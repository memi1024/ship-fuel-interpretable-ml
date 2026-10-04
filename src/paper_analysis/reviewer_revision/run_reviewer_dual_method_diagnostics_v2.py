#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_reviewer_dual_method_diagnostics.py

Purpose
-------
Generate side-by-side diagnostics for the new reviewer comments while preserving
the locked F31 experimental design.

This script deliberately provides:
A) RECOMMENDED / journal-conservative analyses
B) ORIGINAL-STYLE comparison analyses, where scientifically defensible

It does NOT:
- retune the locked XGBoost model;
- rebuild the official L1 split;
- retrain the H3 model;
- treat source pathway as causally separable from ship type;
- infer rudder "absence" from a non-confirmation result.

Main modules
------------
1) hypothesis_inference
   H1:
      A. vessel-cluster bootstrap CI for L2-L1 and L3-L1 pooled RMSE differences
      B. paired vessel sign-flip permutation test for per-vessel RMSE differences
   H3:
      A. vessel-bootstrap CIs for FT weighted CII deterioration and FD total-fuel
         changes, plus pairwise ship-type contrasts
      B. original-style ship-type × reduction interaction test using cluster-robust
         OLS/Wald F test, plus vessel-label permutation sensitivity

2) lovo_bulk
   A. vessel-level normalized Wasserstein-1 + one-dimensional KDE overlap for
      SOG, draught, and wave height, summarised by ship type
   B. original-style three-feature multivariate KDE overlap diagnostic

3) wave_c4
   Post-hoc stratification of the +10% wave-height perturbation by ship type,
   empirical wave tertile, speed tertile, relative-wave sector, and selected
   two-way strata. Uses the locked L1 model; no refit.

4) rudder
   Response-level rudder non-confirmation table using row-level paired prediction
   deltas, tail thresholds, and vessel-cluster bootstrap CIs. No predictor-SD-based
   MDE claim.

5) time_resolution
   If a paired row-level 5-min->10-min vs direct-10-min prediction CSV is supplied
   (or auto-discovered), produces:
      A. paired vessel-cluster bootstrap and sign-flip comparison
      B. residual ACF/DW/BP and within-window speed-variability diagnostics
   If the paired file is absent, the module writes an exact input-schema note and
   skips without affecting the other modules.

6) source_pathway
   A. conservative descriptive source-pathway summaries for L2/L3, SHAP and H3,
      with an explicit structural-confounding statement
   B. original-style between-pathway descriptive/permutation comparisons
      (NON-CAUSAL; source=ship-type bundle in this dataset)

Expected locked study
---------------------
Cruise cohort: 489,620
Vessels: 21
L1 train/test: 391,696 / 97,924
Canonical features: 17

Recommended Windows command
---------------------------
python src/paper_analysis/reviewer_revision/run_reviewer_dual_method_diagnostics_v2.py ^
  --work-dir "src/paper_analysis" ^
  --main-output "results" ^
  --raw-cruise "C:\\path\\to\\final_fixed31_cruise.csv" ^
  --out-dir "revision_runs/reviewer_dual_method_diagnostics"

Optional time-resolution input:
  --temporal-resolution-csv "C:\\path\\to\\paired_5min_vs_10min_predictions.csv"

See REQUIRED_TIME_RESOLUTION_SCHEMA.txt produced by this script if the paired
row-level file is not auto-discovered.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd

from scipy.stats import gaussian_kde, wasserstein_distance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KernelDensity
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import acf


VERSION = "2026-08-19.reviewer-dual-method-v2-rowid-fix"

EXPECTED_CRUISE = 489_620
EXPECTED_TRAIN = 391_696
EXPECTED_TEST = 97_924
EXPECTED_TEMPORAL_TEST = 97_934
EXPECTED_VESSELS = 21

PRIMARY_FEATURES = [
    "speed_kn",
    "heading_sin",
    "heading_cos",
    "draught_m",
    "trim_m",
    "rudder_deg",
    "rel_wind_speed_kn",
    "rel_wind_sin",
    "rel_wind_cos",
    "wave_height_m",
    "rel_wave_sin",
    "rel_wave_cos",
    "wave_period_s",
    "sst_c",
    "mslp_hpa",
    "ship_type_bulk",
    "ship_type_container",
]

SHIFT_FEATURES = ["speed_kn", "draught_m", "wave_height_m"]


# ---------------------------------------------------------------------
# CLI / IO
# ---------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument(
        "--work-dir",
        default="src/paper_analysis",
        help="Folder containing f31_core_memory_safe.py and column_overrides_fixed31.json.",
    )
    p.add_argument(
        "--main-output",
        default="results",
    )
    p.add_argument(
        "--raw-cruise",
        default="data/final_fixed31_cruise.csv",
    )
    p.add_argument("--core-path", default=None)
    p.add_argument("--column-overrides", default=None)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--csv-chunksize", type=int, default=25_000)
    p.add_argument("--trajectory-gap-minutes", type=float, default=30.0)
    p.add_argument("--bootstrap", type=int, default=5_000)
    p.add_argument("--permutations", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=20260819)

    p.add_argument(
        "--steps",
        default="all",
        help=(
            "Comma-separated: hypothesis_inference,lovo_bulk,wave_c4,"
            "rudder,time_resolution,source_pathway or all"
        ),
    )

    p.add_argument("--wave-sample-n", type=int, default=20_000)
    p.add_argument("--wave-min-group-n", type=int, default=200)

    p.add_argument("--kde-sample-n", type=int, default=5_000)
    p.add_argument("--kde-eval-n", type=int, default=3_000)

    p.add_argument(
        "--temporal-resolution-csv",
        default=None,
        help=(
            "Optional paired row-level CSV for 5-min->10-min aggregate versus "
            "direct 10-min predictions."
        ),
    )
    p.add_argument("--acf-max-lag", type=int, default=12)

    p.add_argument(
        "--l4-by-vessel-csv",
        default=None,
        help="Optional L4 by-vessel performance table for source-pathway summary.",
    )

    return p.parse_args()


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p


def atomic_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def normalize_ship_type(x):
    s = str(x).strip().lower()
    if "bulk" in s:
        return "bulk"
    if "container" in s:
        return "container"
    if "tank" in s:
        return "tanker"
    return s


def met_source_from_ship_type(x):
    st = normalize_ship_type(x)
    return "ERA5/container" if st == "container" else "pre-matched/bulk+tanker"


def find_variant(folder: Path, canonical: str) -> Path:
    stems = [canonical]
    if canonical.endswith(".py"):
        stems += [
            canonical.replace(".py", "(1).py"),
            canonical.replace(".py", "(2).py"),
        ]
    if canonical.endswith(".json"):
        stems += [
            canonical.replace(".json", "(1).json"),
            canonical.replace(".json", "(2).json"),
        ]
    for name in stems:
        p = folder / name
        if p.exists():
            return p
    raise FileNotFoundError(f"Could not resolve {canonical} in {folder}")


def find_one(root: Path, patterns, required=True, prefer_contains=()):
    hits = []
    for pattern in patterns:
        hits.extend(root.rglob(pattern))
    # de-duplicate
    uniq = []
    seen = set()
    for h in hits:
        k = str(h.resolve())
        if k not in seen:
            uniq.append(h)
            seen.add(k)
    hits = uniq

    if prefer_contains and hits:
        scored = []
        for p in hits:
            s = str(p).lower()
            score = sum(1 for token in prefer_contains if token.lower() in s)
            scored.append((score, len(str(p)), p))
        scored.sort(key=lambda z: (-z[0], z[1], str(z[2])))
        hits = [z[2] for z in scored]
    else:
        hits = sorted(hits, key=lambda p: (len(str(p)), str(p)))

    if hits:
        return hits[0]
    if required:
        raise FileNotFoundError(
            f"Could not find any of {patterns} under {root}"
        )
    return None


def load_core(args):
    work = Path(args.work_dir)
    core_path = (
        Path(args.core_path)
        if args.core_path
        else find_variant(work, "f31_core_memory_safe.py")
    )
    if not core_path.exists():
        raise FileNotFoundError(core_path)

    name = "f31_core_memory_safe_reviewer"
    spec = importlib.util.spec_from_file_location(name, core_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {core_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)

    override_path = (
        Path(args.column_overrides)
        if args.column_overrides
        else find_variant(work, "column_overrides_fixed31.json")
    )
    return mod, core_path, override_path


def load_locked_data(core, args, overrides_path):
    ns = SimpleNamespace(
        column_overrides=str(overrides_path),
        trajectory_gap_minutes=float(args.trajectory_gap_minutes),
        csv_chunksize=int(args.csv_chunksize),
    )
    raw_path = Path(args.raw_cruise)
    if not raw_path.exists():
        raise FileNotFoundError(raw_path)

    print("Reading cruise cohort through exact F31 core...")
    df = core.read_csv_memory_safe(raw_path, ns, _PrintLogger())
    bundle = core.canonicalize_dataframe(df, ns, _PrintLogger())
    del df
    gc.collect()

    raw = bundle.raw.reset_index(drop=True)
    X = bundle.feature_df.reset_index(drop=True)

    # The canonical F31 bundle does not necessarily retain the temporary row_id
    # used by some older diagnostic scripts.  After reset_index(drop=True), the
    # canonical row position is the stable tie-breaker used for deterministic
    # temporal ordering.  Reconstruct it explicitly without changing row order.
    if "row_id" not in raw.columns:
        raw["row_id"] = np.arange(len(raw), dtype=np.int64)
        print(
            "[INFO] canonical raw has no row_id; "
            "reconstructed row_id from locked canonical row order."
        )

    raw["ship_type"] = raw["ship_type"].map(normalize_ship_type)

    if len(raw) != EXPECTED_CRUISE:
        raise AssertionError(
            f"Expected {EXPECTED_CRUISE:,} cruise rows, got {len(raw):,}"
        )
    if raw["vessel_id"].nunique() != EXPECTED_VESSELS:
        raise AssertionError(
            f"Expected {EXPECTED_VESSELS} vessels, got "
            f"{raw['vessel_id'].nunique()}"
        )
    if list(X.columns) != PRIMARY_FEATURES:
        raise AssertionError(
            "Canonical feature order differs from locked study:\n"
            + repr(list(X.columns))
        )

    return raw, X


class _PrintLogger:
    def info(self, msg, *args):
        try:
            print(msg % args if args else msg)
        except Exception:
            print(msg, *args)
    def warning(self, msg, *args):
        try:
            print("[WARN] " + (msg % args if args else msg))
        except Exception:
            print("[WARN]", msg, *args)


def resolve_locked_artifacts(main: Path):
    art = main / "14_artifacts"

    split = art / "record_split_indices.npz"
    model = art / "xgb_record_split.joblib"
    l1_pred = art / "xgb_record_test_prediction.npy"

    for p in [split, model, l1_pred]:
        if not p.exists():
            raise FileNotFoundError(p)

    lovo = find_one(
        main,
        ["LOVO_predictions_xgb.npy", "LOVO_prediction_xgb.npy"],
        required=True,
        prefer_contains=("05_lovo",),
    )
    temporal = find_one(
        main,
        ["temporal_prediction_xgb.npy", "temporal_predictions_xgb.npy"],
        required=True,
        prefer_contains=("15_revision_diagnostics", "02_temporal"),
    )

    return {
        "split": split,
        "model": model,
        "l1_pred": l1_pred,
        "lovo_pred": lovo,
        "temporal_pred": temporal,
    }


def known_vessel_temporal_split(raw, frac=0.8):
    """
    Exact known-vessel first-80% / last-20% temporal split.

    Timestamp is the primary ordering key.  row_id is only a deterministic
    tie-breaker.  If an older/core data bundle does not expose row_id, the
    canonical dataframe index is used as the equivalent stable row-order key.
    """
    train_idx, test_idx = [], []

    for _, inds in raw.groupby("vessel_id").groups.items():
        sub = raw.loc[inds].copy()

        if "row_id" in sub.columns:
            tie_col = "row_id"
        else:
            tie_col = "_canonical_row_order"
            sub[tie_col] = sub.index.to_numpy(dtype=np.int64)

        sort_cols = ["timestamp", tie_col]
        sub = sub.sort_values(sort_cols, kind="mergesort")

        cut = max(
            1,
            min(len(sub) - 1, int(math.floor(len(sub) * frac))),
        )
        train_idx.extend(sub.index[:cut].tolist())
        test_idx.extend(sub.index[cut:].tolist())

    return (
        np.asarray(sorted(train_idx), dtype=int),
        np.asarray(sorted(test_idx), dtype=int),
    )


def validate_predictions(raw, X, core, artifacts):
    split = np.load(artifacts["split"])
    tr = np.asarray(split["train_idx"], dtype=int)
    te = np.asarray(split["test_idx"], dtype=int)

    if len(tr) != EXPECTED_TRAIN or len(te) != EXPECTED_TEST:
        raise AssertionError(
            f"Locked L1 split changed: train={len(tr)}, test={len(te)}"
        )

    model = joblib.load(artifacts["model"])
    p_saved = np.asarray(np.load(artifacts["l1_pred"]), dtype=float)
    p_now = np.asarray(
        model.predict(
            core.feature_matrix_from_raw(
                raw.iloc[te].reset_index(drop=True),
                list(X.columns),
            )
        ),
        dtype=float,
    )

    if not np.allclose(
        p_saved, p_now, rtol=1e-7, atol=1e-9, equal_nan=True
    ):
        diff = float(np.max(np.abs(p_saved - p_now)))
        raise AssertionError(
            f"Locked L1 prediction alignment failed, max diff={diff:.9g}"
        )

    lovo = np.asarray(np.load(artifacts["lovo_pred"]), dtype=float)
    if lovo.shape != (len(raw),) or not np.isfinite(lovo).all():
        raise AssertionError(
            f"LOVO prediction shape/finite failure: {lovo.shape}"
        )

    temporal_tr, temporal_te = known_vessel_temporal_split(raw, 0.8)

    if len(temporal_te) != EXPECTED_TEMPORAL_TEST:
        raise AssertionError(
            "Temporal split count changed: "
            f"{len(temporal_te):,} != {EXPECTED_TEMPORAL_TEST:,}. "
            "Stop rather than silently re-defining L2."
        )

    temporal = np.asarray(
        np.load(artifacts["temporal_pred"]), dtype=float
    )
    if temporal.shape != (len(temporal_te),) or not np.isfinite(temporal).all():
        raise AssertionError(
            "Temporal prediction cache does not match exact first80-last20 split."
        )

    # A shape match alone is not enough: verify that the reconstructed test-row
    # order reproduces the locked L2 metrics.  This catches ordering mistakes.
    temporal_y = raw.iloc[temporal_te]["target"].to_numpy(float)
    temporal_rmse = float(
        np.sqrt(mean_squared_error(temporal_y, temporal))
    )
    temporal_r2 = float(r2_score(temporal_y, temporal))

    if not (
        abs(temporal_rmse - 0.0916) <= 0.0015
        and abs(temporal_r2 - 0.6369) <= 0.015
    ):
        raise AssertionError(
            "Temporal-cache ordering/alignment check failed: "
            f"reconstructed RMSE={temporal_rmse:.6f}, "
            f"R2={temporal_r2:.6f}. "
            "Expected approximately 0.0916 / 0.6369 from the locked L2 run. "
            "Do not continue until the temporal-row ordering is resolved."
        )

    print(
        "[PASS] Locked L2 temporal cache alignment | "
        f"n={len(temporal_te):,} | "
        f"RMSE={temporal_rmse:.6f} | R2={temporal_r2:.6f}"
    )

    return {
        "record_tr": tr,
        "record_te": te,
        "model": model,
        "l1_pred": p_saved,
        "lovo_pred": lovo,
        "temporal_tr": temporal_tr,
        "temporal_te": temporal_te,
        "temporal_pred": temporal,
    }


# ---------------------------------------------------------------------
# Generic statistics
# ---------------------------------------------------------------------

def metric_dict(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return {
        "n": int(len(y)),
        "RMSE": float(np.sqrt(mean_squared_error(y, p))),
        "MAE": float(mean_absolute_error(y, p)),
        "R2": float(r2_score(y, p)),
        "SSE": float(np.sum((p - y) ** 2)),
    }


def per_vessel_metrics(raw_subset, pred, validation):
    rows = []
    z = raw_subset.reset_index(drop=True)
    pred = np.asarray(pred, dtype=float)
    for vessel, idx in z.groupby("vessel_id").groups.items():
        ii = np.asarray(list(idx), dtype=int)
        y = z.loc[ii, "target"].to_numpy(float)
        p = pred[ii]
        m = metric_dict(y, p)
        rows.append({
            "validation": validation,
            "vessel_id": str(vessel),
            "ship_type": normalize_ship_type(z.loc[ii[0], "ship_type"]),
            **m,
        })
    return pd.DataFrame(rows)


def cluster_bootstrap_pooled_rmse_difference(
    a_byv: pd.DataFrame,
    b_byv: pd.DataFrame,
    reps: int,
    seed: int,
):
    """
    Difference = RMSE_B - RMSE_A.
    Resampling unit is vessel. Each validation keeps its own record counts/SSE.
    """
    a = a_byv.set_index("vessel_id")
    b = b_byv.set_index("vessel_id")
    common = sorted(set(a.index).intersection(b.index))
    if len(common) < 3:
        raise ValueError("Too few common vessels.")

    a = a.loc[common]
    b = b.loc[common]
    nA = a["n"].to_numpy(float)
    sA = a["SSE"].to_numpy(float)
    nB = b["n"].to_numpy(float)
    sB = b["SSE"].to_numpy(float)

    point_a = math.sqrt(sA.sum() / nA.sum())
    point_b = math.sqrt(sB.sum() / nB.sum())
    point = point_b - point_a

    rng = np.random.default_rng(seed)
    boot = np.empty(reps, dtype=float)
    k = len(common)
    for i in range(reps):
        draw = rng.integers(0, k, size=k)
        rA = math.sqrt(sA[draw].sum() / nA[draw].sum())
        rB = math.sqrt(sB[draw].sum() / nB[draw].sum())
        boot[i] = rB - rA

    return {
        "n_vessels": k,
        "RMSE_A": point_a,
        "RMSE_B": point_b,
        "delta_RMSE_B_minus_A": point,
        "bootstrap_CI95_low": float(np.quantile(boot, .025)),
        "bootstrap_CI95_high": float(np.quantile(boot, .975)),
        "bootstrap_prob_delta_gt_0": float(np.mean(boot > 0)),
    }


def signflip_permutation_mean_difference(
    a_byv: pd.DataFrame,
    b_byv: pd.DataFrame,
    reps: int,
    seed: int,
):
    """
    Original-style paired permutation/sign-flip test on per-vessel RMSE.
    H_A: RMSE_B > RMSE_A.
    """
    a = a_byv.set_index("vessel_id")
    b = b_byv.set_index("vessel_id")
    common = sorted(set(a.index).intersection(b.index))
    d = (
        b.loc[common, "RMSE"].to_numpy(float)
        - a.loc[common, "RMSE"].to_numpy(float)
    )
    obs = float(d.mean())

    rng = np.random.default_rng(seed)
    exceed = 0
    chunk = 10_000
    done = 0
    while done < reps:
        r = min(chunk, reps - done)
        signs = rng.choice(
            np.array([-1.0, 1.0]),
            size=(r, len(d)),
            replace=True,
        )
        stats = (signs * d[None, :]).mean(axis=1)
        exceed += int(np.sum(stats >= obs))
        done += r

    p_one = (exceed + 1.0) / (reps + 1.0)
    return {
        "n_vessels": len(common),
        "mean_per_vessel_delta_RMSE": obs,
        "one_sided_signflip_p": float(p_one),
    }


def cluster_bootstrap_group_mean(values, vessels, reps, seed):
    values = np.asarray(values, dtype=float)
    vessels = np.asarray(vessels).astype(str)
    uniq = np.unique(vessels)
    by = {
        v: values[vessels == v]
        for v in uniq
    }
    rng = np.random.default_rng(seed)
    means = np.empty(reps)
    for i in range(reps):
        draw = rng.choice(uniq, size=len(uniq), replace=True)
        vals = np.concatenate([by[v] for v in draw])
        means[i] = float(np.mean(vals))
    return (
        float(np.mean(values)),
        float(np.quantile(means, .025)),
        float(np.quantile(means, .975)),
    )


# ---------------------------------------------------------------------
# 1) HYPOTHESIS INFERENCE: two methods
# ---------------------------------------------------------------------

def h1_inference(raw, locked, args, out):
    d = ensure_dir(out / "01_hypothesis_inference" / "H1")

    te = locked["record_te"]
    tte = locked["temporal_te"]

    L1 = per_vessel_metrics(
        raw.iloc[te].reset_index(drop=True),
        locked["l1_pred"],
        "L1",
    )
    L2 = per_vessel_metrics(
        raw.iloc[tte].reset_index(drop=True),
        locked["temporal_pred"],
        "L2",
    )
    L3 = per_vessel_metrics(
        raw.reset_index(drop=True),
        locked["lovo_pred"],
        "L3",
    )

    atomic_csv(
        pd.concat([L1, L2, L3], ignore_index=True),
        d / "Table_H1_0_per_vessel_metrics.csv",
    )

    rows_a = []
    rows_b = []
    for j, (B, name) in enumerate([(L2, "L2_minus_L1"), (L3, "L3_minus_L1")]):
        a = cluster_bootstrap_pooled_rmse_difference(
            L1, B, args.bootstrap, args.seed + 100 + j
        )
        a.update({
            "contrast": name,
            "method": "A_recommended_vessel_cluster_bootstrap",
            "decision_rule": "CI excludes 0 on positive side => supported RMSE decay",
        })
        rows_a.append(a)

        b = signflip_permutation_mean_difference(
            L1, B, args.permutations, args.seed + 120 + j
        )
        b.update({
            "contrast": name,
            "method": "B_original_style_paired_vessel_signflip_permutation",
            "alternative": "RMSE_B > RMSE_A",
        })
        rows_b.append(b)

    atomic_csv(
        pd.DataFrame(rows_a),
        d / "Table_H1_A_cluster_bootstrap_RMSE_decay.csv",
    )
    atomic_csv(
        pd.DataFrame(rows_b),
        d / "Table_H1_B_signflip_permutation_RMSE_decay.csv",
    )

    write_text(
        d / "H1_INTERPRETATION.txt",
        """Primary recommendation:
Use Table_H1_A as the main inferential result. The vessel-cluster bootstrap
respects the 21-vessel dependence structure and estimates the uncertainty of the
pooled RMSE deterioration.

Use Table_H1_B only as a secondary/original-style paired permutation sensitivity.
Do not treat individual ten-minute rows as iid replicates.

For manuscript wording, prefer:
'RMSE deterioration from L1 to L2/L3 was supported by a vessel-cluster bootstrap
95% confidence interval that excluded zero.'
R2 should remain an important descriptive effect-size/generalisation metric, not
the main p-value target.
""",
    )


def ft_weighted_cii_change(g):
    # constant CO2 conversion factor cancels in the ratio
    b_f = g["baseline_predicted_fuel_t"].sum()
    b_tw = g["baseline_transport_work"].sum()
    f_f = g["FT_predicted_fuel_t"].sum()
    f_tw = g["FT_transport_work"].sum()
    return ((f_f / f_tw) / (b_f / b_tw) - 1.0) * 100.0


def fd_total_fuel_change(g):
    return (
        g["FD_predicted_total_fuel_t"].sum()
        / g["baseline_predicted_fuel_t"].sum()
        - 1.0
    ) * 100.0


def bootstrap_vessel_aggregate(g, stat_fn, reps, seed):
    vessels = sorted(g["vessel_id"].astype(str).unique())
    by = {
        v: g[g["vessel_id"].astype(str) == v].copy()
        for v in vessels
    }
    point = float(stat_fn(g))
    rng = np.random.default_rng(seed)
    vals = np.empty(reps)
    for i in range(reps):
        draw = rng.choice(vessels, size=len(vessels), replace=True)
        zz = pd.concat([by[v] for v in draw], ignore_index=True)
        vals[i] = stat_fn(zz)
    return (
        point,
        float(np.quantile(vals, .025)),
        float(np.quantile(vals, .975)),
        vals,
    )


def independent_bootstrap_contrast(
    ga, gb, stat_fn, reps, seed
):
    va = sorted(ga["vessel_id"].astype(str).unique())
    vb = sorted(gb["vessel_id"].astype(str).unique())
    map_a = {
        v: ga[ga["vessel_id"].astype(str) == v].copy()
        for v in va
    }
    map_b = {
        v: gb[gb["vessel_id"].astype(str) == v].copy()
        for v in vb
    }
    point = float(stat_fn(ga) - stat_fn(gb))
    rng = np.random.default_rng(seed)
    vals = np.empty(reps)
    for i in range(reps):
        da = rng.choice(va, size=len(va), replace=True)
        db = rng.choice(vb, size=len(vb), replace=True)
        za = pd.concat([map_a[v] for v in da], ignore_index=True)
        zb = pd.concat([map_b[v] for v in db], ignore_index=True)
        vals[i] = stat_fn(za) - stat_fn(zb)
    return (
        point,
        float(np.quantile(vals, .025)),
        float(np.quantile(vals, .975)),
    )


def cluster_robust_interaction_test(df, outcome):
    x = df.copy()
    x["reduction_cat"] = x["reduction_pct"].astype(str)
    model = smf.ols(
        f"{outcome} ~ C(reduction_cat) * C(ship_type)",
        data=x,
    )
    fit = model.fit(
        cov_type="cluster",
        cov_kwds={"groups": x["vessel_id"].astype(str)},
    )

    names = list(fit.params.index)
    interaction_idx = [
        i for i, n in enumerate(names)
        if ":" in n
    ]
    if not interaction_idx:
        return {
            "outcome": outcome,
            "interaction_terms": 0,
            "F": np.nan,
            "p_value": np.nan,
            "df_num": np.nan,
            "df_denom": np.nan,
        }

    R = np.zeros((len(interaction_idx), len(names)))
    for r, i in enumerate(interaction_idx):
        R[r, i] = 1.0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        test = fit.f_test(R)

    return {
        "outcome": outcome,
        "interaction_terms": len(interaction_idx),
        "F": float(np.asarray(test.fvalue).squeeze()),
        "p_value": float(np.asarray(test.pvalue).squeeze()),
        "df_num": float(test.df_num),
        "df_denom": float(test.df_denom),
    }


def ordinary_interaction_F(df, outcome):
    """
    Fast ordinary two-way interaction F statistic for permutation sensitivity.
    Uses direct least squares to avoid statsmodels formula overhead inside
    thousands of vessel-label permutations.
    """
    x = df.copy()
    y = x[outcome].to_numpy(float)

    reductions = sorted(x["reduction_pct"].unique())
    ship_types = ["bulk", "container", "tanker"]

    # Baselines: first reduction level and bulk.
    r_dummies = []
    for r in reductions[1:]:
        r_dummies.append(
            (x["reduction_pct"].to_numpy(float) == float(r)).astype(float)
        )

    s_dummies = []
    for s in ship_types[1:]:
        s_dummies.append(
            (x["ship_type"].astype(str).to_numpy() == s).astype(float)
        )

    cols_reduced = [np.ones(len(x))] + r_dummies + s_dummies
    cols_full = list(cols_reduced)

    for rd in r_dummies:
        for sd in s_dummies:
            cols_full.append(rd * sd)

    Xr = np.column_stack(cols_reduced)
    Xf = np.column_stack(cols_full)

    br, *_ = np.linalg.lstsq(Xr, y, rcond=None)
    bf, *_ = np.linalg.lstsq(Xf, y, rcond=None)

    er = y - Xr @ br
    ef = y - Xf @ bf
    ssr_r = float(er @ er)
    ssr_f = float(ef @ ef)

    df_num = Xf.shape[1] - Xr.shape[1]
    df_den = len(y) - Xf.shape[1]
    if df_num <= 0 or df_den <= 0 or ssr_f <= 0:
        return np.nan

    return float(
        ((ssr_r - ssr_f) / df_num) / (ssr_f / df_den)
    )


def vessel_label_permutation_interaction(
    df, outcome, reps, seed
):
    """
    Permute ship-type labels at the vessel-profile level.
    This is a non-causal sensitivity test.
    """
    x = df.copy()
    vessels = (
        x[["vessel_id", "ship_type"]]
        .drop_duplicates()
        .sort_values("vessel_id")
        .reset_index(drop=True)
    )
    observed = ordinary_interaction_F(x, outcome)
    labels = vessels["ship_type"].to_numpy(str)
    vids = vessels["vessel_id"].to_numpy(str)

    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(reps):
        perm = rng.permutation(labels)
        mp = dict(zip(vids, perm))
        z = x.copy()
        z["ship_type"] = z["vessel_id"].astype(str).map(mp)
        f = ordinary_interaction_F(z, outcome)
        exceed += int(f >= observed)

    return {
        "outcome": outcome,
        "observed_interaction_F": observed,
        "vessel_label_permutation_p": (exceed + 1) / (reps + 1),
        "permutations": reps,
    }


def h3_inference(main, args, out):
    d = ensure_dir(out / "01_hypothesis_inference" / "H3")

    ft_path = find_one(
        main,
        ["Table_15_FT_L1_aligned_by_vessel*.csv"],
        required=True,
    )
    fd_path = find_one(
        main,
        ["Table_16_FD_fuel_time_decomposition_by_vessel*.csv"],
        required=True,
    )
    ft = pd.read_csv(ft_path)
    fd = pd.read_csv(fd_path)
    ft["ship_type"] = ft["ship_type"].map(normalize_ship_type)
    fd["ship_type"] = fd["ship_type"].map(normalize_ship_type)

    # A: primary bootstrap uncertainty for aggregate endpoints
    a_rows = []
    pair_rows = []
    types = ["bulk", "container", "tanker"]

    for outcome_name, data, stat_fn in [
        ("FT_weighted_CII_change_pct", ft, ft_weighted_cii_change),
        ("FD_total_fuel_change_pct", fd, fd_total_fuel_change),
    ]:
        for ri, r in enumerate(sorted(data["reduction_pct"].unique())):
            z = data[data["reduction_pct"] == r].copy()
            for ti, st in enumerate(["Fleet"] + types):
                g = z if st == "Fleet" else z[z["ship_type"] == st]
                point, lo, hi, boots = bootstrap_vessel_aggregate(
                    g,
                    stat_fn,
                    args.bootstrap,
                    args.seed + 2000 + 100 * ri + ti
                    + (500 if outcome_name.startswith("FD") else 0),
                )
                a_rows.append({
                    "outcome": outcome_name,
                    "reduction_pct": r,
                    "scope": st,
                    "n_vessels": g["vessel_id"].nunique(),
                    "estimate_pct": point,
                    "vessel_bootstrap_CI95_low": lo,
                    "vessel_bootstrap_CI95_high": hi,
                    "bootstrap_prob_above_zero": float(np.mean(boots > 0)),
                    "bootstrap_prob_below_zero": float(np.mean(boots < 0)),
                    "interpretation": (
                        "statistically supported deterioration"
                        if lo > 0
                        else "statistically supported improvement"
                        if hi < 0
                        else "direction not statistically resolved at 95%"
                    ),
                })

            for ai, a in enumerate(types):
                for b in types[ai + 1:]:
                    ga = z[z["ship_type"] == a]
                    gb = z[z["ship_type"] == b]
                    point, lo, hi = independent_bootstrap_contrast(
                        ga, gb, stat_fn, args.bootstrap,
                        args.seed + 2500 + 100 * ri + ai * 10
                        + types.index(b)
                        + (500 if outcome_name.startswith("FD") else 0),
                    )
                    pair_rows.append({
                        "outcome": outcome_name,
                        "reduction_pct": r,
                        "contrast": f"{a}_minus_{b}",
                        "estimate_difference_pp": point,
                        "vessel_bootstrap_CI95_low": lo,
                        "vessel_bootstrap_CI95_high": hi,
                        "CI_excludes_zero": bool(lo > 0 or hi < 0),
                    })

    atomic_csv(
        pd.DataFrame(a_rows),
        d / "Table_H3_A_vessel_bootstrap_endpoint_CI.csv",
    )
    atomic_csv(
        pd.DataFrame(pair_rows),
        d / "Table_H3_A_pairwise_shiptype_contrasts.csv",
    )

    # B: original-style global interaction on vessel-level outcomes
    ft_reg = ft[
        ["vessel_id", "ship_type", "reduction_pct", "FT_CII_change_pct"]
    ].copy()
    fd_reg = fd[
        [
            "vessel_id",
            "ship_type",
            "reduction_pct",
            "FD_total_fuel_change_pct_recomputed",
        ]
    ].rename(
        columns={
            "FD_total_fuel_change_pct_recomputed": "FD_total_fuel_change_pct"
        }
    )

    interaction_rows = []
    permutation_rows = []

    for j, (z, outcome) in enumerate([
        (ft_reg, "FT_CII_change_pct"),
        (fd_reg, "FD_total_fuel_change_pct"),
    ]):
        res = cluster_robust_interaction_test(z, outcome)
        res["method"] = "B_original_style_cluster_robust_interaction"
        interaction_rows.append(res)

        # keep permutations manageable; user can raise --permutations
        pr = vessel_label_permutation_interaction(
            z, outcome, args.permutations, args.seed + 3000 + j
        )
        pr["method"] = "B2_vessel_profile_label_permutation"
        permutation_rows.append(pr)

    atomic_csv(
        pd.DataFrame(interaction_rows),
        d / "Table_H3_B_cluster_robust_shiptype_x_reduction_interaction.csv",
    )
    atomic_csv(
        pd.DataFrame(permutation_rows),
        d / "Table_H3_B2_vessel_label_permutation_interaction.csv",
    )

    write_text(
        d / "H3_INTERPRETATION.txt",
        """Recommended hierarchy:
1) Main text: Table_H3_A vessel-bootstrap confidence intervals for the actual
   operational endpoints (FT weighted CII-proxy change; FD total-fuel change).
2) Main/supplement: pairwise ship-type contrast CIs.
3) Supplementary sensitivity only: cluster-robust ship-type × reduction global
   interaction test.

Reason: the three reductions are repeated model-based perturbations on the same
21 vessels. A naive ordinary two-way ANOVA would overstate independence. The
cluster-robust interaction test is provided only to reproduce the reviewer's
original statistical framing more safely.

FD is not a second independent hypothesis. It is the fixed-distance mechanism
arm/decomposition of H3.
""",
    )


# ---------------------------------------------------------------------
# 2) LOVO bulk diagnosis: two methods
# ---------------------------------------------------------------------

def one_dim_kde_overlap(train, test, sample_n, seed):
    train = np.asarray(train, dtype=float)
    test = np.asarray(test, dtype=float)
    train = train[np.isfinite(train)]
    test = test[np.isfinite(test)]
    if len(train) < 10 or len(test) < 10:
        return np.nan

    rng = np.random.default_rng(seed)
    if len(train) > sample_n:
        train = rng.choice(train, size=sample_n, replace=False)
    if len(test) > sample_n:
        test = rng.choice(test, size=sample_n, replace=False)

    s = float(np.std(train, ddof=1))
    if s <= 1e-12:
        return 1.0 if np.std(test, ddof=1) <= 1e-12 else 0.0

    lo = float(np.quantile(np.concatenate([train, test]), .002))
    hi = float(np.quantile(np.concatenate([train, test]), .998))
    if not np.isfinite(lo + hi) or hi <= lo:
        return np.nan

    try:
        kt = gaussian_kde(train)
        ke = gaussian_kde(test)
        grid = np.linspace(lo, hi, 512)
        pt = kt(grid)
        pe = ke(grid)
        ov = np.trapz(np.minimum(pt, pe), grid)
        return float(np.clip(ov, 0.0, 1.0))
    except Exception:
        # Histogram-overlap fallback
        bins = np.linspace(lo, hi, 65)
        ht, _ = np.histogram(train, bins=bins, density=True)
        he, _ = np.histogram(test, bins=bins, density=True)
        widths = np.diff(bins)
        ov = np.sum(np.minimum(ht, he) * widths)
        return float(np.clip(ov, 0.0, 1.0))


def multivariate_kde_overlap(
    train3, test3, sample_n, eval_n, seed
):
    """
    Original-style 3-feature KDE overlap.

    Estimate OVL = integral min(p, q) dx using mixture-sample identity:
    OVL = E_m[ 2 min(p,q)/(p+q) ], m=(p+q)/2.
    """
    A = np.asarray(train3, dtype=float)
    B = np.asarray(test3, dtype=float)
    A = A[np.isfinite(A).all(axis=1)]
    B = B[np.isfinite(B).all(axis=1)]
    if len(A) < 50 or len(B) < 50:
        return np.nan

    rng = np.random.default_rng(seed)
    if len(A) > sample_n:
        A = A[rng.choice(len(A), sample_n, replace=False)]
    if len(B) > sample_n:
        B = B[rng.choice(len(B), sample_n, replace=False)]

    mu = A.mean(axis=0)
    sd = A.std(axis=0, ddof=1)
    sd[sd < 1e-9] = 1.0
    Az = (A - mu) / sd
    Bz = (B - mu) / sd

    n_eff = max(100, min(len(Az), len(Bz)))
    # Scott-type scale for d=3 in standardised space
    bw = float(np.clip(n_eff ** (-1.0 / 7.0), 0.15, 0.80))

    kde_a = KernelDensity(
        kernel="gaussian", bandwidth=bw, algorithm="auto"
    ).fit(Az)
    kde_b = KernelDensity(
        kernel="gaussian", bandwidth=bw, algorithm="auto"
    ).fit(Bz)

    na = min(eval_n // 2, len(Az))
    nb = min(eval_n - na, len(Bz))
    Ea = Az[rng.choice(len(Az), na, replace=False)]
    Eb = Bz[rng.choice(len(Bz), nb, replace=False)]
    E = np.vstack([Ea, Eb])

    la = kde_a.score_samples(E)
    lb = kde_b.score_samples(E)

    # stable 2*min(p,q)/(p+q)
    lmin = np.minimum(la, lb)
    lden = np.logaddexp(la, lb)
    w = 2.0 * np.exp(lmin - lden)
    return float(np.mean(w))


def run_lovo_bulk(raw, locked, main, args, out):
    d = ensure_dir(out / "02_LOVO_bulk_diagnostics")

    # Prefer existing exact W1 table if present
    d2_path = find_one(
        main,
        ["Table_D2_L3_LOVO_distribution_shift_and_performance*.csv"],
        required=False,
    )
    if d2_path is not None:
        existing = pd.read_csv(d2_path)
        existing["ship_type"] = existing["ship_type"].map(normalize_ship_type)
    else:
        existing = None

    lovo_byv = per_vessel_metrics(
        raw.reset_index(drop=True),
        locked["lovo_pred"],
        "L3",
    ).set_index("vessel_id")

    rows = []
    vessels = sorted(raw["vessel_id"].astype(str).unique())

    for vi, vessel in enumerate(vessels):
        te = raw["vessel_id"].astype(str).to_numpy() == vessel
        tr = ~te
        st = normalize_ship_type(raw.loc[te, "ship_type"].iloc[0])

        row = {
            "vessel_id": vessel,
            "ship_type": st,
            "met_source": met_source_from_ship_type(st),
            "LOVO_RMSE": float(lovo_byv.loc[vessel, "RMSE"]),
            "LOVO_R2": float(lovo_byv.loc[vessel, "R2"]),
            "test_n": int(te.sum()),
        }

        for fi, feat in enumerate(SHIFT_FEATURES):
            a = raw.loc[tr, feat].to_numpy(float)
            b = raw.loc[te, feat].to_numpy(float)

            if existing is not None:
                q = existing[
                    (existing["vessel_id"].astype(str) == vessel)
                    & (existing["feature"] == feat)
                ]
                if len(q):
                    nw1 = float(q.iloc[0]["normalized_wasserstein1"])
                    w1 = float(q.iloc[0]["wasserstein1"])
                else:
                    w1 = float(wasserstein_distance(a, b))
                    nw1 = w1 / max(float(np.std(a, ddof=1)), 1e-12)
            else:
                w1 = float(wasserstein_distance(a, b))
                nw1 = w1 / max(float(np.std(a, ddof=1)), 1e-12)

            ov = one_dim_kde_overlap(
                a, b, args.kde_sample_n,
                args.seed + 4000 + vi * 20 + fi,
            )
            row[f"{feat}_W1"] = w1
            row[f"{feat}_normalized_W1"] = nw1
            row[f"{feat}_KDE_overlap_1D"] = ov

        row["KDE_overlap_3D_speed_draught_wave"] = (
            multivariate_kde_overlap(
                raw.loc[tr, SHIFT_FEATURES].to_numpy(float),
                raw.loc[te, SHIFT_FEATURES].to_numpy(float),
                args.kde_sample_n,
                args.kde_eval_n,
                args.seed + 5000 + vi,
            )
        )
        rows.append(row)
        print(
            f"LOVO overlap {vi+1:02d}/{len(vessels)} | "
            f"{vessel} | {st}"
        )

    tab = pd.DataFrame(rows)
    atomic_csv(
        tab,
        d / "Table_LOVO_A_B_vessel_shift_overlap_diagnostics.csv",
    )

    # Recommended summary
    agg_specs = {}
    for feat in SHIFT_FEATURES:
        agg_specs[f"{feat}_normalized_W1_median"] = (
            f"{feat}_normalized_W1", "median"
        )
        agg_specs[f"{feat}_KDE_overlap_1D_median"] = (
            f"{feat}_KDE_overlap_1D", "median"
        )

    summary = (
        tab.groupby("ship_type")
        .agg(
            vessels=("vessel_id", "nunique"),
            positive_R2_vessels=("LOVO_R2", lambda s: int((s > 0).sum())),
            median_LOVO_R2=("LOVO_R2", "median"),
            mean_LOVO_R2=("LOVO_R2", "mean"),
            median_LOVO_RMSE=("LOVO_RMSE", "median"),
            median_speed_nW1=("speed_kn_normalized_W1", "median"),
            median_draught_nW1=("draught_m_normalized_W1", "median"),
            median_wave_nW1=("wave_height_m_normalized_W1", "median"),
            median_speed_1D_overlap=("speed_kn_KDE_overlap_1D", "median"),
            median_draught_1D_overlap=("draught_m_KDE_overlap_1D", "median"),
            median_wave_1D_overlap=("wave_height_m_KDE_overlap_1D", "median"),
            median_3D_KDE_overlap=(
                "KDE_overlap_3D_speed_draught_wave", "median"
            ),
        )
        .reset_index()
    )
    atomic_csv(
        summary,
        d / "Table_LOVO_A_shiptype_summary_recommended.csv",
    )

    # Original-style direct bulk vs non-bulk comparisons (diagnostic only)
    b = tab[tab["ship_type"] == "bulk"]
    nb = tab[tab["ship_type"] != "bulk"]
    comp_rows = []
    metrics = [
        "LOVO_R2",
        "speed_kn_normalized_W1",
        "draught_m_normalized_W1",
        "wave_height_m_normalized_W1",
        "KDE_overlap_3D_speed_draught_wave",
    ]
    rng = np.random.default_rng(args.seed + 5500)

    for metric in metrics:
        obs = float(b[metric].mean() - nb[metric].mean())
        allv = tab[["vessel_id", "ship_type", metric]].copy()
        vals = allv[metric].to_numpy(float)
        n_bulk = len(b)
        exceed = 0
        perm_abs = abs(obs)
        for _ in range(args.permutations):
            order = rng.permutation(len(vals))
            x = vals[order[:n_bulk]]
            y = vals[order[n_bulk:]]
            stat = float(np.mean(x) - np.mean(y))
            exceed += int(abs(stat) >= perm_abs)
        comp_rows.append({
            "metric": metric,
            "bulk_mean": float(b[metric].mean()),
            "nonbulk_mean": float(nb[metric].mean()),
            "bulk_minus_nonbulk": obs,
            "two_sided_vessel_label_permutation_p":
                (exceed + 1) / (args.permutations + 1),
            "note": "diagnostic only; not a causal ship-type/source effect test",
        })

    atomic_csv(
        pd.DataFrame(comp_rows),
        d / "Table_LOVO_B_original_style_bulk_vs_nonbulk_comparison.csv",
    )

    write_text(
        d / "LOVO_INTERPRETATION.txt",
        """Method A (recommended):
Report held-out-vessel normalized Wasserstein-1 together with 1D density overlap,
then summarise by ship type. This directly tests whether the six bulk carriers
occupy more weakly represented operating envelopes.

Method B (original-style):
The 3D KDE overlap combines SOG, draught and wave height into one feature-space
overlap index. It is useful as a diagnostic, but it is bandwidth-dependent and
should be supplementary rather than the sole explanation for negative R2.

Do not conclude that distribution shift proves the cause of LOVO failure. A
negative LOVO R2 can also reflect concept drift, unobserved vessel-specific
propulsion structure, hull/engine condition, and omitted current effects.
""",
    )


# ---------------------------------------------------------------------
# 3) Wave C4 stratified diagnostic
# ---------------------------------------------------------------------

def qcut_safe(s, labels):
    try:
        return pd.qcut(s, q=len(labels), labels=labels, duplicates="drop")
    except Exception:
        return pd.Series(
            ["all"] * len(s), index=s.index, dtype="object"
        )


def relative_sector(sin_v, cos_v):
    angle = (
        np.degrees(np.arctan2(np.asarray(sin_v), np.asarray(cos_v)))
        + 360.0
    ) % 360.0
    out = np.full(len(angle), "cross", dtype=object)
    head = (angle <= 45.0) | (angle >= 315.0)
    following = (angle >= 135.0) & (angle <= 225.0)
    out[head] = "head"
    out[following] = "following"
    return out


def vessel_bootstrap_summary(
    g, reps, seed
):
    vessels = sorted(g["vessel_id"].astype(str).unique())
    by = {
        v: g[g["vessel_id"].astype(str) == v]
        for v in vessels
    }
    up_v = np.array([
        float((by[v]["delta"] > 0).mean())
        for v in vessels
    ])
    mean_v = np.array([
        float(by[v]["delta"].mean())
        for v in vessels
    ])

    rng = np.random.default_rng(seed)
    boot_up = np.empty(reps)
    boot_mean = np.empty(reps)
    n = len(vessels)
    for i in range(reps):
        draw = rng.integers(0, n, size=n)
        boot_up[i] = up_v[draw].mean()
        boot_mean[i] = mean_v[draw].mean()

    return {
        "vessel_balanced_expected_pct": 100.0 * float(up_v.mean()),
        "vessel_balanced_expected_CI95_low":
            100.0 * float(np.quantile(boot_up, .025)),
        "vessel_balanced_expected_CI95_high":
            100.0 * float(np.quantile(boot_up, .975)),
        "vessel_balanced_mean_delta": float(mean_v.mean()),
        "vessel_balanced_mean_delta_CI95_low":
            float(np.quantile(boot_mean, .025)),
        "vessel_balanced_mean_delta_CI95_high":
            float(np.quantile(boot_mean, .975)),
    }


def summarize_wave_group(rows, group_cols, label, args, seed):
    out = []
    grouped = rows.groupby(group_cols, dropna=False)
    for i, (key, g) in enumerate(grouped):
        if not isinstance(key, tuple):
            key = (key,)
        if len(g) < args.wave_min_group_n:
            continue
        if g["vessel_id"].nunique() < 2:
            continue

        base = {
            "stratification": label,
            "n": len(g),
            "n_vessels": g["vessel_id"].nunique(),
            "expected_direction_pct":
                100.0 * float((g["delta"] > 0).mean()),
            "reverse_direction_pct":
                100.0 * float((g["delta"] < 0).mean()),
            "zero_pct":
                100.0 * float((g["delta"] == 0).mean()),
            "median_delta_t_per_10min": float(g["delta"].median()),
            "mean_delta_t_per_10min": float(g["delta"].mean()),
        }
        for c, v in zip(group_cols, key):
            base[c] = v
        base.update(
            vessel_bootstrap_summary(
                g,
                args.bootstrap,
                seed + i,
            )
        )
        base["C4_local_assessment"] = (
            "expected-direction majority"
            if base["expected_direction_pct"] > 50.0
            else "localized non-confirmation"
        )
        out.append(base)
    return out


def run_wave_c4(raw, X, core, locked, args, out):
    d = ensure_dir(out / "03_wave_C4_stratified")

    tr = locked["record_tr"]
    te = locked["record_te"]
    model = locked["model"]

    raw_train = raw.iloc[tr].reset_index(drop=True)
    raw_test = raw.iloc[te].reset_index(drop=True)

    # Support: perturbed wave remains within ship-type-specific L1 training range.
    support_table = (
        raw_train.groupby("ship_type")["wave_height_m"]
        .agg(["min", "max"])
    )
    perturbed_wave = (
        raw_test["wave_height_m"].astype(float).to_numpy() * 1.10
    )
    st = raw_test["ship_type"].astype(str).to_numpy()
    supported = np.zeros(len(raw_test), dtype=bool)
    for ship_type, r in support_table.iterrows():
        m = st == ship_type
        supported[m] = (
            (perturbed_wave[m] >= float(r["min"]))
            & (perturbed_wave[m] <= float(r["max"]))
        )

    base_raw = raw_test.loc[supported].reset_index(drop=True).copy()
    base_pred = locked["l1_pred"][supported]

    pert = base_raw.copy()
    pert["wave_height_m"] = (
        pert["wave_height_m"].astype(float) * 1.10
    )
    pertX = core.feature_matrix_from_raw(
        pert,
        list(X.columns),
    )
    pred_pert = np.asarray(model.predict(pertX), dtype=float)

    rows = base_raw[
        [
            "row_id",
            "vessel_id",
            "ship_type",
            "timestamp",
            "speed_kn",
            "wave_height_m",
            "rel_wave_sin",
            "rel_wave_cos",
        ]
    ].copy()
    rows["prediction_base"] = base_pred
    rows["prediction_wave_plus10"] = pred_pert
    rows["delta"] = pred_pert - base_pred
    rows["wave_state_empirical"] = qcut_safe(
        rows["wave_height_m"],
        ["low", "moderate", "high"],
    ).astype(str)
    rows["speed_band_empirical"] = qcut_safe(
        rows["speed_kn"],
        ["low", "middle", "high"],
    ).astype(str)
    rows["relative_wave_sector"] = relative_sector(
        rows["rel_wave_sin"],
        rows["rel_wave_cos"],
    )

    rows.to_csv(
        d / "wave_plus10_full_supported_row_level.csv.gz",
        index=False,
        compression="gzip",
    )

    # full supported + fixed-size sensitivity sample
    rng = np.random.default_rng(args.seed + 6100)
    n = min(args.wave_sample_n, len(rows))
    sample_idx = np.sort(
        rng.choice(len(rows), size=n, replace=False)
    )
    sample = rows.iloc[sample_idx].reset_index(drop=True)
    sample.to_csv(
        d / "wave_plus10_sample20k_row_level.csv.gz",
        index=False,
        compression="gzip",
    )

    all_summaries = []
    for frame_name, frame, seed0 in [
        ("full_supported_L1", rows, args.seed + 6200),
        ("sample20k_sensitivity", sample, args.seed + 6600),
    ]:
        global_row = {
            "dataset": frame_name,
            "stratification": "global",
            "n": len(frame),
            "n_vessels": frame["vessel_id"].nunique(),
            "expected_direction_pct":
                100.0 * float((frame["delta"] > 0).mean()),
            "reverse_direction_pct":
                100.0 * float((frame["delta"] < 0).mean()),
            "median_delta_t_per_10min": float(frame["delta"].median()),
            "mean_delta_t_per_10min": float(frame["delta"].mean()),
        }
        global_row.update(
            vessel_bootstrap_summary(
                frame,
                args.bootstrap,
                seed0,
            )
        )
        all_summaries.append(global_row)

        group_specs = [
            (["ship_type"], "ship_type"),
            (["wave_state_empirical"], "wave_state_tertile"),
            (["speed_band_empirical"], "speed_tertile"),
            (["relative_wave_sector"], "relative_wave_sector"),
            (
                ["ship_type", "wave_state_empirical"],
                "ship_type_x_wave_state",
            ),
            (
                ["ship_type", "relative_wave_sector"],
                "ship_type_x_wave_sector",
            ),
        ]
        for gi, (cols, label) in enumerate(group_specs):
            zz = summarize_wave_group(
                frame, cols, label, args,
                seed0 + 100 + 50 * gi,
            )
            for r in zz:
                r["dataset"] = frame_name
            all_summaries.extend(zz)

    tab = pd.DataFrame(all_summaries)
    atomic_csv(
        tab,
        d / "Table_WAVE_C4_posthoc_stratified_diagnostics.csv",
    )

    write_text(
        d / "WAVE_C4_INTERPRETATION.txt",
        """This module is explicitly POST-HOC. It diagnoses where the already observed
global wave C4 reverse responses occur; it is not a new a-priori criterion.

Recommended interpretation:
- If a supported subgroup has <=50% expected-direction responses, call it a
  'localized non-confirmation' for that subgroup.
- If reverse responses are dispersed across groups, describe broad conditional
  heterogeneity rather than inventing one failure domain.
- Do not infer from SHAP sign alone; use paired perturbation deltas.
- Do not claim a causal wave-resistance effect.
""",
    )


# ---------------------------------------------------------------------
# 4) Rudder: response-level non-confirmation, no predictor-SD MDE
# ---------------------------------------------------------------------

def run_rudder(raw, X, core, locked, main, args, out):
    d = ensure_dir(out / "04_rudder_nonconfirmation")

    existing = find_one(
        main,
        ["rudder_plus1_row_level.csv.gz", "rudder_plus1_row_level*.csv.gz"],
        required=False,
        prefer_contains=("15_revision_diagnostics", "08_rudder"),
    )

    if existing is not None:
        rows = pd.read_csv(existing)
        print(f"Reusing rudder row-level perturbation: {existing}")
    else:
        rng = np.random.default_rng(args.seed + 822)
        te = locked["record_te"]
        n = min(20_000, len(te))
        local = np.sort(
            rng.choice(len(te), size=n, replace=False)
        )
        global_idx = te[local]

        base_raw = raw.iloc[global_idx].reset_index(drop=True).copy()
        Xb = core.feature_matrix_from_raw(
            base_raw, list(X.columns)
        )
        Xp = Xb.copy()

        r = Xp["rudder_deg"].to_numpy(float)
        sign = np.sign(r)
        sign[sign == 0] = 1.0
        Xp["rudder_deg"] = sign * (np.abs(r) + 1.0)

        pb = np.asarray(locked["model"].predict(Xb), dtype=float)
        pp = np.asarray(locked["model"].predict(Xp), dtype=float)
        rows = base_raw[
            [
                "row_id",
                "vessel_id",
                "ship_type",
                "timestamp",
                "rudder_deg",
                "speed_kn",
                "target",
            ]
        ].copy()
        rows["abs_rudder_original"] = np.abs(
            rows["rudder_deg"].to_numpy(float)
        )
        rows["prediction_base"] = pb
        rows["prediction_abs_rudder_plus1deg"] = pp
        rows["delta"] = pp - pb

    rows["ship_type"] = rows["ship_type"].map(normalize_ship_type)
    rows.to_csv(
        d / "rudder_plus1_row_level_used.csv.gz",
        index=False,
        compression="gzip",
    )

    thresholds = [0.0, 0.5, 1.0, 2.0]
    out_rows = []
    for i, th in enumerate(thresholds):
        g = (
            rows
            if th == 0.0
            else rows[rows["abs_rudder_original"] > th]
        ).copy()
        if len(g) < 3:
            continue
        delta = g["delta"].to_numpy(float)
        sd = float(np.std(delta, ddof=1))
        mean = float(np.mean(delta))
        vb = vessel_bootstrap_summary(
            g, args.bootstrap, args.seed + 7000 + i
        )
        out_rows.append({
            "tail_rule": (
                "all audit rows"
                if th == 0.0
                else f"|rudder| > {th:g} deg"
            ),
            "n": len(g),
            "n_vessels": g["vessel_id"].nunique(),
            "prediction_up_pct":
                100.0 * float((delta > 0).mean()),
            "prediction_down_pct":
                100.0 * float((delta < 0).mean()),
            "mean_delta_t_per_10min": mean,
            "median_delta_t_per_10min": float(np.median(delta)),
            "row_level_sd_delta": sd,
            "row_level_Cohens_d":
                mean / sd if sd > 0 else np.nan,
            "vessel_cluster_mean_delta_CI95_low":
                vb["vessel_balanced_mean_delta_CI95_low"],
            "vessel_cluster_mean_delta_CI95_high":
                vb["vessel_balanced_mean_delta_CI95_high"],
            "assessment": (
                "positive response supported"
                if (
                    vb["vessel_balanced_mean_delta_CI95_low"] > 0
                    and (delta > 0).mean() > .5
                )
                else "non-confirmation"
            ),
        })

    atomic_csv(
        pd.DataFrame(out_rows),
        d / "Table_RUDDER_response_level_nonconfirmation.csv",
    )

    write_text(
        d / "RUDDER_METHOD_PATCH.txt",
        """Recommended manuscript replacement:

'The rudder perturbation was evaluated on the response scale rather than by
deriving statistical power from the marginal distribution of rudder angle.
For each supported observation, absolute rudder angle was increased by 1 degree
and the paired prediction change was recorded. Mean and median paired changes,
row-level standardised effect size, vessel-cluster bootstrap 95% confidence
intervals, and sensitivity across alternative absolute-rudder tail thresholds
were reported. A confidence interval spanning zero or a response direction that
changes across reasonable tail thresholds is interpreted as non-confirmation,
not as evidence that rudder-induced resistance is physically absent.'

Do NOT write 'the dataset lacks statistical power' unless a defensible
response-scale clustered MDE analysis is separately added.
""",
    )


# ---------------------------------------------------------------------
# 5) Time-resolution: both reviewer/original and recommended methods
# ---------------------------------------------------------------------

TIME_COL_CANDIDATES = {
    "vessel_id": [
        "vessel_id", "vessel", "ship_id", "mmsi",
    ],
    "ship_type": [
        "ship_type", "vessel_type", "type",
    ],
    "timestamp": [
        "timestamp", "time", "datetime", "utc_time",
    ],
    "target": [
        "target", "y_true", "observed_10min", "fuel_10min",
        "actual_10min", "target_10min",
    ],
    "pred5": [
        "pred_5min_agg", "prediction_5min_aggregate",
        "pred_5min_to_10min", "pred_5_to_10",
        "y_pred_5min_aggregate", "prediction_5min_agg",
    ],
    "pred10": [
        "pred_10min_direct", "prediction_10min_direct",
        "pred_direct_10min", "y_pred_10min",
        "prediction_10min",
    ],
    "speed_var": [
        "speed_std_within10", "speed_sd_within10",
        "within10_speed_sd", "speed_variability",
        "abs_speed_diff_5min", "speed_change_within10",
    ],
    "target_5a": ["target_5a", "y_true_5a", "fuel_5a"],
    "target_5b": ["target_5b", "y_true_5b", "fuel_5b"],
    "pred_5a": ["pred_5a", "y_pred_5a", "prediction_5a"],
    "pred_5b": ["pred_5b", "y_pred_5b", "prediction_5b"],
    "speed_5a": ["speed_5a", "sog_5a", "speed_first5"],
    "speed_5b": ["speed_5b", "sog_5b", "speed_second5"],
}


def resolve_col(cols, candidates, required=False):
    lower = {str(c).lower(): c for c in cols}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    if required:
        raise KeyError(
            f"Could not resolve any of columns: {candidates}"
        )
    return None


def discover_time_csv(main, explicit=None):
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(p)
        return p

    candidates = []
    for p in main.rglob("*.csv"):
        name = p.name.lower()
        if any(
            tok in name
            for tok in [
                "5min", "5_min", "temporal_resolution",
                "time_resolution", "resolution",
            ]
        ):
            candidates.append(p)

    for p in sorted(candidates, key=lambda x: len(str(x))):
        try:
            cols = pd.read_csv(p, nrows=0).columns
            pred5 = resolve_col(cols, TIME_COL_CANDIDATES["pred5"])
            pred10 = resolve_col(cols, TIME_COL_CANDIDATES["pred10"])
            target = resolve_col(cols, TIME_COL_CANDIDATES["target"])
            if pred5 and pred10 and target:
                return p
        except Exception:
            pass
    return None


def time_serial_by_vessel(z, residual_col, method, max_lag):
    rows = []
    acf_rows = []
    if "timestamp" not in z.columns:
        return pd.DataFrame(), pd.DataFrame()

    for vessel, g in z.groupby("vessel_id"):
        g = g.sort_values("timestamp")
        e = g[residual_col].to_numpy(float)
        if len(e) < 4:
            continue
        rows.append({
            "method": method,
            "vessel_id": vessel,
            "n": len(g),
            "durbin_watson": float(durbin_watson(e)),
        })
        av = acf(
            e,
            nlags=min(max_lag, len(e) - 1),
            fft=True,
            missing="drop",
        )
        for lag in range(1, len(av)):
            acf_rows.append({
                "method": method,
                "vessel_id": vessel,
                "lag": lag,
                "acf": float(av[lag]),
            })
    return pd.DataFrame(rows), pd.DataFrame(acf_rows)


def time_breusch_pagan(z, residual_col, pred_col, speed_var_col=None):
    cols = [pred_col]
    if speed_var_col:
        cols.append(speed_var_col)
    X = z[cols].astype(float).copy()

    if "ship_type" in z.columns:
        dd = pd.get_dummies(
            z["ship_type"].astype(str),
            prefix="ship",
            drop_first=True,
            dtype=float,
        )
        X = pd.concat([X.reset_index(drop=True), dd.reset_index(drop=True)], axis=1)

    X = sm.add_constant(X, has_constant="add")
    e = z[residual_col].to_numpy(float)
    lm, lmp, fval, fp = het_breuschpagan(e, X.to_numpy(float))
    return {
        "LM": float(lm),
        "LM_p_value": float(lmp),
        "F": float(fval),
        "F_p_value": float(fp),
        "auxiliary_variables": "+".join(X.columns.astype(str)),
    }


def run_time_resolution(main, args, out):
    d = ensure_dir(out / "05_time_resolution_dual_method")
    p = discover_time_csv(main, args.temporal_resolution_csv)

    if p is None:
        write_text(
            d / "REQUIRED_TIME_RESOLUTION_SCHEMA.txt",
            """No paired row-level 5-min versus 10-min prediction CSV was found.

Create/export ONE row per paired 10-min evaluation window with at least:
    vessel_id
    target_10min
    pred_5min_agg
    pred_10min_direct

Strongly recommended:
    timestamp
    ship_type
    speed_std_within10
or:
    speed_5a
    speed_5b

For direct error-cancellation diagnostics, also include if available:
    target_5a, pred_5a
    target_5b, pred_5b

Then rerun:
python run_reviewer_dual_method_diagnostics.py ... ^
  --steps time_resolution ^
  --temporal-resolution-csv "C:\\path\\to\\paired_5min_vs_10min_predictions.csv"

The script will NOT manufacture residual diagnostics from the aggregate Table 10
alone.
""",
        )
        print("[SKIP] time_resolution: no paired row-level CSV found.")
        return

    raw = pd.read_csv(p)
    cols = raw.columns

    vessel_c = resolve_col(
        cols, TIME_COL_CANDIDATES["vessel_id"], required=True
    )
    target_c = resolve_col(
        cols, TIME_COL_CANDIDATES["target"], required=True
    )
    p5_c = resolve_col(
        cols, TIME_COL_CANDIDATES["pred5"], required=True
    )
    p10_c = resolve_col(
        cols, TIME_COL_CANDIDATES["pred10"], required=True
    )
    ship_c = resolve_col(cols, TIME_COL_CANDIDATES["ship_type"])
    time_c = resolve_col(cols, TIME_COL_CANDIDATES["timestamp"])
    svar_c = resolve_col(cols, TIME_COL_CANDIDATES["speed_var"])
    s5a = resolve_col(cols, TIME_COL_CANDIDATES["speed_5a"])
    s5b = resolve_col(cols, TIME_COL_CANDIDATES["speed_5b"])

    z = pd.DataFrame({
        "vessel_id": raw[vessel_c].astype(str),
        "target": pd.to_numeric(raw[target_c], errors="coerce"),
        "pred_5min_agg": pd.to_numeric(raw[p5_c], errors="coerce"),
        "pred_10min_direct": pd.to_numeric(raw[p10_c], errors="coerce"),
    })
    if ship_c:
        z["ship_type"] = raw[ship_c].map(normalize_ship_type)
    if time_c:
        z["timestamp"] = pd.to_datetime(
            raw[time_c], errors="coerce", utc=True
        )

    if svar_c:
        z["speed_variability"] = pd.to_numeric(
            raw[svar_c], errors="coerce"
        )
    elif s5a and s5b:
        a = pd.to_numeric(raw[s5a], errors="coerce")
        b = pd.to_numeric(raw[s5b], errors="coerce")
        z["speed_variability"] = np.abs(a - b)

    z = z.dropna(
        subset=["target", "pred_5min_agg", "pred_10min_direct"]
    ).reset_index(drop=True)
    z["resid_5"] = z["pred_5min_agg"] - z["target"]
    z["resid_10"] = z["pred_10min_direct"] - z["target"]

    # A recommended paired inference
    by5 = per_vessel_metrics(
        z.rename(columns={"target": "target"})[
            ["vessel_id", "target"]
        ].assign(
            ship_type=z.get("ship_type", "unknown")
        ),
        z["pred_5min_agg"].to_numpy(float),
        "5min_agg",
    )
    by10 = per_vessel_metrics(
        z.rename(columns={"target": "target"})[
            ["vessel_id", "target"]
        ].assign(
            ship_type=z.get("ship_type", "unknown")
        ),
        z["pred_10min_direct"].to_numpy(float),
        "10min_direct",
    )

    a = cluster_bootstrap_pooled_rmse_difference(
        by5, by10, args.bootstrap, args.seed + 8000
    )
    a.update({
        "contrast": "direct10_minus_5min_aggregate",
        "method": "A_recommended_vessel_cluster_bootstrap",
    })
    atomic_csv(
        pd.DataFrame([a]),
        d / "Table_TIME_A_cluster_bootstrap_RMSE_difference.csv",
    )

    b = signflip_permutation_mean_difference(
        by5, by10, args.permutations, args.seed + 8100
    )
    b.update({
        "contrast": "direct10_minus_5min_aggregate",
        "method": "A2_paired_vessel_signflip",
    })
    atomic_csv(
        pd.DataFrame([b]),
        d / "Table_TIME_A2_signflip_RMSE_difference.csv",
    )

    # Overall metrics
    overall = []
    for name, pc in [
        ("5min_to_10min_aggregate", "pred_5min_agg"),
        ("direct_10min", "pred_10min_direct"),
    ]:
        m = metric_dict(z["target"], z[pc])
        overall.append({"method": name, **m})
    atomic_csv(
        pd.DataFrame(overall),
        d / "Table_TIME_0_overall_metrics.csv",
    )

    # B original requested residual diagnostics
    serial_frames = []
    acf_frames = []
    for name, resid in [
        ("5min_to_10min_aggregate", "resid_5"),
        ("direct_10min", "resid_10"),
    ]:
        s, aacf = time_serial_by_vessel(
            z, resid, name, args.acf_max_lag
        )
        if len(s):
            serial_frames.append(s)
        if len(aacf):
            acf_frames.append(aacf)

    if serial_frames:
        serial = pd.concat(serial_frames, ignore_index=True)
        atomic_csv(
            serial,
            d / "Table_TIME_B1_DurbinWatson_by_vessel.csv",
        )
    if acf_frames:
        aa = pd.concat(acf_frames, ignore_index=True)
        atomic_csv(
            aa,
            d / "Table_TIME_B2_ACF_by_vessel.csv",
        )

    bp_rows = []
    for name, resid, predc in [
        ("5min_to_10min_aggregate", "resid_5", "pred_5min_agg"),
        ("direct_10min", "resid_10", "pred_10min_direct"),
    ]:
        try:
            rr = time_breusch_pagan(
                z, resid, predc,
                "speed_variability"
                if "speed_variability" in z.columns
                else None,
            )
            rr["method"] = name
            bp_rows.append(rr)
        except Exception as exc:
            bp_rows.append({
                "method": name,
                "error": str(exc),
            })
    atomic_csv(
        pd.DataFrame(bp_rows),
        d / "Table_TIME_B3_BreuschPagan.csv",
    )

    # Within-window non-stationarity
    if "speed_variability" in z.columns:
        z["speed_variability_tertile"] = qcut_safe(
            z["speed_variability"],
            ["low", "moderate", "high"],
        ).astype(str)
        rows = []
        for level, g in z.groupby("speed_variability_tertile"):
            if len(g) < 50:
                continue
            m5 = metric_dict(g["target"], g["pred_5min_agg"])
            m10 = metric_dict(g["target"], g["pred_10min_direct"])
            rows.append({
                "speed_variability_tertile": level,
                "n": len(g),
                "median_speed_variability":
                    float(g["speed_variability"].median()),
                "RMSE_5min_aggregate": m5["RMSE"],
                "RMSE_direct_10min": m10["RMSE"],
                "direct_minus_aggregate_RMSE":
                    m10["RMSE"] - m5["RMSE"],
            })
        atomic_csv(
            pd.DataFrame(rows),
            d / "Table_TIME_B4_within_window_nonstationarity.csv",
        )

    # Direct two-half error cancellation if available
    t5a = resolve_col(cols, TIME_COL_CANDIDATES["target_5a"])
    t5b = resolve_col(cols, TIME_COL_CANDIDATES["target_5b"])
    p5a = resolve_col(cols, TIME_COL_CANDIDATES["pred_5a"])
    p5b = resolve_col(cols, TIME_COL_CANDIDATES["pred_5b"])
    if all([t5a, t5b, p5a, p5b]):
        e1 = (
            pd.to_numeric(raw[p5a], errors="coerce")
            - pd.to_numeric(raw[t5a], errors="coerce")
        )
        e2 = (
            pd.to_numeric(raw[p5b], errors="coerce")
            - pd.to_numeric(raw[t5b], errors="coerce")
        )
        good = e1.notna() & e2.notna()
        corr = float(np.corrcoef(e1[good], e2[good])[0, 1])
        atomic_csv(
            pd.DataFrame([{
                "n_windows": int(good.sum()),
                "correlation_first5_second5_residual": corr,
                "interpretation": (
                    "negative correlation supports partial error cancellation"
                    if corr < 0
                    else "no evidence of systematic negative error cancellation"
                ),
            }]),
            d / "Table_TIME_B5_two_half_error_cancellation.csv",
        )

    write_text(
        d / "TIME_RESOLUTION_INTERPRETATION.txt",
        """Do not pre-label the 5-min advantage as a 'de-noising effect'.

Use:
- paired vessel-cluster RMSE CI as the main inferential comparison;
- DW/ACF/BP as residual diagnostics;
- the speed-variability strata to test whether direct 10-min modelling is more
  sensitive to within-window non-stationarity;
- the two-half residual correlation, if available, to test whether aggregation
  benefits from partial error cancellation.

If these diagnostics do not support a clear mechanism, write:
'The 5-min approach showed a small paired advantage, but the mechanism could not
be uniquely attributed to de-noising or within-window non-stationarity.'
""",
    )


# ---------------------------------------------------------------------
# 6) Source-pathway: original-style comparisons + conservative framing
# ---------------------------------------------------------------------

def source_summary_by_vessel(perv):
    z = perv.copy()
    z["met_source"] = z["ship_type"].map(met_source_from_ship_type)
    return (
        z.groupby("met_source")
        .agg(
            vessels=("vessel_id", "nunique"),
            mean_RMSE=("RMSE", "mean"),
            median_RMSE=("RMSE", "median"),
            mean_R2=("R2", "mean"),
            median_R2=("R2", "median"),
            positive_R2_vessels=("R2", lambda s: int((s > 0).sum())),
        )
        .reset_index()
    )


def simple_source_permutation(
    perv, metric, reps, seed
):
    z = perv.copy()
    z["met_source"] = z["ship_type"].map(met_source_from_ship_type)
    A = z[z["met_source"] == "ERA5/container"][metric].to_numpy(float)
    B = z[z["met_source"] == "pre-matched/bulk+tanker"][metric].to_numpy(float)
    obs = float(np.mean(A) - np.mean(B))
    vals = np.concatenate([A, B])
    nA = len(A)
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(reps):
        perm = rng.permutation(vals)
        st = float(np.mean(perm[:nA]) - np.mean(perm[nA:]))
        exceed += int(abs(st) >= abs(obs))
    return {
        "metric": metric,
        "ERA5_mean": float(np.mean(A)),
        "prematched_mean": float(np.mean(B)),
        "ERA5_minus_prematched": obs,
        "two_sided_label_permutation_p":
            (exceed + 1) / (reps + 1),
        "causal_interpretation_allowed": False,
    }


def run_source_pathway(raw, X, locked, main, args, out):
    d = ensure_dir(out / "06_source_pathway")

    # L2 / L3 per-vessel
    L2 = per_vessel_metrics(
        raw.iloc[locked["temporal_te"]].reset_index(drop=True),
        locked["temporal_pred"],
        "L2",
    )
    L3 = per_vessel_metrics(
        raw.reset_index(drop=True),
        locked["lovo_pred"],
        "L3",
    )
    atomic_csv(
        source_summary_by_vessel(L2),
        d / "Table_SOURCE_A_L2_by_pathway.csv",
    )
    atomic_csv(
        source_summary_by_vessel(L3),
        d / "Table_SOURCE_A_L3_by_pathway.csv",
    )

    comp = []
    for j, (name, z) in enumerate([("L2", L2), ("L3", L3)]):
        for k, metric in enumerate(["RMSE", "R2"]):
            rr = simple_source_permutation(
                z, metric, args.permutations,
                args.seed + 9000 + 100 * j + k,
            )
            rr["validation"] = name
            rr["note"] = (
                "NON-CAUSAL: source pathway is structurally confounded with ship type"
            )
            comp.append(rr)
    atomic_csv(
        pd.DataFrame(comp),
        d / "Table_SOURCE_B_original_style_L2_L3_between_pathway_comparison.csv",
    )

    # Optional L4
    l4_path = Path(args.l4_by_vessel_csv) if args.l4_by_vessel_csv else None
    if l4_path is None:
        l4_path = find_one(
            main,
            [
                "*L4*by_vessel*.csv",
                "*adaptation*by_vessel*.csv",
                "*target*vessel*by_vessel*.csv",
            ],
            required=False,
        )
    if l4_path is not None and l4_path.exists():
        try:
            l4 = pd.read_csv(l4_path)
            # normalize common names
            rename = {}
            for c in l4.columns:
                cl = c.lower()
                if cl in {"r2", "r^2"}:
                    rename[c] = "R2"
                elif cl == "rmse":
                    rename[c] = "RMSE"
            l4 = l4.rename(columns=rename)
            if all(c in l4.columns for c in ["vessel_id", "ship_type", "RMSE", "R2"]):
                l4["ship_type"] = l4["ship_type"].map(normalize_ship_type)
                atomic_csv(
                    source_summary_by_vessel(l4),
                    d / "Table_SOURCE_A_L4_by_pathway.csv",
                )
        except Exception as exc:
            write_text(
                d / "L4_SOURCE_SUMMARY_ERROR.txt",
                str(exc),
            )

    # SHAP pathway summary
    shap_v_path = find_one(
        main,
        ["interventional_SHAP_values.npy"],
        required=False,
    )
    shap_rows_path = find_one(
        main,
        ["interventional_SHAP_sample_rows.csv"],
        required=False,
    )
    if shap_v_path and shap_rows_path:
        sv = np.load(shap_v_path)
        sr = pd.read_csv(shap_rows_path)
        sr["ship_type"] = sr["ship_type"].map(normalize_ship_type)
        if sv.shape[0] == len(sr) and sv.shape[1] == len(PRIMARY_FEATURES):
            shap_rows = []
            assoc_rows = []
            for src in ["ERA5/container", "pre-matched/bulk+tanker"]:
                m = (
                    sr["ship_type"].map(met_source_from_ship_type)
                    == src
                ).to_numpy()
                if m.sum() < 10:
                    continue
                imp = np.mean(np.abs(sv[m]), axis=0)
                imp = imp / imp.sum() * 100.0
                for feat, val in zip(PRIMARY_FEATURES, imp):
                    shap_rows.append({
                        "met_source": src,
                        "feature": feat,
                        "normalized_mean_abs_SHAP_pct": float(val),
                    })
                for feat in ["speed_kn", "wave_height_m", "rel_wind_speed_kn"]:
                    j = PRIMARY_FEATURES.index(feat)
                    rho = pd.Series(sr.loc[m, feat]).corr(
                        pd.Series(sv[m, j]), method="spearman"
                    )
                    assoc_rows.append({
                        "met_source": src,
                        "feature": feat,
                        "feature_SHAP_Spearman_rho": float(rho),
                    })
            atomic_csv(
                pd.DataFrame(shap_rows),
                d / "Table_SOURCE_A_SHAP_importance_by_pathway.csv",
            )
            atomic_csv(
                pd.DataFrame(assoc_rows),
                d / "Table_SOURCE_A_SHAP_direction_by_pathway.csv",
            )

    # H3 pathway summaries
    ft_path = find_one(
        main,
        ["Table_15_FT_L1_aligned_by_vessel*.csv"],
        required=False,
    )
    fd_path = find_one(
        main,
        ["Table_16_FD_fuel_time_decomposition_by_vessel*.csv"],
        required=False,
    )
    if ft_path and fd_path:
        ft = pd.read_csv(ft_path)
        fd = pd.read_csv(fd_path)
        ft["ship_type"] = ft["ship_type"].map(normalize_ship_type)
        fd["ship_type"] = fd["ship_type"].map(normalize_ship_type)
        ft["met_source"] = ft["ship_type"].map(met_source_from_ship_type)
        fd["met_source"] = fd["ship_type"].map(met_source_from_ship_type)

        rows = []
        for r in sorted(ft["reduction_pct"].unique()):
            for src in ["ERA5/container", "pre-matched/bulk+tanker"]:
                gf = ft[
                    (ft["reduction_pct"] == r)
                    & (ft["met_source"] == src)
                ]
                gd = fd[
                    (fd["reduction_pct"] == r)
                    & (fd["met_source"] == src)
                ]
                rows.append({
                    "reduction_pct": r,
                    "met_source": src,
                    "n_vessels": gf["vessel_id"].nunique(),
                    "FT_weighted_CII_change_pct":
                        float(ft_weighted_cii_change(gf)),
                    "FD_total_fuel_change_pct":
                        float(fd_total_fuel_change(gd)),
                    "note": (
                        "Descriptive only: source pathway and ship type are confounded."
                    ),
                })
        atomic_csv(
            pd.DataFrame(rows),
            d / "Table_SOURCE_A_H3_by_pathway.csv",
        )

    write_text(
        d / "SOURCE_CONFOUNDING_STATEMENT.txt",
        """Recommended exact manuscript statement:

'Environmental-data source pathway and ship type are structurally confounded in
the present dataset: all containership observations use ERA5-derived weather
fields, whereas bulk-carrier and tanker observations use pre-matched
environmental records. Consequently, source-stratified differences in L2/L3
transfer, SHAP behaviour, or H3 scenario outcomes cannot be causally attributed
to source quality independently of vessel type. These comparisons are reported
as descriptive robustness diagnostics only.'

Important:
The original-style between-pathway permutation tables are supplied only for
comparison with the reviewer's request. They MUST NOT be described as causal
source effects.
""",
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    args = parse_args()
    main_out = Path(args.main_output)
    out = Path(args.out_dir) if args.out_dir else (
        main_out / "20_reviewer_dual_method_diagnostics"
    )
    ensure_dir(out)

    steps = (
        [
            "hypothesis_inference",
            "lovo_bulk",
            "wave_c4",
            "rudder",
            "time_resolution",
            "source_pathway",
        ]
        if args.steps.strip().lower() == "all"
        else [x.strip() for x in args.steps.split(",") if x.strip()]
    )

    write_text(
        out / "RUN_CONFIGURATION.json",
        json.dumps(
            {
                "version": VERSION,
                "steps": steps,
                "bootstrap": args.bootstrap,
                "permutations": args.permutations,
                "seed": args.seed,
                "main_output": str(main_out),
                "raw_cruise": str(args.raw_cruise),
            },
            indent=2,
        ),
    )

    print(f"Version: {VERSION}")
    print(f"Output: {out}")

    core, core_path, overrides = load_core(args)
    print(f"Core: {core_path}")
    print(f"Overrides: {overrides}")

    raw, X = load_locked_data(core, args, overrides)
    artifacts = resolve_locked_artifacts(main_out)
    locked = validate_predictions(raw, X, core, artifacts)

    print(
        "[PASS] Locked cohort/model alignment | "
        f"cruise={len(raw):,} | vessels={raw.vessel_id.nunique()} | "
        f"L1 train/test={len(locked['record_tr']):,}/{len(locked['record_te']):,}"
    )

    if "hypothesis_inference" in steps:
        print("\n[1] H1/H3 dual-method inference...")
        h1_inference(raw, locked, args, out)
        h3_inference(main_out, args, out)

    if "lovo_bulk" in steps:
        print("\n[2] LOVO bulk diagnostics...")
        run_lovo_bulk(raw, locked, main_out, args, out)

    if "wave_c4" in steps:
        print("\n[3] Wave C4 post-hoc stratification...")
        run_wave_c4(raw, X, core, locked, args, out)

    if "rudder" in steps:
        print("\n[4] Rudder response-level non-confirmation...")
        run_rudder(raw, X, core, locked, main_out, args, out)

    if "time_resolution" in steps:
        print("\n[5] Time-resolution dual-method diagnostics...")
        run_time_resolution(main_out, args, out)

    if "source_pathway" in steps:
        print("\n[6] Source-pathway descriptive diagnostics...")
        run_source_pathway(raw, X, locked, main_out, args, out)

    write_text(
        out / "README_METHOD_SELECTION.txt",
        """METHOD SELECTION FOR ADVISOR DISCUSSION

Recommended/main:
- H1: vessel-cluster bootstrap pooled RMSE difference
- H3: vessel-bootstrap endpoint CIs + ship-type contrast CIs
- LOVO bulk: per-vessel normalized W1 + 1D overlap; 3D KDE supplementary
- Wave C4: full supported L1 post-hoc subgroup diagnosis
- Rudder: paired response distribution + vessel-cluster CI + tail sensitivity
- Time resolution: paired vessel-cluster RMSE inference + residual diagnostics
- Source pathway: descriptive only with explicit structural-confounding statement

Original-style comparison supplied:
- H1 paired vessel sign-flip permutation
- H3 ship-type × reduction cluster-robust interaction F test
- LOVO 3D KDE overlap and bulk-vs-nonbulk permutation comparison
- Time-resolution DW/ACF/BP + non-stationarity/error-cancellation diagnostics
- Source-pathway between-group permutation comparison (NON-CAUSAL)

Do not select a method by whichever produces the smallest p-value. Select based
on the scientific estimand and dependence structure.
""",
    )

    print("\nDONE")
    print(f"All available outputs written to: {out}")


if __name__ == "__main__":
    main()
