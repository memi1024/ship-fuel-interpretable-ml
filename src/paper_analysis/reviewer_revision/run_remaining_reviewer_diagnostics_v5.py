#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_remaining_reviewer_diagnostics_v5.py

One-pass script for the remaining reviewer diagnostics after the first dual-method run.

It does NOT retrain the locked F31 XGBoost model and does NOT redefine the L1/L2/L3
cohorts.  It reuses locked artifacts and existing 5-min/10-min V3 predictions.

Modules
-------
1. source_shap_fixed
   Fixes the pandas-index alignment bug in source-stratified SHAP-direction correlations.
   Uses scipy.stats.spearmanr on NumPy arrays and verifies the overall rho values against
   the locked full-sample values.

2. lovo_target_variance
   Adds per-vessel target SD/IQR/CV and RMSE-to-target-SD diagnostics to explain why
   negative R2 can coexist with moderate absolute RMSE.

3. rudder_dual_estimand
   Reuses the original locked 20,000-row rudder perturbation file if available.
   Reports BOTH:
      A. pooled row-weighted mean delta + vessel-cluster bootstrap CI
         (same estimand as the original revision diagnostic)
      B. vessel-balanced mean delta + vessel bootstrap CI
   This removes the point-estimate/CI estimand mismatch.

4. time_resolution_from_v3
   Reuses the existing V3 equal-budget 5-min vs 10-min experiment:
      revision_runs/time_resolution_equal_budget
   No retraining.
   Builds strict paired 10-min test windows and produces:
      - 2,000-rep paired vessel-cluster bootstrap RMSE difference
      - paired vessel sign-flip permutation sensitivity
      - DW / ACF / Koenker-BP residual diagnostics
      - within-window speed-variability stratification
      - first-half vs second-half 5-min residual error-cancellation diagnostic
   It reports all / cruise / maneuver scopes separately so that the manuscript Table 10
   scope can be identified without silently changing the estimand.

5. wave_scope_note
   Writes an explicit reconciliation note:
      primary locked audit = 20,000-row audit sample (61.67% expected direction);
      post-hoc full-support diagnostic = 97,793 supported L1 rows (~61.45%).
   The full-support diagnostic diagnoses failure domains and does not replace Table 12.

Recommended command
-------------------
python "src/paper_analysis/reviewer_revision/run_remaining_reviewer_diagnostics_v5.py" ^
  --work-dir "src/paper_analysis" ^
  --main-output "results" ^
  --raw-cruise "data/final_fixed31_cruise.csv" ^
  --reviewer-output "revision_runs/reviewer_dual_method_diagnostics" ^
  --time-root "revision_runs/time_resolution_equal_budget" ^
  --bootstrap 2000
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import acf


VERSION = "2026-08-19.remaining-reviewer-v5-conditional-load"

LOCKED_RHO = {
    "speed_kn": 0.9783,
    "wave_height_m": 0.8306,
    "rel_wind_speed_kn": 0.9106,
}
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


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--work-dir", default="src/paper_analysis")
    p.add_argument(
        "--main-output",
        default="results",
    )
    p.add_argument(
        "--raw-cruise",
        default="data/final_fixed31_cruise.csv",
    )
    p.add_argument(
        "--reviewer-output",
        default="revision_runs/reviewer_dual_method_diagnostics",
    )
    p.add_argument(
        "--time-root",
        default="revision_runs/time_resolution_equal_budget",
    )
    p.add_argument("--base-reviewer-script", default=None)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--permutations", type=int, default=100000)
    p.add_argument("--seed", type=int, default=20260819)
    p.add_argument("--csv-chunksize", type=int, default=25000)
    p.add_argument("--trajectory-gap-minutes", type=float, default=30.0)
    p.add_argument("--acf-max-lag", type=int, default=12)
    p.add_argument("--time-csv-chunksize", type=int, default=100000)
    p.add_argument(
        "--steps",
        default=(
            "source_shap_fixed,lovo_target_variance,"
            "rudder_dual_estimand,time_resolution_from_v3,wave_scope_note"
        ),
    )
    return p.parse_args()


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def atomic_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8-sig")
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


def source_label(ship_type):
    st = normalize_ship_type(ship_type)
    return "ERA5/container" if st == "container" else "pre-matched/bulk+tanker"


def load_base_module(args):
    work = Path(args.work_dir)
    if args.base_reviewer_script:
        path = Path(args.base_reviewer_script)
    else:
        candidates = [
            work / "reviewer_revision" / "run_reviewer_dual_method_diagnostics_v2.py",
            work / "reviewer_revision" / "run_reviewer_dual_method_diagnostics.py",
            work / "run_reviewer_dual_method_diagnostics_v2.py",
            work / "run_reviewer_dual_method_diagnostics.py",
        ]
        path = next((p for p in candidates if p.exists()), None)
        if path is None:
            raise FileNotFoundError(
                "Could not find run_reviewer_dual_method_diagnostics_v2.py "
                f"under {work}"
            )

    spec = importlib.util.spec_from_file_location(
        "reviewer_dual_base_v2", path
    )
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    print(f"[BASE] {path}")
    return mod


def load_locked(base, args):
    ns = SimpleNamespace(
        work_dir=args.work_dir,
        main_output=args.main_output,
        raw_cruise=args.raw_cruise,
        core_path=None,
        column_overrides=None,
        trajectory_gap_minutes=args.trajectory_gap_minutes,
        csv_chunksize=args.csv_chunksize,
    )
    core, core_path, overrides = base.load_core(ns)
    raw, X = base.load_locked_data(core, ns, overrides)
    artifacts = base.resolve_locked_artifacts(Path(args.main_output))
    locked = base.validate_predictions(raw, X, core, artifacts)
    print(
        "[PASS] locked F31 alignment | "
        f"cruise={len(raw):,} | vessels={raw['vessel_id'].nunique()} | "
        f"L1={len(locked['record_te']):,}"
    )
    return core, raw, X, locked


# ---------------------------------------------------------------------
# 1. SOURCE-SPECIFIC SHAP — FIXED
# ---------------------------------------------------------------------

def find_first(root: Path, patterns, prefer=()):
    hits = []
    for pat in patterns:
        hits.extend(root.rglob(pat))
    hits = list({str(p.resolve()): p for p in hits}.values())
    if not hits:
        return None

    scored = []
    for p in hits:
        s = str(p).lower()
        score = sum(token.lower() in s for token in prefer)
        scored.append((score, -len(str(p)), p))
    scored.sort(key=lambda z: (-z[0], z[1], str(z[2])))
    return scored[0][2]


def run_source_shap_fixed(main: Path, out: Path):
    d = ensure_dir(out / "07_remaining" / "01_source_SHAP_fixed")

    shap_path = find_first(
        main,
        ["interventional_SHAP_values.npy"],
        prefer=("15_revision_diagnostics", "07_interventional_shap"),
    )
    rows_path = find_first(
        main,
        ["interventional_SHAP_sample_rows.csv"],
        prefer=("15_revision_diagnostics", "07_interventional_shap"),
    )
    if shap_path is None or rows_path is None:
        raise FileNotFoundError(
            "Could not locate interventional_SHAP_values.npy and "
            "interventional_SHAP_sample_rows.csv"
        )

    sv = np.asarray(np.load(shap_path), dtype=float)
    sr = pd.read_csv(rows_path, low_memory=False)

    if sv.ndim != 2 or sv.shape[1] != len(PRIMARY_FEATURES):
        raise AssertionError(
            f"Unexpected SHAP shape {sv.shape}; expected p={len(PRIMARY_FEATURES)}"
        )
    if len(sr) != sv.shape[0]:
        raise AssertionError(
            f"SHAP rows mismatch: values={sv.shape[0]}, metadata={len(sr)}"
        )

    sr["ship_type"] = sr["ship_type"].map(normalize_ship_type)
    sr["met_source"] = sr["ship_type"].map(source_label)

    direction_rows = []
    groups = [("Overall", np.ones(len(sr), dtype=bool))]
    groups += [
        (src, sr["met_source"].eq(src).to_numpy())
        for src in ["ERA5/container", "pre-matched/bulk+tanker"]
    ]

    for group_name, mask in groups:
        for feat in ["speed_kn", "wave_height_m", "rel_wind_speed_kn"]:
            j = PRIMARY_FEATURES.index(feat)
            x = pd.to_numeric(sr.loc[mask, feat], errors="coerce").to_numpy(float)
            y = sv[mask, j].astype(float)
            good = np.isfinite(x) & np.isfinite(y)
            rho, p = spearmanr(x[good], y[good])
            direction_rows.append({
                "scope": group_name,
                "feature": feat,
                "n": int(good.sum()),
                "feature_SHAP_Spearman_rho": float(rho),
                "p_value_iid_descriptive_only": float(p),
                "note": (
                    "rho computed positionally with scipy.spearmanr; "
                    "p-value is descriptive only because rows are clustered/serial"
                ),
            })

    direction = pd.DataFrame(direction_rows)
    atomic_csv(
        direction,
        d / "Table_SOURCE_SHAP_direction_FIXED.csv",
    )

    # Hard validation against the locked overall values.
    for feat, expected in LOCKED_RHO.items():
        got = float(
            direction.loc[
                (direction["scope"] == "Overall")
                & (direction["feature"] == feat),
                "feature_SHAP_Spearman_rho",
            ].iloc[0]
        )
        if abs(got - expected) > 0.01:
            raise AssertionError(
                f"Overall SHAP rho check failed for {feat}: "
                f"{got:.4f} vs locked {expected:.4f}"
            )

    importance_rows = []
    for group_name, mask in groups:
        imp = np.mean(np.abs(sv[mask]), axis=0)
        denom = float(imp.sum())
        pct = 100.0 * imp / denom if denom > 0 else np.full_like(imp, np.nan)
        for feat, raw_imp, v in zip(PRIMARY_FEATURES, imp, pct):
            importance_rows.append({
                "scope": group_name,
                "feature": feat,
                "mean_abs_SHAP": float(raw_imp),
                "normalized_mean_abs_SHAP_pct": float(v),
            })
    atomic_csv(
        pd.DataFrame(importance_rows),
        d / "Table_SOURCE_SHAP_importance_FIXED.csv",
    )

    write_text(
        d / "SOURCE_SHAP_NOTE.txt",
        """The previous source-specific SHAP direction table must be discarded because
Pandas Series index alignment produced invalid subgroup correlations. This rerun uses
NumPy arrays + scipy.stats.spearmanr and first verifies that the Overall values recover
the locked interventional SHAP correlations.

All source-stratified SHAP results remain descriptive because environmental source
pathway is structurally confounded with ship type.
""",
    )
    print("[OK] source-specific SHAP direction fixed")


# ---------------------------------------------------------------------
# 2. LOVO TARGET-VARIANCE DIAGNOSTIC
# ---------------------------------------------------------------------

def run_lovo_target_variance(raw, locked, out: Path):
    d = ensure_dir(out / "07_remaining" / "02_LOVO_target_variance")
    pred = np.asarray(locked["lovo_pred"], dtype=float)

    rows = []
    r = raw.reset_index(drop=True)
    for vessel, idx in r.groupby("vessel_id").groups.items():
        ii = np.asarray(list(idx), dtype=int)
        y = r.loc[ii, "target"].to_numpy(float)
        p = pred[ii]
        rmse = float(np.sqrt(mean_squared_error(y, p)))
        r2 = float(r2_score(y, p))
        mean = float(np.mean(y))
        sd = float(np.std(y, ddof=1))
        q25, q75 = np.quantile(y, [0.25, 0.75])
        iqr = float(q75 - q25)
        rows.append({
            "vessel_id": str(vessel),
            "ship_type": normalize_ship_type(r.loc[ii[0], "ship_type"]),
            "n": len(ii),
            "LOVO_RMSE": rmse,
            "LOVO_R2": r2,
            "target_mean": mean,
            "target_SD": sd,
            "target_IQR": iqr,
            "target_CV_abs": sd / abs(mean) if abs(mean) > 1e-12 else np.nan,
            "RMSE_over_target_SD": rmse / sd if sd > 1e-12 else np.nan,
            "RMSE_over_target_IQR": rmse / iqr if iqr > 1e-12 else np.nan,
            "target_min": float(np.min(y)),
            "target_max": float(np.max(y)),
        })

    tab = pd.DataFrame(rows)
    atomic_csv(
        tab,
        d / "Table_LOVO_target_variance_by_vessel.csv",
    )

    summary = (
        tab.groupby("ship_type")
        .agg(
            vessels=("vessel_id", "nunique"),
            positive_R2=("LOVO_R2", lambda s: int((s > 0).sum())),
            median_R2=("LOVO_R2", "median"),
            median_RMSE=("LOVO_RMSE", "median"),
            median_target_SD=("target_SD", "median"),
            median_target_IQR=("target_IQR", "median"),
            median_target_CV=("target_CV_abs", "median"),
            median_RMSE_over_target_SD=("RMSE_over_target_SD", "median"),
        )
        .reset_index()
    )
    atomic_csv(
        summary,
        d / "Table_LOVO_target_variance_by_shiptype.csv",
    )
    print("[OK] LOVO target-variance diagnostic")


# ---------------------------------------------------------------------
# 3. RUDDER — TWO ESTIMANDS, ALIGNED POINT + CI
# ---------------------------------------------------------------------

def pooled_cluster_boot_mean(df, reps, seed):
    vessels = np.unique(df["vessel_id"].astype(str).to_numpy())
    by = {
        v: df.loc[df["vessel_id"].astype(str).eq(v), "delta"].to_numpy(float)
        for v in vessels
    }
    rng = np.random.default_rng(seed)
    vals = np.empty(reps, dtype=float)
    for b in range(reps):
        draw = rng.choice(vessels, size=len(vessels), replace=True)
        pooled = np.concatenate([by[v] for v in draw])
        vals[b] = float(np.mean(pooled))
    return (
        float(df["delta"].mean()),
        float(np.quantile(vals, .025)),
        float(np.quantile(vals, .975)),
    )


def vessel_balanced_boot_mean(df, reps, seed):
    vm = (
        df.groupby("vessel_id", as_index=False)["delta"]
        .mean()["delta"]
        .to_numpy(float)
    )
    rng = np.random.default_rng(seed)
    vals = np.empty(reps, dtype=float)
    for b in range(reps):
        draw = rng.choice(vm, size=len(vm), replace=True)
        vals[b] = float(np.mean(draw))
    return (
        float(np.mean(vm)),
        float(np.quantile(vals, .025)),
        float(np.quantile(vals, .975)),
    )


def run_rudder_dual(raw, X, locked, main: Path, out: Path, args):
    d = ensure_dir(out / "07_remaining" / "03_rudder_dual_estimand")
    row_path = find_first(
        main,
        ["rudder_plus1_row_level.csv.gz"],
        prefer=("15_revision_diagnostics", "08_rudder_power"),
    )

    if row_path is not None:
        rows = pd.read_csv(row_path)
        source = str(row_path)
    else:
        # Exact fallback consistent with run_f31_revision_diagnostics:
        # seed=20260808 + 822, n=20,000.
        record_te = np.asarray(locked["record_te"], dtype=int)
        rng = np.random.default_rng(20260808 + 822)
        n = min(20000, len(record_te))
        local = np.sort(rng.choice(len(record_te), size=n, replace=False))
        global_idx = record_te[local]

        base_raw = raw.iloc[global_idx].reset_index(drop=True).copy()
        Xb = X.iloc[global_idx].reset_index(drop=True).copy()
        Xp = Xb.copy()
        rud = Xp["rudder_deg"].to_numpy(float)
        sign = np.sign(rud)
        sign[sign == 0] = 1.0
        Xp["rudder_deg"] = sign * (np.abs(rud) + 1.0)

        pb = np.asarray(locked["model"].predict(Xb), dtype=float)
        pp = np.asarray(locked["model"].predict(Xp), dtype=float)

        rows = base_raw[
            ["row_id", "vessel_id", "ship_type", "timestamp",
             "rudder_deg", "speed_kn", "target"]
        ].copy()
        rows["abs_rudder_original"] = np.abs(rows["rudder_deg"].to_numpy(float))
        rows["prediction_base"] = pb
        rows["prediction_abs_rudder_plus1deg"] = pp
        rows["delta"] = pp - pb
        source = "regenerated_exact_original_seed_20260808_plus_822"

    rows["vessel_id"] = rows["vessel_id"].astype(str)
    thresholds = [0.0, 0.5, 1.0, 2.0]
    out_rows = []

    for i, th in enumerate(thresholds):
        g = rows.copy() if th == 0 else rows.loc[
            rows["abs_rudder_original"] > th
        ].copy()
        if len(g) < 3:
            continue

        delta = g["delta"].to_numpy(float)
        sd = float(np.std(delta, ddof=1))
        pooled, plo, phi = pooled_cluster_boot_mean(
            g, args.bootstrap, args.seed + 12000 + i
        )
        vb, vlo, vhi = vessel_balanced_boot_mean(
            g, args.bootstrap, args.seed + 12100 + i
        )

        out_rows.append({
            "tail": "all" if th == 0 else f"|rudder|>{th:g}deg",
            "n_rows": len(g),
            "n_vessels": g["vessel_id"].nunique(),
            "prediction_up_pct": 100 * float((delta > 1e-8).mean()),
            "prediction_down_pct": 100 * float((delta < -1e-8).mean()),
            "median_delta": float(np.median(delta)),
            "row_level_Cohens_d": pooled / sd if sd > 0 else np.nan,

            "A_pooled_row_weighted_mean_delta": pooled,
            "A_vessel_cluster_CI95_low": plo,
            "A_vessel_cluster_CI95_high": phi,

            "B_vessel_balanced_mean_delta": vb,
            "B_vessel_bootstrap_CI95_low": vlo,
            "B_vessel_bootstrap_CI95_high": vhi,

            "source_row_level_file": source,
        })

    tab = pd.DataFrame(out_rows)
    atomic_csv(
        tab,
        d / "Table_RUDDER_dual_estimand_aligned.csv",
    )

    # Sanity check against the locked primary threshold point estimate.
    primary = tab.loc[tab["tail"] == "|rudder|>1deg"]
    if len(primary):
        got = float(primary["A_pooled_row_weighted_mean_delta"].iloc[0])
        if abs(got - (-0.000328)) > 0.00015:
            warnings.warn(
                f"Primary rudder pooled mean {got:.6f} differs from locked ~-0.000328"
            )

    write_text(
        d / "RUDDER_ESTIMAND_GUIDE.txt",
        """Recommended main-text estimand:
A = pooled row-weighted mean paired prediction change, with vessels resampled as
clusters. This matches the original revision diagnostic construction.

B = equal-vessel-weight mean paired prediction change, with vessel bootstrap.
Report B only as a sensitivity result if useful.

Never pair the A point estimate with the B confidence interval.
""",
    )
    print("[OK] rudder dual-estimand table")


# ---------------------------------------------------------------------
# 4. 5-MIN vs 10-MIN — EXISTING V3 PREDICTIONS, NO RETRAINING
# ---------------------------------------------------------------------

def require(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _read_filtered_chunks(
    path: Path,
    usecols,
    key_col: str,
    keep_ids: set[str],
    chunksize: int = 100_000,
) -> pd.DataFrame:
    """
    Memory-safe reader for large V3 paired-data CSVs.

    Only rows whose key_col is in keep_ids are retained in memory.
    This is important because 05_paired_5min_model_ready.csv may be much larger
    than the final test-window prediction table.
    """
    blocks = []
    total = 0
    kept = 0

    for i, chunk in enumerate(
        pd.read_csv(
            path,
            usecols=usecols,
            chunksize=chunksize,
            low_memory=True,
        ),
        start=1,
    ):
        total += len(chunk)
        key = chunk[key_col].astype(str)
        mask = key.isin(keep_ids)
        if mask.any():
            sub = chunk.loc[mask].copy()
            blocks.append(sub)
            kept += len(sub)

        if i % 10 == 0:
            print(
                f"[MEMSAFE] {path.name}: read {total:,} rows; "
                f"kept {kept:,} rows for test windows"
            )

        del chunk

    if not blocks:
        raise RuntimeError(
            f"No rows from {path} matched the paired test-window IDs."
        )

    out = pd.concat(blocks, ignore_index=True)
    print(
        f"[MEMSAFE] {path.name}: finished {total:,} rows; "
        f"retained {len(out):,}"
    )
    return out


def load_v3_paired(time_root: Path, chunksize: int = 100_000):
    paired_path = require(
        time_root / "05_predictions" / "02_paired_10min_test_predictions.csv"
    )
    native5_path = require(
        time_root / "05_predictions" / "01_native_5min_test_predictions.csv"
    )
    five_ready_path = require(
        time_root / "00_data" / "05_paired_5min_model_ready.csv"
    )
    map_path = require(
        time_root / "00_data" / "07_five_to_ten_minute_window_map.csv"
    )

    # The paired prediction file contains only the final paired TEST windows and
    # is therefore safe to read normally.  Use its IDs to filter the much larger
    # model-ready/map CSVs during streaming reads.
    paired = pd.read_csv(paired_path, low_memory=True)
    paired["ten_minute_window_id"] = (
        paired["ten_minute_window_id"].astype(str)
    )
    test_ids = set(paired["ten_minute_window_id"].tolist())

    native = pd.read_csv(native5_path, low_memory=True)
    native["ten_minute_window_id"] = (
        native["ten_minute_window_id"].astype(str)
    )
    native = native.loc[
        native["ten_minute_window_id"].isin(test_ids)
    ].copy()

    five = _read_filtered_chunks(
        five_ready_path,
        usecols=[
            "ten_minute_window_id",
            "speed_kn",
            "five_minute_position",
        ],
        key_col="ten_minute_window_id",
        keep_ids=test_ids,
        chunksize=chunksize,
    )
    five["ten_minute_window_id"] = (
        five["ten_minute_window_id"].astype(str)
    )

    mapping = _read_filtered_chunks(
        map_path,
        usecols=[
            "ten_minute_window_id",
            "pseudo_ship_group_id",
            "trajectory_segment_id",
            "ship_type",
            "window_timestamp_utc",
            "five_minute_position",
        ],
        key_col="ten_minute_window_id",
        keep_ids=test_ids,
        chunksize=chunksize,
    )
    mapping["ten_minute_window_id"] = (
        mapping["ten_minute_window_id"].astype(str)
    )

    identity = (
        mapping.sort_values(
            ["ten_minute_window_id", "five_minute_position"]
        )
        .groupby("ten_minute_window_id", as_index=False)
        .agg(
            vessel_id=("pseudo_ship_group_id", "first"),
            trajectory_segment_id=("trajectory_segment_id", "first"),
            mapped_ship_type=("ship_type", "first"),
            map_rows=("five_minute_position", "size"),
        )
    )

    speed = (
        five.sort_values(
            ["ten_minute_window_id", "five_minute_position"]
        )
        .groupby("ten_minute_window_id", as_index=False)
        .agg(
            speed_5a=("speed_kn", "first"),
            speed_5b=("speed_kn", "last"),
            speed_mean_within10=("speed_kn", "mean"),
            speed_std_within10=(
                "speed_kn",
                lambda s: float(np.std(s.to_numpy(float), ddof=0)),
            ),
            five_ready_rows=("speed_kn", "size"),
        )
    )
    speed["abs_speed_diff_5min"] = (
        speed["speed_5b"] - speed["speed_5a"]
    ).abs()

    needed = {
        "ten_minute_window_id",
        "five_minute_position",
        "fuel_t_5min",
        "predicted_fuel_t_5min",
    }
    missing = needed - set(native.columns)
    if missing:
        raise KeyError(f"Native 5-min predictions missing {sorted(missing)}")

    native = native.sort_values(
        ["ten_minute_window_id", "five_minute_position"]
    ).copy()
    native["_half"] = native.groupby("ten_minute_window_id").cumcount() + 1

    actual = native.pivot(
        index="ten_minute_window_id",
        columns="_half",
        values="fuel_t_5min",
    ).rename(columns={1: "target_5a", 2: "target_5b"})
    pred = native.pivot(
        index="ten_minute_window_id",
        columns="_half",
        values="predicted_fuel_t_5min",
    ).rename(columns={1: "pred_5a", 2: "pred_5b"})
    halves = actual.join(pred, how="outer").reset_index()

    out = (
        paired.merge(identity, on="ten_minute_window_id", how="left", validate="one_to_one")
        .merge(speed, on="ten_minute_window_id", how="left", validate="one_to_one")
        .merge(halves, on="ten_minute_window_id", how="left", validate="one_to_one")
    )

    rename = {
        "window_timestamp_utc": "timestamp",
        "actual_fuel_t_10min": "target_10min",
        "predicted_t_10min_from_5min_model": "pred_5min_agg",
        "predicted_t_10min_from_10min_model": "pred_10min_direct",
    }
    out = out.rename(columns=rename)

    if "ship_type" not in out.columns and "mapped_ship_type" in out.columns:
        out["ship_type"] = out["mapped_ship_type"]
    out["ship_type"] = out["ship_type"].map(normalize_ship_type)
    out["voyage_phase"] = out["voyage_phase"].astype(str).str.lower()

    required_cols = [
        "vessel_id",
        "target_10min",
        "pred_5min_agg",
        "pred_10min_direct",
        "target_5a",
        "pred_5a",
        "target_5b",
        "pred_5b",
    ]
    if out[required_cols].isna().any().any():
        bad = out[required_cols].isna().sum()
        raise AssertionError(f"Missing paired V3 data:\n{bad[bad > 0]}")

    if not (out["map_rows"] == 2).all():
        raise AssertionError("Not all test windows map to two 5-min rows")
    if not (out["five_ready_rows"] == 2).all():
        raise AssertionError("Not all model-ready test windows contain two 5-min rows")

    target_diff = np.abs(
        out["target_5a"] + out["target_5b"] - out["target_10min"]
    )
    if float(target_diff.max()) > 1e-8:
        raise AssertionError(
            f"5-min target sum mismatch, max={target_diff.max():.3e}"
        )

    return out


def metrics(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    rmse = float(np.sqrt(mean_squared_error(y, p)))
    return {
        "n": len(y),
        "RMSE": rmse,
        "R2": float(r2_score(y, p)),
        "SSE": float(np.sum((p - y) ** 2)),
    }


def per_vessel_sse(df, pred_col):
    rows = []
    for vessel, g in df.groupby("vessel_id"):
        y = g["target_10min"].to_numpy(float)
        p = g[pred_col].to_numpy(float)
        m = metrics(y, p)
        rows.append({
            "vessel_id": str(vessel),
            "ship_type": normalize_ship_type(g["ship_type"].iloc[0]),
            **m,
        })
    return pd.DataFrame(rows)


def paired_cluster_boot_rmse(df, reps, seed):
    vessels = sorted(df["vessel_id"].astype(str).unique())
    by = {
        v: df.loc[df["vessel_id"].astype(str).eq(v)].copy()
        for v in vessels
    }

    def pooled_rmse(block, col):
        y = block["target_10min"].to_numpy(float)
        p = block[col].to_numpy(float)
        return float(np.sqrt(np.mean((p - y) ** 2)))

    point5 = pooled_rmse(df, "pred_5min_agg")
    point10 = pooled_rmse(df, "pred_10min_direct")
    point = point10 - point5

    rng = np.random.default_rng(seed)
    vals = np.empty(reps)
    for b in range(reps):
        draw = rng.choice(vessels, size=len(vessels), replace=True)
        z = pd.concat([by[v] for v in draw], ignore_index=True)
        vals[b] = (
            pooled_rmse(z, "pred_10min_direct")
            - pooled_rmse(z, "pred_5min_agg")
        )
    return {
        "n_vessels": len(vessels),
        "RMSE_5min_aggregated": point5,
        "RMSE_direct_10min": point10,
        "delta_RMSE_direct10_minus_5agg": point,
        "cluster_boot_CI95_low": float(np.quantile(vals, .025)),
        "cluster_boot_CI95_high": float(np.quantile(vals, .975)),
        "cluster_boot_prob_delta_gt_0": float(np.mean(vals > 0)),
        "bootstrap_reps": reps,
    }


def signflip_permutation_rmse(df, reps, seed):
    a = per_vessel_sse(df, "pred_5min_agg").set_index("vessel_id")
    b = per_vessel_sse(df, "pred_10min_direct").set_index("vessel_id")
    common = sorted(set(a.index) & set(b.index))
    diff = (
        b.loc[common, "RMSE"].to_numpy(float)
        - a.loc[common, "RMSE"].to_numpy(float)
    )
    observed = float(diff.mean())
    rng = np.random.default_rng(seed)
    exceed = 0
    done = 0
    chunk = 10000
    while done < reps:
        n = min(chunk, reps - done)
        signs = rng.choice([-1.0, 1.0], size=(n, len(diff)))
        st = (signs * diff[None, :]).mean(axis=1)
        exceed += int(np.sum(st >= observed))
        done += n
    return {
        "n_vessels": len(common),
        "mean_vessel_delta_RMSE_direct_minus_agg": observed,
        "one_sided_signflip_p": (exceed + 1) / (reps + 1),
        "permutations": reps,
    }


def residual_diagnostics(df, scope, max_lag):
    dw_rows = []
    acf_rows = []
    for pred_col, label in [
        ("pred_5min_agg", "5min_aggregated"),
        ("pred_10min_direct", "direct_10min"),
    ]:
        for vessel, g in df.groupby("vessel_id"):
            g = g.copy()
            if "timestamp" in g.columns:
                g["timestamp"] = pd.to_datetime(g["timestamp"], errors="coerce")
                g = g.sort_values("timestamp")
            e = (
                g[pred_col].to_numpy(float)
                - g["target_10min"].to_numpy(float)
            )
            if len(e) < 4:
                continue
            dw_rows.append({
                "scope": scope,
                "method": label,
                "vessel_id": str(vessel),
                "n": len(e),
                "Durbin_Watson": float(durbin_watson(e)),
            })
            aa = acf(
                e,
                nlags=min(max_lag, len(e) - 1),
                fft=True,
                missing="drop",
            )
            for lag in range(1, len(aa)):
                acf_rows.append({
                    "scope": scope,
                    "method": label,
                    "vessel_id": str(vessel),
                    "lag": lag,
                    "ACF": float(aa[lag]),
                })
    return pd.DataFrame(dw_rows), pd.DataFrame(acf_rows)


def bp_rows(df, scope):
    rows = []
    ship = pd.get_dummies(
        df["ship_type"], prefix="ship", drop_first=True, dtype=float
    ).reset_index(drop=True)

    for pred_col, label in [
        ("pred_5min_agg", "5min_aggregated"),
        ("pred_10min_direct", "direct_10min"),
    ]:
        e = (
            df[pred_col].to_numpy(float)
            - df["target_10min"].to_numpy(float)
        )
        X = pd.DataFrame({
            "prediction": df[pred_col].to_numpy(float),
            "speed_std_within10": df["speed_std_within10"].to_numpy(float),
        })
        X = pd.concat([X.reset_index(drop=True), ship], axis=1)
        X = sm.add_constant(X, has_constant="add")
        lm, lmp, fval, fp = het_breuschpagan(e, X.to_numpy(float))
        rows.append({
            "scope": scope,
            "method": label,
            "LM": float(lm),
            "LM_p_iid_descriptive": float(lmp),
            "F": float(fval),
            "F_p_iid_descriptive": float(fp),
            "note": "descriptive only; 10-min windows are serially clustered within vessel",
        })
    return rows


def run_time_resolution(time_root: Path, out: Path, args):
    d = ensure_dir(out / "07_remaining" / "04_time_resolution_V3")
    all_rows = load_v3_paired(time_root, args.time_csv_chunksize)
    all_rows.to_csv(
        d / "paired_5min_vs_10min_test_windows_from_V3.csv.gz",
        index=False,
        compression="gzip",
    )

    scope_defs = {
        "all_V3_test_windows": np.ones(len(all_rows), dtype=bool),
        "cruise_only": all_rows["voyage_phase"].eq("cruise").to_numpy(),
        "maneuver_only": all_rows["voyage_phase"].eq("maneuver").to_numpy(),
    }

    summary_rows = []
    boot_rows = []
    perm_rows = []
    dw_all, acf_all, bp_all = [], [], []
    var_rows = []
    cancel_rows = []

    manuscript_target = {
        "n": 18197,
        "RMSE5": 0.0327,
        "RMSE10": 0.0337,
    }

    for si, (scope, mask) in enumerate(scope_defs.items()):
        z = all_rows.loc[mask].copy().reset_index(drop=True)
        if len(z) == 0:
            continue

        m5 = metrics(z["target_10min"], z["pred_5min_agg"])
        m10 = metrics(z["target_10min"], z["pred_10min_direct"])

        score = (
            abs(len(z) - manuscript_target["n"]) / manuscript_target["n"]
            + abs(m5["RMSE"] - manuscript_target["RMSE5"]) / manuscript_target["RMSE5"]
            + abs(m10["RMSE"] - manuscript_target["RMSE10"]) / manuscript_target["RMSE10"]
        )
        summary_rows.append({
            "scope": scope,
            "n_windows": len(z),
            "n_vessels": z["vessel_id"].nunique(),
            "RMSE_5min_aggregated": m5["RMSE"],
            "R2_5min_aggregated": m5["R2"],
            "RMSE_direct_10min": m10["RMSE"],
            "R2_direct_10min": m10["R2"],
            "delta_RMSE_direct_minus_aggregate": m10["RMSE"] - m5["RMSE"],
            "manuscript_Table10_match_score_lower_is_better": score,
        })

        br = paired_cluster_boot_rmse(
            z, args.bootstrap, args.seed + 15000 + si
        )
        br["scope"] = scope
        boot_rows.append(br)

        pr = signflip_permutation_rmse(
            z, args.permutations, args.seed + 15100 + si
        )
        pr["scope"] = scope
        perm_rows.append(pr)

        dw, aa = residual_diagnostics(z, scope, args.acf_max_lag)
        if len(dw):
            dw_all.append(dw)
        if len(aa):
            acf_all.append(aa)
        bp_all.extend(bp_rows(z, scope))

        z["speed_variability_tertile"] = pd.qcut(
            z["speed_std_within10"],
            q=3,
            labels=["low", "moderate", "high"],
            duplicates="drop",
        ).astype(str)
        for level, g in z.groupby("speed_variability_tertile"):
            mm5 = metrics(g["target_10min"], g["pred_5min_agg"])
            mm10 = metrics(g["target_10min"], g["pred_10min_direct"])
            var_rows.append({
                "scope": scope,
                "speed_variability_tertile": level,
                "n": len(g),
                "median_speed_std_within10": float(g["speed_std_within10"].median()),
                "RMSE_5min_aggregated": mm5["RMSE"],
                "RMSE_direct_10min": mm10["RMSE"],
                "delta_RMSE_direct_minus_aggregate": mm10["RMSE"] - mm5["RMSE"],
            })

        # Error-cancellation: residual convention prediction - observation.
        z["e5a"] = z["pred_5a"] - z["target_5a"]
        z["e5b"] = z["pred_5b"] - z["target_5b"]
        rho, p = spearmanr(z["e5a"], z["e5b"])
        pearson = float(np.corrcoef(z["e5a"], z["e5b"])[0, 1])

        v_corr = []
        for vessel, g in z.groupby("vessel_id"):
            if len(g) >= 20:
                c = np.corrcoef(g["e5a"], g["e5b"])[0, 1]
                if np.isfinite(c):
                    v_corr.append(c)

        cancel_rows.append({
            "scope": scope,
            "n_windows": len(z),
            "Pearson_first5_second5_residual": pearson,
            "Spearman_first5_second5_residual": float(rho),
            "Spearman_p_iid_descriptive": float(p),
            "median_within_vessel_Pearson": (
                float(np.median(v_corr)) if v_corr else np.nan
            ),
            "vessels_with_defined_correlation": len(v_corr),
            "interpretation": (
                "negative association is compatible with partial error cancellation"
                if pearson < 0
                else "no negative half-window residual association supporting error cancellation"
            ),
        })

    summary = pd.DataFrame(summary_rows).sort_values(
        "manuscript_Table10_match_score_lower_is_better"
    )
    atomic_csv(summary, d / "Table_TIME_0_scope_match_and_metrics.csv")
    atomic_csv(pd.DataFrame(boot_rows), d / "Table_TIME_A_2000_vessel_cluster_bootstrap.csv")
    atomic_csv(pd.DataFrame(perm_rows), d / "Table_TIME_A2_vessel_signflip_permutation.csv")
    if dw_all:
        atomic_csv(pd.concat(dw_all, ignore_index=True), d / "Table_TIME_B1_DW_by_vessel.csv")
    if acf_all:
        atomic_csv(pd.concat(acf_all, ignore_index=True), d / "Table_TIME_B2_ACF_by_vessel.csv")
    atomic_csv(pd.DataFrame(bp_all), d / "Table_TIME_B3_Koenker_BP_descriptive.csv")
    atomic_csv(pd.DataFrame(var_rows), d / "Table_TIME_B4_speed_variability_strata.csv")
    atomic_csv(pd.DataFrame(cancel_rows), d / "Table_TIME_B5_half_window_error_cancellation.csv")

    best = summary.iloc[0]
    write_text(
        d / "TIME_SCOPE_GUIDE.txt",
        f"""The script reports all V3 scopes separately and does not silently choose a
new modelling cohort.

Closest scope to the manuscript Table 10 target (n≈18,197; RMSE≈0.0327/0.0337):
    scope={best['scope']}
    n={int(best['n_windows'])}
    RMSE 5-min aggregate={best['RMSE_5min_aggregated']:.6f}
    RMSE direct 10-min={best['RMSE_direct_10min']:.6f}

Before manuscript insertion, confirm that this is the exact scope used to produce
the published Table 10. If it is not close, locate the earlier V2/V3 run directory
rather than forcing the current files to match.
""",
    )
    print("[OK] V3 time-resolution diagnostics")


# ---------------------------------------------------------------------
# 5. WAVE SCOPE RECONCILIATION NOTE
# ---------------------------------------------------------------------

def run_wave_scope_note(reviewer_out: Path, out: Path):
    d = ensure_dir(out / "07_remaining" / "05_wave_scope_reconciliation")
    wave_path = (
        reviewer_out
        / "03_wave_C4_stratified"
        / "Table_WAVE_C4_posthoc_stratified_diagnostics.csv"
    )

    full_n = None
    full_up = None
    full_down = None
    if wave_path.is_file():
        w = pd.read_csv(wave_path)
        q = w.loc[
            w["dataset"].astype(str).eq("full_supported_L1")
            & w["stratification"].astype(str).eq("global")
        ]
        if len(q):
            full_n = int(q.iloc[0]["n"])
            full_up = float(q.iloc[0]["expected_direction_pct"])
            full_down = float(q.iloc[0]["reverse_direction_pct"])

    tab = pd.DataFrame([
        {
            "role": "PRIMARY pre-specified C4 audit",
            "sample_scope": "locked 20,000-row held-out audit sample; support-aware after perturbation",
            "supported_n": 20000,
            "wave_prediction_up_pct": 61.67,
            "wave_prediction_down_pct": 35.61,
            "use_in_manuscript": "Table 12/13 primary audit result",
        },
        {
            "role": "POST-HOC failure-domain diagnostic",
            "sample_scope": "all supported L1 held-out observations",
            "supported_n": full_n,
            "wave_prediction_up_pct": full_up,
            "wave_prediction_down_pct": full_down,
            "use_in_manuscript": "subgroup diagnostic only; does not replace primary audit",
        },
    ])
    atomic_csv(tab, d / "Table_WAVE_primary_vs_fullsupport_scope.csv")

    write_text(
        d / "MANUSCRIPT_WORDING.txt",
        """Recommended Method sentence for Section 3.4.3:

'For the primary local-perturbation audit, a fixed 20,000-observation sample was
drawn from the locked L1 held-out set before applying perturbation-specific
support checks. The resulting supported sample sizes are reported explicitly in
the perturbation table. This locked audit sample is used for the pre-specified C4
decision.'

Recommended Results bridge after Table 12/13:

'Because the wave-height audit showed a sizeable minority of reverse responses,
a post-hoc failure-domain diagnostic was subsequently repeated over all supported
L1 held-out observations and stratified by ship type, empirical wave-height
tertile, speed tertile, and relative-wave sector. This full-support diagnostic
was used to localise heterogeneity rather than to replace the pre-specified
20,000-observation audit.'

Recommended table note:

'Table values refer to the fixed 20,000-observation primary perturbation audit;
supported n may be slightly smaller when the perturbed value leaves the
ship-type-specific training support.'

With these statements, 61.67% (primary 20k audit) and ~61.45% (full-support
post-hoc diagnostic) are not competing estimates. They answer different stages
of the audit.
""",
    )
    print("[OK] wave primary/full-support scope note")


def main():
    args = parse_args()
    steps = [x.strip() for x in args.steps.split(",") if x.strip()]
    main_out = Path(args.main_output)
    reviewer_out = Path(args.reviewer_output)

    # Only the first three modules require the 489,620-row locked F31 cohort.
    # Time-resolution V3 and wave-scope reconciliation use existing output files
    # and can run without loading the large cruise CSV/model at all.
    f31_steps = {
        "source_shap_fixed",
        "lovo_target_variance",
        "rudder_dual_estimand",
    }
    need_f31 = any(step in f31_steps for step in steps)

    core = raw = X = locked = None
    if need_f31:
        base = load_base_module(args)
        core, raw, X, locked = load_locked(base, args)
    else:
        print(
            "[SKIP] Locked F31 cohort/model loading: requested steps "
            "use existing output files only."
        )

    if "source_shap_fixed" in steps:
        run_source_shap_fixed(main_out, reviewer_out)

    if "lovo_target_variance" in steps:
        run_lovo_target_variance(raw, locked, reviewer_out)

    if "rudder_dual_estimand" in steps:
        run_rudder_dual(raw, X, locked, main_out, reviewer_out, args)

    if "time_resolution_from_v3" in steps:
        run_time_resolution(Path(args.time_root), reviewer_out, args)

    if "wave_scope_note" in steps:
        run_wave_scope_note(reviewer_out, reviewer_out)

    write_text(
        reviewer_out / "07_remaining" / "RUN_MANIFEST.json",
        json.dumps(
            {
                "version": VERSION,
                "steps": steps,
                "bootstrap_reps": args.bootstrap,
                "permutations": args.permutations,
                "seed": args.seed,
                "main_output": args.main_output,
                "raw_cruise": args.raw_cruise,
                "time_root": args.time_root,
                "no_F31_retraining": True,
                "no_time_resolution_retraining": True,
            },
            indent=2,
            ensure_ascii=False,
        ),
    )

    print("\nDONE")
    print(
        "Outputs: "
        + str(reviewer_out / "07_remaining")
    )


if __name__ == "__main__":
    main()
