#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A6b_zero_addback_retraining_sensitivity_v2.py

A6 reviewer-facing zero-fuel add-back retraining sensitivity.

Baseline:
    canonical Fixed31 official L1 training rows.
Sensitivity:
    the same official L1 training rows + A6b Fixed31-eligible literal-zero rows.

Both:
    - locked XGBoost hyperparameters;
    - no retuning;
    - exact same official canonical L1 test rows;
    - exact same canonical H3 support reference/sample.

Implementation alignment:
    - L1 metrics + SHAP follow A5_vessel_balanced_training_robustness.py;
    - H3 FT/FD + inference follow
      A3_multimodel_H3_robustness_FINAL_v2_memorysafe.py.

The analysis is a stress test of label-validity/exclusion sensitivity. It does
not assert that literal zero fuel is physically correct.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from _revision_common import (
    DEFAULT_SEED,
    atomic_csv,
    atomic_json,
    by_vessel_metrics,
    canonical_loader_args,
    fit_locked_model,
    import_analysis_module,
    interventional_tree_shap,
    load_analysis_modules,
    load_fixed31_cruise,
    load_locked_params,
    load_official_split,
    metric_row,
    setup_logger,
    shap_summary,
)


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--fixed31-cruise", required=True)
    p.add_argument("--addback-csv", required=True)
    p.add_argument("--analysis-dir", default="src/paper_analysis")
    p.add_argument(
        "--column-overrides",
        default="src/paper_analysis/column_overrides_fixed31.json",
    )
    p.add_argument(
        "--hyperparams-csv",
        default="src/paper_analysis/02_best_hyperparameters.csv",
    )
    p.add_argument("--split-path", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument(
        "--a3-module",
        default=None,
        help=(
            "Path to A3_multimodel_H3_robustness_FINAL_v2_memorysafe.py; "
            "defaults to the copy beside this script."
        ),
    )
    p.add_argument("--reductions", default="0.05,0.10,0.15")
    p.add_argument("--bootstrap-reps", type=int, default=2000)
    p.add_argument("--permutations", type=int, default=100000)

    # Match A5's locked-training seed for the training-scheme sensitivity.
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)

    # A3-final resampling seed. Point estimates are unaffected by this seed;
    # it controls bootstrap/permutation reproducibility.
    p.add_argument("--h3-inference-seed", type=int, default=20260819)

    p.add_argument("--n-jobs", type=int, default=4)
    p.add_argument("--trajectory-gap-minutes", type=float, default=30.0)

    # Match A5 defaults exactly.
    p.add_argument("--shap-sample-n", type=int, default=12000)
    p.add_argument("--shap-background-n", type=int, default=256)

    p.add_argument("--fd-chunk-rows", type=int, default=3000)
    p.add_argument("--skip-shap", action="store_true")
    p.add_argument("--skip-h3", action="store_true")
    return p.parse_args()


def load_a3_module(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"A3 final module not found: {path}")
    spec = importlib.util.spec_from_file_location("a3_final_for_a6", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import A3 module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def normalize_ship_type(x) -> str:
    s = str(x).strip().lower()
    if "bulk" in s:
        return "bulk"
    if "container" in s:
        return "container"
    if "tank" in s:
        return "tanker"
    return s


def add_scenario(df: pd.DataFrame, scenario: str) -> pd.DataFrame:
    z = df.copy()
    z.insert(0, "scenario", scenario)
    return z


def canonicalize_addback(
    addback_path: Path,
    analysis_dir: str,
    column_overrides: str,
    trajectory_gap_minutes: float,
    base_feature_cols: list[str],
    logger,
):
    """
    Canonicalize add-back rows separately from Fixed31.

    This is deliberately separate so adding the sensitivity rows cannot change
    canonicalization, ordering, or content of the historical Fixed31 cohort.
    """
    core, _ = load_analysis_modules(analysis_dir)
    src = pd.read_csv(addback_path, low_memory=False)

    if len(src) == 0:
        raise AssertionError("A6 add-back table is empty.")

    if "ship_type" in src.columns:
        ship = src["ship_type"].astype("string").str.strip().str.lower()
        if not ship.eq("container").all():
            raise AssertionError("All A6 add-back rows must be container.")
    if "voyage_phase" in src.columns:
        phase = src["voyage_phase"].astype("string").str.strip().str.lower()
        if not phase.eq("cruise").all():
            raise AssertionError("All A6 add-back rows must be cruise.")
    if "fuel_t_10min" not in src.columns:
        raise KeyError("Add-back CSV lacks fuel_t_10min.")
    fuel = pd.to_numeric(src["fuel_t_10min"], errors="coerce")
    if not (np.isfinite(fuel) & fuel.eq(0.0)).all():
        raise AssertionError(
            "All A6 add-back targets must be finite literal zero."
        )

    args = canonical_loader_args(column_overrides, trajectory_gap_minutes)
    bundle = core.canonicalize_dataframe(src, args, logger)
    raw_add = bundle.raw.reset_index(drop=True)
    X_add = bundle.feature_df.reset_index(drop=True)
    add_feature_cols = list(bundle.feature_cols)

    if len(raw_add) != len(src) or len(X_add) != len(src):
        raise AssertionError(
            "Canonicalization changed the A6 add-back row count."
        )
    if add_feature_cols != list(base_feature_cols):
        raise AssertionError(
            "Add-back canonical feature schema/order differs from Fixed31.\n"
            f"Fixed31={base_feature_cols}\nAddback={add_feature_cols}"
        )
    if not np.allclose(
        raw_add["target"].to_numpy(float),
        0.0,
        atol=0.0,
        rtol=0.0,
    ):
        raise AssertionError(
            "Canonicalized A6 add-back target is no longer exact zero."
        )
    if not np.isfinite(X_add.to_numpy(dtype=float)).all():
        raise AssertionError(
            "A6 add-back canonical feature matrix contains non-finite values."
        )

    return raw_add, X_add


def paired_h3_delta(
    baseline: pd.DataFrame,
    addback: pd.DataFrame,
    *,
    outcome: str,
    stat_fn,
    reps: int,
    seed: int,
):
    """
    Paired vessel bootstrap of addback - baseline endpoint.

    Vessel draws are identical in the two scenarios for each replicate.
    """
    types = ["bulk", "container", "tanker"]
    rows = []

    b = baseline.copy()
    a = addback.copy()
    b["ship_type"] = b["ship_type"].map(normalize_ship_type)
    a["ship_type"] = a["ship_type"].map(normalize_ship_type)

    for ri, reduction in enumerate(sorted(b["reduction_pct"].unique())):
        br = b[b["reduction_pct"] == reduction].copy()
        ar = a[a["reduction_pct"] == reduction].copy()

        for si, scope in enumerate(["Fleet"] + types):
            bg = br if scope == "Fleet" else br[br["ship_type"] == scope]
            ag = ar if scope == "Fleet" else ar[ar["ship_type"] == scope]

            vb = sorted(bg["vessel_id"].astype(str).unique())
            va = sorted(ag["vessel_id"].astype(str).unique())
            if vb != va:
                raise AssertionError(
                    f"Baseline/add-back vessel sets differ: "
                    f"{scope}, reduction={reduction}"
                )

            # One row per vessel per reduction is expected.
            if len(bg) != len(vb) or len(ag) != len(va):
                raise AssertionError(
                    "H3 paired delta expects one endpoint row per vessel "
                    "per reduction."
                )

            point_b = float(stat_fn(bg))
            point_a = float(stat_fn(ag))

            by_b = {
                v: bg[bg["vessel_id"].astype(str) == v].copy()
                for v in vb
            }
            by_a = {
                v: ag[ag["vessel_id"].astype(str) == v].copy()
                for v in va
            }

            rng = np.random.default_rng(seed + ri * 100 + si)
            vals = np.empty(int(reps), dtype=float)
            for i in range(int(reps)):
                draw = rng.choice(vb, size=len(vb), replace=True)
                sb = pd.concat([by_b[v] for v in draw], ignore_index=True)
                sa = pd.concat([by_a[v] for v in draw], ignore_index=True)
                vals[i] = float(stat_fn(sa) - stat_fn(sb))

            rows.append({
                "outcome": outcome,
                "reduction_pct": float(reduction),
                "scope": scope,
                "n_vessels": len(vb),
                "baseline_estimate_pct": point_b,
                "zero_addback_estimate_pct": point_a,
                "delta_addback_minus_baseline_pp": point_a - point_b,
                "paired_vessel_bootstrap_CI95_low":
                    float(np.nanquantile(vals, 0.025)),
                "paired_vessel_bootstrap_CI95_high":
                    float(np.nanquantile(vals, 0.975)),
            })
    return rows


def main():
    args = parse_args()
    out = Path(args.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(
        "A6_zero_addback",
        out,
        "A6b_zero_addback_retraining.log",
    )

    fixed31 = Path(args.fixed31_cruise).expanduser().resolve()
    addback = Path(args.addback_csv).expanduser().resolve()
    split_path = Path(args.split_path).expanduser().resolve()
    for p, label in (
        (fixed31, "canonical Fixed31"),
        (addback, "A6b add-back CSV"),
        (split_path, "official split"),
    ):
        if not p.is_file():
            raise FileNotFoundError(f"{label} not found: {p}")

    script_dir = Path(__file__).resolve().parent
    a3_path = (
        Path(args.a3_module).expanduser().resolve()
        if args.a3_module
        else script_dir / "A3_multimodel_H3_robustness_FINAL_v2_memorysafe.py"
    )
    a3 = load_a3_module(a3_path)

    # ------------------------------------------------------------------
    # 1. Load canonical Fixed31 EXACTLY as A3/A5 do.
    # ------------------------------------------------------------------
    core, runner, raw, X, feature_cols, _ = load_fixed31_cruise(
        fixed31,
        args.analysis_dir,
        args.column_overrides,
        args.trajectory_gap_minutes,
        logger,
        allow_noncanonical_counts=False,
    )
    if len(feature_cols) != 17:
        raise AssertionError(
            f"Expected canonical17 features, observed {len(feature_cols)}."
        )

    # Canonicalize add-back separately so it cannot perturb Fixed31.
    raw_add, X_add = canonicalize_addback(
        addback,
        args.analysis_dir,
        args.column_overrides,
        args.trajectory_gap_minutes,
        feature_cols,
        logger,
    )
    n_add = len(raw_add)

    # ------------------------------------------------------------------
    # 2. Exact historical L1 split on Fixed31 only.
    # ------------------------------------------------------------------
    tr, te, split_source = load_official_split(
        split_path,
        raw,
        core,
        seed=args.seed,
        allow_rebuild=False,
        logger=logger,
    )

    raw_tr = raw.iloc[tr].reset_index(drop=True)
    raw_te = raw.iloc[te].reset_index(drop=True)
    X_tr = X.iloc[tr].reset_index(drop=True)
    X_te = X.iloc[te].reset_index(drop=True)
    y_tr = raw_tr["target"].to_numpy(float)
    y_te = raw_te["target"].to_numpy(float)

    X_aug_tr = pd.concat([X_tr, X_add], ignore_index=True)
    y_aug_tr = np.concatenate(
        [y_tr, raw_add["target"].to_numpy(float)]
    )

    if len(X_tr) != 391_696 or len(X_te) != 97_924:
        raise AssertionError(
            f"Official L1 counts changed: train={len(X_tr):,}, "
            f"test={len(X_te):,}"
        )
    if len(X_aug_tr) != len(X_tr) + n_add:
        raise AssertionError("Augmented training size mismatch.")

    logger.info(
        "A6 design locked | baseline train=%d | add-back=%d | "
        "augmented train=%d | fixed test=%d",
        len(X_tr), n_add, len(X_aug_tr), len(X_te),
    )

    params = load_locked_params(
        runner,
        args.hyperparams_csv,
        logger,
    )
    if "xgb" not in params:
        raise KeyError("XGB missing from locked hyperparameter registry.")

    cii = None if args.skip_h3 else import_analysis_module(
        args.analysis_dir,
        "run_cii_l1_aligned_final",
    )
    reductions = [
        float(x) for x in args.reductions.split(",") if x.strip()
    ]

    models = {}
    predictions = {}
    perf_rows = []
    vessel_parts = []

    # Train both before SHAP/H3, exactly analogous to A5.
    for scenario, X_fit, y_fit in (
        ("baseline", X_tr, y_tr),
        ("zero_addback", X_aug_tr, y_aug_tr),
    ):
        logger.info("Fitting xgb / %s", scenario)
        model = fit_locked_model(
            runner,
            "xgb",
            params["xgb"],
            X_fit,
            y_fit,
            seed=args.seed,
            n_jobs=args.n_jobs,
        )
        pred = np.asarray(model.predict(X_te), dtype=float)
        models[scenario] = model
        predictions[scenario] = pred

        perf_rows.append({
            "scenario": scenario,
            "train_rows": int(len(X_fit)),
            "addback_rows": int(n_add if scenario == "zero_addback" else 0),
            **metric_row(core, y_te, pred),
        })
        vv = by_vessel_metrics(core, raw_te, pred)
        vv.insert(0, "scenario", scenario)
        vessel_parts.append(vv)

    perf = pd.DataFrame(perf_rows)
    base_perf = perf[perf["scenario"] == "baseline"].iloc[0]
    add_perf = perf[perf["scenario"] == "zero_addback"].iloc[0]
    numeric_metrics = [
        c for c in perf.columns
        if c not in {"scenario", "train_rows", "addback_rows"}
        and pd.api.types.is_numeric_dtype(perf[c])
    ]
    for metric in numeric_metrics:
        b = float(base_perf[metric])
        a = float(add_perf[metric])
        perf.loc[
            perf["scenario"] == "zero_addback",
            f"delta_vs_baseline_{metric}",
        ] = a - b

    atomic_csv(perf, out / "A6_L1_overall.csv")
    atomic_csv(
        pd.concat(vessel_parts, ignore_index=True),
        out / "A6_L1_by_vessel.csv",
    )

    pred_out = raw_te[
        ["vessel_id", "ship_type", "target"]
    ].copy()
    pred_out.insert(
        0, "official_test_position", np.arange(len(raw_te), dtype=int)
    )
    for scenario, pred in predictions.items():
        pred_out[f"pred_{scenario}"] = pred
        pred_out[f"residual_{scenario}"] = (
            raw_te["target"].to_numpy(float) - pred
        )
    atomic_csv(pred_out, out / "A6_L1_test_predictions.csv")

    # Type-level L1 metrics, still on exactly the same test rows.
    type_rows = []
    for scenario, pred in predictions.items():
        for ship_type, idx in raw_te.groupby("ship_type").groups.items():
            idx = np.asarray(list(idx), dtype=int)
            type_rows.append({
                "scenario": scenario,
                "ship_type": normalize_ship_type(ship_type),
                "n_test": int(len(idx)),
                **metric_row(
                    core,
                    y_te[idx],
                    pred[idx],
                ),
            })
    atomic_csv(
        pd.DataFrame(type_rows),
        out / "A6_L1_by_ship_type.csv",
    )

    # ------------------------------------------------------------------
    # 3. SHAP — exact A5 convention: common canonical background + heldout
    # sample, same defaults and same seed for both fits.
    # ------------------------------------------------------------------
    shap_manifest = {"available": False}
    if not args.skip_shap:
        shap_imp = {}
        shap_dir = {}
        shap_idx = {}

        for scenario, model in models.items():
            logger.info("Computing A5-aligned SHAP / %s", scenario)
            ev, sv, idx = interventional_tree_shap(
                model,
                X_tr,
                X_te,
                background_n=args.shap_background_n,
                sample_n=args.shap_sample_n,
                seed=args.seed + 500,
            )
            imp, direction = shap_summary(ev, sv)
            shap_imp[scenario] = imp
            shap_dir[scenario] = direction
            shap_idx[scenario] = idx
            atomic_csv(
                add_scenario(imp, scenario),
                out / f"A6_SHAP_global_importance_{scenario}.csv",
            )
            atomic_csv(
                add_scenario(direction, scenario),
                out / f"A6_SHAP_direction_{scenario}.csv",
            )
            del ev, sv
            gc.collect()

        if not np.array_equal(
            shap_idx["baseline"],
            shap_idx["zero_addback"],
        ):
            raise AssertionError("SHAP held-out samples differ.")

        b = shap_imp["baseline"].rename(columns={
            "mean_abs_SHAP": "baseline_mean_abs_SHAP",
            "rank": "baseline_rank",
        })
        a = shap_imp["zero_addback"].rename(columns={
            "mean_abs_SHAP": "zero_addback_mean_abs_SHAP",
            "rank": "zero_addback_rank",
        })
        bd = shap_dir["baseline"].rename(columns={
            "spearman_feature_vs_SHAP":
                "baseline_spearman_feature_vs_SHAP"
        })
        ad = shap_dir["zero_addback"].rename(columns={
            "spearman_feature_vs_SHAP":
                "zero_addback_spearman_feature_vs_SHAP"
        })
        comp = (
            b.merge(a, on="feature")
            .merge(bd, on="feature")
            .merge(ad, on="feature")
        )
        comp["rank_change_addback_minus_baseline"] = (
            comp["zero_addback_rank"] - comp["baseline_rank"]
        )
        comp["direction_same_sign"] = (
            np.sign(comp["baseline_spearman_feature_vs_SHAP"])
            == np.sign(comp["zero_addback_spearman_feature_vs_SHAP"])
        )
        comp = comp.sort_values("baseline_rank").reset_index(drop=True)
        atomic_csv(comp, out / "A6_SHAP_comparison.csv")

        rank_rho = float(spearmanr(
            comp["baseline_rank"].to_numpy(float),
            comp["zero_addback_rank"].to_numpy(float),
        ).statistic)
        top_b = set(comp.nsmallest(5, "baseline_rank")["feature"])
        top_a = set(comp.nsmallest(5, "zero_addback_rank")["feature"])
        union = top_b | top_a
        shap_manifest = {
            "available": True,
            "A5_aligned": True,
            "background_n": int(min(args.shap_background_n, len(X_tr))),
            "sample_n": int(min(args.shap_sample_n, len(X_te))),
            "common_canonical_training_background": True,
            "common_heldout_test_sample": True,
            "global_rank_spearman": rank_rho,
            "top5_baseline": sorted(map(str, top_b)),
            "top5_zero_addback": sorted(map(str, top_a)),
            "top5_overlap_n": int(len(top_b & top_a)),
            "top5_jaccard":
                float(len(top_b & top_a) / len(union)) if union else np.nan,
            "direction_same_sign_n":
                int(comp["direction_same_sign"].sum()),
            "features": int(len(comp)),
            "note": "SHAP is associational, not causal.",
        }
        atomic_json(shap_manifest, out / "A6_SHAP_summary.json")

    # ------------------------------------------------------------------
    # 4. H3 — exact A3 FINAL v2 machinery, but support reference is fixed
    # to canonical raw_tr for BOTH A6 training scenarios.
    # ------------------------------------------------------------------
    h3_ft = {}
    h3_fd = {}
    h3_support = {}
    endpoint_rows = []
    pair_rows = []
    interaction_rows = []

    if not args.skip_h3:
        for si, scenario in enumerate(("baseline", "zero_addback")):
            logger.info(
                "Running A3-final H3 machinery / %s", scenario
            )
            ft, fd, support, _ = a3.calculate_ft_fd_memory_safe(
                cii=cii,
                model=models[scenario],
                raw_train=raw_tr,   # fixed canonical support reference
                raw_test=raw_te,
                feature_cols=feature_cols,
                saved_baseline_pred=predictions[scenario],
                reductions=reductions,
                default_cf=3.114,
                logger=logger,
                chunk_rows=args.fd_chunk_rows,
            )
            ft["ship_type"] = ft["ship_type"].map(normalize_ship_type)
            fd["ship_type"] = fd["ship_type"].map(normalize_ship_type)
            h3_ft[scenario] = ft
            h3_fd[scenario] = fd
            h3_support[scenario] = support

            atomic_csv(
                add_scenario(ft, scenario),
                out / f"A6_H3_FT_by_vessel_{scenario}.csv",
            )
            atomic_csv(
                add_scenario(fd, scenario),
                out / f"A6_H3_FD_by_vessel_{scenario}.csv",
            )
            atomic_csv(
                add_scenario(support, scenario),
                out / f"A6_H3_support_{scenario}.csv",
            )

            ep, pairs, inter = a3.h3_inference_for_model(
                model_name=f"xgb_{scenario}",
                ft=ft,
                fd=fd,
                bootstrap_reps=args.bootstrap_reps,
                permutations=args.permutations,
                seed=args.h3_inference_seed + si * 10000,
            )
            for row in ep:
                row["scenario"] = scenario
            for row in pairs:
                row["scenario"] = scenario
            for row in inter:
                row["scenario"] = scenario
            endpoint_rows.extend(ep)
            pair_rows.extend(pairs)
            interaction_rows.extend(inter)

        # Same support rows/counts are mandatory.
        bs = h3_support["baseline"].reset_index(drop=True)
        az = h3_support["zero_addback"].reset_index(drop=True)
        if list(bs.columns) != list(az.columns) or not bs.equals(az):
            raise AssertionError(
                "Baseline and zero-addback H3 support audits differ."
            )

        atomic_csv(
            pd.DataFrame(endpoint_rows),
            out / "A6_H3_endpoint_vessel_bootstrap_CI.csv",
        )
        atomic_csv(
            pd.DataFrame(pair_rows),
            out / "A6_H3_pairwise_shiptype_contrasts.csv",
        )
        atomic_csv(
            pd.DataFrame(interaction_rows),
            out / "A6_H3_shiptype_x_reduction_interaction.csv",
        )

        delta_rows = []
        delta_rows.extend(paired_h3_delta(
            h3_ft["baseline"],
            h3_ft["zero_addback"],
            outcome="FT_CII_change_pct",
            stat_fn=a3.ft_weighted_cii_change,
            reps=args.bootstrap_reps,
            seed=args.h3_inference_seed + 30000,
        ))
        delta_rows.extend(paired_h3_delta(
            h3_fd["baseline"],
            h3_fd["zero_addback"],
            outcome="FD_total_fuel_change_pct",
            stat_fn=a3.fd_total_fuel_change,
            reps=args.bootstrap_reps,
            seed=args.h3_inference_seed + 40000,
        ))
        atomic_csv(
            pd.DataFrame(delta_rows),
            out / "A6_H3_baseline_vs_addback_paired_delta.csv",
        )

    # ------------------------------------------------------------------
    # 5. Manifest.
    # ------------------------------------------------------------------
    manifest = {
        "task": "A6 zero-fuel add-back retraining sensitivity",
        "model": "xgb",
        "model_seed": int(args.seed),
        "training_alignment": "A5 locked-training convention",
        "model_selection": "locked manuscript hyperparameters; no retuning",
        "split_source": split_source,
        "canonical_train_rows": int(len(X_tr)),
        "eligible_addback_rows": int(n_add),
        "augmented_train_rows": int(len(X_aug_tr)),
        "canonical_test_rows": int(len(X_te)),
        "test_policy": "official canonical L1 test rows unchanged",
        "support_policy": (
            "canonical official L1 training rows define H3 support for both "
            "baseline and add-back fits"
        ),
        "feature_cols": feature_cols,
        "shap": shap_manifest,
        "h3_skipped": bool(args.skip_h3),
        "h3_alignment": (
            "A3_multimodel_H3_robustness_FINAL_v2_memorysafe.py"
        ),
        "h3_inference_seed": int(args.h3_inference_seed),
        "reductions": reductions,
        "bootstrap_reps": int(args.bootstrap_reps),
        "permutations": int(args.permutations),
        "fd_chunk_rows": int(args.fd_chunk_rows),
        "interpretation": (
            "Literal-zero stress test under an alternative label-validity "
            "assumption; zero fuel is not asserted to be physically correct."
        ),
    }
    atomic_json(manifest, out / "A6_manifest.json")

    logger.info(
        "A6 COMPLETE | baseline train=%d | zero-addback train=%d | "
        "fixed test=%d | addback=%d",
        len(X_tr),
        len(X_aug_tr),
        len(X_te),
        n_add,
    )


if __name__ == "__main__":
    main()
