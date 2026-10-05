#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A1 — strict temporal inner tuning for the L2 known-vessel 80→20 design."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

from _revision_common import (
    DEFAULT_SEED, atomic_csv, atomic_json, by_vessel_metrics,
    expanding_temporal_folds, fit_locked_model, load_fixed31_cruise,
    load_locked_params, metric_row, run_optuna_cv, setup_logger,
)


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--fixed31-cruise", required=True)
    p.add_argument("--analysis-dir", default="src/paper_analysis")
    p.add_argument("--column-overrides", default="src/paper_analysis/column_overrides_fixed31.json")
    p.add_argument("--hyperparams-csv", default="src/paper_analysis/02_best_hyperparameters.csv")
    p.add_argument("--output-dir", default="revision_runs/A1_strict_temporal")
    p.add_argument("--models", default="xgb,lgbm")
    p.add_argument("--outer-train-fraction", type=float, default=0.80)
    p.add_argument("--inner-folds", type=int, default=3)
    p.add_argument("--inner-initial-train-fraction", type=float, default=0.50)
    p.add_argument("--n-trials", type=int, default=8)
    p.add_argument("--early-stopping-rounds", type=int, default=80)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--n-jobs", type=int, default=4)
    p.add_argument("--trajectory-gap-minutes", type=float, default=30.0)
    p.add_argument("--allow-noncanonical-counts", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.output_dir).resolve()
    logger = setup_logger("A1", out, "A1_strict_temporal.log")

    core, runner, raw, X, feature_cols, _ = load_fixed31_cruise(
        args.fixed31_cruise, args.analysis_dir, args.column_overrides,
        args.trajectory_gap_minutes, logger, args.allow_noncanonical_counts,
    )
    params = load_locked_params(runner, args.hyperparams_csv, logger)
    models = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    unsupported = [m for m in models if m not in {"xgb", "lgbm"}]
    if unsupported:
        raise ValueError(f"A1 nested tuning supports xgb/lgbm only; got {unsupported}")

    outer_tr, outer_te = core.known_vessel_temporal_split(raw, args.outer_train_fraction)
    raw_tr = raw.iloc[outer_tr].reset_index(drop=True)
    X_tr = X.iloc[outer_tr].reset_index(drop=True)
    raw_te = raw.iloc[outer_te].reset_index(drop=True)
    X_te = X.iloc[outer_te].reset_index(drop=True)

    folds = expanding_temporal_folds(
        raw_tr, args.inner_folds, args.inner_initial_train_fraction
    )
    fold_manifest = []
    for i, (tr, va) in enumerate(folds, start=1):
        fold_manifest.append({
            "fold": i, "train_rows": int(len(tr)), "valid_rows": int(len(va)),
            "train_max_timestamp": str(raw_tr.iloc[tr]["timestamp"].max()),
            "valid_min_timestamp": str(raw_tr.iloc[va]["timestamp"].min()),
            "valid_max_timestamp": str(raw_tr.iloc[va]["timestamp"].max()),
        })
    atomic_csv(pd.DataFrame(fold_manifest), out / "A1_inner_temporal_folds.csv")

    overall_rows, byv_parts, selected_rows, paired_parts = [], [], [], []

    for model_name in models:
        logger.info("=== A1 model=%s ===", model_name)
        selected, search = run_optuna_cv(
            model_name=model_name,
            locked_params=params[model_name],
            X=X_tr, raw=raw_tr, splits=folds, runner=runner,
            n_trials=args.n_trials, seed=args.seed + 100,
            n_jobs_model=args.n_jobs,
            early_stopping_rounds=args.early_stopping_rounds,
            logger=logger,
        )
        atomic_csv(search, out / f"A1_{model_name}_inner_search.csv")

        m_locked = fit_locked_model(
            runner, model_name, params[model_name], X_tr,
            raw_tr["target"].to_numpy(float),
            seed=args.seed, n_jobs=args.n_jobs,
        )
        p_locked = np.asarray(m_locked.predict(X_te), dtype=float)

        m_strict = fit_locked_model(
            runner, model_name, selected, X_tr,
            raw_tr["target"].to_numpy(float),
            seed=args.seed, n_jobs=args.n_jobs,
        )
        p_strict = np.asarray(m_strict.predict(X_te), dtype=float)

        for design, pred in [("locked", p_locked), ("strict_temporal_tuned", p_strict)]:
            overall_rows.append({
                "model": model_name, "design": design,
                **metric_row(core, raw_te["target"].to_numpy(float), pred),
            })
            vv = by_vessel_metrics(core, raw_te, pred)
            vv.insert(0, "design", design)
            vv.insert(0, "model", model_name)
            byv_parts.append(vv)

        locked_v = byv_parts[-2][["vessel_id", "ship_type", "RMSE"]].rename(
            columns={"RMSE": "RMSE_locked"}
        )
        strict_v = byv_parts[-1][["vessel_id", "ship_type", "RMSE"]].rename(
            columns={"RMSE": "RMSE_strict"}
        )
        pair = locked_v.merge(strict_v, on=["vessel_id", "ship_type"], validate="one_to_one")
        pair["RMSE_delta_strict_minus_locked"] = pair["RMSE_strict"] - pair["RMSE_locked"]
        pair.insert(0, "model", model_name)
        paired_parts.append(pair)

        selected_rows.append({
            "model": model_name,
            "selected_parameters_json": json.dumps(selected, sort_keys=True),
            "locked_parameters_json": json.dumps(params[model_name], sort_keys=True),
            "inner_cv_design": "within-outer-L2 expanding chronological folds by vessel",
            "inner_folds": args.inner_folds,
            "outer_design": "first 80% of each vessel -> final 20%",
            "outer_train_rows": len(raw_tr),
            "outer_test_rows": len(raw_te),
        })

    overall = pd.DataFrame(overall_rows)
    byv = pd.concat(byv_parts, ignore_index=True)
    paired = pd.concat(paired_parts, ignore_index=True)

    atomic_csv(overall, out / "A1_L2_overall_locked_vs_strict.csv")
    atomic_csv(byv, out / "A1_L2_by_vessel_locked_vs_strict.csv")
    atomic_csv(paired, out / "A1_L2_paired_vessel_RMSE_delta.csv")
    atomic_csv(pd.DataFrame(selected_rows), out / "A1_selected_hyperparameters.csv")
    atomic_json({
        "task": "A1 strict temporal nested-tuning sensitivity",
        "scientific_boundary": (
            "This is a revision-only sensitivity analysis. It does not replace or rewrite "
            "the historical manuscript hyperparameter registry."
        ),
        "models": models,
        "feature_cols": feature_cols,
        "seed": args.seed,
    }, out / "A1_manifest.json")

    logger.info("A1 COMPLETE: %s", out)


if __name__ == "__main__":
    main()
