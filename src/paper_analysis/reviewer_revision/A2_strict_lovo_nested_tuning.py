#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A2 — strict nested LOVO: each target vessel is excluded from fitting AND tuning."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import numpy as np
import pandas as pd

from _revision_common import (
    DEFAULT_SEED, atomic_csv, atomic_json, fit_locked_model,
    load_fixed31_cruise, load_locked_params, metric_row, run_optuna_cv,
    setup_logger, vessel_group_folds,
)


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--fixed31-cruise", required=True)
    p.add_argument("--analysis-dir", default="src/paper_analysis")
    p.add_argument("--column-overrides", default="src/paper_analysis/column_overrides_fixed31.json")
    p.add_argument("--hyperparams-csv", default="src/paper_analysis/02_best_hyperparameters.csv")
    p.add_argument("--output-dir", default="revision_runs/A2_strict_lovo")
    p.add_argument("--models", default="xgb,lgbm")
    p.add_argument("--target-vessels", default="all",
                   help="Comma-separated vessel IDs or 'all'. Useful for batched runs.")
    p.add_argument("--inner-folds", type=int, default=3)
    p.add_argument("--n-trials", type=int, default=6)
    p.add_argument("--early-stopping-rounds", type=int, default=80)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--n-jobs", type=int, default=4)
    p.add_argument("--trajectory-gap-minutes", type=float, default=30.0)
    p.add_argument("--allow-noncanonical-counts", action="store_true")
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.output_dir).resolve()
    detail_dir = out / "per_target"
    detail_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("A2", out, "A2_strict_lovo.log")

    core, runner, raw, X, feature_cols, _ = load_fixed31_cruise(
        args.fixed31_cruise, args.analysis_dir, args.column_overrides,
        args.trajectory_gap_minutes, logger, args.allow_noncanonical_counts,
    )
    params = load_locked_params(runner, args.hyperparams_csv, logger)
    models = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    if any(m not in {"xgb", "lgbm"} for m in models):
        raise ValueError("A2 nested tuning supports xgb/lgbm only.")

    all_vessels = sorted(raw["vessel_id"].astype(str).unique())
    if args.target_vessels.lower() == "all":
        targets = all_vessels
    else:
        targets = [x.strip() for x in args.target_vessels.split(",") if x.strip()]
        missing = sorted(set(targets) - set(all_vessels))
        if missing:
            raise ValueError(f"Unknown target vessels: {missing}")

    fold_rows, selected_rows = [], []

    for target_no, target in enumerate(targets, start=1):
        te_mask = raw["vessel_id"].astype(str).eq(target).to_numpy()
        tr_mask = ~te_mask
        raw_src = raw.loc[tr_mask].reset_index(drop=True)
        X_src = X.loc[tr_mask].reset_index(drop=True)
        raw_tgt = raw.loc[te_mask].reset_index(drop=True)
        X_tgt = X.loc[te_mask].reset_index(drop=True)
        ship_type = str(raw_tgt["ship_type"].iloc[0])

        logger.info(
            "=== target %d/%d vessel=%s type=%s source_rows=%d target_rows=%d ===",
            target_no, len(targets), target, ship_type, len(raw_src), len(raw_tgt),
        )

        inner_splits = vessel_group_folds(raw_src, args.inner_folds)

        for model_name in models:
            target_model_dir = detail_dir / str(target) / model_name
            target_model_dir.mkdir(parents=True, exist_ok=True)
            done_path = target_model_dir / "DONE.json"

            if args.resume and done_path.is_file():
                done = json.loads(done_path.read_text(encoding="utf-8"))
                fold_rows.extend(done["result_rows"])
                selected_rows.append(done["selected_row"])
                logger.info("Resume: %s / %s", target, model_name)
                continue

            selected, search = run_optuna_cv(
                model_name=model_name,
                locked_params=params[model_name],
                X=X_src, raw=raw_src, splits=inner_splits, runner=runner,
                n_trials=args.n_trials, seed=args.seed + target_no * 100,
                n_jobs_model=args.n_jobs,
                early_stopping_rounds=args.early_stopping_rounds,
                logger=logger,
            )
            atomic_csv(search, target_model_dir / "inner_search.csv")

            result_rows = []
            for design, use_params in [
                ("locked", params[model_name]),
                ("nested_lovo_tuned", selected),
            ]:
                model = fit_locked_model(
                    runner, model_name, use_params, X_src,
                    raw_src["target"].to_numpy(float),
                    seed=args.seed + target_no, n_jobs=args.n_jobs,
                )
                pred = np.asarray(model.predict(X_tgt), dtype=float)
                result_rows.append({
                    "model": model_name,
                    "design": design,
                    "vessel_id": target,
                    "ship_type": ship_type,
                    **metric_row(core, raw_tgt["target"].to_numpy(float), pred),
                })
                del model, pred
                gc.collect()

            locked_rmse = result_rows[0]["RMSE"]
            nested_rmse = result_rows[1]["RMSE"]
            for row in result_rows:
                row["RMSE_delta_nested_minus_locked"] = nested_rmse - locked_rmse
            fold_rows.extend(result_rows)

            selected_row = {
                "model": model_name,
                "vessel_id": target,
                "ship_type": ship_type,
                "source_vessels": int(raw_src["vessel_id"].nunique()),
                "selected_parameters_json": json.dumps(selected, sort_keys=True),
                "locked_parameters_json": json.dumps(params[model_name], sort_keys=True),
                "inner_cv_design": "GroupKFold by source-vessel ID; target vessel absent from all inner folds",
                "inner_folds": len(inner_splits),
            }
            selected_rows.append(selected_row)

            atomic_json(
                {"result_rows": result_rows, "selected_row": selected_row},
                done_path,
            )

    folds = pd.DataFrame(fold_rows)
    selected_tab = pd.DataFrame(selected_rows)
    atomic_csv(folds, out / "A2_LOVO_by_vessel_locked_vs_nested.csv")
    atomic_csv(selected_tab, out / "A2_selected_hyperparameters_by_target.csv")

    overall_rows = []
    for (model, design), g in folds.groupby(["model", "design"]):
        # Equal-vessel macro summaries are the key outer-fold summary.
        overall_rows.append({
            "model": model,
            "design": design,
            "vessels": int(g["vessel_id"].nunique()),
            "mean_vessel_RMSE": float(g["RMSE"].mean()),
            "median_vessel_RMSE": float(g["RMSE"].median()),
            "mean_vessel_R2": float(g["R2"].mean()),
            "median_vessel_R2": float(g["R2"].median()),
            "positive_R2_vessels": int((g["R2"] > 0).sum()),
        })
    overall = pd.DataFrame(overall_rows)
    atomic_csv(overall, out / "A2_LOVO_overall_locked_vs_nested.csv")

    paired = []
    for model, g in folds.groupby("model"):
        a = g[g["design"] == "locked"][["vessel_id", "ship_type", "RMSE", "R2"]].rename(
            columns={"RMSE": "RMSE_locked", "R2": "R2_locked"}
        )
        b = g[g["design"] == "nested_lovo_tuned"][["vessel_id", "ship_type", "RMSE", "R2"]].rename(
            columns={"RMSE": "RMSE_nested", "R2": "R2_nested"}
        )
        z = a.merge(b, on=["vessel_id", "ship_type"], validate="one_to_one")
        z["RMSE_delta_nested_minus_locked"] = z["RMSE_nested"] - z["RMSE_locked"]
        z["R2_delta_nested_minus_locked"] = z["R2_nested"] - z["R2_locked"]
        z.insert(0, "model", model)
        paired.append(z)
    if paired:
        atomic_csv(pd.concat(paired, ignore_index=True), out / "A2_paired_vessel_delta.csv")

    atomic_json({
        "task": "A2 strict nested LOVO sensitivity",
        "target_vessels": targets,
        "models": models,
        "scientific_boundary": (
            "Each target vessel is excluded from parameter selection and model fitting. "
            "This is a revision sensitivity analysis and does not replace the manuscript's "
            "historical locked hyperparameter registry."
        ),
    }, out / "A2_manifest.json")
    logger.info("A2 COMPLETE: %s", out)


if __name__ == "__main__":
    main()
