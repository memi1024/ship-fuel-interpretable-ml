#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A5 — equal-vessel-total-weight training robustness."""
from __future__ import annotations

import argparse
import gc
from pathlib import Path
import numpy as np
import pandas as pd

from _revision_common import (
    DEFAULT_SEED, atomic_csv, atomic_json, by_vessel_metrics,
    fit_locked_model, import_analysis_module, interventional_tree_shap,
    load_fixed31_cruise, load_locked_params, load_official_split,
    metric_row, setup_logger, shap_summary, vessel_balanced_weights,
)


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--fixed31-cruise", required=True)
    p.add_argument("--analysis-dir", default="src/paper_analysis")
    p.add_argument("--column-overrides", default="src/paper_analysis/column_overrides_fixed31.json")
    p.add_argument("--hyperparams-csv", default="src/paper_analysis/02_best_hyperparameters.csv")
    p.add_argument("--split-path", required=True)
    p.add_argument("--output-dir", default="revision_runs/A5_vessel_balanced")
    p.add_argument("--model", default="xgb", choices=["xgb", "lgbm", "rf"])
    p.add_argument("--shap-sample-n", type=int, default=12000)
    p.add_argument("--shap-background-n", type=int, default=256)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--n-jobs", type=int, default=4)
    p.add_argument("--trajectory-gap-minutes", type=float, default=30.0)
    p.add_argument("--allow-rebuild-split", action="store_true")
    p.add_argument("--allow-noncanonical-counts", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.output_dir).resolve()
    logger = setup_logger("A5", out, "A5_vessel_balanced_training.log")

    core, runner, raw, X, feature_cols, _ = load_fixed31_cruise(
        args.fixed31_cruise, args.analysis_dir, args.column_overrides,
        args.trajectory_gap_minutes, logger, args.allow_noncanonical_counts,
    )
    params = load_locked_params(runner, args.hyperparams_csv, logger)
    tr, te, split_source = load_official_split(
        args.split_path, raw, core, seed=args.seed,
        allow_rebuild=args.allow_rebuild_split, logger=logger,
    )
    rtr, rte = raw.iloc[tr].reset_index(drop=True), raw.iloc[te].reset_index(drop=True)
    xtr, xte = X.iloc[tr].reset_index(drop=True), X.iloc[te].reset_index(drop=True)
    ytr, yte = rtr["target"].to_numpy(float), rte["target"].to_numpy(float)
    weights = vessel_balanced_weights(rtr)

    models = {}
    predictions = {}
    for label, w in [("unweighted", None), ("equal_vessel_weight", weights)]:
        logger.info("Fitting %s / %s", args.model, label)
        model = fit_locked_model(
            runner, args.model, params[args.model], xtr, ytr,
            seed=args.seed, n_jobs=args.n_jobs, sample_weight=w,
        )
        pred = np.asarray(model.predict(xte), dtype=float)
        models[label] = model
        predictions[label] = pred

    perf_rows, vessel_parts = [], []
    for label, pred in predictions.items():
        perf_rows.append({"training": label, **metric_row(core, yte, pred)})
        vv = by_vessel_metrics(core, rte, pred)
        vv.insert(0, "training", label)
        vessel_parts.append(vv)
    atomic_csv(pd.DataFrame(perf_rows), out / "A5_L1_overall.csv")
    atomic_csv(pd.concat(vessel_parts, ignore_index=True), out / "A5_L1_by_vessel.csv")

    # Common held-out SHAP sample for both fits.
    shap_imp, shap_dir = [], []
    common_eval_idx = None
    for i, (label, model) in enumerate(models.items()):
        ev, sv, idx = interventional_tree_shap(
            model, xtr, xte,
            background_n=args.shap_background_n,
            sample_n=args.shap_sample_n,
            seed=args.seed + 500,
        )
        if common_eval_idx is None:
            common_eval_idx = idx
        imp, direction = shap_summary(ev, sv)
        imp.insert(0, "training", label)
        direction.insert(0, "training", label)
        shap_imp.append(imp)
        shap_dir.append(direction)
    atomic_csv(pd.concat(shap_imp, ignore_index=True), out / "A5_SHAP_global_importance.csv")
    atomic_csv(pd.concat(shap_dir, ignore_index=True), out / "A5_SHAP_direction.csv")

    cii = import_analysis_module(args.analysis_dir, "run_cii_l1_aligned_final")
    ft_parts = []
    for label, model in models.items():
        ft_v, fd_v, _, _ = cii.calculate_ft_fd(
            model=model, raw_train=rtr, raw_test=rte, feature_cols=feature_cols,
            saved_baseline_pred=predictions[label],
            reductions=[0.05, 0.10, 0.15], default_cf=3.114, logger=logger,
        )
        ft_s, combo_s = cii.create_summary(ft_v, fd_v)
        combo_s.insert(0, "training", label)
        ft_parts.append(combo_s)
    atomic_csv(pd.concat(ft_parts, ignore_index=True), out / "A5_FT_FD_summary.csv")

    atomic_csv(pd.DataFrame({
        "vessel_id": rtr["vessel_id"],
        "sample_weight": weights,
    }), out / "A5_training_weights.csv")
    atomic_json({
        "task": "A5 equal-vessel-total-weight robustness",
        "model": args.model,
        "split_source": split_source,
        "weight_definition": "each vessel receives equal total training weight; mean row weight normalized to 1",
        "note": "Locked hyperparameters are unchanged; only training observation weights differ.",
    }, out / "A5_manifest.json")
    logger.info("A5 COMPLETE: %s", out)


if __name__ == "__main__":
    main()
