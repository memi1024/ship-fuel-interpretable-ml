#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A7_joint_support_H3_sensitivity.py

Reviewer-facing A7 joint-support sensitivity for H3.

Question addressed
------------------
The original H3 speed-reduction scenarios enforce empirical support for the
modified speed, but speed-only support does not guarantee that the resulting
combination of speed, draught, trim, wind and wave height is represented in
the training data.

This script therefore compares:

    1) speed_only:
       the canonical H3 support mask;

    2) joint_qXX:
       canonical speed support INTERSECTED with a multivariable k-nearest-
       neighbour support rule learned exclusively from the official L1
       training partition.

Default joint-support variables
-------------------------------
    speed_kn
    draught_m
    trim_m
    rel_wind_speed_kn
    wave_height_m

These are the revision helper's JOINT_SUPPORT_DEFAULT_FEATURES and deliberately
exclude ship-type dummies. Joint support is estimated separately within each
ship type.

Joint-support rule
------------------
For each ship type:
  * standardise the joint-support variables using training mean/SD only;
  * build a cKDTree on the official L1 training rows;
  * for every training row, calculate the Euclidean distance to its k-th
    nearest OTHER training neighbour;
  * set the support threshold to a training-only quantile (default 0.99) of
    that k-NN distance distribution;
  * construct each test counterfactual by reducing speed only, while holding
    the other joint-support variables fixed;
  * retain the test counterfactual when its k-th-nearest training-neighbour
    distance is no larger than the learned threshold.

The same source-row support mask is used for FT and FD. This prevents FT and FD
from answering the H3 question on different source observations.

Outputs include:
  - training-only support thresholds;
  - row-, vessel-, ship-type- and fleet-level retention;
  - speed-only and joint-support FT/FD vessel endpoints;
  - vessel-bootstrap endpoint intervals;
  - pairwise ship-type contrasts;
  - cluster-robust and 100k vessel-label permutation interaction tests;
  - paired vessel-bootstrap change in H3 endpoints after imposing joint support.

This is a support/overlap sensitivity analysis. It does not make the fixed-
covariate speed-reduction scenario a voyage-level causal intervention.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from _revision_common import (
    DEFAULT_SEED,
    JOINT_SUPPORT_DEFAULT_FEATURES,
    atomic_csv,
    atomic_json,
    fit_locked_model,
    import_analysis_module,
    load_analysis_modules,
    canonical_loader_args,
    load_locked_params,
    load_official_split,
    setup_logger,
)


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--fixed31-cruise", required=True)
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
        required=True,
        help="A3_multimodel_H3_robustness_FINAL_v2_memorysafe.py",
    )
    p.add_argument("--model", default="xgb", choices=["xgb", "lgbm", "rf"])
    p.add_argument("--reductions", default="0.05,0.10,0.15")
    p.add_argument(
        "--joint-features",
        default=",".join(JOINT_SUPPORT_DEFAULT_FEATURES),
        help=(
            "Continuous joint-support variables. Ship type is handled by "
            "separate within-type support models and must not be included."
        ),
    )
    p.add_argument(
        "--joint-quantiles",
        default="0.99",
        help=(
            "Comma-separated training-only kNN-distance quantiles. "
            "Example: 0.95,0.99"
        ),
    )
    p.add_argument(
        "--knn-k",
        type=int,
        default=5,
        help="k for standardised within-ship-type kNN support distance.",
    )
    p.add_argument("--bootstrap-reps", type=int, default=2000)
    p.add_argument("--permutations", type=int, default=100000)
    p.add_argument("--seed", type=int, default=20260819)
    p.add_argument("--n-jobs", type=int, default=4)
    p.add_argument("--trajectory-gap-minutes", type=float, default=30.0)
    p.add_argument("--fd-chunk-rows", type=int, default=3000)
    p.add_argument(
        "--save-row-support-audit",
        action="store_true",
        help="Save row-level support distances/masks as gzip CSV.",
    )
    return p.parse_args()


def load_fixed31_cruise_memorysafe(
    fixed31_csv,
    analysis_dir,
    column_overrides,
    trajectory_gap_minutes,
    logger,
):
    """
    A7-local memory-safer Fixed31 loader.

    Difference from _revision_common.load_fixed31_cruise:
      * pandas CSV parsing uses low_memory=True, which avoids the large
        single-pass tokenisation buffer that can trigger C-parser OOM;
      * the source DataFrame/bundle are released immediately after
        canonicalisation.

    Canonicalisation itself is unchanged: the same f31_core_memory_safe
    canonicalize_dataframe() and the same column overrides are used.
    """
    core, runner = load_analysis_modules(analysis_dir)
    p = Path(fixed31_csv).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Fixed31 cruise CSV not found: {p}")

    logger.info("Reading Fixed31 cruise CSV [memory-safe parser]: %s", p)
    df = pd.read_csv(p, low_memory=True)

    args = canonical_loader_args(
        column_overrides,
        trajectory_gap_minutes,
    )
    bundle = core.canonicalize_dataframe(df, args, logger)
    raw = bundle.raw.reset_index(drop=True)
    X = bundle.feature_df.reset_index(drop=True)
    feature_cols = list(bundle.feature_cols)
    column_map = bundle.column_map

    del df, bundle
    gc.collect()

    # Same canonical guards used by the revision pipeline.
    if "phase" in raw.columns:
        phases = set(raw["phase"].dropna().astype(str).unique())
        if phases and phases != {"cruise"}:
            raise AssertionError(
                f"A7 requires cruise-only cohort; observed phases={sorted(phases)}"
            )
    if len(raw) != 489_620:
        raise AssertionError(
            f"Fixed31 cruise row count changed: {len(raw):,} != 489,620"
        )
    if int(raw["vessel_id"].nunique()) != 21:
        raise AssertionError(
            f"Vessel count changed: {raw['vessel_id'].nunique()} != 21"
        )

    logger.info(
        "Loaded canonical cruise cohort: rows=%d vessels=%d features=%d",
        len(raw),
        raw["vessel_id"].nunique(),
        X.shape[1],
    )
    return core, runner, raw, X, feature_cols, column_map


def load_module(path: Path, name: str):
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
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


def parse_float_list(s: str):
    return [float(x.strip()) for x in str(s).split(",") if x.strip()]


def parse_feature_list(s: str):
    return [x.strip() for x in str(s).split(",") if x.strip()]


def quantile_label(q: float) -> str:
    pct = q * 100.0
    if abs(pct - round(pct)) < 1e-10:
        return f"joint_q{int(round(pct))}"
    return f"joint_q{str(pct).replace('.', 'p')}"


def make_joint_support_models(
    raw_train: pd.DataFrame,
    features: list[str],
    quantiles: list[float],
    k: int,
    logger,
):
    """
    Learn within-ship-type scaling, cKDTree and train-only kNN thresholds.

    Train threshold uses distance to the k-th nearest OTHER training row:
    cKDTree query k+1 includes the row itself at distance 0.
    """
    if k < 1:
        raise ValueError("--knn-k must be >= 1")
    for f in features:
        if f not in raw_train.columns:
            raise KeyError(f"Joint-support feature missing from raw train: {f}")

    models = {}
    threshold_rows = []

    for ship_type, g0 in raw_train.groupby("ship_type", sort=True):
        st = normalize_ship_type(ship_type)
        g = g0.reset_index(drop=True)
        X = g[features].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        if not np.isfinite(X).all():
            raise AssertionError(
                f"Non-finite joint-support training values for {st}."
            )
        n = len(X)
        if n <= k:
            raise ValueError(f"{st}: n_train={n} <= k={k}")

        mean = X.mean(axis=0)
        sd = X.std(axis=0, ddof=1)
        bad = (~np.isfinite(sd)) | (sd <= 0)
        if np.any(bad):
            names = [features[i] for i in np.flatnonzero(bad)]
            raise AssertionError(
                f"{st}: zero/nonfinite train SD for {names}"
            )

        Z = (X - mean) / sd
        tree = cKDTree(Z)

        # self + k other neighbours => k+1 query.
        d, _ = tree.query(Z, k=k + 1, workers=-1)
        train_kdist = np.asarray(d[:, k], dtype=float)
        if not np.isfinite(train_kdist).all():
            raise AssertionError(f"{st}: nonfinite train kNN distance.")

        thresholds = {}
        for q in quantiles:
            if not (0.5 < q < 1.0):
                raise ValueError(
                    f"Joint quantile must be in (0.5,1), got {q}"
                )
            threshold = float(np.quantile(train_kdist, q))
            thresholds[q] = threshold
            threshold_rows.append({
                "ship_type": st,
                "n_train": n,
                "knn_k": int(k),
                "threshold_quantile": q,
                "threshold_distance": threshold,
                "train_kdist_median": float(np.median(train_kdist)),
                "train_kdist_p95": float(np.quantile(train_kdist, 0.95)),
                "train_kdist_p99": float(np.quantile(train_kdist, 0.99)),
                **{
                    f"mean__{f}": float(mean[j])
                    for j, f in enumerate(features)
                },
                **{
                    f"sd__{f}": float(sd[j])
                    for j, f in enumerate(features)
                },
            })

        models[st] = {
            "mean": mean,
            "sd": sd,
            "tree": tree,
            "thresholds": thresholds,
            "n_train": n,
        }
        logger.info(
            "Joint support fitted | type=%s n=%d k=%d "
            "train kdist median=%.4f p95=%.4f p99=%.4f",
            st,
            n,
            k,
            float(np.median(train_kdist)),
            float(np.quantile(train_kdist, 0.95)),
            float(np.quantile(train_kdist, 0.99)),
        )
        del X, Z, d, train_kdist
        gc.collect()

    return models, pd.DataFrame(threshold_rows)


def joint_distance_for_counterfactual(
    raw_test: pd.DataFrame,
    reduction: float,
    features: list[str],
    support_models: dict,
    k: int,
):
    """
    kNN distance for FT-style counterfactual source rows:
    speed is changed; all other joint-support variables remain fixed.
    """
    cf = raw_test[["ship_type"] + features].copy()
    cf["speed_kn"] = pd.to_numeric(
        cf["speed_kn"], errors="coerce"
    ) * (1.0 - float(reduction))

    dist = np.full(len(cf), np.nan, dtype=float)

    for ship_type, idx in cf.groupby("ship_type").groups.items():
        st = normalize_ship_type(ship_type)
        if st not in support_models:
            raise KeyError(f"No joint support model for ship type={st}")
        loc = np.asarray(list(idx), dtype=int)
        X = (
            cf.loc[loc, features]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(float)
        )
        finite = np.isfinite(X).all(axis=1)
        if finite.any():
            m = support_models[st]
            Z = (X[finite] - m["mean"]) / m["sd"]
            d, _ = m["tree"].query(Z, k=k, workers=-1)
            if k == 1:
                kth = np.asarray(d, dtype=float)
            else:
                kth = np.asarray(d[:, k - 1], dtype=float)
            dist[loc[finite]] = kth

    return dist


def build_support_masks(
    cii,
    raw_train: pd.DataFrame,
    raw_test: pd.DataFrame,
    reductions: list[float],
    features: list[str],
    quantiles: list[float],
    k: int,
    support_models: dict,
    logger,
):
    """
    Return:
      masks[scheme][reduction] -> bool array
      row_audit -> long row-level audit
      summary -> fleet/type/vessel retention tables
    """
    masks = {"speed_only": {}}
    for q in quantiles:
        masks[quantile_label(q)] = {}

    row_parts = []
    summary_rows = []

    for reduction in reductions:
        rpct = reduction * 100.0
        speed_mask = np.asarray(
            cii.support_mask(
                raw_train=raw_train,
                raw_test=raw_test,
                reduction=float(reduction),
            ),
            dtype=bool,
        )

        # Preserve the historical canonical speed-support count assertion.
        if rpct in cii.EXPECTED_SUPPORTED:
            expected = int(cii.EXPECTED_SUPPORTED[rpct])
            observed = int(speed_mask.sum())
            if observed != expected:
                raise AssertionError(
                    f"{rpct:.0f}% canonical speed support changed: "
                    f"{observed} != {expected}"
                )

        dist = joint_distance_for_counterfactual(
            raw_test,
            reduction,
            features,
            support_models,
            k,
        )
        masks["speed_only"][reduction] = speed_mask

        audit = pd.DataFrame({
            "test_row": np.arange(len(raw_test), dtype=int),
            "vessel_id": raw_test["vessel_id"].astype(str).to_numpy(),
            "ship_type": raw_test["ship_type"].map(
                normalize_ship_type
            ).to_numpy(),
            "reduction_pct": rpct,
            "speed_support": speed_mask,
            "joint_knn_distance": dist,
        })

        for q in quantiles:
            scheme = quantile_label(q)
            joint_ok = np.zeros(len(raw_test), dtype=bool)

            for st, idx in audit.groupby("ship_type").groups.items():
                loc = np.asarray(list(idx), dtype=int)
                threshold = support_models[st]["thresholds"][q]
                joint_ok[loc] = (
                    np.isfinite(dist[loc]) & (dist[loc] <= threshold)
                )

            final = speed_mask & joint_ok
            masks[scheme][reduction] = final
            audit[f"{scheme}_joint_only"] = joint_ok
            audit[f"{scheme}_final_support"] = final

            logger.info(
                "%.0f%% support | speed=%d | %s=%d | "
                "retained_vs_speed=%.2f%%",
                rpct,
                int(speed_mask.sum()),
                scheme,
                int(final.sum()),
                100.0 * final.sum() / max(int(speed_mask.sum()), 1),
            )

        row_parts.append(audit)

        # Fleet/type/vessel retention for each scheme.
        base_meta = raw_test[
            ["vessel_id", "ship_type"]
        ].copy().reset_index(drop=True)
        base_meta["ship_type"] = base_meta["ship_type"].map(
            normalize_ship_type
        )
        base_meta["speed_support"] = speed_mask

        for scheme, by_red in masks.items():
            final = by_red[reduction]
            tmp = base_meta.copy()
            tmp["final_support"] = final

            scopes = [("Fleet", tmp)]
            scopes += [
                (f"type:{st}", g.copy())
                for st, g in tmp.groupby("ship_type")
            ]
            scopes += [
                (f"vessel:{v}", g.copy())
                for v, g in tmp.groupby("vessel_id")
            ]
            for scope, g in scopes:
                ii = g.index.to_numpy(dtype=int)
                n_total = len(g)
                n_speed = int(speed_mask[ii].sum())
                n_final = int(final[ii].sum())
                summary_rows.append({
                    "scheme": scheme,
                    "reduction_pct": rpct,
                    "scope": scope,
                    "n_total_test": n_total,
                    "n_speed_supported": n_speed,
                    "n_final_supported": n_final,
                    "retained_pct_of_test":
                        100.0 * n_final / max(n_total, 1),
                    "retained_pct_of_speed_support":
                        100.0 * n_final / max(n_speed, 1),
                })

    return (
        masks,
        pd.concat(row_parts, ignore_index=True),
        pd.DataFrame(summary_rows),
    )


def calculate_ft_fd_with_masks(
    cii,
    model,
    raw_test: pd.DataFrame,
    feature_cols,
    saved_baseline_pred: np.ndarray,
    reductions,
    masks_by_reduction: dict,
    default_cf: float,
    logger,
    chunk_rows: int = 3000,
):
    """
    Exact A3 memory-safe FT/FD calculation except that the source-row support
    mask is supplied by A7.

    The SAME source-row mask is used for FT and FD.
    """
    base = raw_test.copy().reset_index(drop=True)
    if len(saved_baseline_pred) != len(base):
        raise ValueError("Baseline prediction length mismatch.")

    base["_source_row_id"] = np.arange(len(base), dtype=int)
    base["_baseline_pred"] = np.asarray(saved_baseline_pred, dtype=float)
    base["_baseline_distance_nm"] = cii.baseline_distance_old_core(base)
    base["_cf_used"] = cii.co2_factor_series(base, default_cf)
    if base["dwt"].isna().all():
        raise KeyError("DWT is required for CII analysis.")

    ft_vessel_rows = []
    fd_vessel_rows = []
    support_audit_rows = []
    interval_min = float(cii.INTERVAL_MIN)

    for reduction in reductions:
        reduction = float(reduction)
        rpct = reduction * 100.0
        speed_ratio = 1.0 - reduction
        valid = np.asarray(masks_by_reduction[reduction], dtype=bool)
        if len(valid) != len(base):
            raise AssertionError("Support mask length mismatch.")

        b = base.loc[valid].copy()
        logger.info(
            "A7 scenario calculation %.0f%% | supported rows=%d",
            rpct, len(b),
        )
        if b.empty:
            raise RuntimeError(f"{rpct:.0f}%: no supported rows.")

        ft = b.copy()
        ft["speed_kn"] = ft["speed_kn"].astype(float) * speed_ratio
        ft_pred = np.asarray(
            model.predict(
                cii.core.feature_matrix_from_raw(ft, feature_cols)
            ),
            dtype=float,
        )
        ft["_baseline_pred"] = b["_baseline_pred"].to_numpy(float)
        ft["_scenario_pred"] = ft_pred
        ratio = np.divide(
            ft["speed_kn"].to_numpy(float),
            b["speed_kn"].to_numpy(float),
            out=np.zeros(len(ft), dtype=float),
            where=b["speed_kn"].to_numpy(float) != 0.0,
        )
        ft["_scenario_distance_nm"] = (
            b["_baseline_distance_nm"].to_numpy(float) * ratio
        )

        for vessel, bg in b.groupby("vessel_id", sort=False):
            fg = ft.loc[ft["vessel_id"] == vessel].copy()
            if fg.empty:
                raise RuntimeError(
                    f"{rpct:.0f}% / {vessel}: empty FT subset."
                )

            ship_type = normalize_ship_type(bg["ship_type"].iloc[0])
            dwt_series = pd.to_numeric(bg["dwt"], errors="coerce").dropna()
            if dwt_series.empty:
                raise RuntimeError(f"{vessel}: missing DWT.")
            dwt = float(dwt_series.median())

            base_fuel = float(bg["_baseline_pred"].sum())
            base_dist = float(bg["_baseline_distance_nm"].sum())
            base_duration_min = len(bg) * interval_min
            base_co2_g = float(np.sum(
                bg["_baseline_pred"].to_numpy(float)
                * bg["_cf_used"].to_numpy(float) * 1e6
            ))
            base_tw = dwt * base_dist
            base_cii = base_co2_g / base_tw if base_tw > 0 else np.nan

            ft_fuel = float(fg["_scenario_pred"].sum())
            ft_dist = float(fg["_scenario_distance_nm"].sum())
            ft_duration_min = len(fg) * interval_min
            ft_co2_g = float(np.sum(
                fg["_scenario_pred"].to_numpy(float)
                * fg["_cf_used"].to_numpy(float) * 1e6
            ))
            ft_tw = dwt * ft_dist
            ft_cii = ft_co2_g / ft_tw if ft_tw > 0 else np.nan

            fd_fuel = 0.0
            fd_dist = 0.0
            fd_duration_min = 0.0
            fd_co2_g = 0.0
            fd_added_fuel = 0.0
            full_count = 0
            partial_count = 0
            total_count = 0

            fd_source = bg.copy()
            fd_source["distance_nm_used"] = (
                fd_source["_baseline_distance_nm"]
            )
            nsrc = len(fd_source)

            for start in range(0, nsrc, max(int(chunk_rows), 1)):
                chunk = fd_source.iloc[
                    start:start + max(int(chunk_rows), 1)
                ].copy()
                expanded = cii.fdmod.build_fd_explicit_intervals(
                    supported_base=chunk,
                    reduction=reduction,
                )
                if expanded.empty:
                    continue

                expanded = cii.fdmod.predict_fd_explicit_intervals(
                    model=model,
                    expanded=expanded,
                    feature_cols=feature_cols,
                    clip_negative_for_cii=False,
                )
                cf_map = (
                    chunk.set_index("_source_row_id")["_cf_used"].to_dict()
                )
                expanded["_cf_used"] = (
                    expanded["_source_row_id"].map(cf_map).astype(float)
                )

                fuel = expanded[
                    "_predicted_interval_fuel_t"
                ].to_numpy(float)
                fd_fuel += float(np.sum(fuel))
                fd_dist += float(
                    expanded["_interval_distance_nm"].sum()
                )
                fd_duration_min += float(
                    expanded["_interval_minutes"].sum()
                )
                fd_co2_g += float(np.sum(
                    fuel
                    * expanded["_cf_used"].to_numpy(float)
                    * 1e6
                ))
                added = expanded["_is_added_time"].astype(bool).to_numpy()
                fd_added_fuel += float(np.sum(fuel[added]))
                full_count += int(
                    (expanded["_interval_kind"] == "full_10min").sum()
                )
                partial_count += int(
                    (expanded["_interval_kind"] == "partial").sum()
                )
                total_count += int(len(expanded))

                del expanded, chunk
                gc.collect()

            if total_count == 0:
                raise RuntimeError(
                    f"{rpct:.0f}% / {vessel}: "
                    "explicit FD returned no rows."
                )

            fd_tw = dwt * fd_dist
            fd_cii = fd_co2_g / fd_tw if fd_tw > 0 else np.nan
            fd_added_time_min = fd_duration_min - base_duration_min
            ft_cii_ratio = (
                ft_cii / base_cii
                if np.isfinite(base_cii) and base_cii > 0
                else np.nan
            )
            fd_cii_ratio = (
                fd_cii / base_cii
                if np.isfinite(base_cii) and base_cii > 0
                else np.nan
            )
            fuel_rate_ratio = (
                ft_fuel / base_fuel if base_fuel > 0 else np.nan
            )
            epsilon = cii.finite_change_elasticity(
                fuel_rate_ratio=fuel_rate_ratio,
                speed_ratio=speed_ratio,
            )

            common = {
                "reduction_pct": rpct,
                "vessel_id": vessel,
                "ship_type": ship_type,
                "n_supported": int(len(bg)),
                "speed_ratio": speed_ratio,
                "DWT": dwt,
                "baseline_predicted_fuel_t": base_fuel,
                "baseline_distance_nm": base_dist,
                "baseline_duration_min": base_duration_min,
                "baseline_transport_work": base_tw,
                "baseline_CII_proxy": base_cii,
                "fuel_rate_ratio": fuel_rate_ratio,
                "finite_change_fuel_rate_elasticity": epsilon,
            }

            ft_vessel_rows.append({
                **common,
                "FT_predicted_fuel_t": ft_fuel,
                "FT_fuel_change_pct":
                    ((ft_fuel / base_fuel - 1.0) * 100.0)
                    if base_fuel > 0 else np.nan,
                "FT_distance_nm": ft_dist,
                "FT_distance_change_pct":
                    ((ft_dist / base_dist - 1.0) * 100.0)
                    if base_dist > 0 else np.nan,
                "FT_duration_min": ft_duration_min,
                "FT_duration_change_pct": 0.0,
                "FT_transport_work": ft_tw,
                "FT_CII_proxy": ft_cii,
                "FT_CII_ratio": ft_cii_ratio,
                "FT_CII_change_pct":
                    ((ft_cii_ratio - 1.0) * 100.0)
                    if np.isfinite(ft_cii_ratio) else np.nan,
                "FT_improved": bool(
                    np.isfinite(ft_cii)
                    and np.isfinite(base_cii)
                    and ft_cii < base_cii
                ),
            })

            fd_vessel_rows.append({
                **common,
                "FD_predicted_total_fuel_t": fd_fuel,
                "FD_total_fuel_change_pct":
                    ((fd_fuel / base_fuel - 1.0) * 100.0)
                    if base_fuel > 0 else np.nan,
                "FD_added_sailing_time_min": fd_added_time_min,
                "FD_duration_min": fd_duration_min,
                "FD_duration_change_pct":
                    ((fd_duration_min / base_duration_min - 1.0) * 100.0)
                    if base_duration_min > 0 else np.nan,
                "FD_added_time_fuel_t": fd_added_fuel,
                "FD_added_time_fuel_pct_of_baseline":
                    (fd_added_fuel / base_fuel * 100.0)
                    if base_fuel > 0 else np.nan,
                "FD_distance_nm": fd_dist,
                "FD_distance_error_nm": fd_dist - base_dist,
                "FD_distance_error_pct":
                    ((fd_dist / base_dist - 1.0) * 100.0)
                    if base_dist > 0 else np.nan,
                "FD_transport_work": fd_tw,
                "FD_CII_proxy": fd_cii,
                "FD_CII_ratio": fd_cii_ratio,
                "FD_CII_change_pct":
                    ((fd_cii_ratio - 1.0) * 100.0)
                    if np.isfinite(fd_cii_ratio) else np.nan,
                "FD_improved": bool(
                    np.isfinite(fd_cii)
                    and np.isfinite(base_cii)
                    and fd_cii < base_cii
                ),
                "FT_minus_FD_CII_change_pct_points":
                    ((ft_cii_ratio - fd_cii_ratio) * 100.0)
                    if np.isfinite(ft_cii_ratio)
                    and np.isfinite(fd_cii_ratio)
                    else np.nan,
                "FD_explicit_full_10min_intervals": full_count,
                "FD_explicit_partial_intervals": partial_count,
                "FD_explicit_total_intervals": total_count,
            })

            del fg, fd_source
            gc.collect()

        support_audit_rows.append({
            "reduction_pct": rpct,
            "L1_test_rows": len(base),
            "supported_rows": int(valid.sum()),
            "vessels": int(b["vessel_id"].nunique()),
            "ship_types": int(b["ship_type"].nunique()),
        })
        del b, ft
        gc.collect()

    return (
        pd.DataFrame(ft_vessel_rows),
        pd.DataFrame(fd_vessel_rows),
        pd.DataFrame(support_audit_rows),
    )


def paired_scheme_delta(
    speed_df: pd.DataFrame,
    joint_df: pd.DataFrame,
    outcome: str,
    stat_fn,
    reps: int,
    seed: int,
):
    """
    Paired vessel-bootstrap joint-support minus speed-only endpoint delta.

    If joint support removes an entire vessel, the paired comparison is
    calculated on the common vessel set and the loss is explicitly reported.
    """
    types = ["bulk", "container", "tanker"]
    rows = []

    for ri, reduction in enumerate(
        sorted(speed_df["reduction_pct"].unique())
    ):
        s0 = speed_df[
            speed_df["reduction_pct"] == reduction
        ].copy()
        j0 = joint_df[
            joint_df["reduction_pct"] == reduction
        ].copy()

        for si, scope in enumerate(["Fleet"] + types):
            sg = s0 if scope == "Fleet" else s0[s0["ship_type"] == scope]
            jg = j0 if scope == "Fleet" else j0[j0["ship_type"] == scope]

            sv = sorted(sg["vessel_id"].astype(str).unique())
            jv = sorted(jg["vessel_id"].astype(str).unique())
            common = sorted(set(sv) & set(jv))
            if not common:
                continue

            sc = sg[sg["vessel_id"].astype(str).isin(common)].copy()
            jc = jg[jg["vessel_id"].astype(str).isin(common)].copy()

            point_speed = float(stat_fn(sc))
            point_joint = float(stat_fn(jc))

            by_s = {
                v: sc[sc["vessel_id"].astype(str) == v].copy()
                for v in common
            }
            by_j = {
                v: jc[jc["vessel_id"].astype(str) == v].copy()
                for v in common
            }

            rng = np.random.default_rng(seed + ri * 100 + si)
            vals = np.empty(int(reps), dtype=float)
            for b in range(int(reps)):
                draw = rng.choice(
                    common, size=len(common), replace=True
                )
                ds = pd.concat(
                    [by_s[v] for v in draw], ignore_index=True
                )
                dj = pd.concat(
                    [by_j[v] for v in draw], ignore_index=True
                )
                vals[b] = float(stat_fn(dj) - stat_fn(ds))

            rows.append({
                "outcome": outcome,
                "reduction_pct": reduction,
                "scope": scope,
                "speed_only_vessels": len(sv),
                "joint_support_vessels": len(jv),
                "common_vessels_for_paired_delta": len(common),
                "speed_only_estimate_pct": point_speed,
                "joint_support_estimate_pct": point_joint,
                "delta_joint_minus_speed_only_pp":
                    point_joint - point_speed,
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
        "A7_joint_support",
        out,
        "A7_joint_support_H3.log",
    )

    a3 = load_module(
        Path(args.a3_module).expanduser().resolve(),
        "a3_final_for_a7",
    )

    core, runner, raw, X, feature_cols, _ = load_fixed31_cruise_memorysafe(
        args.fixed31_cruise,
        args.analysis_dir,
        args.column_overrides,
        args.trajectory_gap_minutes,
        logger,
    )
    params = load_locked_params(
        runner, args.hyperparams_csv, logger
    )
    if args.model not in params:
        raise KeyError(
            f"{args.model} absent from locked hyperparameter registry."
        )

    tr, te, split_source = load_official_split(
        args.split_path,
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

    reductions = parse_float_list(args.reductions)
    quantiles = parse_float_list(args.joint_quantiles)
    joint_features = parse_feature_list(args.joint_features)

    if "speed_kn" not in joint_features:
        raise ValueError("Joint support must include speed_kn.")
    forbidden = {
        "ship_type",
        "ship_type_bulk",
        "ship_type_container",
        "ship_type_tanker",
    }
    if forbidden.intersection(joint_features):
        raise ValueError(
            "Ship-type variables must not be in joint distance; "
            "support is already fit separately by ship type."
        )

    logger.info(
        "A7 design | model=%s | train=%d test=%d | "
        "joint_features=%s | k=%d | quantiles=%s",
        args.model,
        len(raw_tr),
        len(raw_te),
        ",".join(joint_features),
        args.knn_k,
        ",".join(map(str, quantiles)),
    )

    cii = import_analysis_module(
        args.analysis_dir, "run_cii_l1_aligned_final"
    )

    # Fit once BEFORE building support geometry, then immediately release the
    # large design matrices. A7 changes support, not model fitting.
    logger.info(
        "Fitting locked %s once on official L1 training partition...",
        args.model,
    )
    model = fit_locked_model(
        runner,
        args.model,
        params[args.model],
        X_tr,
        y_tr,
        seed=args.seed,
        n_jobs=args.n_jobs,
    )
    baseline_pred = np.asarray(
        model.predict(X_te), dtype=float
    )

    del X, X_tr, X_te, y_tr, raw, tr, te
    gc.collect()
    logger.info("Released canonical design matrices before joint-support audit.")

    # Train-only joint support geometry.
    support_models, thresholds = make_joint_support_models(
        raw_tr,
        joint_features,
        quantiles,
        args.knn_k,
        logger,
    )
    atomic_csv(
        thresholds,
        out / "A7_joint_support_thresholds.csv",
    )

    # Canonical speed support + joint support.
    masks, row_audit, retention = build_support_masks(
        cii,
        raw_tr,
        raw_te,
        reductions,
        joint_features,
        quantiles,
        args.knn_k,
        support_models,
        logger,
    )
    atomic_csv(
        retention,
        out / "A7_support_retention_all_scopes.csv",
    )
    if args.save_row_support_audit:
        row_audit.to_csv(
            out / "A7_row_support_audit.csv.gz",
            index=False,
            compression="gzip",
        )

    all_ft = []
    all_fd = []
    all_support = []
    all_endpoint_rows = []
    all_pair_rows = []
    all_interaction_rows = []
    scheme_ft = {}
    scheme_fd = {}

    schemes = ["speed_only"] + [
        quantile_label(q)
        for q in quantiles
    ]

    for si, scheme in enumerate(schemes):
        logger.info("=== A7 support scheme: %s ===", scheme)
        ft, fd, supp = calculate_ft_fd_with_masks(
            cii=cii,
            model=model,
            raw_test=raw_te,
            feature_cols=feature_cols,
            saved_baseline_pred=baseline_pred,
            reductions=reductions,
            masks_by_reduction=masks[scheme],
            default_cf=3.114,
            logger=logger,
            chunk_rows=args.fd_chunk_rows,
        )
        ft["ship_type"] = ft["ship_type"].map(normalize_ship_type)
        fd["ship_type"] = fd["ship_type"].map(normalize_ship_type)
        ft.insert(0, "support_scheme", scheme)
        fd.insert(0, "support_scheme", scheme)
        supp.insert(0, "support_scheme", scheme)

        atomic_csv(
            ft,
            out / f"A7_{scheme}_FT_by_vessel.csv",
        )
        atomic_csv(
            fd,
            out / f"A7_{scheme}_FD_by_vessel.csv",
        )
        atomic_csv(
            supp,
            out / f"A7_{scheme}_scenario_support.csv",
        )

        scheme_ft[scheme] = ft.drop(
            columns=["support_scheme"]
        ).copy()
        scheme_fd[scheme] = fd.drop(
            columns=["support_scheme"]
        ).copy()

        ep, pairs, inter = a3.h3_inference_for_model(
            model_name=args.model,
            ft=scheme_ft[scheme],
            fd=scheme_fd[scheme],
            bootstrap_reps=args.bootstrap_reps,
            permutations=args.permutations,
            seed=args.seed + si * 10000,
        )
        for r in ep:
            r["support_scheme"] = scheme
        for r in pairs:
            r["support_scheme"] = scheme
        for r in inter:
            r["support_scheme"] = scheme
        all_endpoint_rows.extend(ep)
        all_pair_rows.extend(pairs)
        all_interaction_rows.extend(inter)
        all_ft.append(ft)
        all_fd.append(fd)
        all_support.append(supp)

    atomic_csv(
        pd.concat(all_ft, ignore_index=True),
        out / "A7_FT_by_vessel_all_support_schemes.csv",
    )
    atomic_csv(
        pd.concat(all_fd, ignore_index=True),
        out / "A7_FD_by_vessel_all_support_schemes.csv",
    )
    atomic_csv(
        pd.concat(all_support, ignore_index=True),
        out / "A7_scenario_support_all_schemes.csv",
    )
    atomic_csv(
        pd.DataFrame(all_endpoint_rows),
        out / "A7_H3_endpoint_vessel_bootstrap_CI.csv",
    )
    atomic_csv(
        pd.DataFrame(all_pair_rows),
        out / "A7_H3_pairwise_shiptype_contrasts.csv",
    )
    atomic_csv(
        pd.DataFrame(all_interaction_rows),
        out / "A7_H3_shiptype_x_reduction_interaction.csv",
    )

    # Direct paired comparison of speed-only vs each joint support criterion.
    delta_rows = []
    for qi, q in enumerate(quantiles):
        scheme = quantile_label(q)

        rows = paired_scheme_delta(
            scheme_ft["speed_only"],
            scheme_ft[scheme],
            outcome="FT_CII_change_pct",
            stat_fn=a3.ft_weighted_cii_change,
            reps=args.bootstrap_reps,
            seed=args.seed + 50000 + qi * 1000,
        )
        for r in rows:
            r["joint_support_scheme"] = scheme
            r["analysis_role"] = "formal_H3"
        delta_rows.extend(rows)

        rows = paired_scheme_delta(
            scheme_fd["speed_only"],
            scheme_fd[scheme],
            outcome="FD_total_fuel_change_pct",
            stat_fn=a3.fd_total_fuel_change,
            reps=args.bootstrap_reps,
            seed=args.seed + 60000 + qi * 1000,
        )
        for r in rows:
            r["joint_support_scheme"] = scheme
            r["analysis_role"] = "sensitivity"
        delta_rows.extend(rows)

    atomic_csv(
        pd.DataFrame(delta_rows),
        out / "A7_speed_only_vs_joint_paired_delta.csv",
    )

    # Compact type/fleet retention table for manuscript/rebuttal.
    compact = retention[
        retention["scope"].str.startswith("type:")
        | retention["scope"].eq("Fleet")
    ].copy()
    atomic_csv(
        compact,
        out / "A7_support_retention_fleet_and_type.csv",
    )

    manifest = {
        "task": "A7 joint-support H3 sensitivity",
        "reviewer_issue": (
            "Speed-only empirical support does not guarantee support for "
            "the full speed/draught/trim/wind/wave counterfactual state."
        ),
        "model": args.model,
        "model_selection": (
            "locked manuscript hyperparameters; no retuning"
        ),
        "split_source": split_source,
        "train_rows": int(len(raw_tr)),
        "test_rows": int(len(raw_te)),
        "joint_features": joint_features,
        "ship_type_in_distance": False,
        "support_fit_scope": (
            "separate within-ship-type models using official L1 training "
            "rows only"
        ),
        "distance": (
            "Euclidean distance after within-ship-type train z-standardisation"
        ),
        "knn_k": int(args.knn_k),
        "threshold_definition": (
            "quantile of each training row's distance to its k-th nearest "
            "OTHER training neighbour"
        ),
        "joint_quantiles": quantiles,
        "reductions": reductions,
        "support_intersection": (
            "canonical speed support AND joint kNN support"
        ),
        "same_source_mask_for_FT_and_FD": True,
        "formal_H3_endpoint": "FT_CII_change_pct",
        "FD_role": "sensitivity/mechanism evidence",
        "bootstrap_reps": int(args.bootstrap_reps),
        "permutations": int(args.permutations),
        "fd_chunk_rows": int(args.fd_chunk_rows),
        "interpretation": (
            "Support/overlap sensitivity for a fixed-covariate local "
            "scenario; not a realistic voyage-level causal intervention."
        ),
    }
    atomic_json(
        manifest,
        out / "A7_manifest.json",
    )

    logger.info("A7 COMPLETE: %s", out)


if __name__ == "__main__":
    main()
