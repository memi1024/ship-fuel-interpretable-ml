#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A3 — Final reviewer-facing multi-model H3 robustness.

Purpose
-------
Repeat the manuscript H3 operational sensitivity/inference chain for
XGBoost, LightGBM and Random Forest using:

1. the same official L1 split;
2. each model's locked manuscript hyperparameters;
3. the same final FT/FD scenario implementation and support logic;
4. vessel-cluster bootstrap endpoint intervals;
5. pairwise ship-type contrasts;
6. cluster-robust ship-type x reduction interaction tests; and
7. vessel-label permutation interaction tests.

This is a revision-only robustness analysis. It does not replace the
historical locked model registry or the manuscript's primary XGBoost chain.
FT remains the formal H3 endpoint; FD remains sensitivity/mechanism evidence.
"""
from __future__ import annotations

import argparse
import gc
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from _revision_common import (
    atomic_csv,
    atomic_json,
    fit_locked_model,
    import_analysis_module,
    load_fixed31_cruise,
    load_locked_params,
    load_official_split,
    setup_logger,
)


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--fixed31-cruise", required=True)
    p.add_argument("--analysis-dir", default="src/paper_analysis")
    p.add_argument("--column-overrides", default="src/paper_analysis/column_overrides_fixed31.json")
    p.add_argument("--hyperparams-csv", default="src/paper_analysis/02_best_hyperparameters.csv")
    p.add_argument(
        "--split-path",
        required=True,
        help="Historical 14_artifacts/record_split_indices.npz is strongly preferred.",
    )
    p.add_argument("--output-dir", default="revision_runs/A3_multimodel_H3_final")
    p.add_argument("--models", default="xgb,lgbm,rf")
    p.add_argument("--reductions", default="0.05,0.10,0.15")
    p.add_argument("--bootstrap-reps", type=int, default=2000)
    p.add_argument("--permutations", type=int, default=100000)
    p.add_argument("--seed", type=int, default=20260819)
    p.add_argument("--n-jobs", type=int, default=4)
    p.add_argument("--trajectory-gap-minutes", type=float, default=30.0)
    p.add_argument("--allow-rebuild-split", action="store_true")
    p.add_argument("--allow-noncanonical-counts", action="store_true")
    p.add_argument(
        "--save-fd-interval-audit",
        action="store_true",
        help="Save the very large explicit-FD interval audit for each model. Off by default.",
    )
    p.add_argument(
        "--resume-existing-models",
        action="store_true",
        help="Reuse existing A3_<model>_FT/FD_by_vessel.csv files instead of refitting that model.",
    )
    p.add_argument(
        "--memory-safe-fd",
        action="store_true",
        help="Use exact chunked explicit-FD prediction/aggregation to reduce RAM use.",
    )
    p.add_argument(
        "--fd-chunk-rows",
        type=int,
        default=3000,
        help="Source rows per chunk for memory-safe explicit-FD construction.",
    )
    return p.parse_args()


def normalize_ship_type(x) -> str:
    s = str(x).strip().lower()
    if "bulk" in s:
        return "bulk"
    if "container" in s:
        return "container"
    if "tank" in s:
        return "tanker"
    return s


def calculate_ft_fd_memory_safe(
    cii,
    model,
    raw_train: pd.DataFrame,
    raw_test: pd.DataFrame,
    feature_cols,
    saved_baseline_pred: np.ndarray,
    reductions,
    default_cf: float,
    logger,
    chunk_rows: int = 3000,
):
    """Exact memory-safe equivalent of cii.calculate_ft_fd().

    FT is unchanged. FD still uses the canonical explicit-interval builder and
    predictor, but materialises/predicts one vessel and a small source-row chunk
    at a time, immediately aggregates the interval results, and discards the
    chunk. This preserves the explicit full-10-min + final-partial calculation
    while avoiding fleet-wide expanded DataFrames and retained interval audits.
    """
    base = raw_test.copy().reset_index(drop=True)
    if len(saved_baseline_pred) != len(base):
        raise ValueError("Saved baseline prediction length does not match L1 test.")

    base["_source_row_id"] = np.arange(len(base), dtype=int)
    base["_baseline_pred"] = np.asarray(saved_baseline_pred, dtype=float)
    base["_baseline_distance_nm"] = cii.baseline_distance_old_core(base)
    base["_cf_used"] = cii.co2_factor_series(base, default_cf)
    if base["dwt"].isna().all():
        raise KeyError("DWT is required for CII analysis.")

    ft_vessel_rows, fd_vessel_rows, support_audit_rows = [], [], []
    interval_min = float(cii.INTERVAL_MIN)

    for reduction in reductions:
        reduction = float(reduction)
        rpct = reduction * 100.0
        speed_ratio = 1.0 - reduction
        logger.info("Running %.0f%% speed reduction [memory-safe FD]...", rpct)

        valid = cii.support_mask(raw_train=raw_train, raw_test=base, reduction=reduction)
        n_supported = int(valid.sum())
        if rpct in cii.EXPECTED_SUPPORTED:
            expected_n = cii.EXPECTED_SUPPORTED[rpct]
            if n_supported != expected_n:
                raise AssertionError(
                    f"{rpct:.0f}% support count changed: current={n_supported}, "
                    f"expected={expected_n}. Stop because this is no longer the same scenario sample."
                )

        b = base.loc[valid].copy()

        # Fixed-Time: identical to canonical implementation.
        ft = b.copy()
        ft["speed_kn"] = ft["speed_kn"].astype(float) * speed_ratio
        ft_pred = np.asarray(
            model.predict(cii.core.feature_matrix_from_raw(ft, feature_cols)), dtype=float
        )
        ft["_baseline_pred"] = b["_baseline_pred"].to_numpy(float)
        ft["_scenario_pred"] = ft_pred
        ratio = np.divide(
            ft["speed_kn"].to_numpy(float),
            b["speed_kn"].to_numpy(float),
            out=np.zeros(len(ft), dtype=float),
            where=b["speed_kn"].to_numpy(float) != 0.0,
        )
        ft["_scenario_distance_nm"] = b["_baseline_distance_nm"].to_numpy(float) * ratio

        # Process FD independently vessel-by-vessel and chunk-by-chunk.
        for vessel, bg in b.groupby("vessel_id", sort=False):
            fg = ft.loc[ft["vessel_id"] == vessel].copy()
            if fg.empty:
                raise RuntimeError(f"{rpct:.0f}% / {vessel}: empty FT subset.")

            ship_type = str(bg["ship_type"].iloc[0])
            dwt = float(pd.to_numeric(bg["dwt"], errors="coerce").dropna().median())

            base_fuel = float(bg["_baseline_pred"].sum())
            base_dist = float(bg["_baseline_distance_nm"].sum())
            base_duration_min = len(bg) * interval_min
            base_co2_g = float(np.sum(
                bg["_baseline_pred"].to_numpy(float)
                * bg["_cf_used"].to_numpy(float) * 1e6
            ))
            base_tw = dwt * base_dist
            base_cii = base_co2_g / base_tw if base_tw > 0.0 else np.nan

            ft_fuel = float(fg["_scenario_pred"].sum())
            ft_dist = float(fg["_scenario_distance_nm"].sum())
            ft_duration_min = len(fg) * interval_min
            ft_co2_g = float(np.sum(
                fg["_scenario_pred"].to_numpy(float)
                * fg["_cf_used"].to_numpy(float) * 1e6
            ))
            ft_tw = dwt * ft_dist
            ft_cii = ft_co2_g / ft_tw if ft_tw > 0.0 else np.nan

            fd_fuel = 0.0
            fd_dist = 0.0
            fd_duration_min = 0.0
            fd_co2_g = 0.0
            fd_added_fuel = 0.0
            full_count = partial_count = total_count = 0

            fd_source = bg.copy()
            fd_source["distance_nm_used"] = fd_source["_baseline_distance_nm"]
            nsrc = len(fd_source)
            for start in range(0, nsrc, max(int(chunk_rows), 1)):
                chunk = fd_source.iloc[start:start + max(int(chunk_rows), 1)].copy()
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
                cf_map = chunk.set_index("_source_row_id")["_cf_used"].to_dict()
                expanded["_cf_used"] = expanded["_source_row_id"].map(cf_map).astype(float)

                fuel = expanded["_predicted_interval_fuel_t"].to_numpy(float)
                fd_fuel += float(np.sum(fuel))
                fd_dist += float(expanded["_interval_distance_nm"].sum())
                fd_duration_min += float(expanded["_interval_minutes"].sum())
                fd_co2_g += float(np.sum(fuel * expanded["_cf_used"].to_numpy(float) * 1e6))
                added = expanded["_is_added_time"].astype(bool).to_numpy()
                fd_added_fuel += float(np.sum(fuel[added]))
                full_count += int((expanded["_interval_kind"] == "full_10min").sum())
                partial_count += int((expanded["_interval_kind"] == "partial").sum())
                total_count += int(len(expanded))

                del expanded, chunk
                gc.collect()

            if total_count == 0:
                raise RuntimeError(f"{rpct:.0f}% / {vessel}: explicit FD expansion returned no rows.")

            fd_tw = dwt * fd_dist
            fd_cii = fd_co2_g / fd_tw if fd_tw > 0.0 else np.nan
            fd_added_time_min = fd_duration_min - base_duration_min
            ft_cii_ratio = ft_cii / base_cii if np.isfinite(base_cii) and base_cii > 0 else np.nan
            fd_cii_ratio = fd_cii / base_cii if np.isfinite(base_cii) and base_cii > 0 else np.nan
            fuel_rate_ratio = ft_fuel / base_fuel if base_fuel > 0 else np.nan
            epsilon = cii.finite_change_elasticity(
                fuel_rate_ratio=fuel_rate_ratio,
                speed_ratio=speed_ratio,
            )
            retained_pct = len(bg) / max(int((base["vessel_id"] == vessel).sum()), 1) * 100.0

            common = {
                "reduction_pct": rpct,
                "vessel_id": vessel,
                "ship_type": ship_type,
                "n_supported": int(len(bg)),
                "retained_pct": retained_pct,
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
                "FT_fuel_change_pct": ((ft_fuel / base_fuel - 1.0) * 100.0) if base_fuel > 0 else np.nan,
                "FT_distance_nm": ft_dist,
                "FT_distance_change_pct": ((ft_dist / base_dist - 1.0) * 100.0) if base_dist > 0 else np.nan,
                "FT_duration_min": ft_duration_min,
                "FT_duration_change_pct": 0.0,
                "FT_transport_work": ft_tw,
                "FT_CII_proxy": ft_cii,
                "FT_CII_ratio": ft_cii_ratio,
                "FT_CII_change_pct": ((ft_cii_ratio - 1.0) * 100.0) if np.isfinite(ft_cii_ratio) else np.nan,
                "FT_improved": bool(np.isfinite(ft_cii) and np.isfinite(base_cii) and ft_cii < base_cii),
            })
            fd_vessel_rows.append({
                **common,
                "FD_predicted_total_fuel_t": fd_fuel,
                "FD_total_fuel_change_pct": ((fd_fuel / base_fuel - 1.0) * 100.0) if base_fuel > 0 else np.nan,
                "FD_added_sailing_time_min": fd_added_time_min,
                "FD_duration_min": fd_duration_min,
                "FD_duration_change_pct": ((fd_duration_min / base_duration_min - 1.0) * 100.0) if base_duration_min > 0 else np.nan,
                "FD_added_time_fuel_t": fd_added_fuel,
                "FD_added_time_fuel_pct_of_baseline": (fd_added_fuel / base_fuel * 100.0) if base_fuel > 0 else np.nan,
                "FD_distance_nm": fd_dist,
                "FD_distance_error_nm": fd_dist - base_dist,
                "FD_distance_error_pct": ((fd_dist / base_dist - 1.0) * 100.0) if base_dist > 0 else np.nan,
                "FD_transport_work": fd_tw,
                "FD_CII_proxy": fd_cii,
                "FD_CII_ratio": fd_cii_ratio,
                "FD_CII_change_pct": ((fd_cii_ratio - 1.0) * 100.0) if np.isfinite(fd_cii_ratio) else np.nan,
                "FD_improved": bool(np.isfinite(fd_cii) and np.isfinite(base_cii) and fd_cii < base_cii),
                "FT_minus_FD_CII_change_pct_points": ((ft_cii_ratio - fd_cii_ratio) * 100.0) if np.isfinite(ft_cii_ratio) and np.isfinite(fd_cii_ratio) else np.nan,
                "FD_explicit_full_10min_intervals": full_count,
                "FD_explicit_partial_intervals": partial_count,
                "FD_explicit_total_intervals": total_count,
            })

            del fg, fd_source
            gc.collect()

        support_audit_rows.append({
            "reduction_pct": rpct,
            "L1_test_rows": len(base),
            "supported_rows_A": n_supported,
            "supported_rows_B": n_supported,
            "same_support_mask_A_B": True,
            "expected_supported_rows": cii.EXPECTED_SUPPORTED.get(rpct, np.nan),
            "supported_count_matches_expected": (
                n_supported == cii.EXPECTED_SUPPORTED[rpct]
                if rpct in cii.EXPECTED_SUPPORTED else np.nan
            ),
            "vessels_A_B": int(b["vessel_id"].nunique()),
        })
        del b, ft
        gc.collect()

    return (
        pd.DataFrame(ft_vessel_rows),
        pd.DataFrame(fd_vessel_rows),
        pd.DataFrame(support_audit_rows),
        pd.DataFrame(),
    )


# -----------------------------------------------------------------------------
# H3 endpoint estimands — copied/aligned to run_final_manuscript_inference.py
# -----------------------------------------------------------------------------

def ft_weighted_cii_change(g: pd.DataFrame) -> float:
    """Group/fleet weighted FT CII-proxy change, matching final inference."""
    b_f = g["baseline_predicted_fuel_t"].sum()
    b_tw = g["baseline_transport_work"].sum()
    f_f = g["FT_predicted_fuel_t"].sum()
    f_tw = g["FT_transport_work"].sum()
    if b_f <= 0 or b_tw <= 0 or f_tw <= 0:
        return np.nan
    return float(((f_f / f_tw) / (b_f / b_tw) - 1.0) * 100.0)


def fd_total_fuel_change(g: pd.DataFrame) -> float:
    """Group/fleet FD total-fuel change, matching final inference."""
    base = g["baseline_predicted_fuel_t"].sum()
    if base <= 0:
        return np.nan
    return float((g["FD_predicted_total_fuel_t"].sum() / base - 1.0) * 100.0)


def bootstrap_vessel_aggregate(g: pd.DataFrame, stat_fn, reps: int, seed: int):
    vessels = sorted(g["vessel_id"].astype(str).unique())
    if not vessels:
        return np.nan, np.nan, np.nan
    by = {v: g[g["vessel_id"].astype(str) == v].copy() for v in vessels}
    point = float(stat_fn(g))
    rng = np.random.default_rng(seed)
    vals = np.empty(reps, dtype=float)
    for i in range(reps):
        draw = rng.choice(vessels, size=len(vessels), replace=True)
        vals[i] = stat_fn(pd.concat([by[v] for v in draw], ignore_index=True))
    return point, float(np.nanquantile(vals, 0.025)), float(np.nanquantile(vals, 0.975))


def independent_bootstrap_contrast(ga: pd.DataFrame, gb: pd.DataFrame, stat_fn, reps: int, seed: int):
    va = sorted(ga["vessel_id"].astype(str).unique())
    vb = sorted(gb["vessel_id"].astype(str).unique())
    if not va or not vb:
        return np.nan, np.nan, np.nan
    a = {v: ga[ga["vessel_id"].astype(str) == v].copy() for v in va}
    b = {v: gb[gb["vessel_id"].astype(str) == v].copy() for v in vb}
    point = float(stat_fn(ga) - stat_fn(gb))
    rng = np.random.default_rng(seed)
    vals = np.empty(reps, dtype=float)
    for i in range(reps):
        da = rng.choice(va, size=len(va), replace=True)
        db = rng.choice(vb, size=len(vb), replace=True)
        vals[i] = stat_fn(pd.concat([a[v] for v in da], ignore_index=True)) - stat_fn(
            pd.concat([b[v] for v in db], ignore_index=True)
        )
    return point, float(np.nanquantile(vals, 0.025)), float(np.nanquantile(vals, 0.975))


# -----------------------------------------------------------------------------
# Interaction inference — aligned to final manuscript inference implementation
# -----------------------------------------------------------------------------

def cluster_robust_interaction(df: pd.DataFrame, outcome: str):
    try:
        import statsmodels.formula.api as smf
    except ImportError as exc:
        raise ImportError("A3 final inference requires statsmodels.") from exc

    x = df[["vessel_id", "ship_type", "reduction_pct", outcome]].dropna().copy()
    x["reduction_cat"] = x["reduction_pct"].astype(str)
    fit = smf.ols(f"{outcome} ~ C(reduction_cat) * C(ship_type)", data=x).fit(
        cov_type="cluster",
        cov_kwds={"groups": x["vessel_id"].astype(str)},
    )
    names = list(fit.params.index)
    idx = [i for i, name in enumerate(names) if ":" in name]
    if not idx:
        raise RuntimeError(f"No interaction coefficients found for {outcome}.")
    R = np.zeros((len(idx), len(names)))
    for r, i in enumerate(idx):
        R[r, i] = 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        test = fit.f_test(R)
    return (
        float(np.asarray(test.fvalue).squeeze()),
        float(np.asarray(test.pvalue).squeeze()),
        float(test.df_num),
        float(test.df_denom),
    )


def ordinary_interaction_f(df: pd.DataFrame, outcome: str) -> float:
    y = df[outcome].to_numpy(float)
    reductions = sorted(df["reduction_pct"].unique())
    ship_types = ["bulk", "container", "tanker"]

    r_dummies = [
        (df["reduction_pct"].to_numpy(float) == float(r)).astype(float)
        for r in reductions[1:]
    ]
    s_dummies = [
        (df["ship_type"].astype(str).to_numpy() == s).astype(float)
        for s in ship_types[1:]
    ]
    reduced = [np.ones(len(df))] + r_dummies + s_dummies
    full = list(reduced) + [rd * sd for rd in r_dummies for sd in s_dummies]
    Xr, Xf = np.column_stack(reduced), np.column_stack(full)
    br, *_ = np.linalg.lstsq(Xr, y, rcond=None)
    bf, *_ = np.linalg.lstsq(Xf, y, rcond=None)
    er, ef = y - Xr @ br, y - Xf @ bf
    ssr_r, ssr_f = float(er @ er), float(ef @ ef)
    df_num = Xf.shape[1] - Xr.shape[1]
    df_den = len(y) - Xf.shape[1]
    if df_num <= 0 or df_den <= 0 or ssr_f <= 0:
        return np.nan
    return float(((ssr_r - ssr_f) / df_num) / (ssr_f / df_den))


def interaction_permutation(df: pd.DataFrame, outcome: str, reps: int, seed: int):
    """
    Vessel-label permutation test used in the final manuscript inference.

    Ship-type labels are permuted at vessel level, preserving the three repeated
    reduction observations belonging to each vessel.
    """
    vessels = (
        df[["vessel_id", "ship_type"]]
        .drop_duplicates()
        .sort_values("vessel_id")
    )
    vids = vessels["vessel_id"].astype(str).to_numpy()
    labels = vessels["ship_type"].astype(str).to_numpy()
    observed = ordinary_interaction_f(df, outcome)
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(reps):
        mapping = dict(zip(vids, rng.permutation(labels)))
        z = df.copy()
        z["ship_type"] = z["vessel_id"].astype(str).map(mapping)
        exceed += int(ordinary_interaction_f(z, outcome) >= observed)
    return observed, float((exceed + 1.0) / (reps + 1.0))


def h3_inference_for_model(
    model_name: str,
    ft: pd.DataFrame,
    fd: pd.DataFrame,
    bootstrap_reps: int,
    permutations: int,
    seed: int,
):
    """Return endpoint CIs, pairwise contrasts, and global interaction tests."""
    types = ["bulk", "container", "tanker"]
    endpoint_rows = []
    pair_rows = []
    interaction_rows = []

    analyses = [
        ("FT_CII_change_pct", ft, ft_weighted_cii_change, "formal_H3"),
        ("FD_total_fuel_change_pct", fd, fd_total_fuel_change, "sensitivity"),
    ]

    for oi, (outcome, data, stat_fn, role) in enumerate(analyses):
        for ri, reduction in enumerate(sorted(data["reduction_pct"].unique())):
            z = data[data["reduction_pct"] == reduction].copy()

            for si, scope in enumerate(["Fleet"] + types):
                g = z if scope == "Fleet" else z[z["ship_type"] == scope]
                point, lo, hi = bootstrap_vessel_aggregate(
                    g,
                    stat_fn,
                    bootstrap_reps,
                    seed + 2000 + oi * 500 + ri * 100 + si,
                )
                endpoint_rows.append({
                    "model": model_name,
                    "analysis_role": role,
                    "outcome": outcome,
                    "reduction_pct": reduction,
                    "scope": scope,
                    "n_vessels": int(g["vessel_id"].nunique()),
                    "estimate_pct": point,
                    "vessel_bootstrap_CI95_low": lo,
                    "vessel_bootstrap_CI95_high": hi,
                })

            for ai, a in enumerate(types):
                for b in types[ai + 1:]:
                    ga = z[z["ship_type"] == a]
                    gb = z[z["ship_type"] == b]
                    point, lo, hi = independent_bootstrap_contrast(
                        ga,
                        gb,
                        stat_fn,
                        bootstrap_reps,
                        seed + 3000 + oi * 500 + ri * 100 + ai * 10 + types.index(b),
                    )
                    pair_rows.append({
                        "model": model_name,
                        "analysis_role": role,
                        "outcome": outcome,
                        "reduction_pct": reduction,
                        "contrast": f"{a}-{b}",
                        "estimate_difference_pp": point,
                        "vessel_bootstrap_CI95_low": lo,
                        "vessel_bootstrap_CI95_high": hi,
                        "CI_excludes_zero": bool(np.isfinite(lo) and np.isfinite(hi) and (lo > 0 or hi < 0)),
                    })

        reg = data[["vessel_id", "ship_type", "reduction_pct", outcome]].dropna().copy()
        F, p, df_num, df_den = cluster_robust_interaction(reg, outcome)
        perm_F, perm_p = interaction_permutation(
            reg,
            outcome,
            permutations,
            seed + 4000 + oi,
        )
        interaction_rows.append({
            "model": model_name,
            "analysis_role": role,
            "outcome": outcome,
            "cluster_robust_F": F,
            "df_num": df_num,
            "df_denom": df_den,
            "cluster_robust_p": p,
            "permutation_F": perm_F,
            "permutation_p": perm_p,
            "permutations": permutations,
            "cluster_unit": "vessel_id",
            "permutation_unit": "vessel_id ship-type label",
        })

    return endpoint_rows, pair_rows, interaction_rows


def main():
    args = parse_args()
    out = Path(args.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("A3_final", out, "A3_multimodel_H3_FINAL.log")

    core, runner, raw, X, feature_cols, _ = load_fixed31_cruise(
        args.fixed31_cruise,
        args.analysis_dir,
        args.column_overrides,
        args.trajectory_gap_minutes,
        logger,
        args.allow_noncanonical_counts,
    )
    params = load_locked_params(runner, args.hyperparams_csv, logger)
    tr, te, split_source = load_official_split(
        args.split_path,
        raw,
        core,
        seed=args.seed,
        allow_rebuild=args.allow_rebuild_split,
        logger=logger,
    )

    raw_tr = raw.iloc[tr].reset_index(drop=True)
    raw_te = raw.iloc[te].reset_index(drop=True)
    X_tr = X.iloc[tr].reset_index(drop=True)
    X_te = X.iloc[te].reset_index(drop=True)
    y_tr = raw_tr["target"].to_numpy(float)
    del raw, X, tr, te
    gc.collect()

    cii = import_analysis_module(args.analysis_dir, "run_cii_l1_aligned_final")
    reductions = [float(x) for x in args.reductions.split(",") if x.strip()]
    models = [m.strip().lower() for m in args.models.split(",") if m.strip()]

    if set(models) != {"xgb", "lgbm", "rf"}:
        logger.warning(
            "Reviewer-facing A3 is intended for xgb,lgbm,rf; requested models=%s",
            models,
        )

    all_ft = []
    all_fd = []
    all_summary = []
    all_endpoints = []
    all_pairs = []
    all_interactions = []
    model_status = []

    for mi, model_name in enumerate(models):
        if model_name not in params:
            raise KeyError(f"Model missing from locked registry: {model_name}")

        logger.info("=== A3 FINAL model=%s ===", model_name)
        ft_path = out / f"A3_{model_name}_FT_by_vessel.csv"
        fd_path = out / f"A3_{model_name}_FD_by_vessel.csv"
        support_path = out / f"A3_{model_name}_support_audit.csv"
        summary_path = out / f"A3_{model_name}_FT_FD_summary.csv"

        reuse = bool(args.resume_existing_models and ft_path.exists() and fd_path.exists())
        if reuse:
            logger.info("Reusing completed scenario outputs for %s: %s / %s", model_name, ft_path, fd_path)
            ft_v = pd.read_csv(ft_path)
            fd_v = pd.read_csv(fd_path)
            support_audit = pd.read_csv(support_path) if support_path.exists() else pd.DataFrame()
            if "model" not in ft_v.columns:
                ft_v.insert(0, "model", model_name)
            if "model" not in fd_v.columns:
                fd_v.insert(0, "model", model_name)
            if not support_audit.empty and "model" not in support_audit.columns:
                support_audit.insert(0, "model", model_name)
            ft_v["ship_type"] = ft_v["ship_type"].map(normalize_ship_type)
            fd_v["ship_type"] = fd_v["ship_type"].map(normalize_ship_type)
            if summary_path.exists():
                combo_s = pd.read_csv(summary_path)
                if "model" not in combo_s.columns:
                    combo_s.insert(0, "model", model_name)
            else:
                _, combo_s = cii.create_summary(ft_v.drop(columns=["model"]), fd_v.drop(columns=["model"]))
                combo_s.insert(0, "model", model_name)
            interval_audit = pd.DataFrame()
            model = None
            base_pred = None
        else:
            logger.info("Fitting locked %s on official L1 training partition...", model_name)
            model = fit_locked_model(
                runner, model_name, params[model_name], X_tr, y_tr,
                seed=args.seed, n_jobs=args.n_jobs,
            )
            base_pred = np.asarray(model.predict(X_te), dtype=float)

            # Free the large design matrices before scenario expansion once no later
            # model still needs to be fitted (important for RF recovery runs).
            if mi == len(models) - 1:
                del X_tr, X_te, y_tr
                gc.collect()

            logger.info("Running final FT/FD scenario machinery for %s...", model_name)
            if args.memory_safe_fd:
                ft_v, fd_v, support_audit, interval_audit = calculate_ft_fd_memory_safe(
                    cii=cii, model=model, raw_train=raw_tr, raw_test=raw_te,
                    feature_cols=feature_cols, saved_baseline_pred=base_pred,
                    reductions=reductions, default_cf=3.114, logger=logger,
                    chunk_rows=args.fd_chunk_rows,
                )
            else:
                ft_v, fd_v, support_audit, interval_audit = cii.calculate_ft_fd(
                    model=model, raw_train=raw_tr, raw_test=raw_te,
                    feature_cols=feature_cols, saved_baseline_pred=base_pred,
                    reductions=reductions, default_cf=3.114, logger=logger,
                )

            ft_v = ft_v.copy(); fd_v = fd_v.copy(); support_audit = support_audit.copy()
            ft_v["ship_type"] = ft_v["ship_type"].map(normalize_ship_type)
            fd_v["ship_type"] = fd_v["ship_type"].map(normalize_ship_type)
            for df in (ft_v, fd_v):
                df.insert(0, "model", model_name)
            if not support_audit.empty:
                support_audit.insert(0, "model", model_name)

            _, combo_s = cii.create_summary(ft_v.drop(columns=["model"]), fd_v.drop(columns=["model"]))
            combo_s.insert(0, "model", model_name)
            atomic_csv(ft_v, ft_path)
            atomic_csv(fd_v, fd_path)
            atomic_csv(combo_s, summary_path)
            atomic_csv(support_audit, support_path)

            if args.save_fd_interval_audit and not interval_audit.empty:
                interval_path = out / f"A3_{model_name}_FD_interval_audit.csv.gz"
                logger.info("Saving large FD interval audit: %s", interval_path)
                interval_audit.to_csv(interval_path, index=False, compression="gzip")
            else:
                logger.info("FD interval audit not retained/saved in this run.")

        # Formal H3 inference chain for this model.
        logger.info(
            "Running endpoint bootstrap, pairwise contrasts, cluster-robust interaction, "
            "and %d vessel-label permutations for %s...",
            args.permutations,
            model_name,
        )
        ep, pairs, inter = h3_inference_for_model(
            model_name=model_name,
            ft=ft_v.drop(columns=["model"]),
            fd=fd_v.drop(columns=["model"]),
            bootstrap_reps=args.bootstrap_reps,
            permutations=args.permutations,
            seed=args.seed + mi * 10000,
        )
        all_endpoints.extend(ep)
        all_pairs.extend(pairs)
        all_interactions.extend(inter)

        all_ft.append(ft_v)
        all_fd.append(fd_v)
        all_summary.append(combo_s)

        model_status.append({
            "model": model_name,
            "FT_rows": int(len(ft_v)),
            "FD_rows": int(len(fd_v)),
            "vessels": int(ft_v["vessel_id"].nunique()),
            "reductions": ";".join(map(str, sorted(ft_v["reduction_pct"].unique()))),
            "support_min_retained_pct": float(ft_v["retained_pct"].min()),
        })

        del model, base_pred, interval_audit
        gc.collect()

    ft_all = pd.concat(all_ft, ignore_index=True)
    fd_all = pd.concat(all_fd, ignore_index=True)
    summary_all = pd.concat(all_summary, ignore_index=True)
    endpoint_df = pd.DataFrame(all_endpoints)
    pair_df = pd.DataFrame(all_pairs)
    interaction_df = pd.DataFrame(all_interactions)

    atomic_csv(ft_all, out / "A3_FT_by_vessel_all_models.csv")
    atomic_csv(fd_all, out / "A3_FD_by_vessel_all_models.csv")
    atomic_csv(summary_all, out / "A3_FT_FD_summary_all_models.csv")
    atomic_csv(endpoint_df, out / "A3_H3_endpoint_vessel_bootstrap_CI_all_models.csv")
    atomic_csv(pair_df, out / "A3_H3_pairwise_shiptype_contrasts_all_models.csv")
    atomic_csv(interaction_df, out / "A3_H3_shiptype_x_reduction_interaction_all_models.csv")
    atomic_csv(pd.DataFrame(model_status), out / "A3_model_run_status.csv")

    # Compact cross-model concordance table for the formal FT H3 endpoint.
    concordance_rows = []
    ft_ep = endpoint_df[
        (endpoint_df["analysis_role"] == "formal_H3") &
        (endpoint_df["scope"].isin(["bulk", "container", "tanker"]))
    ].copy()
    for (scope, reduction), g in ft_ep.groupby(["scope", "reduction_pct"]):
        signs = []
        for _, r in g.sort_values("model").iterrows():
            est = float(r["estimate_pct"])
            sign = "positive" if est > 0 else ("negative" if est < 0 else "zero")
            signs.append(f"{r['model']}:{sign}")
        concordance_rows.append({
            "scope": scope,
            "reduction_pct": reduction,
            "models": int(g["model"].nunique()),
            "direction_by_model": ";".join(signs),
            "all_same_direction": len({s.split(":", 1)[1] for s in signs}) == 1,
        })
    atomic_csv(pd.DataFrame(concordance_rows), out / "A3_FT_direction_concordance_across_models.csv")

    atomic_json(
        {
            "task": "A3 multimodel H3 robustness — final inference-aligned version",
            "models": models,
            "split_source": split_source,
            "reductions": reductions,
            "bootstrap_reps": args.bootstrap_reps,
            "permutations": args.permutations,
            "formal_H3_endpoint": "FT_CII_change_pct",
            "FD_role": "sensitivity/mechanism evidence, not a second formal H3 test",
            "model_selection": "locked manuscript registry; no retuning in A3",
            "support_logic": (
                "canonical support_mask + exact explicit-FD; chunked memory-safe aggregation when --memory-safe-fd is used"
            ),
            "inference_alignment": (
                "Endpoint bootstrap, pairwise ship-type contrasts, cluster-robust interaction, "
                "and vessel-label permutation interaction mirror run_final_manuscript_inference.py."
            ),
            "fd_interval_audit_saved": bool(args.save_fd_interval_audit and not args.memory_safe_fd),
            "resume_existing_models": bool(args.resume_existing_models),
            "memory_safe_fd": bool(args.memory_safe_fd),
            "fd_chunk_rows": int(args.fd_chunk_rows),
        },
        out / "A3_manifest.json",
    )

    logger.info("A3 FINAL COMPLETE: %s", out)


if __name__ == "__main__":
    main()
