#!/usr/bin/env python3
# -*- coding: utf-8 -*-

r"""L1-aligned CII analysis with fixed-time and explicit fixed-distance scenarios.

The analysis reuses the established 489,620-row cruise cohort, 391,696/97,924 L1
split, fitted XGBoost model, and baseline L1 predictions. It does not retrain the
reference model or create an alternative CII-specific split. FT retains the ten-minute
interval duration and reduces travelled distance with speed. FD retains source-segment
distance, reconstructs the required sailing duration as full and partial ten-minute
intervals, predicts every materialised interval with the same XGBoost model, and sums
predicted fuel. The same support mask is applied to FT and FD for each reduction level.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import joblib
import numpy as np
import pandas as pd

import f31_core_memory_safe as core
import f31_cii_ft_fd_explicit_intervals as fdmod


SCRIPT_VERSION = "2026-08-17.l1-aligned-ft-fd-final-v1"
INTERVAL_MIN = 10.0

# Locked study counts.
EXPECTED_CRUISE = 489_620
EXPECTED_TRAIN = 391_696
EXPECTED_TEST = 97_924
EXPECTED_VESSELS = 21

# The already-observed scenario-specific supported counts from the same L1
# test set. These are used as an additional hard audit.
EXPECTED_SUPPORTED = {
    5.0: 96_940,
    10.0: 95_244,
    15.0: 92_275,
}


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logger(out_dir: Path) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("cii_l1_aligned")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)

    fh = logging.FileHandler(
        out_dir / "run_cii_l1_aligned.log",
        encoding="utf-8",
    )
    fh.setFormatter(fmt)

    logger.addHandler(sh)
    logger.addHandler(fh)

    return logger


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Run only the L1-aligned FT/FD CII supplement using the same "
            "saved model, split and baseline predictions as the preceding "
            "predictive analysis."
        ),
    )

    p.add_argument(
        "--work-dir",
        default=r"data\ship_fuel_v4",
    )
    p.add_argument(
        "--data-dir",
        default=r"data\05_clean23\all_phase_equal_cleaning\05_combined",
    )
    p.add_argument(
        "--input-csv",
        default=None,
        help="Optional exact path to final_fixed31_all_phases_model_ready.csv.",
    )
    p.add_argument(
        "--original-output-dir",
        default=r"data\11_f31_complete_empirics",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help=(
            "New output folder. Defaults to "
            "<original-output-dir>\\11_cii_l1_aligned_final"
        ),
    )
    p.add_argument(
        "--column-overrides",
        default=None,
    )

    p.add_argument(
        "--model-path",
        default=None,
        help="Optional exact path to xgb_record_split.joblib.",
    )
    p.add_argument(
        "--split-path",
        default=None,
        help="Optional exact path to record_split_indices.npz.",
    )
    p.add_argument(
        "--prediction-path",
        default=None,
        help="Optional exact path to xgb_record_test_prediction.npy.",
    )

    p.add_argument("--co2-factor", type=float, default=3.114)
    p.add_argument("--csv-chunksize", type=int, default=25_000)
    p.add_argument("--trajectory-gap-minutes", type=float, default=30.0)

    p.add_argument(
        "--reductions",
        default="0.05,0.10,0.15",
    )

    p.add_argument(
        "--prediction-atol",
        type=float,
        default=1e-9,
    )
    p.add_argument(
        "--prediction-rtol",
        type=float,
        default=1e-7,
    )

    return p.parse_args()


# ---------------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------------

def require_file(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def resolve_column_overrides(
    explicit: str | None,
    work_dir: Path,
) -> str | None:
    if explicit:
        p = Path(explicit)
        require_file(p, "Column-overrides JSON")
        return str(p)

    candidates = [
        work_dir / "column_overrides_fixed31.json",
        work_dir / "column_overrides_fixed31(2).json",
    ]

    for p in candidates:
        if p.exists():
            return str(p)

    hits = sorted(
        work_dir.glob("column_overrides_fixed31*.json")
    )
    return str(hits[0]) if hits else None


def parse_reductions(text: str) -> list[float]:
    out = [
        float(x.strip())
        for x in text.split(",")
        if x.strip()
    ]

    if not out:
        raise ValueError("No reductions supplied.")

    for r in out:
        if not (0.0 < r < 1.0):
            raise ValueError(
                f"Reduction must be between 0 and 1: {r}"
            )

    return out


def baseline_distance_old_core(
    raw_test: pd.DataFrame,
) -> pd.Series:
    """
    Preserve the exact distance convention in the original core CII function.

    If distance_nm exists and is not entirely missing, use it directly.
    Otherwise use SOG * 10/60.
    """
    if (
        "distance_nm" not in raw_test.columns
        or raw_test["distance_nm"].isna().all()
    ):
        return (
            raw_test["speed_kn"].astype(float)
            * (INTERVAL_MIN / 60.0)
        )

    return pd.to_numeric(
        raw_test["distance_nm"],
        errors="coerce",
    )


def co2_factor_series(
    raw_test: pd.DataFrame,
    default_cf: float,
) -> pd.Series:
    if (
        "co2_factor" in raw_test.columns
        and raw_test["co2_factor"].notna().any()
    ):
        return pd.to_numeric(
            raw_test["co2_factor"],
            errors="coerce",
        ).fillna(default_cf)

    return pd.Series(
        float(default_cf),
        index=raw_test.index,
        dtype=float,
    )


def support_mask(
    raw_train: pd.DataFrame,
    raw_test: pd.DataFrame,
    reduction: float,
) -> np.ndarray:
    """
    Exact ship-type support logic used by the original scenario function.
    """
    sc = raw_test.copy()
    sc["speed_kn"] = (
        sc["speed_kn"].astype(float)
        * (1.0 - reduction)
    )

    support = (
        raw_train
        .groupby("ship_type")["speed_kn"]
        .agg(["min", "max"])
    )

    valid = np.ones(len(sc), dtype=bool)

    for st, inds in sc.groupby("ship_type").groups.items():
        if st not in support.index:
            valid[np.asarray(list(inds), dtype=int)] = False
            continue

        lo = float(support.loc[st, "min"])
        hi = float(support.loc[st, "max"])

        idx = np.asarray(list(inds), dtype=int)

        valid[idx] = (
            (sc.loc[idx, "speed_kn"].to_numpy(float) >= lo)
            &
            (sc.loc[idx, "speed_kn"].to_numpy(float) <= hi)
        )

    return valid


def finite_change_elasticity(
    fuel_rate_ratio: float,
    speed_ratio: float,
) -> float:
    if (
        np.isfinite(fuel_rate_ratio)
        and np.isfinite(speed_ratio)
        and fuel_rate_ratio > 0.0
        and speed_ratio > 0.0
        and not np.isclose(speed_ratio, 1.0)
    ):
        return float(
            np.log(fuel_rate_ratio)
            / np.log(speed_ratio)
        )

    return np.nan


# ---------------------------------------------------------------------------
# Aggregate helper
# ---------------------------------------------------------------------------

def weighted_cii(
    cii: np.ndarray,
    transport_work: np.ndarray,
) -> float:
    return float(
        np.average(
            np.asarray(cii, dtype=float),
            weights=np.maximum(
                np.asarray(transport_work, dtype=float),
                1e-12,
            ),
        )
    )


# ---------------------------------------------------------------------------
# Main scenario calculation
# ---------------------------------------------------------------------------

def calculate_ft_fd(
    model,
    raw_train: pd.DataFrame,
    raw_test: pd.DataFrame,
    feature_cols: Sequence[str],
    saved_baseline_pred: np.ndarray,
    reductions: Sequence[float],
    default_cf: float,
    logger: logging.Logger,
):
    """
    Calculate Scenario A and B from the SAME L1 test set.

    Baseline predictions are NOT regenerated. They come directly from the
    saved xgb_record_test_prediction.npy used by the preceding L1 analysis.
    """
    base = raw_test.copy().reset_index(drop=True)

    if len(saved_baseline_pred) != len(base):
        raise ValueError(
            "Saved baseline prediction length does not match L1 test."
        )

    base["_source_row_id"] = np.arange(
        len(base),
        dtype=int,
    )
    base["_baseline_pred"] = np.asarray(
        saved_baseline_pred,
        dtype=float,
    )
    base["_baseline_distance_nm"] = (
        baseline_distance_old_core(base)
    )
    base["_cf_used"] = co2_factor_series(
        base,
        default_cf,
    )

    if base["dwt"].isna().all():
        raise KeyError("DWT is required for CII analysis.")

    ft_vessel_rows = []
    fd_vessel_rows = []
    support_audit_rows = []
    interval_audit_parts = []

    for reduction in reductions:
        reduction = float(reduction)
        rpct = reduction * 100.0
        speed_ratio = 1.0 - reduction

        logger.info(
            "Running %.0f%% speed reduction...",
            rpct,
        )

        valid = support_mask(
            raw_train=raw_train,
            raw_test=base,
            reduction=reduction,
        )

        n_supported = int(valid.sum())

        if rpct in EXPECTED_SUPPORTED:
            expected_n = EXPECTED_SUPPORTED[rpct]
            if n_supported != expected_n:
                raise AssertionError(
                    f"{rpct:.0f}% support count changed: "
                    f"current={n_supported}, expected={expected_n}. "
                    "Stop because this is no longer the same scenario sample."
                )

        b = base.loc[valid].copy()

        # ===============================================================
        # A: Fixed-Time
        # ===============================================================
        ft = b.copy()
        ft["speed_kn"] = (
            ft["speed_kn"].astype(float)
            * speed_ratio
        )

        ft_pred = np.asarray(
            model.predict(
                core.feature_matrix_from_raw(
                    ft,
                    feature_cols,
                )
            ),
            dtype=float,
        )

        # SAME baseline predictions from saved L1 artifact.
        ft["_baseline_pred"] = b["_baseline_pred"].to_numpy(float)
        ft["_scenario_pred"] = ft_pred

        # Exact original FT distance convention.
        ratio = np.divide(
            ft["speed_kn"].to_numpy(float),
            b["speed_kn"].to_numpy(float),
            out=np.zeros(len(ft), dtype=float),
            where=b["speed_kn"].to_numpy(float) != 0.0,
        )

        ft["_scenario_distance_nm"] = (
            b["_baseline_distance_nm"].to_numpy(float)
            * ratio
        )

        # ===============================================================
        # B: Fixed-Distance, explicit intervals
        # ===============================================================
        fd_source = b.copy()

        # build_fd_explicit_intervals() expects this exact name.
        fd_source["distance_nm_used"] = (
            fd_source["_baseline_distance_nm"]
        )

        fd_expanded = fdmod.build_fd_explicit_intervals(
            supported_base=fd_source,
            reduction=reduction,
        )

        if fd_expanded.empty:
            raise RuntimeError(
                f"{rpct:.0f}% explicit FD expansion returned no rows."
            )

        fd_expanded = (
            fdmod.predict_fd_explicit_intervals(
                model=model,
                expanded=fd_expanded,
                feature_cols=feature_cols,
                clip_negative_for_cii=False,
            )
        )

        # Carry the CO2 factor from the SAME original test record.
        cf_map = (
            fd_source
            .set_index("_source_row_id")["_cf_used"]
            .to_dict()
        )

        fd_expanded["_cf_used"] = (
            fd_expanded["_source_row_id"]
            .map(cf_map)
            .astype(float)
        )

        fd_expanded["_reduction_pct"] = rpct

        # Detailed proof that extra intervals were explicitly constructed.
        audit_cols = [
            "_reduction_pct",
            "_source_row_id",
            "vessel_id",
            "ship_type",
            "_baseline_speed_kn",
            "_reduced_speed_kn",
            "_segment_distance_nm",
            "_required_minutes",
            "_interval_sequence",
            "_interval_kind",
            "_interval_minutes",
            "_interval_distance_nm",
            "_is_added_time",
            "_predicted_fuel_t_per_10min",
            "_predicted_interval_fuel_t",
        ]

        if "timestamp" in fd_expanded.columns:
            audit_cols.insert(3, "timestamp")

        interval_audit_parts.append(
            fd_expanded[audit_cols].copy()
        )

        # ===============================================================
        # Vessel aggregation
        # ===============================================================
        for vessel, bg in b.groupby("vessel_id"):
            source_ids = set(
                bg["_source_row_id"]
                .astype(int)
                .tolist()
            )

            fg = ft[
                ft["_source_row_id"].isin(source_ids)
            ].copy()

            dg = fd_expanded[
                fd_expanded["_source_row_id"].isin(source_ids)
            ].copy()

            if fg.empty or dg.empty:
                raise RuntimeError(
                    f"{rpct:.0f}% / {vessel}: empty FT or FD subset."
                )

            ship_type = str(bg["ship_type"].iloc[0])
            dwt = float(
                pd.to_numeric(
                    bg["dwt"],
                    errors="coerce",
                )
                .dropna()
                .median()
            )

            # Baseline
            base_fuel = float(
                bg["_baseline_pred"].sum()
            )
            base_dist = float(
                bg["_baseline_distance_nm"].sum()
            )
            base_duration_min = (
                len(bg) * INTERVAL_MIN
            )

            base_co2_g = float(
                np.sum(
                    bg["_baseline_pred"].to_numpy(float)
                    * bg["_cf_used"].to_numpy(float)
                    * 1e6
                )
            )

            base_tw = dwt * base_dist
            base_cii = (
                base_co2_g / base_tw
                if base_tw > 0.0
                else np.nan
            )

            # FT
            ft_fuel = float(
                fg["_scenario_pred"].sum()
            )
            ft_dist = float(
                fg["_scenario_distance_nm"].sum()
            )
            ft_duration_min = (
                len(fg) * INTERVAL_MIN
            )

            ft_co2_g = float(
                np.sum(
                    fg["_scenario_pred"].to_numpy(float)
                    * fg["_cf_used"].to_numpy(float)
                    * 1e6
                )
            )

            ft_tw = dwt * ft_dist
            ft_cii = (
                ft_co2_g / ft_tw
                if ft_tw > 0.0
                else np.nan
            )

            # FD explicit
            fd_fuel = float(
                dg["_predicted_interval_fuel_t"].sum()
            )
            fd_dist = float(
                dg["_interval_distance_nm"].sum()
            )
            fd_duration_min = float(
                dg["_interval_minutes"].sum()
            )

            fd_co2_g = float(
                np.sum(
                    dg["_predicted_interval_fuel_t"].to_numpy(float)
                    * dg["_cf_used"].to_numpy(float)
                    * 1e6
                )
            )

            fd_tw = dwt * fd_dist
            fd_cii = (
                fd_co2_g / fd_tw
                if fd_tw > 0.0
                else np.nan
            )

            # Explicitly predicted fuel located beyond the original 10 min.
            added_mask = dg["_is_added_time"].astype(bool)

            fd_added_fuel = float(
                dg.loc[
                    added_mask,
                    "_predicted_interval_fuel_t",
                ].sum()
            )

            fd_added_time_min = (
                fd_duration_min
                - base_duration_min
            )

            ft_cii_ratio = (
                ft_cii / base_cii
                if np.isfinite(base_cii)
                and base_cii > 0.0
                else np.nan
            )

            fd_cii_ratio = (
                fd_cii / base_cii
                if np.isfinite(base_cii)
                and base_cii > 0.0
                else np.nan
            )

            fuel_rate_ratio = (
                ft_fuel / base_fuel
                if base_fuel > 0.0
                else np.nan
            )

            epsilon = finite_change_elasticity(
                fuel_rate_ratio=fuel_rate_ratio,
                speed_ratio=speed_ratio,
            )

            retained_pct = (
                len(bg)
                / max(
                    int(
                        (
                            base["vessel_id"]
                            == vessel
                        ).sum()
                    ),
                    1,
                )
                * 100.0
            )

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
                "FT_fuel_change_pct": (
                    (ft_fuel / base_fuel - 1.0) * 100.0
                    if base_fuel > 0.0
                    else np.nan
                ),
                "FT_distance_nm": ft_dist,
                "FT_distance_change_pct": (
                    (ft_dist / base_dist - 1.0) * 100.0
                    if base_dist > 0.0
                    else np.nan
                ),
                "FT_duration_min": ft_duration_min,
                "FT_duration_change_pct": 0.0,
                "FT_transport_work": ft_tw,
                "FT_CII_proxy": ft_cii,
                "FT_CII_ratio": ft_cii_ratio,
                "FT_CII_change_pct": (
                    (ft_cii_ratio - 1.0) * 100.0
                    if np.isfinite(ft_cii_ratio)
                    else np.nan
                ),
                "FT_improved": bool(
                    np.isfinite(ft_cii)
                    and np.isfinite(base_cii)
                    and ft_cii < base_cii
                ),
            })

            fd_vessel_rows.append({
                **common,
                "FD_predicted_total_fuel_t": fd_fuel,
                "FD_total_fuel_change_pct": (
                    (fd_fuel / base_fuel - 1.0) * 100.0
                    if base_fuel > 0.0
                    else np.nan
                ),

                "FD_added_sailing_time_min": fd_added_time_min,
                "FD_duration_min": fd_duration_min,
                "FD_duration_change_pct": (
                    (fd_duration_min / base_duration_min - 1.0)
                    * 100.0
                    if base_duration_min > 0.0
                    else np.nan
                ),

                "FD_added_time_fuel_t": fd_added_fuel,
                "FD_added_time_fuel_pct_of_baseline": (
                    fd_added_fuel / base_fuel * 100.0
                    if base_fuel > 0.0
                    else np.nan
                ),

                "FD_distance_nm": fd_dist,
                "FD_distance_error_nm": (
                    fd_dist - base_dist
                ),
                "FD_distance_error_pct": (
                    (fd_dist / base_dist - 1.0) * 100.0
                    if base_dist > 0.0
                    else np.nan
                ),

                "FD_transport_work": fd_tw,
                "FD_CII_proxy": fd_cii,
                "FD_CII_ratio": fd_cii_ratio,
                "FD_CII_change_pct": (
                    (fd_cii_ratio - 1.0) * 100.0
                    if np.isfinite(fd_cii_ratio)
                    else np.nan
                ),
                "FD_improved": bool(
                    np.isfinite(fd_cii)
                    and np.isfinite(base_cii)
                    and fd_cii < base_cii
                ),

                "FT_minus_FD_CII_change_pct_points": (
                    (ft_cii_ratio - fd_cii_ratio)
                    * 100.0
                    if np.isfinite(ft_cii_ratio)
                    and np.isfinite(fd_cii_ratio)
                    else np.nan
                ),

                "FD_explicit_full_10min_intervals": int(
                    (
                        dg["_interval_kind"]
                        == "full_10min"
                    ).sum()
                ),
                "FD_explicit_partial_intervals": int(
                    (
                        dg["_interval_kind"]
                        == "partial"
                    ).sum()
                ),
                "FD_explicit_total_intervals": int(
                    len(dg)
                ),
            })

        # Per-reduction source-sample audit.
        support_audit_rows.append({
            "reduction_pct": rpct,
            "L1_test_rows": len(base),
            "supported_rows_A": n_supported,
            "supported_rows_B": n_supported,
            "same_support_mask_A_B": True,
            "expected_supported_rows": EXPECTED_SUPPORTED.get(
                rpct,
                np.nan,
            ),
            "supported_count_matches_expected": (
                n_supported
                == EXPECTED_SUPPORTED[rpct]
                if rpct in EXPECTED_SUPPORTED
                else np.nan
            ),
            "vessels_A_B": int(
                b["vessel_id"].nunique()
            ),
        })

    ft_vessel = pd.DataFrame(ft_vessel_rows)
    fd_vessel = pd.DataFrame(fd_vessel_rows)
    support_audit = pd.DataFrame(support_audit_rows)

    interval_audit = (
        pd.concat(
            interval_audit_parts,
            ignore_index=True,
        )
        if interval_audit_parts
        else pd.DataFrame()
    )

    return (
        ft_vessel,
        fd_vessel,
        support_audit,
        interval_audit,
    )


# ---------------------------------------------------------------------------
# Summary tables
# ---------------------------------------------------------------------------

def create_summary(
    ft_vessel: pd.DataFrame,
    fd_vessel: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return:
        ft_summary
        combined_summary
    """
    merge_keys = [
        "reduction_pct",
        "vessel_id",
        "ship_type",
        "n_supported",
        "retained_pct",
        "speed_ratio",
        "DWT",
        "baseline_predicted_fuel_t",
        "baseline_distance_nm",
        "baseline_duration_min",
        "baseline_transport_work",
        "baseline_CII_proxy",
        "fuel_rate_ratio",
        "finite_change_fuel_rate_elasticity",
    ]

    merged = ft_vessel.merge(
        fd_vessel,
        on=merge_keys,
        how="inner",
        validate="one_to_one",
    )

    if len(merged) != len(ft_vessel) or len(merged) != len(fd_vessel):
        raise AssertionError(
            "FT and FD vessel tables do not contain the same scenario/vessel rows."
        )

    ft_rows = []
    combo_rows = []

    for rpct, rg in merged.groupby("reduction_pct"):
        scopes = [("Fleet", rg)] + [
            (str(st), g)
            for st, g in rg.groupby("ship_type")
        ]

        for scope, sg in scopes:
            base_fuel = float(
                sg["baseline_predicted_fuel_t"].sum()
            )
            ft_fuel = float(
                sg["FT_predicted_fuel_t"].sum()
            )
            fd_fuel = float(
                sg["FD_predicted_total_fuel_t"].sum()
            )

            base_cii = weighted_cii(
                sg["baseline_CII_proxy"].to_numpy(float),
                sg["baseline_transport_work"].to_numpy(float),
            )

            ft_cii = weighted_cii(
                sg["FT_CII_proxy"].to_numpy(float),
                sg["FT_transport_work"].to_numpy(float),
            )

            fd_cii = weighted_cii(
                sg["FD_CII_proxy"].to_numpy(float),
                sg["FD_transport_work"].to_numpy(float),
            )

            ft_ratio = (
                ft_cii / base_cii
                if base_cii > 0.0
                else np.nan
            )
            fd_ratio = (
                fd_cii / base_cii
                if base_cii > 0.0
                else np.nan
            )

            ft_change = (
                (ft_ratio - 1.0) * 100.0
                if np.isfinite(ft_ratio)
                else np.nan
            )
            fd_change = (
                (fd_ratio - 1.0) * 100.0
                if np.isfinite(fd_ratio)
                else np.nan
            )

            ft_rows.append({
                "reduction_pct": float(rpct),
                "scope": scope,
                "predicted_fuel_change_pct": (
                    (ft_fuel / base_fuel - 1.0)
                    * 100.0
                    if base_fuel > 0.0
                    else np.nan
                ),
                "CII_proxy_change_pct_weighted": ft_change,
                "median_vessel_CII_proxy_change_pct": float(
                    sg["FT_CII_change_pct"].median()
                ),
                "mean_vessel_CII_proxy_change_pct": float(
                    sg["FT_CII_change_pct"].mean()
                ),
                "vessels_improved": int(
                    sg["FT_improved"].sum()
                ),
                "vessels_total": int(len(sg)),
                "mean_retained_pct": float(
                    sg["retained_pct"].mean()
                ),
                "n_supported_total": int(
                    sg["n_supported"].sum()
                ),
            })

            speed_ratio = float(
                sg["speed_ratio"].iloc[0]
            )

            fuel_rate_ratio = (
                ft_fuel / base_fuel
                if base_fuel > 0.0
                else np.nan
            )

            epsilon = finite_change_elasticity(
                fuel_rate_ratio,
                speed_ratio,
            )

            combo_rows.append({
                "reduction_pct": float(rpct),
                "scope": scope,
                "vessels": int(len(sg)),
                "n_supported_total": int(
                    sg["n_supported"].sum()
                ),
                "mean_retained_pct": float(
                    sg["retained_pct"].mean()
                ),

                "speed_ratio": speed_ratio,
                "fuel_rate_ratio": fuel_rate_ratio,
                "finite_change_fuel_rate_elasticity": epsilon,

                # A / FT
                "FT_fuel_change_pct": (
                    (ft_fuel / base_fuel - 1.0)
                    * 100.0
                    if base_fuel > 0.0
                    else np.nan
                ),
                "FT_CII_change_pct_weighted": ft_change,
                "FT_vessels_improved": int(
                    sg["FT_improved"].sum()
                ),

                # B / FD
                "FD_added_sailing_time_min": float(
                    sg["FD_added_sailing_time_min"].sum()
                ),
                "FD_duration_change_pct": (
                    (
                        sg["FD_duration_min"].sum()
                        / sg["baseline_duration_min"].sum()
                    )
                    - 1.0
                ) * 100.0,
                "FD_added_time_fuel_t": float(
                    sg["FD_added_time_fuel_t"].sum()
                ),
                "FD_added_time_fuel_pct_of_baseline": (
                    sg["FD_added_time_fuel_t"].sum()
                    / base_fuel
                    * 100.0
                    if base_fuel > 0.0
                    else np.nan
                ),
                "FD_total_fuel_change_pct": (
                    (fd_fuel / base_fuel - 1.0)
                    * 100.0
                    if base_fuel > 0.0
                    else np.nan
                ),
                "FD_CII_change_pct_weighted": fd_change,
                "FD_vessels_improved": int(
                    sg["FD_improved"].sum()
                ),

                "FT_minus_FD_CII_change_pct_points": (
                    ft_change - fd_change
                    if np.isfinite(ft_change)
                    and np.isfinite(fd_change)
                    else np.nan
                ),

                "FD_max_abs_distance_error_nm": float(
                    np.nanmax(
                        np.abs(
                            sg["FD_distance_error_nm"]
                            .to_numpy(float)
                        )
                    )
                ),
            })

    return (
        pd.DataFrame(ft_rows),
        pd.DataFrame(combo_rows),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    t0 = time.time()

    work_dir = Path(args.work_dir)
    original_output = Path(
        args.original_output_dir
    )
    artifacts = original_output / "14_artifacts"

    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else original_output
        / "11_cii_l1_aligned_final"
    )

    logger = setup_logger(out_dir)

    logger.info(
        "L1-aligned FT/FD CII runner %s",
        SCRIPT_VERSION,
    )
    logger.info(
        "No training; no new split; no alternate CII dataset."
    )

    model_path = (
        Path(args.model_path)
        if args.model_path
        else artifacts / "xgb_record_split.joblib"
    )
    split_path = (
        Path(args.split_path)
        if args.split_path
        else artifacts / "record_split_indices.npz"
    )
    prediction_path = (
        Path(args.prediction_path)
        if args.prediction_path
        else artifacts / "xgb_record_test_prediction.npy"
    )

    require_file(
        model_path,
        "Locked XGBoost model",
    )
    require_file(
        split_path,
        "Locked L1 split",
    )
    require_file(
        prediction_path,
        "Locked L1 test prediction",
    )

    column_overrides = resolve_column_overrides(
        args.column_overrides,
        work_dir,
    )

    core_args = SimpleNamespace(
        input_csv=args.input_csv,
        data_dir=args.data_dir,
        column_overrides=column_overrides,
        csv_chunksize=args.csv_chunksize,
        trajectory_gap_minutes=args.trajectory_gap_minutes,
    )

    # ---------------------------------------------------------------
    # Reconstruct SAME cruise cohort.
    # ---------------------------------------------------------------
    logger.info(
        "Loading and canonicalising the original Fixed31 data..."
    )

    df = core.load_phase_files(
        core_args,
        logger,
    )
    bundle = core.canonicalize_dataframe(
        df,
        core_args,
        logger,
    )

    cruise_mask = (
        bundle.raw["phase"]
        == "cruise"
    ).to_numpy()

    raw_cruise = (
        bundle.raw.loc[cruise_mask]
        .reset_index(drop=True)
    )
    X_cruise = (
        bundle.feature_df.loc[cruise_mask]
        .reset_index(drop=True)
    )

    if len(raw_cruise) != EXPECTED_CRUISE:
        raise AssertionError(
            f"Cruise count changed: {len(raw_cruise)} "
            f"!= {EXPECTED_CRUISE}"
        )

    if (
        raw_cruise["vessel_id"].nunique()
        != EXPECTED_VESSELS
    ):
        raise AssertionError(
            "Vessel count differs from locked study."
        )

    # Exact original feature order from the canonical feature frame.
    feature_cols = list(X_cruise.columns)

    if len(feature_cols) != 17:
        raise AssertionError(
            f"Expected 17 model features; got {len(feature_cols)}"
        )

    logger.info(
        "Cruise cohort locked: %d rows | %d vessels | %d features",
        len(raw_cruise),
        raw_cruise["vessel_id"].nunique(),
        len(feature_cols),
    )

    # ---------------------------------------------------------------
    # Restore SAME L1 split.
    # ---------------------------------------------------------------
    split = np.load(split_path)

    train_idx = np.asarray(
        split["train_idx"],
        dtype=int,
    )
    test_idx = np.asarray(
        split["test_idx"],
        dtype=int,
    )

    if len(train_idx) != EXPECTED_TRAIN:
        raise AssertionError(
            f"Train count changed: {len(train_idx)} != {EXPECTED_TRAIN}"
        )

    if len(test_idx) != EXPECTED_TEST:
        raise AssertionError(
            f"Test count changed: {len(test_idx)} != {EXPECTED_TEST}"
        )

    if len(np.intersect1d(train_idx, test_idx)):
        raise AssertionError(
            "Train/test overlap found."
        )

    if (
        len(train_idx) + len(test_idx)
        != len(raw_cruise)
    ):
        raise AssertionError(
            "L1 split no longer covers the cruise cohort exactly."
        )

    raw_train = (
        raw_cruise.iloc[train_idx]
        .reset_index(drop=True)
    )
    raw_test = (
        raw_cruise.iloc[test_idx]
        .reset_index(drop=True)
    )

    logger.info(
        "L1 split locked: train=%d | test=%d",
        len(raw_train),
        len(raw_test),
    )

    # ---------------------------------------------------------------
    # Load SAME model + SAME saved baseline prediction.
    # ---------------------------------------------------------------
    model = joblib.load(model_path)

    saved_pred = np.asarray(
        np.load(prediction_path),
        dtype=float,
    )

    if len(saved_pred) != EXPECTED_TEST:
        raise AssertionError(
            "Saved L1 prediction length does not equal 97,924."
        )

    # Full exact alignment audit.
    current_pred = np.asarray(
        model.predict(
            core.feature_matrix_from_raw(
                raw_test,
                feature_cols,
            )
        ),
        dtype=float,
    )

    pred_diff = np.abs(
        current_pred - saved_pred
    )

    pred_ok = np.allclose(
        current_pred,
        saved_pred,
        rtol=args.prediction_rtol,
        atol=args.prediction_atol,
        equal_nan=True,
    )

    pred_audit = pd.DataFrame([{
        "n_test": len(saved_pred),
        "mean_abs_difference": float(
            np.mean(pred_diff)
        ),
        "max_abs_difference": float(
            np.max(pred_diff)
        ),
        "alignment_passed": bool(pred_ok),
    }])

    pred_audit.to_csv(
        out_dir
        / "Audit_01_locked_L1_prediction_alignment.csv",
        index=False,
    )

    if not pred_ok:
        raise AssertionError(
            "The saved model does not reproduce the locked "
            "xgb_record_test_prediction.npy. Stop."
        )

    logger.info(
        "LOCKED L1 PREDICTION ALIGNMENT PASSED."
    )

    # ---------------------------------------------------------------
    # FT + FD on SAME test records / SAME support masks.
    # ---------------------------------------------------------------
    reductions = parse_reductions(
        args.reductions
    )

    (
        ft_vessel,
        fd_vessel,
        support_audit,
        interval_audit,
    ) = calculate_ft_fd(
        model=model,
        raw_train=raw_train,
        raw_test=raw_test,
        feature_cols=feature_cols,
        saved_baseline_pred=saved_pred,
        reductions=reductions,
        default_cf=args.co2_factor,
        logger=logger,
    )

    ft_summary, combined_summary = (
        create_summary(
            ft_vessel,
            fd_vessel,
        )
    )

    # ---------------------------------------------------------------
    # Hard final audits.
    # ---------------------------------------------------------------
    if not support_audit[
        "same_support_mask_A_B"
    ].all():
        raise AssertionError(
            "A and B do not use the same support mask."
        )

    max_fd_dist_error = float(
        np.nanmax(
            np.abs(
                fd_vessel[
                    "FD_distance_error_nm"
                ].to_numpy(float)
            )
        )
    )

    if max_fd_dist_error > 1e-6:
        raise AssertionError(
            "FD failed to preserve baseline distance: "
            f"max vessel error={max_fd_dist_error} nm"
        )

    # ---------------------------------------------------------------
    # Save final tables.
    # ---------------------------------------------------------------
    ft_vessel.to_csv(
        out_dir
        / "Table_15_FT_L1_aligned_by_vessel.csv",
        index=False,
    )

    ft_summary.to_csv(
        out_dir
        / "Table_15_FT_L1_aligned_summary.csv",
        index=False,
    )

    fd_vessel.to_csv(
        out_dir
        / "Table_16_FD_explicit_L1_aligned_by_vessel.csv",
        index=False,
    )

    # A manuscript-friendly Scenario-B summary extracted from the combined
    # result while retaining the corresponding FT fields for audit.
    fd_summary_cols = [
        "reduction_pct",
        "scope",
        "vessels",
        "n_supported_total",
        "mean_retained_pct",
        "FD_added_sailing_time_min",
        "FD_duration_change_pct",
        "FD_added_time_fuel_t",
        "FD_added_time_fuel_pct_of_baseline",
        "FD_total_fuel_change_pct",
        "FD_CII_change_pct_weighted",
        "FD_vessels_improved",
        "finite_change_fuel_rate_elasticity",
        "FD_max_abs_distance_error_nm",
    ]

    combined_summary[
        fd_summary_cols
    ].to_csv(
        out_dir
        / "Table_16_FD_explicit_L1_aligned_summary.csv",
        index=False,
    )

    combined_summary.to_csv(
        out_dir
        / "Table_15_vs_16_L1_aligned_review.csv",
        index=False,
    )

    support_audit.to_csv(
        out_dir
        / "Audit_02_same_support_mask_A_B.csv",
        index=False,
    )

    interval_audit.to_csv(
        out_dir
        / "Audit_03_FD_explicit_added_intervals.csv",
        index=False,
    )

    # Compact final audit.
    final_audit = pd.DataFrame([{
        "cruise_rows": len(raw_cruise),
        "train_rows": len(raw_train),
        "test_rows": len(raw_test),
        "vessels": raw_cruise["vessel_id"].nunique(),
        "features": len(feature_cols),
        "saved_prediction_alignment": True,
        "same_support_mask_A_B": bool(
            support_audit[
                "same_support_mask_A_B"
            ].all()
        ),
        "max_FD_distance_error_nm": max_fd_dist_error,
        "model_retrained": False,
        "new_split_created": False,
        "alternate_CII_dataset_used": False,
    }])

    final_audit.to_csv(
        out_dir
        / "Audit_04_final_method_consistency.csv",
        index=False,
    )

    # ---------------------------------------------------------------
    # Console summary.
    # ---------------------------------------------------------------
    logger.info(
        "=== FINAL L1-ALIGNED CII RESULTS ==="
    )

    fleet = combined_summary[
        combined_summary["scope"]
        == "Fleet"
    ]

    for _, row in fleet.iterrows():
        logger.info(
            "%.0f%% | supported=%d | "
            "FT fuel=%+.6f%% | FT CII=%+.6f%% | "
            "FD duration=%+.6f%% | FD added fuel=%.3f t | "
            "FD total fuel=%+.6f%% | FD CII=%+.6f%% | "
            "FT-FD CII diff=%+.6f pp",
            row["reduction_pct"],
            int(row["n_supported_total"]),
            row["FT_fuel_change_pct"],
            row["FT_CII_change_pct_weighted"],
            row["FD_duration_change_pct"],
            row["FD_added_time_fuel_t"],
            row["FD_total_fuel_change_pct"],
            row["FD_CII_change_pct_weighted"],
            row["FT_minus_FD_CII_change_pct_points"],
        )

    logger.info(
        "Finished. No training was performed."
    )
    logger.info(
        "Outputs: %s",
        out_dir,
    )
    logger.info(
        "Elapsed: %.2f min",
        (time.time() - t0) / 60.0,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.exception(
            "L1-aligned FT/FD supplement failed"
        )
        raise
