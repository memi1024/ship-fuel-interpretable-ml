#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
F31 CII speed-reduction scenarios:
Scenario A = fixed time (FT)
Scenario B = fixed distance (FD), with EXPLICIT added sailing intervals

Purpose
-------
This version does NOT estimate Scenario-B fuel by multiplying the reduced-speed
fuel by a voyage-duration coefficient such as 1/(1-r).

Instead, for every supported baseline 10-min record:

1) Preserve the baseline segment distance for Scenario B.
2) Reduce speed to V_r = V_0 * (1-r).
3) Compute the actual time required to traverse that fixed-distance segment:
       required_minutes = distance_nm / V_r * 60
4) Split the required time into:
       - N complete 10-min intervals
       - at most one final partial interval
5) Materialise those added intervals as rows.
6) Send every materialised row through the fitted ML model.
7) Sum the predicted fuel of all complete intervals.
8) For only the final incomplete interval, convert the model's t/10-min output
   to the exact partial-duration fuel:
       fuel_partial = prediction_10min * partial_minutes / 10

The partial-interval conversion is a unit conversion required because the
trained target is t/10 min. It is not a voyage-level duration approximation.

Important modelling boundary
----------------------------
For added intervals, non-speed covariates are held at the source record values.
This is a local partial-equilibrium scenario, consistent with the existing
sensitivity design. It does not forecast future weather, auxiliary-engine load,
routing changes, schedule effects, or other time-varying operating states.

Outputs
-------
1) Table_C3_FT_by_vessel.csv
2) Table_C4_FD_explicit_by_vessel.csv
3) Table_C5_FT_FD_explicit_summary.csv
4) Table_C6_FD_added_interval_audit.csv
5) Table_C7_FT_FD_comparison.csv

"""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple, Dict, Optional

import numpy as np
import pandas as pd

import f31_core_memory_safe as core


INTERVAL_MINUTES = 10.0
EPS = 1e-10


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _predict_10min(
    model,
    raw: pd.DataFrame,
    feature_cols: Sequence[str],
    clip_negative_for_cii: bool = False,
) -> np.ndarray:
    """Predict main-engine fuel in t / 10 min."""
    if len(raw) == 0:
        return np.asarray([], dtype=float)

    X = core.feature_matrix_from_raw(raw, feature_cols)
    pred = np.asarray(model.predict(X), dtype=float)

    if clip_negative_for_cii:
        pred = np.maximum(pred, 0.0)

    return pred


def _distance_used(raw: pd.DataFrame) -> pd.Series:
    """
    Prefer the stored interval distance when available.
    Otherwise use SOG × 10 min.
    Invalid/non-positive stored values are replaced row-wise by the kinematic
    10-min distance.
    """
    kinematic = raw["speed_kn"].astype(float) * (INTERVAL_MINUTES / 60.0)

    if "distance_nm" not in raw.columns:
        return kinematic

    observed = pd.to_numeric(raw["distance_nm"], errors="coerce")
    valid = np.isfinite(observed.to_numpy(float)) & (observed.to_numpy(float) > 0.0)

    out = kinematic.copy()
    out.loc[valid] = observed.loc[valid].astype(float)
    return out.astype(float)


def _co2_factor(raw: pd.DataFrame, default_cf: float) -> pd.Series:
    if "co2_factor" in raw.columns and raw["co2_factor"].notna().any():
        return pd.to_numeric(raw["co2_factor"], errors="coerce").fillna(default_cf)
    return pd.Series(float(default_cf), index=raw.index, dtype=float)


def _speed_support(train_raw: pd.DataFrame) -> pd.DataFrame:
    return (
        train_raw.groupby("ship_type")["speed_kn"]
        .agg(["min", "max"])
        .astype(float)
    )


def _supported_after_speed_reduction(
    raw: pd.DataFrame,
    support: pd.DataFrame,
    reduction: float,
) -> np.ndarray:
    """Ship-type-specific empirical support check for reduced SOG."""
    reduced_speed = raw["speed_kn"].astype(float).to_numpy() * (1.0 - reduction)
    ship_types = raw["ship_type"].astype(str).to_numpy()

    valid = np.ones(len(raw), dtype=bool)

    for i, (st, vr) in enumerate(zip(ship_types, reduced_speed)):
        if st not in support.index:
            valid[i] = False
            continue

        lo = float(support.loc[st, "min"])
        hi = float(support.loc[st, "max"])

        valid[i] = (
            np.isfinite(vr)
            and vr > 0.0
            and vr >= lo
            and vr <= hi
        )

    return valid


def _finite_change_elasticity(
    fuel_rate_ratio: float,
    speed_ratio: float,
) -> float:
    """
    Finite-change elasticity:
        epsilon = ln(q_r / q_0) / ln(V_r / V_0)
    """
    if (
        np.isfinite(fuel_rate_ratio)
        and np.isfinite(speed_ratio)
        and fuel_rate_ratio > 0.0
        and speed_ratio > 0.0
        and not np.isclose(speed_ratio, 1.0)
    ):
        return float(np.log(fuel_rate_ratio) / np.log(speed_ratio))
    return np.nan


# ---------------------------------------------------------------------------
# Scenario B: explicit interval construction
# ---------------------------------------------------------------------------

def build_fd_explicit_intervals(
    supported_base: pd.DataFrame,
    reduction: float,
) -> pd.DataFrame:
    """
    Materialise all time intervals required to traverse each baseline segment
    at the reduced speed while preserving the baseline segment distance.

    No voyage-level duration multiplier is used for fuel.

    Each baseline row may create:
      - zero or more complete 10-min rows;
      - at most one partial final row.

    The added rows inherit the source row's non-speed covariates because this
    is a local partial-equilibrium sensitivity experiment.

    Returns
    -------
    expanded : DataFrame
        Contains copies of the source raw rows plus audit fields:
        _source_row_id
        _segment_distance_nm
        _reduced_speed_kn
        _required_minutes
        _interval_sequence
        _interval_kind
        _interval_minutes
        _interval_distance_nm
        _is_added_time
    """
    reduction = float(reduction)
    if not 0.0 < reduction < 1.0:
        raise ValueError("reduction must be strictly between 0 and 1")

    rows = []

    for _, src in supported_base.iterrows():
        source = src.copy()

        source_row_id = int(source["_source_row_id"])
        base_speed = float(source["speed_kn"])
        reduced_speed = base_speed * (1.0 - reduction)
        segment_distance = float(source["distance_nm_used"])

        if (
            not np.isfinite(reduced_speed)
            or reduced_speed <= 0.0
            or not np.isfinite(segment_distance)
            or segment_distance <= 0.0
        ):
            continue

        # Exact kinematic time required to preserve THIS baseline segment.
        required_minutes = segment_distance / reduced_speed * 60.0

        if not np.isfinite(required_minutes) or required_minutes <= 0.0:
            continue

        # Number of complete 10-min intervals.
        n_full = int(np.floor((required_minutes + EPS) / INTERVAL_MINUTES))
        remainder = required_minutes - n_full * INTERVAL_MINUTES

        # Floating-point guard.
        if remainder < EPS:
            remainder = 0.0
        elif remainder > INTERVAL_MINUTES - EPS:
            n_full += 1
            remainder = 0.0

        seq = 0
        cumulative_minutes = 0.0
        cumulative_distance = 0.0

        # Complete 10-min intervals.
        for _j in range(n_full):
            seq += 1
            duration_min = INTERVAL_MINUTES
            distance_nm = reduced_speed * duration_min / 60.0

            r = source.copy()
            r["speed_kn"] = reduced_speed
            r["_source_row_id"] = source_row_id
            r["_segment_distance_nm"] = segment_distance
            r["_baseline_speed_kn"] = base_speed
            r["_reduced_speed_kn"] = reduced_speed
            r["_required_minutes"] = required_minutes
            r["_interval_sequence"] = seq
            r["_interval_kind"] = "full_10min"
            r["_interval_minutes"] = duration_min
            r["_interval_distance_nm"] = distance_nm

            cumulative_minutes += duration_min
            cumulative_distance += distance_nm

            # Added-time flag is based on whether this explicit interval extends
            # past the original 10-min observation window.
            r["_is_added_time"] = bool(cumulative_minutes > INTERVAL_MINUTES + EPS)

            rows.append(r)

        # Final partial interval if required.
        if remainder > 0.0:
            seq += 1

            # Use the exact remaining distance to ensure distance preservation.
            remaining_distance = segment_distance - cumulative_distance

            # Numerical guard; kinematically this equals V * remainder / 60.
            if remaining_distance < 0.0 and abs(remaining_distance) < 1e-8:
                remaining_distance = 0.0

            r = source.copy()
            r["speed_kn"] = reduced_speed
            r["_source_row_id"] = source_row_id
            r["_segment_distance_nm"] = segment_distance
            r["_baseline_speed_kn"] = base_speed
            r["_reduced_speed_kn"] = reduced_speed
            r["_required_minutes"] = required_minutes
            r["_interval_sequence"] = seq
            r["_interval_kind"] = "partial"
            r["_interval_minutes"] = remainder
            r["_interval_distance_nm"] = remaining_distance

            cumulative_minutes += remainder
            cumulative_distance += remaining_distance

            r["_is_added_time"] = bool(cumulative_minutes > INTERVAL_MINUTES + EPS)

            rows.append(r)

    if not rows:
        return pd.DataFrame()

    expanded = pd.DataFrame(rows).reset_index(drop=True)

    return expanded


def predict_fd_explicit_intervals(
    model,
    expanded: pd.DataFrame,
    feature_cols: Sequence[str],
    clip_negative_for_cii: bool = False,
) -> pd.DataFrame:
    """
    Predict every explicitly materialised FD interval.

    Full 10-min intervals use the model prediction directly.
    The final partial interval uses only a unit conversion:
        t/10min × partial_minutes/10
    """
    if expanded.empty:
        return expanded.copy()

    out = expanded.copy()

    pred_10min = _predict_10min(
        model=model,
        raw=out,
        feature_cols=feature_cols,
        clip_negative_for_cii=clip_negative_for_cii,
    )

    out["_predicted_fuel_t_per_10min"] = pred_10min

    # Required dimensional conversion for partial intervals.
    # For complete intervals this factor equals exactly 1.
    out["_predicted_interval_fuel_t"] = (
        out["_predicted_fuel_t_per_10min"].to_numpy(float)
        * out["_interval_minutes"].to_numpy(float)
        / INTERVAL_MINUTES
    )

    return out


# ---------------------------------------------------------------------------
# Main FT + explicit-FD calculation
# ---------------------------------------------------------------------------

def speed_reduction_scenarios_ft_fd_explicit(
    model,
    train_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
    feature_cols: List[str],
    reductions: Sequence[float] = (0.05, 0.10, 0.15),
    default_cf: float = 3.114,
    clip_negative_predictions_for_cii: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute Scenario A (FT) and Scenario B (FD) independently.

    Returns
    -------
    ft_vessel_df
    fd_vessel_df
    summary_df
    fd_interval_audit_df
    """
    base = test_raw.copy().reset_index(drop=True)

    required = {"speed_kn", "ship_type", "vessel_id", "dwt"}
    missing = sorted(required - set(base.columns))
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    if base["dwt"].isna().all():
        raise KeyError("DWT is required for CII calculations")

    base["_source_row_id"] = np.arange(len(base), dtype=int)
    base["distance_nm_used"] = _distance_used(base)
    base["cf_used"] = _co2_factor(base, default_cf)

    base_pred_all = _predict_10min(
        model=model,
        raw=base,
        feature_cols=feature_cols,
        clip_negative_for_cii=clip_negative_predictions_for_cii,
    )
    base["_baseline_pred_10min"] = base_pred_all

    support = _speed_support(train_raw)

    ft_rows = []
    fd_rows = []
    summary_rows = []
    all_fd_interval_audits = []

    for reduction in reductions:
        reduction = float(reduction)
        if not 0.0 < reduction < 1.0:
            raise ValueError(f"Invalid reduction: {reduction}")

        speed_ratio = 1.0 - reduction

        # Support is determined using the reduced speed.
        valid = _supported_after_speed_reduction(
            raw=base,
            support=support,
            reduction=reduction,
        )

        b = base.loc[valid].copy().reset_index(drop=True)

        if b.empty:
            continue

        # ==============================================================
        # Scenario A: fixed time
        # ==============================================================
        ft = b.copy()
        ft["speed_kn"] = ft["speed_kn"].astype(float) * speed_ratio

        ft_pred = _predict_10min(
            model=model,
            raw=ft,
            feature_cols=feature_cols,
            clip_negative_for_cii=clip_negative_predictions_for_cii,
        )

        ft["_scenario_pred_10min"] = ft_pred

        # Same 10-min observation duration; distance shrinks with speed.
        # Scale the measured/used baseline interval distance to preserve the
        # existing pipeline's distance basis.
        ft["_scenario_distance_nm"] = (
            ft["distance_nm_used"].astype(float) * speed_ratio
        )

        # ==============================================================
        # Scenario B: fixed distance, EXPLICIT added intervals
        # ==============================================================
        fd_expanded = build_fd_explicit_intervals(
            supported_base=b,
            reduction=reduction,
        )

        fd_expanded = predict_fd_explicit_intervals(
            model=model,
            expanded=fd_expanded,
            feature_cols=feature_cols,
            clip_negative_for_cii=clip_negative_predictions_for_cii,
        )

        if fd_expanded.empty:
            continue

        fd_expanded["_reduction_pct"] = reduction * 100.0

        # Copy CO2 factor from each source record.
        source_cf = b.set_index("_source_row_id")["cf_used"].to_dict()
        fd_expanded["_cf_used"] = (
            fd_expanded["_source_row_id"].map(source_cf).astype(float)
        )

        # Keep a detailed audit file so every additional interval is visible.
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

        all_fd_interval_audits.append(fd_expanded[audit_cols].copy())

        # ==============================================================
        # Vessel-level aggregation
        # ==============================================================
        for vessel, bg in b.groupby("vessel_id"):
            ship_type = str(bg["ship_type"].iloc[0])
            dwt = float(pd.to_numeric(bg["dwt"], errors="coerce").dropna().median())

            source_ids = set(bg["_source_row_id"].astype(int).tolist())

            fg = ft[ft["_source_row_id"].isin(source_ids)].copy()
            dg = fd_expanded[
                fd_expanded["_source_row_id"].isin(source_ids)
            ].copy()

            if fg.empty or dg.empty:
                continue

            # ---------------- Baseline ----------------
            base_fuel = float(bg["_baseline_pred_10min"].sum())
            base_distance = float(bg["distance_nm_used"].sum())
            base_duration_min = float(len(bg) * INTERVAL_MINUTES)

            base_co2_g = float(
                np.sum(
                    bg["_baseline_pred_10min"].to_numpy(float)
                    * bg["cf_used"].to_numpy(float)
                    * 1e6
                )
            )

            base_tw = dwt * base_distance
            base_cii = (
                base_co2_g / base_tw
                if base_tw > 0.0
                else np.nan
            )

            # ---------------- Scenario A / FT ----------------
            ft_fuel = float(fg["_scenario_pred_10min"].sum())
            ft_distance = float(fg["_scenario_distance_nm"].sum())
            ft_duration_min = float(len(fg) * INTERVAL_MINUTES)

            ft_co2_g = float(
                np.sum(
                    fg["_scenario_pred_10min"].to_numpy(float)
                    * fg["cf_used"].to_numpy(float)
                    * 1e6
                )
            )

            ft_tw = dwt * ft_distance
            ft_cii = (
                ft_co2_g / ft_tw
                if ft_tw > 0.0
                else np.nan
            )

            # ---------------- Scenario B / explicit FD ----------------
            fd_fuel = float(dg["_predicted_interval_fuel_t"].sum())
            fd_distance = float(dg["_interval_distance_nm"].sum())
            fd_duration_min = float(dg["_interval_minutes"].sum())

            fd_co2_g = float(
                np.sum(
                    dg["_predicted_interval_fuel_t"].to_numpy(float)
                    * dg["_cf_used"].to_numpy(float)
                    * 1e6
                )
            )

            fd_tw = dwt * fd_distance
            fd_cii = (
                fd_co2_g / fd_tw
                if fd_tw > 0.0
                else np.nan
            )

            # Fuel in the explicit FD intervals that lie beyond the original
            # 10-min windows. This is a direct sum, not a duration coefficient.
            fd_added_time_fuel = float(
                dg.loc[
                    dg["_is_added_time"].astype(bool),
                    "_predicted_interval_fuel_t",
                ].sum()
            )

            fd_added_time_minutes = max(
                fd_duration_min - base_duration_min,
                0.0,
            )

            # Number of explicitly materialised intervals.
            n_fd_full = int((dg["_interval_kind"] == "full_10min").sum())
            n_fd_partial = int((dg["_interval_kind"] == "partial").sum())
            n_fd_total = int(len(dg))

            fuel_rate_ratio = (
                ft_fuel / base_fuel
                if base_fuel > 0.0
                else np.nan
            )

            elasticity = _finite_change_elasticity(
                fuel_rate_ratio=fuel_rate_ratio,
                speed_ratio=speed_ratio,
            )

            ft_cii_ratio = (
                ft_cii / base_cii
                if np.isfinite(base_cii) and base_cii > 0.0
                else np.nan
            )
            fd_cii_ratio = (
                fd_cii / base_cii
                if np.isfinite(base_cii) and base_cii > 0.0
                else np.nan
            )

            retained_pct = (
                len(bg)
                / max(int((base["vessel_id"] == vessel).sum()), 1)
                * 100.0
            )

            common = {
                "reduction_pct": reduction * 100.0,
                "vessel_id": vessel,
                "ship_type": ship_type,
                "n_supported_source_records": int(len(bg)),
                "retained_pct": retained_pct,
                "DWT": dwt,
                "speed_ratio": speed_ratio,
                "baseline_predicted_fuel_t": base_fuel,
                "baseline_distance_nm": base_distance,
                "baseline_duration_min": base_duration_min,
                "baseline_CII_proxy": base_cii,
                "fuel_rate_ratio_from_FT_10min_predictions": fuel_rate_ratio,
                "finite_change_fuel_rate_elasticity": elasticity,
            }

            ft_rows.append({
                **common,
                "FT_duration_min": ft_duration_min,
                "FT_duration_change_pct": 0.0,
                "FT_distance_nm": ft_distance,
                "FT_distance_change_pct": (
                    (ft_distance / base_distance - 1.0) * 100.0
                    if base_distance > 0.0 else np.nan
                ),
                "FT_predicted_fuel_t": ft_fuel,
                "FT_fuel_change_pct": (
                    (ft_fuel / base_fuel - 1.0) * 100.0
                    if base_fuel > 0.0 else np.nan
                ),
                "FT_CII_proxy": ft_cii,
                "FT_CII_ratio": ft_cii_ratio,
                "FT_CII_change_pct": (
                    (ft_cii_ratio - 1.0) * 100.0
                    if np.isfinite(ft_cii_ratio) else np.nan
                ),
                "FT_improved": bool(ft_cii < base_cii)
                if np.isfinite(ft_cii) and np.isfinite(base_cii)
                else False,
            })

            fd_rows.append({
                **common,
                "FD_duration_min": fd_duration_min,
                "FD_added_sailing_time_min": fd_added_time_minutes,
                "FD_duration_change_pct": (
                    (fd_duration_min / base_duration_min - 1.0) * 100.0
                    if base_duration_min > 0.0 else np.nan
                ),
                "FD_distance_nm": fd_distance,
                "FD_distance_error_nm": fd_distance - base_distance,
                "FD_distance_error_pct": (
                    (fd_distance / base_distance - 1.0) * 100.0
                    if base_distance > 0.0 else np.nan
                ),
                "FD_explicit_full_10min_intervals": n_fd_full,
                "FD_explicit_partial_intervals": n_fd_partial,
                "FD_explicit_total_intervals": n_fd_total,
                "FD_predicted_total_fuel_t": fd_fuel,
                "FD_added_time_fuel_t": fd_added_time_fuel,
                "FD_added_time_fuel_pct_of_baseline": (
                    fd_added_time_fuel / base_fuel * 100.0
                    if base_fuel > 0.0 else np.nan
                ),
                "FD_total_fuel_change_pct": (
                    (fd_fuel / base_fuel - 1.0) * 100.0
                    if base_fuel > 0.0 else np.nan
                ),
                "FD_CII_proxy": fd_cii,
                "FD_CII_ratio": fd_cii_ratio,
                "FD_CII_change_pct": (
                    (fd_cii_ratio - 1.0) * 100.0
                    if np.isfinite(fd_cii_ratio) else np.nan
                ),
                "FD_improved": bool(fd_cii < base_cii)
                if np.isfinite(fd_cii) and np.isfinite(base_cii)
                else False,
                "FT_minus_FD_CII_change_pct_points": (
                    (ft_cii_ratio - fd_cii_ratio) * 100.0
                    if np.isfinite(ft_cii_ratio) and np.isfinite(fd_cii_ratio)
                    else np.nan
                ),
            })

    ft_vessel_df = pd.DataFrame(ft_rows)
    fd_vessel_df = pd.DataFrame(fd_rows)

    if all_fd_interval_audits:
        fd_interval_audit_df = pd.concat(
            all_fd_interval_audits,
            ignore_index=True,
        )
    else:
        fd_interval_audit_df = pd.DataFrame()

    # ------------------------------------------------------------------
    # Fleet / ship-type summary
    # ------------------------------------------------------------------
    if ft_vessel_df.empty or fd_vessel_df.empty:
        return (
            ft_vessel_df,
            fd_vessel_df,
            pd.DataFrame(),
            fd_interval_audit_df,
        )

    merged = ft_vessel_df.merge(
        fd_vessel_df,
        on=[
            "reduction_pct",
            "vessel_id",
            "ship_type",
            "n_supported_source_records",
            "retained_pct",
            "DWT",
            "speed_ratio",
            "baseline_predicted_fuel_t",
            "baseline_distance_nm",
            "baseline_duration_min",
            "baseline_CII_proxy",
            "fuel_rate_ratio_from_FT_10min_predictions",
            "finite_change_fuel_rate_elasticity",
        ],
        how="inner",
        validate="one_to_one",
    )

    for reduction_pct, rg in merged.groupby("reduction_pct"):
        groups = [("Fleet", rg)] + [
            (str(st), g)
            for st, g in rg.groupby("ship_type")
        ]

        for scope, sg in groups:
            base_fuel = float(sg["baseline_predicted_fuel_t"].sum())
            ft_fuel = float(sg["FT_predicted_fuel_t"].sum())
            fd_fuel = float(sg["FD_predicted_total_fuel_t"].sum())

            base_distance = float(sg["baseline_distance_nm"].sum())
            ft_distance = float(sg["FT_distance_nm"].sum())
            fd_distance = float(sg["FD_distance_nm"].sum())

            base_duration = float(sg["baseline_duration_min"].sum())
            ft_duration = float(sg["FT_duration_min"].sum())
            fd_duration = float(sg["FD_duration_min"].sum())

            # Aggregate CII through transport-work weights.
            base_w = (
                sg["DWT"].to_numpy(float)
                * sg["baseline_distance_nm"].to_numpy(float)
            )
            ft_w = (
                sg["DWT"].to_numpy(float)
                * sg["FT_distance_nm"].to_numpy(float)
            )
            fd_w = (
                sg["DWT"].to_numpy(float)
                * sg["FD_distance_nm"].to_numpy(float)
            )

            base_cii = float(
                np.average(
                    sg["baseline_CII_proxy"].to_numpy(float),
                    weights=np.maximum(base_w, 1e-12),
                )
            )
            ft_cii = float(
                np.average(
                    sg["FT_CII_proxy"].to_numpy(float),
                    weights=np.maximum(ft_w, 1e-12),
                )
            )
            fd_cii = float(
                np.average(
                    sg["FD_CII_proxy"].to_numpy(float),
                    weights=np.maximum(fd_w, 1e-12),
                )
            )

            ft_ratio = (
                ft_cii / base_cii
                if base_cii > 0.0 else np.nan
            )
            fd_ratio = (
                fd_cii / base_cii
                if base_cii > 0.0 else np.nan
            )

            fuel_rate_ratio = (
                ft_fuel / base_fuel
                if base_fuel > 0.0 else np.nan
            )
            speed_ratio = float(sg["speed_ratio"].iloc[0])

            elasticity = _finite_change_elasticity(
                fuel_rate_ratio=fuel_rate_ratio,
                speed_ratio=speed_ratio,
            )

            summary_rows.append({
                "reduction_pct": reduction_pct,
                "scope": scope,
                "vessels": int(len(sg)),
                "speed_ratio": speed_ratio,

                "baseline_distance_nm": base_distance,
                "FT_distance_nm": ft_distance,
                "FD_distance_nm": fd_distance,

                "baseline_duration_min": base_duration,
                "FT_duration_min": ft_duration,
                "FD_duration_min": fd_duration,
                "FD_added_sailing_time_min": fd_duration - base_duration,
                "FD_duration_change_pct": (
                    (fd_duration / base_duration - 1.0) * 100.0
                    if base_duration > 0.0 else np.nan
                ),

                "baseline_predicted_fuel_t": base_fuel,
                "FT_predicted_fuel_t": ft_fuel,
                "FT_fuel_change_pct": (
                    (ft_fuel / base_fuel - 1.0) * 100.0
                    if base_fuel > 0.0 else np.nan
                ),
                "FD_predicted_total_fuel_t": fd_fuel,
                "FD_total_fuel_change_pct": (
                    (fd_fuel / base_fuel - 1.0) * 100.0
                    if base_fuel > 0.0 else np.nan
                ),

                # Directly summed from explicit added-time intervals.
                "FD_added_time_fuel_t": float(
                    sg["FD_added_time_fuel_t"].sum()
                ),
                "FD_added_time_fuel_pct_of_baseline": (
                    float(sg["FD_added_time_fuel_t"].sum()) / base_fuel * 100.0
                    if base_fuel > 0.0 else np.nan
                ),

                "fuel_rate_ratio_from_FT_10min_predictions": fuel_rate_ratio,
                "finite_change_fuel_rate_elasticity": elasticity,

                "FT_CII_proxy": ft_cii,
                "FT_CII_ratio": ft_ratio,
                "FT_CII_change_pct_weighted": (
                    (ft_ratio - 1.0) * 100.0
                    if np.isfinite(ft_ratio) else np.nan
                ),

                "FD_CII_proxy": fd_cii,
                "FD_CII_ratio": fd_ratio,
                "FD_CII_change_pct_weighted": (
                    (fd_ratio - 1.0) * 100.0
                    if np.isfinite(fd_ratio) else np.nan
                ),

                "FT_minus_FD_CII_change_pct_points": (
                    (ft_ratio - fd_ratio) * 100.0
                    if np.isfinite(ft_ratio) and np.isfinite(fd_ratio)
                    else np.nan
                ),

                "FT_vessels_improved": int(sg["FT_improved"].sum()),
                "FD_vessels_improved": int(sg["FD_improved"].sum()),
                "mean_retained_pct": float(sg["retained_pct"].mean()),

                "FD_explicit_full_10min_intervals": int(
                    sg["FD_explicit_full_10min_intervals"].sum()
                ),
                "FD_explicit_partial_intervals": int(
                    sg["FD_explicit_partial_intervals"].sum()
                ),
                "FD_explicit_total_intervals": int(
                    sg["FD_explicit_total_intervals"].sum()
                ),
            })

    summary_df = pd.DataFrame(summary_rows)

    return (
        ft_vessel_df,
        fd_vessel_df,
        summary_df,
        fd_interval_audit_df,
    )


# ---------------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------------

def save_ft_fd_explicit_outputs(
    model,
    train_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
    feature_cols: List[str],
    output_dir,
    default_cf: float,
    reductions: Sequence[float] = (0.05, 0.10, 0.15),
    clip_negative_predictions_for_cii: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    (
        ft_vessel_df,
        fd_vessel_df,
        summary_df,
        fd_interval_audit_df,
    ) = speed_reduction_scenarios_ft_fd_explicit(
        model=model,
        train_raw=train_raw,
        test_raw=test_raw,
        feature_cols=feature_cols,
        reductions=reductions,
        default_cf=default_cf,
        clip_negative_predictions_for_cii=clip_negative_predictions_for_cii,
    )

    # Table 15: Scenario A
    ft_vessel_df.to_csv(
        out / "Table_C3_FT_by_vessel.csv",
        index=False,
    )

    # Table 16: Scenario B
    fd_vessel_df.to_csv(
        out / "Table_C4_FD_explicit_by_vessel.csv",
        index=False,
    )

    # Full combined summary
    summary_df.to_csv(
        out / "Table_C5_FT_FD_explicit_summary.csv",
        index=False,
    )

    # Row-level audit: proves that extra sailing intervals were actually
    # constructed and predicted.
    fd_interval_audit_df.to_csv(
        out / "Table_C6_FD_added_interval_audit.csv",
        index=False,
    )

    # Compact FT/FD comparison.
    comparison_cols = [
        "reduction_pct",
        "scope",
        "vessels",
        "FT_distance_nm",
        "FD_distance_nm",
        "FT_duration_min",
        "FD_duration_min",
        "FD_added_sailing_time_min",
        "FD_duration_change_pct",
        "FT_predicted_fuel_t",
        "FT_fuel_change_pct",
        "FD_predicted_total_fuel_t",
        "FD_total_fuel_change_pct",
        "FD_added_time_fuel_t",
        "FD_added_time_fuel_pct_of_baseline",
        "finite_change_fuel_rate_elasticity",
        "FT_CII_change_pct_weighted",
        "FD_CII_change_pct_weighted",
        "FT_minus_FD_CII_change_pct_points",
        "FT_vessels_improved",
        "FD_vessels_improved",
        "mean_retained_pct",
    ]

    if not summary_df.empty:
        summary_df[comparison_cols].to_csv(
            out / "Table_C7_FT_FD_comparison.csv",
            index=False,
        )

    return (
        ft_vessel_df,
        fd_vessel_df,
        summary_df,
        fd_interval_audit_df,
    )


# ---------------------------------------------------------------------------
# Fixed-distance interval reconstruction interface
# ---------------------------------------------------------------------------

def run_speed_scenarios_explicit(
    model,
    raw_train: pd.DataFrame,
    raw_test: pd.DataFrame,
    feature_cols: List[str],
    args,
    logger=None,
):
    """
    Drop-in-style wrapper for the existing empirical runner.
    """
    out = Path(args.output_dir) / "11_cii"

    if logger is not None:
        logger.info(
            "Running FT + explicit-interval FD CII scenarios; "
            "FD fuel is NOT computed with a voyage-duration multiplier."
        )

    results = save_ft_fd_explicit_outputs(
        model=model,
        train_raw=raw_train,
        test_raw=raw_test,
        feature_cols=feature_cols,
        output_dir=out,
        default_cf=args.co2_factor,
        reductions=(0.05, 0.10, 0.15),
        clip_negative_predictions_for_cii=False,
    )

    ft_vessel_df, fd_vessel_df, summary_df, fd_interval_audit_df = results

    if logger is not None and not summary_df.empty:
        fleet = summary_df[summary_df["scope"] == "Fleet"].copy()

        for _, row in fleet.iterrows():
            logger.info(
                "Speed reduction %.0f%% | FT fuel %.3f%% | "
                "FD fuel %.3f%% | explicit added-time fuel %.3f t | "
                "FT CII %.3f%% | FD CII %.3f%%",
                row["reduction_pct"],
                row["FT_fuel_change_pct"],
                row["FD_total_fuel_change_pct"],
                row["FD_added_time_fuel_t"],
                row["FT_CII_change_pct_weighted"],
                row["FD_CII_change_pct_weighted"],
            )

    return summary_df


if __name__ == "__main__":
    print(
        "Fixed-distance interval reconstruction module.\n"
        "Call run_speed_scenarios_explicit(...) from the analysis pipeline.\n"
    )
