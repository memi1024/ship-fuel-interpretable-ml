# -*- coding: utf-8 -*-
"""
A6b_build_zero_addback_candidates.py

Reviewer-facing A6b builder for the zero-fuel-underway sensitivity analysis.

Purpose
-------
Identify container zero-fuel-underway rows from the V15 flagged table, restore the
literal zero target ONLY for this sensitivity branch, recompute the V15 speed
stability screen with those rows present, and then pass the candidates through
all downstream NON-FUEL-SPECIFIC Fixed28/Fixed29/Fixed31 QC.

The canonical Fixed31 dataset is never modified.

Design
------
1) Candidate rule:
       ship_type == "container"
       fuel_t_10min_raw == 0
       speed_kn_raw > 1 kn

2) Recompute V15 speed stability using all base-valid observations after
   restoring the tagged container zero rows.  The original V15 rule is:
       - continuous block breaks when gap between valid rows > 15 min;
       - trailing 3-record sample SD;
       - first two valid rows in a block retained;
       - thereafter SD <= 0.5 kn (default).

3) Select cruise candidates and require Fixed27 complete-case eligibility.

4) Replay non-fuel-specific downstream QC:
   Fixed28:
       - bad-vessel exclusion;
       - draught/trim validity;
       - fuel POWER UPPER bound (zero naturally passes);
       - wave period 0.5..25 s;
       - surface temperature -5..37 C;
       - model completeness.
       EXCEPTION: the tagged zero rows are not rejected by fuel < 1e-5.

   Fixed29:
       - derived fore/aft draught > 0;
       - speed <= 1.5 * service speed when service speed is valid;
       - model completeness.

   Fixed31:
       - derived fore/aft draught finite and > 0;
       - design draught finite and > 0;
       - minimum end draught / design draught >= 0.05;
       - model completeness.
       EXCEPTIONS: tagged zero rows are not rejected by the severe-low-fuel
       rule or the final minimum-positive-fuel recheck.

5) Align surviving candidate rows to the exact canonical Fixed31 column order.
   These rows are intended to be APPENDED TO TRAINING ONLY.  The official
   Fixed31 test rows and test indices remain unchanged.

This is deliberately a stress-test branch.  It does not claim that zero fuel is
physically correct; it tests sensitivity to the asymmetric exclusion rule.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd


KEY_COLUMNS = [
    "ship_type",
    "pseudo_ship_group_id",
    "trajectory_segment_id",
    "timestamp_utc",
]

STATIC4_PREDICTORS = [
    "design_draught_m",
    "deadweight_t",
    "service_speed_kn",
    "main_engine_power_kw",
]

FIXED23_PREDICTORS = [
    "speed_kn",
    "course_sin",
    "course_cos",
    "heading_sin",
    "heading_cos",
    "mean_draught_m",
    "rudder_deg",
    "distance_nm_10min",
    "rel_wind_speed_kn",
    "relative_wind_sin",
    "relative_wind_cos",
    "wave_height_m",
    "wave_period_s",
    "relative_wave_sin",
    "relative_wave_cos",
    "surface_pressure_pa",
    "surface_temperature_c",
    "rel_wind_speed_x_speed",
    "wave_height_x_speed",
    "draught_x_speed",
    "trim_m",
    "trim_x_speed",
    "draught_x_trim",
]

MODEL_VARIABLES = [
    "fuel_t_10min",
    "speed_kn",
    "heading_sin",
    "heading_cos",
    "mean_draught_m",
    "rudder_deg",
    "trim_m",
    "rel_wind_speed_kn",
    "relative_wind_sin",
    "relative_wind_cos",
    "wave_height_m",
    "wave_period_s",
    "relative_wave_sin",
    "relative_wave_cos",
    "surface_pressure_pa",
    "surface_temperature_c",
]

INTERACTIONS = {
    "rel_wind_speed_x_speed": ("rel_wind_speed_kn", "speed_kn"),
    "wave_height_x_speed": ("wave_height_m", "speed_kn"),
    "draught_x_speed": ("mean_draught_m", "speed_kn"),
    "trim_x_speed": ("trim_m", "speed_kn"),
    "draught_x_trim": ("mean_draught_m", "trim_m"),
}

BAD_SHIP_ID = "TANKER_F25527857_P0001"
SFOC_MAX_G_KWH = 250.0
MIN_FUEL_T_10MIN = 1e-5
WAVE_PERIOD_MIN = 0.5
WAVE_PERIOD_MAX = 25.0
TEMPERATURE_MIN = -5.0
TEMPERATURE_MAX = 37.0
SPEED_SERVICE_RATIO_LIMIT = 1.5
MIN_END_DESIGN_RATIO_DELETE = 0.05


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build Fixed31-eligible zero-fuel add-back candidate rows."
    )
    p.add_argument(
        "--v15-flags",
        required=True,
        type=Path,
        help="D:/PAPERDATA/05_clean23/cleaned_10min_with_flags.csv",
    )
    p.add_argument(
        "--canonical-fixed31",
        required=True,
        type=Path,
        help="Canonical Fixed31 cruise dataset used by the manuscript.",
    )
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--ship-type", default="container")
    p.add_argument("--speed-threshold", type=float, default=1.0)
    p.add_argument("--speed-std-threshold", type=float, default=0.5)
    p.add_argument("--gap-minutes", type=float, default=15.0)
    p.add_argument("--chunksize", type=int, default=100_000)
    return p.parse_args()


def setup_logging(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                out / "A6b_build_zero_addback.log",
                mode="w",
                encoding="utf-8",
            ),
        ],
        force=True,
    )


def finite_numeric(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    return x.where(np.isfinite(x))


def flag1(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)
    return pd.to_numeric(s, errors="coerce").eq(1)


def candidate_mask(df: pd.DataFrame, ship_type: str, threshold: float) -> pd.Series:
    ship = df["ship_type"].astype("string").str.strip().str.lower()
    speed_raw = finite_numeric(df["speed_kn_raw"])
    fuel_raw = finite_numeric(df["fuel_t_10min_raw"])
    return (
        ship.eq(ship_type.lower())
        & speed_raw.gt(threshold)
        & fuel_raw.eq(0.0)
    )


def composite_key_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df[KEY_COLUMNS].copy()


def stable_row_id(df: pd.DataFrame) -> pd.Series:
    # Match the A6a row-id implementation.
    keys = df[KEY_COLUMNS].astype("string").fillna("<NA>")
    h = pd.util.hash_pandas_object(keys, index=False).astype("uint64")
    return h.map(lambda x: f"{int(x):016x}")


def require_columns(header: list[str], required: list[str], label: str) -> None:
    missing = sorted(set(required) - set(header))
    if missing:
        raise KeyError(f"{label} missing required columns: {', '.join(missing)}")


def finite_all(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for col in columns:
        x = pd.to_numeric(df[col], errors="coerce")
        mask &= np.isfinite(x)
    return mask


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(block_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def scan_candidate_inclusive_stability(
    path: Path,
    ship_type: str,
    speed_threshold: float,
    speed_std_threshold: float,
    gap_minutes: float,
    chunksize: int,
) -> pd.DataFrame:
    """
    Stream the V15 flagged table in its canonical sorted order and compute the
    candidate-inclusive V15 stability result only for tagged zero rows.

    State is maintained per (ship_type, vessel, trajectory segment), so chunk
    boundaries do not change the rolling calculation.
    """
    header = list(pd.read_csv(path, nrows=0, encoding="utf-8-sig").columns)
    required = [
        *KEY_COLUMNS,
        "primary_key_valid",
        "complete_window_flag",
        "speed_kn_raw",
        "speed_kn",
        "fuel_t_10min_raw",
        "fuel_t_10min",
        "voyage_phase",
    ]
    require_columns(header, required, "V15 flagged table")

    usecols = required
    states: dict[tuple[str, str, str], dict] = {}
    candidate_rows = []
    total = 0
    candidate_total = 0

    reader = pd.read_csv(
        path,
        usecols=usecols,
        chunksize=chunksize,
        encoding="utf-8-sig",
        low_memory=True,
    )

    last_sort_key = None

    for chunk_no, df in enumerate(reader, start=1):
        total += len(df)
        df["ship_type"] = df["ship_type"].astype("string").str.strip().str.lower()
        df["pseudo_ship_group_id"] = (
            df["pseudo_ship_group_id"].astype("string").str.strip()
        )
        df["trajectory_segment_id"] = (
            df["trajectory_segment_id"].astype("string").str.strip()
        )
        ts = pd.to_datetime(df["timestamp_utc"], errors="coerce", utc=True)

        speed_raw = finite_numeric(df["speed_kn_raw"])
        speed = finite_numeric(df["speed_kn"])
        fuel_raw = finite_numeric(df["fuel_t_10min_raw"])
        fuel_clean = finite_numeric(df["fuel_t_10min"])
        is_candidate = (
            df["ship_type"].eq(ship_type.lower())
            & speed_raw.gt(speed_threshold)
            & fuel_raw.eq(0.0)
        )
        candidate_total += int(is_candidate.sum())

        fuel_alt = fuel_clean.copy()
        fuel_alt.loc[is_candidate] = 0.0

        base_valid_alt = (
            flag1(df["primary_key_valid"])
            & flag1(df["complete_window_flag"])
            & speed.notna()
            & fuel_alt.notna()
        )

        # itertuples is used intentionally: only ~1.04M rows and stateful rolling
        # logic must continue exactly across chunk boundaries.
        local = pd.DataFrame({
            "ship_type": df["ship_type"],
            "vessel": df["pseudo_ship_group_id"],
            "segment": df["trajectory_segment_id"],
            "timestamp_raw": df["timestamp_utc"],
            "timestamp": ts,
            "speed": speed,
            "is_candidate": is_candidate,
            "base_valid_alt": base_valid_alt,
            "voyage_phase": df["voyage_phase"],
        })

        for row in local.itertuples(index=False):
            group = (str(row.ship_type), str(row.vessel), str(row.segment))
            timestamp = row.timestamp

            # The V15 producer writes this file ordered by
            # ship_type/vessel/segment/timestamp.  Detect gross order failures.
            sort_key = (
                group[0],
                group[1],
                group[2],
                timestamp.value if not pd.isna(timestamp) else -1,
            )
            if last_sort_key is not None and sort_key < last_sort_key:
                raise RuntimeError(
                    "V15 flagged input is not in canonical key/time order. "
                    "Do not use streaming stability replay on an unsorted file."
                )
            last_sort_key = sort_key

            speed_n = 0
            speed_std = math.nan
            stable = False
            block_id = math.nan

            if bool(row.base_valid_alt) and not pd.isna(timestamp):
                state = states.get(group)
                if state is None:
                    state = {
                        "last_valid_timestamp": None,
                        "block_id": 0,
                        "speeds": deque(maxlen=3),
                    }
                    states[group] = state

                last_ts = state["last_valid_timestamp"]
                new_block = (
                    last_ts is None
                    or (timestamp - last_ts).total_seconds() / 60.0 > gap_minutes
                )
                if new_block:
                    state["block_id"] += 1
                    state["speeds"].clear()

                state["speeds"].append(float(row.speed))
                state["last_valid_timestamp"] = timestamp

                speed_n = len(state["speeds"])
                block_id = state["block_id"]
                if speed_n >= 2:
                    speed_std = float(np.std(
                        np.asarray(state["speeds"], dtype=float),
                        ddof=1,
                    ))
                stable = (
                    speed_n < 3
                    or (
                        np.isfinite(speed_std)
                        and speed_std <= speed_std_threshold
                    )
                )

            if bool(row.is_candidate):
                candidate_rows.append({
                    "ship_type": str(row.ship_type),
                    "pseudo_ship_group_id": str(row.vessel),
                    "trajectory_segment_id": str(row.segment),
                    "timestamp_utc": str(row.timestamp_raw),
                    "voyage_phase_scan": str(row.voyage_phase),
                    "a6_base_valid_alt": int(bool(row.base_valid_alt)),
                    "a6_stability_block_id_alt": block_id,
                    "a6_speed_n_3_alt": speed_n,
                    "a6_speed_std_3_alt": speed_std,
                    "a6_speed_stable_alt": int(bool(stable)),
                })

        if chunk_no % 10 == 0:
            logging.info(
                "stability pass | chunks=%d rows=%s candidates=%s",
                chunk_no, f"{total:,}", f"{candidate_total:,}"
            )

    out = pd.DataFrame(candidate_rows)
    if out.empty:
        raise RuntimeError("No zero-fuel-underway candidates found.")

    dup = out.duplicated(KEY_COLUMNS, keep=False)
    if dup.any():
        raise RuntimeError(
            f"Candidate composite key is not unique: {int(dup.sum())} rows."
        )

    logging.info(
        "stability pass complete | rows=%s candidates=%s",
        f"{total:,}", f"{len(out):,}"
    )
    return out


def load_candidate_rows(
    path: Path,
    canonical_columns: list[str],
    ship_type: str,
    speed_threshold: float,
    chunksize: int,
) -> pd.DataFrame:
    header = list(pd.read_csv(path, nrows=0, encoding="utf-8-sig").columns)

    extras = [
        "speed_kn_raw",
        "fuel_t_10min_raw",
        "fuel_zero_underway_flag",
        "primary_key_valid",
        "complete_window_flag",
        "voyage_phase",
    ]
    requested = list(dict.fromkeys(canonical_columns + extras + KEY_COLUMNS))
    usecols = [c for c in requested if c in header]

    # Every canonical column must be recoverable from the V15 flagged table,
    # except fuel_t_10min and speed_std_3, which are explicitly replaced below
    # but still should normally exist.
    missing_canonical = [c for c in canonical_columns if c not in header]
    if missing_canonical:
        raise KeyError(
            "Canonical Fixed31 contains columns not present in the V15 flagged "
            "table: " + ", ".join(missing_canonical)
        )

    blocks = []
    reader = pd.read_csv(
        path,
        usecols=usecols,
        chunksize=chunksize,
        encoding="utf-8-sig",
        low_memory=True,
    )
    for df in reader:
        mask = candidate_mask(df, ship_type, speed_threshold)
        if mask.any():
            blocks.append(df.loc[mask].copy())

    if not blocks:
        raise RuntimeError("No candidate rows found during extraction pass.")

    out = pd.concat(blocks, ignore_index=True)
    out["ship_type"] = out["ship_type"].astype("string").str.strip().str.lower()
    out["pseudo_ship_group_id"] = (
        out["pseudo_ship_group_id"].astype("string").str.strip()
    )
    out["trajectory_segment_id"] = (
        out["trajectory_segment_id"].astype("string").str.strip()
    )
    return out


def apply_downstream_candidate_qc(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    x = df.copy()

    # Restore literal-zero target for the sensitivity branch.
    x["fuel_t_10min"] = 0.0
    x["speed_std_3"] = pd.to_numeric(
        x["a6_speed_std_3_alt"], errors="coerce"
    )

    # Recalculate interactions to mirror Fixed28/Fixed29/Fixed31 producers.
    for out, (left, right) in INTERACTIONS.items():
        x[out] = (
            pd.to_numeric(x[left], errors="coerce")
            * pd.to_numeric(x[right], errors="coerce")
        )

    # Stage flags are cumulative.
    x["pass_cruise"] = (
        x["voyage_phase"].astype("string").str.lower().eq("cruise")
    )
    x["pass_alt_speed_stability"] = (
        pd.to_numeric(x["a6_base_valid_alt"], errors="coerce").eq(1)
        & pd.to_numeric(x["a6_speed_stable_alt"], errors="coerce").eq(1)
    )

    fixed27_vars = ["fuel_t_10min"] + FIXED23_PREDICTORS + STATIC4_PREDICTORS
    x["pass_fixed27_complete"] = finite_all(x, fixed27_vars)

    # Fixed28 non-fuel-specific QC plus the fuel upper bound.
    draught = pd.to_numeric(x["mean_draught_m"], errors="coerce")
    trim = pd.to_numeric(x["trim_m"], errors="coerce")
    power = pd.to_numeric(x["main_engine_power_kw"], errors="coerce")
    fuel = pd.to_numeric(x["fuel_t_10min"], errors="coerce")
    wave_period = pd.to_numeric(x["wave_period_s"], errors="coerce")
    temperature = pd.to_numeric(x["surface_temperature_c"], errors="coerce")

    x["pass_fixed28_bad_ship"] = (
        x["pseudo_ship_group_id"].astype("string") != BAD_SHIP_ID
    )
    x["pass_fixed28_draught_trim"] = (
        np.isfinite(draught) & draught.gt(0) & np.isfinite(trim)
    )
    theoretical_max = power * SFOC_MAX_G_KWH / 6_000_000.0
    valid_power = np.isfinite(power) & power.gt(0)
    x["pass_fixed28_fuel_upper"] = ~(
        valid_power & fuel.gt(theoretical_max)
    )
    x["fixed28_min_fuel_exception_applied"] = fuel.lt(MIN_FUEL_T_10MIN)
    x["pass_fixed28_wave_period"] = (
        np.isfinite(wave_period)
        & wave_period.ge(WAVE_PERIOD_MIN)
        & wave_period.le(WAVE_PERIOD_MAX)
    )
    x["pass_fixed28_temperature"] = (
        np.isfinite(temperature)
        & temperature.ge(TEMPERATURE_MIN)
        & temperature.le(TEMPERATURE_MAX)
    )
    x["pass_fixed28_model_complete"] = finite_all(x, MODEL_VARIABLES)

    # Fixed29.
    mean_draught = pd.to_numeric(x["mean_draught_m"], errors="coerce")
    x["derived_aft_draught_m"] = mean_draught + trim / 2.0
    x["derived_fore_draught_m"] = mean_draught - trim / 2.0
    x["pass_fixed29_derived_draught"] = (
        np.isfinite(x["derived_aft_draught_m"])
        & np.isfinite(x["derived_fore_draught_m"])
        & x["derived_aft_draught_m"].gt(0)
        & x["derived_fore_draught_m"].gt(0)
    )

    speed = pd.to_numeric(x["speed_kn"], errors="coerce")
    service = pd.to_numeric(x["service_speed_kn"], errors="coerce")
    valid_service = np.isfinite(service) & service.gt(0)
    ratio = np.where(valid_service, speed / service, np.nan)
    x["a6_speed_service_ratio"] = ratio
    x["pass_fixed29_speed_service"] = ~(
        valid_service & pd.Series(ratio, index=x.index).gt(
            SPEED_SERVICE_RATIO_LIMIT
        )
    )
    x["pass_fixed29_model_complete"] = finite_all(x, MODEL_VARIABLES)

    # Fixed31 non-fuel-specific draught QC.
    design = pd.to_numeric(x["design_draught_m"], errors="coerce")
    min_end = pd.concat(
        [
            pd.to_numeric(x["derived_fore_draught_m"], errors="coerce"),
            pd.to_numeric(x["derived_aft_draught_m"], errors="coerce"),
        ],
        axis=1,
    ).min(axis=1)
    min_ratio = min_end / design
    x["a6_minimum_end_draught_m"] = min_end
    x["a6_minimum_end_design_ratio"] = min_ratio
    x["pass_fixed31_draught"] = (
        np.isfinite(x["derived_fore_draught_m"])
        & np.isfinite(x["derived_aft_draught_m"])
        & np.isfinite(design)
        & design.gt(0)
        & x["derived_fore_draught_m"].gt(0)
        & x["derived_aft_draught_m"].gt(0)
        & np.isfinite(min_ratio)
        & min_ratio.ge(MIN_END_DESIGN_RATIO_DELETE)
    )
    x["fixed31_severe_low_fuel_exception_applied"] = True
    x["fixed31_minimum_positive_exception_applied"] = True
    x["pass_fixed31_model_complete"] = finite_all(x, MODEL_VARIABLES)

    cumulative = pd.Series(True, index=x.index)
    stage_columns = [
        "pass_cruise",
        "pass_alt_speed_stability",
        "pass_fixed27_complete",
        "pass_fixed28_bad_ship",
        "pass_fixed28_draught_trim",
        "pass_fixed28_fuel_upper",
        "pass_fixed28_wave_period",
        "pass_fixed28_temperature",
        "pass_fixed28_model_complete",
        "pass_fixed29_derived_draught",
        "pass_fixed29_speed_service",
        "pass_fixed29_model_complete",
        "pass_fixed31_draught",
        "pass_fixed31_model_complete",
    ]

    flow = []
    for stage in stage_columns:
        cumulative &= x[stage].astype(bool)
        flow.append({
            "stage": stage,
            "surviving_candidates": int(cumulative.sum()),
            "removed_at_stage": int((~x[stage].astype(bool) & cumulative.shift(
                fill_value=True
            )).sum()) if False else np.nan,
        })

    # Compute removed-at-stage correctly from sequential masks.
    cumulative = pd.Series(True, index=x.index)
    flow = []
    previous_n = int(cumulative.sum())
    for stage in stage_columns:
        cumulative &= x[stage].astype(bool)
        now = int(cumulative.sum())
        flow.append({
            "stage": stage,
            "surviving_candidates": now,
            "removed_at_stage": previous_n - now,
        })
        previous_n = now

    x["a6b_final_eligible"] = cumulative
    return x, pd.DataFrame(flow)


def main() -> None:
    args = parse_args()
    setup_logging(args.output_dir)

    for p in (args.v15_flags, args.canonical_fixed31):
        if not p.is_file():
            raise FileNotFoundError(p)

    canonical_header = list(
        pd.read_csv(args.canonical_fixed31, nrows=0, encoding="utf-8-sig").columns
    )
    require_columns(
        canonical_header,
        ["ship_type", "pseudo_ship_group_id", "fuel_t_10min", "speed_kn"],
        "canonical Fixed31",
    )

    logging.info("Pass 1/3: recompute candidate-inclusive V15 speed stability.")
    stability = scan_candidate_inclusive_stability(
        args.v15_flags,
        args.ship_type,
        args.speed_threshold,
        args.speed_std_threshold,
        args.gap_minutes,
        args.chunksize,
    )

    logging.info("Pass 2/3: extract tagged zero-underway candidate rows.")
    candidates = load_candidate_rows(
        args.v15_flags,
        canonical_header,
        args.ship_type,
        args.speed_threshold,
        args.chunksize,
    )

    candidates = candidates.merge(
        stability,
        on=KEY_COLUMNS,
        how="left",
        validate="one_to_one",
    )
    if candidates["a6_speed_stable_alt"].isna().any():
        raise RuntimeError("Some candidate rows have no stability replay result.")

    candidates["a6_row_id"] = stable_row_id(candidates)

    # Independent consistency check against source V15 flag when present.
    if "fuel_zero_underway_flag" in candidates.columns:
        mismatches = int(
            (~flag1(candidates["fuel_zero_underway_flag"])).sum()
        )
        if mismatches:
            raise RuntimeError(
                f"{mismatches} candidates disagree with V15 zero-underway flag."
            )

    logging.info("Pass 3/3: replay downstream non-fuel QC for candidates.")
    audited, flow = apply_downstream_candidate_qc(candidates)

    # Prepend initial candidate count to the flow.
    flow = pd.concat(
        [
            pd.DataFrame([{
                "stage": "initial_zero_underway_candidates_all_phases",
                "surviving_candidates": int(len(audited)),
                "removed_at_stage": 0,
            }]),
            flow,
        ],
        ignore_index=True,
    )
    flow.to_csv(
        args.output_dir / "A6b_candidate_flow.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Vessel-level survival.
    vessel_rows = []
    for vessel, g in audited.groupby("pseudo_ship_group_id", dropna=False):
        vessel_rows.append({
            "pseudo_ship_group_id": vessel,
            "initial_candidates": int(len(g)),
            "cruise_candidates": int(g["pass_cruise"].sum()),
            "speed_stable_candidates": int(
                (g["pass_cruise"] & g["pass_alt_speed_stability"]).sum()
            ),
            "fixed31_eligible_addback": int(g["a6b_final_eligible"].sum()),
        })
    pd.DataFrame(vessel_rows).to_csv(
        args.output_dir / "A6b_candidate_flow_by_vessel.csv",
        index=False,
        encoding="utf-8-sig",
    )

    audit_cols = list(dict.fromkeys(
        ["a6_row_id"] + KEY_COLUMNS
        + [
            "voyage_phase",
            "speed_kn_raw",
            "speed_kn",
            "fuel_t_10min_raw",
            "fuel_t_10min",
            "a6_base_valid_alt",
            "a6_stability_block_id_alt",
            "a6_speed_n_3_alt",
            "a6_speed_std_3_alt",
            "a6_speed_stable_alt",
        ]
        + [c for c in audited.columns if c.startswith("pass_")]
        + [
            "fixed28_min_fuel_exception_applied",
            "fixed31_severe_low_fuel_exception_applied",
            "fixed31_minimum_positive_exception_applied",
            "a6_speed_service_ratio",
            "a6_minimum_end_draught_m",
            "a6_minimum_end_design_ratio",
            "a6b_final_eligible",
        ]
    ))
    audit_cols = [c for c in audit_cols if c in audited.columns]
    audited[audit_cols].to_csv(
        args.output_dir / "A6b_candidate_row_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    eligible = audited.loc[audited["a6b_final_eligible"]].copy()

    # Ensure the candidate target and recomputed stability field are what the
    # sensitivity branch intends to expose to modeling.
    eligible["fuel_t_10min"] = 0.0
    if "speed_std_3" in eligible.columns:
        eligible["speed_std_3"] = eligible["a6_speed_std_3_alt"]

    for out, (left, right) in INTERACTIONS.items():
        if out in eligible.columns:
            eligible[out] = (
                pd.to_numeric(eligible[left], errors="coerce")
                * pd.to_numeric(eligible[right], errors="coerce")
            )

    # Check that no add-back key already exists in the canonical Fixed31.
    canonical_keys = pd.read_csv(
        args.canonical_fixed31,
        usecols=[c for c in KEY_COLUMNS if c in canonical_header],
        encoding="utf-8-sig",
        low_memory=True,
    )
    if all(c in canonical_keys.columns for c in KEY_COLUMNS):
        base_key_set = set(map(tuple, canonical_keys[KEY_COLUMNS].astype(str).to_numpy()))
        add_key_tuples = list(map(tuple, eligible[KEY_COLUMNS].astype(str).to_numpy()))
        overlap = sum(k in base_key_set for k in add_key_tuples)
    else:
        overlap = 0
        logging.warning(
            "Canonical Fixed31 lacks one or more composite-key fields; "
            "exact overlap check was skipped."
        )

    if overlap:
        raise RuntimeError(
            f"{overlap} eligible add-back rows already exist in canonical Fixed31."
        )

    # Modeling file: exact canonical column order only.
    eligible_model = eligible[canonical_header].copy()
    eligible_model.to_csv(
        args.output_dir / "A6b_zero_addback_fixed31_eligible.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Separate traceability file with row IDs + canonical modeling columns.
    trace = eligible[["a6_row_id"] + canonical_header].copy()
    trace.to_csv(
        args.output_dir / "A6b_zero_addback_fixed31_eligible_with_ids.csv",
        index=False,
        encoding="utf-8-sig",
    )

    baseline_n = len(canonical_keys)
    added_n = len(eligible_model)
    extension = pd.DataFrame([{
        "canonical_fixed31_rows": baseline_n,
        "eligible_addback_rows": added_n,
        "augmented_rows_if_appended": baseline_n + added_n,
        "appended_index_start_zero_based": baseline_n if added_n else np.nan,
        "appended_index_end_zero_based":
            baseline_n + added_n - 1 if added_n else np.nan,
        "test_set_policy": "canonical official test indices unchanged",
        "training_policy": "append all eligible A6 zero rows to training only",
    }])
    extension.to_csv(
        args.output_dir / "A6b_training_extension_plan.csv",
        index=False,
        encoding="utf-8-sig",
    )

    manifest = {
        "task": "A6b_build_zero_addback_candidates",
        "v15_flags": str(args.v15_flags.resolve()),
        "canonical_fixed31": str(args.canonical_fixed31.resolve()),
        "canonical_fixed31_sha256": sha256_file(args.canonical_fixed31),
        "candidate_rule": (
            f"ship_type={args.ship_type.lower()}, "
            f"fuel_t_10min_raw==0, speed_kn_raw>{args.speed_threshold:g}"
        ),
        "speed_stability": {
            "gap_minutes_new_block_if_greater_than": args.gap_minutes,
            "trailing_points": 3,
            "sample_std_threshold_kn": args.speed_std_threshold,
            "first_two_valid_rows_retained": True,
            "candidate_inclusive_replay": True,
        },
        "fuel_specific_exceptions_for_tagged_rows_only": {
            "Fixed28_minimum_fuel_1e-5": True,
            "Fixed31_severe_low_fuel_removal": True,
            "Fixed31_minimum_positive_fuel_recheck": True,
        },
        "all_other_downstream_qc": "replayed for candidates",
        "initial_candidates_all_phases": int(len(audited)),
        "initial_cruise_candidates": int(audited["pass_cruise"].sum()),
        "eligible_addback_rows": int(added_n),
        "canonical_key_overlap": int(overlap),
        "modeling_policy": (
            "append eligible rows to training only; canonical Fixed31 test rows "
            "and official test indices remain unchanged"
        ),
        "interpretation": (
            "literal-zero stress test; does not assert that zero fuel is a "
            "physically valid measurement"
        ),
    }
    (args.output_dir / "A6b_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    logging.info(
        "A6b BUILD COMPLETE | all candidates=%s | cruise=%s | eligible=%s | "
        "canonical overlap=%s",
        f"{len(audited):,}",
        f"{int(audited['pass_cruise'].sum()):,}",
        f"{added_n:,}",
        f"{overlap:,}",
    )


if __name__ == "__main__":
    main()
