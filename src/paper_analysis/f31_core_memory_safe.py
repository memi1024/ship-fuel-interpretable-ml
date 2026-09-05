#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
F31 journal-final empirical pipeline for ship fuel-consumption prediction,
generalisation, physical-consistency auditing, and main-engine CII-proxy analysis.

Design principles
-----------------
1) One locked data audit, one common record-level split, and shared tuning samples/folds.
2) Three-stage hyperparameter procedure:
   Stage A: Optuna/Bayesian broad search on a 30k training subset.
   Stage B: re-evaluate the top 3 candidates on up to 100k training rows.
   Stage C: fit the selected configuration on the full outer training data.
3) Explicit separation of:
   - record-level interpolation,
   - known-vessel temporal forecasting,
   - locked-hyperparameter leave-one-vessel-out (LOVO) zero-shot transfer,
   - optional new-vessel 80% adaptation.
4) SHAP is used for model-behaviour interpretation, not as a causal coefficient or
   a direct CII mathematical weight.
5) CII calculations are observation-period main-engine proxies only.

The script is intentionally defensive about column naming. It resolves common aliases,
prints the mapping, and writes it to run_manifest.json. If a required field cannot be
resolved, execution stops with a clear error rather than silently guessing.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import platform
import shutil
import sys
import time
import traceback
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

try:
    import statsmodels.api as sm
except Exception:
    sm = None

from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

try:
    import optuna
except Exception as exc:  # pragma: no cover
    raise RuntimeError("optuna is required. Install with: pip install optuna") from exc

try:
    from xgboost import XGBRegressor
except Exception as exc:  # pragma: no cover
    raise RuntimeError("xgboost is required. Install with: pip install xgboost") from exc

try:
    from lightgbm import LGBMRegressor
except Exception as exc:  # pragma: no cover
    raise RuntimeError("lightgbm is required. Install with: pip install lightgbm") from exc

try:
    import shap
except Exception as exc:  # pragma: no cover
    raise RuntimeError("shap is required. Install with: pip install shap") from exc

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Constants and aliases
# -----------------------------------------------------------------------------

SCRIPT_VERSION = "2026-08-08.final3-memory-safe-csv"

PHASE_ORDER = ["cruise", "maneuver", "anchor_berth"]
SHIP_TYPES = ["bulk", "container", "tanker"]

COLUMN_ALIASES: Dict[str, List[str]] = {
    "vessel_id": [
        "pseudo_ship_group_id", "ship_group_id", "vessel_id", "ship_id", "imo",
        "IMO", "mmsi", "MMSI", "ship_no", "vessel",
    ],
    "ship_type": [
        "ship_type", "vessel_type", "type", "ship_category", "vessel_category",
    ],
    "timestamp": [
        "timestamp", "timestamp_utc", "datetime", "date_time", "utc_time", "time_utc", "ais_time",
        "base_datetime", "DateTime", "UTC", "time",
    ],
    "phase": ["phase", "voyage_phase", "operating_phase", "operation_phase", "regime"],
    "target": [
        "FOC10", "foc10", "foc_10min", "foc_10min_t", "fuel_10min_t",
        "fuel_consumption_10min", "fuel_consumption_t_10min", "fuel_t_10min",
        "main_engine_fuel_10min_t", "ME_FOC_10min_t",
    ],
    "speed_kn": [
        "speed_kn", "sog_kn", "SOG", "sog", "speed_over_ground", "vessel_speed_kn",
        "Speed_kn",
    ],
    "heading_deg": ["heading_deg", "heading", "Heading", "course_deg", "cog_deg"],
    "heading_sin": ["heading_sin", "sin_heading", "heading_sine"],
    "heading_cos": ["heading_cos", "cos_heading", "heading_cosine"],
    "draught_m": [
        "mean_draught_m", "mean_draft_m", "draught_m", "draft_m", "mean_draught",
        "mean_draft", "draught", "draft",
    ],
    "trim_m": ["trim_m", "trim", "Trim"],
    "rudder_deg": [
        "rudder_angle_deg", "rudder_deg", "rudder_angle", "mean_rudder_angle_deg",
        "rudder",
    ],
    "rel_wind_speed_kn": [
        "relative_wind_speed_kn", "rel_wind_speed_kn", "relative_wind_speed",
        "rel_wind_speed", "wind_speed_rel_kn", "apparent_wind_speed_kn",
    ],
    "rel_wind_dir_deg": [
        "relative_wind_direction_deg", "rel_wind_direction_deg", "relative_wind_direction",
        "rel_wind_dir_deg", "wind_direction_relative_deg", "apparent_wind_direction_deg",
    ],
    "rel_wind_sin": [
        "relative_wind_sin", "rel_wind_sin", "relative_wind_direction_sin",
        "sin_relative_wind", "wind_rel_sin",
    ],
    "rel_wind_cos": [
        "relative_wind_cos", "rel_wind_cos", "relative_wind_direction_cos",
        "cos_relative_wind", "wind_rel_cos",
    ],
    "wave_height_m": [
        "significant_wave_height_m", "significant_wave_height", "swh_m", "swh",
        "wave_height_m", "Hs",
    ],
    "rel_wave_dir_deg": [
        "relative_wave_direction_deg", "rel_wave_direction_deg", "relative_wave_direction",
        "rel_wave_dir_deg", "wave_direction_relative_deg",
    ],
    "rel_wave_sin": [
        "relative_wave_sin", "rel_wave_sin", "relative_wave_direction_sin",
        "sin_relative_wave", "wave_rel_sin",
    ],
    "rel_wave_cos": [
        "relative_wave_cos", "rel_wave_cos", "relative_wave_direction_cos",
        "cos_relative_wave", "wave_rel_cos",
    ],
    "wave_period_s": [
        "mean_wave_period_s", "mean_wave_period", "wave_period_s", "wave_period",
        "mwp_s", "mwp",
    ],
    "sst_c": [
        "sea_surface_temperature_c", "surface_temperature_c", "sst_c",
        "sea_surface_temperature", "surface_temperature",
    ],
    "mslp_hpa": [
        "mean_sea_level_pressure_hpa", "sea_level_pressure_hpa", "surface_pressure_hpa",
        "mslp_hpa", "msl_hpa", "slp_hpa", "pressure_hpa",
        "mean_sea_level_pressure", "sea_level_pressure", "surface_pressure",
        "mean_sea_level_pressure_pa", "sea_level_pressure_pa", "surface_pressure_pa",
        "mslp", "msl", "slp", "msl_pressure", "pressure_msl",
    ],
    "trajectory_group": [
        "trajectory_group", "trajectory_segment_id", "trajectory_id", "segment_id", "continuous_trajectory_id",
        "voyage_segment_id", "track_segment_id",
    ],
    "distance_nm": [
        "distance_nm", "distance_nm_10min", "interval_distance_nm", "distance_10min_nm", "distance_travelled_nm",
        "distance_traveled_nm", "sailing_distance_nm",
    ],
    "dwt": ["DWT", "dwt", "deadweight", "deadweight_t", "deadweight_tonnes"],
    "co2_factor": [
        "co2_factor", "fuel_co2_factor", "cf", "CF", "co2_t_per_t_fuel",
    ],
    "true_wind_speed_kn": ["true_wind_speed_kn", "tws_kn", "true_wind_speed"],
    "true_wind_dir_deg": ["true_wind_direction_deg", "twd_deg", "true_wind_direction"],
}

REQUIRED_CANONICAL = [
    "vessel_id", "ship_type", "target", "speed_kn", "draught_m", "trim_m",
    "rudder_deg", "rel_wind_speed_kn", "wave_height_m", "wave_period_s",
    "sst_c", "mslp_hpa",
]

CANONICAL_FEATURES = [
    "speed_kn", "heading_sin", "heading_cos", "draught_m", "trim_m", "rudder_deg",
    "rel_wind_speed_kn", "rel_wind_sin", "rel_wind_cos", "wave_height_m",
    "rel_wave_sin", "rel_wave_cos", "wave_period_s", "sst_c", "mslp_hpa",
    "ship_type_bulk", "ship_type_container",
]

OPERATIONAL_FEATURES = [
    "speed_kn", "heading_sin", "heading_cos", "draught_m", "trim_m", "rudder_deg",
    "ship_type_bulk", "ship_type_container",
]

WEATHER_FEATURES = [
    "rel_wind_speed_kn", "rel_wind_sin", "rel_wind_cos", "wave_height_m",
    "rel_wave_sin", "rel_wave_cos", "wave_period_s", "sst_c", "mslp_hpa",
    "ship_type_bulk", "ship_type_container",
]

RIDGE_EXTRA_FEATURES = [
    "speed_sq", "speed_cu", "speed_x_wind", "speed_x_wave", "speed_x_draught",
    "speed_x_trim", "draught_x_trim",
]

PHYSICAL_EXPECTATION = {
    "speed_kn": "positive_non_linear",
    "wave_height_m": "generally_positive_context_dependent",
    "rudder_abs": "larger_absolute_angle_generally_positive",
    "draught_m": "context_dependent",
    "trim_m": "context_dependent_possible_optimum",
    "rel_wind_speed_kn": "direction_dependent",
    "rel_wind_direction": "cyclic_context_dependent",
    "rel_wave_direction": "cyclic_context_dependent",
}


# -----------------------------------------------------------------------------
# Utility classes/functions
# -----------------------------------------------------------------------------

@dataclass
class DataBundle:
    raw: pd.DataFrame
    feature_df: pd.DataFrame
    feature_cols: List[str]
    column_map: Dict[str, Optional[str]]


def setup_logging(out_dir: Path) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"
    logger = logging.getLogger("f31")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def normalize_name(x: str) -> str:
    return str(x).strip().lower().replace(" ", "_").replace("-", "_")


def find_alias(columns: Sequence[str], aliases: Sequence[str]) -> Optional[str]:
    exact = {str(c): str(c) for c in columns}
    lower = {str(c).lower(): str(c) for c in columns}
    normalized = {normalize_name(c): str(c) for c in columns}
    for a in aliases:
        if a in exact:
            return exact[a]
        if a.lower() in lower:
            return lower[a.lower()]
        n = normalize_name(a)
        if n in normalized:
            return normalized[n]
    return None


def resolve_columns(df: pd.DataFrame, overrides: Optional[Dict[str, str]] = None) -> Dict[str, Optional[str]]:
    overrides = overrides or {}
    mapping: Dict[str, Optional[str]] = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        if canonical in overrides and overrides[canonical]:
            col = overrides[canonical]
            if col not in df.columns:
                raise KeyError(f"Column override {canonical}={col!r} not found in dataset")
            mapping[canonical] = col
        else:
            mapping[canonical] = find_alias(df.columns, aliases)
    return mapping


def normalize_ship_type(series: pd.Series) -> pd.Series:
    def f(v):
        s = str(v).strip().lower()
        if any(k in s for k in ["bulk", "散货"]):
            return "bulk"
        if any(k in s for k in ["container", "集装箱"]):
            return "container"
        if any(k in s for k in ["tanker", "oil", "油轮", "化学品"]):
            return "tanker"
        return s.replace(" ", "_")
    return series.map(f)


def normalize_phase(series: pd.Series) -> pd.Series:
    def f(v):
        s = str(v).strip().lower().replace("/", "_").replace(" ", "_")
        if any(k in s for k in ["cruise", "cruising", "sailing", "巡航"]):
            return "cruise"
        if any(k in s for k in ["maneuver", "manoeuv", "机动"]):
            return "maneuver"
        if any(k in s for k in ["anchor", "berth", "anchorage", "锚", "靠泊"]):
            return "anchor_berth"
        return s
    return series.map(f)


def to_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def angle_from_sincos(sin_v: pd.Series, cos_v: pd.Series) -> pd.Series:
    ang = np.degrees(np.arctan2(sin_v.astype(float), cos_v.astype(float)))
    return pd.Series(np.mod(ang, 360.0), index=sin_v.index)


def derive_trajectory_groups(df: pd.DataFrame, gap_minutes: float = 30.0) -> pd.Series:
    """Create vessel-specific continuous segments using time gaps when possible."""
    if "trajectory_group" in df.columns and df["trajectory_group"].notna().any():
        return df["trajectory_group"].astype(str)
    if "timestamp" in df.columns and df["timestamp"].notna().any():
        out = pd.Series(index=df.index, dtype="object")
        for vessel, idx in df.groupby("vessel_id").groups.items():
            sub = df.loc[idx].sort_values("timestamp")
            diffs = sub["timestamp"].diff().dt.total_seconds().div(60.0)
            cuts = diffs.isna() | (diffs > gap_minutes) | (diffs < 0)
            seg = cuts.cumsum().astype(int)
            out.loc[sub.index] = [f"{vessel}__seg{z}" for z in seg]
        return out.astype(str)
    # Conservative fallback: vessel itself becomes group.
    return df["vessel_id"].astype(str)


def _load_column_overrides_for_csv(args) -> Dict[str, str]:
    if not getattr(args, "column_overrides", None):
        return {}
    with open(args.column_overrides, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {str(k): str(v) for k, v in data.items() if v}


def _select_memory_safe_usecols(path: Path, args, logger: logging.Logger) -> Tuple[List[str], Dict[str, str]]:
    """Resolve only the raw columns needed by the modelling pipeline from the CSV header.

    Reading the full Fixed31 CSV with ``low_memory=False`` can require several times the
    on-disk size in RAM because pandas first tokenises and type-infers every column.  The
    final pipeline uses only a subset of the audit columns, so we resolve aliases from the
    header and load only those fields.
    """
    header = pd.read_csv(path, nrows=0)
    raw_cols = list(header.columns)
    overrides = _load_column_overrides_for_csv(args)

    # Fields actually consumed downstream. Direction-in-degrees columns are retained only
    # as fallbacks when sine/cosine columns are absent. Timestamp is required by the
    # known-vessel temporal and adaptation analyses.
    needed_keys = [
        "vessel_id", "ship_type", "timestamp", "phase", "target", "speed_kn",
        "heading_deg", "heading_sin", "heading_cos", "draught_m", "trim_m",
        "rudder_deg", "rel_wind_speed_kn", "rel_wind_dir_deg", "rel_wind_sin",
        "rel_wind_cos", "wave_height_m", "rel_wave_dir_deg", "rel_wave_sin",
        "rel_wave_cos", "wave_period_s", "sst_c", "mslp_hpa",
        "trajectory_group", "distance_nm", "dwt", "co2_factor",
        "true_wind_speed_kn", "true_wind_dir_deg",
    ]

    selected: List[str] = []
    canonical_by_raw: Dict[str, str] = {}
    for key in needed_keys:
        col = None
        if key in overrides:
            candidate = overrides[key]
            if candidate not in raw_cols:
                raise KeyError(f"Column override {key}={candidate!r} not found in {path}")
            col = candidate
        elif key in COLUMN_ALIASES:
            col = find_alias(raw_cols, COLUMN_ALIASES[key])
        if col is not None and col not in selected:
            selected.append(col)
            canonical_by_raw[col] = key

    # Preserve any source phase fallback field if a future Fixed31 export contains it.
    if "__phase_from_file" in raw_cols and "__phase_from_file" not in selected:
        selected.append("__phase_from_file")

    # Fail early if the header itself cannot support the required model fields.
    missing_header = []
    for key in REQUIRED_CANONICAL:
        if key in overrides:
            ok = overrides[key] in raw_cols
        else:
            ok = find_alias(raw_cols, COLUMN_ALIASES.get(key, [])) is not None
        if not ok:
            missing_header.append(key)
    if missing_header:
        raise KeyError(
            f"Required columns are absent from CSV header {path.name}: " + ", ".join(missing_header)
        )

    logger.info(
        "Memory-safe CSV plan for %s: loading %d/%d columns",
        path.name, len(selected), len(raw_cols),
    )
    logger.info("Selected raw columns: %s", selected)
    return selected, canonical_by_raw


def _memory_safe_dtype_map(usecols: Sequence[str], canonical_by_raw: Dict[str, str]) -> Dict[str, object]:
    """Use compact dtypes while retaining float64 for target/CII aggregation fields."""
    string_keys = {"vessel_id", "ship_type", "timestamp", "phase", "trajectory_group"}
    high_precision_keys = {"target", "distance_nm", "dwt", "co2_factor"}
    dtypes: Dict[str, object] = {}
    for raw in usecols:
        key = canonical_by_raw.get(raw)
        if key in string_keys:
            dtypes[raw] = "string"
        elif key in high_precision_keys:
            dtypes[raw] = "float64"
        elif key is not None:
            dtypes[raw] = "float32"
    return dtypes


def read_csv_memory_safe(path: Path, args, logger: logging.Logger) -> pd.DataFrame:
    """Read a Fixed31 CSV in bounded-memory chunks and only load modelling columns."""
    usecols, canonical_by_raw = _select_memory_safe_usecols(path, args, logger)
    dtypes = _memory_safe_dtype_map(usecols, canonical_by_raw)
    chunksize = max(5000, int(getattr(args, "csv_chunksize", 50000)))

    try:
        file_mb = path.stat().st_size / (1024 ** 2)
    except Exception:
        file_mb = float("nan")
    logger.info(
        "Reading %s (%.1f MiB) in chunks of %d rows with low_memory=True",
        path, file_mb, chunksize,
    )

    chunks: List[pd.DataFrame] = []
    rows = 0

    def _consume(reader, engine_name: str):
        nonlocal chunks, rows
        chunks = []
        rows = 0
        for i, chunk in enumerate(reader, start=1):
            chunks.append(chunk)
            rows += len(chunk)
            if i == 1 or i % 5 == 0:
                logger.info("  [%s] CSV chunks read: %d | rows=%d", engine_name, i, rows)

    try:
        reader = pd.read_csv(
            path,
            usecols=usecols,
            dtype=dtypes,
            chunksize=chunksize,
            low_memory=True,
            engine="c",
        )
        _consume(reader, "C")
    except (MemoryError, pd.errors.ParserError) as exc:
        logger.warning(
            "C-engine chunked CSV read failed for %s: %s", path, exc
        )
        logger.warning(
            "Retrying with the Python CSV engine at a smaller bounded chunk size. "
            "This is slower but avoids the C tokenizer's large temporary buffers."
        )
        fallback_chunksize = min(chunksize, 10000)
        try:
            reader = pd.read_csv(
                path,
                usecols=usecols,
                dtype=dtypes,
                chunksize=fallback_chunksize,
                engine="python",
            )
            _consume(reader, "python")
        except Exception as exc2:
            logger.error("Python-engine fallback also failed for %s: %s", path, exc2)
            logger.error(
                "Verify 64-bit Python and close other model-training processes. "
                "Diagnostic: python -c \"import struct; print(struct.calcsize('P')*8)\""
            )
            raise

    if not chunks:
        return pd.DataFrame(columns=usecols)

    # Concatenate only compact selected columns; this is far smaller than concatenating
    # the entire audit export.  ``copy=False`` reduces an avoidable second copy.
    out = pd.concat(chunks, ignore_index=True, sort=False, copy=False)
    del chunks
    logger.info("Finished memory-safe read: %d rows x %d columns", len(out), out.shape[1])
    return out


def load_phase_files(args, logger: logging.Logger) -> pd.DataFrame:
    """Load all-phase combined file if available, else concatenate the three phase files.

    Uses a bounded-memory reader because the complete Fixed31 audit export can exceed
    available RAM when pandas tokenises all columns with low_memory=False.
    """
    candidates = []
    if args.input_csv:
        candidates.append(Path(args.input_csv))
    if args.data_dir:
        d = Path(args.data_dir)
        candidates.append(d / "final_fixed31_all_phases_model_ready.csv")
    for p in candidates:
        if p.exists():
            logger.info("Loading combined dataset: %s", p)
            return read_csv_memory_safe(p, args, logger)

    if not args.data_dir:
        raise FileNotFoundError("Provide --input-csv or --data-dir containing final Fixed31 CSV files")
    d = Path(args.data_dir)
    phase_files = {
        "cruise": d / "final_fixed31_cruise.csv",
        "maneuver": d / "final_fixed31_maneuver.csv",
        "anchor_berth": d / "final_fixed31_anchor_berth.csv",
    }
    missing = [str(p) for p in phase_files.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Combined file not found and one or more phase files are missing:\n" + "\n".join(missing)
        )
    frames = []
    for phase, p in phase_files.items():
        x = read_csv_memory_safe(p, args, logger)
        x["__phase_from_file"] = phase
        frames.append(x)
        logger.info("Loaded %-13s %10d rows from %s", phase, len(x), p)
    return pd.concat(frames, ignore_index=True, sort=False, copy=False)


def canonicalize_dataframe(df: pd.DataFrame, args, logger: logging.Logger) -> DataBundle:
    overrides = {}
    if args.column_overrides:
        with open(args.column_overrides, "r", encoding="utf-8") as f:
            overrides = json.load(f)
    mapping = resolve_columns(df, overrides)

    # Pressure fields in input datasets have appeared under several short ERA5-style
    # names (e.g. msl / mslp / slp). If the normal alias table misses the field,
    # attempt a conservative pressure-only fallback. We auto-select only when there
    # is exactly one plausible candidate; otherwise the error reports the candidates
    # and asks for an explicit override.
    if mapping.get("mslp_hpa") is None:
        pressure_tokens = (
            "mean_sea_level_pressure", "sea_level_pressure", "surface_pressure",
            "mslp", "pressure_msl", "msl_pressure", "slp"
        )
        candidates = []
        for c in df.columns:
            n = normalize_name(c)
            if n == "msl" or any(tok in n for tok in pressure_tokens):
                candidates.append(str(c))
        candidates = list(dict.fromkeys(candidates))
        if len(candidates) == 1:
            mapping["mslp_hpa"] = candidates[0]
            logger.warning(
                "Auto-resolved mslp_hpa from pressure-like column %r. "
                "Verify its physical meaning and units in the data dictionary.",
                candidates[0],
            )
        elif len(candidates) > 1:
            logger.warning("Multiple pressure-like columns found: %s", candidates)

    missing_req = [k for k in REQUIRED_CANONICAL if mapping.get(k) is None]
    if missing_req:
        pressure_hint = ""
        if "mslp_hpa" in missing_req:
            pressure_like = [
                str(c) for c in df.columns
                if any(t in normalize_name(c) for t in ("press", "msl", "slp"))
            ]
            if pressure_like:
                pressure_hint = " Pressure-like columns detected: " + ", ".join(pressure_like[:20]) + "."
        raise KeyError(
            "Required columns could not be resolved: " + ", ".join(missing_req) +
            ". Use --column-overrides JSON to map canonical names to actual columns." +
            pressure_hint
        )

    out = pd.DataFrame(index=df.index)
    for key, col in mapping.items():
        if col is not None:
            out[key] = df[col]

    if "__phase_from_file" in df.columns and (mapping.get("phase") is None):
        out["phase"] = df["__phase_from_file"]
        mapping["phase"] = "__phase_from_file"

    out["vessel_id"] = out["vessel_id"].astype(str).str.strip()
    out["ship_type"] = normalize_ship_type(out["ship_type"])
    if "phase" in out.columns:
        out["phase"] = normalize_phase(out["phase"])
    else:
        out["phase"] = "unknown"

    if "timestamp" in out.columns:
        out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce", utc=True)
    else:
        out["timestamp"] = pd.NaT

    numeric_keys = [
        "target", "speed_kn", "heading_deg", "heading_sin", "heading_cos", "draught_m",
        "trim_m", "rudder_deg", "rel_wind_speed_kn", "rel_wind_dir_deg", "rel_wind_sin",
        "rel_wind_cos", "wave_height_m", "rel_wave_dir_deg", "rel_wave_sin", "rel_wave_cos",
        "wave_period_s", "sst_c", "mslp_hpa", "distance_nm", "dwt", "co2_factor",
        "true_wind_speed_kn", "true_wind_dir_deg",
    ]
    for k in numeric_keys:
        if k in out.columns:
            out[k] = to_numeric(out[k])

    # Canonical mslp_hpa must be expressed in hPa. ERA5 native mean sea-level
    # pressure is commonly stored in Pa. Detect this from the value scale rather
    # than the column spelling, so names such as `msl` are handled safely.
    if "mslp_hpa" in out.columns:
        p_med = float(out["mslp_hpa"].dropna().median()) if out["mslp_hpa"].notna().any() else np.nan
        if np.isfinite(p_med) and 20000.0 <= p_med <= 120000.0:
            logger.warning(
                "Pressure median %.2f suggests Pa units; converting canonical mslp_hpa by /100.",
                p_med,
            )
            out["mslp_hpa"] = out["mslp_hpa"] / 100.0
        elif np.isfinite(p_med) and not (800.0 <= p_med <= 1200.0):
            logger.warning(
                "Canonical pressure median %.2f is outside the usual 800-1200 hPa range. "
                "Check the source column and units.",
                p_med,
            )

    # Directional derivations.
    if "heading_sin" not in out.columns or "heading_cos" not in out.columns:
        if "heading_deg" not in out.columns:
            raise KeyError("Need heading_sin/cos columns or a heading_deg column")
        rad = np.deg2rad(out["heading_deg"] % 360.0)
        out["heading_sin"] = np.sin(rad)
        out["heading_cos"] = np.cos(rad)
    if "rel_wind_sin" not in out.columns or "rel_wind_cos" not in out.columns:
        if "rel_wind_dir_deg" not in out.columns:
            raise KeyError("Need relative wind sin/cos columns or relative wind direction in degrees")
        rad = np.deg2rad(out["rel_wind_dir_deg"] % 360.0)
        out["rel_wind_sin"] = np.sin(rad)
        out["rel_wind_cos"] = np.cos(rad)
    if "rel_wave_sin" not in out.columns or "rel_wave_cos" not in out.columns:
        if "rel_wave_dir_deg" not in out.columns:
            raise KeyError("Need relative wave sin/cos columns or relative wave direction in degrees")
        rad = np.deg2rad(out["rel_wave_dir_deg"] % 360.0)
        out["rel_wave_sin"] = np.sin(rad)
        out["rel_wave_cos"] = np.cos(rad)

    # Recover angular values for sector analysis if only sine/cosine were present.
    if "rel_wind_dir_deg" not in out.columns:
        out["rel_wind_dir_deg"] = angle_from_sincos(out["rel_wind_sin"], out["rel_wind_cos"])
    if "rel_wave_dir_deg" not in out.columns:
        out["rel_wave_dir_deg"] = angle_from_sincos(out["rel_wave_sin"], out["rel_wave_cos"])

    out["ship_type_bulk"] = (out["ship_type"] == "bulk").astype(int)
    out["ship_type_container"] = (out["ship_type"] == "container").astype(int)

    # Create / normalize trajectory groups.
    if mapping.get("trajectory_group") is not None:
        out["trajectory_group"] = df[mapping["trajectory_group"]].astype(str)
    out["trajectory_group"] = derive_trajectory_groups(out, args.trajectory_gap_minutes)

    feature_cols = [c for c in CANONICAL_FEATURES if c in out.columns]
    missing_feat = [c for c in CANONICAL_FEATURES if c not in feature_cols]
    if missing_feat:
        raise KeyError("Model features missing after canonicalization: " + ", ".join(missing_feat))

    # Complete-case modelling cohort only. Preserve full raw audit separately if desired.
    complete_cols = ["target", "vessel_id", "ship_type", "phase"] + feature_cols
    model_mask = out[complete_cols].notna().all(axis=1)
    n_bad = int((~model_mask).sum())
    if n_bad:
        logger.warning("Dropping %d rows with missing canonical model fields after loading Fixed31 data", n_bad)
    out = out.loc[model_mask].copy().reset_index(drop=True)
    feat = out[feature_cols].copy()

    logger.info("Canonical column mapping:")
    for k in sorted(mapping):
        logger.info("  %-22s <- %s", k, mapping[k])
    logger.info("Canonical modelling rows: %d", len(out))
    logger.info("Vessels: %d | types=%s", out["vessel_id"].nunique(), out["ship_type"].value_counts().to_dict())
    logger.info("Phases: %s", out["phase"].value_counts().to_dict())

    return DataBundle(raw=out, feature_df=feat, feature_cols=feature_cols, column_map=mapping)


def strict_fixed31_audit(df: pd.DataFrame, args, logger: logging.Logger) -> Dict:
    counts = df["phase"].value_counts().to_dict()
    audit = {
        "phase_counts": {k: int(v) for k, v in counts.items()},
        "total": int(len(df)),
        "vessels": int(df["vessel_id"].nunique()),
        "ship_type_vessels": df.groupby("ship_type")["vessel_id"].nunique().astype(int).to_dict(),
    }
    expected = {
        "cruise": args.expected_cruise_n,
        "maneuver": args.expected_maneuver_n,
        "anchor_berth": args.expected_anchor_n,
    }
    expected_total = sum(expected.values())
    audit["expected_phase_counts"] = expected
    audit["expected_total"] = expected_total
    audit["phase_count_match"] = all(counts.get(k, 0) == v for k, v in expected.items())
    audit["vessel_count_match"] = df["vessel_id"].nunique() == args.expected_vessels
    if args.strict_audit:
        if not audit["phase_count_match"]:
            raise AssertionError(f"Fixed31 phase counts do not match expected values: observed={counts}, expected={expected}")
        if not audit["vessel_count_match"]:
            raise AssertionError(
                f"Expected {args.expected_vessels} vessels but found {df['vessel_id'].nunique()}"
            )
    logger.info("Fixed31 audit: total=%d, phase_counts=%s, vessels=%d", len(df), counts, df["vessel_id"].nunique())
    return audit


def select_primary_scope(bundle: DataBundle, scope: str, logger: logging.Logger) -> DataBundle:
    if scope == "all_phases":
        return bundle
    if scope == "cruise":
        m = bundle.raw["phase"] == "cruise"
        logger.info("Primary scope restricted to cruise: %d rows", int(m.sum()))
        raw = bundle.raw.loc[m].reset_index(drop=True)
        feat = bundle.feature_df.loc[m].reset_index(drop=True)
        return DataBundle(raw=raw, feature_df=feat, feature_cols=bundle.feature_cols, column_map=bundle.column_map)
    raise ValueError(f"Unknown primary scope: {scope}")


def stratified_record_split(raw: pd.DataFrame, test_size: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    y = raw["target"]
    try:
        q = pd.qcut(y.rank(method="first"), q=10, labels=False, duplicates="drop")
    except Exception:
        q = pd.cut(y, bins=10, labels=False, duplicates="drop")
    strata = raw["ship_type"].astype(str) + "__" + q.astype(str)
    idx = np.arange(len(raw))
    train_idx, test_idx = train_test_split(
        idx, test_size=test_size, random_state=seed, stratify=strata
    )
    return np.sort(train_idx), np.sort(test_idx)


def ridge_feature_matrix(X: pd.DataFrame) -> pd.DataFrame:
    Z = X.copy()
    speed = Z["speed_kn"]
    Z["speed_sq"] = speed ** 2
    Z["speed_cu"] = speed ** 3
    Z["speed_x_wind"] = speed * Z["rel_wind_speed_kn"]
    Z["speed_x_wave"] = speed * Z["wave_height_m"]
    Z["speed_x_draught"] = speed * Z["draught_m"]
    Z["speed_x_trim"] = speed * Z["trim_m"]
    Z["draught_x_trim"] = Z["draught_m"] * Z["trim_m"]
    return Z


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(math.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    denom = np.maximum(np.abs(y_true), eps)
    mape = float(np.mean(np.abs(err) / denom) * 100.0)
    smape = float(np.mean(2.0 * np.abs(err) / np.maximum(np.abs(y_true) + np.abs(y_pred), eps)) * 100.0)
    try:
        r2 = float(r2_score(y_true, y_pred))
    except Exception:
        r2 = float("nan")
    total_true = float(np.sum(y_true))
    bias_pct = float((np.sum(y_pred) - total_true) / total_true * 100.0) if abs(total_true) > eps else float("nan")
    target_sd = float(np.std(y_true, ddof=1)) if len(y_true) > 1 else float("nan")
    nrmse_sd = float(rmse / target_sd) if target_sd and np.isfinite(target_sd) and target_sd > 0 else float("nan")
    return {
        "n": int(len(y_true)), "MSE": mse, "RMSE": rmse, "MAE": mae,
        "MAPE_pct": mape, "sMAPE_pct": smape, "R2": r2,
        "target_sd": target_sd, "NRMSE_sd": nrmse_sd,
        "cumulative_bias_pct": bias_pct,
    }


def per_group_metrics(raw: pd.DataFrame, y_pred: np.ndarray, group_cols: Sequence[str]) -> pd.DataFrame:
    tmp = raw[list(group_cols) + ["target"]].copy()
    tmp["prediction"] = np.asarray(y_pred)
    rows = []
    for key, g in tmp.groupby(list(group_cols), dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(group_cols, key))
        row.update(metric_dict(g["target"].values, g["prediction"].values))
        rows.append(row)
    return pd.DataFrame(rows)


def macro_vessel_summary(vessel_metrics: pd.DataFrame) -> Dict[str, float]:
    out = {}
    for m in ["RMSE", "MAE", "R2", "NRMSE_sd", "cumulative_bias_pct"]:
        if m in vessel_metrics:
            vals = pd.to_numeric(vessel_metrics[m], errors="coerce")
            out[f"mean_vessel_{m}"] = float(vals.mean())
            out[f"median_vessel_{m}"] = float(vals.median())
    if "R2" in vessel_metrics:
        out["positive_R2_vessel_proportion"] = float((vessel_metrics["R2"] > 0).mean())
    return out


def sample_indices(n: int, max_n: int, seed: int) -> np.ndarray:
    if n <= max_n:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=max_n, replace=False))


def make_group_folds(groups: pd.Series, n_splits: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    groups = pd.Series(groups).astype(str).reset_index(drop=True)
    unique = groups.nunique()
    n_splits = min(n_splits, unique)
    if n_splits < 2:
        raise ValueError("Need at least two unique groups for grouped CV")
    gkf = GroupKFold(n_splits=n_splits)
    dummy = np.zeros(len(groups))
    return list(gkf.split(dummy, dummy, groups=groups.values))


def cv_rmse(model, X: pd.DataFrame, y: np.ndarray, folds: List[Tuple[np.ndarray, np.ndarray]], ridge_mode=False) -> float:
    vals = []
    Xuse = ridge_feature_matrix(X) if ridge_mode else X
    for tr, va in folds:
        m = clone(model)
        m.fit(Xuse.iloc[tr], y[tr])
        p = m.predict(Xuse.iloc[va])
        vals.append(math.sqrt(mean_squared_error(y[va], p)))
    return float(np.mean(vals))


def model_factory(name: str, params: Dict, seed: int, n_jobs: int):
    name = name.lower()
    if name == "lr":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ])
    if name == "ridge_interaction":
        alpha = float(params.get("alpha", 1.0))
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=alpha, random_state=seed)),
        ])
    if name == "dt":
        return DecisionTreeRegressor(
            random_state=seed,
            max_depth=int(params.get("max_depth", 14)),
            min_samples_leaf=int(params.get("min_samples_leaf", 5)),
            min_samples_split=int(params.get("min_samples_split", 10)),
            max_features=float(params.get("max_features", 1.0)),
        )
    if name == "rf":
        return RandomForestRegressor(
            random_state=seed,
            n_jobs=n_jobs,
            n_estimators=int(params.get("n_estimators", 500)),
            max_depth=int(params.get("max_depth", 24)),
            min_samples_leaf=int(params.get("min_samples_leaf", 2)),
            min_samples_split=int(params.get("min_samples_split", 4)),
            max_features=float(params.get("max_features", 0.8)),
            max_samples=float(params.get("max_samples", 0.9)),
        )
    if name == "xgb":
        return XGBRegressor(
            random_state=seed,
            n_jobs=n_jobs,
            tree_method="hist",
            objective="reg:squarederror",
            eval_metric="rmse",
            n_estimators=int(params.get("n_estimators", 800)),
            max_depth=int(params.get("max_depth", 7)),
            learning_rate=float(params.get("learning_rate", 0.05)),
            subsample=float(params.get("subsample", 0.85)),
            colsample_bytree=float(params.get("colsample_bytree", 0.85)),
            min_child_weight=float(params.get("min_child_weight", 2.0)),
            reg_alpha=float(params.get("reg_alpha", 0.0)),
            reg_lambda=float(params.get("reg_lambda", 1.0)),
        )
    if name == "lgbm":
        return LGBMRegressor(
            random_state=seed,
            n_jobs=n_jobs,
            verbosity=-1,
            n_estimators=int(params.get("n_estimators", 800)),
            num_leaves=int(params.get("num_leaves", 63)),
            max_depth=int(params.get("max_depth", 10)),
            learning_rate=float(params.get("learning_rate", 0.05)),
            feature_fraction=float(params.get("feature_fraction", 0.85)),
            bagging_fraction=float(params.get("bagging_fraction", 0.85)),
            bagging_freq=1,
            min_child_samples=int(params.get("min_child_samples", 20)),
            reg_alpha=float(params.get("reg_alpha", 0.0)),
            reg_lambda=float(params.get("reg_lambda", 0.0)),
        )
    if name == "ann":
        hidden = params.get("hidden_layer_sizes", [128, 64])
        if isinstance(hidden, str):
            hidden = [int(v) for v in hidden.split("x")]
        hidden = tuple(hidden)
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", MLPRegressor(
                random_state=seed,
                hidden_layer_sizes=hidden,
                activation="relu",
                solver="adam",
                alpha=float(params.get("alpha", 1e-4)),
                learning_rate_init=float(params.get("learning_rate_init", 1e-3)),
                batch_size=int(params.get("batch_size", 512)),
                max_iter=int(params.get("max_iter", 400)),
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,
            )),
        ])
    raise ValueError(f"Unknown model {name}")


def suggest_params(trial: optuna.Trial, name: str) -> Dict:
    if name == "ridge_interaction":
        return {"alpha": trial.suggest_float("alpha", 1e-4, 1e3, log=True)}
    if name == "dt":
        return {
            "max_depth": trial.suggest_int("max_depth", 4, 30),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 40),
            "max_features": trial.suggest_float("max_features", 0.5, 1.0),
        }
    if name == "rf":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 250, 900, step=50),
            "max_depth": trial.suggest_int("max_depth", 8, 32),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 12),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 30),
            "max_features": trial.suggest_float("max_features", 0.5, 1.0),
            "max_samples": trial.suggest_float("max_samples", 0.65, 1.0),
        }
    if name == "xgb":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 300, 1600, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 0.5, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
        }
    if name == "lgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 300, 1600, step=100),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "max_depth": trial.suggest_int("max_depth", 5, 14),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 80),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 30.0, log=True),
        }
    if name == "ann":
        return {
            "hidden_layer_sizes": trial.suggest_categorical(
                "hidden_layer_sizes", ["64", "128", "128x64", "256x128", "256x128x64"]
            ),
            "alpha": trial.suggest_float("alpha", 1e-6, 1e-2, log=True),
            "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 5e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [256, 512, 1024, 2048]),
            "max_iter": 400,
        }
    raise ValueError(name)


def three_stage_tune(
    name: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    groups_train: pd.Series,
    stage_a_idx: np.ndarray,
    stage_b_idx: np.ndarray,
    args,
    logger: logging.Logger,
) -> Tuple[Dict, Dict]:
    if name == "lr":
        return {}, {"stage": "not_tuned"}

    ridge_mode = name == "ridge_interaction"
    Xa = X_train.iloc[stage_a_idx].reset_index(drop=True)
    ya = y_train[stage_a_idx]
    ga = groups_train.iloc[stage_a_idx].reset_index(drop=True)
    folds_a = make_group_folds(ga, args.cv_folds)

    logger.info("Tuning %s Stage A: n=%d, trials=%d", name, len(Xa), args.stage_a_trials)
    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    def objective(trial):
        params = suggest_params(trial, name)
        model = model_factory(name, params, args.seed, args.n_jobs)
        return cv_rmse(model, Xa, ya, folds_a, ridge_mode=ridge_mode)

    study.optimize(objective, n_trials=args.stage_a_trials, show_progress_bar=False)
    trials = [t for t in study.trials if t.value is not None and np.isfinite(t.value)]
    trials = sorted(trials, key=lambda t: t.value)
    top = trials[: min(3, len(trials))]
    if not top:
        raise RuntimeError(f"No valid Optuna trials for {name}")

    Xb = X_train.iloc[stage_b_idx].reset_index(drop=True)
    yb = y_train[stage_b_idx]
    gb = groups_train.iloc[stage_b_idx].reset_index(drop=True)
    folds_b = make_group_folds(gb, args.cv_folds)
    stage_b_rows = []
    logger.info("Tuning %s Stage B: n=%d, candidates=%d", name, len(Xb), len(top))
    for rank, t in enumerate(top, start=1):
        params = dict(t.params)
        if name == "ann":
            params["max_iter"] = 400
        model = model_factory(name, params, args.seed, args.n_jobs)
        score = cv_rmse(model, Xb, yb, folds_b, ridge_mode=ridge_mode)
        stage_b_rows.append({
            "stage_a_rank": rank, "stage_a_rmse": float(t.value),
            "stage_b_rmse": float(score), "params": params,
        })
    stage_b_rows.sort(key=lambda z: z["stage_b_rmse"])
    best = stage_b_rows[0]
    logger.info("Selected %s params with Stage B CV RMSE %.6f", name, best["stage_b_rmse"])
    meta = {
        "stage_a_best_value": float(study.best_value),
        "stage_a_trials": int(len(study.trials)),
        "stage_b_candidates": stage_b_rows,
        "selected_stage_b_rmse": best["stage_b_rmse"],
    }
    return best["params"], meta


def fit_predict_model(name: str, params: Dict, X_train: pd.DataFrame, y_train: np.ndarray,
                      X_test: pd.DataFrame, seed: int, n_jobs: int):
    model = model_factory(name, params, seed, n_jobs)
    if name == "ridge_interaction":
        Xtr = ridge_feature_matrix(X_train)
        Xte = ridge_feature_matrix(X_test)
    else:
        Xtr, Xte = X_train, X_test
    model.fit(Xtr, y_train)
    pred = model.predict(Xte)
    return model, pred


def cluster_bootstrap_rmse_difference(
    vessel_ids: pd.Series,
    y: np.ndarray,
    pred_reduced: np.ndarray,
    pred_full: np.ndarray,
    B: int,
    seed: int,
) -> Dict[str, float]:
    tmp = pd.DataFrame({
        "vessel_id": vessel_ids.astype(str).values,
        "sq_red": (pred_reduced - y) ** 2,
        "sq_full": (pred_full - y) ** 2,
        "ae_red": np.abs(pred_reduced - y),
        "ae_full": np.abs(pred_full - y),
    })
    agg = tmp.groupby("vessel_id").agg(
        n=("sq_red", "size"),
        sse_red=("sq_red", "sum"), sse_full=("sq_full", "sum"),
        sae_red=("ae_red", "sum"), sae_full=("ae_full", "sum"),
    )
    vids = agg.index.to_numpy()
    rng = np.random.default_rng(seed)
    diffs_rmse = []
    diffs_mae = []
    for _ in range(B):
        draw = rng.choice(vids, size=len(vids), replace=True)
        a = agg.loc[draw]
        n = a["n"].sum()
        rmse_red = math.sqrt(a["sse_red"].sum() / n)
        rmse_full = math.sqrt(a["sse_full"].sum() / n)
        mae_red = a["sae_red"].sum() / n
        mae_full = a["sae_full"].sum() / n
        diffs_rmse.append(rmse_red - rmse_full)
        diffs_mae.append(mae_red - mae_full)
    return {
        "delta_RMSE_reduced_minus_full": float(np.mean(diffs_rmse)),
        "delta_RMSE_CI_low": float(np.quantile(diffs_rmse, 0.025)),
        "delta_RMSE_CI_high": float(np.quantile(diffs_rmse, 0.975)),
        "delta_MAE_reduced_minus_full": float(np.mean(diffs_mae)),
        "delta_MAE_CI_low": float(np.quantile(diffs_mae, 0.025)),
        "delta_MAE_CI_high": float(np.quantile(diffs_mae, 0.975)),
    }


def save_scatter_dependency(x, shap_values, path: Path, xlabel: str, ylabel: str = "SHAP value"):
    fig = plt.figure(figsize=(7.5, 5.5))
    ax = fig.add_subplot(111)
    ax.scatter(x, shap_values, s=8, alpha=0.35)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def fit_poly_diagnostics(x: np.ndarray, y: np.ndarray, max_degree=3) -> pd.DataFrame:
    rows = []
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    for deg in range(1, max_degree + 1):
        coef = np.polyfit(x, y, deg)
        pred = np.polyval(coef, x)
        resid = y - pred
        rss = float(np.sum(resid ** 2))
        n = len(y)
        k = deg + 1
        aic = float(n * np.log(max(rss / max(n, 1), 1e-15)) + 2 * k)
        tss = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - rss / tss if tss > 0 else float("nan")
        rows.append({"degree": deg, "AIC": aic, "R2": r2, "coefficients": json.dumps(coef.tolist())})
    return pd.DataFrame(rows)


def semantic_wind_sector(angle_deg: pd.Series, zero_is: str) -> pd.Series:
    a = np.mod(angle_deg.astype(float), 360.0)
    raw = pd.Series(index=angle_deg.index, dtype="object")
    raw[((a <= 45) | (a >= 315))] = "0deg_sector"
    raw[((a > 45) & (a < 135)) | ((a > 225) & (a < 315))] = "cross_sector"
    raw[(a >= 135) & (a <= 225)] = "180deg_sector"
    if zero_is == "headwind":
        return raw.map({"0deg_sector": "headwind", "cross_sector": "crosswind", "180deg_sector": "following_wind"})
    if zero_is == "following":
        return raw.map({"0deg_sector": "following_wind", "cross_sector": "crosswind", "180deg_sector": "headwind"})
    return raw


def get_tree_shap(model, X: pd.DataFrame) -> np.ndarray:
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)
    if isinstance(sv, list):
        sv = sv[0]
    return np.asarray(sv)


def physical_status_from_spearman(feature: str, rho: float) -> str:
    if feature == "speed_kn":
        if rho >= 0.2:
            return "Consistent"
        if rho <= -0.2:
            return "Inconsistent"
        return "Weak/uncertain"
    if feature == "wave_height_m":
        if rho >= 0.1:
            return "Consistent/conditional"
        if rho <= -0.1:
            return "Potentially inconsistent; inspect interactions"
        return "Context-dependent"
    return "Not directionally testable / context-dependent"


def partial_dependence_2d_density_aware(
    model,
    X_background: pd.DataFrame,
    support_raw: pd.DataFrame,
    speed_col: str,
    wind_col: str,
    grid_n: int,
    support_min: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    if len(X_background) > 500:
        bg = X_background.iloc[np.sort(rng.choice(len(X_background), 500, replace=False))].copy()
    else:
        bg = X_background.copy()
    svals = support_raw[speed_col].astype(float)
    wvals = support_raw[wind_col].astype(float)
    sgrid = np.unique(np.quantile(svals, np.linspace(0.02, 0.98, grid_n)))
    wgrid = np.unique(np.quantile(wvals, np.linspace(0.02, 0.98, grid_n)))
    # Support half-widths from adjacent grid spacing.
    s_half = np.median(np.diff(sgrid)) / 2 if len(sgrid) > 1 else max(np.std(svals), 1e-6)
    w_half = np.median(np.diff(wgrid)) / 2 if len(wgrid) > 1 else max(np.std(wvals), 1e-6)
    rows = []
    for s in sgrid:
        for w in wgrid:
            support = int(((np.abs(svals - s) <= s_half) & (np.abs(wvals - w) <= w_half)).sum())
            xmod = bg.copy()
            xmod[speed_col] = s
            xmod[wind_col] = w
            pred = float(np.mean(model.predict(xmod)))
            rows.append({
                "speed_kn": float(s), "rel_wind_speed_kn": float(w),
                "PDP_prediction": pred, "support_n": support,
                "supported": bool(support >= support_min),
            })
    return pd.DataFrame(rows)


def plot_density_pdp(df: pd.DataFrame, path: Path):
    supported = df[df["supported"]].copy()
    if supported.empty:
        return
    piv = supported.pivot(index="rel_wind_speed_kn", columns="speed_kn", values="PDP_prediction")
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    im = ax.imshow(
        piv.values,
        origin="lower",
        aspect="auto",
        extent=[piv.columns.min(), piv.columns.max(), piv.index.min(), piv.index.max()],
    )
    ax.set_xlabel("Speed over ground (kn)")
    ax.set_ylabel("Relative wind speed (kn)")
    ax.set_title("Density-aware 2D partial dependence (unsupported cells masked)")
    fig.colorbar(im, ax=ax, label="Predicted fuel consumption")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def perturb_raw(raw: pd.DataFrame, kind: str, pct: float = 0.0, abs_add: float = 0.0) -> pd.DataFrame:
    z = raw.copy()
    if kind == "speed":
        z["speed_kn"] = z["speed_kn"] * (1.0 + pct)
    elif kind == "wave":
        z["wave_height_m"] = z["wave_height_m"] * (1.0 + pct)
    elif kind == "rudder_abs":
        r = z["rudder_deg"].astype(float)
        sign = np.sign(r)
        sign[sign == 0] = 1.0
        z["rudder_deg"] = sign * (np.abs(r) + abs_add)
    else:
        raise ValueError(kind)
    return z


def feature_matrix_from_raw(raw: pd.DataFrame, feature_cols: Sequence[str]) -> pd.DataFrame:
    # All model features are directly stored/derived in canonical raw dataframe.
    return raw[list(feature_cols)].copy()


def local_response_audit(
    model,
    train_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
    feature_cols: List[str],
    sample_n: int,
    seed: int,
) -> pd.DataFrame:
    idx = sample_indices(len(test_raw), sample_n, seed)
    base_raw = test_raw.iloc[idx].copy().reset_index(drop=True)
    base_X = feature_matrix_from_raw(base_raw, feature_cols)
    base_pred = model.predict(base_X)

    # Ship-type speed support from training data.
    speed_support = train_raw.groupby("ship_type")["speed_kn"].agg(["min", "max"])
    perturbations = [
        ("speed_plus_5pct", "speed", 0.05, 0.0),
        ("wave_height_plus_10pct", "wave", 0.10, 0.0),
        ("abs_rudder_plus_1deg", "rudder_abs", 0.0, 1.0),
    ]
    rows = []
    for label, kind, pct, add in perturbations:
        pr = perturb_raw(base_raw, kind, pct=pct, abs_add=add)
        valid = np.ones(len(pr), dtype=bool)
        if kind == "speed":
            for st, inds in pr.groupby("ship_type").groups.items():
                if st in speed_support.index:
                    lo, hi = speed_support.loc[st, ["min", "max"]]
                    valid[np.array(list(inds), dtype=int)] = (
                        (pr.loc[inds, "speed_kn"] >= lo) & (pr.loc[inds, "speed_kn"] <= hi)
                    ).values
        pred = np.full(len(pr), np.nan)
        if valid.any():
            pred[valid] = model.predict(feature_matrix_from_raw(pr.loc[valid], feature_cols))
        delta = pred - base_pred
        tmp = base_raw[["ship_type", "phase"]].copy()
        tmp["valid"] = valid
        tmp["delta"] = delta
        tmp["label"] = label
        for scope_cols in [[], ["ship_type"], ["phase"], ["ship_type", "phase"]]:
            if scope_cols:
                grouped = tmp.groupby(scope_cols, dropna=False)
            else:
                grouped = [((), tmp)]
            for key, g in grouped:
                gv = g[g["valid"] & g["delta"].notna()]
                if isinstance(key, tuple):
                    key_t = key
                else:
                    key_t = (key,)
                row = {"perturbation": label, "scope": "overall" if not scope_cols else "+".join(scope_cols)}
                for c, v in zip(scope_cols, key_t):
                    row[c] = v
                row["n_total"] = int(len(g))
                row["n_supported"] = int(len(gv))
                if len(gv):
                    tol = 1e-8
                    row["pred_increase_pct"] = float((gv["delta"] > tol).mean() * 100)
                    row["pred_unchanged_pct"] = float((np.abs(gv["delta"]) <= tol).mean() * 100)
                    row["pred_decrease_pct"] = float((gv["delta"] < -tol).mean() * 100)
                    row["median_delta"] = float(gv["delta"].median())
                rows.append(row)
    return pd.DataFrame(rows)


def compute_cii_by_vessel(raw: pd.DataFrame, pred: np.ndarray, default_cf: float) -> pd.DataFrame:
    x = raw.copy()
    x["prediction"] = np.asarray(pred)
    if "distance_nm" not in x.columns or x["distance_nm"].isna().all():
        x["distance_nm_used"] = x["speed_kn"] * (10.0 / 60.0)
    else:
        x["distance_nm_used"] = x["distance_nm"]
    if "co2_factor" in x.columns and x["co2_factor"].notna().any():
        x["cf_used"] = x["co2_factor"].fillna(default_cf)
    else:
        x["cf_used"] = default_cf
    if "dwt" not in x.columns or x["dwt"].isna().all():
        raise KeyError("DWT is required for CII-proxy calculations. Provide a DWT column or column override.")
    rows = []
    for vessel, g in x.groupby("vessel_id"):
        dwt = float(g["dwt"].dropna().median())
        dist = float(g["distance_nm_used"].sum())
        cf_obs_weighted = float(np.average(g["cf_used"], weights=np.maximum(g["target"], 1e-12)))
        # Exact row-wise CO2 to support mixed fuel factors if present.
        obs_co2_g = float(np.sum(g["target"] * g["cf_used"] * 1e6))
        pred_co2_g = float(np.sum(g["prediction"] * g["cf_used"] * 1e6))
        denom = dwt * dist
        obs_proxy = obs_co2_g / denom if denom > 0 else np.nan
        pred_proxy = pred_co2_g / denom if denom > 0 else np.nan
        rows.append({
            "vessel_id": vessel,
            "ship_type": g["ship_type"].iloc[0],
            "n": len(g), "DWT": dwt, "distance_nm": dist,
            "observed_fuel_t": float(g["target"].sum()),
            "predicted_fuel_t": float(g["prediction"].sum()),
            "observed_CII_proxy": obs_proxy,
            "predicted_CII_proxy": pred_proxy,
            "absolute_error": abs(pred_proxy - obs_proxy),
            "relative_error_pct": (pred_proxy - obs_proxy) / obs_proxy * 100 if obs_proxy else np.nan,
            "absolute_percentage_error_pct": abs(pred_proxy - obs_proxy) / abs(obs_proxy) * 100 if obs_proxy else np.nan,
            "representative_CF": cf_obs_weighted,
        })
    return pd.DataFrame(rows)


def cii_summary(vessel_df: pd.DataFrame, raw: pd.DataFrame, pred: np.ndarray) -> Dict[str, float]:
    rho = spearmanr(vessel_df["observed_CII_proxy"], vessel_df["predicted_CII_proxy"], nan_policy="omit").statistic
    total_obs = float(np.sum(raw["target"]))
    total_pred = float(np.sum(pred))
    return {
        "vessels": int(len(vessel_df)),
        "CII_proxy_MAE": float(vessel_df["absolute_error"].mean()),
        "CII_proxy_median_absolute_error": float(vessel_df["absolute_error"].median()),
        "mean_relative_error_pct": float(vessel_df["relative_error_pct"].mean()),
        "median_absolute_percentage_error_pct": float(vessel_df["absolute_percentage_error_pct"].median()),
        "vessel_rank_spearman": float(rho),
        "observed_fuel_t": total_obs,
        "predicted_fuel_t": total_pred,
        "cumulative_fuel_bias_pct": (total_pred - total_obs) / total_obs * 100.0,
    }


def speed_reduction_scenarios(
    model,
    train_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
    feature_cols: List[str],
    reductions: Sequence[float],
    default_cf: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    base = test_raw.copy().reset_index(drop=True)
    base_X = feature_matrix_from_raw(base, feature_cols)
    base_pred = model.predict(base_X)
    if "distance_nm" not in base.columns or base["distance_nm"].isna().all():
        base["distance_nm_used"] = base["speed_kn"] * (10.0 / 60.0)
    else:
        base["distance_nm_used"] = base["distance_nm"]
    support = train_raw.groupby("ship_type")["speed_kn"].agg(["min", "max"])
    if "dwt" not in base.columns or base["dwt"].isna().all():
        raise KeyError("DWT required for CII scenario analysis")
    if "co2_factor" in base.columns and base["co2_factor"].notna().any():
        cf = base["co2_factor"].fillna(default_cf)
    else:
        cf = pd.Series(default_cf, index=base.index)

    vessel_rows = []
    for reduction in reductions:
        sc = base.copy()
        sc["speed_kn"] = sc["speed_kn"] * (1.0 - reduction)
        valid = np.ones(len(sc), dtype=bool)
        for st, inds in sc.groupby("ship_type").groups.items():
            lo, hi = support.loc[st, ["min", "max"]]
            valid[np.array(list(inds), dtype=int)] = (
                (sc.loc[inds, "speed_kn"] >= lo) & (sc.loc[inds, "speed_kn"] <= hi)
            ).values
        sc = sc.loc[valid].copy()
        b = base.loc[valid].copy()
        bpred = base_pred[valid]
        scpred = model.predict(feature_matrix_from_raw(sc, feature_cols))
        # Fixed ten-minute formulation: interval distance changes with SOG.
        ratio = np.divide(sc["speed_kn"].values, b["speed_kn"].values, out=np.zeros(len(sc)), where=b["speed_kn"].values != 0)
        sc["distance_scenario_nm"] = b["distance_nm_used"].values * ratio
        sc["base_pred"] = bpred
        sc["scenario_pred"] = scpred
        sc["cf_used"] = cf.loc[b.index].values

        for vessel, g in sc.groupby("vessel_id"):
            dwt = float(g["dwt"].dropna().median())
            base_fuel = float(g["base_pred"].sum())
            sc_fuel = float(g["scenario_pred"].sum())
            base_dist = float(base.loc[g.index, "distance_nm_used"].sum()) if set(g.index).issubset(base.index) else np.nan
            # Index after loc remains original, so direct base.loc works.
            base_dist = float(base.loc[g.index, "distance_nm_used"].sum())
            sc_dist = float(g["distance_scenario_nm"].sum())
            base_co2 = float(np.sum(g["base_pred"] * g["cf_used"] * 1e6))
            sc_co2 = float(np.sum(g["scenario_pred"] * g["cf_used"] * 1e6))
            base_proxy = base_co2 / (dwt * base_dist) if dwt * base_dist > 0 else np.nan
            sc_proxy = sc_co2 / (dwt * sc_dist) if dwt * sc_dist > 0 else np.nan
            vessel_rows.append({
                "reduction_pct": reduction * 100,
                "vessel_id": vessel,
                "ship_type": g["ship_type"].iloc[0],
                "n_supported": len(g),
                "retained_pct": len(g) / max((base["vessel_id"] == vessel).sum(), 1) * 100,
                "baseline_predicted_fuel_t": base_fuel,
                "scenario_predicted_fuel_t": sc_fuel,
                "fuel_change_pct": (sc_fuel - base_fuel) / base_fuel * 100 if base_fuel else np.nan,
                "baseline_distance_nm": base_dist,
                "scenario_distance_nm": sc_dist,
                "distance_change_pct": (sc_dist - base_dist) / base_dist * 100 if base_dist else np.nan,
                "baseline_transport_work": dwt * base_dist,
                "scenario_transport_work": dwt * sc_dist,
                "baseline_CII_proxy": base_proxy,
                "scenario_CII_proxy": sc_proxy,
                "CII_proxy_change_pct": (sc_proxy - base_proxy) / base_proxy * 100 if base_proxy else np.nan,
                "improved": bool(sc_proxy < base_proxy) if np.isfinite(sc_proxy) and np.isfinite(base_proxy) else False,
            })
    vessels = pd.DataFrame(vessel_rows)
    summary_rows = []
    for reduction, rg in vessels.groupby("reduction_pct"):
        for scope, sg in [("Fleet", rg)] + [(st, g) for st, g in rg.groupby("ship_type")]:
            base_fuel = sg["baseline_predicted_fuel_t"].sum()
            sc_fuel = sg["scenario_predicted_fuel_t"].sum()
            # Exact aggregation of vessel-level proxies uses the CII denominators
            # (DWT × distance) as weights, which is algebraically equivalent to
            # total CO2 divided by total capacity-distance over the supported records.
            base_weights = sg["baseline_transport_work"].values
            sc_weights = sg["scenario_transport_work"].values
            base_proxy = np.average(sg["baseline_CII_proxy"], weights=np.maximum(base_weights, 1e-12))
            sc_proxy = np.average(sg["scenario_CII_proxy"], weights=np.maximum(sc_weights, 1e-12))
            summary_rows.append({
                "reduction_pct": reduction,
                "scope": scope,
                "predicted_fuel_change_pct": (sc_fuel - base_fuel) / base_fuel * 100 if base_fuel else np.nan,
                "CII_proxy_change_pct_weighted": (sc_proxy - base_proxy) / base_proxy * 100 if base_proxy else np.nan,
                "median_vessel_CII_proxy_change_pct": float(sg["CII_proxy_change_pct"].median()),
                "mean_vessel_CII_proxy_change_pct": float(sg["CII_proxy_change_pct"].mean()),
                "vessels_improved": int(sg["improved"].sum()),
                "vessels_total": int(len(sg)),
                "mean_retained_pct": float(sg["retained_pct"].mean()),
            })
    return vessels, pd.DataFrame(summary_rows)


def known_vessel_temporal_split(raw: pd.DataFrame, frac_train: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:
    if raw["timestamp"].isna().all():
        raise ValueError("Temporal validation requires a timestamp column")
    train_idx, test_idx = [], []
    for vessel, inds in raw.groupby("vessel_id").groups.items():
        sub = raw.loc[inds].sort_values("timestamp")
        cut = max(1, min(len(sub) - 1, int(math.floor(len(sub) * frac_train))))
        train_idx.extend(sub.index[:cut].tolist())
        test_idx.extend(sub.index[cut:].tolist())
    return np.array(sorted(train_idx)), np.array(sorted(test_idx))


def run_lovo(
    model_name: str,
    params: Dict,
    X: pd.DataFrame,
    raw: pd.DataFrame,
    args,
    logger: logging.Logger,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    pred_all = np.full(len(raw), np.nan)
    fold_rows = []
    for i, vessel in enumerate(sorted(raw["vessel_id"].unique()), start=1):
        te = raw["vessel_id"].values == vessel
        tr = ~te
        logger.info("LOVO %02d/%02d | hold out vessel=%s | train=%d test=%d", i, raw["vessel_id"].nunique(), vessel, tr.sum(), te.sum())
        model = model_factory(model_name, params, args.seed, args.n_jobs)
        model.fit(X.loc[tr], raw.loc[tr, "target"].values)
        p = model.predict(X.loc[te])
        pred_all[te] = p
        row = {"vessel_id": vessel, "ship_type": raw.loc[te, "ship_type"].iloc[0]}
        row.update(metric_dict(raw.loc[te, "target"].values, p))
        fold_rows.append(row)
    overall = pd.DataFrame([metric_dict(raw["target"].values, pred_all)])
    overall.insert(0, "model", model_name)
    folds = pd.DataFrame(fold_rows)
    return overall, folds, pred_all


def run_adaptation(
    model_name: str,
    params: Dict,
    X: pd.DataFrame,
    raw: pd.DataFrame,
    args,
    logger: logging.Logger,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    if raw["timestamp"].isna().all():
        raise ValueError("New-vessel adaptation requires timestamp")
    pred_all = np.full(len(raw), np.nan)
    tested = np.zeros(len(raw), dtype=bool)
    fold_rows = []
    vessels = sorted(raw["vessel_id"].unique())
    for i, vessel in enumerate(vessels, start=1):
        target_idx = raw.index[raw["vessel_id"] == vessel]
        target_sorted = raw.loc[target_idx].sort_values("timestamp")
        cut = max(1, min(len(target_sorted) - 1, int(math.floor(len(target_sorted) * 0.8))))
        adapt_idx = target_sorted.index[:cut]
        test_idx = target_sorted.index[cut:]
        source_idx = raw.index[raw["vessel_id"] != vessel]
        train_idx = np.concatenate([source_idx.values, adapt_idx.values])
        logger.info("Adapt %02d/%02d | vessel=%s | source+adapt=%d test=%d", i, len(vessels), vessel, len(train_idx), len(test_idx))
        model = model_factory(model_name, params, args.seed, args.n_jobs)
        model.fit(X.loc[train_idx], raw.loc[train_idx, "target"].values)
        p = model.predict(X.loc[test_idx])
        pred_all[test_idx] = p
        tested[test_idx] = True
        row = {"vessel_id": vessel, "ship_type": raw.loc[test_idx, "ship_type"].iloc[0]}
        row.update(metric_dict(raw.loc[test_idx, "target"].values, p))
        fold_rows.append(row)
    overall = pd.DataFrame([metric_dict(raw.loc[tested, "target"].values, pred_all[tested])])
    overall.insert(0, "model", model_name)
    return overall, pd.DataFrame(fold_rows), pred_all


def maybe_copy_consistency_files(args, out_dir: Path, logger: logging.Logger):
    if not args.consistency_dir:
        return []
    src = Path(args.consistency_dir)
    copied = []
    if src.exists():
        dest = out_dir / "00_data_audit" / "consistency_files"
        dest.mkdir(parents=True, exist_ok=True)
        for name in ["01_cruise_fixed27_consistency.json", "02_cruise_fixed31_consistency.json"]:
            p = src / name
            if p.exists():
                shutil.copy2(p, dest / name)
                copied.append(str(dest / name))
                logger.info("Copied consistency audit: %s", p)
    return copied


def write_ready_to_paste_summary(
    out_dir: Path,
    record_table: pd.DataFrame,
    phase_table: pd.DataFrame,
    ablation_table: Optional[pd.DataFrame],
    temporal_overall: Optional[pd.DataFrame],
    lovo_overall: Optional[pd.DataFrame],
    adaptation_overall: Optional[pd.DataFrame],
    gen_ladder: Optional[pd.DataFrame],
    scenario_summary: Optional[pd.DataFrame],
    selected_generalisation_model: str,
):
    lines = [
        "# Auto-generated manuscript-ready results notes",
        "",
        "These paragraphs are generated from the CSV outputs. Verify journal style and the exact model naming before pasting.",
        "",
        "## Record-level comparison",
    ]
    if not record_table.empty:
        best = record_table.sort_values("RMSE").iloc[0]
        lines.append(
            f"Under the common record-level holdout design, {best['model']} achieved the lowest RMSE "
            f"({best['RMSE']:.6f} t/10 min), with MAE {best['MAE']:.6f} t/10 min and R² {best['R2']:.6f}. "
            "These metrics represent within-fleet record-level interpolation and should not be interpreted as zero-shot transfer to unseen vessels."
        )
    if lovo_overall is not None and not lovo_overall.empty:
        r = lovo_overall.iloc[0]
        lines += ["", "## LOVO zero-shot generalisation", (
            f"Locked-hyperparameter leave-one-vessel-out validation using {selected_generalisation_model} produced "
            f"RMSE {r['RMSE']:.6f}, MAE {r['MAE']:.6f}, and R² {r['R2']:.6f}. "
            "The deterioration relative to record-level testing indicates material between-vessel heterogeneity and limits direct zero-shot transfer. "
            "Because hyperparameters were fixed rather than re-optimised inside each outer fold, this experiment should be described as locked-hyperparameter LOVO rather than fully nested cross-validation."
        )]
    if temporal_overall is not None and not temporal_overall.empty:
        r = temporal_overall.iloc[0]
        lines += ["", "## Known-vessel temporal forecasting", (
            f"When the first 80% of each vessel history was used for training and the final 20% for testing, "
            f"{selected_generalisation_model} achieved RMSE {r['RMSE']:.6f}, MAE {r['MAE']:.6f}, and R² {r['R2']:.6f}. "
            "This design evaluates future prediction for vessels whose past data are already available."
        )]
    if adaptation_overall is not None and not adaptation_overall.empty:
        r = adaptation_overall.iloc[0]
        lines += ["", "## New-vessel adaptation", (
            f"After adding the first 80% of each target vessel's observations to the training pool and testing on its remaining 20%, "
            f"RMSE was {r['RMSE']:.6f}, MAE {r['MAE']:.6f}, and R² {r['R2']:.6f}. "
            "The comparison with zero-shot LOVO quantifies the value of vessel-specific adaptation."
        )]
    if ablation_table is not None and not ablation_table.empty:
        full = ablation_table[ablation_table["configuration"] == "AIS+Weather+controls"]
        if not full.empty:
            f = full.iloc[0]
            lines += ["", "## Multi-source feature ablation", (
                f"The full AIS-plus-weather specification achieved RMSE {f['RMSE']:.6f} and R² {f['R2']:.6f}. "
                "Reduced operational-only and weather-only configurations were evaluated on the identical test set to quantify the incremental value of multi-source fusion."
            )]
    if scenario_summary is not None and not scenario_summary.empty:
        fleet = scenario_summary[scenario_summary["scope"] == "Fleet"]
        if not fleet.empty:
            lines += ["", "## Main-engine CII-proxy speed sensitivity"]
            for _, r in fleet.sort_values("reduction_pct").iterrows():
                lines.append(
                    f"A {r['reduction_pct']:.0f}% hypothetical speed reduction changed predicted fuel by "
                    f"{r['predicted_fuel_change_pct']:.3f}% and the transport-work-weighted main-engine CII proxy by "
                    f"{r['CII_proxy_change_pct_weighted']:.3f}%; {int(r['vessels_improved'])}/{int(r['vessels_total'])} vessels improved."
                )
            lines.append("These are model-based fixed-time sensitivity results, not causal estimates or statutory CII ratings.")
    if gen_ladder is not None and not gen_ladder.empty:
        lines += ["", "## Generalisation ladder", "See Table_E19_generalisation_ladder.csv for the full four-level comparison."]

    (out_dir / "AUTO_RESULTS_SUMMARY.md").write_text("\n\n".join(lines), encoding="utf-8")


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data-dir", type=str, default=None,
                   help="Directory containing final_fixed31_*.csv files")
    p.add_argument("--input-csv", type=str, default=None,
                   help="Optional combined final_fixed31_all_phases_model_ready.csv path")
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--column-overrides", type=str, default=None,
                   help="JSON mapping canonical names to actual dataset columns")
    p.add_argument("--consistency-dir", type=str, default=None,
                   help="Directory containing cruise Fixed27/Fixed31 consistency JSON files")
    p.add_argument("--primary-scope", choices=["all_phases", "cruise"], default="all_phases",
                   help="Primary record-level model cohort. all_phases follows the revised design; cruise preserves the prior F31 core cohort.")
    p.add_argument("--seed", type=int, default=20260808)
    p.add_argument("--n-jobs", type=int, default=8)
    p.add_argument("--test-size", type=float, default=0.20)
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--stage-a-trials", type=int, default=40)
    p.add_argument("--stage-a-n", type=int, default=30000)
    p.add_argument("--stage-b-n", type=int, default=100000)
    p.add_argument("--csv-chunksize", type=int, default=50000,
                   help="Rows per pandas CSV chunk for bounded-memory Fixed31 loading")
    p.add_argument("--trajectory-gap-minutes", type=float, default=30.0)
    p.add_argument("--reuse-hyperparams", type=str, default=None,
                   help="Optional JSON generated by a previous run; skips Optuna and reuses locked parameters")
    p.add_argument("--ablation-model", choices=["rf", "xgb", "lgbm"], default="lgbm")
    p.add_argument("--explain-model", choices=["rf", "xgb", "lgbm"], default="xgb")
    p.add_argument("--generalisation-model", choices=["auto_best_tree", "rf", "xgb", "lgbm"], default="auto_best_tree")
    p.add_argument("--run-new-vessel-adaptation", action="store_true")
    p.add_argument("--skip-lovo", action="store_true")
    p.add_argument("--skip-shap", action="store_true")
    p.add_argument("--skip-cii", action="store_true")
    p.add_argument("--shap-sample-n", type=int, default=12000)
    p.add_argument("--local-audit-n", type=int, default=20000)
    p.add_argument("--pdp-grid-n", type=int, default=20)
    p.add_argument("--pdp-support-min", type=int, default=30)
    p.add_argument("--bootstrap-reps", type=int, default=2000)
    p.add_argument("--co2-factor", type=float, default=3.114,
                   help="Fallback t CO2 per t fuel. Replace with the correct fuel-specific IMO factor if a row-wise factor is unavailable.")
    p.add_argument("--cii-scenario-phase", choices=["cruise", "cruise_maneuver", "primary"], default="cruise",
                   help="Rows used for hypothetical speed-reduction scenarios. The default avoids applying slow-steaming counterfactuals to anchoring/berthing windows.")
    p.add_argument("--wind-zero-is", choices=["unknown", "headwind", "following"], default="unknown")
    p.add_argument("--expected-vessels", type=int, default=21)
    p.add_argument("--expected-cruise-n", type=int, default=489620)
    p.add_argument("--expected-maneuver-n", type=int, default=15851)
    p.add_argument("--expected-anchor-n", type=int, default=41583)
    p.add_argument("--strict-audit", action="store_true")
    p.add_argument("--quick-check", action="store_true",
                   help="Reduce trial counts and samples for a fast pipeline execution check")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    logger = setup_logging(out_dir)
    t0 = time.time()

    if args.quick_check:
        args.stage_a_trials = min(args.stage_a_trials, 2)
        args.stage_a_n = min(args.stage_a_n, 1500)
        args.stage_b_n = min(args.stage_b_n, 3000)
        args.shap_sample_n = min(args.shap_sample_n, 1000)
        args.local_audit_n = min(args.local_audit_n, 1000)
        args.bootstrap_reps = min(args.bootstrap_reps, 100)
        args.pdp_grid_n = min(args.pdp_grid_n, 8)
        args.n_jobs = min(args.n_jobs, 2)
        logger.info("QUICK SMOKE MODE ENABLED")

    for sub in [
        "00_data_audit", "01_phase_accuracy", "02_ridge_interaction",
        "03_known_vessel_temporal", "04_shap_physical", "05_cii_scenarios",
        "06_generalisation", "07_model_artifacts",
    ]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    logger.info("F31 final journal pipeline version %s", SCRIPT_VERSION)
    logger.info("Primary scope: %s", args.primary_scope)
    logger.warning(
        "CII CO2 conversion factor: %.6f t CO2/t fuel.",
        args.co2_factor,
    )
    if args.wind_zero_is == "unknown":
        logger.warning("Relative-wind 0-degree convention is unknown; wind sectors will not be labelled head/following.")

    df = load_phase_files(args, logger)
    bundle_all = canonicalize_dataframe(df, args, logger)
    audit = strict_fixed31_audit(bundle_all.raw, args, logger)
    copied_consistency = maybe_copy_consistency_files(args, out_dir, logger)

    # Data-retention flow supplied in the final Fixed31 audit; kept as an explicit reproducibility table.
    retention_rows = [
        {"phase": "cruise", "Fixed27_input": 529512, "Fixed28_deleted": 39147, "Fixed29_deleted": 420, "Fixed30_empirical_rule_deleted": 0, "Fixed31_deleted": 325, "Fixed31_final": 489620},
        {"phase": "maneuver", "Fixed27_input": 19334, "Fixed28_deleted": 3479, "Fixed29_deleted": 0, "Fixed30_empirical_rule_deleted": 0, "Fixed31_deleted": 4, "Fixed31_final": 15851},
        {"phase": "anchor_berth", "Fixed27_input": 87644, "Fixed28_deleted": 45751, "Fixed29_deleted": 3, "Fixed30_empirical_rule_deleted": 0, "Fixed31_deleted": 307, "Fixed31_final": 41583},
    ]
    pd.DataFrame(retention_rows).to_csv(out_dir / "00_data_audit" / "Table_E0_data_retention_flow.csv", index=False)
    bundle_all.raw.groupby(["ship_type", "vessel_id", "phase"]).size().rename("n").reset_index().to_csv(
        out_dir / "00_data_audit" / "Table_E0b_rows_by_vessel_phase.csv", index=False
    )

    # Journal-ready descriptive statistics from the audited final dataset.
    desc_vars = [
        "target", "speed_kn", "draught_m", "rudder_deg", "trim_m",
        "rel_wind_speed_kn", "wave_height_m", "wave_period_s", "mslp_hpa", "sst_c",
    ]
    desc_rows = []
    for scope, g in [("Overall", bundle_all.raw)] + [(st, sg) for st, sg in bundle_all.raw.groupby("ship_type")]:
        for v in desc_vars:
            if v in g.columns:
                desc_rows.append({
                    "scope": scope, "variable": v, "n": int(g[v].notna().sum()),
                    "mean": float(g[v].mean()), "std": float(g[v].std(ddof=1)),
                    "median": float(g[v].median()),
                })
    pd.DataFrame(desc_rows).to_csv(out_dir / "00_data_audit" / "Table_E0c_descriptive_stats_by_ship_type.csv", index=False)

    corr_vars = [
        "target", "speed_kn", "draught_m", "rudder_deg", "trim_m",
        "rel_wind_speed_kn", "wave_height_m", "wave_period_s", "mslp_hpa", "sst_c",
    ]
    corr = bundle_all.raw[corr_vars].corr(method="pearson")
    corr.to_csv(out_dir / "00_data_audit" / "Table_E0d_pearson_correlation_matrix.csv")
    fig = plt.figure(figsize=(8.5, 7.0))
    ax = fig.add_subplot(111)
    im = ax.imshow(corr.values, aspect="auto", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr_vars)))
    ax.set_yticks(range(len(corr_vars)))
    ax.set_xticklabels(corr_vars, rotation=55, ha="right", fontsize=8)
    ax.set_yticklabels(corr_vars, fontsize=8)
    for i in range(len(corr_vars)):
        for j in range(len(corr_vars)):
            ax.text(j, i, f"{corr.iloc[i,j]:.2f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=ax, label="Pearson r")
    fig.tight_layout()
    fig.savefig(out_dir / "00_data_audit" / "Figure_E0_pearson_correlation_matrix.png", dpi=220)
    plt.close(fig)

    bundle = select_primary_scope(bundle_all, args.primary_scope, logger)
    raw = bundle.raw.reset_index(drop=True)
    X = bundle.feature_df.reset_index(drop=True)
    y = raw["target"].to_numpy(dtype=float)

    train_idx, test_idx = stratified_record_split(raw, args.test_size, args.seed)
    X_train, X_test = X.iloc[train_idx].reset_index(drop=True), X.iloc[test_idx].reset_index(drop=True)
    y_train, y_test = y[train_idx], y[test_idx]
    raw_train = raw.iloc[train_idx].reset_index(drop=True)
    raw_test = raw.iloc[test_idx].reset_index(drop=True)
    groups_train = raw_train["trajectory_group"].astype(str).reset_index(drop=True)

    stage_a_idx = sample_indices(len(X_train), args.stage_a_n, args.seed + 11)
    stage_b_idx = sample_indices(len(X_train), args.stage_b_n, args.seed + 29)

    model_names = ["lr", "ridge_interaction", "dt", "rf", "xgb", "lgbm", "ann"]
    params_by_model = {}
    tuning_meta = {}
    if args.reuse_hyperparams:
        data = json.loads(Path(args.reuse_hyperparams).read_text(encoding="utf-8"))
        params_by_model = data.get("params_by_model", data)
        logger.info("Reusing locked hyperparameters from %s", args.reuse_hyperparams)
    else:
        for name in model_names:
            params, meta = three_stage_tune(
                name, X_train, y_train, groups_train,
                stage_a_idx, stage_b_idx, args, logger
            )
            params_by_model[name] = params
            tuning_meta[name] = meta
        save_json({"params_by_model": params_by_model, "tuning_meta": tuning_meta}, out_dir / "07_model_artifacts" / "locked_hyperparameters.json")

    # Stage C full outer-training fit and common record-level test evaluation.
    fitted = {}
    predictions = {}
    record_rows = []
    phase_rows = []
    vessel_metric_frames = []
    shiptype_rows = []
    logger.info("Stage C: fitting all models on full outer training set")
    for name in model_names:
        logger.info("Fit final %s", name)
        model, pred = fit_predict_model(name, params_by_model.get(name, {}), X_train, y_train, X_test, args.seed, args.n_jobs)
        fitted[name] = model
        predictions[name] = pred
        row = {"model": name}
        row.update(metric_dict(y_test, pred))
        record_rows.append(row)

        # Per-phase evaluation on identical outer test rows.
        tmp = raw_test.copy()
        for phase, gidx in tmp.groupby("phase").groups.items():
            inds = np.array(list(gidx), dtype=int)
            r = {"model": name, "phase": phase}
            r.update(metric_dict(y_test[inds], pred[inds]))
            phase_rows.append(r)

        # Ship-type micro metrics.
        for st, gidx in raw_test.groupby("ship_type").groups.items():
            inds = np.array(list(gidx), dtype=int)
            r = {"model": name, "ship_type": st}
            r.update(metric_dict(y_test[inds], pred[inds]))
            shiptype_rows.append(r)

        vm = per_group_metrics(raw_test, pred, ["vessel_id", "ship_type"])
        vm.insert(0, "model", name)
        vessel_metric_frames.append(vm)

    record_table = pd.DataFrame(record_rows).sort_values("RMSE")
    record_table.to_csv(out_dir / "01_phase_accuracy" / "Table_E0_record_level_model_comparison.csv", index=False)
    phase_table = pd.DataFrame(phase_rows)
    phase_table.to_csv(out_dir / "01_phase_accuracy" / "Table_E1_accuracy_by_operating_phase.csv", index=False)
    shiptype_table = pd.DataFrame(shiptype_rows)
    shiptype_table.to_csv(out_dir / "01_phase_accuracy" / "Table_E2_accuracy_by_ship_type.csv", index=False)
    vessel_metrics = pd.concat(vessel_metric_frames, ignore_index=True)
    vessel_metrics.to_csv(out_dir / "01_phase_accuracy" / "Table_E2b_accuracy_by_vessel.csv", index=False)

    macro_rows = []
    for name, g in vessel_metrics.groupby("model"):
        r = {"model": name}
        r.update(macro_vessel_summary(g))
        macro_rows.append(r)
    pd.DataFrame(macro_rows).to_csv(out_dir / "01_phase_accuracy" / "Table_E2c_equal_vessel_macro_metrics.csv", index=False)

    # Explicit LR vs Physics-informed Ridge-Interaction baseline.
    ridge_comp = record_table[record_table["model"].isin(["lr", "ridge_interaction"])].copy()
    ridge_comp.to_csv(out_dir / "02_ridge_interaction" / "Table_E3_lr_vs_ridge_interaction.csv", index=False)
    ridge_vm = vessel_metrics[vessel_metrics["model"].isin(["lr", "ridge_interaction"])].copy()
    ridge_vm.to_csv(out_dir / "02_ridge_interaction" / "Table_E4_lr_vs_ridge_by_vessel.csv", index=False)

    # Choose generalisation model after record-level comparison unless explicitly locked.
    if args.generalisation_model == "auto_best_tree":
        tree_best = record_table[record_table["model"].isin(["rf", "xgb", "lgbm"])].sort_values("RMSE").iloc[0]
        gen_model_name = str(tree_best["model"])
    else:
        gen_model_name = args.generalisation_model
    logger.info("Generalisation ladder model: %s", gen_model_name)

    # Feature ablation on identical test observations, locked hyperparameters.
    ablation_configs = {
        "AIS+controls": [c for c in OPERATIONAL_FEATURES if c in X.columns],
        "Weather+controls": [c for c in WEATHER_FEATURES if c in X.columns],
        "AIS+Weather+controls": list(bundle.feature_cols),
    }
    ablation_rows = []
    ablation_preds = {}
    abl_model = args.ablation_model
    for label, cols in ablation_configs.items():
        logger.info("Ablation %s using %s with %d features", label, abl_model, len(cols))
        m = model_factory(abl_model, params_by_model[abl_model], args.seed, args.n_jobs)
        m.fit(X_train[cols], y_train)
        p = m.predict(X_test[cols])
        ablation_preds[label] = p
        r = {"configuration": label, "model": abl_model, "n_features": len(cols)}
        r.update(metric_dict(y_test, p))
        ablation_rows.append(r)
    ablation_table = pd.DataFrame(ablation_rows)
    full_rmse = float(ablation_table.loc[ablation_table["configuration"] == "AIS+Weather+controls", "RMSE"].iloc[0])
    ablation_table["delta_RMSE_vs_full"] = ablation_table["RMSE"] - full_rmse
    ablation_table["RMSE_worsening_pct_vs_full"] = (ablation_table["RMSE"] / full_rmse - 1.0) * 100.0
    full_pred = ablation_preds["AIS+Weather+controls"]
    ci_cols = []
    for label in ["AIS+controls", "Weather+controls"]:
        ci = cluster_bootstrap_rmse_difference(
            raw_test["vessel_id"], y_test, ablation_preds[label], full_pred,
            args.bootstrap_reps, args.seed + 100 + len(ci_cols),
        )
        ci["configuration"] = label
        ci_cols.append(ci)
    pd.DataFrame(ci_cols).to_csv(out_dir / "02_ridge_interaction" / "Table_E4b_ablation_cluster_bootstrap.csv", index=False)
    ablation_table.to_csv(out_dir / "02_ridge_interaction" / "Table_E4c_multisource_feature_ablation.csv", index=False)

    # Known-vessel temporal validation using locked hyperparameters.
    temporal_overall = temporal_by_type = temporal_by_vessel = None
    temporal_pred = None
    try:
        tr_t, te_t = known_vessel_temporal_split(raw, 0.8)
        gm = model_factory(gen_model_name, params_by_model[gen_model_name], args.seed, args.n_jobs)
        gm.fit(X.loc[tr_t], y[tr_t])
        temporal_pred = gm.predict(X.loc[te_t])
        temporal_overall = pd.DataFrame([{"model": gen_model_name, **metric_dict(y[te_t], temporal_pred)}])
        temporal_by_type = per_group_metrics(raw.loc[te_t].reset_index(drop=True), temporal_pred, ["ship_type"])
        temporal_by_vessel = per_group_metrics(raw.loc[te_t].reset_index(drop=True), temporal_pred, ["vessel_id", "ship_type"])
        temporal_overall.to_csv(out_dir / "03_known_vessel_temporal" / "Table_E5_known_vessel_temporal_overall.csv", index=False)
        temporal_by_type.to_csv(out_dir / "03_known_vessel_temporal" / "Table_E7_known_vessel_temporal_by_ship_type.csv", index=False)
        temporal_by_vessel.to_csv(out_dir / "03_known_vessel_temporal" / "Table_E8_known_vessel_temporal_by_vessel.csv", index=False)
    except Exception as exc:
        logger.warning("Known-vessel temporal validation failed: %s", exc)

    # LOVO and optional adaptation.
    lovo_overall = lovo_folds = adaptation_overall = adaptation_folds = None
    lovo_pred = adaptation_pred = None
    if not args.skip_lovo:
        lovo_overall, lovo_folds, lovo_pred = run_lovo(
            gen_model_name, params_by_model[gen_model_name], X, raw, args, logger
        )
        lovo_overall.to_csv(out_dir / "06_generalisation" / "Table_E6_LOVO_overall.csv", index=False)
        lovo_folds.to_csv(out_dir / "06_generalisation" / "Table_E6b_LOVO_by_vessel.csv", index=False)
    if args.run_new_vessel_adaptation:
        adaptation_overall, adaptation_folds, adaptation_pred = run_adaptation(
            gen_model_name, params_by_model[gen_model_name], X, raw, args, logger
        )
        adaptation_overall.to_csv(out_dir / "06_generalisation" / "Table_E6c_new_vessel_80pct_adaptation_overall.csv", index=False)
        adaptation_folds.to_csv(out_dir / "06_generalisation" / "Table_E6d_new_vessel_80pct_adaptation_by_vessel.csv", index=False)

    # Standardised LR coefficients with vessel-cluster robust confidence intervals.
    lr_coef_table = pd.DataFrame()
    if sm is not None:
        try:
            mu = X_train.mean(axis=0)
            sd = X_train.std(axis=0, ddof=0).replace(0, 1.0)
            Z = (X_train - mu) / sd
            Zc = sm.add_constant(Z, has_constant="add")
            ols = sm.OLS(y_train, Zc).fit(cov_type="cluster", cov_kwds={"groups": raw_train["vessel_id"].values})
            ci = ols.conf_int(alpha=0.05)
            coef_rows = []
            for feat in X_train.columns:
                coef_rows.append({
                    "feature": feat,
                    "standardized_LR_coefficient": float(ols.params[feat]),
                    "CI95_low": float(ci.loc[feat, 0]),
                    "CI95_high": float(ci.loc[feat, 1]),
                    "p_value_cluster_robust": float(ols.pvalues[feat]),
                })
            lr_coef_table = pd.DataFrame(coef_rows)
            lr_coef_table.to_csv(out_dir / "04_shap_physical" / "Table_E12e_LR_standardized_coefficients_cluster_CI.csv", index=False)
        except Exception as exc:
            logger.warning("Cluster-robust LR coefficient table failed: %s", exc)
    else:
        logger.warning("statsmodels is required for the cluster-robust LR coefficient table")

    # SHAP and physical audit on outer record-level test data.
    shap_global = grouped_shap = physical_matrix = local_audit = pd.DataFrame()
    if not args.skip_shap:
        explain_name = args.explain_model
        emodel = fitted[explain_name]
        shap_idx = sample_indices(len(X_test), args.shap_sample_n, args.seed + 77)
        Xsh = X_test.iloc[shap_idx].copy().reset_index(drop=True)
        rawsh = raw_test.iloc[shap_idx].copy().reset_index(drop=True)
        logger.info("Computing SHAP for %s on %d held-out rows", explain_name, len(Xsh))
        sv = get_tree_shap(emodel, Xsh)
        imp = np.mean(np.abs(sv), axis=0)
        shap_global = pd.DataFrame({"feature": Xsh.columns, "mean_abs_SHAP": imp}).sort_values("mean_abs_SHAP", ascending=False)
        shap_global["normalized_importance"] = shap_global["mean_abs_SHAP"] / shap_global["mean_abs_SHAP"].sum()
        shap_global.to_csv(out_dir / "04_shap_physical" / "Table_E10_global_shap_importance.csv", index=False)

        group_map = {
            "heading_sin": "heading_direction", "heading_cos": "heading_direction",
            "rel_wind_sin": "relative_wind_direction", "rel_wind_cos": "relative_wind_direction",
            "rel_wave_sin": "relative_wave_direction", "rel_wave_cos": "relative_wave_direction",
            "ship_type_bulk": "ship_type", "ship_type_container": "ship_type",
        }
        gg = shap_global.copy()
        gg["grouped_feature"] = gg["feature"].map(group_map).fillna(gg["feature"])
        grouped_shap = gg.groupby("grouped_feature", as_index=False)["mean_abs_SHAP"].sum().sort_values("mean_abs_SHAP", ascending=False)
        grouped_shap["normalized_importance"] = grouped_shap["mean_abs_SHAP"] / grouped_shap["mean_abs_SHAP"].sum()
        grouped_shap["operational_pathway"] = grouped_shap["grouped_feature"].map({
            "speed_kn": "directly_operational",
            "trim_m": "conditionally_operational",
            "rudder_deg": "conditionally_operational",
            "draught_m": "loading_related",
            "relative_wind_direction": "route/exposure_related",
            "relative_wave_direction": "route/exposure_related",
            "rel_wind_speed_kn": "environmental_exposure",
            "wave_height_m": "environmental_exposure",
            "wave_period_s": "environmental_exposure",
            "sst_c": "environmental_exposure",
            "mslp_hpa": "environmental_exposure",
            "ship_type": "structural_non_operational",
            "heading_direction": "route/heading_related",
        }).fillna("other")
        grouped_shap.to_csv(out_dir / "04_shap_physical" / "Table_E11_grouped_shap_operational_mapping.csv", index=False)

        # Direction matrix with three-state logic.
        rows = []
        for feat in ["speed_kn", "wave_height_m", "draught_m", "trim_m", "rel_wind_speed_kn"]:
            if feat not in Xsh.columns:
                continue
            j = Xsh.columns.get_loc(feat)
            rho = float(spearmanr(Xsh[feat], sv[:, j], nan_policy="omit").statistic)
            rows.append({
                "feature": feat,
                "prior_expectation": PHYSICAL_EXPECTATION.get(feat, "context_dependent"),
                "SHAP_value_spearman": rho,
                "status": physical_status_from_spearman(feat, rho),
            })
        # Signed rudder uses |rudder| for the audit.
        if "rudder_deg" in Xsh.columns:
            j = Xsh.columns.get_loc("rudder_deg")
            rho = float(spearmanr(np.abs(Xsh["rudder_deg"]), sv[:, j], nan_policy="omit").statistic)
            rows.append({
                "feature": "abs(rudder_deg)",
                "prior_expectation": PHYSICAL_EXPECTATION["rudder_abs"],
                "SHAP_value_spearman": rho,
                "status": "Not directionally testable / context-dependent",
            })
        for feat in ["relative_wind_direction", "relative_wave_direction", "heading_direction"]:
            rows.append({
                "feature": feat,
                "prior_expectation": "cyclic/context-dependent",
                "SHAP_value_spearman": np.nan,
                "status": "Not directionally testable / context-dependent",
            })
        physical_matrix = pd.DataFrame(rows)
        physical_matrix.to_csv(out_dir / "04_shap_physical" / "Table_E12_SHAP_physical_direction_matrix.csv", index=False)
        if not lr_coef_table.empty:
            matrix = lr_coef_table.merge(physical_matrix, on="feature", how="left")
            def lr_direction(r):
                lo, hi = r["CI95_low"], r["CI95_high"]
                if lo > 0: return "positive"
                if hi < 0: return "negative"
                return "uncertain"
            matrix["LR_direction_95CI"] = matrix.apply(lr_direction, axis=1)
            matrix.to_csv(out_dir / "04_shap_physical" / "Table_E12f_regression_SHAP_physical_matrix.csv", index=False)

        # Speed dependence + binned density + polynomial diagnostics.
        if "speed_kn" in Xsh.columns:
            j = Xsh.columns.get_loc("speed_kn")
            save_scatter_dependency(Xsh["speed_kn"], sv[:, j], out_dir / "04_shap_physical" / "Figure_E1_speed_SHAP_dependence.png", "Speed over ground (kn)")
            bins = pd.qcut(Xsh["speed_kn"], q=min(30, max(5, Xsh["speed_kn"].nunique() // 20)), duplicates="drop")
            speed_bins = pd.DataFrame({"speed_kn": Xsh["speed_kn"], "shap": sv[:, j], "bin": bins}).groupby("bin", observed=True).agg(
                speed_mean=("speed_kn", "mean"), shap_mean=("shap", "mean"), n=("shap", "size")
            ).reset_index(drop=True)
            speed_bins.to_csv(out_dir / "04_shap_physical" / "Table_E12a_speed_SHAP_binned_density.csv", index=False)
            poly = fit_poly_diagnostics(speed_bins["speed_mean"].values, speed_bins["shap_mean"].values, 3)
            poly.to_csv(out_dir / "04_shap_physical" / "Table_E12b_speed_polynomial_diagnostics.csv", index=False)

        # Relative-wind dependence by angular sector.
        if "rel_wind_speed_kn" in Xsh.columns:
            j = Xsh.columns.get_loc("rel_wind_speed_kn")
            wind_df = pd.DataFrame({
                "rel_wind_speed_kn": Xsh["rel_wind_speed_kn"],
                "wind_SHAP": sv[:, j],
                "rel_wind_dir_deg": rawsh["rel_wind_dir_deg"],
                "speed_kn": Xsh["speed_kn"],
            })
            wind_df["wind_sector"] = semantic_wind_sector(wind_df["rel_wind_dir_deg"], args.wind_zero_is)
            wind_df.to_csv(out_dir / "04_shap_physical" / "Table_E12c_wind_SHAP_by_sector_raw.csv", index=False)
            sector_rows = []
            for sector, g in wind_df.groupby("wind_sector"):
                rho = spearmanr(g["rel_wind_speed_kn"], g["wind_SHAP"], nan_policy="omit").statistic
                sector_rows.append({"wind_sector": sector, "n": len(g), "speed_SHAP_spearman": rho})
            pd.DataFrame(sector_rows).to_csv(out_dir / "04_shap_physical" / "Table_E12d_wind_SHAP_sector_summary.csv", index=False)
            save_scatter_dependency(wind_df["rel_wind_speed_kn"], wind_df["wind_SHAP"], out_dir / "04_shap_physical" / "Figure_E2_wind_SHAP_dependence.png", "Relative wind speed (kn)")

        # Density-aware 2D PDP using held-out support counts.
        pdp = partial_dependence_2d_density_aware(
            emodel, Xsh, rawsh, "speed_kn", "rel_wind_speed_kn",
            args.pdp_grid_n, args.pdp_support_min, args.seed,
        )
        pdp.to_csv(out_dir / "04_shap_physical" / "Table_E13_speed_wind_density_aware_PDP.csv", index=False)
        plot_density_pdp(pdp, out_dir / "04_shap_physical" / "Figure_E3_speed_wind_density_aware_PDP.png")

        local_audit = local_response_audit(
            emodel, raw_train, raw_test, bundle.feature_cols,
            args.local_audit_n, args.seed + 99,
        )
        local_audit.to_csv(out_dir / "04_shap_physical" / "Table_E14_local_physical_response_audit.csv", index=False)

    # CII model-choice sensitivity and speed scenarios.
    cii_model_summary = None
    scenario_summary = None
    if not args.skip_cii:
        cii_summaries = []
        for name in model_names:
            try:
                vdf = compute_cii_by_vessel(raw_test, predictions[name], args.co2_factor)
                vdf.insert(0, "model", name)
                vdf.to_csv(out_dir / "05_cii_scenarios" / f"Table_E15_CII_proxy_by_vessel_{name}.csv", index=False)
                s = {"model": name, **cii_summary(vdf, raw_test, predictions[name])}
                cii_summaries.append(s)
            except KeyError as exc:
                logger.warning("CII model-choice analysis failed: %s", exc)
                cii_summaries = []
                break
        if cii_summaries:
            cii_model_summary = pd.DataFrame(cii_summaries)
            cii_model_summary.to_csv(out_dir / "05_cii_scenarios" / "Table_E16_model_choice_CII_proxy_summary.csv", index=False)

        try:
            scenario_model = fitted[args.explain_model]
            if args.cii_scenario_phase == "cruise":
                scenario_train = raw_train.loc[raw_train["phase"] == "cruise"].copy()
                scenario_test = raw_test.loc[raw_test["phase"] == "cruise"].copy()
            elif args.cii_scenario_phase == "cruise_maneuver":
                scenario_train = raw_train.loc[raw_train["phase"].isin(["cruise", "maneuver"])].copy()
                scenario_test = raw_test.loc[raw_test["phase"].isin(["cruise", "maneuver"])].copy()
            else:
                scenario_train = raw_train.copy()
                scenario_test = raw_test.copy()
            if scenario_train.empty or scenario_test.empty:
                raise ValueError(f"No rows available for CII scenario phase scope: {args.cii_scenario_phase}")
            logger.info("CII speed-reduction scenario scope=%s: train=%d, test=%d", args.cii_scenario_phase, len(scenario_train), len(scenario_test))
            scenario_vessel, scenario_summary = speed_reduction_scenarios(
                scenario_model, scenario_train, scenario_test, bundle.feature_cols,
                [0.05, 0.10, 0.15], args.co2_factor,
            )
            scenario_vessel.insert(0, "scenario_phase_scope", args.cii_scenario_phase)
            scenario_summary.insert(0, "scenario_phase_scope", args.cii_scenario_phase)
            scenario_vessel.to_csv(out_dir / "05_cii_scenarios" / "Table_E17_speed_reduction_CII_proxy_by_vessel.csv", index=False)
            scenario_summary.to_csv(out_dir / "05_cii_scenarios" / "Table_E18_speed_reduction_CII_proxy_scenarios_summary.csv", index=False)
        except KeyError as exc:
            logger.warning("CII speed-scenario analysis failed: %s", exc)

    # Generalisation ladder on one locked model.
    gen_rows = []
    rec = record_table[record_table["model"] == gen_model_name].iloc[0]
    gen_rows.append({
        "validation_design": "Random record 80/20",
        "target_vessel_history_visible": "Partly visible",
        "deployment_question": "Record-level interpolation on represented vessels",
        "model": gen_model_name,
        **{k: rec[k] for k in ["n", "RMSE", "MAE", "MAPE_pct", "sMAPE_pct", "R2", "cumulative_bias_pct"]},
    })
    if temporal_overall is not None and not temporal_overall.empty:
        r = temporal_overall.iloc[0]
        gen_rows.append({
            "validation_design": "All vessels first 80% -> last 20%",
            "target_vessel_history_visible": "First 80% visible",
            "deployment_question": "Future prediction for known vessels",
            "model": gen_model_name,
            **{k: r[k] for k in ["n", "RMSE", "MAE", "MAPE_pct", "sMAPE_pct", "R2", "cumulative_bias_pct"]},
        })
    if lovo_overall is not None and not lovo_overall.empty:
        r = lovo_overall.iloc[0]
        gen_rows.append({
            "validation_design": "Locked-hyperparameter LOVO zero-shot",
            "target_vessel_history_visible": "No",
            "deployment_question": "Direct transfer to an unseen vessel",
            "model": gen_model_name,
            **{k: r[k] for k in ["n", "RMSE", "MAE", "MAPE_pct", "sMAPE_pct", "R2", "cumulative_bias_pct"]},
        })
    if adaptation_overall is not None and not adaptation_overall.empty:
        r = adaptation_overall.iloc[0]
        gen_rows.append({
            "validation_design": "LOVO + target vessel first 80% adaptation",
            "target_vessel_history_visible": "First 80% visible",
            "deployment_question": "Personalised prediction after new-vessel history accumulates",
            "model": gen_model_name,
            **{k: r[k] for k in ["n", "RMSE", "MAE", "MAPE_pct", "sMAPE_pct", "R2", "cumulative_bias_pct"]},
        })
    gen_ladder = pd.DataFrame(gen_rows)
    gen_ladder.to_csv(out_dir / "Table_E19_generalisation_ladder.csv", index=False)

    # Persist model artifacts where practical.
    try:
        import joblib
        for name in ["rf", "xgb", "lgbm", args.explain_model]:
            if name in fitted:
                joblib.dump(fitted[name], out_dir / "07_model_artifacts" / f"{name}_record_split.joblib")
    except Exception as exc:
        logger.warning("Model serialization warning: %s", exc)

    manifest = {
        "script_version": SCRIPT_VERSION,
        "started_at_epoch": t0,
        "elapsed_seconds": time.time() - t0,
        "python": sys.version,
        "platform": platform.platform(),
        "package_versions": {
            "numpy": np.__version__, "pandas": pd.__version__,
            "sklearn": __import__("sklearn").__version__,
            "optuna": optuna.__version__, "xgboost": __import__("xgboost").__version__,
            "lightgbm": __import__("lightgbm").__version__, "shap": shap.__version__,
        },
        "args": vars(args),
        "column_map": bundle_all.column_map,
        "all_phase_audit": audit,
        "primary_rows": int(len(raw)),
        "primary_vessels": int(raw["vessel_id"].nunique()),
        "outer_train_n": int(len(train_idx)),
        "outer_test_n": int(len(test_idx)),
        "feature_columns": bundle.feature_cols,
        "locked_params": params_by_model,
        "generalisation_model": gen_model_name,
        "explain_model": args.explain_model,
        "ablation_model": args.ablation_model,
        "consistency_files_copied": copied_consistency,
        "method_notes": {
            "LOVO": "Locked hyperparameters; each vessel completely excluded from model fitting in its fold; not fully nested hyperparameter validation.",
            "CII": "Observation-period main-engine proxy only; not statutory attained CII or rating.",
            "CII_speed_scenario_scope": args.cii_scenario_phase,
            "speed_scenario": "Fixed 10-min local sensitivity on the declared scenario phase scope; distance changes proportionally with SOG; relative wind held fixed unless separately recomputed upstream.",
            "SHAP": "Model attribution; not causal effect and not a direct optimisation weight.",
        },
    }
    save_json(manifest, out_dir / "run_manifest.json")

    write_ready_to_paste_summary(
        out_dir, record_table, phase_table, ablation_table,
        temporal_overall, lovo_overall, adaptation_overall,
        gen_ladder, scenario_summary, gen_model_name,
    )

    logger.info("Pipeline completed successfully in %.1f minutes", (time.time() - t0) / 60.0)
    logger.info("Main output: %s", out_dir)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
