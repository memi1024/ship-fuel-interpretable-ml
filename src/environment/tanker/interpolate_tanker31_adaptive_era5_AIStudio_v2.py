# -*- coding: utf-8 -*-
"""
AI Studio - fixed31 tanker + cruise adaptive ERA5 interpolation
==============================================================

Target
------
Use cleaned fixed31 tanker+cruise coordinates and the adaptive ERA5 manifest
to interpolate seven harmonised environmental fields:

    wind_s      10 m wind speed, knots
    wind_d      meteorological wind FROM direction, degree true
    wave_h      significant wave height, m
    wave_d      mean wave direction, degree true
    wave_p      mean wave period, s
    surface_t   sea-surface temperature, deg C
    surface_p   surface pressure, Pa

Interpolation
-------------
Spatial  : strict bilinear
Temporal : linear
Wave dir : circular sine/cosine interpolation
Wind dir : calculated after interpolating u10 and v10

Important
---------
* Strict ERA5 SST only.
* NEVER uses 2 m air temperature.
* NEVER uses nearest-neighbour fallback.
* Supports adaptive spatial parts and cross-month pad files.
* Uses era5_request_manifest.csv to route observations to their ERA5 part.
* Processes one spatial part at a time to keep memory use low.
* Supports checkpoint/resume.
* A checkpoint is saved ONLY when all ERA5 files required for that part exist.
  This prevents incomplete uploads/downloads from being permanently skipped.
"""

from __future__ import annotations

import ast
import gc
import json
import math
import os
import re
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from netCDF4 import Dataset, num2date

try:
    from netCDF4 import set_chunk_cache
except ImportError:
    set_chunk_cache = None


# =============================================================================
# 1. USER SETTINGS
# =============================================================================

REPO_ROOT = Path(__file__).resolve().parents[3]

# Optional explicit locations. Environment variables are preferred for
# machine-specific paths; otherwise the script searches repository-local data.
TARGET_CSV_OVERRIDE = os.environ.get("SHIP_FUEL_TANKER_FIXED31_COORDS")
ERA5_ROOT_OVERRIDE = os.environ.get("SHIP_FUEL_TANKER_ERA5_ROOT")

SEARCH_ROOTS = [
    REPO_ROOT / "data",
    REPO_ROOT / "data" / "era5" / "tanker",
    Path.cwd(),
    # Backward-compatible AI Studio mount locations.
    Path("/home/aistudio/data/datasets"),
    Path("/home/aistudio/data"),
    Path("/data"),
    Path("/home/aistudio/tanker_work"),
    Path("/home/aistudio"),
]

TARGET_FILENAME = "06_final_fixed31_with_coordinates.csv"
MANIFEST_FILENAME = "era5_request_manifest.csv"

OUTPUT_ROOT = Path(os.environ.get(
    "SHIP_FUEL_TANKER_HARMONIZED_OUTPUT_DIR",
    str(REPO_ROOT / "revision_runs" / "era5_tanker"),
))
OUTPUT_CSV = OUTPUT_ROOT / "tanker31_cruise_with_era5_7fields.csv"
AUDIT_JSON = OUTPUT_ROOT / "tanker31_cruise_with_era5_7fields_audit.json"
MONTHLY_AUDIT_CSV = OUTPUT_ROOT / "tanker31_cruise_era5_coverage_by_month.csv"
UNROUTED_CSV = OUTPUT_ROOT / "tanker31_cruise_unrouted_rows.csv"

WORK_DIR = OUTPUT_ROOT / "_part_checkpoints"

SHIP_TYPE_KEYWORD = "tanker"
VOYAGE_PHASE = "cruise"

TIME_COL = "timestamp_utc"
LAT_COL = "latitude_deg"
LON_COL = "longitude_deg"

KEY_COLUMNS = [
    "ship_type",
    "pseudo_ship_group_id",
    "trajectory_segment_id",
    TIME_COL,
    "voyage_phase",
    LAT_COL,
    LON_COL,
]

# Must be identical to the adaptive downloader planning grid.
TILE_LAT_DEG = 10.0
TILE_LON_DEG = 10.0

MAX_TIME_GAP_HOURS = 2.0
MAX_CACHED_TIME_SLICES = 2

CALM_WIND_THRESHOLD_MS = 0.01
MS_TO_KNOT = 1.9438444924406048
KELVIN_TO_CELSIUS = 273.15

# False = allow testing with a partially uploaded ERA5 dataset.
# Missing parts remain NaN and are reported.
# For final production run, True is recommended after all ERA5 files are uploaded.
REQUIRE_ALL_ERA5_FILES = False

USE_CHECKPOINTS = True

CODE_BUILD = "TANKER_FIXED31_AISTUDIO_ADAPTIVE_7FIELDS_V2_2026-09-28"


# Reduce HDF5 / netCDF cache growth.
if set_chunk_cache is not None:
    try:
        set_chunk_cache(
            size=8 * 1024 * 1024,
            nelems=1009,
            preemption=0.5,
        )
    except Exception:
        pass


# =============================================================================
# 2. ERA5 VARIABLES
# =============================================================================

SURFACE_VARIABLES: Dict[str, Sequence[str]] = {
    "u10": ("u10", "10m_u_component_of_wind"),
    "v10": ("v10", "10m_v_component_of_wind"),
    "sp": ("sp", "surface_pressure"),
    "sst": ("sst", "sea_surface_temperature"),
}

WAVE_VARIABLES: Dict[str, Sequence[str]] = {
    "swh": (
        "swh",
        "significant_height_of_combined_wind_waves_and_swell",
    ),
    "mwd": ("mwd", "mean_wave_direction"),
    "mwp": ("mwp", "mean_wave_period"),
}

TIME_COORD_CANDIDATES = (
    "valid_time",
    "time",
    "forecast_time",
    "date",
)

LAT_COORD_CANDIDATES = ("latitude", "lat")
LON_COORD_CANDIDATES = ("longitude", "lon")

FINAL_FIELDS = (
    "wind_s",
    "wind_d",
    "wave_h",
    "wave_d",
    "wave_p",
    "surface_t",
    "surface_p",
)


# =============================================================================
# 3. GENERAL HELPERS
# =============================================================================

def normalize_name(value: object) -> str:
    return re.sub(
        r"[^0-9a-z]+",
        "",
        str(value).strip().lower(),
    )


def choose_column(
    dataframe: pd.DataFrame,
    candidates: Sequence[str],
    required: bool = True,
) -> Optional[str]:
    mapping = {
        normalize_name(c): c
        for c in dataframe.columns
    }

    for candidate in candidates:
        key = normalize_name(candidate)
        if key in mapping:
            return mapping[key]

    if required:
        raise KeyError(
            f"Cannot find any of {list(candidates)}.\n"
            f"Available columns: {list(dataframe.columns)}"
        )

    return None


def search_for_filename(filename: str) -> List[Path]:
    matches = []
    seen = set()

    for root in SEARCH_ROOTS:
        if not root.exists():
            continue

        try:
            for path in root.rglob(filename):
                try:
                    resolved = path.resolve()
                except Exception:
                    resolved = path

                key = str(resolved)

                if key not in seen:
                    matches.append(path)
                    seen.add(key)
        except Exception:
            # Some mounted locations may reject recursive traversal.
            continue

    return matches


def discover_unique_file(filename: str) -> Path:
    matches = search_for_filename(filename)

    if not matches:
        raise FileNotFoundError(
            f"Cannot find '{filename}'.\n"
            f"Searched roots:\n  "
            + "\n  ".join(str(p) for p in SEARCH_ROOTS)
        )

    # Prefer writable / extracted tanker_work copy if available.
    matches = sorted(
        matches,
        key=lambda p: (
            0 if "/tanker_work/" in str(p).replace("\\", "/") else 1,
            len(str(p)),
            str(p),
        ),
    )

    if len(matches) > 1:
        print(
            f"WARNING: found {len(matches)} copies of {filename}:"
        )
        for item in matches:
            print("  ", item)
        print("Using:", matches[0])

    return matches[0]


def discover_paths() -> Tuple[Path, Path]:
    if TARGET_CSV_OVERRIDE:
        target_csv = Path(TARGET_CSV_OVERRIDE)
    else:
        target_csv = discover_unique_file(
            TARGET_FILENAME
        )

    if ERA5_ROOT_OVERRIDE:
        era5_root = Path(ERA5_ROOT_OVERRIDE)
    else:
        manifest_path = discover_unique_file(
            MANIFEST_FILENAME
        )
        era5_root = manifest_path.parent

    return target_csv, era5_root


def parse_utc(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(
        series,
        errors="coerce",
        utc=True,
    )

    return (
        parsed
        .dt.tz_convert("UTC")
        .dt.tz_localize(None)
    )


def normalize_lon_180(values):
    arr = np.asarray(values, dtype=np.float64)
    return ((arr + 180.0) % 360.0) - 180.0


def tile_indices(
    latitude: float,
    longitude: float,
) -> Tuple[int, int]:
    lat = float(latitude)
    lon = float(normalize_lon_180([longitude])[0])

    if lat >= 90.0:
        lat = np.nextafter(90.0, -np.inf)

    if lon >= 180.0:
        lon = np.nextafter(180.0, -np.inf)

    lat_i = int(
        math.floor(
            (lat + 90.0) / TILE_LAT_DEG
        )
    )

    lon_i = int(
        math.floor(
            (lon + 180.0) / TILE_LON_DEG
        )
    )

    return lat_i, lon_i


def parse_tile_cells(value) -> set:
    if pd.isna(value):
        return set()

    text = str(value).strip()

    if not text:
        return set()

    # Preferred representation: Python list/tuple literal.
    try:
        obj = ast.literal_eval(text)
        result = set()

        if isinstance(obj, dict):
            obj = list(obj.keys())

        if isinstance(obj, (list, tuple, set)):
            for item in obj:
                if (
                    isinstance(item, (list, tuple))
                    and len(item) >= 2
                ):
                    result.add(
                        (int(item[0]), int(item[1]))
                    )

                elif isinstance(item, str):
                    numbers = re.findall(
                        r"-?\d+",
                        item,
                    )

                    if len(numbers) >= 2:
                        result.add(
                            (
                                int(numbers[0]),
                                int(numbers[1]),
                            )
                        )

        if result:
            return result

    except Exception:
        pass

    # Fallback: handles "3:12;3:13", "(3,12),(3,13)", etc.
    result = set()

    for pair in re.findall(
        r"(-?\d+)\s*[,:\s]\s*(-?\d+)",
        text,
    ):
        result.add(
            (int(pair[0]), int(pair[1]))
        )

    return result


def lon_inside(
    longitude: float,
    west: float,
    east: float,
) -> bool:
    lon = float(
        normalize_lon_180([longitude])[0]
    )
    west = float(
        normalize_lon_180([west])[0]
    )
    east = float(
        normalize_lon_180([east])[0]
    )

    if west <= east:
        return (
            west - 1e-9
            <= lon
            <= east + 1e-9
        )

    # Dateline-crossing interval.
    return (
        lon >= west - 1e-9
        or lon <= east + 1e-9
    )


# =============================================================================
# 4. MANIFEST
# =============================================================================

@dataclass
class ManifestColumns:
    plan_year: str
    plan_month: str
    part: str

    request_year: Optional[str]
    request_month: Optional[str]

    tile_cells: Optional[str]

    core_north: Optional[str]
    core_south: Optional[str]
    core_west: Optional[str]
    core_east: Optional[str]

    surface_file: Optional[str]
    wave_file: Optional[str]


def detect_manifest_columns(
    manifest: pd.DataFrame,
) -> ManifestColumns:

    return ManifestColumns(
        plan_year=choose_column(
            manifest,
            ("plan_year", "year"),
        ),

        plan_month=choose_column(
            manifest,
            ("plan_month", "month"),
        ),

        part=choose_column(
            manifest,
            ("part", "part_number", "part_no"),
        ),

        request_year=choose_column(
            manifest,
            (
                "request_year",
                "req_year",
                "download_year",
            ),
            required=False,
        ),

        request_month=choose_column(
            manifest,
            (
                "request_month",
                "req_month",
                "download_month",
            ),
            required=False,
        ),

        tile_cells=choose_column(
            manifest,
            (
                "tile_cells",
                "tiles",
                "planning_tiles",
            ),
            required=False,
        ),

        core_north=choose_column(
            manifest,
            (
                "core_north",
                "north_core",
                "core_lat_max",
            ),
            required=False,
        ),

        core_south=choose_column(
            manifest,
            (
                "core_south",
                "south_core",
                "core_lat_min",
            ),
            required=False,
        ),

        core_west=choose_column(
            manifest,
            (
                "core_west",
                "west_core",
                "core_lon_min",
            ),
            required=False,
        ),

        core_east=choose_column(
            manifest,
            (
                "core_east",
                "east_core",
                "core_lon_max",
            ),
            required=False,
        ),

        surface_file=choose_column(
            manifest,
            (
                "surface_file",
                "surface_path",
                "surface_target",
                "surface_filename",
            ),
            required=False,
        ),

        wave_file=choose_column(
            manifest,
            (
                "wave_file",
                "wave_path",
                "wave_target",
                "wave_filename",
            ),
            required=False,
        ),
    )


def part_table_from_manifest(
    manifest: pd.DataFrame,
    cols: ManifestColumns,
) -> pd.DataFrame:

    output = []

    grouping_columns = [
        cols.plan_year,
        cols.plan_month,
        cols.part,
    ]

    for (
        plan_year,
        plan_month,
        part_number,
    ), sub in manifest.groupby(
        grouping_columns,
        sort=True,
    ):
        row = {
            "plan_year": int(plan_year),
            "plan_month": int(plan_month),
            "part": int(part_number),
            "tile_cells": set(),
            "core_north": np.nan,
            "core_south": np.nan,
            "core_west": np.nan,
            "core_east": np.nan,
        }

        if cols.tile_cells:
            for value in sub[cols.tile_cells]:
                row["tile_cells"].update(
                    parse_tile_cells(value)
                )

        for output_name, column_name in (
            ("core_north", cols.core_north),
            ("core_south", cols.core_south),
            ("core_west", cols.core_west),
            ("core_east", cols.core_east),
        ):
            if column_name:
                values = pd.to_numeric(
                    sub[column_name],
                    errors="coerce",
                ).dropna()

                if len(values):
                    row[output_name] = float(
                        values.iloc[0]
                    )

        output.append(row)

    return pd.DataFrame(output)


def build_tile_route(
    part_table: pd.DataFrame,
) -> Dict[Tuple[int, int, int, int], int]:

    route = {}

    for _, row in part_table.iterrows():
        plan_year = int(row["plan_year"])
        plan_month = int(row["plan_month"])
        part = int(row["part"])

        for lat_i, lon_i in row["tile_cells"]:
            key = (
                plan_year,
                plan_month,
                int(lat_i),
                int(lon_i),
            )

            if (
                key in route
                and route[key] != part
            ):
                raise RuntimeError(
                    "The same planning tile belongs "
                    f"to multiple parts: {key}; "
                    f"{route[key]} vs {part}"
                )

            route[key] = part

    return route


def route_by_core_bounds(
    part_table: pd.DataFrame,
    year: int,
    month: int,
    latitude: float,
    longitude: float,
) -> Optional[int]:

    candidates = part_table[
        (
            part_table["plan_year"]
            == int(year)
        )
        & (
            part_table["plan_month"]
            == int(month)
        )
    ]

    hits = []

    for _, row in candidates.iterrows():
        bounds = [
            row["core_north"],
            row["core_south"],
            row["core_west"],
            row["core_east"],
        ]

        if not all(
            np.isfinite(bounds)
        ):
            continue

        if (
            float(row["core_south"]) - 1e-9
            <= latitude
            <= float(row["core_north"]) + 1e-9
            and lon_inside(
                longitude,
                float(row["core_west"]),
                float(row["core_east"]),
            )
        ):
            hits.append(
                int(row["part"])
            )

    if len(hits) == 1:
        return hits[0]

    return None


def route_observations(
    target: pd.DataFrame,
    part_table: pd.DataFrame,
    tile_route: Dict[
        Tuple[int, int, int, int],
        int,
    ],
) -> pd.Series:

    result = np.full(
        len(target),
        np.nan,
        dtype=np.float64,
    )

    times = target["_time"].to_numpy()
    lats = target[LAT_COL].to_numpy(
        dtype=np.float64
    )
    lons = target[LON_COL].to_numpy(
        dtype=np.float64
    )

    for i, (
        timestamp,
        latitude,
        longitude,
    ) in enumerate(
        zip(times, lats, lons)
    ):
        if (
            pd.isna(timestamp)
            or not np.isfinite(latitude)
            or not np.isfinite(longitude)
        ):
            continue

        timestamp = pd.Timestamp(
            timestamp
        )

        lat_i, lon_i = tile_indices(
            latitude,
            longitude,
        )

        part = tile_route.get(
            (
                timestamp.year,
                timestamp.month,
                lat_i,
                lon_i,
            )
        )

        # Fallback only to the exact core box from the manifest,
        # NOT nearest-neighbour ERA5 interpolation.
        if part is None:
            part = route_by_core_bounds(
                part_table,
                timestamp.year,
                timestamp.month,
                latitude,
                longitude,
            )

        if part is not None:
            result[i] = int(part)

    return pd.Series(
        result,
        index=target.index,
        dtype="Float64",
    )


# =============================================================================
# 5. ERA5 FILE RESOLUTION
# =============================================================================

def resolve_manifest_path(
    value,
    era5_root: Path,
) -> Optional[Path]:

    if value is None or pd.isna(value):
        return None

    text = str(value).strip()

    if not text:
        return None

    # Windows path saved inside the manifest.
    if re.match(
        r"^[A-Za-z]:[\\/]",
        text,
    ):
        filename = Path(
            text.replace("\\", "/")
        ).name

        matches = list(
            era5_root.rglob(filename)
        )

        return (
            matches[0]
            if matches
            else None
        )

    path = Path(text)

    if path.is_absolute():
        return (
            path
            if path.exists()
            else None
        )

    direct = era5_root / path

    if direct.exists():
        return direct

    matches = list(
        era5_root.rglob(path.name)
    )

    return (
        matches[0]
        if matches
        else None
    )


def constructed_file_path(
    era5_root: Path,
    kind: str,
    plan_year: int,
    plan_month: int,
    part: int,
    request_year: int,
    request_month: int,
) -> Path:

    folder = era5_root / kind

    base = (
        f"era5_{kind}_"
        f"{plan_year:04d}_{plan_month:02d}_"
        f"part{part}"
    )

    if (
        request_year == plan_year
        and request_month == plan_month
    ):
        filename = base + ".nc"

    else:
        filename = (
            base
            + f"_pad_"
            + f"{request_year:04d}_"
            + f"{request_month:02d}.nc"
        )

    return folder / filename


def files_for_part(
    manifest: pd.DataFrame,
    cols: ManifestColumns,
    era5_root: Path,
    plan_year: int,
    plan_month: int,
    part: int,
    kind: str,
) -> List[Path]:

    mask = (
        (
            pd.to_numeric(
                manifest[cols.plan_year],
                errors="coerce",
            )
            == plan_year
        )
        & (
            pd.to_numeric(
                manifest[cols.plan_month],
                errors="coerce",
            )
            == plan_month
        )
        & (
            pd.to_numeric(
                manifest[cols.part],
                errors="coerce",
            )
            == part
        )
    )

    subset = manifest.loc[mask].copy()

    if subset.empty:
        return []

    path_column = (
        cols.surface_file
        if kind == "surface"
        else cols.wave_file
    )

    output = []

    for _, row in subset.iterrows():
        path = None

        if path_column:
            path = resolve_manifest_path(
                row[path_column],
                era5_root,
            )

        if path is None:
            request_year = (
                int(row[cols.request_year])
                if cols.request_year
                else plan_year
            )

            request_month = (
                int(row[cols.request_month])
                if cols.request_month
                else plan_month
            )

            path = constructed_file_path(
                era5_root,
                kind,
                plan_year,
                plan_month,
                part,
                request_year,
                request_month,
            )

        output.append(path)

    # Stable unique paths.
    unique = []
    seen = set()

    for path in output:
        key = str(path)

        if key not in seen:
            unique.append(path)
            seen.add(key)

    return unique


# =============================================================================
# 6. NETCDF INSPECTION / LOCAL COLLECTION
# =============================================================================

def find_nc_name(
    nc: Dataset,
    candidates: Sequence[str],
) -> str:

    for name in candidates:
        if name in nc.variables:
            return name

    normalized = {
        normalize_name(name): name
        for name in nc.variables
    }

    for candidate in candidates:
        key = normalize_name(candidate)

        if key in normalized:
            return normalized[key]

    raise KeyError(
        f"Cannot find NetCDF variable from "
        f"{list(candidates)}.\n"
        f"Available: {list(nc.variables.keys())}"
    )


def nc_times_to_ns(
    variable,
) -> np.ndarray:

    units = getattr(
        variable,
        "units",
        None,
    )

    if units is None:
        raise ValueError(
            f"Time variable {variable.name} "
            "has no units attribute."
        )

    calendar_name = getattr(
        variable,
        "calendar",
        "standard",
    )

    converted = num2date(
        variable[:],
        units=units,
        calendar=calendar_name,
        only_use_cftime_datetimes=False,
        only_use_python_datetimes=False,
    )

    output = []

    for value in np.asarray(
        converted
    ).ravel():
        try:
            timestamp = pd.Timestamp(
                value
            )
        except Exception:
            timestamp = pd.Timestamp(
                value.strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            )

        if timestamp.tzinfo is not None:
            timestamp = (
                timestamp
                .tz_convert("UTC")
                .tz_localize(None)
            )

        output.append(
            timestamp.value
        )

    return np.asarray(
        output,
        dtype=np.int64,
    )


@dataclass
class NCMeta:
    path: Path
    time_name: str
    lat_name: str
    lon_name: str

    variable_names: Dict[str, str]

    times_ns: np.ndarray
    latitude: np.ndarray
    longitude: np.ndarray


def inspect_nc(
    path: Path,
    variable_candidates: Dict[
        str,
        Sequence[str],
    ],
) -> NCMeta:

    with Dataset(
        str(path),
        mode="r",
    ) as nc:

        time_name = find_nc_name(
            nc,
            TIME_COORD_CANDIDATES,
        )

        lat_name = find_nc_name(
            nc,
            LAT_COORD_CANDIDATES,
        )

        lon_name = find_nc_name(
            nc,
            LON_COORD_CANDIDATES,
        )

        actual_variables = {
            logical_name: find_nc_name(
                nc,
                aliases,
            )
            for (
                logical_name,
                aliases,
            ) in variable_candidates.items()
        }

        times_ns = nc_times_to_ns(
            nc.variables[time_name]
        )

        latitude = np.asarray(
            nc.variables[lat_name][:],
            dtype=np.float64,
        )

        longitude = np.asarray(
            nc.variables[lon_name][:],
            dtype=np.float64,
        )

    return NCMeta(
        path=path,
        time_name=time_name,
        lat_name=lat_name,
        lon_name=lon_name,
        variable_names=actual_variables,
        times_ns=times_ns,
        latitude=latitude,
        longitude=longitude,
    )


class LocalERA5Collection:
    """
    ERA5 collection for ONE adaptive spatial part.

    Main and cross-month pad files of the same spatial part
    are combined into one strictly increasing local timeline.
    """

    def __init__(
        self,
        paths: Sequence[Path],
        variable_candidates: Dict[
            str,
            Sequence[str],
        ],
        label: str,
    ):
        self.requested_paths = [
            Path(p)
            for p in paths
        ]

        self.missing_paths = [
            p
            for p in self.requested_paths
            if not p.is_file()
        ]

        existing_paths = [
            p
            for p in self.requested_paths
            if p.is_file()
        ]

        if not existing_paths:
            raise FileNotFoundError(
                f"{label}: no existing ERA5 files."
            )

        self.label = label
        self.variable_candidates = (
            variable_candidates
        )

        self.files = [
            inspect_nc(
                path,
                variable_candidates,
            )
            for path in existing_paths
        ]

        reference_lat = (
            self.files[0].latitude
        )

        reference_lon = (
            self.files[0].longitude
        )

        for meta in self.files[1:]:
            if (
                meta.latitude.shape
                != reference_lat.shape
                or meta.longitude.shape
                != reference_lon.shape
                or not np.allclose(
                    meta.latitude,
                    reference_lat,
                    equal_nan=True,
                )
                or not np.allclose(
                    meta.longitude,
                    reference_lon,
                    equal_nan=True,
                )
            ):
                raise RuntimeError(
                    f"{label}: inconsistent spatial grids. "
                    f"Problem file: {meta.path.name}"
                )

        self.latitude = reference_lat
        self.longitude = reference_lon

        time_parts = []
        file_parts = []
        local_parts = []

        for file_id, meta in enumerate(
            self.files
        ):
            count = len(
                meta.times_ns
            )

            time_parts.append(
                meta.times_ns
            )

            file_parts.append(
                np.full(
                    count,
                    file_id,
                    dtype=np.int32,
                )
            )

            local_parts.append(
                np.arange(
                    count,
                    dtype=np.int32,
                )
            )

        all_times = np.concatenate(
            time_parts
        )

        all_files = np.concatenate(
            file_parts
        )

        all_locals = np.concatenate(
            local_parts
        )

        order = np.argsort(
            all_times,
            kind="stable",
        )

        all_times = all_times[order]
        all_files = all_files[order]
        all_locals = all_locals[order]

        unique_mask = np.r_[
            True,
            all_times[1:]
            != all_times[:-1],
        ]

        self.times_ns = (
            all_times[unique_mask]
        )

        self.file_ids = (
            all_files[unique_mask]
        )

        self.local_ids = (
            all_locals[unique_mask]
        )

        self.cache = OrderedDict()

    def read_time_slice(
        self,
        global_index: int,
    ) -> Dict[str, np.ndarray]:

        global_index = int(
            global_index
        )

        if global_index in self.cache:
            cached = self.cache.pop(
                global_index
            )

            self.cache[
                global_index
            ] = cached

            return cached

        file_id = int(
            self.file_ids[
                global_index
            ]
        )

        local_time_index = int(
            self.local_ids[
                global_index
            ]
        )

        meta = self.files[
            file_id
        ]

        result = {}

        with Dataset(
            str(meta.path),
            mode="r",
        ) as nc:

            for (
                logical_name,
                actual_name,
            ) in meta.variable_names.items():

                variable = nc.variables[
                    actual_name
                ]

                indexers = []
                remaining_dimensions = []

                for dim in variable.dimensions:
                    if dim == meta.time_name:
                        indexers.append(
                            local_time_index
                        )

                    elif dim in (
                        meta.lat_name,
                        meta.lon_name,
                    ):
                        indexers.append(
                            slice(None)
                        )

                        remaining_dimensions.append(
                            dim
                        )

                    elif dim.lower() == "expver":
                        indexers.append(
                            slice(None)
                        )

                        remaining_dimensions.append(
                            dim
                        )

                    elif len(
                        nc.dimensions[dim]
                    ) == 1:
                        indexers.append(0)

                    else:
                        raise RuntimeError(
                            f"{meta.path.name}: "
                            f"unsupported dimension "
                            f"{dim} in {actual_name}"
                        )

                raw = np.ma.asarray(
                    variable[
                        tuple(indexers)
                    ]
                )

                data = np.ma.filled(
                    raw,
                    np.nan,
                ).astype(
                    np.float64
                )

                lower_dimensions = [
                    d.lower()
                    for d in remaining_dimensions
                ]

                if (
                    "expver"
                    in lower_dimensions
                ):
                    axis = (
                        lower_dimensions
                        .index("expver")
                    )

                    moved = np.moveaxis(
                        data,
                        axis,
                        0,
                    )

                    collapsed = np.full(
                        moved.shape[1:],
                        np.nan,
                        dtype=np.float64,
                    )

                    for layer in moved:
                        fill_mask = (
                            ~np.isfinite(
                                collapsed
                            )
                            & np.isfinite(
                                layer
                            )
                        )

                        collapsed[
                            fill_mask
                        ] = layer[
                            fill_mask
                        ]

                    data = collapsed

                    remaining_dimensions.pop(
                        axis
                    )

                lat_axis = (
                    remaining_dimensions
                    .index(
                        meta.lat_name
                    )
                )

                lon_axis = (
                    remaining_dimensions
                    .index(
                        meta.lon_name
                    )
                )

                data = np.moveaxis(
                    data,
                    [lat_axis, lon_axis],
                    [0, 1],
                )

                data = np.squeeze(
                    data
                )

                expected_shape = (
                    len(meta.latitude),
                    len(meta.longitude),
                )

                if (
                    data.shape
                    != expected_shape
                ):
                    raise RuntimeError(
                        f"{meta.path.name}:"
                        f"{actual_name} spatial "
                        f"shape={data.shape}, "
                        f"expected={expected_shape}"
                    )

                result[
                    logical_name
                ] = np.asarray(
                    data,
                    dtype=np.float32,
                )

        self.cache[
            global_index
        ] = result

        while (
            len(self.cache)
            > MAX_CACHED_TIME_SLICES
        ):
            self.cache.popitem(
                last=False
            )

        return result

    def clear(self):
        self.cache.clear()
        gc.collect()


# =============================================================================
# 7. INTERPOLATION
# =============================================================================

def align_target_longitudes(
    longitude_axis: np.ndarray,
    target_longitude: np.ndarray,
) -> np.ndarray:

    axis = np.asarray(
        longitude_axis,
        dtype=np.float64,
    )

    target = np.asarray(
        target_longitude,
        dtype=np.float64,
    )

    finite_axis = axis[
        np.isfinite(axis)
    ]

    if not len(finite_axis):
        return target.copy()

    axis_min = float(
        finite_axis.min()
    )

    axis_max = float(
        finite_axis.max()
    )

    # ERA5 file may use 0..360.
    if (
        axis_min >= -1e-9
        and axis_max > 180.0
    ):
        return target % 360.0

    # Otherwise use -180..180.
    return normalize_lon_180(
        target
    )


def bracket_axis(
    coordinates: np.ndarray,
    points: np.ndarray,
):

    coordinates = np.asarray(
        coordinates,
        dtype=np.float64,
    )

    points = np.asarray(
        points,
        dtype=np.float64,
    )

    order = np.argsort(
        coordinates
    )

    sorted_coordinates = (
        coordinates[order]
    )

    if len(
        sorted_coordinates
    ) < 2:
        raise ValueError(
            "Coordinate axis requires "
            "at least two points."
        )

    valid = (
        np.isfinite(points)
        & (
            points
            >= sorted_coordinates[0]
            - 1e-10
        )
        & (
            points
            <= sorted_coordinates[-1]
            + 1e-10
        )
    )

    clipped = np.clip(
        points,
        sorted_coordinates[0],
        sorted_coordinates[-1],
    )

    position = np.searchsorted(
        sorted_coordinates,
        clipped,
        side="right",
    )

    position = np.clip(
        position,
        1,
        len(sorted_coordinates) - 1,
    )

    lower_sorted = (
        position - 1
    )

    upper_sorted = position

    x0 = sorted_coordinates[
        lower_sorted
    ]

    x1 = sorted_coordinates[
        upper_sorted
    ]

    denominator = x1 - x0

    weight = np.zeros(
        len(points),
        dtype=np.float64,
    )

    nonzero = (
        np.abs(denominator)
        > 1e-15
    )

    weight[nonzero] = (
        clipped[nonzero]
        - x0[nonzero]
    ) / denominator[nonzero]

    return (
        order[lower_sorted],
        order[upper_sorted],
        weight,
        valid,
    )


def time_brackets(
    timeline: np.ndarray,
    targets: np.ndarray,
):

    position = np.searchsorted(
        timeline,
        targets,
        side="left",
    )

    exact = np.zeros(
        len(targets),
        dtype=bool,
    )

    inside = (
        position
        < len(timeline)
    )

    exact[inside] = (
        timeline[
            position[inside]
        ]
        == targets[inside]
    )

    lower = (
        position - 1
    )

    upper = (
        position.copy()
    )

    lower[exact] = (
        position[exact]
    )

    upper[exact] = (
        position[exact]
    )

    valid = (
        (lower >= 0)
        & (
            upper
            < len(timeline)
        )
    )

    lower = np.clip(
        lower,
        0,
        len(timeline) - 1,
    )

    upper = np.clip(
        upper,
        0,
        len(timeline) - 1,
    )

    weight = np.zeros(
        len(targets),
        dtype=np.float64,
    )

    different = (
        valid
        & (
            lower
            != upper
        )
    )

    denominator = (
        timeline[
            upper[different]
        ]
        - timeline[
            lower[different]
        ]
    )

    weight[different] = (
        targets[different]
        - timeline[
            lower[different]
        ]
    ) / denominator

    max_gap_ns = int(
        MAX_TIME_GAP_HOURS
        * 3600
        * 1e9
    )

    gap = (
        timeline[upper]
        - timeline[lower]
    )

    valid &= (
        (~different)
        | (
            gap
            <= max_gap_ns
        )
    )

    return (
        lower,
        upper,
        weight,
        valid,
    )


def bilinear_strict(
    field: np.ndarray,
    lat0,
    lat1,
    lon0,
    lon1,
    wy,
    wx,
):

    q00 = field[
        lat0,
        lon0,
    ]

    q01 = field[
        lat0,
        lon1,
    ]

    q10 = field[
        lat1,
        lon0,
    ]

    q11 = field[
        lat1,
        lon1,
    ]

    return (
        (1.0 - wy)
        * (1.0 - wx)
        * q00
        + (1.0 - wy)
        * wx
        * q01
        + wy
        * (1.0 - wx)
        * q10
        + wy
        * wx
        * q11
    )


def interpolate_collection(
    collection: LocalERA5Collection,
    target_times_ns: np.ndarray,
    target_latitude: np.ndarray,
    target_longitude: np.ndarray,
    circular_variables: Sequence[str] = (),
) -> Dict[str, np.ndarray]:

    count = len(
        target_times_ns
    )

    aligned_longitude = (
        align_target_longitudes(
            collection.longitude,
            target_longitude,
        )
    )

    (
        lat0,
        lat1,
        wy,
        valid_lat,
    ) = bracket_axis(
        collection.latitude,
        target_latitude,
    )

    (
        lon0,
        lon1,
        wx,
        valid_lon,
    ) = bracket_axis(
        collection.longitude,
        aligned_longitude,
    )

    (
        t0,
        t1,
        wt,
        valid_time,
    ) = time_brackets(
        collection.times_ns,
        target_times_ns,
    )

    valid = (
        valid_lat
        & valid_lon
        & valid_time
    )

    output = {
        name: np.full(
            count,
            np.nan,
            dtype=np.float64,
        )
        for name in (
            collection
            .variable_candidates
        )
    }

    rows = np.flatnonzero(
        valid
    )

    if not len(rows):
        return output

    pair_key = (
        t0[rows].astype(
            np.int64
        )
        * (
            len(
                collection.times_ns
            )
            + 1
        )
        + t1[rows].astype(
            np.int64
        )
    )

    order = np.argsort(
        pair_key,
        kind="stable",
    )

    rows = rows[order]
    pair_key = pair_key[order]

    boundaries = np.flatnonzero(
        np.r_[
            True,
            pair_key[1:]
            != pair_key[:-1],
            True,
        ]
    )

    circular = set(
        circular_variables
    )

    for start, stop in zip(
        boundaries[:-1],
        boundaries[1:],
    ):
        r = rows[
            start:stop
        ]

        lower_index = int(
            t0[r[0]]
        )

        upper_index = int(
            t1[r[0]]
        )

        lower_fields = (
            collection
            .read_time_slice(
                lower_index
            )
        )

        if (
            lower_index
            == upper_index
        ):
            upper_fields = (
                lower_fields
            )

        else:
            upper_fields = (
                collection
                .read_time_slice(
                    upper_index
                )
            )

        for name in (
            collection
            .variable_candidates
        ):
            if name in circular:
                lower_rad = np.deg2rad(
                    lower_fields[name]
                )

                upper_rad = np.deg2rad(
                    upper_fields[name]
                )

                s0 = bilinear_strict(
                    np.sin(lower_rad),
                    lat0[r],
                    lat1[r],
                    lon0[r],
                    lon1[r],
                    wy[r],
                    wx[r],
                )

                c0 = bilinear_strict(
                    np.cos(lower_rad),
                    lat0[r],
                    lat1[r],
                    lon0[r],
                    lon1[r],
                    wy[r],
                    wx[r],
                )

                s1 = bilinear_strict(
                    np.sin(upper_rad),
                    lat0[r],
                    lat1[r],
                    lon0[r],
                    lon1[r],
                    wy[r],
                    wx[r],
                )

                c1 = bilinear_strict(
                    np.cos(upper_rad),
                    lat0[r],
                    lat1[r],
                    lon0[r],
                    lon1[r],
                    wy[r],
                    wx[r],
                )

                sine_value = (
                    (1.0 - wt[r])
                    * s0
                    + wt[r]
                    * s1
                )

                cosine_value = (
                    (1.0 - wt[r])
                    * c0
                    + wt[r]
                    * c1
                )

                angle = (
                    np.degrees(
                        np.arctan2(
                            sine_value,
                            cosine_value,
                        )
                    )
                    + 360.0
                ) % 360.0

                angle[
                    np.hypot(
                        sine_value,
                        cosine_value,
                    )
                    < 1e-12
                ] = np.nan

                output[
                    name
                ][r] = angle

            else:
                lower_value = (
                    bilinear_strict(
                        lower_fields[name],
                        lat0[r],
                        lat1[r],
                        lon0[r],
                        lon1[r],
                        wy[r],
                        wx[r],
                    )
                )

                upper_value = (
                    bilinear_strict(
                        upper_fields[name],
                        lat0[r],
                        lat1[r],
                        lon0[r],
                        lon1[r],
                        wy[r],
                        wx[r],
                    )
                )

                output[
                    name
                ][r] = (
                    (1.0 - wt[r])
                    * lower_value
                    + wt[r]
                    * upper_value
                )

    return output


# =============================================================================
# 8. TARGET DATA
# =============================================================================

def load_target(
    target_csv: Path,
) -> pd.DataFrame:

    header = (
        pd.read_csv(
            target_csv,
            nrows=0,
        )
        .columns
        .tolist()
    )

    missing = [
        column
        for column in KEY_COLUMNS
        if column not in header
    ]

    if missing:
        raise KeyError(
            f"Target CSV is missing: {missing}\n"
            f"Available columns: {header}"
        )

    target = pd.read_csv(
        target_csv,
        usecols=KEY_COLUMNS,
        low_memory=False,
    )

    total_rows = len(
        target
    )

    target = target[
        target[
            "ship_type"
        ]
        .astype(str)
        .str.contains(
            SHIP_TYPE_KEYWORD,
            case=False,
            na=False,
        )
    ].copy()

    target = target[
        target[
            "voyage_phase"
        ]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq(
            VOYAGE_PHASE.lower()
        )
    ].copy()

    target["_time"] = parse_utc(
        target[TIME_COL]
    )

    target[LAT_COL] = pd.to_numeric(
        target[LAT_COL],
        errors="coerce",
    )

    target[LON_COL] = pd.to_numeric(
        target[LON_COL],
        errors="coerce",
    )

    valid = (
        target["_time"].notna()
        & target[LAT_COL].between(
            -90.0,
            90.0,
        )
        & target[LON_COL].between(
            -360.0,
            360.0,
        )
    )

    invalid_count = int(
        (~valid).sum()
    )

    target = target.loc[
        valid
    ].copy()

    # Keep original row number in source file.
    target["_source_row"] = (
        target.index.astype(
            np.int64
        )
    )

    target = target.reset_index(
        drop=True
    )

    print(
        f"Source rows before filter : "
        f"{total_rows:,}"
    )

    print(
        f"Valid tanker+cruise rows  : "
        f"{len(target):,}"
    )

    print(
        f"Invalid time/position     : "
        f"{invalid_count:,}"
    )

    return target


# =============================================================================
# 9. PER-PART PROCESSING
# =============================================================================

def empty_output(
    rows: pd.DataFrame,
) -> pd.DataFrame:

    output = rows[
        KEY_COLUMNS
        + ["_source_row"]
    ].copy()

    for field in FINAL_FIELDS:
        output[field] = np.nan

    output["era5_plan_year"] = (
        rows["_time"]
        .dt.year
        .to_numpy()
    )

    output["era5_plan_month"] = (
        rows["_time"]
        .dt.month
        .to_numpy()
    )

    output["era5_part"] = (
        pd.to_numeric(
            rows["_part"],
            errors="coerce",
        )
    )

    output["era5_status"] = (
        "not_processed"
    )

    return output


def process_part(
    rows: pd.DataFrame,
    manifest: pd.DataFrame,
    manifest_cols: ManifestColumns,
    era5_root: Path,
    plan_year: int,
    plan_month: int,
    part: int,
) -> pd.DataFrame:

    checkpoint_path = (
        WORK_DIR
        / (
            f"part_"
            f"{plan_year:04d}_"
            f"{plan_month:02d}_"
            f"{part:03d}.csv"
        )
    )

    if (
        USE_CHECKPOINTS
        and checkpoint_path.is_file()
    ):
        print(
            "  checkpoint found -> skip"
        )

        return pd.read_csv(
            checkpoint_path
        )

    output = empty_output(
        rows
    )

    surface_paths = files_for_part(
        manifest,
        manifest_cols,
        era5_root,
        plan_year,
        plan_month,
        part,
        "surface",
    )

    wave_paths = files_for_part(
        manifest,
        manifest_cols,
        era5_root,
        plan_year,
        plan_month,
        part,
        "wave",
    )

    missing_surface = [
        path
        for path in surface_paths
        if not path.is_file()
    ]

    missing_wave = [
        path
        for path in wave_paths
        if not path.is_file()
    ]

    files_complete = (
        bool(surface_paths)
        and bool(wave_paths)
        and not missing_surface
        and not missing_wave
    )

    if missing_surface or missing_wave:
        print(
            "  ERA5 files incomplete:"
        )

        print(
            f"    surface missing: "
            f"{len(missing_surface)}"
        )

        print(
            f"    wave missing   : "
            f"{len(missing_wave)}"
        )

        if REQUIRE_ALL_ERA5_FILES:
            missing_all = (
                missing_surface
                + missing_wave
            )

            raise FileNotFoundError(
                "Missing ERA5 files:\n  "
                + "\n  ".join(
                    str(p)
                    for p in missing_all
                )
            )

    target_times_ns = (
        rows["_time"]
        .astype(
            "datetime64[ns]"
        )
        .astype(
            np.int64
        )
        .to_numpy()
    )

    target_latitude = (
        rows[LAT_COL]
        .to_numpy(
            dtype=np.float64
        )
    )

    target_longitude = (
        rows[LON_COL]
        .to_numpy(
            dtype=np.float64
        )
    )

    surface_result = None
    wave_result = None

    # Surface collection.
    try:
        surface_collection = (
            LocalERA5Collection(
                surface_paths,
                SURFACE_VARIABLES,
                (
                    f"surface "
                    f"{plan_year:04d}-"
                    f"{plan_month:02d} "
                    f"part{part}"
                ),
            )
        )

        surface_result = (
            interpolate_collection(
                surface_collection,
                target_times_ns,
                target_latitude,
                target_longitude,
                circular_variables=(),
            )
        )

        surface_collection.clear()

        del surface_collection

    except Exception as exc:
        print(
            "  SURFACE WARNING: "
            f"{type(exc).__name__}: "
            f"{exc}"
        )

    # Wave collection.
    try:
        wave_collection = (
            LocalERA5Collection(
                wave_paths,
                WAVE_VARIABLES,
                (
                    f"wave "
                    f"{plan_year:04d}-"
                    f"{plan_month:02d} "
                    f"part{part}"
                ),
            )
        )

        wave_result = (
            interpolate_collection(
                wave_collection,
                target_times_ns,
                target_latitude,
                target_longitude,
                circular_variables=(
                    "mwd",
                ),
            )
        )

        wave_collection.clear()

        del wave_collection

    except Exception as exc:
        print(
            "  WAVE WARNING: "
            f"{type(exc).__name__}: "
            f"{exc}"
        )

    # Derived wind fields + surface variables.
    if surface_result is not None:
        u10 = surface_result[
            "u10"
        ]

        v10 = surface_result[
            "v10"
        ]

        wind_speed_ms = np.hypot(
            u10,
            v10,
        )

        output["wind_s"] = (
            wind_speed_ms
            * MS_TO_KNOT
        )

        # Meteorological FROM direction.
        wind_direction = (
            np.degrees(
                np.arctan2(
                    -u10,
                    -v10,
                )
            )
            + 360.0
        ) % 360.0

        wind_direction[
            ~np.isfinite(
                wind_speed_ms
            )
            | (
                wind_speed_ms
                < CALM_WIND_THRESHOLD_MS
            )
        ] = np.nan

        output[
            "wind_d"
        ] = wind_direction

        output[
            "surface_t"
        ] = (
            surface_result["sst"]
            - KELVIN_TO_CELSIUS
        )

        output[
            "surface_p"
        ] = surface_result["sp"]

    if wave_result is not None:
        output[
            "wave_h"
        ] = wave_result["swh"]

        output[
            "wave_d"
        ] = wave_result["mwd"]

        output[
            "wave_p"
        ] = wave_result["mwp"]

    complete_mask = (
        output[
            list(FINAL_FIELDS)
        ]
        .notna()
        .all(axis=1)
    )

    any_mask = (
        output[
            list(FINAL_FIELDS)
        ]
        .notna()
        .any(axis=1)
    )

    output.loc[
        complete_mask,
        "era5_status",
    ] = "complete_7fields"

    output.loc[
        (~complete_mask)
        & any_mask,
        "era5_status",
    ] = "partial_fields"

    output.loc[
        (~complete_mask)
        & (~any_mask),
        "era5_status",
    ] = "missing_era5"

    # Critical checkpoint rule:
    # save only when every manifest-required ERA5 file for the part exists.
    if (
        USE_CHECKPOINTS
        and files_complete
    ):
        output.to_csv(
            checkpoint_path,
            index=False,
            encoding="utf-8-sig",
        )

        print(
            "  checkpoint saved "
            "(all required ERA5 files present)"
        )

    elif USE_CHECKPOINTS:
        print(
            "  checkpoint NOT saved "
            "(ERA5 files incomplete; "
            "this part will be recalculated later)"
        )

    gc.collect()

    return output


# =============================================================================
# 10. REPORTING
# =============================================================================

def print_field_stats(
    dataframe: pd.DataFrame,
):

    print(
        "\nSeven-field non-missing statistics:"
    )

    print(
        "-" * 88
    )

    total = len(
        dataframe
    )

    for field in FINAL_FIELDS:
        count = int(
            dataframe[
                field
            ]
            .notna()
            .sum()
        )

        percent = (
            100.0
            * count
            / total
        )

        print(
            f"{field:10s}: "
            f"{count:8,d}/{total:,} "
            f"= {percent:6.2f}%"
        )

    complete = (
        dataframe[
            list(FINAL_FIELDS)
        ]
        .notna()
        .all(axis=1)
    )

    print(
        f"{'ALL 7':10s}: "
        f"{int(complete.sum()):8,d}/"
        f"{total:,} "
        f"= {100.0 * complete.mean():6.2f}%"
    )


# =============================================================================
# 11. MAIN
# =============================================================================

def main():

    try:
        sys.stdout.reconfigure(
            line_buffering=True
        )
    except Exception:
        pass

    start_clock = time.time()

    print(
        "=" * 96
    )

    print(
        "FIXED31 TANKER + CRUISE "
        "- ADAPTIVE ERA5 "
        "SEVEN-FIELD INTERPOLATION"
    )

    print(
        "=" * 96
    )

    print(
        "CODE BUILD             :",
        CODE_BUILD,
    )

    print(
        "Spatial interpolation  : bilinear"
    )

    print(
        "Temporal interpolation : linear"
    )

    print(
        "Wave direction         : circular sin/cos"
    )

    print(
        "Wind direction         : derived after u10/v10 interpolation"
    )

    print(
        "Sea temperature        : ERA5 SST only"
    )

    print(
        "2 m air temperature    : NEVER USED"
    )

    print(
        "Nearest neighbour      : NEVER USED"
    )

    print(
        "=" * 96
    )

    target_csv, era5_root = (
        discover_paths()
    )

    manifest_path = (
        era5_root
        / MANIFEST_FILENAME
    )

    print(
        "Target CSV :",
        target_csv,
    )

    print(
        "ERA5 root  :",
        era5_root,
    )

    print(
        "Manifest   :",
        manifest_path,
    )

    print(
        "Output CSV :",
        OUTPUT_CSV,
    )

    print(
        "Checkpoint :",
        WORK_DIR,
    )

    if not manifest_path.is_file():
        raise FileNotFoundError(
            manifest_path
        )

    WORK_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # 1. Target
    # -------------------------------------------------------------------------
    print(
        "\n[1/7] Reading fixed31 target data..."
    )

    target = load_target(
        target_csv
    )

    # -------------------------------------------------------------------------
    # 2. Manifest
    # -------------------------------------------------------------------------
    print(
        "\n[2/7] Reading adaptive request manifest..."
    )

    manifest = pd.read_csv(
        manifest_path,
        low_memory=False,
    )

    print(
        f"Manifest rows: "
        f"{len(manifest):,}"
    )

    print(
        "Manifest columns:"
    )

    print(
        list(
            manifest.columns
        )
    )

    manifest_cols = (
        detect_manifest_columns(
            manifest
        )
    )

    print(
        "\nDetected manifest fields:"
    )

    print(
        "  plan_year     :",
        manifest_cols.plan_year,
    )

    print(
        "  plan_month    :",
        manifest_cols.plan_month,
    )

    print(
        "  part          :",
        manifest_cols.part,
    )

    print(
        "  request_year  :",
        manifest_cols.request_year,
    )

    print(
        "  request_month :",
        manifest_cols.request_month,
    )

    print(
        "  tile_cells    :",
        manifest_cols.tile_cells,
    )

    print(
        "  surface_file  :",
        manifest_cols.surface_file,
    )

    print(
        "  wave_file     :",
        manifest_cols.wave_file,
    )

    part_table = (
        part_table_from_manifest(
            manifest,
            manifest_cols,
        )
    )

    tile_route = (
        build_tile_route(
            part_table
        )
    )

    print(
        f"Plan-month spatial parts : "
        f"{len(part_table):,}"
    )

    print(
        f"Planning-tile routes     : "
        f"{len(tile_route):,}"
    )

    # -------------------------------------------------------------------------
    # 3. Route observations
    # -------------------------------------------------------------------------
    print(
        "\n[3/7] Routing observations to ERA5 parts..."
    )

    target["_part"] = (
        route_observations(
            target,
            part_table,
            tile_route,
        )
    )

    routed_mask = (
        target["_part"]
        .notna()
    )

    routed_count = int(
        routed_mask.sum()
    )

    unrouted_count = (
        len(target)
        - routed_count
    )

    print(
        f"Routed rows   : "
        f"{routed_count:,}/"
        f"{len(target):,} "
        f"= "
        f"{100.0 * routed_mask.mean():.2f}%"
    )

    print(
        f"Unrouted rows : "
        f"{unrouted_count:,}"
    )

    if unrouted_count:
        target.loc[
            ~routed_mask,
            KEY_COLUMNS
            + ["_source_row"],
        ].to_csv(
            UNROUTED_CSV,
            index=False,
            encoding="utf-8-sig",
        )

        print(
            "Unrouted CSV :",
            UNROUTED_CSV,
        )

    # -------------------------------------------------------------------------
    # 4. Process groups
    # -------------------------------------------------------------------------
    print(
        "\n[4/7] Interpolating routed observations part-by-part..."
    )

    routed_target = (
        target.loc[
            routed_mask
        ]
        .copy()
    )

    routed_target["_part"] = (
        routed_target[
            "_part"
        ]
        .astype(int)
    )

    routed_target["_year"] = (
        routed_target[
            "_time"
        ]
        .dt.year
    )

    routed_target["_month"] = (
        routed_target[
            "_time"
        ]
        .dt.month
    )

    groups = list(
        routed_target.groupby(
            [
                "_year",
                "_month",
                "_part",
            ],
            sort=True,
        )
    )

    print(
        f"Groups to process: "
        f"{len(groups):,}"
    )

    result_parts = []

    for group_number, (
        (
            year,
            month,
            part,
        ),
        rows,
    ) in enumerate(
        groups,
        start=1,
    ):

        print(
            "\n"
            + "-" * 88
        )

        print(
            f"[{group_number}/"
            f"{len(groups)}] "
            f"{int(year):04d}-"
            f"{int(month):02d} "
            f"part{int(part)} "
            f"rows={len(rows):,}"
        )

        result = process_part(
            rows,
            manifest,
            manifest_cols,
            era5_root,
            int(year),
            int(month),
            int(part),
        )

        result_parts.append(
            result
        )

    # -------------------------------------------------------------------------
    # 5. Append unrouted rows
    # -------------------------------------------------------------------------
    print(
        "\n[5/7] Adding unrouted rows..."
    )

    if unrouted_count:
        unrouted_rows = (
            target.loc[
                ~routed_mask
            ]
            .copy()
        )

        unrouted_output = (
            empty_output(
                unrouted_rows
            )
        )

        unrouted_output[
            "era5_status"
        ] = "unrouted"

        result_parts.append(
            unrouted_output
        )

    # -------------------------------------------------------------------------
    # 6. Merge and audit
    # -------------------------------------------------------------------------
    print(
        "\n[6/7] Merging and auditing..."
    )

    result = pd.concat(
        result_parts,
        ignore_index=True,
    )

    result = (
        result
        .sort_values(
            "_source_row",
            kind="stable",
        )
        .reset_index(
            drop=True
        )
    )

    print_field_stats(
        result
    )

    result["_month_key"] = (
        pd.to_datetime(
            result[TIME_COL],
            errors="coerce",
            utc=True,
        )
        .dt.strftime(
            "%Y-%m"
        )
    )

    monthly_rows = []

    for (
        month_key,
        sub,
    ) in result.groupby(
        "_month_key",
        dropna=False,
        sort=True,
    ):

        complete = (
            sub[
                list(FINAL_FIELDS)
            ]
            .notna()
            .all(axis=1)
        )

        monthly_rows.append(
            {
                "month": month_key,
                "rows": len(sub),
                "complete_7fields": int(
                    complete.sum()
                ),
                "complete_7fields_pct": float(
                    100.0
                    * complete.mean()
                ),
                "unrouted": int(
                    (
                        sub[
                            "era5_status"
                        ]
                        == "unrouted"
                    ).sum()
                ),
                "partial_fields": int(
                    (
                        sub[
                            "era5_status"
                        ]
                        == "partial_fields"
                    ).sum()
                ),
                "missing_era5": int(
                    (
                        sub[
                            "era5_status"
                        ]
                        == "missing_era5"
                    ).sum()
                ),
            }
        )

    monthly_audit = (
        pd.DataFrame(
            monthly_rows
        )
    )

    print(
        "\nCoverage by month:"
    )

    print(
        monthly_audit.to_string(
            index=False,
            float_format=(
                lambda x: f"{x:.2f}"
            ),
        )
    )

    # -------------------------------------------------------------------------
    # 7. Save
    # -------------------------------------------------------------------------
    print(
        "\n[7/7] Writing outputs..."
    )

    output_columns = (
        KEY_COLUMNS
        + list(
            FINAL_FIELDS
        )
        + [
            "era5_plan_year",
            "era5_plan_month",
            "era5_part",
            "era5_status",
        ]
    )

    result[
        output_columns
    ].to_csv(
        OUTPUT_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    monthly_audit.to_csv(
        MONTHLY_AUDIT_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    complete = (
        result[
            list(FINAL_FIELDS)
        ]
        .notna()
        .all(axis=1)
    )

    audit = {
        "code_build": CODE_BUILD,
        "target_csv": str(
            target_csv
        ),
        "era5_root": str(
            era5_root
        ),
        "manifest": str(
            manifest_path
        ),
        "rows_tanker_cruise": int(
            len(result)
        ),
        "routed_rows": int(
            (
                result[
                    "era5_status"
                ]
                != "unrouted"
            ).sum()
        ),
        "unrouted_rows": int(
            (
                result[
                    "era5_status"
                ]
                == "unrouted"
            ).sum()
        ),
        "complete_7fields_rows": int(
            complete.sum()
        ),
        "complete_7fields_pct": float(
            100.0
            * complete.mean()
        ),
        "spatial_interpolation": (
            "strict_bilinear"
        ),
        "temporal_interpolation": (
            "linear"
        ),
        "max_time_gap_hours": (
            MAX_TIME_GAP_HOURS
        ),
        "nearest_neighbor_used": False,
        "wave_direction_interpolation": (
            "circular_sin_cos"
        ),
        "wind_direction_method": (
            "meteorological FROM direction "
            "derived after interpolation "
            "of u10 and v10"
        ),
        "surface_temperature_source": (
            "ERA5 sea_surface_temperature"
        ),
        "surface_temperature_fallback": None,
        "surface_temperature_output_unit": (
            "degC"
        ),
        "surface_pressure_output_unit": (
            "Pa"
        ),
        "wind_speed_output_unit": (
            "kn"
        ),
        "checkpoint_rule": (
            "saved only when every "
            "manifest-required surface and "
            "wave file for the part exists"
        ),
        "output_csv": str(
            OUTPUT_CSV
        ),
        "monthly_audit_csv": str(
            MONTHLY_AUDIT_CSV
        ),
        "unrouted_csv": str(
            UNROUTED_CSV
        ),
        "checkpoint_dir": str(
            WORK_DIR
        ),
    }

    AUDIT_JSON.write_text(
        json.dumps(
            audit,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\n"
        + "=" * 96
    )

    print(
        "FINISHED"
    )

    print(
        "=" * 96
    )

    print(
        "Output CSV    :",
        OUTPUT_CSV,
    )

    print(
        "Audit JSON    :",
        AUDIT_JSON,
    )

    print(
        "Monthly audit :",
        MONTHLY_AUDIT_CSV,
    )

    print(
        "Unrouted CSV  :",
        UNROUTED_CSV,
    )

    print(
        "Complete 7 fields:",
        f"{int(complete.sum()):,}/"
        f"{len(result):,}",
        f"= "
        f"{100.0 * complete.mean():.2f}%",
    )

    print(
        "Elapsed:",
        f"{(time.time() - start_clock) / 60.0:.2f} min",
    )

    print(
        "=" * 96
    )


if __name__ == "__main__":
    main()
