# -*- coding: utf-8 -*-
from __future__ import annotations

import gc
import json
import os
import re
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
from netCDF4 import Dataset, num2date

# ============================================================
# Paths
# ============================================================
# Repository-safe defaults. Override any location with the corresponding
# environment variable when running on AI Studio or another machine.
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get(
    "SHIP_FUEL_BULK_ERA5_ROOT",
    str(REPO_ROOT / "data" / "era5" / "bulk"),
))
INPUT_CSV = Path(os.environ.get(
    "SHIP_FUEL_BULK_INPUT_CSV",
    str(REPO_ROOT / "data" / "bulk_model_10min_features.csv"),
))
OUTPUT_CSV = Path(os.environ.get(
    "SHIP_FUEL_BULK_HARMONIZED_OUTPUT",
    str(REPO_ROOT / "revision_runs" / "era5_bulk" / "bulk_model_10min_features_harmonized_era5.csv"),
))
AUDIT_JSON = Path(os.environ.get(
    "SHIP_FUEL_BULK_HARMONIZED_AUDIT",
    str(REPO_ROOT / "revision_runs" / "era5_bulk" / "bulk_model_10min_features_harmonized_era5_audit.json"),
))

WIND_WAVE_EXTRACTED = DATA_ROOT / "wind_wave_extracted_ym"
PRESSURE_TEMP_DIR = DATA_ROOT / "pressure_temp"
SST_DIR = DATA_ROOT / "sst"

WORK_DIR = Path(os.environ.get(
    "SHIP_FUEL_BULK_HARMONIZED_WORK",
    str(REPO_ROOT / "revision_runs" / "era5_bulk" / "_chunks"),
))

# Conservative memory settings
CHUNK_ROWS = 10_000
MAX_CACHED_TIME_SLICES = 2
MAX_TIME_GAP_HOURS = 2.0

MS_TO_KNOT = 1.9438444924406048
KELVIN_TO_CELSIUS = 273.15

CODE_BUILD = "AISTUDIO_FINAL_2026-08-23_STRICT_SST_LOW_MEM"

# Expected study period
EXPECTED_MONTHS = [
    f"{year}_{month:02d}"
    for year, month in (
        [(2022, m) for m in range(3, 13)]
        + [(2023, m) for m in range(1, 8)]
    )
]

# ============================================================
# Column aliases
# ============================================================
TIME_CANDIDATES = ["timestamp_utc", "timestamp", "UTC时间(-)", "UTC时间", "time", "datetime"]
LAT_CANDIDATES = ["latitude_deg", "latitude", "lat", "纬度"]
LON_CANDIDATES = ["longitude_deg", "longitude", "lon", "经度"]

HEADING_CANDIDATES = ["heading_deg", "heading", "hdg", "艏向角(deg)"]
COURSE_CANDIDATES = ["course_deg", "course", "cog", "direct", "航向角(deg)"]
SPEED_CANDIDATES = ["speed_kn", "speed", "sog", "对地航速(kn)"]

# ============================================================
# Helpers
# ============================================================
def normalize_name(value: object) -> str:
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", str(value).strip().lower())


def find_column(columns, candidates, required=True):
    index = {normalize_name(c): c for c in columns}
    for candidate in candidates:
        key = normalize_name(candidate)
        if key in index:
            return index[key]
    if required:
        raise KeyError(f"Cannot find required column from {candidates}")
    return None


def find_nc_variable(nc: Dataset, candidates):
    for name in candidates:
        if name in nc.variables:
            return name
    lower_map = {name.lower(): name for name in nc.variables}
    for name in candidates:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    raise KeyError(f"Cannot find NetCDF variable from {candidates}")


def parse_month_from_parent(path: Path) -> str:
    parent = path.parent.name
    if re.fullmatch(r"\d{4}_\d{2}", parent):
        return parent
    raise ValueError(
        f"风浪解压目录必须使用 YYYY_MM 作为月份目录，当前文件：{path}\n"
        f"当前父目录：{parent}\n"
        "例如：data/era5/bulk/wind_wave_extracted_ym/2022_03/data_stream-oper_stepType-instant.nc"
    )


def discover_wind_wave_files():
    oper = {}
    wave = {}

    for path in WIND_WAVE_EXTRACTED.rglob("data_stream-oper_stepType-instant.nc"):
        month = parse_month_from_parent(path)
        if month in oper:
            raise RuntimeError(f"Duplicate oper file for {month}: {oper[month]} ; {path}")
        oper[month] = path

    for path in WIND_WAVE_EXTRACTED.rglob("data_stream-wave_stepType-instant.nc"):
        month = parse_month_from_parent(path)
        if month in wave:
            raise RuntimeError(f"Duplicate wave file for {month}: {wave[month]} ; {path}")
        wave[month] = path

    missing_oper = [m for m in EXPECTED_MONTHS if m not in oper]
    missing_wave = [m for m in EXPECTED_MONTHS if m not in wave]

    if missing_oper or missing_wave:
        raise FileNotFoundError(
            "风浪解压文件不完整。\n"
            f"缺少 oper: {missing_oper}\n"
            f"缺少 wave: {missing_wave}"
        )

    return [oper[m] for m in EXPECTED_MONTHS], [wave[m] for m in EXPECTED_MONTHS]


def discover_monthly_files(folder: Path, pattern_prefix: str):
    found = {}
    regex = re.compile(rf"{re.escape(pattern_prefix)}_(\d{{4}})_(\d{{2}})\.nc$", re.I)

    for path in folder.glob(f"{pattern_prefix}_*.nc"):
        match = regex.search(path.name)
        if match:
            month = f"{match.group(1)}_{match.group(2)}"
            found[month] = path

    return found


def nc_times_ns(nc: Dataset, time_name: str):
    var = nc.variables[time_name]
    values = num2date(
        var[:],
        units=var.units,
        calendar=getattr(var, "calendar", "standard"),
        only_use_cftime_datetimes=False,
    )

    result = []
    for value in np.asarray(values).ravel():
        try:
            ts = pd.Timestamp(value)
        except Exception:
            ts = pd.Timestamp(value.strftime("%Y-%m-%d %H:%M:%S"))
        if ts.tzinfo is not None:
            ts = ts.tz_convert("UTC").tz_localize(None)
        result.append(ts.value)

    return np.asarray(result, dtype=np.int64)


class ERA5Collection:
    def __init__(self, file_specs, variables, label):
        self.variables = variables
        self.label = label
        self.files = []
        self.cache = OrderedDict()

        if not file_specs:
            raise FileNotFoundError(f"No files found for {label}")

        for spec in file_specs:
            path = Path(spec["path"])
            month = spec["month"]

            with Dataset(path, "r") as nc:
                time_name = find_nc_variable(nc, ["valid_time", "time"])
                lat_name = find_nc_variable(nc, ["latitude", "lat"])
                lon_name = find_nc_variable(nc, ["longitude", "lon"])

                actual_variables = {
                    key: find_nc_variable(nc, aliases)
                    for key, aliases in variables.items()
                }

                times = nc_times_ns(nc, time_name)
                lat = np.asarray(nc.variables[lat_name][:], dtype=np.float64)
                lon = np.asarray(nc.variables[lon_name][:], dtype=np.float64)

            self.files.append({
                "month": month,
                "path": path,
                "time_name": time_name,
                "lat_name": lat_name,
                "lon_name": lon_name,
                "variables": actual_variables,
                "times": times,
                "lat": lat,
                "lon": lon,
            })

        self.lat = self.files[0]["lat"]
        self.lon = self.files[0]["lon"]

        for meta in self.files[1:]:
            if len(meta["lat"]) != len(self.lat) or len(meta["lon"]) != len(self.lon):
                raise RuntimeError(f"{label}: inconsistent grid dimensions")

        all_times = []
        all_file_ids = []
        all_local_ids = []

        for file_id, meta in enumerate(self.files):
            n = len(meta["times"])
            all_times.append(meta["times"])
            all_file_ids.append(np.full(n, file_id, dtype=np.int32))
            all_local_ids.append(np.arange(n, dtype=np.int32))

        timeline = np.concatenate(all_times)
        file_ids = np.concatenate(all_file_ids)
        local_ids = np.concatenate(all_local_ids)

        order = np.argsort(timeline, kind="stable")
        timeline = timeline[order]
        file_ids = file_ids[order]
        local_ids = local_ids[order]

        unique = np.r_[True, timeline[1:] != timeline[:-1]]

        self.times = timeline[unique]
        self.file_ids = file_ids[unique]
        self.local_ids = local_ids[unique]

    def read_slice(self, global_index: int):
        global_index = int(global_index)

        if global_index in self.cache:
            data = self.cache.pop(global_index)
            self.cache[global_index] = data
            return data

        file_id = int(self.file_ids[global_index])
        local_id = int(self.local_ids[global_index])
        meta = self.files[file_id]

        result = {}

        with Dataset(meta["path"], "r") as nc:
            for key, actual_name in meta["variables"].items():
                var = nc.variables[actual_name]
                indexers = []
                remaining_dims = []

                for dim in var.dimensions:
                    if dim == meta["time_name"]:
                        indexers.append(local_id)
                    elif dim in (meta["lat_name"], meta["lon_name"]):
                        indexers.append(slice(None))
                        remaining_dims.append(dim)
                    elif dim.lower() == "expver":
                        indexers.append(slice(None))
                        remaining_dims.append(dim)
                    elif len(nc.dimensions[dim]) == 1:
                        indexers.append(0)
                    else:
                        raise RuntimeError(
                            f"{meta['path'].name}: unsupported dimension {dim}"
                        )

                data = np.ma.filled(
                    np.ma.asarray(var[tuple(indexers)]),
                    np.nan,
                ).astype(np.float64)

                lower_dims = [d.lower() for d in remaining_dims]

                if "expver" in lower_dims:
                    axis = lower_dims.index("expver")
                    moved = np.moveaxis(data, axis, 0)
                    collapsed = np.full(moved.shape[1:], np.nan, dtype=np.float64)

                    for layer in moved:
                        mask = ~np.isfinite(collapsed) & np.isfinite(layer)
                        collapsed[mask] = layer[mask]

                    data = collapsed
                    remaining_dims.pop(axis)

                lat_axis = remaining_dims.index(meta["lat_name"])
                lon_axis = remaining_dims.index(meta["lon_name"])

                data = np.moveaxis(data, [lat_axis, lon_axis], [0, 1])
                result[key] = np.asarray(np.squeeze(data), dtype=np.float32)

        self.cache[global_index] = result

        while len(self.cache) > MAX_CACHED_TIME_SLICES:
            self.cache.popitem(last=False)

        return result

    def clear_cache(self):
        self.cache.clear()
        gc.collect()


def bracket_axis(axis, points):
    axis = np.asarray(axis, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64)

    order = np.argsort(axis)
    sorted_axis = axis[order]

    valid = (
        np.isfinite(points)
        & (points >= sorted_axis[0])
        & (points <= sorted_axis[-1])
    )

    clipped = np.clip(points, sorted_axis[0], sorted_axis[-1])
    pos = np.searchsorted(sorted_axis, clipped, side="right")
    pos = np.clip(pos, 1, len(sorted_axis) - 1)

    i0 = pos - 1
    i1 = pos

    x0 = sorted_axis[i0]
    x1 = sorted_axis[i1]

    weight = (clipped - x0) / (x1 - x0)

    return order[i0], order[i1], weight, valid


def time_brackets(timeline, targets):
    pos = np.searchsorted(timeline, targets, side="left")

    exact = np.zeros(len(targets), dtype=bool)
    inside = pos < len(timeline)
    exact[inside] = timeline[pos[inside]] == targets[inside]

    lower = pos - 1
    upper = pos.copy()

    lower[exact] = pos[exact]
    upper[exact] = pos[exact]

    valid = (lower >= 0) & (upper < len(timeline))

    lower = np.clip(lower, 0, len(timeline) - 1)
    upper = np.clip(upper, 0, len(timeline) - 1)

    weight = np.zeros(len(targets), dtype=np.float64)

    different = valid & (lower != upper)

    denominator = timeline[upper[different]] - timeline[lower[different]]
    weight[different] = (
        targets[different] - timeline[lower[different]]
    ) / denominator

    max_gap_ns = int(MAX_TIME_GAP_HOURS * 3600 * 1e9)
    gap = timeline[upper] - timeline[lower]
    valid &= (~different) | (gap <= max_gap_ns)

    return lower, upper, weight, valid


def bilinear(field, lat0, lat1, lon0, lon1, wy, wx):
    q00 = field[lat0, lon0]
    q01 = field[lat0, lon1]
    q10 = field[lat1, lon0]
    q11 = field[lat1, lon1]

    return (
        (1.0 - wy) * (1.0 - wx) * q00
        + (1.0 - wy) * wx * q01
        + wy * (1.0 - wx) * q10
        + wy * wx * q11
    )


def interpolate_collection(
    collection: ERA5Collection,
    target_time,
    target_lat,
    target_lon,
    circular=(),
):
    n = len(target_time)

    lat0, lat1, wy, valid_lat = bracket_axis(collection.lat, target_lat)
    lon0, lon1, wx, valid_lon = bracket_axis(collection.lon, target_lon)
    t0, t1, wt, valid_time = time_brackets(collection.times, target_time)

    valid = valid_lat & valid_lon & valid_time
    rows = np.where(valid)[0]

    result = {
        key: np.full(n, np.nan, dtype=np.float64)
        for key in collection.variables
    }

    if len(rows) == 0:
        return result

    pair_key = (
        t0[rows].astype(np.int64) * (len(collection.times) + 1)
        + t1[rows].astype(np.int64)
    )

    order = np.argsort(pair_key, kind="stable")
    rows = rows[order]
    pair_key = pair_key[order]

    boundaries = np.flatnonzero(
        np.r_[True, pair_key[1:] != pair_key[:-1], True]
    )

    circular = set(circular)

    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        r = rows[start:stop]

        lower_idx = int(t0[r[0]])
        upper_idx = int(t1[r[0]])

        lower_fields = collection.read_slice(lower_idx)
        upper_fields = (
            lower_fields
            if lower_idx == upper_idx
            else collection.read_slice(upper_idx)
        )

        for name in collection.variables:
            if name in circular:
                lower_rad = np.deg2rad(lower_fields[name])
                upper_rad = np.deg2rad(upper_fields[name])

                s0 = bilinear(
                    np.sin(lower_rad),
                    lat0[r], lat1[r], lon0[r], lon1[r],
                    wy[r], wx[r],
                )
                c0 = bilinear(
                    np.cos(lower_rad),
                    lat0[r], lat1[r], lon0[r], lon1[r],
                    wy[r], wx[r],
                )

                s1 = bilinear(
                    np.sin(upper_rad),
                    lat0[r], lat1[r], lon0[r], lon1[r],
                    wy[r], wx[r],
                )
                c1 = bilinear(
                    np.cos(upper_rad),
                    lat0[r], lat1[r], lon0[r], lon1[r],
                    wy[r], wx[r],
                )

                s = (1.0 - wt[r]) * s0 + wt[r] * s1
                c = (1.0 - wt[r]) * c0 + wt[r] * c1

                angle = (np.degrees(np.arctan2(s, c)) + 360.0) % 360.0
                angle[np.hypot(s, c) < 1e-12] = np.nan

                result[name][r] = angle

            else:
                v0 = bilinear(
                    lower_fields[name],
                    lat0[r], lat1[r], lon0[r], lon1[r],
                    wy[r], wx[r],
                )
                v1 = bilinear(
                    upper_fields[name],
                    lat0[r], lat1[r], lon0[r], lon1[r],
                    wy[r], wx[r],
                )

                result[name][r] = (
                    (1.0 - wt[r]) * v0 + wt[r] * v1
                )

    return result


def angle_difference_deg(direction_to, ship_direction):
    return (
        (direction_to - ship_direction + 180.0) % 360.0
    ) - 180.0


def make_relative_features(df):
    heading_col = find_column(df.columns, HEADING_CANDIDATES, required=False)
    course_col = find_column(df.columns, COURSE_CANDIDATES, required=False)
    speed_col = find_column(df.columns, SPEED_CANDIDATES, required=True)

    speed = pd.to_numeric(df[speed_col], errors="coerce").to_numpy(dtype=np.float64, copy=True)

    if heading_col is not None:
        heading = pd.to_numeric(df[heading_col], errors="coerce").to_numpy(dtype=np.float64, copy=True)
    else:
        heading = np.full(len(df), np.nan)

    if course_col is not None:
        course = pd.to_numeric(df[course_col], errors="coerce").to_numpy(dtype=np.float64, copy=True)
    else:
        course = np.full(len(df), np.nan)

    ship_direction = np.where(np.isfinite(heading), heading, course)
    ship_direction = ship_direction % 360.0

    # ERA5 wind_d and mwd are meteorological/oceanographic FROM directions.
    # Convert to vector TO direction before relative-angle calculation.
    wind_to = (df["wind_direction_deg"].to_numpy(np.float64) + 180.0) % 360.0
    wave_to = (df["wave_direction_deg"].to_numpy(np.float64) + 180.0) % 360.0

    rel_wind_angle = angle_difference_deg(wind_to, ship_direction)
    rel_wave_angle = angle_difference_deg(wave_to, ship_direction)

    rel_wind_rad = np.deg2rad(rel_wind_angle)
    rel_wave_rad = np.deg2rad(rel_wave_angle)

    wind_speed = df["wind_speed_kn"].to_numpy(np.float64)

    rel_wind_speed_sq = (
        wind_speed ** 2
        + speed ** 2
        - 2.0 * wind_speed * speed * np.cos(rel_wind_rad)
    )
    rel_wind_speed_sq = np.where(
        np.isfinite(rel_wind_speed_sq),
        np.maximum(rel_wind_speed_sq, 0.0),
        np.nan,
    )

    df["wind_vector_to_direction_deg"] = wind_to
    df["wave_vector_to_direction_deg"] = wave_to

    df["relative_wind_angle_deg"] = rel_wind_angle
    df["relative_wave_angle_deg"] = rel_wave_angle

    df["relative_wind_sin"] = np.sin(rel_wind_rad)
    df["relative_wind_cos"] = np.cos(rel_wind_rad)

    df["relative_wave_sin"] = np.sin(rel_wave_rad)
    df["relative_wave_cos"] = np.cos(rel_wave_rad)

    df["rel_wind_speed_kn"] = np.sqrt(rel_wind_speed_sq)

    df["rel_wind_speed_x_speed"] = df["rel_wind_speed_kn"] * speed
    df["wave_height_x_speed"] = df["wave_height_m"] * speed

    return df


# ============================================================
# File discovery
# ============================================================
oper_paths, wave_paths = discover_wind_wave_files()

pressure_map = discover_monthly_files(
    PRESSURE_TEMP_DIR,
    "era5_pres_temp",
)
sst_map = discover_monthly_files(
    SST_DIR,
    "era5_sst",
)

missing_pressure = [m for m in EXPECTED_MONTHS if m not in pressure_map]
if missing_pressure:
    raise FileNotFoundError(
        f"Missing pressure/temp files for: {missing_pressure}"
    )

oper_specs = [
    {"month": m, "path": p}
    for m, p in zip(EXPECTED_MONTHS, oper_paths)
]
wave_specs = [
    {"month": m, "path": p}
    for m, p in zip(EXPECTED_MONTHS, wave_paths)
]
pressure_specs = [
    {"month": m, "path": pressure_map[m]}
    for m in EXPECTED_MONTHS
]

# Temperature: SST preferred; t2m fallback by month
missing_sst = [m for m in EXPECTED_MONTHS if m not in sst_map]
if missing_sst:
    raise FileNotFoundError(
        f"Missing SST files for: {missing_sst}. "
        "Harmonized analysis uses SST only; no t2m fallback."
    )

sst_specs = [
    {"month": m, "path": sst_map[m]}
    for m in EXPECTED_MONTHS
]

print("=" * 78)
print(f"CODE BUILD             : {CODE_BUILD}")
print(f"DATA ROOT              : {DATA_ROOT}")
print(f"WORK DIR               : {WORK_DIR}")
print("AI Studio mode         : YES")
print(f"RUNNING FILE           : {Path(__file__).resolve()}")
print("READONLY FIX           : np.where, no in-place lat/lon assignment")
print("Bulk harmonized ERA5 interpolation")
print("Spatial interpolation : bilinear")
print("Temporal interpolation: linear")
print("Nearest-neighbor      : NO")
print(f"Input                  : {INPUT_CSV}")
print(f"Output                 : {OUTPUT_CSV}")
print(f"oper files             : {len(oper_specs)}")
print(f"wave files             : {len(wave_specs)}")
print(f"pressure files         : {len(pressure_specs)}")
print(f"SST files              : {len(sst_specs)}")
print("t2m fallback           : NO (strict SST only)")
print("=" * 78)

# ============================================================
# Collections
# ============================================================
wind = ERA5Collection(
    oper_specs,
    {
        "u10": ["u10", "10m_u_component_of_wind"],
        "v10": ["v10", "10m_v_component_of_wind"],
    },
    "wind",
)

wave = ERA5Collection(
    wave_specs,
    {
        "swh": ["swh", "significant_height_of_combined_wind_waves_and_swell"],
        "mwd": ["mwd", "mean_wave_direction"],
        "mwp": ["mwp", "mean_wave_period"],
    },
    "wave",
)

pressure = ERA5Collection(
    pressure_specs,
    {
        "sp": ["sp", "surface_pressure"],
    },
    "pressure",
)

sst = ERA5Collection(
    sst_specs,
    {
        "sst": ["sst", "sea_surface_temperature"],
    },
    "sst",
)

# ============================================================
# Input/output setup
# ============================================================
if not INPUT_CSV.is_file():
    raise FileNotFoundError(f"Input not found: {INPUT_CSV}")

WORK_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

header = pd.read_csv(INPUT_CSV, nrows=0).columns

time_col = find_column(header, TIME_CANDIDATES, required=True)
lat_col = find_column(header, LAT_CANDIDATES, required=True)
lon_col = find_column(header, LON_CANDIDATES, required=True)

print(f"time column: {time_col}")
print(f"lat column : {lat_col}")
print(f"lon column : {lon_col}")

chunk_files = []
audit_chunks = []

# ============================================================
# Process chunks
# ============================================================
for chunk_no, df in enumerate(
    pd.read_csv(INPUT_CSV, chunksize=CHUNK_ROWS, low_memory=True),
    start=1,
):
    chunk_path = WORK_DIR / f"chunk_{chunk_no:05d}.csv"
    chunk_audit_path = WORK_DIR / f"chunk_{chunk_no:05d}.json"

    if chunk_path.is_file() and chunk_audit_path.is_file():
        print(f"[skip] chunk {chunk_no} already completed")
        chunk_files.append(chunk_path)
        with chunk_audit_path.open("r", encoding="utf-8") as f:
            audit_chunks.append(json.load(f))
        continue

    print(f"[run ] chunk {chunk_no}: {len(df):,} rows")

    parsed_time = pd.to_datetime(
        df[time_col],
        errors="coerce",
        utc=True,
    ).dt.tz_localize(None)

    # Do not mutate pandas-backed arrays in place.
    # Some pandas/NumPy combinations can still expose a read-only array even
    # when to_numpy(copy=True) is requested. np.asarray + np.where creates
    # independent writable result arrays without any in-place assignment.
    latitude_raw = np.asarray(
        pd.to_numeric(df[lat_col], errors="coerce"),
        dtype=np.float64,
    )
    longitude_raw = np.asarray(
        pd.to_numeric(df[lon_col], errors="coerce"),
        dtype=np.float64,
    )

    latitude = np.where(
        np.isfinite(latitude_raw)
        & (latitude_raw >= -90.0)
        & (latitude_raw <= 90.0),
        latitude_raw,
        np.nan,
    ).astype(np.float64, copy=True)

    longitude = np.where(
        np.isfinite(longitude_raw)
        & (longitude_raw >= -180.0)
        & (longitude_raw <= 180.0),
        longitude_raw,
        np.nan,
    ).astype(np.float64, copy=True)

    target_time = np.full(
        len(df),
        np.iinfo(np.int64).min,
        dtype=np.int64,
    )

    valid_time = parsed_time.notna().to_numpy()
    target_time[valid_time] = (
        parsed_time[valid_time]
        .astype("datetime64[ns]")
        .astype(np.int64)
    )

    # Wind
    wind_values = interpolate_collection(
        wind,
        target_time,
        latitude,
        longitude,
    )

    u10 = wind_values["u10"]
    v10 = wind_values["v10"]

    wind_speed_ms = np.hypot(u10, v10)
    wind_speed_kn = wind_speed_ms * MS_TO_KNOT

    wind_direction = (
        np.degrees(np.arctan2(-u10, -v10))
        + 360.0
    ) % 360.0

    wind_direction[
        ~np.isfinite(wind_speed_ms)
        | (wind_speed_ms < 0.01)
    ] = np.nan

    # Wave
    wave_values = interpolate_collection(
        wave,
        target_time,
        latitude,
        longitude,
        circular=("mwd",),
    )

    # Surface pressure
    pressure_values = interpolate_collection(
        pressure,
        target_time,
        latitude,
        longitude,
    )

    # Strict SST only: no 2 m air-temperature fallback
    sst_values = interpolate_collection(
        sst,
        target_time,
        latitude,
        longitude,
    )

    surface_temperature_c = (
        sst_values["sst"] - KELVIN_TO_CELSIUS
    )
    temperature_source = np.where(
        np.isfinite(sst_values["sst"]),
        "sst",
        "missing",
    )

    # Replace environmental fields
    df["wind_speed_kn"] = wind_speed_kn
    df["wind_direction_deg"] = wind_direction
    df["wave_height_m"] = wave_values["swh"]
    df["wave_direction_deg"] = wave_values["mwd"]
    df["wave_period_s"] = wave_values["mwp"]
    df["surface_pressure_pa"] = pressure_values["sp"]
    df["surface_temperature_c"] = surface_temperature_c

    # Useful audit-only provenance column
    df["surface_temperature_source_harmonized"] = temperature_source

    # Recompute dependent environmental features
    df = make_relative_features(df)

    # Write checkpoint chunk
    df.to_csv(
        chunk_path,
        index=False,
        encoding="utf-8-sig",
    )

    chunk_audit = {
        "chunk": chunk_no,
        "rows": len(df),
        "wind_nonmissing": int(df["wind_speed_kn"].notna().sum()),
        "wave_nonmissing": int(df["wave_height_m"].notna().sum()),
        "pressure_nonmissing": int(df["surface_pressure_pa"].notna().sum()),
        "temperature_nonmissing": int(df["surface_temperature_c"].notna().sum()),
        "temperature_sst_rows": int(
            (df["surface_temperature_source_harmonized"] == "sst").sum()
        ),
        "temperature_t2m_fallback_rows": 0,
    }

    chunk_audit_path.write_text(
        json.dumps(chunk_audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    chunk_files.append(chunk_path)
    audit_chunks.append(chunk_audit)

    wind.clear_cache()
    wave.clear_cache()
    pressure.clear_cache()
    sst.clear_cache()

    del df
    del wind_values
    del wave_values
    del pressure_values
    del sst_values
    gc.collect()

# ============================================================
# Merge completed chunks
# ============================================================
print("[merge] writing final CSV")

OUTPUT_CSV.unlink(missing_ok=True)

first = True
total_rows = 0

for chunk_path in sorted(chunk_files):
    for df in pd.read_csv(
        chunk_path,
        chunksize=CHUNK_ROWS,
        low_memory=True,
    ):
        df.to_csv(
            OUTPUT_CSV,
            mode="w" if first else "a",
            header=first,
            index=False,
            encoding="utf-8-sig" if first else "utf-8",
        )
        total_rows += len(df)
        first = False
        del df
        gc.collect()

# ============================================================
# Final audit
# ============================================================
audit = {
    "input": str(INPUT_CSV),
    "output": str(OUTPUT_CSV),
    "rows_output": total_rows,
    "spatial_interpolation": "bilinear",
    "temporal_interpolation": "linear",
    "nearest_neighbor_used": False,
    "wave_direction_interpolation": "circular_sin_cos",
    "wind_direction_method": "derived_after_interpolating_u10_v10",
    "wind_direction_convention": "meteorological_from",
    "surface_temperature_primary": "ERA5 sea_surface_temperature",
    "surface_temperature_fallback": None,
    "wind_oper_files": [str(item["path"]) for item in oper_specs],
    "wave_files": [str(item["path"]) for item in wave_specs],
    "pressure_temp_files": [str(item["path"]) for item in pressure_specs],
    "sst_files": [str(item["path"]) for item in sst_specs],
    "sst_missing_months": [],
    "chunk_rows": CHUNK_ROWS,
    "max_cached_time_slices_per_collection": MAX_CACHED_TIME_SLICES,
    "chunks": audit_chunks,
}

AUDIT_JSON.write_text(
    json.dumps(audit, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print("=" * 78)
print("Finished")
print(f"Rows   : {total_rows:,}")
print(f"Output : {OUTPUT_CSV}")
print(f"Audit  : {AUDIT_JSON}")
print("=" * 78)
