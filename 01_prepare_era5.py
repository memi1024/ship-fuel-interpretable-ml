# -*- coding: utf-8 -*-
"""Inspect vessel-data time/position coverage and download all seven ERA5 fields.

Stage 1: prepare the ERA5 fields used for containership environmental matching.

The seven fields are downloaded from the start, so there is NO separate 2-m
temperature supplementation step and NO 2m_temperature request. Temperature is
represented by ERA5 sea-surface temperature (SST), matching the final
harmonised environmental definition used by the interpolation code.

Downloaded fields
-----------------
Atmospheric/surface file (monthly):
    u10, v10, surface_pressure, sea_surface_temperature
Wave file (monthly):
    significant wave height, mean wave direction, mean wave period

Files are named to be directly compatible with
``era5_interpolate_to_continership_7fields.py``:
    era5_surface_wind_raw_YYYY_MM.nc
    era5_wave_raw_YYYY_MM.nc
"""
from __future__ import annotations

import argparse
import calendar
import json
import math
import re
from datetime import timedelta
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

SURFACE_VARIABLES = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "surface_pressure",
    "sea_surface_temperature",
]
WAVE_VARIABLES = [
    "significant_height_of_combined_wind_waves_and_swell",
    "mean_wave_direction",
    "mean_wave_period",
]

TIME_ALIASES = (
    "UTC时间(-)", "UTC时间", "UTC Time", "utc_time", "timestamp_utc",
    "timestamp", "datetime", "日期时间", "时间", "time",
)
LAT_ALIASES = (
    "纬度(deg)", "纬度", "latitude_deg", "latitude", "lat",
)
LON_ALIASES = (
    "经度(deg)", "经度", "longitude_deg", "longitude", "lng", "lon",
)


def normalize_name(value: object) -> str:
    text = str(value).strip().lower()
    return re.sub(r"[\s_\-\/\\()（）\[\]{}]+", "", text)


def resolve_column(columns: Iterable[str], aliases: Sequence[str], keywords: Sequence[str]) -> str:
    cols = list(map(str, columns))
    lookup = {normalize_name(c): c for c in cols}
    for a in aliases:
        hit = lookup.get(normalize_name(a))
        if hit is not None:
            return hit
    lowered = [(c, normalize_name(c)) for c in cols]
    for keyword in keywords:
        key = normalize_name(keyword)
        hits = [c for c, norm in lowered if key in norm]
        if len(hits) == 1:
            return hits[0]
    raise KeyError(f"Cannot resolve column from aliases={aliases}; available columns={cols}")


def read_header(path: Path, sheet: str | int) -> list[str]:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls", ".xlsm"}:
        return pd.read_excel(path, sheet_name=sheet, nrows=0).columns.astype(str).tolist()
    if suffix in {".csv", ".txt", ".tsv"}:
        sep = "\t" if suffix == ".tsv" else None
        return pd.read_csv(path, nrows=0, sep=sep, engine="python").columns.astype(str).tolist()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path).columns.astype(str).tolist()
    raise ValueError(f"Unsupported input type: {path.suffix}")


def iter_selected(path: Path, columns: list[str], sheet: str | int, chunksize: int):
    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt", ".tsv"}:
        sep = "\t" if suffix == ".tsv" else None
        yield from pd.read_csv(
            path, usecols=columns, chunksize=chunksize, sep=sep, engine="python", low_memory=True
        )
        return
    if suffix in {".xlsx", ".xls", ".xlsm"}:
        # Excel is read once; this is the source class used by the
        # containership interpolation producer.
        yield pd.read_excel(path, sheet_name=sheet, usecols=columns)
        return
    if suffix in {".parquet", ".pq"}:
        yield pd.read_parquet(path, columns=columns)
        return
    raise ValueError(f"Unsupported input type: {path.suffix}")


def inspect_extent(path: Path, sheet: str | int, chunksize: int) -> dict:
    header = read_header(path, sheet)
    time_col = resolve_column(header, TIME_ALIASES, ("utc", "time", "时间"))
    lat_col = resolve_column(header, LAT_ALIASES, ("latitude", "lat", "纬度"))
    lon_col = resolve_column(header, LON_ALIASES, ("longitude", "lon", "经度"))

    row_count = 0
    valid_time = valid_lat = valid_lon = valid_all = 0
    tmin = tmax = None
    lat_min = math.inf
    lat_max = -math.inf
    lon_min = math.inf
    lon_max = -math.inf

    for chunk in iter_selected(path, [time_col, lat_col, lon_col], sheet, chunksize):
        row_count += len(chunk)
        times = pd.to_datetime(chunk[time_col], errors="coerce", utc=True)
        lat = pd.to_numeric(chunk[lat_col], errors="coerce")
        lon = pd.to_numeric(chunk[lon_col], errors="coerce")
        lat = lat.where(lat.between(-90, 90))
        lon = lon.where(lon.between(-180, 180))

        vt = times.notna()
        vlat = lat.notna()
        vlon = lon.notna()
        vall = vt & vlat & vlon
        valid_time += int(vt.sum())
        valid_lat += int(vlat.sum())
        valid_lon += int(vlon.sum())
        valid_all += int(vall.sum())

        if vt.any():
            cmin = times[vt].min()
            cmax = times[vt].max()
            tmin = cmin if tmin is None or cmin < tmin else tmin
            tmax = cmax if tmax is None or cmax > tmax else tmax
        if vlat.any():
            lat_min = min(lat_min, float(lat[vlat].min()))
            lat_max = max(lat_max, float(lat[vlat].max()))
        if vlon.any():
            lon_min = min(lon_min, float(lon[vlon].min()))
            lon_max = max(lon_max, float(lon[vlon].max()))

    if tmin is None or tmax is None or not np.isfinite([lat_min, lat_max, lon_min, lon_max]).all():
        raise RuntimeError("No valid time/latitude/longitude extent could be determined.")

    return {
        "input": str(path.resolve()),
        "rows": row_count,
        "columns": {"time": time_col, "latitude": lat_col, "longitude": lon_col},
        "valid_rows": {
            "time": valid_time,
            "latitude": valid_lat,
            "longitude": valid_lon,
            "time_lat_lon": valid_all,
        },
        "time_min_utc": tmin.isoformat(),
        "time_max_utc": tmax.isoformat(),
        "latitude_min": lat_min,
        "latitude_max": lat_max,
        "longitude_min": lon_min,
        "longitude_max": lon_max,
    }


def outward_area(extent: dict, grid: float, margin_cells: int) -> list[float]:
    margin = grid * margin_cells
    north = math.ceil(extent["latitude_max"] / grid) * grid + margin
    south = math.floor(extent["latitude_min"] / grid) * grid - margin
    west = math.floor(extent["longitude_min"] / grid) * grid - margin
    east = math.ceil(extent["longitude_max"] / grid) * grid + margin
    return [min(90.0, north), max(-180.0, west), max(-90.0, south), min(180.0, east)]


def month_iter(start: pd.Timestamp, end: pd.Timestamp):
    year, month = start.year, start.month
    while (year, month) <= (end.year, end.month):
        yield year, month
        month += 1
        if month == 13:
            year += 1
            month = 1


def days_for_month(year: int, month: int, start: pd.Timestamp, end: pd.Timestamp) -> list[str]:
    first = pd.Timestamp(year=year, month=month, day=1, tz="UTC")
    last = pd.Timestamp(year=year, month=month, day=calendar.monthrange(year, month)[1], tz="UTC")
    lo = max(first, start.normalize())
    hi = min(last, end.normalize())
    if lo > hi:
        return []
    return [f"{d.day:02d}" for d in pd.date_range(lo, hi, freq="D")]


def retrieve_one(client, target: Path, variables: list[str], year: int, month: int,
                 days: list[str], area: list[float], grid: float, force: bool):
    if target.exists() and target.stat().st_size > 0 and not force:
        print(f"[skip] {target.name}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    if partial.exists():
        partial.unlink()
    request = {
        "product_type": ["reanalysis"],
        "variable": variables,
        "year": [str(year)],
        "month": [f"{month:02d}"],
        "day": days,
        "time": [f"{h:02d}:00" for h in range(24)],
        "area": area,
        "grid": [grid, grid],
        "data_format": "netcdf",
        "download_format": "unarchived",
    }
    print(f"[download] {target.name} | {year}-{month:02d} | days={days[0]}..{days[-1]}")
    client.retrieve("reanalysis-era5-single-levels", request, str(partial))
    partial.replace(target)


def parse_sheet(value: str):
    return int(value) if value.isdigit() else value


def main() -> int:
    p = argparse.ArgumentParser(description="Inspect data extent and download complete ERA5 seven-field set.")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True,
                   help="ERA5 raw directory; monthly files and manifests are written here.")
    p.add_argument("--sheet", default="0", help="Excel sheet index/name; default 0.")
    p.add_argument("--chunksize", type=int, default=100_000)
    p.add_argument("--grid", type=float, default=0.5)
    p.add_argument("--margin-cells", type=int, default=1,
                   help="Extra grid cells around observed bounds to preserve bilinear neighbours.")
    p.add_argument("--time-buffer-hours", type=float, default=2.0,
                   help="Extend first/last observation time for temporal interpolation neighbours.")
    p.add_argument("--inspect-only", action="store_true")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sheet = parse_sheet(str(args.sheet))

    extent = inspect_extent(args.input, sheet, args.chunksize)
    area = outward_area(extent, args.grid, args.margin_cells)
    start = pd.Timestamp(extent["time_min_utc"]) - pd.Timedelta(hours=args.time_buffer_hours)
    end = pd.Timestamp(extent["time_max_utc"]) + pd.Timedelta(hours=args.time_buffer_hours)

    manifest = {
        "purpose": "era5_7field_download",
        "extent": extent,
        "request_time_start_utc": start.isoformat(),
        "request_time_end_utc": end.isoformat(),
        "area_north_west_south_east": area,
        "grid_degrees": [args.grid, args.grid],
        "spatial_margin_cells": args.margin_cells,
        "surface_variables": SURFACE_VARIABLES,
        "wave_variables": WAVE_VARIABLES,
        "temperature_definition": "sea_surface_temperature",
        "two_m_temperature_requested": False,
        "note": "All seven environmental variables are downloaded from the start; no later t2m supplement/fallback is used.",
    }
    manifest_path = args.output_dir / "era5_request_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 78)
    print("Observed data extent")
    print(f"rows       : {extent['rows']:,}")
    print(f"time UTC   : {extent['time_min_utc']} -> {extent['time_max_utc']}")
    print(f"latitude   : {extent['latitude_min']:.6f} -> {extent['latitude_max']:.6f}")
    print(f"longitude  : {extent['longitude_min']:.6f} -> {extent['longitude_max']:.6f}")
    print(f"ERA5 area  : {area}  [N,W,S,E]")
    print("temperature: ERA5 SST only; 2m_temperature is NOT requested")
    print(f"manifest   : {manifest_path}")
    print("=" * 78)

    if args.inspect_only:
        return 0

    try:
        import cdsapi
    except ImportError as exc:
        raise SystemExit("cdsapi is required for download: python -m pip install cdsapi") from exc

    client = cdsapi.Client()
    for year, month in month_iter(start, end):
        days = days_for_month(year, month, start, end)
        if not days:
            continue
        retrieve_one(
            client,
            args.output_dir / f"era5_surface_wind_raw_{year:04d}_{month:02d}.nc",
            SURFACE_VARIABLES,
            year, month, days, area, args.grid, args.force,
        )
        retrieve_one(
            client,
            args.output_dir / f"era5_wave_raw_{year:04d}_{month:02d}.nc",
            WAVE_VARIABLES,
            year, month, days, area, args.grid, args.force,
        )

    print("ERA5 seven-field download complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
