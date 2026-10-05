# -*- coding: utf-8 -*-
r"""
Inspect ONLY the cleaned tanker "31-version" dataset and estimate the ERA5
retrieval scope/time. This script NEVER calls cdsapi and NEVER downloads data.

Purpose
-------
1. Read the final cleaned tanker dataset (CSV or XLSX).
2. Report row/time/latitude/longitude coverage by month.
3. Rebuild the same adaptive ERA5 planning logic used by the final tanker
   downloader:
      - 10 x 10 degree planning tiles
      - max ~60 x 60 degree request rectangle
      - min coarse-box fill 30%
      - 1 degree spatial padding
      - observed days +/- 1 calendar day (cross-month/year supported)
      - 0.5 x 0.5 degree ERA5 output grid
      - surface = u10, v10, surface pressure, SST
      - wave = SWH, mean wave direction, mean wave period
4. Estimate request count and request-volume proxies.
5. If the CURRENT manifest exists, compare the 31-version plan against the
   currently-running original tanker plan.
6. Write two preview CSV reports. No CDS request is submitted.

Usage
-----
Option A: edit DATA_FILE below, then run
    python inspect_tanker31_era5_scope.py

Option B: pass the 31-version file directly
    python inspect_tanker31_cruise_era5_scope.py "C:\\path\\to\\tanker_31.csv"

CSV and XLSX are supported. XLSX reading requires openpyxl.
"""

import csv
import math
import os
import sys
from collections import defaultdict, deque
from datetime import date, datetime, timedelta, timezone
from pathlib import Path


# =============================================================================
# USER SETTINGS
# =============================================================================

REPO_ROOT = Path(__file__).resolve().parents[1]
# Pass the data path on the command line or override these defaults with
# environment variables. Raw/restricted data are intentionally not in Git.
DATA_FILE = Path(os.environ.get(
    "SHIP_FUEL_TANKER_FIXED31_COORDS",
    str(REPO_ROOT / "data" / "06_final_fixed31_with_coordinates.csv"),
))

# Existing final downloader manifest. If present, the script compares the new
# 31-version plan against the current running plan. If absent, comparison is
# simply skipped.
CURRENT_MANIFEST = Path(os.environ.get(
    "SHIP_FUEL_TANKER_ERA5_MANIFEST",
    str(REPO_ROOT / "data" / "era5" / "tanker" / "era5_request_manifest.csv"),
))

# Reports are written next to the 31-version dataset.
REPORT_PREFIX = "tanker31_cruise_era5_scope"

GRID = [0.5, 0.5]
TILE_LON_DEG = 10.0
TILE_LAT_DEG = 10.0
MAX_REQUEST_LON_SPAN_DEG = 60.0
MAX_REQUEST_LAT_SPAN_DEG = 60.0
MIN_BOX_FILL = 0.30
PADDING_DEG = 1.0
DAY_PADDING_DAYS = 1
PROGRESS_EVERY = 1_000_000

# Two CDS requests are made per month-slice: one surface + one wave.
REQUEST_GROUPS = 2
N_VARIABLES = 7

# Queue time varies greatly. These are sensitivity scenarios, not promises.
REQUESTS_PER_HOUR_SCENARIOS = [0.5, 1.0, 2.0, 4.0]

# ERA5 variables, shown in reports only. No data are downloaded here.
SURFACE = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "surface_pressure",
    "sea_surface_temperature",
]
WAVE = [
    "significant_height_of_combined_wind_waves_and_swell",
    "mean_wave_direction",
    "mean_wave_period",
]


# =============================================================================
# BASIC PARSING
# =============================================================================

def normalise_name(name):
    return str(name).strip().lower()


def find_index(header, candidates):
    lookup = {normalise_name(c): i for i, c in enumerate(header)}
    for candidate in candidates:
        if candidate in lookup:
            return lookup[candidate], header[lookup[candidate]]
    raise ValueError(
        f"Could not find any of {candidates}. Available columns include: {header[:50]}"
    )


def parse_timestamp(value):
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
        return dt

    s = str(value).strip()
    if not s:
        return None

    try:
        numeric = s.replace(".", "", 1)
        if numeric.isdigit():
            x = float(s)
            if x > 1e12:
                x /= 1000.0
            if x > 1e8:
                return datetime.fromtimestamp(x, tz=timezone.utc).replace(tzinfo=None)
    except Exception:
        pass

    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
        return dt
    except Exception:
        pass

    for fmt in [
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y/%m/%d %H:%M",
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%m/%d/%Y %H:%M:%S",
        "%m/%d/%Y %H:%M",
    ]:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def safe_float(value):
    try:
        x = float(str(value).strip())
        if math.isfinite(x):
            return x
    except Exception:
        pass
    return None


def normalize_lon(lon):
    return ((lon + 180.0) % 360.0) - 180.0


def set_csv_field_limit():
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def resolve_data_file():
    if len(sys.argv) >= 2:
        return Path(sys.argv[1].strip('"'))
    return DATA_FILE


# =============================================================================
# INPUT ITERATORS (CSV / XLSX)
# =============================================================================

def iter_rows(path):
    suffix = path.suffix.lower()

    if suffix == ".csv":
        set_csv_field_limit()
        fh = path.open("r", encoding="utf-8-sig", errors="replace", newline="")
        reader = csv.reader(fh)
        try:
            header = next(reader)
        except StopIteration:
            fh.close()
            raise ValueError("Input CSV is empty.")

        def gen():
            try:
                for row in reader:
                    yield row
            finally:
                fh.close()

        return header, gen()

    if suffix in {".xlsx", ".xlsm"}:
        try:
            from openpyxl import load_workbook
        except ImportError as exc:
            raise ImportError(
                "XLSX input requires openpyxl. Install with: pip install openpyxl"
            ) from exc

        wb = load_workbook(path, read_only=True, data_only=True)
        ws = wb.active
        it = ws.iter_rows(values_only=True)
        try:
            header = ["" if v is None else str(v) for v in next(it)]
        except StopIteration:
            wb.close()
            raise ValueError("Input workbook is empty.")

        def gen():
            try:
                for row in it:
                    yield list(row)
            finally:
                wb.close()

        return header, gen()

    raise ValueError("Supported input types: .csv, .xlsx, .xlsm")


# =============================================================================
# ADAPTIVE PLANNING GRID (same logic as final downloader)
# =============================================================================

N_LON_TILES = int(round(360.0 / TILE_LON_DEG))
N_LAT_TILES = int(round(180.0 / TILE_LAT_DEG))
MAX_LON_TILES = max(1, int(math.floor(MAX_REQUEST_LON_SPAN_DEG / TILE_LON_DEG + 1e-9)))
MAX_LAT_TILES = max(1, int(math.floor(MAX_REQUEST_LAT_SPAN_DEG / TILE_LAT_DEG + 1e-9)))


def point_to_tile(lat, lon):
    lon = normalize_lon(lon)
    lon_idx = int(math.floor((lon + 180.0) / TILE_LON_DEG))
    lon_idx = min(max(lon_idx, 0), N_LON_TILES - 1)

    lat_for_bin = min(lat, 90.0 - 1e-10)
    lat_idx = int(math.floor((lat_for_bin + 90.0) / TILE_LAT_DEG))
    lat_idx = min(max(lat_idx, 0), N_LAT_TILES - 1)
    return lat_idx, lon_idx


def bbox_of_cells(cells):
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), max(rows), min(cols), max(cols)


def box_shape(rect):
    r0, r1, c0, c1 = rect
    return r1 - r0 + 1, c1 - c0 + 1


def split_at_largest_gap(values):
    vals = sorted(set(values))
    best_gap = 0
    best_left = None
    for a, b in zip(vals, vals[1:]):
        gap = b - a - 1
        if gap > best_gap:
            best_gap = gap
            best_left = a
    return best_left if best_gap > 0 else None


def balanced_threshold(values):
    vals = sorted(values)
    if not vals or vals[0] == vals[-1]:
        return None
    mid = (vals[0] + vals[-1]) / 2.0
    candidates = sorted(set(vals))[:-1]
    return min(candidates, key=lambda x: abs(x - mid))


def partition_cells(cells):
    cells = set(cells)
    if not cells:
        return []

    rect = bbox_of_cells(cells)
    h, w = box_shape(rect)
    fill = len(cells) / float(h * w)

    if w <= MAX_LON_TILES and h <= MAX_LAT_TILES and (fill >= MIN_BOX_FILL or len(cells) == 1):
        return [(rect, cells)]

    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    col_gap_split = split_at_largest_gap(cols)
    row_gap_split = split_at_largest_gap(rows)

    choose_axis = None
    threshold = None

    if col_gap_split is not None or row_gap_split is not None:
        col_gap = row_gap = 0
        if col_gap_split is not None:
            sc = sorted(set(cols))
            i = sc.index(col_gap_split)
            col_gap = sc[i + 1] - sc[i] - 1
        if row_gap_split is not None:
            sr = sorted(set(rows))
            i = sr.index(row_gap_split)
            row_gap = sr[i + 1] - sr[i] - 1

        if col_gap >= row_gap and col_gap_split is not None:
            choose_axis, threshold = "lon", col_gap_split
        elif row_gap_split is not None:
            choose_axis, threshold = "lat", row_gap_split

    if choose_axis is None:
        lon_pressure = w / MAX_LON_TILES
        lat_pressure = h / MAX_LAT_TILES
        if lon_pressure > 1 or lat_pressure > 1:
            choose_axis = "lon" if lon_pressure >= lat_pressure else "lat"
        else:
            choose_axis = "lon" if w >= h else "lat"

        vals = cols if choose_axis == "lon" else rows
        threshold = balanced_threshold(vals)
        if threshold is None:
            choose_axis = "lat" if choose_axis == "lon" else "lon"
            vals = rows if choose_axis == "lat" else cols
            threshold = balanced_threshold(vals)

    if threshold is None:
        return [(rect, cells)]

    if choose_axis == "lon":
        a = {(r, c) for r, c in cells if c <= threshold}
    else:
        a = {(r, c) for r, c in cells if r <= threshold}
    b = cells - a

    if not a or not b:
        return [(rect, cells)]
    return partition_cells(a) + partition_cells(b)


def connected_components(tiles):
    # Deliberately no dateline wrapping: one rectangle must not cross +/-180.
    remaining = set(tiles)
    comps = []
    while remaining:
        start = remaining.pop()
        comp = {start}
        q = deque([start])
        while q:
            r, c = q.popleft()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < N_LAT_TILES and 0 <= nc < N_LON_TILES):
                        continue
                    nxt = (nr, nc)
                    if nxt in remaining:
                        remaining.remove(nxt)
                        comp.add(nxt)
                        q.append(nxt)
        comps.append(comp)
    return comps


def rectangle_core_bounds(rect):
    r0, r1, c0, c1 = rect
    south = -90.0 + r0 * TILE_LAT_DEG
    north = -90.0 + (r1 + 1) * TILE_LAT_DEG
    west = -180.0 + c0 * TILE_LON_DEG
    east = -180.0 + (c1 + 1) * TILE_LON_DEG
    return north, west, south, east


def rectangle_to_area(rect):
    north, west, south, east = rectangle_core_bounds(rect)
    south = max(-90.0, south - PADDING_DEG)
    north = min(90.0, north + PADDING_DEG)
    west = max(-180.0, west - PADDING_DEG)
    east = min(180.0, east + PADDING_DEG)
    return [
        float(math.ceil(north)),
        float(math.floor(west)),
        float(math.floor(south)),
        float(math.ceil(east)),
    ]


def expand_dates(days, year, month):
    out = set()
    for d in days:
        base = date(year, month, d)
        for offset in range(-DAY_PADDING_DAYS, DAY_PADDING_DAYS + 1):
            out.add(base + timedelta(days=offset))
    return sorted(out)


def group_dates_by_month(dates):
    grouped = defaultdict(set)
    for dt in dates:
        grouped[(dt.year, dt.month)].add(dt.day)
    return [(ym, sorted(days)) for ym, days in sorted(grouped.items())]


def compact_days(days):
    if not days:
        return ""
    ranges = []
    start = prev = days[0]
    for d in days[1:]:
        if d == prev + 1:
            prev = d
        else:
            ranges.append((start, prev))
            start = prev = d
    ranges.append((start, prev))
    return ",".join(f"{a:02d}" if a == b else f"{a:02d}-{b:02d}" for a, b in ranges)


def compact_request_groups(request_groups):
    return "; ".join(
        f"{year:04d}-{month:02d}:{compact_days(days)}"
        for (year, month), days in request_groups
    )


def build_month_parts(stat, year, month):
    parts = []
    for comp in connected_components(stat["tiles"]):
        for rect, assigned_cells in partition_cells(comp):
            obs_days = set()
            for cell in assigned_cells:
                obs_days.update(stat["tile_days"][cell])
            request_dates = expand_dates(obs_days, year, month)
            parts.append(
                {
                    "rect": rect,
                    "cells": assigned_cells,
                    "area": rectangle_to_area(rect),
                    "observed_days": sorted(obs_days),
                    "dates": request_dates,
                    "request_groups": group_dates_by_month(request_dates),
                }
            )
    parts.sort(key=lambda p: (p["rect"][0], p["rect"][2], p["rect"][1], p["rect"][3]))
    return parts


# =============================================================================
# 0.5 DEG TARGET-GRID DIAGNOSTICS
# =============================================================================

def grid_bracket_indices(value, origin, step, n_points):
    """Return the lower/upper grid-node indices needed for linear interpolation."""
    pos = (value - origin) / step
    lo = int(math.floor(pos + 1e-12))
    hi = int(math.ceil(pos - 1e-12))
    lo = min(max(lo, 0), n_points - 1)
    hi = min(max(hi, 0), n_points - 1)
    if lo == hi:
        # Keep one neighbour where possible so a local linear interpolation still
        # has two grid nodes available along this dimension.
        if hi < n_points - 1:
            hi += 1
        elif lo > 0:
            lo -= 1
    return lo, hi


N_GRID_LAT = int(round(180.0 / GRID[0])) + 1
N_GRID_LON = int(round(360.0 / GRID[1])) + 1


def required_grid_nodes(lat, lon):
    lon = normalize_lon(lon)
    lat_lo, lat_hi = grid_bracket_indices(lat, -90.0, GRID[0], N_GRID_LAT)
    lon_lo, lon_hi = grid_bracket_indices(lon, -180.0, GRID[1], N_GRID_LON)
    return {
        (lat_lo, lon_lo),
        (lat_lo, lon_hi),
        (lat_hi, lon_lo),
        (lat_hi, lon_hi),
    }


def required_hours(dt):
    floor_dt = dt.replace(minute=0, second=0, microsecond=0)
    if dt == floor_dt:
        # Exact ERA5 hour: only that hour is needed temporally.
        return {floor_dt}
    return {floor_dt, floor_dt + timedelta(hours=1)}


# =============================================================================
# VOLUME / TIME ESTIMATION
# =============================================================================

def area_grid_points(area):
    north, west, south, east = area
    n_lat = int(round((north - south) / GRID[0])) + 1
    n_lon = int(round((east - west) / GRID[1])) + 1
    return max(n_lat, 1) * max(n_lon, 1)


def plan_volume(parts_by_month):
    """Return request slices and a raw gridpoint-hour-field proxy."""
    total_parts = 0
    total_slices = 0
    total_part_days = 0
    total_gridpoint_hours = 0
    total_field_values = 0

    for parts in parts_by_month.values():
        total_parts += len(parts)
        for p in parts:
            total_part_days += len(p["dates"])
            gp = area_grid_points(p["area"])
            for _, days in p["request_groups"]:
                total_slices += 1
                gph = gp * len(days) * 24
                total_gridpoint_hours += gph
                total_field_values += gph * N_VARIABLES

    return {
        "parts": total_parts,
        "slices": total_slices,
        "requests": total_slices * REQUEST_GROUPS,
        "part_days": total_part_days,
        "gridpoint_hours": total_gridpoint_hours,
        "field_values": total_field_values,
        "float32_gb_equiv": total_field_values * 4 / 1e9,
    }


def manifest_volume(path):
    if not path.is_file():
        return None

    slices = 0
    gridpoint_hours = 0
    field_values = 0
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                area = [
                    float(row["north"]),
                    float(row["west"]),
                    float(row["south"]),
                    float(row["east"]),
                ]
                day_count = int(row["day_count"])
            except Exception:
                continue
            gp = area_grid_points(area)
            gph = gp * day_count * 24
            slices += 1
            gridpoint_hours += gph
            field_values += gph * N_VARIABLES

    return {
        "slices": slices,
        "requests": slices * REQUEST_GROUPS,
        "gridpoint_hours": gridpoint_hours,
        "field_values": field_values,
        "float32_gb_equiv": field_values * 4 / 1e9,
    }


def format_duration_hours(hours):
    if hours < 24:
        return f"{hours:.1f} h"
    return f"{hours / 24.0:.1f} days"


# =============================================================================
# SCAN + REPORT
# =============================================================================

def scan(path):
    if not path.is_file():
        raise FileNotFoundError(
            f"31-version tanker file not found: {path}\n"
            "Edit DATA_FILE at the top of the script or pass the path on the command line."
        )

    header, rows = iter_rows(path)
    time_i, time_name = find_index(
        header, ["time", "datetime", "date_time", "timestamp", "timestamp_utc", "utc", "date"]
    )
    lat_i, lat_name = find_index(header, ["lat", "latitude", "latitude_deg"])
    lon_i, lon_name = find_index(header, ["lon", "lng", "longitude", "longitude_deg"])
    ship_i, ship_name = find_index(header, ["ship_type"])
    phase_i, phase_name = find_index(header, ["voyage_phase"])
    needed_i = max(time_i, lat_i, lon_i, ship_i, phase_i)

    print("Input:", path)
    print("Detected columns:")
    print("  time      :", time_name)
    print("  latitude  :", lat_name)
    print("  longitude :", lon_name)
    print("  ship type :", ship_name)
    print("  phase     :", phase_name)
    print("Filter:")
    print("  ship_type contains 'tanker' (case-insensitive)")
    print("  voyage_phase == 'cruise' (case-insensitive)")
    print("\nScanning CLEANED fixed31 TANKER + CRUISE rows only (NO DOWNLOAD)...")

    stats = defaultdict(
        lambda: {
            "rows": 0,
            "tiles": set(),
            "tile_days": defaultdict(set),
            "lat_min": None,
            "lat_max": None,
            "lon_bins": set(),
            "required_grid_nodes": set(),
            "required_hours": set(),
            "observed_days": set(),
        }
    )

    total = valid = bad_time = bad_position = short_rows = 0
    t_min = t_max = None
    lat_min = lat_max = None
    global_lon_bins = set()
    global_grid_nodes = set()
    global_hours = set()

    for row in rows:
        total += 1
        if len(row) <= needed_i:
            short_rows += 1
            continue

        ship_value = str(row[ship_i]).strip().lower()
        phase_value = str(row[phase_i]).strip().lower()
        if "tanker" not in ship_value or phase_value != "cruise":
            continue

        dt = parse_timestamp(row[time_i])
        lat = safe_float(row[lat_i])
        lon = safe_float(row[lon_i])

        if dt is None:
            bad_time += 1
            continue
        if lat is None or lon is None or not (-90 <= lat <= 90) or not (-360 <= lon <= 360):
            bad_position += 1
            continue

        lon = normalize_lon(lon)
        ym = (dt.year, dt.month)
        cell = point_to_tile(lat, lon)
        s = stats[ym]
        s["rows"] += 1
        s["tiles"].add(cell)
        s["tile_days"][cell].add(dt.day)
        s["observed_days"].add(dt.day)
        s["lat_min"] = lat if s["lat_min"] is None else min(s["lat_min"], lat)
        s["lat_max"] = lat if s["lat_max"] is None else max(s["lat_max"], lat)

        lon_bin = int(math.floor((lon + 180.0) / GRID[1]))
        lon_bin = min(max(lon_bin, 0), int(360 / GRID[1]) - 1)
        s["lon_bins"].add(lon_bin)
        global_lon_bins.add(lon_bin)

        nodes = required_grid_nodes(lat, lon)
        hours = required_hours(dt)
        s["required_grid_nodes"].update(nodes)
        s["required_hours"].update(hours)
        global_grid_nodes.update(nodes)
        global_hours.update(hours)

        valid += 1
        t_min = dt if t_min is None else min(t_min, dt)
        t_max = dt if t_max is None else max(t_max, dt)
        lat_min = lat if lat_min is None else min(lat_min, lat)
        lat_max = lat if lat_max is None else max(lat_max, lat)

        if total % PROGRESS_EVERY == 0:
            print(f"  scanned {total:,} rows; valid={valid:,}; months={len(stats)}")

    if valid == 0:
        raise ValueError("No valid timestamp + latitude + longitude rows were found.")

    return {
        "stats": stats,
        "total": total,
        "valid": valid,
        "bad_time": bad_time,
        "bad_position": bad_position,
        "short_rows": short_rows,
        "t_min": t_min,
        "t_max": t_max,
        "lat_min": lat_min,
        "lat_max": lat_max,
        "global_lon_bins": global_lon_bins,
        "global_grid_nodes": global_grid_nodes,
        "global_hours": global_hours,
    }


def circular_span_from_bins(bins, bin_deg=0.5):
    if not bins:
        return 0.0
    n = int(round(360.0 / bin_deg))
    occ = sorted(bins)
    if len(occ) == 1:
        return bin_deg
    largest_gap = 0
    for i, cur in enumerate(occ):
        nxt = occ[(i + 1) % len(occ)]
        if i == len(occ) - 1:
            nxt += n
        gap = nxt - cur - 1
        largest_gap = max(largest_gap, gap)
    return (n - largest_gap) * bin_deg


def build_plan(stats):
    monthly_plan = {}
    for (year, month), stat in sorted(stats.items()):
        monthly_plan[(year, month)] = build_month_parts(stat, year, month)
    return monthly_plan


def write_reports(path, stats, plan):
    out_dir = path.parent
    monthly_file = out_dir / f"{REPORT_PREFIX}_monthly.csv"
    plan_file = out_dir / f"{REPORT_PREFIX}_plan_preview.csv"

    with monthly_file.open("w", encoding="utf-8-sig", newline="") as fh:
        fieldnames = [
            "year", "month", "rows", "lat_min", "lat_max", "circular_lon_span_deg",
            "observed_days", "occupied_10deg_tiles", "required_0p5_grid_nodes",
            "required_era5_hours", "adaptive_parts", "month_slices", "part_days",
            "coarse_box_fill",
        ]
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for ym in sorted(stats):
            year, month = ym
            s = stats[ym]
            parts = plan[ym]
            box_tiles = sum(box_shape(p["rect"])[0] * box_shape(p["rect"])[1] for p in parts)
            fill = len(s["tiles"]) / box_tiles if box_tiles else 1.0
            w.writerow({
                "year": year,
                "month": month,
                "rows": s["rows"],
                "lat_min": s["lat_min"],
                "lat_max": s["lat_max"],
                "circular_lon_span_deg": round(circular_span_from_bins(s["lon_bins"], GRID[1]), 2),
                "observed_days": compact_days(sorted(s["observed_days"])),
                "occupied_10deg_tiles": len(s["tiles"]),
                "required_0p5_grid_nodes": len(s["required_grid_nodes"]),
                "required_era5_hours": len(s["required_hours"]),
                "adaptive_parts": len(parts),
                "month_slices": sum(len(p["request_groups"]) for p in parts),
                "part_days": sum(len(p["dates"]) for p in parts),
                "coarse_box_fill": round(fill, 4),
            })

    with plan_file.open("w", encoding="utf-8-sig", newline="") as fh:
        fieldnames = [
            "plan_year", "plan_month", "part", "north", "west", "south", "east",
            "request_year", "request_month", "days", "day_count", "grid_points",
            "gridpoint_hours", "surface_variables", "wave_variables",
        ]
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for (year, month), parts in sorted(plan.items()):
            for i, p in enumerate(parts, 1):
                gp = area_grid_points(p["area"])
                north, west, south, east = p["area"]
                for (ry, rm), days in p["request_groups"]:
                    w.writerow({
                        "plan_year": year,
                        "plan_month": month,
                        "part": i,
                        "north": north,
                        "west": west,
                        "south": south,
                        "east": east,
                        "request_year": ry,
                        "request_month": rm,
                        "days": compact_days(days),
                        "day_count": len(days),
                        "grid_points": gp,
                        "gridpoint_hours": gp * len(days) * 24,
                        "surface_variables": ";".join(SURFACE),
                        "wave_variables": ";".join(WAVE),
                    })

    return monthly_file, plan_file


def main():
    path = resolve_data_file()
    result = scan(path)
    stats = result["stats"]
    plan = build_plan(stats)
    volume = plan_volume(plan)
    baseline = manifest_volume(CURRENT_MANIFEST)

    print("\n" + "=" * 92)
    print("CLEANED FIXED31 TANKER + CRUISE: OVERALL RANGE")
    print("=" * 92)
    print("Rows total/valid       :", f"{result['total']:,} / {result['valid']:,}")
    print("Bad time/position      :", f"{result['bad_time']:,} / {result['bad_position']:,}")
    print("Short rows             :", f"{result['short_rows']:,}")
    print("Time range             :", result["t_min"], "->", result["t_max"])
    print("Latitude range         :", f"{result['lat_min']:.6f} .. {result['lat_max']:.6f}")
    print("Dateline-aware lon span:", f"~{circular_span_from_bins(result['global_lon_bins'], GRID[1]):.1f} deg")
    print("Months                 :", len(stats))
    print("Unique 0.5deg grid nodes needed spatially :", f"{len(result['global_grid_nodes']):,}")
    print("Unique ERA5 hourly timestamps needed       :", f"{len(result['global_hours']):,}")

    print("\nMonthly summary:")
    print("-" * 92)
    print("month    rows      lat-range           lon-span  tiles  0.5nodes  hours  parts  slices")
    print("-" * 92)
    for ym in sorted(stats):
        year, month = ym
        s = stats[ym]
        parts = plan[ym]
        slices = sum(len(p["request_groups"]) for p in parts)
        print(
            f"{year}-{month:02d}  {s['rows']:>8,}  "
            f"{s['lat_min']:>7.2f}..{s['lat_max']:<7.2f}  "
            f"{circular_span_from_bins(s['lon_bins'], GRID[1]):>7.1f}  "
            f"{len(s['tiles']):>5}  {len(s['required_grid_nodes']):>8,}  "
            f"{len(s['required_hours']):>5,}  {len(parts):>5}  {slices:>6}"
        )

    print("\n" + "=" * 92)
    print("ERA5 DOWNLOAD ESTIMATE FOR FIXED31 TANKER + CRUISE")
    print("=" * 92)
    print("Adaptive spatial parts :", f"{volume['parts']:,}")
    print("Monthly request slices :", f"{volume['slices']:,}")
    print("CDS requests            :", f"{volume['requests']:,}", "(surface + wave)")
    print("Summed spatial-part days:", f"{volume['part_days']:,}")
    print("Gridpoint-hours proxy   :", f"{volume['gridpoint_hours']:,}")
    print("7-field values proxy    :", f"{volume['field_values']:,}")
    print("Float32 raw equivalent  :", f"~{volume['float32_gb_equiv']:.1f} GB")
    print("  NOTE: this is an uncompressed comparison proxy, NOT predicted NetCDF file size.")

    print("\nQueue/processing-time sensitivity (very approximate):")
    for rate in REQUESTS_PER_HOUR_SCENARIOS:
        hours = volume["requests"] / rate
        print(f"  at {rate:g} completed request(s)/hour -> ~{format_duration_hours(hours)}")

    if baseline:
        print("\nComparison with current manifest:")
        print("  Current manifest      :", CURRENT_MANIFEST)
        print("  Current requests      :", f"{baseline['requests']:,}")
        print("  31-version requests   :", f"{volume['requests']:,}")
        req_ratio = volume["requests"] / baseline["requests"] if baseline["requests"] else float("nan")
        vol_ratio = volume["field_values"] / baseline["field_values"] if baseline["field_values"] else float("nan")
        print("  Request-count ratio   :", f"{req_ratio:.1%}")
        print("  Area x day x 7-field volume ratio:", f"{vol_ratio:.1%}")
        if vol_ratio < 1:
            print("  Estimated volume saving:", f"~{(1.0 - vol_ratio):.1%}")
        else:
            print("  Estimated volume change:", f"~{(vol_ratio - 1.0):+.1%}")
        print("  If CDS conditions stay similar, the volume ratio is the better rough time comparison.")
    else:
        print("\nCurrent manifest not found; relative comparison skipped:")
        print(" ", CURRENT_MANIFEST)

    monthly_file, plan_file = write_reports(path, stats, plan)
    print("\nReports written:")
    print(" ", monthly_file)
    print(" ", plan_file)
    print("\nNO CDS request was submitted. This script is inspection/planning only.")


if __name__ == "__main__":
    main()
