# -*- coding: utf-8 -*-
r"""
Final adaptive/day-aware ERA5 downloader for tanker harmonisation.

Environmental-variable structure (matched to the formal harmonisation pipeline)
-------------------------------------------------------------------------------
surface:
    10m_u_component_of_wind
    10m_v_component_of_wind
    surface_pressure
    sea_surface_temperature

wave:
    significant_height_of_combined_wind_waves_and_swell
    mean_wave_direction
    mean_wave_period

2 m air temperature is intentionally NOT requested.

The downloader:
- streams the tanker CSV once;
- plans only occupied 10 x 10 degree tiles;
- adaptively groups tiles into request rectangles no larger than about 60 x 60
  degrees before padding;
- never makes one rectangle cross the +/-180 degree dateline;
- adds 1 degree spatial padding;
- requests only observed days plus +/-1 calendar day, including across month/year
  boundaries;
- saves surface and wave data separately;
- writes era5_request_manifest.csv before any CDS request is submitted.

The manifest is the contract used by interpolate_tanker31_adaptive_era5_AIStudio_v2.py.
Do not rename/move downloaded NetCDF files without updating the manifest.
"""

import csv
import math
import os
import sys
import time
from collections import defaultdict, deque
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import cdsapi


# -----------------------------------------------------------------------------
# USER SETTINGS
# -----------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_FILE = Path(os.environ.get(
    "SHIP_FUEL_TANKER_SOURCE_CSV",
    str(REPO_ROOT / "data" / "raw" / "Oil Tanker Carrier.csv"),
))
ROOT = Path(os.environ.get(
    "SHIP_FUEL_TANKER_ERA5_ROOT",
    str(REPO_ROOT / "data" / "era5" / "tanker"),
))

GRID = [0.5, 0.5]
TIMES = [f"{h:02d}:00" for h in range(24)]

TILE_LON_DEG = 10.0
TILE_LAT_DEG = 10.0
MAX_REQUEST_LON_SPAN_DEG = 60.0
MAX_REQUEST_LAT_SPAN_DEG = 60.0
MIN_BOX_FILL = 0.30
PADDING_DEG = 1.0
DAY_PADDING_DAYS = 1

CONFIRM_BEFORE_DOWNLOAD = True
MAX_RETRIES = 3
RETRY_WAIT_SECONDS = 20
PROGRESS_EVERY = 1_000_000


# -----------------------------------------------------------------------------
# OUTPUTS AND VARIABLES
# -----------------------------------------------------------------------------

SURFACE_DIR = ROOT / "surface"
WAVE_DIR = ROOT / "wave"
MANIFEST_FILE = ROOT / "era5_request_manifest.csv"

SURFACE_DIR.mkdir(parents=True, exist_ok=True)
WAVE_DIR.mkdir(parents=True, exist_ok=True)

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


# -----------------------------------------------------------------------------
# BASIC PARSING
# -----------------------------------------------------------------------------

def normalise_name(name):
    return str(name).strip().lower()


def find_index(header, candidates):
    lookup = {normalise_name(c): i for i, c in enumerate(header)}
    for candidate in candidates:
        if candidate in lookup:
            return lookup[candidate], header[lookup[candidate]]
    raise ValueError(
        f"Could not find any of {candidates}. Available columns include: {header[:40]}"
    )


def parse_timestamp(value):
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


# -----------------------------------------------------------------------------
# ADAPTIVE PLANNING GRID
# -----------------------------------------------------------------------------

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
    """8-neighbour components, deliberately without dateline wrapping."""
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


def tile_cells_text(cells):
    return "|".join(f"{r}:{c}" for r, c in sorted(cells))


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


# -----------------------------------------------------------------------------
# FILE NAMING + MANIFEST
# -----------------------------------------------------------------------------

def output_file(folder, prefix, plan_year, plan_month, part_number, request_year, request_month):
    base = f"{prefix}_{plan_year}_{plan_month:02d}_part{part_number}"
    if (request_year, request_month) != (plan_year, plan_month):
        base += f"_pad_{request_year}_{request_month:02d}"
    return folder / f"{base}.nc"


def manifest_rows(monthly_plan):
    rows = []
    for (plan_year, plan_month), parts in sorted(monthly_plan.items()):
        n_parts = len(parts)
        for part_number, part in enumerate(parts, 1):
            core_north, core_west, core_south, core_east = rectangle_core_bounds(part["rect"])
            north, west, south, east = part["area"]
            for (request_year, request_month), days in part["request_groups"]:
                surface_file = output_file(
                    SURFACE_DIR, "era5_surface", plan_year, plan_month, part_number,
                    request_year, request_month,
                )
                wave_file = output_file(
                    WAVE_DIR, "era5_wave", plan_year, plan_month, part_number,
                    request_year, request_month,
                )
                rows.append(
                    {
                        "plan_year": plan_year,
                        "plan_month": plan_month,
                        "part": part_number,
                        "part_count": n_parts,
                        "core_north": core_north,
                        "core_west": core_west,
                        "core_south": core_south,
                        "core_east": core_east,
                        "north": north,
                        "west": west,
                        "south": south,
                        "east": east,
                        "request_year": request_year,
                        "request_month": request_month,
                        "days": compact_days(days),
                        "day_count": len(days),
                        "observed_days_plan_month": compact_days(part["observed_days"]),
                        "tile_cells": tile_cells_text(part["cells"]),
                        "surface_variables": ";".join(SURFACE),
                        "wave_variables": ";".join(WAVE),
                        "grid_lat_deg": GRID[0],
                        "grid_lon_deg": GRID[1],
                        "planning_tile_lat_deg": TILE_LAT_DEG,
                        "planning_tile_lon_deg": TILE_LON_DEG,
                        "spatial_padding_deg": PADDING_DEG,
                        "day_padding_days": DAY_PADDING_DAYS,
                        "surface_file": str(surface_file.relative_to(ROOT)),
                        "wave_file": str(wave_file.relative_to(ROOT)),
                    }
                )
    return rows


def write_manifest(monthly_plan):
    rows = manifest_rows(monthly_plan)
    if not rows:
        raise ValueError("Manifest would be empty.")
    fieldnames = list(rows[0].keys())
    with MANIFEST_FILE.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print("Manifest written:", MANIFEST_FILE)
    print("Manifest rows   :", len(rows))
    return rows


# -----------------------------------------------------------------------------
# STREAM INPUT ONCE + BUILD PLAN
# -----------------------------------------------------------------------------

def scan_monthly_plan(csv_path):
    if not csv_path.is_file():
        raise FileNotFoundError(f"Tanker CSV not found: {csv_path}")

    set_csv_field_limit()
    stats = defaultdict(
        lambda: {
            "rows": 0,
            "tiles": set(),
            "tile_days": defaultdict(set),
        }
    )

    total = valid_total = bad_time = bad_position = short_rows = 0
    global_t_min = global_t_max = None
    global_lat_min = global_lat_max = None

    with csv_path.open("r", encoding="utf-8-sig", errors="replace", newline="") as fh:
        reader = csv.reader(fh)
        try:
            header = next(reader)
        except StopIteration:
            raise ValueError("The tanker CSV is empty.")

        time_i, time_name = find_index(header, ["time", "datetime", "date_time", "timestamp", "utc", "date"])
        lat_i, lat_name = find_index(header, ["lat", "latitude"])
        lon_i, lon_name = find_index(header, ["lon", "lng", "longitude"])
        needed_i = max(time_i, lat_i, lon_i)

        print("Detected columns:")
        print("  time      :", time_name)
        print("  latitude  :", lat_name)
        print("  longitude :", lon_name)
        print("Scanning tanker CSV in streaming mode (very low memory)...")

        for row in reader:
            total += 1
            if len(row) <= needed_i:
                short_rows += 1
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

            valid_total += 1
            global_t_min = dt if global_t_min is None else min(global_t_min, dt)
            global_t_max = dt if global_t_max is None else max(global_t_max, dt)
            global_lat_min = lat if global_lat_min is None else min(global_lat_min, lat)
            global_lat_max = lat if global_lat_max is None else max(global_lat_max, lat)

            if total % PROGRESS_EVERY == 0:
                print(f"  scanned {total:,} rows; valid={valid_total:,}; months={len(stats)}")

    if valid_total == 0:
        raise ValueError("No valid timestamp + latitude + longitude rows were found.")

    print("\nOverall scan:")
    print("  total rows     :", f"{total:,}")
    print("  valid rows     :", f"{valid_total:,}")
    print("  bad timestamps :", f"{bad_time:,}")
    print("  bad positions  :", f"{bad_position:,}")
    print("  short rows     :", f"{short_rows:,}")
    print("  time range     :", global_t_min, "->", global_t_max)
    print("  latitude       :", global_lat_min, "->", global_lat_max)
    print("  months found   :", len(stats))
    print("  ERA5 grid      :", GRID)
    print("  surface vars   : u10 + v10 + surface pressure + SST")
    print("  wave vars      : SWH + mean wave direction + mean wave period")
    print("  2m temperature : NOT REQUESTED\n")

    monthly_plan = {}
    total_parts = total_part_days = total_slices = 0
    total_occupied_tiles = total_box_tiles = 0

    print("Monthly ERA5 request plan:")
    print("-" * 108)
    print(
        f"Adaptive strategy: {TILE_LON_DEG:.0f}x{TILE_LAT_DEG:.0f} deg planning tiles; "
        f"max box ~{MAX_REQUEST_LON_SPAN_DEG:.0f}x{MAX_REQUEST_LAT_SPAN_DEG:.0f} deg; "
        f"min fill={MIN_BOX_FILL:.0%}; padding={PADDING_DEG:.1f} deg; day padding=+/-{DAY_PADDING_DAYS}"
    )
    print("-" * 108)

    for (year, month) in sorted(stats):
        s = stats[(year, month)]
        parts = build_month_parts(s, year, month)
        monthly_plan[(year, month)] = parts

        total_parts += len(parts)
        total_occupied_tiles += len(s["tiles"])
        month_box_tiles = 0
        month_part_days = 0

        for p in parts:
            h, w = box_shape(p["rect"])
            month_box_tiles += h * w
            month_part_days += len(p["dates"])
            total_slices += len(p["request_groups"])

        total_box_tiles += month_box_tiles
        total_part_days += month_part_days
        fill = len(s["tiles"]) / month_box_tiles if month_box_tiles else 1.0

        print(
            f"{year}-{month:02d}: rows={s['rows']:,}, occupied tiles={len(s['tiles'])}, "
            f"part(s)={len(parts)}, coarse-box fill~{fill:.0%}, summed part-days={month_part_days}"
        )
        for i, p in enumerate(parts, 1):
            print(
                f"    part{i}: area={p['area']}, dates={compact_request_groups(p['request_groups'])} "
                f"({len(p['dates'])} day(s), {len(p['request_groups'])} month-slice(s))"
            )

    print("-" * 108)
    overall_fill = total_occupied_tiles / total_box_tiles if total_box_tiles else 1.0
    print(f"Total monthly spatial parts : {total_parts}")
    print(f"Overall coarse-box fill     : {overall_fill:.1%}")
    print(f"Summed spatial-part days    : {total_part_days}")
    print(f"Monthly request slices      : {total_slices}")
    print(f"Total CDS downloads planned : {total_slices * 2} (surface + wave)")
    return monthly_plan


# -----------------------------------------------------------------------------
# CDS DOWNLOAD
# -----------------------------------------------------------------------------

def request_one_area(client, variables, target, request_year, request_month, days, area, label):
    temp = Path(str(target) + ".part")
    if target.exists() and target.stat().st_size > 0:
        print("Skip:", target)
        return True
    if temp.exists():
        temp.unlink()

    request = {
        "product_type": ["reanalysis"],
        "variable": variables,
        "year": [str(request_year)],
        "month": [f"{request_month:02d}"],
        "day": [f"{d:02d}" for d in days],
        "time": TIMES,
        "area": area,
        "grid": GRID,
        "data_format": "netcdf",
        "download_format": "unarchived",
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(
                f"Request {label} {request_year}-{request_month:02d} days={compact_days(days)} "
                f"(attempt {attempt}/{MAX_RETRIES})"
            )
            print("  area:", area)
            client.retrieve("reanalysis-era5-single-levels", request, str(temp))
            os.replace(temp, target)
            print("Done:", target)
            return True
        except Exception as exc:
            print(f"Failed {label}: {exc}")
            if temp.exists():
                try:
                    temp.unlink()
                except OSError:
                    pass
            if attempt == MAX_RETRIES:
                print("GIVE UP for now; re-run later. Completed files will be skipped.")
                return False
            print(f"Retrying in {RETRY_WAIT_SECONDS} seconds...")
            time.sleep(RETRY_WAIT_SECONDS)
    return False


def main():
    monthly_plan = scan_monthly_plan(DATA_FILE)
    rows = write_manifest(monthly_plan)

    if CONFIRM_BEFORE_DOWNLOAD:
        answer = input(
            "\nReview the FINAL SURFACE/WAVE adaptive request plan and manifest above. "
            "Type YES to start ERA5 downloads: "
        ).strip()
        if answer != "YES":
            print("Stopped before download. Manifest was kept; no CDS request was submitted.")
            return

    client = cdsapi.Client()
    failures = 0

    for row in rows:
        area = [float(row["north"]), float(row["west"]), float(row["south"]), float(row["east"])]
        days = []
        for token in str(row["days"]).split(","):
            if not token:
                continue
            if "-" in token:
                a, b = map(int, token.split("-"))
                days.extend(range(a, b + 1))
            else:
                days.append(int(token))

        sy = int(row["request_year"])
        sm = int(row["request_month"])
        py = int(row["plan_year"])
        pm = int(row["plan_month"])
        part = int(row["part"])

        ok1 = request_one_area(
            client, SURFACE, ROOT / Path(row["surface_file"]), sy, sm, days, area,
            f"surface plan={py}-{pm:02d} part={part}",
        )
        ok2 = request_one_area(
            client, WAVE, ROOT / Path(row["wave_file"]), sy, sm, days, area,
            f"wave plan={py}-{pm:02d} part={part}",
        )
        failures += int(not ok1) + int(not ok2)

    print("\nFinished:", ROOT)
    print("Surface :", SURFACE_DIR)
    print("Wave    :", WAVE_DIR)
    print("Manifest:", MANIFEST_FILE)
    print("Failed download items:", failures)
    if failures:
        print("Re-run the script later; existing completed files will be skipped.")


if __name__ == "__main__":
    main()
