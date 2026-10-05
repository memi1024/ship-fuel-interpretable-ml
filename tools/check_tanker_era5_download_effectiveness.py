# -*- coding: utf-8 -*-
"""
Check ERA5 download validity/completeness for tanker adaptive downloads.

Metrics:
1) Manifest file completion:
   - expected surface files
   - expected wave files
   - valid existing files
   - paired surface+wave completion
2) NetCDF structural validity:
   - file exists
   - non-zero size
   - xarray can open it
   - expected variables are present
   - at least one time value exists
3) Fixed31 tanker+cruise record coverage:
   - uses the manifest planning tile assignment
   - checks whether both temporal bracketing hours are covered by valid
     surface + wave files for the routed spatial part
   - reports how many target records are currently matchable

This script DOES NOT download anything and DOES NOT modify the CSV/NetCDF files.
"""

import ast
import math
import os
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd
import xarray as xr


# ---------------------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
ERA5_ROOT = Path(os.environ.get(
    "SHIP_FUEL_TANKER_ERA5_ROOT",
    str(REPO_ROOT / "data" / "era5" / "tanker"),
))
MANIFEST = ERA5_ROOT / "era5_request_manifest.csv"

TARGET_CSV = Path(os.environ.get(
    "SHIP_FUEL_TANKER_FIXED31_COORDS",
    str(REPO_ROOT / "data" / "06_final_fixed31_with_coordinates.csv"),
))

# Only evaluate final tanker cruise records.
SHIP_TYPE_KEYWORD = "tanker"
VOYAGE_PHASE = "cruise"

TIME_COL = "timestamp_utc"
LAT_COL = "latitude_deg"
LON_COL = "longitude_deg"

# Must match the downloader planning grid.
TILE_LON_DEG = 10.0
TILE_LAT_DEG = 10.0

SURFACE_EXPECTED = {
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "surface_pressure",
    "sea_surface_temperature",
}

WAVE_EXPECTED = {
    "significant_height_of_combined_wind_waves_and_swell",
    "mean_wave_direction",
    "mean_wave_period",
}

MIN_FILE_BYTES = 1024


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------

def norm_lon(lon):
    return ((float(lon) + 180.0) % 360.0) - 180.0


def month_key(ts):
    return int(ts.year), int(ts.month)


def tile_indices(lat, lon):
    """
    Return planning-tile integer indices consistent with a global
    10x10 degree grid starting at lat=-90, lon=-180.
    """
    lon = norm_lon(lon)
    lat = float(lat)

    # Clamp exact north/east boundary values into the last valid tile.
    if lat >= 90:
        lat = np.nextafter(90.0, -np.inf)
    if lon >= 180:
        lon = np.nextafter(180.0, -np.inf)

    lat_i = int(math.floor((lat + 90.0) / TILE_LAT_DEG))
    lon_i = int(math.floor((lon + 180.0) / TILE_LON_DEG))
    return lat_i, lon_i


def parse_tile_cells(value):
    """
    Manifest may store tile_cells in several textual forms.
    Return a set of (lat_i, lon_i).
    """
    if pd.isna(value):
        return set()

    s = str(value).strip()
    if not s:
        return set()

    # Preferred: Python-literal list/tuple.
    try:
        obj = ast.literal_eval(s)
        out = set()
        if isinstance(obj, (list, tuple, set)):
            for item in obj:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    out.add((int(item[0]), int(item[1])))
                elif isinstance(item, str):
                    parts = item.replace("(", "").replace(")", "").split(",")
                    if len(parts) >= 2:
                        out.add((int(parts[0]), int(parts[1])))
        if out:
            return out
    except Exception:
        pass

    # Fallback formats such as "3:12;3:13" or "3,12;3,13".
    out = set()
    for token in s.replace("|", ";").split(";"):
        token = token.strip().replace("(", "").replace(")", "")
        if not token:
            continue
        if ":" in token:
            a, b = token.split(":", 1)
        elif "," in token:
            a, b = token.split(",", 1)
        else:
            continue
        try:
            out.add((int(a.strip()), int(b.strip())))
        except Exception:
            pass
    return out


def parse_days(value):
    """
    Convert a manifest days field such as:
      01-17,21-30
      31
      01-30
    to a set of integer days.
    """
    if pd.isna(value):
        return set()

    s = str(value).strip()
    out = set()
    if not s:
        return out

    for piece in s.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "-" in piece:
            a, b = piece.split("-", 1)
            try:
                a, b = int(a), int(b)
                out.update(range(a, b + 1))
            except Exception:
                pass
        else:
            try:
                out.add(int(piece))
            except Exception:
                pass
    return out


def resolve_file_path(value):
    if pd.isna(value):
        return None
    p = Path(str(value))
    if not p.is_absolute():
        p = ERA5_ROOT / p
    return p


def detect_time_name(ds):
    candidates = ["valid_time", "time", "datetime", "date"]
    for c in candidates:
        if c in ds.coords or c in ds.variables:
            return c
    # Some CDS NetCDFs can use forecast reference style names.
    for c in ds.coords:
        if "time" in c.lower():
            return c
    return None


def canonical_var_names(ds):
    """
    Return both raw data_var names and long/standard metadata names to make
    CDS variable naming checks more tolerant.
    """
    names = set(ds.data_vars)
    for name, da in ds.data_vars.items():
        for key in ("long_name", "standard_name", "short_name"):
            v = da.attrs.get(key)
            if v:
                names.add(str(v))
    return names


def variable_group_ok(ds, expected):
    """
    CDS NetCDF short names often differ from API request names:
      u10, v10, sp, sst, swh, mwd, mwp
    Therefore accept either API-style or standard short names.
    """
    raw = set(ds.data_vars)

    aliases = {
        "10m_u_component_of_wind": {"10m_u_component_of_wind", "u10"},
        "10m_v_component_of_wind": {"10m_v_component_of_wind", "v10"},
        "surface_pressure": {"surface_pressure", "sp"},
        "sea_surface_temperature": {"sea_surface_temperature", "sst"},
        "significant_height_of_combined_wind_waves_and_swell": {
            "significant_height_of_combined_wind_waves_and_swell", "swh"
        },
        "mean_wave_direction": {"mean_wave_direction", "mwd"},
        "mean_wave_period": {"mean_wave_period", "mwp"},
    }

    missing = []
    for e in expected:
        if not (raw & aliases[e]):
            missing.append(e)
    return len(missing) == 0, missing


def inspect_nc(path, expected_group):
    result = {
        "exists": False,
        "size_bytes": 0,
        "open_ok": False,
        "vars_ok": False,
        "time_ok": False,
        "valid": False,
        "error": "",
    }

    if path is None:
        result["error"] = "no path"
        return result

    if not path.exists():
        result["error"] = "missing"
        return result

    result["exists"] = True

    try:
        size = path.stat().st_size
        result["size_bytes"] = size
        if size < MIN_FILE_BYTES:
            result["error"] = f"too small ({size} bytes)"
            return result
    except Exception as exc:
        result["error"] = f"stat failed: {exc}"
        return result

    try:
        with xr.open_dataset(path, decode_times=True) as ds:
            # Trigger light metadata read only, not full arrays.
            result["open_ok"] = True

            ok, missing = variable_group_ok(ds, expected_group)
            result["vars_ok"] = ok
            if not ok:
                result["error"] = "missing vars: " + ",".join(missing)

            tname = detect_time_name(ds)
            if tname is not None:
                try:
                    result["time_ok"] = int(ds[tname].size) > 0
                except Exception:
                    result["time_ok"] = True

            result["valid"] = (
                result["open_ok"]
                and result["vars_ok"]
                and result["time_ok"]
            )

    except Exception as exc:
        result["error"] = f"open failed: {exc}"

    return result


def choose_col(df, candidates, required=True):
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    if required:
        raise ValueError(
            f"Could not find any of {candidates}. "
            f"Available columns include: {list(df.columns)[:80]}"
        )
    return None


# ---------------------------------------------------------------------
# MANIFEST VALIDATION
# ---------------------------------------------------------------------

def validate_manifest():
    if not MANIFEST.is_file():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST}")

    m = pd.read_csv(MANIFEST)

    # Flexible names, in case downloader naming evolved.
    surface_col = choose_col(
        m,
        ["surface_file", "surface_path", "surface_target"]
    )
    wave_col = choose_col(
        m,
        ["wave_file", "wave_path", "wave_target"]
    )

    plan_year_col = choose_col(m, ["plan_year"])
    plan_month_col = choose_col(m, ["plan_month"])
    part_col = choose_col(m, ["part", "part_number"])
    request_year_col = choose_col(m, ["request_year"])
    request_month_col = choose_col(m, ["request_month"])
    days_col = choose_col(m, ["days", "request_days"])
    tile_col = choose_col(m, ["tile_cells"], required=False)

    rows = []

    print("\nChecking NetCDF files listed in manifest...")
    print("-" * 88)

    cache = {}

    for i, r in m.iterrows():
        spath = resolve_file_path(r[surface_col])
        wpath = resolve_file_path(r[wave_col])

        skey = str(spath)
        wkey = str(wpath)

        if skey not in cache:
            cache[skey] = inspect_nc(spath, SURFACE_EXPECTED)
        if wkey not in cache:
            cache[wkey] = inspect_nc(wpath, WAVE_EXPECTED)

        sres = cache[skey]
        wres = cache[wkey]

        row = {
            "manifest_row": i + 1,
            "plan_year": int(r[plan_year_col]),
            "plan_month": int(r[plan_month_col]),
            "part": int(r[part_col]),
            "request_year": int(r[request_year_col]),
            "request_month": int(r[request_month_col]),
            "days": str(r[days_col]),
            "tile_cells": r[tile_col] if tile_col else "",
            "surface_file": str(spath),
            "wave_file": str(wpath),
            "surface_exists": sres["exists"],
            "surface_valid": sres["valid"],
            "surface_size_mb": sres["size_bytes"] / (1024 ** 2),
            "surface_error": sres["error"],
            "wave_exists": wres["exists"],
            "wave_valid": wres["valid"],
            "wave_size_mb": wres["size_bytes"] / (1024 ** 2),
            "wave_error": wres["error"],
            "pair_valid": sres["valid"] and wres["valid"],
        }
        rows.append(row)

    status = pd.DataFrame(rows)

    total = len(status)
    svalid = int(status["surface_valid"].sum())
    wvalid = int(status["wave_valid"].sum())
    pvalid = int(status["pair_valid"].sum())

    print(f"Manifest slices          : {total:,}")
    print(f"Valid surface files      : {svalid:,}/{total:,} = {100*svalid/total:.1f}%")
    print(f"Valid wave files         : {wvalid:,}/{total:,} = {100*wvalid/total:.1f}%")
    print(f"Valid paired slices      : {pvalid:,}/{total:,} = {100*pvalid/total:.1f}%")

    surface_mb = status.loc[status["surface_valid"], "surface_size_mb"].sum()
    wave_mb = status.loc[status["wave_valid"], "wave_size_mb"].sum()
    print(f"Valid downloaded size    : {(surface_mb + wave_mb)/1024:.2f} GB")
    print(f"  surface                : {surface_mb/1024:.2f} GB")
    print(f"  wave                   : {wave_mb/1024:.2f} GB")

    return m, status, {
        "surface_col": surface_col,
        "wave_col": wave_col,
        "plan_year_col": plan_year_col,
        "plan_month_col": plan_month_col,
        "part_col": part_col,
        "request_year_col": request_year_col,
        "request_month_col": request_month_col,
        "days_col": days_col,
        "tile_col": tile_col,
    }


# ---------------------------------------------------------------------
# TARGET RECORD COVERAGE
# ---------------------------------------------------------------------

def build_part_index(manifest, cols, status):
    """
    Build:
      (plan_year, plan_month, tile_lat_i, tile_lon_i) -> part
    and a per-part month/day validity lookup.
    """
    tile_col = cols["tile_col"]
    if not tile_col:
        raise ValueError(
            "Manifest does not contain tile_cells, so exact fixed31 routing "
            "cannot be checked."
        )

    route = {}
    validity = {}

    status_by_row = status.set_index("manifest_row")

    for idx, r in manifest.iterrows():
        py = int(r[cols["plan_year_col"]])
        pm = int(r[cols["plan_month_col"]])
        part = int(r[cols["part_col"]])

        cells = parse_tile_cells(r[tile_col])
        for cell in cells:
            key = (py, pm, int(cell[0]), int(cell[1]))
            if key in route and route[key] != part:
                # This should not normally occur if planning tiles were assigned uniquely.
                route[key] = None
            else:
                route[key] = part

        ry = int(r[cols["request_year_col"]])
        rm = int(r[cols["request_month_col"]])
        days = parse_days(r[cols["days_col"]])

        s = status_by_row.loc[idx + 1]
        pair_valid = bool(s["pair_valid"])

        validity[(py, pm, part, ry, rm)] = {
            "days": days,
            "pair_valid": pair_valid,
        }

    return route, validity


def target_coverage(route, validity):
    if not TARGET_CSV.is_file():
        raise FileNotFoundError(f"Target CSV not found: {TARGET_CSV}")

    usecols = [
        "ship_type",
        TIME_COL,
        "voyage_phase",
        LAT_COL,
        LON_COL,
    ]

    # Read only columns needed for this coverage check.
    df = pd.read_csv(TARGET_CSV, usecols=lambda c: c in usecols)

    needed = {TIME_COL, LAT_COL, LON_COL}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Target CSV missing required columns: {sorted(missing)}")

    if "ship_type" in df.columns:
        df = df[
            df["ship_type"].astype(str).str.contains(
                SHIP_TYPE_KEYWORD, case=False, na=False
            )
        ]

    if "voyage_phase" in df.columns:
        df = df[
            df["voyage_phase"].astype(str).str.strip().str.lower()
            == VOYAGE_PHASE.lower()
        ]

    df = df.copy()
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce", utc=True)
    df[LAT_COL] = pd.to_numeric(df[LAT_COL], errors="coerce")
    df[LON_COL] = pd.to_numeric(df[LON_COL], errors="coerce")

    valid_base = (
        df[TIME_COL].notna()
        & df[LAT_COL].between(-90, 90)
        & df[LON_COL].between(-360, 360)
    )
    df = df.loc[valid_base].copy()

    total = len(df)
    if total == 0:
        raise ValueError("No valid tanker+cruise rows found in target CSV.")

    # Strip timezone so year/month/day handling is straightforward.
    ts = df[TIME_COL].dt.tz_convert("UTC").dt.tz_localize(None)

    # Hourly linear interpolation needs floor(t) and ceil(t).
    t0 = ts.dt.floor("h")
    t1 = ts.dt.ceil("h")
    # Exact-hour record only needs that hour; using same value is fine.
    t1 = t1.where(t1 != t0, t0)

    routed = 0
    matchable = 0
    no_route = 0
    missing_time_slice = 0

    monthly = {}

    for lat, lon, obs_t, a, b in zip(
        df[LAT_COL].to_numpy(),
        df[LON_COL].to_numpy(),
        ts.to_numpy(),
        t0.to_numpy(),
        t1.to_numpy(),
    ):
        obs_t = pd.Timestamp(obs_t)
        a = pd.Timestamp(a)
        b = pd.Timestamp(b)

        py, pm = obs_t.year, obs_t.month
        ti, tj = tile_indices(lat, lon)

        part = route.get((py, pm, ti, tj))
        mk = f"{py:04d}-{pm:02d}"
        if mk not in monthly:
            monthly[mk] = {
                "rows": 0,
                "routed": 0,
                "matchable": 0,
                "no_route": 0,
                "missing_time_slice": 0,
            }
        monthly[mk]["rows"] += 1

        if part is None:
            no_route += 1
            monthly[mk]["no_route"] += 1
            continue

        routed += 1
        monthly[mk]["routed"] += 1

        needed_hours = [a] if a == b else [a, b]
        ok = True

        for h in needed_hours:
            key = (py, pm, int(part), int(h.year), int(h.month))
            info = validity.get(key)
            if (
                info is None
                or not info["pair_valid"]
                or int(h.day) not in info["days"]
            ):
                ok = False
                break

        if ok:
            matchable += 1
            monthly[mk]["matchable"] += 1
        else:
            missing_time_slice += 1
            monthly[mk]["missing_time_slice"] += 1

    coverage = 100.0 * matchable / total
    route_pct = 100.0 * routed / total

    print("\nFixed31 tanker+cruise CURRENTLY MATCHABLE coverage")
    print("-" * 88)
    print(f"Valid target rows        : {total:,}")
    print(f"Routed to manifest part  : {routed:,}/{total:,} = {route_pct:.2f}%")
    print(f"Currently matchable      : {matchable:,}/{total:,} = {coverage:.2f}%")
    print(f"No manifest route        : {no_route:,}")
    print(f"Missing downloaded slice : {missing_time_slice:,}")

    mdf = (
        pd.DataFrame.from_dict(monthly, orient="index")
        .reset_index()
        .rename(columns={"index": "month"})
        .sort_values("month")
    )
    mdf["route_pct"] = 100 * mdf["routed"] / mdf["rows"]
    mdf["matchable_pct"] = 100 * mdf["matchable"] / mdf["rows"]

    print("\nMonthly target coverage:")
    print("-" * 88)
    for _, r in mdf.iterrows():
        print(
            f"{r['month']}: rows={int(r['rows']):,}, "
            f"routed={r['route_pct']:.1f}%, "
            f"matchable={r['matchable_pct']:.1f}%"
        )

    return mdf, {
        "target_rows": total,
        "routed": routed,
        "matchable": matchable,
        "route_pct": route_pct,
        "matchable_pct": coverage,
        "no_route": no_route,
        "missing_time_slice": missing_time_slice,
    }


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    print("=" * 88)
    print("TANKER ERA5 DOWNLOAD VALIDITY / COVERAGE CHECK")
    print("=" * 88)
    print("ERA5 root :", ERA5_ROOT)
    print("Manifest  :", MANIFEST)
    print("Target CSV:", TARGET_CSV)
    print("Mode      : CHECK ONLY -- NO DOWNLOAD, NO FILE MODIFICATION")

    manifest, status, cols = validate_manifest()

    # Save slice-level validation report.
    status_path = ERA5_ROOT / "era5_download_validity_report.csv"
    status.to_csv(status_path, index=False, encoding="utf-8-sig")
    print("\nSlice validity report:")
    print(" ", status_path)

    try:
        route, validity = build_part_index(manifest, cols, status)
        monthly_cov, summary = target_coverage(route, validity)

        monthly_path = ERA5_ROOT / "era5_fixed31_cruise_coverage_by_month.csv"
        monthly_cov.to_csv(monthly_path, index=False, encoding="utf-8-sig")
        print("\nMonthly coverage report:")
        print(" ", monthly_path)

        print("\n" + "=" * 88)
        print("KEY RESULT")
        print("=" * 88)
        print(
            f"Current ERA5 data can support "
            f"{summary['matchable_pct']:.2f}% of valid fixed31 tanker+cruise rows "
            f"for hourly spatiotemporal interpolation."
        )
        print(
            "This is the most useful 'effective rate' while the download is still incomplete."
        )

    except Exception as exc:
        print("\nTarget-record coverage could not be calculated:")
        print(" ", exc)
        print(
            "The manifest/file-completion report above is still valid. "
            "If needed, inspect the manifest column names/tile_cells format."
        )

    print("\nFinished. No CDS request was submitted.")


if __name__ == "__main__":
    main()
