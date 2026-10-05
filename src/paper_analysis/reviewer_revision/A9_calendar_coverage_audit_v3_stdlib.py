#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
A9 v3 (stdlib-only) — exact L2/L4 calendar coverage audit.

Why this version exists
-----------------------
This version deliberately avoids NumPy, pandas, OpenBLAS, MKL and other native
numerical libraries. It is intended for Windows systems where importing
NumPy/pandas can fail with an OpenBLAS memory-allocation error.

It reads only three CSV columns:
- vessel ID
- ship type
- timestamp

It then reproduces the manuscript's chronological split exactly:
    cut = floor(n_records * 0.80), constrained to [1, n-1]

For each vessel:
- first 80% records = L2 temporal-training contribution
                   = L4 target-vessel history
- final 20% records = L2/L4 target-vessel test period

No model is trained. No source file is modified.
"""

from __future__ import annotations

import argparse
import csv
import math
from datetime import datetime, timezone
from pathlib import Path


EXPECTED_ROWS = 489_620
EXPECTED_VESSELS = 21

VESSEL_CANDIDATES = (
    "vessel_id",
    "pseudo_ship_group_id",
    "ship_id",
)

TIME_CANDIDATES = (
    "timestamp",
    "timestamp_utc",
    "time",
    "datetime",
)

TYPE_CANDIDATES = (
    "ship_type",
    "vessel_type",
)


def norm_name(value: str) -> str:
    return str(value).strip().lower()


def resolve_column(fieldnames, candidates, label):
    lookup = {norm_name(name): name for name in fieldnames}
    for candidate in candidates:
        if norm_name(candidate) in lookup:
            return lookup[norm_name(candidate)]
    raise KeyError(
        f"Could not resolve {label}. Tried {list(candidates)}. "
        f"Available columns include: {fieldnames[:40]}"
    )


def parse_timestamp(value: str) -> datetime:
    text = str(value).strip()
    if not text:
        raise ValueError("blank timestamp")

    # Normalise common UTC suffix.
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"

    # First try Python ISO parser.
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        dt = None

    # Common fallbacks used in CSV exports.
    if dt is None:
        formats = (
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
            "%Y/%m/%d %H:%M:%S",
            "%Y/%m/%d %H:%M",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%S.%f",
        )
        for fmt in formats:
            try:
                dt = datetime.strptime(text, fmt)
                break
            except ValueError:
                pass

    if dt is None:
        raise ValueError(f"unrecognised timestamp: {text!r}")

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)

    return dt


def iso_utc(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def elapsed_days(start: datetime, end: datetime) -> float:
    return (end - start).total_seconds() / 86400.0


def inclusive_calendar_days(start: datetime, end: datetime) -> int:
    return (end.date() - start.date()).days + 1


def median(values):
    vals = sorted(float(v) for v in values)
    n = len(vals)
    if n == 0:
        return ""
    middle = n // 2
    if n % 2:
        return vals[middle]
    return (vals[middle - 1] + vals[middle]) / 2.0


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument(
        "--fixed31-cruise",
        required=True,
        help="Path to final_fixed31_cruise.csv",
    )
    p.add_argument(
        "--output-dir",
        default="revision_runs/A9_calendar_coverage",
    )
    p.add_argument(
        "--train-fraction",
        type=float,
        default=0.80,
    )
    p.add_argument(
        "--allow-noncanonical-counts",
        action="store_true",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if not (0.0 < args.train_fraction < 1.0):
        raise ValueError("--train-fraction must be between 0 and 1.")

    source = Path(args.fixed31_cruise)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not source.is_file():
        raise FileNotFoundError(f"Input file not found: {source}")

    # vessel -> {"ship_type": str, "timestamps": [datetime, ...]}
    vessels = {}
    row_count = 0
    invalid_timestamp_rows = 0

    print("Reading Fixed31 cruise CSV with Python standard library only...")
    print(f"Source: {source}")

    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)

        if not reader.fieldnames:
            raise RuntimeError("CSV header is missing.")

        vessel_col = resolve_column(
            reader.fieldnames, VESSEL_CANDIDATES, "vessel ID"
        )
        type_col = resolve_column(
            reader.fieldnames, TYPE_CANDIDATES, "ship type"
        )
        time_col = resolve_column(
            reader.fieldnames, TIME_CANDIDATES, "timestamp"
        )

        print("Resolved columns:")
        print(f"  vessel ID : {vessel_col}")
        print(f"  ship type : {type_col}")
        print(f"  timestamp : {time_col}")

        for row_number, row in enumerate(reader, start=2):
            row_count += 1

            vessel = str(row.get(vessel_col, "")).strip()
            ship_type = str(row.get(type_col, "")).strip().lower()

            if not vessel:
                raise RuntimeError(
                    f"Blank vessel ID at CSV row {row_number}."
                )
            if not ship_type:
                raise RuntimeError(
                    f"Blank ship type at CSV row {row_number}."
                )

            try:
                timestamp = parse_timestamp(row.get(time_col, ""))
            except Exception as exc:
                invalid_timestamp_rows += 1
                raise RuntimeError(
                    f"Invalid timestamp at CSV row {row_number}: "
                    f"{row.get(time_col, '')!r}. Error: {exc}"
                ) from exc

            entry = vessels.setdefault(
                vessel,
                {
                    "ship_type": ship_type,
                    "timestamps": [],
                },
            )

            if entry["ship_type"] != ship_type:
                raise RuntimeError(
                    f"Vessel {vessel!r} maps to more than one ship type: "
                    f"{entry['ship_type']!r}, {ship_type!r}"
                )

            entry["timestamps"].append(timestamp)

            if row_count % 100_000 == 0:
                print(f"  read {row_count:,} rows...")

    print(f"Finished reading {row_count:,} rows.")

    if not args.allow_noncanonical_counts:
        if row_count != EXPECTED_ROWS:
            raise AssertionError(
                f"Expected {EXPECTED_ROWS:,} rows, found {row_count:,}."
            )
        if len(vessels) != EXPECTED_VESSELS:
            raise AssertionError(
                f"Expected {EXPECTED_VESSELS} vessels, found {len(vessels)}."
            )

    by_vessel_rows = []

    for vessel in sorted(vessels):
        ship_type = vessels[vessel]["ship_type"]
        times = vessels[vessel]["timestamps"]
        times.sort()

        n_total = len(times)
        if n_total < 2:
            raise RuntimeError(
                f"Vessel {vessel!r} has fewer than two observations."
            )

        cut = max(
            1,
            min(
                n_total - 1,
                int(math.floor(n_total * args.train_fraction)),
            ),
        )

        first80 = times[:cut]
        last20 = times[cut:]

        full_start = times[0]
        full_end = times[-1]

        first80_start = first80[0]
        first80_end = first80[-1]

        last20_start = last20[0]
        last20_end = last20[-1]

        by_vessel_rows.append({
            "vessel_id": vessel,
            "ship_type": ship_type,
            "n_total": n_total,

            "full_start_utc": iso_utc(full_start),
            "full_end_utc": iso_utc(full_end),
            "full_elapsed_days": elapsed_days(full_start, full_end),
            "full_inclusive_calendar_days":
                inclusive_calendar_days(full_start, full_end),

            "first80_n": len(first80),
            "first80_pct_records": 100.0 * len(first80) / n_total,
            "first80_start_utc": iso_utc(first80_start),
            "first80_end_utc": iso_utc(first80_end),
            "first80_elapsed_days":
                elapsed_days(first80_start, first80_end),
            "first80_inclusive_calendar_days":
                inclusive_calendar_days(first80_start, first80_end),

            "last20_n": len(last20),
            "last20_pct_records": 100.0 * len(last20) / n_total,
            "last20_start_utc": iso_utc(last20_start),
            "last20_end_utc": iso_utc(last20_end),
            "last20_elapsed_days":
                elapsed_days(last20_start, last20_end),
            "last20_inclusive_calendar_days":
                inclusive_calendar_days(last20_start, last20_end),
        })

    by_vessel_rows.sort(
        key=lambda r: (r["ship_type"], r["vessel_id"])
    )

    by_vessel_path = (
        output_dir / "A9_L2_L4_calendar_coverage_by_vessel.csv"
    )

    by_vessel_fields = list(by_vessel_rows[0].keys())

    with by_vessel_path.open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=by_vessel_fields
        )
        writer.writeheader()
        writer.writerows(by_vessel_rows)

    # Fleet + ship-type summary
    scopes = [("Fleet", by_vessel_rows)]

    ship_types = sorted(
        {row["ship_type"] for row in by_vessel_rows}
    )

    for ship_type in ship_types:
        scopes.append(
            (
                ship_type,
                [
                    row
                    for row in by_vessel_rows
                    if row["ship_type"] == ship_type
                ],
            )
        )

    summary_rows = []

    for scope_name, rows in scopes:
        summary_rows.append({
            "scope": scope_name,
            "vessels": len(rows),
            "earliest_full_start_utc":
                min(row["full_start_utc"] for row in rows),
            "latest_full_end_utc":
                max(row["full_end_utc"] for row in rows),

            "full_elapsed_days_median":
                median(row["full_elapsed_days"] for row in rows),
            "full_elapsed_days_min":
                min(row["full_elapsed_days"] for row in rows),
            "full_elapsed_days_max":
                max(row["full_elapsed_days"] for row in rows),

            "first80_elapsed_days_median":
                median(row["first80_elapsed_days"] for row in rows),
            "first80_elapsed_days_min":
                min(row["first80_elapsed_days"] for row in rows),
            "first80_elapsed_days_max":
                max(row["first80_elapsed_days"] for row in rows),

            "last20_elapsed_days_median":
                median(row["last20_elapsed_days"] for row in rows),
            "last20_elapsed_days_min":
                min(row["last20_elapsed_days"] for row in rows),
            "last20_elapsed_days_max":
                max(row["last20_elapsed_days"] for row in rows),

            "first80_records_total":
                sum(row["first80_n"] for row in rows),
            "last20_records_total":
                sum(row["last20_n"] for row in rows),
        })

    summary_path = (
        output_dir / "A9_L2_L4_calendar_coverage_summary.csv"
    )

    with summary_path.open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(summary_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    type_counts = {
        ship_type: sum(
            1
            for row in by_vessel_rows
            if row["ship_type"] == ship_type
        )
        for ship_type in ship_types
    }

    manifest_row = {
        "source_file": str(source.resolve()),
        "rows": row_count,
        "vessels": len(vessels),
        "bulk_vessels": type_counts.get("bulk", 0),
        "container_vessels": type_counts.get("container", 0),
        "tanker_vessels": type_counts.get("tanker", 0),
        "train_fraction": args.train_fraction,
        "vessel_column": vessel_col,
        "ship_type_column": type_col,
        "timestamp_column": time_col,
        "invalid_timestamp_rows": invalid_timestamp_rows,
        "implementation": "python_stdlib_only_no_numpy_no_pandas",
        "status": "PASS",
    }

    manifest_path = (
        output_dir / "A9_calendar_audit_manifest.csv"
    )

    with manifest_path.open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(manifest_row.keys())
        )
        writer.writeheader()
        writer.writerow(manifest_row)

    print()
    print("=" * 78)
    print("A9 L2/L4 CALENDAR COVERAGE AUDIT: PASS")
    print("=" * 78)
    print(f"Rows        : {row_count:,}")
    print(f"Vessels     : {len(vessels)}")
    print(f"Output dir  : {output_dir.resolve()}")
    print()
    print("Files written:")
    print(f"  {by_vessel_path}")
    print(f"  {summary_path}")
    print(f"  {manifest_path}")
    print()
    print("Fleet summary:")
    fleet = summary_rows[0]
    print(
        "  first-80% calendar duration, median/min/max days = "
        f"{fleet['first80_elapsed_days_median']:.3f} / "
        f"{fleet['first80_elapsed_days_min']:.3f} / "
        f"{fleet['first80_elapsed_days_max']:.3f}"
    )


if __name__ == "__main__":
    main()
