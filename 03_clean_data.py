# -*- coding: utf-8 -*-
"""Stage 3: standardise source data and construct the Fixed31 modelling dataset.

Sequence
--------
1. Heterogeneous source standardisation and quality control.
2. Vessel identity, trajectory, 5-to-10-minute aggregation, feature construction, and merge.
3. Equal-phase quality checks and Fixed31 construction.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

V15_STEPS = [
    "init",
    "stage-container", "build-container",
    "stage-bulk", "build-bulk",
    "stage-tanker", "build-tanker",
    "feature-container", "feature-bulk", "feature-tanker",
    "merge", "clean23",
]


def run(cmd: list[str]) -> None:
    print("\n$", subprocess.list2cmdline(cmd), flush=True)
    subprocess.run(cmd, check=True)


def ensure_csv(source: Path, target_dir: Path) -> Path:
    if source.suffix.lower() == ".csv":
        return source
    if source.suffix.lower() not in {".xlsx", ".xls", ".xlsm"}:
        raise ValueError(f"Container interpolated file must be CSV/Excel: {source}")
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{source.stem}.csv"
    if target.exists() and target.stat().st_size > 0:
        print(f"[reuse] converted container CSV: {target}")
        return target
    print(f"[convert] {source} -> {target}")
    df = pd.read_excel(source)
    df.to_csv(target, index=False, encoding="utf-8-sig")
    return target


def first_stage_output(clean_dir: Path, ship_type: str, source: Path) -> Path:
    return clean_dir / f"clean_{ship_type}_{source.stem}.csv"



def run_synthetic_check() -> int:
    """Run a small in-memory execution check for the source-standardisation core."""
    cleaning_dir = Path(__file__).resolve().parent / "src" / "cleaning"
    sys.path.insert(0, str(cleaning_dir))
    from ship_data_cleaning import (
        CleaningStats, StreamState, create_standardized_chunk,
    )

    n = 24
    ts = pd.date_range("2024-01-01", periods=n, freq="10min", tz="UTC")

    container = pd.DataFrame({
        "ship_id": ["DEMO_CONTAINER"] * n,
        "ship type": ["container"] * n,
        "timestamp": ts,
        "DWT": [50000.0] * n,
        "ServiceSpeed": [18.0] * n,
        "SOG": np.linspace(10.0, 16.0, n),
        "course": np.linspace(0.0, 300.0, n),
        "heading": np.linspace(2.0, 302.0, n),
        "rudder": np.zeros(n),
        "艏吃水": np.full(n, 9.5),
        "艉吃水": np.full(n, 9.7),
        "wind_s": np.full(n, 12.0),
        "wind_d": np.full(n, 90.0),
        "wave_h": np.full(n, 1.2),
        "wave_p": np.full(n, 7.0),
        "wave_d": np.full(n, 110.0),
        "surface_p": np.full(n, 101300.0),
        "surface_t": np.full(n, 18.0),
        "fuel_rate_kg_h": np.full(n, 1200.0),
        "lat": np.linspace(20.0, 20.2, n),
        "lon": np.linspace(120.0, 120.2, n),
    })

    common = pd.DataFrame({
        "id": ["DEMO_BULK"] * n,
        "ShipType": ["bulk"] * n,
        "timestamp": ts,
        "Draught": [12.0] * n,
        "Deadweight": [80000.0] * n,
        "ServiceSpeed": [14.0] * n,
        "Total KW Main Eng": [10000.0] * n,
        "speed": np.linspace(9.0, 14.0, n),
        "direct": np.linspace(0.0, 300.0, n),
        "hdg": np.linspace(1.0, 301.0, n),
        "rudder": np.zeros(n),
        "df": np.full(n, 11.8),
        "da": np.full(n, 12.2),
        "dmp": np.full(n, 12.0),
        "trim": np.full(n, 0.4),
        "wind_s": np.full(n, 10.0),
        "wind_d": np.full(n, 90.0),
        "wave_h": np.full(n, 1.5),
        "wave_p": np.full(n, 7.0),
        "wave_d": np.full(n, 120.0),
        "surface_p": np.full(n, 1013.0),
        "surface_t": np.full(n, 290.0),
        "me_fo": np.full(n, 0.12),
        "lat": np.linspace(20.0, 20.2, n),
        "lon": np.linspace(120.0, 120.2, n),
    })

    outputs = {}
    for kind, frame in [("container", container), ("bulk", common)]:
        z = create_standardized_chunk(
            frame, kind, f"demo_{kind}.csv", StreamState(), "t/5min", None, CleaningStats()
        )
        outputs[kind] = z

    tanker = common.copy()
    tanker["id"] = "DEMO_TANKER"
    tanker["ShipType"] = "tanker"
    outputs["tanker"] = create_standardized_chunk(
        tanker, "tanker", "demo_tanker.csv", StreamState(), "t/5min", None, CleaningStats()
    )

    required = {
        "ship_type", "ship_id", "timestamp_utc", "speed_kn", "mean_draught_m",
        "rudder_deg", "wind_speed_kn", "wave_height_m", "surface_pressure_pa",
        "surface_temperature_c",
    }
    for kind, z in outputs.items():
        missing = sorted(required.difference(z.columns))
        if missing:
            raise AssertionError(f"{kind}: required standardised fields missing: {missing}")
        if len(z) != n:
            raise AssertionError(f"{kind}: row count changed during standardisation check")
        if not np.isfinite(pd.to_numeric(z["speed_kn"], errors="coerce")).all():
            raise AssertionError(f"{kind}: non-finite speed after standardisation")

    if not np.isfinite(outputs["container"]["fuel_t_10min"]).all():
        raise AssertionError("container: 10-min fuel conversion failed")
    for kind in ["bulk", "tanker"]:
        if not np.isfinite(outputs[kind]["fuel_t_5min"]).all():
            raise AssertionError(f"{kind}: 5-min source fuel conversion failed")

    print("STAGE 3 SYNTHETIC CHECK PASSED")
    print("Synthetic rows checked:", n * 3)
    print("No files were written.")
    return 0

def main() -> int:
    p = argparse.ArgumentParser(description="Run source cleaning -> V15.0 -> Fixed31 equal-phase cleaning.")
    p.add_argument("--synthetic-check", action="store_true", help="Run a small in-memory execution check.")
    here = Path(__file__).resolve().parent
    ref = here / "src" / "cleaning"
    p.add_argument("--code-dir", type=Path, default=ref)
    p.add_argument("--container-interpolated", type=Path, default=None,
                   help="Output from 02_interpolate_era5.py (Excel or CSV).")
    p.add_argument("--bulk", type=Path, default=None, help="Original bulk.csv.")
    p.add_argument("--tanker", type=Path, default=None, help="Original tank.csv; source weather is retained.")
    p.add_argument("--work-root", type=Path, default=None,
                   help="Working directory for converted/first-stage files.")
    p.add_argument("--paperdata-root", type=Path, default=None,
                   help="V15 output root containing 00_meta..05_clean23.")
    p.add_argument("--bulk-fuel-unit", choices=["t/5min", "kg/5min", "kg/h"], default="t/5min")
    p.add_argument("--tanker-fuel-unit", choices=["t/5min", "kg/5min", "kg/h"], default="t/5min")
    p.add_argument("--stage1-chunksize", type=int, default=200_000)
    p.add_argument("--v15-chunksize", type=int, default=20_000)
    p.add_argument("--v15-memory-limit", default="1500MB")
    p.add_argument("--v15-threads", type=int, default=1)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    if args.synthetic_check:
        return run_synthetic_check()

    required_cli = {
        "--container-interpolated": args.container_interpolated,
        "--bulk": args.bulk,
        "--tanker": args.tanker,
        "--work-root": args.work_root,
        "--paperdata-root": args.paperdata_root,
    }
    missing_cli = [name for name, value in required_cli.items() if value is None]
    if missing_cli:
        p.error("Stage 3 requires " + ", ".join(missing_cli))

    required_scripts = [
        "ship_data_cleaning.py",
        "three_ship_identity_clean23_v15_0.py",
        "clean_all_phases_equal_pipeline.py",
        "create_final_fixed28_quality_checked.py",
        "create_final_fixed29_model_ready.py",
        "audit_fixed29_physical_ranges.py",
        "create_final_fixed31_model_ready.py",
    ]
    for name in required_scripts:
        path = args.code_dir / name
        if not path.is_file():
            raise FileNotFoundError(path)
    for source in [args.container_interpolated, args.bulk, args.tanker]:
        if not source.is_file():
            raise FileNotFoundError(source)

    args.work_root.mkdir(parents=True, exist_ok=True)
    args.paperdata_root.mkdir(parents=True, exist_ok=True)
    input_dir = args.work_root / "00_inputs"
    clean_dir = args.work_root / "01_first_stage_cleaned"
    clean_dir.mkdir(parents=True, exist_ok=True)

    container_csv = ensure_csv(args.container_interpolated, input_dir)

    # ------------------------------------------------------------------
    # A. First-stage standardisation/QC
    # ------------------------------------------------------------------
    cleaner = args.code_dir / "ship_data_cleaning.py"
    stage1_specs = [
        ("container", container_csv, "t/5min"),
        ("bulk", args.bulk, args.bulk_fuel_unit),
        ("tanker", args.tanker, args.tanker_fuel_unit),
    ]
    for ship_type, source, unit in stage1_specs:
        out = first_stage_output(clean_dir, ship_type, source)
        if out.exists() and out.stat().st_size > 0 and not args.force:
            print(f"[reuse] first-stage {ship_type}: {out}")
            continue
        cmd = [
            sys.executable, str(cleaner),
            "--input", str(source),
            "--output-dir", str(clean_dir),
            "--ship-type", ship_type,
            "--chunksize", str(args.stage1_chunksize),
        ]
        if ship_type in {"bulk", "tanker"}:
            cmd += ["--me-fo-unit", unit]
        run(cmd)

    cleaned_container = first_stage_output(clean_dir, "container", container_csv)
    cleaned_bulk = first_stage_output(clean_dir, "bulk", args.bulk)
    cleaned_tanker = first_stage_output(clean_dir, "tanker", args.tanker)
    for path in [cleaned_container, cleaned_bulk, cleaned_tanker]:
        if not path.is_file():
            raise FileNotFoundError(path)

    # ------------------------------------------------------------------
    # B. Identity, aggregation, feature construction, Clean23 / Fixed27
    # ------------------------------------------------------------------
    v15 = args.code_dir / "three_ship_identity_clean23_v15_0.py"
    common = [
        "--output-root", str(args.paperdata_root),
        "--cleaned-container", str(cleaned_container),
        "--cleaned-bulk", str(cleaned_bulk),
        "--cleaned-tanker", str(cleaned_tanker),
        # Raw/reference sources preserve original-row/static-field lineage.
        "--raw-container", str(container_csv),
        "--raw-bulk", str(args.bulk),
        "--raw-tanker", str(args.tanker),
        "--chunksize", str(args.v15_chunksize),
        "--memory-limit", str(args.v15_memory_limit),
        "--threads", str(args.v15_threads),
    ]
    for step in V15_STEPS:
        cmd = [sys.executable, str(v15), "--step", step] + common
        if args.force:
            cmd.append("--force")
        run(cmd)

    clean23_dir = args.paperdata_root / "05_clean23"
    all_phases = clean23_dir / "final_fixed23_all_phases.csv"
    fixed27_cruise = clean23_dir / "final_fixed27_cruise.csv"
    for path in [all_phases, fixed27_cruise]:
        if not path.is_file():
            raise FileNotFoundError(path)

    # ------------------------------------------------------------------
    # C. Fixed28 -> Fixed29 -> Fixed30 audit -> Fixed31 for all phases
    # ------------------------------------------------------------------
    equal = args.code_dir / "clean_all_phases_equal_pipeline.py"
    equal_root = clean23_dir / "all_phase_equal_cleaning"
    cmd = [
        sys.executable, str(equal),
        "--script-dir", str(args.code_dir),
        "--all-phases", str(all_phases),
        "--cruise-fixed27", str(fixed27_cruise),
        "--output-root", str(equal_root),
    ]
    cmd.append("--skip-final-reference-check")
    if args.force:
        cmd.append("--force")
    run(cmd)

    final_combined = equal_root / "05_combined" / "final_fixed31_all_phases_model_ready.csv"
    if not final_combined.is_file():
        raise FileNotFoundError(final_combined)

    print("\nCleaning pipeline complete.")
    print("Final all-phase Fixed31:", final_combined)
    print("Cruise cohort will be selected inside the model pipeline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
