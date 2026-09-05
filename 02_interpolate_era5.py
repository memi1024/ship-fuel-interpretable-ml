# -*- coding: utf-8 -*-
"""Stage 2: interpolate seven ERA5 fields to containership observations."""
from __future__ import annotations

import argparse
import importlib.util
import re
from pathlib import Path


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("_era5_container_producer", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def available_months(raw_dir: Path) -> list[tuple[int, int]]:
    pattern = re.compile(r"era5_surface_wind_raw_(\d{4})_(\d{2})\.nc$", re.I)
    months = []
    for path in raw_dir.glob("era5_surface_wind_raw_*.nc"):
        m = pattern.search(path.name)
        if m:
            months.append((int(m.group(1)), int(m.group(2))))
    months = sorted(set(months))
    if not months:
        raise FileNotFoundError(f"No era5_surface_wind_raw_YYYY_MM.nc files in {raw_dir}")
    for y, m in months:
        wave = raw_dir / f"era5_wave_raw_{y:04d}_{m:02d}.nc"
        if not wave.is_file():
            raise FileNotFoundError(f"Missing matching wave file: {wave}")
    return months


def parse_sheet(value: str):
    return int(value) if value.isdigit() else value


def main() -> int:
    p = argparse.ArgumentParser(description="Interpolate seven ERA5 fields to containership observations.")
    here = Path(__file__).resolve().parent
    p.add_argument("--producer", type=Path,
                   default=here / "src" / "environment" / "era5_interpolate_to_continership_7fields.py")
    p.add_argument("--input", type=Path, required=True,
                   help="Original containership Excel file.")
    p.add_argument("--era5-dir", type=Path, required=True,
                   help="Directory produced by 01_prepare_era5.py.")
    p.add_argument("--output", type=Path, required=True,
                   help="Interpolated Excel output.")
    p.add_argument("--sheet", default="0")
    p.add_argument("--work-dir", type=Path, default=None)
    p.add_argument("--validation-mode", choices=["none", "metadata", "full"], default="metadata")
    args = p.parse_args()

    for path in [args.producer, args.input]:
        if not path.is_file():
            raise FileNotFoundError(path)
    if not args.era5_dir.is_dir():
        raise FileNotFoundError(args.era5_dir)

    months = available_months(args.era5_dir)
    work = args.work_dir or (args.output.parent / "era5_interpolation_work")
    work.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    m = load_module(args.producer)
    m.INPUT_EXCEL = args.input
    m.INPUT_SHEET = parse_sheet(str(args.sheet))
    m.RAW_DIR = args.era5_dir
    m.OUTPUT_EXCEL = args.output
    m.SURFACE_PATTERN = str(args.era5_dir / "era5_surface_wind_raw_*.nc")
    m.WAVE_PATTERN = str(args.era5_dir / "era5_wave_raw_*.nc")
    m.START_YEAR, m.START_MONTH = months[0]
    m.END_YEAR, m.END_MONTH = months[-1]
    m.VALIDATION_MODE = args.validation_mode
    m.VALIDATION_CACHE = work / "era5_deep_validation_cache.json"
    m.VALIDATION_REPORT = work / "era5_file_validation_report.csv"
    m.CHECKPOINT_DIR = work / "checkpoints"
    m.SURFACE_CHECKPOINT = m.CHECKPOINT_DIR / "surface_result_checkpoint.npz"
    m.WAVE_CHECKPOINT = m.CHECKPOINT_DIR / "wave_result_checkpoint.npz"

    print("Interpolation module:", args.producer)
    print("ERA5 months:", f"{months[0][0]}-{months[0][1]:02d}", "->", f"{months[-1][0]}-{months[-1][1]:02d}")
    print("Output:", args.output)
    m.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
