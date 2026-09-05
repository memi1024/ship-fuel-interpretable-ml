#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 5: run the paired bulk-carrier ERA5 harmonisation control."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--fixed31-cruise", type=Path, required=True)
    p.add_argument("--bulk-harmonized-csv", type=Path, required=True)
    p.add_argument("--main-output", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=20260808)
    p.add_argument("--n-jobs", type=int, default=4)
    p.add_argument("--bootstrap-reps", type=int, default=2000)
    p.add_argument("--csv-chunksize", type=int, default=25000)
    args = p.parse_args()

    here = Path(__file__).resolve().parent
    analysis = here / "src" / "paper_analysis"
    cmd = [
        sys.executable,
        str(analysis / "run_bulk_era5_harmonisation_control.py"),
        "--fixed31-cruise",
        str(args.fixed31_cruise),
        "--bulk-harmonized-csv",
        str(args.bulk_harmonized_csv),
        "--main-output",
        str(args.main_output),
        "--output-dir",
        str(args.output_dir),
        "--analysis-dir",
        str(analysis),
        "--column-overrides",
        str(analysis / "column_overrides_fixed31.json"),
        "--hyperparams-csv",
        str(analysis / "02_best_hyperparameters.csv"),
        "--seed",
        str(args.seed),
        "--n-jobs",
        str(args.n_jobs),
        "--bootstrap-reps",
        str(args.bootstrap_reps),
        "--csv-chunksize",
        str(args.csv_chunksize),
    ]
    print("$", subprocess.list2cmdline(cmd), flush=True)
    subprocess.run(cmd, cwd=str(analysis), check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
