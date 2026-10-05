#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fast preflight: paths, imports, Fixed31 canonicalization, locked registry and optional L1 split."""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from _revision_common import (
    DEFAULT_SEED, atomic_json, load_fixed31_cruise, load_locked_params,
    load_official_split, setup_logger,
)


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--fixed31-cruise", required=True)
    p.add_argument("--analysis-dir", default="src/paper_analysis")
    p.add_argument("--column-overrides", default="src/paper_analysis/column_overrides_fixed31.json")
    p.add_argument("--hyperparams-csv", default="src/paper_analysis/02_best_hyperparameters.csv")
    p.add_argument("--split-path", default=None)
    p.add_argument("--output-dir", default="revision_runs/preflight")
    p.add_argument("--trajectory-gap-minutes", type=float, default=30.0)
    p.add_argument("--allow-rebuild-split", action="store_true")
    p.add_argument("--allow-noncanonical-counts", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.output_dir).resolve()
    logger = setup_logger("revision_preflight", out, "preflight.log")
    core, runner, raw, X, feature_cols, mapping = load_fixed31_cruise(
        args.fixed31_cruise, args.analysis_dir, args.column_overrides,
        args.trajectory_gap_minutes, logger, args.allow_noncanonical_counts,
    )
    params = load_locked_params(runner, args.hyperparams_csv, logger)
    split_info = None
    if args.split_path:
        tr, te, src = load_official_split(
            args.split_path, raw, core, seed=DEFAULT_SEED,
            allow_rebuild=args.allow_rebuild_split, logger=logger,
        )
        split_info = {"train": len(tr), "test": len(te), "source": src}
    audit = {
        "status": "PASS",
        "rows": len(raw),
        "vessels": int(raw["vessel_id"].nunique()),
        "ship_type_vessels": raw.groupby("ship_type")["vessel_id"].nunique().astype(int).to_dict(),
        "features": feature_cols,
        "feature_count": len(feature_cols),
        "locked_models": sorted(params),
        "column_mapping": mapping,
        "split": split_info,
    }
    atomic_json(audit, out / "preflight_audit.json")
    logger.info("PREFLIGHT PASS")


if __name__ == "__main__":
    main()
