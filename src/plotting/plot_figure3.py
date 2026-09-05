# -*- coding: utf-8 -*-
"""Figure 3 wrapper for key predictor and fuel distributions."""
from __future__ import annotations
import argparse
from pathlib import Path
from render_all_figures import load_canonical_cruise, fig3_distributions


def main() -> int:
    p=argparse.ArgumentParser()
    p.add_argument('--fixed31-cruise',type=Path,required=True)
    p.add_argument('--core-path',type=Path,required=True)
    p.add_argument('--column-overrides',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args()
    raw=load_canonical_cruise(a.fixed31_cruise,a.core_path,a.column_overrides)
    fig3_distributions(raw,a.output_dir)
    return 0

if __name__=='__main__':
    raise SystemExit(main())
