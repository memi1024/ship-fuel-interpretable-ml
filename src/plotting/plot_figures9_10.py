# -*- coding: utf-8 -*-
"""Figure 9/10 wrapper for interventional-SHAP and fixed-time outputs."""
from __future__ import annotations
import argparse
from pathlib import Path
from render_all_figures import fig9_direction_shap, fig10_ft


def main() -> int:
    p=argparse.ArgumentParser()
    p.add_argument('--revision-output',type=Path,required=True)
    p.add_argument('--scenario-output',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--strict',action='store_true')
    a=p.parse_args()
    fig9_direction_shap(a.revision_output,a.output_dir,a.strict)
    fig10_ft(a.scenario_output,a.output_dir,a.strict)
    return 0

if __name__=='__main__':
    raise SystemExit(main())
