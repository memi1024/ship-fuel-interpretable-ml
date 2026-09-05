# -*- coding: utf-8 -*-
"""Figure 8 wrapper for GAM/LOWESS outputs."""
from __future__ import annotations
import argparse
from pathlib import Path
from render_all_figures import fig8_gam_lowess


def main() -> int:
    p=argparse.ArgumentParser()
    p.add_argument('--revision-output',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--strict',action='store_true')
    a=p.parse_args()
    fig8_gam_lowess(a.revision_output,a.output_dir,a.strict)
    return 0

if __name__=='__main__':
    raise SystemExit(main())
