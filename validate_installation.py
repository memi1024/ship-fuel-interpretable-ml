# -*- coding: utf-8 -*-
"""Run the Stage 3 and Stage 4 synthetic execution checks."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run_stage(script: str) -> None:
    cmd = [sys.executable, str(ROOT / script), "--synthetic-check"]
    print("$", subprocess.list2cmdline(cmd), flush=True)
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def main() -> int:
    run_stage("03_clean_data.py")
    run_stage("04_reproduce_paper.py")
    print("STAGE 3+4 SYNTHETIC EXECUTION CHECK PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
