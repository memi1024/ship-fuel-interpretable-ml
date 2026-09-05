# -*- coding: utf-8 -*-
"""Stage 4: run the manuscript analysis from the Fixed31 dataset."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


MAIN_STEPS = (
    "data,base,phase,ablation,temporal,lovo,adaptation,shiptype,"
    "shap,shap_stability,summary"
)
REVISION_STEPS = "residuals,shift,cubic,ablation,shap,rudder"


def run(cmd: list[str], cwd: Path) -> None:
    print("\n$", subprocess.list2cmdline(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def run_synthetic_check(seed: int = 42) -> int:
    """Run a small in-memory execution check of the Stage-4 model and CII functions."""
    import numpy as np
    import pandas as pd

    analysis_dir = Path(__file__).resolve().parent / "src" / "paper_analysis"
    sys.path.insert(0, str(analysis_dir))
    from f31_core_memory_safe import (
        CANONICAL_FEATURES,
        compute_cii_by_vessel,
        metric_dict,
        model_factory,
        speed_reduction_scenarios,
        stratified_record_split,
    )

    rng = np.random.default_rng(seed)
    specs = [
        ("B1", "bulk", 80000.0, 12.0),
        ("B2", "bulk", 82000.0, 12.5),
        ("C1", "container", 50000.0, 15.0),
        ("C2", "container", 52000.0, 15.5),
        ("T1", "tanker", 90000.0, 13.0),
        ("T2", "tanker", 92000.0, 13.5),
    ]
    frames = []
    for j, (vid, ship_type, dwt, speed_mu) in enumerate(specs):
        n = 300
        speed = np.clip(rng.normal(speed_mu, 2.5, n), 8.2, 20.0)
        heading = rng.uniform(0.0, 360.0, n)
        wind_dir = rng.uniform(0.0, 360.0, n)
        wave_dir = rng.uniform(0.0, 360.0, n)
        wind = rng.uniform(5.0, 20.0, n)
        wave = rng.uniform(0.2, 3.0, n)
        draught = rng.normal(12.0, 0.4, n)
        trim = rng.normal(0.0, 0.2, n)
        rudder = rng.normal(0.0, 0.3, n)
        fuel = (
            0.04
            + 0.0001 * speed**3
            + 0.004 * wave
            + 0.0007 * wind
            + 0.002 * (draught - 12.0)
            + rng.normal(0.0, 0.008, n)
        )
        frames.append(
            pd.DataFrame(
                {
                    "vessel_id": vid,
                    "ship_type": ship_type,
                    "timestamp": pd.date_range("2024-01-01", periods=n, freq="10min")
                    + pd.Timedelta(days=30 * j),
                    "target": np.clip(fuel, 0.03, None),
                    "speed_kn": speed,
                    "heading_sin": np.sin(np.deg2rad(heading)),
                    "heading_cos": np.cos(np.deg2rad(heading)),
                    "draught_m": draught,
                    "trim_m": trim,
                    "rudder_deg": rudder,
                    "rel_wind_speed_kn": wind,
                    "rel_wind_sin": np.sin(np.deg2rad(wind_dir)),
                    "rel_wind_cos": np.cos(np.deg2rad(wind_dir)),
                    "wave_height_m": wave,
                    "rel_wave_sin": np.sin(np.deg2rad(wave_dir)),
                    "rel_wave_cos": np.cos(np.deg2rad(wave_dir)),
                    "wave_period_s": rng.uniform(4.0, 10.0, n),
                    "sst_c": rng.uniform(10.0, 25.0, n),
                    "mslp_hpa": rng.normal(1013.0, 7.0, n),
                    "ship_type_bulk": int(ship_type == "bulk"),
                    "ship_type_container": int(ship_type == "container"),
                    "dwt": dwt,
                    "distance_nm": speed * (10.0 / 60.0),
                }
            )
        )

    raw = pd.concat(frames, ignore_index=True)
    tr, te = stratified_record_split(raw, test_size=0.20, seed=seed)
    model = model_factory(
        "xgb",
        {
            "n_estimators": 25,
            "max_depth": 3,
            "learning_rate": 0.08,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "min_child_weight": 1.0,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
            "gamma": 0.0,
        },
        seed=seed,
        n_jobs=1,
    )
    model.fit(raw.iloc[tr][CANONICAL_FEATURES], raw.iloc[tr]["target"].to_numpy())
    pred = model.predict(raw.iloc[te][CANONICAL_FEATURES])
    metrics = metric_dict(raw.iloc[te]["target"].to_numpy(), pred)
    cii = compute_cii_by_vessel(raw.iloc[te].reset_index(drop=True), pred, default_cf=3.114)
    vessel_scenarios, scenario_summary = speed_reduction_scenarios(
        model=model,
        train_raw=raw.iloc[tr].reset_index(drop=True),
        test_raw=raw.iloc[te].reset_index(drop=True),
        feature_cols=CANONICAL_FEATURES,
        reductions=[0.05, 0.10, 0.15],
        default_cf=3.114,
    )

    if not np.isfinite(metrics["RMSE"]):
        raise AssertionError("non-finite RMSE")
    if cii.empty or vessel_scenarios.empty or scenario_summary.empty:
        raise AssertionError("empty CII output")
    if sorted(map(float, scenario_summary["reduction_pct"].unique())) != [5.0, 10.0, 15.0]:
        raise AssertionError("speed-reduction levels changed")

    print("STAGE 4 SYNTHETIC CHECK PASSED")
    print(f"Synthetic rows checked: {len(raw)}; train={len(tr)}; test={len(te)}")
    print(f"Finite RMSE: {metrics['RMSE']:.6f}; CII vessels: {len(cii)}")
    print("No files were written.")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--synthetic-check", action="store_true")
    p.add_argument("--cleaning-root", type=Path)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--seed", type=int, default=20260808)
    p.add_argument("--n-jobs", type=int, default=4)
    p.add_argument("--csv-chunksize", type=int, default=25000)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--permutations", type=int, default=100000)
    args = p.parse_args()

    if args.synthetic_check:
        return run_synthetic_check(args.seed)
    if args.cleaning_root is None or args.output_dir is None:
        p.error("--cleaning-root and --output-dir are required")

    here = Path(__file__).resolve().parent
    analysis = here / "src" / "paper_analysis"
    plotting = here / "src" / "plotting"
    data_dir = args.cleaning_root / "05_combined"
    consistency_dir = args.cleaning_root / "90_consistency"
    cruise_csv = data_dir / "final_fixed31_cruise.csv"
    all_phase_csv = data_dir / "final_fixed31_all_phases_model_ready.csv"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    core = analysis / "f31_core_memory_safe.py"
    hp = analysis / "02_best_hyperparameters.csv"
    overrides = analysis / "column_overrides_fixed31.json"

    run(
        [
            sys.executable,
            str(analysis / "run_f31_complete_empirics_v4.py"),
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(args.output_dir),
            "--column-overrides",
            str(overrides),
            "--consistency-dir",
            str(consistency_dir),
            "--hyperparams-csv",
            str(hp),
            "--steps",
            MAIN_STEPS,
            "--seed",
            str(args.seed),
            "--n-jobs",
            str(args.n_jobs),
            "--csv-chunksize",
            str(args.csv_chunksize),
        ],
        analysis,
    )

    revision_out = args.output_dir / "15_revision_diagnostics"
    run(
        [
            sys.executable,
            str(analysis / "run_f31_revision_diagnostics.py"),
            "--input-csv",
            str(cruise_csv),
            "--main-output",
            str(args.output_dir),
            "--output-dir",
            str(revision_out),
            "--core-path",
            str(core),
            "--column-overrides",
            str(overrides),
            "--hyperparams-csv",
            str(hp),
            "--steps",
            REVISION_STEPS,
            "--seed",
            str(args.seed),
            "--n-jobs",
            str(min(args.n_jobs, 4)),
            "--csv-chunksize",
            str(args.csv_chunksize),
        ],
        analysis,
    )

    scenario_out = args.output_dir / "11_cii_l1_aligned_final"
    run(
        [
            sys.executable,
            str(analysis / "run_cii_l1_aligned_final.py"),
            "--work-dir",
            str(analysis),
            "--data-dir",
            str(data_dir),
            "--input-csv",
            str(all_phase_csv),
            "--original-output-dir",
            str(args.output_dir),
            "--output-dir",
            str(scenario_out),
            "--column-overrides",
            str(overrides),
        ],
        analysis,
    )

    run(
        [
            sys.executable,
            str(analysis / "run_priorityA_fd_fuel_time_decomposition.py"),
            "--input-dir",
            str(scenario_out),
            "--output-dir",
            str(scenario_out / "PriorityA_FD_fuel_time"),
        ],
        analysis,
    )

    inference_out = args.output_dir / "17_final_manuscript_inference"
    run(
        [
            sys.executable,
            str(analysis / "run_final_manuscript_inference.py"),
            "--input-csv",
            str(cruise_csv),
            "--main-output",
            str(args.output_dir),
            "--revision-output",
            str(revision_out),
            "--scenario-output",
            str(scenario_out),
            "--output-dir",
            str(inference_out),
            "--core-path",
            str(core),
            "--column-overrides",
            str(overrides),
            "--bootstrap",
            str(args.bootstrap),
            "--permutations",
            str(args.permutations),
            "--seed",
            str(20260819),
            "--csv-chunksize",
            str(args.csv_chunksize),
        ],
        analysis,
    )

    run(
        [
            sys.executable,
            str(plotting / "render_all_figures.py"),
            "--fixed31-cruise",
            str(cruise_csv),
            "--core-path",
            str(core),
            "--column-overrides",
            str(overrides),
            "--main-output",
            str(args.output_dir),
            "--revision-output",
            str(revision_out),
            "--scenario-output",
            str(scenario_out),
            "--output-dir",
            str(args.output_dir / "figures"),
            "--seed",
            str(args.seed),
        ],
        plotting,
    )

    print("\nStage 4 complete:", args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
