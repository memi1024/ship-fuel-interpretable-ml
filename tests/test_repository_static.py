from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def test_entrypoints_exist():
    for name in [
        "01_prepare_era5.py",
        "02_interpolate_era5.py",
        "03_clean_data.py",
        "04_reproduce_paper.py",
        "05_bulk_era5_control.py",
        "validate_installation.py",
    ]:
        assert (ROOT / name).is_file(), name


def test_paper_producers_exist():
    required = [
        "src/cleaning/ship_data_cleaning.py",
        "src/cleaning/three_ship_identity_clean23_v15_0.py",
        "src/environment/era5_interpolate_to_continership_7fields.py",
        "src/paper_analysis/f31_core_memory_safe.py",
        "src/paper_analysis/run_f31_complete_empirics_v4.py",
        "src/paper_analysis/run_f31_revision_diagnostics.py",
        "src/paper_analysis/run_cii_l1_aligned_final.py",
        "src/paper_analysis/run_priorityA_fd_fuel_time_decomposition.py",
        "src/paper_analysis/run_final_manuscript_inference.py",
        "src/paper_analysis/run_bulk_era5_harmonisation_control.py",
        "src/plotting/render_all_figures.py",
    ]
    for rel in required:
        assert (ROOT / rel).is_file(), rel


def test_repository_tree():
    assert not (ROOT / "SOURCE_FILE_MAP.md").exists()
    assert not (ROOT / "config" / "expected_private_files.json").exists()
    assert not (ROOT / "src" / "paper_analysis" / "rebuild_l1_model_snapshots.py").exists()


def test_repository_has_no_generated_binary_artifacts():
    forbidden = {".joblib", ".npy", ".npz", ".pkl", ".pickle", ".parquet", ".nc", ".xlsx", ".xls"}
    hits = [p for p in ROOT.rglob("*") if p.is_file() and p.suffix.lower() in forbidden]
    assert not hits, hits


def test_stage34_synthetic_check_runs():
    cp = subprocess.run(
        [sys.executable, str(ROOT / "validate_installation.py")],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
    )
    assert cp.returncode == 0, cp.stdout + "\n" + cp.stderr
    assert "STAGE 3+4 SYNTHETIC EXECUTION CHECK PASSED" in cp.stdout
