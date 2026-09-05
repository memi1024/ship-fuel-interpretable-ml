#!/usr/bin/env python3
# -*- coding: utf-8 -*-

r"""Fixed-distance total-fuel and voyage-time decomposition.

This post-processing step uses the L1-aligned FT and explicit-FD vessel-level tables.
It reports average fuel-rate change, voyage-time change, total-fuel change, and their
multiplicative identity, together with the kinematic duration benchmark.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


KEYS = ["reduction_pct", "vessel_id", "ship_type"]


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument(
        "--input-dir",
        default=r"data\11_f31_complete_empirics\11_cii_l1_aligned_final",
    )
    p.add_argument("--ft-file", default="Table_15_FT_L1_aligned_by_vessel.csv")
    p.add_argument("--fd-file", default="Table_16_FD_explicit_L1_aligned_by_vessel.csv")
    p.add_argument(
        "--output-dir",
        default=None,
        help="Defaults to <input-dir>\\PriorityA_FD_fuel_time",
    )
    p.add_argument("--distance-tol-nm", type=float, default=1e-6)
    p.add_argument("--identity-tol", type=float, default=1e-10)
    return p.parse_args()


def require_columns(df: pd.DataFrame, cols, label: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{label} missing columns: {missing}")


def pct(ratio):
    return (ratio - 1.0) * 100.0


def main():
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else input_dir / "PriorityA_FD_fuel_time"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    ft_path = input_dir / args.ft_file
    fd_path = input_dir / args.fd_file

    if not ft_path.exists():
        raise FileNotFoundError(ft_path)
    if not fd_path.exists():
        raise FileNotFoundError(fd_path)

    ft = pd.read_csv(ft_path)
    fd = pd.read_csv(fd_path)

    ft_required = KEYS + [
        "n_supported",
        "retained_pct",
        "baseline_predicted_fuel_t",
        "baseline_distance_nm",
        "baseline_duration_min",
    ]
    fd_required = KEYS + [
        "n_supported",
        "retained_pct",
        "baseline_predicted_fuel_t",
        "baseline_distance_nm",
        "baseline_duration_min",
        "FD_predicted_total_fuel_t",
        "FD_duration_min",
        "FD_added_sailing_time_min",
        "FD_added_time_fuel_t",
        "FD_distance_nm",
        "FD_distance_error_nm",
        "FD_total_fuel_change_pct",
        "FD_CII_change_pct",
    ]

    require_columns(ft, ft_required, "FT table")
    require_columns(fd, fd_required, "FD table")

    if len(ft) != len(fd):
        raise AssertionError(f"FT/FD row count differs: {len(ft)} vs {len(fd)}")

    # -----------------------------------------------------------------
    # A. Strict FT/FD source-sample alignment check.
    # -----------------------------------------------------------------
    chk = ft[ft_required].merge(
        fd[fd_required],
        on=KEYS,
        how="outer",
        suffixes=("_FT", "_FD"),
        indicator=True,
        validate="one_to_one",
    )

    if not (chk["_merge"] == "both").all():
        raise AssertionError("FT and FD do not contain identical scenario/vessel keys.")

    alignment_rows = []

    exact_cols = ["n_supported"]
    numeric_cols = [
        "retained_pct",
        "baseline_predicted_fuel_t",
        "baseline_distance_nm",
        "baseline_duration_min",
    ]

    for c in exact_cols:
        a = chk[f"{c}_FT"].to_numpy()
        b = chk[f"{c}_FD"].to_numpy()
        ok = np.array_equal(a, b)
        alignment_rows.append({
            "check": f"FT_vs_FD_{c}",
            "passed": bool(ok),
            "max_abs_difference": float(np.max(np.abs(a - b))),
        })
        if not ok:
            raise AssertionError(f"FT/FD {c} differs.")

    for c in numeric_cols:
        a = chk[f"{c}_FT"].to_numpy(float)
        b = chk[f"{c}_FD"].to_numpy(float)
        diff = np.abs(a - b)
        ok = np.allclose(a, b, atol=1e-10, rtol=1e-10, equal_nan=True)
        alignment_rows.append({
            "check": f"FT_vs_FD_{c}",
            "passed": bool(ok),
            "max_abs_difference": float(np.nanmax(diff)),
        })
        if not ok:
            raise AssertionError(f"FT/FD {c} differs.")

    alignment = pd.DataFrame(alignment_rows)
    alignment.to_csv(
        output_dir / "Audit_PriorityA_FD_FT_sample_alignment.csv",
        index=False,
    )

    # Work from FD only after alignment has passed.
    z = fd.copy()

    # -----------------------------------------------------------------
    # B. Vessel-level fuel-rate / time / total-fuel decomposition.
    # -----------------------------------------------------------------
    z["baseline_avg_fuel_rate_t_per_10min"] = (
        z["baseline_predicted_fuel_t"]
        / z["baseline_duration_min"]
        * 10.0
    )
    z["FD_avg_fuel_rate_t_per_10min"] = (
        z["FD_predicted_total_fuel_t"]
        / z["FD_duration_min"]
        * 10.0
    )

    z["FD_fuel_rate_ratio"] = (
        z["FD_avg_fuel_rate_t_per_10min"]
        / z["baseline_avg_fuel_rate_t_per_10min"]
    )
    z["FD_fuel_rate_change_pct"] = pct(z["FD_fuel_rate_ratio"])

    z["FD_voyage_time_ratio"] = (
        z["FD_duration_min"]
        / z["baseline_duration_min"]
    )
    z["FD_voyage_time_change_pct_recomputed"] = pct(z["FD_voyage_time_ratio"])

    z["FD_theoretical_time_change_pct"] = (
        1.0 / (1.0 - z["reduction_pct"] / 100.0) - 1.0
    ) * 100.0
    z["FD_realised_minus_theoretical_time_pp"] = (
        z["FD_voyage_time_change_pct_recomputed"]
        - z["FD_theoretical_time_change_pct"]
    )

    z["FD_total_fuel_ratio"] = (
        z["FD_predicted_total_fuel_t"]
        / z["baseline_predicted_fuel_t"]
    )
    z["FD_total_fuel_change_pct_recomputed"] = pct(z["FD_total_fuel_ratio"])

    # Multiplicative decomposition identity.
    z["FD_decomposition_ratio_product"] = (
        z["FD_fuel_rate_ratio"] * z["FD_voyage_time_ratio"]
    )
    z["FD_decomposition_identity_error"] = (
        z["FD_total_fuel_ratio"] - z["FD_decomposition_ratio_product"]
    )

    # Fixed-distance redundancy: fuel per nm change == total fuel change.
    z["baseline_fuel_per_nm_t"] = (
        z["baseline_predicted_fuel_t"] / z["baseline_distance_nm"]
    )
    z["FD_fuel_per_nm_t"] = (
        z["FD_predicted_total_fuel_t"] / z["FD_distance_nm"]
    )
    z["FD_fuel_per_nm_change_pct"] = pct(
        z["FD_fuel_per_nm_t"] / z["baseline_fuel_per_nm_t"]
    )
    z["FD_fuel_per_nm_minus_total_fuel_change_pp"] = (
        z["FD_fuel_per_nm_change_pct"]
        - z["FD_total_fuel_change_pct_recomputed"]
    )

    z["FD_lower_total_fuel"] = (
        z["FD_total_fuel_change_pct_recomputed"] < 0.0
    )

    # Simple mechanism label, useful for audit/discussion, not mandatory in paper.
    z["FD_mechanism"] = np.where(
        z["FD_total_fuel_change_pct_recomputed"] > 0,
        "time penalty dominates fuel-rate reduction",
        np.where(
            z["FD_total_fuel_change_pct_recomputed"] < 0,
            "fuel-rate reduction dominates time penalty",
            "approximately balanced",
        ),
    )

    vessel_cols = [
        "reduction_pct",
        "vessel_id",
        "ship_type",
        "n_supported",
        "retained_pct",
        "baseline_predicted_fuel_t",
        "baseline_duration_min",
        "FD_predicted_total_fuel_t",
        "FD_duration_min",
        "FD_added_sailing_time_min",
        "FD_added_time_fuel_t",
        "FD_fuel_rate_change_pct",
        "FD_voyage_time_change_pct_recomputed",
        "FD_theoretical_time_change_pct",
        "FD_realised_minus_theoretical_time_pp",
        "FD_total_fuel_change_pct_recomputed",
        "FD_lower_total_fuel",
        "FD_mechanism",
        "FD_decomposition_identity_error",
        "FD_fuel_per_nm_minus_total_fuel_change_pp",
        "FD_distance_error_nm",
    ]

    z[vessel_cols].to_csv(
        output_dir / "Table_16_FD_fuel_time_decomposition_by_vessel.csv",
        index=False,
    )

    # -----------------------------------------------------------------
    # C. Fleet / ship-type aggregation from underlying totals.
    # -----------------------------------------------------------------
    rows = []

    for rpct, rg in z.groupby("reduction_pct", sort=True):
        scopes = [("Fleet", rg)] + [
            (str(st), g)
            for st, g in rg.groupby("ship_type", sort=True)
        ]

        for scope, sg in scopes:
            base_fuel = float(sg["baseline_predicted_fuel_t"].sum())
            fd_fuel = float(sg["FD_predicted_total_fuel_t"].sum())
            base_time = float(sg["baseline_duration_min"].sum())
            fd_time = float(sg["FD_duration_min"].sum())

            base_rate = base_fuel / base_time * 10.0
            fd_rate = fd_fuel / fd_time * 10.0

            fuel_rate_ratio = fd_rate / base_rate
            time_ratio = fd_time / base_time
            total_fuel_ratio = fd_fuel / base_fuel

            theoretical_time_change = (
                1.0 / (1.0 - float(rpct) / 100.0) - 1.0
            ) * 100.0

            rows.append({
                "reduction_pct": float(rpct),
                "scope": scope,
                "vessels": int(len(sg)),
                "n_supported_total": int(sg["n_supported"].sum()),
                "mean_retained_pct": float(sg["retained_pct"].mean()),

                "predicted_avg_fuel_rate_change_pct": pct(fuel_rate_ratio),
                "voyage_time_change_pct": pct(time_ratio),
                "kinematic_time_benchmark_pct": theoretical_time_change,
                "realised_minus_kinematic_time_pp": (
                    pct(time_ratio) - theoretical_time_change
                ),

                "added_sailing_time_hours": float(
                    sg["FD_added_sailing_time_min"].sum() / 60.0
                ),
                "added_time_fuel_t": float(
                    sg["FD_added_time_fuel_t"].sum()
                ),
                "added_time_fuel_pct_of_baseline": float(
                    sg["FD_added_time_fuel_t"].sum()
                    / base_fuel
                    * 100.0
                ),

                "predicted_total_fuel_change_pct": pct(total_fuel_ratio),
                "vessels_with_lower_total_fuel": int(
                    (sg["FD_total_fuel_change_pct_recomputed"] < 0.0).sum()
                ),
                "vessels_total": int(len(sg)),

                "decomposition_identity_error_ratio": float(
                    total_fuel_ratio - fuel_rate_ratio * time_ratio
                ),
                "max_abs_FD_distance_error_nm": float(
                    np.nanmax(np.abs(sg["FD_distance_error_nm"].to_numpy(float)))
                ),

                "mechanism": (
                    "time penalty dominates fuel-rate reduction"
                    if total_fuel_ratio > 1.0
                    else "fuel-rate reduction dominates time penalty"
                    if total_fuel_ratio < 1.0
                    else "approximately balanced"
                ),
            })

    summary = pd.DataFrame(rows)

    # Manuscript table: omit CII and fuel/nm because they are redundant here.
    manuscript_cols = [
        "reduction_pct",
        "scope",
        "n_supported_total",
        "mean_retained_pct",
        "predicted_avg_fuel_rate_change_pct",
        "voyage_time_change_pct",
        "added_sailing_time_hours",
        "added_time_fuel_t",
        "predicted_total_fuel_change_pct",
        "vessels_with_lower_total_fuel",
        "vessels_total",
    ]

    summary[manuscript_cols].to_csv(
        output_dir / "Table_16_FD_fuel_time_decomposition_FINAL.csv",
        index=False,
    )

    # -----------------------------------------------------------------
    # D. Hard audit.
    # -----------------------------------------------------------------
    max_identity_error = float(
        np.nanmax(np.abs(z["FD_decomposition_identity_error"].to_numpy(float)))
    )
    max_distance_error = float(
        np.nanmax(np.abs(z["FD_distance_error_nm"].to_numpy(float)))
    )
    max_fuel_nm_redundancy_error = float(
        np.nanmax(
            np.abs(
                z["FD_fuel_per_nm_minus_total_fuel_change_pp"]
                .to_numpy(float)
            )
        )
    )
    max_existing_total_diff = float(
        np.nanmax(
            np.abs(
                z["FD_total_fuel_change_pct_recomputed"].to_numpy(float)
                - z["FD_total_fuel_change_pct"].to_numpy(float)
            )
        )
    )
    max_cii_total_diff = float(
        np.nanmax(
            np.abs(
                z["FD_CII_change_pct"].to_numpy(float)
                - z["FD_total_fuel_change_pct_recomputed"].to_numpy(float)
            )
        )
    )

    audit = pd.DataFrame([
        {
            "check": "multiplicative_identity_total_fuel_equals_rate_times_time",
            "max_abs_error": max_identity_error,
            "tolerance": args.identity_tol,
            "passed": max_identity_error <= args.identity_tol,
        },
        {
            "check": "FD_distance_consistency",
            "max_abs_error": max_distance_error,
            "tolerance": args.distance_tol_nm,
            "passed": max_distance_error <= args.distance_tol_nm,
        },
        {
            "check": "fuel_per_nm_change_equals_total_fuel_change_under_FD",
            "max_abs_error": max_fuel_nm_redundancy_error,
            "tolerance": 1e-8,
            "passed": max_fuel_nm_redundancy_error <= 1e-8,
        },
        {
            "check": "recomputed_total_fuel_change_matches_existing_FD_column",
            "max_abs_error": max_existing_total_diff,
            "tolerance": 1e-8,
            "passed": max_existing_total_diff <= 1e-8,
        },
        {
            "check": "FD_CII_change_is_redundant_with_total_fuel_change",
            "max_abs_error": max_cii_total_diff,
            "tolerance": 1e-8,
            "passed": max_cii_total_diff <= 1e-8,
        },
    ])

    audit.to_csv(
        output_dir / "Audit_PriorityA_FD_identity_checks.csv",
        index=False,
    )

    if not audit["passed"].all():
        raise AssertionError(
            "At least one Priority-A audit failed. Inspect Audit_PriorityA_FD_identity_checks.csv"
        )

    # -----------------------------------------------------------------
    # Console summary
    # -----------------------------------------------------------------
    print("\n=== PRIORITY A: FD FUEL–TIME DECOMPOSITION ===\n")

    display = summary[
        [
            "reduction_pct",
            "scope",
            "predicted_avg_fuel_rate_change_pct",
            "voyage_time_change_pct",
            "predicted_total_fuel_change_pct",
            "added_time_fuel_t",
            "mechanism",
        ]
    ].copy()

    print(display.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    print("\nAll audits PASSED.")
    print(f"Outputs: {output_dir}")
    print("\nNo model fitting or new prediction was performed.")


if __name__ == "__main__":
    main()
