#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone overnight runner for the missing Fixed31 tanker ERA5 harmonization control.

Purpose
-------
This script targets the remaining harmonization experiment needed for the manuscript:
tanker × head-sea wave-C4, under a strict original-vs-ERA5 paired-control design.

Locked design
-------------
- Fixed31 cruise cohort: 489,620 rows
- Official L1 split: 391,696 train / 97,924 test
- Canonical XGB feature specification: 17 features
- Locked XGB hyperparameters from 02_best_hyperparameters.csv
- No retuning
- Same target, same rows, same L1 split, same model family/specification
- The ONLY treatment is replacing tanker environmental variables with harmonized ERA5
- Harmonized ERA5 missing values are retained; XGBoost native missing routing preserves
  the exact official cohort.

Tanker ERA5 transformation
--------------------------
Input seven ERA5 fields:
    wind_s, wind_d, wave_h, wave_d, wave_p, surface_t, surface_p

Converted to canonical model environment exactly as the bulk harmonization:
- wind_d / wave_d are FROM directions
- vector TO direction = FROM + 180 deg
- relative angle = wrapped(TO - ship heading) in [-180, 180)
- relative wind speed =
    sqrt(wind^2 + ship_speed^2 - 2*wind*ship_speed*cos(relative_wind_angle))
- wave height = ERA5 SWH
- wave period = ERA5 mean wave period
- SST = ERA5 sea-surface temperature in deg C
- pressure = Pa / 100 -> hPa

Primary Wave-C4 estimand
------------------------
At +10% wave height:
- official L1 tanker test rows only
- strict seven-field ERA5-complete rows
- complete derived harmonized environmental model features
- perturbed wave height inside BOTH original and harmonized tanker training support
- same paired rows in both branches
- primary head/cross/following analysis requires sector agreement across pathways
  ("stable sector")
- head = <=45 or >=315 deg; following = 135-225 deg; otherwise cross
- vessel-balanced bootstrap inference

Additional reviewer-facing outputs
----------------------------------
1) exact alignment/provenance audit
2) original-vs-harmonized tanker predictive metrics on the same L1 test rows
3) tanker by-vessel predictive metrics
4) paired vessel bootstrap RMSE difference
5) +10% primary wave-C4 with stable-sector head-sea table
6) +5/+10/+15% wave perturbation sensitivity
7) sector-migration audit
8) secondary wave/speed-tertile subgroup diagnostics
9) paired vessel sign-flip sensitivity for the primary head-sea pathway contrast
10) manuscript-ready summary files
11) hard interpolation-method audit (bilinear/linear/no nearest-neighbour/SST-only)
12) seven-field unit/range/coverage audit + monthly coverage cross-check

Python compatibility: 3.7+
"""

from __future__ import print_function

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


VERSION = "2026-10-02.tanker-c4-harmonization-overnight-v2-audited"

EXPECTED_ROWS = 489620
EXPECTED_TRAIN = 391696
EXPECTED_TEST = 97924
EXPECTED_VESSELS = 21
EXPECTED_TANKER_ROWS = 245402
EXPECTED_TANKER_COMPLETE7 = 215479

# Provenance hashes from the already-validated bulk harmonization control manifest.
EXPECTED_SHA16 = {
    "final_fixed31_cruise.csv": "f14a0735a9539cf2",
    "record_split_indices.npz": "25320775de28dade",
    "02_best_hyperparameters.csv": "cb5bdd35008c8c85",
}

PRIMARY_FEATURES = [
    "speed_kn",
    "heading_sin",
    "heading_cos",
    "draught_m",
    "trim_m",
    "rudder_deg",
    "rel_wind_speed_kn",
    "rel_wind_sin",
    "rel_wind_cos",
    "wave_height_m",
    "rel_wave_sin",
    "rel_wave_cos",
    "wave_period_s",
    "sst_c",
    "mslp_hpa",
    "ship_type_bulk",
    "ship_type_container",
]

OPERATIONAL_FEATURES = [
    "speed_kn",
    "heading_sin",
    "heading_cos",
    "draught_m",
    "trim_m",
    "rudder_deg",
    "ship_type_bulk",
    "ship_type_container",
]

ENV_FEATURES = [
    "rel_wind_speed_kn",
    "rel_wind_sin",
    "rel_wind_cos",
    "wave_height_m",
    "rel_wave_sin",
    "rel_wave_cos",
    "wave_period_s",
    "sst_c",
    "mslp_hpa",
]

TANKER_ERA5_FIELDS = [
    "wind_s",
    "wind_d",
    "wave_h",
    "wave_d",
    "wave_p",
    "surface_t",
    "surface_p",
]

ERA5_SANITY_RULES = {
    "wind_s": {"unit": "kn", "lower": 0.0, "upper": 250.0, "lower_inclusive": True, "upper_inclusive": True},
    "wind_d": {"unit": "deg_FROM", "lower": 0.0, "upper": 360.0, "lower_inclusive": True, "upper_inclusive": False},
    "wave_h": {"unit": "m", "lower": 0.0, "upper": 50.0, "lower_inclusive": True, "upper_inclusive": True},
    "wave_d": {"unit": "deg_FROM", "lower": 0.0, "upper": 360.0, "lower_inclusive": True, "upper_inclusive": False},
    "wave_p": {"unit": "s", "lower": 0.0, "upper": 60.0, "lower_inclusive": False, "upper_inclusive": True},
    "surface_t": {"unit": "degC", "lower": -5.0, "upper": 50.0, "lower_inclusive": True, "upper_inclusive": True},
    "surface_p": {"unit": "Pa", "lower": 75000.0, "upper": 115000.0, "lower_inclusive": True, "upper_inclusive": True},
}

PRIMARY_WAVE_PCT = 10.0


# ---------------------------------------------------------------------
# CLI / utilities
# ---------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Run the missing tanker ERA5 harmonization + head-sea C4 control."
    )
    p.add_argument("--home", default=".")
    p.add_argument("--fixed31", default=None)
    p.add_argument("--tanker-era5", default=None)
    p.add_argument("--tanker-audit", default=None)
    p.add_argument("--tanker-monthly-audit", default=None)
    p.add_argument(
        "--allow-missing-interpolation-audit",
        action="store_true",
        help="Allow missing interpolation audit JSON for debugging only."
    )
    p.add_argument("--split", default=None)
    p.add_argument("--hyperparams", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument(
        "--steps",
        default="all",
        help="Comma list: alignment,fit,performance,c4,summary or all"
    )
    p.add_argument("--resume", action="store_true")
    p.add_argument("--n-jobs", type=int, default=2)
    p.add_argument("--seed", type=int, default=20260808)
    p.add_argument("--bootstrap", type=int, default=5000)
    p.add_argument("--permutations", type=int, default=100000)
    p.add_argument("--wave-levels", default="0.05,0.10,0.15")
    p.add_argument("--wave-min-group-n", type=int, default=200)
    p.add_argument(
        "--expected-tanker-complete7",
        type=int,
        default=EXPECTED_TANKER_COMPLETE7,
        help="Set -1 to disable the hard complete-7 count audit."
    )
    p.add_argument(
        "--skip-hash-check",
        action="store_true",
        help="Skip strict SHA16 checks for the three locked non-ERA5 inputs."
    )
    return p.parse_args()


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    return Path(path)


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def sha256_short(path):
    h = hashlib.sha256()
    with open(str(path), "rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()[:16].lower()


def atomic_csv(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8-sig")
    tmp.replace(path)


def logprint(log_path, msg):
    line = str(msg)
    print(line, flush=True)
    with open(str(log_path), "a", encoding="utf-8") as f:
        f.write(line + "\n")


def discover_one(explicit, basename, roots, required=True):
    if explicit:
        p = Path(explicit)
        if p.is_file():
            return p.resolve()
        raise FileNotFoundError("Explicit file not found: %s" % p)

    # Prefer the supplied --home roots rather than a machine-specific path.
    first_root = Path(roots[0]) if roots else Path(".")
    direct = [
        first_root / basename,
        first_root / "work" / basename,
    ]
    for p in direct:
        if p.is_file():
            return p.resolve()

    hits = []
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        try:
            hits.extend(root.rglob(basename))
        except Exception:
            pass

    # Prefer shorter paths; identical basenames are later protected by SHA where applicable.
    hits = sorted({str(x.resolve()): x.resolve() for x in hits}.values(),
                  key=lambda x: (len(str(x)), str(x)))
    if hits:
        return hits[0]

    if required:
        raise FileNotFoundError(
            "Could not find %s under %s" % (basename, [str(x) for x in roots])
        )
    return None


def resolve_paths(args):
    home = Path(args.home)
    roots = [
        home,
        home / "work",
        home / "data" / "datasets",
    ]

    paths = {
        "fixed31": discover_one(
            args.fixed31, "final_fixed31_cruise.csv", roots, True
        ),
        "tanker_era5": discover_one(
            args.tanker_era5, "tanker31_cruise_with_era5_7fields.csv", roots, True
        ),
        "tanker_audit": discover_one(
            args.tanker_audit, "tanker31_cruise_with_era5_7fields_audit.json",
            roots, False
        ),
        "tanker_monthly_audit": discover_one(
            args.tanker_monthly_audit,
            "tanker31_cruise_era5_coverage_by_month.csv",
            roots, False
        ),
        "split": discover_one(
            args.split, "record_split_indices.npz", roots, True
        ),
        "hyperparams": discover_one(
            args.hyperparams, "02_best_hyperparameters.csv", roots, True
        ),
        "out": Path(args.output_dir) if args.output_dir else (
            home / "tanker_C4_harmonization_overnight"
        ),
    }
    paths["out"].mkdir(parents=True, exist_ok=True)
    return paths


def normalize_ship_type(series):
    z = series.astype(str).str.strip().str.lower()
    out = z.copy()
    out[z.str.contains("bulk", na=False)] = "bulk"
    out[z.str.contains("container", na=False)] = "container"
    out[z.str.contains("tank", na=False)] = "tanker"
    return out


def load_split(path):
    z = np.load(str(path))
    train_key = next((k for k in z.files if "train" in k.lower()), None)
    test_key = next((k for k in z.files if "test" in k.lower()), None)
    if train_key is None or test_key is None:
        raise KeyError("Cannot find train/test arrays in %s; keys=%s" % (path, z.files))
    tr = np.asarray(z[train_key], dtype=int)
    te = np.asarray(z[test_key], dtype=int)
    return tr, te, train_key, test_key


def load_xgb_params(path):
    tab = pd.read_csv(path)
    names = tab["model"].astype(str).str.strip().str.lower()
    row = tab.loc[names.isin(["xgb", "xgboost"])]
    if row.empty:
        raise ValueError("No XGB/XGBoost row in %s" % path)
    raw = row.iloc[0]["best_parameters_json"]
    if pd.isna(raw) or str(raw).strip() in ["", "{}"]:
        return {}
    return json.loads(str(raw))


def xgb_factory(params, seed, n_jobs):
    return XGBRegressor(
        random_state=int(seed),
        n_jobs=int(n_jobs),
        tree_method="hist",
        objective="reg:squarederror",
        eval_metric="rmse",
        n_estimators=int(params.get("n_estimators", 5000)),
        max_depth=int(params.get("max_depth", 6)),
        learning_rate=float(params.get("learning_rate", 0.0275)),
        subsample=float(params.get("subsample", 0.6869)),
        colsample_bytree=float(params.get("colsample_bytree", 0.8536)),
        min_child_weight=float(params.get("min_child_weight", 7.5)),
        gamma=float(params.get("gamma", 0.0)),
        reg_alpha=float(params.get("reg_alpha", 0.1)),
        reg_lambda=float(params.get("reg_lambda", 1.0)),
    )


# ---------------------------------------------------------------------
# Canonical Fixed31
# ---------------------------------------------------------------------

def build_canonical_original(df):
    required = [
        "ship_type",
        "pseudo_ship_group_id",
        "timestamp_utc",
        "fuel_t_10min",
        "speed_kn",
        "heading_sin",
        "heading_cos",
        "mean_draught_m",
        "trim_m",
        "rudder_deg",
        "rel_wind_speed_kn",
        "relative_wind_sin",
        "relative_wind_cos",
        "wave_height_m",
        "relative_wave_sin",
        "relative_wave_cos",
        "wave_period_s",
        "surface_temperature_c",
        "surface_pressure_pa",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError("Fixed31 missing required columns: %s" % missing)

    raw = pd.DataFrame(index=np.arange(len(df)))
    raw["row_id"] = np.arange(len(df), dtype=np.int64)
    raw["vessel_id"] = df["pseudo_ship_group_id"].astype(str).str.strip().to_numpy()
    raw["ship_type"] = normalize_ship_type(df["ship_type"]).to_numpy()
    raw["timestamp"] = pd.to_datetime(df["timestamp_utc"], errors="coerce", utc=True)
    raw["target"] = pd.to_numeric(df["fuel_t_10min"], errors="coerce").to_numpy(float)
    raw["speed_kn"] = pd.to_numeric(df["speed_kn"], errors="coerce").to_numpy(float)
    raw["heading_sin"] = pd.to_numeric(df["heading_sin"], errors="coerce").to_numpy(float)
    raw["heading_cos"] = pd.to_numeric(df["heading_cos"], errors="coerce").to_numpy(float)
    raw["ship_direction_deg"] = (
        np.degrees(
            np.arctan2(
                raw["heading_sin"].to_numpy(float),
                raw["heading_cos"].to_numpy(float),
            )
        )
        + 360.0
    ) % 360.0

    raw["draught_m"] = pd.to_numeric(df["mean_draught_m"], errors="coerce").to_numpy(float)
    raw["trim_m"] = pd.to_numeric(df["trim_m"], errors="coerce").to_numpy(float)
    raw["rudder_deg"] = pd.to_numeric(df["rudder_deg"], errors="coerce").to_numpy(float)

    raw["rel_wind_speed_kn"] = pd.to_numeric(
        df["rel_wind_speed_kn"], errors="coerce"
    ).to_numpy(float)
    raw["rel_wind_sin"] = pd.to_numeric(
        df["relative_wind_sin"], errors="coerce"
    ).to_numpy(float)
    raw["rel_wind_cos"] = pd.to_numeric(
        df["relative_wind_cos"], errors="coerce"
    ).to_numpy(float)

    raw["wave_height_m"] = pd.to_numeric(
        df["wave_height_m"], errors="coerce"
    ).to_numpy(float)
    raw["rel_wave_sin"] = pd.to_numeric(
        df["relative_wave_sin"], errors="coerce"
    ).to_numpy(float)
    raw["rel_wave_cos"] = pd.to_numeric(
        df["relative_wave_cos"], errors="coerce"
    ).to_numpy(float)
    raw["wave_period_s"] = pd.to_numeric(
        df["wave_period_s"], errors="coerce"
    ).to_numpy(float)

    raw["sst_c"] = pd.to_numeric(
        df["surface_temperature_c"], errors="coerce"
    ).to_numpy(float)
    raw["mslp_hpa"] = (
        pd.to_numeric(df["surface_pressure_pa"], errors="coerce").to_numpy(float)
        / 100.0
    )

    raw["ship_type_bulk"] = (raw["ship_type"] == "bulk").astype(np.int8)
    raw["ship_type_container"] = (raw["ship_type"] == "container").astype(np.int8)

    if "trajectory_segment_id" in df.columns:
        raw["trajectory_group"] = df["trajectory_segment_id"].astype(str).to_numpy()
    else:
        raw["trajectory_group"] = np.nan

    X = raw[PRIMARY_FEATURES].copy()
    return raw.reset_index(drop=True), X.reset_index(drop=True)


def angle_difference_deg(direction_to, ship_direction):
    return ((direction_to - ship_direction + 180.0) % 360.0) - 180.0



def _audit_row(check, observed, expected, passed, detail=""):
    return {
        "check": str(check),
        "observed": observed,
        "expected": expected,
        "PASS": bool(passed),
        "detail": str(detail),
    }


def validate_tanker_interpolation_audit(audit_path, monthly_path, out_dir, args, log_path):
    d = ensure_dir(out_dir)
    rows = []

    if audit_path is None or not Path(audit_path).is_file():
        if args.allow_missing_interpolation_audit:
            rows.append(_audit_row(
                "interpolation_audit_json_present", False, True, False,
                "Bypassed with --allow-missing-interpolation-audit"
            ))
            atomic_csv(pd.DataFrame(rows), d / "Table_A0_interpolation_method_hard_audit.csv")
            logprint(log_path, "[WARN] interpolation audit JSON missing; strict method audit bypassed")
            return None, None
        raise FileNotFoundError(
            "Strict run requires tanker31_cruise_with_era5_7fields_audit.json"
        )

    with open(str(audit_path), "r", encoding="utf-8") as f:
        audit = json.load(f)

    def add(check, key, expected, predicate=None, detail=""):
        observed = audit.get(key, None)
        if predicate is None:
            passed = observed == expected
        else:
            try:
                passed = bool(predicate(observed))
            except Exception:
                passed = False
        rows.append(_audit_row(check, observed, expected, passed, detail))

    add("code_build_is_V2", "code_build", "V2 adaptive tanker interpolator",
        lambda x: x is not None and "V2" in str(x).upper())
    add("rows_tanker_cruise", "rows_tanker_cruise", EXPECTED_TANKER_ROWS,
        lambda x: int(x) == EXPECTED_TANKER_ROWS)

    if int(args.expected_tanker_complete7) >= 0:
        add("complete_7fields_rows", "complete_7fields_rows",
            int(args.expected_tanker_complete7),
            lambda x: int(x) == int(args.expected_tanker_complete7))

    add("spatial_interpolation", "spatial_interpolation", "strict_bilinear",
        lambda x: str(x).strip().lower() in {"strict_bilinear", "bilinear"})
    add("temporal_interpolation", "temporal_interpolation", "linear",
        lambda x: str(x).strip().lower() == "linear")
    add("max_time_gap_hours", "max_time_gap_hours", 2.0,
        lambda x: np.isfinite(float(x)) and abs(float(x) - 2.0) < 1e-12)
    add("nearest_neighbor_used", "nearest_neighbor_used", False,
        lambda x: x is False)
    add("wave_direction_interpolation", "wave_direction_interpolation",
        "circular_sin_cos",
        lambda x: str(x).strip().lower() == "circular_sin_cos")
    add("wind_direction_method", "wind_direction_method",
        "derived after interpolating u10/v10",
        lambda x: x is not None and "u10" in str(x).lower() and "v10" in str(x).lower())
    add("surface_temperature_source", "surface_temperature_source",
        "ERA5 sea_surface_temperature",
        lambda x: str(x).strip().lower() == "era5 sea_surface_temperature")
    add("surface_temperature_fallback", "surface_temperature_fallback", None,
        lambda x: x is None)

    tab = pd.DataFrame(rows)
    atomic_csv(tab, d / "Table_A0_interpolation_method_hard_audit.csv")
    save_json(audit, d / "source_tanker_interpolation_audit_copy.json")

    failed = tab.loc[~tab["PASS"].astype(bool)]
    if not failed.empty:
        raise AssertionError(
            "Interpolation-method hard audit failed: %s"
            % failed["check"].tolist()
        )

    monthly_summary = None
    if monthly_path is not None and Path(monthly_path).is_file():
        monthly = pd.read_csv(monthly_path)
        need = ["month", "rows", "complete_7fields", "complete_7fields_pct"]
        miss = [c for c in need if c not in monthly.columns]
        if miss:
            raise KeyError("Monthly coverage audit missing columns: %s" % miss)

        rows_sum = int(pd.to_numeric(monthly["rows"], errors="coerce").sum())
        complete_sum = int(pd.to_numeric(monthly["complete_7fields"], errors="coerce").sum())

        if rows_sum != EXPECTED_TANKER_ROWS:
            raise AssertionError(
                "Monthly coverage row sum changed: %d vs %d"
                % (rows_sum, EXPECTED_TANKER_ROWS)
            )
        if int(args.expected_tanker_complete7) >= 0 and complete_sum != int(args.expected_tanker_complete7):
            raise AssertionError(
                "Monthly complete-7 sum changed: %d vs %d"
                % (complete_sum, int(args.expected_tanker_complete7))
            )

        atomic_csv(monthly, d / "Table_A0c_monthly_coverage_audit.csv")
        monthly_summary = {
            "months": int(len(monthly)),
            "rows_sum": rows_sum,
            "complete_7fields_sum": complete_sum,
            "overall_complete_7fields_pct": float(100.0 * complete_sum / max(rows_sum, 1)),
            "monthly_complete_pct_min": float(pd.to_numeric(
                monthly["complete_7fields_pct"], errors="coerce").min()),
            "monthly_complete_pct_max": float(pd.to_numeric(
                monthly["complete_7fields_pct"], errors="coerce").max()),
        }
        for c in ["unrouted", "partial_fields", "missing_era5"]:
            if c in monthly.columns:
                monthly_summary[c + "_sum"] = int(
                    pd.to_numeric(monthly[c], errors="coerce").sum()
                )
        save_json(monthly_summary, d / "monthly_coverage_summary.json")
        logprint(
            log_path,
            "[PASS] monthly coverage cross-check | months=%d | rows=%d | complete7=%d (%.2f%%)"
            % (monthly_summary["months"], rows_sum, complete_sum,
               monthly_summary["overall_complete_7fields_pct"])
        )
    else:
        logprint(
            log_path,
            "[WARN] monthly coverage CSV not found; overall audit JSON and row-level checks still enforced"
        )

    logprint(
        log_path,
        "[PASS] interpolation hard audit | bilinear + linear time + NO nearest-neighbour + circular wave direction + u10/v10 wind + SST only"
    )
    return audit, monthly_summary


def audit_tanker_era5_7fields(df, out_dir, log_path):
    d = ensure_dir(out_dir)
    rows = []
    bad_any = np.zeros(len(df), dtype=bool)

    for field in TANKER_ERA5_FIELDS:
        rule = ERA5_SANITY_RULES[field]
        x = pd.to_numeric(df[field], errors="coerce").to_numpy(float)
        finite = np.isfinite(x)

        lower = float(rule["lower"])
        upper = float(rule["upper"])
        low_ok = x >= lower if rule["lower_inclusive"] else x > lower
        high_ok = x <= upper if rule["upper_inclusive"] else x < upper
        good = finite & low_ok & high_ok
        bad = finite & ~good
        bad_any |= bad

        xf = x[finite]
        if len(xf):
            q01, q50, q99 = np.quantile(xf, [0.01, 0.50, 0.99])
            min_v = float(np.min(xf))
            max_v = float(np.max(xf))
        else:
            q01 = q50 = q99 = min_v = max_v = np.nan

        rows.append({
            "field": field,
            "declared_unit": rule["unit"],
            "n_rows": int(len(x)),
            "nonmissing": int(finite.sum()),
            "missing": int((~finite).sum()),
            "nonmissing_pct": float(100.0 * finite.mean()),
            "min": min_v,
            "p01": float(q01),
            "median": float(q50),
            "p99": float(q99),
            "max": max_v,
            "sanity_lower": lower,
            "sanity_upper": upper,
            "out_of_range_count": int(bad.sum()),
            "PASS": bool(int(bad.sum()) == 0),
        })

    tab = pd.DataFrame(rows)
    atomic_csv(tab, d / "Table_A0b_era5_7field_unit_range_audit.csv")

    complete7 = (
        df[TANKER_ERA5_FIELDS]
        .apply(pd.to_numeric, errors="coerce")
        .notna()
        .all(axis=1)
        .to_numpy()
    )

    coverage = {
        "rows": int(len(df)),
        "complete_7fields_rows": int(complete7.sum()),
        "complete_7fields_pct": float(100.0 * complete7.mean()) if len(df) else np.nan,
        "any_out_of_range_rows": int(bad_any.sum()),
    }
    save_json(
        coverage,
        d / "era5_7field_rowlevel_coverage_and_range_summary.json"
    )

    if bad_any.any():
        key_cols = [c for c in [
            "pseudo_ship_group_id", "trajectory_segment_id", "timestamp_utc",
            "latitude_deg", "longitude_deg"
        ] if c in df.columns]
        bad_sample = df.loc[bad_any, key_cols + TANKER_ERA5_FIELDS].head(500).copy()
        atomic_csv(bad_sample, d / "FAILED_era5_out_of_range_rows_sample.csv")
        failed_fields = tab.loc[~tab["PASS"], "field"].tolist()
        raise AssertionError(
            "ERA5 seven-field unit/range audit failed for %s" % failed_fields
        )

    logprint(
        log_path,
        "[PASS] ERA5 7-field unit/range audit | complete7=%d/%d = %.2f%%"
        % (int(complete7.sum()), int(len(df)), float(100.0 * complete7.mean()))
    )
    return coverage, tab


def build_tanker_harmonized_overlay(raw_o, X_o, tanker_path, tanker_audit_path,
                                    out_dir, args, log_path):
    d = ensure_dir(out_dir)

    header = pd.read_csv(tanker_path, nrows=0)
    required = ["pseudo_ship_group_id", "timestamp_utc"] + TANKER_ERA5_FIELDS
    missing = [c for c in required if c not in header.columns]
    if missing:
        raise KeyError("Tanker ERA5 file missing columns: %s" % missing)

    use_trajectory = (
        "trajectory_segment_id" in header.columns
        and raw_o["trajectory_group"].notna().any()
    )
    if use_trajectory:
        required.append("trajectory_segment_id")

    usecols = list(required)
    if "era5_status" in header.columns:
        usecols.append("era5_status")

    t = pd.read_csv(tanker_path, usecols=usecols, low_memory=False)

    range_coverage, range_table = audit_tanker_era5_7fields(
        t, d, log_path
    )

    if len(t) != EXPECTED_TANKER_ROWS:
        raise AssertionError(
            "Expected %d tanker ERA5 rows, got %d"
            % (EXPECTED_TANKER_ROWS, len(t))
        )

    t["_t"] = pd.to_datetime(t["timestamp_utc"], errors="coerce", utc=True)
    t["_vessel"] = t["pseudo_ship_group_id"].astype(str).str.strip()
    if use_trajectory:
        t["_traj"] = t["trajectory_segment_id"].astype(str)

    tanker_mask = raw_o["ship_type"].eq("tanker").to_numpy()
    if int(tanker_mask.sum()) != EXPECTED_TANKER_ROWS:
        raise AssertionError(
            "Official Fixed31 tanker rows changed: %d vs expected %d"
            % (int(tanker_mask.sum()), EXPECTED_TANKER_ROWS)
        )

    key_cols = ["row_id", "vessel_id", "timestamp", "speed_kn", "ship_direction_deg"]
    if use_trajectory:
        key_cols.append("trajectory_group")

    keys = raw_o.loc[tanker_mask, key_cols].copy()
    keys = keys.rename(columns={"vessel_id": "_vessel", "timestamp": "_t"})
    keys["_vessel"] = keys["_vessel"].astype(str).str.strip()

    if use_trajectory:
        keys = keys.rename(columns={"trajectory_group": "_traj"})
        keys["_traj"] = keys["_traj"].astype(str)
        merge_keys = ["_vessel", "_t", "_traj"]
    else:
        merge_keys = ["_vessel", "_t"]
        if t.duplicated(merge_keys).any() or keys.duplicated(merge_keys).any():
            raise AssertionError(
                "Trajectory key unavailable and vessel+timestamp is not unique. "
                "Upload a tanker ERA5 file that preserves trajectory_segment_id."
            )

    hm = t.merge(keys, on=merge_keys, how="inner", validate="one_to_one")
    if len(hm) != EXPECTED_TANKER_ROWS:
        raise AssertionError(
            "Tanker ERA5 alignment incomplete: matched=%d expected=%d"
            % (len(hm), EXPECTED_TANKER_ROWS)
        )
    if hm["row_id"].duplicated().any():
        raise AssertionError("Tanker alignment duplicated official row_id")

    seven = hm[TANKER_ERA5_FIELDS].apply(pd.to_numeric, errors="coerce")
    complete7 = seven.notna().all(axis=1).to_numpy()
    n_complete7 = int(complete7.sum())

    expected_complete = int(args.expected_tanker_complete7)
    if expected_complete >= 0 and n_complete7 != expected_complete:
        raise AssertionError(
            "Tanker complete-7 count changed: %d vs expected %d"
            % (n_complete7, expected_complete)
        )

    if "era5_status" in hm.columns:
        status_complete = (
            hm["era5_status"].astype(str).eq("complete_7fields").to_numpy()
        )
        mismatch = int(np.sum(status_complete != complete7))
        if mismatch:
            raise AssertionError(
                "era5_status and finite seven-field mask disagree on %d rows"
                % mismatch
            )

    speed = pd.to_numeric(hm["speed_kn"], errors="coerce").to_numpy(float)
    ship_dir = pd.to_numeric(
        hm["ship_direction_deg"], errors="coerce"
    ).to_numpy(float)

    wind_s = pd.to_numeric(hm["wind_s"], errors="coerce").to_numpy(float)
    wind_d = pd.to_numeric(hm["wind_d"], errors="coerce").to_numpy(float)
    wave_h = pd.to_numeric(hm["wave_h"], errors="coerce").to_numpy(float)
    wave_d = pd.to_numeric(hm["wave_d"], errors="coerce").to_numpy(float)
    wave_p = pd.to_numeric(hm["wave_p"], errors="coerce").to_numpy(float)
    surface_t = pd.to_numeric(hm["surface_t"], errors="coerce").to_numpy(float)
    surface_p = pd.to_numeric(hm["surface_p"], errors="coerce").to_numpy(float)

    wind_to = (wind_d + 180.0) % 360.0
    wave_to = (wave_d + 180.0) % 360.0
    rel_wind_angle = angle_difference_deg(wind_to, ship_dir)
    rel_wave_angle = angle_difference_deg(wave_to, ship_dir)
    rel_wind_rad = np.deg2rad(rel_wind_angle)
    rel_wave_rad = np.deg2rad(rel_wave_angle)

    rel_wind_speed_sq = (
        wind_s ** 2
        + speed ** 2
        - 2.0 * wind_s * speed * np.cos(rel_wind_rad)
    )
    rel_wind_speed_sq = np.where(
        np.isfinite(rel_wind_speed_sq),
        np.maximum(rel_wind_speed_sq, 0.0),
        np.nan,
    )

    hm["rel_wind_speed_kn_h"] = np.sqrt(rel_wind_speed_sq)
    hm["rel_wind_sin_h"] = np.sin(rel_wind_rad)
    hm["rel_wind_cos_h"] = np.cos(rel_wind_rad)
    hm["wave_height_m_h"] = wave_h
    hm["rel_wave_sin_h"] = np.sin(rel_wave_rad)
    hm["rel_wave_cos_h"] = np.cos(rel_wave_rad)
    hm["wave_period_s_h"] = wave_p
    hm["sst_c_h"] = surface_t
    hm["mslp_hpa_h"] = surface_p / 100.0
    hm["complete_7fields"] = complete7

    mapping = {
        "rel_wind_speed_kn_h": "rel_wind_speed_kn",
        "rel_wind_sin_h": "rel_wind_sin",
        "rel_wind_cos_h": "rel_wind_cos",
        "wave_height_m_h": "wave_height_m",
        "rel_wave_sin_h": "rel_wave_sin",
        "rel_wave_cos_h": "rel_wave_cos",
        "wave_period_s_h": "wave_period_s",
        "sst_c_h": "sst_c",
        "mslp_hpa_h": "mslp_hpa",
    }

    raw_h = raw_o.copy(deep=True)
    X_h = X_o.copy(deep=True)
    rid = hm["row_id"].to_numpy(dtype=int)

    overlay_audit = []
    for src, canon in mapping.items():
        vals = pd.to_numeric(hm[src], errors="coerce").to_numpy(float)
        old = raw_o.loc[rid, canon].to_numpy(float)
        raw_h.loc[rid, canon] = vals
        X_h.loc[rid, canon] = vals
        same = np.isclose(old, vals, rtol=1e-10, atol=1e-12, equal_nan=True)
        overlay_audit.append({
            "canonical_feature": canon,
            "changed_rows": int((~same).sum()),
            "original_missing": int(np.isnan(old).sum()),
            "harmonized_missing": int(np.isnan(vals).sum()),
        })

    # Hard negative controls.
    op_exact = np.allclose(
        X_o[OPERATIONAL_FEATURES].to_numpy(float),
        X_h[OPERATIONAL_FEATURES].to_numpy(float),
        rtol=0.0, atol=0.0, equal_nan=True,
    )
    target_exact = np.allclose(
        raw_o["target"].to_numpy(float),
        raw_h["target"].to_numpy(float),
        rtol=0.0, atol=0.0, equal_nan=True,
    )
    non_tanker = ~tanker_mask
    non_tanker_env_exact = np.allclose(
        X_o.loc[non_tanker, ENV_FEATURES].to_numpy(float),
        X_h.loc[non_tanker, ENV_FEATURES].to_numpy(float),
        rtol=0.0, atol=0.0, equal_nan=True,
    )
    if not op_exact:
        raise AssertionError("Operational negative-control identity failed")
    if not target_exact:
        raise AssertionError("Target identity failed")
    if not non_tanker_env_exact:
        raise AssertionError("Tanker overlay changed non-tanker environment")

    model_env_complete = (
        hm[
            [
                "rel_wind_speed_kn_h",
                "rel_wind_sin_h",
                "rel_wind_cos_h",
                "wave_height_m_h",
                "rel_wave_sin_h",
                "rel_wave_cos_h",
                "wave_period_s_h",
                "sst_c_h",
                "mslp_hpa_h",
            ]
        ]
        .notna()
        .all(axis=1)
        .to_numpy()
    )
    hm["model_env_complete"] = model_env_complete

    atomic_csv(pd.DataFrame(overlay_audit), d / "Table_A1_tanker_environment_overlay_audit.csv")

    prov_cols = [
        "row_id", "_vessel", "_t", "complete_7fields", "model_env_complete",
    ]
    if use_trajectory:
        prov_cols.append("_traj")
    prov_cols += TANKER_ERA5_FIELDS + [
        "rel_wind_speed_kn_h",
        "rel_wind_sin_h",
        "rel_wind_cos_h",
        "wave_height_m_h",
        "rel_wave_sin_h",
        "rel_wave_cos_h",
        "wave_period_s_h",
        "sst_c_h",
        "mslp_hpa_h",
    ]
    prov = hm[prov_cols].copy()
    prov = prov.rename(columns={
        "_vessel": "pseudo_ship_group_id",
        "_t": "timestamp_utc",
        "_traj": "trajectory_segment_id",
    })
    atomic_csv(prov, d / "tanker_model_features_harmonized_era5.csv")

    complete_row_ids = set(
        hm.loc[hm["complete_7fields"], "row_id"].astype(int).tolist()
    )
    model_complete_row_ids = set(
        hm.loc[hm["model_env_complete"], "row_id"].astype(int).tolist()
    )

    manifest = {
        "version": VERSION,
        "tanker_rows": int(len(hm)),
        "tanker_vessels": int(raw_o.loc[tanker_mask, "vessel_id"].nunique()),
        "complete_7fields_rows": n_complete7,
        "complete_7fields_pct": float(100.0 * n_complete7 / max(len(hm), 1)),
        "model_environment_complete_rows": int(model_env_complete.sum()),
        "model_environment_complete_pct": float(100.0 * model_env_complete.mean()),
        "alignment_keys": merge_keys,
        "wind_wave_direction_convention": "ERA5 FROM -> +180 deg -> vector TO",
        "relative_angle_definition": "wrapped TO minus ship heading in [-180,180)",
        "relative_wind_speed_formula": (
            "sqrt(wind^2 + ship^2 - 2*wind*ship*cos(relative_angle))"
        ),
        "surface_temperature": "ERA5 SST only",
        "surface_pressure_conversion": "Pa/100 -> hPa",
        "source_file": str(tanker_path),
        "seven_field_range_coverage_audit": range_coverage,
        "operational_negative_control_exact": bool(op_exact),
        "target_exact": bool(target_exact),
        "non_tanker_environment_exact": bool(non_tanker_env_exact),
    }

    if tanker_audit_path is not None and Path(tanker_audit_path).exists():
        try:
            with open(str(tanker_audit_path), "r", encoding="utf-8") as f:
                manifest["interpolation_audit"] = json.load(f)
        except Exception as exc:
            manifest["interpolation_audit_read_error"] = repr(exc)

    save_json(manifest, d / "tanker_harmonization_manifest.json")

    logprint(
        log_path,
        "[PASS] tanker overlay | rows=%d | complete7=%d | model-env-complete=%d"
        % (len(hm), n_complete7, int(model_env_complete.sum()))
    )
    return (
        raw_h,
        X_h,
        hm,
        complete_row_ids,
        model_complete_row_ids,
        manifest,
    )


# ---------------------------------------------------------------------
# Design validation / fitting
# ---------------------------------------------------------------------

def validate_design(raw_o, X_o, raw_h, X_h, tr, te, paths, params, args, log_path):
    if len(raw_o) != EXPECTED_ROWS:
        raise AssertionError("Fixed31 row count changed: %d" % len(raw_o))
    if raw_o["vessel_id"].nunique() != EXPECTED_VESSELS:
        raise AssertionError(
            "Vessel count changed: %d" % raw_o["vessel_id"].nunique()
        )
    if int(raw_o["ship_type"].eq("tanker").sum()) != EXPECTED_TANKER_ROWS:
        raise AssertionError("Official tanker row count changed")
    if len(tr) != EXPECTED_TRAIN or len(te) != EXPECTED_TEST:
        raise AssertionError(
            "L1 split count changed: train=%d test=%d" % (len(tr), len(te))
        )
    if len(np.intersect1d(tr, te)) != 0:
        raise AssertionError("Train/test overlap")
    union = np.unique(np.concatenate([tr, te]))
    if len(union) != EXPECTED_ROWS:
        raise AssertionError("Train/test split does not cover Fixed31")
    if int(union.min()) != 0 or int(union.max()) != EXPECTED_ROWS - 1:
        raise AssertionError("Split index range changed")
    if list(X_o.columns) != PRIMARY_FEATURES:
        raise AssertionError("Original 17-feature order changed")
    if list(X_h.columns) != PRIMARY_FEATURES:
        raise AssertionError("Harmonized 17-feature order changed")
    if X_o.isna().any().any():
        raise AssertionError("Original canonical features unexpectedly contain NaNs")
    if raw_o["target"].isna().any():
        raise AssertionError("Original target contains NaNs")

    hashes = {}
    for k in ["fixed31", "split", "hyperparams"]:
        p = paths[k]
        h = sha256_short(p)
        hashes[p.name] = h
        if not args.skip_hash_check:
            exp = EXPECTED_SHA16.get(p.name)
            if exp is not None and h != exp:
                raise AssertionError(
                    "SHA16 mismatch for %s: %s vs expected %s"
                    % (p.name, h, exp)
                )

    manifest = {
        "version": VERSION,
        "design": {
            "rows": EXPECTED_ROWS,
            "vessels": EXPECTED_VESSELS,
            "tanker_rows": EXPECTED_TANKER_ROWS,
            "train_rows": EXPECTED_TRAIN,
            "test_rows": EXPECTED_TEST,
            "features": PRIMARY_FEATURES,
            "seed": int(args.seed),
            "xgb_params": params,
            "missing_policy": (
                "preserve official cohort; XGBoost native missing routing for "
                "harmonized ERA5 NaNs"
            ),
        },
        "hashes_sha16": hashes,
        "files": {
            k: str(v) for k, v in paths.items()
            if k != "out" and v is not None
        },
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    try:
        import xgboost
        manifest["software"]["xgboost"] = xgboost.__version__
    except Exception:
        pass

    logprint(
        log_path,
        "[PASS] locked design | rows=%d | train/test=%d/%d | tanker=%d"
        % (EXPECTED_ROWS, len(tr), len(te), EXPECTED_TANKER_ROWS)
    )
    return manifest


def fit_model_cached(Xtr, ytr, Xte, params, args, model_path, pred_path,
                     label, log_path):
    model_path = Path(model_path)
    pred_path = Path(pred_path)
    if args.resume and model_path.exists() and pred_path.exists():
        logprint(log_path, "[resume] %s" % label)
        return joblib.load(str(model_path)), np.load(str(pred_path))

    logprint(
        log_path,
        "[fit] %s | train=%d | test=%d | p=%d"
        % (label, len(Xtr), len(Xte), Xtr.shape[1])
    )
    t0 = time.time()
    model = xgb_factory(params, args.seed, args.n_jobs)
    model.fit(Xtr, ytr)
    pred = np.asarray(model.predict(Xte), dtype=float)
    joblib.dump(model, str(model_path))
    np.save(str(pred_path), pred)
    logprint(
        log_path,
        "[fit done] %s | %.1f min" % (label, (time.time() - t0) / 60.0)
    )
    return model, pred


def fit_control_models(raw_o, X_o, raw_h, X_h, tr, te, params, args, out, log_path):
    art = ensure_dir(Path(out) / "99_artifacts")
    ytr = raw_o.iloc[tr]["target"].to_numpy(float)

    original_model, original_pred = fit_model_cached(
        X_o.iloc[tr][PRIMARY_FEATURES],
        ytr,
        X_o.iloc[te][PRIMARY_FEATURES],
        params, args,
        art / "xgb_original_dynamic_physical.joblib",
        art / "pred_original_dynamic_physical.npy",
        "original dynamic_physical",
        log_path,
    )

    harmonized_model, harmonized_pred = fit_model_cached(
        X_h.iloc[tr][PRIMARY_FEATURES],
        ytr,
        X_h.iloc[te][PRIMARY_FEATURES],
        params, args,
        art / "xgb_tanker_harmonized_dynamic_physical.joblib",
        art / "pred_tanker_harmonized_dynamic_physical.npy",
        "tanker-harmonized dynamic_physical",
        log_path,
    )

    return (
        {"original": original_model, "harmonized": harmonized_model},
        {"original": original_pred, "harmonized": harmonized_pred},
    )


# ---------------------------------------------------------------------
# Predictive performance
# ---------------------------------------------------------------------

def metric_dict(y, pred):
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    mse = float(mean_squared_error(y, pred))
    rmse = math.sqrt(mse)
    mae = float(mean_absolute_error(y, pred))
    try:
        r2 = float(r2_score(y, pred))
    except Exception:
        r2 = np.nan
    sd = float(np.std(y, ddof=1)) if len(y) > 1 else np.nan
    return {
        "n": int(len(y)),
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "target_sd": sd,
        "RMSE_over_target_SD": (
            float(rmse / sd) if np.isfinite(sd) and sd > 0 else np.nan
        ),
        "mean_error_pred_minus_obs": float(np.mean(pred - y)),
    }


def paired_vessel_bootstrap_rmse(vessel_ids, y, pred_o, pred_h, reps, seed):
    ids = np.asarray(vessel_ids).astype(str)
    y = np.asarray(y, dtype=float)
    po = np.asarray(pred_o, dtype=float)
    ph = np.asarray(pred_h, dtype=float)
    vessels = np.unique(ids)
    idx = {v: np.flatnonzero(ids == v) for v in vessels}

    point = (
        math.sqrt(mean_squared_error(y, ph))
        - math.sqrt(mean_squared_error(y, po))
    )

    rng = np.random.default_rng(seed)
    vals = np.empty(reps, dtype=float)
    for i in range(reps):
        draw = rng.choice(vessels, size=len(vessels), replace=True)
        ii = np.concatenate([idx[v] for v in draw])
        rh = math.sqrt(mean_squared_error(y[ii], ph[ii]))
        ro = math.sqrt(mean_squared_error(y[ii], po[ii]))
        vals[i] = rh - ro

    return {
        "delta_RMSE_harmonized_minus_original": float(point),
        "bootstrap_CI95_low": float(np.quantile(vals, 0.025)),
        "bootstrap_CI95_high": float(np.quantile(vals, 0.975)),
        "bootstrap_prob_delta_below_zero": float(np.mean(vals < 0)),
        "bootstrap_prob_delta_above_zero": float(np.mean(vals > 0)),
        "bootstrap_reps": int(reps),
        "bootstrap_unit": "vessel",
    }


def run_performance(raw_o, raw_h, X_h, te, preds, complete_row_ids,
                    model_complete_row_ids, args, out, log_path):
    d = ensure_dir(Path(out) / "02_predictive_performance")
    te = np.asarray(te, dtype=int)
    test = raw_o.iloc[te].reset_index(drop=True)
    y = test["target"].to_numpy(float)

    is_tanker = test["ship_type"].eq("tanker").to_numpy()
    global_ids = te
    complete7 = np.array(
        [int(x) in complete_row_ids for x in global_ids], dtype=bool
    )
    model_complete = np.array(
        [int(x) in model_complete_row_ids for x in global_ids], dtype=bool
    )

    scopes = [
        ("all_L1_test", np.ones(len(te), dtype=bool)),
        ("tanker_all", is_tanker),
        ("tanker_complete7", is_tanker & complete7),
        ("tanker_model_env_complete", is_tanker & model_complete),
    ]

    rows = []
    for scope, mask in scopes:
        if not mask.any():
            continue
        for branch in ["original", "harmonized"]:
            m = metric_dict(y[mask], np.asarray(preds[branch])[mask])
            m.update({"scope": scope, "branch": branch})
            rows.append(m)
    tab = pd.DataFrame(rows)
    atomic_csv(tab, d / "Table_P1_original_vs_harmonized_metrics.csv")

    # Paired tanker RMSE inference on the strict model-environment-complete subset.
    mask = is_tanker & model_complete
    boot = paired_vessel_bootstrap_rmse(
        test.loc[mask, "vessel_id"].to_numpy(str),
        y[mask],
        np.asarray(preds["original"])[mask],
        np.asarray(preds["harmonized"])[mask],
        int(args.bootstrap),
        int(args.seed + 12000),
    )
    atomic_csv(pd.DataFrame([boot]), d / "Table_P2_tanker_paired_vessel_bootstrap_RMSE.csv")

    # By-vessel metrics on same strict rows.
    bv = []
    base = test.loc[mask, ["vessel_id", "target"]].copy()
    base["pred_original"] = np.asarray(preds["original"])[mask]
    base["pred_harmonized"] = np.asarray(preds["harmonized"])[mask]
    for vessel, g in base.groupby("vessel_id"):
        for branch, col in [
            ("original", "pred_original"),
            ("harmonized", "pred_harmonized"),
        ]:
            mm = metric_dict(g["target"], g[col])
            mm.update({"vessel_id": vessel, "branch": branch})
            bv.append(mm)
    atomic_csv(pd.DataFrame(bv), d / "Table_P3_tanker_by_vessel_metrics.csv")

    logprint(
        log_path,
        "[PASS] predictive performance | strict tanker test rows=%d"
        % int(mask.sum())
    )


# ---------------------------------------------------------------------
# Wave C4
# ---------------------------------------------------------------------

def relative_sector(sin_v, cos_v):
    s = np.asarray(sin_v, dtype=float)
    c = np.asarray(cos_v, dtype=float)
    out = np.full(len(s), "missing", dtype=object)
    good = np.isfinite(s) & np.isfinite(c)
    angle = np.full(len(s), np.nan, dtype=float)
    angle[good] = (np.degrees(np.arctan2(s[good], c[good])) + 360.0) % 360.0
    out[good] = "cross"
    head = good & ((angle <= 45.0) | (angle >= 315.0))
    following = good & (angle >= 135.0) & (angle <= 225.0)
    out[head] = "head"
    out[following] = "following"
    return out


def qcut_safe(s, labels):
    try:
        return pd.qcut(s, q=len(labels), labels=labels, duplicates="drop").astype(str)
    except Exception:
        return pd.Series(["all"] * len(s), index=s.index, dtype=object)


def sign_flip_test(values, permutations, seed):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return np.nan, np.nan, 0
    obs = float(np.mean(x))

    # Exact enumeration when feasible.
    if n <= 18:
        total = 2 ** n
        exceed = 0
        for mask in range(total):
            signs = np.ones(n, dtype=float)
            for j in range(n):
                if (mask >> j) & 1:
                    signs[j] = -1.0
            stat = float(np.mean(x * signs))
            exceed += int(abs(stat) >= abs(obs) - 1e-15)
        p = float(exceed / float(total))
        return obs, p, total

    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(int(permutations)):
        signs = rng.choice([-1.0, 1.0], size=n)
        stat = float(np.mean(x * signs))
        exceed += int(abs(stat) >= abs(obs))
    p = float((exceed + 1) / (int(permutations) + 1))
    return obs, p, int(permutations)


def paired_vessel_c4_summary(g, reps, seed, permutations):
    vessels = sorted(g["vessel_id"].astype(str).unique())
    if len(vessels) == 0:
        raise ValueError("Empty C4 group")

    by = {v: g[g["vessel_id"].astype(str) == v] for v in vessels}

    o_up = np.array(
        [float((by[v]["delta_original"] > 0).mean()) for v in vessels],
        dtype=float,
    )
    h_up = np.array(
        [float((by[v]["delta_harmonized"] > 0).mean()) for v in vessels],
        dtype=float,
    )
    o_mean = np.array(
        [float(by[v]["delta_original"].mean()) for v in vessels],
        dtype=float,
    )
    h_mean = np.array(
        [float(by[v]["delta_harmonized"].mean()) for v in vessels],
        dtype=float,
    )

    rng = np.random.default_rng(seed)
    n = len(vessels)
    bo = np.empty(reps, dtype=float)
    bh = np.empty(reps, dtype=float)
    bd_up = np.empty(reps, dtype=float)
    bmo = np.empty(reps, dtype=float)
    bmh = np.empty(reps, dtype=float)
    bd_mean = np.empty(reps, dtype=float)

    for i in range(reps):
        draw = rng.integers(0, n, size=n)
        bo[i] = o_up[draw].mean()
        bh[i] = h_up[draw].mean()
        bd_up[i] = bh[i] - bo[i]
        bmo[i] = o_mean[draw].mean()
        bmh[i] = h_mean[draw].mean()
        bd_mean[i] = bmh[i] - bmo[i]

    _, p_up, nperm_up = sign_flip_test(
        100.0 * (h_up - o_up), permutations, seed + 500000
    )
    _, p_mean, nperm_mean = sign_flip_test(
        h_mean - o_mean, permutations, seed + 600000
    )

    return {
        "n": int(len(g)),
        "n_vessels": int(n),

        "original_expected_direction_pct":
            100.0 * float((g["delta_original"] > 0).mean()),
        "harmonized_expected_direction_pct":
            100.0 * float((g["delta_harmonized"] > 0).mean()),

        "original_mean_delta_t_per_10min":
            float(g["delta_original"].mean()),
        "harmonized_mean_delta_t_per_10min":
            float(g["delta_harmonized"].mean()),
        "original_median_delta_t_per_10min":
            float(g["delta_original"].median()),
        "harmonized_median_delta_t_per_10min":
            float(g["delta_harmonized"].median()),

        "original_vessel_balanced_expected_pct":
            100.0 * float(o_up.mean()),
        "harmonized_vessel_balanced_expected_pct":
            100.0 * float(h_up.mean()),

        "original_vessel_balanced_expected_CI95_low":
            100.0 * float(np.quantile(bo, 0.025)),
        "original_vessel_balanced_expected_CI95_high":
            100.0 * float(np.quantile(bo, 0.975)),
        "harmonized_vessel_balanced_expected_CI95_low":
            100.0 * float(np.quantile(bh, 0.025)),
        "harmonized_vessel_balanced_expected_CI95_high":
            100.0 * float(np.quantile(bh, 0.975)),

        "delta_vessel_balanced_expected_pct_harmonized_minus_original":
            100.0 * float(h_up.mean() - o_up.mean()),
        "delta_vessel_balanced_expected_CI95_low":
            100.0 * float(np.quantile(bd_up, 0.025)),
        "delta_vessel_balanced_expected_CI95_high":
            100.0 * float(np.quantile(bd_up, 0.975)),
        "paired_vessel_signflip_p_expected_pct":
            float(p_up),
        "paired_vessel_signflip_permutations_expected_pct":
            int(nperm_up),

        "original_vessel_balanced_mean_delta":
            float(o_mean.mean()),
        "harmonized_vessel_balanced_mean_delta":
            float(h_mean.mean()),

        "original_vessel_balanced_mean_delta_CI95_low":
            float(np.quantile(bmo, 0.025)),
        "original_vessel_balanced_mean_delta_CI95_high":
            float(np.quantile(bmo, 0.975)),
        "harmonized_vessel_balanced_mean_delta_CI95_low":
            float(np.quantile(bmh, 0.025)),
        "harmonized_vessel_balanced_mean_delta_CI95_high":
            float(np.quantile(bmh, 0.975)),

        "delta_vessel_balanced_mean_delta_harmonized_minus_original":
            float(h_mean.mean() - o_mean.mean()),
        "delta_vessel_balanced_mean_delta_CI95_low":
            float(np.quantile(bd_mean, 0.025)),
        "delta_vessel_balanced_mean_delta_CI95_high":
            float(np.quantile(bd_mean, 0.975)),
        "paired_vessel_signflip_p_mean_delta":
            float(p_mean),
        "paired_vessel_signflip_permutations_mean_delta":
            int(nperm_mean),
    }


def summarize_paired_c4(rows, group_cols, label, args, seed):
    out = []
    grouped = rows.groupby(group_cols, dropna=False)
    for i, (key, g) in enumerate(grouped):
        if not isinstance(key, tuple):
            key = (key,)
        if len(g) < int(args.wave_min_group_n):
            continue
        if g["vessel_id"].nunique() < 2:
            continue
        base = {"stratification": label}
        for c, v in zip(group_cols, key):
            base[c] = v
        base.update(
            paired_vessel_c4_summary(
                g,
                reps=int(args.bootstrap),
                seed=int(seed + i),
                permutations=int(args.permutations),
            )
        )
        out.append(base)
    return out


def c4_for_level(level, raw_o, raw_h, te, models, preds,
                 complete_row_ids, model_complete_row_ids,
                 args, log_path):
    tr_dummy = None
    te = np.asarray(te, dtype=int)

    # Training support is derived outside this function via attributes attached
    # by caller in args. This avoids repeated full-data slicing.
    support_o = args._support_o
    support_h = args._support_h

    is_tanker_test = raw_o.iloc[te]["ship_type"].eq("tanker").to_numpy()
    test_global = te[is_tanker_test]
    test_local = np.flatnonzero(is_tanker_test)

    base_o = raw_o.iloc[test_global].reset_index(drop=True).copy()
    base_h = raw_h.iloc[test_global].reset_index(drop=True).copy()

    complete7 = np.array(
        [int(rid) in complete_row_ids for rid in test_global], dtype=bool
    )
    model_env_complete = np.array(
        [int(rid) in model_complete_row_ids for rid in test_global], dtype=bool
    )

    wave_o = pd.to_numeric(
        base_o["wave_height_m"], errors="coerce"
    ).to_numpy(float)
    wave_h = pd.to_numeric(
        base_h["wave_height_m"], errors="coerce"
    ).to_numpy(float)

    pert_o = wave_o * (1.0 + level)
    pert_h = wave_h * (1.0 + level)

    sup_o = (
        np.isfinite(pert_o)
        & (pert_o >= support_o[0])
        & (pert_o <= support_o[1])
    )
    sup_h = (
        np.isfinite(pert_h)
        & (pert_h >= support_h[0])
        & (pert_h <= support_h[1])
    )

    sector_o = relative_sector(
        base_o["rel_wave_sin"], base_o["rel_wave_cos"]
    )
    sector_h = relative_sector(
        base_h["rel_wave_sin"], base_h["rel_wave_cos"]
    )
    finite_sector = (sector_o != "missing") & (sector_h != "missing")

    joint = (
        complete7
        & model_env_complete
        & sup_o
        & sup_h
        & finite_sector
    )
    if int(joint.sum()) == 0:
        raise RuntimeError(
            "No tanker rows survived joint support at %.0f%% wave perturbation"
            % (100.0 * level)
        )

    gidx = test_global[joint]
    lidx = test_local[joint]
    bo = raw_o.iloc[gidx].reset_index(drop=True).copy()
    bh = raw_h.iloc[gidx].reset_index(drop=True).copy()

    pred_base_o = np.asarray(preds["original"], dtype=float)[lidx]
    pred_base_h = np.asarray(preds["harmonized"], dtype=float)[lidx]

    po = bo.copy()
    ph = bh.copy()
    po["wave_height_m"] = po["wave_height_m"].astype(float) * (1.0 + level)
    ph["wave_height_m"] = ph["wave_height_m"].astype(float) * (1.0 + level)

    pred_pert_o = np.asarray(
        models["original"].predict(po[PRIMARY_FEATURES]), dtype=float
    )
    pred_pert_h = np.asarray(
        models["harmonized"].predict(ph[PRIMARY_FEATURES]), dtype=float
    )

    rows = pd.DataFrame({
        "row_id": gidx.astype(int),
        "vessel_id": bo["vessel_id"].astype(str).to_numpy(),
        "timestamp": bo["timestamp"].astype(str).to_numpy(),
        "speed_kn": bo["speed_kn"].to_numpy(float),
        "wave_perturbation_pct": 100.0 * level,
        "wave_height_original_m": bo["wave_height_m"].to_numpy(float),
        "wave_height_harmonized_m": bh["wave_height_m"].to_numpy(float),
        "sector_original": relative_sector(
            bo["rel_wave_sin"], bo["rel_wave_cos"]
        ),
        "sector_harmonized": relative_sector(
            bh["rel_wave_sin"], bh["rel_wave_cos"]
        ),
        "prediction_base_original": pred_base_o,
        "prediction_wave_perturbed_original": pred_pert_o,
        "prediction_base_harmonized": pred_base_h,
        "prediction_wave_perturbed_harmonized": pred_pert_h,
    })
    rows["delta_original"] = (
        rows["prediction_wave_perturbed_original"]
        - rows["prediction_base_original"]
    )
    rows["delta_harmonized"] = (
        rows["prediction_wave_perturbed_harmonized"]
        - rows["prediction_base_harmonized"]
    )
    rows["delta_harmonized_minus_original"] = (
        rows["delta_harmonized"] - rows["delta_original"]
    )
    rows["sector_stable"] = (
        rows["sector_original"] == rows["sector_harmonized"]
    )

    logprint(
        log_path,
        "[C4 %.0f%%] tanker_test=%d | joint=%d | stable=%d"
        % (
            100.0 * level,
            len(test_global),
            len(rows),
            int(rows["sector_stable"].sum()),
        )
    )

    audit = {
        "wave_perturbation_pct": 100.0 * level,
        "official_tanker_test_rows": int(len(test_global)),
        "complete7_rows_within_tanker_test": int(complete7.sum()),
        "model_env_complete_rows_within_tanker_test":
            int(model_env_complete.sum()),
        "original_support_rows": int(sup_o.sum()),
        "harmonized_support_rows": int(sup_h.sum()),
        "joint_support_rows": int(joint.sum()),
        "joint_support_vessels": int(rows["vessel_id"].nunique()),
        "stable_sector_rows": int(rows["sector_stable"].sum()),
        "stable_sector_pct": float(100.0 * rows["sector_stable"].mean()),
        "original_training_wave_support_m": list(support_o),
        "harmonized_training_wave_support_m": list(support_h),
    }
    return rows, audit


def run_c4(raw_o, raw_h, tr, te, models, preds,
           complete_row_ids, model_complete_row_ids,
           args, out, log_path):
    d = ensure_dir(Path(out) / "03_wave_c4")

    tr = np.asarray(tr, dtype=int)
    train_o = raw_o.iloc[tr].reset_index(drop=True)
    train_h = raw_h.iloc[tr].reset_index(drop=True)
    to = pd.to_numeric(
        train_o.loc[train_o["ship_type"].eq("tanker"), "wave_height_m"],
        errors="coerce",
    ).dropna()
    th = pd.to_numeric(
        train_h.loc[train_h["ship_type"].eq("tanker"), "wave_height_m"],
        errors="coerce",
    ).dropna()
    if to.empty or th.empty:
        raise RuntimeError("Cannot derive tanker training wave support")

    args._support_o = (float(to.min()), float(to.max()))
    args._support_h = (float(th.min()), float(th.max()))

    levels = [
        float(x.strip()) for x in args.wave_levels.split(",")
        if x.strip()
    ]
    if not levels:
        raise ValueError("No wave levels")
    for x in levels:
        if x <= 0:
            raise ValueError("Wave perturbation must be >0")

    sensitivity_rows = []
    audits = []
    primary_rows = None

    for li, level in enumerate(levels):
        rows, audit = c4_for_level(
            level, raw_o, raw_h, te, models, preds,
            complete_row_ids, model_complete_row_ids,
            args, log_path,
        )
        audits.append(audit)

        stable = rows[rows["sector_stable"]].copy()
        head = stable[stable["sector_harmonized"].eq("head")].copy()
        overall = rows.copy()

        if len(overall) >= args.wave_min_group_n and overall["vessel_id"].nunique() >= 2:
            rr = paired_vessel_c4_summary(
                overall, int(args.bootstrap),
                int(args.seed + 30000 + li),
                int(args.permutations),
            )
            rr.update({
                "wave_perturbation_pct": 100.0 * level,
                "scope": "all_joint_support_tanker",
            })
            sensitivity_rows.append(rr)

        if len(head) >= args.wave_min_group_n and head["vessel_id"].nunique() >= 2:
            rr = paired_vessel_c4_summary(
                head, int(args.bootstrap),
                int(args.seed + 31000 + li),
                int(args.permutations),
            )
            rr.update({
                "wave_perturbation_pct": 100.0 * level,
                "scope": "stable_head_sea_PRIMARY",
            })
            sensitivity_rows.append(rr)

        if abs(100.0 * level - PRIMARY_WAVE_PCT) < 1e-8:
            primary_rows = rows.copy()

    atomic_csv(
        pd.DataFrame(sensitivity_rows),
        d / "Table_C4_01_wave_5_10_15_sensitivity.csv",
    )
    save_json(audits, d / "C4_support_audit_all_levels.json")

    if primary_rows is None:
        raise ValueError(
            "Primary 10%% wave level is missing from --wave-levels; include 0.10"
        )

    rows = primary_rows
    atomic_csv(
        rows,
        d / "Table_C4_02_primary10_row_level_joint_support.csv",
    )

    migration = (
        rows.groupby(["sector_original", "sector_harmonized"], dropna=False)
        .size()
        .reset_index(name="n")
    )
    migration["pct_of_joint_support"] = (
        100.0 * migration["n"] / max(len(rows), 1)
    )
    atomic_csv(
        migration,
        d / "Table_C4_03_sector_migration.csv",
    )

    stable = rows[rows["sector_stable"]].copy()
    stable_tab = pd.DataFrame(
        summarize_paired_c4(
            stable,
            ["sector_harmonized"],
            "stable_relative_wave_sector_PRIMARY",
            args,
            args.seed + 40000,
        )
    )
    atomic_csv(
        stable_tab,
        d / "Table_C4_04_stable_sector_PRIMARY.csv",
    )

    secondary_tab = pd.DataFrame(
        summarize_paired_c4(
            rows,
            ["sector_harmonized"],
            "harmonized_ERA5_sector_SECONDARY",
            args,
            args.seed + 41000,
        )
    )
    atomic_csv(
        secondary_tab,
        d / "Table_C4_05_harmonized_sector_SECONDARY.csv",
    )

    # Secondary subgroup diagnostics on primary 10% rows.
    rows2 = rows.copy()
    rows2["wave_state_harmonized"] = qcut_safe(
        rows2["wave_height_harmonized_m"],
        ["low", "moderate", "high"],
    )
    rows2["speed_band_empirical"] = qcut_safe(
        rows2["speed_kn"],
        ["low", "middle", "high"],
    )

    extra = []
    specs = [
        (["wave_state_harmonized"], "harmonized_wave_tertile"),
        (["speed_band_empirical"], "speed_tertile"),
        (
            ["wave_state_harmonized", "sector_harmonized"],
            "harmonized_wave_tertile_x_sector",
        ),
        (
            ["speed_band_empirical", "sector_harmonized"],
            "speed_tertile_x_harmonized_sector",
        ),
    ]
    for si, (cols, label) in enumerate(specs):
        extra.extend(
            summarize_paired_c4(
                rows2,
                cols,
                label,
                args,
                args.seed + 42000 + 1000 * si,
            )
        )
    atomic_csv(
        pd.DataFrame(extra),
        d / "Table_C4_06_secondary_strata.csv",
    )

    head_primary = stable_tab[
        stable_tab.get(
            "sector_harmonized", pd.Series(dtype=object)
        ).astype(str).eq("head")
    ].copy()
    if head_primary.empty:
        raise RuntimeError(
            "No stable-sector head-sea primary group met n/vessel criteria"
        )
    head_primary.insert(0, "wave_perturbation_pct", PRIMARY_WAVE_PCT)
    atomic_csv(
        head_primary,
        d / "Table_C4_07_HEADSEA_PRIMARY_manuscript.csv",
    )

    logprint(
        log_path,
        "[PASS] C4 primary head-sea table written | n=%d | vessels=%d"
        % (
            int(head_primary.iloc[0]["n"]),
            int(head_primary.iloc[0]["n_vessels"]),
        )
    )


# ---------------------------------------------------------------------
# Manuscript summary
# ---------------------------------------------------------------------

def run_summary(out, log_path):
    out = Path(out)
    d = ensure_dir(out / "04_summary")

    summary_rows = []

    p = out / "02_predictive_performance" / "Table_P1_original_vs_harmonized_metrics.csv"
    if p.exists():
        z = pd.read_csv(p)
        for scope in ["tanker_all", "tanker_complete7", "tanker_model_env_complete"]:
            g = z[z["scope"].eq(scope)]
            if set(g["branch"]) >= {"original", "harmonized"}:
                oo = g[g["branch"].eq("original")].iloc[0]
                hh = g[g["branch"].eq("harmonized")].iloc[0]
                summary_rows.append({
                    "experiment": "predictive_performance",
                    "endpoint": scope + "_RMSE",
                    "original": oo["RMSE"],
                    "harmonized": hh["RMSE"],
                    "delta_harmonized_minus_original":
                        hh["RMSE"] - oo["RMSE"],
                })
                summary_rows.append({
                    "experiment": "predictive_performance",
                    "endpoint": scope + "_R2",
                    "original": oo["R2"],
                    "harmonized": hh["R2"],
                    "delta_harmonized_minus_original":
                        hh["R2"] - oo["R2"],
                })

    p = out / "03_wave_c4" / "Table_C4_07_HEADSEA_PRIMARY_manuscript.csv"
    if p.exists():
        z = pd.read_csv(p)
        if not z.empty:
            r = z.iloc[0]
            summary_rows.append({
                "experiment": "tanker_wave_C4_primary",
                "endpoint": "stable_headsea_vessel_balanced_expected_direction_pct",
                "original": r["original_vessel_balanced_expected_pct"],
                "harmonized": r["harmonized_vessel_balanced_expected_pct"],
                "delta_harmonized_minus_original":
                    r["delta_vessel_balanced_expected_pct_harmonized_minus_original"],
            })
            summary_rows.append({
                "experiment": "tanker_wave_C4_primary",
                "endpoint": "stable_headsea_vessel_balanced_mean_prediction_delta",
                "original": r["original_vessel_balanced_mean_delta"],
                "harmonized": r["harmonized_vessel_balanced_mean_delta"],
                "delta_harmonized_minus_original":
                    r["delta_vessel_balanced_mean_delta_harmonized_minus_original"],
            })

    atomic_csv(
        pd.DataFrame(summary_rows),
        d / "tanker_harmonization_summary.csv",
    )

    # Human-readable summary.
    md = []
    md.append("# Tanker ERA5 harmonization control — manuscript-ready summary")
    md.append("")
    md.append("This run isolates tanker meteorological preprocessing as the treatment:")
    md.append("same Fixed31 cohort, same L1 split, same locked XGB hyperparameters, same target.")
    md.append("")

    align = out / "01_alignment" / "tanker_harmonization_manifest.json"
    if align.exists():
        with open(str(align), "r", encoding="utf-8") as f:
            a = json.load(f)
        md.append("## Alignment")
        md.append("- Tanker rows: %s" % a.get("tanker_rows"))
        md.append("- ERA5 complete seven-field rows: %s (%.2f%%)" % (
            a.get("complete_7fields_rows"),
            a.get("complete_7fields_pct", np.nan),
        ))
        md.append("- Model-environment-complete rows: %s (%.2f%%)" % (
            a.get("model_environment_complete_rows"),
            a.get("model_environment_complete_pct", np.nan),
        ))
        md.append("")
        md.append("## ERA5 harmonisation audit")
        md.append(
            "- Hard method audit: bilinear spatial interpolation; linear temporal "
            "interpolation; no nearest-neighbour fallback; circular wave-direction "
            "interpolation; wind direction derived from interpolated u10/v10; SST only."
        )
        md.append(
            "- Output units/ranges audited before modelling: wind speed in kn; "
            "directions in [0,360) deg FROM; wave height in m; wave period in s; "
            "SST in degC; surface pressure in Pa."
        )
        md.append("")

    hp = out / "03_wave_c4" / "Table_C4_07_HEADSEA_PRIMARY_manuscript.csv"
    if hp.exists():
        h = pd.read_csv(hp)
        if not h.empty:
            r = h.iloc[0]
            md.append("## Primary +10% wave-height stable head-sea result")
            md.append("- Paired rows: %d; vessels: %d" % (
                int(r["n"]), int(r["n_vessels"])
            ))
            md.append(
                "- Vessel-balanced expected-direction: original %.2f%% "
                "[%.2f, %.2f] vs harmonized %.2f%% [%.2f, %.2f]."
                % (
                    r["original_vessel_balanced_expected_pct"],
                    r["original_vessel_balanced_expected_CI95_low"],
                    r["original_vessel_balanced_expected_CI95_high"],
                    r["harmonized_vessel_balanced_expected_pct"],
                    r["harmonized_vessel_balanced_expected_CI95_low"],
                    r["harmonized_vessel_balanced_expected_CI95_high"],
                )
            )
            md.append(
                "- Harmonized-minus-original difference: %.2f percentage points "
                "[%.2f, %.2f]; paired vessel sign-flip p=%.6g."
                % (
                    r["delta_vessel_balanced_expected_pct_harmonized_minus_original"],
                    r["delta_vessel_balanced_expected_CI95_low"],
                    r["delta_vessel_balanced_expected_CI95_high"],
                    r["paired_vessel_signflip_p_expected_pct"],
                )
            )
            md.append(
                "- Vessel-balanced mean prediction delta: original %.6f vs "
                "harmonized %.6f t/10 min."
                % (
                    r["original_vessel_balanced_mean_delta"],
                    r["harmonized_vessel_balanced_mean_delta"],
                )
            )
            md.append("")

    md.append("## Interpretation boundary")
    md.append(
        "These are paired predictive-response diagnostics under two environmental "
        "preprocessing pathways. Harmonisation reduces one specific source/pathway "
        "inconsistency; it does not remove broader vessel-type confounding arising "
        "from vessel design, propulsion, routes, operating envelopes, or other "
        "observed/unobserved differences. The results are not causal estimates of "
        "meteorological data-source effects."
    )

    (d / "MANUSCRIPT_READY_SUMMARY.md").write_text(
        "\n".join(md), encoding="utf-8"
    )
    logprint(log_path, "[PASS] manuscript-ready summary written")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    t0 = time.time()
    args = parse_args()
    paths = resolve_paths(args)

    out = paths["out"]
    ensure_dir(out / "01_alignment")
    ensure_dir(out / "02_predictive_performance")
    ensure_dir(out / "03_wave_c4")
    ensure_dir(out / "04_summary")
    ensure_dir(out / "99_artifacts")

    log_path = out / "run_overnight.log"
    if not (args.resume and log_path.exists()):
        log_path.write_text("", encoding="utf-8")

    steps = [x.strip().lower() for x in args.steps.split(",") if x.strip()]
    if "all" in steps:
        steps = ["alignment", "fit", "performance", "c4", "summary"]

    logprint(log_path, "=" * 88)
    logprint(log_path, "TANKER ERA5 HARMONIZATION + HEAD-SEA C4 OVERNIGHT RUN")
    logprint(log_path, "version: %s" % VERSION)
    logprint(log_path, "steps  : %s" % steps)
    logprint(log_path, "output : %s" % out)
    logprint(log_path, "=" * 88)

    for k in [
        "fixed31", "tanker_era5", "tanker_audit",
        "tanker_monthly_audit", "split", "hyperparams"
    ]:
        logprint(log_path, "%-14s %s" % (k + ":", paths[k]))

    logprint(log_path, "[load] Fixed31")
    fixed = pd.read_csv(paths["fixed31"], low_memory=False)
    raw_o, X_o = build_canonical_original(fixed)
    del fixed
    gc.collect()

    tr, te, train_key, test_key = load_split(paths["split"])
    params = load_xgb_params(paths["hyperparams"])

    logprint(log_path, "[audit] tanker interpolation method / provenance")
    interpolation_audit, monthly_summary = validate_tanker_interpolation_audit(
        paths["tanker_audit"],
        paths["tanker_monthly_audit"],
        out / "01_alignment",
        args,
        log_path,
    )

    logprint(log_path, "[overlay] tanker ERA5")
    (
        raw_h,
        X_h,
        hm,
        complete_row_ids,
        model_complete_row_ids,
        tanker_manifest,
    ) = build_tanker_harmonized_overlay(
        raw_o,
        X_o,
        paths["tanker_era5"],
        paths["tanker_audit"],
        out / "01_alignment",
        args,
        log_path,
    )

    design_manifest = validate_design(
        raw_o, X_o, raw_h, X_h, tr, te, paths, params, args, log_path
    )
    design_manifest["split_keys"] = {
        "train": train_key,
        "test": test_key,
    }
    design_manifest["tanker_harmonization"] = tanker_manifest
    design_manifest["interpolation_method_hard_audit"] = interpolation_audit
    design_manifest["monthly_coverage_summary"] = monthly_summary
    save_json(
        design_manifest,
        out / "01_alignment" / "locked_design_manifest.json",
    )

    # Missingness by official split.
    split_label = np.full(len(raw_o), "", dtype=object)
    split_label[tr] = "train"
    split_label[te] = "test"
    tanker_mask = raw_o["ship_type"].eq("tanker").to_numpy()
    miss = []
    for c in ENV_FEATURES:
        for sp in ["all", "train", "test"]:
            m = (
                tanker_mask
                if sp == "all"
                else (tanker_mask & (split_label == sp))
            )
            miss.append({
                "feature": c,
                "split": sp,
                "rows": int(m.sum()),
                "original_missing": int(X_o.loc[m, c].isna().sum()),
                "harmonized_missing": int(X_h.loc[m, c].isna().sum()),
                "harmonized_missing_pct": (
                    float(100.0 * X_h.loc[m, c].isna().mean())
                    if m.sum() else np.nan
                ),
            })
    atomic_csv(
        pd.DataFrame(miss),
        out / "01_alignment" / "Table_A2_tanker_missingness_by_split.csv",
    )

    if steps == ["alignment"]:
        logprint(log_path, "Alignment-only run complete.")
        return

    models = None
    preds = None
    if any(x in steps for x in ["fit", "performance", "c4", "summary"]):
        models, preds = fit_control_models(
            raw_o, X_o, raw_h, X_h, tr, te, params, args, out, log_path
        )

    if "performance" in steps:
        run_performance(
            raw_o, raw_h, X_h, te, preds,
            complete_row_ids, model_complete_row_ids,
            args, out, log_path,
        )

    if "c4" in steps:
        run_c4(
            raw_o, raw_h, tr, te, models, preds,
            complete_row_ids, model_complete_row_ids,
            args, out, log_path,
        )

    if "summary" in steps:
        run_summary(out, log_path)

    logprint(log_path, "=" * 88)
    logprint(
        log_path,
        "DONE | elapsed %.2f h | output=%s"
        % ((time.time() - t0) / 3600.0, out)
    )
    logprint(log_path, "=" * 88)


if __name__ == "__main__":
    main()
