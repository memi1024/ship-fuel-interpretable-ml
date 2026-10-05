#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone AIStudio runner for the Fixed31 bulk-carrier ERA5 harmonization control.

Locked design reconstructed from the uploaded F31 final/revision scripts:
- exact Fixed31 cruise cohort: 489,620 rows / 21 vessels
- exact official L1 split: 391,696 train / 97,924 test
- target: fuel_t_10min (t / 10 min)
- locked 17-feature XGB specification
- XGB hyperparameters read from 02_best_hyperparameters.csv, no retuning
- seed: 20260808
- ablation: operational_core / weather_only / dynamic_physical / dynamic_interaction
- bulk-only six-fold locked-hyperparameter LOVO (each official bulk vessel held out)
- interventional TreeSHAP: deterministic official-test sample, training-derived background
- fixed-time (FT) 5/10/15% speed-reduction CII-proxy scenario with original support logic
- vessel-bootstrap FT endpoint confidence intervals

IMPORTANT DESIGN CHOICE FOR THIS CONTROL
----------------------------------------
The historical F31 loader applies complete-case filtering after canonicalization.
The harmonized ERA5 branch contains a small number of strict-SST/wave NaNs. Dropping
those rows would change the official cohort and invalidate the paired control.
Therefore this runner:
  1) builds the locked 489,620-row canonical cohort from ORIGINAL Fixed31;
  2) copies it in memory;
  3) overlays only the official six bulk vessels' harmonized environmental values;
  4) preserves every official row and the exact locked split;
  5) lets XGBoost use its native missing-value routing for harmonized ERA5 NaNs.

The already-generated harmonized control CSV remains a provenance/data product, but
model fitting here uses an in-memory overlay to avoid CSV float round-trip noise in
non-environmental variables.

Python 3.7 compatible.
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

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import joblib


VERSION = "2026-08-25.bulk-harmonization-control-v1"
EXPECTED_ROWS = 489620
EXPECTED_TRAIN = 391696
EXPECTED_TEST = 97924
EXPECTED_VESSELS = 21
EXPECTED_BULK_ROWS = 178760
EXPECTED_BULK_VESSELS = 6
EXPECTED_SUPPORTED = {5.0: 96940, 10.0: 95244, 15.0: 92275}

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
    "speed_kn", "heading_sin", "heading_cos", "draught_m", "trim_m",
    "rudder_deg", "ship_type_bulk", "ship_type_container",
]

WEATHER_FEATURES = [
    "rel_wind_speed_kn", "rel_wind_sin", "rel_wind_cos", "wave_height_m",
    "rel_wave_sin", "rel_wave_cos", "wave_period_s", "sst_c", "mslp_hpa",
    "ship_type_bulk", "ship_type_container",
]

# Master-column -> canonical-column. Pressure is converted Pa -> hPa below.
HARM_MASTER_TO_CANONICAL = {
    "rel_wind_speed_kn": "rel_wind_speed_kn",
    "relative_wind_sin": "rel_wind_sin",
    "relative_wind_cos": "rel_wind_cos",
    "wave_height_m": "wave_height_m",
    "relative_wave_sin": "rel_wave_sin",
    "relative_wave_cos": "rel_wave_cos",
    "wave_period_s": "wave_period_s",
    "surface_temperature_c": "sst_c",
    "surface_pressure_pa": "mslp_hpa",
}

OFFICIAL_BULK_IDS_EXPECTED = {
    "BULK_F1C32B494_P0001",
    "BULK_F37F22765_P0001",
    "BULK_F851F5ED7_P0001",
    "BULK_F88779A60_P0001",
    "BULK_FCDD60B9A_P0001",
    "BULK_FFF122C98_P0001",
}


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Run the paired Fixed31 bulk ERA5 harmonization control on AIStudio."
    )
    p.add_argument("--home", default=".")
    p.add_argument("--fixed31", default=None)
    p.add_argument("--harmonized-bulk", default=None)
    p.add_argument("--harmonized-audit", default=None)
    p.add_argument("--split", default=None)
    p.add_argument("--hyperparams", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument(
        "--steps",
        default="alignment",
        help="Comma list: alignment,ablation,lovo,shap,cii,wave_c4,summary or all"
    )
    p.add_argument("--resume", action="store_true")
    p.add_argument("--n-jobs", type=int, default=2)
    p.add_argument("--seed", type=int, default=20260808)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--shap-eval-n", type=int, default=12000)
    p.add_argument("--shap-background-n", type=int, default=256)
    p.add_argument("--shap-batch-size", type=int, default=500)
    p.add_argument("--co2-factor", type=float, default=3.114)
    p.add_argument("--reductions", default="0.05,0.10,0.15")
    return p.parse_args()


def resolve_paths(args):
    home = Path(args.home)
    paths = {
        "fixed31": Path(args.fixed31) if args.fixed31 else home / "final_fixed31_cruise.csv",
        "harmonized_bulk": Path(args.harmonized_bulk) if args.harmonized_bulk else home / "bulk_model_10min_features_harmonized_era5.csv",
        "harmonized_audit": Path(args.harmonized_audit) if args.harmonized_audit else home / "bulk_model_10min_features_harmonized_era5_audit.json",
        "split": Path(args.split) if args.split else home / "record_split_indices.npz",
        "hyperparams": Path(args.hyperparams) if args.hyperparams else home / "02_best_hyperparameters.csv",
        "out": Path(args.output_dir) if args.output_dir else home / "harmonization_control_results",
    }
    for key in ["fixed31", "harmonized_bulk", "split", "hyperparams"]:
        if not paths[key].exists():
            raise FileNotFoundError("%s not found: %s" % (key, paths[key]))
    paths["out"].mkdir(parents=True, exist_ok=True)
    return paths


def ensure_dirs(out):
    names = [
        "01_alignment",
        "02_ablation",
        "03_bulk_lovo",
        "04_harmonized_shap",
        "05_ft_cii",
        "06_wave_c4",
        "07_summary",
        "99_artifacts",
    ]
    for n in names:
        (out / n).mkdir(parents=True, exist_ok=True)


def sha256_short(path):
    h = hashlib.sha256()
    with open(str(path), "rb") as f:
        while True:
            block = f.read(1024 * 1024)
            if not block:
                break
            h.update(block)
    return h.hexdigest()[:16]


def normalize_ship_type(s):
    z = s.astype(str).str.strip().str.lower()
    out = z.copy()
    out[z.str.contains("bulk", na=False)] = "bulk"
    out[z.str.contains("container", na=False)] = "container"
    out[z.str.contains("tank", na=False)] = "tanker"
    return out


def build_canonical_original(df):
    required = [
        "ship_type", "pseudo_ship_group_id", "timestamp_utc", "fuel_t_10min",
        "speed_kn", "heading_sin", "heading_cos", "mean_draught_m", "trim_m",
        "rudder_deg", "rel_wind_speed_kn", "relative_wind_sin", "relative_wind_cos",
        "wave_height_m", "relative_wave_sin", "relative_wave_cos", "wave_period_s",
        "surface_temperature_c", "surface_pressure_pa", "distance_nm_10min",
        "deadweight_t",
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
    raw["draught_m"] = pd.to_numeric(df["mean_draught_m"], errors="coerce").to_numpy(float)
    raw["trim_m"] = pd.to_numeric(df["trim_m"], errors="coerce").to_numpy(float)
    raw["rudder_deg"] = pd.to_numeric(df["rudder_deg"], errors="coerce").to_numpy(float)
    raw["rel_wind_speed_kn"] = pd.to_numeric(df["rel_wind_speed_kn"], errors="coerce").to_numpy(float)
    raw["rel_wind_sin"] = pd.to_numeric(df["relative_wind_sin"], errors="coerce").to_numpy(float)
    raw["rel_wind_cos"] = pd.to_numeric(df["relative_wind_cos"], errors="coerce").to_numpy(float)
    raw["wave_height_m"] = pd.to_numeric(df["wave_height_m"], errors="coerce").to_numpy(float)
    raw["rel_wave_sin"] = pd.to_numeric(df["relative_wave_sin"], errors="coerce").to_numpy(float)
    raw["rel_wave_cos"] = pd.to_numeric(df["relative_wave_cos"], errors="coerce").to_numpy(float)
    raw["wave_period_s"] = pd.to_numeric(df["wave_period_s"], errors="coerce").to_numpy(float)
    raw["sst_c"] = pd.to_numeric(df["surface_temperature_c"], errors="coerce").to_numpy(float)
    raw["mslp_hpa"] = pd.to_numeric(df["surface_pressure_pa"], errors="coerce").to_numpy(float) / 100.0
    raw["ship_type_bulk"] = (raw["ship_type"] == "bulk").astype(np.int8)
    raw["ship_type_container"] = (raw["ship_type"] == "container").astype(np.int8)
    raw["distance_nm"] = pd.to_numeric(df["distance_nm_10min"], errors="coerce").to_numpy(float)
    raw["dwt"] = pd.to_numeric(df["deadweight_t"], errors="coerce").to_numpy(float)
    if "trajectory_segment_id" in df.columns:
        raw["trajectory_group"] = df["trajectory_segment_id"].astype(str).to_numpy()
    else:
        raw["trajectory_group"] = raw["vessel_id"].astype(str) + "_ROW"

    X = raw[PRIMARY_FEATURES].copy()
    return raw.reset_index(drop=True), X.reset_index(drop=True)


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
    row = tab.loc[tab["model"].astype(str).str.upper() == "XGB"]
    if row.empty:
        raise ValueError("No XGB row in %s" % path)
    return json.loads(str(row.iloc[0]["best_parameters_json"]))


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


def metric_dict(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(math.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    denom = np.maximum(np.abs(y_true), eps)
    mape = float(np.mean(np.abs(err) / denom) * 100.0)
    smape = float(np.mean(2.0 * np.abs(err) / np.maximum(np.abs(y_true) + np.abs(y_pred), eps)) * 100.0)
    try:
        r2 = float(r2_score(y_true, y_pred))
    except Exception:
        r2 = float("nan")
    total_true = float(np.sum(y_true))
    bias_pct = float((np.sum(y_pred) - total_true) / total_true * 100.0) if abs(total_true) > eps else float("nan")
    target_sd = float(np.std(y_true, ddof=1)) if len(y_true) > 1 else float("nan")
    nrmse_sd = float(rmse / target_sd) if np.isfinite(target_sd) and target_sd > 0 else float("nan")
    return {
        "n": int(len(y_true)), "MSE": mse, "RMSE": rmse, "MAE": mae,
        "MAPE_pct": mape, "sMAPE_pct": smape, "R2": r2,
        "target_sd": target_sd, "NRMSE_sd": nrmse_sd,
        "cumulative_bias_pct": bias_pct,
    }


def interaction_feature_matrix(X):
    z = X.copy()
    z["speed_x_wind"] = z["speed_kn"] * z["rel_wind_speed_kn"]
    z["speed_x_wave"] = z["speed_kn"] * z["wave_height_m"]
    z["speed_x_draught"] = z["speed_kn"] * z["draught_m"]
    z["speed_x_trim"] = z["speed_kn"] * z["trim_m"]
    z["draught_x_trim"] = z["draught_m"] * z["trim_m"]
    return z


def deterministic_shap_sample(record_te, n, seed):
    rng = np.random.default_rng(seed)
    if len(record_te) <= n:
        local = np.arange(len(record_te), dtype=int)
    else:
        local = np.sort(rng.choice(len(record_te), size=n, replace=False))
    return local, record_te[local]


def overlay_harmonized_bulk(raw_o, X_o, master_path, out_dir):
    needed = ["pseudo_ship_group_id", "timestamp_utc"] + list(HARM_MASTER_TO_CANONICAL.keys())
    optional_interactions = ["rel_wind_speed_x_speed", "wave_height_x_speed"]
    header = pd.read_csv(master_path, nrows=0)
    for c in optional_interactions:
        if c in header.columns:
            needed.append(c)
    h = pd.read_csv(master_path, usecols=needed, low_memory=False)
    h["_t"] = pd.to_datetime(h["timestamp_utc"], errors="coerce", utc=True)

    bulk_mask = raw_o["ship_type"].eq("bulk").to_numpy()
    official_ids = set(raw_o.loc[bulk_mask, "vessel_id"].astype(str).unique())
    if official_ids != OFFICIAL_BULK_IDS_EXPECTED:
        raise AssertionError("Official bulk IDs changed: %s" % sorted(official_ids))

    keys = raw_o.loc[bulk_mask, ["row_id", "vessel_id", "timestamp"]].copy()
    keys = keys.rename(columns={"vessel_id": "pseudo_ship_group_id", "timestamp": "_t"})

    hm = h[h["pseudo_ship_group_id"].astype(str).isin(official_ids)].merge(
        keys, on=["pseudo_ship_group_id", "_t"], how="inner"
    )
    if len(hm) != EXPECTED_BULK_ROWS:
        raise AssertionError("Expected %d matched bulk rows; got %d" % (EXPECTED_BULK_ROWS, len(hm)))
    if hm["row_id"].duplicated().any():
        raise AssertionError("Harmonized matched subset is not one-to-one by official Fixed31 row_id")

    raw_h = raw_o.copy(deep=True)
    X_h = X_o.copy(deep=True)
    rid = hm["row_id"].to_numpy(dtype=int)

    audit_rows = []
    for src, canon in HARM_MASTER_TO_CANONICAL.items():
        vals = pd.to_numeric(hm[src], errors="coerce").to_numpy(float)
        if src == "surface_pressure_pa":
            vals = vals / 100.0
        old = raw_o.loc[rid, canon].to_numpy(float)
        raw_h.loc[rid, canon] = vals
        X_h.loc[rid, canon] = vals
        same = np.isclose(old, vals, rtol=1e-10, atol=1e-12, equal_nan=True)
        audit_rows.append({
            "canonical_feature": canon,
            "master_column": src,
            "changed_rows": int((~same).sum()),
            "original_missing": int(np.isnan(old).sum()),
            "harmonized_missing": int(np.isnan(vals).sum()),
        })

    # Verify official precomputed environment interactions when available, but model
    # dynamic_interaction will be reconstructed from the canonical feature matrix.
    interaction_audit = []
    for master_col, calc in [
        ("rel_wind_speed_x_speed", raw_h.loc[rid, "rel_wind_speed_kn"].to_numpy(float) * raw_h.loc[rid, "speed_kn"].to_numpy(float)),
        ("wave_height_x_speed", raw_h.loc[rid, "wave_height_m"].to_numpy(float) * raw_h.loc[rid, "speed_kn"].to_numpy(float)),
    ]:
        if master_col in hm.columns:
            v = pd.to_numeric(hm[master_col], errors="coerce").to_numpy(float)
            ok = np.isclose(v, calc, rtol=1e-10, atol=1e-10, equal_nan=True)
            interaction_audit.append({
                "master_column": master_col,
                "rows": int(len(v)),
                "mismatched_vs_recomputed": int((~ok).sum()),
                "max_abs_diff": float(np.nanmax(np.abs(v - calc))) if np.isfinite(v - calc).any() else 0.0,
            })

    # Hard negative-control identity checks.
    op_exact = np.allclose(
        X_o[OPERATIONAL_FEATURES].to_numpy(float),
        X_h[OPERATIONAL_FEATURES].to_numpy(float),
        rtol=0.0, atol=0.0, equal_nan=True,
    )
    target_exact = np.allclose(raw_o["target"], raw_h["target"], rtol=0.0, atol=0.0, equal_nan=True)
    if not op_exact or not target_exact:
        raise AssertionError("Operational negative-control or target identity failed before modelling")

    pd.DataFrame(audit_rows).to_csv(out_dir / "environment_overlay_audit.csv", index=False)
    pd.DataFrame(interaction_audit).to_csv(out_dir / "interaction_recompute_audit.csv", index=False)
    return raw_h, X_h, hm, audit_rows, interaction_audit


def validate_design(raw_o, X_o, raw_h, X_h, tr, te, paths, params, args):
    if len(raw_o) != EXPECTED_ROWS or len(raw_h) != EXPECTED_ROWS:
        raise AssertionError("Fixed31 row count changed")
    if raw_o["vessel_id"].nunique() != EXPECTED_VESSELS:
        raise AssertionError("Expected %d vessels, got %d" % (EXPECTED_VESSELS, raw_o["vessel_id"].nunique()))
    bulk = raw_o["ship_type"].eq("bulk")
    if int(bulk.sum()) != EXPECTED_BULK_ROWS or raw_o.loc[bulk, "vessel_id"].nunique() != EXPECTED_BULK_VESSELS:
        raise AssertionError("Official bulk cohort changed")
    if len(tr) != EXPECTED_TRAIN or len(te) != EXPECTED_TEST:
        raise AssertionError("Official L1 split counts changed")
    if len(np.intersect1d(tr, te)) != 0:
        raise AssertionError("Train/test split overlaps")
    if len(np.unique(np.concatenate([tr, te]))) != EXPECTED_ROWS:
        raise AssertionError("Train/test split does not cover all Fixed31 rows")
    if max(int(tr.max()), int(te.max())) != EXPECTED_ROWS - 1:
        raise AssertionError("Split max index changed")
    if list(X_o.columns) != PRIMARY_FEATURES or list(X_h.columns) != PRIMARY_FEATURES:
        raise AssertionError("Locked 17-feature order changed")
    if X_o.isna().any().any():
        raise AssertionError("Original Fixed31 canonical features unexpectedly contain missing values")
    if raw_o["target"].isna().any():
        raise AssertionError("Original target contains missing values")

    interp = None
    if paths["harmonized_audit"].exists():
        with open(str(paths["harmonized_audit"]), "r", encoding="utf-8") as f:
            interp = json.load(f)
        checks = {
            "rows_output": 298475,
            "spatial_interpolation": "bilinear",
            "temporal_interpolation": "linear",
            "nearest_neighbor_used": False,
        }
        for k, expected in checks.items():
            if k in interp and interp[k] != expected:
                raise AssertionError("Interpolation audit mismatch %s=%r expected %r" % (k, interp[k], expected))

    missing_env = {}
    for c in ["rel_wind_speed_kn", "rel_wind_sin", "rel_wind_cos", "wave_height_m", "rel_wave_sin", "rel_wave_cos", "wave_period_s", "sst_c", "mslp_hpa"]:
        missing_env[c] = int(X_h.loc[bulk, c].isna().sum())

    manifest = {
        "version": VERSION,
        "design": {
            "rows": EXPECTED_ROWS,
            "vessels": EXPECTED_VESSELS,
            "bulk_rows": EXPECTED_BULK_ROWS,
            "bulk_vessels": EXPECTED_BULK_VESSELS,
            "train_rows": EXPECTED_TRAIN,
            "test_rows": EXPECTED_TEST,
            "primary_features": PRIMARY_FEATURES,
            "operational_features": OPERATIONAL_FEATURES,
            "weather_features": WEATHER_FEATURES,
            "target": "fuel_t_10min -> canonical target",
            "seed": args.seed,
            "xgb_params": params,
            "missing_policy": "preserve exact official cohort; XGBoost native missing routing for harmonized ERA5 NaNs",
        },
        "bulk_environment_missing": missing_env,
        "files": {},
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "interpolation_audit": interp,
    }
    try:
        import xgboost
        manifest["software"]["xgboost"] = xgboost.__version__
    except Exception:
        pass
    for k in ["fixed31", "harmonized_bulk", "harmonized_audit", "split", "hyperparams"]:
        p = paths[k]
        if p.exists():
            manifest["files"][k] = {"path": str(p), "bytes": p.stat().st_size, "sha256_16": sha256_short(p)}
    return manifest


def save_json(obj, path):
    with open(str(path), "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def fit_model_cached(Xtr, ytr, Xte, params, args, model_path, pred_path, label):
    if args.resume and model_path.exists() and pred_path.exists():
        print("[resume]", label)
        return joblib.load(str(model_path)), np.load(str(pred_path))
    print("[fit]", label, "train=", len(Xtr), "test=", len(Xte), "features=", Xtr.shape[1])
    model = xgb_factory(params, args.seed, args.n_jobs)
    model.fit(Xtr, ytr)
    pred = np.asarray(model.predict(Xte), dtype=float)
    joblib.dump(model, str(model_path))
    np.save(str(pred_path), pred)
    return model, pred


def cluster_bootstrap_metric_diff(vessel_ids, y, pred_a, pred_b, reps, seed):
    ids = np.asarray(vessel_ids).astype(str)
    y = np.asarray(y, dtype=float)
    a = np.asarray(pred_a, dtype=float)
    b = np.asarray(pred_b, dtype=float)
    uniq = np.unique(ids)
    rng = np.random.default_rng(seed)
    index_by = {u: np.flatnonzero(ids == u) for u in uniq}
    vals = np.empty(reps, dtype=float)
    for i in range(reps):
        draw = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([index_by[u] for u in draw])
        ra = math.sqrt(mean_squared_error(y[idx], a[idx]))
        rb = math.sqrt(mean_squared_error(y[idx], b[idx]))
        vals[i] = ra - rb
    return float(vals.mean()), float(np.quantile(vals, .025)), float(np.quantile(vals, .975))


def run_ablation(raw_o, X_o, raw_h, X_h, tr, te, params, args, out):
    d = out / "02_ablation"
    art = out / "99_artifacts"
    ytr = raw_o.iloc[tr]["target"].to_numpy(float)
    yte = raw_o.iloc[te]["target"].to_numpy(float)
    rawte = raw_o.iloc[te].reset_index(drop=True)

    configs = {
        "operational_core": lambda X: X[OPERATIONAL_FEATURES],
        "weather_only": lambda X: X[WEATHER_FEATURES],
        "dynamic_physical": lambda X: X[PRIMARY_FEATURES],
        "dynamic_interaction": lambda X: interaction_feature_matrix(X[PRIMARY_FEATURES]),
    }

    # Negative control: operational matrices MUST be exactly identical.
    op_train_same = np.allclose(
        configs["operational_core"](X_o.iloc[tr]).to_numpy(float),
        configs["operational_core"](X_h.iloc[tr]).to_numpy(float),
        rtol=0.0, atol=0.0, equal_nan=True,
    )
    op_test_same = np.allclose(
        configs["operational_core"](X_o.iloc[te]).to_numpy(float),
        configs["operational_core"](X_h.iloc[te]).to_numpy(float),
        rtol=0.0, atol=0.0, equal_nan=True,
    )
    if not (op_train_same and op_test_same):
        raise AssertionError("Operational-core negative control failed")

    preds = {}
    models = {}

    # Fit operational core once because both branches have exactly the same X and y.
    xtr_op = configs["operational_core"](X_o.iloc[tr])
    xte_op = configs["operational_core"](X_o.iloc[te])
    op_model, op_pred = fit_model_cached(
        xtr_op, ytr, xte_op, params, args,
        art / "xgb_shared_operational_core.joblib",
        art / "pred_shared_operational_core.npy",
        "shared operational_core negative control",
    )
    for branch in ["original", "harmonized"]:
        preds[(branch, "operational_core")] = op_pred.copy()
        models[(branch, "operational_core")] = op_model

    for branch, X in [("original", X_o), ("harmonized", X_h)]:
        for label in ["weather_only", "dynamic_physical", "dynamic_interaction"]:
            xtr = configs[label](X.iloc[tr])
            xte = configs[label](X.iloc[te])
            model, pred = fit_model_cached(
                xtr, ytr, xte, params, args,
                art / ("xgb_%s_%s.joblib" % (branch, label)),
                art / ("pred_%s_%s.npy" % (branch, label)),
                "%s %s" % (branch, label),
            )
            models[(branch, label)] = model
            preds[(branch, label)] = pred
            del xtr, xte
            gc.collect()

    rows = []
    by_ship = []
    by_vessel = []
    for branch in ["original", "harmonized"]:
        for label in configs:
            p = preds[(branch, label)]
            rows.append({"branch": branch, "configuration": label, "scope": "fleet", "n_features": int(configs[label](X_o.iloc[:1]).shape[1]), **metric_dict(yte, p)})
            for st in ["bulk", "container", "tanker"]:
                m = rawte["ship_type"].eq(st).to_numpy()
                by_ship.append({"branch": branch, "configuration": label, "ship_type": st, "n_features": int(configs[label](X_o.iloc[:1]).shape[1]), **metric_dict(yte[m], p[m])})
            for vessel, idx in rawte.groupby("vessel_id").groups.items():
                ii = np.asarray(list(idx), dtype=int)
                by_vessel.append({
                    "branch": branch,
                    "configuration": label,
                    "vessel_id": vessel,
                    "ship_type": rawte.loc[ii[0], "ship_type"],
                    **metric_dict(yte[ii], p[ii]),
                })

    pd.DataFrame(rows).to_csv(d / "Table_A1_ablation_fleet_by_branch.csv", index=False)
    by_ship_df = pd.DataFrame(by_ship)
    by_ship_df.to_csv(d / "Table_A2_ablation_by_shiptype_branch.csv", index=False)
    pd.DataFrame(by_vessel).to_csv(d / "Table_A3_ablation_by_vessel_branch.csv", index=False)

    # Direct original-vs-harmonized paired control on bulk test rows.
    bulk = rawte["ship_type"].eq("bulk").to_numpy()
    cmp_rows = []
    for j, label in enumerate(configs):
        po = preds[("original", label)]
        ph = preds[("harmonized", label)]
        mo = metric_dict(yte[bulk], po[bulk])
        mh = metric_dict(yte[bulk], ph[bulk])
        dm, lo, hi = cluster_bootstrap_metric_diff(
            rawte.loc[bulk, "vessel_id"].to_numpy(str), yte[bulk], ph[bulk], po[bulk],
            args.bootstrap, args.seed + 4500 + j,
        )
        cmp_rows.append({
            "configuration": label,
            "bulk_test_n": int(bulk.sum()),
            "original_RMSE": mo["RMSE"],
            "harmonized_RMSE": mh["RMSE"],
            "delta_RMSE_harmonized_minus_original": mh["RMSE"] - mo["RMSE"],
            "delta_R2_harmonized_minus_original": mh["R2"] - mo["R2"],
            "cluster_bootstrap_delta_RMSE_mean": dm,
            "cluster_bootstrap_CI95_low": lo,
            "cluster_bootstrap_CI95_high": hi,
        })
    pd.DataFrame(cmp_rows).to_csv(d / "Table_A4_bulk_original_vs_harmonized.csv", index=False)

    neg = {
        "operational_train_exact": bool(op_train_same),
        "operational_test_exact": bool(op_test_same),
        "shared_operational_fit": True,
        "prediction_max_abs_difference": 0.0,
        "interpretation": "operational_core contains no harmonized environmental variable; identical inputs are a hard negative control",
    }
    save_json(neg, d / "operational_core_negative_control.json")
    print("[PASS] ablation finished; operational negative control exact")
    return models, preds


def run_lovo(raw_o, X_o, raw_h, X_h, params, args, out):
    d = out / "03_bulk_lovo"
    chk = d / "checkpoints"
    chk.mkdir(parents=True, exist_ok=True)
    bulk_ids = sorted(raw_o.loc[raw_o["ship_type"].eq("bulk"), "vessel_id"].unique())
    if set(bulk_ids) != OFFICIAL_BULK_IDS_EXPECTED:
        raise AssertionError("Bulk LOVO vessel set changed")

    rows = []
    for vi, vessel in enumerate(bulk_ids):
        te = raw_o["vessel_id"].eq(vessel).to_numpy()
        tr = ~te
        for branch, raw, X in [("original", raw_o, X_o), ("harmonized", raw_h, X_h)]:
            result_path = chk / ("%s__%s.json" % (branch, vessel))
            pred_path = chk / ("%s__%s_pred.npy" % (branch, vessel))
            if args.resume and result_path.exists() and pred_path.exists():
                with open(str(result_path), "r", encoding="utf-8") as f:
                    r = json.load(f)
                rows.append(r)
                print("[resume] LOVO", branch, vessel)
                continue
            print("[LOVO]", vi + 1, "/", len(bulk_ids), branch, vessel, "train=", int(tr.sum()), "test=", int(te.sum()))
            model = xgb_factory(params, args.seed, args.n_jobs)
            model.fit(X.loc[tr, PRIMARY_FEATURES], raw.loc[tr, "target"].to_numpy(float))
            p = np.asarray(model.predict(X.loc[te, PRIMARY_FEATURES]), dtype=float)
            m = metric_dict(raw.loc[te, "target"].to_numpy(float), p)
            r = {
                "branch": branch,
                "vessel_id": vessel,
                "ship_type": "bulk",
                "train_rows": int(tr.sum()),
                "test_rows": int(te.sum()),
                **m,
            }
            np.save(str(pred_path), p)
            save_json(r, result_path)
            rows.append(r)
            del model, p
            gc.collect()

    tab = pd.DataFrame(rows).sort_values(["vessel_id", "branch"]).reset_index(drop=True)
    tab.to_csv(d / "Table_L1_bulk_LOVO_by_vessel_branch.csv", index=False)

    wide = tab.pivot(index="vessel_id", columns="branch", values=["RMSE", "R2", "NRMSE_sd", "target_sd"])
    cmp_rows = []
    for vessel in bulk_ids:
        cmp_rows.append({
            "vessel_id": vessel,
            "original_RMSE": float(wide.loc[vessel, ("RMSE", "original")]),
            "harmonized_RMSE": float(wide.loc[vessel, ("RMSE", "harmonized")]),
            "delta_RMSE_harmonized_minus_original": float(wide.loc[vessel, ("RMSE", "harmonized")] - wide.loc[vessel, ("RMSE", "original")]),
            "original_R2": float(wide.loc[vessel, ("R2", "original")]),
            "harmonized_R2": float(wide.loc[vessel, ("R2", "harmonized")]),
            "delta_R2_harmonized_minus_original": float(wide.loc[vessel, ("R2", "harmonized")] - wide.loc[vessel, ("R2", "original")]),
            "original_RMSE_over_target_SD": float(wide.loc[vessel, ("NRMSE_sd", "original")]),
            "harmonized_RMSE_over_target_SD": float(wide.loc[vessel, ("NRMSE_sd", "harmonized")]),
            "target_SD": float(wide.loc[vessel, ("target_sd", "original")]),
        })
    cmp = pd.DataFrame(cmp_rows)
    cmp.to_csv(d / "Table_L2_bulk_LOVO_original_vs_harmonized.csv", index=False)

    summaries = []
    for branch in ["original", "harmonized"]:
        g = tab[tab["branch"] == branch]
        summaries.append({
            "branch": branch,
            "vessels": int(g["vessel_id"].nunique()),
            "positive_R2": int((g["R2"] > 0).sum()),
            "median_R2": float(g["R2"].median()),
            "median_RMSE": float(g["RMSE"].median()),
            "median_target_SD": float(g["target_sd"].median()),
            "median_RMSE_over_target_SD": float(g["NRMSE_sd"].median()),
        })
    pd.DataFrame(summaries).to_csv(d / "Table_L3_bulk_LOVO_summary.csv", index=False)
    print("[PASS] six-fold bulk LOVO finished")


def ensure_full_models(raw_o, X_o, raw_h, X_h, tr, te, params, args, out):
    art = out / "99_artifacts"
    models = {}
    preds = {}
    ytr = raw_o.iloc[tr]["target"].to_numpy(float)
    for branch, X in [("original", X_o), ("harmonized", X_h)]:
        model_path = art / ("xgb_%s_dynamic_physical.joblib" % branch)
        pred_path = art / ("pred_%s_dynamic_physical.npy" % branch)
        model, pred = fit_model_cached(
            X.iloc[tr][PRIMARY_FEATURES], ytr, X.iloc[te][PRIMARY_FEATURES],
            params, args, model_path, pred_path, "%s dynamic_physical" % branch,
        )
        models[branch] = model
        preds[branch] = pred
    return models, preds


def run_shap(raw_o, X_o, raw_h, X_h, tr, te, params, args, out):
    try:
        import shap
    except Exception as exc:
        raise RuntimeError("SHAP step requires shap; import failed: %r" % exc)

    d = out / "04_harmonized_shap"
    models, _ = ensure_full_models(raw_o, X_o, raw_h, X_h, tr, te, params, args, out)
    local_idx, global_idx = deterministic_shap_sample(te, min(args.shap_eval_n, len(te)), args.seed + 800)
    rng = np.random.default_rng(args.seed + 801)
    bg_n = min(args.shap_background_n, len(tr))
    bg_global = np.sort(rng.choice(tr, size=bg_n, replace=False))
    np.save(str(d / "official_test_sample_global_indices.npy"), global_idx)
    np.save(str(d / "training_background_global_indices.npy"), bg_global)

    all_direction = []
    for branch, raw, X in [("original", raw_o, X_o), ("harmonized", raw_h, X_h)]:
        sv_path = d / ("%s_interventional_SHAP_values.npy" % branch)
        if args.resume and sv_path.exists():
            sv = np.load(str(sv_path))
            print("[resume] SHAP", branch)
        else:
            Xsh = X.iloc[global_idx][PRIMARY_FEATURES].reset_index(drop=True)
            background = X.iloc[bg_global][PRIMARY_FEATURES].reset_index(drop=True)
            print("[SHAP]", branch, "eval=", len(Xsh), "background=", len(background))
            explainer = shap.TreeExplainer(
                models[branch],
                data=background,
                feature_perturbation="interventional",
                model_output="raw",
            )
            sv = np.empty((len(Xsh), len(PRIMARY_FEATURES)), dtype=np.float64)
            for start in range(0, len(Xsh), args.shap_batch_size):
                stop = min(len(Xsh), start + args.shap_batch_size)
                vals = explainer.shap_values(Xsh.iloc[start:stop], check_additivity=False)
                if isinstance(vals, list):
                    vals = vals[0]
                sv[start:stop] = np.asarray(vals, dtype=float)
                print("  SHAP", branch, start, ":", stop)
            np.save(str(sv_path), sv)

        Xsh = X.iloc[global_idx][PRIMARY_FEATURES].reset_index(drop=True)
        rawsh = raw.iloc[global_idx].reset_index(drop=True)
        Xsh.to_csv(d / ("%s_interventional_SHAP_sample_features.csv" % branch), index=False)

        if sv.shape != (len(global_idx), len(PRIMARY_FEATURES)):
            raise AssertionError("Unexpected SHAP shape for %s: %s" % (branch, sv.shape))

        scopes = [("overall", np.ones(len(rawsh), dtype=bool))]
        for st in ["bulk", "container", "tanker"]:
            scopes.append((st, rawsh["ship_type"].eq(st).to_numpy()))
        for scope, mask in scopes:
            for feat in ["speed_kn", "wave_height_m", "rel_wind_speed_kn"]:
                j = PRIMARY_FEATURES.index(feat)
                x = Xsh.loc[mask, feat].to_numpy(float)
                y = sv[mask, j].astype(float)
                good = np.isfinite(x) & np.isfinite(y)
                rho, pval = spearmanr(x[good], y[good]) if good.sum() >= 3 else (np.nan, np.nan)
                all_direction.append({
                    "branch": branch,
                    "scope": scope,
                    "feature": feat,
                    "n": int(good.sum()),
                    "feature_SHAP_Spearman_rho": float(rho) if np.isfinite(rho) else np.nan,
                    "p_value_iid_descriptive_only": float(pval) if np.isfinite(pval) else np.nan,
                })

        # Additivity diagnostic on 200 deterministic positions.
        ncheck = min(200, len(Xsh))
        ci = np.linspace(0, len(Xsh) - 1, ncheck).round().astype(int)
        pred = np.asarray(models[branch].predict(Xsh.iloc[ci]), dtype=float)
        # Recreate explainer only if resumed, to recover expected_value.
        if args.resume and sv_path.exists():
            background = X.iloc[bg_global][PRIMARY_FEATURES].reset_index(drop=True)
            explainer = shap.TreeExplainer(models[branch], data=background, feature_perturbation="interventional", model_output="raw")
        ev = np.asarray(explainer.expected_value).reshape(-1)
        evs = float(ev[0]) if ev.size else np.nan
        recon = evs + sv[ci].sum(axis=1)
        pd.DataFrame([{
            "branch": branch,
            "n_checked": ncheck,
            "mean_abs_additivity_gap": float(np.mean(np.abs(pred - recon))),
            "max_abs_additivity_gap": float(np.max(np.abs(pred - recon))),
            "background_n": int(bg_n),
        }]).to_csv(d / ("%s_SHAP_additivity.csv" % branch), index=False)

    tab = pd.DataFrame(all_direction)
    tab.to_csv(d / "Table_SHAP1_direction_by_branch_scope.csv", index=False)
    bulk = tab[tab["scope"] == "bulk"].pivot(index="feature", columns="branch", values="feature_SHAP_Spearman_rho").reset_index()
    bulk["delta_rho_harmonized_minus_original"] = bulk["harmonized"] - bulk["original"]
    bulk.to_csv(d / "Table_SHAP2_bulk_original_vs_harmonized_direction.csv", index=False)
    print("[PASS] harmonized interventional SHAP finished")


def baseline_distance(raw_test):
    if "distance_nm" not in raw_test.columns or raw_test["distance_nm"].isna().all():
        return raw_test["speed_kn"].astype(float) * (10.0 / 60.0)
    return pd.to_numeric(raw_test["distance_nm"], errors="coerce")


def support_mask(raw_train, raw_test, reduction):
    sc_speed = raw_test["speed_kn"].to_numpy(float) * (1.0 - float(reduction))
    support = raw_train.groupby("ship_type")["speed_kn"].agg(["min", "max"])
    valid = np.ones(len(raw_test), dtype=bool)
    sts = raw_test["ship_type"].astype(str).to_numpy()
    for st in np.unique(sts):
        m = sts == st
        if st not in support.index:
            valid[m] = False
            continue
        lo = float(support.loc[st, "min"])
        hi = float(support.loc[st, "max"])
        valid[m] = np.isfinite(sc_speed[m]) & (sc_speed[m] >= lo) & (sc_speed[m] <= hi)
    return valid


def ft_weighted_cii_change(g):
    b_f = g["baseline_predicted_fuel_t"].sum()
    b_tw = g["baseline_transport_work"].sum()
    f_f = g["FT_predicted_fuel_t"].sum()
    f_tw = g["FT_transport_work"].sum()
    return ((f_f / f_tw) / (b_f / b_tw) - 1.0) * 100.0


def bootstrap_vessel_aggregate(g, stat_fn, reps, seed):
    vessels = sorted(g["vessel_id"].astype(str).unique())
    by = {v: g[g["vessel_id"].astype(str) == v].copy() for v in vessels}
    point = float(stat_fn(g))
    rng = np.random.default_rng(seed)
    vals = np.empty(reps, dtype=float)
    for i in range(reps):
        draw = rng.choice(vessels, size=len(vessels), replace=True)
        zz = pd.concat([by[v] for v in draw], ignore_index=True)
        vals[i] = stat_fn(zz)
    return point, float(np.quantile(vals, .025)), float(np.quantile(vals, .975)), vals


def paired_bootstrap_branch_delta(go, gh, stat_fn, reps, seed):
    vo = sorted(go["vessel_id"].astype(str).unique())
    vh = sorted(gh["vessel_id"].astype(str).unique())
    if vo != vh:
        raise AssertionError("Paired branch bootstrap vessel sets differ")
    mo = {v: go[go["vessel_id"].astype(str) == v].copy() for v in vo}
    mh = {v: gh[gh["vessel_id"].astype(str) == v].copy() for v in vo}
    point = float(stat_fn(gh) - stat_fn(go))
    rng = np.random.default_rng(seed)
    vals = np.empty(reps, dtype=float)
    for i in range(reps):
        draw = rng.choice(vo, size=len(vo), replace=True)
        zo = pd.concat([mo[v] for v in draw], ignore_index=True)
        zh = pd.concat([mh[v] for v in draw], ignore_index=True)
        vals[i] = stat_fn(zh) - stat_fn(zo)
    return point, float(np.quantile(vals, .025)), float(np.quantile(vals, .975))


def calculate_ft_branch(branch, model, baseline_pred, raw_train, raw_test, X_test, reductions, cf, args):
    base = raw_test.copy().reset_index(drop=True)
    base["_baseline_pred"] = np.asarray(baseline_pred, dtype=float)
    base["_baseline_distance_nm"] = baseline_distance(base).to_numpy(float)
    base["_cf_used"] = float(cf)
    rows = []
    support_rows = []

    for reduction in reductions:
        rpct = float(reduction) * 100.0
        ratio_speed = 1.0 - float(reduction)
        valid = support_mask(raw_train, base, reduction)
        n_supported = int(valid.sum())
        expected = EXPECTED_SUPPORTED.get(rpct)
        if expected is not None and n_supported != expected:
            raise AssertionError("FT support count changed at %.0f%%: %d vs expected %d" % (rpct, n_supported, expected))
        b = base.loc[valid].copy()
        ft = b.copy()
        ft["speed_kn"] = ft["speed_kn"].astype(float) * ratio_speed
        # Full model uses canonical 17 features; only speed changes in FT.
        ft_pred = np.asarray(model.predict(ft[PRIMARY_FEATURES]), dtype=float)
        ft["_scenario_pred"] = ft_pred
        ratio = np.divide(
            ft["speed_kn"].to_numpy(float), b["speed_kn"].to_numpy(float),
            out=np.zeros(len(ft), dtype=float), where=b["speed_kn"].to_numpy(float) != 0.0,
        )
        ft["_scenario_distance_nm"] = b["_baseline_distance_nm"].to_numpy(float) * ratio

        for vessel, bg in b.groupby("vessel_id"):
            idx = bg.index
            fg = ft.loc[idx]
            dwt = float(pd.to_numeric(bg["dwt"], errors="coerce").dropna().median())
            base_fuel = float(bg["_baseline_pred"].sum())
            base_dist = float(bg["_baseline_distance_nm"].sum())
            base_co2 = float(np.sum(bg["_baseline_pred"].to_numpy(float) * float(cf) * 1e6))
            base_tw = dwt * base_dist
            base_cii = base_co2 / base_tw if base_tw > 0 else np.nan

            ft_fuel = float(fg["_scenario_pred"].sum())
            ft_dist = float(fg["_scenario_distance_nm"].sum())
            ft_co2 = float(np.sum(fg["_scenario_pred"].to_numpy(float) * float(cf) * 1e6))
            ft_tw = dwt * ft_dist
            ft_cii = ft_co2 / ft_tw if ft_tw > 0 else np.nan
            ft_ratio = ft_cii / base_cii if np.isfinite(base_cii) and base_cii > 0 else np.nan

            rows.append({
                "branch": branch,
                "reduction_pct": rpct,
                "vessel_id": vessel,
                "ship_type": str(bg["ship_type"].iloc[0]),
                "n_supported": int(len(bg)),
                "retained_pct": float(len(bg) / max((base["vessel_id"] == vessel).sum(), 1) * 100.0),
                "DWT": dwt,
                "baseline_predicted_fuel_t": base_fuel,
                "baseline_distance_nm": base_dist,
                "baseline_transport_work": base_tw,
                "baseline_CII_proxy": base_cii,
                "FT_predicted_fuel_t": ft_fuel,
                "FT_fuel_change_pct": (ft_fuel / base_fuel - 1.0) * 100.0 if base_fuel > 0 else np.nan,
                "FT_distance_nm": ft_dist,
                "FT_distance_change_pct": (ft_dist / base_dist - 1.0) * 100.0 if base_dist > 0 else np.nan,
                "FT_transport_work": ft_tw,
                "FT_CII_proxy": ft_cii,
                "FT_CII_ratio": ft_ratio,
                "FT_CII_change_pct": (ft_ratio - 1.0) * 100.0 if np.isfinite(ft_ratio) else np.nan,
            })
        support_rows.append({"branch": branch, "reduction_pct": rpct, "L1_test_rows": len(base), "supported_rows": n_supported, "expected_supported_rows": expected})
    return pd.DataFrame(rows), pd.DataFrame(support_rows)


def run_cii(raw_o, X_o, raw_h, X_h, tr, te, params, args, out):
    d = out / "05_ft_cii"
    models, preds = ensure_full_models(raw_o, X_o, raw_h, X_h, tr, te, params, args, out)
    reductions = [float(x.strip()) for x in args.reductions.split(",") if x.strip()]
    all_v = []
    all_s = []
    for branch, raw, X in [("original", raw_o, X_o), ("harmonized", raw_h, X_h)]:
        v, s = calculate_ft_branch(
            branch, models[branch], preds[branch],
            raw.iloc[tr].reset_index(drop=True),
            raw.iloc[te].reset_index(drop=True),
            X.iloc[te].reset_index(drop=True),
            reductions, args.co2_factor, args,
        )
        all_v.append(v)
        all_s.append(s)
    vessels = pd.concat(all_v, ignore_index=True)
    supports = pd.concat(all_s, ignore_index=True)
    vessels.to_csv(d / "Table_C1_FT_by_vessel_branch.csv", index=False)
    supports.to_csv(d / "Table_C2_FT_support_audit.csv", index=False)

    endpoint_rows = []
    for branch in ["original", "harmonized"]:
        b = vessels[vessels["branch"] == branch]
        for ri, r in enumerate(sorted(b["reduction_pct"].unique())):
            z = b[b["reduction_pct"] == r]
            for ti, scope in enumerate(["Fleet", "bulk", "container", "tanker"]):
                g = z if scope == "Fleet" else z[z["ship_type"] == scope]
                point, lo, hi, boots = bootstrap_vessel_aggregate(
                    g, ft_weighted_cii_change, args.bootstrap,
                    args.seed + 2000 + 100 * ri + ti + (10000 if branch == "harmonized" else 0),
                )
                endpoint_rows.append({
                    "branch": branch,
                    "reduction_pct": r,
                    "scope": scope,
                    "n_vessels": int(g["vessel_id"].nunique()),
                    "FT_weighted_CII_change_pct": point,
                    "vessel_bootstrap_CI95_low": lo,
                    "vessel_bootstrap_CI95_high": hi,
                    "bootstrap_prob_above_zero": float(np.mean(boots > 0)),
                    "bootstrap_prob_below_zero": float(np.mean(boots < 0)),
                })
    pd.DataFrame(endpoint_rows).to_csv(d / "Table_C3_FT_vessel_bootstrap_endpoint_CI_by_branch.csv", index=False)

    pair = []
    for ri, r in enumerate(sorted(vessels["reduction_pct"].unique())):
        go = vessels[(vessels["branch"] == "original") & (vessels["ship_type"] == "bulk") & (vessels["reduction_pct"] == r)]
        gh = vessels[(vessels["branch"] == "harmonized") & (vessels["ship_type"] == "bulk") & (vessels["reduction_pct"] == r)]
        point, lo, hi = paired_bootstrap_branch_delta(go, gh, ft_weighted_cii_change, args.bootstrap, args.seed + 15000 + ri)
        pair.append({
            "reduction_pct": r,
            "scope": "bulk",
            "delta_FT_weighted_CII_change_pp_harmonized_minus_original": point,
            "paired_vessel_bootstrap_CI95_low": lo,
            "paired_vessel_bootstrap_CI95_high": hi,
            "CI_excludes_zero": bool(lo > 0 or hi < 0),
        })
    pd.DataFrame(pair).to_csv(d / "Table_C4_bulk_original_vs_harmonized_FT_CII.csv", index=False)
    print("[PASS] L1-aligned FT CII harmonization control finished")


def run_wave_c4_placeholder(out):
    d = out / "06_wave_c4"
    text = (
        "SKIPPED: the current harmonized input covers bulk carriers only.\n"
        "The tanker x head-sea C4 harmonization experiment requires a tanker harmonized ERA5 master built with the same bilinear + temporal-linear protocol.\n"
        "Do not use the bulk harmonized branch to answer the tanker C4 question.\n"
    )
    (d / "SKIPPED_NEEDS_TANKER_HARMONIZATION.txt").write_text(text, encoding="utf-8")
    print("[SKIP] wave C4 requires tanker harmonization")


def run_summary(out):
    d = out / "07_summary"
    rows = []
    p = out / "02_ablation" / "Table_A4_bulk_original_vs_harmonized.csv"
    if p.exists():
        z = pd.read_csv(p)
        for _, r in z.iterrows():
            rows.append({"experiment": "ablation", "endpoint": str(r["configuration"]) + " bulk RMSE delta", "original": r["original_RMSE"], "harmonized": r["harmonized_RMSE"], "delta": r["delta_RMSE_harmonized_minus_original"]})
    p = out / "03_bulk_lovo" / "Table_L3_bulk_LOVO_summary.csv"
    if p.exists():
        z = pd.read_csv(p).set_index("branch")
        if {"original", "harmonized"}.issubset(z.index):
            for endpoint in ["positive_R2", "median_R2", "median_RMSE", "median_RMSE_over_target_SD"]:
                rows.append({"experiment": "bulk_lovo", "endpoint": endpoint, "original": z.loc["original", endpoint], "harmonized": z.loc["harmonized", endpoint], "delta": z.loc["harmonized", endpoint] - z.loc["original", endpoint]})
    p = out / "04_harmonized_shap" / "Table_SHAP2_bulk_original_vs_harmonized_direction.csv"
    if p.exists():
        z = pd.read_csv(p)
        for _, r in z.iterrows():
            rows.append({"experiment": "shap", "endpoint": str(r["feature"]) + " bulk feature-SHAP rho", "original": r["original"], "harmonized": r["harmonized"], "delta": r["delta_rho_harmonized_minus_original"]})
    p = out / "05_ft_cii" / "Table_C4_bulk_original_vs_harmonized_FT_CII.csv"
    if p.exists():
        z = pd.read_csv(p)
        # Retrieve point endpoints from C3.
        c3 = pd.read_csv(out / "05_ft_cii" / "Table_C3_FT_vessel_bootstrap_endpoint_CI_by_branch.csv")
        for _, r in z.iterrows():
            red = r["reduction_pct"]
            oo = c3[(c3["branch"] == "original") & (c3["scope"] == "bulk") & (c3["reduction_pct"] == red)].iloc[0]
            hh = c3[(c3["branch"] == "harmonized") & (c3["scope"] == "bulk") & (c3["reduction_pct"] == red)].iloc[0]
            rows.append({"experiment": "ft_cii", "endpoint": "bulk FT weighted CII change %.0f%% speed reduction" % red, "original": oo["FT_weighted_CII_change_pct"], "harmonized": hh["FT_weighted_CII_change_pct"], "delta": r["delta_FT_weighted_CII_change_pp_harmonized_minus_original"]})
    pd.DataFrame(rows).to_csv(d / "harmonization_control_summary.csv", index=False)
    print("[PASS] summary written")


def main():
    t0 = time.time()
    args = parse_args()
    paths = resolve_paths(args)
    ensure_dirs(paths["out"])
    steps = [x.strip().lower() for x in args.steps.split(",") if x.strip()]
    if "all" in steps:
        steps = ["alignment", "ablation", "lovo", "shap", "cii", "wave_c4", "summary"]

    print("=" * 80)
    print("F31 BULK HARMONIZATION CONTROL")
    print("version:", VERSION)
    print("output :", paths["out"])
    print("steps  :", steps)
    print("=" * 80)

    print("[load] Fixed31")
    fixed = pd.read_csv(paths["fixed31"], low_memory=False)
    raw_o, X_o = build_canonical_original(fixed)
    if len(raw_o) != EXPECTED_ROWS:
        raise AssertionError("Expected %d Fixed31 rows, got %d" % (EXPECTED_ROWS, len(raw_o)))
    del fixed
    gc.collect()

    tr, te, train_key, test_key = load_split(paths["split"])
    params = load_xgb_params(paths["hyperparams"])

    print("[overlay] harmonized bulk ERA5")
    raw_h, X_h, hm, overlay_audit, interaction_audit = overlay_harmonized_bulk(
        raw_o, X_o, paths["harmonized_bulk"], paths["out"] / "01_alignment"
    )

    manifest = validate_design(raw_o, X_o, raw_h, X_h, tr, te, paths, params, args)
    manifest["split_keys"] = {"train": train_key, "test": test_key}
    manifest["overlay_audit"] = overlay_audit
    manifest["interaction_audit"] = interaction_audit
    save_json(manifest, paths["out"] / "01_alignment" / "harmonization_control_manifest.json")

    # Additional alignment table.
    bulk = raw_o["ship_type"].eq("bulk")
    split_label = np.full(len(raw_o), "", dtype=object)
    split_label[tr] = "train"
    split_label[te] = "test"
    miss = []
    for c in ["rel_wind_speed_kn", "wave_height_m", "wave_period_s", "sst_c", "mslp_hpa"]:
        for sp in ["all", "train", "test"]:
            m = bulk.to_numpy() if sp == "all" else (bulk.to_numpy() & (split_label == sp))
            miss.append({"feature": c, "split": sp, "rows": int(m.sum()), "missing": int(X_h.loc[m, c].isna().sum()), "missing_pct": float(X_h.loc[m, c].isna().mean() * 100.0) if m.sum() else np.nan})
    pd.DataFrame(miss).to_csv(paths["out"] / "01_alignment" / "bulk_missingness_by_split.csv", index=False)
    print("[PASS] alignment | rows=%d | bulk=%d | matched=%d" % (len(raw_o), int(bulk.sum()), len(hm)))

    if "ablation" in steps:
        run_ablation(raw_o, X_o, raw_h, X_h, tr, te, params, args, paths["out"])
    if "lovo" in steps:
        run_lovo(raw_o, X_o, raw_h, X_h, params, args, paths["out"])
    if "shap" in steps:
        run_shap(raw_o, X_o, raw_h, X_h, tr, te, params, args, paths["out"])
    if "cii" in steps:
        run_cii(raw_o, X_o, raw_h, X_h, tr, te, params, args, paths["out"])
    if "wave_c4" in steps:
        run_wave_c4_placeholder(paths["out"])
    if "summary" in steps:
        run_summary(paths["out"])

    print("=" * 80)
    print("DONE in %.1f min" % ((time.time() - t0) / 60.0))
    print("Output:", paths["out"])
    print("=" * 80)


if __name__ == "__main__":
    main()
