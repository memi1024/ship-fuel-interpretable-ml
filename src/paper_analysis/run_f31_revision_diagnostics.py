#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Physical, residual, ablation, and temporal diagnostics for the Fixed31 analysis.

The script uses the established record split and fixed XGBoost hyperparameters. It
computes known-vessel temporal validation, residual dependence diagnostics, distribution
shift, the cubic benchmark, feature ablation, interventional TreeSHAP with GAM/LOWESS
response-shape diagnostics, and rudder perturbation analysis.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import logging
import math
import os
import platform
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd

from scipy.stats import chi2, ks_2samp, spearmanr, wasserstein_distance
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import statsmodels.api as sm
from statsmodels.gam.api import BSplines, GLMGam
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.power import TTestPower
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import acf

try:
    import shap
except Exception as exc:
    shap = None
    _SHAP_IMPORT_ERROR = exc
else:
    _SHAP_IMPORT_ERROR = None


VERSION = "2026-08-17.revision-diagnostics-v1"
DEFAULT_STEPS = [
    "temporal", "residuals", "shift", "cubic", "ablation", "shap", "rudder"
]
SHIFT_FEATURES = ["speed_kn", "draught_m", "wave_height_m"]
PRIMARY_FEATURES = [
    "speed_kn", "heading_sin", "heading_cos", "draught_m", "trim_m",
    "rudder_deg", "rel_wind_speed_kn", "rel_wind_sin", "rel_wind_cos",
    "wave_height_m", "rel_wave_sin", "rel_wave_cos", "wave_period_s",
    "sst_c", "mslp_hpa", "ship_type_bulk", "ship_type_container",
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Physical, residual, ablation, and temporal diagnostics for the Fixed31 analysis."
    )
    p.add_argument(
        "--data-dir",
        default=r"data\05_clean23\all_phase_equal_cleaning\05_combined",
        help="Directory containing final_fixed31_cruise.csv."
    )
    p.add_argument(
        "--input-csv",
        default=None,
        help="Optional explicit cruise CSV; overrides --data-dir."
    )
    p.add_argument(
        "--main-output",
        default=r"data\11_f31_complete_empirics",
        help="Existing complete-empirics output root."
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Revision output root. Default: <main-output>/15_revision_diagnostics."
    )
    p.add_argument("--core-path", default=None)
    p.add_argument("--column-overrides", default="column_overrides_fixed31.json")
    p.add_argument("--hyperparams-csv", default="02_best_hyperparameters.csv")
    p.add_argument("--record-split", default=None)
    p.add_argument("--record-model", default=None)
    p.add_argument("--record-prediction", default=None)
    p.add_argument("--lovo-prediction", default=None)
    p.add_argument(
        "--steps",
        default="all",
        help="Comma list from temporal,residuals,shift,cubic,ablation,shap,rudder or all."
    )
    p.add_argument("--resume", action="store_true")
    p.add_argument("--n-jobs", type=int, default=2)
    p.add_argument("--csv-chunksize", type=int, default=25000)
    p.add_argument("--seed", type=int, default=20260808)
    p.add_argument("--test-size", type=float, default=0.20)
    p.add_argument("--trajectory-gap-minutes", type=float, default=30.0)
    p.add_argument("--expected-cruise-n", type=int, default=489620)
    p.add_argument("--expected-vessels", type=int, default=21)

    # Residual diagnostics
    p.add_argument("--acf-max-lag", type=int, default=12)

    # Cluster bootstrap
    p.add_argument("--bootstrap-reps", type=int, default=2000)

    # SHAP
    p.add_argument("--shap-eval-n", type=int, default=12000)
    p.add_argument("--shap-background-n", type=int, default=256)
    p.add_argument("--shap-batch-size", type=int, default=500)

    # GAM / LOWESS
    p.add_argument("--gam-basis-df", type=int, default=10)
    p.add_argument("--gam-alpha-grid", default="-4,-3,-2,-1,0,1,2,3,4")
    p.add_argument("--gam-grid-n", type=int, default=200)
    p.add_argument("--support-q-low", type=float, default=0.02)
    p.add_argument("--support-q-high", type=float, default=0.98)
    p.add_argument("--loess-frac", type=float, default=0.25)
    p.add_argument("--loess-bootstrap-reps", type=int, default=200)

    # Rudder audit / power
    p.add_argument("--rudder-sample-n", type=int, default=20000)
    p.add_argument("--rudder-near-zero-deg", type=float, default=1.0)
    p.add_argument("--rudder-sensitivity-thresholds", default="0.5,1.0,2.0")
    return p.parse_args()


def parse_steps(s: str) -> List[str]:
    s = str(s).strip().lower()
    if s in {"all", "*"}:
        return list(DEFAULT_STEPS)
    vals = [v.strip() for v in s.split(",") if v.strip()]
    unknown = [v for v in vals if v not in DEFAULT_STEPS]
    if unknown:
        raise ValueError(f"Unknown steps={unknown}; valid={DEFAULT_STEPS}")
    return vals


def setup_logger(out: Path) -> logging.Logger:
    out.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("f31_revision")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    fh = logging.FileHandler(out / "run_physical_validation.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger


def _candidate_path(base: Path, requested: Optional[str], names: Sequence[str]) -> Optional[Path]:
    if requested:
        p = Path(requested)
        if p.exists():
            return p.resolve()
        q = base / requested
        if q.exists():
            return q.resolve()
    for name in names:
        p = base / name
        if p.exists():
            return p.resolve()
    return None


def resolve_script_file(requested: Optional[str], filename: str) -> Path:
    p = Path(requested) if requested else Path(__file__).resolve().parent / filename
    if not p.is_absolute():
        p = Path(__file__).resolve().parent / p
    if not p.is_file():
        raise FileNotFoundError(f"Required file not found: {p}")
    return p.resolve()


def load_core(core_path: Optional[str]):
    p = resolve_script_file(core_path, "f31_core_memory_safe.py")
    module_name = "f31_core_memory_safe_revision"
    spec = importlib.util.spec_from_file_location(module_name, p)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import core from {p}")
    module = importlib.util.module_from_spec(spec)

    # Python 3.13/3.14 dataclasses may inspect sys.modules while class
    # decorators are executing. Dynamic modules therefore must be registered
    # before exec_module(); otherwise @dataclass can fail with
    # AttributeError: 'NoneType' object has no attribute '__dict__'.
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        # Avoid leaving a half-initialized module behind after a failed import.
        sys.modules.pop(module_name, None)
        raise
    return module, p


def sha256_short(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def atomic_csv(df: pd.DataFrame, path: Path, **kwargs) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False, **kwargs)
    tmp.replace(path)


def metric_dict(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    err = p - y
    rmse = math.sqrt(mean_squared_error(y, p))
    mae = mean_absolute_error(y, p)
    return {
        "n": int(len(y)),
        "RMSE": float(rmse),
        "MAE": float(mae),
        "R2": float(r2_score(y, p)),
        "mean_residual_pred_minus_obs": float(np.mean(err)),
        "median_residual_pred_minus_obs": float(np.median(err)),
    }


def interaction_feature_matrix(X: pd.DataFrame) -> pd.DataFrame:
    z = X.copy()
    z["speed_x_wind"] = z["speed_kn"] * z["rel_wind_speed_kn"]
    z["speed_x_wave"] = z["speed_kn"] * z["wave_height_m"]
    z["speed_x_draught"] = z["speed_kn"] * z["draught_m"]
    z["speed_x_trim"] = z["speed_kn"] * z["trim_m"]
    z["draught_x_trim"] = z["draught_m"] * z["trim_m"]
    return z


def load_locked_xgb_params(hp_path: Path) -> Dict:
    tab = pd.read_csv(hp_path)
    name = tab["model"].astype(str).str.upper()
    row = tab.loc[name == "XGB"]
    if row.empty:
        raise ValueError(f"No XGB row in {hp_path}")
    return json.loads(str(row.iloc[0]["best_parameters_json"]))


def load_cruise(core, args, overrides_path: Path, logger) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if args.input_csv:
        cruise_path = Path(args.input_csv)
    else:
        cruise_path = Path(args.data_dir) / "final_fixed31_cruise.csv"
    if not cruise_path.exists():
        raise FileNotFoundError(f"Cruise CSV not found: {cruise_path}")

    ns = SimpleNamespace(
        column_overrides=str(overrides_path),
        trajectory_gap_minutes=float(args.trajectory_gap_minutes),
        csv_chunksize=int(args.csv_chunksize),
    )
    logger.info("Loading cruise cohort: %s", cruise_path)
    df = core.read_csv_memory_safe(cruise_path, ns, logger)
    bundle = core.canonicalize_dataframe(df, ns, logger)
    del df
    gc.collect()

    raw = bundle.raw.reset_index(drop=True)
    X = bundle.feature_df.reset_index(drop=True)

    if len(raw) != args.expected_cruise_n:
        raise AssertionError(f"Expected cruise n={args.expected_cruise_n}, got {len(raw)}")
    if raw["vessel_id"].nunique() != args.expected_vessels:
        raise AssertionError(
            f"Expected {args.expected_vessels} vessels, got {raw['vessel_id'].nunique()}"
        )
    if list(X.columns) != PRIMARY_FEATURES:
        raise AssertionError(
            "Canonical feature order differs from locked 17-feature specification.\n"
            f"Observed={list(X.columns)}"
        )
    raw = raw.copy()
    raw["row_id"] = np.arange(len(raw), dtype=np.int64)
    return raw, X


def resolve_artifacts(args, main: Path, here: Path) -> Dict[str, Path]:
    art = main / "14_artifacts"
    result = {}

    def choose(requested, candidates, required=True):
        search_bases = [here, art, main, main / "05_lovo", main / "04_temporal",
                        main / "16_tree_model_sensitivity"]
        if requested:
            p = Path(requested)
            if p.exists():
                return p.resolve()
            for b in search_bases:
                q = b / requested
                if q.exists():
                    return q.resolve()
        for c in candidates:
            for b in search_bases:
                q = b / c
                if q.exists():
                    return q.resolve()
        if required:
            raise FileNotFoundError(f"Missing required artifact; candidates={candidates}")
        return None

    result["split"] = choose(
        args.record_split, ["record_split_indices.npz"]
    )
    result["model"] = choose(
        args.record_model, ["xgb_record_split.joblib"]
    )
    result["record_pred"] = choose(
        args.record_prediction, ["xgb_record_test_prediction.npy"]
    )
    result["lovo_pred"] = choose(
        args.lovo_prediction,
        ["LOVO_predictions_xgb.npy", "LOVO_prediction_xgb.npy"],
    )
    result["temporal_pred"] = choose(
        None, ["temporal_prediction_xgb.npy", "temporal_predictions_xgb.npy"],
        required=False
    )
    return result


def validate_locked_assets(raw, X, artifacts, logger):
    z = np.load(artifacts["split"])
    tr = np.asarray(z["train_idx"], dtype=int)
    te = np.asarray(z["test_idx"], dtype=int)
    if len(tr) + len(te) != len(raw):
        raise AssertionError("Official record split size does not match cruise cohort.")
    if np.intersect1d(tr, te).size:
        raise AssertionError("Official record split overlaps.")
    if not np.array_equal(np.sort(np.concatenate([tr, te])), np.arange(len(raw))):
        raise AssertionError("Official split does not cover the cruise cohort exactly.")

    p = np.load(artifacts["record_pred"])
    if len(p) != len(te) or not np.isfinite(p).all():
        raise AssertionError("Cached L1 prediction does not match official test split.")

    lovo = np.load(artifacts["lovo_pred"])
    if lovo.shape != (len(raw),) or not np.isfinite(lovo).all():
        raise AssertionError("Cached LOVO prediction must be a complete finite cruise-length vector.")

    model = joblib.load(artifacts["model"])
    names = list(getattr(model, "feature_names_in_", []))
    if names and names != list(X.columns):
        raise AssertionError(f"Cached XGB feature order mismatch: {names}")

    logger.info(
        "Locked assets validated | L1 train=%d test=%d | LOVO=%d | model=%s",
        len(tr), len(te), len(lovo), type(model).__name__
    )
    return tr, te, p.astype(float), lovo.astype(float), model


def known_vessel_temporal_split(raw: pd.DataFrame, frac=0.8) -> Tuple[np.ndarray, np.ndarray]:
    train_idx, test_idx = [], []
    for _, inds in raw.groupby("vessel_id").groups.items():
        sub = raw.loc[inds].sort_values(["timestamp", "row_id"], kind="mergesort")
        cut = max(1, min(len(sub) - 1, int(math.floor(len(sub) * frac))))
        train_idx.extend(sub.index[:cut].tolist())
        test_idx.extend(sub.index[cut:].tolist())
    return np.asarray(sorted(train_idx), dtype=int), np.asarray(sorted(test_idx), dtype=int)


def get_or_run_temporal(
    raw, X, model_template, artifacts, out, args, logger
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = out / "02_temporal"
    d.mkdir(parents=True, exist_ok=True)
    tr, te = known_vessel_temporal_split(raw, 0.8)
    np.save(d / "temporal_train_indices.npy", tr)
    np.save(d / "temporal_test_indices.npy", te)

    local_pred = d / "temporal_prediction_xgb.npy"
    source_pred = artifacts.get("temporal_pred")

    if args.resume and local_pred.exists():
        pred = np.load(local_pred).astype(float)
        logger.info("Temporal prediction reuse: %s", local_pred)
    elif source_pred is not None and source_pred.exists():
        pred = np.load(source_pred).astype(float)
        logger.info("Temporal prediction imported from existing artifact: %s", source_pred)
        np.save(local_pred, pred)
    else:
        logger.info(
            "No temporal prediction cache found; fitting ONE locked XGB on first-80%% rows."
        )
        m = clone(model_template)
        try:
            m.set_params(n_jobs=args.n_jobs)
        except Exception:
            pass
        m.fit(X.iloc[tr], raw.iloc[tr]["target"].to_numpy(float))
        pred = np.asarray(m.predict(X.iloc[te]), dtype=float)
        np.save(local_pred, pred)
        joblib.dump(m, d / "xgb_temporal_first80.joblib")
        del m
        gc.collect()

    if pred.shape != (len(te),) or not np.isfinite(pred).all():
        raise AssertionError(
            f"Temporal prediction length/finite check failed: {pred.shape}, expected {(len(te),)}"
        )

    met = pd.DataFrame([{
        "model": "xgb",
        "validation": "known_vessel_first80_last20",
        **metric_dict(raw.iloc[te]["target"].to_numpy(float), pred),
    }])
    atomic_csv(met, d / "Table_T_revision_temporal_overall.csv")
    logger.info(
        "Temporal ready | train=%d test=%d RMSE=%.6f R2=%.6f",
        len(tr), len(te), met.iloc[0]["RMSE"], met.iloc[0]["R2"]
    )
    return tr, te, pred


def residual_row_frame(raw_subset: pd.DataFrame, pred: np.ndarray, validation: str) -> pd.DataFrame:
    z = raw_subset[
        ["row_id", "vessel_id", "ship_type", "timestamp", "target",
         "speed_kn", "draught_m", "wave_height_m",
         "ship_type_bulk", "ship_type_container"]
    ].copy()
    z["validation"] = validation
    z["y_pred"] = np.asarray(pred, dtype=float)
    z["residual"] = z["y_pred"] - z["target"].astype(float)
    return z


def per_vessel_serial_diagnostics(rows: pd.DataFrame, max_lag: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    summary = []
    acf_rows = []
    for vessel, g in rows.groupby("vessel_id", sort=True):
        g = g.sort_values(["timestamp", "row_id"], kind="mergesort")
        e = g["residual"].to_numpy(float)
        if len(e) < 3:
            continue
        dw = float(durbin_watson(e))
        ts = pd.to_datetime(g["timestamp"], errors="coerce", utc=True)
        dt_min = ts.diff().dt.total_seconds().div(60.0).dropna()
        summary.append({
            "validation": g["validation"].iloc[0],
            "vessel_id": vessel,
            "ship_type": g["ship_type"].iloc[0],
            "n": len(g),
            "durbin_watson": dw,
            "median_time_gap_min": float(dt_min.median()) if len(dt_min) else np.nan,
            "p90_time_gap_min": float(dt_min.quantile(.90)) if len(dt_min) else np.nan,
        })
        nlags = min(max_lag, len(e) - 1)
        av = acf(e, nlags=nlags, fft=True, missing="drop")
        for lag in range(1, len(av)):
            acf_rows.append({
                "validation": g["validation"].iloc[0],
                "vessel_id": vessel,
                "ship_type": g["ship_type"].iloc[0],
                "lag": lag,
                "acf": float(av[lag]),
            })
    return pd.DataFrame(summary), pd.DataFrame(acf_rows)


def serial_summary(per_vessel: pd.DataFrame, acf_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for validation, g in per_vessel.groupby("validation"):
        row = {
            "validation": validation,
            "vessels": int(g["vessel_id"].nunique()),
            "DW_median": float(g["durbin_watson"].median()),
            "DW_IQR_low": float(g["durbin_watson"].quantile(.25)),
            "DW_IQR_high": float(g["durbin_watson"].quantile(.75)),
        }
        a = acf_df[acf_df["validation"] == validation]
        for lag in sorted(a["lag"].unique()):
            z = a[a["lag"] == lag]["acf"]
            row[f"ACF_lag{lag}_median"] = float(z.median())
            row[f"ACF_lag{lag}_IQR_low"] = float(z.quantile(.25))
            row[f"ACF_lag{lag}_IQR_high"] = float(z.quantile(.75))
        rows.append(row)
    return pd.DataFrame(rows)


def breusch_pagan_table(rows: pd.DataFrame) -> pd.DataFrame:
    out = []
    for validation, g in rows.groupby("validation"):
        exog = pd.DataFrame({
            "const": 1.0,
            "y_pred": g["y_pred"].to_numpy(float),
            "speed_kn": g["speed_kn"].to_numpy(float),
            "draught_m": g["draught_m"].to_numpy(float),
            "wave_height_m": g["wave_height_m"].to_numpy(float),
            "ship_type_bulk": g["ship_type_bulk"].to_numpy(float),
            "ship_type_container": g["ship_type_container"].to_numpy(float),
        })
        resid = g["residual"].to_numpy(float)
        lm, lm_p, fval, f_p = het_breuschpagan(
            resid, exog.to_numpy(float), robust=True
        )
        aux = sm.OLS(resid ** 2, exog.to_numpy(float)).fit()
        out.append({
            "validation": validation,
            "n": len(g),
            "test": "Koenker-Breusch-Pagan",
            "LM": float(lm),
            "LM_p_value": float(lm_p),
            "F": float(fval),
            "F_p_value": float(f_p),
            "auxiliary_R2_residual_sq": float(aux.rsquared),
            "auxiliary_variables":
                "y_pred+speed_kn+draught_m+wave_height_m+ship_type_bulk+ship_type_container",
        })
    return pd.DataFrame(out)


def run_residuals(raw, record_te, record_pred, temporal_te, temporal_pred, out, args, logger):
    d = out / "03_residual_diagnostics"
    d.mkdir(parents=True, exist_ok=True)
    l1 = residual_row_frame(
        raw.iloc[record_te].reset_index(drop=True), record_pred, "L1_random_record"
    )
    l2 = residual_row_frame(
        raw.iloc[temporal_te].reset_index(drop=True), temporal_pred, "L2_known_vessel_temporal"
    )
    rows = pd.concat([l1, l2], ignore_index=True)

    serial_parts = []
    acf_parts = []
    for validation, g in rows.groupby("validation"):
        s, a = per_vessel_serial_diagnostics(g, args.acf_max_lag)
        serial_parts.append(s)
        acf_parts.append(a)
    serial = pd.concat(serial_parts, ignore_index=True)
    acfs = pd.concat(acf_parts, ignore_index=True)
    summary = serial_summary(serial, acfs)
    bp = breusch_pagan_table(rows)

    atomic_csv(serial, d / "Table_R1_DurbinWatson_by_vessel.csv")
    atomic_csv(acfs, d / "Table_R2_residual_ACF_by_vessel.csv")
    atomic_csv(summary, d / "Table_R3_serial_diagnostic_summary.csv")
    atomic_csv(bp, d / "Table_R4_BreuschPagan.csv")
    rows.to_csv(d / "residual_rows_L1_L2.csv.gz", index=False, compression="gzip")

    logger.info("Residual diagnostics written: %s", d)


def distribution_compare(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return {
            "train_n": len(a), "test_n": len(b), "wasserstein1": np.nan,
            "normalized_wasserstein1": np.nan, "KS_D": np.nan, "KS_p_value": np.nan,
            "train_mean": np.nan, "test_mean": np.nan, "train_sd": np.nan,
        }
    w = float(wasserstein_distance(a, b))
    sd = float(np.std(a, ddof=1))
    ks = ks_2samp(a, b, alternative="two-sided", method="auto")
    return {
        "train_n": int(len(a)),
        "test_n": int(len(b)),
        "wasserstein1": w,
        "normalized_wasserstein1": w / sd if sd > 0 else np.nan,
        "KS_D": float(ks.statistic),
        "KS_p_value": float(ks.pvalue),
        "train_mean": float(np.mean(a)),
        "test_mean": float(np.mean(b)),
        "train_sd": sd,
    }


def run_shift(raw, temporal_tr, temporal_te, lovo_pred, out, logger):
    d = out / "04_distribution_shift"
    d.mkdir(parents=True, exist_ok=True)

    # L2 pooled + per vessel
    rows_l2 = []
    for feat in SHIFT_FEATURES:
        x = distribution_compare(
            raw.iloc[temporal_tr][feat].to_numpy(float),
            raw.iloc[temporal_te][feat].to_numpy(float),
        )
        rows_l2.append({
            "validation": "L2_known_vessel_temporal",
            "scope": "pooled",
            "vessel_id": "",
            "ship_type": "",
            "feature": feat,
            **x,
        })
    for vessel, idx in raw.groupby("vessel_id").groups.items():
        idx = np.asarray(list(idx), dtype=int)
        tr = np.intersect1d(idx, temporal_tr, assume_unique=False)
        te = np.intersect1d(idx, temporal_te, assume_unique=False)
        for feat in SHIFT_FEATURES:
            rows_l2.append({
                "validation": "L2_known_vessel_temporal",
                "scope": "vessel",
                "vessel_id": vessel,
                "ship_type": raw.loc[idx[0], "ship_type"],
                "feature": feat,
                **distribution_compare(
                    raw.loc[tr, feat].to_numpy(float),
                    raw.loc[te, feat].to_numpy(float),
                ),
            })
    L2 = pd.DataFrame(rows_l2)
    atomic_csv(L2, d / "Table_D1_L2_temporal_distribution_shift.csv")

    # L3 LOVO shift + performance
    perf_rows = []
    for vessel, idx in raw.groupby("vessel_id").groups.items():
        ii = np.asarray(list(idx), dtype=int)
        perf_rows.append({
            "vessel_id": vessel,
            "ship_type": raw.loc[ii[0], "ship_type"],
            **metric_dict(raw.loc[ii, "target"].to_numpy(float), lovo_pred[ii]),
        })
    perf = pd.DataFrame(perf_rows)

    rows_l3 = []
    all_idx = np.arange(len(raw))
    for vessel, idx in raw.groupby("vessel_id").groups.items():
        te = np.asarray(list(idx), dtype=int)
        tr_mask = np.ones(len(raw), dtype=bool)
        tr_mask[te] = False
        tr = all_idx[tr_mask]
        p = perf[perf["vessel_id"] == vessel].iloc[0].to_dict()
        for feat in SHIFT_FEATURES:
            rows_l3.append({
                "validation": "L3_LOVO_zero_shot",
                "scope": "heldout_vessel",
                "vessel_id": vessel,
                "ship_type": p["ship_type"],
                "feature": feat,
                **distribution_compare(
                    raw.loc[tr, feat].to_numpy(float),
                    raw.loc[te, feat].to_numpy(float),
                ),
                "LOVO_RMSE": p["RMSE"],
                "LOVO_MAE": p["MAE"],
                "LOVO_R2": p["R2"],
            })
    L3 = pd.DataFrame(rows_l3)
    atomic_csv(L3, d / "Table_D2_L3_LOVO_distribution_shift_and_performance.csv")
    atomic_csv(perf, d / "Table_D3_LOVO_performance_reconstructed.csv")

    logger.info("Distribution-shift diagnostics written: %s", d)


def cubic_design(speed: np.ndarray, ship_type: Sequence[str], mu: float, sd: float) -> np.ndarray:
    z = (np.asarray(speed, dtype=float) - mu) / max(sd, 1e-12)
    st = np.asarray(ship_type).astype(str)
    return np.column_stack([
        z,
        z ** 2,
        z ** 3,
        (st == "bulk").astype(float),
        (st == "container").astype(float),
    ])


def run_minimal_cubic_lovo(raw, out, args, logger):
    d = out / "05_minimal_cubic_lovo"
    d.mkdir(parents=True, exist_ok=True)
    pred_path = d / "minimal_cubic_LOVO_predictions.npy"
    if args.resume and pred_path.exists():
        pred_all = np.load(pred_path)
        logger.info("Minimal cubic LOVO prediction reuse: %s", pred_path)
    else:
        pred_all = np.full(len(raw), np.nan, dtype=float)
        vessels = sorted(raw["vessel_id"].unique())
        for i, vessel in enumerate(vessels, start=1):
            te = (raw["vessel_id"].to_numpy(str) == str(vessel))
            tr = ~te
            s_tr = raw.loc[tr, "speed_kn"].to_numpy(float)
            mu = float(np.mean(s_tr))
            sd = float(np.std(s_tr, ddof=0))
            Xtr = cubic_design(
                s_tr, raw.loc[tr, "ship_type"].to_numpy(str), mu, sd
            )
            Xte = cubic_design(
                raw.loc[te, "speed_kn"].to_numpy(float),
                raw.loc[te, "ship_type"].to_numpy(str), mu, sd
            )
            m = LinearRegression(fit_intercept=True, copy_X=False, n_jobs=1)
            m.fit(Xtr, raw.loc[tr, "target"].to_numpy(float))
            pred_all[te] = m.predict(Xte)
            logger.info(
                "Minimal cubic LOVO %02d/%02d | vessel=%s | test=%d",
                i, len(vessels), vessel, int(te.sum())
            )
        np.save(pred_path, pred_all)

    if not np.isfinite(pred_all).all():
        raise AssertionError("Minimal cubic LOVO contains non-finite predictions.")

    overall = pd.DataFrame([{
        "model": "minimal_shiptype_cubic_speed",
        **metric_dict(raw["target"].to_numpy(float), pred_all),
    }])
    byv = []
    for vessel, idx in raw.groupby("vessel_id").groups.items():
        ii = np.asarray(list(idx), dtype=int)
        byv.append({
            "vessel_id": vessel,
            "ship_type": raw.loc[ii[0], "ship_type"],
            **metric_dict(raw.loc[ii, "target"].to_numpy(float), pred_all[ii]),
        })
    byv = pd.DataFrame(byv)
    atomic_csv(overall, d / "Table_M1_minimal_cubic_LOVO_overall.csv")
    atomic_csv(byv, d / "Table_M2_minimal_cubic_LOVO_by_vessel.csv")
    logger.info(
        "Minimal cubic LOVO | RMSE=%.6f R2=%.6f",
        overall.iloc[0]["RMSE"], overall.iloc[0]["R2"]
    )


def cluster_bootstrap_delta(
    vessel_ids: Sequence[str],
    y: np.ndarray,
    pred_a: np.ndarray,
    pred_ref: np.ndarray,
    reps: int,
    seed: int,
) -> Dict[str, float]:
    ids = np.asarray(vessel_ids).astype(str)
    y = np.asarray(y, dtype=float)
    a = np.asarray(pred_a, dtype=float)
    b = np.asarray(pred_ref, dtype=float)
    uniq = np.unique(ids)
    idx_map = {v: np.flatnonzero(ids == v) for v in uniq}
    rng = np.random.default_rng(seed)
    vals = np.empty(reps, dtype=float)
    for j in range(reps):
        draw = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_map[v] for v in draw])
        ra = math.sqrt(mean_squared_error(y[idx], a[idx]))
        rb = math.sqrt(mean_squared_error(y[idx], b[idx]))
        vals[j] = ra - rb
    return {
        "delta_RMSE_boot_mean": float(np.mean(vals)),
        "CI95_low": float(np.quantile(vals, .025)),
        "CI95_high": float(np.quantile(vals, .975)),
    }


def source_label(ship_type: pd.Series) -> pd.Series:
    # Locked study design: container meteorology from ERA5; bulk+tanker pre-matched.
    return np.where(
        ship_type.astype(str).str.lower().eq("container"),
        "ERA5_container",
        "pre_matched_bulk_tanker",
    )


def run_ablation(raw, X, record_tr, record_te, record_model, record_pred, out, args, logger):
    d = out / "06_xgb_ablation"
    d.mkdir(parents=True, exist_ok=True)
    ytr = raw.iloc[record_tr]["target"].to_numpy(float)
    yte = raw.iloc[record_te]["target"].to_numpy(float)
    rawte = raw.iloc[record_te].reset_index(drop=True)

    configs = {
        "operational_core": (X.iloc[record_tr][OPERATIONAL_FEATURES],
                             X.iloc[record_te][OPERATIONAL_FEATURES]),
        "weather_only": (X.iloc[record_tr][WEATHER_FEATURES],
                         X.iloc[record_te][WEATHER_FEATURES]),
        "dynamic_physical": (X.iloc[record_tr][PRIMARY_FEATURES],
                             X.iloc[record_te][PRIMARY_FEATURES]),
        "dynamic_interaction": (
            interaction_feature_matrix(X.iloc[record_tr][PRIMARY_FEATURES]),
            interaction_feature_matrix(X.iloc[record_te][PRIMARY_FEATURES]),
        ),
    }

    preds: Dict[str, np.ndarray] = {
        "dynamic_physical": np.asarray(record_pred, dtype=float)
    }

    for label, (xtr, xte) in configs.items():
        if label == "dynamic_physical":
            np.save(d / f"prediction_{label}.npy", preds[label])
            continue
        pred_path = d / f"prediction_{label}.npy"
        model_path = d / f"xgb_{label}.joblib"
        if args.resume and pred_path.exists():
            preds[label] = np.load(pred_path).astype(float)
            logger.info("Ablation prediction reuse: %s", label)
            continue

        logger.info("Locked-XGB ablation fit: %s | features=%d", label, xtr.shape[1])
        m = clone(record_model)
        try:
            m.set_params(n_jobs=args.n_jobs)
        except Exception:
            pass
        m.fit(xtr, ytr)
        p = np.asarray(m.predict(xte), dtype=float)
        preds[label] = p
        np.save(pred_path, p)
        joblib.dump(m, model_path)
        del m
        gc.collect()

    rows = []
    for label, (_, xte) in configs.items():
        rows.append({
            "configuration": label,
            "model": "xgb_locked",
            "n_features": int(xte.shape[1]),
            **metric_dict(yte, preds[label]),
        })
    tab = pd.DataFrame(rows)
    ref_rmse = float(tab.loc[tab["configuration"] == "dynamic_physical", "RMSE"].iloc[0])
    tab["delta_RMSE_vs_dynamic_physical"] = tab["RMSE"] - ref_rmse
    tab["RMSE_worsening_pct_vs_dynamic_physical"] = (tab["RMSE"] / ref_rmse - 1.0) * 100.0
    atomic_csv(tab, d / "Table_A1_XGB_multisource_feature_ablation.csv")

    boot = []
    for j, label in enumerate(["operational_core", "weather_only", "dynamic_interaction"]):
        boot.append({
            "scope": "pooled",
            "source": "all",
            "configuration": label,
            "reference": "dynamic_physical",
            "n_vessels": int(rawte["vessel_id"].nunique()),
            **cluster_bootstrap_delta(
                rawte["vessel_id"].to_numpy(str), yte,
                preds[label], preds["dynamic_physical"],
                args.bootstrap_reps, args.seed + 500 + j,
            ),
        })
    atomic_csv(pd.DataFrame(boot), d / "Table_A2_XGB_vessel_cluster_bootstrap.csv")

    # Long row-level prediction cache.
    long_parts = []
    base = rawte[["row_id", "vessel_id", "ship_type", "target"]].copy()
    base["met_source"] = source_label(base["ship_type"])
    for label, p in preds.items():
        z = base.copy()
        z["configuration"] = label
        z["prediction"] = p
        z["residual"] = p - z["target"].to_numpy(float)
        long_parts.append(z)
    long = pd.concat(long_parts, ignore_index=True)
    long.to_csv(d / "ablation_predictions_long.csv.gz", index=False, compression="gzip")

    # Per-vessel metrics.
    byv = []
    for label, p in preds.items():
        for vessel, idx in rawte.groupby("vessel_id").groups.items():
            ii = np.asarray(list(idx), dtype=int)
            byv.append({
                "vessel_id": vessel,
                "ship_type": rawte.loc[ii[0], "ship_type"],
                "met_source": source_label(rawte.loc[ii, "ship_type"])[0],
                "configuration": label,
                **metric_dict(yte[ii], p[ii]),
            })
    atomic_csv(pd.DataFrame(byv), d / "Table_A3_XGB_ablation_by_vessel.csv")

    # Source-stratified observation-weighted metrics + source-specific vessel bootstrap.
    src_rows = []
    src_boot = []
    sources = pd.Series(source_label(rawte["ship_type"]), index=rawte.index)
    for src in sorted(sources.unique()):
        msk = sources.to_numpy(str) == src
        vids = rawte.loc[msk, "vessel_id"].to_numpy(str)
        for label, p in preds.items():
            src_rows.append({
                "source": src,
                "configuration": label,
                "n": int(msk.sum()),
                "n_vessels": int(pd.Series(vids).nunique()),
                **metric_dict(yte[msk], p[msk]),
            })
        for j, label in enumerate(["operational_core", "weather_only", "dynamic_interaction"]):
            src_boot.append({
                "source": src,
                "configuration": label,
                "reference": "dynamic_physical",
                "n": int(msk.sum()),
                "n_vessels": int(pd.Series(vids).nunique()),
                **cluster_bootstrap_delta(
                    vids, yte[msk], preds[label][msk],
                    preds["dynamic_physical"][msk],
                    args.bootstrap_reps,
                    args.seed + 600 + j + (100 if src.startswith("ERA5") else 0),
                ),
            })

    source_tab = pd.DataFrame(src_rows)
    ref_map = source_tab[source_tab.configuration == "dynamic_physical"].set_index("source")["RMSE"]
    source_tab["delta_RMSE_vs_dynamic_physical"] = [
        r.RMSE - ref_map.loc[r.source] for r in source_tab.itertuples()
    ]
    atomic_csv(source_tab, d / "Table_A4_XGB_ablation_by_meteorological_source.csv")
    atomic_csv(pd.DataFrame(src_boot), d / "Table_A5_source_stratified_vessel_bootstrap.csv")

    (d / "SOURCE_CONFOUNDING_NOTE.txt").write_text(
        "Meteorological source is inferred from the locked study design: "
        "containers=ERA5; bulk+tanker=pre-matched. Source is therefore heavily "
        "confounded with vessel type. The source-stratified results are robustness "
        "diagnostics and must not be interpreted as causal source effects.\n",
        encoding="utf-8",
    )
    logger.info("Locked-XGB ablation and source diagnostics written: %s", d)


def deterministic_shap_sample(record_te: np.ndarray, n: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    if len(record_te) <= n:
        local = np.arange(len(record_te), dtype=int)
    else:
        local = np.sort(rng.choice(len(record_te), size=n, replace=False))
    return local, record_te[local]


def compute_interventional_shap(
    raw, X, record_tr, record_te, model, artifacts, out, args, logger
) -> Tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    if shap is None:
        raise RuntimeError(f"SHAP could not be imported: {_SHAP_IMPORT_ERROR}")

    d = out / "07_interventional_shap"
    d.mkdir(parents=True, exist_ok=True)

    sv_path = d / "interventional_SHAP_values.npy"
    x_path = d / "interventional_SHAP_sample_features.csv"
    row_path = d / "interventional_SHAP_sample_rows.csv"
    bg_idx_path = d / "interventional_SHAP_background_indices.npy"

    if args.resume and sv_path.exists() and x_path.exists() and row_path.exists():
        logger.info("Interventional SHAP reuse: %s", sv_path)
        return np.load(sv_path), pd.read_csv(x_path), pd.read_csv(row_path)

    local_idx, global_idx = deterministic_shap_sample(
        record_te, min(args.shap_eval_n, len(record_te)), args.seed + 800
    )
    Xsh = X.iloc[global_idx].reset_index(drop=True)
    rawsh = raw.iloc[global_idx].reset_index(drop=True)

    rng = np.random.default_rng(args.seed + 801)
    bg_n = min(args.shap_background_n, len(record_tr))
    bg_global = np.sort(rng.choice(record_tr, size=bg_n, replace=False))
    background = X.iloc[bg_global].reset_index(drop=True)
    np.save(bg_idx_path, bg_global)

    logger.info(
        "Interventional TreeSHAP | eval=%d background=%d batches=%d",
        len(Xsh), len(background), math.ceil(len(Xsh) / args.shap_batch_size)
    )
    explainer = shap.TreeExplainer(
        model,
        data=background,
        feature_perturbation="interventional",
        model_output="raw",
    )

    sv = np.empty((len(Xsh), Xsh.shape[1]), dtype=np.float64)
    for start in range(0, len(Xsh), args.shap_batch_size):
        stop = min(len(Xsh), start + args.shap_batch_size)
        block = Xsh.iloc[start:stop]
        vals = explainer.shap_values(block, check_additivity=False)
        if isinstance(vals, list):
            vals = vals[0]
        sv[start:stop] = np.asarray(vals, dtype=float)
        logger.info("  SHAP rows %d:%d complete", start, stop)

    np.save(sv_path, sv)
    Xsh.to_csv(x_path, index=False)

    row_meta = rawsh[
        ["row_id", "vessel_id", "ship_type", "timestamp", "target",
         "speed_kn", "draught_m", "rudder_deg", "wave_height_m",
         "rel_wind_speed_kn"]
    ].copy()
    row_meta["official_test_position"] = local_idx
    row_meta.to_csv(row_path, index=False)

    # Additivity diagnostic.
    ncheck = min(200, len(Xsh))
    idx = np.linspace(0, len(Xsh) - 1, ncheck).round().astype(int)
    pred = np.asarray(model.predict(Xsh.iloc[idx]), dtype=float)
    expected = np.asarray(explainer.expected_value).reshape(-1)
    expected_scalar = float(expected[0]) if expected.size else float("nan")
    recon = expected_scalar + sv[idx].sum(axis=1)
    add = pd.DataFrame([{
        "n_checked": ncheck,
        "expected_value": expected_scalar,
        "mean_abs_additivity_gap": float(np.mean(np.abs(pred - recon))),
        "max_abs_additivity_gap": float(np.max(np.abs(pred - recon))),
        "feature_perturbation": "interventional",
        "background_n": len(background),
    }])
    atomic_csv(add, d / "Table_SHAP0_additivity_check.csv")

    imp = pd.DataFrame({
        "feature": Xsh.columns,
        "mean_abs_SHAP": np.mean(np.abs(sv), axis=0),
    }).sort_values("mean_abs_SHAP", ascending=False)
    imp["normalized_importance"] = imp["mean_abs_SHAP"] / imp["mean_abs_SHAP"].sum()
    imp["rank"] = np.arange(1, len(imp) + 1)
    atomic_csv(imp, d / "Table_SHAP1_interventional_global_importance.csv")

    direction = []
    for feat in ["speed_kn", "wave_height_m", "draught_m", "trim_m", "rel_wind_speed_kn"]:
        j = Xsh.columns.get_loc(feat)
        direction.append({
            "feature": feat,
            "spearman_feature_vs_SHAP": float(
                spearmanr(Xsh[feat], sv[:, j], nan_policy="omit").statistic
            ),
        })
    j = Xsh.columns.get_loc("rudder_deg")
    direction.append({
        "feature": "abs_rudder_deg",
        "spearman_feature_vs_SHAP": float(
            spearmanr(np.abs(Xsh["rudder_deg"]), sv[:, j], nan_policy="omit").statistic
        ),
    })
    atomic_csv(pd.DataFrame(direction), d / "Table_SHAP2_interventional_direction.csv")
    return sv, Xsh, row_meta


def gam_fit_alpha_grid(x: np.ndarray, y: np.ndarray, basis_df: int, log10_alphas: Sequence[float]):
    x2 = np.asarray(x, dtype=float)[:, None]
    bs = BSplines(x2, df=[basis_df], degree=[3])
    exog = np.ones((len(x), 1), dtype=float)
    rows = []
    best = None
    for lg in log10_alphas:
        alpha = 10.0 ** float(lg)
        try:
            res = GLMGam(
                y, exog=exog, smoother=bs, alpha=[alpha],
                family=sm.families.Gaussian()
            ).fit()
            rows.append({
                "log10_alpha": float(lg),
                "alpha": alpha,
                "AIC": float(res.aic),
                "GCV": float(res.gcv),
                "hat_matrix_trace": float(res.hat_matrix_trace),
            })
            score = float(res.gcv)
            if best is None or score < best[0]:
                best = (score, alpha, res, bs)
        except Exception as exc:
            rows.append({
                "log10_alpha": float(lg), "alpha": alpha,
                "AIC": np.nan, "GCV": np.nan, "hat_matrix_trace": np.nan,
                "error": f"{type(exc).__name__}: {exc}",
            })
    if best is None:
        raise RuntimeError("All GAM alpha candidates failed.")
    return best[2], best[3], pd.DataFrame(rows), best[1]


def run_gam_loess(sv, Xsh, row_meta, out, args, logger):
    d = out / "07_interventional_shap"
    speed_j = Xsh.columns.get_loc("speed_kn")
    x = Xsh["speed_kn"].to_numpy(float)
    y = np.asarray(sv[:, speed_j], dtype=float)

    qlo, qhi = np.quantile(x, [args.support_q_low, args.support_q_high])
    grid = np.linspace(qlo, qhi, args.gam_grid_n)
    log_alphas = [float(v) for v in args.gam_alpha_grid.split(",") if v.strip()]

    res, bs, alpha_tab, selected_alpha = gam_fit_alpha_grid(
        x, y, args.gam_basis_df, log_alphas
    )
    atomic_csv(alpha_tab, d / "Table_GAM0_alpha_selection.csv")

    exog_grid = np.ones((len(grid), 1), dtype=float)
    pred = res.get_prediction(
        exog=exog_grid, exog_smooth=grid[:, None], transform=True
    )
    sf = pred.summary_frame(alpha=0.05)
    fit = np.asarray(sf["mean"], dtype=float)
    fit_lo = np.asarray(sf["mean_ci_lower"], dtype=float)
    fit_hi = np.asarray(sf["mean_ci_upper"], dtype=float)

    # Effective degrees of freedom: exclude the unpenalized intercept.
    edf = np.asarray(res.edf, dtype=float)
    smooth_edf = float(np.sum(edf[1:])) if edf.size > 1 else float(np.sum(edf))

    # Joint Wald test for all smooth-basis coefficients.
    params = np.asarray(res.params, dtype=float)
    cov = np.asarray(res.cov_params(), dtype=float)
    beta = params[1:]
    cov_b = cov[1:, 1:]
    rank = int(np.linalg.matrix_rank(cov_b))
    wald = float(beta @ np.linalg.pinv(cov_b) @ beta)
    smooth_p = float(chi2.sf(wald, max(rank, 1)))

    # Delta-method derivatives from finite differences of the spline design.
    xrng = max(qhi - qlo, 1e-6)
    h = xrng / 5000.0
    xp = (grid + h)[:, None]
    xm = (grid - h)[:, None]
    xpp = (grid + h)[:, None]
    xmm = (grid - h)[:, None]
    B0 = bs.transform(grid[:, None])
    Bp = bs.transform(xp)
    Bm = bs.transform(xm)
    X0 = np.column_stack([np.ones(len(grid)), B0])
    Xp = np.column_stack([np.ones(len(grid)), Bp])
    Xm = np.column_stack([np.ones(len(grid)), Bm])
    D1 = (Xp - Xm) / (2.0 * h)
    D2 = (Xp - 2.0 * X0 + Xm) / (h ** 2)

    d1 = D1 @ params
    d2 = D2 @ params
    v1 = np.einsum("ij,jk,ik->i", D1, cov, D1)
    v2 = np.einsum("ij,jk,ik->i", D2, cov, D2)
    se1 = np.sqrt(np.clip(v1, 0, None))
    se2 = np.sqrt(np.clip(v2, 0, None))
    d1_lo, d1_hi = d1 - 1.96 * se1, d1 + 1.96 * se1
    d2_lo, d2_hi = d2 - 1.96 * se2, d2 + 1.96 * se2

    curve = pd.DataFrame({
        "speed_kn": grid,
        "GAM_fit": fit,
        "GAM_CI95_low": fit_lo,
        "GAM_CI95_high": fit_hi,
        "d1": d1,
        "d1_CI95_low": d1_lo,
        "d1_CI95_high": d1_hi,
        "d2": d2,
        "d2_CI95_low": d2_lo,
        "d2_CI95_high": d2_hi,
        "support_q_low": args.support_q_low,
        "support_q_high": args.support_q_high,
    })
    atomic_csv(curve, d / "Table_GAM1_speed_SHAP_GAM_curve_and_derivatives.csv")

    summary = pd.DataFrame([{
        "n_raw_SHAP": len(x),
        "selected_alpha": selected_alpha,
        "basis_df": args.gam_basis_df,
        "smooth_effective_df": smooth_edf,
        "joint_smooth_Wald_chi2": wald,
        "joint_smooth_df": rank,
        "smooth_term_p_value": smooth_p,
        "support_speed_low": qlo,
        "support_speed_high": qhi,
        "fraction_grid_d1_positive": float(np.mean(d1 > 0)),
        "fraction_grid_d1_lower95_positive": float(np.mean(d1_lo > 0)),
        "fraction_grid_d2_positive": float(np.mean(d2 > 0)),
        "fraction_grid_d2_lower95_positive": float(np.mean(d2_lo > 0)),
    }])
    atomic_csv(summary, d / "Table_GAM2_speed_SHAP_GAM_summary.csv")

    # LOWESS primary fit.
    delta = 0.01 * xrng
    smoothed = lowess(
        y, x, frac=args.loess_frac, it=1, delta=delta,
        is_sorted=False, missing="drop", return_sorted=True
    )
    loess_fit = np.interp(grid, smoothed[:, 0], smoothed[:, 1])

    # Vessel-cluster bootstrap confidence band.
    vessel = row_meta["vessel_id"].astype(str).to_numpy()
    uniq = np.unique(vessel)
    index_by = {v: np.flatnonzero(vessel == v) for v in uniq}
    rng = np.random.default_rng(args.seed + 910)
    boots = np.full((args.loess_bootstrap_reps, len(grid)), np.nan, dtype=float)
    for b in range(args.loess_bootstrap_reps):
        draw = rng.choice(uniq, size=len(uniq), replace=True)
        ii = np.concatenate([index_by[v] for v in draw])
        sb = lowess(
            y[ii], x[ii], frac=args.loess_frac, it=1, delta=delta,
            is_sorted=False, missing="drop", return_sorted=True
        )
        boots[b] = np.interp(grid, sb[:, 0], sb[:, 1])
        if (b + 1) % 50 == 0:
            logger.info("LOWESS vessel bootstrap %d/%d", b + 1, args.loess_bootstrap_reps)

    loess_tab = pd.DataFrame({
        "speed_kn": grid,
        "LOWESS_fit": loess_fit,
        "LOWESS_cluster_boot_CI95_low": np.nanquantile(boots, .025, axis=0),
        "LOWESS_cluster_boot_CI95_high": np.nanquantile(boots, .975, axis=0),
        "frac": args.loess_frac,
        "bootstrap_reps": args.loess_bootstrap_reps,
    })
    atomic_csv(loess_tab, d / "Table_LOESS1_speed_SHAP_curve.csv")
    logger.info("GAM/LOWESS outputs written: %s", d)


def vessel_bootstrap_mean(values: np.ndarray, vessel: np.ndarray, reps: int, seed: int):
    vessel = np.asarray(vessel).astype(str)
    values = np.asarray(values, dtype=float)
    uniq = np.unique(vessel)
    idx_map = {v: np.flatnonzero(vessel == v) for v in uniq}
    rng = np.random.default_rng(seed)
    means = np.empty(reps, dtype=float)
    for i in range(reps):
        draw = rng.choice(uniq, size=len(uniq), replace=True)
        ii = np.concatenate([idx_map[v] for v in draw])
        means[i] = np.mean(values[ii])
    return (
        float(np.mean(means)),
        float(np.quantile(means, .025)),
        float(np.quantile(means, .975)),
    )


def run_rudder(raw, X, record_te, model, out, args, logger):
    d = out / "08_rudder_power"
    d.mkdir(parents=True, exist_ok=True)
    row_path = d / "rudder_plus1_row_level.csv.gz"

    if args.resume and row_path.exists():
        rows = pd.read_csv(row_path)
        logger.info("Rudder row-level perturbation reuse: %s", row_path)
    else:
        rng = np.random.default_rng(args.seed + 822)
        n = min(args.rudder_sample_n, len(record_te))
        local = np.sort(rng.choice(len(record_te), size=n, replace=False))
        global_idx = record_te[local]

        base_raw = raw.iloc[global_idx].reset_index(drop=True).copy()
        Xb = X.iloc[global_idx].reset_index(drop=True).copy()
        Xp = Xb.copy()

        r = Xp["rudder_deg"].to_numpy(float)
        sign = np.sign(r)
        sign[sign == 0] = 1.0
        Xp["rudder_deg"] = sign * (np.abs(r) + 1.0)

        base_pred = np.asarray(model.predict(Xb), dtype=float)
        pert_pred = np.asarray(model.predict(Xp), dtype=float)
        delta = pert_pred - base_pred

        rows = base_raw[
            ["row_id", "vessel_id", "ship_type", "timestamp",
             "rudder_deg", "speed_kn", "target"]
        ].copy()
        rows["abs_rudder_original"] = np.abs(rows["rudder_deg"].to_numpy(float))
        rows["prediction_base"] = base_pred
        rows["prediction_abs_rudder_plus1deg"] = pert_pred
        rows["delta"] = delta
        rows["official_test_position"] = local
        rows.to_csv(row_path, index=False, compression="gzip")

    thresholds = sorted(set(
        [args.rudder_near_zero_deg] +
        [float(v) for v in args.rudder_sensitivity_thresholds.split(",") if v.strip()]
    ))

    results = []
    power_solver = TTestPower()
    for th in thresholds:
        tail = rows[rows["abs_rudder_original"] > th].copy()
        dlt = tail["delta"].to_numpy(float)
        n = len(dlt)
        nv = int(tail["vessel_id"].nunique())
        mean = float(np.mean(dlt)) if n else np.nan
        sd = float(np.std(dlt, ddof=1)) if n > 1 else np.nan
        median = float(np.median(dlt)) if n else np.nan
        d_obs = mean / sd if n > 1 and sd > 0 else np.nan
        mde_row = (
            float(power_solver.solve_power(
                effect_size=None, nobs=n, alpha=.05, power=.80, alternative="larger"
            ))
            if n >= 3 else np.nan
        )

        # Vessel-level sensitivity: one mean delta per vessel.
        vmean = tail.groupby("vessel_id")["delta"].mean().to_numpy(float)
        mde_vessel = (
            float(power_solver.solve_power(
                effect_size=None, nobs=len(vmean), alpha=.05, power=.80,
                alternative="larger"
            ))
            if len(vmean) >= 3 else np.nan
        )

        if n and nv:
            bm, blo, bhi = vessel_bootstrap_mean(
                dlt, tail["vessel_id"].to_numpy(str),
                args.bootstrap_reps, args.seed + 1200 + int(round(th * 100))
            )
        else:
            bm = blo = bhi = np.nan

        results.append({
            "threshold_abs_rudder_deg": th,
            "is_primary_threshold": bool(abs(th - args.rudder_near_zero_deg) < 1e-12),
            "tail_definition": f"abs(rudder_deg) > {th:g}",
            "n_tail_rows": n,
            "n_tail_vessels": nv,
            "mean_delta": mean,
            "sd_delta": sd,
            "median_delta": median,
            "positive_fraction": float(np.mean(dlt > 1e-8)) if n else np.nan,
            "negative_fraction": float(np.mean(dlt < -1e-8)) if n else np.nan,
            "observed_Cohens_d_row_level": d_obs,
            "MDE_Cohens_d_row_level_nominal_alpha05_power80_one_sided": mde_row,
            "MDE_Cohens_d_vessel_mean_sensitivity_alpha05_power80_one_sided": mde_vessel,
            "cluster_boot_mean_delta": bm,
            "cluster_boot_CI95_low": blo,
            "cluster_boot_CI95_high": bhi,
        })

    atomic_csv(pd.DataFrame(results), d / "Table_RUD1_rudder_tail_power.csv")
    descriptive = pd.DataFrame([{
        "n_audit": len(rows),
        "rudder_mean_deg": float(rows["rudder_deg"].mean()),
        "rudder_sd_deg": float(rows["rudder_deg"].std(ddof=1)),
        "abs_rudder_median_deg": float(rows["abs_rudder_original"].median()),
        "abs_rudder_q90_deg": float(rows["abs_rudder_original"].quantile(.90)),
        "abs_rudder_q95_deg": float(rows["abs_rudder_original"].quantile(.95)),
        "abs_rudder_q99_deg": float(rows["abs_rudder_original"].quantile(.99)),
        "overall_delta_median": float(rows["delta"].median()),
        "overall_positive_fraction": float((rows["delta"] > 1e-8).mean()),
        "overall_negative_fraction": float((rows["delta"] < -1e-8).mean()),
    }])
    atomic_csv(descriptive, d / "Table_RUD0_rudder_audit_descriptive.csv")

    logger.info("Rudder row-level audit and power outputs written: %s", d)


def write_manifest(
    out, args, paths, core_path, hp_path, overrides_path, steps,
    record_tr, record_te, raw
):
    manifest = {
        "version": VERSION,
        "steps_requested": steps,
        "analysis_choices": {
            "residual_sign": "prediction_minus_observation",
            "acf_max_lag": args.acf_max_lag,
            "BP": "Koenker-Breusch-Pagan robust=True",
            "shift_features": SHIFT_FEATURES,
            "minimal_cubic": "ship_type + standardized_speed + speed^2 + speed^3; OLS-equivalent LinearRegression",
            "ablation_model": "locked XGB; dynamic_physical reuses official cached model/prediction",
            "source_mapping": {
                "container": "ERA5_container",
                "bulk": "pre_matched_bulk_tanker",
                "tanker": "pre_matched_bulk_tanker",
            },
            "source_caveat": "source is structurally confounded with vessel type",
            "SHAP": {
                "feature_perturbation": "interventional",
                "model_output": "raw",
                "background_n": args.shap_background_n,
                "eval_n": args.shap_eval_n,
            },
            "GAM_support_quantiles": [args.support_q_low, args.support_q_high],
            "LOWESS_frac": args.loess_frac,
            "rudder_primary_near_zero_deg": args.rudder_near_zero_deg,
            "rudder_power": "one-sample/paired-style one-sided t-test MDE alpha=.05 power=.80",
        },
        "cohort": {
            "cruise_rows": len(raw),
            "vessels": int(raw["vessel_id"].nunique()),
            "record_train_rows": len(record_tr),
            "record_test_rows": len(record_te),
        },
        "files": {},
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "statsmodels": getattr(sm, "__version__", "unknown"),
            "shap": getattr(shap, "__version__", None) if shap is not None else None,
        },
    }
    all_paths = {
        "core": core_path,
        "hyperparams": hp_path,
        "column_overrides": overrides_path,
        **paths,
    }
    for key, p in all_paths.items():
        if p is not None and Path(p).exists():
            pp = Path(p)
            manifest["files"][key] = {
                "path": str(pp),
                "sha256_16": sha256_short(pp),
                "bytes": pp.stat().st_size,
            }
    (out / "revision_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8"
    )


def main() -> int:
    args = parse_args()
    steps = parse_steps(args.steps)
    main_out = Path(args.main_output)
    out = Path(args.output_dir) if args.output_dir else main_out / "15_revision_diagnostics"
    logger = setup_logger(out)
    started = time.time()

    here = Path(__file__).resolve().parent
    core, core_path = load_core(args.core_path)
    hp_path = resolve_script_file(args.hyperparams_csv, "02_best_hyperparameters.csv")
    overrides_path = resolve_script_file(args.column_overrides, "column_overrides_fixed31.json")
    _ = load_locked_xgb_params(hp_path)  # validation; model artifact remains source of fit params
    artifacts = resolve_artifacts(args, main_out, here)

    logger.info("F31 revision diagnostics %s", VERSION)
    logger.info("Steps: %s", steps)
    raw, X = load_cruise(core, args, overrides_path, logger)
    record_tr, record_te, record_pred, lovo_pred, record_model = validate_locked_assets(
        raw, X, artifacts, logger
    )
    write_manifest(
        out, args, artifacts, core_path, hp_path, overrides_path,
        steps, record_tr, record_te, raw
    )

    temporal_tr = temporal_te = temporal_pred = None
    if any(s in steps for s in ["temporal", "residuals", "shift"]):
        temporal_tr, temporal_te, temporal_pred = get_or_run_temporal(
            raw, X, record_model, artifacts, out, args, logger
        )

    if "residuals" in steps:
        run_residuals(
            raw, record_te, record_pred, temporal_te, temporal_pred, out, args, logger
        )

    if "shift" in steps:
        run_shift(
            raw, temporal_tr, temporal_te, lovo_pred, out, logger
        )

    if "cubic" in steps:
        run_minimal_cubic_lovo(raw, out, args, logger)

    if "ablation" in steps:
        run_ablation(
            raw, X, record_tr, record_te, record_model, record_pred,
            out, args, logger
        )

    if "shap" in steps:
        sv, Xsh, row_meta = compute_interventional_shap(
            raw, X, record_tr, record_te, record_model, artifacts,
            out, args, logger
        )
        run_gam_loess(sv, Xsh, row_meta, out, args, logger)

    if "rudder" in steps:
        run_rudder(raw, X, record_te, record_model, out, args, logger)

    logger.info(
        "Revision diagnostics complete | elapsed_seconds=%.1f | output=%s",
        time.time() - started, out
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
