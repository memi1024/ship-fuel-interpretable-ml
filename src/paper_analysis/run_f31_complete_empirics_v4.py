#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Predictive-validation pipeline for the Fixed31 ship-fuel analysis.

The primary cohort contains 489,620 cruising records. The workflow runs the seven-model
record-level comparison, operating-regime robustness, feature ablation, temporal
extrapolation, LOVO transfer, target-vessel adaptation, ship-type-specific modelling,
SHAP physical diagnostics, explanation stability, and generalisation summaries using
fixed model-selection settings.
"""

from __future__ import annotations

import argparse
import csv
from array import array
import json
import gc
import faulthandler
import logging
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import statsmodels.api as sm
except Exception:
    sm = None

try:
    from lime.lime_tabular import LimeTabularExplainer
except Exception:
    LimeTabularExplainer = None

import f31_core_memory_safe as core

SCRIPT_VERSION = "2026-08-09.complete-extension-v4-cruise-stream-cache"
BASE_MODELS = ["lr", "dt", "rf", "xgb", "lgbm", "ann"]
ALL_MODELS = ["lr", "ridge_interaction", "dt", "rf", "xgb", "lgbm", "ann"]
TREE_MODELS = ["rf", "xgb", "lgbm"]
DEFAULT_STEPS = [
    "data", "base", "phase", "ablation", "temporal", "lovo", "adaptation",
    "shiptype", "shap", "shap_stability", "summary",
]


def setup_logger(out_dir: Path) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("f31_complete")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(out_dir / "run_complete.log", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def ensure_dirs(out: Path):
    names = [
        "00_data_audit", "01_record_level", "02_phase_robustness", "03_ablation",
        "04_temporal", "05_lovo", "06_adaptation", "07_shiptype_specific",
        "08_shap_physical", "09_shap_stability", "10_lime", "11_cii",
        "12_optimization", "13_generalisation", "14_artifacts",
    ]
    for n in names:
        (out / n).mkdir(parents=True, exist_ok=True)


def parse_steps(s: str) -> List[str]:
    s = s.strip().lower()
    if s in {"all", "*"}:
        return list(DEFAULT_STEPS)
    vals = [x.strip() for x in s.split(",") if x.strip()]
    unknown = [x for x in vals if x not in DEFAULT_STEPS]
    if unknown:
        raise ValueError(f"Unknown steps: {unknown}. Valid: {DEFAULT_STEPS}")
    return vals


def normalize_model_name(x: str) -> str:
    n = str(x).strip().lower()
    return {
        "randomforest": "rf", "random_forest": "rf", "rf": "rf",
        "xgboost": "xgb", "xgb": "xgb",
        "lightgbm": "lgbm", "lgbm": "lgbm",
        "decisiontree": "dt", "decision_tree": "dt", "dt": "dt",
        "linearregression": "lr", "linear_regression": "lr", "lr": "lr",
        "ann": "ann", "mlp": "ann",
    }.get(n, n)


def load_locked_hyperparams(csv_path: Path, logger: logging.Logger) -> Dict[str, Dict]:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Locked hyperparameter CSV not found: {csv_path}. "
            "Copy 02_best_hyperparameters.csv beside the runner or pass --hyperparams-csv."
        )
    df = pd.read_csv(csv_path)
    required = {"model", "best_parameters_json"}
    if not required.issubset(df.columns):
        raise ValueError(f"Hyperparameter CSV must contain {sorted(required)}; got {list(df.columns)}")
    out: Dict[str, Dict] = {}
    for _, r in df.iterrows():
        name = normalize_model_name(r["model"])
        raw = r["best_parameters_json"]
        params = {} if pd.isna(raw) or str(raw).strip() in {"", "{}"} else json.loads(str(raw))
        out[name] = params
    missing = [m for m in BASE_MODELS if m not in out]
    if missing:
        raise ValueError(f"Locked parameter registry is missing: {missing}")
    logger.info("Loaded locked hyperparameters for %s", sorted(out))
    return out


def corrected_model_factory(name: str, params: Dict, seed: int, n_jobs: int):
    """Factory compatible with the user's existing official parameter CSV."""
    name = normalize_model_name(name)
    params = dict(params or {})
    if name == "lr":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median", copy=False)),
            ("scaler", StandardScaler(copy=False)),
            ("model", LinearRegression(copy_X=False, n_jobs=1)),
        ])
    if name == "ridge_interaction":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median", copy=False)),
            ("scaler", StandardScaler(copy=False)),
            ("model", Ridge(alpha=float(params.get("alpha", 1.0)), copy_X=False)),
        ])
    if name == "dt":
        return DecisionTreeRegressor(
            random_state=seed,
            max_depth=None if params.get("max_depth") in {None, -1} else int(params.get("max_depth", 18)),
            min_samples_leaf=int(params.get("min_samples_leaf", 7)),
            min_samples_split=int(params.get("min_samples_split", 26)),
            max_features=params.get("max_features", None),
            ccp_alpha=float(params.get("ccp_alpha", 0.0)),
        )
    if name == "rf":
        bootstrap = bool(params.get("bootstrap", True))
        kw = dict(
            random_state=seed,
            n_jobs=n_jobs,
            n_estimators=int(params.get("n_estimators", 350)),
            max_depth=None if params.get("max_depth") in {None, -1} else int(params.get("max_depth", 24)),
            min_samples_leaf=int(params.get("min_samples_leaf", 2)),
            min_samples_split=int(params.get("min_samples_split", 4)),
            max_features=params.get("max_features", 0.7344727825741865),
            bootstrap=bootstrap,
        )
        if bootstrap:
            kw["max_samples"] = params.get("max_samples", None)
        return RandomForestRegressor(**kw)
    if name == "xgb":
        return XGBRegressor(
            random_state=seed,
            n_jobs=n_jobs,
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
    if name == "lgbm":
        # LGBM accepts colsample_bytree/subsample aliases; using them preserves the
        # official registry exactly instead of silently dropping those parameters.
        return LGBMRegressor(
            random_state=seed,
            n_jobs=n_jobs,
            verbosity=-1,
            n_estimators=int(params.get("n_estimators", 5000)),
            num_leaves=int(params.get("num_leaves", 125)),
            max_depth=int(params.get("max_depth", -1)),
            learning_rate=float(params.get("learning_rate", 0.0197)),
            colsample_bytree=float(params.get("colsample_bytree", params.get("feature_fraction", 0.885))),
            subsample=float(params.get("subsample", params.get("bagging_fraction", 0.827))),
            subsample_freq=1,
            min_child_samples=int(params.get("min_child_samples", 22)),
            reg_alpha=float(params.get("reg_alpha", 0.0)),
            reg_lambda=float(params.get("reg_lambda", 0.0)),
        )
    if name == "ann":
        hidden = params.get("hidden_layer_sizes", [256, 128, 64])
        if isinstance(hidden, str):
            hidden = [int(v) for v in hidden.replace("x", ",").split(",") if v.strip()]
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median", copy=False)),
            ("scaler", StandardScaler(copy=False)),
            ("model", MLPRegressor(
                random_state=seed,
                hidden_layer_sizes=tuple(hidden),
                activation=str(params.get("activation", "relu")),
                solver="adam",
                alpha=float(params.get("alpha", 1e-4)),
                learning_rate_init=float(params.get("learning_rate_init", 1e-3)),
                batch_size=int(params.get("batch_size", 1024)),
                beta_1=float(params.get("beta_1", 0.9)),
                beta_2=float(params.get("beta_2", 0.999)),
                max_iter=int(params.get("max_iter", 400)),
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,
            )),
        ])
    raise ValueError(name)


# Ensure core helpers that instantiate models also use the corrected factory.
core.model_factory = corrected_model_factory


def interaction_feature_matrix(X: pd.DataFrame) -> pd.DataFrame:
    z = X.copy()
    z["speed_x_wind"] = z["speed_kn"] * z["rel_wind_speed_kn"]
    z["speed_x_wave"] = z["speed_kn"] * z["wave_height_m"]
    z["speed_x_draught"] = z["speed_kn"] * z["draught_m"]
    z["speed_x_trim"] = z["speed_kn"] * z["trim_m"]
    z["draught_x_trim"] = z["draught_m"] * z["trim_m"]
    return z


def ridge_feature_matrix(X: pd.DataFrame) -> pd.DataFrame:
    z = interaction_feature_matrix(X)
    z["speed_sq"] = z["speed_kn"] ** 2
    z["speed_cu"] = z["speed_kn"] ** 3
    return z


def fit_predict(name: str, params: Dict, Xtr: pd.DataFrame, ytr: np.ndarray,
                Xte: pd.DataFrame, seed: int, n_jobs: int):
    m = corrected_model_factory(name, params, seed, n_jobs)
    if name == "ridge_interaction":
        m.fit(ridge_feature_matrix(Xtr), ytr)
        p = m.predict(ridge_feature_matrix(Xte))
    else:
        m.fit(Xtr, ytr)
        p = m.predict(Xte)
    return m, np.asarray(p)


def predict_named(model, name: str, X: pd.DataFrame) -> np.ndarray:
    if name == "ridge_interaction":
        return np.asarray(model.predict(ridge_feature_matrix(X)))
    return np.asarray(model.predict(X))


def choose_rows_by_groups(raw_train: pd.DataFrame, max_n: int, seed: int) -> np.ndarray:
    """Select complete trajectory groups until roughly max_n rows are retained."""
    if len(raw_train) <= max_n:
        return np.arange(len(raw_train))
    rng = np.random.default_rng(seed)
    sizes = raw_train.groupby("trajectory_group").size()
    groups = sizes.index.to_numpy().copy()
    rng.shuffle(groups)
    chosen, total = [], 0
    for g in groups:
        chosen.append(g)
        total += int(sizes.loc[g])
        if total >= max_n:
            break
    mask = raw_train["trajectory_group"].isin(chosen).to_numpy()
    idx = np.flatnonzero(mask)
    if len(idx) > max_n * 1.25:
        idx = rng.choice(idx, size=max_n, replace=False)
    return np.sort(idx)


def tune_ridge_alpha(X_train: pd.DataFrame, y_train: np.ndarray, raw_train: pd.DataFrame,
                     args, logger: logging.Logger) -> Dict:
    idx = choose_rows_by_groups(raw_train, min(args.ridge_cv_n, len(raw_train)), args.seed + 91)
    Xs = ridge_feature_matrix(X_train.iloc[idx].reset_index(drop=True))
    ys = y_train[idx]
    gs = raw_train.iloc[idx]["trajectory_group"].astype(str).reset_index(drop=True)
    folds = core.make_group_folds(gs, min(args.cv_folds, gs.nunique()))
    alphas = np.logspace(-4, 3, args.ridge_alpha_grid)
    rows = []
    for alpha in alphas:
        m = corrected_model_factory("ridge_interaction", {"alpha": alpha}, args.seed, args.n_jobs)
        vals = []
        for tr, va in folds:
            mc = clone(m)
            mc.fit(Xs.iloc[tr], ys[tr])
            pred = mc.predict(Xs.iloc[va])
            vals.append(math.sqrt(mean_squared_error(ys[va], pred)))
        rows.append((float(alpha), float(np.mean(vals)), float(np.std(vals))))
    tab = pd.DataFrame(rows, columns=["alpha", "CV_RMSE_mean", "CV_RMSE_std"]).sort_values("CV_RMSE_mean")
    tab.to_csv(Path(args.output_dir) / "01_record_level" / "Table_R1_ridge_alpha_cv.csv", index=False)
    best = float(tab.iloc[0]["alpha"])
    logger.info("Ridge-Interaction selected alpha=%g, grouped CV RMSE=%.6f", best, tab.iloc[0]["CV_RMSE_mean"])
    return {"alpha": best}


def metric_plus_dispersion(y: np.ndarray, pred: np.ndarray) -> Dict:
    d = core.metric_dict(y, pred)
    sd = float(np.std(y, ddof=1)) if len(y) > 1 else np.nan
    d["target_std"] = sd
    d["NRMSE_by_target_sd"] = d["RMSE"] / sd if np.isfinite(sd) and sd > 0 else np.nan
    return d


def save_model(path: Path, model):
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def get_or_fit_record_model(name: str, params: Dict, X_train, y_train, X_test,
                            args, logger, resume=True):
    path = Path(args.output_dir) / "14_artifacts" / f"{name}_record_split.joblib"
    pred_path = Path(args.output_dir) / "14_artifacts" / f"{name}_record_test_prediction.npy"
    reuse = bool(resume or getattr(args, "_artifact_reuse", False))
    if reuse and path.exists() and pred_path.exists():
        logger.info("Artifact reuse: loading %s record-split artifact", name)
        return joblib.load(path), np.load(pred_path)
    logger.info("Fitting record-split model: %s | train=%d test=%d", name, len(X_train), len(X_test))
    t0 = time.time()
    model, pred = fit_predict(name, params, X_train, y_train, X_test, args.seed, args.n_jobs)
    logger.info("Finished fit/predict: %s in %.1fs; saving artifact", name, time.time()-t0)
    save_model(path, model)
    np.save(pred_path, pred)
    logger.info("Saved record-split artifact: %s", name)
    return model, pred


def save_record_split_indices(out: Path, train_idx: np.ndarray, test_idx: np.ndarray):
    np.savez_compressed(out / "14_artifacts" / "record_split_indices.npz", train_idx=train_idx, test_idx=test_idx)


def load_or_create_record_split(raw: pd.DataFrame, args, logger):
    p = Path(args.output_dir) / "14_artifacts" / "record_split_indices.npz"
    if args.resume and p.exists():
        z = np.load(p)
        tr, te = z["train_idx"], z["test_idx"]
        if len(tr) + len(te) == len(raw):
            logger.info("Resume: reusing saved record-level split (%d/%d)", len(tr), len(te))
            return tr, te
        logger.warning("Saved split size differs from current cruise cohort; rebuilding split")
    tr, te = core.stratified_record_split(raw, args.test_size, args.seed)
    save_record_split_indices(Path(args.output_dir), tr, te)
    return tr, te


def run_data_outputs(bundle_all, args, logger):
    out = Path(args.output_dir) / "00_data_audit"
    raw = bundle_all.raw
    counts = raw.groupby(["ship_type", "vessel_id", "phase"]).size().rename("n").reset_index()
    counts.to_csv(out / "Table_D1_rows_by_vessel_phase.csv", index=False)
    phase_counts = raw.groupby("phase").size().rename("n").reset_index()
    phase_counts["pct"] = 100 * phase_counts["n"] / phase_counts["n"].sum()
    phase_counts.to_csv(out / "Table_D2_phase_counts.csv", index=False)
    retention = pd.DataFrame([
        {"phase":"cruise", "Fixed27_input":529512, "Fixed28_deleted":39147, "Fixed29_deleted":420,
         "Fixed30_empirical_rule_deleted":0, "Fixed31_deleted":325, "Fixed31_final":489620},
        {"phase":"maneuver", "Fixed27_input":19334, "Fixed28_deleted":3479, "Fixed29_deleted":0,
         "Fixed30_empirical_rule_deleted":0, "Fixed31_deleted":4, "Fixed31_final":15851},
        {"phase":"anchor_berth", "Fixed27_input":87644, "Fixed28_deleted":45751, "Fixed29_deleted":3,
         "Fixed30_empirical_rule_deleted":0, "Fixed31_deleted":307, "Fixed31_final":41583},
    ])
    retention.to_csv(out / "Table_D3_fixed27_to_fixed31_flow.csv", index=False)
    vars_ = ["target","speed_kn","draught_m","rudder_deg","trim_m","rel_wind_speed_kn",
             "wave_height_m","wave_period_s","mslp_hpa","sst_c"]
    rows=[]
    scopes=[("Overall", raw)] + [(st,g) for st,g in raw.groupby("ship_type")]
    for scope,g in scopes:
        for v in vars_:
            rows.append({"scope":scope,"variable":v,"n":int(g[v].notna().sum()),
                         "mean":float(g[v].mean()),"std":float(g[v].std(ddof=1)),
                         "median":float(g[v].median()),"q05":float(g[v].quantile(.05)),"q95":float(g[v].quantile(.95))})
    pd.DataFrame(rows).to_csv(out / "Table_D4_descriptive_stats.csv", index=False)
    logger.info("Data audit outputs written")


def run_base_models(X_train, y_train, X_test, y_test, raw_test, params, args, logger):
    out = Path(args.output_dir) / "01_record_level"
    fitted, preds, rows, vessel_frames = {}, {}, [], []
    ridge_params = params.get("ridge_interaction") or tune_ridge_alpha(X_train, y_train, args._raw_train, args, logger)
    params["ridge_interaction"] = ridge_params
    for name in ALL_MODELS:
        m,p = get_or_fit_record_model(name, params.get(name, {}), X_train, y_train, X_test, args, logger, args.resume)
        fitted[name], preds[name] = m,p
        r={"model":name, **metric_plus_dispersion(y_test,p)}
        rows.append(r)
        vm=core.per_group_metrics(raw_test,p,["vessel_id","ship_type"])
        vm.insert(0,"model",name)
        vessel_frames.append(vm)
    rec=pd.DataFrame(rows).sort_values("RMSE")
    rec.to_csv(out / "Table_B1_record_level_model_comparison.csv",index=False)
    vmet=pd.concat(vessel_frames,ignore_index=True)
    vmet.to_csv(out / "Table_B2_record_level_by_vessel.csv",index=False)
    macro=[]
    for name,g in vmet.groupby("model"):
        macro.append({"model":name, **core.macro_vessel_summary(g)})
    pd.DataFrame(macro).to_csv(out / "Table_B3_equal_vessel_macro_metrics.csv",index=False)
    rec[rec.model.isin(["lr","ridge_interaction"])].to_csv(out / "Table_B4_LR_vs_RidgeInteraction.csv",index=False)
    return fitted,preds,rec,vmet,params



def ensure_ridge_params(params, X_train, y_train, raw_train, args, logger):
    if params.get("ridge_interaction"):
        return params
    p = Path(args.output_dir) / "14_artifacts" / "ridge_interaction_selected_params.json"
    if p.exists():
        try:
            params["ridge_interaction"] = json.loads(p.read_text(encoding="utf-8"))
            logger.info("Resume: reusing saved Ridge-Interaction params: %s", params["ridge_interaction"])
            return params
        except Exception as e:
            logger.warning("Could not reuse saved Ridge params (%s); retuning once", e)
    logger.info("Selecting Ridge-Interaction alpha (one-time; base hyperparameters remain locked)")
    params["ridge_interaction"] = tune_ridge_alpha(X_train, y_train, raw_train, args, logger)
    p.write_text(json.dumps(params["ridge_interaction"], indent=2), encoding="utf-8")
    gc.collect()
    return params


def run_base_models_disk_safe(X_train, y_train, X_test, y_test, raw_test, params, args, logger):
    """Run the seven-model record split without retaining all fitted models in RAM."""
    out = Path(args.output_dir) / "01_record_level"
    params = ensure_ridge_params(params, X_train, y_train, args._raw_train, args, logger)
    rows, vessel_frames = [], []
    for name in ALL_MODELS:
        logger.info("BASE sequential model %s/%s: %s", ALL_MODELS.index(name)+1, len(ALL_MODELS), name)
        model = pred = None
        try:
            model, pred = get_or_fit_record_model(name, params.get(name, {}), X_train, y_train, X_test, args, logger, args.resume)
            rows.append({"model": name, **metric_plus_dispersion(y_test, pred)})
            vm = core.per_group_metrics(raw_test, pred, ["vessel_id", "ship_type"])
            vm.insert(0, "model", name)
            vessel_frames.append(vm)
        finally:
            del model, pred
            gc.collect()
    rec = pd.DataFrame(rows).sort_values("RMSE")
    rec.to_csv(out / "Table_B1_record_level_model_comparison.csv", index=False)
    vmet = pd.concat(vessel_frames, ignore_index=True)
    vmet.to_csv(out / "Table_B2_record_level_by_vessel.csv", index=False)
    macro=[]
    for name,g in vmet.groupby("model"):
        macro.append({"model":name, **core.macro_vessel_summary(g)})
    pd.DataFrame(macro).to_csv(out / "Table_B3_equal_vessel_macro_metrics.csv", index=False)
    rec[rec.model.isin(["lr","ridge_interaction"])].to_csv(out / "Table_B4_LR_vs_RidgeInteraction.csv", index=False)
    logger.info("BASE sequential comparison finished")
    return rec, vmet, params


def run_phase_robustness_disk_safe(phase_eval, X_train, y_train, X_test, y_test, raw_test, params, args, logger):
    """Evaluate operating-regime robustness with RF-safe recovery.

    V3 never deserializes the large RandomForest record-split artifact in this step.
    Cruise predictions are reused from the saved .npy artifact.  For non-cruise
    predictions, RF is refitted once from the locked hyperparameters in the fresh
    phase process, then its manoeuvre/anchor predictions are cached.  On later
    -Resume runs those cached predictions are reused and RF is not refitted.
    """
    out=Path(args.output_dir)/"02_phase_robustness"
    art=Path(args.output_dir)/"14_artifacts"
    out.mkdir(parents=True, exist_ok=True)
    params = ensure_ridge_params(params, X_train, y_train, args._raw_train, args, logger)
    rows=[]; vessel_rows=[]

    for name in ALL_MODELS:
        logger.info("PHASE V3 model %s/%s: %s", ALL_MODELS.index(name)+1, len(ALL_MODELS), name)

        # Cruise evaluation never needs a fitted model: reuse the exact common-test prediction.
        cruise_pred_path = art / f"{name}_record_test_prediction.npy"
        if not cruise_pred_path.exists():
            raise FileNotFoundError(f"Missing base prediction artifact: {cruise_pred_path}. Run -Steps base once.")
        cruise_pred = np.load(cruise_pred_path)
        rows.append({"model":name,"phase":"cruise","evaluation_role":"primary_holdout", **metric_plus_dispersion(y_test,cruise_pred)})
        vm=core.per_group_metrics(raw_test,cruise_pred,["vessel_id","ship_type"]); vm["model"]=name; vm["phase"]="cruise"; vessel_rows.append(vm)
        del cruise_pred

        required=[]
        for phase,role in [("maneuver","robustness_domain_shift"),("anchor_berth","descriptive_OOD")]:
            if phase not in phase_eval:
                continue
            pred_cache = out / f"prediction_{name}_{phase}.npy"
            if args.resume and pred_cache.exists():
                pp=np.load(pred_cache)
                rp,_=phase_eval[phase]
                logger.info("Phase prediction reuse: %s / %s", name, phase)
                rows.append({"model":name,"phase":phase,"evaluation_role":role, **metric_plus_dispersion(rp.target.to_numpy(float),pp)})
                vv=core.per_group_metrics(rp,pp,["vessel_id","ship_type"]); vv["model"]=name; vv["phase"]=phase; vessel_rows.append(vv)
                del pp
            else:
                required.append((phase,role,pred_cache))

        if not required:
            gc.collect()
            continue

        model=None
        try:
            if name == "rf":
                # Loading the saved RF artifact repeatedly caused native Windows termination
                # on the user's machine.  Refit once from the already locked parameters.
                logger.warning("RF phase-safe mode: NOT loading rf_record_split.joblib; refitting RF once from locked params for non-cruise predictions")
                t0=time.time()
                model=corrected_model_factory("rf", params["rf"], args.seed, max(1, min(args.n_jobs, 1)))
                model.fit(X_train, y_train)
                logger.info("RF phase-safe refit finished in %.1fs", time.time()-t0)
            else:
                model_path=art/f"{name}_record_split.joblib"
                if not model_path.exists():
                    raise FileNotFoundError(f"Missing model artifact: {model_path}. Run -Steps base once.")
                logger.info("Loading %s artifact for non-cruise phase prediction", name)
                model=joblib.load(model_path, mmap_mode="r")

            for phase,role,pred_cache in required:
                rp,Xp=phase_eval[phase]
                pp=predict_named(model,name,Xp)
                np.save(pred_cache,pp)
                logger.info("Saved phase prediction cache: %s", pred_cache.name)
                rows.append({"model":name,"phase":phase,"evaluation_role":role, **metric_plus_dispersion(rp.target.to_numpy(float),pp)})
                vv=core.per_group_metrics(rp,pp,["vessel_id","ship_type"]); vv["model"]=name; vv["phase"]=phase; vessel_rows.append(vv)
                del pp
        finally:
            del model
            gc.collect()

    tab=pd.DataFrame(rows);tab.to_csv(out/"Table_P1_operating_regime_robustness.csv",index=False)
    vv=pd.concat(vessel_rows,ignore_index=True);vv.to_csv(out/"Table_P2_operating_regime_by_vessel.csv",index=False)
    macro=[]
    for (m,ph),g in vv.groupby(["model","phase"]): macro.append({"model":m,"phase":ph,**core.macro_vessel_summary(g)})
    pd.DataFrame(macro).to_csv(out/"Table_P3_operating_regime_equal_vessel_macro.csv",index=False)
    logger.info("PHASE V3 completed; cached non-cruise predictions can be reused on later -Resume runs")
    return tab


def collect_cii_predictions_disk_safe(X_train, y_train, X_test, params, args, logger):
    """Get record-test predictions for all models without retaining fitted models."""
    params = ensure_ridge_params(params, X_train, y_train, args._raw_train, args, logger)
    preds={}
    art=Path(args.output_dir)/"14_artifacts"
    for name in ALL_MODELS:
        pp=art/f"{name}_record_test_prediction.npy"
        if (args.resume or getattr(args, "_artifact_reuse", False)) and pp.exists():
            logger.info("CII artifact reuse: loading prediction array for %s",name)
            preds[name]=np.load(pp)
            continue
        m,p=get_or_fit_record_model(name,params.get(name,{}),X_train,y_train,X_test,args,logger,args.resume)
        preds[name]=p
        del m
        gc.collect()
    return preds, params

def run_phase_robustness(bundle_all, fitted, params, cruise_raw_test, cruise_y_test, cruise_preds, args, logger):
    out=Path(args.output_dir)/"02_phase_robustness"
    raw_all=bundle_all.raw.reset_index(drop=True)
    X_all=bundle_all.feature_df.reset_index(drop=True)
    rows=[]; vessel_rows=[]
    for name in ALL_MODELS:
        # cruise is the common outer test set
        p=cruise_preds[name]
        r={"model":name,"phase":"cruise","evaluation_role":"primary_holdout", **metric_plus_dispersion(cruise_y_test,p)}
        rows.append(r)
        vm=core.per_group_metrics(cruise_raw_test,p,["vessel_id","ship_type"]); vm["model"]=name; vm["phase"]="cruise"; vessel_rows.append(vm)
        for phase,role in [("maneuver","robustness_domain_shift"),("anchor_berth","descriptive_OOD")]:
            mask=(raw_all.phase==phase).to_numpy()
            if not mask.any(): continue
            Xp=X_all.loc[mask].reset_index(drop=True); rp=raw_all.loc[mask].reset_index(drop=True)
            pp=predict_named(fitted[name],name,Xp)
            rr={"model":name,"phase":phase,"evaluation_role":role, **metric_plus_dispersion(rp.target.to_numpy(float),pp)}
            rows.append(rr)
            vv=core.per_group_metrics(rp,pp,["vessel_id","ship_type"]); vv["model"]=name; vv["phase"]=phase; vessel_rows.append(vv)
    tab=pd.DataFrame(rows)
    tab.to_csv(out/"Table_P1_operating_regime_robustness.csv",index=False)
    vv=pd.concat(vessel_rows,ignore_index=True)
    vv.to_csv(out/"Table_P2_operating_regime_by_vessel.csv",index=False)
    # equal-vessel boxplot data are in P2; also create a compact macro table
    macro=[]
    for (m,ph),g in vv.groupby(["model","phase"]):
        macro.append({"model":m,"phase":ph,**core.macro_vessel_summary(g)})
    pd.DataFrame(macro).to_csv(out/"Table_P3_operating_regime_equal_vessel_macro.csv",index=False)
    return tab


def cluster_bootstrap_metric_diff(vessel_ids, y, pred_a, pred_b, reps, seed):
    # positive delta means A has larger/worse RMSE than B
    ids=np.asarray(vessel_ids).astype(str); y=np.asarray(y); a=np.asarray(pred_a); b=np.asarray(pred_b)
    uniq=np.unique(ids); rng=np.random.default_rng(seed); vals=[]
    index_by={u:np.flatnonzero(ids==u) for u in uniq}
    for _ in range(reps):
        draw=rng.choice(uniq,size=len(uniq),replace=True)
        idx=np.concatenate([index_by[u] for u in draw])
        ra=math.sqrt(mean_squared_error(y[idx],a[idx])); rb=math.sqrt(mean_squared_error(y[idx],b[idx]))
        vals.append(ra-rb)
    vals=np.asarray(vals)
    return float(vals.mean()),float(np.quantile(vals,.025)),float(np.quantile(vals,.975))


def run_ablation(X_train,y_train,X_test,y_test,raw_test,params,args,logger):
    out=Path(args.output_dir)/"03_ablation"; model_name=args.ablation_model
    base_cols=list(X_train.columns)
    op=[c for c in core.OPERATIONAL_FEATURES if c in base_cols]
    wx=[c for c in core.WEATHER_FEATURES if c in base_cols]
    configs={
        "operational_core":(X_train[op],X_test[op]),
        "weather_only":(X_train[wx],X_test[wx]),
        "dynamic_physical":(X_train[base_cols],X_test[base_cols]),
        "dynamic_interaction":(interaction_feature_matrix(X_train),interaction_feature_matrix(X_test)),
    }
    preds={}; rows=[]
    for label,(tr,te) in configs.items():
        logger.info("Ablation %s | %s | %d features",label,model_name,tr.shape[1])
        m=corrected_model_factory(model_name,params[model_name],args.seed,args.n_jobs); m.fit(tr,y_train); p=m.predict(te)
        preds[label]=np.asarray(p)
        rows.append({"configuration":label,"model":model_name,"n_features":tr.shape[1],**metric_plus_dispersion(y_test,p)})
    tab=pd.DataFrame(rows)
    full=float(tab.loc[tab.configuration=="dynamic_physical","RMSE"].iloc[0])
    tab["delta_RMSE_vs_dynamic_physical"]=tab.RMSE-full
    tab["RMSE_worsening_pct_vs_dynamic_physical"]=(tab.RMSE/full-1)*100
    tab.to_csv(out/"Table_A1_multisource_feature_ablation.csv",index=False)
    boot=[]
    for i,label in enumerate(["operational_core","weather_only","dynamic_interaction"]):
        mean,lo,hi=cluster_bootstrap_metric_diff(raw_test.vessel_id,y_test,preds[label],preds["dynamic_physical"],args.bootstrap_reps,args.seed+501+i)
        boot.append({"configuration":label,"reference":"dynamic_physical","delta_RMSE_mean":mean,"CI95_low":lo,"CI95_high":hi})
    pd.DataFrame(boot).to_csv(out/"Table_A2_vessel_cluster_bootstrap_RMSE_differences.csv",index=False)
    # per-vessel paired error differences
    rows=[]
    for vessel,inds in raw_test.groupby("vessel_id").groups.items():
        ii=np.array(list(inds),dtype=int); yt=y_test[ii]
        ref=math.sqrt(mean_squared_error(yt,preds["dynamic_physical"][ii]))
        for label,p in preds.items():
            rm=math.sqrt(mean_squared_error(yt,p[ii]))
            rows.append({"vessel_id":vessel,"ship_type":raw_test.loc[ii,"ship_type"].iloc[0],"configuration":label,"RMSE":rm,"delta_RMSE_vs_dynamic_physical":rm-ref})
    pd.DataFrame(rows).to_csv(out/"Table_A3_ablation_by_vessel.csv",index=False)
    return tab


def run_temporal(model_names: List[str], X, raw, params,args,logger):
    out=Path(args.output_dir)/"04_temporal"; tr,te=core.known_vessel_temporal_split(raw,0.8)
    rows=[]; byv=[]
    for name in model_names:
        logger.info("Known-vessel temporal 80->20: %s",name)
        m,_=fit_predict(name,params[name],X.loc[tr],raw.loc[tr,"target"].to_numpy(float),X.loc[te],args.seed,args.n_jobs)
        p=predict_named(m,name,X.loc[te])
        rows.append({"model":name,**metric_plus_dispersion(raw.loc[te,"target"].to_numpy(float),p)})
        vv=core.per_group_metrics(raw.loc[te].reset_index(drop=True),p,["vessel_id","ship_type"]); vv["model"]=name; byv.append(vv)
    pd.DataFrame(rows).to_csv(out/"Table_T1_known_vessel_temporal_overall.csv",index=False)
    pd.concat(byv,ignore_index=True).to_csv(out/"Table_T2_known_vessel_temporal_by_vessel.csv",index=False)
    return pd.DataFrame(rows)


def run_lovo_models(model_names: List[str],X,raw,params,args,logger):
    out=Path(args.output_dir)/"05_lovo"; overall=[]; folds=[]
    for name in model_names:
        logger.info("Running locked-hyperparameter LOVO for %s",name)
        ov,fd,p=core.run_lovo(name,params[name],X,raw,args,logger)
        overall.append(ov); fd=fd.copy(); fd["model"]=name; folds.append(fd)
        np.save(out/f"LOVO_predictions_{name}.npy",p)
    O=pd.concat(overall,ignore_index=True); F=pd.concat(folds,ignore_index=True)
    O.to_csv(out/"Table_L1_LOVO_overall.csv",index=False); F.to_csv(out/"Table_L2_LOVO_by_vessel.csv",index=False)
    macro=[]
    for name,g in F.groupby("model"):
        macro.append({"model":name,**core.macro_vessel_summary(g)})
    pd.DataFrame(macro).to_csv(out/"Table_L3_LOVO_equal_vessel_macro.csv",index=False)
    return O


def run_adaptation_models(model_names: List[str],X,raw,params,args,logger):
    out=Path(args.output_dir)/"06_adaptation"; overall=[]; folds=[]
    for name in model_names:
        logger.info("Running new-vessel 80%% adaptation for %s",name)
        ov,fd,p=core.run_adaptation(name,params[name],X,raw,args,logger)
        overall.append(ov); fd=fd.copy(); fd["model"]=name; folds.append(fd)
        np.save(out/f"adaptation_predictions_{name}.npy",p)
    O=pd.concat(overall,ignore_index=True);F=pd.concat(folds,ignore_index=True)
    O.to_csv(out/"Table_N1_new_vessel_adaptation_overall.csv",index=False);F.to_csv(out/"Table_N2_new_vessel_adaptation_by_vessel.csv",index=False)
    return O


def paired_vessel_bootstrap_shiptype(vdf: pd.DataFrame,reps:int,seed:int):
    rng=np.random.default_rng(seed); rows=[]
    for st,g in vdf.groupby("ship_type"):
        wide=g.pivot(index="vessel_id",columns="strategy",values="RMSE").dropna()
        if not {"pooled","ship_type_specific"}.issubset(wide.columns) or wide.empty: continue
        ids=wide.index.to_numpy(); vals=[]
        for _ in range(reps):
            draw=rng.choice(ids,size=len(ids),replace=True)
            vals.append(float(np.mean(wide.loc[draw,"ship_type_specific"].to_numpy()-wide.loc[draw,"pooled"].to_numpy())))
        rows.append({"ship_type":st,"n_vessels":len(ids),"mean_vessel_RMSE_diff_specific_minus_pooled":float(np.mean(vals)),
                     "CI95_low":float(np.quantile(vals,.025)),"CI95_high":float(np.quantile(vals,.975))})
    return pd.DataFrame(rows)


def run_shiptype_specific(model_name,X_train,y_train,raw_train,X_test,y_test,raw_test,pooled_model,args,logger):
    out=Path(args.output_dir)/"07_shiptype_specific"; rows=[]; vrows=[]
    cols=[c for c in X_train.columns if not c.startswith("ship_type_")]
    pooled_pred=predict_named(pooled_model,model_name,X_test)
    for st in core.SHIP_TYPES:
        tr=(raw_train.ship_type==st).to_numpy(); te=(raw_test.ship_type==st).to_numpy()
        if tr.sum()==0 or te.sum()==0: continue
        # pooled result on identical target observations
        rows.append({"ship_type":st,"strategy":"pooled","model":model_name,**metric_plus_dispersion(y_test[te],pooled_pred[te])})
        vv=core.per_group_metrics(raw_test.loc[te].reset_index(drop=True),pooled_pred[te],["vessel_id","ship_type"]);vv["strategy"]="pooled";vrows.append(vv)
        m=corrected_model_factory(model_name,args._params[model_name],args.seed,args.n_jobs);m.fit(X_train.loc[tr,cols],y_train[tr]);p=m.predict(X_test.loc[te,cols])
        rows.append({"ship_type":st,"strategy":"ship_type_specific","model":model_name,**metric_plus_dispersion(y_test[te],p)})
        vv=core.per_group_metrics(raw_test.loc[te].reset_index(drop=True),p,["vessel_id","ship_type"]);vv["strategy"]="ship_type_specific";vrows.append(vv)
        save_model(out/f"{model_name}_{st}_specific.joblib",m)
    T=pd.DataFrame(rows);V=pd.concat(vrows,ignore_index=True)
    T.to_csv(out/"Table_S1_pooled_vs_shiptype_specific.csv",index=False);V.to_csv(out/"Table_S2_paired_by_vessel.csv",index=False)
    paired_vessel_bootstrap_shiptype(V,args.bootstrap_reps,args.seed+701).to_csv(out/"Table_S3_paired_vessel_bootstrap.csv",index=False)
    return T


def standardized_lr_cluster_ci(X_train,y_train,raw_train,out_path,logger):
    if sm is None:
        raise ImportError("statsmodels is required for cluster-robust LR coefficients")
        return pd.DataFrame()
    mu=X_train.mean();sd=X_train.std(ddof=0).replace(0,1);Z=(X_train-mu)/sd;Zc=sm.add_constant(Z,has_constant="add")
    try:
        fit=sm.OLS(y_train,Zc).fit(cov_type="cluster",cov_kwds={"groups":raw_train.vessel_id.to_numpy()});ci=fit.conf_int(.05)
        rows=[]
        for f in X_train.columns:
            rows.append({"feature":f,"standardized_LR_coefficient":float(fit.params[f]),"CI95_low":float(ci.loc[f,0]),"CI95_high":float(ci.loc[f,1]),"p_value_cluster_robust":float(fit.pvalues[f])})
        df=pd.DataFrame(rows);df.to_csv(out_path,index=False);return df
    except Exception as e:
        logger.warning("Cluster-robust LR coefficient estimation failed: %s",e);return pd.DataFrame()


def standardized_ridge_coefficients(ridge_model,X_train,out_path):
    Xr=ridge_feature_matrix(X_train);pipe=ridge_model
    coef=pipe.named_steps["model"].coef_
    df=pd.DataFrame({"feature":Xr.columns,"standardized_Ridge_coefficient":coef})
    df.to_csv(out_path,index=False);return df


def shap_direction_table(Xsh,sv):
    rows=[]
    for f in ["speed_kn","wave_height_m","draught_m","trim_m","rel_wind_speed_kn"]:
        j=Xsh.columns.get_loc(f);rho=float(spearmanr(Xsh[f],sv[:,j],nan_policy="omit").statistic)
        rows.append({"feature":f,"SHAP_value_spearman":rho,"SHAP_direction":"positive" if rho>.1 else "negative" if rho<-.1 else "mixed/weak",
                     "physical_expectation":core.PHYSICAL_EXPECTATION.get(f,"context_dependent"),"assessment":core.physical_status_from_spearman(f,rho)})
    j=Xsh.columns.get_loc("rudder_deg");rho=float(spearmanr(np.abs(Xsh.rudder_deg),sv[:,j],nan_policy="omit").statistic)
    rows.append({"feature":"rudder_deg","SHAP_value_spearman":rho,"SHAP_direction":"absolute-angle audit","physical_expectation":"larger |rudder| generally higher demand","assessment":"context-dependent"})
    for f in ["heading_direction","relative_wind_direction","relative_wave_direction"]:
        rows.append({"feature":f,"SHAP_value_spearman":np.nan,"SHAP_direction":"cyclic","physical_expectation":"cyclic/context-dependent","assessment":"Not directionally testable / context-dependent"})
    return pd.DataFrame(rows)


def run_shap_physical(explain_name,model,X_train,y_train,raw_train,X_test,y_test,raw_test,ridge_model,args,logger):
    out=Path(args.output_dir)/"08_shap_physical"
    idx=core.sample_indices(len(X_test),min(args.shap_sample_n,len(X_test)),args.seed+800)
    Xsh=X_test.iloc[idx].reset_index(drop=True);rawsh=raw_test.iloc[idx].reset_index(drop=True)
    logger.info("SHAP on held-out %s rows=%d",explain_name,len(Xsh))
    sv=core.get_tree_shap(model,Xsh)
    np.save(out/"heldout_SHAP_values.npy",sv);Xsh.to_csv(out/"heldout_SHAP_sample_features.csv",index=False)
    imp=np.mean(np.abs(sv),axis=0);glob=pd.DataFrame({"feature":Xsh.columns,"mean_abs_SHAP":imp}).sort_values("mean_abs_SHAP",ascending=False)
    glob["normalized_importance"]=glob.mean_abs_SHAP/glob.mean_abs_SHAP.sum();glob.to_csv(out/"Table_H1_global_SHAP_importance.csv",index=False)
    gm={"heading_sin":"heading_direction","heading_cos":"heading_direction","rel_wind_sin":"relative_wind_direction","rel_wind_cos":"relative_wind_direction","rel_wave_sin":"relative_wave_direction","rel_wave_cos":"relative_wave_direction","ship_type_bulk":"ship_type","ship_type_container":"ship_type"}
    g=glob.copy();g["grouped_feature"]=g.feature.map(gm).fillna(g.feature);grp=g.groupby("grouped_feature",as_index=False).mean_abs_SHAP.sum().sort_values("mean_abs_SHAP",ascending=False);grp["normalized_importance"]=grp.mean_abs_SHAP/grp.mean_abs_SHAP.sum();grp.to_csv(out/"Table_H2_grouped_SHAP_importance.csv",index=False)
    direction=shap_direction_table(Xsh,sv);direction.to_csv(out/"Table_H3_SHAP_physical_direction.csv",index=False)
    # LR and Ridge coefficient matrix
    lr=standardized_lr_cluster_ci(X_train,y_train,raw_train,out/"Table_H4_LR_coefficients_cluster_CI.csv",logger)
    rr=standardized_ridge_coefficients(ridge_model,X_train,out/"Table_H5_RidgeInteraction_coefficients.csv")
    matrix=direction.merge(lr,on="feature",how="left").merge(rr,on="feature",how="left")
    if not matrix.empty:
        matrix["LR_direction_95CI"]=np.where(matrix.CI95_low>0,"positive",np.where(matrix.CI95_high<0,"negative","uncertain"))
    matrix.to_csv(out/"Table_H6_LR_Ridge_SHAP_physical_matrix.csv",index=False)
    # Speed dependence + polynomial comparison
    j=Xsh.columns.get_loc("speed_kn");core.save_scatter_dependency(Xsh.speed_kn,sv[:,j],out/"Figure_H1_speed_SHAP_dependence.png","Speed over ground (kn)")
    bins=pd.qcut(Xsh.speed_kn,q=min(30,max(8,Xsh.speed_kn.nunique()//20)),duplicates="drop")
    sb=pd.DataFrame({"speed":Xsh.speed_kn,"shap":sv[:,j],"bin":bins}).groupby("bin",observed=True).agg(speed_mean=("speed","mean"),SHAP_mean=("shap","mean"),n=("shap","size")).reset_index(drop=True)
    sb.to_csv(out/"Table_H7_speed_SHAP_binned_density.csv",index=False);core.fit_poly_diagnostics(sb.speed_mean.to_numpy(),sb.SHAP_mean.to_numpy(),3).to_csv(out/"Table_H8_speed_linear_quadratic_cubic_diagnostics.csv",index=False)
    # Wind sectors
    j=Xsh.columns.get_loc("rel_wind_speed_kn");wd=pd.DataFrame({"rel_wind_speed_kn":Xsh.rel_wind_speed_kn,"wind_SHAP":sv[:,j],"rel_wind_dir_deg":rawsh.rel_wind_dir_deg,"speed_kn":Xsh.speed_kn});wd["wind_sector"]=core.semantic_wind_sector(wd.rel_wind_dir_deg,args.wind_zero_is);wd.to_csv(out/"Table_H9_wind_SHAP_by_sector_raw.csv",index=False)
    ws=[]
    for sec,gg in wd.groupby("wind_sector"):
        ws.append({"wind_sector":sec,"n":len(gg),"wind_speed_SHAP_spearman":float(spearmanr(gg.rel_wind_speed_kn,gg.wind_SHAP,nan_policy="omit").statistic)})
    pd.DataFrame(ws).to_csv(out/"Table_H10_wind_SHAP_sector_summary.csv",index=False);core.save_scatter_dependency(wd.rel_wind_speed_kn,wd.wind_SHAP,out/"Figure_H2_wind_SHAP_dependence.png","Relative wind speed (kn)")
    # density-aware 2D PDP
    pdp=core.partial_dependence_2d_density_aware(model,Xsh,rawsh,"speed_kn","rel_wind_speed_kn",args.pdp_grid_n,args.pdp_support_min,args.seed+811);pdp.to_csv(out/"Table_H11_speed_wind_density_aware_2D_PDP.csv",index=False);core.plot_density_pdp(pdp,out/"Figure_H3_speed_wind_density_aware_2D_PDP.png")
    # local physical response audit
    audit=core.local_response_audit(model,raw_train,raw_test,list(X_train.columns),args.local_audit_n,args.seed+822);audit.to_csv(out/"Table_H12_local_physical_response_audit.csv",index=False)
    return glob,grp,direction,sv,Xsh,rawsh


def rank_series_from_importance(df: pd.DataFrame) -> pd.Series:
    s=df.set_index("feature")["mean_abs_SHAP"].sort_values(ascending=False);return s.rank(ascending=False,method="average")


def run_shap_stability(explain_name,params,X_train,y_train,raw_train,Xsh,reference_global,args,logger):
    out=Path(args.output_dir)/"09_shap_stability";ref_rank=rank_series_from_importance(reference_global);ref_top=set(reference_global.head(args.shap_stability_topk).feature)
    rows=[];details=[]
    rng=np.random.default_rng(args.seed+900)
    groups=raw_train.trajectory_group.astype(str).unique()
    for rep in range(1,args.shap_stability_repeats+1):
        chosen=rng.choice(groups,size=max(2,int(len(groups)*args.shap_stability_group_fraction)),replace=False)
        idx=np.flatnonzero(raw_train.trajectory_group.astype(str).isin(chosen).to_numpy())
        if len(idx)>args.shap_stability_train_max:
            idx=rng.choice(idx,size=args.shap_stability_train_max,replace=False)
        logger.info("SHAP stability rep %d/%d train=%d",rep,args.shap_stability_repeats,len(idx))
        m=corrected_model_factory(explain_name,params[explain_name],args.seed+rep,args.n_jobs);m.fit(X_train.iloc[idx],y_train[idx]);sv=core.get_tree_shap(m,Xsh);imp=np.mean(np.abs(sv),axis=0);d=pd.DataFrame({"feature":Xsh.columns,"mean_abs_SHAP":imp}).sort_values("mean_abs_SHAP",ascending=False);rank=rank_series_from_importance(d).reindex(ref_rank.index)
        rho=float(spearmanr(ref_rank.values,rank.values,nan_policy="omit").statistic);top=set(d.head(args.shap_stability_topk).feature);over=len(ref_top&top)/max(1,len(ref_top|top))
        rows.append({"repeat":rep,"train_n":len(idx),"rank_spearman_vs_reference":rho,"topk_jaccard_vs_reference":over,"topk_overlap_count":len(ref_top&top)})
        d["repeat"]=rep;details.append(d)
    pd.DataFrame(rows).to_csv(out/"Table_ST1_SHAP_stability_summary.csv",index=False);pd.concat(details,ignore_index=True).to_csv(out/"Table_ST2_SHAP_stability_importance_by_repeat.csv",index=False)


def select_lime_cases(raw_test,y_test,pred):
    err=np.abs(y_test-pred);rows=[]
    for st in core.SHIP_TYPES:
        inds=np.flatnonzero((raw_test.ship_type==st).to_numpy())
        if len(inds)==0: continue
        e=err[inds]
        for label,q in [("low_error",.10),("median_error",.50),("high_error",.90)]:
            target=np.quantile(e,q);ii=inds[np.argmin(np.abs(e-target))];rows.append((st,label,int(ii)))
    return rows


def run_lime_validation(explain_name,model,X_train,y_train,X_test,y_test,raw_test,record_pred,heldout_sv,args,logger):
    out=Path(args.output_dir)/"10_lime"
    if LimeTabularExplainer is None:
        raise ImportError("The optional LIME analysis requires the 'lime' package")
    rng=np.random.default_rng(args.seed+1000);bg_idx=rng.choice(len(X_train),size=min(args.lime_background_n,len(X_train)),replace=False);bg=X_train.iloc[bg_idx].to_numpy(float)
    cases=select_lime_cases(raw_test,y_test,record_pred);rows=[];weight_rows=[]
    # SHAP for the exact selected cases so comparison is not based on a different subsample
    case_idx=[c[2] for c in cases];case_X=X_test.iloc[case_idx];case_sv=core.get_tree_shap(model,case_X)
    predict_fn=lambda arr: model.predict(pd.DataFrame(arr,columns=X_train.columns))
    for case_pos,(st,label,ii) in enumerate(cases):
        x=X_test.iloc[ii].to_numpy(float);sh=case_sv[case_pos];sh_order=np.argsort(-np.abs(sh));sh_top=set(sh_order[:args.lime_top_features])
        for s in range(args.lime_seeds):
            expl=LimeTabularExplainer(bg,feature_names=list(X_train.columns),mode="regression",discretize_continuous=False,random_state=args.seed+1100+case_pos*100+s)
            exp=expl.explain_instance(x,predict_fn,num_features=min(args.lime_top_features,X_train.shape[1]),num_samples=args.lime_num_samples)
            lw=dict(exp.local_exp.get(1,exp.local_exp.get(0,[])));lime_top=set(lw.keys());common=lime_top&sh_top
            sign_ag=[]
            for j in common:
                sign_ag.append(np.sign(lw[j])==np.sign(sh[j]))
                weight_rows.append({"case_id":case_pos,"ship_type":st,"case_type":label,"seed":s,"feature":X_train.columns[j],"LIME_weight":lw[j],"SHAP_value":sh[j],"sign_agreement":bool(np.sign(lw[j])==np.sign(sh[j]))})
            rows.append({"case_id":case_pos,"vessel_id":raw_test.iloc[ii].vessel_id,"ship_type":st,"case_type":label,"test_row_index":ii,"abs_prediction_error":float(abs(y_test[ii]-record_pred[ii])),"seed":s,"local_fidelity_R2":float(exp.score),"topk_overlap_count":len(common),"topk_jaccard":len(common)/max(1,len(lime_top|sh_top)),"sign_agreement_rate_on_overlap":float(np.mean(sign_ag)) if sign_ag else np.nan})
    pd.DataFrame(rows).to_csv(out/"Table_LIME1_case_fidelity_stability_SHAP_agreement.csv",index=False);pd.DataFrame(weight_rows).to_csv(out/"Table_LIME2_feature_weights_vs_SHAP.csv",index=False)


def run_cii_models(raw_test,preds,args,logger):
    out=Path(args.output_dir)/"11_cii";summ=[]
    for name,p in preds.items():
        try:
            v=core.compute_cii_by_vessel(raw_test,p,args.co2_factor);v.insert(0,"model",name);v.to_csv(out/f"Table_C1_CII_proxy_by_vessel_{name}.csv",index=False);summ.append({"model":name,**core.cii_summary(v,raw_test,p)})
        except Exception as e:
            logger.warning("CII failed for %s: %s",name,e)
    S=pd.DataFrame(summ);S.to_csv(out/"Table_C2_model_choice_CII_proxy_summary.csv",index=False);return S


def run_speed_scenarios(model,raw_train,raw_test,feature_cols,args,logger):
    out=Path(args.output_dir)/"11_cii";v,s=core.speed_reduction_scenarios(model,raw_train,raw_test,feature_cols,[.05,.10,.15],args.co2_factor);v.to_csv(out/"Table_C3_speed_reduction_by_vessel.csv",index=False);s.to_csv(out/"Table_C4_speed_reduction_summary.csv",index=False);return s


def cii_from_raw_and_pred(raw:pd.DataFrame,pred:np.ndarray,cf:float)->float:
    d=raw.distance_nm.to_numpy(float) if "distance_nm" in raw.columns and raw.distance_nm.notna().any() else raw.speed_kn.to_numpy(float)*(10/60)
    dwt=raw.dwt.to_numpy(float);co2=np.asarray(pred)*cf
    den=np.nansum(dwt*d);num=np.nansum(co2)
    return num/den*1e6 if den>0 else np.nan


def constrained_speed_trim_search(model,raw_train,raw_test,feature_cols,grouped_shap,args,logger):
    out=Path(args.output_dir)/"12_optimization";speed_grid=np.array(args.opt_speed_multipliers);trim_grid=np.array(args.opt_trim_offsets)
    ranks=grouped_shap.reset_index(drop=True).copy();ranks["SHAP_rank"]=np.arange(1,len(ranks)+1);rank_map=dict(zip(ranks.grouped_feature,ranks.SHAP_rank))
    rows=[];best_rows=[]
    for vessel,g in raw_test.groupby("vessel_id"):
        st=g.ship_type.iloc[0];tr=raw_train[raw_train.vessel_id==vessel]
        if tr.empty: tr=raw_train[raw_train.ship_type==st]
        s_lo,s_hi=tr.speed_kn.quantile([.01,.99]);t_lo,t_hi=tr.trim_m.quantile([.01,.99])
        base=g.copy();base_pred=model.predict(base[list(feature_cols)]);base_cii=cii_from_raw_and_pred(base,base_pred,args.co2_factor)
        for smul in speed_grid:
            for toff in trim_grid:
                z=base.copy();z["speed_kn"]=z.speed_kn*smul;z["trim_m"]=z.trim_m+toff
                valid=((z.speed_kn>=s_lo)&(z.speed_kn<=s_hi)&(z.trim_m>=t_lo)&(z.trim_m<=t_hi)).to_numpy();ret=float(valid.mean())
                if valid.sum()<2 or ret<args.opt_min_retained: continue
                zz=z.loc[valid].copy();bb=base.loc[valid].copy();p=model.predict(zz[list(feature_cols)]);bp=model.predict(bb[list(feature_cols)])
                # preserve the fixed-10-min design: scale observed interval distance by speed multiplier
                if "distance_nm" in zz.columns:
                    zz["distance_nm"]=bb.distance_nm.to_numpy(float)*smul
                cii=cii_from_raw_and_pred(zz,p,args.co2_factor);bcii=cii_from_raw_and_pred(bb,bp,args.co2_factor)
                rows.append({"vessel_id":vessel,"ship_type":st,"speed_multiplier":smul,"trim_offset_m":toff,"retained_pct":ret*100,"baseline_proxy_same_support":bcii,"scenario_proxy":cii,"proxy_change_pct":(cii/bcii-1)*100 if bcii else np.nan,"predicted_fuel_change_pct":(p.sum()/bp.sum()-1)*100 if bp.sum() else np.nan,"distance_change_pct":smul*100-100,"speed_SHAP_rank":rank_map.get("speed_kn",np.nan),"trim_SHAP_rank":rank_map.get("trim_m",np.nan)})
        cand=pd.DataFrame([r for r in rows if r["vessel_id"]==vessel])
        if not cand.empty:
            best=cand.sort_values(["scenario_proxy","retained_pct"],ascending=[True,False]).iloc[0].to_dict();best_rows.append(best)
    pd.DataFrame(rows).to_csv(out/"Table_O1_speed_trim_grid_all_candidates.csv",index=False);B=pd.DataFrame(best_rows);B.to_csv(out/"Table_O2_best_constrained_scenario_by_vessel.csv",index=False)
    if not B.empty:
        B.groupby("ship_type").agg(vessels=("vessel_id","nunique"),median_proxy_change_pct=("proxy_change_pct","median"),mean_proxy_change_pct=("proxy_change_pct","mean"),median_speed_multiplier=("speed_multiplier","median"),median_trim_offset_m=("trim_offset_m","median")).reset_index().to_csv(out/"Table_O3_optimization_summary_by_ship_type.csv",index=False)
    return B


def make_generalisation_ladder(record_table,temporal,lovo,adapt,args):
    out=Path(args.output_dir)/"13_generalisation";rows=[]
    for model in args.generalisation_models:
        rr=record_table[record_table.model==model]
        if not rr.empty:
            r=rr.iloc[0];rows.append({"model":model,"validation_design":"Random record 80/20","target_vessel_history":"partly visible","question":"within-fleet record interpolation",**{k:r[k] for k in ["n","RMSE","MAE","MAPE_pct","sMAPE_pct","R2","cumulative_bias_pct"]}})
        if temporal is not None:
            tt=temporal[temporal.model==model]
            if not tt.empty:
                r=tt.iloc[0];rows.append({"model":model,"validation_design":"All vessels first 80% -> last 20%","target_vessel_history":"first 80% visible","question":"future prediction for known vessels",**{k:r[k] for k in ["n","RMSE","MAE","MAPE_pct","sMAPE_pct","R2","cumulative_bias_pct"]}})
        if lovo is not None:
            ll=lovo[lovo.model==model]
            if not ll.empty:
                r=ll.iloc[0];rows.append({"model":model,"validation_design":"Locked-hyperparameter LOVO zero-shot","target_vessel_history":"not visible","question":"direct transfer to unseen vessel",**{k:r[k] for k in ["n","RMSE","MAE","MAPE_pct","sMAPE_pct","R2","cumulative_bias_pct"]}})
        if adapt is not None:
            aa=adapt[adapt.model==model]
            if not aa.empty:
                r=aa.iloc[0];rows.append({"model":model,"validation_design":"LOVO + target vessel first 80% adaptation","target_vessel_history":"first 80% visible","question":"personalized future prediction for new vessel",**{k:r[k] for k in ["n","RMSE","MAE","MAPE_pct","sMAPE_pct","R2","cumulative_bias_pct"]}})
    D=pd.DataFrame(rows);D.to_csv(out/"Table_G1_generalisation_ladder.csv",index=False);return D


def write_summary(out:Path,record,ablation,temporal,lovo,shiptype,cii,opt):
    lines=["# F31 complete empirical run: result map",""]
    if record is not None and not record.empty:
        b=record.sort_values("RMSE").iloc[0];lines.append(f"Record-level best RMSE: {b.model} = {b.RMSE:.6f} (within-fleet interpolation).")
    if ablation is not None and not ablation.empty:
        f=ablation[ablation.configuration=="dynamic_physical"].iloc[0];lines.append(f"Full multi-source RMSE: {f.RMSE:.6f} using {f.model}.")
    if lovo is not None and not lovo.empty:
        for _,r in lovo.iterrows(): lines.append(f"Locked-hyperparameter LOVO {r.model}: RMSE {r.RMSE:.6f}, R2 {r.R2:.6f}.")
    if temporal is not None and not temporal.empty:
        for _,r in temporal.iterrows(): lines.append(f"Known-vessel temporal {r.model}: RMSE {r.RMSE:.6f}, R2 {r.R2:.6f}.")
    if shiptype is not None and not shiptype.empty: lines.append("Pooled vs ship-type-specific paired results: see 07_shiptype_specific/Table_S1...")
    if cii is not None and not cii.empty: lines.append("Model-choice CII-proxy sensitivity: see 11_cii/Table_C2...")
    if opt is not None and not opt.empty: lines.append("Constrained speed/trim scenario search is a model-based sensitivity search, not a causal or statutory optimum.")
    (out/"RESULTS_INDEX.md").write_text("\n\n".join(lines),encoding="utf-8")


def parse_args():
    p=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data-dir",default=r"data\05_clean23\all_phase_equal_cleaning\05_combined")
    p.add_argument("--input-csv",default=None)
    p.add_argument("--output-dir",default=r"data\11_f31_complete_empirics")
    p.add_argument("--column-overrides",default=None)
    p.add_argument("--consistency-dir",default=r"data\05_clean23\all_phase_equal_cleaning\90_consistency")
    p.add_argument("--hyperparams-csv",default="02_best_hyperparameters.csv")
    p.add_argument("--steps",default="all",help="Comma-separated modules or all")
    p.add_argument("--resume",action="store_true")
    p.add_argument("--seed",type=int,default=20260808)
    p.add_argument("--n-jobs",type=int,default=4)
    p.add_argument("--csv-chunksize",type=int,default=25000)
    p.add_argument("--test-size",type=float,default=.20)
    p.add_argument("--trajectory-gap-minutes",type=float,default=30)
    p.add_argument("--cv-folds",type=int,default=5)
    p.add_argument("--bootstrap-reps",type=int,default=2000)
    p.add_argument("--ridge-cv-n",type=int,default=100000)
    p.add_argument("--ridge-alpha-grid",type=int,default=24)
    p.add_argument("--ablation-model",choices=TREE_MODELS,default="lgbm")
    p.add_argument("--explain-model",choices=TREE_MODELS,default="xgb")
    p.add_argument("--generalisation-models",default="xgb",help="Comma list, e.g. xgb or rf,xgb,lgbm")
    p.add_argument("--shiptype-model",choices=TREE_MODELS,default="xgb")
    p.add_argument("--shap-sample-n",type=int,default=12000)
    p.add_argument("--local-audit-n",type=int,default=20000)
    p.add_argument("--pdp-grid-n",type=int,default=20)
    p.add_argument("--pdp-support-min",type=int,default=30)
    p.add_argument("--wind-zero-is",choices=["unknown","headwind","following"],default="unknown")
    p.add_argument("--shap-stability-repeats",type=int,default=5)
    p.add_argument("--shap-stability-topk",type=int,default=5)
    p.add_argument("--shap-stability-group-fraction",type=float,default=.80)
    p.add_argument("--shap-stability-train-max",type=int,default=120000)
    p.add_argument("--lime-background-n",type=int,default=10000)
    p.add_argument("--lime-seeds",type=int,default=10)
    p.add_argument("--lime-num-samples",type=int,default=4000)
    p.add_argument("--lime-top-features",type=int,default=10)
    p.add_argument("--co2-factor",type=float,default=3.114)
    p.add_argument("--opt-speed-multipliers",default="0.85,0.90,0.95,1.00")
    p.add_argument("--opt-trim-offsets",default="-0.50,-0.25,0,0.25,0.50")
    p.add_argument("--opt-min-retained",type=float,default=.95)
    p.add_argument("--expected-vessels",type=int,default=21)
    p.add_argument("--expected-cruise-n",type=int,default=489620)
    p.add_argument("--expected-maneuver-n",type=int,default=15851)
    p.add_argument("--expected-anchor-n",type=int,default=41583)
    p.add_argument("--strict-audit",action="store_true")
    p.add_argument("--quick-check",action="store_true")
    # compatibility fields required by core functions
    p.add_argument("--stage-a-trials",type=int,default=40);p.add_argument("--stage-a-n",type=int,default=30000);p.add_argument("--stage-b-n",type=int,default=100000)
    return p.parse_args()



def _stdlib_selected_csv_reader(path: Path, args, logger: logging.Logger) -> pd.DataFrame:
    """Low-overhead CSV reader used when pandas tokenizers cannot obtain Windows commit memory.

    It reads only the same modelling columns selected by the core memory-safe loader, parses
    numeric fields directly into compact ``array`` buffers, and therefore avoids the large
    temporary token/object matrices used by pandas' C/Python CSV engines.
    """
    usecols, canonical_by_raw = core._select_memory_safe_usecols(path, args, logger)
    dtype_map = core._memory_safe_dtype_map(usecols, canonical_by_raw)
    numeric_cols = {c for c, d in dtype_map.items() if str(d) in {"float32", "float64"}}
    buffers = {}
    for c in usecols:
        if c in numeric_cols:
            buffers[c] = array("f" if str(dtype_map[c]) == "float32" else "d")
        else:
            buffers[c] = []

    logger.warning("V4 low-memory stream reader activated for %s", path)
    try:
        csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    except Exception:
        pass

    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as fh:
        reader = csv.reader(fh)
        try:
            header = next(reader)
        except StopIteration:
            return pd.DataFrame(columns=usecols)
        header = [str(x).strip() for x in header]
        pos = {c: i for i, c in enumerate(header)}
        missing = [c for c in usecols if c not in pos]
        if missing:
            raise KeyError(f"V4 stream reader could not find selected columns in {path.name}: {missing}")
        indices = [pos[c] for c in usecols]
        max_i = max(indices)

        rows = 0
        for row in reader:
            if len(row) <= max_i:
                raise ValueError(f"Malformed CSV row {rows+2}: expected at least {max_i+1} fields, got {len(row)}")
            for c, i in zip(usecols, indices):
                v = row[i]
                if c in numeric_cols:
                    if v is None or str(v).strip() == "":
                        buffers[c].append(float("nan"))
                    else:
                        try:
                            buffers[c].append(float(v))
                        except Exception:
                            buffers[c].append(float("nan"))
                else:
                    buffers[c].append(v)
            rows += 1
            if rows == 1 or rows % 50000 == 0:
                logger.info("  [V4-stream] rows read: %d", rows)

    data = {}
    for c in usecols:
        d = dtype_map.get(c)
        if c in numeric_cols:
            npdtype = np.float32 if str(d) == "float32" else np.float64
            data[c] = np.asarray(buffers[c], dtype=npdtype)
        else:
            # Object columns are intentional here; canonicalization immediately normalizes
            # them and pandas StringDtype would add another temporary allocation.
            data[c] = buffers[c]
    out = pd.DataFrame(data, copy=False)
    del buffers
    gc.collect()
    logger.info("V4 stream read finished: %d rows x %d columns", len(out), out.shape[1])
    return out


def _cruise_cache_paths(out: Path):
    art = out / "14_artifacts"
    return art / "cruise_canonical_v4.pkl", art / "cruise_canonical_v4.meta.json"


def _build_or_load_cruise_bundle_v4(args, out: Path, logger: logging.Logger):
    """Load only the 489,620-row cruise cohort for cruise-only extension modules.

    The first successful V4 run creates a canonical pickle cache. Later ``-Resume`` runs
    bypass CSV parsing entirely, which is substantially safer on 16-GB Windows machines.
    """
    cache, meta = _cruise_cache_paths(out)
    if args.resume and cache.exists():
        logger.info("Resume: loading canonical cruise cache: %s", cache)
        raw = pd.read_pickle(cache)
        if len(raw) != args.expected_cruise_n:
            raise AssertionError(f"Cruise cache has {len(raw)} rows; expected {args.expected_cruise_n}")
        if raw["vessel_id"].nunique() != args.expected_vessels:
            raise AssertionError(f"Cruise cache has {raw['vessel_id'].nunique()} vessels; expected {args.expected_vessels}")
        bad_phase = set(raw["phase"].dropna().astype(str).unique()) - {"cruise"}
        if bad_phase:
            raise AssertionError(f"Cruise cache contains non-cruise phases: {sorted(bad_phase)}")
        feature_cols = [c for c in core.CANONICAL_FEATURES if c in raw.columns]
        if len(feature_cols) != len(core.CANONICAL_FEATURES):
            missing = [c for c in core.CANONICAL_FEATURES if c not in raw.columns]
            raise KeyError(f"Cruise cache missing model features: {missing}")
        X = raw[feature_cols].copy()
        return raw.reset_index(drop=True), X.reset_index(drop=True), {"loader":"v4_canonical_cache","cache":str(cache)}

    cruise_path = Path(args.data_dir) / "final_fixed31_cruise.csv"
    if not cruise_path.exists():
        raise FileNotFoundError(f"V4 cruise-only loader requires {cruise_path}")
    logger.info("V4 cruise-only mode: bypassing the 547,054-row combined CSV")
    logger.info("V4 source: %s", cruise_path)
    df = _stdlib_selected_csv_reader(cruise_path, args, logger)
    bundle = core.canonicalize_dataframe(df, args, logger)
    del df
    gc.collect()
    raw = bundle.raw.reset_index(drop=True)
    X = bundle.feature_df.reset_index(drop=True)
    if len(raw) != args.expected_cruise_n:
        raise AssertionError(f"Cruise source produced {len(raw)} canonical rows; expected {args.expected_cruise_n}")
    if raw["vessel_id"].nunique() != args.expected_vessels:
        raise AssertionError(f"Cruise source has {raw['vessel_id'].nunique()} vessels; expected {args.expected_vessels}")
    if set(raw["phase"].dropna().astype(str).unique()) != {"cruise"}:
        raise AssertionError(f"Cruise source contains unexpected phases: {raw['phase'].value_counts().to_dict()}")

    # Save canonical raw only; all model features are columns of raw and can be reconstructed.
    tmp = cache.with_suffix(".tmp.pkl")
    raw.to_pickle(tmp)
    os.replace(tmp, cache)
    meta_obj = {
        "script_version": SCRIPT_VERSION,
        "rows": int(len(raw)),
        "vessels": int(raw["vessel_id"].nunique()),
        "phase_counts": {str(k): int(v) for k, v in raw["phase"].value_counts().to_dict().items()},
        "source": str(cruise_path),
        "features": list(X.columns),
    }
    meta.write_text(json.dumps(meta_obj, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Cruise cache saved: %s", cache)
    return raw, X, meta_obj


def _validate_v4_record_alignment(X: pd.DataFrame, test_idx: np.ndarray, args, logger: logging.Logger):
    """Verify that V4 cruise row order matches the already locked official record split."""
    art = Path(args.output_dir) / "14_artifacts"
    pred_path = art / "xgb_record_test_prediction.npy"
    model_path = art / "xgb_record_split.joblib"
    stamp = art / "v4_cruise_alignment_verified.json"
    if args.resume and stamp.exists():
        logger.info("Resume: record-order alignment was already verified")
        return
    if not pred_path.exists() or not model_path.exists():
        raise FileNotFoundError("Reference XGBoost model and prediction are required for alignment verification")
    logger.info("Record-alignment check: comparing XGB predictions against saved official test predictions")
    saved = np.load(pred_path)
    if len(saved) != len(test_idx):
        raise AssertionError(f"Saved XGB prediction length {len(saved)} != test split {len(test_idx)}")
    model = joblib.load(model_path, mmap_mode="r")
    now = predict_named(model, "xgb", X.iloc[test_idx].reset_index(drop=True))
    if not np.allclose(now, saved, rtol=1e-6, atol=1e-7, equal_nan=True):
        diff = float(np.nanmax(np.abs(now - saved)))
        raise AssertionError(
            "V4 cruise-file row order does not reproduce the locked official XGB test predictions; "
            f"max_abs_diff={diff:.6g}. Refusing to continue."
        )
    stamp.write_text(json.dumps({"verified":True,"n":int(len(saved)),"max_abs_diff":float(np.nanmax(np.abs(now-saved)))}, indent=2), encoding="utf-8")
    logger.info("Record alignment verified on %d official test predictions", len(saved))
    del model, now, saved
    gc.collect()


def main():
    faulthandler.enable(all_threads=True)
    args=parse_args();args.steps=parse_steps(args.steps);args.generalisation_models=[normalize_model_name(x) for x in args.generalisation_models.split(",") if x.strip()]
    args.opt_speed_multipliers=[float(x) for x in args.opt_speed_multipliers.split(",")];args.opt_trim_offsets=[float(x) for x in args.opt_trim_offsets.split(",")]
    if args.quick_check:
        args.bootstrap_reps=min(args.bootstrap_reps,50);args.shap_sample_n=min(args.shap_sample_n,500);args.local_audit_n=min(args.local_audit_n,500);args.pdp_grid_n=min(args.pdp_grid_n,6);args.shap_stability_repeats=min(args.shap_stability_repeats,2);args.shap_stability_train_max=min(args.shap_stability_train_max,3000);args.lime_seeds=min(args.lime_seeds,2);args.lime_num_samples=min(args.lime_num_samples,500);args.lime_background_n=min(args.lime_background_n,1000);args.ridge_cv_n=min(args.ridge_cv_n,3000);args.ridge_alpha_grid=min(args.ridge_alpha_grid,5);args.n_jobs=min(args.n_jobs,2)
    out=Path(args.output_dir);ensure_dirs(out);logger=setup_logger(out);t0=time.time();logger.info("Predictive-validation pipeline %s",SCRIPT_VERSION);logger.info("Steps: %s",args.steps);logger.info("PRIMARY COHORT LOCKED TO CRUISE ONLY")
    here=Path(__file__).resolve().parent
    hp=Path(args.hyperparams_csv);hp=hp if hp.is_absolute() else here/hp
    if args.column_overrides is None:
        cand=here/"column_overrides_fixed31.json";args.column_overrides=str(cand) if cand.exists() else None
    params=load_locked_hyperparams(hp,logger)
    args._artifact_reuse = bool(args.resume)

    logger.info("STEP data-load BEGIN")
    phase_eval={}
    needs_all_phase = any(st in args.steps for st in ("data", "phase"))
    if needs_all_phase:
        # Data audit and operating-regime robustness genuinely require all three phases.
        df=core.load_phase_files(args,logger)
        bundle_all=core.canonicalize_dataframe(df,args,logger)
        audit=core.strict_fixed31_audit(bundle_all.raw,args,logger)
        core.maybe_copy_consistency_files(args,out,logger)
        if "data" in args.steps: run_data_outputs(bundle_all,args,logger)
        if "phase" in args.steps:
            for ph in ["maneuver","anchor_berth"]:
                pm=(bundle_all.raw.phase==ph).to_numpy()
                if pm.any():
                    phase_eval[ph]=(bundle_all.raw.loc[pm].reset_index(drop=True),bundle_all.feature_df.loc[pm].reset_index(drop=True))
        mask=(bundle_all.raw.phase=="cruise").to_numpy()
        raw=bundle_all.raw.loc[mask].reset_index(drop=True)
        X=bundle_all.feature_df.loc[mask].reset_index(drop=True)
        del bundle_all, df, mask
        gc.collect()
    else:
        # All remaining modules use cruise only. Do not parse the 288-MiB all-phase CSV.
        raw, X, load_meta = _build_or_load_cruise_bundle_v4(args,out,logger)
        audit={
            "loader":"cruise_only",
            "total":int(len(raw)),
            "phase_counts":{str(k):int(v) for k,v in raw.phase.value_counts().to_dict().items()},
            "vessels":int(raw.vessel_id.nunique()),
            "source_meta":load_meta,
            "note":"Full Fixed31 phase audit remains in 00_data_audit from prior all-phase runs.",
        }
        core.maybe_copy_consistency_files(args,out,logger)
    logger.info("STEP data-load END | cruise rows=%d; non-cruise eval rows=%d",len(raw),sum(len(v[0]) for v in phase_eval.values()))

    if len(raw)!=args.expected_cruise_n: logger.warning("Cruise cohort has %d rows; expected %d",len(raw),args.expected_cruise_n)
    train_idx,test_idx=load_or_create_record_split(raw,args,logger)
    if not needs_all_phase:
        _validate_v4_record_alignment(X,test_idx,args,logger)
    Xtr=X.iloc[train_idx].reset_index(drop=True);Xte=X.iloc[test_idx].reset_index(drop=True)
    y=raw.target.to_numpy(float);ytr=y[train_idx];yte=y[test_idx]
    rtr=raw.iloc[train_idx].reset_index(drop=True);rte=raw.iloc[test_idx].reset_index(drop=True)
    args._raw_train=rtr;args._params=params

    record=ablation=temporal=lovo=adapt=shiptype=cii=opt=None
    grp_shap=None;sv=Xsh=rawsh=None

    if "base" in args.steps:
        logger.info("STEP base BEGIN")
        record,vmet,params=run_base_models_disk_safe(Xtr,ytr,Xte,yte,rte,params,args,logger);args._params=params
        args._artifact_reuse = True
        logger.info("STEP base END")
    if "phase" in args.steps:
        logger.info("STEP phase BEGIN")
        run_phase_robustness_disk_safe(phase_eval,Xtr,ytr,Xte,yte,rte,params,args,logger)
        args._artifact_reuse = True
        phase_eval.clear();gc.collect();logger.info("STEP phase END")
    if "ablation" in args.steps:
        logger.info("STEP ablation BEGIN");ablation=run_ablation(Xtr,ytr,Xte,yte,rte,params,args,logger);gc.collect();logger.info("STEP ablation END")
    if "temporal" in args.steps:
        logger.info("STEP temporal BEGIN");temporal=run_temporal(args.generalisation_models,X,raw,params,args,logger);gc.collect();logger.info("STEP temporal END")
    if "lovo" in args.steps:
        logger.info("STEP lovo BEGIN");lovo=run_lovo_models(args.generalisation_models,X,raw,params,args,logger);gc.collect();logger.info("STEP lovo END")
    if "adaptation" in args.steps:
        logger.info("STEP adaptation BEGIN");adapt=run_adaptation_models(args.generalisation_models,X,raw,params,args,logger);gc.collect();logger.info("STEP adaptation END")
    if "shiptype" in args.steps:
        logger.info("STEP shiptype BEGIN")
        m,p=get_or_fit_record_model(args.shiptype_model,params[args.shiptype_model],Xtr,ytr,Xte,args,logger,args.resume)
        shiptype=run_shiptype_specific(args.shiptype_model,Xtr,ytr,rtr,Xte,yte,rte,m,args,logger)
        del m,p;gc.collect();logger.info("STEP shiptype END")
    if "shap" in args.steps:
        logger.info("STEP shap BEGIN")
        params=ensure_ridge_params(params,Xtr,ytr,rtr,args,logger);args._params=params
        em,ep=get_or_fit_record_model(args.explain_model,params[args.explain_model],Xtr,ytr,Xte,args,logger,args.resume)
        rm,rp=get_or_fit_record_model("ridge_interaction",params["ridge_interaction"],Xtr,ytr,Xte,args,logger,args.resume)
        glob,grp_shap,dirn,sv,Xsh,rawsh=run_shap_physical(args.explain_model,em,Xtr,ytr,rtr,Xte,yte,rte,rm,args,logger)
        del em,ep,rm,rp;gc.collect();logger.info("STEP shap END")
    if "shap_stability" in args.steps:
        logger.info("STEP shap_stability BEGIN")
        if Xsh is None:
            idx=core.sample_indices(len(Xte),min(args.shap_sample_n,len(Xte)),args.seed+800);Xsh=Xte.iloc[idx].reset_index(drop=True)
            em,ep=get_or_fit_record_model(args.explain_model,params[args.explain_model],Xtr,ytr,Xte,args,logger,args.resume)
            sv0=core.get_tree_shap(em,Xsh);imp=np.mean(np.abs(sv0),axis=0);glob=pd.DataFrame({"feature":Xsh.columns,"mean_abs_SHAP":imp}).sort_values("mean_abs_SHAP",ascending=False)
            del em,ep,sv0;gc.collect()
        run_shap_stability(args.explain_model,params,Xtr,ytr,rtr,Xsh,glob,args,logger);gc.collect();logger.info("STEP shap_stability END")
    if "lime" in args.steps:
        logger.info("STEP lime BEGIN")
        em,ep=get_or_fit_record_model(args.explain_model,params[args.explain_model],Xtr,ytr,Xte,args,logger,args.resume)
        run_lime_validation(args.explain_model,em,Xtr,ytr,Xte,yte,rte,ep,sv,args,logger)
        del em,ep;gc.collect();logger.info("STEP lime END")
    if "cii" in args.steps:
        logger.info("STEP cii BEGIN")
        preds,params=collect_cii_predictions_disk_safe(Xtr,ytr,Xte,params,args,logger);args._params=params
        cii=run_cii_models(rte,preds,args,logger)
        del preds;gc.collect()
        em,ep=get_or_fit_record_model(args.explain_model,params[args.explain_model],Xtr,ytr,Xte,args,logger,args.resume)
        run_speed_scenarios(em,rtr,rte,list(X.columns),args,logger)
        del em,ep;gc.collect();logger.info("STEP cii END")
    if "optimization" in args.steps:
        logger.info("STEP optimization BEGIN")
        em,ep=get_or_fit_record_model(args.explain_model,params[args.explain_model],Xtr,ytr,Xte,args,logger,args.resume)
        if grp_shap is None:
            idx=core.sample_indices(len(Xte),min(args.shap_sample_n,len(Xte)),args.seed+800);Xs=Xte.iloc[idx].reset_index(drop=True);svv=core.get_tree_shap(em,Xs);gg=pd.DataFrame({"feature":Xs.columns,"mean_abs_SHAP":np.mean(np.abs(svv),axis=0)});gm={"heading_sin":"heading_direction","heading_cos":"heading_direction","rel_wind_sin":"relative_wind_direction","rel_wind_cos":"relative_wind_direction","rel_wave_sin":"relative_wave_direction","rel_wave_cos":"relative_wave_direction","ship_type_bulk":"ship_type","ship_type_container":"ship_type"};gg["grouped_feature"]=gg.feature.map(gm).fillna(gg.feature);grp_shap=gg.groupby("grouped_feature",as_index=False).mean_abs_SHAP.sum().sort_values("mean_abs_SHAP",ascending=False);del Xs,svv,gg;gc.collect()
        opt=constrained_speed_trim_search(em,rtr,rte,list(X.columns),grp_shap,args,logger)
        del em,ep;gc.collect();logger.info("STEP optimization END")
    if "summary" in args.steps:
        logger.info("STEP summary BEGIN")
        if record is None and (Path(args.output_dir)/"01_record_level"/"Table_B1_record_level_model_comparison.csv").exists(): record=pd.read_csv(Path(args.output_dir)/"01_record_level"/"Table_B1_record_level_model_comparison.csv")
        if temporal is None and (Path(args.output_dir)/"04_temporal"/"Table_T1_known_vessel_temporal_overall.csv").exists(): temporal=pd.read_csv(Path(args.output_dir)/"04_temporal"/"Table_T1_known_vessel_temporal_overall.csv")
        if lovo is None and (Path(args.output_dir)/"05_lovo"/"Table_L1_LOVO_overall.csv").exists(): lovo=pd.read_csv(Path(args.output_dir)/"05_lovo"/"Table_L1_LOVO_overall.csv")
        if adapt is None and (Path(args.output_dir)/"06_adaptation"/"Table_N1_new_vessel_adaptation_overall.csv").exists(): adapt=pd.read_csv(Path(args.output_dir)/"06_adaptation"/"Table_N1_new_vessel_adaptation_overall.csv")
        if record is not None: make_generalisation_ladder(record,temporal,lovo,adapt,args)
        write_summary(out,record,ablation,temporal,lovo,shiptype,cii,opt);logger.info("STEP summary END")

    manifest={"script_version":SCRIPT_VERSION,"elapsed_seconds":time.time()-t0,"steps":args.steps,"primary_scope":"cruise","cruise_rows":len(raw),"outer_train_rows":len(train_idx),"outer_test_rows":len(test_idx),"vessels":raw.vessel_id.nunique(),"locked_hyperparams_source":str(hp),"locked_params":params,"audit":audit,"method_notes":{"LOVO":"locked hyperparameters, complete vessel exclusion from fitting; not nested tuning","phase":"cruise-only primary model; manoeuvre/anchor are robustness/OOD evaluation","SHAP":"attribution only; not causal and not a direct optimization weight","CII":"observation-period main-engine proxy; no statutory A-E rating","optimization":"constrained model-based speed/trim scenario search within 1st-99th percentile support; not causal or full fixed-distance voyage optimization"}}
    (out/"run_manifest_complete.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False,default=str),encoding="utf-8")
    logger.info("COMPLETE PIPELINE FINISHED in %.1f min. Output: %s",(time.time()-t0)/60,out)


if __name__=="__main__":
    try: main()
    except Exception:
        traceback.print_exc();sys.exit(1)
