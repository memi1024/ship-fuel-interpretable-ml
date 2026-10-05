#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared helpers for the manuscript-revision reruns.

Design goals
------------
1. Reuse the canonical Fixed31 loader and the corrected model factory already
   present in src/paper_analysis.
2. Never silently rebuild the official L1 split unless the caller explicitly
   allows it.
3. Keep the locked manuscript hyperparameter registry distinct from the new
   revision-only nested-tuning sensitivity analyses.
4. Make every expensive script resumable at the file level.
"""
from __future__ import annotations

import gc
import importlib
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

for _name, _value in (
    ("OMP_NUM_THREADS", "1"),
    ("OPENBLAS_NUM_THREADS", "1"),
    ("MKL_NUM_THREADS", "1"),
    ("NUMEXPR_NUM_THREADS", "1"),
    ("LIGHTGBM_NUM_THREADS", "1"),
    ("OMP_WAIT_POLICY", "PASSIVE"),
    ("KMP_BLOCKTIME", "0"),
):
    os.environ.setdefault(_name, _value)

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold

EXPECTED_CRUISE_ROWS = 489_620
EXPECTED_VESSELS = 21
DEFAULT_SEED = 20260808

ENV_FEATURES = [
    "rel_wind_speed_kn", "rel_wind_sin", "rel_wind_cos",
    "wave_height_m", "rel_wave_sin", "rel_wave_cos",
    "wave_period_s", "sst_c", "mslp_hpa",
]

JOINT_SUPPORT_DEFAULT_FEATURES = [
    "speed_kn", "draught_m", "trim_m",
    "rel_wind_speed_kn", "wave_height_m",
]


def setup_logger(name: str, out_dir: Path, filename: Optional[str] = None) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if filename:
        fh = logging.FileHandler(out_dir / filename, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def atomic_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8-sig")
    tmp.replace(path)


def atomic_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def require_file(path: str | Path, label: str) -> Path:
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"{label} not found: {p}")
    return p


def require_dir(path: str | Path, label: str) -> Path:
    p = Path(path).expanduser().resolve()
    if not p.is_dir():
        raise FileNotFoundError(f"{label} not found: {p}")
    return p


def load_analysis_modules(analysis_dir: str | Path):
    """Import the canonical analysis core and v4 runner from src/paper_analysis."""
    adir = require_dir(analysis_dir, "analysis directory")
    if str(adir) not in sys.path:
        sys.path.insert(0, str(adir))
    core = importlib.import_module("f31_core_memory_safe")
    runner = importlib.import_module("run_f31_complete_empirics_v4")
    return core, runner


def import_analysis_module(analysis_dir: str | Path, module_name: str):
    adir = require_dir(analysis_dir, "analysis directory")
    if str(adir) not in sys.path:
        sys.path.insert(0, str(adir))
    return importlib.import_module(module_name)


def canonical_loader_args(column_overrides: Optional[str], trajectory_gap_minutes: float):
    return SimpleNamespace(
        column_overrides=str(Path(column_overrides).resolve()) if column_overrides else None,
        trajectory_gap_minutes=float(trajectory_gap_minutes),
    )


def load_fixed31_cruise(
    fixed31_csv: str | Path,
    analysis_dir: str | Path,
    column_overrides: Optional[str],
    trajectory_gap_minutes: float,
    logger: logging.Logger,
    allow_noncanonical_counts: bool = False,
):
    """
    Load the supplied Fixed31 cruise CSV through the same canonicalization code
    used by the manuscript pipeline.
    """
    core, runner = load_analysis_modules(analysis_dir)
    p = require_file(fixed31_csv, "Fixed31 cruise CSV")
    logger.info("Reading Fixed31 cruise CSV: %s", p)
    df = pd.read_csv(p, low_memory=False)
    args = canonical_loader_args(column_overrides, trajectory_gap_minutes)
    bundle = core.canonicalize_dataframe(df, args, logger)
    raw = bundle.raw.reset_index(drop=True)
    X = bundle.feature_df.reset_index(drop=True)

    # final_fixed31_cruise.csv should already be cruise-only. If phase exists,
    # enforce that no non-cruise observations slipped into the revision reruns.
    if "phase" in raw.columns:
        phases = set(raw["phase"].dropna().astype(str).unique())
        if phases and phases != {"cruise"}:
            raise AssertionError(
                f"Revision reruns require the cruise cohort only; observed phases={sorted(phases)}"
            )

    if not allow_noncanonical_counts:
        if len(raw) != EXPECTED_CRUISE_ROWS:
            raise AssertionError(
                f"Fixed31 cruise row count changed: {len(raw):,} != {EXPECTED_CRUISE_ROWS:,}"
            )
        if int(raw["vessel_id"].nunique()) != EXPECTED_VESSELS:
            raise AssertionError(
                f"Vessel count changed: {raw['vessel_id'].nunique()} != {EXPECTED_VESSELS}"
            )

    logger.info(
        "Loaded canonical cruise cohort: rows=%d vessels=%d features=%d",
        len(raw), raw["vessel_id"].nunique(), X.shape[1],
    )
    return core, runner, raw, X, list(bundle.feature_cols), bundle.column_map


def load_locked_params(runner, hyperparams_csv: str | Path, logger: logging.Logger) -> Dict[str, Dict]:
    p = require_file(hyperparams_csv, "locked hyperparameter registry")
    return runner.load_locked_hyperparams(p, logger)


def load_official_split(
    split_path: str | Path,
    raw: pd.DataFrame,
    core,
    *,
    test_size: float = 0.20,
    seed: int = DEFAULT_SEED,
    allow_rebuild: bool = False,
    logger: Optional[logging.Logger] = None,
) -> Tuple[np.ndarray, np.ndarray, str]:
    p = Path(split_path).expanduser().resolve()
    source = "official_artifact"
    if p.is_file():
        z = np.load(p)
        tr = np.asarray(z["train_idx"], dtype=int)
        te = np.asarray(z["test_idx"], dtype=int)
    else:
        if not allow_rebuild:
            raise FileNotFoundError(
                f"Official split artifact not found: {p}\n"
                "Pass the historical record_split_indices.npz, or use --allow-rebuild-split "
                "only for a clearly labelled canonical reconstruction."
            )
        tr, te = core.stratified_record_split(raw, test_size, seed)
        source = "canonical_rebuild"

    if len(tr) + len(te) != len(raw):
        raise AssertionError("Split does not cover the full cruise cohort.")
    if len(np.intersect1d(tr, te)):
        raise AssertionError("Train/test split overlaps.")
    if len(np.unique(np.concatenate([tr, te]))) != len(raw):
        raise AssertionError("Split indices are not a one-to-one full coverage of the cohort.")
    if logger:
        logger.info("L1 split: train=%d test=%d source=%s", len(tr), len(te), source)
    return np.sort(tr), np.sort(te), source


def fit_locked_model(
    runner,
    model_name: str,
    params: Mapping[str, Any],
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    *,
    seed: int,
    n_jobs: int,
    sample_weight: Optional[np.ndarray] = None,
):
    model = runner.corrected_model_factory(model_name, dict(params), seed, n_jobs)
    if sample_weight is None:
        model.fit(X_train, y_train)
    else:
        try:
            model.fit(X_train, y_train, sample_weight=sample_weight)
        except TypeError:
            # Only relevant for sklearn Pipelines. The revision weighted-training
            # script defaults to tree models, but this keeps failure messages clear.
            model.fit(X_train, y_train, model__sample_weight=sample_weight)
    return model


def fit_boosting_with_early_stop(
    runner,
    model_name: str,
    params: Mapping[str, Any],
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_valid: pd.DataFrame,
    y_valid: np.ndarray,
    *,
    seed: int,
    n_jobs: int,
    early_stopping_rounds: int,
):
    """Fit XGB/LGBM with best-effort version-compatible early stopping."""
    name = model_name.lower()
    model = runner.corrected_model_factory(name, dict(params), seed, n_jobs)
    best_iteration = np.nan

    if name == "xgb":
        try:
            model.set_params(early_stopping_rounds=int(early_stopping_rounds))
            model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
        except TypeError:
            # Older XGBoost sklearn wrapper.
            model.fit(
                X_train, y_train,
                eval_set=[(X_valid, y_valid)],
                verbose=False,
                early_stopping_rounds=int(early_stopping_rounds),
            )
        bi = getattr(model, "best_iteration", None)
        if bi is not None:
            try:
                best_iteration = int(bi) + 1
            except Exception:
                pass

    elif name == "lgbm":
        try:
            import lightgbm as lgb
            model.fit(
                X_train, y_train,
                eval_set=[(X_valid, y_valid)],
                callbacks=[lgb.early_stopping(int(early_stopping_rounds), verbose=False)],
            )
        except Exception:
            # Fallback if callback API differs.
            model.fit(X_train, y_train)
        bi = getattr(model, "best_iteration_", None)
        if bi is not None:
            try:
                best_iteration = int(bi)
            except Exception:
                pass
    else:
        model.fit(X_train, y_train)

    pred = np.asarray(model.predict(X_valid), dtype=float)
    rmse = float(math.sqrt(mean_squared_error(y_valid, pred)))
    return model, pred, rmse, best_iteration


def expanding_temporal_folds(
    raw_outer_train: pd.DataFrame,
    n_splits: int = 3,
    initial_train_fraction: float = 0.50,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Strict chronological inner folds inside the outer L2 training partition.

    Each vessel contributes an expanding training prefix followed by the next
    chronological validation block. No inner validation observation precedes a
    training observation from the same vessel.
    """
    if raw_outer_train["timestamp"].isna().all():
        raise ValueError("Strict temporal tuning requires timestamps.")
    if n_splits < 2:
        raise ValueError("n_splits must be >=2")
    if not (0.2 <= initial_train_fraction < 1.0):
        raise ValueError("initial_train_fraction must be in [0.2,1).")

    ordered: Dict[str, np.ndarray] = {}
    for vessel, inds in raw_outer_train.groupby("vessel_id").groups.items():
        loc = raw_outer_train.loc[inds].sort_values("timestamp").index.to_numpy(dtype=int)
        if len(loc) >= n_splits + 2:
            ordered[str(vessel)] = loc

    edges = np.linspace(initial_train_fraction, 1.0, n_splits + 1)
    folds: List[Tuple[np.ndarray, np.ndarray]] = []
    for k in range(n_splits):
        train_parts, valid_parts = [], []
        left_frac, right_frac = float(edges[k]), float(edges[k + 1])
        for loc in ordered.values():
            n = len(loc)
            left = max(1, min(n - 1, int(math.floor(n * left_frac))))
            right = max(left + 1, min(n, int(math.floor(n * right_frac))))
            tr = loc[:left]
            va = loc[left:right]
            if len(va):
                train_parts.append(tr)
                valid_parts.append(va)
        if not train_parts or not valid_parts:
            raise RuntimeError(f"Could not construct temporal fold {k+1}.")
        folds.append((np.sort(np.concatenate(train_parts)), np.sort(np.concatenate(valid_parts))))
    return folds


def vessel_group_folds(raw_source: pd.DataFrame, n_splits: int = 3) -> List[Tuple[np.ndarray, np.ndarray]]:
    groups = raw_source["vessel_id"].astype(str).to_numpy()
    unique = np.unique(groups)
    folds = min(int(n_splits), len(unique))
    if folds < 2:
        raise ValueError("Need at least two source vessels for grouped inner CV.")
    splitter = GroupKFold(n_splits=folds)
    idx = np.arange(len(raw_source))
    return [(np.asarray(tr), np.asarray(va)) for tr, va in splitter.split(idx, groups=groups)]


def suggest_params(trial, model_name: str) -> Dict[str, Any]:
    """
    Revision-only search spaces. They deliberately include the locked manuscript
    setting but are not claimed to reproduce the historical focused-tuning run.
    """
    name = model_name.lower()
    if name == "xgb":
        return {
            "n_estimators": 5000,
            "learning_rate": trial.suggest_float("learning_rate", 0.008, 0.080, log=True),
            "max_depth": trial.suggest_int("max_depth", 4, 7),
            "min_child_weight": trial.suggest_float("min_child_weight", 2.0, 60.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.50, 0.85),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.75, 0.98),
            "gamma": trial.suggest_float("gamma", 0.0, 0.05),
            "reg_alpha": trial.suggest_categorical(
                "reg_alpha", [0.0, 1e-8, 1e-6, 1e-4, 1e-2, 0.1]
            ),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 0.05, log=True),
        }
    if name == "lgbm":
        return {
            "n_estimators": 5000,
            "learning_rate": trial.suggest_float("learning_rate", 0.008, 0.060, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 48, 160),
            "max_depth": -1,
            "min_child_samples": trial.suggest_int("min_child_samples", 15, 160, log=True),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.70, 0.98),
            "subsample": trial.suggest_float("subsample", 0.75, 1.00),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-9, 0.05, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-7, 0.05, log=True),
        }
    raise ValueError(f"Nested tuning implemented for xgb/lgbm, got {model_name!r}")


def enqueue_locked_if_compatible(study, model_name: str, locked: Mapping[str, Any]) -> None:
    """Seed Optuna with the manuscript setting when it lies inside the revision space."""
    name = model_name.lower()
    p = dict(locked)
    try:
        if name == "xgb":
            seed = {
                "learning_rate": float(p["learning_rate"]),
                "max_depth": int(p["max_depth"]),
                "min_child_weight": float(p["min_child_weight"]),
                "subsample": float(p["subsample"]),
                "colsample_bytree": float(p["colsample_bytree"]),
                "gamma": float(p.get("gamma", 0.0)),
                "reg_alpha": float(p.get("reg_alpha", 0.0)),
                "reg_lambda": float(p["reg_lambda"]),
            }
        elif name == "lgbm":
            seed = {
                "learning_rate": float(p["learning_rate"]),
                "num_leaves": int(p["num_leaves"]),
                "min_child_samples": int(p["min_child_samples"]),
                "colsample_bytree": float(p.get("colsample_bytree", p.get("feature_fraction"))),
                "subsample": float(p.get("subsample", p.get("bagging_fraction"))),
                "reg_alpha": max(float(p.get("reg_alpha", 0.0)), 1e-9),
                "reg_lambda": max(float(p.get("reg_lambda", 1e-7)), 1e-7),
            }
        else:
            return
        study.enqueue_trial(seed)
    except Exception:
        # Seeding is a convenience, never a hidden requirement.
        return


def run_optuna_cv(
    *,
    model_name: str,
    locked_params: Mapping[str, Any],
    X: pd.DataFrame,
    raw: pd.DataFrame,
    splits: Sequence[Tuple[np.ndarray, np.ndarray]],
    runner,
    n_trials: int,
    seed: int,
    n_jobs_model: int,
    early_stopping_rounds: int,
    logger: logging.Logger,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    try:
        import optuna
    except ImportError as exc:
        raise SystemExit("A1/A2 require Optuna: pip install optuna") from exc

    rows: List[Dict[str, Any]] = []

    def objective(trial):
        params = suggest_params(trial, model_name)
        fold_scores, fold_best = [], []
        for fold_no, (tr, va) in enumerate(splits, start=1):
            model = None
            try:
                model, pred, score, best_it = fit_boosting_with_early_stop(
                    runner,
                    model_name,
                    params,
                    X.iloc[tr],
                    raw.iloc[tr]["target"].to_numpy(float),
                    X.iloc[va],
                    raw.iloc[va]["target"].to_numpy(float),
                    seed=seed + fold_no,
                    n_jobs=n_jobs_model,
                    early_stopping_rounds=early_stopping_rounds,
                )
                fold_scores.append(score)
                fold_best.append(best_it)
            finally:
                del model
                gc.collect()
        finite_best = np.asarray([v for v in fold_best if np.isfinite(v)], dtype=float)
        mean_best = float(np.median(finite_best)) if finite_best.size else np.nan
        row = {
            "trial": int(trial.number),
            "model": model_name.lower(),
            "mean_cv_RMSE": float(np.mean(fold_scores)),
            "std_cv_RMSE": float(np.std(fold_scores, ddof=1)) if len(fold_scores) > 1 else 0.0,
            "median_best_iteration": mean_best,
            "parameters_json": json.dumps(params, sort_keys=True),
        }
        for i, s in enumerate(fold_scores, start=1):
            row[f"fold{i}_RMSE"] = float(s)
        rows.append(row)
        logger.info(
            "%s trial=%d mean_RMSE=%.6f best_iter=%s",
            model_name, trial.number, row["mean_cv_RMSE"],
            "NA" if not np.isfinite(mean_best) else f"{mean_best:.0f}",
        )
        return row["mean_cv_RMSE"]

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    enqueue_locked_if_compatible(study, model_name, locked_params)
    study.optimize(objective, n_trials=int(n_trials), n_jobs=1, show_progress_bar=False)

    tab = pd.DataFrame(rows).sort_values(["mean_cv_RMSE", "std_cv_RMSE"]).reset_index(drop=True)
    best_row = tab.iloc[0]
    params = json.loads(best_row["parameters_json"])
    if np.isfinite(best_row["median_best_iteration"]):
        params["n_estimators"] = int(max(1, round(float(best_row["median_best_iteration"]))))
    return params, tab


def metric_row(core, y: np.ndarray, pred: np.ndarray) -> Dict[str, Any]:
    return dict(core.metric_dict(np.asarray(y, dtype=float), np.asarray(pred, dtype=float)))


def by_vessel_metrics(core, raw_test: pd.DataFrame, pred: np.ndarray) -> pd.DataFrame:
    return core.per_group_metrics(raw_test.reset_index(drop=True), np.asarray(pred), ["vessel_id", "ship_type"])


def vessel_balanced_weights(raw_train: pd.DataFrame) -> np.ndarray:
    counts = raw_train["vessel_id"].value_counts()
    w = raw_train["vessel_id"].map(lambda v: 1.0 / float(counts.loc[v])).to_numpy(float)
    w = w * (len(w) / w.sum())
    return w


def interventional_tree_shap(
    model,
    X_train: pd.DataFrame,
    X_eval: pd.DataFrame,
    *,
    background_n: int,
    sample_n: int,
    seed: int,
):
    try:
        import shap
    except ImportError as exc:
        raise SystemExit("SHAP is required for A5: pip install shap") from exc
    rng = np.random.default_rng(seed)
    bg_idx = rng.choice(len(X_train), size=min(background_n, len(X_train)), replace=False)
    ev_idx = rng.choice(len(X_eval), size=min(sample_n, len(X_eval)), replace=False)
    bg = X_train.iloc[np.sort(bg_idx)].copy()
    ev = X_eval.iloc[np.sort(ev_idx)].copy()
    explainer = shap.TreeExplainer(model, data=bg, feature_perturbation="interventional")
    sv = explainer.shap_values(ev)
    if isinstance(sv, list):
        sv = sv[0]
    return ev, np.asarray(sv, dtype=float), np.sort(ev_idx)


def shap_summary(X_eval: pd.DataFrame, shap_values: np.ndarray) -> Tuple[pd.DataFrame, pd.DataFrame]:
    imp = pd.DataFrame({
        "feature": list(X_eval.columns),
        "mean_abs_SHAP": np.mean(np.abs(shap_values), axis=0),
    }).sort_values("mean_abs_SHAP", ascending=False).reset_index(drop=True)
    imp["rank"] = np.arange(1, len(imp) + 1)

    rows = []
    for j, feature in enumerate(X_eval.columns):
        rho = spearmanr(
            pd.to_numeric(X_eval.iloc[:, j], errors="coerce").to_numpy(float),
            shap_values[:, j],
            nan_policy="omit",
        ).statistic
        rows.append({"feature": feature, "spearman_feature_vs_SHAP": float(rho)})
    return imp, pd.DataFrame(rows)


def circular_distance_deg(angle: np.ndarray, center: float) -> np.ndarray:
    a = np.asarray(angle, dtype=float)
    return np.abs((a - center + 180.0) % 360.0 - 180.0)


def head_sea_mask(raw: pd.DataFrame, half_width_deg: float = 45.0) -> np.ndarray:
    if "rel_wave_dir_deg" not in raw.columns:
        raise KeyError("rel_wave_dir_deg is required for the tanker × head-sea audit.")
    return circular_distance_deg(raw["rel_wave_dir_deg"].to_numpy(float), 0.0) <= float(half_width_deg)


def bootstrap_mean_ci(
    df: pd.DataFrame,
    value_col: str,
    *,
    vessel_col: str = "vessel_id",
    reps: int = 2000,
    seed: int = DEFAULT_SEED,
) -> Tuple[float, float, float]:
    vessel_values = df.groupby(vessel_col)[value_col].mean()
    vals = vessel_values.to_numpy(float)
    vals = vals[np.isfinite(vals)]
    if not len(vals):
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    boot = np.empty(int(reps), dtype=float)
    for b in range(int(reps)):
        boot[b] = float(np.mean(rng.choice(vals, size=len(vals), replace=True)))
    return float(np.mean(vals)), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def safe_model_cache_path(out_dir: Path, model_name: str, label: str) -> Path:
    return out_dir / "cache" / f"{label}_{model_name.lower()}.joblib"


def save_joblib(obj: Any, path: Path) -> None:
    import joblib
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)


def load_joblib(path: Path) -> Any:
    import joblib
    return joblib.load(path)
