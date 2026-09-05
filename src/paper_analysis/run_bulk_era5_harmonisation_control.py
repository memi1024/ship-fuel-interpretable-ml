#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Paired bulk-carrier ERA5 harmonisation control.

The analysis keeps the Fixed31 cruise cohort, record identities, target, operational
fields, L1 split, XGBoost specification, random seed, and fixed hyperparameters
unchanged while replacing the bulk-carrier environmental representation. It evaluates
record-level ablation, six-vessel bulk LOVO transfer, interventional SHAP directionality,
and fixed-time CII-proxy sensitivity.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, r2_score

try:
    import shap
except Exception as exc:  # pragma: no cover - environment dependent
    shap = None
    _SHAP_IMPORT_ERROR = exc
else:
    _SHAP_IMPORT_ERROR = None


ENV_FEATURES = [
    "rel_wind_speed_kn", "rel_wind_sin", "rel_wind_cos",
    "wave_height_m", "rel_wave_sin", "rel_wave_cos",
    "wave_period_s", "sst_c", "mslp_hpa",
]
AUDITED_SHAP_FEATURES = ["speed_kn", "wave_height_m", "rel_wind_speed_kn"]


def setup_logger(out_dir: Path) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("bulk_era5_control")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    fh = logging.FileHandler(out_dir / "run_bulk_era5_control.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def load_modules(analysis_dir: Path):
    analysis_dir = analysis_dir.resolve()
    if str(analysis_dir) not in sys.path:
        sys.path.insert(0, str(analysis_dir))
    import f31_core_memory_safe as core  # type: ignore
    import run_f31_complete_empirics_v4 as complete  # type: ignore
    return core, complete


def core_args(column_overrides: Path, csv_chunksize: int) -> SimpleNamespace:
    return SimpleNamespace(
        column_overrides=str(column_overrides),
        csv_chunksize=int(csv_chunksize),
        trajectory_gap_minutes=30.0,
    )


def load_official_cruise(core, cruise_csv: Path, column_overrides: Path,
                         csv_chunksize: int, logger: logging.Logger):
    ns = core_args(column_overrides, csv_chunksize)
    raw_csv = core.read_csv_memory_safe(cruise_csv, ns, logger)
    bundle = core.canonicalize_dataframe(raw_csv, ns, logger)
    raw = bundle.raw.copy().reset_index(drop=True)
    raw["row_id"] = np.arange(len(raw), dtype=np.int64)
    X = bundle.feature_df.copy().reset_index(drop=True)
    return raw, X, list(bundle.feature_cols)


def _read_harmonized_header(core, path: Path) -> Tuple[List[str], Dict[str, str | None]]:
    header = pd.read_csv(path, nrows=0)
    mapping = core.resolve_columns(header)
    return list(header.columns), mapping


def load_harmonized_bulk_environment(core, path: Path, logger: logging.Logger) -> pd.DataFrame:
    """Load only matching keys + environmental fields from the harmonised bulk CSV."""
    header, mapping = _read_harmonized_header(core, path)
    required = ["vessel_id", "timestamp", "speed_kn"] + ENV_FEATURES
    missing = [k for k in required if mapping.get(k) is None]
    if missing:
        raise KeyError(
            "Harmonised bulk CSV is missing columns required for paired alignment/control: "
            + ", ".join(missing)
            + ". Expected the output of the bulk ERA5 interpolation producer, including "
              "relative wind/wave sine-cosine fields."
        )
    optional = ["ship_type", "target", "draught_m"]
    keys = required + [k for k in optional if mapping.get(k) is not None]
    usecols = list(dict.fromkeys(mapping[k] for k in keys if mapping.get(k) is not None))
    logger.info("Reading harmonised bulk environment: %s", path)
    h = pd.read_csv(path, usecols=usecols, low_memory=True)

    out = pd.DataFrame(index=h.index)
    for key in keys:
        col = mapping.get(key)
        if col is not None:
            out[key] = h[col]
    out["vessel_id"] = out["vessel_id"].astype(str).str.strip()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce", utc=True)
    if "ship_type" in out:
        out["ship_type"] = core.normalize_ship_type(out["ship_type"])
        out = out[out["ship_type"].eq("bulk")].copy()

    for c in ["speed_kn", "target", "draught_m"] + ENV_FEATURES:
        if c in out:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    # Harmonised interpolation output stores ERA5 surface pressure in Pa. Canonical
    # modelling uses hPa, matching the main Fixed31 pipeline.
    pmed = float(out["mslp_hpa"].dropna().median()) if out["mslp_hpa"].notna().any() else np.nan
    if np.isfinite(pmed) and pmed > 20000:
        out["mslp_hpa"] = out["mslp_hpa"] / 100.0
        logger.info("Converted harmonised pressure from Pa to hPa (median before conversion %.1f).", pmed)

    out = out[out["timestamp"].notna()].copy()
    logger.info("Harmonised candidate rows: %d", len(out))
    return out


def _make_alignment_key(df: pd.DataFrame, use_speed: bool = False) -> pd.Series:
    ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    # int64 nanosecond timestamps are deterministic across CSV timezone spellings.
    ts_ns = ts.astype("int64").astype(str)
    key = df["vessel_id"].astype(str).str.strip() + "|" + ts_ns
    if use_speed:
        sp = pd.to_numeric(df["speed_kn"], errors="coerce").round(5).map(
            lambda x: "nan" if pd.isna(x) else f"{x:.5f}"
        )
        key = key + "|" + sp
    return key


def align_harmonized_to_official(original_raw: pd.DataFrame, harmonized: pd.DataFrame,
                                  min_match_fraction: float, logger: logging.Logger) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (harmonised_full_raw, alignment_audit).

    Matching is first attempted on vessel + UTC timestamp. If either side has duplicate
    keys, speed rounded to 1e-5 kn is added. This preserves official record identities and
    avoids row-order assumptions.
    """
    bulk_mask = original_raw["ship_type"].eq("bulk")
    ob = original_raw.loc[bulk_mask].copy()
    if not len(ob):
        raise ValueError("Official cruise cohort contains no bulk-carrier rows.")

    for use_speed in (False, True):
        ok = _make_alignment_key(ob, use_speed=use_speed)
        hk = _make_alignment_key(harmonized, use_speed=use_speed)
        odup = int(ok.duplicated(keep=False).sum())
        hdup = int(hk.duplicated(keep=False).sum())
        if odup == 0 and hdup == 0:
            left = ob[["row_id", "vessel_id", "timestamp", "speed_kn"]].copy()
            left["_key"] = ok.values
            right = harmonized.copy()
            right["_key"] = hk.values
            env = right[["_key"] + ENV_FEATURES].copy()
            merged = left.merge(env, on="_key", how="left", validate="one_to_one")
            matched = merged[ENV_FEATURES].notna().any(axis=1)
            frac = float(matched.mean())
            logger.info(
                "Paired alignment using %s: matched %d/%d (%.3f%%)",
                "vessel+time+speed" if use_speed else "vessel+time",
                int(matched.sum()), len(merged), frac * 100,
            )
            if frac < min_match_fraction:
                if not use_speed:
                    continue
                raise RuntimeError(
                    f"Harmonised bulk alignment coverage {frac:.4%} is below required "
                    f"{min_match_fraction:.4%}. Verify vessel IDs/timestamps and that the "
                    "harmonised CSV was generated from the same bulk 10-min feature lineage."
                )

            # Important: a row can be correctly matched yet all ERA5 environmental
            # values can be missing. Determine identity matching by key existence rather
            # than environmental nonmissingness for the final audit.
            right_keys = set(env["_key"].astype(str))
            identity_match = merged["_key"].astype(str).isin(right_keys)
            identity_frac = float(identity_match.mean())
            if identity_frac < min_match_fraction:
                raise RuntimeError(
                    f"Identity match coverage {identity_frac:.4%} is below required "
                    f"{min_match_fraction:.4%}."
                )

            full = original_raw.copy()
            # Work in float64 for the replaced environmental block.  The canonical
            # Fixed31 frame may store some columns as float32, while the ERA5
            # interpolation output is normally float64; promoting once avoids dtype
            # warnings and preserves the interpolated precision.
            for c in ENV_FEATURES:
                full[c] = pd.to_numeric(full[c], errors="coerce").astype("float64")
                values = pd.to_numeric(merged.set_index("row_id")[c], errors="coerce")
                idx = values.index.to_numpy(dtype=int)
                full.loc[idx, c] = values.to_numpy(dtype=float)

            audit = pd.DataFrame([{
                "official_bulk_rows": int(len(ob)),
                "harmonized_candidate_rows": int(len(harmonized)),
                "identity_matched_rows": int(identity_match.sum()),
                "identity_match_fraction": identity_frac,
                "rows_with_any_harmonized_environment": int(matched.sum()),
                "matching_key": "vessel+timestamp+speed" if use_speed else "vessel+timestamp",
                "official_duplicate_keys": odup,
                "harmonized_duplicate_keys": hdup,
            }])
            return full, audit

        logger.warning(
            "Alignment key %s has duplicates (official=%d, harmonized=%d); trying stricter key.",
            "vessel+time+speed" if use_speed else "vessel+time", odup, hdup,
        )

    raise RuntimeError("Could not construct a unique paired alignment key for bulk records.")


def load_xgb_params(complete, hyperparams_csv: Path, logger: logging.Logger) -> Dict:
    """Load the locked XGBoost row without requiring unrelated model rows.

    The official registry contains all Table-7 models, but the paired bulk control uses
    only XGBoost.  Reading that row directly keeps this producer independently runnable
    while preserving the exact locked XGBoost specification when the official registry
    is supplied.
    """
    if not hyperparams_csv.is_file():
        raise FileNotFoundError(f"Locked hyperparameter CSV not found: {hyperparams_csv}")
    df = pd.read_csv(hyperparams_csv)
    if not {"model", "best_parameters_json"}.issubset(df.columns):
        raise ValueError("Hyperparameter CSV must contain model and best_parameters_json columns.")
    names = df["model"].map(complete.normalize_model_name)
    rows = df.loc[names.eq("xgb")]
    if rows.empty:
        raise KeyError("Locked hyperparameter registry does not contain XGB.")
    raw = rows.iloc[0]["best_parameters_json"]
    params = {} if pd.isna(raw) or str(raw).strip() in {"", "{}"} else json.loads(str(raw))
    logger.info("Loaded locked XGBoost hyperparameters from %s", hyperparams_csv)
    return params


def get_record_split(core, raw: pd.DataFrame, main_output: Path | None,
                     seed: int, logger: logging.Logger) -> Tuple[np.ndarray, np.ndarray, str]:
    if main_output is not None:
        p = main_output / "14_artifacts" / "record_split_indices.npz"
        if p.is_file():
            z = np.load(p)
            tr = np.asarray(z["train_idx"], dtype=int)
            te = np.asarray(z["test_idx"], dtype=int)
            if len(tr) + len(te) == len(raw):
                logger.info("Reusing official L1 split: %s", p)
                return tr, te, str(p)
            raise ValueError(
                f"Existing split contains {len(tr)+len(te)} rows but cruise cohort has {len(raw)}."
            )
    try:
        tr, te = core.stratified_record_split(raw, 0.20, seed)
        logger.info("Recreated deterministic L1 split from the manuscript-specified strata and seed.")
        return tr, te, "recreated_from_seed"
    except ValueError as exc:
        # The manuscript dataset (489,620 cruise records) has ample observations in
        # every vessel-type x target-quantile stratum.  Tiny synthetic/demo datasets
        # can contain singleton strata, making that exact split mathematically
        # impossible.  Permit a deterministic ship-type-stratified fallback only for
        # small synthetic checks; the study-data path does not use it.
        if len(raw) >= 10000:
            raise
        from sklearn.model_selection import train_test_split

        logger.warning(
            "Full vessel-type x target-quantile stratification is unavailable on this "
            "small demo dataset (%d rows): %s. Falling back to deterministic "
            "ship-type stratification for small synthetic checks only.",
            len(raw), exc,
        )
        idx = np.arange(len(raw))
        strata = raw["ship_type"].astype(str)
        tr, te = train_test_split(
            idx, test_size=0.20, random_state=seed, stratify=strata
        )
        return np.sort(tr), np.sort(te), "small_demo_ship_type_fallback"


def fit_xgb(complete, params: Dict, X: pd.DataFrame, y: np.ndarray,
            train_idx: np.ndarray, seed: int, n_jobs: int):
    model = complete.corrected_model_factory("xgb", params, seed, n_jobs)
    model.fit(X.iloc[train_idx], y[train_idx])
    return model


def rmse(y: np.ndarray, p: np.ndarray) -> float:
    return math.sqrt(mean_squared_error(y, p))


def cluster_bootstrap_rmse_delta(raw_bulk_test: pd.DataFrame, y: np.ndarray,
                                 p_original: np.ndarray, p_harmonized: np.ndarray,
                                 reps: int, seed: int) -> Tuple[float, float]:
    """Vessel-cluster bootstrap CI for harmonised minus original RMSE."""
    df = raw_bulk_test[["vessel_id"]].copy().reset_index(drop=True)
    df["y"] = y
    df["po"] = p_original
    df["ph"] = p_harmonized
    vessels = np.asarray(sorted(df["vessel_id"].astype(str).unique()))
    by = {v: df.index[df["vessel_id"].astype(str).eq(v)].to_numpy() for v in vessels}
    rng = np.random.default_rng(seed)
    vals = np.empty(reps, dtype=float)
    for b in range(reps):
        draw = rng.choice(vessels, size=len(vessels), replace=True)
        idx = np.concatenate([by[v] for v in draw])
        vals[b] = rmse(df.loc[idx, "y"].to_numpy(float), df.loc[idx, "ph"].to_numpy(float)) - \
                  rmse(df.loc[idx, "y"].to_numpy(float), df.loc[idx, "po"].to_numpy(float))
    return float(np.quantile(vals, .025)), float(np.quantile(vals, .975))


def fit_configuration(complete, params: Dict, X: pd.DataFrame, y: np.ndarray,
                      tr: np.ndarray, te: np.ndarray, cols: Sequence[str],
                      seed: int, n_jobs: int):
    m = complete.corrected_model_factory("xgb", params, seed, n_jobs)
    m.fit(X.loc[tr, list(cols)], y[tr])
    return m, np.asarray(m.predict(X.loc[te, list(cols)]), dtype=float)


def run_l1_ablation(core, complete, raw_o, X_o, raw_h, X_h, y, tr, te, params,
                    out: Path, seed: int, n_jobs: int, bootstrap_reps: int, logger):
    bulk_te_mask = raw_o.loc[te, "ship_type"].eq("bulk").to_numpy()
    bulk_te = te[bulk_te_mask]
    rows = []
    preds = {}
    configs = {
        "operational_core": list(core.OPERATIONAL_FEATURES),
        "weather_only": list(core.WEATHER_FEATURES),
        "dynamic_physical": list(core.CANONICAL_FEATURES),
    }
    for cfg, cols in configs.items():
        mo, po_all = fit_configuration(complete, params, X_o, y, tr, te, cols, seed, n_jobs)
        mh, ph_all = fit_configuration(complete, params, X_h, y, tr, te, cols, seed, n_jobs)
        po = po_all[bulk_te_mask]
        ph = ph_all[bulk_te_mask]
        yo = y[bulk_te]
        ro = rmse(yo, po); rh = rmse(yo, ph)
        rows.append({
            "configuration": cfg,
            "original_bulk_RMSE": ro,
            "harmonized_bulk_RMSE": rh,
            "harmonized_minus_original_RMSE": rh - ro,
        })
        preds[cfg] = (mo, mh, po, ph)
        if cfg == "dynamic_physical":
            lo, hi = cluster_bootstrap_rmse_delta(
                raw_o.loc[bulk_te].reset_index(drop=True), yo, po, ph,
                bootstrap_reps, seed + 410,
            )
            rows[-1]["paired_delta_CI95_low"] = lo
            rows[-1]["paired_delta_CI95_high"] = hi
            joblib.dump(mo, out / "bulk_control_original_L1_xgb.joblib")
            joblib.dump(mh, out / "bulk_control_harmonized_L1_xgb.joblib")
            np.save(out / "bulk_control_original_L1_bulk_predictions.npy", po)
            np.save(out / "bulk_control_harmonized_L1_bulk_predictions.npy", ph)

    tab = pd.DataFrame(rows)
    fused_o = float(tab.loc[tab.configuration.eq("dynamic_physical"), "original_bulk_RMSE"].iloc[0])
    fused_h = float(tab.loc[tab.configuration.eq("dynamic_physical"), "harmonized_bulk_RMSE"].iloc[0])
    for cfg in ["operational_core", "weather_only"]:
        m = tab.configuration.eq(cfg)
        tab.loc[m, "original_delta_RMSE_vs_fused"] = tab.loc[m, "original_bulk_RMSE"] - fused_o
        tab.loc[m, "harmonized_delta_RMSE_vs_fused"] = tab.loc[m, "harmonized_bulk_RMSE"] - fused_h
    tab.to_csv(out / "Table_bulk_ERA5_L1_feature_ablation.csv", index=False)
    logger.info("Bulk L1 harmonisation control complete: original %.6f -> harmonized %.6f", fused_o, fused_h)
    return preds["dynamic_physical"][0], preds["dynamic_physical"][1], tab


def run_lovo(core, complete, raw_o, X_o, raw_h, X_h, y, params, out, seed, n_jobs, logger):
    rows = []
    bulk_vessels = sorted(raw_o.loc[raw_o.ship_type.eq("bulk"), "vessel_id"].astype(str).unique())
    for i, vessel in enumerate(bulk_vessels, start=1):
        test = np.flatnonzero(raw_o["vessel_id"].astype(str).eq(vessel).to_numpy())
        train = np.flatnonzero(~raw_o["vessel_id"].astype(str).eq(vessel).to_numpy())
        logger.info("Bulk LOVO %d/%d: %s", i, len(bulk_vessels), vessel)
        mo = fit_xgb(complete, params, X_o, y, train, seed, n_jobs)
        mh = fit_xgb(complete, params, X_h, y, train, seed, n_jobs)
        po = np.asarray(mo.predict(X_o.iloc[test]), dtype=float)
        ph = np.asarray(mh.predict(X_h.iloc[test]), dtype=float)
        yt = y[test]
        sd = float(np.std(yt, ddof=1))
        rows.append({
            "vessel_id": vessel,
            "n": len(test),
            "original_RMSE": rmse(yt, po),
            "harmonized_RMSE": rmse(yt, ph),
            "original_R2": float(r2_score(yt, po)),
            "harmonized_R2": float(r2_score(yt, ph)),
            "original_RMSE_target_SD": rmse(yt, po) / sd if sd > 0 else np.nan,
            "harmonized_RMSE_target_SD": rmse(yt, ph) / sd if sd > 0 else np.nan,
        })
        del mo, mh
    d = pd.DataFrame(rows)
    d.to_csv(out / "Table_bulk_ERA5_LOVO_by_vessel.csv", index=False)
    summary = pd.DataFrame([{
        "vessels": len(d),
        "original_positive_R2_vessels": int((d.original_R2 > 0).sum()),
        "harmonized_positive_R2_vessels": int((d.harmonized_R2 > 0).sum()),
        "original_median_R2": float(d.original_R2.median()),
        "harmonized_median_R2": float(d.harmonized_R2.median()),
        "original_median_RMSE": float(d.original_RMSE.median()),
        "harmonized_median_RMSE": float(d.harmonized_RMSE.median()),
        "original_median_RMSE_target_SD": float(d.original_RMSE_target_SD.median()),
        "harmonized_median_RMSE_target_SD": float(d.harmonized_RMSE_target_SD.median()),
    }])
    summary.to_csv(out / "Table_bulk_ERA5_LOVO_summary.csv", index=False)
    return summary


def deterministic_global_test_sample(te: np.ndarray, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if len(te) <= n:
        return np.asarray(te, dtype=int)
    local = np.sort(rng.choice(len(te), size=n, replace=False))
    return np.asarray(te, dtype=int)[local]


def interventional_shap_values(model, Xeval: pd.DataFrame, background: pd.DataFrame,
                               batch_size: int, logger: logging.Logger) -> np.ndarray:
    if shap is None:
        raise RuntimeError(f"SHAP import failed: {_SHAP_IMPORT_ERROR}")
    explainer = shap.TreeExplainer(
        model, data=background, feature_perturbation="interventional", model_output="raw"
    )
    sv = np.empty((len(Xeval), Xeval.shape[1]), dtype=np.float64)
    for start in range(0, len(Xeval), batch_size):
        stop = min(len(Xeval), start + batch_size)
        vals = explainer.shap_values(Xeval.iloc[start:stop], check_additivity=False)
        if isinstance(vals, list):
            vals = vals[0]
        sv[start:stop] = np.asarray(vals, dtype=float)
        logger.info("  SHAP %d:%d", start, stop)
    return sv


def run_shap_direction(raw_o, X_o, X_h, tr, te, model_o, model_h, out, seed,
                       eval_n, background_n, batch_size, logger):
    global_eval = deterministic_global_test_sample(te, eval_n, seed + 800)
    bulk_eval = global_eval[raw_o.loc[global_eval, "ship_type"].eq("bulk").to_numpy()]
    if not len(bulk_eval):
        raise RuntimeError("Deterministic L1 SHAP sample contains no bulk rows.")
    rng = np.random.default_rng(seed + 801)
    bg = np.sort(rng.choice(tr, size=min(background_n, len(tr)), replace=False))
    Xeo = X_o.iloc[bulk_eval].reset_index(drop=True)
    Xeh = X_h.iloc[bulk_eval].reset_index(drop=True)
    BGo = X_o.iloc[bg].reset_index(drop=True)
    BGh = X_h.iloc[bg].reset_index(drop=True)
    logger.info("Bulk paired SHAP: eval=%d, background=%d", len(bulk_eval), len(bg))
    svo = interventional_shap_values(model_o, Xeo, BGo, batch_size, logger)
    svh = interventional_shap_values(model_h, Xeh, BGh, batch_size, logger)
    rows = []
    for feat in AUDITED_SHAP_FEATURES:
        j = X_o.columns.get_loc(feat)
        ro = float(spearmanr(Xeo[feat], svo[:, j], nan_policy="omit").statistic)
        rh = float(spearmanr(Xeh[feat], svh[:, j], nan_policy="omit").statistic)
        rows.append({"feature": feat, "original_rho": ro, "harmonized_rho": rh, "delta_rho": rh - ro})
    pd.DataFrame(rows).to_csv(out / "Table_bulk_ERA5_SHAP_direction.csv", index=False)
    np.save(out / "bulk_control_original_interventional_SHAP.npy", svo)
    np.save(out / "bulk_control_harmonized_interventional_SHAP.npy", svh)
    pd.DataFrame({"row_id": bulk_eval}).to_csv(out / "bulk_control_SHAP_eval_rows.csv", index=False)
    np.save(out / "bulk_control_SHAP_background_rows.npy", bg)
    return pd.DataFrame(rows)


def aggregate_vessel_cii(g: pd.DataFrame) -> float:
    bw = np.maximum(g["baseline_transport_work"].to_numpy(float), 1e-12)
    sw = np.maximum(g["scenario_transport_work"].to_numpy(float), 1e-12)
    bp = float(np.average(g["baseline_CII_proxy"], weights=bw))
    sp = float(np.average(g["scenario_CII_proxy"], weights=sw))
    return (sp - bp) / bp * 100.0


def bootstrap_ft_delta(vo: pd.DataFrame, vh: pd.DataFrame, reduction: float,
                       reps: int, seed: int) -> Tuple[float, float]:
    a = vo[vo.reduction_pct.eq(reduction)].copy().set_index("vessel_id")
    b = vh[vh.reduction_pct.eq(reduction)].copy().set_index("vessel_id")
    vessels = sorted(set(a.index.astype(str)) & set(b.index.astype(str)))
    if not vessels:
        return np.nan, np.nan
    a.index = a.index.astype(str); b.index = b.index.astype(str)
    rng = np.random.default_rng(seed)
    vals = np.empty(reps, dtype=float)
    for i in range(reps):
        draw = rng.choice(vessels, size=len(vessels), replace=True)
        ao = a.loc[list(draw)].reset_index(drop=True)
        bh = b.loc[list(draw)].reset_index(drop=True)
        vals[i] = aggregate_vessel_cii(bh) - aggregate_vessel_cii(ao)
    return float(np.quantile(vals, .025)), float(np.quantile(vals, .975))


def run_ft(core, raw_o, X_o, raw_h, X_h, tr, te, model_o, model_h, feature_cols,
           out, reductions, default_cf, bootstrap_reps, seed, logger):
    train_o = raw_o.iloc[tr].copy().reset_index(drop=True)
    test_o = raw_o.iloc[te].copy().reset_index(drop=True)
    train_h = raw_h.iloc[tr].copy().reset_index(drop=True)
    test_h = raw_h.iloc[te].copy().reset_index(drop=True)
    vo, so = core.speed_reduction_scenarios(
        model_o, train_o, test_o, feature_cols, reductions, default_cf
    )
    vh, sh = core.speed_reduction_scenarios(
        model_h, train_h, test_h, feature_cols, reductions, default_cf
    )
    vo = vo[vo.ship_type.eq("bulk")].copy()
    vh = vh[vh.ship_type.eq("bulk")].copy()
    vo.to_csv(out / "Table_bulk_ERA5_FT_original_by_vessel.csv", index=False)
    vh.to_csv(out / "Table_bulk_ERA5_FT_harmonized_by_vessel.csv", index=False)
    rows = []
    for r in [x * 100 for x in reductions]:
        go = vo[vo.reduction_pct.eq(r)]
        gh = vh[vh.reduction_pct.eq(r)]
        po = aggregate_vessel_cii(go)
        ph = aggregate_vessel_cii(gh)
        lo, hi = bootstrap_ft_delta(vo, vh, r, bootstrap_reps, seed + int(r * 10) + 900)
        rows.append({
            "reduction_pct": r,
            "original_bulk_FT_CII_change_pct": po,
            "harmonized_bulk_FT_CII_change_pct": ph,
            "harmonized_minus_original_pp": ph - po,
            "paired_delta_CI95_low": lo,
            "paired_delta_CI95_high": hi,
            "original_favourable_vessels": int(go.improved.sum()),
            "harmonized_favourable_vessels": int(gh.improved.sum()),
            "vessels_total": int(len(gh)),
        })
    tab = pd.DataFrame(rows)
    tab.to_csv(out / "Table_bulk_ERA5_FT_CII_summary.csv", index=False)
    return tab


def make_table24_summary(l1: pd.DataFrame, lovo: pd.DataFrame,
                         shap_tab: pd.DataFrame, ft: pd.DataFrame, out: Path):
    fused = l1[l1.configuration.eq("dynamic_physical")].iloc[0]
    ls = lovo.iloc[0]
    st = shap_tab.set_index("feature")
    f15 = ft.loc[np.isclose(ft.reduction_pct, 15.0)].iloc[0]
    rows = [
        {
            "test": "L1 fused RMSE (bulk)",
            "original_control": fused.original_bulk_RMSE,
            "harmonized_result": fused.harmonized_bulk_RMSE,
            "change": fused.harmonized_minus_original_RMSE,
        },
        {
            "test": "L3 LOVO (bulk) median R2",
            "original_control": ls.original_median_R2,
            "harmonized_result": ls.harmonized_median_R2,
            "change": ls.harmonized_median_R2 - ls.original_median_R2,
        },
        {
            "test": "SHAP direction (bulk): speed rho",
            "original_control": st.loc["speed_kn", "original_rho"],
            "harmonized_result": st.loc["speed_kn", "harmonized_rho"],
            "change": st.loc["speed_kn", "delta_rho"],
        },
        {
            "test": "SHAP direction (bulk): wave rho",
            "original_control": st.loc["wave_height_m", "original_rho"],
            "harmonized_result": st.loc["wave_height_m", "harmonized_rho"],
            "change": st.loc["wave_height_m", "delta_rho"],
        },
        {
            "test": "SHAP direction (bulk): wind rho",
            "original_control": st.loc["rel_wind_speed_kn", "original_rho"],
            "harmonized_result": st.loc["rel_wind_speed_kn", "harmonized_rho"],
            "change": st.loc["rel_wind_speed_kn", "delta_rho"],
        },
        {
            "test": "FT CII at 15% (bulk)",
            "original_control": f15.original_bulk_FT_CII_change_pct,
            "harmonized_result": f15.harmonized_bulk_FT_CII_change_pct,
            "change": f15.harmonized_minus_original_pp,
        },
    ]
    pd.DataFrame(rows).to_csv(out / "Table_24_bulk_ERA5_harmonisation_summary.csv", index=False)


def main() -> int:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Reproduce the four paired bulk-carrier ERA5 harmonisation endpoints.",
    )
    here = Path(__file__).resolve().parent
    p.add_argument("--fixed31-cruise", type=Path, required=True)
    p.add_argument("--bulk-harmonized-csv", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--analysis-dir", type=Path, default=here)
    p.add_argument("--column-overrides", type=Path, default=here / "column_overrides_fixed31.json")
    p.add_argument("--hyperparams-csv", type=Path, default=here / "02_best_hyperparameters.csv")
    p.add_argument("--main-output", type=Path, default=None,
                   help="Existing main output. If supplied, reuses its official record_split_indices.npz.")
    p.add_argument("--seed", type=int, default=20260808)
    p.add_argument("--n-jobs", type=int, default=4)
    p.add_argument("--csv-chunksize", type=int, default=25000)
    p.add_argument("--min-match-fraction", type=float, default=0.995)
    p.add_argument("--bootstrap-reps", type=int, default=2000)
    p.add_argument("--shap-eval-n", type=int, default=12000,
                   help="Deterministic global L1 held-out sample size before selecting bulk rows.")
    p.add_argument("--shap-background-n", type=int, default=256)
    p.add_argument("--shap-batch-size", type=int, default=1000)
    p.add_argument("--default-co2-factor", type=float, default=3.114)
    p.add_argument("--skip-lovo", action="store_true")
    p.add_argument("--skip-shap", action="store_true")
    p.add_argument("--skip-ft", action="store_true")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(args.output_dir)
    core, complete = load_modules(args.analysis_dir)

    raw_o, X_o, feature_cols = load_official_cruise(
        core, args.fixed31_cruise, args.column_overrides, args.csv_chunksize, logger
    )
    if len(raw_o) != 489620:
        logger.warning("Official manuscript cruise count is 489,620; current input has %d rows.", len(raw_o))
    h = load_harmonized_bulk_environment(core, args.bulk_harmonized_csv, logger)
    raw_h, audit = align_harmonized_to_official(raw_o, h, args.min_match_fraction, logger)
    audit.to_csv(args.output_dir / "bulk_ERA5_alignment_audit.csv", index=False)
    X_h = raw_h[feature_cols].copy()

    # Direct paired-integrity checks: target and operational variables are never replaced.
    for c in ["target", "speed_kn", "draught_m", "trim_m", "rudder_deg", "vessel_id", "timestamp"]:
        a = raw_o[c]
        b = raw_h[c]
        if pd.api.types.is_numeric_dtype(a):
            same = np.allclose(a.to_numpy(float), b.to_numpy(float), equal_nan=True)
        else:
            same = a.astype(str).equals(b.astype(str))
        if not same:
            raise AssertionError(f"Paired integrity violation: non-environmental field changed: {c}")

    params = load_xgb_params(complete, args.hyperparams_csv, logger)
    tr, te, split_source = get_record_split(core, raw_o, args.main_output, args.seed, logger)
    y = raw_o["target"].to_numpy(float)
    np.savez_compressed(args.output_dir / "paired_record_split_indices.npz", train_idx=tr, test_idx=te)

    model_o, model_h, l1 = run_l1_ablation(
        core, complete, raw_o, X_o, raw_h, X_h, y, tr, te, params,
        args.output_dir, args.seed, args.n_jobs, args.bootstrap_reps, logger,
    )

    if args.skip_lovo:
        lovo = pd.DataFrame([{
            "original_median_R2": np.nan, "harmonized_median_R2": np.nan,
            "original_positive_R2_vessels": np.nan, "harmonized_positive_R2_vessels": np.nan,
        }])
    else:
        lovo = run_lovo(
            core, complete, raw_o, X_o, raw_h, X_h, y, params,
            args.output_dir, args.seed, args.n_jobs, logger,
        )

    if args.skip_shap:
        shap_tab = pd.DataFrame([
            {"feature": f, "original_rho": np.nan, "harmonized_rho": np.nan, "delta_rho": np.nan}
            for f in AUDITED_SHAP_FEATURES
        ])
    else:
        shap_tab = run_shap_direction(
            raw_o, X_o, X_h, tr, te, model_o, model_h, args.output_dir,
            args.seed, args.shap_eval_n, args.shap_background_n, args.shap_batch_size, logger,
        )

    if args.skip_ft:
        ft = pd.DataFrame([{
            "reduction_pct": 15.0,
            "original_bulk_FT_CII_change_pct": np.nan,
            "harmonized_bulk_FT_CII_change_pct": np.nan,
            "harmonized_minus_original_pp": np.nan,
        }])
    else:
        ft = run_ft(
            core, raw_o, X_o, raw_h, X_h, tr, te, model_o, model_h, feature_cols,
            args.output_dir, [0.05, 0.10, 0.15], args.default_co2_factor,
            args.bootstrap_reps, args.seed, logger,
        )

    make_table24_summary(l1, lovo, shap_tab, ft, args.output_dir)

    manifest = {
        "producer": Path(__file__).name,
        "method": "paired bulk-carrier ERA5 harmonisation control",
        "fixed31_cruise": str(args.fixed31_cruise),
        "bulk_harmonized_csv": str(args.bulk_harmonized_csv),
        "split_source": split_source,
        "seed": args.seed,
        "n_jobs": args.n_jobs,
        "bootstrap_reps": args.bootstrap_reps,
        "shap_eval_n_global": args.shap_eval_n,
        "shap_background_n": args.shap_background_n,
        "environment_features_replaced": ENV_FEATURES,
        "operational_and_target_fields_held_fixed": True,
        "xgboost_native_missing_routing": True,
    }
    (args.output_dir / "bulk_ERA5_control_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    logger.info("Bulk ERA5 harmonisation control complete: %s", args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
