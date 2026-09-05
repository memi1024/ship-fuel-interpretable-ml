#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Statistical inference and local physical-audit tables.

Outputs include model-choice CII aggregation, H1 L2/L3 versus L1 inference, H3 endpoint
confidence intervals and ship-type contrasts, full-support wave-height localisation, and
pooled and vessel-balanced rudder estimands.
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import warnings
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

EXPECTED_CRUISE = 489_620
EXPECTED_TRAIN = 391_696
EXPECTED_TEST = 97_924
EXPECTED_TEMPORAL_TEST = 97_934
EXPECTED_VESSELS = 21

MODEL_NAMES = ["lr", "ridge_interaction", "dt", "rf", "xgb", "lgbm", "ann"]
DISPLAY_NAMES = {
    "lr": "LR",
    "ridge_interaction": "Ridge-Interaction",
    "dt": "DT",
    "rf": "RF",
    "xgb": "XGB",
    "lgbm": "LGBM",
    "ann": "ANN",
}


class Logger:
    def info(self, msg, *args):
        print(msg % args if args else msg)

    def warning(self, msg, *args):
        print("WARNING: " + (msg % args if args else msg))


def atomic_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def normalize_ship_type(x) -> str:
    s = str(x).strip().lower()
    if "bulk" in s:
        return "bulk"
    if "container" in s:
        return "container"
    if "tank" in s:
        return "tanker"
    return s


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_canonical_data(input_csv: Path, core, column_overrides: Path, chunksize: int):
    ns = SimpleNamespace(
        column_overrides=str(column_overrides),
        trajectory_gap_minutes=30.0,
        csv_chunksize=int(chunksize),
    )
    df = core.read_csv_memory_safe(input_csv, ns, Logger())
    bundle = core.canonicalize_dataframe(df, ns, Logger())
    raw = bundle.raw.reset_index(drop=True)
    X = bundle.feature_df.reset_index(drop=True)
    if "row_id" not in raw.columns:
        raw["row_id"] = np.arange(len(raw), dtype=np.int64)
    raw["ship_type"] = raw["ship_type"].map(normalize_ship_type)

    if len(raw) != EXPECTED_CRUISE:
        raise AssertionError(f"cruise rows: {len(raw)} != {EXPECTED_CRUISE}")
    if raw["vessel_id"].nunique() != EXPECTED_VESSELS:
        raise AssertionError(f"vessels: {raw['vessel_id'].nunique()} != {EXPECTED_VESSELS}")
    return raw, X


def known_vessel_temporal_split(raw: pd.DataFrame, frac: float = 0.8):
    train_idx, test_idx = [], []
    for _, inds in raw.groupby("vessel_id").groups.items():
        sub = raw.loc[inds].sort_values(["timestamp", "row_id"], kind="mergesort")
        cut = max(1, min(len(sub) - 1, int(math.floor(len(sub) * frac))))
        train_idx.extend(sub.index[:cut].tolist())
        test_idx.extend(sub.index[cut:].tolist())
    return np.asarray(sorted(train_idx), dtype=int), np.asarray(sorted(test_idx), dtype=int)


def load_locked_artifacts(raw, X, main_output: Path, revision_output: Path):
    art = main_output / "14_artifacts"
    split = np.load(art / "record_split_indices.npz")
    tr = np.asarray(split["train_idx"], dtype=int)
    te = np.asarray(split["test_idx"], dtype=int)
    if len(tr) != EXPECTED_TRAIN or len(te) != EXPECTED_TEST:
        raise AssertionError("L1 split size changed")

    model = joblib.load(art / "xgb_record_split.joblib")
    l1_pred = np.asarray(np.load(art / "xgb_record_test_prediction.npy"), dtype=float)
    current = np.asarray(model.predict(X.iloc[te]), dtype=float)
    if not np.allclose(l1_pred, current, rtol=1e-7, atol=1e-9, equal_nan=True):
        raise AssertionError("L1 model/prediction alignment failed")

    lovo_pred = np.asarray(np.load(main_output / "05_lovo" / "LOVO_predictions_xgb.npy"), dtype=float)
    temporal_pred = np.asarray(
        np.load(revision_output / "02_temporal" / "temporal_prediction_xgb.npy"),
        dtype=float,
    )
    temporal_tr, temporal_te = known_vessel_temporal_split(raw)

    if lovo_pred.shape != (len(raw),):
        raise AssertionError("LOVO prediction length changed")
    if len(temporal_te) != EXPECTED_TEMPORAL_TEST or temporal_pred.shape != (len(temporal_te),):
        raise AssertionError("L2 temporal prediction length changed")

    return {
        "record_tr": tr,
        "record_te": te,
        "model": model,
        "l1_pred": l1_pred,
        "lovo_pred": lovo_pred,
        "temporal_tr": temporal_tr,
        "temporal_te": temporal_te,
        "temporal_pred": temporal_pred,
    }


def metric_dict(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return {
        "n": int(len(y)),
        "RMSE": float(np.sqrt(mean_squared_error(y, p))),
        "MAE": float(mean_absolute_error(y, p)),
        "R2": float(r2_score(y, p)),
        "SSE": float(np.sum((p - y) ** 2)),
    }


def per_vessel_metrics(raw_subset, pred, validation):
    z = raw_subset.reset_index(drop=True)
    pred = np.asarray(pred, dtype=float)
    rows = []
    for vessel, idx in z.groupby("vessel_id").groups.items():
        ii = np.asarray(list(idx), dtype=int)
        y = z.loc[ii, "target"].to_numpy(float)
        p = pred[ii]
        rows.append({
            "validation": validation,
            "vessel_id": str(vessel),
            "ship_type": normalize_ship_type(z.loc[ii[0], "ship_type"]),
            **metric_dict(y, p),
        })
    return pd.DataFrame(rows)


def pooled_rmse_bootstrap_difference(a_byv, b_byv, reps, seed):
    a = a_byv.set_index("vessel_id")
    b = b_byv.set_index("vessel_id")
    common = sorted(set(a.index).intersection(b.index))
    a = a.loc[common]
    b = b.loc[common]
    n_a, s_a = a["n"].to_numpy(float), a["SSE"].to_numpy(float)
    n_b, s_b = b["n"].to_numpy(float), b["SSE"].to_numpy(float)
    rmse_a = math.sqrt(s_a.sum() / n_a.sum())
    rmse_b = math.sqrt(s_b.sum() / n_b.sum())
    rng = np.random.default_rng(seed)
    boot = np.empty(reps, dtype=float)
    k = len(common)
    for i in range(reps):
        draw = rng.integers(0, k, size=k)
        ra = math.sqrt(s_a[draw].sum() / n_a[draw].sum())
        rb = math.sqrt(s_b[draw].sum() / n_b[draw].sum())
        boot[i] = rb - ra
    return rmse_a, rmse_b, rmse_b - rmse_a, float(np.quantile(boot, .025)), float(np.quantile(boot, .975))


def signflip_test(a_byv, b_byv, reps, seed):
    a = a_byv.set_index("vessel_id")
    b = b_byv.set_index("vessel_id")
    common = sorted(set(a.index).intersection(b.index))
    d = b.loc[common, "RMSE"].to_numpy(float) - a.loc[common, "RMSE"].to_numpy(float)
    obs = float(d.mean())
    rng = np.random.default_rng(seed)
    exceed = 0
    done = 0
    chunk = 10_000
    while done < reps:
        r = min(chunk, reps - done)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(r, len(d)), replace=True)
        stats = (signs * d[None, :]).mean(axis=1)
        exceed += int(np.sum(stats >= obs))
        done += r
    return obs, float((exceed + 1.0) / (reps + 1.0)), len(common)


def run_h1(raw, locked, output_dir: Path, bootstrap: int, permutations: int, seed: int):
    l1 = per_vessel_metrics(raw.iloc[locked["record_te"]], locked["l1_pred"], "L1")
    l2 = per_vessel_metrics(raw.iloc[locked["temporal_te"]], locked["temporal_pred"], "L2")
    l3 = per_vessel_metrics(raw, locked["lovo_pred"], "L3")

    rows = []
    for j, (other, name) in enumerate([(l2, "L2-L1"), (l3, "L3-L1")]):
        ra, rb, delta, lo, hi = pooled_rmse_bootstrap_difference(
            l1, other, bootstrap, seed + 100 + j
        )
        mean_delta, p, n_vessels = signflip_test(
            l1, other, permutations, seed + 120 + j
        )
        rows.append({
            "contrast": name,
            "n_vessels": n_vessels,
            "L1_pooled_RMSE": ra,
            "comparison_pooled_RMSE": rb,
            "delta_RMSE": delta,
            "vessel_cluster_bootstrap_CI95_low": lo,
            "vessel_cluster_bootstrap_CI95_high": hi,
            "mean_per_vessel_delta_RMSE": mean_delta,
            "one_sided_signflip_p": p,
            "permutations": permutations,
        })
    atomic_csv(pd.DataFrame(rows), output_dir / "Table_H1_RMSE_decay_inference.csv")


def run_cii_model_choice(raw, locked, core, main_output: Path):
    out = main_output / "11_cii"
    out.mkdir(parents=True, exist_ok=True)
    raw_test = raw.iloc[locked["record_te"]].reset_index(drop=True)
    rows = []
    for name in MODEL_NAMES:
        pred = np.asarray(
            np.load(main_output / "14_artifacts" / f"{name}_record_test_prediction.npy"),
            dtype=float,
        )
        v = core.compute_cii_by_vessel(raw_test, pred, 3.114)
        v.insert(0, "model", DISPLAY_NAMES[name])
        atomic_csv(v, out / f"Table_C1_CII_proxy_by_vessel_{name}.csv")
        rows.append({"model": DISPLAY_NAMES[name], **core.cii_summary(v, raw_test, pred)})
    atomic_csv(pd.DataFrame(rows), out / "Table_C2_model_choice_CII_proxy_summary.csv")


def ft_weighted_cii_change(g):
    b_f = g["baseline_predicted_fuel_t"].sum()
    b_tw = g["baseline_transport_work"].sum()
    f_f = g["FT_predicted_fuel_t"].sum()
    f_tw = g["FT_transport_work"].sum()
    return ((f_f / f_tw) / (b_f / b_tw) - 1.0) * 100.0


def fd_total_fuel_change(g):
    return (g["FD_predicted_total_fuel_t"].sum() / g["baseline_predicted_fuel_t"].sum() - 1.0) * 100.0


def bootstrap_vessel_aggregate(g, stat_fn, reps, seed):
    vessels = sorted(g["vessel_id"].astype(str).unique())
    by = {v: g[g["vessel_id"].astype(str) == v].copy() for v in vessels}
    point = float(stat_fn(g))
    rng = np.random.default_rng(seed)
    vals = np.empty(reps, dtype=float)
    for i in range(reps):
        draw = rng.choice(vessels, size=len(vessels), replace=True)
        vals[i] = stat_fn(pd.concat([by[v] for v in draw], ignore_index=True))
    return point, float(np.quantile(vals, .025)), float(np.quantile(vals, .975))


def independent_bootstrap_contrast(ga, gb, stat_fn, reps, seed):
    va = sorted(ga["vessel_id"].astype(str).unique())
    vb = sorted(gb["vessel_id"].astype(str).unique())
    a = {v: ga[ga["vessel_id"].astype(str) == v].copy() for v in va}
    b = {v: gb[gb["vessel_id"].astype(str) == v].copy() for v in vb}
    point = float(stat_fn(ga) - stat_fn(gb))
    rng = np.random.default_rng(seed)
    vals = np.empty(reps, dtype=float)
    for i in range(reps):
        da = rng.choice(va, size=len(va), replace=True)
        db = rng.choice(vb, size=len(vb), replace=True)
        vals[i] = stat_fn(pd.concat([a[v] for v in da], ignore_index=True)) - stat_fn(
            pd.concat([b[v] for v in db], ignore_index=True)
        )
    return point, float(np.quantile(vals, .025)), float(np.quantile(vals, .975))


def cluster_robust_interaction(df, outcome):
    x = df.copy()
    x["reduction_cat"] = x["reduction_pct"].astype(str)
    fit = smf.ols(f"{outcome} ~ C(reduction_cat) * C(ship_type)", data=x).fit(
        cov_type="cluster", cov_kwds={"groups": x["vessel_id"].astype(str)}
    )
    names = list(fit.params.index)
    idx = [i for i, name in enumerate(names) if ":" in name]
    R = np.zeros((len(idx), len(names)))
    for r, i in enumerate(idx):
        R[r, i] = 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        test = fit.f_test(R)
    return float(np.asarray(test.fvalue).squeeze()), float(np.asarray(test.pvalue).squeeze()), float(test.df_num), float(test.df_denom)


def ordinary_interaction_f(df, outcome):
    y = df[outcome].to_numpy(float)
    reductions = sorted(df["reduction_pct"].unique())
    ship_types = ["bulk", "container", "tanker"]
    r_dummies = [(df["reduction_pct"].to_numpy(float) == float(r)).astype(float) for r in reductions[1:]]
    s_dummies = [(df["ship_type"].astype(str).to_numpy() == s).astype(float) for s in ship_types[1:]]
    reduced = [np.ones(len(df))] + r_dummies + s_dummies
    full = list(reduced) + [rd * sd for rd in r_dummies for sd in s_dummies]
    Xr, Xf = np.column_stack(reduced), np.column_stack(full)
    br, *_ = np.linalg.lstsq(Xr, y, rcond=None)
    bf, *_ = np.linalg.lstsq(Xf, y, rcond=None)
    er, ef = y - Xr @ br, y - Xf @ bf
    ssr_r, ssr_f = float(er @ er), float(ef @ ef)
    df_num, df_den = Xf.shape[1] - Xr.shape[1], len(y) - Xf.shape[1]
    return float(((ssr_r - ssr_f) / df_num) / (ssr_f / df_den))


def interaction_permutation(df, outcome, reps, seed):
    vessels = df[["vessel_id", "ship_type"]].drop_duplicates().sort_values("vessel_id")
    vids = vessels["vessel_id"].astype(str).to_numpy()
    labels = vessels["ship_type"].astype(str).to_numpy()
    observed = ordinary_interaction_f(df, outcome)
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(reps):
        mapping = dict(zip(vids, rng.permutation(labels)))
        z = df.copy()
        z["ship_type"] = z["vessel_id"].astype(str).map(mapping)
        exceed += int(ordinary_interaction_f(z, outcome) >= observed)
    return observed, float((exceed + 1) / (reps + 1))


def run_h3(scenario_output: Path, output_dir: Path, bootstrap: int, permutations: int, seed: int):
    ft = pd.read_csv(scenario_output / "Table_15_FT_L1_aligned_by_vessel.csv")
    fd = pd.read_csv(
        scenario_output / "PriorityA_FD_fuel_time" / "Table_16_FD_fuel_time_decomposition_by_vessel.csv"
    )
    ft["ship_type"] = ft["ship_type"].map(normalize_ship_type)
    fd["ship_type"] = fd["ship_type"].map(normalize_ship_type)

    endpoint_rows, pair_rows = [], []
    types = ["bulk", "container", "tanker"]
    analyses = [
        ("FT_CII_proxy_change_pct", ft, ft_weighted_cii_change),
        ("FD_total_fuel_change_pct", fd, fd_total_fuel_change),
    ]
    for oi, (outcome, data, stat_fn) in enumerate(analyses):
        for ri, reduction in enumerate(sorted(data["reduction_pct"].unique())):
            z = data[data["reduction_pct"] == reduction].copy()
            for si, scope in enumerate(["Fleet"] + types):
                g = z if scope == "Fleet" else z[z["ship_type"] == scope]
                point, lo, hi = bootstrap_vessel_aggregate(
                    g, stat_fn, bootstrap, seed + 2000 + oi * 500 + ri * 100 + si
                )
                endpoint_rows.append({
                    "outcome": outcome,
                    "reduction_pct": reduction,
                    "scope": scope,
                    "n_vessels": int(g["vessel_id"].nunique()),
                    "estimate_pct": point,
                    "vessel_bootstrap_CI95_low": lo,
                    "vessel_bootstrap_CI95_high": hi,
                })
            for ai, a in enumerate(types):
                for b in types[ai + 1:]:
                    ga, gb = z[z["ship_type"] == a], z[z["ship_type"] == b]
                    point, lo, hi = independent_bootstrap_contrast(
                        ga, gb, stat_fn, bootstrap,
                        seed + 3000 + oi * 500 + ri * 100 + ai * 10 + types.index(b),
                    )
                    pair_rows.append({
                        "outcome": outcome,
                        "reduction_pct": reduction,
                        "contrast": f"{a}-{b}",
                        "estimate_difference_pp": point,
                        "vessel_bootstrap_CI95_low": lo,
                        "vessel_bootstrap_CI95_high": hi,
                    })

    atomic_csv(pd.DataFrame(endpoint_rows), output_dir / "Table_H3_endpoint_vessel_bootstrap_CI.csv")
    atomic_csv(pd.DataFrame(pair_rows), output_dir / "Table_H3_pairwise_shiptype_contrasts.csv")

    ft_reg = ft[["vessel_id", "ship_type", "reduction_pct", "FT_CII_change_pct"]].copy()
    fd_reg = fd[["vessel_id", "ship_type", "reduction_pct", "FD_total_fuel_change_pct_recomputed"]].rename(
        columns={"FD_total_fuel_change_pct_recomputed": "FD_total_fuel_change_pct"}
    )
    global_rows = []
    for j, (z, outcome) in enumerate([
        (ft_reg, "FT_CII_change_pct"),
        (fd_reg, "FD_total_fuel_change_pct"),
    ]):
        F, p, df_num, df_den = cluster_robust_interaction(z, outcome)
        perm_F, perm_p = interaction_permutation(z, outcome, permutations, seed + 4000 + j)
        global_rows.append({
            "outcome": outcome,
            "cluster_robust_F": F,
            "df_num": df_num,
            "df_denom": df_den,
            "cluster_robust_p": p,
            "permutation_F": perm_F,
            "permutation_p": perm_p,
            "permutations": permutations,
        })
    atomic_csv(pd.DataFrame(global_rows), output_dir / "Table_H3_shiptype_x_reduction_interaction.csv")


def relative_sector(sin_v, cos_v):
    angle = (np.degrees(np.arctan2(np.asarray(sin_v), np.asarray(cos_v))) + 360.0) % 360.0
    out = np.full(len(angle), "cross", dtype=object)
    out[(angle <= 45.0) | (angle >= 315.0)] = "head"
    out[(angle >= 135.0) & (angle <= 225.0)] = "following"
    return out


def vessel_balanced_wave_summary(g, reps, seed):
    by = g.groupby("vessel_id")["delta"]
    vessel_expected = by.apply(lambda s: float((s > 0).mean())).to_numpy(float)
    rng = np.random.default_rng(seed)
    boot = np.empty(reps, dtype=float)
    for i in range(reps):
        draw = rng.integers(0, len(vessel_expected), size=len(vessel_expected))
        boot[i] = vessel_expected[draw].mean()
    return (
        100.0 * float(vessel_expected.mean()),
        100.0 * float(np.quantile(boot, .025)),
        100.0 * float(np.quantile(boot, .975)),
    )


def wave_row(label, g, bootstrap, seed):
    vb, lo, hi = vessel_balanced_wave_summary(g, bootstrap, seed)
    return {
        "subgroup": label,
        "n": int(len(g)),
        "n_vessels": int(g["vessel_id"].nunique()),
        "expected_direction_pct": 100.0 * float((g["delta"] > 0).mean()),
        "vessel_balanced_expected_pct": vb,
        "vessel_balanced_CI95_low": lo,
        "vessel_balanced_CI95_high": hi,
    }


def run_wave_localisation(raw, X, locked, core, output_dir: Path, bootstrap: int, seed: int):
    train = raw.iloc[locked["record_tr"]].reset_index(drop=True)
    test = raw.iloc[locked["record_te"]].reset_index(drop=True)
    support = train.groupby("ship_type")["wave_height_m"].agg(["min", "max"])
    perturbed = test["wave_height_m"].to_numpy(float) * 1.10
    supported = np.zeros(len(test), dtype=bool)
    st = test["ship_type"].astype(str).to_numpy()
    for ship_type, limits in support.iterrows():
        m = st == ship_type
        supported[m] = (perturbed[m] >= float(limits["min"])) & (perturbed[m] <= float(limits["max"]))

    base = test.loc[supported].reset_index(drop=True).copy()
    pert = base.copy()
    pert["wave_height_m"] = pert["wave_height_m"].astype(float) * 1.10
    pred_pert = np.asarray(
        locked["model"].predict(core.feature_matrix_from_raw(pert, list(X.columns))), dtype=float
    )
    rows = base[["vessel_id", "ship_type", "rel_wave_sin", "rel_wave_cos"]].copy()
    rows["delta"] = pred_pert - locked["l1_pred"][supported]
    rows["relative_wave_sector"] = relative_sector(rows["rel_wave_sin"], rows["rel_wave_cos"])

    groups = [
        ("Global", rows),
        ("Following", rows[rows["relative_wave_sector"] == "following"]),
        ("Cross", rows[rows["relative_wave_sector"] == "cross"]),
        ("Head", rows[rows["relative_wave_sector"] == "head"]),
        ("Tanker × head", rows[(rows["ship_type"] == "tanker") & (rows["relative_wave_sector"] == "head")]),
    ]
    out = [wave_row(label, g, bootstrap, seed + 5000 + i) for i, (label, g) in enumerate(groups)]
    atomic_csv(pd.DataFrame(out), output_dir / "Table_WAVE_full_support_localisation.csv")


def pooled_cluster_boot_mean(df, reps, seed):
    vessels = np.unique(df["vessel_id"].astype(str).to_numpy())
    by = {v: df.loc[df["vessel_id"].astype(str).eq(v), "delta"].to_numpy(float) for v in vessels}
    rng = np.random.default_rng(seed)
    vals = np.empty(reps, dtype=float)
    for i in range(reps):
        draw = rng.choice(vessels, size=len(vessels), replace=True)
        vals[i] = float(np.mean(np.concatenate([by[v] for v in draw])))
    return float(df["delta"].mean()), float(np.quantile(vals, .025)), float(np.quantile(vals, .975))


def vessel_balanced_boot_mean(df, reps, seed):
    means = df.groupby("vessel_id")["delta"].mean().to_numpy(float)
    rng = np.random.default_rng(seed)
    vals = np.empty(reps, dtype=float)
    for i in range(reps):
        vals[i] = float(np.mean(rng.choice(means, size=len(means), replace=True)))
    return float(means.mean()), float(np.quantile(vals, .025)), float(np.quantile(vals, .975))


def run_rudder(revision_output: Path, output_dir: Path, bootstrap: int, seed: int):
    rows = pd.read_csv(revision_output / "08_rudder_power" / "rudder_plus1_row_level.csv.gz")
    rows["vessel_id"] = rows["vessel_id"].astype(str)
    out = []
    for i, threshold in enumerate([0.0, 0.5, 1.0, 2.0]):
        g = rows if threshold == 0.0 else rows[rows["abs_rudder_original"] > threshold]
        pooled, plo, phi = pooled_cluster_boot_mean(g, bootstrap, seed + 6000 + i)
        balanced, blo, bhi = vessel_balanced_boot_mean(g, bootstrap, seed + 6100 + i)
        out.append({
            "threshold": "all" if threshold == 0.0 else f"|rudder|>{threshold:g}°",
            "n": int(len(g)),
            "n_vessels": int(g["vessel_id"].nunique()),
            "pooled_mean_delta": pooled,
            "pooled_vessel_cluster_CI95_low": plo,
            "pooled_vessel_cluster_CI95_high": phi,
            "vessel_balanced_mean_delta": balanced,
            "vessel_balanced_bootstrap_CI95_low": blo,
            "vessel_balanced_bootstrap_CI95_high": bhi,
        })
    atomic_csv(pd.DataFrame(out), output_dir / "Table_RUDDER_dual_estimand.csv")


def main() -> int:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--input-csv", type=Path, required=True)
    p.add_argument("--main-output", type=Path, required=True)
    p.add_argument("--revision-output", type=Path, required=True)
    p.add_argument("--scenario-output", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--core-path", type=Path, required=True)
    p.add_argument("--column-overrides", type=Path, required=True)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--permutations", type=int, default=100000)
    p.add_argument("--seed", type=int, default=20260819)
    p.add_argument("--csv-chunksize", type=int, default=25000)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    core = load_module(args.core_path, "f31_core_final_inference")
    raw, X = load_canonical_data(args.input_csv, core, args.column_overrides, args.csv_chunksize)
    locked = load_locked_artifacts(raw, X, args.main_output, args.revision_output)

    run_cii_model_choice(raw, locked, core, args.main_output)
    run_h1(raw, locked, args.output_dir, args.bootstrap, args.permutations, args.seed)
    run_h3(args.scenario_output, args.output_dir, args.bootstrap, args.permutations, args.seed)
    run_wave_localisation(raw, X, locked, core, args.output_dir, args.bootstrap, args.seed)
    run_rudder(args.revision_output, args.output_dir, args.bootstrap, args.seed)

    print("Final manuscript inference complete:", args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
