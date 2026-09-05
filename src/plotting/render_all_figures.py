# -*- coding: utf-8 -*-
"""Render manuscript Figures 1--11 from the analysis outputs."""
from __future__ import annotations

import argparse
import importlib.util
import logging
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess


SHIP_ORDER = ["bulk", "container", "tanker"]
RUNG_ORDER = ["L1", "L2", "L3", "L4"]


def setup_logger() -> logging.Logger:
    log = logging.getLogger("figure_renderer")
    if not log.handlers:
        log.addHandler(logging.StreamHandler(sys.stdout))
    log.setLevel(logging.INFO)
    return log


def load_core(path: Path):
    spec = importlib.util.spec_from_file_location("f31_core_plot", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_canonical_cruise(csv_path: Path, core_path: Path, overrides: Path) -> pd.DataFrame:
    core = load_core(core_path)
    log = setup_logger()
    ns = SimpleNamespace(
        column_overrides=str(overrides),
        trajectory_gap_minutes=30.0,
        csv_chunksize=25000,
    )
    df = core.read_csv_memory_safe(csv_path, ns, log)
    bundle = core.canonicalize_dataframe(df, ns, log)
    raw = bundle.raw.reset_index(drop=True)
    if "phase" in raw.columns:
        raw = raw.loc[raw["phase"].astype(str).eq("cruise")].reset_index(drop=True)
    return raw


def save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("[figure]", path)


def missing(path: Path, strict: bool = True) -> bool:
    if path.exists():
        return False
    raise FileNotFoundError(f"Required figure input not found: {path}")


def fig1_framework(out: Path):
    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.axis("off")
    boxes = [
        (0.08, 0.67, "Layer 1\nPredictive validity\nL1–L4 generalisation"),
        (0.39, 0.67, "Layer 2\nExplanatory validity\nSHAP + physical audit"),
        (0.70, 0.67, "Layer 3\nOperational validity\nCII proxy + FT/FD"),
    ]
    for x, y, text in boxes:
        ax.text(x, y, text, transform=ax.transAxes, ha="left", va="center",
                bbox=dict(boxstyle="round,pad=0.6", fc="white", ec="black"), fontsize=11)
    ax.annotate("", xy=(0.38, 0.67), xytext=(0.28, 0.67), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(0.69, 0.67), xytext=(0.59, 0.67), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="->"))
    ax.text(0.5, 0.31,
            "Review checkpoints: transferability → physical consistency → operational sensitivity",
            transform=ax.transAxes, ha="center", fontsize=11)
    ax.text(0.5, 0.18,
            "Non-confirmation is retained as a boundary condition rather than re-labelled as confirmation.",
            transform=ax.transAxes, ha="center", fontsize=9)
    ax.set_title("Three-layer sequential validation framework", fontsize=13)
    save(fig, out / "Figure_1_three_layer_framework.png")


def fig2_correlation(raw: pd.DataFrame, out: Path):
    cols = [
        "target", "speed_kn", "draught_m", "trim_m", "rudder_deg",
        "rel_wind_speed_kn", "wave_height_m", "wave_period_s", "mslp_hpa", "sst_c",
    ]
    labels = ["Fuel", "SOG", "Draught", "Trim", "Rudder", "Rel. wind", "Wave H", "Wave T", "Pressure", "SST"]
    d = raw[cols].apply(pd.to_numeric, errors="coerce")
    c = d.corr(method="pearson")
    fig, ax = plt.subplots(figsize=(8.5, 7.3))
    im = ax.imshow(c.to_numpy(), vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{c.iloc[i,j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson r")
    ax.set_title("Pearson correlation matrix: cruising cohort")
    save(fig, out / "Figure_2_correlation_matrix.png")


def fig3_distributions(raw: pd.DataFrame, out: Path):
    specs = [
        ("speed_kn", "SOG (kn)"),
        ("draught_m", "Mean draught (m)"),
        ("wave_height_m", "Significant wave height (m)"),
        ("rel_wind_speed_kn", "Relative wind speed (kn)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    y = pd.to_numeric(raw["target"], errors="coerce").to_numpy(float)
    for ax, (col, label) in zip(axes.ravel(), specs):
        x = pd.to_numeric(raw[col], errors="coerce").to_numpy(float)
        m = np.isfinite(x) & np.isfinite(y)
        hb = ax.hexbin(x[m], y[m], gridsize=55, mincnt=1, bins="log", cmap="viridis")
        ax.set_xlabel(label)
        ax.set_ylabel("Fuel (t/10 min)")
        fig.colorbar(hb, ax=ax, label="log count")
    fig.suptitle("Joint distributions of key predictors and ten-minute fuel consumption")
    fig.tight_layout()
    save(fig, out / "Figure_3_key_predictor_distributions.png")


def bootstrap_absolute_rmse(long: pd.DataFrame, reps: int, seed: int):
    rng = np.random.default_rng(seed)
    rows = []
    for cfg, g in long.groupby("configuration"):
        stats = []
        vessels = sorted(g["vessel_id"].astype(str).unique())
        for v in vessels:
            z = g[g["vessel_id"].astype(str) == v]
            e = z["prediction"].to_numpy(float) - z["target"].to_numpy(float)
            stats.append((len(z), float(np.dot(e, e))))
        n = np.asarray([s[0] for s in stats], dtype=float)
        sse = np.asarray([s[1] for s in stats], dtype=float)
        vals = np.empty(reps)
        for b in range(reps):
            draw = rng.integers(0, len(vessels), size=len(vessels))
            vals[b] = math.sqrt(float(sse[draw].sum() / n[draw].sum()))
        rows.append({
            "configuration": cfg,
            "rmse_boot_low": float(np.quantile(vals, .025)),
            "rmse_boot_high": float(np.quantile(vals, .975)),
        })
    return pd.DataFrame(rows)


def fig4_ablation(revision: Path, out: Path, seed: int, strict: bool):
    table = revision / "06_xgb_ablation" / "Table_A1_XGB_multisource_feature_ablation.csv"
    longp = revision / "06_xgb_ablation" / "ablation_predictions_long.csv.gz"
    if missing(table, strict) or missing(longp, strict): return
    d = pd.read_csv(table)
    long = pd.read_csv(longp)
    ci = bootstrap_absolute_rmse(long, 2000, seed + 400)
    d = d.merge(ci, on="configuration", how="left")
    order = ["operational_core", "weather_only", "dynamic_physical", "dynamic_interaction"]
    d["_ord"] = d["configuration"].map({v:i for i,v in enumerate(order)})
    d = d.sort_values("_ord")
    x = np.arange(len(d))
    y = d["RMSE"].to_numpy(float)
    lo = y - d["rmse_boot_low"].to_numpy(float)
    hi = d["rmse_boot_high"].to_numpy(float) - y
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    ax.bar(x, y)
    ax.errorbar(x, y, yerr=np.vstack([lo, hi]), fmt="none", capsize=4)
    ax.set_xticks(x, d["configuration"], rotation=20, ha="right")
    ax.set_ylabel("RMSE (t/10 min)")
    ax.set_title("XGBoost feature-ablation performance")
    ref = float(d.loc[d.configuration.eq("dynamic_physical"), "RMSE"].iloc[0])
    for xi, yi in zip(x, y):
        ax.text(xi, yi, f"{yi:.4f}\n{(yi/ref-1)*100:+.1f}%", ha="center", va="bottom", fontsize=8)
    save(fig, out / "Figure_4_feature_ablation.png")


def fig5_lovo(main: Path, out: Path, strict: bool):
    p = main / "05_lovo" / "Table_L2_LOVO_by_vessel.csv"
    if missing(p, strict): return
    d = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    markers = {"bulk":"o", "container":"s", "tanker":"^"}
    for st, g in d.groupby("ship_type"):
        ax.scatter(g["NRMSE_sd"], g["R2"], label=st, marker=markers.get(st, "o"), s=45)
    ax.axvline(1.0, ls="--")
    ax.axhline(0.0, ls="--")
    ax.set_xlabel("LOVO RMSE / vessel target SD")
    ax.set_ylabel("LOVO R²")
    ax.set_title("Vessel-level LOVO transfer diagnostics")
    ax.legend()
    save(fig, out / "Figure_5_LOVO_transfer.png")


def fig6_generalisation(main: Path, out: Path, strict: bool):
    paths = {
        "L1": main / "01_record_level" / "Table_B2_record_level_by_vessel.csv",
        "L2": main / "04_temporal" / "Table_T2_known_vessel_temporal_by_vessel.csv",
        "L3": main / "05_lovo" / "Table_L2_LOVO_by_vessel.csv",
        "L4": main / "06_adaptation" / "Table_N2_new_vessel_adaptation_by_vessel.csv",
    }
    for p in paths.values():
        if missing(p, strict): return
    parts = []
    for rung, p in paths.items():
        d = pd.read_csv(p)
        if rung == "L1" and "model" in d.columns:
            d = d[d["model"].astype(str).str.lower().eq("xgb")]
        d = d.copy(); d["rung"] = rung; parts.append(d)
    z = pd.concat(parts, ignore_index=True)
    agg = z.groupby(["ship_type","rung"], as_index=False).agg(RMSE=("RMSE","mean"), R2=("R2","mean"))
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.6))
    for st in SHIP_ORDER:
        g = agg[agg.ship_type.eq(st)].set_index("rung").reindex(RUNG_ORDER)
        axes[0].plot(RUNG_ORDER, g["RMSE"], marker="o", label=st)
        axes[1].plot(RUNG_ORDER, g["R2"], marker="o", label=st)
    axes[0].set_ylabel("Mean vessel RMSE")
    axes[1].set_ylabel("Mean vessel R²")
    for ax in axes:
        ax.set_xlabel("Generalisation rung")
        ax.grid(alpha=.2)
    axes[0].legend()
    fig.suptitle("Generalisation performance across L1–L4 by vessel type")
    fig.tight_layout()
    save(fig, out / "Figure_6_generalisation_ladder.png")


def fig7_shap(revision: Path, out: Path, strict: bool):
    p = revision / "07_interventional_shap" / "Table_SHAP1_interventional_global_importance.csv"
    if missing(p, strict): return
    d = pd.read_csv(p).sort_values("normalized_importance", ascending=True)
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    ax.barh(d["feature"], d["normalized_importance"] * 100)
    ax.set_xlabel("Normalised mean |SHAP| importance (%)")
    ax.set_title("XGBoost interventional SHAP global attribution")
    save(fig, out / "Figure_7_SHAP_global_importance.png")


def fig8_gam_lowess(revision: Path, out: Path, strict: bool):
    d = revision / "07_interventional_shap"
    gp = d / "Table_GAM1_speed_SHAP_GAM_curve_and_derivatives.csv"
    lp = d / "Table_LOESS1_speed_SHAP_curve.csv"
    if missing(gp, strict) or missing(lp, strict): return
    g = pd.read_csv(gp); l = pd.read_csv(lp)
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    ax.plot(g["speed_kn"], g["GAM_fit"], label="GAM")
    ax.fill_between(g["speed_kn"], g["GAM_CI95_low"], g["GAM_CI95_high"], alpha=.18)
    ax.plot(l["speed_kn"], l["LOWESS_fit"], label="LOWESS")
    ax.fill_between(l["speed_kn"], l["LOWESS_cluster_boot_CI95_low"], l["LOWESS_cluster_boot_CI95_high"], alpha=.12)
    ax.axhline(0, lw=.8)
    ax.set_xlabel("SOG (kn)")
    ax.set_ylabel("Interventional SHAP value")
    ax.set_title("Speed–SHAP response estimated by GAM and LOWESS")
    ax.legend()
    save(fig, out / "Figure_8_GAM_LOWESS_speed_SHAP.png")


def angle_from_sincos(sin_v, cos_v):
    return (np.degrees(np.arctan2(np.asarray(sin_v,float), np.asarray(cos_v,float))) + 360.0) % 360.0


def sector(angle):
    a = np.asarray(angle, float) % 360
    out = np.full(len(a), "cross", dtype=object)
    out[(a <= 45) | (a >= 315)] = "head"
    out[(a >= 135) & (a <= 225)] = "following"
    return out


def smooth_sector(ax, x, y, sec, xlabel):
    for s in ["head", "cross", "following"]:
        m = (sec == s) & np.isfinite(x) & np.isfinite(y)
        if m.sum() < 50: continue
        sm = lowess(y[m], x[m], frac=.25, it=1, return_sorted=True)
        ax.plot(sm[:,0], sm[:,1], label=s)
    ax.axhline(0, lw=.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Interventional SHAP value")
    ax.legend(title="sector")


def fig9_direction_shap(revision: Path, out: Path, strict: bool):
    d = revision / "07_interventional_shap"
    vp = d / "interventional_SHAP_values.npy"
    xp = d / "interventional_SHAP_sample_features.csv"
    if missing(vp, strict) or missing(xp, strict): return
    sv = np.load(vp); X = pd.read_csv(xp)
    if sv.shape != (len(X), X.shape[1]):
        raise AssertionError(f"SHAP/features mismatch: {sv.shape} vs {X.shape}")
    wind_ang = angle_from_sincos(X["rel_wind_sin"], X["rel_wind_cos"])
    wave_ang = angle_from_sincos(X["rel_wave_sin"], X["rel_wave_cos"])
    wind_sec = sector(wind_ang); wave_sec = sector(wave_ang)
    jw = X.columns.get_loc("rel_wind_speed_kn"); jh = X.columns.get_loc("wave_height_m")
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.8))
    smooth_sector(axes[0], X["rel_wind_speed_kn"].to_numpy(float), sv[:,jw], wind_sec, "Relative wind speed (kn)")
    smooth_sector(axes[1], X["wave_height_m"].to_numpy(float), sv[:,jh], wave_sec, "Significant wave height (m)")
    axes[0].set_title("Wind response")
    axes[1].set_title("Wave response")
    fig.suptitle("Direction-conditioned interventional SHAP responses")
    fig.tight_layout()
    save(fig, out / "Figure_9_direction_conditioned_SHAP.png")


def fig10_ft(scenario: Path, out: Path, strict: bool):
    p = scenario / "Table_15_FT_L1_aligned_by_vessel.csv"
    if missing(p, strict): return
    d = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=(8.4, 5.5))
    for st in SHIP_ORDER:
        g = d[d.ship_type.eq(st)]
        agg = g.groupby("reduction_pct")["FT_CII_change_pct"].mean().sort_index()
        ax.plot(agg.index, agg.values, marker="o", label=st)
        # show vessel points with small deterministic x offsets
        for r, gg in g.groupby("reduction_pct"):
            offs = np.linspace(-.45, .45, len(gg))
            ax.scatter(np.full(len(gg), r) + offs, gg["FT_CII_change_pct"], s=14, alpha=.45)
    ax.axhline(0, lw=.8)
    ax.set_xlabel("Speed reduction (%)")
    ax.set_ylabel("FT CII-proxy change (%)")
    ax.set_title("Fixed-time CII-proxy responses to speed reduction")
    ax.legend()
    save(fig, out / "Figure_10_FT_CII_by_vessel_type.png")


def fig11_ft_fd(scenario: Path, out: Path, strict: bool):
    ft = scenario / "Table_15_FT_L1_aligned_summary.csv"
    fd = scenario / "PriorityA_FD_fuel_time" / "Table_16_FD_fuel_time_decomposition_FINAL.csv"
    if missing(ft, strict) or missing(fd, strict): return
    a = pd.read_csv(ft); b = pd.read_csv(fd)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.8))
    for st in ["Fleet"] + SHIP_ORDER:
        g = a[a.scope.astype(str).str.lower().eq(st.lower())].sort_values("reduction_pct")
        if not g.empty:
            axes[0].plot(g.reduction_pct, g.CII_proxy_change_pct_weighted, marker="o", label=st)
        h = b[b.scope.astype(str).str.lower().eq(st.lower())].sort_values("reduction_pct")
        if not h.empty:
            axes[1].plot(h.reduction_pct, h.predicted_total_fuel_change_pct, marker="o", label=st)
    axes[0].axhline(0, lw=.8); axes[1].axhline(0, lw=.8)
    axes[0].set_title("FT: CII-proxy change")
    axes[1].set_title("FD: total-fuel change")
    axes[0].set_ylabel("Change (%)"); axes[1].set_ylabel("Change (%)")
    for ax in axes: ax.set_xlabel("Speed reduction (%)")
    axes[0].legend(fontsize=8); axes[1].legend(fontsize=8)
    fig.suptitle("Slow-steaming responses under FT and FD constraints")
    fig.tight_layout()
    save(fig, out / "Figure_11_FT_FD_responses.png")


def main() -> int:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--fixed31-cruise", type=Path, required=True)
    p.add_argument("--core-path", type=Path, required=True)
    p.add_argument("--column-overrides", type=Path, required=True)
    p.add_argument("--main-output", type=Path, required=True)
    p.add_argument("--revision-output", type=Path, required=True)
    p.add_argument("--scenario-output", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=20260808)
    p.add_argument("--strict", action="store_true", default=True, help=argparse.SUPPRESS)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw = load_canonical_cruise(args.fixed31_cruise, args.core_path, args.column_overrides)
    fig1_framework(args.output_dir)
    fig2_correlation(raw, args.output_dir)
    fig3_distributions(raw, args.output_dir)
    fig4_ablation(args.revision_output, args.output_dir, args.seed, args.strict)
    fig5_lovo(args.main_output, args.output_dir, args.strict)
    fig6_generalisation(args.main_output, args.output_dir, args.strict)
    fig7_shap(args.revision_output, args.output_dir, args.strict)
    fig8_gam_lowess(args.revision_output, args.output_dir, args.strict)
    fig9_direction_shap(args.revision_output, args.output_dir, args.strict)
    fig10_ft(args.scenario_output, args.output_dir, args.strict)
    fig11_ft_fd(args.scenario_output, args.output_dir, args.strict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
