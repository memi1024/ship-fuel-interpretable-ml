#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A8_C1_C5_threshold_sensitivity.py

Reviewer-facing threshold-sensitivity analysis for the manuscript's
pre-specified physical-plausibility criteria C1-C5.

Controlling principles
----------------------
1. This script READS already-computed empirical outputs. It does not refit the
   model, recompute SHAP, or recreate missing historical evidence from
   manuscript prose.
2. Missing C1/C3/C4 source tables remain explicitly missing. Manuscript numbers
   are NEVER used as fallback data.
3. The historical decision rules remain the primary rules. Alternative
   thresholds are revision-only sensitivity settings and do not replace the
   manuscript-era protocol.

Primary manuscript rules
------------------------
C1 Direction:
    Majority sign reversal in the supported region -> non-confirmation.

C2 Curvature:
    Positive curvature is confirmed when the lower 95% CI of d2 is > 0 over
    >50% of the empirically supported range.

C3 Empirical support:
    Joint-feature grid cells with fewer than 30 nearby held-out observations
    are unsupported/masked.

C4 Local perturbation:
    Reverse-response proportion >35% in any supported sector -> conditional
    non-confirmation.

C5 Cross-refit stability:
    Mean Spearman rho >= 0.95 AND mean top-five Jaccard >= 0.80.

Default revision-only sensitivity bands
---------------------------------------
C1 speed-direction reversal threshold: 45%, 50% (primary), 55%.
    When no retained table contains an explicit supported-region reversal
    percentage, C1-speed is reconstructed from the retained 30-bin
    density-balanced speed-SHAP table by counting adjacent-bin reversals.
    This is labelled a reviewer-facing speed-specific reconstruction and is
    not presented as a full multifeature C1 reconstruction.
C2 curvature-support threshold: 40%, 50% (primary), 60%.
C3 nearby-observation threshold: 20, 30 (primary), 40.
C4 reverse-response threshold: 25%, 35% (primary), 45%.
C5 stability:
    relaxed = rho>=0.90 & Jaccard>=0.75
    primary = rho>=0.95 & Jaccard>=0.80

The defaults can be overridden from the command line.

Known authoritative source paths from the master handoff
---------------------------------------------------------
C2:
  <results-root>/15_revision_diagnostics/07_interventional_shap/
  Table_GAM2_speed_SHAP_GAM_summary.csv

C5:
  <results-root>/09_shap_stability/
  Table_ST1_SHAP_stability_summary.csv

The master handoff does not establish exact C1/C3/C4 filenames. Therefore this
script requires those paths explicitly if they are to be evaluated. It scans
the results tree and writes a candidate inventory to help locate them, but it
does not silently select an ambiguous historical file.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


PRIMARY = {
    "C1_reverse_pct": 50.0,
    "C2_curvature_support_pct": 50.0,
    "C3_min_nearby_n": 30,
    "C4_reverse_pct": 35.0,
    "C5_rho": 0.95,
    "C5_jaccard": 0.80,
}


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument(
        "--results-root",
        default="results",
        help="Historical manuscript-results root. Read-only.",
    )
    p.add_argument("--output-dir", required=True)

    p.add_argument(
        "--c1-source",
        action="append",
        default=[],
        help=(
            "Existing empirical C1 source CSV. Repeat for multiple files. "
            "Must contain reverse-response or expected-direction evidence."
        ),
    )
    p.add_argument(
        "--c1-speed-binned-source",
        default=None,
        help=(
            "Retained density-balanced speed-SHAP binned CSV used for a "
            "reviewer-facing C1 speed-direction reconstruction. If omitted, "
            "<results-root>/08_shap_physical/"
            "Table_H7_speed_SHAP_binned_density.csv is attempted."
        ),
    )
    p.add_argument(
        "--c2-source",
        default=None,
        help=(
            "Existing C2 GAM summary CSV. If omitted, the master-handoff "
            "default path under 15_revision_diagnostics is attempted."
        ),
    )
    p.add_argument(
        "--c3-source",
        action="append",
        default=[],
        help=(
            "Existing C3 cell-level support CSV. Repeat if needed. "
            "Must contain per-grid-cell nearby-observation counts."
        ),
    )
    p.add_argument(
        "--c4-source",
        action="append",
        default=[],
        help=(
            "Existing C4 perturbation/sector source CSV. Repeat for multiple "
            "tables, e.g. primary perturbation and sector-localisation tables."
        ),
    )
    p.add_argument(
        "--c5-source",
        default=None,
        help=(
            "Existing C5 SHAP-stability summary CSV. If omitted, the "
            "master-handoff default path under 09_shap_stability is attempted."
        ),
    )

    p.add_argument(
        "--c1-reverse-thresholds",
        default="45,50,55",
        help="Revision sensitivity thresholds in percent; 50 is primary.",
    )
    p.add_argument(
        "--c2-curvature-thresholds",
        default="40,50,60",
        help=(
            "Required percent of supported range with lower95(d2)>0; "
            "50 is primary."
        ),
    )
    p.add_argument(
        "--c3-nearby-thresholds",
        default="20,30,40",
        help="Minimum nearby held-out observations per grid cell; 30 primary.",
    )
    p.add_argument(
        "--c4-reverse-thresholds",
        default="25,35,45",
        help="Reverse-response percent threshold; 35 is primary.",
    )
    p.add_argument(
        "--c5-threshold-pairs",
        default="0.90:0.75,0.95:0.80",
        help=(
            "Comma-separated rho:Jaccard threshold pairs. "
            "0.95:0.80 is primary."
        ),
    )
    p.add_argument(
        "--scan-max-files",
        type=int,
        default=5000,
        help="Maximum CSV files to inventory below results-root.",
    )
    return p.parse_args()


def norm(s) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s).strip().lower()).strip("_")


def parse_float_list(s):
    return [float(x.strip()) for x in str(s).split(",") if x.strip()]


def parse_int_list(s):
    return [int(float(x.strip())) for x in str(s).split(",") if x.strip()]


def parse_pairs(s):
    out = []
    for token in str(s).split(","):
        token = token.strip()
        if not token:
            continue
        a, b = token.split(":", 1)
        out.append((float(a), float(b)))
    return out


def to_pct(x):
    x = float(x)
    if abs(x) <= 1.0:
        return 100.0 * x
    return x


def atomic_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def atomic_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    tmp.replace(path)


def read_csv_existing(path) -> pd.DataFrame:
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(p)
    return pd.read_csv(p, low_memory=False)


def existing_or_none(path):
    if path is None:
        return None
    p = Path(path).expanduser().resolve()
    return p if p.is_file() else None


def first_numeric_from_wide(df, predicates):
    """
    Find a numeric wide-format column whose normalised name satisfies every
    predicate. Returns (value, column) or (None, None).
    """
    for col in df.columns:
        nc = norm(col)
        if all(pred(nc) for pred in predicates):
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(vals):
                return float(vals.iloc[0]), col
    return None, None


def numeric_metric_from_long(df, regex_patterns):
    """
    Search long/key-value-like tables. Concatenates all non-numeric text
    columns as a row label, then returns the first numeric value in a row whose
    label matches all regex_patterns.
    """
    if df.empty:
        return None, None
    text_cols = [
        c for c in df.columns
        if not pd.api.types.is_numeric_dtype(df[c])
    ]
    num_cols = [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
    ]
    if not text_cols:
        return None, None
    labels = df[text_cols].astype(str).agg(" | ".join, axis=1).map(norm)
    for i, label in labels.items():
        if all(re.search(pat, label) for pat in regex_patterns):
            # Prefer columns literally named value/estimate/result.
            ordered = sorted(
                num_cols,
                key=lambda c: (
                    0 if norm(c) in {"value", "estimate", "result"} else 1,
                    list(df.columns).index(c),
                ),
            )
            for c in ordered:
                v = pd.to_numeric(pd.Series([df.loc[i, c]]), errors="coerce").iloc[0]
                if pd.notna(v):
                    return float(v), f"row:{i};col:{c}"
    return None, None



def extract_c1_speed_binned(df: pd.DataFrame, source_name: str):
    """
    Reviewer-facing C1 speed-direction reconstruction from the retained
    density-balanced speed-SHAP table.

    The manuscript C1 rule is majority sign reversal in the supported region.
    The retained H7 table does not store a row-level reversal percentage, but
    it does store ordered density-balanced speed bins and mean SHAP values.
    We therefore count adjacent-bin SHAP reversals after ordering by speed.

    This is deliberately labelled speed-specific. It must not be described as
    a complete multifeature reconstruction of the historical C1 audit.
    """
    speed_col = find_col(
        df,
        aliases=["speed_mean", "speed_kn", "speed", "sog_mean", "sog"],
    )
    shap_col = find_col(
        df,
        aliases=["shap_mean", "mean_shap", "mean_shap_value", "shap_value_mean"],
    )
    n_col = find_col(
        df,
        aliases=["n", "count", "bin_n", "bin_count"],
    )

    if speed_col is None or shap_col is None:
        return None, pd.DataFrame()

    work = pd.DataFrame({
        "speed": pd.to_numeric(df[speed_col], errors="coerce"),
        "shap_mean": pd.to_numeric(df[shap_col], errors="coerce"),
    })
    if n_col is not None:
        work["n"] = pd.to_numeric(df[n_col], errors="coerce")
    else:
        work["n"] = np.nan

    work = work[np.isfinite(work["speed"]) & np.isfinite(work["shap_mean"])].copy()
    work = work.sort_values("speed", kind="mergesort").reset_index(drop=True)

    if len(work) < 2:
        return None, pd.DataFrame()

    # Duplicate speed-bin centres make an adjacent directional comparison
    # ambiguous. Do not silently aggregate them.
    if work["speed"].duplicated().any():
        raise ValueError(
            f"C1 speed-binned source contains duplicate speed values: {source_name}"
        )

    rows = []
    for i in range(len(work) - 1):
        left = work.iloc[i]
        right = work.iloc[i + 1]
        dspeed = float(right["speed"] - left["speed"])
        dshap = float(right["shap_mean"] - left["shap_mean"])

        if not np.isfinite(dspeed) or dspeed <= 0:
            continue

        if dshap < 0:
            direction = "reverse"
        elif dshap > 0:
            direction = "expected"
        else:
            direction = "unchanged"

        rows.append({
            "source_file": str(source_name),
            "pair_index": i + 1,
            "speed_left": float(left["speed"]),
            "speed_right": float(right["speed"]),
            "speed_delta": dspeed,
            "shap_left": float(left["shap_mean"]),
            "shap_right": float(right["shap_mean"]),
            "shap_delta": dshap,
            "direction": direction,
            "left_bin_n": (
                float(left["n"]) if pd.notna(left["n"]) else np.nan
            ),
            "right_bin_n": (
                float(right["n"]) if pd.notna(right["n"]) else np.nan
            ),
        })

    audit = pd.DataFrame(rows)
    if audit.empty:
        return None, audit

    n_pairs = int(len(audit))
    n_reverse = int((audit["direction"] == "reverse").sum())
    n_expected = int((audit["direction"] == "expected").sum())
    n_unchanged = int((audit["direction"] == "unchanged").sum())
    reverse_pct = 100.0 * n_reverse / n_pairs

    summary = {
        "source_file": str(source_name),
        "speed_column": speed_col,
        "shap_column": shap_col,
        "n_column": n_col,
        "n_bins": int(len(work)),
        "n_adjacent_pairs": n_pairs,
        "n_expected": n_expected,
        "n_reverse": n_reverse,
        "n_unchanged": n_unchanged,
        "reverse_response_pct": float(reverse_pct),
        "speed_min": float(work["speed"].min()),
        "speed_max": float(work["speed"].max()),
        "shap_min": float(work["shap_mean"].min()),
        "shap_max": float(work["shap_mean"].max()),
        "reconstruction_scope": "speed_only",
        "reconstruction_status": "reviewer_facing_from_retained_binned_output",
    }
    return summary, audit


def extract_c2(df):
    """
    Extract:
      curvature_support_pct = percentage of supported evaluation range where
      lower 95% confidence bound of d2 is >0.
      monotonic_support_pct = optional d1/lower95 positive percentage.
    """
    # Curvature
    curv, src = first_numeric_from_wide(
        df,
        [
            lambda s: ("d2" in s or "second_derivative" in s or "curvature" in s),
            lambda s: ("lower" in s or "ci" in s),
            lambda s: ("positive" in s or "above_zero" in s or "gt_0" in s or "gt0" in s),
        ],
    )
    if curv is None:
        curv, src = numeric_metric_from_long(
            df,
            [
                r"(d2|second_derivative|curvature)",
                r"(lower|ci)",
                r"(positive|above_zero|gt_?0)",
            ],
        )

    # Some GAM summaries use a direct curvature-support column without "lower".
    if curv is None:
        curv, src = first_numeric_from_wide(
            df,
            [
                lambda s: ("curvature" in s or "d2" in s),
                lambda s: ("support" in s or "positive_pct" in s or "positive_fraction" in s),
            ],
        )

    mono, mono_src = first_numeric_from_wide(
        df,
        [
            lambda s: ("d1" in s or "first_derivative" in s or "monotonic" in s),
            lambda s: ("positive" in s or "above_zero" in s or "gt_0" in s or "gt0" in s),
        ],
    )
    if mono is None:
        mono, mono_src = numeric_metric_from_long(
            df,
            [
                r"(d1|first_derivative|monotonic)",
                r"(positive|above_zero|gt_?0)",
            ],
        )

    return {
        "curvature_support_pct": None if curv is None else to_pct(curv),
        "curvature_source_field": src,
        "monotonic_support_pct": None if mono is None else to_pct(mono),
        "monotonic_source_field": mono_src,
    }


def find_col(df, aliases=(), contains_all=()):
    cols = {norm(c): c for c in df.columns}
    for a in aliases:
        if norm(a) in cols:
            return cols[norm(a)]
    # IMPORTANT: all([]) is True in Python. Without this guard, an alias-only
    # lookup that fails would incorrectly return the first DataFrame column.
    if contains_all:
        for c in df.columns:
            nc = norm(c)
            if all(x in nc for x in contains_all):
                return c
    return None


def extract_label_series(df):
    preferred = [
        "feature", "scope", "sector", "subgroup", "condition",
        "perturbation", "variable", "group", "label", "name",
    ]
    cols = [c for c in preferred if c in {norm(x) for x in df.columns}]
    actual = []
    nmap = {norm(c): c for c in df.columns}
    for c in cols:
        actual.append(nmap[c])
    if actual:
        return df[actual].apply(
            lambda row: " | ".join("" if pd.isna(x) else str(x) for x in row.to_list()),
            axis=1,
        )
    text_cols = [
        c for c in df.columns
        if not pd.api.types.is_numeric_dtype(df[c])
    ]
    if text_cols:
        return df[text_cols].apply(
            lambda row: " | ".join("" if pd.isna(x) else str(x) for x in row.to_list()),
            axis=1,
        )
    return pd.Series([f"row_{i}" for i in range(len(df))], index=df.index)


def extract_reverse_rows(df, source_name, criterion):
    """
    Extract row-level reverse-response percentages.

    Preferred evidence:
      reverse_pct directly;
      prediction_down_pct for positive perturbations;
      expected_direction_pct + neutral_pct => exact reverse.

    We do NOT use 100 - expected_direction_pct when neutral/ties are unknown.
    """
    labels = extract_label_series(df)
    reverse_col = (
        find_col(df, aliases=[
            "reverse_pct", "reverse_response_pct",
            "reverse_direction_pct", "opposite_direction_pct",
        ])
        or find_col(df, contains_all=["reverse", "pct"])
    )

    down_col = (
        find_col(df, aliases=[
            "prediction_down_pct", "pred_decrease_pct",
            "down_pct", "decrease_pct",
        ])
        or find_col(df, contains_all=["down", "pct"])
    )
    expected_col = (
        find_col(df, aliases=[
            "expected_direction_pct", "expected_pct",
            "expected_response_pct",
        ])
        or find_col(df, contains_all=["expected", "pct"])
    )
    neutral_col = (
        find_col(df, aliases=[
            "neutral_pct", "unchanged_pct", "tie_pct", "no_change_pct",
        ])
    )

    rows = []
    for i in df.index:
        # For C4, rudder perturbation is an H2d identifiability boundary and is
        # not part of the C4 speed/wave reverse-response threshold analysis.
        if criterion == "C4":
            pcol = find_col(df, aliases=["perturbation"])
            if pcol is not None:
                pval = norm(df.loc[i, pcol])
                if pval and ("rudder" in pval):
                    continue

        val = None
        method = None
        if reverse_col is not None:
            x = pd.to_numeric(pd.Series([df.loc[i, reverse_col]]), errors="coerce").iloc[0]
            if pd.notna(x):
                val = to_pct(x)
                method = f"direct:{reverse_col}"
        elif down_col is not None:
            # Valid for the manuscript's positive physical perturbations
            # (SOG +5%, wave height +10%) and positive-direction C1 priors.
            x = pd.to_numeric(pd.Series([df.loc[i, down_col]]), errors="coerce").iloc[0]
            if pd.notna(x):
                val = to_pct(x)
                method = f"prediction_down:{down_col}"
        elif expected_col is not None and neutral_col is not None:
            e = pd.to_numeric(pd.Series([df.loc[i, expected_col]]), errors="coerce").iloc[0]
            n = pd.to_numeric(pd.Series([df.loc[i, neutral_col]]), errors="coerce").iloc[0]
            if pd.notna(e) and pd.notna(n):
                val = 100.0 - to_pct(e) - to_pct(n)
                method = f"100-expected-neutral:{expected_col},{neutral_col}"

        if val is not None and np.isfinite(val):
            rows.append({
                "criterion": criterion,
                "source_file": source_name,
                "source_row": int(i),
                "label": str(labels.loc[i]),
                "reverse_response_pct": float(val),
                "extraction_method": method,
            })
    return pd.DataFrame(rows)


def extract_c3_cells(df, source_name):
    count_col = None
    aliases = [
        "n_nearby", "nearby_n", "nearby_count",
        "n_nearby_heldout", "heldout_count",
        "support_n", "cell_n", "cell_count",
    ]
    count_col = find_col(df, aliases=aliases)
    if count_col is None:
        # Conservative generic matching.
        for c in df.columns:
            nc = norm(c)
            if (
                ("nearby" in nc or "support" in nc or "cell" in nc)
                and ("count" in nc or nc.endswith("_n") or nc.startswith("n_"))
            ):
                count_col = c
                break
    if count_col is None:
        return None, None

    counts = pd.to_numeric(df[count_col], errors="coerce")
    counts = counts[np.isfinite(counts)]
    if not len(counts):
        return None, count_col
    return counts.to_numpy(float), count_col


def extract_c5(df):
    rho_col = None
    jac_col = None

    # Prefer explicit mean columns.
    for c in df.columns:
        nc = norm(c)
        if rho_col is None and "spearman" in nc and "mean" in nc:
            rho_col = c
        if jac_col is None and "jaccard" in nc and "mean" in nc:
            jac_col = c

    if rho_col is not None:
        vals = pd.to_numeric(df[rho_col], errors="coerce").dropna()
        rho = float(vals.iloc[0]) if len(vals) else None
        rho_method = f"summary:{rho_col}"
    else:
        candidates = [
            c for c in df.columns if "spearman" in norm(c)
        ]
        rho = None
        rho_method = None
        for c in candidates:
            vals = pd.to_numeric(df[c], errors="coerce").dropna()
            if len(vals):
                rho = float(vals.mean())
                rho_method = f"mean_rows:{c}"
                break

    if jac_col is not None:
        vals = pd.to_numeric(df[jac_col], errors="coerce").dropna()
        jac = float(vals.iloc[0]) if len(vals) else None
        jac_method = f"summary:{jac_col}"
    else:
        candidates = [
            c for c in df.columns if "jaccard" in norm(c)
        ]
        jac = None
        jac_method = None
        for c in candidates:
            vals = pd.to_numeric(df[c], errors="coerce").dropna()
            if len(vals):
                jac = float(vals.mean())
                jac_method = f"mean_rows:{c}"
                break

    return {
        "mean_spearman_rho": rho,
        "rho_source_field": rho_method,
        "mean_top5_jaccard": jac,
        "jaccard_source_field": jac_method,
    }


def inventory_csvs(root: Path, max_files: int):
    rows = []
    if not root.is_dir():
        return pd.DataFrame(rows)
    paths = list(root.rglob("*.csv"))[: int(max_files)]
    for p in paths:
        try:
            head = pd.read_csv(p, nrows=5, low_memory=False)
            ncols = [norm(c) for c in head.columns]
            joined = " ".join(ncols)
            score_c1 = sum(k in joined for k in ["reverse", "direction", "shap"])
            score_c3 = sum(k in joined for k in ["support", "nearby", "cell"])
            score_c4 = sum(k in joined for k in ["perturb", "reverse", "sector", "prediction_down"])
            score_c5 = sum(k in joined for k in ["spearman", "jaccard", "stability"])
            rows.append({
                "path": str(p),
                "rows_previewed": len(head),
                "n_columns": len(head.columns),
                "columns": "|".join(map(str, head.columns)),
                "candidate_C1_score": score_c1,
                "candidate_C3_score": score_c3,
                "candidate_C4_score": score_c4,
                "candidate_C5_score": score_c5,
            })
        except Exception as exc:
            rows.append({
                "path": str(p),
                "rows_previewed": np.nan,
                "n_columns": np.nan,
                "columns": "",
                "candidate_C1_score": 0,
                "candidate_C3_score": 0,
                "candidate_C4_score": 0,
                "candidate_C5_score": 0,
                "read_error": repr(exc),
            })
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    root = Path(args.results_root).expanduser().resolve()
    out = Path(args.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    c1_thresholds = parse_float_list(args.c1_reverse_thresholds)
    c2_thresholds = parse_float_list(args.c2_curvature_thresholds)
    c3_thresholds = parse_int_list(args.c3_nearby_thresholds)
    c4_thresholds = parse_float_list(args.c4_reverse_thresholds)
    c5_pairs = parse_pairs(args.c5_threshold_pairs)

    # Retained historical C1 speed-binned table confirmed during the
    # reviewer rerun. This is not claimed to be a complete multifeature C1
    # source; it supports a speed-specific reviewer-facing reconstruction.
    c1_speed_default = (
        root
        / "08_shap_physical"
        / "Table_H7_speed_SHAP_binned_density.csv"
    )

    # Exact source defaults established by the master handoff.
    c2_default = (
        root
        / "15_revision_diagnostics"
        / "07_interventional_shap"
        / "Table_GAM2_speed_SHAP_GAM_summary.csv"
    )
    c5_default = (
        root
        / "09_shap_stability"
        / "Table_ST1_SHAP_stability_summary.csv"
    )

    c1_speed_path = (
        existing_or_none(args.c1_speed_binned_source)
        if args.c1_speed_binned_source
        else existing_or_none(c1_speed_default)
    )
    c2_path = existing_or_none(args.c2_source) if args.c2_source else existing_or_none(c2_default)
    c5_path = existing_or_none(args.c5_source) if args.c5_source else existing_or_none(c5_default)

    # Candidate inventory is diagnostic only; never used to silently choose a
    # missing historical source.
    inventory = inventory_csvs(root, args.scan_max_files)
    if not inventory.empty:
        atomic_csv(inventory, out / "A8_candidate_source_inventory.csv")

    source_audit = []
    all_rows = []

    # ---------------- C1 ----------------
    # Priority:
    #   (a) explicit supported-region reverse-response tables, if supplied;
    #   (b) otherwise a speed-specific reconstruction from retained H7
    #       density-balanced speed-SHAP bins.
    #
    # The latter is NOT a complete multifeature C1 reconstruction.
    c1_extracted = []
    if args.c1_source:
        for source in args.c1_source:
            p = existing_or_none(source)
            if p is None:
                source_audit.append({
                    "criterion": "C1",
                    "scope": "explicit_supported_region",
                    "requested_source": str(source),
                    "status": "missing_file",
                    "note": "No manuscript-number fallback used.",
                })
                continue
            df = read_csv_existing(p)
            rr = extract_reverse_rows(df, str(p), "C1")
            source_audit.append({
                "criterion": "C1",
                "scope": "explicit_supported_region",
                "requested_source": str(p),
                "status": "usable" if len(rr) else "schema_not_resolved",
                "rows_in_source": len(df),
                "rows_with_reverse_evidence": len(rr),
                "note": (
                    "C1 uses majority sign reversal in supported region. "
                    "No Spearman threshold is substituted."
                ),
            })
            if len(rr):
                c1_extracted.append(rr)

    c1_speed_summary = None
    c1_speed_audit = pd.DataFrame()
    if c1_speed_path is not None:
        df = read_csv_existing(c1_speed_path)
        try:
            c1_speed_summary, c1_speed_audit = extract_c1_speed_binned(
                df, str(c1_speed_path)
            )
        except Exception as exc:
            source_audit.append({
                "criterion": "C1",
                "scope": "speed_binned_reconstruction",
                "requested_source": str(c1_speed_path),
                "status": "reconstruction_error",
                "rows_in_source": len(df),
                "note": repr(exc),
            })
            c1_speed_summary = None
            c1_speed_audit = pd.DataFrame()
        else:
            source_audit.append({
                "criterion": "C1",
                "scope": "speed_binned_reconstruction",
                "requested_source": str(c1_speed_path),
                "status": (
                    "usable_speed_specific"
                    if c1_speed_summary is not None
                    else "schema_not_resolved"
                ),
                "rows_in_source": len(df),
                "n_bins": (
                    c1_speed_summary["n_bins"]
                    if c1_speed_summary is not None else np.nan
                ),
                "n_adjacent_pairs": (
                    c1_speed_summary["n_adjacent_pairs"]
                    if c1_speed_summary is not None else np.nan
                ),
                "n_reverse": (
                    c1_speed_summary["n_reverse"]
                    if c1_speed_summary is not None else np.nan
                ),
                "note": (
                    "Reviewer-facing speed-specific reconstruction from retained "
                    "density-balanced speed-SHAP bins. It does not reconstruct "
                    "full multifeature C1."
                ),
            })
            if not c1_speed_audit.empty:
                atomic_csv(
                    c1_speed_audit,
                    out / "A8_C1_speed_adjacent_bin_audit.csv",
                )
            if c1_speed_summary is not None:
                atomic_csv(
                    pd.DataFrame([c1_speed_summary]),
                    out / "A8_C1_speed_reconstruction_summary.csv",
                )
    else:
        source_audit.append({
            "criterion": "C1",
            "scope": "speed_binned_reconstruction",
            "requested_source": str(
                args.c1_speed_binned_source or c1_speed_default
            ),
            "status": "missing_file",
            "note": (
                "Retained H7 speed-binned table was not found; no manuscript "
                "number fallback used."
            ),
        })

    if c1_extracted:
        # If a genuine explicit supported-region reverse table is supplied,
        # retain it as the broader C1 sensitivity source.
        c1 = pd.concat(c1_extracted, ignore_index=True)
        atomic_csv(c1, out / "A8_C1_extracted_reverse_response.csv")
        max_reverse = float(c1["reverse_response_pct"].max())
        for t in c1_thresholds:
            all_rows.append({
                "criterion": "C1",
                "scope": "explicit_supported_region",
                "sensitivity_setting": f"reverse>{t:g}%",
                "primary_setting": math.isclose(t, PRIMARY["C1_reverse_pct"]),
                "empirical_quantity": (
                    "max reverse-response proportion in supplied "
                    "supported-region evidence"
                ),
                "empirical_value": max_reverse,
                "threshold_1": t,
                "threshold_2": np.nan,
                "decision": (
                    "non_confirmation"
                    if max_reverse > t
                    else "direction_supported"
                ),
                "available": True,
                "provenance_note": (
                    "Explicit supported-region reverse-response source."
                ),
            })
    elif c1_speed_summary is not None:
        reverse_pct = float(c1_speed_summary["reverse_response_pct"])
        n_reverse = int(c1_speed_summary["n_reverse"])
        n_pairs = int(c1_speed_summary["n_adjacent_pairs"])
        for t in c1_thresholds:
            all_rows.append({
                "criterion": "C1",
                "scope": "speed_binned_reconstruction",
                "sensitivity_setting": f"reverse>{t:g}%",
                "primary_setting": math.isclose(t, PRIMARY["C1_reverse_pct"]),
                "empirical_quantity": (
                    "adjacent density-balanced speed-bin SHAP reversal proportion"
                ),
                "empirical_value": reverse_pct,
                "auxiliary_value": n_reverse,
                "auxiliary_quantity": "number of reverse adjacent-bin comparisons",
                "denominator": n_pairs,
                "threshold_1": t,
                "threshold_2": np.nan,
                "decision": (
                    "speed_direction_non_confirmation"
                    if reverse_pct > t
                    else "speed_direction_supported"
                ),
                "available": True,
                "provenance_note": (
                    "Reviewer-facing speed-specific reconstruction from retained "
                    "H7 binned output; full multifeature C1 reversal sensitivity "
                    "is not reconstructed."
                ),
            })
    else:
        for t in c1_thresholds:
            all_rows.append({
                "criterion": "C1",
                "scope": "not_evaluable",
                "sensitivity_setting": f"reverse>{t:g}%",
                "primary_setting": math.isclose(t, PRIMARY["C1_reverse_pct"]),
                "empirical_quantity": "reverse-response proportion",
                "empirical_value": np.nan,
                "threshold_1": t,
                "threshold_2": np.nan,
                "decision": "not_evaluable_missing_source",
                "available": False,
                "provenance_note": (
                    "Neither explicit reverse-response evidence nor retained "
                    "speed-binned H7 evidence was available."
                ),
            })

    # Explicitly record the remaining scope boundary so a speed-specific
    # reconstruction cannot be mistaken for a full multifeature C1 recovery.
    c1_scope_status = pd.DataFrame([
        {
            "scope": "speed_direction",
            "evaluable": c1_speed_summary is not None or bool(c1_extracted),
            "status": (
                "speed_specific_reconstruction_available"
                if c1_speed_summary is not None and not c1_extracted
                else (
                    "explicit_supported_region_evidence_available"
                    if c1_extracted else "not_evaluable"
                )
            ),
            "note": (
                "Threshold sensitivity can be evaluated for speed direction "
                "from retained evidence."
            ),
        },
        {
            "scope": "full_multifeature_C1",
            "evaluable": bool(c1_extracted),
            "status": (
                "available_from_explicit_reverse_source"
                if c1_extracted else "not_fully_reconstructable"
            ),
            "note": (
                "Aggregate Spearman/direction summaries for wind, wave and "
                "draught are not converted into a reversal percentage. "
                "No new multifeature C1 statistic is invented."
            ),
        },
    ])
    atomic_csv(c1_scope_status, out / "A8_C1_scope_status.csv")

    # ---------------- C2 ----------------
    if c2_path is not None:
        df = read_csv_existing(c2_path)
        c2 = extract_c2(df)
        source_audit.append({
            "criterion": "C2",
            "requested_source": str(c2_path),
            "status": "usable" if c2["curvature_support_pct"] is not None else "schema_not_resolved",
            "rows_in_source": len(df),
            "curvature_source_field": c2["curvature_source_field"],
            "monotonic_source_field": c2["monotonic_source_field"],
            "note": "Curvature uses lower95(d2)>0 support; no manuscript-number fallback.",
        })
        curv = c2["curvature_support_pct"]
        mono = c2["monotonic_support_pct"]
    else:
        source_audit.append({
            "criterion": "C2",
            "requested_source": str(args.c2_source or c2_default),
            "status": "missing_file",
            "note": "Known master-handoff path was absent; no fallback used.",
        })
        curv = None
        mono = None

    for t in c2_thresholds:
        all_rows.append({
            "criterion": "C2",
            "sensitivity_setting": f"lower95(d2)>0 over >{t:g}% support",
            "primary_setting": math.isclose(t, PRIMARY["C2_curvature_support_pct"]),
            "empirical_quantity": "percent supported range with lower95(d2)>0",
            "empirical_value": np.nan if curv is None else curv,
            "auxiliary_value": np.nan if mono is None else mono,
            "auxiliary_quantity": "d1/monotonic positive support percent when available",
            "threshold_1": t,
            "threshold_2": np.nan,
            "decision": (
                "not_evaluable_missing_source"
                if curv is None
                else (
                    "positive_curvature_confirmed"
                    if curv > t
                    else "curvature_not_confirmed"
                )
            ),
            "available": curv is not None,
        })

    # ---------------- C3 ----------------
    c3_arrays = []
    if args.c3_source:
        for source in args.c3_source:
            p = existing_or_none(source)
            if p is None:
                source_audit.append({
                    "criterion": "C3",
                    "requested_source": str(source),
                    "status": "missing_file",
                    "note": "No 151/400 manuscript fallback used.",
                })
                continue
            df = read_csv_existing(p)
            counts, field = extract_c3_cells(df, str(p))
            source_audit.append({
                "criterion": "C3",
                "requested_source": str(p),
                "status": "usable" if counts is not None else "schema_not_resolved",
                "rows_in_source": len(df),
                "count_source_field": field,
                "note": "Requires cell-level nearby-observation counts.",
            })
            if counts is not None:
                c3_arrays.append(counts)
    else:
        source_audit.append({
            "criterion": "C3",
            "requested_source": "",
            "status": "not_supplied",
            "note": (
                "Exact C3 cell-level table not established by master handoff; "
                "151/400 manuscript result is not used as synthetic source data."
            ),
        })

    c3_counts = np.concatenate(c3_arrays) if c3_arrays else None
    for t in c3_thresholds:
        if c3_counts is None:
            supported_n = np.nan
            total_n = np.nan
            pct = np.nan
            decision = "not_evaluable_missing_cell_level_source"
            avail = False
        else:
            supported_n = int(np.sum(c3_counts >= t))
            total_n = int(len(c3_counts))
            pct = 100.0 * supported_n / total_n if total_n else np.nan
            decision = "support_mask_recomputed"
            avail = True
        all_rows.append({
            "criterion": "C3",
            "sensitivity_setting": f"nearby_n>={t}",
            "primary_setting": int(t) == PRIMARY["C3_min_nearby_n"],
            "empirical_quantity": "supported grid cells",
            "empirical_value": supported_n,
            "auxiliary_value": pct,
            "auxiliary_quantity": "supported grid cells percent",
            "threshold_1": t,
            "threshold_2": np.nan,
            "denominator": total_n,
            "decision": decision,
            "available": avail,
        })

    # ---------------- C4 ----------------
    c4_extracted = []
    if args.c4_source:
        for source in args.c4_source:
            p = existing_or_none(source)
            if p is None:
                source_audit.append({
                    "criterion": "C4",
                    "requested_source": str(source),
                    "status": "missing_file",
                    "note": "No manuscript-number fallback used.",
                })
                continue
            df = read_csv_existing(p)
            rr = extract_reverse_rows(df, str(p), "C4")
            source_audit.append({
                "criterion": "C4",
                "requested_source": str(p),
                "status": "usable" if len(rr) else "schema_not_resolved",
                "rows_in_source": len(df),
                "rows_with_reverse_evidence": len(rr),
                "note": (
                    "C4 requires reverse-response evidence. "
                    "100-expected is NOT used unless neutral/tie percentage is known."
                ),
            })
            if len(rr):
                c4_extracted.append(rr)
    else:
        source_audit.append({
            "criterion": "C4",
            "requested_source": "",
            "status": "not_supplied",
            "note": (
                "Exact C4 source table(s) not established by master handoff; "
                "61.67/49.79 manuscript numbers are not used as fallback."
            ),
        })

    if c4_extracted:
        c4 = pd.concat(c4_extracted, ignore_index=True)
        atomic_csv(c4, out / "A8_C4_extracted_reverse_response.csv")
        max_reverse = float(c4["reverse_response_pct"].max())
        max_row = c4.loc[c4["reverse_response_pct"].idxmax()]
        max_label = str(max_row["label"])
        for t in c4_thresholds:
            all_rows.append({
                "criterion": "C4",
                "sensitivity_setting": f"reverse>{t:g}%",
                "primary_setting": math.isclose(t, PRIMARY["C4_reverse_pct"]),
                "empirical_quantity": "maximum reverse-response proportion across supplied supported sectors",
                "empirical_value": max_reverse,
                "auxiliary_value": max_label,
                "auxiliary_quantity": "sector/row attaining maximum reverse response",
                "threshold_1": t,
                "threshold_2": np.nan,
                "decision": (
                    "conditional_non_confirmation"
                    if max_reverse > t
                    else "perturbation_direction_supported"
                ),
                "available": True,
            })
    else:
        for t in c4_thresholds:
            all_rows.append({
                "criterion": "C4",
                "sensitivity_setting": f"reverse>{t:g}%",
                "primary_setting": math.isclose(t, PRIMARY["C4_reverse_pct"]),
                "empirical_quantity": "maximum reverse-response proportion across supported sectors",
                "empirical_value": np.nan,
                "threshold_1": t,
                "threshold_2": np.nan,
                "decision": "not_evaluable_missing_source",
                "available": False,
            })

    # ---------------- C5 ----------------
    if c5_path is not None:
        df = read_csv_existing(c5_path)
        c5 = extract_c5(df)
        source_audit.append({
            "criterion": "C5",
            "requested_source": str(c5_path),
            "status": (
                "usable"
                if c5["mean_spearman_rho"] is not None
                and c5["mean_top5_jaccard"] is not None
                else "schema_not_resolved"
            ),
            "rows_in_source": len(df),
            "rho_source_field": c5["rho_source_field"],
            "jaccard_source_field": c5["jaccard_source_field"],
            "note": "Cross-refit stability; both rho and top-five Jaccard are required.",
        })
        rho = c5["mean_spearman_rho"]
        jac = c5["mean_top5_jaccard"]
    else:
        source_audit.append({
            "criterion": "C5",
            "requested_source": str(args.c5_source or c5_default),
            "status": "missing_file",
            "note": "Known master-handoff path absent; no fallback used.",
        })
        rho = None
        jac = None

    for rho_t, jac_t in c5_pairs:
        all_rows.append({
            "criterion": "C5",
            "sensitivity_setting": f"rho>={rho_t:g};Jaccard>={jac_t:g}",
            "primary_setting": (
                math.isclose(rho_t, PRIMARY["C5_rho"])
                and math.isclose(jac_t, PRIMARY["C5_jaccard"])
            ),
            "empirical_quantity": "mean Spearman rho",
            "empirical_value": np.nan if rho is None else rho,
            "auxiliary_value": np.nan if jac is None else jac,
            "auxiliary_quantity": "mean top-five Jaccard",
            "threshold_1": rho_t,
            "threshold_2": jac_t,
            "decision": (
                "not_evaluable_missing_source"
                if rho is None or jac is None
                else (
                    "stability_supported"
                    if rho >= rho_t and jac >= jac_t
                    else "stability_not_confirmed"
                )
            ),
            "available": rho is not None and jac is not None,
        })

    summary = pd.DataFrame(all_rows)
    source_df = pd.DataFrame(source_audit)

    atomic_csv(summary, out / "A8_C1_C5_threshold_sensitivity.csv")
    atomic_csv(source_df, out / "A8_source_audit.csv")

    primary_summary = summary[summary["primary_setting"] == True].copy()
    atomic_csv(primary_summary, out / "A8_primary_rule_reproduction.csv")

    availability = (
        summary.groupby("criterion", as_index=False)["available"]
        .max()
        .rename(columns={"available": "criterion_evaluable"})
    )
    atomic_csv(availability, out / "A8_criterion_availability.csv")

    manifest = {
        "task": "A8 C1-C5 decision-threshold sensitivity",
        "analysis_type": "read-only sensitivity from existing empirical outputs",
        "results_root": str(root),
        "primary_rules": PRIMARY,
        "revision_sensitivity": {
            "C1_reverse_pct": c1_thresholds,
            "C2_curvature_support_pct": c2_thresholds,
            "C3_min_nearby_n": c3_thresholds,
            "C4_reverse_pct": c4_thresholds,
            "C5_rho_jaccard_pairs": c5_pairs,
        },
        "known_master_handoff_sources": {
            "C2": str(c2_default),
            "C5": str(c5_default),
        },
        "retained_historical_sources_confirmed_during_revision": {
            "C1_speed_binned": str(c1_speed_default),
        },
        "explicit_sources": {
            "C1": list(args.c1_source),
            "C1_speed_binned": args.c1_speed_binned_source,
            "C2": args.c2_source,
            "C3": list(args.c3_source),
            "C4": list(args.c4_source),
            "C5": args.c5_source,
        },
        "no_manuscript_number_fallback": True,
        "missing_source_policy": (
            "Criterion remains not evaluable; manuscript values are not "
            "converted into synthetic historical source tables."
        ),
        "interpretation": (
            "Threshold sensitivity tests decision-boundary dependence only; "
            "it does not turn SHAP attribution into causal evidence."
        ),
        "C1_scope_boundary": (
            "If no explicit supported-region reverse-response source is "
            "supplied, C1 is reconstructed only for speed from retained "
            "density-balanced H7 bins. This must not be reported as a complete "
            "multifeature C1 reconstruction."
        ),
    }
    atomic_json(manifest, out / "A8_manifest.json")

    print("=" * 78)
    print("A8 C1-C5 THRESHOLD SENSITIVITY COMPLETE")
    print(f"Output: {out}")
    print()
    print("Criterion availability:")
    for _, r in availability.iterrows():
        label = "AVAILABLE" if r["criterion_evaluable"] else "MISSING SOURCE"
        if r["criterion"] == "C1" and r["criterion_evaluable"]:
            if c1_extracted:
                label = "AVAILABLE (explicit supported-region evidence)"
            elif c1_speed_summary is not None:
                label = "AVAILABLE (speed-specific reconstruction)"
        print(f"  {r['criterion']}: {label}")
    print()
    print("Primary-rule rows:")
    if len(primary_summary):
        cols = [
            "criterion", "empirical_value", "auxiliary_value",
            "decision", "available",
        ]
        keep = [c for c in cols if c in primary_summary.columns]
        print(primary_summary[keep].to_string(index=False))
    print("=" * 78)


if __name__ == "__main__":
    main()
