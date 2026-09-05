#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
从fixed28生成最终建模数据fixed29。

新增处理：
1. 根据平均吃水和纵倾反推首、尾吃水；
2. 删除反推首吃水或尾吃水<=0的记录；
3. 删除航速超过1.5倍服务航速的记录；
4. 船内分布异常但未超过1.5倍服务航速的记录仅输出复核表；
5. 不按纵倾正负或大小删除数据；
6. 重新计算交互变量。
"""

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd


INPUT_FILE = Path(
    r"data\05_clean23\final_fixed28_quality_checked.csv"
)

OUTPUT_FILE = Path(
    r"data\05_clean23\final_fixed29_model_ready.csv"
)

AUDIT_DIR = Path(
    r"data\05_clean23\final_fixed29_audit"
)

SHIP_TYPE = "ship_type"
SHIP_ID = "pseudo_ship_group_id"

SPEED_SERVICE_RATIO_LIMIT = 1.5
SPEED_QUANTILE = 0.999
SPEED_IQR_MULTIPLIER = 3.0

INTERACTIONS = {
    "rel_wind_speed_x_speed": (
        "rel_wind_speed_kn",
        "speed_kn",
    ),
    "wave_height_x_speed": (
        "wave_height_m",
        "speed_kn",
    ),
    "draught_x_speed": (
        "mean_draught_m",
        "speed_kn",
    ),
    "trim_x_speed": (
        "trim_m",
        "speed_kn",
    ),
    "draught_x_trim": (
        "mean_draught_m",
        "trim_m",
    ),
}

MODEL_VARIABLES = [
    "fuel_t_10min",
    "speed_kn",
    "heading_sin",
    "heading_cos",
    "mean_draught_m",
    "rudder_deg",
    "trim_m",
    "rel_wind_speed_kn",
    "relative_wind_sin",
    "relative_wind_cos",
    "wave_height_m",
    "wave_period_s",
    "relative_wave_sin",
    "relative_wave_cos",
    "surface_pressure_pa",
    "surface_temperature_c",
]


def configure_console():
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        if stream is not None:
            try:
                stream.reconfigure(
                    encoding="utf-8",
                    errors="backslashreplace",
                )
            except Exception:
                pass


def save_csv(frame, filename):
    frame.to_csv(
        AUDIT_DIR / filename,
        index=False,
        encoding="utf-8-sig",
    )


def main():

    configure_console()

    if not INPUT_FILE.is_file():
        raise FileNotFoundError(
            f"找不到输入文件：{INPUT_FILE}"
        )

    AUDIT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("[1/7] 读取fixed28数据……")

    df = pd.read_csv(
        INPUT_FILE,
        low_memory=False,
    )

    df.insert(
        0,
        "__fixed28_row_number",
        np.arange(
            2,
            len(df) + 2,
            dtype=np.int64,
        ),
    )

    required = [
        SHIP_TYPE,
        SHIP_ID,
        "mean_draught_m",
        "trim_m",
        "speed_kn",
        "service_speed_kn",
    ] + MODEL_VARIABLES

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise KeyError(
            "缺少字段："
            + ", ".join(sorted(set(missing)))
        )

    df[SHIP_TYPE] = (
        df[SHIP_TYPE]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    df[SHIP_ID] = (
        df[SHIP_ID]
        .astype("string")
        .str.strip()
    )

    numeric_columns = list(
        dict.fromkeys(
            MODEL_VARIABLES
            + [
                "service_speed_kn",
            ]
        )
    )

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    input_rows = len(df)
    input_ships = (
        df[[SHIP_TYPE, SHIP_ID]]
        .drop_duplicates()
        .shape[0]
    )

    # ========================================================
    # 1. 根据平均吃水和纵倾反推首尾吃水
    # ========================================================

    print("[2/7] 检查反推首尾吃水……")

    # 纵倾定义：尾吃水 - 首吃水
    df["derived_aft_draught_m"] = (
        df["mean_draught_m"]
        + df["trim_m"] / 2.0
    )

    df["derived_fore_draught_m"] = (
        df["mean_draught_m"]
        - df["trim_m"] / 2.0
    )

    invalid_draught = (
        ~np.isfinite(df["mean_draught_m"])
        | ~np.isfinite(df["trim_m"])
        | ~np.isfinite(
            df["derived_aft_draught_m"]
        )
        | ~np.isfinite(
            df["derived_fore_draught_m"]
        )
        | df["derived_aft_draught_m"].le(0)
        | df["derived_fore_draught_m"].le(0)
    )

    removed_draught = df.loc[
        invalid_draught,
        [
            "__fixed28_row_number",
            SHIP_TYPE,
            SHIP_ID,
            "mean_draught_m",
            "trim_m",
            "derived_fore_draught_m",
            "derived_aft_draught_m",
            "fuel_t_10min",
            "speed_kn",
        ],
    ].copy()

    removed_draught["reason"] = (
        "derived_fore_or_aft_draught_not_positive"
    )

    save_csv(
        removed_draught,
        "01_removed_invalid_derived_draught.csv",
    )

    df = df.loc[
        ~invalid_draught
    ].copy()

    # ========================================================
    # 2. 删除超过1.5倍服务航速的记录
    # ========================================================

    print("[3/7] 删除超过1.5倍服务航速的记录……")

    valid_service_speed = (
        np.isfinite(df["service_speed_kn"])
        & df["service_speed_kn"].gt(0)
    )

    df["speed_service_ratio"] = np.where(
        valid_service_speed,
        df["speed_kn"]
        / df["service_speed_kn"],
        np.nan,
    )

    invalid_speed = (
        valid_service_speed
        & df["speed_service_ratio"].gt(
            SPEED_SERVICE_RATIO_LIMIT
        )
    )

    removed_speed = df.loc[
        invalid_speed,
        [
            "__fixed28_row_number",
            SHIP_TYPE,
            SHIP_ID,
            "speed_kn",
            "service_speed_kn",
            "speed_service_ratio",
            "fuel_t_10min",
            "mean_draught_m",
            "trim_m",
        ],
    ].copy()

    removed_speed["reason"] = (
        "speed_above_1.5_times_service_speed"
    )

    save_csv(
        removed_speed,
        "02_removed_speed_service_ratio.csv",
    )

    df = df.loc[
        ~invalid_speed
    ].copy()

    # ========================================================
    # 3. 船内分布异常仅复核，不删除
    # ========================================================

    print("[4/7] 生成剩余航速分布复核表……")

    speed_summary = []
    distribution_blocks = []

    for (ship_type, ship_id), group in df.groupby(
        [SHIP_TYPE, SHIP_ID],
        dropna=False,
    ):

        speed = group["speed_kn"]

        q1 = float(speed.quantile(0.25))
        q3 = float(speed.quantile(0.75))
        iqr_upper = (
            q3
            + SPEED_IQR_MULTIPLIER
            * (q3 - q1)
        )

        q999 = float(
            speed.quantile(SPEED_QUANTILE)
        )

        distribution_flag = (
            speed.gt(q999)
            & speed.gt(iqr_upper)
        )

        speed_summary.append({
            SHIP_TYPE: ship_type,
            SHIP_ID: ship_id,
            "rows": len(group),
            "speed_median": float(
                speed.median()
            ),
            "speed_q99": float(
                speed.quantile(0.99)
            ),
            "speed_q99_9": q999,
            "speed_max": float(
                speed.max()
            ),
            "iqr_upper_fence": iqr_upper,
            "distribution_review_rows": int(
                distribution_flag.sum()
            ),
        })

        if distribution_flag.any():

            flagged = group.loc[
                distribution_flag,
                [
                    "__fixed28_row_number",
                    SHIP_TYPE,
                    SHIP_ID,
                    "speed_kn",
                    "service_speed_kn",
                    "speed_service_ratio",
                    "fuel_t_10min",
                ],
            ].copy()

            flagged["within_ship_q99_9"] = q999
            flagged["iqr_upper_fence"] = (
                iqr_upper
            )

            distribution_blocks.append(flagged)

    speed_summary = pd.DataFrame(
        speed_summary
    )

    distribution_review = (
        pd.concat(
            distribution_blocks,
            ignore_index=True,
        )
        if distribution_blocks
        else pd.DataFrame()
    )

    save_csv(
        speed_summary,
        "03_speed_distribution_by_ship.csv",
    )

    save_csv(
        distribution_review,
        "04_speed_distribution_review_rows.csv",
    )

    # ========================================================
    # 4. 重新计算交互变量
    # ========================================================

    print("[5/7] 重新计算交互变量……")

    for interaction, (
        left,
        right,
    ) in INTERACTIONS.items():

        df[interaction] = (
            df[left] * df[right]
        )

    # ========================================================
    # 5. 最终完整性检查
    # ========================================================

    print("[6/7] 最终完整性检查……")

    matrix = np.column_stack([
        df[column].to_numpy(
            dtype=float,
            na_value=np.nan,
        )
        for column in MODEL_VARIABLES
    ])

    complete = np.isfinite(
        matrix
    ).all(axis=1)

    completeness = pd.DataFrame([{
        "rows": len(df),
        "complete_model_rows": int(
            complete.sum()
        ),
        "incomplete_model_rows": int(
            (~complete).sum()
        ),
        "complete_model_row_pct": (
            100 * complete.mean()
            if len(df)
            else np.nan
        ),
    }])

    save_csv(
        completeness,
        "05_final_model_completeness.csv",
    )

    if not complete.all():
        raise RuntimeError(
            "最终数据仍有模型变量缺失。"
        )

    final_distribution = (
        df.groupby(
            [SHIP_TYPE, SHIP_ID],
            dropna=False,
        )
        .agg(
            rows=(SHIP_ID, "size"),
            speed_mean=("speed_kn", "mean"),
            speed_max=("speed_kn", "max"),
            draught_min=(
                "mean_draught_m",
                "min",
            ),
            fore_draught_min=(
                "derived_fore_draught_m",
                "min",
            ),
            aft_draught_min=(
                "derived_aft_draught_m",
                "min",
            ),
            trim_min=("trim_m", "min"),
            trim_max=("trim_m", "max"),
            target_mean=(
                "fuel_t_10min",
                "mean",
            ),
        )
        .reset_index()
        .sort_values(
            [SHIP_TYPE, SHIP_ID]
        )
    )

    save_csv(
        final_distribution,
        "06_final_ship_distribution.csv",
    )

    internal_columns = [
        "__fixed28_row_number",
        "derived_fore_draught_m",
        "derived_aft_draught_m",
        "speed_service_ratio",
    ]

    output = df.drop(
        columns=[
            column
            for column in internal_columns
            if column in df.columns
        ]
    )

    output.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    final_ships = (
        output[[SHIP_TYPE, SHIP_ID]]
        .drop_duplicates()
        .shape[0]
    )

    manifest = {
        "input": str(INPUT_FILE),
        "output": str(OUTPUT_FILE),
        "input_rows": int(input_rows),
        "input_ships": int(input_ships),
        "removed_invalid_derived_draught_rows": int(
            len(removed_draught)
        ),
        "removed_speed_rows": int(
            len(removed_speed)
        ),
        "speed_rule": (
            "speed_kn > 1.5 * service_speed_kn"
        ),
        "distribution_only_rows_deleted": 0,
        "final_rows": int(len(output)),
        "final_ships": int(final_ships),
    }

    (
        AUDIT_DIR
        / "07_cleaning_manifest.json"
    ).write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    summary = [
        "FINAL FIXED29 MODEL-READY DATA",
        "=" * 78,
        f"输入记录：{input_rows:,}",
        f"输入船舶：{input_ships}",
        (
            "删除反推首尾吃水无效记录："
            f"{len(removed_draught):,}"
        ),
        (
            "删除超过1.5倍服务航速记录："
            f"{len(removed_speed):,}"
        ),
        (
            "剩余船内分布复核记录："
            f"{len(distribution_review):,}"
        ),
        f"最终记录：{len(output):,}",
        f"最终船舶：{final_ships}",
        f"输出文件：{OUTPUT_FILE}",
    ]

    (
        AUDIT_DIR
        / "00_summary.txt"
    ).write_text(
        "\n".join(summary),
        encoding="utf-8-sig",
    )

    print("[7/7] 完成。")
    print(f"最终数据：{OUTPUT_FILE}")
    print(f"审计目录：{AUDIT_DIR}")
    print(f"删除吃水异常：{len(removed_draught):,}")
    print(f"删除航速异常：{len(removed_speed):,}")
    print(f"最终记录：{len(output):,}")
    print(f"最终船舶：{final_ships}")


if __name__ == "__main__":

    try:
        main()

    except KeyboardInterrupt:
        print(
            "\n用户中断运行。",
            file=sys.stderr,
        )
        raise SystemExit(130)

    except Exception as exc:
        print(
            f"\n运行失败："
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        raise