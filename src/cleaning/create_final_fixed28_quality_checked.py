#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
生成质量复核后的 fixed28 数据。

处理规则：
1. 整船删除 TANKER_F25527857_P0001。
2. 删除剩余记录中无有效平均吃水或纵倾缺失的记录。
3. 删除燃油目标异常。
4. 航速仅复核，不自动删除。
5. 纵倾定义为：尾吃水 - 首吃水。
   不按纵倾大小删除，只统计每艘船 trim_m < -2 的数量。
6. 波周期保留 0.5～25 秒。
7. 表面温度保留 -5～37 摄氏度。
8. 重新计算所有交互变量。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 文件路径
# ============================================================

INPUT_FILE = Path(
    r"data\05_clean23\final_fixed27_cruise.csv"
)

OUTPUT_FILE = Path(
    r"data\05_clean23\final_fixed28_quality_checked.csv"
)

AUDIT_DIR = Path(
    r"data\05_clean23\final_fixed28_quality_audit"
)


# ============================================================
# 基本设置
# ============================================================

SHIP_TYPE = "ship_type"
SHIP_ID = "pseudo_ship_group_id"
TARGET = "fuel_t_10min"
POWER = "main_engine_power_kw"

BAD_SHIP_ID = "TANKER_F25527857_P0001"

# 燃油功率QC参数
SFOC_MAX_G_KWH = 250.0
FUEL_POWER_FACTOR = 1.0
MIN_FUEL_T_10MIN = 1e-5

# 波周期范围
WAVE_PERIOD_MIN = 0.5
WAVE_PERIOD_MAX = 25.0

# 表面温度范围
TEMPERATURE_MIN = -5.0
TEMPERATURE_MAX = 37.0

# 航速复核阈值，只用于标记，不自动删除
SPEED_SERVICE_RATIO_LIMIT = 1.50
SPEED_UPPER_QUANTILE = 0.999
SPEED_IQR_MULTIPLIER = 3.0


MODEL_VARIABLES = [
    TARGET,
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


# ============================================================
# 辅助函数
# ============================================================

def configure_console() -> None:
    """尽量避免Windows终端中文乱码。"""

    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)

        if stream is None:
            continue

        try:
            stream.reconfigure(
                encoding="utf-8",
                errors="backslashreplace",
            )
        except (AttributeError, ValueError, OSError):
            pass


def save_csv(
    frame: pd.DataFrame,
    filename: str,
) -> None:
    """保存审计CSV。"""

    frame.to_csv(
        AUDIT_DIR / filename,
        index=False,
        encoding="utf-8-sig",
    )


def require_columns(
    frame: pd.DataFrame,
    columns: list[str],
) -> None:
    """检查必需字段。"""

    missing = [
        column
        for column in columns
        if column not in frame.columns
    ]

    if missing:
        raise KeyError(
            "输入文件缺少必需字段："
            + ", ".join(missing)
        )


def normalize_identifiers(
    frame: pd.DataFrame,
) -> None:
    """统一船型和船号格式。"""

    frame[SHIP_TYPE] = (
        frame[SHIP_TYPE]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    frame[SHIP_ID] = (
        frame[SHIP_ID]
        .astype("string")
        .str.strip()
    )


def convert_numeric(
    frame: pd.DataFrame,
) -> None:
    """转换本次处理需要的数值字段。"""

    numeric_columns = list(
        dict.fromkeys(
            MODEL_VARIABLES
            + [
                POWER,
                "service_speed_kn",
            ]
            + [
                source
                for pair in INTERACTIONS.values()
                for source in pair
            ]
        )
    )

    for column in numeric_columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(
                frame[column],
                errors="coerce",
            )


def build_speed_review(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    航速复核。

    两类标记：
    1. 船内极端分布：
       航速同时超过船内99.9%分位数和Q3+3IQR。
    2. 服务航速比例：
       航速 > 1.5 × 服务航速。

    这些记录只输出，不删除。
    """

    ship_summaries = []
    flagged_blocks = []

    grouped = frame.groupby(
        [SHIP_TYPE, SHIP_ID],
        dropna=False,
        sort=True,
    )

    for (ship_type, ship_id), group in grouped:

        speed = pd.to_numeric(
            group["speed_kn"],
            errors="coerce",
        )

        service_speed_series = pd.to_numeric(
            group["service_speed_kn"],
            errors="coerce",
        )

        finite_speed = speed[
            np.isfinite(speed)
        ]

        finite_service = service_speed_series[
            np.isfinite(service_speed_series)
            & service_speed_series.gt(0)
        ]

        if finite_speed.empty:
            continue

        q1 = float(finite_speed.quantile(0.25))
        q3 = float(finite_speed.quantile(0.75))
        iqr = q3 - q1

        iqr_upper = (
            q3
            + SPEED_IQR_MULTIPLIER * iqr
        )

        quantile_upper = float(
            finite_speed.quantile(
                SPEED_UPPER_QUANTILE
            )
        )

        service_speed = (
            float(finite_service.median())
            if not finite_service.empty
            else np.nan
        )

        service_limit = (
            service_speed
            * SPEED_SERVICE_RATIO_LIMIT
            if np.isfinite(service_speed)
            else np.nan
        )

        distribution_flag = (
            np.isfinite(speed)
            & speed.gt(quantile_upper)
            & speed.gt(iqr_upper)
        )

        if np.isfinite(service_limit):
            service_ratio_flag = (
                np.isfinite(speed)
                & speed.gt(service_limit)
            )
        else:
            service_ratio_flag = pd.Series(
                False,
                index=group.index,
            )

        combined_flag = (
            distribution_flag
            | service_ratio_flag
        )

        ship_summaries.append({
            SHIP_TYPE: ship_type,
            SHIP_ID: ship_id,
            "rows": len(group),
            "service_speed_kn": service_speed,
            "speed_min": float(finite_speed.min()),
            "speed_median": float(
                finite_speed.median()
            ),
            "speed_q95": float(
                finite_speed.quantile(0.95)
            ),
            "speed_q99": float(
                finite_speed.quantile(0.99)
            ),
            "speed_q99_9": quantile_upper,
            "speed_max": float(
                finite_speed.max()
            ),
            "iqr_upper_fence": iqr_upper,
            "service_ratio_limit": (
                SPEED_SERVICE_RATIO_LIMIT
            ),
            "service_ratio_limit_kn": (
                service_limit
            ),
            "distribution_flag_rows": int(
                distribution_flag.sum()
            ),
            "service_ratio_flag_rows": int(
                service_ratio_flag.sum()
            ),
            "combined_review_rows": int(
                combined_flag.sum()
            ),
        })

        if combined_flag.any():

            columns = [
                column
                for column in [
                    "__source_row_number",
                    SHIP_TYPE,
                    SHIP_ID,
                    "speed_kn",
                    "service_speed_kn",
                    TARGET,
                    "mean_draught_m",
                    "trim_m",
                    "rel_wind_speed_kn",
                    "wave_height_m",
                ]
                if column in group.columns
            ]

            flagged = group.loc[
                combined_flag,
                columns,
            ].copy()

            flagged[
                "within_ship_q99_9_kn"
            ] = quantile_upper

            flagged[
                "within_ship_iqr_upper_kn"
            ] = iqr_upper

            flagged[
                "service_ratio_limit_kn"
            ] = service_limit

            flagged[
                "speed_service_ratio"
            ] = np.where(
                np.isfinite(
                    service_speed_series.loc[
                        combined_flag
                    ]
                )
                & service_speed_series.loc[
                    combined_flag
                ].gt(0),
                speed.loc[combined_flag]
                / service_speed_series.loc[
                    combined_flag
                ],
                np.nan,
            )

            flagged[
                "distribution_flag"
            ] = distribution_flag.loc[
                combined_flag
            ].to_numpy()

            flagged[
                "service_ratio_flag"
            ] = service_ratio_flag.loc[
                combined_flag
            ].to_numpy()

            flagged_blocks.append(flagged)

    summary = pd.DataFrame(ship_summaries)

    flagged_rows = (
        pd.concat(
            flagged_blocks,
            ignore_index=True,
        )
        if flagged_blocks
        else pd.DataFrame()
    )

    return summary, flagged_rows


def build_trim_report(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    统计每艘船 trim_m < -2 的数量。

    不执行纵倾删除。
    纵倾约定：尾吃水 - 首吃水。
    """

    temp = frame[
        [
            SHIP_TYPE,
            SHIP_ID,
            "trim_m",
        ]
    ].copy()

    temp["trim_m"] = pd.to_numeric(
        temp["trim_m"],
        errors="coerce",
    )

    temp["below_minus2"] = (
        temp["trim_m"] < -2.0
    )

    report = (
        temp.groupby(
            [SHIP_TYPE, SHIP_ID],
            dropna=False,
        )
        .agg(
            rows=("trim_m", "size"),
            finite_trim_rows=("trim_m", "count"),
            trim_below_minus2_rows=(
                "below_minus2",
                "sum",
            ),
            trim_min=("trim_m", "min"),
            trim_median=("trim_m", "median"),
            trim_max=("trim_m", "max"),
        )
        .reset_index()
    )

    report[
        "trim_below_minus2_pct"
    ] = np.where(
        report["finite_trim_rows"] > 0,
        100
        * report["trim_below_minus2_rows"]
        / report["finite_trim_rows"],
        np.nan,
    )

    return report.sort_values(
        [
            "trim_below_minus2_rows",
            SHIP_TYPE,
            SHIP_ID,
        ],
        ascending=[
            False,
            True,
            True,
        ],
    )


# ============================================================
# 主程序
# ============================================================

def main() -> None:

    configure_console()

    if not INPUT_FILE.is_file():
        raise FileNotFoundError(
            f"找不到输入文件：{INPUT_FILE}"
        )

    if INPUT_FILE.resolve() == OUTPUT_FILE.resolve():
        raise ValueError(
            "输出文件不能覆盖输入文件。"
        )

    AUDIT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("[1/9] 读取数据……")

    df = pd.read_csv(
        INPUT_FILE,
        low_memory=False,
    )

    # 保存原CSV行号。CSV表头为第1行，所以数据从第2行开始。
    df.insert(
        0,
        "__source_row_number",
        np.arange(
            2,
            len(df) + 2,
            dtype=np.int64,
        ),
    )

    require_columns(
        df,
        [
            SHIP_TYPE,
            SHIP_ID,
            POWER,
            "service_speed_kn",
        ]
        + MODEL_VARIABLES,
    )

    normalize_identifiers(df)
    convert_numeric(df)

    input_rows = len(df)

    input_ship_count = (
        df[
            [SHIP_TYPE, SHIP_ID]
        ]
        .drop_duplicates()
        .shape[0]
    )

    # --------------------------------------------------------
    # 1. 整船删除异常油船
    # --------------------------------------------------------

    print("[2/9] 删除无有效吃水的异常油船……")

    bad_ship_mask = (
        df[SHIP_ID] == BAD_SHIP_ID
    )

    bad_ship_rows = df.loc[
        bad_ship_mask
    ].copy()

    df = df.loc[
        ~bad_ship_mask
    ].copy()

    save_csv(
        pd.DataFrame([{
            "removed_ship_id": BAD_SHIP_ID,
            "removed_rows": len(bad_ship_rows),
            "reason": (
                "mean_draught_m和trim_m整船为0"
            ),
        }]),
        "01_removed_ship_summary.csv",
    )

    # --------------------------------------------------------
    # 2. 删除剩余船舶中无有效吃水或纵倾缺失的记录
    # --------------------------------------------------------

    print("[3/9] 删除无有效吃水或纵倾缺失的记录……")

    draught = pd.to_numeric(
        df["mean_draught_m"],
        errors="coerce",
    )

    trim = pd.to_numeric(
        df["trim_m"],
        errors="coerce",
    )

    invalid_draught_trim = (
        ~np.isfinite(draught)
        | draught.le(0)
        | ~np.isfinite(trim)
    )

    removed_draught_trim = df.loc[
        invalid_draught_trim,
        [
            column
            for column in [
                "__source_row_number",
                SHIP_TYPE,
                SHIP_ID,
                "mean_draught_m",
                "trim_m",
                TARGET,
            ]
            if column in df.columns
        ],
    ].copy()

    removed_draught_trim[
        "reason"
    ] = "invalid_draught_or_missing_trim"

    save_csv(
        removed_draught_trim,
        "02_removed_invalid_draught_trim_rows.csv",
    )

    df = df.loc[
        ~invalid_draught_trim
    ].copy()

    # --------------------------------------------------------
    # 3. 删除燃油目标异常
    # --------------------------------------------------------

    print("[4/9] 删除燃油目标异常……")

    fuel = pd.to_numeric(
        df[TARGET],
        errors="coerce",
    )

    power = pd.to_numeric(
        df[POWER],
        errors="coerce",
    )

    theoretical_max = (
        power
        * SFOC_MAX_G_KWH
        * FUEL_POWER_FACTOR
        / 6_000_000.0
    )

    valid_power = (
        np.isfinite(power)
        & power.gt(0)
    )

    fuel_above_limit = (
        valid_power
        & fuel.gt(theoretical_max)
    )

    fuel_too_small = (
        fuel.lt(MIN_FUEL_T_10MIN)
    )

    invalid_fuel = (
        ~np.isfinite(fuel)
        | fuel_above_limit
        | fuel_too_small
    )

    fuel_columns = [
        column
        for column in [
            "__source_row_number",
            SHIP_TYPE,
            SHIP_ID,
            TARGET,
            POWER,
            "speed_kn",
            "mean_draught_m",
            "trim_m",
        ]
        if column in df.columns
    ]

    removed_fuel = df.loc[
        invalid_fuel,
        fuel_columns,
    ].copy()

    removed_fuel[
        "theoretical_max_fuel_t_10min"
    ] = theoretical_max.loc[
        invalid_fuel
    ].to_numpy()

    removed_fuel[
        "fuel_above_power_limit"
    ] = fuel_above_limit.loc[
        invalid_fuel
    ].to_numpy()

    removed_fuel[
        "fuel_below_minimum"
    ] = fuel_too_small.loc[
        invalid_fuel
    ].to_numpy()

    save_csv(
        removed_fuel,
        "03_removed_fuel_target_rows.csv",
    )

    df = df.loc[
        ~invalid_fuel
    ].copy()

    # --------------------------------------------------------
    # 4. 波周期0.5～25秒
    # --------------------------------------------------------

    print("[5/9] 筛选波周期……")

    wave_period = pd.to_numeric(
        df["wave_period_s"],
        errors="coerce",
    )

    invalid_wave_period = (
        ~np.isfinite(wave_period)
        | wave_period.lt(WAVE_PERIOD_MIN)
        | wave_period.gt(WAVE_PERIOD_MAX)
    )

    removed_wave = df.loc[
        invalid_wave_period,
        [
            column
            for column in [
                "__source_row_number",
                SHIP_TYPE,
                SHIP_ID,
                "wave_period_s",
                "wave_height_m",
                TARGET,
            ]
            if column in df.columns
        ],
    ].copy()

    removed_wave[
        "reason"
    ] = "wave_period_outside_0.5_to_25_seconds"

    save_csv(
        removed_wave,
        "04_removed_wave_period_rows.csv",
    )

    df = df.loc[
        ~invalid_wave_period
    ].copy()

    # --------------------------------------------------------
    # 5. 表面温度-5～37℃
    # --------------------------------------------------------

    print("[6/9] 筛选表面温度……")

    temperature = pd.to_numeric(
        df["surface_temperature_c"],
        errors="coerce",
    )

    invalid_temperature = (
        ~np.isfinite(temperature)
        | temperature.lt(TEMPERATURE_MIN)
        | temperature.gt(TEMPERATURE_MAX)
    )

    removed_temperature = df.loc[
        invalid_temperature,
        [
            column
            for column in [
                "__source_row_number",
                SHIP_TYPE,
                SHIP_ID,
                "surface_temperature_c",
                TARGET,
            ]
            if column in df.columns
        ],
    ].copy()

    removed_temperature[
        "reason"
    ] = "surface_temperature_outside_minus5_to_37_c"

    save_csv(
        removed_temperature,
        "05_removed_temperature_rows.csv",
    )

    df = df.loc[
        ~invalid_temperature
    ].copy()

    # --------------------------------------------------------
    # 6. 航速复核，不删除
    # --------------------------------------------------------

    print("[7/9] 生成航速复核表，不删除航速记录……")

    speed_summary, speed_flagged = (
        build_speed_review(df)
    )

    save_csv(
        speed_summary,
        "06_speed_review_by_ship.csv",
    )

    save_csv(
        speed_flagged,
        "07_speed_review_rows.csv",
    )

    # --------------------------------------------------------
    # 7. 纵倾<-2统计，不删除
    # --------------------------------------------------------

    print("[8/9] 统计每艘船纵倾小于-2米的数量……")

    trim_report = build_trim_report(df)

    save_csv(
        trim_report,
        "08_trim_below_minus2_by_ship.csv",
    )

    # --------------------------------------------------------
    # 8. 重新计算交互变量
    # --------------------------------------------------------

    for interaction, (
        source_1,
        source_2,
    ) in INTERACTIONS.items():

        df[interaction] = (
            pd.to_numeric(
                df[source_1],
                errors="coerce",
            )
            * pd.to_numeric(
                df[source_2],
                errors="coerce",
            )
        )

    # --------------------------------------------------------
    # 9. 最终完整性检查
    # --------------------------------------------------------

    matrix = np.column_stack([
        pd.to_numeric(
            df[column],
            errors="coerce",
        ).to_numpy(
            dtype=float,
            na_value=np.nan,
        )
        for column in MODEL_VARIABLES
    ])

    complete_rows = (
        np.isfinite(matrix).all(axis=1)
    )

    completeness = pd.DataFrame([{
        "rows": len(df),
        "complete_model_rows": int(
            complete_rows.sum()
        ),
        "incomplete_model_rows": int(
            (~complete_rows).sum()
        ),
        "complete_model_row_pct": (
            complete_rows.mean() * 100
            if len(df)
            else np.nan
        ),
    }])

    save_csv(
        completeness,
        "09_final_model_completeness.csv",
    )

    if not complete_rows.all():
        raise RuntimeError(
            "最终数据仍包含模型变量缺失，"
            "请检查09_final_model_completeness.csv。"
        )

    final_distribution = (
        df.groupby(
            [SHIP_TYPE, SHIP_ID],
            dropna=False,
        )
        .agg(
            rows=(SHIP_ID, "size"),
            target_mean=(TARGET, "mean"),
            target_std=(TARGET, "std"),
            speed_mean=("speed_kn", "mean"),
            speed_max=("speed_kn", "max"),
            draught_min=(
                "mean_draught_m",
                "min",
            ),
            draught_max=(
                "mean_draught_m",
                "max",
            ),
            trim_min=("trim_m", "min"),
            trim_max=("trim_m", "max"),
        )
        .reset_index()
        .sort_values(
            [SHIP_TYPE, SHIP_ID]
        )
    )

    save_csv(
        final_distribution,
        "10_final_ship_distribution.csv",
    )

    # 删除内部审计行号，再输出正式CSV
    output = df.drop(
        columns=["__source_row_number"]
    )

    output.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    final_ship_count = (
        output[
            [SHIP_TYPE, SHIP_ID]
        ]
        .drop_duplicates()
        .shape[0]
    )

    manifest = {
        "input_file": str(INPUT_FILE),
        "output_file": str(OUTPUT_FILE),
        "input_rows": int(input_rows),
        "input_ships": int(input_ship_count),
        "removed_ship_id": BAD_SHIP_ID,
        "removed_ship_rows": int(
            len(bad_ship_rows)
        ),
        "removed_invalid_draught_trim_rows": int(
            len(removed_draught_trim)
        ),
        "removed_fuel_rows": int(
            len(removed_fuel)
        ),
        "removed_wave_period_rows": int(
            len(removed_wave)
        ),
        "removed_temperature_rows": int(
            len(removed_temperature)
        ),
        "speed_rows_deleted": 0,
        "speed_review_rows": int(
            len(speed_flagged)
        ),
        "trim_rows_deleted_by_value": 0,
        "trim_convention": (
            "aft_draught_minus_fore_draught"
        ),
        "final_rows": int(len(output)),
        "final_ships": int(
            final_ship_count
        ),
        "rules": {
            "wave_period_s": [
                WAVE_PERIOD_MIN,
                WAVE_PERIOD_MAX,
            ],
            "surface_temperature_c": [
                TEMPERATURE_MIN,
                TEMPERATURE_MAX,
            ],
            "fuel_sfoc_max_g_kwh": (
                SFOC_MAX_G_KWH
            ),
            "speed_service_ratio_review": (
                SPEED_SERVICE_RATIO_LIMIT
            ),
            "speed_upper_quantile_review": (
                SPEED_UPPER_QUANTILE
            ),
            "speed_iqr_multiplier_review": (
                SPEED_IQR_MULTIPLIER
            ),
        },
    }

    (
        AUDIT_DIR
        / "11_cleaning_manifest.json"
    ).write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    summary = [
        "FINAL FIXED28 QUALITY CHECK",
        "=" * 78,
        f"输入文件：{INPUT_FILE}",
        f"输出文件：{OUTPUT_FILE}",
        "",
        f"输入记录：{input_rows:,}",
        f"输入船舶：{input_ship_count}",
        "",
        f"整船删除：{BAD_SHIP_ID}",
        f"整船删除记录：{len(bad_ship_rows):,}",
        (
            "删除无有效吃水或纵倾缺失记录："
            f"{len(removed_draught_trim):,}"
        ),
        (
            "删除燃油目标异常："
            f"{len(removed_fuel):,}"
        ),
        (
            "删除波周期异常："
            f"{len(removed_wave):,}"
        ),
        (
            "删除温度异常："
            f"{len(removed_temperature):,}"
        ),
        "",
        "航速：仅复核，没有自动删除。",
        (
            "航速复核记录："
            f"{len(speed_flagged):,}"
        ),
        "",
        "纵倾：尾吃水减首吃水。",
        "未按纵倾数值删除记录。",
        (
            "每艘船纵倾<-2数量见："
            "08_trim_below_minus2_by_ship.csv"
        ),
        "",
        f"最终记录：{len(output):,}",
        f"最终船舶：{final_ship_count}",
    ]

    if final_ship_count == 21:
        summary.append(
            "船舶数量检查：通过，最终为21艘。"
        )
    else:
        summary.append(
            "警告：最终船舶数量不是21艘，"
            "请检查10_final_ship_distribution.csv。"
        )

    (
        AUDIT_DIR
        / "00_summary.txt"
    ).write_text(
        "\n".join(summary),
        encoding="utf-8-sig",
    )

    print("[9/9] 完成。")
    print(f"最终文件：{OUTPUT_FILE}")
    print(f"审计目录：{AUDIT_DIR}")
    print(f"最终记录：{len(output):,}")
    print(f"最终船舶：{final_ship_count}")
    print("请先打开00_summary.txt。")


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
            "\n运行失败："
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        raise