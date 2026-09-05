#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
从fixed29生成最终fixed31建模数据。

自动删除：
1. 反推首吃水或尾吃水不是有限数值，或<=0；
2. 最小端部吃水 < 设计吃水的5%；
3. 严重低燃油掉零：
   - 已通过同船、同航速区间稳健异常检测；
   - 低于该船燃油0.5%分位数；
   - 低于局部燃油中位数的10%。

仅复核、不删除：
1. 最小端部吃水处于设计吃水5%～20%；
2. 被稳健规则标记、但不属于严重掉零的低燃油记录。
"""

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd


# ============================================================
# 路径
# ============================================================

INPUT_FILE = Path(
    r"data\05_clean23\final_fixed29_model_ready.csv"
)

OUTPUT_FILE = Path(
    r"data\05_clean23\final_fixed31_model_ready.csv"
)

AUDIT_DIR = Path(
    r"data\05_clean23\final_fixed31_audit"
)


# ============================================================
# 字段
# ============================================================

SHIP_TYPE = "ship_type"
SHIP_ID = "pseudo_ship_group_id"

TARGET = "fuel_t_10min"
SPEED = "speed_kn"
SERVICE_SPEED = "service_speed_kn"
POWER = "main_engine_power_kw"

MEAN_DRAUGHT = "mean_draught_m"
TRIM = "trim_m"
DESIGN_DRAUGHT = "design_draught_m"

WAVE_PERIOD = "wave_period_s"
TEMPERATURE = "surface_temperature_c"


# ============================================================
# 最终规则
# ============================================================

# 低于设计吃水5%：自动删除
MIN_END_DESIGN_RATIO_DELETE = 0.05

# 5%～20%：仅复核
MIN_END_DESIGN_RATIO_REVIEW = 0.20

# 低燃油稳健检测
SPEED_BIN_WIDTH_KN = 0.5
MIN_SPEED_BIN_ROWS = 50
SHIP_FUEL_LOW_QUANTILE = 0.005
LOW_FUEL_SCALE_MULTIPLIER = 6.0

# 严重低燃油：低于局部中位数10%
SEVERE_FUEL_MEDIAN_RATIO = 0.10

# 已有质量控制规则
MAX_SPEED_SERVICE_RATIO = 1.5

WAVE_PERIOD_MIN = 0.5
WAVE_PERIOD_MAX = 25.0

TEMPERATURE_MIN = -5.0
TEMPERATURE_MAX = 37.0

SFOC_MAX_G_KWH = 250.0
MINIMUM_FUEL = 1e-5


MODEL_VARIABLES = [
    TARGET,
    SPEED,
    "heading_sin",
    "heading_cos",
    MEAN_DRAUGHT,
    "rudder_deg",
    TRIM,
    "rel_wind_speed_kn",
    "relative_wind_sin",
    "relative_wind_cos",
    "wave_height_m",
    WAVE_PERIOD,
    "relative_wave_sin",
    "relative_wave_cos",
    "surface_pressure_pa",
    TEMPERATURE,
]


INTERACTIONS = {
    "rel_wind_speed_x_speed": (
        "rel_wind_speed_kn",
        SPEED,
    ),
    "wave_height_x_speed": (
        "wave_height_m",
        SPEED,
    ),
    "draught_x_speed": (
        MEAN_DRAUGHT,
        SPEED,
    ),
    "trim_x_speed": (
        TRIM,
        SPEED,
    ),
    "draught_x_trim": (
        MEAN_DRAUGHT,
        TRIM,
    ),
}


def configure_console():
    for name in ("stdout", "stderr"):

        stream = getattr(
            sys,
            name,
            None,
        )

        if stream is None:
            continue

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


def require_columns(frame, columns):

    missing = [
        column
        for column in columns
        if column not in frame.columns
    ]

    if missing:
        raise KeyError(
            "输入文件缺少字段："
            + ", ".join(missing)
        )


def normalize_data(frame):

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

    numeric_columns = list(
        dict.fromkeys(
            MODEL_VARIABLES
            + [
                SERVICE_SPEED,
                POWER,
                DESIGN_DRAUGHT,
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


def reconstruct_draught(frame):
    """
    纵倾定义：
        trim = 尾吃水 - 首吃水
    """

    frame["derived_fore_draught_m"] = (
        frame[MEAN_DRAUGHT]
        - frame[TRIM] / 2.0
    )

    frame["derived_aft_draught_m"] = (
        frame[MEAN_DRAUGHT]
        + frame[TRIM] / 2.0
    )

    frame["minimum_end_draught_m"] = (
        frame[
            [
                "derived_fore_draught_m",
                "derived_aft_draught_m",
            ]
        ]
        .min(axis=1)
    )

    frame["minimum_end_design_ratio"] = (
        frame["minimum_end_draught_m"]
        / frame[DESIGN_DRAUGHT]
    )


def add_low_fuel_flags(frame):

    frame = frame.copy()

    # 每0.5节划分一个船内航速区间
    frame["speed_bin_lower_kn"] = (
        np.floor(
            frame[SPEED]
            / SPEED_BIN_WIDTH_KN
        )
        * SPEED_BIN_WIDTH_KN
    )

    grouping = [
        SHIP_TYPE,
        SHIP_ID,
        "speed_bin_lower_kn",
    ]

    grouped = frame.groupby(
        grouping,
        dropna=False,
        sort=False,
    )

    frame["speed_bin_rows"] = (
        grouped[TARGET].transform("size")
    )

    frame["local_fuel_median"] = (
        grouped[TARGET].transform("median")
    )

    frame["local_fuel_mad"] = (
        grouped[TARGET].transform(
            lambda values: float(
                np.median(
                    np.abs(
                        values
                        - np.median(values)
                    )
                )
            )
        )
    )

    frame["local_fuel_q25"] = (
        grouped[TARGET].transform(
            lambda values:
            values.quantile(0.25)
        )
    )

    frame["local_fuel_q75"] = (
        grouped[TARGET].transform(
            lambda values:
            values.quantile(0.75)
        )
    )

    mad_scale = (
        1.4826
        * frame["local_fuel_mad"]
    )

    iqr_scale = (
        (
            frame["local_fuel_q75"]
            - frame["local_fuel_q25"]
        )
        / 1.349
    )

    # 选择两个稳健尺度中较大的一个，
    # 避免阈值过于严格。
    frame["local_fuel_scale"] = (
        pd.concat(
            [
                mad_scale,
                iqr_scale,
            ],
            axis=1,
        )
        .max(axis=1)
    )

    frame["local_fuel_lower_limit"] = (
        frame["local_fuel_median"]
        - LOW_FUEL_SCALE_MULTIPLIER
        * frame["local_fuel_scale"]
    )

    frame["ship_fuel_p0_5"] = (
        frame.groupby(
            [SHIP_TYPE, SHIP_ID],
            dropna=False,
        )[TARGET]
        .transform(
            lambda values:
            values.quantile(
                SHIP_FUEL_LOW_QUANTILE
            )
        )
    )

    frame["fuel_local_median_ratio"] = (
        np.where(
            np.isfinite(
                frame["local_fuel_median"]
            )
            & frame[
                "local_fuel_median"
            ].gt(0),
            frame[TARGET]
            / frame["local_fuel_median"],
            np.nan,
        )
    )

    enough_rows = (
        frame["speed_bin_rows"]
        >= MIN_SPEED_BIN_ROWS
    )

    positive_scale = (
        frame["local_fuel_scale"] > 0
    )

    below_local_limit = (
        frame[TARGET]
        < frame["local_fuel_lower_limit"]
    )

    below_ship_tail = (
        frame[TARGET]
        <= frame["ship_fuel_p0_5"]
    )

    invalid_target = (
        ~np.isfinite(frame[TARGET])
        | frame[TARGET].le(0)
    )

    frame["low_fuel_review_flag"] = (
        invalid_target
        | (
            enough_rows
            & positive_scale
            & below_local_limit
            & below_ship_tail
        )
    )

    frame["severe_low_fuel_flag"] = (
        invalid_target
        | (
            frame["low_fuel_review_flag"]
            & frame[
                "fuel_local_median_ratio"
            ].lt(
                SEVERE_FUEL_MEDIAN_RATIO
            )
        )
    )

    return frame


def check_model_completeness(frame):

    matrix = np.column_stack([
        pd.to_numeric(
            frame[column],
            errors="coerce",
        ).to_numpy(
            dtype=float,
            na_value=np.nan,
        )
        for column in MODEL_VARIABLES
    ])

    complete = np.isfinite(
        matrix
    ).all(axis=1)

    return pd.DataFrame([{
        "rows": len(frame),
        "complete_model_rows": int(
            complete.sum()
        ),
        "incomplete_model_rows": int(
            (~complete).sum()
        ),
        "complete_model_row_pct": (
            complete.mean() * 100
            if len(frame)
            else np.nan
        ),
    }])


def main():

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

    print("[1/8] 读取fixed29数据……")

    df = pd.read_csv(
        INPUT_FILE,
        low_memory=False,
    )

    df.insert(
        0,
        "__source_row_number",
        np.arange(
            2,
            len(df) + 2,
            dtype=np.int64,
        ),
    )

    required = list(
        dict.fromkeys(
            [
                SHIP_TYPE,
                SHIP_ID,
                DESIGN_DRAUGHT,
                SERVICE_SPEED,
                POWER,
            ]
            + MODEL_VARIABLES
        )
    )

    require_columns(df, required)
    normalize_data(df)

    input_rows = len(df)

    input_ships = (
        df[
            [SHIP_TYPE, SHIP_ID]
        ]
        .drop_duplicates()
        .shape[0]
    )

    # ========================================================
    # 1. 吃水处理
    # ========================================================

    print("[2/8] 检查首尾吃水……")

    reconstruct_draught(df)

    hard_invalid_draught = (
        ~np.isfinite(
            df["derived_fore_draught_m"]
        )
        | ~np.isfinite(
            df["derived_aft_draught_m"]
        )
        | ~np.isfinite(
            df[DESIGN_DRAUGHT]
        )
        | df[DESIGN_DRAUGHT].le(0)
        | df[
            "derived_fore_draught_m"
        ].le(0)
        | df[
            "derived_aft_draught_m"
        ].le(0)
    )

    below_five_percent = (
        ~hard_invalid_draught
        & df[
            "minimum_end_design_ratio"
        ].lt(
            MIN_END_DESIGN_RATIO_DELETE
        )
    )

    remove_draught = (
        hard_invalid_draught
        | below_five_percent
    )

    removed_draught = df.loc[
        remove_draught
    ].copy()

    removed_draught[
        "hard_invalid_draught"
    ] = hard_invalid_draught.loc[
        remove_draught
    ].to_numpy()

    removed_draught[
        "below_5pct_design_draught"
    ] = below_five_percent.loc[
        remove_draught
    ].to_numpy()

    save_csv(
        removed_draught,
        "01_removed_draught_rows.csv",
    )

    df = df.loc[
        ~remove_draught
    ].copy()

    retained_draught_review = df.loc[
        df[
            "minimum_end_design_ratio"
        ].ge(
            MIN_END_DESIGN_RATIO_DELETE
        )
        & df[
            "minimum_end_design_ratio"
        ].lt(
            MIN_END_DESIGN_RATIO_REVIEW
        )
    ].copy()

    save_csv(
        retained_draught_review,
        "02_retained_draught_review_rows.csv",
    )

    # ========================================================
    # 2. 低燃油处理
    # ========================================================

    print("[3/8] 检测严重低燃油掉零……")

    df = add_low_fuel_flags(df)

    all_low_fuel_review = df.loc[
        df["low_fuel_review_flag"]
    ].copy()

    severe_low_fuel = df.loc[
        df["severe_low_fuel_flag"]
    ].copy()

    save_csv(
        severe_low_fuel,
        "03_removed_severe_low_fuel_rows.csv",
    )

    df = df.loc[
        ~df["severe_low_fuel_flag"]
    ].copy()

    retained_low_fuel_review = df.loc[
        df["low_fuel_review_flag"]
    ].copy()

    save_csv(
        retained_low_fuel_review,
        "04_retained_low_fuel_review_rows.csv",
    )

    # 船级低燃油统计
    low_fuel_summary = (
        all_low_fuel_review.groupby(
            [SHIP_TYPE, SHIP_ID],
            dropna=False,
        )
        .agg(
            original_review_rows=(
                TARGET,
                "size",
            ),
            removed_severe_rows=(
                "severe_low_fuel_flag",
                "sum",
            ),
            minimum_fuel=(TARGET, "min"),
            minimum_local_median_ratio=(
                "fuel_local_median_ratio",
                "min",
            ),
        )
        .reset_index()
    )

    low_fuel_summary[
        "retained_review_rows"
    ] = (
        low_fuel_summary[
            "original_review_rows"
        ]
        - low_fuel_summary[
            "removed_severe_rows"
        ]
    )

    save_csv(
        low_fuel_summary,
        "05_low_fuel_summary_by_ship.csv",
    )

    # ========================================================
    # 3. 重新计算交互变量
    # ========================================================

    print("[4/8] 重新计算交互变量……")

    for interaction, (
        left,
        right,
    ) in INTERACTIONS.items():

        df[interaction] = (
            df[left] * df[right]
        )

    # ========================================================
    # 4. 重新核对现有规则
    # ========================================================

    print("[5/8] 重新核对现有规则……")

    speed_ratio = (
        df[SPEED]
        / df[SERVICE_SPEED]
    )

    theoretical_fuel_max = (
        df[POWER]
        * SFOC_MAX_G_KWH
        / 6_000_000.0
    )

    range_recheck = pd.DataFrame([
        {
            "check":
            "speed_service_ratio",
            "violation_rows": int(
                (
                    np.isfinite(speed_ratio)
                    & speed_ratio.gt(
                        MAX_SPEED_SERVICE_RATIO
                    )
                ).sum()
            ),
        },
        {
            "check": WAVE_PERIOD,
            "violation_rows": int(
                (
                    ~np.isfinite(
                        df[WAVE_PERIOD]
                    )
                    | df[WAVE_PERIOD].lt(
                        WAVE_PERIOD_MIN
                    )
                    | df[WAVE_PERIOD].gt(
                        WAVE_PERIOD_MAX
                    )
                ).sum()
            ),
        },
        {
            "check": TEMPERATURE,
            "violation_rows": int(
                (
                    ~np.isfinite(
                        df[TEMPERATURE]
                    )
                    | df[TEMPERATURE].lt(
                        TEMPERATURE_MIN
                    )
                    | df[TEMPERATURE].gt(
                        TEMPERATURE_MAX
                    )
                ).sum()
            ),
        },
        {
            "check":
            "fuel_power_upper_limit",
            "violation_rows": int(
                (
                    np.isfinite(df[POWER])
                    & df[POWER].gt(0)
                    & df[TARGET].gt(
                        theoretical_fuel_max
                    )
                ).sum()
            ),
        },
        {
            "check":
            "minimum_positive_fuel",
            "violation_rows": int(
                (
                    ~np.isfinite(df[TARGET])
                    | df[TARGET].lt(
                        MINIMUM_FUEL
                    )
                ).sum()
            ),
        },
    ])

    save_csv(
        range_recheck,
        "06_existing_range_recheck.csv",
    )

    if int(
        range_recheck[
            "violation_rows"
        ].sum()
    ) != 0:

        raise RuntimeError(
            "最终数据仍有范围违规，"
            "请检查06_existing_range_recheck.csv。"
        )

    # ========================================================
    # 5. 完整性检查
    # ========================================================

    print("[6/8] 检查模型变量完整性……")

    completeness = (
        check_model_completeness(df)
    )

    save_csv(
        completeness,
        "07_final_model_completeness.csv",
    )

    if int(
        completeness.iloc[0][
            "incomplete_model_rows"
        ]
    ) != 0:

        raise RuntimeError(
            "最终数据仍有模型变量缺失。"
        )

    # ========================================================
    # 6. 船级分布
    # ========================================================

    distribution = (
        df.groupby(
            [SHIP_TYPE, SHIP_ID],
            dropna=False,
        )
        .agg(
            rows=(SHIP_ID, "size"),
            speed_min=(SPEED, "min"),
            speed_max=(SPEED, "max"),
            mean_draught_min=(
                MEAN_DRAUGHT,
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
            minimum_design_ratio=(
                "minimum_end_design_ratio",
                "min",
            ),
            fuel_min=(TARGET, "min"),
            fuel_p0_5=(
                TARGET,
                lambda values:
                values.quantile(0.005),
            ),
            fuel_median=(TARGET, "median"),
        )
        .reset_index()
        .sort_values(
            [SHIP_TYPE, SHIP_ID]
        )
    )

    save_csv(
        distribution,
        "08_final_ship_distribution.csv",
    )

    # ========================================================
    # 7. 输出正式文件
    # ========================================================

    print("[7/8] 输出fixed31数据……")

    internal_columns = [
        "__source_row_number",
        "derived_fore_draught_m",
        "derived_aft_draught_m",
        "minimum_end_draught_m",
        "minimum_end_design_ratio",
        "speed_bin_lower_kn",
        "speed_bin_rows",
        "local_fuel_median",
        "local_fuel_mad",
        "local_fuel_q25",
        "local_fuel_q75",
        "local_fuel_scale",
        "local_fuel_lower_limit",
        "ship_fuel_p0_5",
        "fuel_local_median_ratio",
        "low_fuel_review_flag",
        "severe_low_fuel_flag",
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
        output[
            [SHIP_TYPE, SHIP_ID]
        ]
        .drop_duplicates()
        .shape[0]
    )

    manifest = {
        "input": str(INPUT_FILE),
        "output": str(OUTPUT_FILE),
        "input_rows": int(input_rows),
        "input_ships": int(input_ships),

        "removed_draught_rows": int(
            len(removed_draught)
        ),

        "removed_hard_draught_rows": int(
            hard_invalid_draught.sum()
        ),

        "removed_below_5pct_design_rows": int(
            below_five_percent.sum()
        ),

        "retained_draught_review_rows": int(
            len(retained_draught_review)
        ),

        "original_low_fuel_review_rows": int(
            len(all_low_fuel_review)
        ),

        "removed_severe_low_fuel_rows": int(
            len(severe_low_fuel)
        ),

        "retained_low_fuel_review_rows": int(
            len(retained_low_fuel_review)
        ),

        "final_rows": int(len(output)),
        "final_ships": int(final_ships),

        "rules": {
            "draught_delete_ratio":
            MIN_END_DESIGN_RATIO_DELETE,

            "draught_review_upper_ratio":
            MIN_END_DESIGN_RATIO_REVIEW,

            "speed_bin_width_kn":
            SPEED_BIN_WIDTH_KN,

            "minimum_speed_bin_rows":
            MIN_SPEED_BIN_ROWS,

            "ship_low_fuel_quantile":
            SHIP_FUEL_LOW_QUANTILE,

            "low_fuel_scale_multiplier":
            LOW_FUEL_SCALE_MULTIPLIER,

            "severe_fuel_local_median_ratio":
            SEVERE_FUEL_MEDIAN_RATIO,
        },
    }

    (
        AUDIT_DIR
        / "09_cleaning_manifest.json"
    ).write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    summary = [
        "FINAL FIXED31 MODEL-READY DATA",
        "=" * 78,

        f"输入记录：{input_rows:,}",
        f"输入船舶：{input_ships}",

        "",

        (
            "删除吃水异常："
            f"{len(removed_draught):,}"
        ),

        (
            "其中低于设计吃水5%："
            f"{int(below_five_percent.sum()):,}"
        ),

        (
            "保留的5%～20%吃水复核记录："
            f"{len(retained_draught_review):,}"
        ),

        "",

        (
            "原低燃油复核记录："
            f"{len(all_low_fuel_review):,}"
        ),

        (
            "删除严重低燃油掉零："
            f"{len(severe_low_fuel):,}"
        ),

        (
            "保留的低燃油复核记录："
            f"{len(retained_low_fuel_review):,}"
        ),

        "",

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

    print("[8/8] 完成。")
    print(f"最终数据：{OUTPUT_FILE}")
    print(f"审计目录：{AUDIT_DIR}")
    print(
        f"删除吃水异常："
        f"{len(removed_draught):,}"
    )
    print(
        f"删除严重低燃油："
        f"{len(severe_low_fuel):,}"
    )
    print(f"最终记录：{len(output):,}")
    print(f"最终船舶：{final_ships}")
    print("请先查看00_summary.txt。")


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