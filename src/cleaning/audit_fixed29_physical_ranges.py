#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
对 final_fixed29_model_ready.csv 进行物理范围审计。

重要原则：
1. 首、尾吃水 <= 0 属于硬错误，自动删除。
2. 相对设计吃水过低仅作为复核标记，默认不删除。
3. 低燃油采用“同船舶 + 相近航速区间”的稳健统计方法，
   默认只标记，不删除。
4. 输出阈值敏感性表、行级日志和候选数据。
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
    r"data\05_clean23\final_fixed30_physical_review.csv"
)

AUDIT_DIR = Path(
    r"data\05_clean23\final_fixed30_physical_audit"
)


# ============================================================
# 是否执行经验阈值删除
#
# 第一次运行保持False，只生成报告。
# 审查报告后再决定是否改为True。
# ============================================================

APPLY_DRAUGHT_REVIEW_FILTER = False
APPLY_LOW_FUEL_REVIEW_FILTER = False


# ============================================================
# 字段
# ============================================================

SHIP_TYPE = "ship_type"
SHIP_ID = "pseudo_ship_group_id"

TARGET = "fuel_t_10min"
SPEED = "speed_kn"
SERVICE_SPEED = "service_speed_kn"

MEAN_DRAUGHT = "mean_draught_m"
TRIM = "trim_m"
DESIGN_DRAUGHT = "design_draught_m"

POWER = "main_engine_power_kw"

WAVE_PERIOD = "wave_period_s"
TEMPERATURE = "surface_temperature_c"


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


# ============================================================
# 复核参数
# ============================================================

# 仅作为敏感性分析，不代表统一国际标准。
DRAUGHT_RATIO_LEVELS = [
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
]

ABSOLUTE_DRAUGHT_LEVELS_M = [
    0.5,
    1.0,
    1.5,
    2.0,
]

# 选中的“宽口径复核范围”
SELECTED_DRAUGHT_RATIO = 0.20
SELECTED_ABSOLUTE_DRAUGHT_M = 0.50

# 低燃油稳健检测
SPEED_BIN_WIDTH_KN = 0.5
MIN_SPEED_BIN_ROWS = 50
SHIP_LOW_FUEL_QUANTILE = 0.005
LOW_FUEL_MAD_MULTIPLIER = 6.0

# 已有项目规则，仅重新核对
SPEED_SERVICE_RATIO_MAX = 1.5
WAVE_PERIOD_MIN = 0.5
WAVE_PERIOD_MAX = 25.0
TEMPERATURE_MIN = -5.0
TEMPERATURE_MAX = 37.0

# 燃油上界复核
SFOC_MAX_G_KWH = 250.0
FUEL_POWER_FACTOR = 1.0


# ============================================================
# 辅助函数
# ============================================================

def configure_console():
    """尽量避免Windows终端中文乱码。"""

    for stream_name in ("stdout", "stderr"):

        stream = getattr(
            sys,
            stream_name,
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
    """保存审计CSV。"""

    frame.to_csv(
        AUDIT_DIR / filename,
        index=False,
        encoding="utf-8-sig",
    )


def require_columns(frame, columns):
    """检查必需字段。"""

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


def normalize_identifiers(frame):
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


def convert_numeric(frame):
    """转换本次审计使用的数值字段。"""

    numeric_columns = list(
        dict.fromkeys(
            MODEL_VARIABLES
            + [
                SERVICE_SPEED,
                DESIGN_DRAUGHT,
                POWER,
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


# ============================================================
# 吃水审计
# ============================================================

def reconstruct_draught(frame):
    """
    项目纵倾定义：
        trim = 尾吃水 - 首吃水

    因此：
        首吃水 = 平均吃水 - trim / 2
        尾吃水 = 平均吃水 + trim / 2
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

    frame["fore_design_draught_ratio"] = (
        frame["derived_fore_draught_m"]
        / frame[DESIGN_DRAUGHT]
    )

    frame["aft_design_draught_ratio"] = (
        frame["derived_aft_draught_m"]
        / frame[DESIGN_DRAUGHT]
    )

    frame["minimum_end_design_ratio"] = (
        frame[
            [
                "fore_design_draught_ratio",
                "aft_design_draught_ratio",
            ]
        ]
        .min(axis=1)
    )


def build_draught_sensitivity(frame):
    """生成每艘船的吃水阈值敏感性统计。"""

    records = []

    grouped = frame.groupby(
        [SHIP_TYPE, SHIP_ID],
        dropna=False,
        sort=True,
    )

    for (ship_type, ship_id), group in grouped:

        row = {
            SHIP_TYPE: ship_type,
            SHIP_ID: ship_id,
            "rows": len(group),
            "design_draught_m": (
                group[DESIGN_DRAUGHT].median()
            ),
            "mean_draught_min_m": (
                group[MEAN_DRAUGHT].min()
            ),
            "fore_draught_min_m": (
                group[
                    "derived_fore_draught_m"
                ].min()
            ),
            "aft_draught_min_m": (
                group[
                    "derived_aft_draught_m"
                ].min()
            ),
            "minimum_end_draught_min_m": (
                group[
                    "minimum_end_draught_m"
                ].min()
            ),
            "minimum_design_ratio_min": (
                group[
                    "minimum_end_design_ratio"
                ].min()
            ),
            "minimum_design_ratio_p0_1": (
                group[
                    "minimum_end_design_ratio"
                ].quantile(0.001)
            ),
            "minimum_design_ratio_p1": (
                group[
                    "minimum_end_design_ratio"
                ].quantile(0.01)
            ),
        }

        for ratio in DRAUGHT_RATIO_LEVELS:

            label = int(ratio * 100)

            row[
                f"rows_below_{label}pct_design"
            ] = int(
                (
                    group[
                        "minimum_end_design_ratio"
                    ] < ratio
                ).sum()
            )

        for level in ABSOLUTE_DRAUGHT_LEVELS_M:

            label = str(level).replace(
                ".",
                "p",
            )

            row[
                f"rows_below_{label}m"
            ] = int(
                (
                    group[
                        "minimum_end_draught_m"
                    ] < level
                ).sum()
            )

        records.append(row)

    return pd.DataFrame(records)


def build_overall_draught_sensitivity(frame):
    """生成全数据的吃水敏感性统计。"""

    records = []

    for ratio in DRAUGHT_RATIO_LEVELS:

        flag = (
            frame[
                "minimum_end_design_ratio"
            ] < ratio
        )

        records.append({
            "criterion": (
                "minimum_end_draught"
                "_divided_by_design_draught"
            ),
            "threshold": ratio,
            "flagged_rows": int(flag.sum()),
            "flagged_pct": (
                flag.mean() * 100
            ),
            "flagged_ships": (
                frame.loc[
                    flag,
                    [SHIP_TYPE, SHIP_ID],
                ]
                .drop_duplicates()
                .shape[0]
            ),
        })

    for level in ABSOLUTE_DRAUGHT_LEVELS_M:

        flag = (
            frame[
                "minimum_end_draught_m"
            ] < level
        )

        records.append({
            "criterion": (
                "minimum_end_draught_m"
            ),
            "threshold": level,
            "flagged_rows": int(flag.sum()),
            "flagged_pct": (
                flag.mean() * 100
            ),
            "flagged_ships": (
                frame.loc[
                    flag,
                    [SHIP_TYPE, SHIP_ID],
                ]
                .drop_duplicates()
                .shape[0]
            ),
        })

    return pd.DataFrame(records)


def selected_draught_review_flag(frame):
    """
    宽口径复核标记。

    默认只输出，不删除。
    """

    absolute_flag = (
        frame[
            "minimum_end_draught_m"
        ] < SELECTED_ABSOLUTE_DRAUGHT_M
    )

    ratio_flag = (
        np.isfinite(frame[DESIGN_DRAUGHT])
        & frame[DESIGN_DRAUGHT].gt(0)
        & (
            frame[
                "minimum_end_design_ratio"
            ] < SELECTED_DRAUGHT_RATIO
        )
    )

    return absolute_flag | ratio_flag


# ============================================================
# 低燃油审计
# ============================================================

def build_low_fuel_review(frame):
    """
    同船舶、相近航速区间内检测异常低燃油。

    同时满足以下条件才标记：
    1. 航速区间至少有50条记录；
    2. 低于该航速区间中位数减6倍稳健尺度；
    3. 同时低于该船燃油的0.5%分位数。

    默认只标记，不删除。
    """

    working = frame.copy()

    working["speed_bin_lower_kn"] = (
        np.floor(
            working[SPEED]
            / SPEED_BIN_WIDTH_KN
        )
        * SPEED_BIN_WIDTH_KN
    )

    group_columns = [
        SHIP_TYPE,
        SHIP_ID,
        "speed_bin_lower_kn",
    ]

    grouped = working.groupby(
        group_columns,
        dropna=False,
        sort=False,
    )

    working["speed_bin_rows"] = (
        grouped[TARGET].transform("size")
    )

    working["speed_bin_fuel_median"] = (
        grouped[TARGET].transform("median")
    )

    working["speed_bin_fuel_mad"] = (
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

    working["speed_bin_fuel_q25"] = (
        grouped[TARGET].transform(
            lambda values: values.quantile(
                0.25
            )
        )
    )

    working["speed_bin_fuel_q75"] = (
        grouped[TARGET].transform(
            lambda values: values.quantile(
                0.75
            )
        )
    )

    mad_scale = (
        1.4826
        * working[
            "speed_bin_fuel_mad"
        ]
    )

    iqr_scale = (
        (
            working[
                "speed_bin_fuel_q75"
            ]
            - working[
                "speed_bin_fuel_q25"
            ]
        )
        / 1.349
    )

    working[
        "speed_bin_robust_scale"
    ] = pd.concat(
        [
            mad_scale,
            iqr_scale,
        ],
        axis=1,
    ).max(axis=1)

    working[
        "speed_bin_fuel_lower_limit"
    ] = (
        working[
            "speed_bin_fuel_median"
        ]
        - LOW_FUEL_MAD_MULTIPLIER
        * working[
            "speed_bin_robust_scale"
        ]
    )

    working[
        "ship_fuel_p0_5"
    ] = (
        working.groupby(
            [SHIP_TYPE, SHIP_ID],
            dropna=False,
        )[TARGET]
        .transform(
            lambda values: values.quantile(
                SHIP_LOW_FUEL_QUANTILE
            )
        )
    )

    enough_rows = (
        working["speed_bin_rows"]
        >= MIN_SPEED_BIN_ROWS
    )

    positive_scale = (
        working[
            "speed_bin_robust_scale"
        ] > 0
    )

    below_local_limit = (
        working[TARGET]
        < working[
            "speed_bin_fuel_lower_limit"
        ]
    )

    below_ship_tail = (
        working[TARGET]
        <= working["ship_fuel_p0_5"]
    )

    invalid_target = (
        ~np.isfinite(working[TARGET])
        | working[TARGET].le(0)
    )

    working[
        "low_fuel_review_flag"
    ] = (
        invalid_target
        | (
            enough_rows
            & positive_scale
            & below_local_limit
            & below_ship_tail
        )
    )

    flagged = working[
        working["low_fuel_review_flag"]
    ].copy()

    summary = (
        working.groupby(
            [SHIP_TYPE, SHIP_ID],
            dropna=False,
        )
        .agg(
            rows=(TARGET, "size"),
            fuel_min=(TARGET, "min"),
            fuel_p0_1=(
                TARGET,
                lambda values: values.quantile(
                    0.001
                ),
            ),
            fuel_p0_5=(
                TARGET,
                lambda values: values.quantile(
                    0.005
                ),
            ),
            fuel_p1=(
                TARGET,
                lambda values: values.quantile(
                    0.01
                ),
            ),
            fuel_median=(TARGET, "median"),
            low_fuel_review_rows=(
                "low_fuel_review_flag",
                "sum",
            ),
        )
        .reset_index()
    )

    summary["low_fuel_review_pct"] = (
        summary[
            "low_fuel_review_rows"
        ]
        / summary["rows"]
        * 100
    )

    return working, summary, flagged


# ============================================================
# 其他复核
# ============================================================

def build_range_recheck(frame):
    """重新检查已经执行过的项目范围。"""

    speed_ratio = (
        frame[SPEED]
        / frame[SERVICE_SPEED]
    )

    theoretical_fuel_max = (
        frame[POWER]
        * SFOC_MAX_G_KWH
        * FUEL_POWER_FACTOR
        / 6_000_000.0
    )

    return pd.DataFrame([
        {
            "check": (
                "speed_over_service_speed"
            ),
            "rule": (
                f"<= {SPEED_SERVICE_RATIO_MAX}"
            ),
            "violation_rows": int(
                (
                    np.isfinite(speed_ratio)
                    & (
                        speed_ratio
                        > SPEED_SERVICE_RATIO_MAX
                    )
                ).sum()
            ),
        },
        {
            "check": WAVE_PERIOD,
            "rule": (
                f"{WAVE_PERIOD_MIN}"
                f" to {WAVE_PERIOD_MAX} s"
            ),
            "violation_rows": int(
                (
                    ~np.isfinite(
                        frame[WAVE_PERIOD]
                    )
                    | (
                        frame[WAVE_PERIOD]
                        < WAVE_PERIOD_MIN
                    )
                    | (
                        frame[WAVE_PERIOD]
                        > WAVE_PERIOD_MAX
                    )
                ).sum()
            ),
        },
        {
            "check": TEMPERATURE,
            "rule": (
                f"{TEMPERATURE_MIN}"
                f" to {TEMPERATURE_MAX} C"
            ),
            "violation_rows": int(
                (
                    ~np.isfinite(
                        frame[TEMPERATURE]
                    )
                    | (
                        frame[TEMPERATURE]
                        < TEMPERATURE_MIN
                    )
                    | (
                        frame[TEMPERATURE]
                        > TEMPERATURE_MAX
                    )
                ).sum()
            ),
        },
        {
            "check": (
                "fuel_power_upper_limit"
            ),
            "rule": (
                "rated power and "
                "250 g/kWh"
            ),
            "violation_rows": int(
                (
                    np.isfinite(frame[POWER])
                    & frame[POWER].gt(0)
                    & frame[TARGET].gt(
                        theoretical_fuel_max
                    )
                ).sum()
            ),
        },
    ])


def model_completeness(frame):
    """检查模型变量是否完整。"""

    matrix = np.column_stack([
        pd.to_numeric(
            frame[column],
            errors="coerce",
        ).to_numpy(dtype=float)
        for column in MODEL_VARIABLES
    ])

    complete = (
        np.isfinite(matrix).all(axis=1)
    )

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
        ),
    }])


# ============================================================
# 主程序
# ============================================================

def main():

    configure_console()

    if not INPUT_FILE.is_file():

        raise FileNotFoundError(
            f"找不到输入文件：{INPUT_FILE}"
        )

    if (
        INPUT_FILE.resolve()
        == OUTPUT_FILE.resolve()
    ):

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

    require_columns(
        df,
        [
            SHIP_TYPE,
            SHIP_ID,
            DESIGN_DRAUGHT,
            SERVICE_SPEED,
            POWER,
        ]
        + MODEL_VARIABLES,
    )

    normalize_identifiers(df)
    convert_numeric(df)

    input_rows = len(df)

    input_ships = (
        df[
            [SHIP_TYPE, SHIP_ID]
        ]
        .drop_duplicates()
        .shape[0]
    )

    # --------------------------------------------------------
    # 1. 反推首尾吃水
    # --------------------------------------------------------

    print("[2/8] 反推首尾吃水……")

    reconstruct_draught(df)

    hard_invalid_draught = (
        ~np.isfinite(
            df["derived_fore_draught_m"]
        )
        | ~np.isfinite(
            df["derived_aft_draught_m"]
        )
        | df[
            "derived_fore_draught_m"
        ].le(0)
        | df[
            "derived_aft_draught_m"
        ].le(0)
    )

    hard_invalid_rows = df.loc[
        hard_invalid_draught
    ].copy()

    save_csv(
        hard_invalid_rows,
        "01_removed_hard_invalid_draught.csv",
    )

    # 硬错误自动删除
    df = df.loc[
        ~hard_invalid_draught
    ].copy()

    # --------------------------------------------------------
    # 2. 吃水敏感性分析
    # --------------------------------------------------------

    print("[3/8] 生成吃水阈值敏感性表……")

    draught_by_ship = (
        build_draught_sensitivity(df)
    )

    draught_overall = (
        build_overall_draught_sensitivity(
            df
        )
    )

    save_csv(
        draught_by_ship,
        "02_draught_sensitivity_by_ship.csv",
    )

    save_csv(
        draught_overall,
        "03_draught_sensitivity_overall.csv",
    )

    draught_review_flag = (
        selected_draught_review_flag(df)
    )

    draught_review_rows = df.loc[
        draught_review_flag
    ].copy()

    draught_review_rows[
        "selected_absolute_floor_m"
    ] = SELECTED_ABSOLUTE_DRAUGHT_M

    draught_review_rows[
        "selected_design_ratio"
    ] = SELECTED_DRAUGHT_RATIO

    save_csv(
        draught_review_rows,
        "04_draught_review_rows.csv",
    )

    # --------------------------------------------------------
    # 3. 低燃油稳健复核
    # --------------------------------------------------------

    print("[4/8] 生成低燃油稳健复核表……")

    (
        df_with_fuel_flags,
        low_fuel_by_ship,
        low_fuel_rows,
    ) = build_low_fuel_review(df)

    save_csv(
        low_fuel_by_ship,
        "05_low_fuel_review_by_ship.csv",
    )

    low_fuel_output_columns = [
        column
        for column in [
            "__source_row_number",
            SHIP_TYPE,
            SHIP_ID,
            SPEED,
            SERVICE_SPEED,
            TARGET,
            POWER,
            MEAN_DRAUGHT,
            TRIM,
            "speed_bin_lower_kn",
            "speed_bin_rows",
            "speed_bin_fuel_median",
            "speed_bin_robust_scale",
            "speed_bin_fuel_lower_limit",
            "ship_fuel_p0_5",
            "low_fuel_review_flag",
        ]
        if column in low_fuel_rows.columns
    ]

    save_csv(
        low_fuel_rows[
            low_fuel_output_columns
        ],
        "06_low_fuel_review_rows.csv",
    )

    low_fuel_source_rows = set(
        low_fuel_rows[
            "__source_row_number"
        ].tolist()
    )

    low_fuel_flag = (
        df[
            "__source_row_number"
        ].isin(low_fuel_source_rows)
    )

    # --------------------------------------------------------
    # 4. 已有范围重新核对
    # --------------------------------------------------------

    print("[5/8] 重新核对已有物理范围……")

    range_recheck = build_range_recheck(
        df
    )

    save_csv(
        range_recheck,
        "07_existing_range_recheck.csv",
    )

    # --------------------------------------------------------
    # 5. 可选删除
    # --------------------------------------------------------

    print("[6/8] 生成候选数据……")

    delete_flag = pd.Series(
        False,
        index=df.index,
    )

    if APPLY_DRAUGHT_REVIEW_FILTER:

        delete_flag = (
            delete_flag
            | draught_review_flag
        )

    if APPLY_LOW_FUEL_REVIEW_FILTER:

        delete_flag = (
            delete_flag
            | low_fuel_flag
        )

    removal_log = df.loc[
        delete_flag
    ].copy()

    if not removal_log.empty:

        removal_log[
            "draught_review_flag"
        ] = draught_review_flag.loc[
            delete_flag
        ].to_numpy()

        removal_log[
            "low_fuel_review_flag"
        ] = low_fuel_flag.loc[
            delete_flag
        ].to_numpy()

    save_csv(
        removal_log,
        "08_optional_filter_removal_log.csv",
    )

    candidate = df.loc[
        ~delete_flag
    ].copy()

    # --------------------------------------------------------
    # 6. 重新计算交互变量
    # --------------------------------------------------------

    for interaction, (
        left,
        right,
    ) in INTERACTIONS.items():

        candidate[interaction] = (
            candidate[left]
            * candidate[right]
        )

    # --------------------------------------------------------
    # 7. 完整性和船级分布
    # --------------------------------------------------------

    completeness = model_completeness(
        candidate
    )

    save_csv(
        completeness,
        "09_candidate_model_completeness.csv",
    )

    if (
        int(
            completeness.iloc[0][
                "incomplete_model_rows"
            ]
        )
        != 0
    ):

        raise RuntimeError(
            "候选数据仍存在模型变量缺失。"
        )

    final_distribution = (
        candidate.groupby(
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
                lambda values: values.quantile(
                    0.005
                ),
            ),
            fuel_median=(TARGET, "median"),
        )
        .reset_index()
        .sort_values(
            [SHIP_TYPE, SHIP_ID]
        )
    )

    save_csv(
        final_distribution,
        "10_candidate_ship_distribution.csv",
    )

    # --------------------------------------------------------
    # 8. 正式输出
    # --------------------------------------------------------

    internal_columns = [
        "__source_row_number",
        "derived_fore_draught_m",
        "derived_aft_draught_m",
        "minimum_end_draught_m",
        "fore_design_draught_ratio",
        "aft_design_draught_ratio",
        "minimum_end_design_ratio",
    ]

    output = candidate.drop(
        columns=[
            column
            for column in internal_columns
            if column in candidate.columns
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
        "removed_hard_invalid_draught_rows": (
            int(hard_invalid_draught.sum())
        ),
        "draught_review_rows": int(
            draught_review_flag.sum()
        ),
        "low_fuel_review_rows": int(
            low_fuel_flag.sum()
        ),
        "apply_draught_review_filter": (
            APPLY_DRAUGHT_REVIEW_FILTER
        ),
        "apply_low_fuel_review_filter": (
            APPLY_LOW_FUEL_REVIEW_FILTER
        ),
        "optional_deleted_rows": int(
            delete_flag.sum()
        ),
        "final_rows": int(len(output)),
        "final_ships": int(final_ships),
        "parameters": {
            "draught_ratio_levels": (
                DRAUGHT_RATIO_LEVELS
            ),
            "absolute_draught_levels_m": (
                ABSOLUTE_DRAUGHT_LEVELS_M
            ),
            "selected_draught_ratio": (
                SELECTED_DRAUGHT_RATIO
            ),
            "selected_absolute_draught_m": (
                SELECTED_ABSOLUTE_DRAUGHT_M
            ),
            "speed_bin_width_kn": (
                SPEED_BIN_WIDTH_KN
            ),
            "minimum_speed_bin_rows": (
                MIN_SPEED_BIN_ROWS
            ),
            "ship_low_fuel_quantile": (
                SHIP_LOW_FUEL_QUANTILE
            ),
            "low_fuel_mad_multiplier": (
                LOW_FUEL_MAD_MULTIPLIER
            ),
        },
    }

    (
        AUDIT_DIR
        / "11_manifest.json"
    ).write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    summary = [
        "FIXED29 PHYSICAL RANGE AUDIT",
        "=" * 78,
        f"输入记录：{input_rows:,}",
        f"输入船舶：{input_ships}",
        "",
        (
            "删除硬吃水错误："
            f"{int(hard_invalid_draught.sum()):,}"
        ),
        (
            "设计吃水/绝对吃水复核记录："
            f"{int(draught_review_flag.sum()):,}"
        ),
        (
            "低燃油稳健复核记录："
            f"{int(low_fuel_flag.sum()):,}"
        ),
        "",
        (
            "应用吃水复核删除："
            f"{APPLY_DRAUGHT_REVIEW_FILTER}"
        ),
        (
            "应用低燃油复核删除："
            f"{APPLY_LOW_FUEL_REVIEW_FILTER}"
        ),
        (
            "经验复核规则实际删除："
            f"{int(delete_flag.sum()):,}"
        ),
        "",
        f"输出记录：{len(output):,}",
        f"输出船舶：{final_ships}",
        f"候选数据：{OUTPUT_FILE}",
        "",
        "优先查看：",
        "03_draught_sensitivity_overall.csv",
        "02_draught_sensitivity_by_ship.csv",
        "04_draught_review_rows.csv",
        "05_low_fuel_review_by_ship.csv",
        "06_low_fuel_review_rows.csv",
        "",
        (
            "注意：默认不删除设计吃水比例"
            "和低燃油复核记录。"
        ),
    ]

    (
        AUDIT_DIR
        / "00_summary.txt"
    ).write_text(
        "\n".join(summary),
        encoding="utf-8-sig",
    )

    print("[8/8] 完成。")
    print(f"审计目录：{AUDIT_DIR}")
    print(f"候选数据：{OUTPUT_FILE}")
    print(
        "吃水复核记录："
        f"{int(draught_review_flag.sum()):,}"
    )
    print(
        "低燃油复核记录："
        f"{int(low_fuel_flag.sum()):,}"
    )
    print(
        "经验规则删除记录："
        f"{int(delete_flag.sum()):,}"
    )
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