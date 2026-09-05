#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
对 cruise、maneuver、anchor_berth 三个航行阶段执行完全相同的清洗链：

    Fixed27输入
        -> create_final_fixed28_quality_checked.py
        -> create_final_fixed29_model_ready.py
        -> audit_fixed29_physical_ranges.py（只审计，默认不额外删除）
        -> create_final_fixed31_model_ready.py

设计原则
--------
1. 三个阶段分别运行，不能混在一起计算统计阈值。
   Fixed31 的低燃油规则使用“同船 + 0.5节航速区间”的局部分布；
   分阶段运行可避免机动/锚泊数据改变巡航阈值。
2. cruise 直接使用既有 final_fixed27_cruise.csv 作为输入。
3. maneuver 和 anchor_berth 从 final_fixed23_all_phases.csv 提取，
   并要求四项静态船舶特征完整，形成与 Fixed27 相同的输入结构。
4. 自动从 all_phases 反向提取一份 cruise Fixed27 候选，并与既有
   final_fixed27_cruise.csv 比较；不一致立即停止。
5. 最终 cruise Fixed31 与既有 final_fixed31_model_ready.csv 逐主键、
   逐字段比较；不一致立即停止。可用 --skip-final-reference-check 跳过，
   但正式运行不建议跳过。
6. 所有原始清洗脚本通过 import 后直接调用 main()，不复制、不改写规则。
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

try:
    import duckdb  # type: ignore
except ImportError:
    duckdb = None


PHASES = ("cruise", "maneuver", "anchor_berth")

STATIC4 = [
    "design_draught_m",
    "deadweight_t",
    "service_speed_kn",
    "main_engine_power_kw",
]

KEY_COLUMNS = [
    "ship_type",
    "pseudo_ship_group_id",
    "trajectory_segment_id",
    "timestamp_utc",
]

SCRIPT_FILES = {
    "fixed28": "create_final_fixed28_quality_checked.py",
    "fixed29": "create_final_fixed29_model_ready.py",
    "fixed30_audit": "audit_fixed29_physical_ranges.py",
    "fixed31": "create_final_fixed31_model_ready.py",
}


def configure_console() -> None:
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        if stream is None:
            continue
        try:
            stream.reconfigure(
                encoding="utf-8",
                errors="backslashreplace",
            )
        except Exception:
            pass


def parse_args() -> argparse.Namespace:
    script_location = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description=(
            "使用现有 Fixed28/29/30/31 规则，分别清洗巡航、机动和锚泊数据，"
            "并严格验证巡航样本一致性。"
        )
    )
    parser.add_argument(
        "--script-dir",
        type=Path,
        default=script_location,
        help="四个原始清洗脚本所在目录；默认是本程序所在目录。",
    )
    parser.add_argument(
        "--all-phases",
        type=Path,
        default=Path(
            r"data\05_clean23\final_fixed23_all_phases.csv"
        ),
        help="包含 cruise、maneuver、anchor_berth 的 Fixed23 文件。",
    )
    parser.add_argument(
        "--cruise-fixed27",
        type=Path,
        default=Path(
            r"data\05_clean23\final_fixed27_cruise.csv"
        ),
        help="既有巡航 Fixed27，作为巡航清洗的唯一正式输入和结构基准。",
    )
    parser.add_argument(
        "--cruise-fixed31-reference",
        type=Path,
        default=Path(
            r"data\05_clean23\final_fixed31_model_ready.csv"
        ),
        help="既有巡航 Fixed31，作为最终一致性基准。",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            r"data\05_clean23\all_phase_equal_cleaning"
        ),
        help="三阶段统一清洗的输出根目录。",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=200_000,
        help="从 all_phases 分块提取阶段数据时的块大小。",
    )
    parser.add_argument(
        "--comparison-absolute-tolerance",
        type=float,
        default=1e-12,
        help="巡航数值一致性比较的绝对误差。",
    )
    parser.add_argument(
        "--comparison-relative-tolerance",
        type=float,
        default=1e-10,
        help="巡航数值一致性比较的相对误差。",
    )
    parser.add_argument(
        "--skip-final-reference-check",
        action="store_true",
        help=(
            "不比较既有 final_fixed31_model_ready.csv。"
            "仅当该参考文件确实不存在时使用。"
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="删除并重建 output-root。",
    )
    return parser.parse_args()


def quote_identifier(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def sql_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").replace("'", "''")


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def require_files(paths: Dict[str, Path]) -> None:
    missing = [
        f"{name}: {path}"
        for name, path in paths.items()
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "以下必需文件不存在：\n" + "\n".join(missing)
        )


def require_columns(
    columns: Sequence[str],
    required: Sequence[str],
    source: Path,
) -> None:
    missing = [
        column
        for column in required
        if column not in columns
    ]
    if missing:
        raise KeyError(
            f"{source} 缺少字段：{', '.join(missing)}"
        )


def load_module(path: Path, name: str) -> ModuleType:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise ImportError(f"无法加载脚本：{path}")

    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def reset_target(path: Path) -> None:
    if path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def csv_row_count(path: Path, chunksize: int = 300_000) -> int:
    total = 0
    for chunk in pd.read_csv(
        path,
        usecols=[0],
        chunksize=chunksize,
        low_memory=True,
    ):
        total += len(chunk)
    return total


def normalized_phase(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.strip()
        .str.lower()
    )


def prepare_fixed27_phase_inputs(
    all_phases_path: Path,
    cruise_reference_path: Path,
    output_dir: Path,
    chunksize: int,
) -> tuple[Dict[str, Path], pd.DataFrame]:
    """
    从 Fixed23 all_phases 中提取三个阶段，并要求四项静态特征完整。

    输出列严格按照既有 final_fixed27_cruise.csv 的列顺序，
    防止三个阶段出现不同的列结构。
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_columns = (
        pd.read_csv(
            cruise_reference_path,
            nrows=0,
            low_memory=False,
        )
        .columns.astype(str)
        .tolist()
    )
    all_columns = (
        pd.read_csv(
            all_phases_path,
            nrows=0,
            low_memory=False,
        )
        .columns.astype(str)
        .tolist()
    )

    require_columns(
        all_columns,
        ["voyage_phase"] + STATIC4,
        all_phases_path,
    )
    require_columns(
        all_columns,
        reference_columns,
        all_phases_path,
    )
    require_columns(
        reference_columns,
        KEY_COLUMNS + STATIC4,
        cruise_reference_path,
    )

    paths = {
        phase: output_dir / f"final_fixed27_{phase}_derived.csv"
        for phase in PHASES
    }
    for path in paths.values():
        path.unlink(missing_ok=True)

    first_write = {phase: True for phase in PHASES}
    total_by_phase = {phase: 0 for phase in PHASES}
    static_complete_by_phase = {phase: 0 for phase in PHASES}
    source_rows = 0

    use_columns = list(
        dict.fromkeys(
            reference_columns
            + ["voyage_phase"]
            + STATIC4
            + (
                ["static4_complete_flag"]
                if "static4_complete_flag" in all_columns
                else []
            )
        )
    )

    reader = pd.read_csv(
        all_phases_path,
        usecols=use_columns,
        chunksize=chunksize,
        low_memory=False,
        encoding="utf-8-sig",
    )

    for block_number, chunk in enumerate(reader, start=1):
        source_rows += len(chunk)
        phases = normalized_phase(chunk["voyage_phase"])

        static_matrix = np.column_stack([
            pd.to_numeric(
                chunk[column],
                errors="coerce",
            ).to_numpy(dtype=float)
            for column in STATIC4
        ])
        static_complete = np.isfinite(static_matrix).all(axis=1)

        if "static4_complete_flag" in chunk.columns:
            recorded_flag = (
                pd.to_numeric(
                    chunk["static4_complete_flag"],
                    errors="coerce",
                )
                .fillna(0)
                .eq(1)
                .to_numpy()
            )
            static_complete &= recorded_flag

        for phase in PHASES:
            phase_mask = phases.eq(phase).to_numpy()
            total_by_phase[phase] += int(phase_mask.sum())

            keep = phase_mask & static_complete
            static_complete_by_phase[phase] += int(keep.sum())
            if not keep.any():
                continue

            selected = chunk.loc[keep, reference_columns].copy()

            if "voyage_phase" in selected.columns:
                selected["voyage_phase"] = phase

            selected.to_csv(
                paths[phase],
                mode="w" if first_write[phase] else "a",
                header=first_write[phase],
                index=False,
                encoding=(
                    "utf-8-sig"
                    if first_write[phase]
                    else "utf-8"
                ),
            )
            first_write[phase] = False

        print(
            f"[提取阶段块 {block_number}] 累计读取 {source_rows:,} 行；"
            + "；".join(
                f"{phase}静态完整 {static_complete_by_phase[phase]:,}"
                for phase in PHASES
            )
        )

    # 即使某阶段为0行，也生成正确表头，随后明确报错。
    empty_header = pd.DataFrame(columns=reference_columns)
    for phase, path in paths.items():
        if not path.exists():
            empty_header.to_csv(
                path,
                index=False,
                encoding="utf-8-sig",
            )

    summary = pd.DataFrame([
        {
            "phase": phase,
            "source_phase_rows": total_by_phase[phase],
            "fixed27_static_complete_rows": static_complete_by_phase[phase],
            "removed_for_incomplete_static4": (
                total_by_phase[phase]
                - static_complete_by_phase[phase]
            ),
            "derived_fixed27_file": str(paths[phase]),
        }
        for phase in PHASES
    ])

    summary.to_csv(
        output_dir / "00_fixed27_phase_extraction_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    empty_phases = [
        phase
        for phase in PHASES
        if static_complete_by_phase[phase] == 0
    ]
    if empty_phases:
        raise RuntimeError(
            "以下阶段没有可用的 Fixed27 静态完整记录："
            + ", ".join(empty_phases)
        )

    return paths, summary


def normalized_key_expression(alias: str, column: str) -> str:
    identifier = f"{alias}.{quote_identifier(column)}"

    if column == "ship_type":
        return (
            f"lower(trim(coalesce(cast({identifier} AS VARCHAR), '<NULL>')))"
        )
    if column == "timestamp_utc":
        return (
            "coalesce("
            f"strftime(try_cast({identifier} AS TIMESTAMP), "
            "'%Y-%m-%d %H:%M:%S.%f'), "
            f"trim(cast({identifier} AS VARCHAR)), '<NULL>')"
        )
    return (
        f"trim(coalesce(cast({identifier} AS VARCHAR), '<NULL>'))"
    )



def comparison_report_pandas(
    left_path: Path,
    right_path: Path,
    report_path: Path,
    keys: Sequence[str],
    absolute_tolerance: float,
    relative_tolerance: float,
    label: str,
) -> Dict[str, Any]:
    """DuckDB不可用时的内存版一致性比较。"""
    print(
        "[warning] 当前Python未安装duckdb，"
        "巡航一致性检查将使用pandas整表读取。"
    )

    left = pd.read_csv(left_path, low_memory=False, encoding="utf-8-sig")
    right = pd.read_csv(right_path, low_memory=False, encoding="utf-8-sig")

    require_columns(left.columns.astype(str).tolist(), keys, left_path)
    require_columns(right.columns.astype(str).tolist(), keys, right_path)

    missing_reference_columns = [
        column for column in right.columns if column not in left.columns
    ]
    if missing_reference_columns:
        raise KeyError(
            f"{left_path} 缺少参考文件字段："
            + ", ".join(missing_reference_columns)
        )

    def normalize_key_frame(frame: pd.DataFrame) -> pd.DataFrame:
        output = pd.DataFrame(index=frame.index)
        for column in keys:
            values = frame[column].astype("string").str.strip()
            if column == "ship_type":
                values = values.str.lower()
            elif column == "timestamp_utc":
                parsed = pd.to_datetime(values, errors="coerce")
                formatted = parsed.dt.strftime("%Y-%m-%d %H:%M:%S.%f")
                values = formatted.fillna(values)
            output[column] = values.fillna("<NULL>")
        return output

    left_keys = normalize_key_frame(left)
    right_keys = normalize_key_frame(right)
    left["__compare_key"] = left_keys.astype(str).agg(chr(31).join, axis=1)
    right["__compare_key"] = right_keys.astype(str).agg(chr(31).join, axis=1)

    left_duplicates = int(left["__compare_key"].duplicated().sum())
    right_duplicates = int(right["__compare_key"].duplicated().sum())

    left_key_set = set(left["__compare_key"])
    right_key_set = set(right["__compare_key"])
    left_only = len(left_key_set - right_key_set)
    right_only = len(right_key_set - left_key_set)

    value_mismatch_rows = 0
    if (
        left_duplicates == 0
        and right_duplicates == 0
        and left_only == 0
        and right_only == 0
    ):
        left_indexed = left.set_index("__compare_key")
        right_indexed = right.set_index("__compare_key")
        left_indexed = left_indexed.loc[right_indexed.index]
        row_mismatch = np.zeros(len(right_indexed), dtype=bool)

        for column in right.columns:
            if column == "__compare_key" or column in keys:
                continue

            left_series = left_indexed[column]
            right_series = right_indexed[column]
            numeric = (
                pd.api.types.is_numeric_dtype(left_series)
                or pd.api.types.is_numeric_dtype(right_series)
            )

            if numeric:
                x = pd.to_numeric(left_series, errors="coerce").to_numpy(float)
                y = pd.to_numeric(right_series, errors="coerce").to_numpy(float)
                both_nan = np.isnan(x) & np.isnan(y)
                one_nan = np.isnan(x) ^ np.isnan(y)
                finite = np.isfinite(x) & np.isfinite(y)
                mismatch = one_nan.copy()
                mismatch[finite] |= ~np.isclose(
                    x[finite],
                    y[finite],
                    rtol=relative_tolerance,
                    atol=absolute_tolerance,
                    equal_nan=True,
                )
                nonfinite_same = (~finite) & (~one_nan) & (~both_nan) & (x == y)
                mismatch[(~finite) & (~one_nan) & (~both_nan)] = ~nonfinite_same[
                    (~finite) & (~one_nan) & (~both_nan)
                ]
            else:
                x = left_series.astype("string").fillna("<NULL>")
                y = right_series.astype("string").fillna("<NULL>")
                mismatch = x.ne(y).to_numpy()

            row_mismatch |= mismatch

        value_mismatch_rows = int(row_mismatch.sum())

    passed = (
        len(left) == len(right)
        and left_duplicates == 0
        and right_duplicates == 0
        and left_only == 0
        and right_only == 0
        and value_mismatch_rows == 0
    )

    report = {
        "label": label,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "comparison_engine": "pandas_fallback",
        "left_file": str(left_path.resolve()),
        "right_reference_file": str(right_path.resolve()),
        "left_rows": int(len(left)),
        "right_rows": int(len(right)),
        "left_duplicate_key_rows": left_duplicates,
        "right_duplicate_key_rows": right_duplicates,
        "left_only_keys": left_only,
        "right_only_keys": right_only,
        "value_mismatch_rows": value_mismatch_rows,
        "absolute_tolerance": absolute_tolerance,
        "relative_tolerance": relative_tolerance,
        "reference_columns_compared": [
            column for column in right.columns if column != "__compare_key"
        ],
        "left_extra_columns_not_compared": [
            column
            for column in left.columns
            if column not in right.columns and column != "__compare_key"
        ],
        "passed": passed,
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if not passed:
        raise RuntimeError(
            f"{label} 一致性检查失败。请查看：{report_path}"
        )
    return report


def comparison_report(
    left_path: Path,
    right_path: Path,
    report_path: Path,
    keys: Sequence[str],
    absolute_tolerance: float,
    relative_tolerance: float,
    label: str,
) -> Dict[str, Any]:
    """
    使用 DuckDB 比较两个 CSV：
    - 行数；
    - 主键唯一性；
    - 主键集合；
    - 参考文件中的全部字段值。

    left 是新生成/推导文件，right 是既有参考文件。
    """
    if duckdb is None:
        return comparison_report_pandas(
            left_path,
            right_path,
            report_path,
            keys,
            absolute_tolerance,
            relative_tolerance,
            label,
        )

    left_header = (
        pd.read_csv(left_path, nrows=0)
        .columns.astype(str)
        .tolist()
    )
    right_header = (
        pd.read_csv(right_path, nrows=0)
        .columns.astype(str)
        .tolist()
    )

    require_columns(left_header, keys, left_path)
    require_columns(right_header, keys, right_path)

    missing_reference_columns = [
        column
        for column in right_header
        if column not in left_header
    ]
    if missing_reference_columns:
        raise KeyError(
            f"{left_path} 缺少参考文件字段："
            + ", ".join(missing_reference_columns)
        )

    con = duckdb.connect()
    try:
        con.execute(
            f"""
            CREATE VIEW left_data AS
            SELECT *
            FROM read_csv_auto(
                '{sql_path(left_path)}',
                header=true,
                sample_size=100000
            )
            """
        )
        con.execute(
            f"""
            CREATE VIEW right_data AS
            SELECT *
            FROM read_csv_auto(
                '{sql_path(right_path)}',
                header=true,
                sample_size=100000
            )
            """
        )

        left_rows = int(
            con.execute(
                "SELECT count(*) FROM left_data"
            ).fetchone()[0]
        )
        right_rows = int(
            con.execute(
                "SELECT count(*) FROM right_data"
            ).fetchone()[0]
        )

        left_key_parts = [
            normalized_key_expression("l", column)
            for column in keys
        ]
        right_key_parts = [
            normalized_key_expression("r", column)
            for column in keys
        ]
        left_key = (
            "concat_ws(chr(31), "
            + ", ".join(left_key_parts)
            + ")"
        )
        right_key = (
            "concat_ws(chr(31), "
            + ", ".join(right_key_parts)
            + ")"
        )

        left_duplicates = int(
            con.execute(
                f"""
                SELECT coalesce(sum(row_count - 1), 0)
                FROM (
                    SELECT {left_key} AS row_key, count(*) AS row_count
                    FROM left_data l
                    GROUP BY row_key
                    HAVING count(*) > 1
                )
                """
            ).fetchone()[0]
        )
        right_duplicates = int(
            con.execute(
                f"""
                SELECT coalesce(sum(row_count - 1), 0)
                FROM (
                    SELECT {right_key} AS row_key, count(*) AS row_count
                    FROM right_data r
                    GROUP BY row_key
                    HAVING count(*) > 1
                )
                """
            ).fetchone()[0]
        )

        left_only = int(
            con.execute(
                f"""
                WITH
                lk AS (
                    SELECT DISTINCT {left_key} AS row_key
                    FROM left_data l
                ),
                rk AS (
                    SELECT DISTINCT {right_key} AS row_key
                    FROM right_data r
                )
                SELECT count(*)
                FROM lk
                LEFT JOIN rk USING(row_key)
                WHERE rk.row_key IS NULL
                """
            ).fetchone()[0]
        )
        right_only = int(
            con.execute(
                f"""
                WITH
                lk AS (
                    SELECT DISTINCT {left_key} AS row_key
                    FROM left_data l
                ),
                rk AS (
                    SELECT DISTINCT {right_key} AS row_key
                    FROM right_data r
                )
                SELECT count(*)
                FROM rk
                LEFT JOIN lk USING(row_key)
                WHERE lk.row_key IS NULL
                """
            ).fetchone()[0]
        )

        left_types = {
            row[0]: str(row[1]).upper()
            for row in con.execute(
                "DESCRIBE SELECT * FROM left_data"
            ).fetchall()
        }
        right_types = {
            row[0]: str(row[1]).upper()
            for row in con.execute(
                "DESCRIBE SELECT * FROM right_data"
            ).fetchall()
        }

        numeric_tokens = (
            "TINYINT",
            "SMALLINT",
            "INTEGER",
            "BIGINT",
            "HUGEINT",
            "UTINYINT",
            "USMALLINT",
            "UINTEGER",
            "UBIGINT",
            "FLOAT",
            "DOUBLE",
            "DECIMAL",
            "REAL",
        )

        mismatch_terms: List[str] = []
        compared_columns = [
            column
            for column in right_header
            if column not in keys
        ]

        for column in compared_columns:
            left_id = f"l.{quote_identifier(column)}"
            right_id = f"r.{quote_identifier(column)}"
            left_type = left_types.get(column, "")
            right_type = right_types.get(column, "")
            numeric = (
                any(token in left_type for token in numeric_tokens)
                or any(token in right_type for token in numeric_tokens)
            )

            if numeric:
                x = f"try_cast({left_id} AS DOUBLE)"
                y = f"try_cast({right_id} AS DOUBLE)"
                mismatch_terms.append(
                    f"""
                    (
                        CASE
                            WHEN {x} IS NULL AND {y} IS NULL THEN false
                            WHEN {x} IS NULL OR {y} IS NULL THEN true
                            WHEN isnan({x}) AND isnan({y}) THEN false
                            WHEN isinf({x}) OR isinf({y}) THEN {x} <> {y}
                            ELSE abs({x} - {y}) >
                                {float(absolute_tolerance)}
                                + {float(relative_tolerance)}
                                * greatest(abs({x}), abs({y}), 1.0)
                        END
                    )
                    """
                )
            else:
                mismatch_terms.append(
                    f"""
                    (
                        cast({left_id} AS VARCHAR)
                        IS DISTINCT FROM
                        cast({right_id} AS VARCHAR)
                    )
                    """
                )

        join_condition = " AND ".join(
            f"{left_part} = {right_part}"
            for left_part, right_part in zip(
                left_key_parts,
                right_key_parts,
            )
        )

        if mismatch_terms and left_only == 0 and right_only == 0:
            mismatch_condition = " OR ".join(mismatch_terms)
            value_mismatch_rows = int(
                con.execute(
                    f"""
                    SELECT count(*)
                    FROM left_data l
                    JOIN right_data r
                      ON {join_condition}
                    WHERE {mismatch_condition}
                    """
                ).fetchone()[0]
            )
        else:
            value_mismatch_rows = 0

        passed = (
            left_rows == right_rows
            and left_duplicates == 0
            and right_duplicates == 0
            and left_only == 0
            and right_only == 0
            and value_mismatch_rows == 0
        )

        report = {
            "label": label,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "left_file": str(left_path.resolve()),
            "right_reference_file": str(right_path.resolve()),
            "left_rows": left_rows,
            "right_rows": right_rows,
            "left_duplicate_key_rows": left_duplicates,
            "right_duplicate_key_rows": right_duplicates,
            "left_only_keys": left_only,
            "right_only_keys": right_only,
            "value_mismatch_rows": value_mismatch_rows,
            "absolute_tolerance": absolute_tolerance,
            "relative_tolerance": relative_tolerance,
            "reference_columns_compared": right_header,
            "left_extra_columns_not_compared": [
                column
                for column in left_header
                if column not in right_header
            ],
            "passed": passed,
        }
    finally:
        con.close()

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    if not report["passed"]:
        raise RuntimeError(
            f"{label} 一致性检查失败。请查看：{report_path}"
        )

    return report


def run_exact_script(
    module: ModuleType,
    input_file: Path,
    output_file: Path,
    audit_dir: Path,
    stage_name: str,
) -> None:
    reset_target(output_file)
    reset_target(audit_dir)

    module.INPUT_FILE = input_file
    module.OUTPUT_FILE = output_file
    module.AUDIT_DIR = audit_dir

    # 物理审计脚本必须保持“只审计、不使用经验阈值额外删行”。
    if hasattr(module, "APPLY_DRAUGHT_REVIEW_FILTER"):
        module.APPLY_DRAUGHT_REVIEW_FILTER = False
    if hasattr(module, "APPLY_LOW_FUEL_REVIEW_FILTER"):
        module.APPLY_LOW_FUEL_REVIEW_FILTER = False

    print("\n" + "=" * 88)
    print(f"{stage_name}")
    print(f"输入：{input_file}")
    print(f"输出：{output_file}")
    print("=" * 88)

    module.main()

    if not output_file.is_file():
        raise RuntimeError(
            f"{stage_name} 未生成输出：{output_file}"
        )


def phase_stage_paths(
    output_root: Path,
    phase: str,
) -> Dict[str, Path]:
    phase_root = output_root / phase
    return {
        "root": phase_root,
        "fixed28": (
            phase_root
            / "01_fixed28"
            / f"final_fixed28_{phase}_quality_checked.csv"
        ),
        "fixed28_audit": (
            phase_root
            / "01_fixed28"
            / "audit"
        ),
        "fixed29": (
            phase_root
            / "02_fixed29"
            / f"final_fixed29_{phase}_model_ready.csv"
        ),
        "fixed29_audit": (
            phase_root
            / "02_fixed29"
            / "audit"
        ),
        "fixed30": (
            phase_root
            / "03_fixed30_physical_audit"
            / f"final_fixed30_{phase}_physical_review.csv"
        ),
        "fixed30_audit": (
            phase_root
            / "03_fixed30_physical_audit"
            / "audit"
        ),
        "fixed31": (
            phase_root
            / "04_fixed31"
            / f"final_fixed31_{phase}_model_ready.csv"
        ),
        "fixed31_audit": (
            phase_root
            / "04_fixed31"
            / "audit"
        ),
    }


def append_phase_output(
    input_file: Path,
    output_file: Path,
    phase: str,
    first: bool,
    chunksize: int,
    expected_columns: List[str] | None,
) -> List[str]:
    current_columns = (
        pd.read_csv(input_file, nrows=0)
        .columns.astype(str)
        .tolist()
    )

    if "voyage_phase" not in current_columns:
        current_columns.append("voyage_phase")

    if expected_columns is not None:
        if current_columns != expected_columns:
            raise RuntimeError(
                f"{input_file} 的最终列结构与其他阶段不一致。"
            )

    for chunk in pd.read_csv(
        input_file,
        chunksize=chunksize,
        low_memory=False,
        encoding="utf-8-sig",
    ):
        if "voyage_phase" in chunk.columns:
            existing = normalized_phase(chunk["voyage_phase"])
            invalid = ~existing.eq(phase)
            if invalid.any():
                examples = existing.loc[invalid].drop_duplicates().tolist()
                raise RuntimeError(
                    f"{input_file} 中存在不属于 {phase} 的 voyage_phase："
                    f"{examples[:10]}"
                )
            chunk["voyage_phase"] = phase
        else:
            chunk["voyage_phase"] = phase

        chunk = chunk[current_columns]
        chunk.to_csv(
            output_file,
            mode="w" if first else "a",
            header=first,
            index=False,
            encoding="utf-8-sig" if first else "utf-8",
        )
        first = False

    return current_columns


def main() -> None:
    configure_console()
    args = parse_args()

    script_paths = {
        name: args.script_dir / filename
        for name, filename in SCRIPT_FILES.items()
    }

    required_files = {
        "all_phases": args.all_phases,
        "cruise_fixed27": args.cruise_fixed27,
        **{
            f"script_{name}": path
            for name, path in script_paths.items()
        },
    }
    if not args.skip_final_reference_check:
        required_files[
            "cruise_fixed31_reference"
        ] = args.cruise_fixed31_reference

    require_files(required_files)

    if args.output_root.exists():
        if not args.force:
            raise FileExistsError(
                f"输出目录已存在：{args.output_root}\n"
                "确认需要重建时添加 --force。"
            )
        shutil.rmtree(args.output_root)

    args.output_root.mkdir(parents=True, exist_ok=True)

    modules = {
        name: load_module(
            path,
            f"_equal_phase_{name}",
        )
        for name, path in script_paths.items()
    }

    manifest: Dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "all_phases_input": str(args.all_phases.resolve()),
        "cruise_fixed27_reference": str(
            args.cruise_fixed27.resolve()
        ),
        "cruise_fixed31_reference": (
            str(args.cruise_fixed31_reference.resolve())
            if not args.skip_final_reference_check
            else None
        ),
        "output_root": str(args.output_root.resolve()),
        "phase_processing_mode": (
            "each_phase_processed_independently"
        ),
        "reason_for_independent_processing": (
            "Fixed31 low-fuel thresholds depend on within-ship "
            "and within-speed-bin distributions."
        ),
        "scripts": {
            name: {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
            }
            for name, path in script_paths.items()
        },
    }

    fixed27_dir = args.output_root / "00_fixed27_phase_inputs"
    derived_inputs, extraction_summary = (
        prepare_fixed27_phase_inputs(
            args.all_phases,
            args.cruise_fixed27,
            fixed27_dir,
            args.chunksize,
        )
    )

    # 上游巡航一致性：从 all_phases 推导出的巡航 Fixed27 必须与既有巡航一致。
    fixed27_comparison = comparison_report(
        derived_inputs["cruise"],
        args.cruise_fixed27,
        args.output_root
        / "90_consistency"
        / "01_cruise_fixed27_consistency.json",
        KEY_COLUMNS,
        args.comparison_absolute_tolerance,
        args.comparison_relative_tolerance,
        "cruise Fixed27 upstream",
    )

    # 正式运行时，巡航直接使用既有 Fixed27；其他阶段使用同结构推导文件。
    phase_inputs = {
        "cruise": args.cruise_fixed27,
        "maneuver": derived_inputs["maneuver"],
        "anchor_berth": derived_inputs["anchor_berth"],
    }

    stage_summary_rows: List[Dict[str, Any]] = []
    final_phase_paths: Dict[str, Path] = {}

    for phase in PHASES:
        phase_paths = phase_stage_paths(
            args.output_root,
            phase,
        )

        fixed27_rows = csv_row_count(
            phase_inputs[phase],
            args.chunksize,
        )

        run_exact_script(
            modules["fixed28"],
            phase_inputs[phase],
            phase_paths["fixed28"],
            phase_paths["fixed28_audit"],
            f"[{phase}] Fixed28完全相同质量清洗",
        )
        fixed28_rows = csv_row_count(
            phase_paths["fixed28"],
            args.chunksize,
        )

        run_exact_script(
            modules["fixed29"],
            phase_paths["fixed28"],
            phase_paths["fixed29"],
            phase_paths["fixed29_audit"],
            f"[{phase}] Fixed29完全相同模型准备清洗",
        )
        fixed29_rows = csv_row_count(
            phase_paths["fixed29"],
            args.chunksize,
        )

        run_exact_script(
            modules["fixed30_audit"],
            phase_paths["fixed29"],
            phase_paths["fixed30"],
            phase_paths["fixed30_audit"],
            f"[{phase}] Fixed30完全相同物理范围审计",
        )
        fixed30_rows = csv_row_count(
            phase_paths["fixed30"],
            args.chunksize,
        )

        run_exact_script(
            modules["fixed31"],
            phase_paths["fixed29"],
            phase_paths["fixed31"],
            phase_paths["fixed31_audit"],
            f"[{phase}] Fixed31完全相同最终清洗",
        )
        fixed31_rows = csv_row_count(
            phase_paths["fixed31"],
            args.chunksize,
        )

        final_phase_paths[phase] = phase_paths["fixed31"]

        stage_summary_rows.append({
            "phase": phase,
            "fixed27_input_rows": fixed27_rows,
            "fixed28_rows": fixed28_rows,
            "fixed28_removed_rows": fixed27_rows - fixed28_rows,
            "fixed29_rows": fixed29_rows,
            "fixed29_removed_rows": fixed28_rows - fixed29_rows,
            "fixed30_audit_candidate_rows": fixed30_rows,
            "fixed31_rows": fixed31_rows,
            "fixed31_removed_from_fixed29": fixed29_rows - fixed31_rows,
            "fixed31_output": str(
                phase_paths["fixed31"].resolve()
            ),
        })

    # 最终巡航必须与既有 final_fixed31_model_ready.csv 完全同样。
    final_cruise_comparison: Dict[str, Any] | None = None
    if not args.skip_final_reference_check:
        final_cruise_comparison = comparison_report(
            final_phase_paths["cruise"],
            args.cruise_fixed31_reference,
            args.output_root
            / "90_consistency"
            / "02_cruise_fixed31_consistency.json",
            KEY_COLUMNS,
            args.comparison_absolute_tolerance,
            args.comparison_relative_tolerance,
            "cruise Fixed31 final",
        )

    # 合并三个最终阶段。各阶段已独立清洗，合并时不再计算任何阈值或删除记录。
    combined_dir = args.output_root / "05_combined"
    combined_dir.mkdir(parents=True, exist_ok=True)
    combined_path = (
        combined_dir
        / "final_fixed31_all_phases_model_ready.csv"
    )
    combined_path.unlink(missing_ok=True)

    expected_columns: List[str] | None = None
    first = True
    for phase in PHASES:
        expected_columns = append_phase_output(
            final_phase_paths[phase],
            combined_path,
            phase,
            first,
            args.chunksize,
            expected_columns,
        )
        first = False

    summary = pd.DataFrame(stage_summary_rows)
    summary.to_csv(
        combined_dir / "00_phase_cleaning_row_counts.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 方便直接使用的三份最终文件。
    for phase in PHASES:
        convenience_path = (
            combined_dir
            / f"final_fixed31_{phase}.csv"
        )
        shutil.copy2(
            final_phase_paths[phase],
            convenience_path,
        )

    manifest.update({
        "fixed27_extraction": extraction_summary.to_dict(
            orient="records"
        ),
        "fixed27_cruise_consistency": fixed27_comparison,
        "fixed31_cruise_consistency": final_cruise_comparison,
        "stage_row_counts": stage_summary_rows,
        "combined_output": str(combined_path.resolve()),
        "combined_rows": int(summary["fixed31_rows"].sum()),
        "cruise_consistency_guarantee": (
            "passed"
            if fixed27_comparison["passed"]
            and (
                args.skip_final_reference_check
                or (
                    final_cruise_comparison is not None
                    and final_cruise_comparison["passed"]
                )
            )
            else "failed"
        ),
    })

    (
        args.output_root / "run_manifest.json"
    ).write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 88)
    print("三阶段完全同规则清洗完成")
    print("=" * 88)
    print(summary.to_string(index=False))
    print(f"\n合并文件：{combined_path}")
    print(
        "巡航一致性："
        + manifest["cruise_consistency_guarantee"]
    )
    print(
        f"完整运行清单：{args.output_root / 'run_manifest.json'}"
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n用户中断运行。", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(
            f"\n运行失败：{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        raise
