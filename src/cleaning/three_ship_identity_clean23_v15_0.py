# -*- coding: utf-8 -*-
from __future__ import annotations

r"""
第二阶段分步低内存处理 V15.0（集装箱燃油/航程修复 + 四项船舶固有特征）。

推荐顺序：
init → stage-container → build-container → stage-bulk → build-bulk
→ stage-tanker → build-tanker → feature-container → feature-bulk
→ feature-tanker → merge → status

每个命令均为独立进程，步骤完成后操作系统回收内存。失败时只重跑当前步骤。
"""

import argparse
import gc
import hashlib
import json
import math
import shutil
import sys
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

try:
    import duckdb
except ImportError as exc:
    raise SystemExit(
        "缺少duckdb。请运行：python -m pip install duckdb"
    ) from exc


CODE_VERSION = "stage2-identity-10min-clean23-static4-v15.0-2026-07-23"
BASE_DIR = Path(r"data")

# V14从三个原始文件重新标准化和清洗，因此原始文件同时作为
# V10身份流水线的动态数据源和静态字段来源。
DEFAULT_CLEANED_CONTAINER = BASE_DIR / "continership_with_era5_7fields_v4.csv"
DEFAULT_CLEANED_BULK = BASE_DIR / "bulk.csv"
DEFAULT_CLEANED_TANKER = BASE_DIR / "tank.csv"
DEFAULT_RAW_CONTAINER = BASE_DIR / "continership_with_era5_7fields_v4.csv"
DEFAULT_RAW_BULK = BASE_DIR / "bulk.csv"
DEFAULT_RAW_TANKER = BASE_DIR / "tank.csv"
DEFAULT_OUTPUT_ROOT = Path(r"data")

STEP_ORDER = [
    "init",
    "stage-container",
    "build-container",
    "stage-bulk",
    "build-bulk",
    "stage-tanker",
    "build-tanker",
    "feature-container",
    "feature-bulk",
    "feature-tanker",
    "merge",
    "clean23",
]


def load_embedded_module(name: str, source: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__file__ = f"<embedded:{name}>"
    module.__package__ = ""
    module.__dict__["__builtins__"] = __builtins__
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


def quote_sql_text(value: str) -> str:
    return value.replace("'", "''")


def sql_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").replace("'", "''")


def safe_mtime(path: Path) -> tuple[str, str]:
    try:
        value = float(path.stat().st_mtime)
        if not math.isfinite(value):
            raise ValueError("non-finite mtime")
        return (
            datetime.fromtimestamp(value, tz=timezone.utc).isoformat(),
            "ok",
        )
    except (OSError, OverflowError, TypeError, ValueError):
        return "", "unavailable_invalid_filesystem_timestamp"


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def file_metadata(role: str, path: Path) -> Dict[str, Any]:
    modified_time_utc, modified_time_status = safe_mtime(path)
    stat = path.stat()
    return {
        "role": role,
        "path": str(path.resolve()),
        "filename": path.name,
        "size_bytes": int(stat.st_size),
        "modified_time_utc": modified_time_utc,
        "modified_time_status": modified_time_status,
        "sha256": sha256_file(path),
    }


def paths(args: argparse.Namespace) -> Dict[str, Path]:
    root = args.output_root
    return {
        "root": root,
        "meta": root / "00_meta",
        "stage": root / "01_staging",
        "identity": root / "02_identity",
        "features": root / "03_features",
        "unified": root / "04_unified",
        "clean23": root / "05_clean23",
        "state": root / "_state",
        "work": root / "_work",
        "container_stage": root / "01_staging" / "container_stage.csv",
        "bulk_stage": root / "01_staging" / "bulk_stage.csv",
        "tanker_stage": root / "01_staging" / "tanker_stage.csv",
        "container_identity":
            root / "02_identity" / "container_model_10min.csv",
        "bulk_identity":
            root / "02_identity" / "bulk_with_pseudo_ship_id.csv",
        "tanker_identity":
            root / "02_identity" / "tanker_with_pseudo_ship_id.csv",
        "bulk_identity_model":
            root / "02_identity" / "bulk_model_10min_identity.csv",
        "tanker_identity_model":
            root / "02_identity" / "tanker_model_10min_identity.csv",
        "container_features":
            root / "03_features" / "container_model_10min_features.csv",
        "bulk_features":
            root / "03_features" / "bulk_model_10min_features.csv",
        "tanker_features":
            root / "03_features" / "tanker_model_10min_features.csv",
        "unified_features":
            root / "04_unified" / "unified_ship_10min_features.csv",
        "clean23_cruise":
            root / "05_clean23" / "final_fixed23_cruise.csv",
    }


def input_paths(args: argparse.Namespace) -> Dict[str, Path]:
    return {
        "cleaned_container": args.cleaned_container,
        "cleaned_bulk": args.cleaned_bulk,
        "cleaned_tanker": args.cleaned_tanker,
        "raw_container": args.raw_container,
        "raw_bulk": args.raw_bulk,
        "raw_tanker": args.raw_tanker,
    }


def ensure_root_dirs(args: argparse.Namespace) -> None:
    p = paths(args)
    for key in ("meta", "stage", "identity", "features",
                "unified", "clean23", "state", "work"):
        p[key].mkdir(parents=True, exist_ok=True)


def require_files(mapping: Dict[str, Path]) -> None:
    missing = [
        f"{name}: {path}"
        for name, path in mapping.items()
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "以下文件不存在：\n" + "\n".join(missing)
        )


def state_path(args: argparse.Namespace, step: str) -> Path:
    return paths(args)["state"] / f"{step}.done.json"


def write_state(
    args: argparse.Namespace,
    step: str,
    outputs: Iterable[Path],
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    output_rows = []
    for path in outputs:
        if path.exists():
            output_rows.append({
                "path": str(path.resolve()),
                "size_bytes": (
                    int(path.stat().st_size) if path.is_file() else None
                ),
            })
    payload = {
        "step": step,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "code_version": CODE_VERSION,
        "outputs": output_rows,
        "extra": extra or {},
    }
    target = state_path(args, step)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def skip_or_prepare(
    args: argparse.Namespace,
    step: str,
    outputs: Iterable[Path],
) -> bool:
    outputs = list(outputs)
    if not args.force and state_path(args, step).is_file():
        if all(path.exists() for path in outputs):
            print(f"[skip] {step}已完成。使用--force可重跑。")
            return True

    if args.force:
        for path in outputs:
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
        state_path(args, step).unlink(missing_ok=True)
    return False


def configure_duckdb(
    con: "duckdb.DuckDBPyConnection",
    args: argparse.Namespace,
    temp_dir: Path,
) -> None:
    temp_dir.mkdir(parents=True, exist_ok=True)
    con.execute(
        f"SET memory_limit='{quote_sql_text(args.memory_limit)}'"
    )
    con.execute(f"SET threads={max(1, int(args.threads))}")
    con.execute(f"SET temp_directory='{sql_path(temp_dir)}'")
    con.execute("SET preserve_insertion_order=false")


def run_init(args: argparse.Namespace, v6: types.ModuleType) -> None:
    p = paths(args)

    if args.reset and p["root"].exists():
        shutil.rmtree(p["root"])
    ensure_root_dirs(args)
    require_files(input_paths(args))

    outputs = [
        p["meta"] / "first_stage_input_provenance.csv",
        p["meta"] / "00_field_mapping.csv",
        p["meta"] / "run_config.json",
    ]
    if skip_or_prepare(args, "init", outputs):
        return

    pd.DataFrame([
        file_metadata(role, path)
        for role, path in input_paths(args).items()
    ]).to_csv(outputs[0], index=False, encoding="utf-8-sig")

    mapping_frames = [
        v6.field_mapping_rows(
            args.cleaned_container, v6.ALIASES,
            v6.STANDARD_CLEAN_COLUMNS,
            "cleaned_container", "container",
        ),
        v6.field_mapping_rows(
            args.cleaned_bulk, v6.ALIASES,
            v6.STANDARD_CLEAN_COLUMNS,
            "cleaned_bulk", "bulk",
        ),
        v6.field_mapping_rows(
            args.cleaned_tanker, v6.ALIASES,
            v6.STANDARD_CLEAN_COLUMNS,
            "cleaned_tanker", "tanker",
        ),
        v6.field_mapping_rows(
            args.raw_container, v6.CONTAINER_RAW_ID_ALIASES,
            list(v6.CONTAINER_RAW_ID_ALIASES.keys()),
            "raw_container", "container",
        ),
        v6.field_mapping_rows(
            args.raw_bulk, v6.RAW_STATIC_ALIASES,
            list(v6.RAW_STATIC_ALIASES.keys()),
            "raw_bulk", "bulk",
        ),
        v6.field_mapping_rows(
            args.raw_tanker, v6.RAW_STATIC_ALIASES,
            list(v6.RAW_STATIC_ALIASES.keys()),
            "raw_tanker", "tanker",
        ),
    ]
    pd.concat(mapping_frames, ignore_index=True).to_csv(
        outputs[1], index=False, encoding="utf-8-sig"
    )

    config = {
        "code_version": CODE_VERSION,
        "output_root": str(args.output_root.resolve()),
        "inputs": {
            key: str(value.resolve())
            for key, value in input_paths(args).items()
        },
        "settings": {
            "chunksize": args.chunksize,
            "memory_limit": args.memory_limit,
            "threads": args.threads,
            "segment_gap_hours": args.segment_gap_hours,
            "jump_speed_kn": args.jump_speed_kn,
            "expected_container_ships": args.expected_container_ships,
            "min_static_match_rate": args.min_static_match_rate,
            "negative_fuel_policy": args.negative_fuel_policy,
            "distance_interval_alignment":
                args.distance_interval_alignment,
            "wind_direction_convention":
                args.wind_direction_convention,
            "wave_direction_convention":
                args.wave_direction_convention,
            "reference_direction": args.reference_direction,
        },
    }
    outputs[2].write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_state(args, "init", outputs)
    print(f"[done] 初始化完成：{p['root']}")


def run_stage_container(
    args: argparse.Namespace,
    v6: types.ModuleType,
) -> None:
    p = paths(args)
    ensure_root_dirs(args)
    require_files({
        "cleaned_container": args.cleaned_container,
        "raw_container": args.raw_container,
    })
    validation = p["meta"] / "container_first_stage_validation.csv"
    outputs = [p["container_stage"], validation]
    if skip_or_prepare(args, "stage-container", outputs):
        return

    frame = v6.stage_container(
        args.cleaned_container,
        args.raw_container,
        p["container_stage"],
        args.chunksize,
    )
    frame.to_csv(validation, index=False, encoding="utf-8-sig")
    write_state(args, "stage-container", outputs)
    print(f"[done] Container暂存：{p['container_stage']}")


def run_stage_bulk_tanker(
    args: argparse.Namespace,
    v6: types.ModuleType,
    ship_type: str,
) -> None:
    p = paths(args)
    ensure_root_dirs(args)
    cleaned = getattr(args, f"cleaned_{ship_type}")
    raw = getattr(args, f"raw_{ship_type}")
    stage = p[f"{ship_type}_stage"]
    validation = p["meta"] / f"{ship_type}_first_stage_validation.csv"
    alignment_report = (
        p["root"] / f"{ship_type}_static_alignment_report.csv"
    )
    outputs = [stage, validation, alignment_report]
    require_files({
        f"cleaned_{ship_type}": cleaned,
        f"raw_{ship_type}": raw,
    })
    if skip_or_prepare(args, f"stage-{ship_type}", outputs):
        return

    frame = v6.stage_bulk_or_tanker(
        cleaned,
        raw,
        stage,
        ship_type,
        args.chunksize,
        args.memory_limit,
        args.threads,
        args.min_static_match_rate,
    )
    frame.to_csv(validation, index=False, encoding="utf-8-sig")

    if not args.keep_work:
        shutil.rmtree(
            p["stage"] / f"_{ship_type}_alignment",
            ignore_errors=True,
        )

    write_state(args, f"stage-{ship_type}", outputs)
    print(f"[done] {ship_type}键控对齐暂存：{stage}")


def open_step_db(
    args: argparse.Namespace,
    name: str,
) -> tuple["duckdb.DuckDBPyConnection", Path]:
    p = paths(args)
    work = p["work"] / name
    work.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(work / f"{name}.duckdb"))
    configure_duckdb(con, args, work / "duckdb_temp")
    return con, work


def close_step_db(
    con: "duckdb.DuckDBPyConnection",
    work_dir: Path,
    keep_work: bool,
) -> None:
    con.close()
    gc.collect()
    if not keep_work:
        shutil.rmtree(work_dir, ignore_errors=True)


def run_build_container(
    args: argparse.Namespace,
    v6: types.ModuleType,
) -> None:
    p = paths(args)
    ensure_root_dirs(args)
    require_files({"container_stage": p["container_stage"]})
    identity_summary = p["identity"] / "container_identity_summary.csv"
    missing_report = p["identity"] / "container_missing_intervals.csv"
    outputs = [p["container_identity"], identity_summary, missing_report]
    if skip_or_prepare(args, "build-container", outputs):
        return

    con, work = open_step_db(args, "build_container")
    try:
        v6.create_container_table(
            con,
            p["container_stage"],
            args.segment_gap_hours,
            args.jump_speed_kn,
            args.hard_max_fuel_t_5min,
            args.max_sfoc_g_kwh,
            args.fuel_power_margin,
            args.expected_container_ships,
        )
        v6.copy_query_to_csv(
            con,
            """
            SELECT * FROM container_model_10min
            ORDER BY pseudo_ship_group_id, timestamp_utc
            """,
            p["container_identity"],
        )
        v6.copy_query_to_csv(
            con,
            """
            SELECT * FROM container_identity_summary
            ORDER BY delivery_date
            """,
            identity_summary,
        )
        v6.build_container_missing_report(con, missing_report)
    finally:
        close_step_db(con, work, args.keep_work)

    write_state(args, "build-container", outputs)
    print(f"[done] Container身份与10分钟表：{p['container_identity']}")


def run_build_bulk_tanker(
    args: argparse.Namespace,
    v6: types.ModuleType,
    ship_type: str,
) -> None:
    p = paths(args)
    ensure_root_dirs(args)
    stage = p[f"{ship_type}_stage"]
    identified_output = p[f"{ship_type}_identity"]
    model_output = p[f"{ship_type}_identity_model"]
    outputs = [identified_output, model_output]
    require_files({f"{ship_type}_stage": stage})
    if skip_or_prepare(args, f"build-{ship_type}", outputs):
        return

    con, work = open_step_db(args, f"build_{ship_type}")
    assignment_file = p["work"] / f"{ship_type}_track_assignments.csv"
    try:
        v6.create_bulk_tanker_tables(
            con,
            ship_type,
            stage,
            assignment_file,
            args.segment_gap_hours,
            args.jump_speed_kn,
            args.duplicate_radius_nm,
            args.recent_track_hours,
            args.positionless_gap_hours,
            args.max_reconnect_days,
            args.ambiguity_ratio,
            args.hard_max_fuel_t_5min,
            args.max_sfoc_g_kwh,
            args.fuel_power_margin,
            args.negative_fuel_policy,
        )
        v6.copy_query_to_csv(
            con,
            f"""
            SELECT * FROM {ship_type}_identified
            ORDER BY pseudo_ship_group_id, timestamp_utc, source_row
            """,
            identified_output,
        )
        v6.copy_query_to_csv(
            con,
            f"""
            SELECT * FROM {ship_type}_model_10min
            ORDER BY pseudo_ship_group_id, timestamp_utc
            """,
            model_output,
        )
    finally:
        close_step_db(con, work, args.keep_work)
        if not args.keep_work:
            assignment_file.unlink(missing_ok=True)

    write_state(args, f"build-{ship_type}", outputs)
    print(f"[done] {ship_type}身份表：{identified_output}")


def feature_input(
    args: argparse.Namespace,
    ship_type: str,
) -> tuple[Path, int]:
    p = paths(args)
    if ship_type == "container":
        return p["container_identity"], 10
    return p[f"{ship_type}_identity"], 5


def run_feature_ship(
    args: argparse.Namespace,
    v7: types.ModuleType,
    ship_type: str,
) -> None:
    p = paths(args)
    ensure_root_dirs(args)
    input_path, cadence = feature_input(args, ship_type)
    output_path = p[f"{ship_type}_features"]
    audit_path = p["features"] / f"{ship_type}_aggregation_audit.csv"
    status_path = p["features"] / f"{ship_type}_aggregation_status.csv"
    mapping_path = p["features"] / f"{ship_type}_field_mapping.csv"
    outputs = [output_path, audit_path, status_path, mapping_path]
    require_files({f"{ship_type}_identity_input": input_path})
    if skip_or_prepare(args, f"feature-{ship_type}", outputs):
        return

    mapping, mapping_frame = v7.resolve_columns(
        v7.read_header(input_path), ship_type, input_path
    )
    v7.ensure_required(
        mapping,
        ["pseudo_ship_group_id", "timestamp_utc"],
        ship_type,
        input_path,
    )
    mapping_frame.to_csv(
        mapping_path, index=False, encoding="utf-8-sig"
    )

    con, work = open_step_db(args, f"feature_{ship_type}")
    try:
        raw_table = v7.create_raw_table(
            con, ship_type, input_path, mapping
        )
        if cadence == 5:
            base_table = v7.create_5min_aggregate(
                con,
                ship_type,
                raw_table,
                nominal_interval_minutes=
                    args.nominal_interval_minutes,
                max_interval_minutes=args.max_interval_minutes,
                minimum_coverage_minutes=
                    args.minimum_coverage_minutes,
                maximum_coverage_minutes=
                    args.maximum_coverage_minutes,
                distance_alignment=
                    args.distance_interval_alignment,
            )
        else:
            base_table = v7.create_10min_normalized(
                con, ship_type, raw_table
            )

        final_table = v7.create_features(
            con,
            ship_type,
            base_table,
            wind_direction_convention=
                args.wind_direction_convention,
            wave_direction_convention=
                args.wave_direction_convention,
            reference_direction=args.reference_direction,
        )
        v7.copy_query_to_csv(
            con,
            f"""
            SELECT * FROM {final_table}
            ORDER BY pseudo_ship_group_id,
                     trajectory_segment_id, timestamp_utc
            """,
            output_path,
        )
        pd.DataFrame([
            v7.build_audit(
                con, ship_type, raw_table, final_table, cadence
            )
        ]).to_csv(
            audit_path, index=False, encoding="utf-8-sig"
        )
        v7.copy_query_to_csv(
            con,
            f"""
            SELECT
                aggregation_status,
                count(*) AS window_count,
                sum(CASE WHEN fuel_t_10min IS NOT NULL
                    THEN 1 ELSE 0 END) AS fuel_nonmissing_count,
                sum(CASE WHEN distance_nm_10min IS NOT NULL
                    THEN 1 ELSE 0 END) AS distance_nonmissing_count,
                min(fuel_t_10min) AS fuel_min,
                avg(fuel_t_10min) AS fuel_mean,
                median(fuel_t_10min) AS fuel_median,
                max(fuel_t_10min) AS fuel_max
            FROM {final_table}
            GROUP BY aggregation_status
            ORDER BY window_count DESC
            """,
            status_path,
        )
    finally:
        close_step_db(con, work, args.keep_work)

    write_state(args, f"feature-{ship_type}", outputs)
    print(f"[done] {ship_type}10分钟特征：{output_path}")


def run_merge(args: argparse.Namespace) -> None:
    p = paths(args)
    ensure_root_dirs(args)
    inputs = {
        "container": p["container_features"],
        "bulk": p["bulk_features"],
        "tanker": p["tanker_features"],
    }
    require_files(inputs)

    audit_output = p["unified"] / "three_ship_aggregation_audit.csv"
    quality_output = p["unified"] / "three_ship_quality_summary.csv"
    outputs = [p["unified_features"], audit_output, quality_output]
    if skip_or_prepare(args, "merge", outputs):
        return

    con, work = open_step_db(args, "merge")
    try:
        for ship_type, path in inputs.items():
            con.execute(
                f"""
                CREATE OR REPLACE VIEW {ship_type}_features AS
                SELECT *
                FROM read_csv_auto(
                    '{sql_path(path)}',
                    header=true,
                    sample_size=100000
                )
                """
            )

        con.execute(
            """
            CREATE OR REPLACE VIEW unified_features AS
            SELECT * FROM container_features
            UNION ALL BY NAME
            SELECT * FROM bulk_features
            UNION ALL BY NAME
            SELECT * FROM tanker_features
            """
        )
        con.execute(
            f"""
            COPY (
                SELECT * FROM unified_features
                ORDER BY ship_type, pseudo_ship_group_id,
                         trajectory_segment_id, timestamp_utc
            )
            TO '{sql_path(p["unified_features"])}'
            (HEADER, DELIMITER ',', FORMAT CSV)
            """
        )
        con.execute(
            f"""
            COPY (
                SELECT
                    ship_type,
                    count(*) AS row_count,
                    count(DISTINCT pseudo_ship_group_id)
                        AS ship_group_count,
                    count(DISTINCT trajectory_segment_id)
                        AS segment_count,
                    min(timestamp_utc) AS minimum_timestamp,
                    max(timestamp_utc) AS maximum_timestamp,
                    sum(CASE WHEN fuel_t_10min IS NOT NULL
                        THEN 1 ELSE 0 END) AS fuel_nonmissing_rows,
                    sum(CASE WHEN distance_nm_10min IS NOT NULL
                        THEN 1 ELSE 0 END) AS distance_nonmissing_rows,
                    sum(CASE WHEN complete_window_flag = 1
                        THEN 1 ELSE 0 END) AS complete_windows
                FROM unified_features
                GROUP BY ship_type
                ORDER BY ship_type
            )
            TO '{sql_path(quality_output)}'
            (HEADER, DELIMITER ',', FORMAT CSV)
            """
        )
    finally:
        close_step_db(con, work, args.keep_work)

    audit_frames = []
    for ship_type in ("container", "bulk", "tanker"):
        path = p["features"] / f"{ship_type}_aggregation_audit.csv"
        if path.is_file():
            audit_frames.append(pd.read_csv(path))
    pd.concat(audit_frames, ignore_index=True).to_csv(
        audit_output, index=False, encoding="utf-8-sig"
    )

    write_state(args, "merge", outputs)
    print(f"[done] 三船型统一表：{p['unified_features']}")



RAW_DRAUGHT_ALIASES: Dict[str, Sequence[str]] = {
    "fore_draught_m": [
        "fore_draught_m", "fore_draft_m", "fore draft", "fore draught",
        "df", "draft_fore", "draught_fore",
        "draft_fore_m", "draught_fore_m", "fwd_draught_m",
        "draught_fwd_m", "bow_draught_m", "艏吃水", "首吃水",
        "修正后的艏吃水(m)", "修正后的艏吃水（m）",
        "修正后的艏吃水m", "修正后的艏吃水",
        "修正艏吃水(m)", "修正艏吃水（m）", "修正艏吃水",
    ],
    "aft_draught_m": [
        "aft_draught_m", "aft_draft_m", "aft draft", "aft draught",
        "da", "draft_aft", "draught_aft",
        "draft_aft_m", "draught_aft_m", "stern_draught_m",
        "艉吃水", "尾吃水",
        "修正后的艉吃水(m)", "修正后的艉吃水（m）",
        "修正后的艉吃水m", "修正后的艉吃水",
        "修正艉吃水(m)", "修正艉吃水（m）", "修正艉吃水",
    ],
    "ship_length_m": [
        "ship_length_m", "ship length m", "ship length (m)",
        "ship_length", "ship length", "vessel_length_m",
        "vessel length m", "vessel length (m)", "vessel length",
        "length_m", "length (m)", "length/m", "length / m",
        "length", "overall_length_m", "overall length m",
        "overall length (m)", "overall length", "length overall m",
        "length overall / m", "length overall (m)",
        "length_overall_m", "length overall",
        "loa_m", "loa (m)", "loa/m", "loa / m", "loa",
        "LengthBP", "length bp", "length bp / m", "length bp (m)",
        "length_bp_m", "length_bp_static", "LBP", "lbp_m",
        "lbp (m)", "lbp/m", "lbp / m",
        "length between perpendiculars", "length between perpendiculars m",
        "length between perpendiculars (m)",
        "船长", "船长(m)", "船长/m", "船长 / m",
        "总长", "总长(m)", "总长/m", "总长 / m",
        "垂线间长", "垂线间长(m)", "垂线间长/m", "垂线间长 / m",
    ],
}


# V15.0：集装箱船燃油从最原始文件重算，航程优先位置法并以航速积分回补。
# 同时按单船提取并固化四项静态特征：设计吃水、载重吨、服务航速、主机额定功率。

RAW_SHIP_STATIC_ALIASES: Dict[str, Sequence[str]] = {
    "design_draught_m": [
        "design_draught_m", "design draught m", "design draught",
        "design draft m", "design draft", "scantling draught",
        "scantling draft", "Draught", "Draft",
        "设计吃水 / M", "设计吃水/M", "设计吃水(m)",
        "设计吃水（m）", "设计吃水", "结构吃水", "型吃水",
    ],
    "deadweight_t": [
        "deadweight_t", "deadweight", "deadweight tonnage", "dwt",
        "dead weight", "deadweight(t)", "deadweight (t)",
        "deadweight / t", "载重吨", "载重吨(t)", "载重吨 / T",
        "载重量", "载重量(t)", "载重量 / T",
    ],
    "service_speed_kn": [
        "service_speed_kn", "service speed", "servicespeed",
        "service speed kn", "design speed", "design speed kn",
        "航速 / Kn", "航速/Kn", "服务航速", "服务航速(kn)",
        "设计航速", "设计航速(kn)",
    ],
    "main_engine_power_kw": [
        "main_engine_power_kw", "Total KW Main Eng",
        "Main Propulsion Total Power Output", "Main Engine Power",
        "Main Engine Power(kW)", "Main Engine Power (kW)",
        "M/E Power", "M/E Power(kW)", "ME Power",
        "Total Main Engine Power", "主机功率", "主机功率(kW)",
        "主机功率 / KW", "主机额定功率", "主机额定功率(kW)",
        "主机总功率", "主机总功率(kW)",
    ],
}

STATIC4_PREDICTORS = [
    "design_draught_m",
    "deadweight_t",
    "service_speed_kn",
    "main_engine_power_kw",
]

RAW_CONTAINER_FUEL_ALIASES: Dict[str, Sequence[str]] = {
    "fuel_rate_kg_h": [
        "fuel_rate_kg_h", "fuel rate kg/h", "fuel flow kg/h",
        "main engine fuel consumption mass flow rate kg/h",
        "主机燃油消耗质量流量(kg/h)",
        "主机燃油消耗质量流量（kg/h）",
        "主机燃油消耗质量流量kg/h",
        "主机燃油质量流量(kg/h)",
    ],
    "shaft_power_kw": [
        "shaft_power_kw", "shaft power kw", "shaft power",
        "轴功率(kW)", "轴功率（kW）", "轴功率kw", "轴功率",
    ],
    "sfoc_g_kwh_existing": [
        "sfoc_g_kwh", "SFOC(g/kWh)", "SFOC（g/kWh）", "SFOC",
        "主机SFOC(g/kWh)", "主机SFOC（g/kWh）", "主机SFOC",
    ],
    "main_engine_rpm": [
        "main_engine_rpm", "main engine rpm", "rpm",
        "主机转速(rpm)", "主机转速（rpm）", "主机转速",
    ],
}

FIXED23_PREDICTORS = [
    "speed_kn",
    "course_sin",
    "course_cos",
    "heading_sin",
    "heading_cos",
    "mean_draught_m",
    "rudder_deg",
    "distance_nm_10min",
    "rel_wind_speed_kn",
    "relative_wind_sin",
    "relative_wind_cos",
    "wave_height_m",
    "wave_period_s",
    "relative_wave_sin",
    "relative_wave_cos",
    "surface_pressure_pa",
    "surface_temperature_c",
    "rel_wind_speed_x_speed",
    "wave_height_x_speed",
    "draught_x_speed",
    "trim_m",
    "trim_x_speed",
    "draught_x_trim",
]


def quote_identifier(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def choose_fuzzy_ship_length_column(
    header: Sequence[str],
    v6: types.ModuleType,
) -> tuple[Optional[str], List[Dict[str, Any]]]:
    """在精确别名失败后，按LBP→LOA→一般船长的优先级保守识别。"""
    exclusions = (
        "breadth", "beam", "width", "型宽", "draft", "draught",
        "吃水", "height", "高度", "ratio", "比例", "area", "面积",
        "wave", "period", "distance", "距离", "longitude", "经度",
    )

    candidates: List[Dict[str, Any]] = []
    for column in header:
        normalized = v6.normalize_name(column)
        if any(token in normalized for token in exclusions):
            continue

        score = 0
        category = ""
        if any(token in normalized for token in (
            "lengthbetweenperpendiculars", "lengthbp", "lbp",
            "垂线间长",
        )):
            score, category = 50, "length_between_perpendiculars"
        elif any(token in normalized for token in (
            "lengthoverall", "overalllength", "loa", "总长",
        )):
            score, category = 40, "length_overall"
        elif any(token in normalized for token in (
            "shiplength", "vessellength", "船长",
        )):
            score, category = 30, "explicit_ship_length"
        elif normalized.startswith("length") or normalized.endswith("length"):
            score, category = 20, "generic_length"
        elif "length" in normalized or "长度" in normalized:
            score, category = 10, "weak_length_name"

        if score <= 0:
            continue
        if normalized.endswith("m") or "meter" in normalized or "metre" in normalized:
            score += 2
        candidates.append({
            "column": str(column),
            "normalized": normalized,
            "score": score,
            "category": category,
        })

    candidates.sort(key=lambda row: (-int(row["score"]), str(row["column"])))
    if not candidates:
        return None, candidates

    best = candidates[0]
    second_score = int(candidates[1]["score"]) if len(candidates) > 1 else -1
    # 同分时不自动猜测；唯一最高分时采用。
    if int(best["score"]) >= 10 and int(best["score"]) > second_score:
        return str(best["column"]), candidates
    return None, candidates



def choose_fuzzy_draught_column(
    header: Sequence[str],
    v6: types.ModuleType,
    position: str,
) -> tuple[Optional[str], List[Dict[str, Any]]]:
    """精确别名失败时，保守识别艏/艉吃水字段；同分时不自动选择。"""
    if position not in ("fore", "aft"):
        raise ValueError("position必须为fore或aft。")

    positive_tokens = (
        ("修正后的艏吃水", "修正艏吃水", "艏吃水", "首吃水", "foredraught", "foredraft", "fwddraught", "bowdraught")
        if position == "fore"
        else ("修正后的艉吃水", "修正艉吃水", "艉吃水", "尾吃水", "aftdraught", "aftdraft", "sterndraught")
    )
    opposite_tokens = (
        ("艉", "尾", "aft", "stern")
        if position == "fore"
        else ("艏", "首", "fore", "fwd", "bow")
    )
    exclusions = (
        "mean", "average", "平均", "design", "设计", "difference", "差",
        "trim", "纵倾", "ratio", "比例", "rate", "变化率",
    )

    candidates: List[Dict[str, Any]] = []
    for column in header:
        normalized = v6.normalize_name(column)
        if any(token in normalized for token in exclusions):
            continue
        if any(token in normalized for token in opposite_tokens):
            continue
        score = 0
        matched = ""
        for token in positive_tokens:
            if token in normalized:
                matched = token
                score = max(score, 50 if token.startswith("修正后的") else 40)
        if score <= 0:
            continue
        if normalized.endswith("m"):
            score += 2
        candidates.append({
            "column": str(column),
            "normalized": normalized,
            "score": score,
            "matched_token": matched,
            "position": position,
        })

    candidates.sort(key=lambda row: (-int(row["score"]), str(row["column"])))
    if not candidates:
        return None, candidates
    second_score = int(candidates[1]["score"]) if len(candidates) > 1 else -1
    if int(candidates[0]["score"]) > second_score:
        return str(candidates[0]["column"]), candidates
    return None, candidates

def extract_raw_draught_columns(
    args: argparse.Namespace,
    v6: types.ModuleType,
    ship_type: str,
    raw_path: Path,
    output_path: Path,
) -> pd.DataFrame:
    """按原始文件行序提取首吃水、尾吃水和船长，供V10身份结果回连。"""
    header = v6.read_header(raw_path)
    selected: Dict[str, Optional[str]] = {
        canonical: v6.choose_column(header, aliases)
        for canonical, aliases in RAW_DRAUGHT_ALIASES.items()
    }
    selection_methods = {
        canonical: ("normalized_exact_alias" if source is not None else "not_found")
        for canonical, source in selected.items()
    }

    fuzzy_draught_candidates: Dict[str, List[Dict[str, Any]]] = {
        "fore_draught_m": [],
        "aft_draught_m": [],
    }
    for canonical, position in (
        ("fore_draught_m", "fore"),
        ("aft_draught_m", "aft"),
    ):
        if selected.get(canonical) is None:
            fuzzy_value, candidates = choose_fuzzy_draught_column(
                header, v6, position
            )
            fuzzy_draught_candidates[canonical] = candidates
            if fuzzy_value is not None:
                selected[canonical] = fuzzy_value
                selection_methods[canonical] = (
                    "fuzzy_draught_name_unique_best"
                )

    fuzzy_length_candidates: List[Dict[str, Any]] = []
    if selected.get("ship_length_m") is None:
        fuzzy_length, fuzzy_length_candidates = (
            choose_fuzzy_ship_length_column(header, v6)
        )
        if fuzzy_length is not None:
            selected["ship_length_m"] = fuzzy_length
            selection_methods["ship_length_m"] = "fuzzy_length_name_unique_best"

    # V14.4：三种船型全部按首尾吃水重算纵倾；Bulk/Tanker明确识别df=艏吃水、da=艉吃水。
    required = ["fore_draught_m", "aft_draught_m", "ship_length_m"]
    missing = [field for field in required if selected.get(field) is None]

    mapping_frame = pd.DataFrame([
        {
            "ship_type": ship_type,
            "raw_file": str(raw_path.resolve()),
            "canonical_field": canonical,
            "recognized_column": source if source is not None else "[not found]",
            "required": canonical in required,
            "selection_method": selection_methods.get(canonical, "not_found"),
        }
        for canonical, source in selected.items()
    ])
    if missing:
        diagnostic = (
            args.output_root / "05_clean23" /
            f"00_{ship_type}_raw_draught_field_mapping.csv"
        )
        diagnostic.parent.mkdir(parents=True, exist_ok=True)
        mapping_frame.to_csv(
            diagnostic, index=False, encoding="utf-8-sig"
        )
        candidate_path = (
            args.output_root / "05_clean23" /
            f"00_{ship_type}_ship_length_candidates.csv"
        )
        pd.DataFrame(fuzzy_length_candidates).to_csv(
            candidate_path, index=False, encoding="utf-8-sig"
        )
        fore_candidate_path = (
            args.output_root / "05_clean23" /
            f"00_{ship_type}_fore_draught_candidates.csv"
        )
        aft_candidate_path = (
            args.output_root / "05_clean23" /
            f"00_{ship_type}_aft_draught_candidates.csv"
        )
        pd.DataFrame(
            fuzzy_draught_candidates["fore_draught_m"]
        ).to_csv(
            fore_candidate_path, index=False, encoding="utf-8-sig"
        )
        pd.DataFrame(
            fuzzy_draught_candidates["aft_draught_m"]
        ).to_csv(
            aft_candidate_path, index=False, encoding="utf-8-sig"
        )
        header_path = (
            args.output_root / "05_clean23" /
            f"00_{ship_type}_raw_header.csv"
        )
        pd.DataFrame({"raw_column": header}).to_csv(
            header_path, index=False, encoding="utf-8-sig"
        )
        raise KeyError(
            f"{ship_type}原始文件缺少纵倾重算字段：{missing}。"
            f"三种船型均要求首吃水、尾吃水和船长。"
            f"字段映射：{diagnostic}；"
            f"艏吃水候选：{fore_candidate_path}；"
            f"艉吃水候选：{aft_candidate_path}；"
            f"船长候选：{candidate_path}；原始表头：{header_path}"
        )

    usecols = list(dict.fromkeys(
        source for source in selected.values() if source is not None
    ))
    rename_map = {
        source: canonical
        for canonical, source in selected.items()
        if source is not None
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.unlink(missing_ok=True)
    first = True
    row_start = 0
    reader = pd.read_csv(
        raw_path,
        usecols=usecols,
        chunksize=args.chunksize,
        encoding=v6.detect_encoding(raw_path),
        low_memory=True,
    )
    for block_no, chunk in enumerate(reader, start=1):
        data = chunk.rename(columns=rename_map).copy()
        for column in ("fore_draught_m", "aft_draught_m", "ship_length_m"):
            if column not in data.columns:
                data[column] = pd.NA
        data.insert(0, "raw_source_row", range(row_start, row_start + len(data)))
        row_start += len(data)
        data[[
            "raw_source_row", "fore_draught_m", "aft_draught_m", "ship_length_m"
        ]].to_csv(
            output_path,
            mode="w" if first else "a",
            header=first,
            index=False,
            encoding="utf-8-sig" if first else "utf-8",
        )
        first = False
        print(
            f"[{ship_type}] 首尾吃水/船长提取块 {block_no}，累计 {row_start:,} 行"
        )
        del chunk, data
        gc.collect()

    return mapping_frame




def parse_numeric_series(series: pd.Series) -> pd.Series:
    """清除千位分隔符和单位字符，保留可解析数值。"""
    values = series.astype("string").str.strip()
    values = values.str.replace("\u00a0", "", regex=False)
    values = values.str.replace(",", "", regex=False)
    values = values.str.replace("，", "", regex=False)
    values = values.str.replace(r"[^0-9eE+\-.]", "", regex=True)
    values = values.replace({
        "": pd.NA, ".": pd.NA, "-": pd.NA, "+": pd.NA,
        "nan": pd.NA, "None": pd.NA, "null": pd.NA,
    })
    return pd.to_numeric(values, errors="coerce")


def extract_raw_static_columns(
    args: argparse.Namespace,
    v6: types.ModuleType,
    ship_type: str,
    raw_path: Path,
    output_path: Path,
) -> pd.DataFrame:
    """按原始行序提取四项船舶固有特征；缺失字段保留为空并写入映射审计。"""
    header = v6.read_header(raw_path)
    selected: Dict[str, Optional[str]] = {
        canonical: v6.choose_column(header, aliases)
        for canonical, aliases in RAW_SHIP_STATIC_ALIASES.items()
    }
    mapping_frame = pd.DataFrame([
        {
            "ship_type": ship_type,
            "raw_file": str(raw_path.resolve()),
            "canonical_field": canonical,
            "recognized_column": source if source is not None else "[not found]",
            "required_for_fixed23": False,
            "required_for_fixed27": True,
            "selection_method": (
                "normalized_exact_alias" if source is not None else "not_found"
            ),
        }
        for canonical, source in selected.items()
    ])

    usecols = list(dict.fromkeys(
        source for source in selected.values() if source is not None
    ))
    output_columns = ["raw_source_row"] + STATIC4_PREDICTORS
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.unlink(missing_ok=True)

    if not usecols:
        pd.DataFrame(columns=output_columns).to_csv(
            output_path, index=False, encoding="utf-8-sig"
        )
        print(
            f"[warning] {ship_type}原始文件未识别到四项静态字段；"
            "Fixed23仍可生成，Fixed27将缺少该船型。"
        )
        return mapping_frame

    first = True
    row_start = 0
    reader = pd.read_csv(
        raw_path,
        usecols=usecols,
        chunksize=args.chunksize,
        encoding=v6.detect_encoding(raw_path),
        low_memory=True,
    )
    for block_no, chunk in enumerate(reader, start=1):
        data = pd.DataFrame(index=chunk.index)
        data.insert(0, "raw_source_row", range(row_start, row_start + len(chunk)))
        row_start += len(chunk)
        for canonical in STATIC4_PREDICTORS:
            source = selected.get(canonical)
            data[canonical] = (
                parse_numeric_series(chunk[source])
                if source is not None else pd.NA
            )
        data[output_columns].to_csv(
            output_path,
            mode="w" if first else "a",
            header=first,
            index=False,
            encoding="utf-8-sig" if first else "utf-8",
        )
        first = False
        print(
            f"[{ship_type}] 船舶静态字段提取块 {block_no}，累计 {row_start:,} 行"
        )
        del chunk, data
        gc.collect()
    return mapping_frame


def extract_raw_container_fuel_columns(
    args: argparse.Namespace,
    v6: types.ModuleType,
    raw_path: Path,
    output_path: Path,
) -> pd.DataFrame:
    """按原始行序提取集装箱燃油流量及发动机审计字段。"""
    header = v6.read_header(raw_path)
    selected: Dict[str, Optional[str]] = {
        canonical: v6.choose_column(header, aliases)
        for canonical, aliases in RAW_CONTAINER_FUEL_ALIASES.items()
    }
    required = ["fuel_rate_kg_h"]
    missing = [field for field in required if selected.get(field) is None]
    mapping_frame = pd.DataFrame([
        {
            "ship_type": "container",
            "raw_file": str(raw_path.resolve()),
            "canonical_field": canonical,
            "recognized_column": source if source is not None else "[not found]",
            "required": canonical in required,
            "selection_method": (
                "normalized_exact_alias" if source is not None else "not_found"
            ),
        }
        for canonical, source in selected.items()
    ])
    diagnostic = (
        args.output_root / "05_clean23" /
        "00_container_raw_fuel_field_mapping.csv"
    )
    diagnostic.parent.mkdir(parents=True, exist_ok=True)
    mapping_frame.to_csv(diagnostic, index=False, encoding="utf-8-sig")
    if missing:
        header_path = (
            args.output_root / "05_clean23" /
            "00_container_raw_fuel_header.csv"
        )
        pd.DataFrame({"raw_column": header}).to_csv(
            header_path, index=False, encoding="utf-8-sig"
        )
        raise KeyError(
            "container原始文件缺少燃油重算字段："
            f"{missing}。字段映射：{diagnostic}；原始表头：{header_path}"
        )

    usecols = list(dict.fromkeys(
        source for source in selected.values() if source is not None
    ))
    rename_map = {
        source: canonical
        for canonical, source in selected.items()
        if source is not None
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.unlink(missing_ok=True)
    first = True
    row_start = 0
    reader = pd.read_csv(
        raw_path,
        usecols=usecols,
        chunksize=args.chunksize,
        encoding=v6.detect_encoding(raw_path),
        low_memory=True,
    )
    output_columns = [
        "raw_source_row", "fuel_rate_kg_h", "shaft_power_kw",
        "sfoc_g_kwh_existing", "main_engine_rpm",
    ]
    for block_no, chunk in enumerate(reader, start=1):
        data = chunk.rename(columns=rename_map).copy()
        for column in output_columns[1:]:
            if column not in data.columns:
                data[column] = pd.NA
        data.insert(0, "raw_source_row", range(row_start, row_start + len(data)))
        row_start += len(data)
        data[output_columns].to_csv(
            output_path,
            mode="w" if first else "a",
            header=first,
            index=False,
            encoding="utf-8-sig" if first else "utf-8",
        )
        first = False
        print(
            f"[container] 原始燃油字段提取块 {block_no}，累计 {row_start:,} 行"
        )
        del chunk, data
        gc.collect()
    return mapping_frame

def vector_clean_sql(sin_column: str, cos_column: str, component: str) -> str:
    sin_id = quote_identifier(sin_column)
    cos_id = quote_identifier(cos_column)
    norm = f"sqrt(power({sin_id}, 2) + power({cos_id}, 2))"
    source = sin_id if component == "sin" else cos_id
    return f"""
        CASE
            WHEN {sin_id} IS NOT NULL AND {cos_id} IS NOT NULL
             AND isfinite({sin_id}) AND isfinite({cos_id})
             AND abs({sin_id}) <= 1.000001
             AND abs({cos_id}) <= 1.000001
             AND {norm} > 0.000001
             AND {norm} <= 1.010001
            THEN {source} / {norm}
            ELSE NULL
        END
    """


def run_clean23(
    args: argparse.Namespace,
    v6: types.ModuleType,
) -> None:
    """复用V10单船身份，回连原始吃水、燃油和四项静态特征，生成Fixed23/Fixed27数据。"""
    p = paths(args)
    ensure_root_dirs(args)
    output_dir = p["clean23"]
    outputs = [
        p["clean23_cruise"],
        output_dir / "final_fixed23_navigation.csv",
        output_dir / "final_fixed23_all_phases.csv",
        output_dir / "model_matrix_23_predictors_plus_target.csv",
        output_dir / "final_fixed27_cruise.csv",
        output_dir / "model_matrix_27_predictors_plus_target.csv",
        output_dir / "05_ship_static_feature_audit.csv",
        output_dir / "cleaning_flow_by_ship_type.csv",
        output_dir / "validity_by_ship.csv",
    ]
    require_files({
        "unified_features": p["unified_features"],
        "container_stage": p["container_stage"],
        "bulk_stage": p["bulk_stage"],
        "tanker_stage": p["tanker_stage"],
        "container_identity": p["container_identity"],
        "container_identity_summary": p["identity"] / "container_identity_summary.csv",
        "bulk_identity": p["bulk_identity"],
        "tanker_identity": p["tanker_identity"],
        "raw_container": args.raw_container,
        "raw_bulk": args.raw_bulk,
        "raw_tanker": args.raw_tanker,
    })
    if skip_or_prepare(args, "clean23", outputs):
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = p["work"] / "clean23"
    if args.force and work_dir.exists():
        shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    mapping_frames = []
    raw_extracts: Dict[str, Path] = {}
    for ship_type, raw_path in (
        ("container", args.raw_container),
        ("bulk", args.raw_bulk),
        ("tanker", args.raw_tanker),
    ):
        target = work_dir / f"{ship_type}_raw_draught_length.csv"
        mapping_frames.append(
            extract_raw_draught_columns(
                args, v6, ship_type, raw_path, target
            )
        )
        raw_extracts[ship_type] = target

    pd.concat(mapping_frames, ignore_index=True).to_csv(
        output_dir / "00_raw_draught_field_mapping.csv",
        index=False,
        encoding="utf-8-sig",
    )

    static_mapping_frames = []
    raw_static_extracts: Dict[str, Path] = {}
    for ship_type, raw_path in (
        ("container", args.raw_container),
        ("bulk", args.raw_bulk),
        ("tanker", args.raw_tanker),
    ):
        target = work_dir / f"{ship_type}_raw_static4.csv"
        static_mapping_frames.append(
            extract_raw_static_columns(
                args, v6, ship_type, raw_path, target
            )
        )
        raw_static_extracts[ship_type] = target

    pd.concat(static_mapping_frames, ignore_index=True).to_csv(
        output_dir / "00_raw_static4_field_mapping.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # V15.0：燃油不再继承统一表中的集装箱结果，而是从最原始文件按source_row回连。
    container_fuel_extract = work_dir / "container_raw_fuel_fields.csv"
    container_fuel_mapping = extract_raw_container_fuel_columns(
        args, v6, args.raw_container, container_fuel_extract
    )
    container_fuel_mapping.to_csv(
        output_dir / "00_container_raw_fuel_field_mapping.csv",
        index=False,
        encoding="utf-8-sig",
    )

    con, db_work = open_step_db(args, "clean23_sql")
    try:
        # 输入视图均只在SQL需要时扫描，避免把大CSV整体放入内存。
        con.execute(
            f"""
            CREATE OR REPLACE VIEW unified_input AS
            SELECT *
            FROM read_csv_auto(
                '{sql_path(p["unified_features"])}',
                header=true, sample_size=100000
            )
            """
        )

        for ship_type in ("container", "bulk", "tanker"):
            con.execute(
                f"""
                CREATE OR REPLACE VIEW {ship_type}_raw_draught AS
                SELECT
                    try_cast(raw_source_row AS BIGINT) AS raw_source_row,
                    try_cast(fore_draught_m AS DOUBLE) AS fore_draught_m,
                    try_cast(aft_draught_m AS DOUBLE) AS aft_draught_m,
                    try_cast(ship_length_m AS DOUBLE) AS ship_length_m
                FROM read_csv_auto(
                    '{sql_path(raw_extracts[ship_type])}',
                    header=true, all_varchar=true, sample_size=100000
                )
                """
            )

        for ship_type in ("container", "bulk", "tanker"):
            con.execute(
                f"""
                CREATE OR REPLACE VIEW {ship_type}_raw_static4 AS
                SELECT
                    try_cast(raw_source_row AS BIGINT) AS raw_source_row,
                    try_cast(design_draught_m AS DOUBLE) AS design_draught_m_raw,
                    try_cast(deadweight_t AS DOUBLE) AS deadweight_t_raw,
                    try_cast(service_speed_kn AS DOUBLE) AS service_speed_kn_raw,
                    try_cast(main_engine_power_kw AS DOUBLE)
                        AS main_engine_power_kw_raw
                FROM read_csv_auto(
                    '{sql_path(raw_static_extracts[ship_type])}',
                    header=true, all_varchar=true, sample_size=100000
                )
                """
            )

        con.execute(
            f"""
            CREATE OR REPLACE VIEW container_raw_fuel AS
            SELECT
                try_cast(raw_source_row AS BIGINT) AS raw_source_row,
                try_cast(fuel_rate_kg_h AS DOUBLE) AS fuel_rate_kg_h,
                try_cast(shaft_power_kw AS DOUBLE) AS shaft_power_kw,
                try_cast(sfoc_g_kwh_existing AS DOUBLE)
                    AS sfoc_g_kwh_existing,
                try_cast(main_engine_rpm AS DOUBLE) AS main_engine_rpm
            FROM read_csv_auto(
                '{sql_path(container_fuel_extract)}',
                header=true, all_varchar=true, sample_size=100000
            )
            """
        )

        con.execute(
            f"""
            CREATE OR REPLACE VIEW container_stage_narrow AS
            SELECT
                try_cast(source_row AS BIGINT) AS source_row,
                try_cast(timestamp_utc AS TIMESTAMP) AS timestamp_utc,
                try_cast(delivery_date AS DATE) AS delivery_date
            FROM read_csv_auto(
                '{sql_path(p["container_stage"])}',
                header=true, all_varchar=true, sample_size=100000
            )
            """
        )
        con.execute(
            f"""
            CREATE OR REPLACE VIEW container_identity_summary_narrow AS
            SELECT
                cast(pseudo_ship_group_id AS VARCHAR) AS pseudo_ship_group_id,
                try_cast(delivery_date AS DATE) AS delivery_date
            FROM read_csv_auto(
                '{sql_path(p["identity"] / "container_identity_summary.csv")}',
                header=true, all_varchar=true, sample_size=100000
            )
            """
        )
        con.execute(
            f"""
            CREATE OR REPLACE VIEW container_identity_narrow AS
            SELECT
                cast(pseudo_ship_group_id AS VARCHAR) AS pseudo_ship_group_id,
                cast(trajectory_segment_id AS VARCHAR) AS trajectory_segment_id,
                try_cast(timestamp_utc AS TIMESTAMP) AS timestamp_utc
            FROM read_csv_auto(
                '{sql_path(p["container_identity"])}',
                header=true, all_varchar=true, sample_size=100000
            )
            """
        )
        con.execute(
            """
            CREATE OR REPLACE TABLE container_draught_10min AS
            WITH stage_mapped AS (
                SELECT
                    m.pseudo_ship_group_id,
                    s.timestamp_utc,
                    r.fore_draught_m,
                    r.aft_draught_m,
                    r.ship_length_m
                FROM container_stage_narrow s
                JOIN container_identity_summary_narrow m USING(delivery_date)
                LEFT JOIN container_raw_draught r
                  ON r.raw_source_row = s.source_row
            )
            SELECT
                i.pseudo_ship_group_id,
                i.trajectory_segment_id,
                i.timestamp_utc,
                avg(s.fore_draught_m) AS fore_draught_m,
                avg(s.aft_draught_m) AS aft_draught_m,
                median(s.ship_length_m) AS ship_length_m
            FROM container_identity_narrow i
            LEFT JOIN stage_mapped s
              ON s.pseudo_ship_group_id = i.pseudo_ship_group_id
             AND s.timestamp_utc = i.timestamp_utc
            GROUP BY
                i.pseudo_ship_group_id,
                i.trajectory_segment_id,
                i.timestamp_utc
            """
        )

        con.execute(
            """
            CREATE OR REPLACE TABLE container_fuel_10min AS
            WITH stage_mapped AS (
                SELECT
                    m.pseudo_ship_group_id,
                    s.timestamp_utc,
                    f.fuel_rate_kg_h,
                    f.shaft_power_kw,
                    f.sfoc_g_kwh_existing,
                    f.main_engine_rpm
                FROM container_stage_narrow s
                JOIN container_identity_summary_narrow m USING(delivery_date)
                LEFT JOIN container_raw_fuel f
                  ON f.raw_source_row = s.source_row
            ), raw_agg AS (
                SELECT
                    i.pseudo_ship_group_id,
                    i.trajectory_segment_id,
                    i.timestamp_utc,
                    avg(s.fuel_rate_kg_h) FILTER (
                        WHERE s.fuel_rate_kg_h IS NOT NULL
                          AND isfinite(s.fuel_rate_kg_h)
                          AND s.fuel_rate_kg_h>=0
                    ) AS fuel_rate_kg_h_recalculated,
                    avg(s.shaft_power_kw) FILTER (
                        WHERE s.shaft_power_kw IS NOT NULL
                          AND isfinite(s.shaft_power_kw)
                          AND s.shaft_power_kw>0
                    ) AS shaft_power_kw_recalculated,
                    avg(s.sfoc_g_kwh_existing) FILTER (
                        WHERE s.sfoc_g_kwh_existing IS NOT NULL
                          AND isfinite(s.sfoc_g_kwh_existing)
                          AND s.sfoc_g_kwh_existing>=0
                    ) AS sfoc_g_kwh_existing,
                    avg(s.main_engine_rpm) FILTER (
                        WHERE s.main_engine_rpm IS NOT NULL
                          AND isfinite(s.main_engine_rpm)
                          AND s.main_engine_rpm>=0
                    ) AS main_engine_rpm_recalculated,
                    count(s.fuel_rate_kg_h) AS fuel_source_rows,
                    sum(CASE WHEN s.fuel_rate_kg_h IS NOT NULL
                                   AND isfinite(s.fuel_rate_kg_h)
                                   AND s.fuel_rate_kg_h>=0
                             THEN 1 ELSE 0 END) AS valid_fuel_source_rows
                FROM container_identity_narrow i
                LEFT JOIN stage_mapped s
                  ON s.pseudo_ship_group_id = i.pseudo_ship_group_id
                 AND s.timestamp_utc = i.timestamp_utc
                GROUP BY
                    i.pseudo_ship_group_id,
                    i.trajectory_segment_id,
                    i.timestamp_utc
            )
            SELECT
                *,
                fuel_rate_kg_h_recalculated/6.0
                    AS fuel_kg_10min_recalculated,
                fuel_rate_kg_h_recalculated/6000.0
                    AS fuel_t_10min_recalculated,
                CASE
                    WHEN shaft_power_kw_recalculated>0
                     AND fuel_rate_kg_h_recalculated>=0
                    THEN fuel_rate_kg_h_recalculated*1000.0
                         /shaft_power_kw_recalculated
                END AS sfoc_g_kwh_recalculated
            FROM raw_agg
            """
        )

        for ship_type in ("bulk", "tanker"):
            stage_path = p[f"{ship_type}_stage"]
            identity_path = p[f"{ship_type}_identity"]
            con.execute(
                f"""
                CREATE OR REPLACE VIEW {ship_type}_stage_narrow AS
                SELECT
                    try_cast(source_row AS BIGINT) AS source_row,
                    try_cast(matched_raw_source_row AS BIGINT)
                        AS matched_raw_source_row
                FROM read_csv_auto(
                    '{sql_path(stage_path)}',
                    header=true, all_varchar=true, sample_size=100000
                )
                """
            )
            con.execute(
                f"""
                CREATE OR REPLACE VIEW {ship_type}_identity_narrow AS
                SELECT
                    try_cast(source_row AS BIGINT) AS source_row,
                    cast(pseudo_ship_group_id AS VARCHAR)
                        AS pseudo_ship_group_id,
                    cast(trajectory_segment_id AS VARCHAR)
                        AS trajectory_segment_id,
                    try_cast(timestamp_utc AS TIMESTAMP) AS timestamp_utc
                FROM read_csv_auto(
                    '{sql_path(identity_path)}',
                    header=true, all_varchar=true, sample_size=100000
                )
                """
            )
            con.execute(
                f"""
                CREATE OR REPLACE TABLE {ship_type}_draught_10min AS
                SELECT
                    i.pseudo_ship_group_id,
                    i.trajectory_segment_id,
                    time_bucket(INTERVAL '10 minutes', i.timestamp_utc)
                        AS timestamp_utc,
                    avg(r.fore_draught_m) AS fore_draught_m,
                    avg(r.aft_draught_m) AS aft_draught_m,
                    median(r.ship_length_m) AS ship_length_m,
                    count(*) AS matched_source_rows,
                    sum(CASE WHEN r.fore_draught_m IS NOT NULL
                              AND r.aft_draught_m IS NOT NULL
                             THEN 1 ELSE 0 END)
                        AS complete_draught_source_rows
                FROM {ship_type}_identity_narrow i
                JOIN {ship_type}_stage_narrow s USING(source_row)
                LEFT JOIN {ship_type}_raw_draught r
                  ON r.raw_source_row = s.matched_raw_source_row
                WHERE i.timestamp_utc IS NOT NULL
                GROUP BY
                    i.pseudo_ship_group_id,
                    i.trajectory_segment_id,
                    time_bucket(INTERVAL '10 minutes', i.timestamp_utc)
                """
            )

        con.execute(
            """
            CREATE OR REPLACE VIEW supplemental_draught AS
            SELECT
                'container'::VARCHAR AS ship_type,
                pseudo_ship_group_id,
                trajectory_segment_id,
                timestamp_utc,
                fore_draught_m,
                aft_draught_m,
                ship_length_m
            FROM container_draught_10min
            UNION ALL BY NAME
            SELECT
                'bulk'::VARCHAR AS ship_type,
                pseudo_ship_group_id,
                trajectory_segment_id,
                timestamp_utc,
                fore_draught_m,
                aft_draught_m,
                ship_length_m
            FROM bulk_draught_10min
            UNION ALL BY NAME
            SELECT
                'tanker'::VARCHAR AS ship_type,
                pseudo_ship_group_id,
                trajectory_segment_id,
                timestamp_utc,
                fore_draught_m,
                aft_draught_m,
                ship_length_m
            FROM tanker_draught_10min
            """
        )

        con.execute(
            """
            CREATE OR REPLACE VIEW ship_static_source_rows AS
            WITH container_rows AS (
                SELECT
                    'container'::VARCHAR AS ship_type,
                    m.pseudo_ship_group_id,
                    r.design_draught_m_raw,
                    r.deadweight_t_raw,
                    r.service_speed_kn_raw,
                    r.main_engine_power_kw_raw
                FROM container_stage_narrow s
                JOIN container_identity_summary_narrow m USING(delivery_date)
                LEFT JOIN container_raw_static4 r
                  ON r.raw_source_row=s.source_row
            ), bulk_rows AS (
                SELECT
                    'bulk'::VARCHAR AS ship_type,
                    i.pseudo_ship_group_id,
                    r.design_draught_m_raw,
                    r.deadweight_t_raw,
                    r.service_speed_kn_raw,
                    r.main_engine_power_kw_raw
                FROM bulk_identity_narrow i
                JOIN bulk_stage_narrow s USING(source_row)
                LEFT JOIN bulk_raw_static4 r
                  ON r.raw_source_row=s.matched_raw_source_row
            ), tanker_rows AS (
                SELECT
                    'tanker'::VARCHAR AS ship_type,
                    i.pseudo_ship_group_id,
                    r.design_draught_m_raw,
                    r.deadweight_t_raw,
                    r.service_speed_kn_raw,
                    r.main_engine_power_kw_raw
                FROM tanker_identity_narrow i
                JOIN tanker_stage_narrow s USING(source_row)
                LEFT JOIN tanker_raw_static4 r
                  ON r.raw_source_row=s.matched_raw_source_row
            )
            SELECT * FROM container_rows
            UNION ALL BY NAME SELECT * FROM bulk_rows
            UNION ALL BY NAME SELECT * FROM tanker_rows
            """
        )
        con.execute(
            """
            CREATE OR REPLACE TABLE ship_static_valid_rows AS
            SELECT
                ship_type,
                pseudo_ship_group_id,
                design_draught_m_raw,
                deadweight_t_raw,
                service_speed_kn_raw,
                main_engine_power_kw_raw,
                CASE WHEN design_draught_m_raw>0
                           AND design_draught_m_raw<=30
                           AND isfinite(design_draught_m_raw)
                     THEN design_draught_m_raw END AS design_draught_m_valid,
                CASE WHEN deadweight_t_raw>0
                           AND deadweight_t_raw<=1000000
                           AND isfinite(deadweight_t_raw)
                     THEN deadweight_t_raw END AS deadweight_t_valid,
                CASE WHEN service_speed_kn_raw>0
                           AND service_speed_kn_raw<=40
                           AND isfinite(service_speed_kn_raw)
                     THEN service_speed_kn_raw END AS service_speed_kn_valid,
                CASE WHEN main_engine_power_kw_raw>0
                           AND main_engine_power_kw_raw<=200000
                           AND isfinite(main_engine_power_kw_raw)
                     THEN main_engine_power_kw_raw END
                    AS main_engine_power_kw_valid
            FROM ship_static_source_rows
            WHERE pseudo_ship_group_id IS NOT NULL
              AND trim(pseudo_ship_group_id)<>''
            """
        )
        con.execute(
            """
            CREATE OR REPLACE TABLE ship_static_final AS
            SELECT
                ship_type,
                pseudo_ship_group_id,
                count(*) AS source_rows,
                median(design_draught_m_valid) AS design_draught_m,
                median(deadweight_t_valid) AS deadweight_t,
                median(service_speed_kn_valid) AS service_speed_kn,
                median(main_engine_power_kw_valid) AS main_engine_power_kw,
                count(design_draught_m_raw) AS design_draught_raw_nonmissing,
                count(design_draught_m_valid) AS design_draught_valid_rows,
                count(DISTINCT round(design_draught_m_valid,6))
                    AS design_draught_valid_unique,
                min(design_draught_m_valid) AS design_draught_valid_min,
                max(design_draught_m_valid) AS design_draught_valid_max,
                count(deadweight_t_raw) AS deadweight_raw_nonmissing,
                count(deadweight_t_valid) AS deadweight_valid_rows,
                count(DISTINCT round(deadweight_t_valid,3))
                    AS deadweight_valid_unique,
                min(deadweight_t_valid) AS deadweight_valid_min,
                max(deadweight_t_valid) AS deadweight_valid_max,
                count(service_speed_kn_raw) AS service_speed_raw_nonmissing,
                count(service_speed_kn_valid) AS service_speed_valid_rows,
                count(DISTINCT round(service_speed_kn_valid,6))
                    AS service_speed_valid_unique,
                min(service_speed_kn_valid) AS service_speed_valid_min,
                max(service_speed_kn_valid) AS service_speed_valid_max,
                count(main_engine_power_kw_raw) AS main_engine_power_raw_nonmissing,
                count(main_engine_power_kw_valid) AS main_engine_power_valid_rows,
                count(DISTINCT round(main_engine_power_kw_valid,3))
                    AS main_engine_power_valid_unique,
                min(main_engine_power_kw_valid) AS main_engine_power_valid_min,
                max(main_engine_power_kw_valid) AS main_engine_power_valid_max
            FROM ship_static_valid_rows
            GROUP BY ship_type,pseudo_ship_group_id
            """
        )

        # V14.2：Container、Bulk、Tanker全部由首尾吃水重新计算纵倾；不依赖任何原始trim字段。
        if args.trim_sign == "aft-minus-fore":
            recomputed_trim = "d.aft_draught_m - d.fore_draught_m"
        else:
            recomputed_trim = "d.fore_draught_m - d.aft_draught_m"

        course_sin_clean = vector_clean_sql("course_sin_raw", "course_cos_raw", "sin")
        course_cos_clean = vector_clean_sql("course_sin_raw", "course_cos_raw", "cos")
        heading_sin_clean = vector_clean_sql("heading_sin_raw", "heading_cos_raw", "sin")
        heading_cos_clean = vector_clean_sql("heading_sin_raw", "heading_cos_raw", "cos")
        wind_sin_clean = vector_clean_sql("relative_wind_sin_raw", "relative_wind_cos_raw", "sin")
        wind_cos_clean = vector_clean_sql("relative_wind_sin_raw", "relative_wind_cos_raw", "cos")
        wave_sin_clean = vector_clean_sql("relative_wave_sin_raw", "relative_wave_cos_raw", "sin")
        wave_cos_clean = vector_clean_sql("relative_wave_sin_raw", "relative_wave_cos_raw", "cos")

        con.execute(
            f"""
            CREATE OR REPLACE TABLE clean23_joined AS
            WITH unified_lagged AS (
                SELECT
                    u.*,
                    lag(try_cast(u.timestamp_utc AS TIMESTAMP)) OVER(
                        PARTITION BY
                            lower(cast(u.ship_type AS VARCHAR)),
                            cast(u.pseudo_ship_group_id AS VARCHAR),
                            cast(u.trajectory_segment_id AS VARCHAR)
                        ORDER BY try_cast(u.timestamp_utc AS TIMESTAMP)
                    ) AS previous_timestamp_utc,
                    lag(try_cast(u.latitude_deg AS DOUBLE)) OVER(
                        PARTITION BY
                            lower(cast(u.ship_type AS VARCHAR)),
                            cast(u.pseudo_ship_group_id AS VARCHAR),
                            cast(u.trajectory_segment_id AS VARCHAR)
                        ORDER BY try_cast(u.timestamp_utc AS TIMESTAMP)
                    ) AS previous_latitude_deg,
                    lag(try_cast(u.longitude_deg AS DOUBLE)) OVER(
                        PARTITION BY
                            lower(cast(u.ship_type AS VARCHAR)),
                            cast(u.pseudo_ship_group_id AS VARCHAR),
                            cast(u.trajectory_segment_id AS VARCHAR)
                        ORDER BY try_cast(u.timestamp_utc AS TIMESTAMP)
                    ) AS previous_longitude_deg
                FROM unified_input u
            )
            SELECT
                cast(u.ship_type AS VARCHAR) AS ship_type,
                cast(u.pseudo_ship_group_id AS VARCHAR)
                    AS pseudo_ship_group_id,
                cast(u.trajectory_segment_id AS VARCHAR)
                    AS trajectory_segment_id,
                try_cast(u.timestamp_utc AS TIMESTAMP) AS timestamp_utc,
                cast(u.id_confidence AS VARCHAR) AS id_confidence,
                try_cast(u.complete_window_flag AS INTEGER)
                    AS complete_window_flag,
                cast(u.aggregation_status AS VARCHAR) AS aggregation_status,
                try_cast(u.speed_kn AS DOUBLE) AS speed_kn_raw,
                try_cast(u.course_sin AS DOUBLE) AS course_sin_raw,
                try_cast(u.course_cos AS DOUBLE) AS course_cos_raw,
                try_cast(u.heading_sin AS DOUBLE) AS heading_sin_raw,
                try_cast(u.heading_cos AS DOUBLE) AS heading_cos_raw,
                try_cast(u.mean_draught_m AS DOUBLE)
                    AS mean_draught_existing_m,
                CASE
                    WHEN lower(cast(u.ship_type AS VARCHAR))='container'
                     AND d.fore_draught_m IS NOT NULL
                     AND d.aft_draught_m IS NOT NULL
                    THEN (d.fore_draught_m+d.aft_draught_m)/2.0
                    ELSE try_cast(u.mean_draught_m AS DOUBLE)
                END AS mean_draught_m_raw,
                CASE
                    WHEN lower(cast(u.ship_type AS VARCHAR))='container'
                     AND d.fore_draught_m IS NOT NULL
                     AND d.aft_draught_m IS NOT NULL
                    THEN 'fore_aft_average'
                    WHEN try_cast(u.mean_draught_m AS DOUBLE) IS NOT NULL
                    THEN 'unified_existing'
                    ELSE 'unavailable'
                END AS mean_draught_source,
                try_cast(u.rudder_deg AS DOUBLE) AS rudder_deg_raw,
                try_cast(u.latitude_deg AS DOUBLE) AS latitude_deg_raw,
                try_cast(u.longitude_deg AS DOUBLE) AS longitude_deg_raw,
                try_cast(u.distance_nm_10min AS DOUBLE)
                    AS distance_nm_10min_existing_raw,
                CASE
                    WHEN try_cast(u.distance_nm_10min AS DOUBLE) IS NOT NULL
                     AND isfinite(try_cast(u.distance_nm_10min AS DOUBLE))
                    THEN try_cast(u.distance_nm_10min AS DOUBLE)
                    WHEN lower(cast(u.ship_type AS VARCHAR))='container'
                     AND try_cast(u.latitude_deg AS DOUBLE) BETWEEN -90 AND 90
                     AND try_cast(u.longitude_deg AS DOUBLE) BETWEEN -180 AND 180
                     AND u.previous_latitude_deg BETWEEN -90 AND 90
                     AND u.previous_longitude_deg BETWEEN -180 AND 180
                     AND date_diff(
                         'minute',u.previous_timestamp_utc,
                         try_cast(u.timestamp_utc AS TIMESTAMP)
                     ) BETWEEN 1 AND 15
                    THEN 2.0*3440.065*asin(
                        sqrt(
                            least(
                                1.0,
                                greatest(
                                    0.0,
                                    power(
                                        sin(radians(
                                            try_cast(u.latitude_deg AS DOUBLE)
                                            -u.previous_latitude_deg
                                        )/2.0),2
                                    )
                                    +cos(radians(u.previous_latitude_deg))
                                     *cos(radians(
                                         try_cast(u.latitude_deg AS DOUBLE)
                                      ))
                                     *power(
                                         sin(radians(
                                             try_cast(u.longitude_deg AS DOUBLE)
                                             -u.previous_longitude_deg
                                         )/2.0),2
                                      )
                                )
                            )
                        )
                    )
                    WHEN lower(cast(u.ship_type AS VARCHAR))='container'
                     AND try_cast(u.speed_kn AS DOUBLE) IS NOT NULL
                     AND isfinite(try_cast(u.speed_kn AS DOUBLE))
                     AND try_cast(u.speed_kn AS DOUBLE) BETWEEN 0 AND 40
                     AND coalesce(try_cast(u.complete_window_flag AS INTEGER),1)=1
                    THEN try_cast(u.speed_kn AS DOUBLE)/6.0
                    ELSE NULL
                END AS distance_nm_10min_raw,
                CASE
                    WHEN try_cast(u.distance_nm_10min AS DOUBLE) IS NOT NULL
                     AND isfinite(try_cast(u.distance_nm_10min AS DOUBLE))
                    THEN 'unified_existing'
                    WHEN lower(cast(u.ship_type AS VARCHAR))='container'
                     AND try_cast(u.latitude_deg AS DOUBLE) BETWEEN -90 AND 90
                     AND try_cast(u.longitude_deg AS DOUBLE) BETWEEN -180 AND 180
                     AND u.previous_latitude_deg BETWEEN -90 AND 90
                     AND u.previous_longitude_deg BETWEEN -180 AND 180
                     AND date_diff(
                         'minute',u.previous_timestamp_utc,
                         try_cast(u.timestamp_utc AS TIMESTAMP)
                     ) BETWEEN 1 AND 15
                    THEN 'haversine_previous_position'
                    WHEN lower(cast(u.ship_type AS VARCHAR))='container'
                     AND try_cast(u.speed_kn AS DOUBLE) IS NOT NULL
                     AND isfinite(try_cast(u.speed_kn AS DOUBLE))
                     AND try_cast(u.speed_kn AS DOUBLE) BETWEEN 0 AND 40
                     AND coalesce(try_cast(u.complete_window_flag AS INTEGER),1)=1
                    THEN 'speed_time_integration_10min'
                    ELSE 'unavailable'
                END AS distance_10min_source,
                try_cast(u.rel_wind_speed_kn AS DOUBLE)
                    AS rel_wind_speed_kn_raw,
                try_cast(u.relative_wind_sin AS DOUBLE)
                    AS relative_wind_sin_raw,
                try_cast(u.relative_wind_cos AS DOUBLE)
                    AS relative_wind_cos_raw,
                try_cast(u.wave_height_m AS DOUBLE) AS wave_height_m_raw,
                try_cast(u.wave_period_s AS DOUBLE) AS wave_period_s_raw,
                try_cast(u.relative_wave_sin AS DOUBLE)
                    AS relative_wave_sin_raw,
                try_cast(u.relative_wave_cos AS DOUBLE)
                    AS relative_wave_cos_raw,
                try_cast(u.surface_pressure_pa AS DOUBLE)
                    AS surface_pressure_raw,
                try_cast(u.surface_temperature_c AS DOUBLE)
                    AS surface_temperature_raw,
                try_cast(u.fuel_t_10min AS DOUBLE)
                    AS fuel_t_10min_existing_raw,
                try_cast(u.fuel_rate_kg_h AS DOUBLE)
                    AS fuel_rate_kg_h_existing_raw,
                f.fuel_rate_kg_h_recalculated
                    AS fuel_rate_kg_h_recalculated_raw,
                f.fuel_kg_10min_recalculated
                    AS fuel_kg_10min_recalculated_raw,
                f.fuel_t_10min_recalculated
                    AS fuel_t_10min_recalculated_raw,
                f.shaft_power_kw_recalculated
                    AS shaft_power_kw_fuel_raw,
                f.sfoc_g_kwh_existing AS sfoc_g_kwh_existing_raw,
                f.sfoc_g_kwh_recalculated AS sfoc_g_kwh_recalculated_raw,
                f.main_engine_rpm_recalculated AS main_engine_rpm_fuel_raw,
                f.fuel_source_rows AS container_fuel_source_rows,
                f.valid_fuel_source_rows AS container_valid_fuel_source_rows,
                CASE
                    WHEN lower(cast(u.ship_type AS VARCHAR))='container'
                    THEN f.fuel_rate_kg_h_recalculated
                    ELSE try_cast(u.fuel_rate_kg_h AS DOUBLE)
                END AS fuel_rate_kg_h_raw,
                CASE
                    WHEN lower(cast(u.ship_type AS VARCHAR))='container'
                    THEN f.fuel_kg_10min_recalculated
                    WHEN try_cast(u.fuel_t_10min AS DOUBLE) IS NOT NULL
                    THEN try_cast(u.fuel_t_10min AS DOUBLE)*1000.0
                END AS fuel_kg_10min_raw,
                CASE
                    WHEN lower(cast(u.ship_type AS VARCHAR))='container'
                    THEN f.fuel_t_10min_recalculated
                    ELSE try_cast(u.fuel_t_10min AS DOUBLE)
                END AS fuel_t_10min_raw,
                CASE
                    WHEN lower(cast(u.ship_type AS VARCHAR))='container'
                     AND f.fuel_t_10min_recalculated IS NOT NULL
                    THEN 'raw_container_fuel_rate_kg_h_recalculated'
                    WHEN lower(cast(u.ship_type AS VARCHAR))='container'
                    THEN 'raw_container_fuel_unavailable'
                    WHEN try_cast(u.fuel_t_10min AS DOUBLE) IS NOT NULL
                    THEN 'unified_existing'
                    ELSE 'unavailable'
                END AS fuel_10min_source,
                NULL::DOUBLE AS trim_existing_m,
                d.fore_draught_m,
                d.aft_draught_m,
                d.ship_length_m,
                st.design_draught_m,
                st.deadweight_t,
                st.service_speed_kn,
                st.main_engine_power_kw,
                CASE
                    WHEN lower(cast(u.ship_type AS VARCHAR))
                         IN ('container','bulk','tanker')
                     AND d.fore_draught_m IS NOT NULL
                     AND d.aft_draught_m IS NOT NULL
                    THEN {recomputed_trim}
                    ELSE NULL
                END AS trim_raw_m,
                CASE
                    WHEN lower(cast(u.ship_type AS VARCHAR))
                         IN ('container','bulk','tanker')
                     AND d.fore_draught_m IS NOT NULL
                     AND d.aft_draught_m IS NOT NULL
                    THEN '{args.trim_sign}_from_fore_aft'
                    ELSE 'unavailable'
                END AS trim_source,
                count(*) OVER(
                    PARTITION BY
                        cast(u.ship_type AS VARCHAR),
                        cast(u.pseudo_ship_group_id AS VARCHAR),
                        cast(u.trajectory_segment_id AS VARCHAR),
                        try_cast(u.timestamp_utc AS TIMESTAMP)
                ) AS primary_key_count
            FROM unified_lagged u
            LEFT JOIN supplemental_draught d
              ON d.ship_type = lower(cast(u.ship_type AS VARCHAR))
             AND d.pseudo_ship_group_id = cast(u.pseudo_ship_group_id AS VARCHAR)
             AND d.trajectory_segment_id = cast(u.trajectory_segment_id AS VARCHAR)
             AND d.timestamp_utc = try_cast(u.timestamp_utc AS TIMESTAMP)
            LEFT JOIN container_fuel_10min f
              ON lower(cast(u.ship_type AS VARCHAR))='container'
             AND f.pseudo_ship_group_id = cast(u.pseudo_ship_group_id AS VARCHAR)
             AND f.trajectory_segment_id = cast(u.trajectory_segment_id AS VARCHAR)
             AND f.timestamp_utc = try_cast(u.timestamp_utc AS TIMESTAMP)
            LEFT JOIN ship_static_final st
              ON st.ship_type=lower(cast(u.ship_type AS VARCHAR))
             AND st.pseudo_ship_group_id=cast(u.pseudo_ship_group_id AS VARCHAR)
            """
        )

        con.execute(
            f"""
            CREATE OR REPLACE TABLE clean23_physical AS
            SELECT
                *,
                CASE
                    WHEN ship_type IN ('container','bulk','tanker')
                     AND pseudo_ship_group_id IS NOT NULL
                     AND trim(pseudo_ship_group_id)<>''
                     AND trajectory_segment_id IS NOT NULL
                     AND trim(trajectory_segment_id)<>''
                     AND timestamp_utc IS NOT NULL
                     AND primary_key_count=1
                    THEN 1 ELSE 0
                END AS primary_key_valid,

                CASE WHEN speed_kn_raw BETWEEN 0 AND 40
                           AND isfinite(speed_kn_raw)
                     THEN speed_kn_raw END AS speed_kn,
                CASE WHEN fuel_rate_kg_h_raw>=0
                           AND isfinite(fuel_rate_kg_h_raw)
                     THEN fuel_rate_kg_h_raw END AS fuel_rate_kg_h,
                CASE WHEN fuel_kg_10min_raw>=0
                           AND isfinite(fuel_kg_10min_raw)
                           AND NOT(speed_kn_raw>1 AND fuel_kg_10min_raw=0)
                     THEN fuel_kg_10min_raw END AS fuel_kg_10min,
                CASE WHEN fuel_t_10min_raw>=0
                           AND isfinite(fuel_t_10min_raw)
                           AND NOT(speed_kn_raw>1 AND fuel_t_10min_raw=0)
                     THEN fuel_t_10min_raw END AS fuel_t_10min,
                CASE
                    WHEN fuel_t_10min_raw IS NULL
                      OR NOT isfinite(fuel_t_10min_raw)
                      OR fuel_t_10min_raw<0 THEN 0
                    WHEN speed_kn_raw>1 AND fuel_t_10min_raw=0 THEN 0
                    ELSE 1
                END AS fuel_valid_flag_recalculated,
                CASE
                    WHEN speed_kn_raw>1 AND fuel_t_10min_raw=0 THEN 1 ELSE 0
                END AS fuel_zero_underway_flag,
                CASE
                    WHEN fuel_rate_kg_h_raw IS NOT NULL
                     AND fuel_t_10min_raw IS NOT NULL
                     AND isfinite(fuel_rate_kg_h_raw)
                     AND isfinite(fuel_t_10min_raw)
                     AND abs(fuel_rate_kg_h_raw/6000.0-fuel_t_10min_raw)<=1e-10
                    THEN 1 ELSE 0
                END AS fuel_unit_consistency_flag,

                {course_sin_clean} AS course_sin,
                {course_cos_clean} AS course_cos,
                {heading_sin_clean} AS heading_sin,
                {heading_cos_clean} AS heading_cos,

                CASE WHEN mean_draught_m_raw BETWEEN 0 AND 30
                           AND isfinite(mean_draught_m_raw)
                     THEN mean_draught_m_raw END AS mean_draught_m,
                CASE WHEN rudder_deg_raw BETWEEN -45 AND 45
                           AND isfinite(rudder_deg_raw)
                     THEN rudder_deg_raw END AS rudder_deg,
                CASE WHEN distance_nm_10min_raw BETWEEN 0 AND 7.0
                           AND isfinite(distance_nm_10min_raw)
                     THEN distance_nm_10min_raw END AS distance_nm_10min,
                CASE WHEN rel_wind_speed_kn_raw BETWEEN 0 AND 100
                           AND isfinite(rel_wind_speed_kn_raw)
                     THEN rel_wind_speed_kn_raw END AS rel_wind_speed_kn,
                {wind_sin_clean} AS relative_wind_sin,
                {wind_cos_clean} AS relative_wind_cos,
                CASE WHEN wave_height_m_raw BETWEEN 0 AND 20
                           AND isfinite(wave_height_m_raw)
                     THEN wave_height_m_raw END AS wave_height_m,
                CASE WHEN wave_period_s_raw>0 AND wave_period_s_raw<=30
                           AND isfinite(wave_period_s_raw)
                     THEN wave_period_s_raw END AS wave_period_s,
                {wave_sin_clean} AS relative_wave_sin,
                {wave_cos_clean} AS relative_wave_cos,
                CASE
                    WHEN surface_pressure_raw BETWEEN 850 AND 1150
                    THEN surface_pressure_raw*100.0
                    WHEN surface_pressure_raw BETWEEN 85000 AND 115000
                    THEN surface_pressure_raw
                END AS surface_pressure_pa,
                CASE
                    WHEN surface_temperature_raw BETWEEN -5 AND 45
                    THEN surface_temperature_raw
                    WHEN surface_temperature_raw BETWEEN 268.15 AND 318.15
                    THEN surface_temperature_raw-273.15
                END AS surface_temperature_c,

                {float(args.trim_warning_fraction)}*ship_length_m
                    AS trim_warning_limit_m,
                least(
                    {float(args.trim_absolute_cap_m)},
                    {float(args.trim_hard_fraction)}*ship_length_m
                ) AS trim_hard_limit_m,
                CASE
                    WHEN ship_length_m>0
                     AND trim_raw_m IS NOT NULL
                     AND isfinite(trim_raw_m)
                     AND abs(trim_raw_m)<=least(
                         {float(args.trim_absolute_cap_m)},
                         {float(args.trim_hard_fraction)}*ship_length_m
                     )
                    THEN trim_raw_m
                END AS trim_m,
                CASE
                    WHEN ship_length_m IS NULL OR ship_length_m<=0
                        THEN 'invalid_ship_length'
                    WHEN trim_raw_m IS NULL OR NOT isfinite(trim_raw_m)
                        THEN 'missing_trim'
                    WHEN abs(trim_raw_m)<=
                         {float(args.trim_warning_fraction)}*ship_length_m
                        THEN 'normal_le_1pct_length'
                    WHEN abs(trim_raw_m)<=least(
                         {float(args.trim_absolute_cap_m)},
                         {float(args.trim_hard_fraction)}*ship_length_m
                    ) THEN 'high_1_to_2_5pct_retained'
                    ELSE 'invalid_gt_2_5pct_or_absolute_cap'
                END AS trim_physical_status,

                sqrt(power(course_sin_raw,2)+power(course_cos_raw,2))
                    AS course_vector_norm_raw,
                sqrt(power(heading_sin_raw,2)+power(heading_cos_raw,2))
                    AS heading_vector_norm_raw,
                sqrt(power(relative_wind_sin_raw,2)+power(relative_wind_cos_raw,2))
                    AS relative_wind_vector_norm_raw,
                sqrt(power(relative_wave_sin_raw,2)+power(relative_wave_cos_raw,2))
                    AS relative_wave_vector_norm_raw
            FROM clean23_joined
            """
        )

        con.execute(
            """
            CREATE OR REPLACE TABLE clean23_enriched AS
            SELECT
                *,
                rel_wind_speed_kn*speed_kn AS rel_wind_speed_x_speed,
                wave_height_m*speed_kn AS wave_height_x_speed,
                mean_draught_m*speed_kn AS draught_x_speed,
                trim_m*speed_kn AS trim_x_speed,
                mean_draught_m*trim_m AS draught_x_trim,
                CASE
                    WHEN speed_kn>8 THEN 'cruise'
                    WHEN speed_kn>=1 AND speed_kn<=8 THEN 'maneuver'
                    WHEN speed_kn<1 THEN 'anchor_berth'
                    ELSE 'unknown'
                END AS voyage_phase,
                CASE
                    WHEN primary_key_valid=1
                     AND complete_window_flag=1
                     AND speed_kn IS NOT NULL
                     AND fuel_t_10min IS NOT NULL
                    THEN 1 ELSE 0
                END AS base_valid_flag
            FROM clean23_physical
            """
        )

        con.execute(
            """
            CREATE OR REPLACE TABLE clean23_base_previous AS
            SELECT
                *,
                lag(timestamp_utc) OVER(
                    PARTITION BY
                        ship_type,pseudo_ship_group_id,trajectory_segment_id
                    ORDER BY timestamp_utc
                ) AS previous_valid_timestamp
            FROM clean23_enriched
            WHERE base_valid_flag=1
            """
        )
        con.execute(
            """
            CREATE OR REPLACE TABLE clean23_base_blocks AS
            SELECT
                *,
                sum(CASE
                    WHEN previous_valid_timestamp IS NULL THEN 1
                    WHEN date_diff(
                        'minute',previous_valid_timestamp,timestamp_utc
                    )>15 THEN 1
                    ELSE 0
                END) OVER(
                    PARTITION BY
                        ship_type,pseudo_ship_group_id,trajectory_segment_id
                    ORDER BY timestamp_utc
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                ) AS stability_block_id
            FROM clean23_base_previous
            """
        )
        con.execute(
            """
            CREATE OR REPLACE TABLE clean23_speed_index AS
            SELECT
                ship_type,pseudo_ship_group_id,trajectory_segment_id,
                timestamp_utc,
                count(speed_kn) OVER(
                    PARTITION BY ship_type,pseudo_ship_group_id,
                                 trajectory_segment_id,stability_block_id
                    ORDER BY timestamp_utc
                    ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
                ) AS speed_n_3,
                stddev_samp(speed_kn) OVER(
                    PARTITION BY ship_type,pseudo_ship_group_id,
                                 trajectory_segment_id,stability_block_id
                    ORDER BY timestamp_utc
                    ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
                ) AS speed_std_3
            FROM clean23_base_blocks
            """
        )

        if args.require_full_speed_window:
            stable_condition = (
                f"s.speed_n_3=3 AND s.speed_std_3<={float(args.speed_std_threshold)}"
            )
        else:
            stable_condition = (
                f"s.speed_n_3<3 OR s.speed_std_3<={float(args.speed_std_threshold)}"
            )
        complete_case_condition = " AND ".join(
            f"e.{quote_identifier(column)} IS NOT NULL "
            f"AND isfinite(e.{quote_identifier(column)})"
            for column in ["fuel_t_10min"] + FIXED23_PREDICTORS
        )
        static4_complete_condition = " AND ".join(
            f"e.{quote_identifier(column)} IS NOT NULL "
            f"AND isfinite(e.{quote_identifier(column)})"
            for column in STATIC4_PREDICTORS
        )
        fixed27_complete_condition = (
            f"({complete_case_condition}) AND ({static4_complete_condition})"
        )

        con.execute(
            f"""
            CREATE OR REPLACE TABLE clean23_final AS
            SELECT
                e.*,
                s.speed_n_3,
                s.speed_std_3,
                CASE
                    WHEN e.base_valid_flag=1 AND ({stable_condition})
                    THEN 1 ELSE 0
                END AS speed_stable_flag,
                CASE WHEN {complete_case_condition}
                    THEN 1 ELSE 0
                END AS fixed23_complete_case_flag,
                CASE WHEN {static4_complete_condition}
                    THEN 1 ELSE 0
                END AS static4_complete_case_flag,
                CASE WHEN {fixed27_complete_condition}
                    THEN 1 ELSE 0
                END AS fixed27_complete_case_flag,
                CASE
                    WHEN e.base_valid_flag=1
                     AND ({stable_condition})
                     AND ({complete_case_condition})
                    THEN 1 ELSE 0
                END AS final_model_ready_flag,
                CASE
                    WHEN e.base_valid_flag=1
                     AND ({stable_condition})
                     AND ({fixed27_complete_condition})
                    THEN 1 ELSE 0
                END AS final_model_ready_fixed27_flag
            FROM clean23_enriched e
            LEFT JOIN clean23_speed_index s USING(
                ship_type,pseudo_ship_group_id,
                trajectory_segment_id,timestamp_utc
            )
            """
        )
        con.execute("CHECKPOINT")

        v6.copy_query_to_csv(
            con,
            """
            SELECT *
            FROM clean23_final
            WHERE primary_key_count>1
            ORDER BY ship_type,pseudo_ship_group_id,
                     trajectory_segment_id,timestamp_utc
            """,
            output_dir / "01_duplicate_primary_keys.csv",
        )
        v6.copy_query_to_csv(
            con,
            """
            SELECT
                ship_type,
                count(*) AS input_windows,
                count(DISTINCT pseudo_ship_group_id) AS pseudo_ship_count,
                count(DISTINCT trajectory_segment_id) AS segment_count,
                sum(primary_key_valid) AS primary_key_valid_windows,
                sum(CASE WHEN primary_key_valid=1
                          AND complete_window_flag=1 THEN 1 ELSE 0 END)
                    AS complete_windows,
                sum(base_valid_flag) AS base_valid_windows,
                sum(CASE WHEN base_valid_flag=1
                          AND speed_stable_flag=1 THEN 1 ELSE 0 END)
                    AS stable_windows,
                sum(final_model_ready_flag) AS final_fixed23_windows,
                sum(final_model_ready_fixed27_flag) AS final_fixed27_windows,
                sum(CASE WHEN final_model_ready_flag=1
                          AND voyage_phase='cruise' THEN 1 ELSE 0 END)
                    AS final_cruise_windows,
                sum(CASE WHEN final_model_ready_fixed27_flag=1
                          AND voyage_phase='cruise' THEN 1 ELSE 0 END)
                    AS final_fixed27_cruise_windows,
                round(
                    100.0*sum(final_model_ready_flag)/nullif(count(*),0),6
                ) AS final_valid_rate_pct
            FROM clean23_final
            GROUP BY ship_type
            ORDER BY ship_type
            """,
            output_dir / "cleaning_flow_by_ship_type.csv",
        )
        v6.copy_query_to_csv(
            con,
            """
            SELECT
                ship_type,pseudo_ship_group_id,
                first(id_confidence) AS id_confidence,
                count(*) AS input_windows,
                count(DISTINCT trajectory_segment_id) AS segment_count,
                sum(base_valid_flag) AS base_valid_windows,
                sum(CASE WHEN base_valid_flag=1
                          AND speed_stable_flag=1 THEN 1 ELSE 0 END)
                    AS stable_windows,
                sum(final_model_ready_flag) AS final_fixed23_windows,
                sum(final_model_ready_fixed27_flag) AS final_fixed27_windows,
                sum(CASE WHEN final_model_ready_flag=1
                          AND voyage_phase='cruise' THEN 1 ELSE 0 END)
                    AS final_cruise_windows,
                sum(CASE WHEN final_model_ready_fixed27_flag=1
                          AND voyage_phase='cruise' THEN 1 ELSE 0 END)
                    AS final_fixed27_cruise_windows,
                round(
                    100.0*sum(final_model_ready_flag)/nullif(count(*),0),6
                ) AS final_valid_rate_pct,
                min(timestamp_utc) AS first_timestamp,
                max(timestamp_utc) AS last_timestamp
            FROM clean23_final
            GROUP BY ship_type,pseudo_ship_group_id
            ORDER BY ship_type,final_valid_rate_pct,pseudo_ship_group_id
            """,
            output_dir / "validity_by_ship.csv",
        )
        v6.copy_query_to_csv(
            con,
            """
            SELECT
                ship_type,
                mean_draught_source,
                count(*) AS windows,
                sum(CASE WHEN fore_draught_m IS NOT NULL
                          AND aft_draught_m IS NOT NULL THEN 1 ELSE 0 END)
                    AS fore_aft_available_windows,
                sum(CASE WHEN mean_draught_m IS NOT NULL THEN 1 ELSE 0 END)
                    AS retained_mean_draught_windows,
                avg(mean_draught_existing_m) AS mean_existing_draught_m,
                avg((fore_draught_m+aft_draught_m)/2.0)
                    AS mean_fore_aft_average_m,
                avg(mean_draught_m) AS mean_retained_draught_m
            FROM clean23_final
            GROUP BY ship_type,mean_draught_source
            ORDER BY ship_type,windows DESC
            """,
            output_dir / "02_mean_draught_audit.csv",
        )
        v6.copy_query_to_csv(
            con,
            """
            SELECT
                ship_type,trim_physical_status,
                count(*) AS windows,
                sum(CASE WHEN fore_draught_m IS NOT NULL
                          AND aft_draught_m IS NOT NULL THEN 1 ELSE 0 END)
                    AS fore_aft_available_windows,
                sum(CASE WHEN trim_m IS NOT NULL THEN 1 ELSE 0 END)
                    AS retained_trim_windows,
                avg(ship_length_m) AS mean_ship_length_m,
                avg(trim_raw_m) AS mean_trim_raw_m,
                avg(trim_m) AS mean_trim_retained_m,
                min(trim_hard_limit_m) AS min_trim_hard_limit_m,
                max(trim_hard_limit_m) AS max_trim_hard_limit_m
            FROM clean23_final
            GROUP BY ship_type,trim_physical_status
            ORDER BY ship_type,windows DESC
            """,
            output_dir / "02_trim_physical_audit.csv",
        )
        v6.copy_query_to_csv(
            con,
            """
            SELECT
                ship_type,
                avg(course_vector_norm_raw) AS course_norm_mean,
                quantile_cont(course_vector_norm_raw,0.01) AS course_norm_p01,
                avg(heading_vector_norm_raw) AS heading_norm_mean,
                quantile_cont(heading_vector_norm_raw,0.01) AS heading_norm_p01,
                avg(relative_wind_vector_norm_raw) AS wind_norm_mean,
                avg(relative_wave_vector_norm_raw) AS wave_norm_mean,
                sum(CASE WHEN course_sin IS NULL OR course_cos IS NULL
                         THEN 1 ELSE 0 END) AS invalid_course_pairs,
                sum(CASE WHEN heading_sin IS NULL OR heading_cos IS NULL
                         THEN 1 ELSE 0 END) AS invalid_heading_pairs,
                sum(CASE WHEN relative_wind_sin IS NULL
                           OR relative_wind_cos IS NULL
                         THEN 1 ELSE 0 END) AS invalid_wind_pairs,
                sum(CASE WHEN relative_wave_sin IS NULL
                           OR relative_wave_cos IS NULL
                         THEN 1 ELSE 0 END) AS invalid_wave_pairs
            FROM clean23_final
            GROUP BY ship_type
            ORDER BY ship_type
            """,
            output_dir / "03_direction_pair_audit.csv",
        )

        v6.copy_query_to_csv(
            con,
            """
            SELECT
                ship_type,
                fuel_10min_source,
                distance_10min_source,
                count(*) AS windows,
                sum(CASE WHEN fuel_rate_kg_h IS NOT NULL THEN 1 ELSE 0 END)
                    AS valid_fuel_rate_windows,
                sum(CASE WHEN fuel_kg_10min IS NOT NULL THEN 1 ELSE 0 END)
                    AS valid_fuel_kg_windows,
                sum(CASE WHEN fuel_t_10min IS NOT NULL THEN 1 ELSE 0 END)
                    AS valid_fuel_t_windows,
                sum(fuel_zero_underway_flag) AS zero_fuel_underway_windows,
                avg(fuel_unit_consistency_flag)
                    AS fuel_unit_consistency_rate,
                sum(CASE WHEN distance_nm_10min IS NOT NULL THEN 1 ELSE 0 END)
                    AS valid_distance_windows,
                avg(fuel_rate_kg_h) AS mean_fuel_rate_kg_h,
                avg(fuel_kg_10min) AS mean_fuel_kg_10min,
                avg(fuel_t_10min) AS mean_fuel_t_10min,
                sum(fuel_t_10min) AS total_fuel_t,
                avg(distance_nm_10min) AS mean_distance_nm_10min
            FROM clean23_final
            GROUP BY ship_type,fuel_10min_source,distance_10min_source
            ORDER BY ship_type,windows DESC
            """,
            output_dir / "03_fuel_distance_recalculation_audit.csv",
        )
        v6.copy_query_to_csv(
            con,
            """
            SELECT
                ship_type,
                pseudo_ship_group_id,
                trajectory_segment_id,
                timestamp_utc,
                speed_kn_raw,
                fuel_rate_kg_h_existing_raw,
                fuel_rate_kg_h_recalculated_raw,
                fuel_rate_kg_h,
                fuel_kg_10min_recalculated_raw,
                fuel_kg_10min,
                fuel_t_10min_existing_raw,
                fuel_t_10min_recalculated_raw,
                fuel_t_10min,
                fuel_t_10min_recalculated_raw-fuel_t_10min_existing_raw
                    AS recalculated_minus_existing_t,
                shaft_power_kw_fuel_raw,
                sfoc_g_kwh_existing_raw,
                sfoc_g_kwh_recalculated_raw,
                main_engine_rpm_fuel_raw,
                container_fuel_source_rows,
                container_valid_fuel_source_rows,
                fuel_valid_flag_recalculated,
                fuel_zero_underway_flag,
                fuel_unit_consistency_flag,
                fuel_10min_source
            FROM clean23_final
            WHERE ship_type='container'
            ORDER BY pseudo_ship_group_id,trajectory_segment_id,timestamp_utc
            """,
            output_dir / "03_container_fuel_recalculation_detail.csv",
        )
        v6.copy_query_to_csv(
            con,
            """
            SELECT
                pseudo_ship_group_id,
                count(*) AS windows,
                sum(CASE WHEN fuel_rate_kg_h_recalculated_raw IS NOT NULL
                         THEN 1 ELSE 0 END) AS raw_rate_available_windows,
                sum(CASE WHEN fuel_t_10min IS NOT NULL
                         THEN 1 ELSE 0 END) AS valid_fuel_windows,
                sum(fuel_zero_underway_flag) AS zero_fuel_underway_windows,
                avg(fuel_unit_consistency_flag)
                    AS fuel_unit_consistency_rate,
                avg(fuel_rate_kg_h_recalculated_raw)
                    AS mean_fuel_rate_kg_h,
                avg(fuel_kg_10min) AS mean_fuel_kg_10min,
                avg(fuel_t_10min) AS mean_fuel_t_10min,
                sum(fuel_t_10min) AS total_recalculated_fuel_t,
                avg(sfoc_g_kwh_existing_raw) AS mean_existing_sfoc_g_kwh,
                avg(sfoc_g_kwh_recalculated_raw)
                    AS mean_recalculated_sfoc_g_kwh,
                min(timestamp_utc) AS first_timestamp,
                max(timestamp_utc) AS last_timestamp
            FROM clean23_final
            WHERE ship_type='container'
            GROUP BY pseudo_ship_group_id
            ORDER BY pseudo_ship_group_id
            """,
            output_dir / "03_container_fuel_recalculation_by_ship.csv",
        )

        v6.copy_query_to_csv(
            con,
            """
            SELECT
                ship_type,
                pseudo_ship_group_id,
                source_rows,
                design_draught_m,
                design_draught_raw_nonmissing,
                design_draught_valid_rows,
                design_draught_valid_unique,
                design_draught_valid_min,
                design_draught_valid_max,
                CASE WHEN design_draught_m IS NOT NULL
                     THEN 'ship_median_from_raw' ELSE 'unavailable' END
                    AS design_draught_source,
                deadweight_t,
                deadweight_raw_nonmissing,
                deadweight_valid_rows,
                deadweight_valid_unique,
                deadweight_valid_min,
                deadweight_valid_max,
                CASE WHEN deadweight_t IS NOT NULL
                     THEN 'ship_median_from_raw' ELSE 'unavailable' END
                    AS deadweight_source,
                service_speed_kn,
                service_speed_raw_nonmissing,
                service_speed_valid_rows,
                service_speed_valid_unique,
                service_speed_valid_min,
                service_speed_valid_max,
                CASE WHEN service_speed_kn IS NOT NULL
                     THEN 'ship_median_from_raw' ELSE 'unavailable' END
                    AS service_speed_source,
                main_engine_power_kw,
                main_engine_power_raw_nonmissing,
                main_engine_power_valid_rows,
                main_engine_power_valid_unique,
                main_engine_power_valid_min,
                main_engine_power_valid_max,
                CASE WHEN main_engine_power_kw IS NOT NULL
                     THEN 'ship_median_from_raw' ELSE 'unavailable' END
                    AS main_engine_power_source,
                CASE WHEN design_draught_m IS NOT NULL
                       AND deadweight_t IS NOT NULL
                       AND service_speed_kn IS NOT NULL
                       AND main_engine_power_kw IS NOT NULL
                     THEN 1 ELSE 0 END AS static4_complete_flag
            FROM ship_static_final
            ORDER BY ship_type,pseudo_ship_group_id
            """,
            output_dir / "05_ship_static_feature_audit.csv",
        )
        static_missing_union_sql = " UNION ALL ".join(
            f"""
            SELECT
                ship_type,
                '{column}'::VARCHAR AS variable,
                count(*) AS ship_groups,
                sum(CASE WHEN {quote_identifier(column)} IS NOT NULL
                          AND isfinite({quote_identifier(column)})
                         THEN 1 ELSE 0 END) AS complete_ship_groups,
                sum(CASE WHEN {quote_identifier(column)} IS NULL
                          OR NOT isfinite({quote_identifier(column)})
                         THEN 1 ELSE 0 END) AS missing_ship_groups,
                round(100.0*sum(CASE
                    WHEN {quote_identifier(column)} IS NOT NULL
                     AND isfinite({quote_identifier(column)})
                    THEN 1 ELSE 0 END)/nullif(count(*),0),6)
                    AS complete_rate_pct
            FROM ship_static_final
            GROUP BY ship_type
            """
            for column in STATIC4_PREDICTORS
        )
        v6.copy_query_to_csv(
            con,
            f"""
            SELECT * FROM ({static_missing_union_sql}) q
            ORDER BY ship_type,complete_rate_pct,variable
            """,
            output_dir / "05_static4_missingness_by_ship_type.csv",
        )

        # V15.0：输出Fixed23与静态四字段缺失率，明确定位某船型为何未进入最终数据。
        diagnostic_variables = ["fuel_t_10min"] + FIXED23_PREDICTORS
        missing_union_sql = " UNION ALL ".join(
            f"""
            SELECT
                ship_type,
                '{column}'::VARCHAR AS variable,
                count(*) AS total_windows,
                sum(CASE WHEN {quote_identifier(column)} IS NOT NULL
                          AND isfinite({quote_identifier(column)})
                         THEN 1 ELSE 0 END) AS finite_windows,
                sum(CASE WHEN {quote_identifier(column)} IS NULL
                          OR NOT isfinite({quote_identifier(column)})
                         THEN 1 ELSE 0 END) AS missing_or_invalid_windows,
                round(100.0*sum(CASE WHEN {quote_identifier(column)} IS NOT NULL
                                      AND isfinite({quote_identifier(column)})
                                     THEN 1 ELSE 0 END)/nullif(count(*),0),6)
                    AS finite_rate_pct
            FROM clean23_final
            GROUP BY ship_type
            """
            for column in diagnostic_variables
        )
        v6.copy_query_to_csv(
            con,
            f"""
            SELECT * FROM ({missing_union_sql}) q
            ORDER BY ship_type, finite_rate_pct, variable
            """,
            output_dir / "04_fixed23_missingness_by_ship_type.csv",
        )
        v6.copy_query_to_csv(
            con,
            f"""
            SELECT * FROM ({missing_union_sql}) q
            WHERE ship_type='container'
            ORDER BY finite_rate_pct, variable
            """,
            output_dir / "04_container_fixed23_missingness.csv",
        )
        v6.copy_query_to_csv(
            con,
            """
            SELECT
                ship_type,
                count(*) AS input_windows,
                count(DISTINCT pseudo_ship_group_id) AS input_ship_groups,
                sum(base_valid_flag) AS base_valid_windows,
                sum(CASE WHEN base_valid_flag=1 AND speed_stable_flag=1
                         THEN 1 ELSE 0 END) AS stable_windows,
                sum(fixed23_complete_case_flag) AS fixed23_complete_windows,
                sum(static4_complete_case_flag) AS static4_complete_windows,
                sum(fixed27_complete_case_flag) AS fixed27_complete_windows,
                sum(final_model_ready_flag) AS final_model_ready_windows,
                sum(final_model_ready_fixed27_flag)
                    AS final_model_ready_fixed27_windows,
                sum(CASE WHEN final_model_ready_flag=1
                          AND voyage_phase='cruise' THEN 1 ELSE 0 END)
                    AS final_cruise_windows,
                sum(CASE WHEN final_model_ready_fixed27_flag=1
                          AND voyage_phase='cruise' THEN 1 ELSE 0 END)
                    AS final_fixed27_cruise_windows
            FROM clean23_final
            GROUP BY ship_type
            ORDER BY ship_type
            """,
            output_dir / "04_ship_type_retention_diagnosis.csv",
        )

        model_columns = [
            "ship_type", "pseudo_ship_group_id", "trajectory_segment_id",
            "timestamp_utc", "id_confidence", "voyage_phase", "speed_std_3",
            "ship_length_m",
        ] + STATIC4_PREDICTORS + ["fuel_t_10min"] + FIXED23_PREDICTORS
        model_sql = ", ".join(quote_identifier(column) for column in model_columns)
        matrix23_sql = ", ".join(
            quote_identifier(column)
            for column in ["fuel_t_10min"] + FIXED23_PREDICTORS
        )
        fixed27_predictors = FIXED23_PREDICTORS + STATIC4_PREDICTORS
        matrix27_sql = ", ".join(
            quote_identifier(column)
            for column in ["fuel_t_10min"] + fixed27_predictors
        )

        v6.copy_query_to_csv(
            con,
            """
            SELECT *
            FROM clean23_final
            ORDER BY ship_type,pseudo_ship_group_id,
                     trajectory_segment_id,timestamp_utc
            """,
            output_dir / "cleaned_10min_with_flags.csv",
        )
        v6.copy_query_to_csv(
            con,
            f"SELECT {model_sql} FROM clean23_final "
            "WHERE final_model_ready_flag=1 "
            "ORDER BY ship_type,pseudo_ship_group_id,"
            "trajectory_segment_id,timestamp_utc",
            output_dir / "final_fixed23_all_phases.csv",
        )
        v6.copy_query_to_csv(
            con,
            f"SELECT {model_sql} FROM clean23_final "
            "WHERE final_model_ready_flag=1 "
            "AND voyage_phase IN ('cruise','maneuver') "
            "ORDER BY ship_type,pseudo_ship_group_id,"
            "trajectory_segment_id,timestamp_utc",
            output_dir / "final_fixed23_navigation.csv",
        )
        v6.copy_query_to_csv(
            con,
            f"SELECT {model_sql} FROM clean23_final "
            "WHERE final_model_ready_flag=1 AND voyage_phase='cruise' "
            "ORDER BY ship_type,pseudo_ship_group_id,"
            "trajectory_segment_id,timestamp_utc",
            p["clean23_cruise"],
        )
        v6.copy_query_to_csv(
            con,
            f"SELECT {matrix23_sql} FROM clean23_final "
            "WHERE final_model_ready_flag=1 AND voyage_phase='cruise'",
            output_dir / "model_matrix_23_predictors_plus_target.csv",
        )
        v6.copy_query_to_csv(
            con,
            f"SELECT {model_sql} FROM clean23_final "
            "WHERE final_model_ready_fixed27_flag=1 "
            "AND voyage_phase='cruise' "
            "ORDER BY ship_type,pseudo_ship_group_id,"
            "trajectory_segment_id,timestamp_utc",
            output_dir / "final_fixed27_cruise.csv",
        )
        v6.copy_query_to_csv(
            con,
            f"SELECT {matrix27_sql} FROM clean23_final "
            "WHERE final_model_ready_fixed27_flag=1 "
            "AND voyage_phase='cruise'",
            output_dir / "model_matrix_27_predictors_plus_target.csv",
        )

        v6.copy_query_to_csv(
            con,
            """
            SELECT
                ship_type,
                count(*) AS rows,
                count(DISTINCT pseudo_ship_group_id) AS ship_groups
            FROM clean23_final
            WHERE final_model_ready_flag=1 AND voyage_phase='cruise'
            GROUP BY ship_type
            ORDER BY ship_type
            """,
            output_dir / "04_final_cruise_ship_type_distribution.csv",
        )
        v6.copy_query_to_csv(
            con,
            """
            SELECT
                ship_type,
                count(*) AS rows,
                count(DISTINCT pseudo_ship_group_id) AS ship_groups
            FROM clean23_final
            WHERE final_model_ready_fixed27_flag=1 AND voyage_phase='cruise'
            GROUP BY ship_type
            ORDER BY ship_type
            """,
            output_dir / "05_final_fixed27_cruise_ship_type_distribution.csv",
        )

        final_distribution = con.execute(
            """
            SELECT ship_type, count(*) AS rows,
                   count(DISTINCT pseudo_ship_group_id) AS ship_groups
            FROM clean23_final
            WHERE final_model_ready_flag=1 AND voyage_phase='cruise'
            GROUP BY ship_type
            ORDER BY ship_type
            """
        ).fetchall()
        print("[audit] 最终巡航船型分布：")
        for ship_type_value, row_count, ship_count in final_distribution:
            print(
                f"  {ship_type_value}: rows={int(row_count):,}, "
                f"ships={int(ship_count):,}"
            )
        if not any(str(row[0]).lower() == "container" for row in final_distribution):
            print(
                "[warning] 最终巡航数据仍无container。请查看："
                f"{output_dir / '04_container_fixed23_missingness.csv'}"
            )

        fixed27_distribution = con.execute(
            """
            SELECT ship_type, count(*) AS rows,
                   count(DISTINCT pseudo_ship_group_id) AS ship_groups
            FROM clean23_final
            WHERE final_model_ready_fixed27_flag=1 AND voyage_phase='cruise'
            GROUP BY ship_type
            ORDER BY ship_type
            """
        ).fetchall()
        print("[audit] Fixed27最终巡航船型分布：")
        for ship_type_value, row_count, ship_count in fixed27_distribution:
            print(
                f"  {ship_type_value}: rows={int(row_count):,}, "
                f"ships={int(ship_count):,}"
            )
        if not fixed27_distribution:
            print(
                "[warning] Fixed27巡航数据为空。请查看："
                f"{output_dir / '05_static4_missingness_by_ship_type.csv'}"
            )

        pd.DataFrame([
            {"order": 0, "variable": "fuel_t_10min", "role": "target"},
            *[
                {"order": index, "variable": variable, "role": "predictor"}
                for index, variable in enumerate(FIXED23_PREDICTORS, start=1)
            ],
        ]).to_csv(
            output_dir / "04_fixed23_variable_manifest.csv",
            index=False,
            encoding="utf-8-sig",
        )
        pd.DataFrame([
            {"order": 0, "variable": "fuel_t_10min", "role": "target"},
            *[
                {
                    "order": index,
                    "variable": variable,
                    "role": (
                        "static_predictor"
                        if variable in STATIC4_PREDICTORS
                        else "dynamic_predictor"
                    ),
                }
                for index, variable in enumerate(
                    FIXED23_PREDICTORS + STATIC4_PREDICTORS,
                    start=1,
                )
            ],
        ]).to_csv(
            output_dir / "05_fixed27_variable_manifest.csv",
            index=False,
            encoding="utf-8-sig",
        )

        summary = {
            "code_version": CODE_VERSION,
            "identity_method": (
                "Container按精确DeliveryDate；Bulk/Tanker复用V10静态指纹和保守轨迹分配"
            ),
            "trim_definition": {
                "all_ship_types": args.trim_sign,
                "source": "recomputed_from_fore_and_aft_draught",
                "container_existing_trim": "not_available_and_not_required",
                "container_mean_draught": "recomputed_from_fore_and_aft",
                "container_fuel_t_10min": "raw fuel_rate_kg_h / 6000",
                "container_distance_nm_10min": "existing value, else haversine previous position, else speed_kn/6 for a complete 10-minute window",
                "warning_fraction_of_length": args.trim_warning_fraction,
                "hard_fraction_of_length": args.trim_hard_fraction,
                "absolute_cap_m": args.trim_absolute_cap_m,
            },
            "speed_stability": {
                "window_points": 3,
                "level": "unified_10min",
                "threshold_kn": args.speed_std_threshold,
                "require_full_window": args.require_full_speed_window,
            },
            "predictor_count_fixed23": len(FIXED23_PREDICTORS),
            "predictor_count_fixed27": len(FIXED23_PREDICTORS + STATIC4_PREDICTORS),
            "target": "fuel_t_10min",
            "container_recalculation": {
                "fuel_source": "raw container file joined by source_row and exact ship/time identity",
                "fuel_rate_kg_h": "re-extracted from 主机燃油消耗质量流量(kg/h)",
                "fuel_kg_10min": "fuel_rate_kg_h * 10/60 = fuel_rate_kg_h/6",
                "fuel_t_10min": "fuel_rate_kg_h * 10/60/1000 = fuel_rate_kg_h/6000",
                "sfoc_g_kwh_recalculated": "fuel_rate_kg_h*1000/shaft_power_kw, audit only",
                "fuel_validity": "finite, non-negative, and not zero when speed>1 kn",
                "distance_nm_10min": "existing value; otherwise great-circle distance for 1-15 minute gaps; otherwise speed_kn/6 for complete container windows",
            },
            "fixed23_predictors": FIXED23_PREDICTORS,
            "static4_predictors": STATIC4_PREDICTORS,
            "fixed27_predictors": FIXED23_PREDICTORS + STATIC4_PREDICTORS,
            "static_feature_method": {
                "source": "raw files linked to pseudo_ship_group_id",
                "within_ship_aggregation": "median of physically valid values",
                "design_draught_m_range": "0 < x <= 30 m",
                "deadweight_t_range": "0 < x <= 1,000,000 t",
                "service_speed_kn_range": "0 < x <= 40 kn",
                "main_engine_power_kw_range": "0 < x <= 200,000 kW",
                "missing_policy": "retain null in cleaned Fixed23; exclude only from Fixed27 complete-case output",
            },
        }
        (output_dir / "05_clean23_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    finally:
        close_step_db(con, db_work, args.keep_work)
        if not args.keep_work:
            shutil.rmtree(work_dir, ignore_errors=True)

    write_state(args, "clean23", outputs)
    print(f"[done] Fixed23巡航数据：{p['clean23_cruise']}")
    print(f"[done] Fixed27巡航数据：{output_dir / 'final_fixed27_cruise.csv'}")

def run_status(args: argparse.Namespace) -> None:
    ensure_root_dirs(args)
    rows = []
    for step in STEP_ORDER:
        marker = state_path(args, step)
        rows.append({
            "step": step,
            "completed": marker.is_file(),
            "state_file": str(marker),
        })
    frame = pd.DataFrame(rows)
    print(frame.to_string(index=False))
    target = paths(args)["meta"] / "step_status.csv"
    frame.to_csv(target, index=False, encoding="utf-8-sig")
    print(f"\n状态表：{target}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="第二阶段分步低内存运行V15.0。"
    )
    parser.add_argument(
        "--step",
        choices=STEP_ORDER + ["status", "all"],
        default="clean23",
        help=(
            "运行步骤；省略时默认执行clean23。V15.0在clean23下自动强制重建05_clean23，重算集装箱燃油/航程，并加入四项船舶固有特征。"
        ),
    )
    parser.add_argument(
        "--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT
    )
    parser.add_argument(
        "--cleaned-container", type=Path,
        default=DEFAULT_CLEANED_CONTAINER
    )
    parser.add_argument(
        "--cleaned-bulk", type=Path,
        default=DEFAULT_CLEANED_BULK
    )
    parser.add_argument(
        "--cleaned-tanker", type=Path,
        default=DEFAULT_CLEANED_TANKER
    )
    parser.add_argument(
        "--raw-container", type=Path, default=DEFAULT_RAW_CONTAINER
    )
    parser.add_argument(
        "--raw-bulk", type=Path, default=DEFAULT_RAW_BULK
    )
    parser.add_argument(
        "--raw-tanker", type=Path, default=DEFAULT_RAW_TANKER
    )

    parser.add_argument("--chunksize", type=int, default=20000)
    parser.add_argument("--memory-limit", default="1500MB")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--keep-work", action="store_true")

    parser.add_argument(
        "--segment-gap-hours", type=float, default=24.0
    )
    parser.add_argument(
        "--jump-speed-kn", type=float, default=35.0
    )
    parser.add_argument(
        "--expected-container-ships", type=int, default=5
    )
    parser.add_argument(
        "--min-static-match-rate", type=float, default=0.995
    )
    parser.add_argument(
        "--hard-max-fuel-t-5min", type=float, default=2.0
    )
    parser.add_argument(
        "--max-sfoc-g-kwh", type=float, default=300.0
    )
    parser.add_argument(
        "--fuel-power-margin", type=float, default=1.5
    )
    parser.add_argument(
        "--negative-fuel-policy",
        choices=["reject", "absolute"], default="reject"
    )

    parser.add_argument(
        "--duplicate-radius-nm", type=float, default=0.25
    )
    parser.add_argument(
        "--recent-track-hours", type=float, default=72.0
    )
    parser.add_argument(
        "--positionless-gap-hours", type=float, default=6.0
    )
    parser.add_argument(
        "--max-reconnect-days", type=float, default=180.0
    )
    parser.add_argument(
        "--ambiguity-ratio", type=float, default=1.8
    )

    parser.add_argument(
        "--nominal-interval-minutes", type=float, default=5.0
    )
    parser.add_argument(
        "--max-interval-minutes", type=float, default=7.5
    )
    parser.add_argument(
        "--minimum-coverage-minutes", type=float, default=9.5
    )
    parser.add_argument(
        "--maximum-coverage-minutes", type=float, default=10.5
    )
    parser.add_argument(
        "--distance-interval-alignment",
        choices=["ending", "starting"], default="ending"
    )
    parser.add_argument(
        "--wind-direction-convention",
        choices=["from", "toward"], default="from"
    )
    parser.add_argument(
        "--wave-direction-convention",
        choices=["from", "toward"], default="from"
    )
    parser.add_argument(
        "--reference-direction",
        choices=["heading", "course"], default="heading"
    )

    parser.add_argument("--speed-std-threshold", type=float, default=0.50)
    parser.add_argument("--require-full-speed-window", action="store_true")
    parser.add_argument(
        "--trim-sign",
        choices=["aft-minus-fore", "fore-minus-aft"],
        default="aft-minus-fore",
    )
    parser.add_argument(
        "--container-trim-sign",
        choices=["as-is", "invert"],
        default="as-is",
        help=(
            "兼容旧CMD保留；V14.2中Container也由首尾吃水重算，"
            "此参数不再影响trim_m。"
        ),
    )
    parser.add_argument("--trim-warning-fraction", type=float, default=0.01)
    parser.add_argument("--trim-hard-fraction", type=float, default=0.025)
    parser.add_argument("--trim-absolute-cap-m", type=float, default=6.0)
    return parser.parse_args()


def execute_step(
    args: argparse.Namespace,
    step: str,
    v6: types.ModuleType,
    v7: types.ModuleType,
) -> None:
    if step == "init":
        run_init(args, v6)
    elif step == "stage-container":
        run_stage_container(args, v6)
    elif step == "build-container":
        run_build_container(args, v6)
    elif step == "stage-bulk":
        run_stage_bulk_tanker(args, v6, "bulk")
    elif step == "build-bulk":
        run_build_bulk_tanker(args, v6, "bulk")
    elif step == "stage-tanker":
        run_stage_bulk_tanker(args, v6, "tanker")
    elif step == "build-tanker":
        run_build_bulk_tanker(args, v6, "tanker")
    elif step == "feature-container":
        run_feature_ship(args, v7, "container")
    elif step == "feature-bulk":
        run_feature_ship(args, v7, "bulk")
    elif step == "feature-tanker":
        run_feature_ship(args, v7, "tanker")
    elif step == "merge":
        run_merge(args)
    elif step == "clean23":
        run_clean23(args, v6)
    elif step == "status":
        run_status(args)
    else:
        raise ValueError(f"未知步骤：{step}")


def main() -> int:
    args = parse_args()
    # V15.0默认用途是重建clean23，重算集装箱燃油/航程，并加入四项船舶固有特征。
    if args.step == "clean23":
        args.force = True
    ensure_root_dirs(args)

    v6 = load_embedded_module("_stepwise_v6", V6_SOURCE)
    v7 = load_embedded_module("_stepwise_v7", V7_SOURCE)

    if args.step == "all":
        print(
            "正式大数据建议逐条命令运行；"
            "all模式用于小数据或已确认内存充足的环境。"
        )
        for step in STEP_ORDER:
            print("\n" + "=" * 80)
            print(f"执行：{step}")
            print("=" * 80)
            execute_step(args, step, v6, v7)
            gc.collect()
    else:
        execute_step(args, args.step, v6, v7)

    return 0

V6_SOURCE = '\n# -*- coding: utf-8 -*-\n"""\npost_cleaning_pipeline_v5_strict_fuel_shiptype.py\n\n低内存版第二阶段船舶后处理流水线（严格5分钟燃油聚合与船型控制版）。\n\n主要流程：\n1. 以分块方式读取第一阶段清洗CSV，只保留后续需要的列；\n2. 油船/散货船与原始静态字段按原始行顺序分块对齐；\n3. 使用DuckDB进行磁盘外排序、窗口计算和10分钟聚合；\n4. 构造静态指纹、推定船舶组、连续轨迹段和ID置信度；\n5. 严格将吨/5分钟燃油按两个有效5分钟槽求和为吨/10分钟；\n6. 输出船型审计、油耗窗口审计、统一10分钟表、特征和固定划分。\n\n与旧版相比：\n- 不会使用 pandas 一次性读取 50万～120万行的完整文件；\n- 不会同时把三种船型的大表保存在内存；\n- DuckDB可将排序和聚合的中间结果溢写到磁盘。\n\n依赖：\n    pip install pandas numpy duckdb\n\n注意：\n- pseudo_ship_group_id 是推定船舶组，不是真实 IMO/MMSI。\n- 本脚本不会把缺失舵角直接填成0。\n- 插补、标准化和异常阈值的最终拟合应仅在训练集内部完成。\n- bulk/tanker 的 fuel_t_5min 与 me_fo 均按吨/5分钟解释；不会按kg/h二次换算。\n- fuel_t_10min 的单位固定为吨/10分钟；fuel_rate_kg_h仅作审计，禁止入模。\n"""\n\nfrom __future__ import annotations\n\nimport argparse\nimport gc\nimport json\nimport math\nimport re\nimport shutil\nimport sys\nfrom itertools import zip_longest\nfrom pathlib import Path\nfrom typing import Dict, Iterable, List, Optional, Sequence, Tuple\n\nimport numpy as np\nimport pandas as pd\n\ntry:\n    import duckdb\nexcept ImportError as exc:\n    raise SystemExit(\n        "缺少 duckdb。请先运行：\\n"\n        "python -m pip install duckdb"\n    ) from exc\n\n\n# =============================================================================\n# 默认路径和参数\n# =============================================================================\n\nBASE_DIR = Path(r"data")\nDEFAULT_CLEAN_DIR = BASE_DIR / "cleaned"\nDEFAULT_OUTPUT_DIR = BASE_DIR / "post_cleaning_v6_strict_fuel_rebuilt"\n\nDEFAULT_RAW_CONTAINER = BASE_DIR / "continership_with_era5_7fields_v4.csv"\nDEFAULT_RAW_BULK = BASE_DIR / "bulk.csv"\nDEFAULT_RAW_TANKER = BASE_DIR / "tank.csv"\n\nDEFAULT_CHUNK_SIZE = 50_000\nDEFAULT_MEMORY_LIMIT = "4GB"\nDEFAULT_THREADS = 1\nDEFAULT_SEGMENT_GAP_HOURS = 24.0\nDEFAULT_JUMP_SPEED_KN = 35.0\nDEFAULT_GROUP_FOLDS = 5\nDEFAULT_DUPLICATE_RADIUS_NM = 0.25\nDEFAULT_RECENT_TRACK_HOURS = 72.0\nDEFAULT_POSITIONLESS_GAP_HOURS = 6.0\nDEFAULT_MAX_RECONNECT_DAYS = 180.0\nDEFAULT_AMBIGUITY_RATIO = 1.8\nDEFAULT_HARD_MAX_FUEL_T_5MIN = 2.0\nDEFAULT_MAX_SFOC_G_KWH = 300.0\nDEFAULT_FUEL_POWER_MARGIN = 1.5\nDEFAULT_EXPECTED_CONTAINER_SHIPS = 5\n\nALLOWED_SHIP_TYPES = ("container", "bulk", "tanker")\nFUEL_INPUT_UNIT_5MIN = "tonnes_per_5min"\nFUEL_OUTPUT_UNIT_10MIN = "tonnes_per_10min"\n\n\n# =============================================================================\n# 统一字段与别名\n# =============================================================================\n\nALIASES: Dict[str, Sequence[str]] = {\n    "ship_type": ["ship_type", "ShipType", "ship type", "船型"],\n    "timestamp_utc": ["timestamp_utc", "timestamp", "UTC时间(-)", "utc time", "time"],\n    "latitude_deg": ["latitude_deg", "lat", "latitude", "纬度"],\n    "longitude_deg": ["longitude_deg", "lon", "lng", "longitude", "经度"],\n    "speed_kn": ["speed_kn", "speed", "对地航速(kn)", "sog"],\n    "course_deg": ["course_deg", "direct", "course", "航向角(deg)"],\n    "heading_deg": ["heading_deg", "hdg", "heading", "艏向角(deg)"],\n    "mean_draught_m": ["mean_draught_m", "dmp", "mean draught"],\n    "trim_m": ["trim_m", "trim"],\n    "rudder_deg": ["rudder_deg", "rudder", "舵角(deg)"],\n    "wind_speed_kn": ["wind_speed_kn", "wind_s", "wind speed"],\n    "wind_direction_deg": ["wind_direction_deg", "wind_d", "wind direction"],\n    "wave_height_m": ["wave_height_m", "wave_h", "wave height"],\n    "wave_period_s": ["wave_period_s", "wave_p", "wave period"],\n    "wave_direction_deg": ["wave_direction_deg", "wave_d", "wave direction"],\n    "surface_pressure_pa": ["surface_pressure_pa", "surface_p", "surface pressure"],\n    "surface_temperature_c": [\n        "surface_temperature_c", "surface_t", "ssurface_t", "surface temperature"\n    ],\n    "fuel_t_5min": ["fuel_t_5min", "me_fo"],\n    "fuel_t_10min": ["fuel_t_10min"],\n    "fuel_rate_kg_h": [\n        "fuel_rate_kg_h", "主机燃油消耗质量流量(kg/h)", "fuel rate"\n    ],\n    "fuel_source": ["fuel_source"],\n    "fuel_valid_flag": ["fuel_valid_flag"],\n    "valid_record_flag": ["valid_record_flag"],\n    "time_valid_flag": ["time_valid_flag"],\n    "position_valid_flag": ["position_valid_flag"],\n    "speed_valid_flag": ["speed_valid_flag"],\n    "design_draught_m": ["design_draught_m", "Draught", "设计吃水 / M"],\n    "deadweight_t": ["deadweight_t", "Deadweight", "Deadweight Tonnage"],\n    "service_speed_kn": ["service_speed_kn", "ServiceSpeed", "航速 / Kn"],\n    "main_engine_power_kw": [\n        "main_engine_power_kw", "Total KW Main Eng",\n        "Main Propulsion Total Power Output"\n    ],\n    "ship_age_years": ["ship_age_years"],\n    "calendar_year": ["calendar_year"],\n}\n\nRAW_STATIC_ALIASES: Dict[str, Sequence[str]] = {\n    "raw_timestamp": [\n        "timestamp", "timestamp_utc", "UTC时间(-)", "utc time", "time",\n        "datetime", "date_time",\n    ],\n    "delivery_date": ["DeliveryDate", "delivery date", "交付日期"],\n    "deadweight_static": ["Deadweight", "Deadweight Tonnage", "deadweight_t"],\n    "gt_static": ["GT", "Gross Tonnage", "gross tonnage"],\n    "power_static": [\n        "Total KW Main Eng", "Main Propulsion Total Power Output",\n        "main_engine_power_kw"\n    ],\n    "service_speed_static": ["ServiceSpeed", "航速 / Kn", "service_speed_kn"],\n    "breadth_static": ["Breadth", "breadth", "型宽"],\n    "length_bp_static": ["LengthBP", "LBP", "垂线间长"],\n    "draught_static": ["Draught", "设计吃水 / M", "design_draught_m"],\n    "class_static": ["Class", "class", "船级"],\n}\n\n# Bulk/Tanker清洗文件可能删除了部分原始行，不能再按绝对行号拼接。\n# 以下动态字段只用于把清洗后的保留行匹配回原始行，再提取静态字段。\nRAW_ALIGNMENT_ALIASES: Dict[str, Sequence[str]] = {\n    "raw_timestamp": RAW_STATIC_ALIASES["raw_timestamp"],\n    "raw_latitude_deg": ALIASES["latitude_deg"],\n    "raw_longitude_deg": ALIASES["longitude_deg"],\n    "raw_speed_kn": ALIASES["speed_kn"],\n    "raw_course_deg": ALIASES["course_deg"],\n    "raw_mean_draught_m": ALIASES["mean_draught_m"],\n    "raw_fuel_t_5min": ALIASES["fuel_t_5min"],\n}\n\n# 集装箱船身份识别优先直接读取原始文件中的精确交付日期。\n# raw_timestamp 仅用于检查第一阶段清洗文件与原始文件是否仍保持逐行对齐。\nCONTAINER_RAW_ID_ALIASES: Dict[str, Sequence[str]] = {\n    "delivery_date": [\n        "DeliveryDate", "delivery_date", "delivery date", "Date of Delivery",\n        "Delivery Time", "deliverytime", "交付日期", "交船日期", "交船时间",\n        "建造日期",\n    ],\n    "raw_timestamp": [\n        "timestamp", "timestamp_utc", "UTC时间(-)", "utc time", "time",\n    ],\n}\n\nSTANDARD_CLEAN_COLUMNS = [\n    "ship_type", "timestamp_utc", "latitude_deg", "longitude_deg",\n    "speed_kn", "course_deg", "heading_deg", "mean_draught_m",\n    "trim_m", "rudder_deg", "wind_speed_kn", "wind_direction_deg",\n    "wave_height_m", "wave_period_s", "wave_direction_deg",\n    "surface_pressure_pa", "surface_temperature_c",\n    "fuel_t_5min", "fuel_t_10min", "fuel_rate_kg_h", "fuel_source",\n    "fuel_valid_flag", "valid_record_flag", "time_valid_flag",\n    "position_valid_flag", "speed_valid_flag",\n    "design_draught_m", "deadweight_t", "service_speed_kn",\n    "main_engine_power_kw", "ship_age_years", "calendar_year",\n]\n\nMODEL_COLUMNS = [\n    "pseudo_ship_group_id", "trajectory_segment_id", "id_confidence",\n    "ship_type", "timestamp_utc", "latitude_deg", "longitude_deg",\n    "distance_nm_10min", "design_draught_m", "deadweight_t",\n    "service_speed_kn", "main_engine_power_kw", "ship_age_years",\n    "calendar_year", "speed_kn", "course_deg", "heading_deg",\n    "mean_draught_m", "trim_m", "rudder_deg", "wind_speed_kn",\n    "wind_direction_deg", "wave_height_m", "wave_period_s",\n    "wave_direction_deg", "surface_pressure_pa", "surface_temperature_c",\n    "fuel_t_10min", "fuel_rate_kg_h", "fuel_source",\n    "resample_count_10min", "coverage_ratio_10min",\n    "complete_window_flag", "time_valid_flag", "position_valid_flag",\n    "speed_valid_flag", "fuel_valid_flag", "valid_record_flag",\n    "wave_missing_flag", "surface_temperature_missing_flag",\n    "rudder_missing_flag",\n]\n\nVALIDATION_COLUMNS = [\n    "timestamp_utc", "latitude_deg", "longitude_deg", "speed_kn",\n    "course_deg", "mean_draught_m", "trim_m", "rudder_deg",\n    "wind_speed_kn", "wind_direction_deg", "wave_height_m",\n    "wave_period_s", "wave_direction_deg", "surface_pressure_pa",\n    "surface_temperature_c",\n]\n\n\n# =============================================================================\n# 通用工具\n# =============================================================================\n\ndef normalize_name(value: object) -> str:\n    """规范化字段名，兼容空格、逗号、括号、斜线、下划线和中英文标点。"""\n    import unicodedata\n\n    text = unicodedata.normalize("NFKC", str(value))\n    text = text.strip().lower().replace("\\ufeff", "")\n    return re.sub(r"[^0-9a-z\\u4e00-\\u9fff]+", "", text)\n\n\ndef choose_column(columns: Sequence[str], aliases: Sequence[str]) -> Optional[str]:\n    mapping = {normalize_name(column): column for column in columns}\n    for alias in aliases:\n        match = mapping.get(normalize_name(alias))\n        if match is not None:\n            return match\n    return None\n\n\ndef detect_encoding(path: Path) -> str:\n    for encoding in ["utf-8-sig", "utf-8", "gb18030", "gbk"]:\n        try:\n            with path.open("r", encoding=encoding) as handle:\n                handle.read(65_536)\n            return encoding\n        except UnicodeDecodeError:\n            continue\n    return "utf-8"\n\n\ndef read_header(path: Path) -> List[str]:\n    return list(\n        pd.read_csv(\n            path,\n            nrows=0,\n            encoding=detect_encoding(path),\n            low_memory=True,\n        ).columns\n    )\n\n\ndef sql_path(path: Path) -> str:\n    return str(path).replace("\\\\", "/").replace("\'", "\'\'")\n\n\ndef quote_sql_text(value: str) -> str:\n    return value.replace("\'", "\'\'")\n\n\ndef discover_cleaned_file(clean_dir: Path, ship_type: str) -> Path:\n    if ship_type == "container":\n        tokens = ["container", "continer", "continership"]\n    elif ship_type == "bulk":\n        tokens = ["bulk"]\n    else:\n        tokens = ["tanker", "tank"]\n\n    excluded = [\n        "10min", "unified", "report", "pseudo", "feature", "candidate",\n        "missing", "validation", "stage"\n    ]\n    candidates: List[Path] = []\n    for path in clean_dir.glob("*.csv"):\n        name = path.name.lower()\n        if any(token in name for token in excluded):\n            continue\n        if any(token in name for token in tokens):\n            candidates.append(path)\n\n    if not candidates:\n        raise FileNotFoundError(\n            f"在 {clean_dir} 中没有找到 {ship_type} 第一阶段清洗文件。"\n        )\n    candidates.sort(key=lambda item: item.stat().st_mtime, reverse=True)\n    return candidates[0]\n\n\ndef build_column_plan(\n    path: Path,\n    aliases: Dict[str, Sequence[str]],\n    targets: Sequence[str],\n) -> Tuple[List[str], Dict[str, str]]:\n    header = read_header(path)\n    usecols: List[str] = []\n    rename_map: Dict[str, str] = {}\n    for target in targets:\n        source = choose_column(header, aliases.get(target, [target]))\n        if source is not None:\n            usecols.append(source)\n            rename_map[source] = target\n    return list(dict.fromkeys(usecols)), rename_map\n\n\ndef field_mapping_rows(\n    path: Path,\n    aliases: Dict[str, Sequence[str]],\n    targets: Sequence[str],\n    file_role: str,\n    expected_ship_type: Optional[str] = None,\n) -> pd.DataFrame:\n    header = read_header(path)\n    rows: List[Dict[str, object]] = []\n    for target in targets:\n        source = choose_column(header, aliases.get(target, [target]))\n        rows.append(\n            {\n                "file_role": file_role,\n                "expected_ship_type": expected_ship_type,\n                "source_file": str(path),\n                "canonical_field": target,\n                "recognized_column": source if source is not None else "[not found]",\n                "match_method": "normalized_exact" if source is not None else "not_found",\n                "normalized_recognized_column": (\n                    normalize_name(source) if source is not None else ""\n                ),\n            }\n        )\n    return pd.DataFrame(rows)\n\n\ndef standardize_clean_chunk(\n    chunk: pd.DataFrame,\n    rename_map: Dict[str, str],\n    ship_type: str,\n) -> pd.DataFrame:\n    if ship_type not in ALLOWED_SHIP_TYPES:\n        raise ValueError(\n            f"无法识别船型：{ship_type}；允许值为{ALLOWED_SHIP_TYPES}。"\n        )\n\n    data = chunk.rename(columns=rename_map).copy()\n    for column in STANDARD_CLEAN_COLUMNS:\n        if column not in data.columns:\n            data[column] = np.nan\n\n    # 船型由当前处理分支强制写入，不信任原CSV中可能不一致的标签。\n    data["ship_type"] = ship_type\n    if not data["ship_type"].eq(ship_type).all():\n        raise RuntimeError(f"{ship_type}标准化阶段船型写入失败。")\n    return data[STANDARD_CLEAN_COLUMNS]\n\n\ndef update_validation_accumulator(\n    accumulator: Dict[str, object],\n    chunk: pd.DataFrame,\n) -> None:\n    accumulator["row_count"] += len(chunk)\n    timestamp = pd.to_datetime(chunk["timestamp_utc"], errors="coerce", utc=True)\n    accumulator["timestamp_valid"] += int(timestamp.notna().sum())\n    for column in VALIDATION_COLUMNS:\n        accumulator["missing"][column] += int(chunk[column].isna().sum())\n\n\ndef validation_rows(\n    accumulator: Dict[str, object],\n    ship_type: str,\n    source_file: Path,\n) -> pd.DataFrame:\n    total = int(accumulator["row_count"])\n    rows: List[Dict[str, object]] = [\n        {\n            "ship_type": ship_type,\n            "source_file": str(source_file),\n            "metric": "row_count",\n            "value": total,\n        },\n        {\n            "ship_type": ship_type,\n            "source_file": str(source_file),\n            "metric": "timestamp_valid_rate",\n            "value": (\n                float(accumulator["timestamp_valid"]) / total if total else np.nan\n            ),\n        },\n    ]\n    for column in VALIDATION_COLUMNS:\n        missing = int(accumulator["missing"][column])\n        rows.extend(\n            [\n                {\n                    "ship_type": ship_type,\n                    "source_file": str(source_file),\n                    "metric": f"column_exists:{column}",\n                    "value": 1,\n                },\n                {\n                    "ship_type": ship_type,\n                    "source_file": str(source_file),\n                    "metric": f"missing_rate:{column}",\n                    "value": missing / total if total else np.nan,\n                },\n            ]\n        )\n    return pd.DataFrame(rows)\n\n\ndef new_validation_accumulator() -> Dict[str, object]:\n    return {\n        "row_count": 0,\n        "timestamp_valid": 0,\n        "missing": {column: 0 for column in VALIDATION_COLUMNS},\n    }\n\n\ndef append_csv(chunk: pd.DataFrame, path: Path, first: bool) -> None:\n    path.parent.mkdir(parents=True, exist_ok=True)\n    chunk.to_csv(\n        path,\n        mode="w" if first else "a",\n        header=first,\n        index=False,\n        encoding="utf-8-sig" if first else "utf-8",\n    )\n\n\n# =============================================================================\n# 分块生成标准化暂存文件\n# =============================================================================\n\ndef stage_container(\n    cleaned_path: Path,\n    raw_path: Path,\n    stage_path: Path,\n    chunk_size: int,\n) -> pd.DataFrame:\n    """分块对齐集装箱船清洗数据与原始精确交付日期。"""\n    clean_usecols, clean_rename = build_column_plan(\n        cleaned_path, ALIASES, STANDARD_CLEAN_COLUMNS\n    )\n    raw_targets = ["delivery_date", "raw_timestamp"]\n    raw_usecols, raw_rename = build_column_plan(\n        raw_path, CONTAINER_RAW_ID_ALIASES, raw_targets\n    )\n\n    if not clean_usecols:\n        raise ValueError(f"无法识别集装箱船清洗文件字段：{cleaned_path}")\n    if (\n        "fuel_t_10min" not in clean_rename.values()\n        and "fuel_rate_kg_h" not in clean_rename.values()\n    ):\n        raise ValueError(\n            "集装箱船清洗文件未识别到fuel_t_10min或fuel_rate_kg_h；"\n            "不能生成统一10分钟燃油目标。"\n        )\n    if "delivery_date" not in raw_rename.values():\n        raise ValueError(\n            f"无法在原始集装箱船文件中识别精确交付日期：{raw_path}。"\n            "请确认字段名属于 DeliveryDate、delivery_date、交付日期或交船日期。"\n        )\n\n    clean_reader = pd.read_csv(\n        cleaned_path,\n        usecols=clean_usecols,\n        chunksize=chunk_size,\n        encoding=detect_encoding(cleaned_path),\n        low_memory=True,\n    )\n    raw_reader = pd.read_csv(\n        raw_path,\n        usecols=raw_usecols,\n        chunksize=chunk_size,\n        encoding=detect_encoding(raw_path),\n        low_memory=True,\n    )\n\n    accumulator = new_validation_accumulator()\n    first = True\n    source_row = 0\n    timestamp_compared = 0\n    timestamp_mismatch = 0\n\n    static_output_columns = list(RAW_STATIC_ALIASES.keys())\n\n    for block_no, pair in enumerate(\n        zip_longest(clean_reader, raw_reader, fillvalue=None), start=1\n    ):\n        clean_chunk, raw_chunk = pair\n        if clean_chunk is None or raw_chunk is None:\n            raise RuntimeError(\n                "集装箱船清洗文件与原始文件的分块数量不同，无法按行对齐。"\n            )\n        if len(clean_chunk) != len(raw_chunk):\n            raise RuntimeError(\n                f"集装箱船第{block_no}块行数不一致："\n                f"cleaned={len(clean_chunk)}, raw={len(raw_chunk)}。"\n            )\n\n        data = standardize_clean_chunk(clean_chunk, clean_rename, "container")\n        raw_data = raw_chunk.rename(columns=raw_rename).reset_index(drop=True)\n        data = data.reset_index(drop=True)\n\n        data.insert(0, "source_row", np.arange(source_row, source_row + len(data)))\n        source_row += len(data)\n\n        data["delivery_date"] = raw_data["delivery_date"].to_numpy()\n\n        # 为 typed_stage_sql(include_static=True) 补齐静态列。\n        data["deadweight_static"] = data["deadweight_t"]\n        data["gt_static"] = np.nan\n        data["power_static"] = data["main_engine_power_kw"]\n        data["service_speed_static"] = data["service_speed_kn"]\n        data["breadth_static"] = np.nan\n        data["length_bp_static"] = np.nan\n        data["draught_static"] = data["design_draught_m"]\n        data["class_static"] = np.nan\n\n        # 若原始文件含时间戳，则检查逐行合并是否仍然可靠。\n        if "raw_timestamp" in raw_data.columns:\n            clean_time = pd.to_datetime(data["timestamp_utc"], errors="coerce", utc=True)\n            raw_time = pd.to_datetime(raw_data["raw_timestamp"], errors="coerce", utc=True)\n            comparable = clean_time.notna() & raw_time.notna()\n            if comparable.any():\n                delta_seconds = (\n                    clean_time[comparable] - raw_time[comparable]\n                ).abs().dt.total_seconds()\n                timestamp_compared += int(comparable.sum())\n                timestamp_mismatch += int((delta_seconds > 1.0).sum())\n\n        update_validation_accumulator(accumulator, data)\n        append_csv(data, stage_path, first)\n        first = False\n        print(f"[container] 暂存块 {block_no}，累计 {source_row:,} 行")\n        del clean_chunk, raw_chunk, raw_data, data\n        gc.collect()\n\n    if timestamp_compared:\n        mismatch_rate = timestamp_mismatch / timestamp_compared\n        print(\n            f"[container] 原始/清洗时间戳逐行核验：比较 {timestamp_compared:,} 行，"\n            f"不一致 {timestamp_mismatch:,} 行（{mismatch_rate:.6%}）"\n        )\n        if mismatch_rate > 0.001:\n            raise RuntimeError(\n                "集装箱船原始文件与第一阶段清洗文件的行顺序可能不一致；"\n                "时间戳不一致率超过0.1%，不能安全地按行合并交付日期。"\n            )\n\n    return validation_rows(accumulator, "container", cleaned_path)\n\ndef stage_bulk_or_tanker(\n    cleaned_path: Path,\n    raw_path: Path,\n    stage_path: Path,\n    ship_type: str,\n    chunk_size: int,\n    memory_limit: str,\n    threads: int,\n    min_static_match_rate: float,\n) -> pd.DataFrame:\n    """按业务键将清洗行匹配回原始行并拼接静态字段。\n\n    旧版使用zip_longest逐块按行拼接。只要第一阶段清洗文件删除过\n    任意原始行，后续行就会整体错位；即使前20块长度都等于50000，\n    也不能证明这些行仍一一对应。\n\n    V6先分别分块暂存清洗表和原始表，再用DuckDB执行分层一对一匹配：\n    1. 时间戳 + 纬度 + 经度；\n    2. 时间戳 + 航速 + 航向 + 5分钟油耗；\n    3. 剩余记录中双方均唯一的时间戳。\n\n    每一层都使用键内出现序号，确保同一原始行最多匹配一次。\n    未达到最低匹配率时立即停止，避免静态字段错配污染船舶身份。\n    """\n    if ship_type not in ("bulk", "tanker"):\n        raise ValueError(\n            "stage_bulk_or_tanker仅允许bulk或tanker，"\n            f"实际收到：{ship_type}"\n        )\n    if not 0.0 <= min_static_match_rate <= 1.0:\n        raise ValueError("min_static_match_rate必须位于0和1之间。")\n\n    clean_usecols, clean_rename = build_column_plan(\n        cleaned_path, ALIASES, STANDARD_CLEAN_COLUMNS\n    )\n\n    raw_targets = list(dict.fromkeys(\n        list(RAW_STATIC_ALIASES.keys())\n        + list(RAW_ALIGNMENT_ALIASES.keys())\n    ))\n    raw_aliases = {\n        **RAW_STATIC_ALIASES,\n        **RAW_ALIGNMENT_ALIASES,\n    }\n    raw_usecols, raw_rename = build_column_plan(\n        raw_path, raw_aliases, raw_targets\n    )\n\n    if not clean_usecols:\n        raise ValueError(f"无法识别第一阶段清洗字段：{cleaned_path}")\n    if "fuel_t_5min" not in clean_rename.values():\n        raise ValueError(\n            f"{ship_type}清洗文件未识别到fuel_t_5min或me_fo。"\n            "本流水线明确要求该字段单位为吨/5分钟；"\n            "为避免把kg/h误当区间油耗，程序停止。"\n        )\n    if "raw_timestamp" not in raw_rename.values():\n        raise ValueError(\n            f"{ship_type}原始文件未识别到时间戳。"\n            "当清洗文件与原始文件行数不同，必须使用时间和动态字段"\n            "进行键控匹配，不能继续按行拼接。"\n        )\n\n    work_dir = stage_path.parent / f"_{ship_type}_alignment"\n    work_dir.mkdir(parents=True, exist_ok=True)\n    clean_only_path = work_dir / f"{ship_type}_clean_only.csv"\n    raw_only_path = work_dir / f"{ship_type}_raw_only.csv"\n    alignment_db = work_dir / f"{ship_type}_alignment.duckdb"\n    alignment_temp = work_dir / "duckdb_temp"\n    alignment_temp.mkdir(parents=True, exist_ok=True)\n\n    accumulator = new_validation_accumulator()\n\n    # A. 分块暂存清洗文件。source_row是清洗表内稳定顺序，不再冒充原始行号。\n    clean_first = True\n    clean_row = 0\n    clean_reader = pd.read_csv(\n        cleaned_path,\n        usecols=clean_usecols,\n        chunksize=chunk_size,\n        encoding=detect_encoding(cleaned_path),\n        low_memory=True,\n    )\n    for block_no, clean_chunk in enumerate(clean_reader, start=1):\n        clean_data = standardize_clean_chunk(\n            clean_chunk,\n            clean_rename,\n            ship_type,\n        ).reset_index(drop=True)\n        clean_data.insert(\n            0,\n            "clean_source_row",\n            np.arange(clean_row, clean_row + len(clean_data)),\n        )\n        clean_row += len(clean_data)\n        update_validation_accumulator(accumulator, clean_data)\n        append_csv(clean_data, clean_only_path, clean_first)\n        clean_first = False\n        print(\n            f"[{ship_type}] 清洗表暂存块 {block_no}，"\n            f"累计 {clean_row:,} 行"\n        )\n        del clean_chunk, clean_data\n        gc.collect()\n\n    # B. 分块暂存原始表。只保留静态字段和用于匹配的动态字段。\n    raw_first = True\n    raw_row = 0\n    raw_reader = pd.read_csv(\n        raw_path,\n        usecols=raw_usecols,\n        chunksize=chunk_size,\n        encoding=detect_encoding(raw_path),\n        low_memory=True,\n    )\n    for block_no, raw_chunk in enumerate(raw_reader, start=1):\n        raw_data = raw_chunk.rename(columns=raw_rename).reset_index(drop=True)\n        for target in raw_targets:\n            if target not in raw_data.columns:\n                raw_data[target] = np.nan\n        raw_data = raw_data[raw_targets]\n        raw_data.insert(\n            0,\n            "raw_source_row",\n            np.arange(raw_row, raw_row + len(raw_data)),\n        )\n        raw_row += len(raw_data)\n        append_csv(raw_data, raw_only_path, raw_first)\n        raw_first = False\n        print(\n            f"[{ship_type}] 原始表暂存块 {block_no}，"\n            f"累计 {raw_row:,} 行"\n        )\n        del raw_chunk, raw_data\n        gc.collect()\n\n    con = duckdb.connect(str(alignment_db))\n    con.execute(f"SET memory_limit=\'{quote_sql_text(memory_limit)}\'")\n    con.execute(f"SET threads={max(1, int(threads))}")\n    con.execute(f"SET temp_directory=\'{sql_path(alignment_temp)}\'")\n    con.execute("SET preserve_insertion_order=false")\n\n    try:\n        clean_path_sql = sql_path(clean_only_path)\n        raw_path_sql = sql_path(raw_only_path)\n\n        # V10低内存修正：完整清洗宽表保留为CSV扫描视图，\n        # DuckDB只物化身份匹配需要的窄列。\n        con.execute(\n            f"""\n            CREATE OR REPLACE VIEW clean_source_all AS\n            SELECT\n                try_cast(clean_source_row AS BIGINT) AS clean_source_row,\n                * EXCLUDE (clean_source_row)\n            FROM read_csv_auto(\n                \'{clean_path_sql}\', header=true, all_varchar=true,\n                sample_size=10000, ignore_errors=false\n            )\n            """\n        )\n        con.execute(\n            """\n            CREATE OR REPLACE TABLE clean_align AS\n            SELECT\n                clean_source_row,\n                try_cast(timestamp_utc AS TIMESTAMP) AS match_ts,\n                try_cast(latitude_deg AS DOUBLE) AS match_lat,\n                try_cast(longitude_deg AS DOUBLE) AS match_lon,\n                try_cast(speed_kn AS DOUBLE) AS match_speed,\n                try_cast(course_deg AS DOUBLE) AS match_course,\n                try_cast(mean_draught_m AS DOUBLE) AS match_draught,\n                try_cast(fuel_t_5min AS DOUBLE) AS match_fuel\n            FROM clean_source_all\n            """\n        )\n        con.execute("CHECKPOINT")\n        con.execute(\n            f"""\n            CREATE OR REPLACE TABLE raw_align AS\n            SELECT\n                try_cast(raw_source_row AS BIGINT) AS raw_source_row,\n                try_cast(raw_timestamp AS TIMESTAMP) AS match_ts,\n                try_cast(raw_latitude_deg AS DOUBLE) AS match_lat,\n                try_cast(raw_longitude_deg AS DOUBLE) AS match_lon,\n                try_cast(raw_speed_kn AS DOUBLE) AS match_speed,\n                try_cast(raw_course_deg AS DOUBLE) AS match_course,\n                try_cast(raw_mean_draught_m AS DOUBLE) AS match_draught,\n                try_cast(raw_fuel_t_5min AS DOUBLE) AS match_fuel,\n                try_cast(delivery_date AS DATE) AS delivery_date,\n                try_cast(deadweight_static AS DOUBLE) AS deadweight_static,\n                try_cast(gt_static AS DOUBLE) AS gt_static,\n                try_cast(power_static AS DOUBLE) AS power_static,\n                try_cast(service_speed_static AS DOUBLE) AS service_speed_static,\n                try_cast(breadth_static AS DOUBLE) AS breadth_static,\n                try_cast(length_bp_static AS DOUBLE) AS length_bp_static,\n                try_cast(draught_static AS DOUBLE) AS draught_static,\n                nullif(trim(cast(class_static AS VARCHAR)), \'\') AS class_static\n            FROM read_csv_auto(\n                \'{raw_path_sql}\', header=true, all_varchar=true,\n                sample_size=100000, ignore_errors=false\n            )\n            """\n        )\n\n        # 第1层：时间戳+位置。6位小数可吸收CSV浮点文本细微差异。\n        con.execute(\n            """\n            CREATE OR REPLACE TABLE strong_clean AS\n            SELECT *,\n                concat(\n                    cast(match_ts AS VARCHAR), \'|\',\n                    cast(round(match_lat, 6) AS VARCHAR), \'|\',\n                    cast(round(match_lon, 6) AS VARCHAR)\n                ) AS match_key,\n                row_number() OVER (\n                    PARTITION BY match_ts, round(match_lat, 6), round(match_lon, 6)\n                    ORDER BY clean_source_row\n                ) AS key_occurrence\n            FROM clean_align\n            WHERE match_ts IS NOT NULL\n              AND match_lat IS NOT NULL\n              AND match_lon IS NOT NULL\n            """\n        )\n        con.execute(\n            """\n            CREATE OR REPLACE TABLE strong_raw AS\n            SELECT *,\n                concat(\n                    cast(match_ts AS VARCHAR), \'|\',\n                    cast(round(match_lat, 6) AS VARCHAR), \'|\',\n                    cast(round(match_lon, 6) AS VARCHAR)\n                ) AS match_key,\n                row_number() OVER (\n                    PARTITION BY match_ts, round(match_lat, 6), round(match_lon, 6)\n                    ORDER BY raw_source_row\n                ) AS key_occurrence\n            FROM raw_align\n            WHERE match_ts IS NOT NULL\n              AND match_lat IS NOT NULL\n              AND match_lon IS NOT NULL\n            """\n        )\n        con.execute(\n            """\n            CREATE OR REPLACE TABLE match_strong AS\n            SELECT c.clean_source_row, r.raw_source_row,\n                   \'timestamp_position\'::VARCHAR AS alignment_method\n            FROM strong_clean c\n            JOIN strong_raw r USING (match_key, key_occurrence)\n            """\n        )\n\n        # 第2层：对剩余记录使用时间戳+航速+航向+燃油。\n        con.execute(\n            """\n            CREATE OR REPLACE TABLE remaining_clean_1 AS\n            SELECT c.*\n            FROM clean_align c\n            LEFT JOIN match_strong m USING (clean_source_row)\n            WHERE m.clean_source_row IS NULL\n            """\n        )\n        con.execute(\n            """\n            CREATE OR REPLACE TABLE remaining_raw_1 AS\n            SELECT r.*\n            FROM raw_align r\n            LEFT JOIN match_strong m USING (raw_source_row)\n            WHERE m.raw_source_row IS NULL\n            """\n        )\n        con.execute(\n            """\n            CREATE OR REPLACE TABLE medium_clean AS\n            SELECT *,\n                concat(\n                    cast(match_ts AS VARCHAR), \'|\',\n                    cast(round(match_speed, 4) AS VARCHAR), \'|\',\n                    cast(round(mod(mod(match_course, 360.0) + 360.0, 360.0), 3) AS VARCHAR), \'|\',\n                    cast(round(match_fuel, 9) AS VARCHAR)\n                ) AS match_key,\n                row_number() OVER (\n                    PARTITION BY match_ts, round(match_speed, 4),\n                        round(mod(mod(match_course, 360.0) + 360.0, 360.0), 3),\n                        round(match_fuel, 9)\n                    ORDER BY clean_source_row\n                ) AS key_occurrence\n            FROM remaining_clean_1\n            WHERE match_ts IS NOT NULL\n              AND match_speed IS NOT NULL\n              AND match_course IS NOT NULL\n              AND match_fuel IS NOT NULL\n            """\n        )\n        con.execute(\n            """\n            CREATE OR REPLACE TABLE medium_raw AS\n            SELECT *,\n                concat(\n                    cast(match_ts AS VARCHAR), \'|\',\n                    cast(round(match_speed, 4) AS VARCHAR), \'|\',\n                    cast(round(mod(mod(match_course, 360.0) + 360.0, 360.0), 3) AS VARCHAR), \'|\',\n                    cast(round(match_fuel, 9) AS VARCHAR)\n                ) AS match_key,\n                row_number() OVER (\n                    PARTITION BY match_ts, round(match_speed, 4),\n                        round(mod(mod(match_course, 360.0) + 360.0, 360.0), 3),\n                        round(match_fuel, 9)\n                    ORDER BY raw_source_row\n                ) AS key_occurrence\n            FROM remaining_raw_1\n            WHERE match_ts IS NOT NULL\n              AND match_speed IS NOT NULL\n              AND match_course IS NOT NULL\n              AND match_fuel IS NOT NULL\n            """\n        )\n        con.execute(\n            """\n            CREATE OR REPLACE TABLE match_medium AS\n            SELECT c.clean_source_row, r.raw_source_row,\n                   \'timestamp_speed_course_fuel\'::VARCHAR AS alignment_method\n            FROM medium_clean c\n            JOIN medium_raw r USING (match_key, key_occurrence)\n            """\n        )\n\n        # 第3层：只接受剩余数据中双方都唯一的时间戳，避免同一时刻多船歧义。\n        con.execute(\n            """\n            CREATE OR REPLACE TABLE matched_12 AS\n            SELECT * FROM match_strong\n            UNION ALL\n            SELECT * FROM match_medium\n            """\n        )\n        con.execute(\n            """\n            CREATE OR REPLACE TABLE remaining_clean_2 AS\n            SELECT c.*\n            FROM clean_align c\n            LEFT JOIN matched_12 m USING (clean_source_row)\n            WHERE m.clean_source_row IS NULL\n            """\n        )\n        con.execute(\n            """\n            CREATE OR REPLACE TABLE remaining_raw_2 AS\n            SELECT r.*\n            FROM raw_align r\n            LEFT JOIN matched_12 m USING (raw_source_row)\n            WHERE m.raw_source_row IS NULL\n            """\n        )\n        con.execute(\n            """\n            CREATE OR REPLACE TABLE unique_clean_ts AS\n            SELECT match_ts, min(clean_source_row) AS clean_source_row\n            FROM remaining_clean_2\n            WHERE match_ts IS NOT NULL\n            GROUP BY match_ts\n            HAVING count(*) = 1\n            """\n        )\n        con.execute(\n            """\n            CREATE OR REPLACE TABLE unique_raw_ts AS\n            SELECT match_ts, min(raw_source_row) AS raw_source_row\n            FROM remaining_raw_2\n            WHERE match_ts IS NOT NULL\n            GROUP BY match_ts\n            HAVING count(*) = 1\n            """\n        )\n        con.execute(\n            """\n            CREATE OR REPLACE TABLE match_unique_time AS\n            SELECT c.clean_source_row, r.raw_source_row,\n                   \'unique_timestamp\'::VARCHAR AS alignment_method\n            FROM unique_clean_ts c\n            JOIN unique_raw_ts r USING (match_ts)\n            """\n        )\n        con.execute(\n            """\n            CREATE OR REPLACE TABLE all_matches AS\n            SELECT * FROM matched_12\n            UNION ALL\n            SELECT * FROM match_unique_time\n            """\n        )\n\n        duplicate_clean_matches = int(con.execute(\n            """\n            SELECT count(*) FROM (\n                SELECT clean_source_row\n                FROM all_matches\n                GROUP BY clean_source_row\n                HAVING count(*) > 1\n            )\n            """\n        ).fetchone()[0])\n        duplicate_raw_matches = int(con.execute(\n            """\n            SELECT count(*) FROM (\n                SELECT raw_source_row\n                FROM all_matches\n                GROUP BY raw_source_row\n                HAVING count(*) > 1\n            )\n            """\n        ).fetchone()[0])\n        if duplicate_clean_matches or duplicate_raw_matches:\n            raise RuntimeError(\n                f"{ship_type}键控对齐出现非一对一匹配："\n                f"clean重复={duplicate_clean_matches}, "\n                f"raw重复={duplicate_raw_matches}"\n            )\n\n        stats = con.execute(\n            """\n            SELECT\n                (SELECT count(*) FROM clean_align) AS clean_rows,\n                (SELECT count(*) FROM raw_align) AS raw_rows,\n                (SELECT count(*) FROM all_matches) AS matched_rows,\n                (SELECT count(*) FROM match_strong) AS strong_matches,\n                (SELECT count(*) FROM match_medium) AS medium_matches,\n                (SELECT count(*) FROM match_unique_time) AS unique_time_matches,\n                (SELECT count(*) FROM clean_align WHERE match_ts IS NULL)\n                    AS invalid_clean_timestamp_rows\n            """\n        ).fetchone()\n        clean_rows = int(stats[0])\n        raw_rows = int(stats[1])\n        matched_rows = int(stats[2])\n        match_rate = matched_rows / clean_rows if clean_rows else 0.0\n\n        alignment_report = pd.DataFrame([\n            {\n                "ship_type": ship_type,\n                "cleaned_path": str(cleaned_path),\n                "raw_path": str(raw_path),\n                "cleaned_rows": clean_rows,\n                "raw_rows": raw_rows,\n                "raw_minus_cleaned_rows": raw_rows - clean_rows,\n                "matched_rows": matched_rows,\n                "unmatched_cleaned_rows": clean_rows - matched_rows,\n                "match_rate": match_rate,\n                "timestamp_position_matches": int(stats[3]),\n                "timestamp_speed_course_fuel_matches": int(stats[4]),\n                "unique_timestamp_matches": int(stats[5]),\n                "invalid_clean_timestamp_rows": int(stats[6]),\n                "minimum_required_match_rate": min_static_match_rate,\n                "alignment_safe": int(match_rate >= min_static_match_rate),\n            }\n        ])\n        alignment_report.to_csv(\n            stage_path.parent.parent / f"{ship_type}_static_alignment_report.csv",\n            index=False,\n            encoding="utf-8-sig",\n        )\n\n        copy_query_to_csv(\n            con,\n            """\n            SELECT\n                c.clean_source_row AS source_row,\n                c.ship_type,\n                c.timestamp_utc,\n                c.latitude_deg,\n                c.longitude_deg,\n                c.speed_kn,\n                c.course_deg,\n                c.heading_deg,\n                c.mean_draught_m,\n                c.trim_m,\n                c.rudder_deg,\n                c.wind_speed_kn,\n                c.wind_direction_deg,\n                c.wave_height_m,\n                c.wave_period_s,\n                c.wave_direction_deg,\n                c.surface_pressure_pa,\n                c.surface_temperature_c,\n                c.fuel_t_5min,\n                c.fuel_t_10min,\n                c.fuel_rate_kg_h,\n                c.fuel_source,\n                c.fuel_valid_flag,\n                c.valid_record_flag,\n                c.time_valid_flag,\n                c.position_valid_flag,\n                c.speed_valid_flag,\n                c.design_draught_m,\n                c.deadweight_t,\n                c.service_speed_kn,\n                c.main_engine_power_kw,\n                c.ship_age_years,\n                c.calendar_year,\n                r.delivery_date,\n                coalesce(r.deadweight_static, try_cast(c.deadweight_t AS DOUBLE))\n                    AS deadweight_static,\n                r.gt_static,\n                coalesce(r.power_static, try_cast(c.main_engine_power_kw AS DOUBLE))\n                    AS power_static,\n                coalesce(r.service_speed_static, try_cast(c.service_speed_kn AS DOUBLE))\n                    AS service_speed_static,\n                r.breadth_static,\n                r.length_bp_static,\n                coalesce(r.draught_static, try_cast(c.design_draught_m AS DOUBLE))\n                    AS draught_static,\n                r.class_static,\n                m.raw_source_row AS matched_raw_source_row,\n                coalesce(m.alignment_method, \'unmatched\') AS static_alignment_method\n            FROM clean_source_all c\n            LEFT JOIN all_matches m USING (clean_source_row)\n            LEFT JOIN raw_align r USING (raw_source_row)\n            ORDER BY c.clean_source_row\n            """,\n            stage_path,\n        )\n\n        if match_rate < min_static_match_rate:\n            unmatched_path = stage_path.parent.parent / f"{ship_type}_unmatched_cleaned_rows.csv"\n            copy_query_to_csv(\n                con,\n                """\n                SELECT c.*\n                FROM clean_align c\n                LEFT JOIN all_matches m USING (clean_source_row)\n                WHERE m.clean_source_row IS NULL\n                ORDER BY clean_source_row\n                """,\n                unmatched_path,\n            )\n            raise RuntimeError(\n                f"{ship_type}清洗行与原始行的静态字段匹配率仅"\n                f"{match_rate:.6%}，低于最低要求{min_static_match_rate:.6%}。"\n                f"已输出：{unmatched_path}。"\n                "禁止退回按行拼接，因为那会把静态参数分配给错误记录。"\n            )\n\n        print(\n            f"[{ship_type}] 键控静态字段对齐完成："\n            f"cleaned={clean_rows:,}, raw={raw_rows:,}, "\n            f"matched={matched_rows:,} ({match_rate:.6%})"\n        )\n\n    finally:\n        con.close()\n\n    return validation_rows(\n        accumulator,\n        ship_type,\n        cleaned_path,\n    )\n\n\n# =============================================================================\n# DuckDB SQL构造\n# =============================================================================\n\ndef dnum(column: str) -> str:\n    return f\'try_cast("{column}" AS DOUBLE)\'\n\n\ndef dint(column: str, default: int = 1) -> str:\n    return f\'coalesce(try_cast("{column}" AS INTEGER), {default})\'\n\n\ndef dtext(column: str) -> str:\n    return f\'nullif(trim(cast("{column}" AS VARCHAR)), \\\'\\\')\'\n\n\ndef typed_stage_sql(stage_file: Path, ship_type: str, include_static: bool) -> str:\n    p = sql_path(stage_file)\n    columns = [\n        "source_row",\n        f"\'{quote_sql_text(ship_type)}\'::VARCHAR AS ship_type",\n        \'try_cast("timestamp_utc" AS TIMESTAMP) AS timestamp_utc\',\n        f\'{dnum("latitude_deg")} AS latitude_deg\',\n        f\'{dnum("longitude_deg")} AS longitude_deg\',\n        f\'{dnum("speed_kn")} AS speed_kn\',\n        f\'mod(mod({dnum("course_deg")}, 360.0) + 360.0, 360.0) AS course_deg\',\n        f\'mod(mod({dnum("heading_deg")}, 360.0) + 360.0, 360.0) AS heading_deg\',\n        f\'{dnum("mean_draught_m")} AS mean_draught_m\',\n        f\'{dnum("trim_m")} AS trim_m\',\n        f\'{dnum("rudder_deg")} AS rudder_deg\',\n        f\'{dnum("wind_speed_kn")} AS wind_speed_kn\',\n        f\'mod(mod({dnum("wind_direction_deg")}, 360.0) + 360.0, 360.0) AS wind_direction_deg\',\n        f\'{dnum("wave_height_m")} AS wave_height_m\',\n        f\'{dnum("wave_period_s")} AS wave_period_s\',\n        f\'mod(mod({dnum("wave_direction_deg")}, 360.0) + 360.0, 360.0) AS wave_direction_deg\',\n        f\'{dnum("surface_pressure_pa")} AS surface_pressure_pa\',\n        f\'{dnum("surface_temperature_c")} AS surface_temperature_c\',\n        f\'{dnum("fuel_t_5min")} AS fuel_t_5min\',\n        f\'{dnum("fuel_t_10min")} AS fuel_t_10min_input\',\n        f\'{dnum("fuel_rate_kg_h")} AS fuel_rate_kg_h_input\',\n        f\'{dtext("fuel_source")} AS fuel_source_input\',\n        f\'{dint("fuel_valid_flag", 0)} AS fuel_valid_flag_input\',\n        f\'{dint("valid_record_flag", 1)} AS valid_record_flag\',\n        f\'{dint("time_valid_flag", 1)} AS time_valid_flag\',\n        f\'{dint("position_valid_flag", 1)} AS position_valid_flag\',\n        f\'{dint("speed_valid_flag", 1)} AS speed_valid_flag\',\n        f\'{dnum("design_draught_m")} AS design_draught_m\',\n        f\'{dnum("deadweight_t")} AS deadweight_t\',\n        f\'{dnum("service_speed_kn")} AS service_speed_kn\',\n        f\'{dnum("main_engine_power_kw")} AS main_engine_power_kw\',\n        f\'{dnum("ship_age_years")} AS ship_age_years\',\n        \'try_cast("calendar_year" AS INTEGER) AS calendar_year_input\',\n    ]\n\n    if include_static:\n        columns.extend(\n            [\n                \'try_cast("delivery_date" AS DATE) AS delivery_date\',\n                f\'{dnum("deadweight_static")} AS deadweight_static\',\n                f\'{dnum("gt_static")} AS gt_static\',\n                f\'{dnum("power_static")} AS power_static\',\n                f\'{dnum("service_speed_static")} AS service_speed_static\',\n                f\'{dnum("breadth_static")} AS breadth_static\',\n                f\'{dnum("length_bp_static")} AS length_bp_static\',\n                f\'{dnum("draught_static")} AS draught_static\',\n                f\'{dtext("class_static")} AS class_static\',\n            ]\n        )\n\n    return f"""\n        SELECT\n            {\', \'.join(columns)}\n        FROM read_csv_auto(\n            \'{p}\',\n            header=true,\n            sample_size=100000,\n            ignore_errors=false,\n            all_varchar=true\n        )\n    """\n\n\ndef haversine_sql(prev_lat: str, prev_lon: str, lat: str, lon: str) -> str:\n    return f"""\n        CASE\n            WHEN {prev_lat} IS NULL OR {prev_lon} IS NULL\n              OR {lat} IS NULL OR {lon} IS NULL\n            THEN NULL\n            ELSE 2.0 * 3440.065 * asin(\n                sqrt(\n                    least(\n                        1.0,\n                        greatest(\n                            0.0,\n                            pow(sin(radians({lat} - {prev_lat}) / 2.0), 2)\n                            + cos(radians({prev_lat})) * cos(radians({lat}))\n                              * pow(sin(radians({lon} - {prev_lon}) / 2.0), 2)\n                        )\n                    )\n                )\n            )\n        END\n    """\n\n\n\ndef haversine_nm_scalar(lat1: float, lon1: float, lat2: float, lon2: float) -> float:\n    """Calculate great-circle distance in nautical miles for scalar coordinates."""\n    if any(pd.isna(value) for value in (lat1, lon1, lat2, lon2)):\n        return math.nan\n    lat1r = math.radians(float(lat1))\n    lon1r = math.radians(float(lon1))\n    lat2r = math.radians(float(lat2))\n    lon2r = math.radians(float(lon2))\n    dlat = lat2r - lat1r\n    dlon = lon2r - lon1r\n    a = (\n        math.sin(dlat / 2.0) ** 2\n        + math.cos(lat1r) * math.cos(lat2r) * math.sin(dlon / 2.0) ** 2\n    )\n    a = min(1.0, max(0.0, a))\n    return 2.0 * 3440.065 * math.asin(math.sqrt(a))\n\n\ndef _valid_position(lat: object, lon: object) -> bool:\n    if pd.isna(lat) or pd.isna(lon):\n        return False\n    return -90.0 <= float(lat) <= 90.0 and -180.0 <= float(lon) <= 180.0\n\n\ndef _cluster_same_timestamp(\n    frame: pd.DataFrame,\n    duplicate_radius_nm: float,\n) -> List[Dict[str, object]]:\n    """Cluster records at the same timestamp that are effectively duplicate positions."""\n    clusters: List[Dict[str, object]] = []\n    for row in frame.itertuples(index=False):\n        lat = row.latitude_deg\n        lon = row.longitude_deg\n        placed = False\n        if _valid_position(lat, lon):\n            for cluster in clusters:\n                if not cluster["has_position"]:\n                    continue\n                distance = haversine_nm_scalar(\n                    float(cluster["lat"]), float(cluster["lon"]), float(lat), float(lon)\n                )\n                if distance <= duplicate_radius_nm:\n                    cluster["source_rows"].append(int(row.source_row))\n                    cluster["lat_values"].append(float(lat))\n                    cluster["lon_values"].append(float(lon))\n                    cluster["lat"] = float(np.median(cluster["lat_values"]))\n                    cluster["lon"] = float(np.median(cluster["lon_values"]))\n                    placed = True\n                    break\n        if not placed:\n            clusters.append(\n                {\n                    "source_rows": [int(row.source_row)],\n                    "lat_values": [float(lat)] if _valid_position(lat, lon) else [],\n                    "lon_values": [float(lon)] if _valid_position(lat, lon) else [],\n                    "lat": float(lat) if _valid_position(lat, lon) else math.nan,\n                    "lon": float(lon) if _valid_position(lat, lon) else math.nan,\n                    "has_position": _valid_position(lat, lon),\n                }\n            )\n    return clusters\n\n\ndef assign_tracks_for_fingerprint(\n    observations: pd.DataFrame,\n    prefix: str,\n    fingerprint: str,\n    max_speed_kn: float,\n    duplicate_radius_nm: float,\n    recent_track_hours: float,\n    positionless_gap_hours: float,\n    max_reconnect_days: float,\n    ambiguity_ratio: float,\n) -> pd.DataFrame:\n    """\n    Conservative identity assignment.\n\n    One stable static fingerprint is treated as one inferred ship unless there is\n    separately verified concurrent-track evidence. Time gaps and position jumps\n    are NOT allowed to create new ships here; they are handled later as trajectory\n    segment boundaries and quality flags.\n\n    This deliberately avoids the previous failure mode in which ordinary gaps,\n    missing positions, or isolated coordinate errors generated tens of thousands\n    of pseudo ships.\n    """\n    columns = [\n        "source_row", "pseudo_ship_group_id", "pseudo_ship_id", "track_no",\n        "assignment_confidence", "assignment_reason",\n        "matched_implied_speed_kn", "match_candidate_count",\n    ]\n    if observations.empty:\n        return pd.DataFrame(columns=columns)\n\n    data = observations.copy()\n    data["timestamp_utc"] = pd.to_datetime(data["timestamp_utc"], errors="coerce")\n    data = data.sort_values(\n        ["timestamp_utc", "source_row"], na_position="last", kind="mergesort"\n    ).reset_index(drop=True)\n\n    fp_short = str(fingerprint)[:8].upper()\n    pseudo_id = f"{prefix}_F{fp_short}_P0001"\n\n    output_rows: List[Dict[str, object]] = []\n    prev_time = None\n    prev_lat = math.nan\n    prev_lon = math.nan\n    prev_has_position = False\n\n    for index, row in enumerate(data.itertuples(index=False)):\n        timestamp = row.timestamp_utc\n        lat = row.latitude_deg\n        lon = row.longitude_deg\n        has_position = _valid_position(lat, lon)\n        implied_speed = math.nan\n\n        if pd.isna(timestamp):\n            reason = "invalid_timestamp_same_fingerprint"\n            confidence = "low"\n        elif prev_time is None:\n            reason = "static_fingerprint_anchor"\n            confidence = "high" if has_position else "medium"\n        else:\n            dt_hours = (pd.Timestamp(timestamp) - pd.Timestamp(prev_time)).total_seconds() / 3600.0\n            if dt_hours == 0:\n                if has_position and prev_has_position:\n                    distance = haversine_nm_scalar(prev_lat, prev_lon, float(lat), float(lon))\n                    if distance <= duplicate_radius_nm:\n                        reason = "duplicate_same_time_position"\n                        confidence = "high"\n                    else:\n                        reason = "simultaneous_far_position_conflict"\n                        confidence = "low"\n                else:\n                    reason = "same_time_missing_position"\n                    confidence = "low"\n            elif dt_hours < 0:\n                reason = "time_order_conflict_same_fingerprint"\n                confidence = "low"\n            elif has_position and prev_has_position:\n                distance = haversine_nm_scalar(prev_lat, prev_lon, float(lat), float(lon))\n                implied_speed = distance / dt_hours\n                if implied_speed > max_speed_kn:\n                    reason = "position_jump_same_fingerprint"\n                    confidence = "low"\n                elif dt_hours > max_reconnect_days * 24.0:\n                    reason = "very_long_gap_same_fingerprint"\n                    confidence = "medium"\n                elif dt_hours > recent_track_hours:\n                    reason = "long_gap_same_fingerprint"\n                    confidence = "medium"\n                else:\n                    reason = "matched_by_static_fingerprint"\n                    confidence = "high"\n            else:\n                if dt_hours <= positionless_gap_hours:\n                    reason = "short_gap_missing_position"\n                    confidence = "medium"\n                else:\n                    reason = "long_gap_missing_position"\n                    confidence = "low"\n\n        output_rows.append(\n            {\n                "source_row": int(row.source_row),\n                "pseudo_ship_group_id": pseudo_id,\n                "pseudo_ship_id": pseudo_id,\n                "track_no": 1,\n                "assignment_confidence": confidence,\n                "assignment_reason": reason,\n                "matched_implied_speed_kn": implied_speed,\n                "match_candidate_count": 1,\n            }\n        )\n\n        if pd.notna(timestamp):\n            prev_time = pd.Timestamp(timestamp)\n            if has_position:\n                prev_lat = float(lat)\n                prev_lon = float(lon)\n                prev_has_position = True\n\n    return pd.DataFrame(output_rows, columns=columns)\n\n\ndef build_track_assignment_file(\n    con: "duckdb.DuckDBPyConnection",\n    base_table: str,\n    ship_type: str,\n    output_path: Path,\n    max_speed_kn: float,\n    duplicate_radius_nm: float,\n    recent_track_hours: float,\n    positionless_gap_hours: float,\n    max_reconnect_days: float,\n    ambiguity_ratio: float,\n) -> None:\n    prefix = "BULK" if ship_type == "bulk" else "TANKER"\n    fingerprints = con.execute(\n        f"SELECT ship_fingerprint, count(*) AS n FROM {base_table} "\n        "GROUP BY ship_fingerprint ORDER BY n DESC"\n    ).fetchall()\n\n    first = True\n    total = 0\n    for index, (fingerprint, count) in enumerate(fingerprints, start=1):\n        observations = con.execute(\n            f"""\n            SELECT source_row, timestamp_utc, latitude_deg, longitude_deg\n            FROM {base_table}\n            WHERE ship_fingerprint = ?\n            ORDER BY timestamp_utc NULLS LAST, source_row\n            """,\n            [fingerprint],\n        ).fetchdf()\n        assignments = assign_tracks_for_fingerprint(\n            observations=observations,\n            prefix=prefix,\n            fingerprint=str(fingerprint),\n            max_speed_kn=max_speed_kn,\n            duplicate_radius_nm=duplicate_radius_nm,\n            recent_track_hours=recent_track_hours,\n            positionless_gap_hours=positionless_gap_hours,\n            max_reconnect_days=max_reconnect_days,\n            ambiguity_ratio=ambiguity_ratio,\n        )\n        append_csv(assignments, output_path, first)\n        first = False\n        total += len(assignments)\n        ship_count = assignments["pseudo_ship_group_id"].nunique() if len(assignments) else 0\n        print(\n            f"[{ship_type}] 指纹 {index}/{len(fingerprints)}："\n            f"{int(count):,} 行，识别 {ship_count:,} 个推定船舶组，累计 {total:,} 行"\n        )\n        del observations, assignments\n        gc.collect()\n\n\ndef create_bulk_tanker_tables(\n    con: "duckdb.DuckDBPyConnection",\n    ship_type: str,\n    stage_file: Path,\n    assignment_file: Path,\n    gap_hours: float,\n    jump_speed_kn: float,\n    duplicate_radius_nm: float,\n    recent_track_hours: float,\n    positionless_gap_hours: float,\n    max_reconnect_days: float,\n    ambiguity_ratio: float,\n    hard_max_fuel_t_5min: float,\n    max_sfoc_g_kwh: float,\n    fuel_power_margin: float,\n    negative_fuel_policy: str,\n) -> None:\n    if ship_type not in ("bulk", "tanker"):\n        raise ValueError(\n            "create_bulk_tanker_tables仅允许bulk或tanker，"\n            f"实际收到：{ship_type}"\n        )\n\n    prefix = "BULK" if ship_type == "bulk" else "TANKER"\n    base = f"{ship_type}_base"\n    identified = f"{ship_type}_identified"\n    model10 = f"{ship_type}_model_10min"\n    effective_fuel = "abs(fuel_t_5min)" if negative_fuel_policy == "absolute" else "fuel_t_5min"\n\n    print(f"[{ship_type}] DuckDB：建立静态指纹和油耗质量基础表")\n    con.execute(f"CREATE OR REPLACE TABLE {base}_raw AS {typed_stage_sql(stage_file, ship_type, True)}")\n    con.execute(\n        f"""\n        CREATE OR REPLACE TABLE {base} AS\n        WITH fp AS (\n            SELECT *,\n                concat_ws(\'|\',\n                    coalesce(strftime(delivery_date, \'%Y-%m-%d\'), \'NA\'),\n                    coalesce(cast(round(deadweight_static / 100.0) * 100.0 AS VARCHAR), \'NA\'),\n                    coalesce(cast(round(gt_static / 100.0) * 100.0 AS VARCHAR), \'NA\'),\n                    coalesce(cast(round(power_static / 10.0) * 10.0 AS VARCHAR), \'NA\'),\n                    coalesce(cast(round(service_speed_static, 1) AS VARCHAR), \'NA\'),\n                    coalesce(cast(round(breadth_static, 1) AS VARCHAR), \'NA\'),\n                    coalesce(cast(round(length_bp_static, 1) AS VARCHAR), \'NA\'),\n                    coalesce(cast(round(draught_static, 1) AS VARCHAR), \'NA\'),\n                    coalesce(upper(trim(class_static)), \'NA\')\n                ) AS ship_fingerprint_text,\n                (\n                    cast(delivery_date IS NOT NULL AS INTEGER)\n                    + cast(deadweight_static IS NOT NULL AS INTEGER)\n                    + cast(gt_static IS NOT NULL AS INTEGER)\n                    + cast(power_static IS NOT NULL AS INTEGER)\n                    + cast(service_speed_static IS NOT NULL AS INTEGER)\n                    + cast(breadth_static IS NOT NULL AS INTEGER)\n                    + cast(length_bp_static IS NOT NULL AS INTEGER)\n                    + cast(draught_static IS NOT NULL AS INTEGER)\n                    + cast(class_static IS NOT NULL AS INTEGER)\n                ) / 9.0 AS fingerprint_completeness\n            FROM {base}_raw\n        ), prepared AS (\n            SELECT fp.*,\n                substr(md5(ship_fingerprint_text), 1, 14) AS ship_fingerprint,\n                {effective_fuel} AS fuel_t_5min_clean,\n                {effective_fuel} * 12000.0 AS fuel_rate_kg_h_raw,\n                coalesce(main_engine_power_kw, power_static)\n                    * {float(max_sfoc_g_kwh)} / 1000.0 * {float(fuel_power_margin)}\n                    AS fuel_power_limit_kg_h\n            FROM fp\n        )\n        SELECT *,\n            fuel_t_5min = 0 AS fuel_zero_flag,\n            fuel_t_5min < 0 AS fuel_negative_flag,\n            fuel_t_5min = 0 AND speed_kn > 3 AS fuel_sailing_zero_flag,\n            fuel_t_5min_clean > {float(hard_max_fuel_t_5min)} AS fuel_hard_limit_exceed_flag,\n            CASE\n                WHEN fuel_power_limit_kg_h IS NULL THEN false\n                ELSE fuel_rate_kg_h_raw > fuel_power_limit_kg_h\n            END AS fuel_power_limit_exceed_flag,\n            CASE\n                WHEN fuel_t_5min_clean > {float(hard_max_fuel_t_5min)} THEN true\n                WHEN fuel_power_limit_kg_h IS NOT NULL\n                 AND fuel_rate_kg_h_raw > fuel_power_limit_kg_h THEN true\n                ELSE false\n            END AS fuel_extreme_flag,\n            CASE\n                WHEN fuel_t_5min IS NULL THEN 0\n                WHEN \'{negative_fuel_policy}\' = \'reject\' AND fuel_t_5min < 0 THEN 0\n                WHEN fuel_t_5min = 0 AND speed_kn > 3 THEN 0\n                WHEN fuel_t_5min_clean > {float(hard_max_fuel_t_5min)} THEN 0\n                WHEN fuel_power_limit_kg_h IS NOT NULL\n                 AND fuel_rate_kg_h_raw > fuel_power_limit_kg_h THEN 0\n                ELSE 1\n            END AS fuel_raw_valid_flag\n        FROM prepared\n        """\n    )\n    con.execute(f"DROP TABLE {base}_raw")\n\n    print(f"[{ship_type}] 保守身份识别：每个稳定静态指纹先作为一个推定船舶；位置跳跃仅切轨迹段")\n    build_track_assignment_file(\n        con=con,\n        base_table=base,\n        ship_type=ship_type,\n        output_path=assignment_file,\n        max_speed_kn=jump_speed_kn,\n        duplicate_radius_nm=duplicate_radius_nm,\n        recent_track_hours=recent_track_hours,\n        positionless_gap_hours=positionless_gap_hours,\n        max_reconnect_days=max_reconnect_days,\n        ambiguity_ratio=ambiguity_ratio,\n    )\n\n    assignment_path = sql_path(assignment_file)\n    con.execute(\n        f"""\n        CREATE OR REPLACE TABLE {ship_type}_assignments AS\n        SELECT\n            try_cast(source_row AS BIGINT) AS source_row,\n            cast(pseudo_ship_group_id AS VARCHAR) AS pseudo_ship_group_id,\n            cast(pseudo_ship_id AS VARCHAR) AS pseudo_ship_id,\n            try_cast(track_no AS INTEGER) AS track_no,\n            cast(assignment_confidence AS VARCHAR) AS assignment_confidence,\n            cast(assignment_reason AS VARCHAR) AS assignment_reason,\n            try_cast(matched_implied_speed_kn AS DOUBLE) AS matched_implied_speed_kn,\n            try_cast(match_candidate_count AS INTEGER) AS match_candidate_count\n        FROM read_csv_auto(\n            \'{assignment_path}\', header=true, all_varchar=true, sample_size=100000\n        )\n        """\n    )\n\n    print(f"[{ship_type}] DuckDB：划分连续轨迹段并计算ID置信度")\n    distance_expr = haversine_sql("prev_lat", "prev_lon", "latitude_deg", "longitude_deg")\n    con.execute(\n        f"""\n        CREATE OR REPLACE TABLE {identified} AS\n        WITH joined AS (\n            SELECT b.*, a.pseudo_ship_group_id, a.pseudo_ship_id, a.track_no,\n                a.assignment_confidence, a.assignment_reason,\n                a.matched_implied_speed_kn, a.match_candidate_count\n            FROM {base} b\n            LEFT JOIN {ship_type}_assignments a USING (source_row)\n        ), lagged AS (\n            SELECT *,\n                lag(timestamp_utc) OVER (\n                    PARTITION BY pseudo_ship_group_id\n                    ORDER BY timestamp_utc NULLS LAST, source_row\n                ) AS prev_time,\n                lag(latitude_deg) OVER (\n                    PARTITION BY pseudo_ship_group_id\n                    ORDER BY timestamp_utc NULLS LAST, source_row\n                ) AS prev_lat,\n                lag(longitude_deg) OVER (\n                    PARTITION BY pseudo_ship_group_id\n                    ORDER BY timestamp_utc NULLS LAST, source_row\n                ) AS prev_lon,\n                row_number() OVER (\n                    PARTITION BY pseudo_ship_group_id\n                    ORDER BY timestamp_utc NULLS LAST, source_row\n                ) AS group_rn\n            FROM joined\n        ), movement AS (\n            SELECT *,\n                date_diff(\'minute\', prev_time, timestamp_utc) AS time_gap_min,\n                {distance_expr} AS distance_nm_interval\n            FROM lagged\n        ), jumps AS (\n            SELECT *,\n                CASE WHEN time_gap_min > 0\n                    THEN distance_nm_interval / (time_gap_min / 60.0)\n                    ELSE NULL END AS implied_speed_kn\n            FROM movement\n        ), boundaries AS (\n            SELECT *,\n                CASE WHEN implied_speed_kn > {float(jump_speed_kn)} THEN 1 ELSE 0 END\n                    AS position_jump_flag,\n                CASE\n                    WHEN group_rn = 1 THEN 1\n                    WHEN timestamp_utc IS NULL THEN 1\n                    WHEN time_gap_min = 0\n                     AND coalesce(distance_nm_interval, 0.0) <= {float(duplicate_radius_nm)} THEN 0\n                    WHEN time_gap_min <= 0 THEN 1\n                    WHEN time_gap_min > {float(gap_hours) * 60.0} THEN 1\n                    WHEN implied_speed_kn > {float(jump_speed_kn)} THEN 1\n                    ELSE 0\n                END AS new_segment_flag\n            FROM jumps\n        ), numbered AS (\n            SELECT *,\n                sum(new_segment_flag) OVER (\n                    PARTITION BY pseudo_ship_group_id\n                    ORDER BY timestamp_utc NULLS LAST, source_row\n                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW\n                ) AS segment_no\n            FROM boundaries\n        ), group_stats AS (\n            SELECT pseudo_ship_group_id,\n                count(*) AS group_record_count,\n                median(fingerprint_completeness) AS group_fingerprint_completeness,\n                avg(CASE WHEN assignment_confidence = \'low\' THEN 1.0 ELSE 0.0 END)\n                    AS low_assignment_rate,\n                avg(CASE WHEN position_jump_flag = 1 THEN 1.0 ELSE 0.0 END)\n                    AS position_jump_rate\n            FROM numbered\n            GROUP BY pseudo_ship_group_id\n        )\n        SELECT n.*,\n            pseudo_ship_group_id || \'_S\' || lpad(cast(segment_no AS VARCHAR), 4, \'0\')\n                AS trajectory_segment_id,\n            CASE\n                WHEN g.group_record_count >= 20\n                 AND g.group_fingerprint_completeness >= 0.80\n                 AND g.low_assignment_rate <= 0.01\n                 AND g.position_jump_rate <= 0.005 THEN \'high\'\n                WHEN g.group_record_count >= 4\n                 AND g.group_fingerprint_completeness >= 0.50\n                 AND g.low_assignment_rate <= 0.10\n                 AND g.position_jump_rate <= 0.02 THEN \'medium\'\n                ELSE \'low\'\n            END AS id_confidence\n        FROM numbered n\n        JOIN group_stats g USING (pseudo_ship_group_id)\n        """\n    )\n    con.execute(f"DROP TABLE {base}")\n    con.execute(f"DROP TABLE {ship_type}_assignments")\n\n    print(f"[{ship_type}] DuckDB：按严格双5分钟槽聚合到10分钟")\n\n    # 先压到5分钟槽级。若同一船舶、轨迹段和5分钟槽出现多行，\n    # 不自动平均燃油，也不把重复行视为多个有效时间槽。\n    con.execute(\n        f"""\n        CREATE OR REPLACE TABLE {model10}_slot AS\n        SELECT\n            pseudo_ship_group_id,\n            trajectory_segment_id,\n            first(id_confidence ORDER BY timestamp_utc, source_row)\n                AS id_confidence,\n            \'{ship_type}\'::VARCHAR AS ship_type,\n            time_bucket(INTERVAL \'5 minutes\', timestamp_utc)\n                AS timestamp_5min,\n            time_bucket(INTERVAL \'10 minutes\', timestamp_utc)\n                AS timestamp_10min,\n            count(*) AS rows_in_5min_slot,\n            sum(CASE WHEN fuel_raw_valid_flag = 1 THEN 1 ELSE 0 END)\n                AS valid_fuel_rows_in_slot,\n            avg(latitude_deg) AS latitude_deg,\n            avg(longitude_deg) AS longitude_deg,\n            first(design_draught_m ORDER BY timestamp_utc, source_row)\n                FILTER (WHERE design_draught_m IS NOT NULL)\n                AS design_draught_m,\n            first(deadweight_t ORDER BY timestamp_utc, source_row)\n                FILTER (WHERE deadweight_t IS NOT NULL)\n                AS deadweight_t,\n            first(service_speed_kn ORDER BY timestamp_utc, source_row)\n                FILTER (WHERE service_speed_kn IS NOT NULL)\n                AS service_speed_kn,\n            first(main_engine_power_kw ORDER BY timestamp_utc, source_row)\n                FILTER (WHERE main_engine_power_kw IS NOT NULL)\n                AS main_engine_power_kw,\n            avg(ship_age_years) AS ship_age_years,\n            avg(speed_kn) AS speed_kn,\n            mod(\n                degrees(\n                    atan2(\n                        avg(sin(radians(course_deg))),\n                        avg(cos(radians(course_deg)))\n                    )\n                ) + 360.0,\n                360.0\n            ) AS course_deg,\n            mod(\n                degrees(\n                    atan2(\n                        avg(sin(radians(heading_deg))),\n                        avg(cos(radians(heading_deg)))\n                    )\n                ) + 360.0,\n                360.0\n            ) AS heading_deg,\n            avg(mean_draught_m) AS mean_draught_m,\n            avg(trim_m) AS trim_m,\n            avg(rudder_deg) AS rudder_deg,\n            avg(wind_speed_kn) AS wind_speed_kn,\n            mod(\n                degrees(\n                    atan2(\n                        avg(sin(radians(wind_direction_deg))),\n                        avg(cos(radians(wind_direction_deg)))\n                    )\n                ) + 360.0,\n                360.0\n            ) AS wind_direction_deg,\n            avg(wave_height_m) AS wave_height_m,\n            avg(wave_period_s) AS wave_period_s,\n            mod(\n                degrees(\n                    atan2(\n                        avg(sin(radians(wave_direction_deg))),\n                        avg(cos(radians(wave_direction_deg)))\n                    )\n                ) + 360.0,\n                360.0\n            ) AS wave_direction_deg,\n            avg(surface_pressure_pa) AS surface_pressure_pa,\n            avg(surface_temperature_c) AS surface_temperature_c,\n            CASE\n                WHEN count(*) = 1\n                 AND sum(CASE\n                     WHEN fuel_raw_valid_flag = 1\n                      AND fuel_t_5min_clean IS NOT NULL\n                      AND fuel_t_5min_clean >= 0\n                     THEN 1 ELSE 0 END\n                 ) = 1\n                THEN max(fuel_t_5min_clean)\n                ELSE NULL\n            END AS fuel_t_5min_valid,\n            min(time_valid_flag) AS time_valid_flag,\n            min(position_valid_flag) AS position_valid_flag,\n            min(speed_valid_flag) AS speed_valid_flag,\n            min(valid_record_flag) AS valid_record_flag\n        FROM {identified}\n        WHERE timestamp_utc IS NOT NULL\n        GROUP BY\n            pseudo_ship_group_id,\n            trajectory_segment_id,\n            time_bucket(INTERVAL \'5 minutes\', timestamp_utc),\n            time_bucket(INTERVAL \'10 minutes\', timestamp_utc)\n        """\n    )\n\n    con.execute(\n        f"""\n        CREATE OR REPLACE TABLE {model10}_pre AS\n        SELECT\n            pseudo_ship_group_id,\n            trajectory_segment_id,\n            first(id_confidence ORDER BY timestamp_5min)\n                AS id_confidence,\n            \'{ship_type}\'::VARCHAR AS ship_type,\n            timestamp_10min AS timestamp_utc,\n            avg(latitude_deg) AS latitude_deg,\n            avg(longitude_deg) AS longitude_deg,\n            first(design_draught_m ORDER BY timestamp_5min)\n                FILTER (WHERE design_draught_m IS NOT NULL)\n                AS design_draught_m,\n            first(deadweight_t ORDER BY timestamp_5min)\n                FILTER (WHERE deadweight_t IS NOT NULL)\n                AS deadweight_t,\n            first(service_speed_kn ORDER BY timestamp_5min)\n                FILTER (WHERE service_speed_kn IS NOT NULL)\n                AS service_speed_kn,\n            first(main_engine_power_kw ORDER BY timestamp_5min)\n                FILTER (WHERE main_engine_power_kw IS NOT NULL)\n                AS main_engine_power_kw,\n            avg(ship_age_years) AS ship_age_years,\n            year(timestamp_10min) AS calendar_year,\n            avg(speed_kn) AS speed_kn,\n            mod(\n                degrees(\n                    atan2(\n                        avg(sin(radians(course_deg))),\n                        avg(cos(radians(course_deg)))\n                    )\n                ) + 360.0,\n                360.0\n            ) AS course_deg,\n            mod(\n                degrees(\n                    atan2(\n                        avg(sin(radians(heading_deg))),\n                        avg(cos(radians(heading_deg)))\n                    )\n                ) + 360.0,\n                360.0\n            ) AS heading_deg,\n            avg(mean_draught_m) AS mean_draught_m,\n            avg(trim_m) AS trim_m,\n            avg(rudder_deg) AS rudder_deg,\n            avg(wind_speed_kn) AS wind_speed_kn,\n            mod(\n                degrees(\n                    atan2(\n                        avg(sin(radians(wind_direction_deg))),\n                        avg(cos(radians(wind_direction_deg)))\n                    )\n                ) + 360.0,\n                360.0\n            ) AS wind_direction_deg,\n            avg(wave_height_m) AS wave_height_m,\n            avg(wave_period_s) AS wave_period_s,\n            mod(\n                degrees(\n                    atan2(\n                        avg(sin(radians(wave_direction_deg))),\n                        avg(cos(radians(wave_direction_deg)))\n                    )\n                ) + 360.0,\n                360.0\n            ) AS wave_direction_deg,\n            avg(surface_pressure_pa) AS surface_pressure_pa,\n            avg(surface_temperature_c) AS surface_temperature_c,\n            sum(rows_in_5min_slot) AS resample_count_10min,\n            count(*) AS unique_5min_slot_count,\n            sum(CASE WHEN rows_in_5min_slot > 1 THEN 1 ELSE 0 END)\n                AS duplicate_5min_slot_count,\n            sum(CASE WHEN fuel_t_5min_valid IS NOT NULL THEN 1 ELSE 0 END)\n                AS fuel_valid_5min_count,\n            sum(fuel_t_5min_valid) AS fuel_t_10min_partial,\n            min(time_valid_flag) AS time_valid_flag,\n            min(position_valid_flag) AS position_valid_flag,\n            min(speed_valid_flag) AS speed_valid_flag,\n            min(valid_record_flag) AS valid_record_flag\n        FROM {model10}_slot\n        GROUP BY\n            pseudo_ship_group_id,\n            trajectory_segment_id,\n            timestamp_10min\n        """\n    )\n\n    agg_distance_expr = haversine_sql(\n        "prev_lat",\n        "prev_lon",\n        "latitude_deg",\n        "longitude_deg",\n    )\n    con.execute(\n        f"""\n        CREATE OR REPLACE TABLE {model10} AS\n        WITH completed AS (\n            SELECT *,\n                CASE\n                    WHEN unique_5min_slot_count < 2\n                        THEN \'missing_5min_slot\'\n                    WHEN unique_5min_slot_count > 2\n                        THEN \'unexpected_more_than_2_slots\'\n                    WHEN duplicate_5min_slot_count > 0\n                        THEN \'duplicate_rows_in_5min_slot\'\n                    WHEN fuel_valid_5min_count < 2\n                        THEN \'invalid_or_missing_fuel\'\n                    WHEN resample_count_10min <> 2\n                        THEN \'unexpected_source_row_count\'\n                    ELSE \'valid_complete_window\'\n                END AS fuel_window_status,\n                CASE\n                    WHEN unique_5min_slot_count = 2\n                     AND duplicate_5min_slot_count = 0\n                     AND fuel_valid_5min_count = 2\n                     AND resample_count_10min = 2\n                    THEN 1 ELSE 0\n                END AS complete_window_flag,\n                least(unique_5min_slot_count / 2.0, 1.0)\n                    AS coverage_ratio_10min,\n                CASE\n                    WHEN unique_5min_slot_count = 2\n                     AND duplicate_5min_slot_count = 0\n                     AND fuel_valid_5min_count = 2\n                     AND resample_count_10min = 2\n                    THEN fuel_t_10min_partial\n                    ELSE NULL\n                END AS fuel_t_10min\n            FROM {model10}_pre\n        ), lagged AS (\n            SELECT *,\n                lag(latitude_deg) OVER (\n                    PARTITION BY\n                        pseudo_ship_group_id,\n                        trajectory_segment_id\n                    ORDER BY timestamp_utc\n                ) AS prev_lat,\n                lag(longitude_deg) OVER (\n                    PARTITION BY\n                        pseudo_ship_group_id,\n                        trajectory_segment_id\n                    ORDER BY timestamp_utc\n                ) AS prev_lon\n            FROM completed\n        )\n        SELECT\n            pseudo_ship_group_id,\n            trajectory_segment_id,\n            id_confidence,\n            ship_type,\n            timestamp_utc,\n            latitude_deg,\n            longitude_deg,\n            {agg_distance_expr} AS distance_nm_10min,\n            design_draught_m,\n            deadweight_t,\n            service_speed_kn,\n            main_engine_power_kw,\n            ship_age_years,\n            calendar_year,\n            speed_kn,\n            course_deg,\n            heading_deg,\n            mean_draught_m,\n            trim_m,\n            rudder_deg,\n            wind_speed_kn,\n            wind_direction_deg,\n            wave_height_m,\n            wave_period_s,\n            wave_direction_deg,\n            surface_pressure_pa,\n            surface_temperature_c,\n            fuel_t_10min,\n            fuel_t_10min * 6000.0 AS fuel_rate_kg_h,\n            \'fuel_t_5min_tonnes_summed_strict_2slot\'\n                AS fuel_source,\n            resample_count_10min,\n            unique_5min_slot_count,\n            duplicate_5min_slot_count,\n            fuel_valid_5min_count,\n            coverage_ratio_10min,\n            complete_window_flag,\n            fuel_window_status,\n            \'{FUEL_INPUT_UNIT_5MIN}\'::VARCHAR AS fuel_input_unit,\n            \'{FUEL_OUTPUT_UNIT_10MIN}\'::VARCHAR AS fuel_output_unit,\n            time_valid_flag,\n            position_valid_flag,\n            speed_valid_flag,\n            CASE\n                WHEN fuel_t_10min IS NOT NULL\n                 AND fuel_t_10min >= 0\n                THEN 1 ELSE 0\n            END AS fuel_valid_flag,\n            valid_record_flag,\n            CASE\n                WHEN wave_height_m IS NULL\n                  OR wave_period_s IS NULL\n                  OR wave_direction_deg IS NULL\n                THEN 1 ELSE 0\n            END AS wave_missing_flag,\n            CASE\n                WHEN surface_temperature_c IS NULL\n                THEN 1 ELSE 0\n            END AS surface_temperature_missing_flag,\n            CASE\n                WHEN rudder_deg IS NULL\n                THEN 1 ELSE 0\n            END AS rudder_missing_flag\n        FROM lagged\n        """\n    )\n    con.execute(f"DROP TABLE {model10}_slot")\n    con.execute(f"DROP TABLE {model10}_pre")\ndef create_container_table(\n    con: "duckdb.DuckDBPyConnection",\n    stage_file: Path,\n    gap_hours: float,\n    jump_speed_kn: float,\n    hard_max_fuel_t_5min: float,\n    max_sfoc_g_kwh: float,\n    fuel_power_margin: float,\n    expected_container_ships: int,\n) -> None:\n    print("[container] DuckDB：使用原始精确交付日期识别5艘集装箱船")\n    con.execute(\n        f"CREATE OR REPLACE TABLE container_raw AS "\n        f"{typed_stage_sql(stage_file, \'container\', True)}"\n    )\n\n    date_rows = con.execute(\n        """\n        SELECT delivery_date, count(*) AS record_count\n        FROM container_raw\n        GROUP BY delivery_date\n        ORDER BY delivery_date NULLS LAST\n        """\n    ).fetchall()\n    missing_delivery_count = int(\n        con.execute(\n            "SELECT count(*) FROM container_raw WHERE delivery_date IS NULL"\n        ).fetchone()[0]\n    )\n    exact_ship_count = int(\n        con.execute(\n            "SELECT count(DISTINCT delivery_date) FROM container_raw "\n            "WHERE delivery_date IS NOT NULL"\n        ).fetchone()[0]\n    )\n\n    print("[container] 精确交付日期分组：")\n    for delivery_date, record_count in date_rows:\n        print(f"  {delivery_date}: {int(record_count):,} 行")\n\n    if missing_delivery_count:\n        raise RuntimeError(\n            f"集装箱船有 {missing_delivery_count:,} 行缺少精确交付日期。"\n            "为避免把记录错误分配给其他船舶，程序已停止。"\n        )\n\n    if expected_container_ships > 0 and exact_ship_count != expected_container_ships:\n        raise RuntimeError(\n            f"按精确交付日期识别出 {exact_ship_count} 艘集装箱船，"\n            f"但预期为 {expected_container_ships}。程序已停止。"\n        )\n\n    distance_expr = haversine_sql(\n        "prev_lat", "prev_lon", "latitude_deg", "longitude_deg"\n    )\n    con.execute(\n        f"""\n        CREATE OR REPLACE TABLE container_identified AS\n        WITH ship_map AS (\n            SELECT\n                delivery_date,\n                row_number() OVER (ORDER BY delivery_date) AS ship_no\n            FROM (\n                SELECT DISTINCT delivery_date\n                FROM container_raw\n                WHERE delivery_date IS NOT NULL\n            ) d\n        ), fuel_prepared AS (\n            SELECT r.*,\n                m.ship_no,\n                coalesce(fuel_t_10min_input, fuel_rate_kg_h_input / 6000.0)\n                    AS fuel_t_10min,\n                coalesce(\n                    fuel_source_input,\n                    CASE\n                        WHEN fuel_t_10min_input IS NOT NULL\n                            THEN \'first_stage_fuel_t_10min\'\n                        WHEN fuel_rate_kg_h_input IS NOT NULL\n                            THEN \'first_stage_fuel_rate\'\n                        ELSE \'missing\'\n                    END\n                ) AS fuel_source\n            FROM container_raw r\n            JOIN ship_map m USING (delivery_date)\n        ), fingerprinted AS (\n            SELECT *,\n                cast(delivery_date AS VARCHAR) AS ship_fingerprint_text,\n                1.0::DOUBLE AS fingerprint_completeness,\n                fuel_t_10min * 6000.0 AS fuel_rate_kg_h_raw,\n                main_engine_power_kw * {float(max_sfoc_g_kwh)} / 1000.0\n                    * {float(fuel_power_margin)} AS fuel_power_limit_kg_h,\n                fuel_t_10min = 0 AS fuel_zero_flag,\n                fuel_t_10min < 0 AS fuel_negative_flag,\n                fuel_t_10min = 0 AND speed_kn > 3 AS fuel_sailing_zero_flag,\n                fuel_t_10min > {float(hard_max_fuel_t_5min) * 2.0}\n                    AS fuel_hard_limit_exceed_flag,\n                CASE\n                    WHEN main_engine_power_kw IS NULL THEN false\n                    ELSE fuel_t_10min * 6000.0\n                        > main_engine_power_kw * {float(max_sfoc_g_kwh)} / 1000.0\n                          * {float(fuel_power_margin)}\n                END AS fuel_power_limit_exceed_flag\n            FROM fuel_prepared\n        ), identified_base AS (\n            SELECT *,\n                substr(md5(ship_fingerprint_text), 1, 14) AS ship_fingerprint,\n                \'CONTAINER_P\' || lpad(cast(ship_no AS VARCHAR), 3, \'0\')\n                    AS pseudo_ship_group_id,\n                \'CONTAINER_P\' || lpad(cast(ship_no AS VARCHAR), 3, \'0\')\n                    AS pseudo_ship_id,\n                CASE\n                    WHEN fuel_t_10min > {float(hard_max_fuel_t_5min) * 2.0}\n                        THEN true\n                    WHEN main_engine_power_kw IS NOT NULL\n                     AND fuel_t_10min * 6000.0\n                        > main_engine_power_kw * {float(max_sfoc_g_kwh)} / 1000.0\n                          * {float(fuel_power_margin)} THEN true\n                    ELSE false\n                END AS fuel_extreme_flag,\n                CASE\n                    WHEN fuel_t_10min IS NULL OR fuel_t_10min < 0 THEN 0\n                    WHEN fuel_t_10min = 0 AND speed_kn > 3 THEN 0\n                    WHEN fuel_t_10min > {float(hard_max_fuel_t_5min) * 2.0} THEN 0\n                    WHEN main_engine_power_kw IS NOT NULL\n                     AND fuel_t_10min * 6000.0\n                        > main_engine_power_kw * {float(max_sfoc_g_kwh)} / 1000.0\n                          * {float(fuel_power_margin)} THEN 0\n                    ELSE 1\n                END AS fuel_raw_valid_flag\n            FROM fingerprinted\n        ), lagged AS (\n            SELECT *,\n                lag(timestamp_utc) OVER (\n                    PARTITION BY pseudo_ship_group_id\n                    ORDER BY timestamp_utc NULLS LAST, source_row\n                ) AS prev_time,\n                lag(latitude_deg) OVER (\n                    PARTITION BY pseudo_ship_group_id\n                    ORDER BY timestamp_utc NULLS LAST, source_row\n                ) AS prev_lat,\n                lag(longitude_deg) OVER (\n                    PARTITION BY pseudo_ship_group_id\n                    ORDER BY timestamp_utc NULLS LAST, source_row\n                ) AS prev_lon,\n                row_number() OVER (\n                    PARTITION BY pseudo_ship_group_id\n                    ORDER BY timestamp_utc NULLS LAST, source_row\n                ) AS group_rn\n            FROM identified_base\n        ), movement AS (\n            SELECT *,\n                date_diff(\'minute\', prev_time, timestamp_utc) AS time_gap_min,\n                {distance_expr} AS distance_nm_10min\n            FROM lagged\n        ), speed_calc AS (\n            SELECT *,\n                CASE WHEN time_gap_min > 0\n                    THEN distance_nm_10min / (time_gap_min / 60.0)\n                    ELSE NULL END AS implied_speed_kn\n            FROM movement\n        ), boundaries AS (\n            SELECT *,\n                CASE WHEN implied_speed_kn > {float(jump_speed_kn)}\n                    THEN 1 ELSE 0 END AS position_jump_flag,\n                CASE\n                    WHEN group_rn = 1 THEN 1\n                    WHEN timestamp_utc IS NULL THEN 1\n                    WHEN time_gap_min = 0\n                     AND coalesce(distance_nm_10min, 0.0) <= 0.25 THEN 0\n                    WHEN time_gap_min <= 0 THEN 1\n                    WHEN time_gap_min > {float(gap_hours) * 60.0} THEN 1\n                    WHEN implied_speed_kn > {float(jump_speed_kn)} THEN 1\n                    ELSE 0\n                END AS new_segment_flag\n            FROM speed_calc\n        ), numbered AS (\n            SELECT *,\n                sum(new_segment_flag) OVER (\n                    PARTITION BY pseudo_ship_group_id\n                    ORDER BY timestamp_utc NULLS LAST, source_row\n                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW\n                ) AS segment_no\n            FROM boundaries\n        ), group_stats AS (\n            SELECT pseudo_ship_group_id,\n                count(*) AS group_record_count,\n                avg(CASE WHEN position_jump_flag = 1 THEN 1.0 ELSE 0.0 END)\n                    AS position_jump_rate\n            FROM numbered\n            GROUP BY pseudo_ship_group_id\n        )\n        SELECT n.*,\n            pseudo_ship_group_id || \'_S\'\n                || lpad(cast(segment_no AS VARCHAR), 4, \'0\')\n                AS trajectory_segment_id,\n            CASE\n                WHEN g.group_record_count >= 20\n                 AND g.position_jump_rate <= 0.005 THEN \'high\'\n                WHEN g.group_record_count >= 4\n                 AND g.position_jump_rate <= 0.02 THEN \'medium\'\n                ELSE \'low\'\n            END AS id_confidence,\n            CASE\n                WHEN group_rn = 1 THEN \'exact_delivery_date_anchor\'\n                WHEN implied_speed_kn > {float(jump_speed_kn)}\n                    THEN \'position_jump_same_delivery_date\'\n                WHEN time_gap_min > {float(gap_hours) * 60.0}\n                    THEN \'long_gap_same_delivery_date\'\n                ELSE \'matched_by_exact_delivery_date\'\n            END AS assignment_reason,\n            CASE\n                WHEN implied_speed_kn > {float(jump_speed_kn)} THEN \'low\'\n                WHEN time_gap_min > {float(gap_hours) * 60.0} THEN \'medium\'\n                ELSE \'high\'\n            END AS assignment_confidence,\n            implied_speed_kn AS matched_implied_speed_kn,\n            1 AS match_candidate_count\n        FROM numbered n\n        JOIN group_stats g USING (pseudo_ship_group_id)\n        """\n    )\n\n    ship_count = int(\n        con.execute(\n            "SELECT count(DISTINCT pseudo_ship_group_id) "\n            "FROM container_identified"\n        ).fetchone()[0]\n    )\n    print(f"[container] 精确交付日期识别出 {ship_count} 艘推定船舶")\n\n    con.execute(\n        """\n        CREATE OR REPLACE TABLE container_identity_summary AS\n        SELECT\n            pseudo_ship_group_id,\n            min(delivery_date) AS delivery_date,\n            count(*) AS record_count,\n            min(timestamp_utc) AS start_time,\n            max(timestamp_utc) AS end_time,\n            count(DISTINCT trajectory_segment_id) AS trajectory_segment_count,\n            sum(CASE WHEN position_jump_flag = 1 THEN 1 ELSE 0 END)\n                AS position_jump_count,\n            quantile_cont(implied_speed_kn, 0.99) AS p99_implied_speed_kn,\n            max(implied_speed_kn) AS max_implied_speed_kn,\n            min(id_confidence) AS id_confidence\n        FROM container_identified\n        GROUP BY pseudo_ship_group_id\n        """\n    )\n\n    con.execute(\n        """\n        CREATE OR REPLACE TABLE container_model_10min AS\n        SELECT\n            pseudo_ship_group_id, trajectory_segment_id, id_confidence,\n            \'container\'::VARCHAR AS ship_type, timestamp_utc,\n            latitude_deg, longitude_deg, distance_nm_10min,\n            design_draught_m, deadweight_t, service_speed_kn,\n            main_engine_power_kw, ship_age_years,\n            coalesce(calendar_year_input, year(timestamp_utc)) AS calendar_year,\n            speed_kn, course_deg, heading_deg, mean_draught_m, trim_m,\n            rudder_deg, wind_speed_kn, wind_direction_deg, wave_height_m,\n            wave_period_s, wave_direction_deg, surface_pressure_pa,\n            surface_temperature_c, fuel_t_10min,\n            fuel_t_10min * 6000.0 AS fuel_rate_kg_h, fuel_source,\n            fuel_zero_flag, fuel_negative_flag, fuel_sailing_zero_flag,\n            fuel_hard_limit_exceed_flag, fuel_power_limit_exceed_flag,\n            fuel_extreme_flag, 1 AS resample_count_10min,\n            1.0 AS coverage_ratio_10min, 1 AS complete_window_flag,\n            time_valid_flag, position_valid_flag, speed_valid_flag,\n            fuel_raw_valid_flag AS fuel_valid_flag, valid_record_flag,\n            CASE WHEN wave_height_m IS NULL OR wave_period_s IS NULL\n                  OR wave_direction_deg IS NULL THEN 1 ELSE 0 END\n                AS wave_missing_flag,\n            CASE WHEN surface_temperature_c IS NULL THEN 1 ELSE 0 END\n                AS surface_temperature_missing_flag,\n            CASE WHEN rudder_deg IS NULL THEN 1 ELSE 0 END\n                AS rudder_missing_flag\n        FROM container_identified\n        """\n    )\n    con.execute("DROP TABLE container_raw")\n\n\n# =============================================================================\n# 报告、统一表、特征与划分\n# =============================================================================\n\ndef copy_query_to_csv(\n    con: "duckdb.DuckDBPyConnection",\n    query: str,\n    output_path: Path,\n) -> None:\n    output_path.parent.mkdir(parents=True, exist_ok=True)\n    con.execute(\n        f"COPY ({query}) TO \'{sql_path(output_path)}\' "\n        "(HEADER, DELIMITER \',\', QUOTE \'\\"\', ESCAPE \'\\"\')"\n    )\n\n\n\ndef build_pseudo_report(con: "duckdb.DuckDBPyConnection", output_path: Path) -> None:\n    query = """\n        SELECT * FROM (\n            SELECT\n                \'container\'::VARCHAR AS ship_type,\n                pseudo_ship_group_id,\n                first(pseudo_ship_id) AS pseudo_ship_id,\n                count(*) AS record_count,\n                min(timestamp_utc) AS start_time,\n                max(timestamp_utc) AS end_time,\n                count(DISTINCT trajectory_segment_id) AS trajectory_segment_count,\n                sum(position_jump_flag) AS position_jump_count,\n                median(fingerprint_completeness) AS fingerprint_completeness,\n                first(id_confidence) AS id_confidence,\n                first(ship_fingerprint) AS ship_fingerprint,\n                first(delivery_date) AS delivery_date,\n                max(time_gap_min) AS max_time_gap_min,\n                max(implied_speed_kn) AS max_implied_speed_kn,\n                median(matched_implied_speed_kn) AS median_assignment_speed_kn,\n                avg(CASE WHEN assignment_confidence = \'low\' THEN 1.0 ELSE 0.0 END)\n                    AS low_assignment_rate,\n                sum(CASE WHEN assignment_reason = \'simultaneous_far_position_conflict\'\n                    THEN 1 ELSE 0 END) AS simultaneous_far_conflict_count\n            FROM container_identified\n            GROUP BY pseudo_ship_group_id\n\n            UNION ALL\n\n            SELECT\n                \'bulk\'::VARCHAR AS ship_type,\n                pseudo_ship_group_id,\n                first(pseudo_ship_id) AS pseudo_ship_id,\n                count(*) AS record_count,\n                min(timestamp_utc) AS start_time,\n                max(timestamp_utc) AS end_time,\n                count(DISTINCT trajectory_segment_id) AS trajectory_segment_count,\n                sum(position_jump_flag) AS position_jump_count,\n                median(fingerprint_completeness) AS fingerprint_completeness,\n                first(id_confidence) AS id_confidence,\n                first(ship_fingerprint) AS ship_fingerprint,\n                first(delivery_date) AS delivery_date,\n                max(time_gap_min) AS max_time_gap_min,\n                max(implied_speed_kn) AS max_implied_speed_kn,\n                median(matched_implied_speed_kn) AS median_assignment_speed_kn,\n                avg(CASE WHEN assignment_confidence = \'low\' THEN 1.0 ELSE 0.0 END)\n                    AS low_assignment_rate,\n                sum(CASE WHEN assignment_reason = \'simultaneous_far_position_conflict\'\n                    THEN 1 ELSE 0 END) AS simultaneous_far_conflict_count\n            FROM bulk_identified\n            GROUP BY pseudo_ship_group_id\n\n            UNION ALL\n\n            SELECT\n                \'tanker\'::VARCHAR AS ship_type,\n                pseudo_ship_group_id,\n                first(pseudo_ship_id) AS pseudo_ship_id,\n                count(*) AS record_count,\n                min(timestamp_utc) AS start_time,\n                max(timestamp_utc) AS end_time,\n                count(DISTINCT trajectory_segment_id) AS trajectory_segment_count,\n                sum(position_jump_flag) AS position_jump_count,\n                median(fingerprint_completeness) AS fingerprint_completeness,\n                first(id_confidence) AS id_confidence,\n                first(ship_fingerprint) AS ship_fingerprint,\n                first(delivery_date) AS delivery_date,\n                max(time_gap_min) AS max_time_gap_min,\n                max(implied_speed_kn) AS max_implied_speed_kn,\n                median(matched_implied_speed_kn) AS median_assignment_speed_kn,\n                avg(CASE WHEN assignment_confidence = \'low\' THEN 1.0 ELSE 0.0 END)\n                    AS low_assignment_rate,\n                sum(CASE WHEN assignment_reason = \'simultaneous_far_position_conflict\'\n                    THEN 1 ELSE 0 END) AS simultaneous_far_conflict_count\n            FROM tanker_identified\n            GROUP BY pseudo_ship_group_id\n        )\n        ORDER BY ship_type, pseudo_ship_group_id\n    """\n    copy_query_to_csv(con, query, output_path)\n\n\ndef build_fuel_report(con: "duckdb.DuckDBPyConnection", output_path: Path) -> None:\n    long_rows: List[Dict[str, object]] = []\n\n    container = con.execute(\n        """\n        SELECT\n            count(*) AS total_records,\n            count(fuel_t_10min) AS fuel_nonmissing_count,\n            count(*) - count(fuel_t_10min) AS fuel_missing_count,\n            (count(*) - count(fuel_t_10min))::DOUBLE / nullif(count(*), 0) AS fuel_missing_rate,\n            sum(CASE WHEN fuel_t_10min < 0 THEN 1 ELSE 0 END) AS fuel_negative_count,\n            sum(CASE WHEN fuel_t_10min = 0 THEN 1 ELSE 0 END) AS fuel_zero_count,\n            sum(CASE WHEN fuel_t_10min = 0 AND speed_kn > 3 THEN 1 ELSE 0 END)\n                AS sailing_zero_count_speed_gt_3kn,\n            sum(CASE WHEN fuel_hard_limit_exceed_flag THEN 1 ELSE 0 END)\n                AS fuel_hard_limit_exceed_count,\n            sum(CASE WHEN fuel_power_limit_exceed_flag THEN 1 ELSE 0 END)\n                AS fuel_power_limit_exceed_count,\n            sum(CASE WHEN fuel_extreme_flag THEN 1 ELSE 0 END) AS fuel_extreme_count,\n            sum(CASE WHEN fuel_valid_flag = 1 THEN 1 ELSE 0 END) AS fuel_valid_count,\n            avg(CASE WHEN fuel_valid_flag = 1 THEN 1.0 ELSE 0.0 END) AS fuel_valid_rate,\n            median(fuel_t_10min) AS fuel_raw_median,\n            quantile_cont(fuel_t_10min, 0.99) AS fuel_raw_q99,\n            median(fuel_t_10min) FILTER (WHERE fuel_valid_flag = 1) AS fuel_valid_median,\n            quantile_cont(fuel_t_10min, 0.99) FILTER (WHERE fuel_valid_flag = 1)\n                AS fuel_valid_q99,\n            median(fuel_rate_kg_h) FILTER (WHERE fuel_valid_flag = 1)\n                AS valid_fuel_rate_median_kg_h,\n            quantile_cont(fuel_rate_kg_h, 0.99) FILTER (WHERE fuel_valid_flag = 1)\n                AS valid_fuel_rate_q99_kg_h\n        FROM container_model_10min\n        """\n    ).fetchdf().iloc[0]\n    for metric, value in container.items():\n        long_rows.append({"ship_type": "container", "metric": metric, "value": value})\n\n    for ship_type in ["bulk", "tanker"]:\n        row = con.execute(\n            f"""\n            SELECT\n                count(*) AS total_records,\n                count(fuel_t_5min) AS fuel_nonmissing_count,\n                count(*) - count(fuel_t_5min) AS fuel_missing_count,\n                (count(*) - count(fuel_t_5min))::DOUBLE / nullif(count(*), 0)\n                    AS fuel_missing_rate,\n                sum(CASE WHEN fuel_negative_flag THEN 1 ELSE 0 END) AS fuel_negative_count,\n                sum(CASE WHEN fuel_zero_flag THEN 1 ELSE 0 END) AS fuel_zero_count,\n                sum(CASE WHEN fuel_sailing_zero_flag THEN 1 ELSE 0 END)\n                    AS sailing_zero_count_speed_gt_3kn,\n                sum(CASE WHEN fuel_hard_limit_exceed_flag THEN 1 ELSE 0 END)\n                    AS fuel_hard_limit_exceed_count,\n                sum(CASE WHEN fuel_power_limit_exceed_flag THEN 1 ELSE 0 END)\n                    AS fuel_power_limit_exceed_count,\n                sum(CASE WHEN fuel_extreme_flag THEN 1 ELSE 0 END) AS fuel_extreme_count,\n                sum(fuel_raw_valid_flag) AS fuel_valid_count,\n                avg(fuel_raw_valid_flag::DOUBLE) AS fuel_valid_rate,\n                median(fuel_t_5min) AS fuel_raw_median,\n                quantile_cont(fuel_t_5min, 0.99) AS fuel_raw_q99,\n                quantile_cont(fuel_t_5min, 0.999) AS fuel_raw_q999,\n                median(fuel_t_5min_clean) FILTER (WHERE fuel_raw_valid_flag = 1)\n                    AS fuel_valid_median,\n                quantile_cont(fuel_t_5min_clean, 0.99)\n                    FILTER (WHERE fuel_raw_valid_flag = 1) AS fuel_valid_q99,\n                median(fuel_t_5min_clean * 12000.0)\n                    FILTER (WHERE fuel_raw_valid_flag = 1)\n                    AS valid_fuel_rate_median_kg_h,\n                quantile_cont(fuel_t_5min_clean * 12000.0, 0.99)\n                    FILTER (WHERE fuel_raw_valid_flag = 1)\n                    AS valid_fuel_rate_q99_kg_h\n            FROM {ship_type}_identified\n            """\n        ).fetchdf().iloc[0]\n        for metric, value in row.items():\n            long_rows.append({"ship_type": ship_type, "metric": metric, "value": value})\n\n    pd.DataFrame(long_rows).to_csv(output_path, index=False, encoding="utf-8-sig")\n\ndef build_container_missing_report(\n    con: "duckdb.DuckDBPyConnection",\n    output_path: Path,\n) -> None:\n    variables = [\n        "wave_height_m", "wave_period_s", "wave_direction_deg",\n        "surface_temperature_c", "rudder_deg"\n    ]\n    frames: List[pd.DataFrame] = []\n    for variable in variables:\n        frame = con.execute(\n            f"""\n            WITH ordered AS (\n                SELECT timestamp_utc,\n                    {variable} IS NULL AS is_missing,\n                    lag({variable} IS NULL) OVER (ORDER BY timestamp_utc) AS prev_missing,\n                    lag(timestamp_utc) OVER (ORDER BY timestamp_utc) AS prev_time\n                FROM container_model_10min\n            ), starts AS (\n                SELECT *,\n                    CASE\n                        WHEN is_missing AND (\n                            coalesce(prev_missing, false) = false\n                            OR date_diff(\'minute\', prev_time, timestamp_utc) > 15\n                            OR date_diff(\'minute\', prev_time, timestamp_utc) <= 0\n                        ) THEN 1 ELSE 0\n                    END AS new_run\n                FROM ordered\n            ), numbered AS (\n                SELECT *,\n                    sum(new_run) OVER (\n                        ORDER BY timestamp_utc\n                        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW\n                    ) AS run_id\n                FROM starts\n            )\n            SELECT\n                \'{variable}\'::VARCHAR AS variable,\n                min(timestamp_utc) AS missing_start,\n                max(timestamp_utc) AS missing_end,\n                date_diff(\'minute\', min(timestamp_utc), max(timestamp_utc)) + 10\n                    AS missing_duration_min,\n                count(*) AS missing_record_count,\n                CASE\n                    WHEN count(*) = 1 THEN \'single_point\'\n                    WHEN date_diff(\'minute\', min(timestamp_utc), max(timestamp_utc)) + 10 <= 30\n                        THEN \'20_30min\'\n                    WHEN date_diff(\'minute\', min(timestamp_utc), max(timestamp_utc)) + 10 <= 360\n                        THEN \'30min_6h\'\n                    WHEN date_diff(\'minute\', min(timestamp_utc), max(timestamp_utc)) + 10 <= 1440\n                        THEN \'6_24h\'\n                    ELSE \'over_24h\'\n                END AS duration_bucket\n            FROM numbered\n            WHERE is_missing\n            GROUP BY run_id\n            ORDER BY missing_start\n            """\n        ).fetchdf()\n        frames.append(frame)\n\n    pd.concat(frames, ignore_index=True).to_csv(\n        output_path, index=False, encoding="utf-8-sig"\n    )\n\n\ndef select_model_columns(table: str) -> str:\n    return ", ".join(MODEL_COLUMNS)\n\n\ndef assert_ship_type_table(\n    con: "duckdb.DuckDBPyConnection",\n    table_name: str,\n    expected_ship_type: str,\n) -> None:\n    if expected_ship_type not in ALLOWED_SHIP_TYPES:\n        raise ValueError(f"非法预期船型：{expected_ship_type}")\n\n    total, invalid, distinct_types = con.execute(\n        f"""\n        SELECT\n            count(*) AS total_rows,\n            sum(CASE\n                WHEN ship_type IS NULL\n                  OR lower(trim(ship_type)) <> ?\n                THEN 1 ELSE 0 END\n            ) AS invalid_rows,\n            count(DISTINCT lower(trim(ship_type))) AS distinct_ship_types\n        FROM {table_name}\n        """,\n        [expected_ship_type],\n    ).fetchone()\n\n    if int(invalid or 0) > 0 or int(distinct_types or 0) != 1:\n        raise RuntimeError(\n            f"{table_name}船型控制失败：expected={expected_ship_type}, "\n            f"total={int(total or 0)}, invalid={int(invalid or 0)}, "\n            f"distinct={int(distinct_types or 0)}"\n        )\n\n\ndef build_ship_type_control_report(\n    con: "duckdb.DuckDBPyConnection",\n    output_path: Path,\n) -> None:\n    rows: List[Dict[str, object]] = []\n    for ship_type, table_name in [\n        ("container", "container_model_10min"),\n        ("bulk", "bulk_model_10min"),\n        ("tanker", "tanker_model_10min"),\n    ]:\n        assert_ship_type_table(con, table_name, ship_type)\n        result = con.execute(\n            f"""\n            SELECT\n                count(*) AS total_rows,\n                count(DISTINCT pseudo_ship_group_id) AS ship_count,\n                count(DISTINCT trajectory_segment_id) AS segment_count,\n                min(timestamp_utc) AS first_timestamp,\n                max(timestamp_utc) AS last_timestamp,\n                count(fuel_t_10min) AS fuel_nonmissing_rows\n            FROM {table_name}\n            """\n        ).fetchone()\n        rows.append(\n            {\n                "expected_ship_type": ship_type,\n                "table_name": table_name,\n                "total_rows": int(result[0] or 0),\n                "pseudo_ship_count": int(result[1] or 0),\n                "trajectory_segment_count": int(result[2] or 0),\n                "first_timestamp": result[3],\n                "last_timestamp": result[4],\n                "fuel_nonmissing_rows": int(result[5] or 0),\n                "ship_type_check": "passed",\n            }\n        )\n    pd.DataFrame(rows).to_csv(\n        output_path,\n        index=False,\n        encoding="utf-8-sig",\n    )\n\n\ndef build_fuel_window_report(\n    con: "duckdb.DuckDBPyConnection",\n    output_path: Path,\n) -> None:\n    frames: List[pd.DataFrame] = []\n\n    container = con.execute(\n        """\n        SELECT\n            \'container\'::VARCHAR AS ship_type,\n            CASE\n                WHEN fuel_valid_flag = 1 THEN \'native_10min_valid\'\n                WHEN fuel_t_10min IS NULL THEN \'native_10min_missing\'\n                ELSE \'native_10min_invalid\'\n            END AS fuel_window_status,\n            count(*) AS window_count,\n            count(fuel_t_10min) AS fuel_nonmissing_count,\n            min(fuel_t_10min) AS fuel_t_10min_min,\n            avg(fuel_t_10min) AS fuel_t_10min_mean,\n            median(fuel_t_10min) AS fuel_t_10min_median,\n            max(fuel_t_10min) AS fuel_t_10min_max\n        FROM container_model_10min\n        GROUP BY 1, 2\n        """\n    ).fetchdf()\n    frames.append(container)\n\n    for ship_type in ("bulk", "tanker"):\n        frame = con.execute(\n            f"""\n            SELECT\n                \'{ship_type}\'::VARCHAR AS ship_type,\n                fuel_window_status,\n                count(*) AS window_count,\n                count(fuel_t_10min) AS fuel_nonmissing_count,\n                min(fuel_t_10min) AS fuel_t_10min_min,\n                avg(fuel_t_10min) AS fuel_t_10min_mean,\n                median(fuel_t_10min) AS fuel_t_10min_median,\n                max(fuel_t_10min) AS fuel_t_10min_max\n            FROM {ship_type}_model_10min\n            GROUP BY 1, 2\n            """\n        ).fetchdf()\n        frames.append(frame)\n\n    pd.concat(frames, ignore_index=True).to_csv(\n        output_path,\n        index=False,\n        encoding="utf-8-sig",\n    )\n\n\ndef build_unit_consistency_report(\n    con: "duckdb.DuckDBPyConnection",\n    output_path: Path,\n) -> None:\n    rows: List[Dict[str, object]] = []\n    for ship_type, table_name in [\n        ("container", "container_model_10min"),\n        ("bulk", "bulk_model_10min"),\n        ("tanker", "tanker_model_10min"),\n    ]:\n        result = con.execute(\n            f"""\n            SELECT\n                count(*) FILTER (\n                    WHERE fuel_t_10min IS NOT NULL\n                      AND fuel_rate_kg_h IS NOT NULL\n                ) AS compared_rows,\n                max(abs(fuel_rate_kg_h - fuel_t_10min * 6000.0))\n                    FILTER (\n                        WHERE fuel_t_10min IS NOT NULL\n                          AND fuel_rate_kg_h IS NOT NULL\n                    ) AS max_absolute_difference_kg_h,\n                avg(abs(fuel_rate_kg_h - fuel_t_10min * 6000.0))\n                    FILTER (\n                        WHERE fuel_t_10min IS NOT NULL\n                          AND fuel_rate_kg_h IS NOT NULL\n                    ) AS mean_absolute_difference_kg_h\n            FROM {table_name}\n            """\n        ).fetchone()\n        max_difference = float(result[1] or 0.0)\n        rows.append(\n            {\n                "ship_type": ship_type,\n                "table_name": table_name,\n                "fuel_t_10min_unit": FUEL_OUTPUT_UNIT_10MIN,\n                "fuel_rate_kg_h_unit": "kg_per_hour",\n                "expected_relation": "fuel_rate_kg_h = fuel_t_10min * 6000",\n                "compared_rows": int(result[0] or 0),\n                "max_absolute_difference_kg_h": max_difference,\n                "mean_absolute_difference_kg_h": float(result[2] or 0.0),\n                "unit_check": (\n                    "passed" if max_difference <= 1e-8 else "failed"\n                ),\n            }\n        )\n\n    report = pd.DataFrame(rows)\n    report.to_csv(\n        output_path,\n        index=False,\n        encoding="utf-8-sig",\n    )\n    if (report["unit_check"] == "failed").any():\n        raise RuntimeError(\n            "fuel_t_10min与fuel_rate_kg_h单位一致性核验失败。"\n        )\n\n\ndef build_unified_tables(\n    con: "duckdb.DuckDBPyConnection",\n    output_dir: Path,\n    folds: int,\n) -> Dict[str, int]:\n    print("[unified] 合并三船型10分钟表")\n    assert_ship_type_table(con, "container_model_10min", "container")\n    assert_ship_type_table(con, "bulk_model_10min", "bulk")\n    assert_ship_type_table(con, "tanker_model_10min", "tanker")\n\n    cols = select_model_columns("")\n    con.execute(\n        f"""\n        CREATE OR REPLACE TABLE unified_full AS\n        SELECT {cols} FROM container_model_10min\n        UNION ALL\n        SELECT {cols} FROM bulk_model_10min\n        UNION ALL\n        SELECT {cols} FROM tanker_model_10min\n        """\n    )\n\n    copy_query_to_csv(\n        con,\n        f"SELECT {cols} FROM unified_full ORDER BY ship_type, pseudo_ship_group_id, timestamp_utc",\n        output_dir / "unified_ship_10min_full.csv",\n    )\n\n    print("[unified] 应用主分析质量条件")\n    con.execute(\n        """\n        CREATE OR REPLACE TABLE model_candidate AS\n        SELECT *\n        FROM unified_full\n        WHERE coalesce(time_valid_flag, 1) = 1\n          AND coalesce(position_valid_flag, 1) = 1\n          AND coalesce(speed_valid_flag, 1) = 1\n          AND coalesce(fuel_valid_flag, 0) = 1\n          AND coalesce(complete_window_flag, 0) = 1\n          AND id_confidence IN (\'high\', \'medium\')\n        """\n    )\n    copy_query_to_csv(\n        con,\n        "SELECT * FROM model_candidate ORDER BY ship_type, pseudo_ship_group_id, timestamp_utc",\n        output_dir / "unified_ship_10min_model_candidate.csv",\n    )\n\n    group_sizes = con.execute(\n        """\n        SELECT\n            ship_type,\n            pseudo_ship_group_id,\n            count(*) AS record_count\n        FROM model_candidate\n        GROUP BY ship_type, pseudo_ship_group_id\n        ORDER BY ship_type, record_count DESC, pseudo_ship_group_id\n        """\n    ).fetchdf()\n\n    # 每个船型内部独立平衡group_fold_id，确保分船型建模时折负载合理。\n    fold_loads_by_type: Dict[str, List[int]] = {\n        ship_type: [0] * folds\n        for ship_type in ALLOWED_SHIP_TYPES\n    }\n    fold_rows: List[Dict[str, object]] = []\n    for _, row in group_sizes.iterrows():\n        ship_type = str(row["ship_type"])\n        if ship_type not in fold_loads_by_type:\n            raise RuntimeError(\n                f"model_candidate出现未知船型：{ship_type}"\n            )\n        loads = fold_loads_by_type[ship_type]\n        fold = int(np.argmin(loads))\n        count = int(row["record_count"])\n        fold_rows.append(\n            {\n                "ship_type": ship_type,\n                "pseudo_ship_group_id": row["pseudo_ship_group_id"],\n                "group_fold_id": fold,\n            }\n        )\n        loads[fold] += count\n\n    fold_map = pd.DataFrame(fold_rows)\n    con.register("fold_map_df", fold_map)\n    con.execute("CREATE OR REPLACE TABLE fold_map AS SELECT * FROM fold_map_df")\n    con.unregister("fold_map_df")\n\n    print("[features] 构造物理增强特征和固定样本划分")\n    con.execute(\n        """\n        CREATE OR REPLACE TABLE features_splits AS\n        WITH ordered AS (\n            SELECT c.*,\n                row_number() OVER (\n                    PARTITION BY ship_type, pseudo_ship_group_id\n                    ORDER BY timestamp_utc\n                ) AS group_row_no,\n                count(*) OVER (\n                    PARTITION BY ship_type, pseudo_ship_group_id\n                ) AS group_row_count\n            FROM model_candidate c\n        ), features AS (\n            SELECT *,\n                pow(speed_kn, 3) AS speed_cubed,\n                CASE WHEN service_speed_kn > 0\n                    THEN speed_kn / service_speed_kn ELSE NULL END\n                    AS speed_service_ratio,\n                CASE WHEN design_draught_m > 0\n                    THEN mean_draught_m / design_draught_m ELSE NULL END\n                    AS draught_design_ratio,\n                abs(rudder_deg) AS abs_rudder_deg,\n                mod(mod(wind_direction_deg - course_deg + 180.0, 360.0) + 360.0, 360.0) - 180.0\n                    AS relative_wind_angle_deg,\n                mod(mod(wave_direction_deg - course_deg + 180.0, 360.0) + 360.0, 360.0) - 180.0\n                    AS relative_wave_angle_deg,\n                wind_speed_kn * cos(radians(\n                    mod(mod(wind_direction_deg - course_deg + 180.0, 360.0) + 360.0, 360.0) - 180.0\n                )) AS headwind_component_kn,\n                wind_speed_kn * abs(sin(radians(\n                    mod(mod(wind_direction_deg - course_deg + 180.0, 360.0) + 360.0, 360.0) - 180.0\n                ))) AS crosswind_component_kn,\n                wave_height_m * cos(radians(\n                    mod(mod(wave_direction_deg - course_deg + 180.0, 360.0) + 360.0, 360.0) - 180.0\n                )) AS headwave_component_m,\n                wave_height_m * abs(sin(radians(\n                    mod(mod(wave_direction_deg - course_deg + 180.0, 360.0) + 360.0, 360.0) - 180.0\n                ))) AS crosswave_component_m,\n                sin(radians(course_deg)) AS course_sin,\n                cos(radians(course_deg)) AS course_cos,\n                sin(radians(wind_direction_deg)) AS wind_direction_sin,\n                cos(radians(wind_direction_deg)) AS wind_direction_cos,\n                sin(radians(wave_direction_deg)) AS wave_direction_sin,\n                cos(radians(wave_direction_deg)) AS wave_direction_cos,\n                CASE\n                    WHEN group_row_no::DOUBLE / group_row_count <= 0.70 THEN \'train\'\n                    WHEN group_row_no::DOUBLE / group_row_count <= 0.85 THEN \'validation\'\n                    ELSE \'test\'\n                END AS time_split_label,\n                \'within_pseudo_ship_future_prediction\'::VARCHAR AS split_task\n            FROM ordered\n        )\n        SELECT f.*,\n            m.group_fold_id,\n            CASE WHEN id_confidence IN (\'high\', \'medium\') THEN 1 ELSE 0 END\n                AS group_cv_eligible_flag\n        FROM features f\n        LEFT JOIN fold_map m\n          USING (ship_type, pseudo_ship_group_id)\n        """\n    )\n\n    copy_query_to_csv(\n        con,\n        "SELECT * FROM features_splits ORDER BY ship_type, pseudo_ship_group_id, timestamp_utc",\n        output_dir / "unified_ship_10min_features_splits.csv",\n    )\n\n    counts = {\n        "container_10min": int(con.execute("SELECT count(*) FROM container_model_10min").fetchone()[0]),\n        "bulk_10min": int(con.execute("SELECT count(*) FROM bulk_model_10min").fetchone()[0]),\n        "tanker_10min": int(con.execute("SELECT count(*) FROM tanker_model_10min").fetchone()[0]),\n        "unified_full": int(con.execute("SELECT count(*) FROM unified_full").fetchone()[0]),\n        "model_candidate": int(con.execute("SELECT count(*) FROM model_candidate").fetchone()[0]),\n        "features_splits": int(con.execute("SELECT count(*) FROM features_splits").fetchone()[0]),\n    }\n    return counts\n\n\n# =============================================================================\n# 命令行与主流程\n# =============================================================================\n\ndef parse_args() -> argparse.Namespace:\n    parser = argparse.ArgumentParser(\n        description="第二阶段低内存后处理V6：键控静态字段对齐、船型控制、严格5分钟燃油聚合、统一10分钟表与固定划分"\n    )\n    parser.add_argument("--clean-dir", type=Path, default=DEFAULT_CLEAN_DIR)\n    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)\n    parser.add_argument("--raw-container", type=Path, default=DEFAULT_RAW_CONTAINER)\n    parser.add_argument("--raw-bulk", type=Path, default=DEFAULT_RAW_BULK)\n    parser.add_argument("--raw-tanker", type=Path, default=DEFAULT_RAW_TANKER)\n    parser.add_argument("--cleaned-container", type=Path)\n    parser.add_argument("--cleaned-bulk", type=Path)\n    parser.add_argument("--cleaned-tanker", type=Path)\n    parser.add_argument("--chunksize", type=int, default=DEFAULT_CHUNK_SIZE)\n    parser.add_argument("--memory-limit", default=DEFAULT_MEMORY_LIMIT)\n    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)\n    parser.add_argument(\n        "--segment-gap-hours", type=float, default=DEFAULT_SEGMENT_GAP_HOURS\n    )\n    parser.add_argument(\n        "--jump-speed-kn", type=float, default=DEFAULT_JUMP_SPEED_KN\n    )\n    parser.add_argument("--group-folds", type=int, default=DEFAULT_GROUP_FOLDS)\n    parser.add_argument("--expected-container-ships", type=int, default=DEFAULT_EXPECTED_CONTAINER_SHIPS,\n                        help="预期集装箱船数量；设为0可关闭数量核验。")\n    parser.add_argument("--duplicate-radius-nm", type=float, default=DEFAULT_DUPLICATE_RADIUS_NM)\n    parser.add_argument("--recent-track-hours", type=float, default=DEFAULT_RECENT_TRACK_HOURS)\n    parser.add_argument("--positionless-gap-hours", type=float, default=DEFAULT_POSITIONLESS_GAP_HOURS)\n    parser.add_argument("--max-reconnect-days", type=float, default=DEFAULT_MAX_RECONNECT_DAYS)\n    parser.add_argument("--ambiguity-ratio", type=float, default=DEFAULT_AMBIGUITY_RATIO)\n    parser.add_argument("--hard-max-fuel-t-5min", type=float, default=DEFAULT_HARD_MAX_FUEL_T_5MIN)\n    parser.add_argument("--max-sfoc-g-kwh", type=float, default=DEFAULT_MAX_SFOC_G_KWH)\n    parser.add_argument("--fuel-power-margin", type=float, default=DEFAULT_FUEL_POWER_MARGIN)\n    parser.add_argument(\n        "--min-static-match-rate",\n        type=float,\n        default=0.995,\n        help=(\n            "Bulk/Tanker清洗行匹配回原始行的最低比例。"\n            "默认0.995；低于该比例停止，防止静态字段错配。"\n        ),\n    )\n    parser.add_argument(\n        "--bulk-tanker-fuel-unit",\n        choices=["t_per_5min"],\n        default="t_per_5min",\n        help=(\n            "bulk/tanker fuel_t_5min或me_fo的单位。"\n            "本版本仅接受吨/5分钟，防止单位误用。"\n        ),\n    )\n    parser.add_argument(\n        "--negative-fuel-policy", choices=["reject", "absolute"], default="reject",\n        help="负油耗处理：reject为默认剔除；只有确认负号是流向约定时才使用absolute。",\n    )\n    parser.add_argument(\n        "--keep-staging",\n        action="store_true",\n        help="处理成功后保留标准化暂存CSV。默认删除以节省磁盘。",\n    )\n    return parser.parse_args()\n\n\ndef main() -> int:\n    args = parse_args()\n    if args.bulk_tanker_fuel_unit != "t_per_5min":\n        raise ValueError(\n            "本版本只允许bulk/tanker燃油单位为吨/5分钟。"\n        )\n\n    args.output_dir.mkdir(parents=True, exist_ok=True)\n    stage_dir = args.output_dir / "_staging"\n    temp_dir = args.output_dir / "_duckdb_temp"\n    stage_dir.mkdir(parents=True, exist_ok=True)\n    temp_dir.mkdir(parents=True, exist_ok=True)\n\n    cleaned_container = args.cleaned_container or discover_cleaned_file(\n        args.clean_dir, "container"\n    )\n    cleaned_bulk = args.cleaned_bulk or discover_cleaned_file(\n        args.clean_dir, "bulk"\n    )\n    cleaned_tanker = args.cleaned_tanker or discover_cleaned_file(\n        args.clean_dir, "tanker"\n    )\n\n    required = [\n        cleaned_container, cleaned_bulk, cleaned_tanker,\n        args.raw_container, args.raw_bulk, args.raw_tanker,\n    ]\n    for path in required:\n        if not path.is_file():\n            raise FileNotFoundError(f"文件不存在：{path}")\n\n    # 运行前先输出字段映射，特别核对bulk/tanker的fuel_t_5min或me_fo。\n    mapping_frames = [\n        field_mapping_rows(\n            cleaned_container,\n            ALIASES,\n            STANDARD_CLEAN_COLUMNS,\n            "cleaned_container",\n            "container",\n        ),\n        field_mapping_rows(\n            cleaned_bulk,\n            ALIASES,\n            STANDARD_CLEAN_COLUMNS,\n            "cleaned_bulk",\n            "bulk",\n        ),\n        field_mapping_rows(\n            cleaned_tanker,\n            ALIASES,\n            STANDARD_CLEAN_COLUMNS,\n            "cleaned_tanker",\n            "tanker",\n        ),\n        field_mapping_rows(\n            args.raw_container,\n            CONTAINER_RAW_ID_ALIASES,\n            list(CONTAINER_RAW_ID_ALIASES.keys()),\n            "raw_container",\n            "container",\n        ),\n        field_mapping_rows(\n            args.raw_bulk,\n            RAW_STATIC_ALIASES,\n            list(RAW_STATIC_ALIASES.keys()),\n            "raw_bulk",\n            "bulk",\n        ),\n        field_mapping_rows(\n            args.raw_tanker,\n            RAW_STATIC_ALIASES,\n            list(RAW_STATIC_ALIASES.keys()),\n            "raw_tanker",\n            "tanker",\n        ),\n    ]\n    pd.concat(mapping_frames, ignore_index=True).to_csv(\n        args.output_dir / "00_field_mapping.csv",\n        index=False,\n        encoding="utf-8-sig",\n    )\n    del mapping_frames\n\n    container_stage = stage_dir / "container_stage.csv"\n    bulk_stage = stage_dir / "bulk_stage.csv"\n    tanker_stage = stage_dir / "tanker_stage.csv"\n\n    validation_frames: List[pd.DataFrame] = []\n\n    print("\\n阶段A：分块生成标准化暂存文件")\n    validation_frames.append(\n        stage_container(\n            cleaned_container, args.raw_container, container_stage, args.chunksize\n        )\n    )\n    validation_frames.append(\n        stage_bulk_or_tanker(\n            cleaned_bulk,\n            args.raw_bulk,\n            bulk_stage,\n            "bulk",\n            args.chunksize,\n            args.memory_limit,\n            args.threads,\n            args.min_static_match_rate,\n        )\n    )\n    validation_frames.append(\n        stage_bulk_or_tanker(\n            cleaned_tanker,\n            args.raw_tanker,\n            tanker_stage,\n            "tanker",\n            args.chunksize,\n            args.memory_limit,\n            args.threads,\n            args.min_static_match_rate,\n        )\n    )\n    pd.concat(validation_frames, ignore_index=True).to_csv(\n        args.output_dir / "first_stage_validation_summary.csv",\n        index=False,\n        encoding="utf-8-sig",\n    )\n    del validation_frames\n    gc.collect()\n\n    db_path = args.output_dir / "post_cleaning_work.duckdb"\n    con = duckdb.connect(str(db_path))\n    con.execute(f"SET memory_limit=\'{quote_sql_text(args.memory_limit)}\'")\n    con.execute(f"SET threads={max(1, int(args.threads))}")\n    con.execute(f"SET temp_directory=\'{sql_path(temp_dir)}\'")\n    con.execute("SET preserve_insertion_order=false")\n\n    try:\n        print("\\n阶段B：DuckDB磁盘外排序、身份识别和聚合")\n        create_container_table(\n            con, container_stage, args.segment_gap_hours, args.jump_speed_kn,\n            args.hard_max_fuel_t_5min, args.max_sfoc_g_kwh,\n            args.fuel_power_margin, args.expected_container_ships,\n        )\n        copy_query_to_csv(\n            con,\n            "SELECT * FROM container_model_10min "\n            "ORDER BY pseudo_ship_group_id, timestamp_utc",\n            args.output_dir / "container_model_10min.csv",\n        )\n        copy_query_to_csv(\n            con,\n            "SELECT * FROM container_identity_summary ORDER BY delivery_date",\n            args.output_dir / "container_identity_summary.csv",\n        )\n        build_container_missing_report(\n            con, args.output_dir / "container_missing_intervals.csv"\n        )\n\n        create_bulk_tanker_tables(\n            con, "bulk", bulk_stage, stage_dir / "bulk_track_assignments.csv",\n            args.segment_gap_hours, args.jump_speed_kn,\n            args.duplicate_radius_nm, args.recent_track_hours,\n            args.positionless_gap_hours, args.max_reconnect_days,\n            args.ambiguity_ratio, args.hard_max_fuel_t_5min,\n            args.max_sfoc_g_kwh, args.fuel_power_margin,\n            args.negative_fuel_policy,\n        )\n        copy_query_to_csv(\n            con,\n            "SELECT * FROM bulk_identified ORDER BY pseudo_ship_group_id, timestamp_utc, source_row",\n            args.output_dir / "bulk_with_pseudo_ship_id.csv",\n        )\n        copy_query_to_csv(\n            con,\n            "SELECT * FROM bulk_model_10min ORDER BY pseudo_ship_group_id, timestamp_utc",\n            args.output_dir / "bulk_model_10min.csv",\n        )\n\n        create_bulk_tanker_tables(\n            con, "tanker", tanker_stage, stage_dir / "tanker_track_assignments.csv",\n            args.segment_gap_hours, args.jump_speed_kn,\n            args.duplicate_radius_nm, args.recent_track_hours,\n            args.positionless_gap_hours, args.max_reconnect_days,\n            args.ambiguity_ratio, args.hard_max_fuel_t_5min,\n            args.max_sfoc_g_kwh, args.fuel_power_margin,\n            args.negative_fuel_policy,\n        )\n        copy_query_to_csv(\n            con,\n            "SELECT * FROM tanker_identified ORDER BY pseudo_ship_group_id, timestamp_utc, source_row",\n            args.output_dir / "tanker_with_pseudo_ship_id.csv",\n        )\n        copy_query_to_csv(\n            con,\n            "SELECT * FROM tanker_model_10min ORDER BY pseudo_ship_group_id, timestamp_utc",\n            args.output_dir / "tanker_model_10min.csv",\n        )\n\n        print("\\n阶段C：输出船型、燃油质量和单位一致性报告")\n        build_pseudo_report(\n            con,\n            args.output_dir / "pseudo_ship_id_report.csv",\n        )\n        build_fuel_report(\n            con,\n            args.output_dir / "fuel_quality_report.csv",\n        )\n        build_ship_type_control_report(\n            con,\n            args.output_dir / "ship_type_control_report.csv",\n        )\n        build_fuel_window_report(\n            con,\n            args.output_dir / "fuel_10min_window_report.csv",\n        )\n        build_unit_consistency_report(\n            con,\n            args.output_dir / "fuel_unit_consistency_report.csv",\n        )\n\n        print("\\n阶段D：统一三船型、构造特征和固定划分")\n        counts = build_unified_tables(\n            con, args.output_dir, max(2, int(args.group_folds))\n        )\n\n        summary = {\n            "inputs": {\n                "cleaned_container": str(cleaned_container),\n                "cleaned_bulk": str(cleaned_bulk),\n                "cleaned_tanker": str(cleaned_tanker),\n                "raw_bulk": str(args.raw_bulk),\n                "raw_tanker": str(args.raw_tanker),\n            },\n            "output_dir": str(args.output_dir),\n            "row_counts": counts,\n            "settings": {\n                "chunksize": args.chunksize,\n                "duckdb_memory_limit": args.memory_limit,\n                "threads": args.threads,\n                "segment_gap_hours": args.segment_gap_hours,\n                "jump_speed_kn": args.jump_speed_kn,\n                "group_folds": args.group_folds,\n                "expected_container_ships": args.expected_container_ships,\n                "duplicate_radius_nm": args.duplicate_radius_nm,\n                "recent_track_hours": args.recent_track_hours,\n                "positionless_gap_hours": args.positionless_gap_hours,\n                "max_reconnect_days": args.max_reconnect_days,\n                "ambiguity_ratio": args.ambiguity_ratio,\n                "hard_max_fuel_t_5min": args.hard_max_fuel_t_5min,\n                "max_sfoc_g_kwh": args.max_sfoc_g_kwh,\n                "fuel_power_margin": args.fuel_power_margin,\n                "min_static_match_rate": args.min_static_match_rate,\n                "bulk_tanker_static_alignment": "hierarchical_one_to_one_business_key",\n                "negative_fuel_policy": args.negative_fuel_policy,\n                "bulk_tanker_fuel_input_unit": FUEL_INPUT_UNIT_5MIN,\n                "unified_fuel_output_unit": FUEL_OUTPUT_UNIT_10MIN,\n            },\n            "notes": [\n                "pseudo_ship_group_id是推定船舶组，不是真实IMO/MMSI。",\n                "bulk/tanker的fuel_t_5min与me_fo均按吨/5分钟解释。",\n                "两个不同且无重复的有效5分钟槽直接求和，生成吨/10分钟fuel_t_10min。",\n                "只有一个5分钟槽时不翻倍；重复槽不自动平均；fuel_t_10min保持缺失。",\n                "fuel_rate_kg_h=fuel_t_10min*6000，仅用于单位和物理审计，禁止作为模型特征。",\n                "船型由处理分支强制写入，并在合并前进行表级船型核验。",\n                "group_fold_id在container、bulk、tanker内部独立做负载均衡。",\n                "长时间缺失未做强制插值。",\n                "本脚本不执行最终完整案例删除；模型插补和标准化仍应仅在训练集内部拟合。",\n            ],\n        }\n        with (args.output_dir / "pipeline_summary.json").open(\n            "w", encoding="utf-8"\n        ) as handle:\n            json.dump(summary, handle, ensure_ascii=False, indent=2)\n\n        print("\\n第二阶段严格燃油与船型控制版完成。")\n        print(f"输出目录：{args.output_dir}")\n        for key, value in counts.items():\n            print(f"  {key}: {value:,}")\n\n    finally:\n        con.close()\n\n    if not args.keep_staging:\n        shutil.rmtree(stage_dir, ignore_errors=True)\n        shutil.rmtree(temp_dir, ignore_errors=True)\n        try:\n            db_path.unlink()\n        except OSError:\n            pass\n\n    return 0\n\n\nif __name__ == "__main__":\n    try:\n        raise SystemExit(main())\n    except Exception as exc:\n        print(f"\\n运行失败：{exc}", file=sys.stderr)\n        raise\n'

V7_SOURCE = '\n# -*- coding: utf-8 -*-\nfrom __future__ import annotations\n\nr"""\n三船型10分钟数据重建与特征生成（V7）。\n\n用途\n----\n1. 将散货船和油船的5分钟识别数据聚合为10分钟数据；\n2. 集装箱船默认读取已经形成的10分钟表，也可通过参数改为5分钟输入；\n3. 按变量物理含义采用不同聚合方法；\n4. 生成航向向量、相对风浪方向、相对风速和指定交互项；\n5. 输出分船型结果、统一结果、字段映射、聚合方法和质量审计。\n\n默认输入\n--------\ndata\\post_cleaning_v6_keyed_alignment\\bulk_with_pseudo_ship_id.csv\ndata\\post_cleaning_v6_keyed_alignment\\tanker_with_pseudo_ship_id.csv\ndata\\post_cleaning_v6_keyed_alignment\\container_model_10min.csv\n\n重要时间语义\n------------\nBulk/Tanker状态变量和燃油率默认按“记录时间为区间起点”处理：\n    [timestamp, timestamp + interval)\n\nV6中的 distance_nm_interval 是由上一位置到当前位置计算，默认按“记录时间为区间终点”\n归入10分钟窗口。可以用 --distance-interval-alignment starting 改为起点标记。\n\n燃油公式\n--------\n每条记录：\n    fuel_t_interval = fuel_rate_kg_h * interval_minutes / 60 / 1000\n\n10分钟窗口：\n    fuel_t_10min = sum(fuel_t_interval)\n\n正式 fuel_t_10min 仅在完整窗口内输出；部分窗口的积分值保留在\nfuel_t_10min_partial 供审计，不对缺失窗口翻倍。\n\n相对风速\n--------\n先根据方向约定将绝对风向转换为风矢量“去向”，再计算与船舶参考方向的夹角：\n    rel_wind_speed_kn =\n        sqrt(wind_speed_kn^2 + speed_kn^2\n             - 2 * wind_speed_kn * speed_kn * cos(relative_wind_angle))\n\n默认：\n    风向和浪向字段为“来自方向”（from）；\n    船舶参考方向优先使用 heading_deg，缺失时回退到 course_deg。\n"""\n\nimport argparse\nimport json\nimport math\nimport re\nimport shutil\nimport sys\nimport unicodedata\nfrom pathlib import Path\nfrom typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple\n\nimport pandas as pd\n\ntry:\n    import duckdb\nexcept ImportError as exc:\n    raise SystemExit(\n        "缺少duckdb。请运行：\\n"\n        "python -m pip install duckdb"\n    ) from exc\n\n\nCODE_VERSION = "2026-07-20-stage2-5min-to-10min-three-ship-features-v7"\n\nDEFAULT_BASE = Path(r"data\\post_cleaning_v6_keyed_alignment")\nDEFAULT_BULK = DEFAULT_BASE / "bulk_with_pseudo_ship_id.csv"\nDEFAULT_TANKER = DEFAULT_BASE / "tanker_with_pseudo_ship_id.csv"\nDEFAULT_CONTAINER = DEFAULT_BASE / "container_model_10min.csv"\nDEFAULT_OUTPUT = Path(r"data\\post_cleaning_v7_10min_features")\n\n\nALIASES: Dict[str, Sequence[str]] = {\n    "pseudo_ship_group_id": [\n        "pseudo_ship_group_id", "pseudo_ship_id", "ship_id", "vessel_id",\n        "imo", "mmsi",\n    ],\n    "trajectory_segment_id": [\n        "trajectory_segment_id", "segment_id", "trajectory_id", "voyage_segment_id",\n    ],\n    "id_confidence": ["id_confidence", "identity_confidence"],\n    "timestamp_utc": [\n        "timestamp_utc", "timestamp", "UTC时间(-)", "utc time", "time",\n        "datetime", "date_time",\n    ],\n    "source_row": ["source_row", "row_id", "original_row"],\n    "latitude_deg": ["latitude_deg", "latitude", "lat", "纬度"],\n    "longitude_deg": ["longitude_deg", "longitude", "lon", "lng", "经度"],\n    "design_draught_m": [\n        "design_draught_m", "design_draught", "draught_static", "设计吃水 / M",\n    ],\n    "deadweight_t": [\n        "deadweight_t", "deadweight", "deadweight_static", "Deadweight Tonnage",\n    ],\n    "service_speed_kn": [\n        "service_speed_kn", "service_speed", "service_speed_static", "ServiceSpeed",\n    ],\n    "main_engine_power_kw": [\n        "main_engine_power_kw", "main_engine_power", "power_static",\n        "Total KW Main Eng", "Main Propulsion Total Power Output",\n    ],\n    "ship_age_years": ["ship_age_years", "ship_age"],\n    "calendar_year": ["calendar_year", "calendar_year_input"],\n    "speed_kn": ["speed_kn", "speed", "sog", "对地航速(kn)"],\n    "course_deg": ["course_deg", "course", "cog", "direct", "航向角(deg)"],\n    "heading_deg": ["heading_deg", "heading", "hdg", "艏向角(deg)"],\n    "heading_sin": ["heading_sin"],\n    "heading_cos": ["heading_cos"],\n    "mean_draught_m": ["mean_draught_m", "mean_draught", "dmp"],\n    "trim_m": ["trim_m", "trim"],\n    "rudder_deg": ["rudder_deg", "rudder", "舵角(deg)"],\n    "wind_speed_kn": ["wind_speed_kn", "wind_speed", "wind_s"],\n    "wind_direction_deg": [\n        "wind_direction_deg", "wind_direction", "wind_d",\n    ],\n    "wave_height_m": ["wave_height_m", "wave_height", "wave_h"],\n    "wave_period_s": ["wave_period_s", "wave_period", "wave_p"],\n    "wave_direction_deg": [\n        "wave_direction_deg", "wave_direction", "wave_d",\n    ],\n    "surface_pressure_pa": [\n        "surface_pressure_pa", "surface_pressure", "surface_p",\n    ],\n    "surface_temperature_c": [\n        "surface_temperature_c", "surface_temperature", "surface_t", "ssurface_t",\n    ],\n    "fuel_rate_kg_h": [\n        "fuel_rate_kg_h", "fuel_rate_kg_h_raw",\n        "主机燃油消耗质量流量(kg/h)", "fuel rate",\n    ],\n    "fuel_t_5min": [\n        "fuel_t_5min", "fuel_t_5min_clean", "me_fo",\n    ],\n    "fuel_t_10min": ["fuel_t_10min"],\n    "distance_nm_interval": [\n        "distance_nm_interval", "distance_interval_nm", "distance_nm_5min",\n    ],\n    "distance_nm_10min": ["distance_nm_10min"],\n    "interval_minutes": [\n        "interval_minutes", "time_interval_minutes", "time_interval_min",\n        "delta_time_min", "dt_minutes",\n    ],\n    "fuel_raw_valid_flag": [\n        "fuel_raw_valid_flag", "fuel_valid_flag",\n    ],\n    "time_valid_flag": ["time_valid_flag"],\n    "position_valid_flag": ["position_valid_flag"],\n    "speed_valid_flag": ["speed_valid_flag"],\n    "valid_record_flag": ["valid_record_flag"],\n    "complete_window_flag": ["complete_window_flag"],\n    "coverage_ratio_10min": ["coverage_ratio_10min"],\n}\n\n\nCANONICAL_FIELDS = list(ALIASES.keys())\n\n\ndef normalize_name(value: Any) -> str:\n    text = unicodedata.normalize("NFKC", str(value))\n    text = text.replace("\\ufeff", "").strip().lower()\n    return re.sub(r"[^0-9a-z\\u4e00-\\u9fff]+", "", text)\n\n\ndef detect_encoding(path: Path) -> str:\n    for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk"):\n        try:\n            with path.open("r", encoding=encoding) as handle:\n                handle.read(65536)\n            return encoding\n        except UnicodeDecodeError:\n            continue\n    return "utf-8"\n\n\ndef read_header(path: Path) -> List[str]:\n    return list(\n        pd.read_csv(\n            path,\n            nrows=0,\n            encoding=detect_encoding(path),\n            low_memory=True,\n        ).columns\n    )\n\n\ndef resolve_columns(\n    columns: Sequence[str],\n    ship_type: str,\n    input_path: Path,\n) -> Tuple[Dict[str, str], pd.DataFrame]:\n    index: Dict[str, List[str]] = {}\n    for column in columns:\n        index.setdefault(normalize_name(column), []).append(str(column))\n\n    mapping: Dict[str, str] = {}\n    used: set[str] = set()\n    rows: List[Dict[str, Any]] = []\n\n    for canonical in CANONICAL_FIELDS:\n        selected: Optional[str] = None\n        matched_alias = ""\n        for alias in ALIASES[canonical]:\n            candidates = [\n                candidate\n                for candidate in index.get(normalize_name(alias), [])\n                if candidate not in used\n            ]\n            if candidates:\n                selected = candidates[0]\n                matched_alias = alias\n                break\n\n        if selected is not None:\n            mapping[canonical] = selected\n            used.add(selected)\n\n        rows.append({\n            "ship_type": ship_type,\n            "input_file": str(input_path),\n            "canonical_field": canonical,\n            "recognized_column": selected if selected is not None else "[not found]",\n            "matched_alias": matched_alias,\n            "match_method": "normalized_exact" if selected is not None else "not_found",\n        })\n\n    return mapping, pd.DataFrame(rows)\n\n\ndef sql_path(path: Path) -> str:\n    return str(path).replace("\\\\", "/").replace("\'", "\'\'")\n\n\ndef quote_identifier(name: str) -> str:\n    return \'"\' + str(name).replace(\'"\', \'""\') + \'"\'\n\n\ndef quote_text(value: str) -> str:\n    return str(value).replace("\'", "\'\'")\n\n\ndef typed_expr(\n    mapping: Dict[str, str],\n    canonical: str,\n    sql_type: str,\n    default_sql: str = "NULL",\n) -> str:\n    source = mapping.get(canonical)\n    if source is None:\n        return f"{default_sql}::{sql_type} AS {canonical}_input"\n    return (\n        f"try_cast({quote_identifier(source)} AS {sql_type}) "\n        f"AS {canonical}_input"\n    )\n\n\ndef text_expr(\n    mapping: Dict[str, str],\n    canonical: str,\n    default_sql: str = "NULL",\n) -> str:\n    source = mapping.get(canonical)\n    if source is None:\n        return f"{default_sql}::VARCHAR AS {canonical}_input"\n    return (\n        f"nullif(trim(cast({quote_identifier(source)} AS VARCHAR)), \'\') "\n        f"AS {canonical}_input"\n    )\n\n\ndef build_typed_source_sql(\n    path: Path,\n    mapping: Dict[str, str],\n) -> str:\n    expressions = [\n        typed_expr(mapping, "source_row", "BIGINT"),\n        text_expr(mapping, "pseudo_ship_group_id"),\n        text_expr(mapping, "trajectory_segment_id"),\n        text_expr(mapping, "id_confidence", "\'unknown\'"),\n        typed_expr(mapping, "timestamp_utc", "TIMESTAMP"),\n        typed_expr(mapping, "latitude_deg", "DOUBLE"),\n        typed_expr(mapping, "longitude_deg", "DOUBLE"),\n        typed_expr(mapping, "design_draught_m", "DOUBLE"),\n        typed_expr(mapping, "deadweight_t", "DOUBLE"),\n        typed_expr(mapping, "service_speed_kn", "DOUBLE"),\n        typed_expr(mapping, "main_engine_power_kw", "DOUBLE"),\n        typed_expr(mapping, "ship_age_years", "DOUBLE"),\n        typed_expr(mapping, "calendar_year", "INTEGER"),\n        typed_expr(mapping, "speed_kn", "DOUBLE"),\n        typed_expr(mapping, "course_deg", "DOUBLE"),\n        typed_expr(mapping, "heading_deg", "DOUBLE"),\n        typed_expr(mapping, "heading_sin", "DOUBLE"),\n        typed_expr(mapping, "heading_cos", "DOUBLE"),\n        typed_expr(mapping, "mean_draught_m", "DOUBLE"),\n        typed_expr(mapping, "trim_m", "DOUBLE"),\n        typed_expr(mapping, "rudder_deg", "DOUBLE"),\n        typed_expr(mapping, "wind_speed_kn", "DOUBLE"),\n        typed_expr(mapping, "wind_direction_deg", "DOUBLE"),\n        typed_expr(mapping, "wave_height_m", "DOUBLE"),\n        typed_expr(mapping, "wave_period_s", "DOUBLE"),\n        typed_expr(mapping, "wave_direction_deg", "DOUBLE"),\n        typed_expr(mapping, "surface_pressure_pa", "DOUBLE"),\n        typed_expr(mapping, "surface_temperature_c", "DOUBLE"),\n        typed_expr(mapping, "fuel_rate_kg_h", "DOUBLE"),\n        typed_expr(mapping, "fuel_t_5min", "DOUBLE"),\n        typed_expr(mapping, "fuel_t_10min", "DOUBLE"),\n        typed_expr(mapping, "distance_nm_interval", "DOUBLE"),\n        typed_expr(mapping, "distance_nm_10min", "DOUBLE"),\n        typed_expr(mapping, "interval_minutes", "DOUBLE"),\n        typed_expr(mapping, "fuel_raw_valid_flag", "INTEGER"),\n        typed_expr(mapping, "time_valid_flag", "INTEGER", "1"),\n        typed_expr(mapping, "position_valid_flag", "INTEGER", "1"),\n        typed_expr(mapping, "speed_valid_flag", "INTEGER", "1"),\n        typed_expr(mapping, "valid_record_flag", "INTEGER", "1"),\n        typed_expr(mapping, "complete_window_flag", "INTEGER"),\n        typed_expr(mapping, "coverage_ratio_10min", "DOUBLE"),\n    ]\n\n    return f"""\n        SELECT\n            {", ".join(expressions)}\n        FROM read_csv_auto(\n            \'{sql_path(path)}\',\n            header=true,\n            all_varchar=true,\n            sample_size=100000,\n            ignore_errors=false\n        )\n    """\n\n\ndef normalize_direction(expr: str) -> str:\n    return f"mod(mod({expr}, 360.0) + 360.0, 360.0)"\n\n\ndef weighted_avg(value: str, weight: str = "interval_minutes_effective") -> str:\n    return (\n        f"sum(({value}) * {weight}) FILTER (WHERE ({value}) IS NOT NULL) "\n        f"/ nullif(sum({weight}) FILTER (WHERE ({value}) IS NOT NULL), 0.0)"\n    )\n\n\ndef circular_component(\n    direction: str,\n    function: str,\n    weight: str = "interval_minutes_effective",\n) -> str:\n    return weighted_avg(f"{function}(radians({direction}))", weight)\n\n\ndef circular_deg(sin_expr: str, cos_expr: str) -> str:\n    return (\n        f"CASE WHEN {sin_expr} IS NULL OR {cos_expr} IS NULL THEN NULL "\n        f"ELSE {normalize_direction(f\'degrees(atan2({sin_expr}, {cos_expr}))\')} END"\n    )\n\n\ndef copy_query_to_csv(\n    con: "duckdb.DuckDBPyConnection",\n    query: str,\n    path: Path,\n) -> None:\n    path.parent.mkdir(parents=True, exist_ok=True)\n    con.execute(\n        f"""\n        COPY ({query})\n        TO \'{sql_path(path)}\'\n        (HEADER, DELIMITER \',\', FORMAT CSV)\n        """\n    )\n\n\ndef ensure_required(\n    mapping: Dict[str, str],\n    fields: Sequence[str],\n    ship_type: str,\n    path: Path,\n) -> None:\n    missing = [field for field in fields if field not in mapping]\n    if missing:\n        raise KeyError(\n            f"{ship_type}输入缺少必要字段：{missing}；文件={path}"\n        )\n\n\ndef create_raw_table(\n    con: "duckdb.DuckDBPyConnection",\n    ship_type: str,\n    path: Path,\n    mapping: Dict[str, str],\n) -> str:\n    table = f"{ship_type}_raw_typed"\n    con.execute(\n        f"CREATE OR REPLACE TABLE {table} AS "\n        + build_typed_source_sql(path, mapping)\n    )\n    return table\n\n\ndef create_5min_aggregate(\n    con: "duckdb.DuckDBPyConnection",\n    ship_type: str,\n    raw_table: str,\n    nominal_interval_minutes: float,\n    max_interval_minutes: float,\n    minimum_coverage_minutes: float,\n    maximum_coverage_minutes: float,\n    distance_alignment: str,\n) -> str:\n    prefix = ship_type\n    prepared = f"{prefix}_prepared"\n    dedup = f"{prefix}_dedup"\n    intervals = f"{prefix}_intervals"\n    state_window = f"{prefix}_state_window"\n    distance_window = f"{prefix}_distance_window"\n    base10 = f"{prefix}_base10"\n\n    # 输入燃油率优先；缺失时允许从吨/5分钟反推kg/h，但在输出中记录来源。\n    con.execute(\n        f"""\n        CREATE OR REPLACE TABLE {prepared} AS\n        SELECT\n            coalesce(\n                pseudo_ship_group_id_input,\n                \'{ship_type.upper()}_UNRESOLVED\'\n            ) AS pseudo_ship_group_id,\n            coalesce(\n                trajectory_segment_id_input,\n                coalesce(\n                    pseudo_ship_group_id_input,\n                    \'{ship_type.upper()}_UNRESOLVED\'\n                ) || \'_SEG0001\'\n            ) AS trajectory_segment_id,\n            coalesce(id_confidence_input, \'unknown\') AS id_confidence,\n            \'{ship_type}\'::VARCHAR AS ship_type,\n            coalesce(\n                source_row_input,\n                row_number() OVER ()\n            ) AS source_row,\n            timestamp_utc_input AS timestamp_utc,\n            latitude_deg_input AS latitude_deg,\n            longitude_deg_input AS longitude_deg,\n            design_draught_m_input AS design_draught_m,\n            deadweight_t_input AS deadweight_t,\n            service_speed_kn_input AS service_speed_kn,\n            main_engine_power_kw_input AS main_engine_power_kw,\n            ship_age_years_input AS ship_age_years,\n            calendar_year_input AS calendar_year_input,\n            speed_kn_input AS speed_kn,\n            {normalize_direction("course_deg_input")} AS course_deg,\n            {normalize_direction("heading_deg_input")} AS heading_deg,\n            mean_draught_m_input AS mean_draught_m,\n            trim_m_input AS trim_m,\n            rudder_deg_input AS rudder_deg,\n            wind_speed_kn_input AS wind_speed_kn,\n            {normalize_direction("wind_direction_deg_input")} AS wind_direction_deg,\n            wave_height_m_input AS wave_height_m,\n            wave_period_s_input AS wave_period_s,\n            {normalize_direction("wave_direction_deg_input")} AS wave_direction_deg,\n            surface_pressure_pa_input AS surface_pressure_pa,\n            surface_temperature_c_input AS surface_temperature_c,\n            CASE\n                WHEN fuel_rate_kg_h_input IS NOT NULL\n                    THEN fuel_rate_kg_h_input\n                WHEN fuel_t_5min_input IS NOT NULL\n                    THEN fuel_t_5min_input * 12000.0\n                ELSE NULL\n            END AS fuel_rate_kg_h_effective,\n            CASE\n                WHEN fuel_rate_kg_h_input IS NOT NULL\n                    THEN \'input_fuel_rate_kg_h\'\n                WHEN fuel_t_5min_input IS NOT NULL\n                    THEN \'derived_from_tonnes_per_5min\'\n                ELSE \'missing\'\n            END AS fuel_rate_source,\n            distance_nm_interval_input AS distance_nm_interval,\n            interval_minutes_input AS interval_minutes_input,\n            CASE\n                WHEN fuel_raw_valid_flag_input IS NOT NULL\n                    THEN fuel_raw_valid_flag_input\n                WHEN coalesce(\n                    fuel_rate_kg_h_input,\n                    fuel_t_5min_input * 12000.0\n                ) >= 0\n                    THEN 1\n                ELSE 0\n            END AS fuel_valid_source_flag,\n            coalesce(time_valid_flag_input, 1) AS time_valid_flag,\n            coalesce(position_valid_flag_input, 1) AS position_valid_flag,\n            coalesce(speed_valid_flag_input, 1) AS speed_valid_flag,\n            coalesce(valid_record_flag_input, 1) AS valid_record_flag\n        FROM {raw_table}\n        WHERE timestamp_utc_input IS NOT NULL\n        """\n    )\n\n    # 同一船舶、轨迹段和时间戳的重复行先压成一个槽，数值取均值，累计量取均值，\n    # 避免重复数据被双重求和；窗口完整标志仍会因重复而置0。\n    con.execute(\n        f"""\n        CREATE OR REPLACE TABLE {dedup} AS\n        SELECT\n            pseudo_ship_group_id,\n            trajectory_segment_id,\n            first(id_confidence ORDER BY source_row) AS id_confidence,\n            first(ship_type ORDER BY source_row) AS ship_type,\n            timestamp_utc,\n            min(source_row) AS source_row,\n            count(*) AS rows_at_timestamp,\n            avg(latitude_deg) AS latitude_deg,\n            avg(longitude_deg) AS longitude_deg,\n            first(design_draught_m ORDER BY source_row)\n                FILTER (WHERE design_draught_m IS NOT NULL) AS design_draught_m,\n            first(deadweight_t ORDER BY source_row)\n                FILTER (WHERE deadweight_t IS NOT NULL) AS deadweight_t,\n            first(service_speed_kn ORDER BY source_row)\n                FILTER (WHERE service_speed_kn IS NOT NULL) AS service_speed_kn,\n            first(main_engine_power_kw ORDER BY source_row)\n                FILTER (WHERE main_engine_power_kw IS NOT NULL)\n                AS main_engine_power_kw,\n            avg(ship_age_years) AS ship_age_years,\n            first(calendar_year_input ORDER BY source_row)\n                FILTER (WHERE calendar_year_input IS NOT NULL)\n                AS calendar_year_input,\n            avg(speed_kn) AS speed_kn,\n            {circular_deg(\n                "avg(sin(radians(course_deg)))",\n                "avg(cos(radians(course_deg)))"\n            )} AS course_deg,\n            {circular_deg(\n                "avg(sin(radians(heading_deg)))",\n                "avg(cos(radians(heading_deg)))"\n            )} AS heading_deg,\n            avg(mean_draught_m) AS mean_draught_m,\n            avg(trim_m) AS trim_m,\n            avg(rudder_deg) AS rudder_deg,\n            avg(wind_speed_kn) AS wind_speed_kn,\n            {circular_deg(\n                "avg(sin(radians(wind_direction_deg)))",\n                "avg(cos(radians(wind_direction_deg)))"\n            )} AS wind_direction_deg,\n            avg(wave_height_m) AS wave_height_m,\n            avg(wave_period_s) AS wave_period_s,\n            {circular_deg(\n                "avg(sin(radians(wave_direction_deg)))",\n                "avg(cos(radians(wave_direction_deg)))"\n            )} AS wave_direction_deg,\n            avg(surface_pressure_pa) AS surface_pressure_pa,\n            avg(surface_temperature_c) AS surface_temperature_c,\n            avg(fuel_rate_kg_h_effective)\n                FILTER (WHERE fuel_valid_source_flag = 1)\n                AS fuel_rate_kg_h_effective,\n            min(fuel_valid_source_flag) AS fuel_valid_source_flag,\n            string_agg(DISTINCT fuel_rate_source, \'|\') AS fuel_rate_source,\n            avg(distance_nm_interval) AS distance_nm_interval,\n            avg(interval_minutes_input) AS interval_minutes_input,\n            min(time_valid_flag) AS time_valid_flag,\n            min(position_valid_flag) AS position_valid_flag,\n            min(speed_valid_flag) AS speed_valid_flag,\n            min(valid_record_flag) AS valid_record_flag\n        FROM {prepared}\n        GROUP BY\n            pseudo_ship_group_id,\n            trajectory_segment_id,\n            timestamp_utc\n        """\n    )\n\n    con.execute(\n        f"""\n        CREATE OR REPLACE TABLE {intervals} AS\n        WITH sequenced AS (\n            SELECT *,\n                lead(timestamp_utc) OVER (\n                    PARTITION BY pseudo_ship_group_id, trajectory_segment_id\n                    ORDER BY timestamp_utc\n                ) AS next_timestamp,\n                lag(timestamp_utc) OVER (\n                    PARTITION BY pseudo_ship_group_id, trajectory_segment_id\n                    ORDER BY timestamp_utc\n                ) AS previous_timestamp\n            FROM {dedup}\n        )\n        SELECT *,\n            CASE\n                WHEN interval_minutes_input > 0\n                 AND interval_minutes_input <= {float(max_interval_minutes)}\n                    THEN interval_minutes_input\n                WHEN next_timestamp > timestamp_utc\n                 AND date_diff(\'second\', timestamp_utc, next_timestamp) / 60.0\n                        <= {float(max_interval_minutes)}\n                    THEN date_diff(\'second\', timestamp_utc, next_timestamp) / 60.0\n                ELSE {float(nominal_interval_minutes)}\n            END AS interval_minutes_effective,\n            CASE\n                WHEN interval_minutes_input > 0\n                 AND interval_minutes_input <= {float(max_interval_minutes)}\n                    THEN \'input_interval_minutes\'\n                WHEN next_timestamp > timestamp_utc\n                 AND date_diff(\'second\', timestamp_utc, next_timestamp) / 60.0\n                        <= {float(max_interval_minutes)}\n                    THEN \'next_timestamp_difference\'\n                ELSE \'nominal_interval_fallback\'\n            END AS interval_minutes_source,\n            time_bucket(INTERVAL \'10 minutes\', timestamp_utc) AS state_window_start,\n            CASE\n                WHEN \'{quote_text(distance_alignment)}\' = \'ending\'\n                    THEN time_bucket(\n                        INTERVAL \'10 minutes\',\n                        timestamp_utc - INTERVAL \'1 microsecond\'\n                    )\n                ELSE time_bucket(INTERVAL \'10 minutes\', timestamp_utc)\n            END AS distance_window_start\n        FROM sequenced\n        """\n    )\n\n    course_sin = circular_component("course_deg", "sin")\n    course_cos = circular_component("course_deg", "cos")\n    heading_sin = circular_component("heading_deg", "sin")\n    heading_cos = circular_component("heading_deg", "cos")\n    wind_sin = circular_component("wind_direction_deg", "sin")\n    wind_cos = circular_component("wind_direction_deg", "cos")\n    wave_sin = circular_component("wave_direction_deg", "sin")\n    wave_cos = circular_component("wave_direction_deg", "cos")\n\n    con.execute(\n        f"""\n        CREATE OR REPLACE TABLE {state_window} AS\n        WITH aggregated AS (\n            SELECT\n                pseudo_ship_group_id,\n                trajectory_segment_id,\n                first(id_confidence ORDER BY timestamp_utc) AS id_confidence,\n                first(ship_type ORDER BY timestamp_utc) AS ship_type,\n                state_window_start AS window_start_utc,\n                state_window_start + INTERVAL \'10 minutes\' AS window_end_utc,\n                min(timestamp_utc) AS source_timestamp_first_utc,\n                max(timestamp_utc) AS source_timestamp_last_utc,\n                count(*) AS source_slot_count_10min,\n                sum(rows_at_timestamp) AS source_row_count_10min,\n                sum(rows_at_timestamp - 1) AS duplicate_source_rows_10min,\n                sum(interval_minutes_effective) AS coverage_minutes_10min,\n                string_agg(DISTINCT interval_minutes_source, \'|\')\n                    AS interval_minutes_source,\n                {weighted_avg("latitude_deg")} AS latitude_deg,\n                {weighted_avg("longitude_deg")} AS longitude_deg,\n                first(design_draught_m ORDER BY timestamp_utc)\n                    FILTER (WHERE design_draught_m IS NOT NULL)\n                    AS design_draught_m,\n                first(deadweight_t ORDER BY timestamp_utc)\n                    FILTER (WHERE deadweight_t IS NOT NULL)\n                    AS deadweight_t,\n                first(service_speed_kn ORDER BY timestamp_utc)\n                    FILTER (WHERE service_speed_kn IS NOT NULL)\n                    AS service_speed_kn,\n                first(main_engine_power_kw ORDER BY timestamp_utc)\n                    FILTER (WHERE main_engine_power_kw IS NOT NULL)\n                    AS main_engine_power_kw,\n                {weighted_avg("ship_age_years")} AS ship_age_years,\n                coalesce(\n                    first(calendar_year_input ORDER BY timestamp_utc)\n                        FILTER (WHERE calendar_year_input IS NOT NULL),\n                    year(state_window_start)\n                ) AS calendar_year,\n                {weighted_avg("speed_kn")} AS speed_kn,\n                {course_sin} AS course_sin,\n                {course_cos} AS course_cos,\n                {heading_sin} AS heading_sin,\n                {heading_cos} AS heading_cos,\n                {weighted_avg("mean_draught_m")} AS mean_draught_m,\n                {weighted_avg("trim_m")} AS trim_m,\n                {weighted_avg("rudder_deg")} AS rudder_deg,\n                {weighted_avg("wind_speed_kn")} AS wind_speed_kn,\n                {wind_sin} AS wind_direction_sin,\n                {wind_cos} AS wind_direction_cos,\n                {weighted_avg("wave_height_m")} AS wave_height_m,\n                {weighted_avg("wave_period_s")} AS wave_period_s,\n                {wave_sin} AS wave_direction_sin,\n                {wave_cos} AS wave_direction_cos,\n                {weighted_avg("surface_pressure_pa")} AS surface_pressure_pa,\n                {weighted_avg("surface_temperature_c")}\n                    AS surface_temperature_c,\n                sum(\n                    fuel_rate_kg_h_effective\n                    * interval_minutes_effective / 60.0 / 1000.0\n                ) FILTER (\n                    WHERE fuel_valid_source_flag = 1\n                      AND fuel_rate_kg_h_effective IS NOT NULL\n                ) AS fuel_t_10min_partial,\n                sum(CASE\n                    WHEN fuel_valid_source_flag = 1\n                     AND fuel_rate_kg_h_effective IS NOT NULL\n                    THEN 1 ELSE 0 END\n                ) AS fuel_valid_source_count,\n                string_agg(DISTINCT fuel_rate_source, \'|\')\n                    AS fuel_rate_source,\n                min(time_valid_flag) AS time_valid_flag,\n                min(position_valid_flag) AS position_valid_flag,\n                min(speed_valid_flag) AS speed_valid_flag,\n                min(valid_record_flag) AS valid_record_flag\n            FROM {intervals}\n            GROUP BY\n                pseudo_ship_group_id,\n                trajectory_segment_id,\n                state_window_start\n        )\n        SELECT *,\n            {circular_deg("course_sin", "course_cos")} AS course_deg,\n            {circular_deg("heading_sin", "heading_cos")} AS heading_deg,\n            {circular_deg("wind_direction_sin", "wind_direction_cos")}\n                AS wind_direction_deg,\n            {circular_deg("wave_direction_sin", "wave_direction_cos")}\n                AS wave_direction_deg,\n            least(coverage_minutes_10min / 10.0, 1.0)\n                AS coverage_ratio_10min,\n            CASE\n                WHEN source_slot_count_10min = 2\n                 AND source_row_count_10min = 2\n                 AND duplicate_source_rows_10min = 0\n                 AND coverage_minutes_10min BETWEEN\n                     {float(minimum_coverage_minutes)}\n                     AND {float(maximum_coverage_minutes)}\n                 AND fuel_valid_source_count = 2\n                    THEN 1 ELSE 0\n            END AS complete_window_flag,\n            CASE\n                WHEN source_slot_count_10min < 2 THEN \'missing_5min_slot\'\n                WHEN source_slot_count_10min > 2 THEN \'more_than_2_slots\'\n                WHEN duplicate_source_rows_10min > 0 THEN \'duplicate_timestamp\'\n                WHEN coverage_minutes_10min < {float(minimum_coverage_minutes)}\n                    THEN \'insufficient_time_coverage\'\n                WHEN coverage_minutes_10min > {float(maximum_coverage_minutes)}\n                    THEN \'excess_time_coverage\'\n                WHEN fuel_valid_source_count < 2 THEN \'invalid_or_missing_fuel_rate\'\n                ELSE \'valid_complete_window\'\n            END AS aggregation_status,\n            CASE\n                WHEN source_slot_count_10min = 2\n                 AND source_row_count_10min = 2\n                 AND duplicate_source_rows_10min = 0\n                 AND coverage_minutes_10min BETWEEN\n                     {float(minimum_coverage_minutes)}\n                     AND {float(maximum_coverage_minutes)}\n                 AND fuel_valid_source_count = 2\n                THEN fuel_t_10min_partial\n                ELSE NULL\n            END AS fuel_t_10min\n        FROM aggregated\n        """\n    )\n\n    con.execute(\n        f"""\n        CREATE OR REPLACE TABLE {distance_window} AS\n        SELECT\n            pseudo_ship_group_id,\n            trajectory_segment_id,\n            distance_window_start AS window_start_utc,\n            sum(distance_nm_interval)\n                FILTER (WHERE distance_nm_interval IS NOT NULL)\n                AS distance_nm_10min,\n            sum(CASE WHEN distance_nm_interval IS NOT NULL THEN 1 ELSE 0 END)\n                AS distance_interval_count,\n            sum(rows_at_timestamp - 1) AS distance_duplicate_source_rows,\n            CASE\n                WHEN sum(CASE\n                    WHEN distance_nm_interval IS NOT NULL THEN 1 ELSE 0 END\n                ) = 2\n                 AND sum(rows_at_timestamp - 1) = 0\n                THEN 1 ELSE 0\n            END AS distance_complete_flag\n        FROM {intervals}\n        GROUP BY\n            pseudo_ship_group_id,\n            trajectory_segment_id,\n            distance_window_start\n        """\n    )\n\n    con.execute(\n        f"""\n        CREATE OR REPLACE TABLE {base10} AS\n        SELECT\n            s.pseudo_ship_group_id,\n            s.trajectory_segment_id,\n            s.id_confidence,\n            s.ship_type,\n            s.window_start_utc AS timestamp_utc,\n            s.window_start_utc,\n            s.window_end_utc,\n            s.source_timestamp_first_utc,\n            s.source_timestamp_last_utc,\n            \'10min_window_start\'::VARCHAR AS timestamp_semantics,\n            s.latitude_deg,\n            s.longitude_deg,\n            d.distance_nm_10min,\n            s.design_draught_m,\n            s.deadweight_t,\n            s.service_speed_kn,\n            s.main_engine_power_kw,\n            s.ship_age_years,\n            s.calendar_year,\n            s.speed_kn,\n            s.course_deg,\n            s.course_sin,\n            s.course_cos,\n            s.heading_deg,\n            s.heading_sin,\n            s.heading_cos,\n            s.mean_draught_m,\n            s.trim_m,\n            s.rudder_deg,\n            s.wind_speed_kn,\n            s.wind_direction_deg,\n            s.wind_direction_sin,\n            s.wind_direction_cos,\n            s.wave_height_m,\n            s.wave_period_s,\n            s.wave_direction_deg,\n            s.wave_direction_sin,\n            s.wave_direction_cos,\n            s.surface_pressure_pa,\n            s.surface_temperature_c,\n            s.fuel_t_10min,\n            s.fuel_t_10min_partial,\n            s.fuel_t_10min * 6000.0 AS fuel_rate_kg_h,\n            s.fuel_rate_source,\n            \'tonnes_per_10min\'::VARCHAR AS fuel_unit,\n            s.source_slot_count_10min,\n            s.source_row_count_10min,\n            s.duplicate_source_rows_10min,\n            s.coverage_minutes_10min,\n            s.coverage_ratio_10min,\n            s.complete_window_flag,\n            s.aggregation_status,\n            d.distance_interval_count,\n            d.distance_duplicate_source_rows,\n            coalesce(d.distance_complete_flag, 0) AS distance_complete_flag,\n            s.interval_minutes_source,\n            s.time_valid_flag,\n            s.position_valid_flag,\n            s.speed_valid_flag,\n            CASE WHEN s.fuel_t_10min IS NOT NULL AND s.fuel_t_10min >= 0\n                THEN 1 ELSE 0 END AS fuel_valid_flag,\n            s.valid_record_flag\n        FROM {state_window} s\n        LEFT JOIN {distance_window} d\n          USING (\n            pseudo_ship_group_id,\n            trajectory_segment_id,\n            window_start_utc\n          )\n        """\n    )\n    return base10\n\n\ndef create_10min_normalized(\n    con: "duckdb.DuckDBPyConnection",\n    ship_type: str,\n    raw_table: str,\n) -> str:\n    table = f"{ship_type}_base10"\n    con.execute(\n        f"""\n        CREATE OR REPLACE TABLE {table} AS\n        WITH prepared AS (\n            SELECT\n                coalesce(\n                    pseudo_ship_group_id_input,\n                    \'{ship_type.upper()}_UNRESOLVED\'\n                ) AS pseudo_ship_group_id,\n                coalesce(\n                    trajectory_segment_id_input,\n                    coalesce(\n                        pseudo_ship_group_id_input,\n                        \'{ship_type.upper()}_UNRESOLVED\'\n                    ) || \'_SEG0001\'\n                ) AS trajectory_segment_id,\n                coalesce(id_confidence_input, \'unknown\') AS id_confidence,\n                \'{ship_type}\'::VARCHAR AS ship_type,\n                timestamp_utc_input AS timestamp_utc,\n                latitude_deg_input AS latitude_deg,\n                longitude_deg_input AS longitude_deg,\n                design_draught_m_input AS design_draught_m,\n                deadweight_t_input AS deadweight_t,\n                service_speed_kn_input AS service_speed_kn,\n                main_engine_power_kw_input AS main_engine_power_kw,\n                ship_age_years_input AS ship_age_years,\n                coalesce(calendar_year_input, year(timestamp_utc_input))\n                    AS calendar_year,\n                speed_kn_input AS speed_kn,\n                {normalize_direction("course_deg_input")} AS course_deg,\n                {normalize_direction("heading_deg_input")} AS heading_deg,\n                heading_sin_input AS heading_sin_input,\n                heading_cos_input AS heading_cos_input,\n                mean_draught_m_input AS mean_draught_m,\n                trim_m_input AS trim_m,\n                rudder_deg_input AS rudder_deg,\n                wind_speed_kn_input AS wind_speed_kn,\n                {normalize_direction("wind_direction_deg_input")}\n                    AS wind_direction_deg,\n                wave_height_m_input AS wave_height_m,\n                wave_period_s_input AS wave_period_s,\n                {normalize_direction("wave_direction_deg_input")}\n                    AS wave_direction_deg,\n                surface_pressure_pa_input AS surface_pressure_pa,\n                surface_temperature_c_input AS surface_temperature_c,\n                coalesce(\n                    fuel_t_10min_input,\n                    fuel_rate_kg_h_input * 10.0 / 60.0 / 1000.0\n                ) AS fuel_t_10min,\n                CASE\n                    WHEN fuel_t_10min_input IS NOT NULL\n                        THEN \'input_fuel_t_10min\'\n                    WHEN fuel_rate_kg_h_input IS NOT NULL\n                        THEN \'derived_from_fuel_rate_kg_h_10min\'\n                    ELSE \'missing\'\n                END AS fuel_rate_source,\n                coalesce(\n                    distance_nm_10min_input,\n                    distance_nm_interval_input\n                ) AS distance_nm_10min,\n                coalesce(complete_window_flag_input, 1)\n                    AS complete_window_flag,\n                coalesce(coverage_ratio_10min_input, 1.0)\n                    AS coverage_ratio_10min,\n                coalesce(time_valid_flag_input, 1) AS time_valid_flag,\n                coalesce(position_valid_flag_input, 1) AS position_valid_flag,\n                coalesce(speed_valid_flag_input, 1) AS speed_valid_flag,\n                coalesce(valid_record_flag_input, 1) AS valid_record_flag\n            FROM {raw_table}\n            WHERE timestamp_utc_input IS NOT NULL\n        ), dedup AS (\n            SELECT\n                pseudo_ship_group_id,\n                trajectory_segment_id,\n                first(id_confidence ORDER BY timestamp_utc) AS id_confidence,\n                first(ship_type ORDER BY timestamp_utc) AS ship_type,\n                timestamp_utc,\n                count(*) AS source_row_count_10min,\n                avg(latitude_deg) AS latitude_deg,\n                avg(longitude_deg) AS longitude_deg,\n                avg(distance_nm_10min) AS distance_nm_10min,\n                first(design_draught_m ORDER BY timestamp_utc)\n                    FILTER (WHERE design_draught_m IS NOT NULL)\n                    AS design_draught_m,\n                first(deadweight_t ORDER BY timestamp_utc)\n                    FILTER (WHERE deadweight_t IS NOT NULL)\n                    AS deadweight_t,\n                first(service_speed_kn ORDER BY timestamp_utc)\n                    FILTER (WHERE service_speed_kn IS NOT NULL)\n                    AS service_speed_kn,\n                first(main_engine_power_kw ORDER BY timestamp_utc)\n                    FILTER (WHERE main_engine_power_kw IS NOT NULL)\n                    AS main_engine_power_kw,\n                avg(ship_age_years) AS ship_age_years,\n                first(calendar_year ORDER BY timestamp_utc) AS calendar_year,\n                avg(speed_kn) AS speed_kn,\n                {circular_deg(\n                    "avg(sin(radians(course_deg)))",\n                    "avg(cos(radians(course_deg)))"\n                )} AS course_deg,\n                avg(sin(radians(course_deg))) AS course_sin,\n                avg(cos(radians(course_deg))) AS course_cos,\n                {circular_deg(\n                    "avg(sin(radians(heading_deg)))",\n                    "avg(cos(radians(heading_deg)))"\n                )} AS heading_deg,\n                coalesce(\n                    avg(heading_sin_input),\n                    avg(sin(radians(heading_deg)))\n                ) AS heading_sin,\n                coalesce(\n                    avg(heading_cos_input),\n                    avg(cos(radians(heading_deg)))\n                ) AS heading_cos,\n                avg(mean_draught_m) AS mean_draught_m,\n                avg(trim_m) AS trim_m,\n                avg(rudder_deg) AS rudder_deg,\n                avg(wind_speed_kn) AS wind_speed_kn,\n                {circular_deg(\n                    "avg(sin(radians(wind_direction_deg)))",\n                    "avg(cos(radians(wind_direction_deg)))"\n                )} AS wind_direction_deg,\n                avg(sin(radians(wind_direction_deg)))\n                    AS wind_direction_sin,\n                avg(cos(radians(wind_direction_deg)))\n                    AS wind_direction_cos,\n                avg(wave_height_m) AS wave_height_m,\n                avg(wave_period_s) AS wave_period_s,\n                {circular_deg(\n                    "avg(sin(radians(wave_direction_deg)))",\n                    "avg(cos(radians(wave_direction_deg)))"\n                )} AS wave_direction_deg,\n                avg(sin(radians(wave_direction_deg)))\n                    AS wave_direction_sin,\n                avg(cos(radians(wave_direction_deg)))\n                    AS wave_direction_cos,\n                avg(surface_pressure_pa) AS surface_pressure_pa,\n                avg(surface_temperature_c) AS surface_temperature_c,\n                avg(fuel_t_10min) AS fuel_t_10min,\n                string_agg(DISTINCT fuel_rate_source, \'|\')\n                    AS fuel_rate_source,\n                min(complete_window_flag) AS complete_window_flag,\n                avg(coverage_ratio_10min) AS coverage_ratio_10min,\n                min(time_valid_flag) AS time_valid_flag,\n                min(position_valid_flag) AS position_valid_flag,\n                min(speed_valid_flag) AS speed_valid_flag,\n                min(valid_record_flag) AS valid_record_flag\n            FROM prepared\n            GROUP BY\n                pseudo_ship_group_id,\n                trajectory_segment_id,\n                timestamp_utc\n        )\n        SELECT\n            pseudo_ship_group_id,\n            trajectory_segment_id,\n            id_confidence,\n            ship_type,\n            timestamp_utc,\n            timestamp_utc AS window_start_utc,\n            timestamp_utc + INTERVAL \'10 minutes\' AS window_end_utc,\n            timestamp_utc AS source_timestamp_first_utc,\n            timestamp_utc AS source_timestamp_last_utc,\n            \'10min_input_timestamp\'::VARCHAR AS timestamp_semantics,\n            latitude_deg,\n            longitude_deg,\n            distance_nm_10min,\n            design_draught_m,\n            deadweight_t,\n            service_speed_kn,\n            main_engine_power_kw,\n            ship_age_years,\n            calendar_year,\n            speed_kn,\n            course_deg,\n            course_sin,\n            course_cos,\n            heading_deg,\n            heading_sin,\n            heading_cos,\n            mean_draught_m,\n            trim_m,\n            rudder_deg,\n            wind_speed_kn,\n            wind_direction_deg,\n            wind_direction_sin,\n            wind_direction_cos,\n            wave_height_m,\n            wave_period_s,\n            wave_direction_deg,\n            wave_direction_sin,\n            wave_direction_cos,\n            surface_pressure_pa,\n            surface_temperature_c,\n            fuel_t_10min,\n            fuel_t_10min AS fuel_t_10min_partial,\n            fuel_t_10min * 6000.0 AS fuel_rate_kg_h,\n            fuel_rate_source,\n            \'tonnes_per_10min\'::VARCHAR AS fuel_unit,\n            1::BIGINT AS source_slot_count_10min,\n            source_row_count_10min,\n            greatest(source_row_count_10min - 1, 0)\n                AS duplicate_source_rows_10min,\n            10.0::DOUBLE AS coverage_minutes_10min,\n            coverage_ratio_10min,\n            CASE\n                WHEN source_row_count_10min = 1\n                    THEN complete_window_flag\n                ELSE 0\n            END AS complete_window_flag,\n            CASE\n                WHEN source_row_count_10min > 1\n                    THEN \'duplicate_10min_timestamp\'\n                WHEN fuel_t_10min IS NULL\n                    THEN \'missing_fuel_t_10min\'\n                ELSE \'input_10min_row\'\n            END AS aggregation_status,\n            CASE WHEN distance_nm_10min IS NOT NULL THEN 1 ELSE 0 END\n                AS distance_interval_count,\n            greatest(source_row_count_10min - 1, 0)\n                AS distance_duplicate_source_rows,\n            CASE\n                WHEN distance_nm_10min IS NOT NULL\n                 AND source_row_count_10min = 1\n                THEN 1 ELSE 0\n            END AS distance_complete_flag,\n            \'input_10min\'::VARCHAR AS interval_minutes_source,\n            time_valid_flag,\n            position_valid_flag,\n            speed_valid_flag,\n            CASE WHEN fuel_t_10min IS NOT NULL AND fuel_t_10min >= 0\n                THEN 1 ELSE 0 END AS fuel_valid_flag,\n            valid_record_flag\n        FROM dedup\n        """\n    )\n    return table\n\n\ndef create_features(\n    con: "duckdb.DuckDBPyConnection",\n    ship_type: str,\n    base_table: str,\n    wind_direction_convention: str,\n    wave_direction_convention: str,\n    reference_direction: str,\n) -> str:\n    table = f"{ship_type}_final"\n    wind_shift = 180.0 if wind_direction_convention == "from" else 0.0\n    wave_shift = 180.0 if wave_direction_convention == "from" else 0.0\n\n    if reference_direction == "heading":\n        reference_sql = "coalesce(heading_deg, course_deg)"\n        source_sql = (\n            "CASE WHEN heading_deg IS NOT NULL THEN \'heading\' "\n            "WHEN course_deg IS NOT NULL THEN \'course_fallback\' ELSE \'missing\' END"\n        )\n    else:\n        reference_sql = "coalesce(course_deg, heading_deg)"\n        source_sql = (\n            "CASE WHEN course_deg IS NOT NULL THEN \'course\' "\n            "WHEN heading_deg IS NOT NULL THEN \'heading_fallback\' ELSE \'missing\' END"\n        )\n\n    con.execute(\n        f"""\n        CREATE OR REPLACE TABLE {table} AS\n        WITH directions AS (\n            SELECT *,\n                {reference_sql} AS reference_direction_deg,\n                {source_sql} AS reference_direction_source,\n                {normalize_direction(f"wind_direction_deg + {wind_shift}")}\n                    AS wind_vector_to_direction_deg,\n                {normalize_direction(f"wave_direction_deg + {wave_shift}")}\n                    AS wave_vector_to_direction_deg\n            FROM {base_table}\n        ), relative_angles AS (\n            SELECT *,\n                CASE\n                    WHEN wind_vector_to_direction_deg IS NULL\n                      OR reference_direction_deg IS NULL\n                    THEN NULL\n                    ELSE mod(\n                        mod(\n                            wind_vector_to_direction_deg\n                            - reference_direction_deg\n                            + 180.0,\n                            360.0\n                        ) + 360.0,\n                        360.0\n                    ) - 180.0\n                END AS relative_wind_angle_deg,\n                CASE\n                    WHEN wave_vector_to_direction_deg IS NULL\n                      OR reference_direction_deg IS NULL\n                    THEN NULL\n                    ELSE mod(\n                        mod(\n                            wave_vector_to_direction_deg\n                            - reference_direction_deg\n                            + 180.0,\n                            360.0\n                        ) + 360.0,\n                        360.0\n                    ) - 180.0\n                END AS relative_wave_angle_deg\n            FROM directions\n        ), derived AS (\n            SELECT *,\n                sin(radians(relative_wind_angle_deg))\n                    AS relative_wind_sin,\n                cos(radians(relative_wind_angle_deg))\n                    AS relative_wind_cos,\n                sin(radians(relative_wave_angle_deg))\n                    AS relative_wave_sin,\n                cos(radians(relative_wave_angle_deg))\n                    AS relative_wave_cos,\n                CASE\n                    WHEN wind_speed_kn IS NULL\n                      OR speed_kn IS NULL\n                      OR relative_wind_angle_deg IS NULL\n                    THEN NULL\n                    ELSE sqrt(\n                        greatest(\n                            0.0,\n                            pow(wind_speed_kn, 2)\n                            + pow(speed_kn, 2)\n                            - 2.0 * wind_speed_kn * speed_kn\n                              * cos(radians(relative_wind_angle_deg))\n                        )\n                    )\n                END AS rel_wind_speed_kn\n            FROM relative_angles\n        )\n        SELECT *,\n            rel_wind_speed_kn * speed_kn\n                AS rel_wind_speed_x_speed,\n            wave_height_m * speed_kn\n                AS wave_height_x_speed,\n            mean_draught_m * speed_kn\n                AS draught_x_speed,\n            trim_m * speed_kn\n                AS trim_x_speed,\n            mean_draught_m * trim_m\n                AS draught_x_trim\n        FROM derived\n        """\n    )\n    return table\n\n\ndef write_method_dictionary(path: Path) -> None:\n    rows = [\n        ("pseudo_ship_group_id", "group key", "不聚合；作为船舶组键"),\n        ("trajectory_segment_id", "group key", "不跨轨迹段聚合"),\n        ("timestamp_utc", "window label", "10分钟窗口起点"),\n        ("latitude_deg", "continuous state", "按有效时间间隔加权平均"),\n        ("longitude_deg", "continuous state", "按有效时间间隔加权平均"),\n        ("speed_kn", "continuous state", "按有效时间间隔加权平均"),\n        ("mean_draught_m", "continuous state", "按有效时间间隔加权平均"),\n        ("trim_m", "continuous state", "按有效时间间隔加权平均"),\n        ("rudder_deg", "continuous state", "按有效时间间隔加权平均"),\n        ("wind_speed_kn", "continuous state", "按有效时间间隔加权平均"),\n        ("wave_height_m", "continuous state", "按有效时间间隔加权平均"),\n        ("wave_period_s", "continuous state", "按有效时间间隔加权平均"),\n        ("surface_pressure_pa", "continuous state", "按有效时间间隔加权平均"),\n        ("surface_temperature_c", "continuous state", "按有效时间间隔加权平均"),\n        ("design_draught_m", "static", "窗口内第一个非缺失值"),\n        ("deadweight_t", "static", "窗口内第一个非缺失值"),\n        ("service_speed_kn", "static", "窗口内第一个非缺失值"),\n        ("main_engine_power_kw", "static", "窗口内第一个非缺失值"),\n        ("course_deg", "circular direction", "正弦和余弦按时间加权平均后atan2重建"),\n        ("heading_deg", "circular direction", "正弦和余弦按时间加权平均后atan2重建"),\n        ("heading_sin", "circular component", "原始heading正弦按时间加权平均"),\n        ("heading_cos", "circular component", "原始heading余弦按时间加权平均"),\n        ("wind_direction_deg", "circular direction", "正弦和余弦按时间加权平均后atan2重建"),\n        ("wave_direction_deg", "circular direction", "正弦和余弦按时间加权平均后atan2重建"),\n        (\n            "fuel_t_10min",\n            "interval total",\n            "sum(fuel_rate_kg_h × interval_minutes / 60 / 1000)，单位吨/10分钟",\n        ),\n        (\n            "distance_nm_10min",\n            "interval total",\n            "原始distance_nm_interval按10分钟窗口求和；默认记录时间为区间终点",\n        ),\n        ("validity flags", "quality flags", "窗口内取最小值，要求所有记录有效"),\n        (\n            "rel_wind_speed_kn",\n            "derived",\n            "绝对风矢量与船速矢量之差的模；用余弦定理计算",\n        ),\n        ("relative_wind_sin/cos", "derived", "相对风向角正弦/余弦"),\n        ("relative_wave_sin/cos", "derived", "相对浪向角正弦/余弦"),\n        ("interaction terms", "derived", "在10分钟聚合完成后相乘"),\n    ]\n    pd.DataFrame(\n        rows,\n        columns=["variable", "variable_type", "aggregation_or_formula"],\n    ).to_csv(path, index=False, encoding="utf-8-sig")\n\n\ndef build_audit(\n    con: "duckdb.DuckDBPyConnection",\n    ship_type: str,\n    raw_table: str,\n    final_table: str,\n    cadence: int,\n) -> Dict[str, Any]:\n    input_rows = int(con.execute(f"SELECT count(*) FROM {raw_table}").fetchone()[0])\n    row = con.execute(\n        f"""\n        SELECT\n            count(*) AS output_rows,\n            count(DISTINCT pseudo_ship_group_id) AS ship_count,\n            count(DISTINCT trajectory_segment_id) AS segment_count,\n            sum(complete_window_flag) AS complete_windows,\n            sum(CASE WHEN fuel_t_10min IS NOT NULL THEN 1 ELSE 0 END)\n                AS fuel_nonmissing,\n            sum(CASE WHEN distance_nm_10min IS NOT NULL THEN 1 ELSE 0 END)\n                AS distance_nonmissing,\n            sum(CASE WHEN duplicate_source_rows_10min > 0 THEN 1 ELSE 0 END)\n                AS duplicate_windows,\n            min(timestamp_utc) AS first_timestamp,\n            max(timestamp_utc) AS last_timestamp,\n            min(fuel_t_10min) AS fuel_min,\n            avg(fuel_t_10min) AS fuel_mean,\n            median(fuel_t_10min) AS fuel_median,\n            max(fuel_t_10min) AS fuel_max\n        FROM {final_table}\n        """\n    ).fetchone()\n    output_rows = int(row[0] or 0)\n    complete = int(row[3] or 0)\n    return {\n        "ship_type": ship_type,\n        "input_cadence_minutes": cadence,\n        "input_rows": input_rows,\n        "output_rows": output_rows,\n        "pseudo_ship_count": int(row[1] or 0),\n        "trajectory_segment_count": int(row[2] or 0),\n        "complete_windows": complete,\n        "complete_window_rate": complete / output_rows if output_rows else None,\n        "fuel_nonmissing_windows": int(row[4] or 0),\n        "distance_nonmissing_windows": int(row[5] or 0),\n        "duplicate_windows": int(row[6] or 0),\n        "first_timestamp": str(row[7]),\n        "last_timestamp": str(row[8]),\n        "fuel_t_10min_min": row[9],\n        "fuel_t_10min_mean": row[10],\n        "fuel_t_10min_median": row[11],\n        "fuel_t_10min_max": row[12],\n    }\n\n\ndef parse_args() -> argparse.Namespace:\n    parser = argparse.ArgumentParser(\n        description="Bulk/Tanker 5分钟转10分钟，并生成三船型方向、相对风浪和交互特征。"\n    )\n    parser.add_argument("--bulk-input", type=Path, default=DEFAULT_BULK)\n    parser.add_argument("--tanker-input", type=Path, default=DEFAULT_TANKER)\n    parser.add_argument("--container-input", type=Path, default=DEFAULT_CONTAINER)\n    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)\n    parser.add_argument(\n        "--container-cadence",\n        type=int,\n        choices=[5, 10],\n        default=10,\n        help="Container输入时间粒度；默认其输入已经是10分钟。",\n    )\n    parser.add_argument("--nominal-interval-minutes", type=float, default=5.0)\n    parser.add_argument("--max-interval-minutes", type=float, default=7.5)\n    parser.add_argument("--minimum-coverage-minutes", type=float, default=9.5)\n    parser.add_argument("--maximum-coverage-minutes", type=float, default=10.5)\n    parser.add_argument(\n        "--distance-interval-alignment",\n        choices=["ending", "starting"],\n        default="ending",\n        help="distance_nm_interval的时间标签。V6 identified数据默认是ending。",\n    )\n    parser.add_argument(\n        "--wind-direction-convention",\n        choices=["from", "toward"],\n        default="from",\n    )\n    parser.add_argument(\n        "--wave-direction-convention",\n        choices=["from", "toward"],\n        default="from",\n    )\n    parser.add_argument(\n        "--reference-direction",\n        choices=["heading", "course"],\n        default="heading",\n    )\n    parser.add_argument("--memory-limit", default="4GB")\n    parser.add_argument("--threads", type=int, default=1)\n    parser.add_argument("--keep-work-db", action="store_true")\n    return parser.parse_args()\n\n\ndef main() -> int:\n    args = parse_args()\n    args.output_dir.mkdir(parents=True, exist_ok=True)\n    temp_dir = args.output_dir / "_duckdb_temp"\n    temp_dir.mkdir(parents=True, exist_ok=True)\n\n    inputs = {\n        "bulk": (args.bulk_input, 5),\n        "tanker": (args.tanker_input, 5),\n        "container": (args.container_input, args.container_cadence),\n    }\n    for ship_type, (path, _) in inputs.items():\n        if not path.is_file():\n            raise FileNotFoundError(f"{ship_type}输入不存在：{path}")\n\n    mapping_frames: List[pd.DataFrame] = []\n    mappings: Dict[str, Dict[str, str]] = {}\n    for ship_type, (path, cadence) in inputs.items():\n        mapping, frame = resolve_columns(read_header(path), ship_type, path)\n        ensure_required(\n            mapping,\n            ["pseudo_ship_group_id", "timestamp_utc"],\n            ship_type,\n            path,\n        )\n        if cadence == 5:\n            if (\n                "fuel_rate_kg_h" not in mapping\n                and "fuel_t_5min" not in mapping\n            ):\n                raise KeyError(\n                    f"{ship_type} 5分钟输入既没有fuel_rate_kg_h，"\n                    f"也没有可反推燃油率的fuel_t_5min：{path}"\n                )\n        else:\n            if (\n                "fuel_t_10min" not in mapping\n                and "fuel_rate_kg_h" not in mapping\n            ):\n                raise KeyError(\n                    f"{ship_type} 10分钟输入既没有fuel_t_10min，"\n                    f"也没有fuel_rate_kg_h：{path}"\n                )\n        mappings[ship_type] = mapping\n        mapping_frames.append(frame)\n\n    pd.concat(mapping_frames, ignore_index=True).to_csv(\n        args.output_dir / "00_field_mapping.csv",\n        index=False,\n        encoding="utf-8-sig",\n    )\n    write_method_dictionary(args.output_dir / "01_aggregation_method_dictionary.csv")\n\n    db_path = args.output_dir / "stage2_10min_features_work.duckdb"\n    con = duckdb.connect(str(db_path))\n    con.execute(f"SET memory_limit=\'{quote_text(args.memory_limit)}\'")\n    con.execute(f"SET threads={max(1, int(args.threads))}")\n    con.execute(f"SET temp_directory=\'{sql_path(temp_dir)}\'")\n    con.execute("SET preserve_insertion_order=false")\n\n    final_tables: Dict[str, str] = {}\n    audits: List[Dict[str, Any]] = []\n\n    try:\n        for ship_type, (path, cadence) in inputs.items():\n            print(f"[{ship_type}] 读取：{path}")\n            raw_table = create_raw_table(\n                con,\n                ship_type,\n                path,\n                mappings[ship_type],\n            )\n            if cadence == 5:\n                base_table = create_5min_aggregate(\n                    con,\n                    ship_type,\n                    raw_table,\n                    nominal_interval_minutes=args.nominal_interval_minutes,\n                    max_interval_minutes=args.max_interval_minutes,\n                    minimum_coverage_minutes=args.minimum_coverage_minutes,\n                    maximum_coverage_minutes=args.maximum_coverage_minutes,\n                    distance_alignment=args.distance_interval_alignment,\n                )\n            else:\n                base_table = create_10min_normalized(\n                    con,\n                    ship_type,\n                    raw_table,\n                )\n\n            final_table = create_features(\n                con,\n                ship_type,\n                base_table,\n                wind_direction_convention=args.wind_direction_convention,\n                wave_direction_convention=args.wave_direction_convention,\n                reference_direction=args.reference_direction,\n            )\n            final_tables[ship_type] = final_table\n\n            output_path = args.output_dir / f"{ship_type}_model_10min_features.csv"\n            copy_query_to_csv(\n                con,\n                f"""\n                SELECT *\n                FROM {final_table}\n                ORDER BY\n                    pseudo_ship_group_id,\n                    trajectory_segment_id,\n                    timestamp_utc\n                """,\n                output_path,\n            )\n            audits.append(\n                build_audit(\n                    con,\n                    ship_type,\n                    raw_table,\n                    final_table,\n                    cadence,\n                )\n            )\n\n            copy_query_to_csv(\n                con,\n                f"""\n                SELECT\n                    aggregation_status,\n                    count(*) AS window_count,\n                    sum(CASE WHEN fuel_t_10min IS NOT NULL THEN 1 ELSE 0 END)\n                        AS fuel_nonmissing_count,\n                    sum(CASE WHEN distance_nm_10min IS NOT NULL THEN 1 ELSE 0 END)\n                        AS distance_nonmissing_count,\n                    min(fuel_t_10min) AS fuel_min,\n                    avg(fuel_t_10min) AS fuel_mean,\n                    median(fuel_t_10min) AS fuel_median,\n                    max(fuel_t_10min) AS fuel_max\n                FROM {final_table}\n                GROUP BY aggregation_status\n                ORDER BY window_count DESC\n                """,\n                args.output_dir / f"{ship_type}_aggregation_status.csv",\n            )\n\n        common_query = " UNION ALL ".join(\n            f"SELECT * FROM {final_tables[ship_type]}"\n            for ship_type in ("container", "bulk", "tanker")\n        )\n        con.execute(\n            f"CREATE OR REPLACE TABLE unified_ship_10min_features AS {common_query}"\n        )\n        copy_query_to_csv(\n            con,\n            """\n            SELECT *\n            FROM unified_ship_10min_features\n            ORDER BY\n                ship_type,\n                pseudo_ship_group_id,\n                trajectory_segment_id,\n                timestamp_utc\n            """,\n            args.output_dir / "unified_ship_10min_features.csv",\n        )\n\n        pd.DataFrame(audits).to_csv(\n            args.output_dir / "02_aggregation_audit.csv",\n            index=False,\n            encoding="utf-8-sig",\n        )\n\n        copy_query_to_csv(\n            con,\n            """\n            SELECT\n                ship_type,\n                count(*) AS row_count,\n                count(DISTINCT pseudo_ship_group_id) AS ship_count,\n                sum(complete_window_flag) AS complete_windows,\n                sum(CASE WHEN fuel_t_10min IS NOT NULL THEN 1 ELSE 0 END)\n                    AS fuel_nonmissing_windows,\n                sum(CASE WHEN distance_nm_10min IS NOT NULL THEN 1 ELSE 0 END)\n                    AS distance_nonmissing_windows,\n                avg(CASE\n                    WHEN abs(\n                        fuel_rate_kg_h - fuel_t_10min * 6000.0\n                    ) <= 1e-8\n                    THEN 1.0 ELSE 0.0 END\n                ) FILTER (\n                    WHERE fuel_t_10min IS NOT NULL\n                      AND fuel_rate_kg_h IS NOT NULL\n                ) AS fuel_unit_consistency_rate\n            FROM unified_ship_10min_features\n            GROUP BY ship_type\n            ORDER BY ship_type\n            """,\n            args.output_dir / "03_three_ship_type_quality_summary.csv",\n        )\n\n        summary = {\n            "code_version": CODE_VERSION,\n            "inputs": {\n                ship_type: {\n                    "path": str(path),\n                    "cadence_minutes": cadence,\n                }\n                for ship_type, (path, cadence) in inputs.items()\n            },\n            "output_dir": str(args.output_dir),\n            "settings": {\n                "nominal_interval_minutes": args.nominal_interval_minutes,\n                "max_interval_minutes": args.max_interval_minutes,\n                "minimum_coverage_minutes": args.minimum_coverage_minutes,\n                "maximum_coverage_minutes": args.maximum_coverage_minutes,\n                "distance_interval_alignment": args.distance_interval_alignment,\n                "wind_direction_convention": args.wind_direction_convention,\n                "wave_direction_convention": args.wave_direction_convention,\n                "reference_direction": args.reference_direction,\n                "memory_limit": args.memory_limit,\n                "threads": args.threads,\n            },\n            "formulas": {\n                "fuel_t_interval": (\n                    "fuel_rate_kg_h * interval_minutes / 60 / 1000"\n                ),\n                "fuel_t_10min": "sum(fuel_t_interval)",\n                "distance_nm_10min": "sum(distance_nm_interval)",\n                "rel_wind_speed_kn": (\n                    "sqrt(wind_speed_kn^2 + speed_kn^2 - "\n                    "2*wind_speed_kn*speed_kn*cos(relative_wind_angle))"\n                ),\n                "rel_wind_speed_x_speed": "rel_wind_speed_kn * speed_kn",\n                "wave_height_x_speed": "wave_height_m * speed_kn",\n                "draught_x_speed": "mean_draught_m * speed_kn",\n                "trim_x_speed": "trim_m * speed_kn",\n                "draught_x_trim": "mean_draught_m * trim_m",\n            },\n            "important_notes": [\n                "Bulk和Tanker按5分钟数据聚合；Container默认按既有10分钟表规范化。",\n                "方向变量使用正余弦向量平均，禁止直接算术平均角度。",\n                "正式fuel_t_10min只在完整双槽窗口输出；部分积分保留在fuel_t_10min_partial。",\n                "distance_nm_interval默认按区间终点时间归窗，适配V6 identified输出。",\n                "fuel_rate_kg_h是fuel_t_10min的审计换算，不应作为预测变量。",\n                "若Container只有聚合后的heading_deg而没有原始5分钟heading，heading_sin/cos只能由聚合角度反推；要得到真正的向量平均应输入Container 5分钟数据并设置--container-cadence 5。",\n            ],\n            "audits": audits,\n        }\n        (args.output_dir / "pipeline_summary.json").write_text(\n            json.dumps(summary, ensure_ascii=False, indent=2, default=str),\n            encoding="utf-8",\n        )\n\n        print("=" * 78)\n        print("三船型10分钟聚合与特征生成完成")\n        for audit in audits:\n            print(\n                f"{audit[\'ship_type\']}: input={audit[\'input_rows\']:,}, "\n                f"output={audit[\'output_rows\']:,}, "\n                f"complete={audit[\'complete_windows\']:,}"\n            )\n        print(f"输出目录：{args.output_dir}")\n        print("=" * 78)\n\n    finally:\n        con.close()\n\n    if not args.keep_work_db:\n        shutil.rmtree(temp_dir, ignore_errors=True)\n        try:\n            db_path.unlink()\n        except OSError:\n            pass\n\n    return 0\n\n\nif __name__ == "__main__":\n    try:\n        raise SystemExit(main())\n    except Exception as exc:\n        print(f"运行失败：{exc}", file=sys.stderr)\n        raise\n\n'

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\n运行失败：{exc}", file=sys.stderr)
        raise