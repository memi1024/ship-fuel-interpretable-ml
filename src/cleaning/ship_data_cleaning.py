from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


# ------------------------------
# 统一输出字段
# ------------------------------
KEY_COLUMNS = [
    "ship_type",
    "design_draught_m",
    "deadweight_t",
    "service_speed_kn",
    "main_engine_power_kw",
    "speed_kn",
    "course_deg",
    "mean_draught_m",
    "rudder_deg",
    "trim_m",
    "wind_speed_kn",
    "wind_direction_deg",
    "wave_height_m",
    "wave_period_s",
    "wave_direction_deg",
    "surface_pressure_pa",
    "surface_temperature_c",
    "fuel_rate_kg_h",
    "fuel_t_10min",
]

AUX_COLUMNS = [
    "ship_id",
    "timestamp_utc",
    "latitude_deg",
    "longitude_deg",
    "heading_deg",
    "distance_nm_interval",
    "distance_nm_10min",
    "fuel_source",
    "mean_draught_source",
    "speed_source",
    "source_interval_min",
    "data_interval_min",
    "calendar_year",
    "ship_age_years",
    "resample_count_10min",
    "coverage_ratio_10min",
    "source_file",
]

QC_COLUMNS = [
    "time_valid_flag",
    "duplicate_flag",
    "duplicate_count",
    "static_conflict_flag",
    "position_valid_flag",
    "speed_valid_flag",
    "draught_valid_flag",
    "weather_valid_flag",
    "fuel_valid_flag",
    "trim_consistency_flag",
    "imputed_flag",
    "outlier_flag",
    "valid_record_flag",
    "invalid_reason",
]

# 油船/散货船清洗层额外保留，后续10分钟聚合时使用
INTERMEDIATE_COLUMNS = ["fuel_t_5min"]
STANDARD_OUTPUT_COLUMNS = KEY_COLUMNS + AUX_COLUMNS + QC_COLUMNS + INTERMEDIATE_COLUMNS


# ------------------------------
# 字段别名映射
# ------------------------------
ALIASES: Dict[str, Dict[str, List[str]]] = {
    "container": {
        "ship_id": ["ship_id", "id", "IMO", "IMO Number", "MMSI"],
        "ship_type": ["ship type", "ShipType"],
        "timestamp": ["UTC时间(-)", "UTC时间", "timestamp", "time"],
        "delivery_date": ["交船日期", "DeliveryDate"],
        "design_draught": ["设计吃水 / M", "设计吃水/M", "设计吃水(m)", "设计吃水"],
        "deadweight": ["Deadweight Tonnage", "Deadweight", "DWT"],
        "service_speed": ["航速 / Kn", "航速/Kn", "设计航速", "ServiceSpeed"],
        "main_power": [
            "Main Propulsion Total Power Output",
            "Main Propulsion Power Output",
            "Total KW Main Eng",
        ],
        "speed": ["对地航速(kn)", "对地航速", "SOG", "sog"],
        "course": ["航向角(deg)", "航向角", "course", "direct"],
        "heading": ["艏向角(deg)", "艏向角", "heading", "hdg"],
        "rudder": ["舵角(deg)", "舵角", "rudder"],
        "fore_draught": ["修正后的艏吃水(m)", "修正后的艏吃水", "艏吃水", "df"],
        "aft_draught": ["修正后的艉吃水(m)", "修正后的艉吃水", "艉吃水", "da"],
        "wind_speed": ["wind_s", "10米风速", "风速"],
        "wind_direction": ["wind_d", "10米气象风向", "风向"],
        "wave_height": ["wave_h", "综合有效波高", "浪高"],
        "wave_period": ["wave_p", "平均波周期", "浪周期"],
        "wave_direction": ["wave_d", "平均波向", "浪向"],
        "surface_pressure": ["surface_p", "表面气压", "气压"],
        "surface_temperature": ["ssurface_t", "surface_t", "海表温度", "表面温度"],
        "fuel_rate": ["主机燃油消耗质量流量(kg/h)", "主机燃油消耗质量流量", "fuel_rate_kg_h"],
        "shaft_power": ["轴功率(kW)", "轴功率", "shaft_power"],
        "sfoc": ["主机SFOC(g/kWh)", "主机SFOC", "SFOC", "sfoc"],
        "lat": ["纬度(deg)", "纬度", "lat", "latitude"],
        "lon": ["经度(deg)", "经度", "lon", "longitude"],
    },
    "common_english": {
        "ship_id": ["id", "ship_id", "IMO", "MMSI"],
        "ship_type": ["ShipType", "ship type", "ship_type"],
        "timestamp": ["timestamp", "UTC时间(-)", "time"],
        "delivery_date": ["DeliveryDate", "交船日期"],
        "design_draught": ["Draught", "design_draught_m"],
        "deadweight": ["Deadweight", "Deadweight Tonnage", "DWT"],
        "service_speed": ["ServiceSpeed", "service_speed_kn"],
        "main_power": ["Total KW Main Eng", "Main Propulsion Total Power Output"],
        # 按用户要求：油船和散货船的sog为空，统一使用speed
        "speed": ["speed"],
        "course": ["direct"],
        "heading": ["hdg"],
        "rudder": ["rudder"],
        "fore_draught": ["df"],
        "aft_draught": ["da"],
        "mean_draught_candidate": ["dmp"],
        "trim": ["trim"],
        "wind_speed": ["wind_s"],
        "wind_direction": ["wind_d"],
        "wave_height": ["wave_h"],
        "wave_period": ["wave_p"],
        "wave_direction": ["wave_d"],
        "surface_pressure": ["surface_p"],
        "surface_temperature": ["surface_t"],
        "fuel_5min": ["me_fo"],
        "lat": ["lat"],
        "lon": ["lon"],
    },
}

CONTAINER_SIGNATURES = {
    "UTC时间(-)",
    "对地航速(kn)",
    "修正后的艏吃水(m)",
    "Main Propulsion Total Power Output",
}

ENGLISH_SIGNATURES = {
    "ShipType",
    "Deadweight",
    "Total KW Main Eng",
    "ServiceSpeed",
    "timestamp",
    "me_fo",
    "speed",
    "direct",
}

TANKER_WORDS = ["tanker", "oil tanker", "chemical tanker", "油船", "油轮", "成品油", "原油"]
BULK_WORDS = ["bulk", "bulk carrier", "bulker", "散货", "干散货"]
CONTAINER_WORDS = ["container", "containership", "集装箱"]


# ------------------------------
# 工具函数
# ------------------------------
def normalize_name(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value)).strip().lower()
    # 去除常见分隔符和标点，以便识别不完全一致的字段名
    text = re.sub(r"[\s_\-\/\\()（）\[\]{}]+", "", text)
    text = text.replace("·", "").replace(".", "")
    return text


def build_column_lookup(columns: Iterable[str]) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    for col in columns:
        key = normalize_name(col)
        if key not in lookup:
            lookup[key] = col
    return lookup


def find_column(columns: Iterable[str], aliases: Iterable[str]) -> Optional[str]:
    lookup = build_column_lookup(columns)
    for alias in aliases:
        actual = lookup.get(normalize_name(alias))
        if actual is not None:
            return actual
    return None


def get_series(df: pd.DataFrame, aliases: Iterable[str], numeric: bool = False) -> pd.Series:
    col = find_column(df.columns, aliases)
    if col is None:
        return pd.Series(np.nan, index=df.index, dtype="float64" if numeric else "object")
    series = df[col]
    if numeric:
        if series.dtype == object:
            series = series.astype(str).str.replace(",", "", regex=False).str.strip()
            series = series.replace({"": np.nan, "nan": np.nan, "None": np.nan, "null": np.nan})
        return pd.to_numeric(series, errors="coerce")
    return series


def wrap_angle(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.mod(360.0)


def standardize_pressure_pa(series: pd.Series) -> Tuple[pd.Series, str]:
    s = pd.to_numeric(series, errors="coerce")
    median = s.dropna().median()
    if pd.isna(median):
        return s, "unknown"
    if 800 <= median <= 1200:  # hPa
        return s * 100.0, "hPa_to_Pa"
    if 80 <= median <= 120:  # kPa
        return s * 1000.0, "kPa_to_Pa"
    return s, "already_Pa_or_unknown"


def standardize_temperature_c(series: pd.Series) -> Tuple[pd.Series, str]:
    s = pd.to_numeric(series, errors="coerce")
    median = s.dropna().median()
    if pd.isna(median):
        return s, "unknown"
    if median > 150:  # Kelvin
        return s - 273.15, "K_to_C"
    return s, "already_C_or_unknown"


def haversine_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    if any(pd.isna(v) for v in [lat1, lon1, lat2, lon2]):
        return np.nan
    if not (-90 <= lat1 <= 90 and -90 <= lat2 <= 90 and -180 <= lon1 <= 180 and -180 <= lon2 <= 180):
        return np.nan
    radius_nm = 3440.065
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * radius_nm * math.asin(min(1.0, math.sqrt(a)))


def detect_encoding(path: Path) -> str:
    candidates = ["utf-8-sig", "utf-8", "gb18030", "gbk", "latin1"]
    for encoding in candidates:
        try:
            with path.open("r", encoding=encoding) as f:
                f.read(65536)
            return encoding
        except UnicodeDecodeError:
            continue
    return "latin1"


def detect_separator(path: Path, encoding: str) -> str:
    with path.open("r", encoding=encoding, errors="replace") as f:
        sample = f.read(65536)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", "\t", ";", "|"])
        return dialect.delimiter
    except csv.Error:
        return ","


def read_sample(path: Path, encoding: Optional[str], sep: Optional[str], nrows: int = 200) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt", ".tsv"}:
        enc = encoding or detect_encoding(path)
        delimiter = sep or detect_separator(path, enc)
        return pd.read_csv(path, encoding=enc, sep=delimiter, nrows=nrows, low_memory=False)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, nrows=nrows)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path).head(nrows)
    raise ValueError(f"不支持的文件格式：{suffix}")


def classify_ship_type(path: Path, sample: pd.DataFrame, override: str) -> str:
    if override != "auto":
        return override

    columns = set(sample.columns.astype(str))
    normalized_columns = {normalize_name(c) for c in columns}

    if sum(normalize_name(c) in normalized_columns for c in CONTAINER_SIGNATURES) >= 2:
        return "container"

    filename = path.stem.lower()
    shiptype_col = find_column(sample.columns, ALIASES["common_english"]["ship_type"])
    values = ""
    if shiptype_col:
        values = " ".join(sample[shiptype_col].dropna().astype(str).str.lower().unique()[:50])

    combined_text = f"{filename} {values}"
    if any(word.lower() in combined_text for word in TANKER_WORDS):
        return "tanker"
    if any(word.lower() in combined_text for word in BULK_WORDS):
        return "bulk"
    if any(word.lower() in combined_text for word in CONTAINER_WORDS):
        return "container"

    if sum(normalize_name(c) in normalized_columns for c in ENGLISH_SIGNATURES) >= 5:
        raise ValueError(
            "检测到油船/散货船共用的英文结构，但无法区分船型。"
            "请根据文件实际类型增加参数 --ship-type tanker 或 --ship-type bulk。"
        )

    raise ValueError("无法自动识别船型，请使用 --ship-type container/tanker/bulk 显式指定。")


@dataclass
class StreamState:
    last_timestamp: Dict[str, pd.Timestamp] = field(default_factory=dict)
    last_position: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    static_values: Dict[str, Dict[str, object]] = field(default_factory=dict)


@dataclass
class CleaningStats:
    rows_in: int = 0
    rows_out: int = 0
    duplicate_rows: int = 0
    invalid_rows: int = 0
    measured_fuel_rows: int = 0
    calculated_fuel_rows: int = 0
    missing_fuel_rows: int = 0
    pressure_conversion: str = "unknown"
    temperature_conversion: str = "unknown"
    missing_counts: Dict[str, int] = field(default_factory=dict)

    def update_missing(self, df: pd.DataFrame) -> None:
        for col in KEY_COLUMNS:
            count = int(df[col].isna().sum()) if col in df.columns else len(df)
            self.missing_counts[col] = self.missing_counts.get(col, 0) + count


# ------------------------------
# 清洗核心
# ------------------------------
def create_standardized_chunk(
    raw: pd.DataFrame,
    ship_kind: str,
    source_file: str,
    state: StreamState,
    me_fo_unit: str,
    explicit_ship_id: Optional[str],
    stats: CleaningStats,
) -> pd.DataFrame:
    raw = raw.copy()
    raw.columns = [str(c).strip() for c in raw.columns]
    schema = ALIASES["container"] if ship_kind == "container" else ALIASES["common_english"]

    std = pd.DataFrame(index=raw.index)

    # 船舶编号：优先原字段，其次显式参数，最后使用文件名
    source_ship_id = get_series(raw, schema["ship_id"], numeric=False)
    fallback_id = explicit_ship_id or Path(source_file).stem
    std["ship_id"] = source_ship_id.astype("string").replace({"<NA>": pd.NA, "nan": pd.NA}).fillna(fallback_id)
    std["ship_id"] = std["ship_id"].astype(str).str.strip().replace({"": fallback_id, "nan": fallback_id})

    source_ship_type = get_series(raw, schema["ship_type"], numeric=False)
    fixed_type = {"container": "container_ship", "tanker": "tanker", "bulk": "bulk_carrier"}[ship_kind]
    std["ship_type"] = source_ship_type.astype("string").replace({"<NA>": pd.NA, "nan": pd.NA}).fillna(fixed_type)

    # 时间
    timestamp_raw = get_series(raw, schema["timestamp"], numeric=False)
    timestamp = pd.to_datetime(timestamp_raw, errors="coerce", utc=True)
    std["timestamp_utc"] = timestamp
    std["time_valid_flag"] = timestamp.notna().astype("int8")
    std["calendar_year"] = timestamp.dt.year.astype("Int64")

    # 静态变量
    std["design_draught_m"] = get_series(raw, schema["design_draught"], numeric=True)
    std["deadweight_t"] = get_series(raw, schema["deadweight"], numeric=True)
    std["service_speed_kn"] = get_series(raw, schema["service_speed"], numeric=True)
    std["main_engine_power_kw"] = get_series(raw, schema["main_power"], numeric=True)

    # 动态变量
    std["speed_kn"] = get_series(raw, schema["speed"], numeric=True)
    std["speed_source"] = "SOG" if ship_kind == "container" else "speed"
    std["course_deg"] = wrap_angle(get_series(raw, schema["course"], numeric=True))
    std["heading_deg"] = wrap_angle(get_series(raw, schema["heading"], numeric=True))
    std["rudder_deg"] = get_series(raw, schema["rudder"], numeric=True)

    # 吃水与纵倾
    fore = get_series(raw, schema["fore_draught"], numeric=True)
    aft = get_series(raw, schema["aft_draught"], numeric=True)
    calculated_mean = (fore + aft) / 2.0
    calculated_trim = aft - fore  # 艉倾为正

    if ship_kind == "container":
        std["mean_draught_m"] = calculated_mean
        std["mean_draught_source"] = np.where(calculated_mean.notna(), "fore_aft_calculated", "missing")
        std["trim_m"] = calculated_trim
        std["trim_consistency_flag"] = np.where(calculated_trim.notna(), 1, 0).astype("int8")
    else:
        mean_candidate = get_series(raw, schema.get("mean_draught_candidate", []), numeric=True)
        # 优先使用艏艉平均；缺失时才使用dmp候选值，避免未经确认直接覆盖
        std["mean_draught_m"] = calculated_mean.combine_first(mean_candidate)
        std["mean_draught_source"] = np.select(
            [calculated_mean.notna(), calculated_mean.isna() & mean_candidate.notna()],
            ["fore_aft_calculated", "dmp_fallback"],
            default="missing",
        )
        trim_raw = get_series(raw, schema["trim"], numeric=True)
        std["trim_m"] = trim_raw.combine_first(calculated_trim)
        tolerance = 0.20  # m，可按数据质量再调整
        comparable = trim_raw.notna() & calculated_trim.notna()
        consistent_same = (trim_raw - calculated_trim).abs() <= tolerance
        consistent_opposite = (trim_raw + calculated_trim).abs() <= tolerance
        std["trim_consistency_flag"] = np.select(
            [~comparable, consistent_same, consistent_opposite],
            [0, 1, -1],  # -1表示很可能符号方向相反
            default=0,
        ).astype("int8")

    # 海气象变量
    std["wind_speed_kn"] = get_series(raw, schema["wind_speed"], numeric=True)
    std["wind_direction_deg"] = wrap_angle(get_series(raw, schema["wind_direction"], numeric=True))
    std["wave_height_m"] = get_series(raw, schema["wave_height"], numeric=True)
    std["wave_period_s"] = get_series(raw, schema["wave_period"], numeric=True)
    std["wave_direction_deg"] = wrap_angle(get_series(raw, schema["wave_direction"], numeric=True))

    pressure_raw = get_series(raw, schema["surface_pressure"], numeric=True)
    std["surface_pressure_pa"], pressure_note = standardize_pressure_pa(pressure_raw)
    temperature_raw = get_series(raw, schema["surface_temperature"], numeric=True)
    std["surface_temperature_c"], temperature_note = standardize_temperature_c(temperature_raw)
    if stats.pressure_conversion == "unknown":
        stats.pressure_conversion = pressure_note
    if stats.temperature_conversion == "unknown":
        stats.temperature_conversion = temperature_note

    # 经纬度
    std["latitude_deg"] = get_series(raw, schema["lat"], numeric=True)
    std["longitude_deg"] = get_series(raw, schema["lon"], numeric=True)

    # 油耗
    if ship_kind == "container":
        measured_rate = get_series(raw, schema["fuel_rate"], numeric=True)
        shaft_power = get_series(raw, schema["shaft_power"], numeric=True)
        sfoc = get_series(raw, schema["sfoc"], numeric=True)
        calculated_rate = shaft_power * sfoc / 1000.0
        std["fuel_rate_kg_h"] = measured_rate.combine_first(calculated_rate)
        std["fuel_source"] = np.select(
            [measured_rate.notna(), measured_rate.isna() & calculated_rate.notna()],
            ["measured", "calculated"],
            default="missing",
        )
        std["fuel_t_10min"] = std["fuel_rate_kg_h"] / 6000.0
        std["fuel_t_5min"] = np.nan
    else:
        me_fo = get_series(raw, schema["fuel_5min"], numeric=True)
        if me_fo_unit == "t/5min":
            std["fuel_t_5min"] = me_fo
            std["fuel_rate_kg_h"] = me_fo * 12000.0
        elif me_fo_unit == "kg/5min":
            std["fuel_t_5min"] = me_fo / 1000.0
            std["fuel_rate_kg_h"] = me_fo * 12.0
        elif me_fo_unit == "kg/h":
            std["fuel_t_5min"] = me_fo / 12000.0
            std["fuel_rate_kg_h"] = me_fo
        else:
            raise ValueError(f"不支持的me_fo单位：{me_fo_unit}")
        std["fuel_source"] = np.where(me_fo.notna(), "measured", "missing")
        # 第一阶段不把单条5分钟记录伪装成10分钟油耗
        std["fuel_t_10min"] = np.nan

    # 交船日期与船龄
    delivery_raw = get_series(raw, schema["delivery_date"], numeric=False)
    delivery = pd.to_datetime(delivery_raw, errors="coerce", utc=True)
    age_days = (timestamp - delivery).dt.total_seconds() / 86400.0
    std["ship_age_years"] = age_days / 365.25

    # 排序后计算重复、采样间隔与区间航程
    # 注意：跨块计算依赖输入文件整体大致按船舶和时间排序
    std["_original_index"] = np.arange(len(std))
    std = std.sort_values(["ship_id", "timestamp_utc", "_original_index"], kind="mergesort", na_position="last")

    std["duplicate_count"] = std.groupby(["ship_id", "timestamp_utc"], dropna=False)["ship_id"].transform("size").astype("int32")
    std["duplicate_flag"] = (std["duplicate_count"] > 1).astype("int8")

    std["source_interval_min"] = std.groupby("ship_id", sort=False)["timestamp_utc"].diff().dt.total_seconds() / 60.0
    std["distance_nm_interval"] = np.nan

    for ship_id, group_idx in std.groupby("ship_id", sort=False).groups.items():
        indices = list(group_idx)
        previous_time = state.last_timestamp.get(str(ship_id))
        previous_position = state.last_position.get(str(ship_id))

        for pos, idx in enumerate(indices):
            current_time = std.at[idx, "timestamp_utc"]
            current_lat = std.at[idx, "latitude_deg"]
            current_lon = std.at[idx, "longitude_deg"]

            if pos == 0 and previous_time is not None and pd.notna(current_time):
                std.at[idx, "source_interval_min"] = (current_time - previous_time).total_seconds() / 60.0
                if current_time == previous_time:
                    std.at[idx, "duplicate_flag"] = 1
                    std.at[idx, "duplicate_count"] = max(2, int(std.at[idx, "duplicate_count"]))

            if pos == 0:
                prev_pos = previous_position
            else:
                prev_idx = indices[pos - 1]
                prev_pos = (std.at[prev_idx, "latitude_deg"], std.at[prev_idx, "longitude_deg"])

            if prev_pos is not None:
                std.at[idx, "distance_nm_interval"] = haversine_nm(
                    prev_pos[0], prev_pos[1], current_lat, current_lon
                )

        valid_times = std.loc[indices, "timestamp_utc"].dropna()
        if not valid_times.empty:
            last_idx = valid_times.index[-1]
            state.last_timestamp[str(ship_id)] = std.at[last_idx, "timestamp_utc"]
        valid_position = std.loc[indices, ["latitude_deg", "longitude_deg"]].dropna()
        if not valid_position.empty:
            last_position = valid_position.iloc[-1]
            state.last_position[str(ship_id)] = (float(last_position.iloc[0]), float(last_position.iloc[1]))

    # 统一当前数据间隔：清洗层仍保留原始尺度；第二阶段重采样后改为10
    std["data_interval_min"] = np.where(ship_kind == "container", 10.0, 5.0)
    std["distance_nm_10min"] = np.where(ship_kind == "container", std["distance_nm_interval"], np.nan)
    std["resample_count_10min"] = np.where(ship_kind == "container", 1, np.nan)
    std["coverage_ratio_10min"] = np.where(ship_kind == "container", 1.0, np.nan)

    # 静态变量一致性与船内缺失补充
    static_cols = ["ship_type", "design_draught_m", "deadweight_t", "service_speed_kn", "main_engine_power_kw"]
    std["static_conflict_flag"] = 0
    for ship_id, group_idx in std.groupby("ship_id", sort=False).groups.items():
        ship_key = str(ship_id)
        state.static_values.setdefault(ship_key, {})
        indices = list(group_idx)
        for col in static_cols:
            non_null = std.loc[indices, col].dropna()
            reference_value = state.static_values[ship_key].get(col)
            if reference_value is None and not non_null.empty:
                reference_value = non_null.mode().iloc[0] if not non_null.mode().empty else non_null.iloc[0]
                state.static_values[ship_key][col] = reference_value
            if reference_value is not None:
                conflict = std.loc[indices, col].notna() & (std.loc[indices, col].astype(str) != str(reference_value))
                std.loc[np.array(indices)[conflict.to_numpy()], "static_conflict_flag"] = 1
                std.loc[indices, col] = std.loc[indices, col].fillna(reference_value)

    # 质量检查
    std["position_valid_flag"] = (
        std["latitude_deg"].between(-90, 90, inclusive="both")
        & std["longitude_deg"].between(-180, 180, inclusive="both")
    ).astype("int8")

    std["speed_valid_flag"] = std["speed_kn"].between(0, 40, inclusive="both").astype("int8")

    draught_ratio = std["mean_draught_m"] / std["design_draught_m"]
    std["draught_valid_flag"] = (
        std["mean_draught_m"].gt(0)
        & std["design_draught_m"].gt(0)
        & draught_ratio.between(0.25, 1.50, inclusive="both")
    ).astype("int8")

    weather_valid = (
        std["wind_speed_kn"].between(0, 80, inclusive="both")
        & std["wave_height_m"].between(0, 20, inclusive="both")
        & std["wave_period_s"].between(1, 30, inclusive="both")
        & std["surface_pressure_pa"].between(85000, 110000, inclusive="both")
        & std["surface_temperature_c"].between(-3, 45, inclusive="both")
        & std["wind_direction_deg"].notna()
        & std["wave_direction_deg"].notna()
    )
    std["weather_valid_flag"] = weather_valid.astype("int8")

    std["fuel_valid_flag"] = std["fuel_rate_kg_h"].ge(0).astype("int8")
    std["imputed_flag"] = 0  # 第一阶段不进行插值

    hard_outlier = (
        (std["speed_kn"].notna() & ~std["speed_kn"].between(0, 40, inclusive="both"))
        | (std["rudder_deg"].notna() & ~std["rudder_deg"].between(-45, 45, inclusive="both"))
        | (std["wind_speed_kn"].notna() & ~std["wind_speed_kn"].between(0, 80, inclusive="both"))
        | (std["wave_height_m"].notna() & ~std["wave_height_m"].between(0, 20, inclusive="both"))
        | (std["wave_period_s"].notna() & ~std["wave_period_s"].between(1, 30, inclusive="both"))
        | (std["fuel_rate_kg_h"].notna() & (std["fuel_rate_kg_h"] < 0))
    )
    std["outlier_flag"] = hard_outlier.astype("int8")

    # 记录有效性：用于建模的严格条件；原始记录仍全部输出，不直接删除
    required_flags = [
        "time_valid_flag",
        "position_valid_flag",
        "speed_valid_flag",
        "draught_valid_flag",
        "weather_valid_flag",
        "fuel_valid_flag",
    ]
    std["valid_record_flag"] = (std[required_flags].min(axis=1).eq(1) & std["outlier_flag"].eq(0)).astype("int8")

    reason_pairs = [
        ("time_valid_flag", "invalid_time"),
        ("position_valid_flag", "invalid_position"),
        ("speed_valid_flag", "invalid_speed"),
        ("draught_valid_flag", "invalid_draught"),
        ("weather_valid_flag", "invalid_or_missing_weather"),
        ("fuel_valid_flag", "invalid_or_missing_fuel"),
        ("static_conflict_flag", "static_conflict"),
        ("outlier_flag", "hard_outlier"),
    ]

    def build_reason(row: pd.Series) -> str:
        reasons: List[str] = []
        for flag, reason in reason_pairs:
            if flag in {"static_conflict_flag", "outlier_flag"}:
                if int(row.get(flag, 0)) == 1:
                    reasons.append(reason)
            elif int(row.get(flag, 0)) == 0:
                reasons.append(reason)
        if int(row.get("duplicate_flag", 0)) == 1:
            reasons.append("duplicate_timestamp")
        return ";".join(reasons)

    std["invalid_reason"] = std.apply(build_reason, axis=1)
    std["source_file"] = source_file

    # 恢复块内原始顺序，便于与原始文件对照
    std = std.sort_values("_original_index", kind="mergesort").drop(columns=["_original_index"])

    # 保证字段齐全
    for col in STANDARD_OUTPUT_COLUMNS:
        if col not in std.columns:
            std[col] = np.nan

    # 统计
    stats.rows_in += len(raw)
    stats.rows_out += len(std)
    stats.duplicate_rows += int(std["duplicate_flag"].sum())
    stats.invalid_rows += int((std["valid_record_flag"] == 0).sum())
    stats.measured_fuel_rows += int((std["fuel_source"] == "measured").sum())
    stats.calculated_fuel_rows += int((std["fuel_source"] == "calculated").sum())
    stats.missing_fuel_rows += int((std["fuel_source"] == "missing").sum())
    stats.update_missing(std)

    return std[STANDARD_OUTPUT_COLUMNS]


def iter_input_chunks(
    path: Path,
    chunksize: int,
    encoding: Optional[str],
    sep: Optional[str],
) -> Iterable[pd.DataFrame]:
    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt", ".tsv"}:
        enc = encoding or detect_encoding(path)
        delimiter = sep or detect_separator(path, enc)
        yield from pd.read_csv(
            path,
            encoding=enc,
            sep=delimiter,
            chunksize=chunksize,
            low_memory=False,
        )
    elif suffix in {".xlsx", ".xls"}:
        print(f"警告：Excel不支持分块读取，将一次性载入内存：{path}", file=sys.stderr)
        yield pd.read_excel(path)
    elif suffix in {".parquet", ".pq"}:
        # 简化处理：若Parquet极大，可进一步改为pyarrow批读取
        yield pd.read_parquet(path)
    else:
        raise ValueError(f"不支持的文件格式：{suffix}")


def process_file(
    input_path: Path,
    output_dir: Path,
    ship_type_override: str,
    chunksize: int,
    encoding: Optional[str],
    sep: Optional[str],
    me_fo_unit: str,
    keep_container_all: bool,
    explicit_ship_id: Optional[str],
) -> Tuple[Path, Path]:
    sample = read_sample(input_path, encoding=encoding, sep=sep)
    ship_kind = classify_ship_type(input_path, sample, ship_type_override)
    print(f"[{input_path.name}] 识别船型：{ship_kind}")

    output_dir.mkdir(parents=True, exist_ok=True)
    clean_path = output_dir / f"clean_{ship_kind}_{input_path.stem}.csv"
    report_path = output_dir / f"cleaning_report_{ship_kind}_{input_path.stem}.json"

    state = StreamState()
    stats = CleaningStats()
    first_chunk = True

    for chunk_number, raw_chunk in enumerate(iter_input_chunks(input_path, chunksize, encoding, sep), start=1):
        std_chunk = create_standardized_chunk(
            raw=raw_chunk,
            ship_kind=ship_kind,
            source_file=input_path.name,
            state=state,
            me_fo_unit=me_fo_unit,
            explicit_ship_id=explicit_ship_id,
            stats=stats,
        )

        if ship_kind == "container" and keep_container_all:
            # 集装箱船保留全部原始字段。统一字段放在前面；若原始字段与统一字段同名则不重复。
            original_to_keep = raw_chunk.drop(columns=[c for c in raw_chunk.columns if c in std_chunk.columns], errors="ignore")
            output_chunk = pd.concat([std_chunk.reset_index(drop=True), original_to_keep.reset_index(drop=True)], axis=1)
        else:
            # 油船和散货船仅输出关键字段、辅助字段和质量标记
            output_chunk = std_chunk

        output_chunk.to_csv(
            clean_path,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            index=False,
            encoding="utf-8-sig",
        )
        first_chunk = False
        print(f"  已处理块 {chunk_number}，累计 {stats.rows_in:,} 行")

    report = {
        "input_file": str(input_path),
        "output_file": str(clean_path),
        "ship_kind": ship_kind,
        "rows_in": stats.rows_in,
        "rows_out": stats.rows_out,
        "duplicate_rows": stats.duplicate_rows,
        "invalid_rows": stats.invalid_rows,
        "valid_rows": stats.rows_out - stats.invalid_rows,
        "valid_ratio": (stats.rows_out - stats.invalid_rows) / stats.rows_out if stats.rows_out else None,
        "fuel_source_counts": {
            "measured": stats.measured_fuel_rows,
            "calculated": stats.calculated_fuel_rows,
            "missing": stats.missing_fuel_rows,
        },
        "unit_detection": {
            "surface_pressure": stats.pressure_conversion,
            "surface_temperature": stats.temperature_conversion,
            "me_fo_unit_assumption": me_fo_unit if ship_kind in {"tanker", "bulk"} else None,
        },
        "missing_counts_key_columns": stats.missing_counts,
        "notes": [
            "油船和散货船的fuel_t_10min在第一阶段保持为空；后续按10分钟窗对fuel_t_5min求和。",
            "集装箱船默认保留全部原始字段；油船和散货船只保留统一关键字段、辅助字段与质量标记。",
            "若输入文件未按船舶和时间排序，跨块source_interval_min和distance_nm_interval可能不准确。",
            "valid_record_flag只用于筛选，不会在本脚本中删除原始记录。",
        ],
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return clean_path, report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="三船型大文件清洗与字段自动识别")
    parser.add_argument("--input", nargs="+", required=True, help="一个或多个CSV/Excel/Parquet文件")
    parser.add_argument("--output-dir", default="cleaned_output", help="输出目录")
    parser.add_argument(
        "--ship-type",
        choices=["auto", "container", "tanker", "bulk"],
        default="auto",
        help="船型；油船/散货船自动识别失败时必须显式指定",
    )
    parser.add_argument("--chunksize", type=int, default=200_000, help="CSV每块行数")
    parser.add_argument("--encoding", default=None, help="CSV编码；默认自动检测")
    parser.add_argument("--sep", default=None, help="CSV分隔符；默认自动检测")
    parser.add_argument(
        "--me-fo-unit",
        choices=["t/5min", "kg/5min", "kg/h"],
        default="t/5min",
        help="油船/散货船me_fo的原始单位",
    )
    parser.add_argument(
        "--drop-container-extra",
        action="store_true",
        help="启用后集装箱船也只保留关键字段；默认保留全部原始非关键字段",
    )
    parser.add_argument(
        "--ship-id",
        default=None,
        help="原始文件没有船舶编号时，可显式指定；否则使用文件名作为ship_id",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)

    results = []
    for input_name in args.input:
        input_path = Path(input_name)
        if not input_path.exists():
            raise FileNotFoundError(f"文件不存在：{input_path}")
        clean_path, report_path = process_file(
            input_path=input_path,
            output_dir=output_dir,
            ship_type_override=args.ship_type,
            chunksize=args.chunksize,
            encoding=args.encoding,
            sep=args.sep,
            me_fo_unit=args.me_fo_unit,
            keep_container_all=not args.drop_container_extra,
            explicit_ship_id=args.ship_id,
        )
        results.append((clean_path, report_path))

    print("\n处理完成：")
    for clean_path, report_path in results:
        print(f"  清洗文件：{clean_path}")
        print(f"  清洗报告：{report_path}")


if __name__ == "__main__":
    main()