from __future__ import annotations

import calendar
import gc
import glob
import hashlib
import json
import math
import os
import sys
import time
import warnings
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from netCDF4 import Dataset, num2date

try:
    from netCDF4 import set_chunk_cache
except ImportError:
    set_chunk_cache = None

warnings.filterwarnings("ignore", category=RuntimeWarning)

# 限制 netCDF-C/HDF5 的全局 chunk cache，避免长时间运行后缓存膨胀。
if set_chunk_cache is not None:
    try:
        set_chunk_cache(
            size=8 * 1024 * 1024,
            nelems=1009,
            preemption=0.5,
        )
    except Exception:
        pass


# =============================================================================
# 1. 用户配置
# =============================================================================

INPUT_EXCEL = Path(r"data\continership.xlsx")
INPUT_SHEET = 0

RAW_DIR = Path(r"data/era5\era5_final_7fields\raw")

OUTPUT_EXCEL = Path(
    r"data\continership_with_era5_7fields_v4.xlsx"
)

SURFACE_PATTERN = str(
    RAW_DIR / "era5_surface_wind_raw_*.nc"
)
WAVE_PATTERN = str(
    RAW_DIR / "era5_wave_raw_*.nc"
)

# 船舶 Excel 中没有时区信息的时间按 UTC 解释。
INPUT_TIMEZONE = "UTC"

START_YEAR = 2023
START_MONTH = 10
END_YEAR = 2024
END_MONTH = 10

REQUIRE_COMPLETE_FILES = True

# -------------------------------------------------------------------------
# 文件有效性检验
# -------------------------------------------------------------------------

# 可选：
#   "full"     首次逐数据块深检全部变量；推荐。
#   "metadata" 只检查变量、维度和时间数。
#   "none"     不预检，仅在插值读取时重试。
VALIDATION_MODE = "metadata"

# 深检时一次读取多少个小时。
# 24 表示按日读取，内存开销通常约几十 MB。
VALIDATION_TIME_CHUNK = 1

VALIDATION_CACHE = (
    RAW_DIR / "era5_deep_validation_cache.json"
)
VALIDATION_REPORT = Path(
    r"data\era5_file_validation_report.csv"
)

# -------------------------------------------------------------------------
# 插值读取
# -------------------------------------------------------------------------

# 单个 ERA5 二维时次读取失败时，重新打开文件的重试次数。
READ_RETRY_COUNT = 5
READ_RETRY_WAIT_SECONDS = 3

# 缓存多少个完整二维时次。
# 风温压每个时次约数 MB，4 个通常很安全。
MAX_CACHED_TIME_SLICES = 2

# 时间轴不允许跨越大于该值的缺口。
MAX_TIME_GAP_HOURS = 2.0

# 海岸附近的严格程度：
# False：四角中任何实际有权重的角点缺失，则结果缺失。
# True：用剩余有效角点重新归一化。
ALLOW_PARTIAL_SPATIAL_NEIGHBORS = False

CALM_WIND_THRESHOLD_MS = 0.01
MS_TO_KNOT = 1.9438444924406048
KELVIN_TO_CELSIUS = 273.15

# 进度同时按行数、时间区间组数和实际经过秒数触发。
PROGRESS_EVERY_ROWS = 500
PROGRESS_EVERY_GROUPS = 10
PROGRESS_EVERY_SECONDS = 10.0

# -------------------------------------------------------------------------
# 检查点
# -------------------------------------------------------------------------

CHECKPOINT_DIR = Path(
    r"data\era5_interp_checkpoints"
)

# 使用与上一版相同的名称，能够恢复已完成的风温压结果。
SURFACE_CHECKPOINT = (
    CHECKPOINT_DIR / "surface_result_checkpoint.npz"
)
WAVE_CHECKPOINT = (
    CHECKPOINT_DIR / "wave_result_checkpoint.npz"
)

CHECKPOINT_EVERY_GROUPS = 250


# =============================================================================
# 2. 名称候选
# =============================================================================

TIME_COLUMN_CANDIDATES = (
    "UTC时间(-)",
    "UTC时间",
    "UTC Time",
    "utc_time",
    "timestamp",
    "datetime",
    "日期时间",
    "时间",
    "time",
)

LAT_COLUMN_CANDIDATES = (
    "纬度(deg)",
    "纬度",
    "latitude",
    "lat",
)

LON_COLUMN_CANDIDATES = (
    "经度(deg)",
    "经度",
    "longitude",
    "lon",
)

TIME_COORDINATE_CANDIDATES = (
    "valid_time",
    "time",
    "forecast_time",
    "date",
)

LAT_COORDINATE_CANDIDATES = (
    "latitude",
    "lat",
)

LON_COORDINATE_CANDIDATES = (
    "longitude",
    "lon",
)

SURFACE_VARIABLE_CANDIDATES: Dict[str, Sequence[str]] = {
    "sp": (
        "sp",
        "surface_pressure",
        "surface_p",
    ),
    "sst": (
        "sst",
        "sea_surface_temperature",
        "surface_t",
    ),
    "u10": (
        "u10",
        "10m_u_component_of_wind",
    ),
    "v10": (
        "v10",
        "10m_v_component_of_wind",
    ),
}

WAVE_VARIABLE_CANDIDATES: Dict[str, Sequence[str]] = {
    "swh": (
        "swh",
        "wave_h",
        "significant_height_of_combined_wind_waves_and_swell",
    ),
    "mwd": (
        "mwd",
        "wave_d",
        "mean_wave_direction",
    ),
    "mwp": (
        "mwp",
        "wave_p",
        "mean_wave_period",
    ),
}


# =============================================================================
# 3. 数据结构
# =============================================================================

@dataclass
class NCFileMeta:
    path: Path
    time_name: str
    lat_name: str
    lon_name: str
    times_ns: np.ndarray
    latitude: np.ndarray
    longitude: np.ndarray
    variables: Dict[str, str]


class ERA5Collection:
    """某一类 ERA5 月文件及其全局时间轴。"""

    def __init__(
        self,
        file_paths: Sequence[Path],
        variable_candidates: Dict[str, Sequence[str]],
        label: str,
    ):
        if not file_paths:
            raise FileNotFoundError(
                f"未找到 {label} 文件。"
            )

        self.label = label
        self.variable_candidates = variable_candidates
        self.files = [
            inspect_nc_file(
                path,
                variable_candidates,
            )
            for path in sorted(file_paths)
        ]

        self._validate_grids()

        self.latitude = self.files[0].latitude
        self.longitude = self.files[0].longitude

        time_parts = []
        file_parts = []
        local_parts = []

        for file_id, meta in enumerate(self.files):
            count = len(meta.times_ns)

            time_parts.append(meta.times_ns)
            file_parts.append(
                np.full(
                    count,
                    file_id,
                    dtype=np.int32,
                )
            )
            local_parts.append(
                np.arange(
                    count,
                    dtype=np.int32,
                )
            )

        all_times = np.concatenate(time_parts)
        all_files = np.concatenate(file_parts)
        all_locals = np.concatenate(local_parts)

        order = np.argsort(
            all_times,
            kind="stable",
        )

        all_times = all_times[order]
        all_files = all_files[order]
        all_locals = all_locals[order]

        unique_mask = np.ones(
            len(all_times),
            dtype=bool,
        )

        if len(all_times) > 1:
            duplicate = (
                all_times[1:]
                == all_times[:-1]
            )
            unique_mask[1:] = ~duplicate

            if duplicate.any():
                print(
                    f"  {label} 检测到 "
                    f"{int(duplicate.sum())} 个重复时次，"
                    "保留第一条。"
                )

        self.times_ns = all_times[unique_mask]
        self.file_ids = all_files[unique_mask]
        self.local_ids = all_locals[unique_mask]

        if not len(self.times_ns):
            raise RuntimeError(
                f"{label} 时间轴为空。"
            )

        if np.any(np.diff(self.times_ns) <= 0):
            raise RuntimeError(
                f"{label} 时间轴不是严格递增。"
            )

        print(
            f"  {label}：文件 {len(self.files)} 个；"
            f"时次 {len(self.times_ns)} 个；"
            f"时间范围 "
            f"{pd.Timestamp(self.times_ns[0])} ~ "
            f"{pd.Timestamp(self.times_ns[-1])}"
        )

    def _validate_grids(self) -> None:
        reference_lat = self.files[0].latitude
        reference_lon = self.files[0].longitude

        for meta in self.files[1:]:
            if (
                meta.latitude.shape
                != reference_lat.shape
                or not np.allclose(
                    meta.latitude,
                    reference_lat,
                    equal_nan=True,
                )
            ):
                raise RuntimeError(
                    f"{self.label} 纬度网格不一致："
                    f"{meta.path.name}"
                )

            if (
                meta.longitude.shape
                != reference_lon.shape
                or not np.allclose(
                    meta.longitude,
                    reference_lon,
                    equal_nan=True,
                )
            ):
                raise RuntimeError(
                    f"{self.label} 经度网格不一致："
                    f"{meta.path.name}"
                )


class TimeSliceCache:
    """
    小型 LRU 缓存。

    缓存的是某一 ERA5 时次全部目标变量的完整二维场，
    不缓存长期打开的 NetCDF 文件句柄。
    """

    def __init__(self, max_items: int):
        self.max_items = max(1, int(max_items))
        self._cache: OrderedDict[
            Tuple[str, int],
            Dict[str, np.ndarray],
        ] = OrderedDict()

    def get(
        self,
        collection: ERA5Collection,
        global_time_index: int,
    ) -> Dict[str, np.ndarray]:
        key = (
            collection.label,
            int(global_time_index),
        )

        if key in self._cache:
            value = self._cache.pop(key)
            self._cache[key] = value
            return value

        value = load_full_time_slice(
            collection,
            int(global_time_index),
        )

        self._cache[key] = value

        while len(self._cache) > self.max_items:
            _, evicted = self._cache.popitem(last=False)
            del evicted
            gc.collect()

        return value

    def clear(self) -> None:
        self._cache.clear()
        gc.collect()


# =============================================================================
# 4. 通用名称与时间处理
# =============================================================================

def normalize_name(value: object) -> str:
    return (
        str(value)
        .strip()
        .lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
        .replace("（", "(")
        .replace("）", ")")
    )


def find_dataframe_column(
    columns: Iterable[object],
    candidates: Sequence[str],
    contains_tokens: Sequence[str],
) -> str:
    columns = list(columns)

    normalized = {
        normalize_name(column): str(column)
        for column in columns
    }

    for candidate in candidates:
        key = normalize_name(candidate)

        if key in normalized:
            return normalized[key]

    for column in columns:
        lower = str(column).lower()

        if any(
            token.lower() in lower
            for token in contains_tokens
        ):
            return str(column)

    raise KeyError(
        f"无法识别字段。候选={list(candidates)}；"
        f"实际={columns}"
    )


def find_nc_name(
    nc: Dataset,
    candidates: Sequence[str],
    required_ndim: Optional[int] = None,
) -> Optional[str]:
    for name in candidates:
        if name in nc.variables:
            variable = nc.variables[name]

            if (
                required_ndim is None
                or variable.ndim == required_ndim
            ):
                return name

    normalized_candidates = {
        normalize_name(name)
        for name in candidates
    }

    for name, variable in nc.variables.items():
        if (
            normalize_name(name)
            in normalized_candidates
        ):
            if (
                required_ndim is None
                or variable.ndim == required_ndim
            ):
                return name

    return None


def netcdf_times_to_ns(
    time_variable,
) -> np.ndarray:
    units = getattr(
        time_variable,
        "units",
        None,
    )
    calendar_name = getattr(
        time_variable,
        "calendar",
        "standard",
    )

    if units is None:
        raise ValueError(
            f"时间变量 {time_variable.name} "
            "缺少 units。"
        )

    raw_values = np.asarray(
        time_variable[:]
    )

    date_values = num2date(
        raw_values,
        units=units,
        calendar=calendar_name,
        only_use_cftime_datetimes=False,
        only_use_python_datetimes=False,
    )

    result = []

    for value in np.asarray(
        date_values
    ).ravel():
        try:
            timestamp = pd.Timestamp(value)
        except Exception:
            timestamp = pd.Timestamp(
                value.strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            )

        if timestamp.tzinfo is not None:
            timestamp = (
                timestamp
                .tz_convert("UTC")
                .tz_localize(None)
            )

        result.append(timestamp.value)

    return np.asarray(
        result,
        dtype=np.int64,
    )


def parse_input_utc(
    series: pd.Series,
) -> pd.Series:
    parsed = pd.to_datetime(
        series,
        errors="coerce",
    )

    try:
        timezone = parsed.dt.tz
    except AttributeError:
        return pd.to_datetime(
            series,
            errors="coerce",
            utc=True,
        ).dt.tz_localize(None)

    if timezone is None:
        localized = parsed.dt.tz_localize(
            INPUT_TIMEZONE,
            ambiguous="NaT",
            nonexistent="NaT",
        )
    else:
        localized = parsed

    return (
        localized
        .dt.tz_convert("UTC")
        .dt.tz_localize(None)
    )


# =============================================================================
# 5. NetCDF 元数据检查
# =============================================================================

def inspect_nc_file(
    path: Path,
    variable_candidates: Dict[str, Sequence[str]],
) -> NCFileMeta:
    with Dataset(str(path), mode="r") as nc:
        time_name = find_nc_name(
            nc,
            TIME_COORDINATE_CANDIDATES,
            required_ndim=1,
        )
        lat_name = find_nc_name(
            nc,
            LAT_COORDINATE_CANDIDATES,
            required_ndim=1,
        )
        lon_name = find_nc_name(
            nc,
            LON_COORDINATE_CANDIDATES,
            required_ndim=1,
        )

        if time_name is None:
            raise KeyError(
                f"{path.name} 缺少时间坐标。"
            )

        if lat_name is None or lon_name is None:
            raise KeyError(
                f"{path.name} 缺少经纬度坐标。"
            )

        variables: Dict[str, str] = {}

        for logical_name, candidates in (
            variable_candidates.items()
        ):
            actual_name = find_nc_name(
                nc,
                candidates,
            )

            if actual_name is None:
                raise KeyError(
                    f"{path.name} 缺少变量 "
                    f"{logical_name}；"
                    f"候选={list(candidates)}；"
                    f"现有={list(nc.variables.keys())}"
                )

            variables[logical_name] = actual_name

        times_ns = netcdf_times_to_ns(
            nc.variables[time_name]
        )
        latitude = np.asarray(
            nc.variables[lat_name][:],
            dtype=np.float64,
        )
        longitude = np.asarray(
            nc.variables[lon_name][:],
            dtype=np.float64,
        )

    return NCFileMeta(
        path=path,
        time_name=time_name,
        lat_name=lat_name,
        lon_name=lon_name,
        times_ns=times_ns,
        latitude=latitude,
        longitude=longitude,
        variables=variables,
    )


def iter_months(
    start_year: int,
    start_month: int,
    end_year: int,
    end_month: int,
):
    year = start_year
    month = start_month

    while (year, month) <= (
        end_year,
        end_month,
    ):
        yield year, month

        month += 1

        if month == 13:
            month = 1
            year += 1


def check_expected_files() -> None:
    missing = []

    for year, month in iter_months(
        START_YEAR,
        START_MONTH,
        END_YEAR,
        END_MONTH,
    ):
        surface = RAW_DIR / (
            f"era5_surface_wind_raw_"
            f"{year:04d}_{month:02d}.nc"
        )
        wave = RAW_DIR / (
            f"era5_wave_raw_"
            f"{year:04d}_{month:02d}.nc"
        )

        if not surface.exists():
            missing.append(surface.name)

        if not wave.exists():
            missing.append(wave.name)

    if missing:
        message = (
            f"缺少 {len(missing)} 个文件：\n  "
            + "\n  ".join(missing)
        )

        if REQUIRE_COMPLETE_FILES:
            raise FileNotFoundError(message)

        print("警告：" + message)


# =============================================================================
# 6. 文件深度有效性检验
# =============================================================================

def file_signature(path: Path) -> Dict[str, int]:
    stat = path.stat()

    return {
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def load_validation_cache() -> dict:
    if not VALIDATION_CACHE.exists():
        return {}

    try:
        with open(
            VALIDATION_CACHE,
            "r",
            encoding="utf-8",
        ) as file_handle:
            data = json.load(file_handle)

        if isinstance(data, dict):
            return data

    except Exception:
        pass

    return {}


def save_validation_cache(
    cache: dict,
) -> None:
    VALIDATION_CACHE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = VALIDATION_CACHE.with_suffix(
        ".tmp"
    )

    with open(
        temporary,
        "w",
        encoding="utf-8",
    ) as file_handle:
        json.dump(
            cache,
            file_handle,
            ensure_ascii=False,
            indent=2,
        )

    os.replace(
        temporary,
        VALIDATION_CACHE,
    )


def collapse_expver(
    data: np.ndarray,
    dimension_names: List[str],
) -> Tuple[np.ndarray, List[str]]:
    expver_axis = None

    for axis, name in enumerate(
        dimension_names
    ):
        if name.lower() == "expver":
            expver_axis = axis
            break

    if expver_axis is None:
        return data, dimension_names

    moved = np.moveaxis(
        data,
        expver_axis,
        0,
    )

    collapsed = np.full(
        moved.shape[1:],
        np.nan,
        dtype=np.float64,
    )

    for layer in moved:
        mask = (
            ~np.isfinite(collapsed)
            & np.isfinite(layer)
        )
        collapsed[mask] = layer[mask]

    new_names = dimension_names.copy()
    new_names.pop(expver_axis)

    return collapsed, new_names


def build_time_chunk_selection(
    nc: Dataset,
    variable,
    meta: NCFileMeta,
    start: int,
    stop: int,
):
    indexers = []
    remaining_dimensions = []

    for dimension in variable.dimensions:
        if dimension == meta.time_name:
            indexers.append(
                slice(start, stop)
            )
            remaining_dimensions.append(dimension)

        elif dimension in (
            meta.lat_name,
            meta.lon_name,
        ):
            indexers.append(slice(None))
            remaining_dimensions.append(dimension)

        elif dimension.lower() == "expver":
            indexers.append(slice(None))
            remaining_dimensions.append(dimension)

        elif len(nc.dimensions[dimension]) == 1:
            indexers.append(0)

        else:
            raise RuntimeError(
                f"变量 {variable.name} 包含"
                f"无法处理的额外维度 {dimension}，"
                f"长度={len(nc.dimensions[dimension])}"
            )

    return tuple(indexers), remaining_dimensions


def validate_one_file(
    meta: NCFileMeta,
    mode: str,
) -> Dict[str, object]:
    start_clock = time.time()

    report = {
        "file": meta.path.name,
        "path": str(meta.path),
        "status": "ok",
        "mode": mode,
        "message": "",
        "elapsed_seconds": 0.0,
    }

    try:
        with Dataset(
            str(meta.path),
            mode="r",
        ) as nc:
            expected_time_count = len(
                meta.times_ns
            )

            if expected_time_count == 0:
                raise RuntimeError(
                    "时间坐标为空。"
                )

            for logical_name, actual_name in (
                meta.variables.items()
            ):
                variable = nc.variables[
                    actual_name
                ]

                if (
                    meta.time_name
                    not in variable.dimensions
                ):
                    raise RuntimeError(
                        f"{actual_name} 不包含时间维度 "
                        f"{meta.time_name}。"
                    )

                axis = variable.dimensions.index(
                    meta.time_name
                )

                if (
                    variable.shape[axis]
                    != expected_time_count
                ):
                    raise RuntimeError(
                        f"{actual_name} 时间长度 "
                        f"{variable.shape[axis]}，"
                        f"坐标长度 {expected_time_count}。"
                    )

        if mode != "full":
            report["message"] = (
                "元数据、变量和时间长度检查通过"
            )
            return report

        for logical_name, actual_name in (
            meta.variables.items()
        ):
            total_times = len(meta.times_ns)

            for start in range(
                0,
                total_times,
                VALIDATION_TIME_CHUNK,
            ):
                stop = min(
                    start
                    + VALIDATION_TIME_CHUNK,
                    total_times,
                )

                try:
                    with Dataset(
                        str(meta.path),
                        mode="r",
                    ) as nc:
                        variable = nc.variables[
                            actual_name
                        ]

                        (
                            selection,
                            remaining_dimensions,
                        ) = build_time_chunk_selection(
                            nc,
                            variable,
                            meta,
                            start,
                            stop,
                        )

                        data = np.ma.asarray(
                            variable[selection]
                        )

                        # 强制访问数据，确保 HDF 数据块已解码。
                        _ = data.count()

                except Exception as chunk_error:
                    # 大块失败后逐小时定位，区分瞬时错误和持久错误。
                    bad_details = []

                    for time_index in range(
                        start,
                        stop,
                    ):
                        success = False
                        errors = []

                        for attempt in range(
                            1,
                            READ_RETRY_COUNT + 1,
                        ):
                            try:
                                with Dataset(
                                    str(meta.path),
                                    mode="r",
                                ) as nc:
                                    variable = nc.variables[
                                        actual_name
                                    ]

                                    (
                                        selection,
                                        _,
                                    ) = build_time_chunk_selection(
                                        nc,
                                        variable,
                                        meta,
                                        time_index,
                                        time_index + 1,
                                    )

                                    one_hour = np.ma.asarray(
                                        variable[selection]
                                    )
                                    _ = one_hour.count()

                                success = True
                                break

                            except Exception as exc:
                                errors.append(
                                    f"{type(exc).__name__}: {exc}"
                                )
                                time.sleep(
                                    READ_RETRY_WAIT_SECONDS
                                )

                        if not success:
                            bad_details.append(
                                f"变量={actual_name}；"
                                f"UTC={pd.Timestamp(meta.times_ns[time_index])}；"
                                f"错误={errors[-1] if errors else chunk_error}"
                            )

                    if bad_details:
                        raise RuntimeError(
                            "发现持续不可读数据块：\n  "
                            + "\n  ".join(bad_details)
                        ) from chunk_error

                print(
                    f"    验证 {meta.path.name} / "
                    f"{actual_name}："
                    f"{stop}/{total_times}"
                )

        report["message"] = (
            "全部变量和全部时次数据块均可读取"
        )

    except Exception as exc:
        report["status"] = "failed"
        report["message"] = (
            f"{type(exc).__name__}: {exc}"
        )

    finally:
        report["elapsed_seconds"] = round(
            time.time() - start_clock,
            2,
        )

    return report


def validate_all_files(
    surface_files: Sequence[Path],
    wave_files: Sequence[Path],
) -> None:
    if VALIDATION_MODE == "none":
        print(
            "  已跳过文件预检 "
            "(VALIDATION_MODE='none')。"
        )
        return

    print("\n[0/6] 检验 ERA5 文件有效性……")
    print(f"  检验模式：{VALIDATION_MODE}")

    cache = load_validation_cache()
    reports = []

    file_specs = [
        (
            path,
            SURFACE_VARIABLE_CANDIDATES,
            "风温压",
        )
        for path in surface_files
    ] + [
        (
            path,
            WAVE_VARIABLE_CANDIDATES,
            "波浪",
        )
        for path in wave_files
    ]

    failed = []

    for index, (
        path,
        candidates,
        category,
    ) in enumerate(
        file_specs,
        start=1,
    ):
        signature = file_signature(path)
        cache_key = str(path.resolve())

        cached = cache.get(cache_key)

        can_skip = (
            cached is not None
            and cached.get("status") == "ok"
            and cached.get("mode") == VALIDATION_MODE
            and cached.get("signature") == signature
        )

        print(
            f"\n  [{index}/{len(file_specs)}] "
            f"{category}：{path.name}"
        )

        if can_skip:
            print(
                "    文件未改变，使用上次深检结果。"
            )

            report = {
                "file": path.name,
                "path": str(path),
                "status": "ok",
                "mode": VALIDATION_MODE,
                "message": "使用缓存的有效性结果",
                "elapsed_seconds": 0.0,
            }

        else:
            meta = inspect_nc_file(
                path,
                candidates,
            )

            report = validate_one_file(
                meta,
                VALIDATION_MODE,
            )

            cache[cache_key] = {
                "status": report["status"],
                "mode": VALIDATION_MODE,
                "message": report["message"],
                "signature": signature,
                "validated_at": pd.Timestamp.now().isoformat(),
            }

            save_validation_cache(cache)

        reports.append(report)

        if report["status"] != "ok":
            failed.append(report)
            print(
                "    检验失败："
                + str(report["message"])
            )
        else:
            print(
                "    检验通过："
                + str(report["message"])
            )

    VALIDATION_REPORT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(reports).to_csv(
        VALIDATION_REPORT,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"\n  检验报告：{VALIDATION_REPORT}"
    )

    if failed:
        details = "\n\n".join(
            f"{item['file']}:\n{item['message']}"
            for item in failed
        )

        raise RuntimeError(
            "ERA5 文件有效性检验未通过。\n\n"
            + details
            + "\n\n请只重新下载上述失败文件。"
        )

    print(
        "\n  全部 ERA5 文件均通过有效性检验。"
    )


# =============================================================================
# 7. 稳健读取完整二维时次
# =============================================================================

def extract_2d_spatial_slice(
    nc: Dataset,
    variable,
    meta: NCFileMeta,
    local_time_index: int,
) -> np.ndarray:
    indexers = []
    remaining_dimensions: List[str] = []

    for dimension in variable.dimensions:
        if dimension == meta.time_name:
            indexers.append(
                int(local_time_index)
            )

        elif dimension in (
            meta.lat_name,
            meta.lon_name,
        ):
            indexers.append(slice(None))
            remaining_dimensions.append(
                dimension
            )

        elif dimension.lower() == "expver":
            indexers.append(slice(None))
            remaining_dimensions.append(
                dimension
            )

        elif len(nc.dimensions[dimension]) == 1:
            indexers.append(0)

        else:
            raise RuntimeError(
                f"{meta.path.name} 的变量 "
                f"{variable.name} 包含额外维度 "
                f"{dimension}，长度 "
                f"{len(nc.dimensions[dimension])}。"
            )

    raw = np.ma.asarray(
        variable[tuple(indexers)],
        dtype=np.float64,
    )

    data = np.ma.filled(
        raw,
        np.nan,
    )

    data, remaining_dimensions = (
        collapse_expver(
            data,
            remaining_dimensions,
        )
    )

    if (
        meta.lat_name
        not in remaining_dimensions
        or meta.lon_name
        not in remaining_dimensions
    ):
        raise RuntimeError(
            f"{variable.name} 读取后"
            "未保留经纬度维度。"
        )

    lat_axis = remaining_dimensions.index(
        meta.lat_name
    )
    lon_axis = remaining_dimensions.index(
        meta.lon_name
    )

    data = np.moveaxis(
        data,
        [lat_axis, lon_axis],
        [0, 1],
    )

    if data.ndim > 2:
        extra_shape = data.shape[2:]

        if any(
            size != 1
            for size in extra_shape
        ):
            raise RuntimeError(
                f"{variable.name} 仍有额外维度："
                f"{data.shape}"
            )

        data = data.reshape(
            data.shape[0],
            data.shape[1],
        )

    expected_shape = (
        len(meta.latitude),
        len(meta.longitude),
    )

    if data.shape != expected_shape:
        raise RuntimeError(
            f"{variable.name} 的空间形状 "
            f"{data.shape}，预期 "
            f"{expected_shape}。"
        )

    return data


def load_full_time_slice(
    collection: ERA5Collection,
    global_time_index: int,
) -> Dict[str, np.ndarray]:
    file_id = int(
        collection.file_ids[
            global_time_index
        ]
    )
    local_time_index = int(
        collection.local_ids[
            global_time_index
        ]
    )

    meta = collection.files[file_id]
    utc_time = pd.Timestamp(
        collection.times_ns[
            global_time_index
        ]
    )

    errors = []

    for attempt in range(
        1,
        READ_RETRY_COUNT + 1,
    ):
        try:
            result = {}

            # 每次尝试重新打开文件，并在读取完成后立即关闭。
            with Dataset(
                str(meta.path),
                mode="r",
            ) as nc:
                for (
                    logical_name,
                    actual_name,
                ) in meta.variables.items():
                    variable = nc.variables[
                        actual_name
                    ]

                    field = extract_2d_spatial_slice(
                        nc,
                        variable,
                        meta,
                        local_time_index,
                    )

                    # 二维 ERA5 场以 float32 缓存，降低内存占用；
                    # 后续插值运算会自动提升精度。
                    result[logical_name] = np.asarray(
                        field,
                        dtype=np.float32,
                    )

                    del field

            return result

        except Exception as exc:
            error = (
                f"第 {attempt}/{READ_RETRY_COUNT} 次失败；"
                f"文件={meta.path.name}；"
                f"UTC={utc_time}；"
                f"{type(exc).__name__}: {exc}"
            )
            errors.append(error)

            print(
                "\n  读取警告："
                + error
            )

            gc.collect()

            if attempt < READ_RETRY_COUNT:
                time.sleep(
                    READ_RETRY_WAIT_SECONDS
                )

    raise RuntimeError(
        "完整二维时次连续读取失败：\n  "
        + "\n  ".join(errors)
    )


# =============================================================================
# 8. 时间和空间括号
# =============================================================================

def bracket_nonperiodic_axis(
    coordinates: np.ndarray,
    points: np.ndarray,
):
    coordinates = np.asarray(
        coordinates,
        dtype=np.float64,
    )
    points = np.asarray(
        points,
        dtype=np.float64,
    )

    order = np.argsort(coordinates)
    sorted_coordinates = coordinates[
        order
    ]

    if len(sorted_coordinates) < 2:
        raise ValueError(
            "坐标轴至少需要两个点。"
        )

    valid = (
        np.isfinite(points)
        & (
            points
            >= sorted_coordinates[0] - 1e-10
        )
        & (
            points
            <= sorted_coordinates[-1] + 1e-10
        )
    )

    clipped = np.clip(
        points,
        sorted_coordinates[0],
        sorted_coordinates[-1],
    )

    position = np.searchsorted(
        sorted_coordinates,
        clipped,
        side="right",
    )

    position = np.clip(
        position,
        1,
        len(sorted_coordinates) - 1,
    )

    lower_sorted = position - 1
    upper_sorted = position

    x0 = sorted_coordinates[
        lower_sorted
    ]
    x1 = sorted_coordinates[
        upper_sorted
    ]

    denominator = x1 - x0

    weight = np.divide(
        clipped - x0,
        denominator,
        out=np.zeros_like(clipped),
        where=denominator != 0,
    )

    return (
        order[lower_sorted].astype(np.int32),
        order[upper_sorted].astype(np.int32),
        np.clip(weight, 0.0, 1.0),
        valid,
    )


def bracket_longitude(
    coordinates: np.ndarray,
    points: np.ndarray,
):
    coordinates = np.asarray(
        coordinates,
        dtype=np.float64,
    )
    points = np.asarray(
        points,
        dtype=np.float64,
    )

    order = np.argsort(coordinates)
    sorted_coordinates = coordinates[
        order
    ]

    differences = np.diff(
        sorted_coordinates
    )
    positive = differences[
        differences > 1e-10
    ]

    if not len(positive):
        raise ValueError(
            "经度坐标无有效间隔。"
        )

    step = float(
        np.median(positive)
    )

    is_global = (
        sorted_coordinates[-1]
        - sorted_coordinates[0]
        + step
        >= 359.0
    )

    if not is_global:
        return bracket_nonperiodic_axis(
            coordinates,
            points,
        )

    base = sorted_coordinates[0]

    normalized = (
        (points - base) % 360.0
    ) + base

    valid = np.isfinite(points)

    position = np.searchsorted(
        sorted_coordinates,
        normalized,
        side="right",
    )

    lower_sorted = (
        position - 1
    ) % len(sorted_coordinates)
    upper_sorted = (
        position
    ) % len(sorted_coordinates)

    x0 = sorted_coordinates[
        lower_sorted
    ].copy()
    x1 = sorted_coordinates[
        upper_sorted
    ].copy()

    upper_wrap = (
        position
        == len(sorted_coordinates)
    )
    lower_wrap = (
        position == 0
    )

    x1[upper_wrap] += 360.0
    x0[lower_wrap] -= 360.0

    adjusted = normalized.copy()
    adjusted[lower_wrap] += 360.0

    weight = (
        adjusted - x0
    ) / (x1 - x0)

    return (
        order[lower_sorted].astype(np.int32),
        order[upper_sorted].astype(np.int32),
        np.clip(weight, 0.0, 1.0),
        valid,
    )


def calculate_time_brackets(
    timeline_ns: np.ndarray,
    target_ns: np.ndarray,
):
    timeline_ns = np.asarray(
        timeline_ns,
        dtype=np.int64,
    )
    target_ns = np.asarray(
        target_ns,
        dtype=np.int64,
    )

    count = len(timeline_ns)

    position = np.searchsorted(
        timeline_ns,
        target_ns,
        side="left",
    )

    exact = np.zeros(
        len(target_ns),
        dtype=bool,
    )

    inside = position < count

    exact[inside] = (
        timeline_ns[position[inside]]
        == target_ns[inside]
    )

    lower = position - 1
    upper = position.copy()

    lower[exact] = position[exact]
    upper[exact] = position[exact]

    valid = (
        (lower >= 0)
        & (upper < count)
    )

    safe_lower = np.clip(
        lower,
        0,
        count - 1,
    )
    safe_upper = np.clip(
        upper,
        0,
        count - 1,
    )

    t0 = timeline_ns[safe_lower]
    t1 = timeline_ns[safe_upper]

    weight = np.zeros(
        len(target_ns),
        dtype=np.float64,
    )

    different = (
        valid
        & (
            safe_lower
            != safe_upper
        )
    )

    denominator = (
        t1[different]
        - t0[different]
    ).astype(np.float64)

    weight[different] = (
        target_ns[different]
        - t0[different]
    ) / denominator

    gap_hours = np.zeros(
        len(target_ns),
        dtype=np.float64,
    )

    gap_hours[different] = (
        t1[different]
        - t0[different]
    ) / 3_600_000_000_000.0

    valid &= (
        ~different
        | (
            gap_hours
            <= MAX_TIME_GAP_HOURS
        )
    )

    return (
        safe_lower.astype(np.int32),
        safe_upper.astype(np.int32),
        weight,
        valid,
    )


# =============================================================================
# 9. 双线性与圆周插值
# =============================================================================

def spatial_weights(
    latitude_weight: np.ndarray,
    longitude_weight: np.ndarray,
) -> np.ndarray:
    wy = latitude_weight
    wx = longitude_weight

    return np.column_stack(
        [
            (1.0 - wy) * (1.0 - wx),
            (1.0 - wy) * wx,
            wy * (1.0 - wx),
            wy * wx,
        ]
    )


def extract_corners(
    field: np.ndarray,
    lat0: np.ndarray,
    lat1: np.ndarray,
    lon0: np.ndarray,
    lon1: np.ndarray,
) -> np.ndarray:
    return np.column_stack(
        [
            field[lat0, lon0],
            field[lat0, lon1],
            field[lat1, lon0],
            field[lat1, lon1],
        ]
    )


def weighted_combine(
    values: np.ndarray,
    weights: np.ndarray,
    allow_partial: bool,
) -> np.ndarray:
    values = np.asarray(
        values,
        dtype=np.float64,
    )
    weights = np.asarray(
        weights,
        dtype=np.float64,
    )

    finite = np.isfinite(values)
    relevant = weights > 1e-12

    if allow_partial:
        effective = np.where(
            finite,
            weights,
            0.0,
        )

        denominator = effective.sum(
            axis=1
        )
        numerator = (
            np.where(
                finite,
                values,
                0.0,
            )
            * effective
        ).sum(axis=1)

        result = np.full(
            len(values),
            np.nan,
            dtype=np.float64,
        )

        valid = denominator > 0

        result[valid] = (
            numerator[valid]
            / denominator[valid]
        )

        return result

    invalid = np.any(
        relevant & ~finite,
        axis=1,
    )

    result = (
        np.where(
            finite,
            values,
            0.0,
        )
        * weights
    ).sum(axis=1)

    result[invalid] = np.nan

    return result


def spatial_linear(
    corners: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    return weighted_combine(
        corners,
        weights,
        ALLOW_PARTIAL_SPATIAL_NEIGHBORS,
    )


def temporal_linear(
    lower: np.ndarray,
    upper: np.ndarray,
    upper_weight: np.ndarray,
) -> np.ndarray:
    values = np.column_stack(
        [lower, upper]
    )
    weights = np.column_stack(
        [
            1.0 - upper_weight,
            upper_weight,
        ]
    )

    return weighted_combine(
        values,
        weights,
        allow_partial=False,
    )


# =============================================================================
# 10. 检查点
# =============================================================================

def calculate_input_signature(
    target_times_ns: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
) -> str:
    digest = hashlib.sha256()

    digest.update(
        np.ascontiguousarray(
            target_times_ns
        ).view(np.uint8)
    )
    digest.update(
        np.ascontiguousarray(
            latitude
        ).view(np.uint8)
    )
    digest.update(
        np.ascontiguousarray(
            longitude
        ).view(np.uint8)
    )

    return digest.hexdigest()


def save_checkpoint(
    path: Path,
    collection: ERA5Collection,
    result: Dict[str, np.ndarray],
    completed_groups: int,
    total_groups: int,
    total_rows: int,
    input_signature: str,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_name(
        path.name + ".tmp"
    )

    payload = {
        "label": np.asarray(
            collection.label
        ),
        "completed_groups": np.asarray(
            completed_groups,
            dtype=np.int64,
        ),
        "total_groups": np.asarray(
            total_groups,
            dtype=np.int64,
        ),
        "total_rows": np.asarray(
            total_rows,
            dtype=np.int64,
        ),
        "timeline_start": np.asarray(
            collection.times_ns[0],
            dtype=np.int64,
        ),
        "timeline_end": np.asarray(
            collection.times_ns[-1],
            dtype=np.int64,
        ),
        "input_signature": np.asarray(
            input_signature
        ),
    }

    for name, values in result.items():
        payload[f"result__{name}"] = values

    with open(
        temporary,
        "wb",
    ) as file_handle:
        np.savez(
            file_handle,
            **payload,
        )

    os.replace(
        temporary,
        path,
    )


def load_checkpoint(
    path: Path,
    collection: ERA5Collection,
    result: Dict[str, np.ndarray],
    total_groups: int,
    total_rows: int,
    input_signature: str,
) -> int:
    if not path.exists():
        return 0

    try:
        with np.load(
            path,
            allow_pickle=False,
        ) as archive:
            label = str(
                archive["label"].item()
            )
            stored_total_groups = int(
                archive["total_groups"].item()
            )
            stored_total_rows = int(
                archive["total_rows"].item()
            )
            timeline_start = int(
                archive["timeline_start"].item()
            )
            timeline_end = int(
                archive["timeline_end"].item()
            )
            completed = int(
                archive["completed_groups"].item()
            )

            # 兼容旧版检查点：旧文件可能没有 input_signature。
            signature_ok = True

            if "input_signature" in archive:
                signature_ok = (
                    str(
                        archive[
                            "input_signature"
                        ].item()
                    )
                    == input_signature
                )

            compatible = (
                label == collection.label
                and stored_total_groups
                == total_groups
                and stored_total_rows
                == total_rows
                and timeline_start
                == int(
                    collection.times_ns[0]
                )
                and timeline_end
                == int(
                    collection.times_ns[-1]
                )
                and signature_ok
                and 0
                <= completed
                <= total_groups
            )

            if not compatible:
                print(
                    f"  忽略不匹配检查点："
                    f"{path.name}"
                )
                return 0

            for name in result:
                key = f"result__{name}"

                if key not in archive:
                    return 0

                stored = archive[key]

                if stored.shape != result[name].shape:
                    return 0

                result[name][:] = stored

        print(
            f"  已恢复检查点：{path.name}；"
            f"时间区间组 {completed}/{total_groups}"
        )

        return completed

    except Exception as exc:
        print(
            f"  检查点读取失败，从头计算："
            f"{type(exc).__name__}: {exc}"
        )
        return 0


# =============================================================================
# 11. 集合插值
# =============================================================================

def interpolate_collection(
    collection: ERA5Collection,
    target_times_ns: np.ndarray,
    target_latitude: np.ndarray,
    target_longitude: np.ndarray,
    circular_variables: Sequence[str],
    checkpoint_path: Path,
    input_signature: str,
) -> Dict[str, np.ndarray]:
    """
    对一个 ERA5 集合执行空间双线性 + 时间线性插值。

    此版本增加三类实时反馈：
    1. 开始时报告有效行数和时间区间组数；
    2. 读取较慢的时间组前报告 UTC 和文件名；
    3. 按行数、组数或经过秒数报告进度、速度和预计剩余时间。
    """
    total_rows = len(target_times_ns)

    result = {
        name: np.full(
            total_rows,
            np.nan,
            dtype=np.float64,
        )
        for name in collection.variable_candidates
    }

    print(
        f"  {collection.label}：计算经纬度和时间括号……",
        flush=True,
    )

    (
        lat0,
        lat1,
        lat_weight,
        valid_lat,
    ) = bracket_nonperiodic_axis(
        collection.latitude,
        target_latitude,
    )

    (
        lon0,
        lon1,
        lon_weight,
        valid_lon,
    ) = bracket_longitude(
        collection.longitude,
        target_longitude,
    )

    (
        lower_global,
        upper_global,
        time_weight,
        valid_time,
    ) = calculate_time_brackets(
        collection.times_ns,
        target_times_ns,
    )

    valid = (
        valid_lat
        & valid_lon
        & valid_time
        & np.isfinite(target_latitude)
        & np.isfinite(target_longitude)
    )

    valid_indices = np.where(valid)[0]

    print(
        f"  {collection.label}：有效记录 "
        f"{len(valid_indices)}/{total_rows}；"
        f"无效或超出范围 {total_rows - len(valid_indices)}",
        flush=True,
    )

    if not len(valid_indices):
        print(
            f"  {collection.label}：没有可插值记录，直接返回空结果。",
            flush=True,
        )
        return result

    key_base = len(collection.times_ns) + 1

    pair_keys = (
        lower_global[valid_indices].astype(np.int64)
        * key_base
        + upper_global[valid_indices].astype(np.int64)
    )

    order = np.argsort(
        pair_keys,
        kind="stable",
    )

    sorted_indices = valid_indices[order]
    sorted_keys = pair_keys[order]

    boundaries = np.flatnonzero(
        np.r_[
            True,
            sorted_keys[1:] != sorted_keys[:-1],
            True,
        ]
    )

    total_groups = len(boundaries) - 1
    average_rows = len(valid_indices) / total_groups

    print(
        f"  {collection.label}：共 {total_groups} 个时间区间组；"
        f"平均每组 {average_rows:.2f} 行。"
        "每个组可能读取 1–2 个完整二维时次。",
        flush=True,
    )

    start_group = load_checkpoint(
        checkpoint_path,
        collection,
        result,
        total_groups,
        total_rows,
        input_signature,
    )

    if start_group == total_groups:
        print(
            f"  {collection.label} 已由完整检查点恢复。",
            flush=True,
        )
        return result

    processed_rows = int(boundaries[start_group])
    completed_groups = start_group

    last_reported_rows = processed_rows
    last_reported_groups = completed_groups
    last_report_clock = time.time()
    interpolation_start_clock = last_report_clock

    cache = TimeSliceCache(MAX_CACHED_TIME_SLICES)
    circular_set = set(circular_variables)

    def describe_time_index(global_index: int) -> Tuple[str, str]:
        file_id = int(collection.file_ids[global_index])
        utc_text = str(pd.Timestamp(collection.times_ns[global_index]))
        filename = collection.files[file_id].path.name
        return utc_text, filename

    def format_eta(seconds: float) -> str:
        if not np.isfinite(seconds) or seconds < 0:
            return "未知"
        seconds = int(round(seconds))
        hours, remainder = divmod(seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        if hours:
            return f"{hours:d}小时{minutes:02d}分"
        if minutes:
            return f"{minutes:d}分{secs:02d}秒"
        return f"{secs:d}秒"

    try:
        for group_number in range(start_group, total_groups):
            start = boundaries[group_number]
            stop = boundaries[group_number + 1]
            rows = sorted_indices[start:stop]

            lower_index = int(lower_global[rows[0]])
            upper_index = int(upper_global[rows[0]])

            should_announce_read = (
                group_number == start_group
                or (group_number + 1) % PROGRESS_EVERY_GROUPS == 0
            )

            if should_announce_read:
                lower_utc, lower_file = describe_time_index(lower_index)
                upper_utc, upper_file = describe_time_index(upper_index)
                print(
                    f"  {collection.label} 正在处理组 "
                    f"{group_number + 1}/{total_groups}；"
                    f"本组 {len(rows)} 行；"
                    f"UTC {lower_utc} -> {upper_utc}；"
                    f"文件 {lower_file} -> {upper_file}",
                    flush=True,
                )

            group_clock = time.time()

            lower_fields = cache.get(
                collection,
                lower_index,
            )

            if lower_index == upper_index:
                upper_fields = lower_fields
            else:
                upper_fields = cache.get(
                    collection,
                    upper_index,
                )

            read_elapsed = time.time() - group_clock

            if read_elapsed >= PROGRESS_EVERY_SECONDS:
                print(
                    f"    本组二维场读取耗时 {read_elapsed:.1f} 秒，"
                    "程序仍在运行。",
                    flush=True,
                )

            grid_weights = spatial_weights(
                lat_weight[rows],
                lon_weight[rows],
            )

            for name in collection.variable_candidates:
                lower_corners = extract_corners(
                    lower_fields[name],
                    lat0[rows],
                    lat1[rows],
                    lon0[rows],
                    lon1[rows],
                )
                upper_corners = extract_corners(
                    upper_fields[name],
                    lat0[rows],
                    lat1[rows],
                    lon0[rows],
                    lon1[rows],
                )

                if name in circular_set:
                    lower_radians = np.deg2rad(lower_corners)
                    upper_radians = np.deg2rad(upper_corners)

                    lower_sine = spatial_linear(
                        np.sin(lower_radians),
                        grid_weights,
                    )
                    lower_cosine = spatial_linear(
                        np.cos(lower_radians),
                        grid_weights,
                    )
                    upper_sine = spatial_linear(
                        np.sin(upper_radians),
                        grid_weights,
                    )
                    upper_cosine = spatial_linear(
                        np.cos(upper_radians),
                        grid_weights,
                    )

                    sine = temporal_linear(
                        lower_sine,
                        upper_sine,
                        time_weight[rows],
                    )
                    cosine = temporal_linear(
                        lower_cosine,
                        upper_cosine,
                        time_weight[rows],
                    )

                    angle = (
                        np.degrees(np.arctan2(sine, cosine))
                        + 360.0
                    ) % 360.0

                    invalid_angle = (
                        ~np.isfinite(sine)
                        | ~np.isfinite(cosine)
                        | (np.hypot(sine, cosine) < 1e-12)
                    )
                    angle[invalid_angle] = np.nan
                    result[name][rows] = angle

                else:
                    lower_value = spatial_linear(
                        lower_corners,
                        grid_weights,
                    )
                    upper_value = spatial_linear(
                        upper_corners,
                        grid_weights,
                    )

                    result[name][rows] = temporal_linear(
                        lower_value,
                        upper_value,
                        time_weight[rows],
                    )

            processed_rows += len(rows)
            completed_groups = group_number + 1
            now = time.time()

            checkpoint_due = (
                completed_groups % CHECKPOINT_EVERY_GROUPS == 0
                or completed_groups == total_groups
            )

            if checkpoint_due:
                checkpoint_clock = time.time()
                print(
                    f"  {collection.label}：保存检查点 "
                    f"{completed_groups}/{total_groups}……",
                    flush=True,
                )
                save_checkpoint(
                    checkpoint_path,
                    collection,
                    result,
                    completed_groups,
                    total_groups,
                    total_rows,
                    input_signature,
                )
                print(
                    f"  {collection.label}：检查点已保存，耗时 "
                    f"{time.time() - checkpoint_clock:.1f} 秒。",
                    flush=True,
                )

            report_due = (
                processed_rows - last_reported_rows >= PROGRESS_EVERY_ROWS
                or completed_groups - last_reported_groups
                >= PROGRESS_EVERY_GROUPS
                or now - last_report_clock >= PROGRESS_EVERY_SECONDS
                or completed_groups == total_groups
            )

            if report_due:
                elapsed = max(now - interpolation_start_clock, 1e-9)
                groups_done_this_run = completed_groups - start_group
                group_rate = groups_done_this_run / elapsed
                remaining_groups = total_groups - completed_groups
                eta_seconds = (
                    remaining_groups / group_rate
                    if group_rate > 0
                    else float("nan")
                )
                percent = completed_groups / total_groups * 100.0

                print(
                    f"  {collection.label} 进度："
                    f"行 {processed_rows}/{len(valid_indices)}；"
                    f"时间组 {completed_groups}/{total_groups} "
                    f"({percent:.2f}%)；"
                    f"速度 {group_rate:.2f} 组/秒；"
                    f"预计剩余 {format_eta(eta_seconds)}",
                    flush=True,
                )

                last_reported_rows = processed_rows
                last_reported_groups = completed_groups
                last_report_clock = now

    except Exception:
        save_checkpoint(
            checkpoint_path,
            collection,
            result,
            completed_groups,
            total_groups,
            total_rows,
            input_signature,
        )
        print(
            f"\n  已保存故障前检查点：{checkpoint_path}",
            flush=True,
        )
        raise

    finally:
        cache.clear()

    return result


# =============================================================================
# 12. 输出统计
# =============================================================================

def print_stats(
    dataframe: pd.DataFrame,
    columns: Sequence[str],
) -> None:
    print("\n" + "=" * 76)
    print("插值结果统计")
    print("=" * 76)

    total = len(dataframe)

    for column in columns:
        count = int(
            dataframe[column]
            .notna()
            .sum()
        )

        percentage = (
            count / total * 100
            if total
            else 0.0
        )

        if count:
            range_text = (
                f"{dataframe[column].min():.4f}"
                f" ~ "
                f"{dataframe[column].max():.4f}"
            )
        else:
            range_text = "无有效值"

        print(
            f"{column:<12}"
            f"{count}/{total} "
            f"({percentage:6.2f}%)；"
            f"范围 {range_text}"
        )


def metadata_sheet() -> pd.DataFrame:
    return pd.DataFrame(
        [
            [
                "wind_s",
                "10米风速",
                "kn",
                "u10、v10",
                "时空插值后计算",
            ],
            [
                "wind_d",
                "10米气象风向（风的来向）",
                "degree true",
                "u10、v10",
                "时空插值后计算",
            ],
            [
                "wave_h",
                "综合有效波高",
                "m",
                "swh",
                "时间线性 + 空间双线性",
            ],
            [
                "wave_d",
                "平均波向",
                "degree true",
                "mwd",
                "正弦余弦圆周插值",
            ],
            [
                "wave_p",
                "平均波周期",
                "s",
                "mwp",
                "时间线性 + 空间双线性",
            ],
            [
                "surface_t",
                "海表温度",
                "℃",
                "sst",
                "插值后 K - 273.15",
            ],
            [
                "surface_p",
                "表面气压",
                "Pa",
                "sp",
                "时间线性 + 空间双线性",
            ],
        ],
        columns=[
            "字段",
            "含义",
            "单位",
            "ERA5来源",
            "处理方法",
        ],
    )


# =============================================================================
# 13. Excel 稳健读取
# =============================================================================

def read_input_excel(
    path: Path,
    sheet_name,
) -> pd.DataFrame:
    """优先使用 calamine 读取大体积 Excel，并给出明确诊断。"""
    file_size_mb = path.stat().st_size / (1024 ** 2)
    print(f"  Excel 文件大小：{file_size_mb:.2f} MB", flush=True)

    # pandas 2.2+ 支持 calamine。它通常比 openpyxl 读取大表更快。
    try:
        import python_calamine  # noqa: F401
        has_calamine = True
    except ImportError:
        has_calamine = False

    engines = ["calamine", "openpyxl"] if has_calamine else ["openpyxl"]
    errors = []

    for engine in engines:
        print(f"  尝试 Excel 引擎：{engine}", flush=True)
        read_clock = time.time()

        try:
            dataframe = pd.read_excel(
                path,
                sheet_name=sheet_name,
                engine=engine,
            )

            print(
                f"  Excel 读取完成：{len(dataframe)} 行，"
                f"{len(dataframe.columns)} 列；"
                f"耗时 {time.time() - read_clock:.1f} 秒",
                flush=True,
            )
            return dataframe

        except MemoryError as exc:
            raise MemoryError(
                "读取 Excel 时内存不足。通常是工作表包含大量实际或带格式的空行/空列，"
                "或者文件数据量超过当前内存。请在 Excel 中按 Ctrl+End 检查最后使用单元格，"
                "删除真实数据范围之外的整行和整列后另存为新文件；也可先另存为 CSV。"
            ) from exc

        except KeyboardInterrupt as exc:
            raise RuntimeError(
                "Excel 读取被手动中断，程序尚未进入 ERA5 插值。"
                "请清理工作表多余空行/空列，或安装 python-calamine 后重试："
                "python -m pip install python-calamine"
            ) from exc

        except Exception as exc:
            errors.append(
                f"{engine}: {type(exc).__name__}: {exc}"
            )
            print(
                f"  {engine} 读取失败："
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )

    extra = (
        "\n建议先运行以下测试确认是否只是数据量问题：\n"
        "pd.read_excel(INPUT_EXCEL, sheet_name=INPUT_SHEET, nrows=1000)"
    )
    raise RuntimeError(
        "所有 Excel 引擎均读取失败：\n  "
        + "\n  ".join(errors)
        + extra
    )


# =============================================================================
# 14. 主程序
# =============================================================================

def main() -> None:
    # 避免某些 IDE、重定向终端或批处理环境缓存 print 输出。
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    start_clock = time.time()

    print("=" * 88)
    print(
        "ERA5 七字段稳健插值 v3："
        "文件深检 + 完整二维时次读取 + 断点续算"
    )
    print("=" * 88)
    print(f"输入 Excel：{INPUT_EXCEL}")
    print(f"ERA5 目录：{RAW_DIR}")
    print(f"输出 Excel：{OUTPUT_EXCEL}")
    print(f"文件检验模式：{VALIDATION_MODE}")
    print(f"检查点目录：{CHECKPOINT_DIR}")
    print("=" * 88)

    if not INPUT_EXCEL.exists():
        raise FileNotFoundError(
            f"Excel 不存在：{INPUT_EXCEL}"
        )

    check_expected_files()

    surface_files = [
        Path(path)
        for path in sorted(
            glob.glob(
                SURFACE_PATTERN
            )
        )
    ]
    wave_files = [
        Path(path)
        for path in sorted(
            glob.glob(
                WAVE_PATTERN
            )
        )
    ]

    validate_all_files(
        surface_files,
        wave_files,
    )

    print("\n[1/6] 读取船舶 Excel……")

    dataframe = read_input_excel(
        INPUT_EXCEL,
        INPUT_SHEET,
    )

    time_column = find_dataframe_column(
        dataframe.columns,
        TIME_COLUMN_CANDIDATES,
        ("utc", "time", "时间"),
    )
    lat_column = find_dataframe_column(
        dataframe.columns,
        LAT_COLUMN_CANDIDATES,
        ("latitude", "lat", "纬度"),
    )
    lon_column = find_dataframe_column(
        dataframe.columns,
        LON_COLUMN_CANDIDATES,
        ("longitude", "lon", "经度"),
    )

    print(
        f"  数据行数：{len(dataframe)}；"
        f"字段数：{len(dataframe.columns)}"
    )
    print(f"  时间列：{time_column}")
    print(f"  纬度列：{lat_column}")
    print(f"  经度列：{lon_column}")

    dataframe[
        "ERA5_interp_time_UTC"
    ] = parse_input_utc(
        dataframe[time_column]
    )

    latitude = pd.to_numeric(
        dataframe[lat_column],
        errors="coerce",
    ).to_numpy(dtype=np.float64)

    longitude = pd.to_numeric(
        dataframe[lon_column],
        errors="coerce",
    ).to_numpy(dtype=np.float64)

    time_series = dataframe[
        "ERA5_interp_time_UTC"
    ]

    time_valid = (
        time_series
        .notna()
        .to_numpy()
    )

    target_times_ns = np.full(
        len(dataframe),
        np.iinfo(np.int64).min,
        dtype=np.int64,
    )

    target_times_ns[time_valid] = (
        time_series[time_valid]
        .astype("datetime64[ns]")
        .astype(np.int64)
    )

    invalid_input = (
        ~time_valid
        | ~np.isfinite(latitude)
        | ~np.isfinite(longitude)
    )

    print(
        f"  无效时间/经纬度记录："
        f"{int(invalid_input.sum())}"
    )

    input_signature = (
        calculate_input_signature(
            target_times_ns,
            latitude,
            longitude,
        )
    )

    print("\n[2/6] 建立风温压索引……")
    surface_collection = ERA5Collection(
        surface_files,
        SURFACE_VARIABLE_CANDIDATES,
        "风温压数据",
    )

    print("\n[3/6] 建立波浪索引……")
    wave_collection = ERA5Collection(
        wave_files,
        WAVE_VARIABLE_CANDIDATES,
        "波浪数据",
    )

    print("\n[4/6] 插值风温压……")
    surface_result = interpolate_collection(
        surface_collection,
        target_times_ns,
        latitude,
        longitude,
        circular_variables=(),
        checkpoint_path=SURFACE_CHECKPOINT,
        input_signature=input_signature,
    )

    print("\n[5/6] 插值波浪……")
    wave_result = interpolate_collection(
        wave_collection,
        target_times_ns,
        latitude,
        longitude,
        circular_variables=("mwd",),
        checkpoint_path=WAVE_CHECKPOINT,
        input_signature=input_signature,
    )

    print("\n[6/6] 计算字段并保存……")

    u10 = surface_result["u10"]
    v10 = surface_result["v10"]

    wind_speed_ms = np.hypot(
        u10,
        v10,
    )

    wind_s = (
        wind_speed_ms
        * MS_TO_KNOT
    )

    wind_d = (
        np.degrees(
            np.arctan2(
                -u10,
                -v10,
            )
        )
        + 360.0
    ) % 360.0

    wind_d[
        ~np.isfinite(
            wind_speed_ms
        )
        | (
            wind_speed_ms
            < CALM_WIND_THRESHOLD_MS
        )
    ] = np.nan

    dataframe["wind_s"] = wind_s
    dataframe["wind_d"] = wind_d
    dataframe["wave_h"] = (
        wave_result["swh"]
    )
    dataframe["wave_d"] = (
        wave_result["mwd"]
    )
    dataframe["wave_p"] = (
        wave_result["mwp"]
    )
    dataframe["surface_t"] = (
        surface_result["sst"]
        - KELVIN_TO_CELSIUS
    )
    dataframe["surface_p"] = (
        surface_result["sp"]
    )

    final_columns = (
        "wind_s",
        "wind_d",
        "wave_h",
        "wave_d",
        "wave_p",
        "surface_t",
        "surface_p",
    )

    print_stats(
        dataframe,
        final_columns,
    )

    preview_columns = [
        time_column,
        "ERA5_interp_time_UTC",
        lat_column,
        lon_column,
        *final_columns,
    ]

    print("\n前20条预览：")
    print(
        dataframe[
            preview_columns
        ]
        .head(20)
        .to_string(index=False)
    )

    OUTPUT_EXCEL.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with pd.ExcelWriter(
        OUTPUT_EXCEL,
        engine="openpyxl",
        datetime_format=(
            "yyyy-mm-dd hh:mm:ss"
        ),
    ) as writer:
        dataframe.to_excel(
            writer,
            sheet_name="continership_ERA5",
            index=False,
        )

        metadata_sheet().to_excel(
            writer,
            sheet_name="ERA5_variables",
            index=False,
        )

    print("\n" + "=" * 88)
    print("全部完成")
    print(f"输出文件：{OUTPUT_EXCEL}")
    print(
        f"总耗时："
        f"{(time.time() - start_clock) / 60:.2f} 分钟"
    )
    print("=" * 88)


if __name__ == "__main__":
    main()