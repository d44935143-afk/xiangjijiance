from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from static_rmse import static_rmse
from matplotlib import font_manager

for font_path in (Path("C:/Windows/Fonts/msyh.ttc"), Path("C:/Windows/Fonts/simhei.ttf")):
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams["font.sans-serif"] = [font_manager.FontProperties(fname=str(font_path)).get_name()]
        break
plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# 配置区：后续实验优先只修改这里
# ============================================================
BASE_DIR = Path(__file__).resolve().parent

CAMERA_FILE = BASE_DIR / "camera_data" / "mubiao1.xlsx"
GNSS_FILE = BASE_DIR / "gnss_data" / "ohlo3000.pos"
OUTPUT_FILE = BASE_DIR / "dandian_count_sampled_plots" / "dandian_count_sampled.xlsx"
PLOT_DIR = BASE_DIR / "dandian_count_sampled_plots"

# 相机文件配置。当前插值文件无表头，第一列为相机 Y 位移。
CAMERA_HAS_HEADER = False
CAMERA_TIME_COLUMN = None
CAMERA_VALUE_COLUMN = 0
CAMERA_TARGET2_COLUMN = 1
CAMERA_SAMPLE_INTERVAL_SECONDS = 30.0
CAMERA_ZERO_AT_START = True

# Excel GNSS 字段可手动指定；None 表示自动识别。
GNSS_COLUMN_MAP = {
    "time": None,
    "X": None,
    "Y": None,
    "Z": None,
    "ratio": None,
    "Q": None,
    "ns": None,
}

# GNSS 质量筛选。
RATIO_THRESHOLD = 3.0
VALID_Q_VALUES = {1}  # 设为 None 可关闭 Q 筛选。
MIN_SATELLITES = None  # 例如设为 6；None 表示不按卫星数筛选。

# GNSS 突变与 Hampel/MAD 异常检测，单位均为 mm。
GNSS_HORIZONTAL_AXIS_JUMP_THRESHOLD_MM = 10.0
GNSS_VERTICAL_AXIS_JUMP_THRESHOLD_MM = 20.0
GNSS_3D_JUMP_THRESHOLD_MM = 30.0
GNSS_RETURN_TOLERANCE_MM = 10.0
MAD_WINDOW = 31
MAD_THRESHOLD = 3.5

# GNSS 长期基准。
GNSS_SMOOTH_WINDOW =1000
GNSS_SMOOTH_MIN_VALID = 10
GNSS_SMOOTH_METHOD = "mean"  # "mean" 或 "median"
GNSS_LONG_MEDIAN_WINDOW_SECONDS = 1800.0  # 默认 30 分钟。
GNSS_LONG_MEDIAN_MIN_VALID = 10

# GNSS 可靠性门控。
CONSISTENCY_WINDOW_SECONDS = 300.0  # 相机/GNSS 增量一致性窗口，默认 5 分钟。
CAMERA_GNSS_INCREMENT_ERROR_THRESHOLD_MM = 10.0
GNSS_LOCAL_VOLATILITY_WINDOW_SECONDS = 300.0
GNSS_LOCAL_VOLATILITY_THRESHOLD_MM = 0.5

# 互补滤波与相机异常检测。
GNSS_CORRECTION_GAIN_K = 0.30
CAMERA_JUMP_THRESHOLD_MM = 20.0
INITIALIZATION_DURATION_SECONDS = 360.0

# 时间同步。
TIME_SYNC_METHOD = "count_uniform"  # 按数量抽取后逐行配对，不按时间差筛选。
MAX_TIME_DIFFERENCE_SECONDS = None

# GNSS 参考坐标。None 时使用前 REFERENCE_SAMPLE_COUNT 个质量合格历元均值。
MANUAL_REFERENCE_XYZ_M = None
REFERENCE_SAMPLE_COUNT = 60

# 完整 3x3 GNSS->相机旋转矩阵。第二行为相机 Y 轴方向。
# 当前矩阵保持之前的 Y 轴投影 [1,1,1]/sqrt(3)，其余两行补成正交基。
# 得到实际外参后，只需替换此矩阵。
ROTATION_MATRIX = np.array(
    [
        [1 / np.sqrt(2), -1 / np.sqrt(2), 0.0],
        [1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3)],
        [-1 / np.sqrt(6), -1 / np.sqrt(6), 2 / np.sqrt(6)],
    ],
    dtype=float,
)

# GNSS 相机 Y 轴（沉降）方向符号：-1 表示正负号取反，1 表示保持原方向。
GNSS_SETTLEMENT_SIGN = 1.0

SAVE_PLOTS = True
SHOW_PLOTS = False
STANDARD_DEVIATION_DDOF = 1  # 1 为样本标准差，0 为总体标准差。


def validate_config():
    """检查配置参数和旋转矩阵是否合法。"""
    if not 0.0 <= GNSS_CORRECTION_GAIN_K <= 1.0:
        raise ValueError("GNSS_CORRECTION_GAIN_K 必须位于 0 到 1 之间。")
    if MAD_WINDOW < 3 or MAD_WINDOW % 2 == 0:
        raise ValueError("MAD_WINDOW 必须是大于等于 3 的奇数。")
    if GNSS_SMOOTH_WINDOW < 1:
        raise ValueError("GNSS_SMOOTH_WINDOW 必须大于 0。")
    if GNSS_LONG_MEDIAN_WINDOW_SECONDS <= 0:
        raise ValueError("GNSS_LONG_MEDIAN_WINDOW_SECONDS 必须大于 0。")
    if CONSISTENCY_WINDOW_SECONDS <= 0:
        raise ValueError("CONSISTENCY_WINDOW_SECONDS 必须大于 0。")
    if ROTATION_MATRIX.shape != (3, 3):
        raise ValueError("ROTATION_MATRIX 必须是 3x3 矩阵。")
    if GNSS_SETTLEMENT_SIGN not in {-1.0, 1.0}:
        raise ValueError("GNSS_SETTLEMENT_SIGN 只能是 -1.0 或 1.0。")
    if not np.allclose(
        ROTATION_MATRIX @ ROTATION_MATRIX.T,
        np.eye(3),
        atol=1e-6,
    ):
        raise ValueError("ROTATION_MATRIX 的行必须构成正交单位基。")


def _find_column(columns, configured, candidates, required=False):
    """根据手动配置或候选名称寻找字段。"""
    if configured is not None:
        if configured not in columns:
            raise ValueError(f"配置的字段不存在：{configured}")
        return configured

    normalized = {str(column).strip().lower(): column for column in columns}
    for candidate in candidates:
        if candidate.lower() in normalized:
            return normalized[candidate.lower()]

    if required:
        raise ValueError(f"无法自动识别字段，候选名称：{candidates}")
    return None


def load_camera_data(path):
    """读取相机时间和位移；无时间列时稍后根据 GNSS 起点生成时间。"""
    header = 0 if CAMERA_HAS_HEADER else None
    data = pd.read_excel(path, header=header)
    if data.empty:
        raise ValueError(f"相机文件为空：{path}")

    value_column = CAMERA_VALUE_COLUMN
    if value_column not in data.columns:
        if isinstance(value_column, int) and value_column < data.shape[1]:
            value_column = data.columns[value_column]
        else:
            raise ValueError(f"相机位移字段不存在：{CAMERA_VALUE_COLUMN}")

    camera_y = pd.to_numeric(data[value_column], errors="coerce")
    if camera_y.isna().any():
        bad_rows = (camera_y[camera_y.isna()].index + 1).tolist()
        raise ValueError(f"相机位移存在非数值，行号：{bad_rows[:20]}")

    if CAMERA_ZERO_AT_START:
        camera_y = camera_y - camera_y.iloc[0]

    target2_column = CAMERA_TARGET2_COLUMN
    if target2_column not in data.columns:
        if isinstance(target2_column, int) and target2_column < data.shape[1]:
            target2_column = data.columns[target2_column]
        else:
            raise ValueError(f"目标二位移列不存在: {CAMERA_TARGET2_COLUMN}")

    target2_y = pd.to_numeric(data[target2_column], errors="coerce")
    if target2_y.isna().any():
        bad_rows = (target2_y[target2_y.isna()].index + 1).tolist()
        raise ValueError(f"目标二位移存在非数值，行号: {bad_rows[:20]}")
    if CAMERA_ZERO_AT_START:
        target2_y = target2_y - target2_y.iloc[0]

    if CAMERA_TIME_COLUMN is None:
        camera_time = pd.Series(pd.NaT, index=data.index, dtype="datetime64[ns]")
    else:
        if CAMERA_TIME_COLUMN not in data.columns:
            raise ValueError(f"相机时间字段不存在：{CAMERA_TIME_COLUMN}")
        camera_time = pd.to_datetime(data[CAMERA_TIME_COLUMN], errors="coerce")
        if camera_time.isna().any():
            raise ValueError("相机时间列存在无法解析的值。")

    return pd.DataFrame(
        {
            "camera_time": camera_time,
            "camera_original": camera_y.to_numpy(dtype=float),
            "target2_original": target2_y.to_numpy(dtype=float),
        }
    )


def _load_gnss_pos(path):
    """读取 RTKPOST POS 文件，保留绝对 ECEF 坐标和质量字段。"""
    raw = pd.read_csv(path, sep=r"\s+", comment="%", header=None)
    if raw.empty or raw.shape[1] < 15:
        raise ValueError(f"POS 文件格式不完整：{path}")

    time = pd.to_datetime(
        raw.iloc[:, 0].astype(str) + " " + raw.iloc[:, 1].astype(str),
        errors="coerce",
    )
    result = pd.DataFrame(
        {
            "time": time,
            "X": pd.to_numeric(raw.iloc[:, 2], errors="coerce"),
            "Y": pd.to_numeric(raw.iloc[:, 3], errors="coerce"),
            "Z": pd.to_numeric(raw.iloc[:, 4], errors="coerce"),
            "Q": pd.to_numeric(raw.iloc[:, 5], errors="coerce"),
            "ns": pd.to_numeric(raw.iloc[:, 6], errors="coerce"),
            "sdx": pd.to_numeric(raw.iloc[:, 7], errors="coerce"),
            "sdy": pd.to_numeric(raw.iloc[:, 8], errors="coerce"),
            "sdz": pd.to_numeric(raw.iloc[:, 9], errors="coerce"),
            "sdxy": pd.to_numeric(raw.iloc[:, 10], errors="coerce"),
            "sdyz": pd.to_numeric(raw.iloc[:, 11], errors="coerce"),
            "sdzx": pd.to_numeric(raw.iloc[:, 12], errors="coerce"),
            "age": pd.to_numeric(raw.iloc[:, 13], errors="coerce"),
            "ratio": pd.to_numeric(raw.iloc[:, 14], errors="coerce"),
        }
    )
    return result


def _load_gnss_excel(path):
    """读取 Excel GNSS，并自动识别常见字段名称。"""
    data = pd.read_excel(path)
    columns = list(data.columns)

    time_col = _find_column(
        columns,
        GNSS_COLUMN_MAP["time"],
        ["time", "时间", "datetime", "timestamp", "gpst"],
        required=True,
    )
    x_col = _find_column(
        columns,
        GNSS_COLUMN_MAP["X"],
        ["x", "x(m)", "x坐标(m)", "x-ecef(m)"],
        required=True,
    )
    y_col = _find_column(
        columns,
        GNSS_COLUMN_MAP["Y"],
        ["y", "y(m)", "y坐标(m)", "y-ecef(m)"],
        required=True,
    )
    z_col = _find_column(
        columns,
        GNSS_COLUMN_MAP["Z"],
        ["z", "z(m)", "z坐标(m)", "z-ecef(m)"],
        required=True,
    )
    ratio_col = _find_column(columns, GNSS_COLUMN_MAP["ratio"], ["ratio"])
    q_col = _find_column(columns, GNSS_COLUMN_MAP["Q"], ["q", "quality"])
    ns_col = _find_column(
        columns,
        GNSS_COLUMN_MAP["ns"],
        ["ns", "卫星数", "satellites"],
    )

    result = pd.DataFrame(
        {
            "time": pd.to_datetime(data[time_col], errors="coerce"),
            "X": pd.to_numeric(data[x_col], errors="coerce"),
            "Y": pd.to_numeric(data[y_col], errors="coerce"),
            "Z": pd.to_numeric(data[z_col], errors="coerce"),
            "ratio": (
                pd.to_numeric(data[ratio_col], errors="coerce")
                if ratio_col is not None
                else np.nan
            ),
            "Q": (
                pd.to_numeric(data[q_col], errors="coerce")
                if q_col is not None
                else np.nan
            ),
            "ns": (
                pd.to_numeric(data[ns_col], errors="coerce")
                if ns_col is not None
                else np.nan
            ),
        }
    )
    return result


def load_gnss_data(path):
    """根据扩展名读取 POS 或 Excel GNSS，并检查基础数据。"""
    if path.suffix.lower() == ".pos":
        data = _load_gnss_pos(path)
    elif path.suffix.lower() in {".xlsx", ".xls"}:
        data = _load_gnss_excel(path)
    else:
        raise ValueError(f"不支持的 GNSS 文件格式：{path.suffix}")

    required = ["time", "X", "Y", "Z"]
    if data[required].isna().any().any():
        raise ValueError("GNSS 时间或 XYZ 坐标存在无法解析的值。")
    data = data.sort_values("time").drop_duplicates("time").reset_index(drop=True)
    return data


def apply_gnss_quality_screen(gnss):
    """根据 ratio、Q 和卫星数标记低质量点，但不覆盖原始坐标。"""
    ratio_bad = pd.Series(False, index=gnss.index)
    q_bad = pd.Series(False, index=gnss.index)
    ns_bad = pd.Series(False, index=gnss.index)

    if "ratio" in gnss and gnss["ratio"].notna().any():
        ratio_bad = gnss["ratio"].notna() & (gnss["ratio"] < RATIO_THRESHOLD)
    if VALID_Q_VALUES is not None and "Q" in gnss and gnss["Q"].notna().any():
        q_bad = gnss["Q"].notna() & ~gnss["Q"].isin(VALID_Q_VALUES)
    if MIN_SATELLITES is not None and "ns" in gnss and gnss["ns"].notna().any():
        ns_bad = gnss["ns"].notna() & (gnss["ns"] < MIN_SATELLITES)

    result = gnss.copy()
    result["gnss_ratio_bad"] = ratio_bad
    result["gnss_q_bad"] = q_bad
    result["gnss_ns_bad"] = ns_bad
    result["gnss_quality_bad"] = ratio_bad | q_bad | ns_bad
    result["gnss_ratio"] = result["ratio"]
    return result


def calculate_gnss_displacement(gnss):
    """建立 ECEF 参考坐标并将绝对坐标转换为毫米级累计位移。"""
    xyz = gnss[["X", "Y", "Z"]].to_numpy(dtype=float)

    if MANUAL_REFERENCE_XYZ_M is not None:
        reference = np.asarray(MANUAL_REFERENCE_XYZ_M, dtype=float)
        if reference.shape != (3,):
            raise ValueError("MANUAL_REFERENCE_XYZ_M 必须包含 X0、Y0、Z0。")
    else:
        good_xyz = gnss.loc[~gnss["gnss_quality_bad"], ["X", "Y", "Z"]]
        reference_samples = good_xyz.head(REFERENCE_SAMPLE_COUNT)
        if len(reference_samples) < 3:
            raise ValueError("质量合格的 GNSS 数据不足，无法计算参考坐标。")
        reference = reference_samples.mean().to_numpy(dtype=float)

    displacement_mm = (xyz - reference) * 1000.0
    result = gnss.copy()
    result[["dX_mm", "dY_mm", "dZ_mm"]] = displacement_mm
    result.attrs["reference_xyz_m"] = reference
    return result


def rolling_mad_filter(series, window, threshold):
    """使用滑动 MAD 计算鲁棒 z-score，并返回异常标记和 z-score。"""
    min_periods = max(5, window // 3)
    rolling_median = series.rolling(
        window=window,
        center=True,
        min_periods=min_periods,
    ).median()
    rolling_mad = series.rolling(
        window=window,
        center=True,
        min_periods=min_periods,
    ).apply(
        lambda values: np.nanmedian(
            np.abs(values - np.nanmedian(values))
        ),
        raw=True,
    )

    deviation = series - rolling_median
    z_score = 0.6745 * deviation / rolling_mad.replace(0.0, np.nan)
    zero_mad_outlier = rolling_mad.eq(0.0) & deviation.abs().gt(1e-9)
    bad = z_score.abs().gt(threshold) | zero_mad_outlier
    return bad.fillna(False), z_score


def detect_gnss_outliers(gnss):
    """检测三轴跳变、跳出后返回的孤立点以及滑动 MAD 异常。"""
    result = gnss.copy()
    xyz = result[["dX_mm", "dY_mm", "dZ_mm"]]
    step = xyz.diff()
    step_3d = np.sqrt((step**2).sum(axis=1))

    result["gnss_step_dx_mm"] = step["dX_mm"]
    result["gnss_step_dy_mm"] = step["dY_mm"]
    result["gnss_step_dz_mm"] = step["dZ_mm"]
    result["gnss_step_3d_mm"] = step_3d
    result["gnss_jump_suspect"] = (
        step["dX_mm"].abs().gt(GNSS_HORIZONTAL_AXIS_JUMP_THRESHOLD_MM)
        | step["dY_mm"].abs().gt(GNSS_HORIZONTAL_AXIS_JUMP_THRESHOLD_MM)
        | step["dZ_mm"].abs().gt(GNSS_VERTICAL_AXIS_JUMP_THRESHOLD_MM)
        | step_3d.gt(GNSS_3D_JUMP_THRESHOLD_MM)
    )

    next_step_3d = step_3d.shift(-1)
    return_distance = np.sqrt(((xyz.shift(-1) - xyz.shift(1)) ** 2).sum(axis=1))
    result["gnss_return_spike"] = (
        result["gnss_jump_suspect"]
        & next_step_3d.gt(GNSS_3D_JUMP_THRESHOLD_MM)
        & return_distance.le(GNSS_RETURN_TOLERANCE_MM)
    ).fillna(False)

    mad_flags = []
    for axis in ["dX_mm", "dY_mm", "dZ_mm"]:
        bad, z_score = rolling_mad_filter(result[axis], MAD_WINDOW, MAD_THRESHOLD)
        result[f"{axis}_mad_z"] = z_score
        mad_flags.append(bad)
    result["gnss_mad_bad"] = mad_flags[0] | mad_flags[1] | mad_flags[2]

    # 大位移只标为 suspect；只有质量差、跳出后返回或 MAD 孤立异常才剔除。
    result["gnss_bad"] = (
        result["gnss_quality_bad"]
        | result["gnss_return_spike"]
        | result["gnss_mad_bad"]
    )
    return result


def transform_gnss_coordinate(gnss):
    """将 GNSS 三轴毫米位移转换到相机坐标系，并保留原始与清洗结果。"""
    displacement = gnss[["dX_mm", "dY_mm", "dZ_mm"]].to_numpy(dtype=float)
    camera_xyz = displacement @ ROTATION_MATRIX.T

    result = gnss.copy()
    result[["gnss_camera_x", "gnss_camera_y", "gnss_camera_z"]] = camera_xyz
    result["gnss_original"] = (
        GNSS_SETTLEMENT_SIGN * result["gnss_camera_y"]
    )
    result["gnss_clean"] = result["gnss_original"].mask(result["gnss_bad"])
    for column in ["X", "Y", "Z", "dX_mm", "dY_mm", "dZ_mm"]:
        result[f"{column}_clean"] = result[column].mask(result["gnss_bad"])
    return result


def seconds_to_samples(time, duration_seconds):
    """根据实际时间间隔将秒数转换为至少 1 个采样点。"""
    interval = time.diff().dt.total_seconds().dropna()
    interval = interval[interval > 0]
    if interval.empty:
        return max(1, int(round(duration_seconds / CAMERA_SAMPLE_INTERVAL_SECONDS)))
    median_interval = float(interval.median())
    return max(1, int(round(duration_seconds / median_interval)))


def smooth_gnss(gnss):
    """计算短窗口统计量和用于融合的长窗口 rolling median。"""
    result = gnss.copy()
    rolling = result["gnss_clean"].rolling(
        window=GNSS_SMOOTH_WINDOW,
        min_periods=min(GNSS_SMOOTH_MIN_VALID, GNSS_SMOOTH_WINDOW),
    )
    result["gnss_mean"] = rolling.mean()
    result["gnss_median"] = rolling.median()

    if GNSS_SMOOTH_METHOD == "mean":
        result["gnss_smooth"] = result["gnss_mean"]
    elif GNSS_SMOOTH_METHOD == "median":
        result["gnss_smooth"] = result["gnss_median"]
    else:
        raise ValueError("GNSS_SMOOTH_METHOD 只能是 'mean' 或 'median'。")

    long_window = seconds_to_samples(
        result["time"],
        GNSS_LONG_MEDIAN_WINDOW_SECONDS,
    )
    result["gnss_long_median"] = result["gnss_clean"].rolling(
        window=long_window,
        center=True,
        min_periods=min(GNSS_LONG_MEDIAN_MIN_VALID, long_window),
    ).median()
    result["gnss_long_window_samples"] = long_window
    return result


def select_gnss_by_camera_count(camera, gnss):
    """按相机条数等间隔抽取 GNSS。

    零基索引为 i * M // N；相机2条、GNSS4条时取第1、3条。
    非整数比例的间隔相差至多一条；GNSS不足时拒绝重复抽取。
    假设两组数据覆盖相同观测时段，不校验实际时间同步。
    """
    n, m = len(camera), len(gnss)
    if n == 0:
        raise ValueError("相机数据为空，无法抽取 GNSS。")
    if m < n:
        raise ValueError(f"GNSS只有{m}条，相机有{n}条，无法无重复抽取。")
    indices = np.array([i * m // n for i in range(n)], dtype=np.int64)
    selected = gnss.iloc[indices].copy().reset_index(drop=True)
    selected["gnss_source_row"] = indices + 1
    selected["camera_source_row"] = np.arange(1, n + 1)
    selected["gnss_total_count"] = m
    selected["camera_total_count"] = n
    return selected


def synchronize_data(camera, gnss):
    """抽取后的 GNSS 与相机按原行序配对。"""
    if len(camera) != len(gnss):
        raise ValueError("抽取后的GNSS条数必须与相机条数一致。")
    result = gnss.copy().reset_index(drop=True)
    camera = camera.reset_index(drop=True)
    result["camera_original"] = camera["camera_original"].to_numpy()
    result["target2_original"] = camera["target2_original"].to_numpy()
    # 无相机时间时采用对应 GNSS 时间，不生成固定30秒时间轴。
    result["matched_camera_time"] = (
        result["time"] if camera["camera_time"].isna().all()
        else camera["camera_time"].to_numpy()
    )
    result["time_difference_s"] = (
        result["time"] - result["matched_camera_time"]
    ).abs().dt.total_seconds()
    result["time_sync_ok"] = True  # 数量配对完成，不表示实际时间同步。
    return result


def detect_camera_outliers(data):
    """计算相机短时增量并标记超过阈值或未完成时间匹配的点。"""
    result = data.copy()
    result["camera_increment"] = result["camera_original"].diff()
    if len(result):
        result.loc[result.index[0], "camera_increment"] = 0.0
    result["camera_bad"] = (
        ~result["time_sync_ok"]
        | result["camera_increment"].abs().gt(CAMERA_JUMP_THRESHOLD_MM)
        | result["camera_increment"].isna()
    )
    return result


def evaluate_gnss_local_volatility(gnss):
    """在完整 GNSS 时间轴上计算局部增量波动，抽取时保留结果。"""
    result = gnss.copy()
    window = seconds_to_samples(result["time"], GNSS_LOCAL_VOLATILITY_WINDOW_SECONDS)
    result["gnss_local_volatility"] = result["gnss_original"].diff().rolling(
        window=window, min_periods=min(window, max(3, window // 2)),
    ).std()
    result["gnss_volatility_ok"] = result["gnss_local_volatility"].le(
        GNSS_LOCAL_VOLATILITY_THRESHOLD_MM
    )
    result["volatility_window_samples"] = window
    return result


def evaluate_gnss_reliability(data):
    """联合质量、突变、MAD、局部波动和增量一致性建立可靠性门控。"""
    result = data.copy()
    consistency_window = seconds_to_samples(
        result["time"],
        CONSISTENCY_WINDOW_SECONDS,
    )

    result["delta_camera_window"] = (
        result["camera_original"]
        - result["camera_original"].shift(consistency_window)
    )
    result["delta_gnss_window"] = (
        result["gnss_original"]
        - result["gnss_original"].shift(consistency_window)
    )
    result["camera_gnss_increment_error"] = (
        result["delta_camera_window"] - result["delta_gnss_window"]
    ).abs()

    result["gnss_ratio_ok"] = ~result["gnss_ratio_bad"]
    result["gnss_solution_ok"] = ~(
        result["gnss_q_bad"] | result["gnss_ns_bad"]
    )
    result["gnss_jump_ok"] = ~result["gnss_jump_suspect"]
    result["gnss_mad_ok"] = ~result["gnss_mad_bad"]
    result["gnss_increment_consistency_ok"] = result[
        "camera_gnss_increment_error"
    ].le(CAMERA_GNSS_INCREMENT_ERROR_THRESHOLD_MM)

    result["gnss_reliable"] = (
        result["time_sync_ok"]
        & result["gnss_ratio_ok"]
        & result["gnss_solution_ok"]
        & result["gnss_jump_ok"]
        & result["gnss_mad_ok"]
        & result["gnss_volatility_ok"]
        & result["gnss_increment_consistency_ok"]
        & result["gnss_long_median"].notna()
    )
    result["consistency_window_samples"] = consistency_window
    return result


def complementary_fusion(data):
    """相机负责预测，只有可靠 GNSS 才能修正长期基准。"""
    result = data.copy()
    count = len(result)
    prediction = np.full(count, np.nan, dtype=float)
    fusion = np.full(count, np.nan, dtype=float)
    gnss_correction_used = np.zeros(count, dtype=bool)

    if count == 0:
        result["prediction"] = prediction
        result["fusion"] = fusion
        result["gnss_correction_used"] = gnss_correction_used
        return result

    initialization_end = result["time"].iloc[0] + pd.Timedelta(
        seconds=INITIALIZATION_DURATION_SECONDS
    )
    initialization_values = result.loc[
        (result["time"] <= initialization_end)
        & ~result["gnss_bad"]
        & result["gnss_clean"].notna(),
        "gnss_clean",
    ]

    if not initialization_values.empty:
        initial_value = initialization_values.mean()
    else:
        first_smooth = result["gnss_smooth"].dropna()
        if not first_smooth.empty:
            initial_value = first_smooth.iloc[0]
        elif pd.notna(result["camera_original"].iloc[0]):
            initial_value = result["camera_original"].iloc[0]
        else:
            raise ValueError("GNSS 和相机均无法提供融合初始值。")

    prediction[0] = initial_value
    fusion[0] = initial_value

    for index in range(1, count):
        if result["camera_bad"].iloc[index]:
            prediction[index] = fusion[index - 1]
        else:
            prediction[index] = (
                fusion[index - 1] + result["camera_increment"].iloc[index]
            )

        if result["gnss_reliable"].iloc[index]:
            fusion[index] = (
                prediction[index]
                + GNSS_CORRECTION_GAIN_K
                * (result["gnss_long_median"].iloc[index] - prediction[index])
            )
            gnss_correction_used[index] = True
        else:
            fusion[index] = prediction[index]

    result["prediction"] = prediction
    result["fusion"] = fusion
    result["gnss_correction_used"] = gnss_correction_used
    result["camera_minus_gnss"] = (
        result["camera_original"] - result["gnss_original"]
    )
    result["fusion_minus_gnss_long"] = (
        result["fusion"] - result["gnss_long_median"]
    )
    # 目标一原始位移与融合位移之差视为相机公共误差。
    # 两个相机目标均已首帧归零，因此公共误差也相对首帧归零。
    common_error = result["camera_original"] - result["fusion"]
    result["common_error"] = common_error - common_error.iloc[0]
    result["target2_compensated"] = (
        result["target2_original"] - result["common_error"]
    )
    return result


def shade_unreliable(ax, data, label=True):
    """在图中用浅红色背景标出 GNSS 不可靠历元。"""
    ax.fill_between(
        data["time"],
        0,
        1,
        where=~data["gnss_reliable"],
        transform=ax.get_xaxis_transform(),
        color="red",
        alpha=0.08,
        step="mid",
        label="GNSS不可靠" if label else None,
    )


def plot_results(data):
    """生成融合诊断图，以及目标二补偿前后的对比图。"""
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    figures = []

    fig1, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(data["time"], data["camera_original"], label="相机原始位移")
    ax1.plot(data["time"], data["gnss_original"], label="GNSS原始位移", alpha=0.75)
    ax1.set(title="目标一相机与GNSS原始位移对比", xlabel="时间", ylabel="位移 / mm")
    ax1.grid(True)
    ax1.legend()
    fig1.tight_layout()
    figures.append((fig1, "01_raw_comparison.png"))

    fig2, ax2 = plt.subplots(figsize=(12, 5))
    ax2.plot(data["time"], data["gnss_original"], label="GNSS原始位移", alpha=0.45)
    ax2.plot(data["time"], data["gnss_mean"], label="滑动均值")
    ax2.plot(data["time"], data["gnss_median"], label="滑动中位数")
    ax2.plot(
        data["time"],
        data["gnss_long_median"],
        label="GNSS长窗口中位数",
        linewidth=2,
    )
    ax2.set(title="GNSS位移平滑结果", xlabel="时间", ylabel="位移 / mm")
    ax2.grid(True)
    ax2.legend()
    fig2.tight_layout()
    figures.append((fig2, "02_gnss_smoothing.png"))

    fig3, ax3 = plt.subplots(figsize=(12, 5))
    ax3.plot(data["time"], data["camera_original"], label="目标一融合前", alpha=0.65)
    ax3.plot(data["time"], data["gnss_original"], label="GNSS原始位移", alpha=0.35)
    ax3.plot(data["time"], data["fusion"], label="目标一融合后", linewidth=2)
    shade_unreliable(ax3, data)
    ax3.set(title="目标一融合前、融合后与GNSS位移对比", xlabel="时间", ylabel="位移 / mm")
    ax3.grid(True)
    ax3.legend()
    fig3.tight_layout()
    figures.append((fig3, "dandian.png"))

    fig4, ax4 = plt.subplots(figsize=(12, 5))
    ax4.plot(
        data["time"],
        data["camera_original"] - data["gnss_long_median"],
        label="相机减GNSS长窗口中位数",
    )
    ax4.plot(
        data["time"],
        data["fusion_minus_gnss_long"],
        label="融合减GNSS长窗口中位数",
    )
    ax4.axhline(0.0, color="black", linewidth=1)
    ax4.set(title="相机与融合结果相对GNSS基准的残差", xlabel="时间", ylabel="残差 / mm")
    ax4.grid(True)
    ax4.legend()
    fig4.tight_layout()
    figures.append((fig4, "04_residuals.png"))

    fig5, ax5 = plt.subplots(figsize=(12, 5))
    ax5.plot(data["time"], data["gnss_original"], label="GNSS原始位移")
    bad = data["gnss_bad"]
    ax5.scatter(
        data.loc[bad, "time"],
        data.loc[bad, "gnss_original"],
        color="red",
        s=18,
        label="GNSS异常点",
        zorder=3,
    )
    shade_unreliable(ax5, data)
    ax5.set(title="GNSS异常点标记", xlabel="时间", ylabel="位移 / mm")
    ax5.grid(True)
    ax5.legend()
    fig5.tight_layout()
    figures.append((fig5, "05_gnss_outliers.png"))

    fig6, ax6 = plt.subplots(figsize=(12, 5))
    ax6.plot(
        data["time"],
        data["target2_original"],
        label="目标二原始位移",
        alpha=0.8,
    )
    ax6.plot(
        data["time"],
        data["common_error"],
        label="目标一估计的公共误差",
        alpha=0.7,
    )
    ax6.plot(
        data["time"],
        data["target2_compensated"],
        label="目标二补偿后",
        linewidth=2,
    )
    ax6.axhline(0.0, color="black", linewidth=1)
    ax6.set(
        title="目标二融合补偿前后位移对比",
        xlabel="时间",
        ylabel="位移 / mm",
    )
    ax6.grid(True)
    ax6.legend()
    fig6.tight_layout()
    figures.append((fig6, "06_target2_compensation.png"))

    if SAVE_PLOTS:
        for figure, filename in figures:
            figure.savefig(PLOT_DIR / filename, dpi=160)

    if SHOW_PLOTS:
        plt.show()
    else:
        for figure, _ in figures:
            plt.close(figure)


def calculate_statistics(data):
    """位移统计仅描述波动和趋势；无独立真值时不计算误差精度。"""
    hours = (data["time"] - data["time"].iloc[0]).dt.total_seconds() / 3600.0
    rows = []
    for quantity, column in [
        ("目标一融合前", "camera_original"),
        ("目标一融合后", "fusion"),
        ("GNSS原始投影", "gnss_original"),
        ("目标二原始位移", "target2_original"),
        ("GNSS剔除异常后", "gnss_clean"),
        ("目标二补偿后", "target2_compensated"),
    ]:
        values = pd.to_numeric(data[column], errors="coerce")
        valid = np.isfinite(values) & np.isfinite(hours)
        y, t = values[valid], hours[valid]
        slope = (float(np.polyfit(t, y, 1)[0])
                 if len(y) > 1 and np.ptp(t) > 0 else np.nan)
        rows.append({
            "quantity": quantity,
            "source_column": column,
            "RMSE(mm)": static_rmse(data, column, GNSS_SETTLEMENT_SIGN * ROTATION_MATRIX[1]),
            "standard_deviation_mm": y.std(ddof=STANDARD_DEVIATION_DDOF),
            "valid_count": len(y),
            "mean_mm": y.mean(),
            "min_mm": y.min(),
            "max_mm": y.max(),
            "range_mm": y.max() - y.min(),
            "slope_mm_per_hour": slope,
        })
    return pd.DataFrame(rows).rename(columns={"quantity": "数据类型", "source_column": "数据列", "standard_deviation_mm": "标准差(mm)", "valid_count": "有效数量", "mean_mm": "均值(mm)", "min_mm": "最小值(mm)", "max_mm": "最大值(mm)", "range_mm": "极差(mm)", "slope_mm_per_hour": "趋势斜率(mm/小时)"})


def save_results(data, reference_xyz_m):
    """将融合明细和关键参数保存为新的 Excel 文件。"""
    output_columns = [
        "gnss_source_row", "camera_source_row",
        "gnss_total_count", "camera_total_count",
        "time",
        "matched_camera_time",
        "time_difference_s",
        "time_sync_ok",
        "camera_original",
        "camera_increment",
        "target2_original",
        "X",
        "Y",
        "Z",
        "X_clean",
        "Y_clean",
        "Z_clean",
        "dX_mm",
        "dY_mm",
        "dZ_mm",
        "dX_mm_clean",
        "dY_mm_clean",
        "dZ_mm_clean",
        "gnss_camera_x",
        "gnss_camera_y",
        "gnss_camera_z",
        "gnss_original",
        "gnss_clean",
        "gnss_mean",
        "gnss_median",
        "gnss_smooth",
        "gnss_long_median",
        "gnss_long_window_samples",
        "gnss_ratio",
        "ratio",
        "Q",
        "ns",
        "gnss_ratio_bad",
        "gnss_q_bad",
        "gnss_ns_bad",
        "gnss_quality_bad",
        "gnss_jump_suspect",
        "gnss_return_spike",
        "gnss_mad_bad",
        "gnss_bad",
        "gnss_ratio_ok",
        "gnss_solution_ok",
        "gnss_jump_ok",
        "gnss_mad_ok",
        "gnss_local_volatility",
        "gnss_volatility_ok",
        "delta_camera_window",
        "delta_gnss_window",
        "camera_gnss_increment_error",
        "gnss_increment_consistency_ok",
        "gnss_reliable",
        "consistency_window_samples",
        "volatility_window_samples",
        "camera_bad",
        "prediction",
        "fusion",
        "gnss_correction_used",
        "common_error",
        "target2_compensated",
        "camera_minus_gnss",
        "fusion_minus_gnss_long",
    ]
    existing_columns = [column for column in output_columns if column in data.columns]

    parameter_rows = [
        ["PAIRING_METHOD", "GNSS零基索引=i*M//N；按数量逐行配对"],
        ["GNSS_TOTAL_COUNT", int(data["gnss_total_count"].iloc[0])],
        ["CAMERA_TOTAL_COUNT", len(data)],
        ["RATIO_THRESHOLD", RATIO_THRESHOLD],
        ["MAD_WINDOW", MAD_WINDOW],
        ["MAD_THRESHOLD", MAD_THRESHOLD],
        ["GNSS_SMOOTH_WINDOW", GNSS_SMOOTH_WINDOW],
        ["GNSS_SMOOTH_METHOD", GNSS_SMOOTH_METHOD],
        ["GNSS_LONG_MEDIAN_WINDOW_SECONDS", GNSS_LONG_MEDIAN_WINDOW_SECONDS],
        ["CONSISTENCY_WINDOW_SECONDS", CONSISTENCY_WINDOW_SECONDS],
        [
            "CAMERA_GNSS_INCREMENT_ERROR_THRESHOLD_MM",
            CAMERA_GNSS_INCREMENT_ERROR_THRESHOLD_MM,
        ],
        [
            "GNSS_LOCAL_VOLATILITY_WINDOW_SECONDS",
            GNSS_LOCAL_VOLATILITY_WINDOW_SECONDS,
        ],
        [
            "GNSS_LOCAL_VOLATILITY_THRESHOLD_MM",
            GNSS_LOCAL_VOLATILITY_THRESHOLD_MM,
        ],
        ["GNSS_CORRECTION_GAIN_K", GNSS_CORRECTION_GAIN_K],
        ["GNSS_SETTLEMENT_SIGN", GNSS_SETTLEMENT_SIGN],
        ["STANDARD_DEVIATION_DDOF", STANDARD_DEVIATION_DDOF],
        ["GNSS_3D_JUMP_THRESHOLD_MM", GNSS_3D_JUMP_THRESHOLD_MM],
        ["CAMERA_JUMP_THRESHOLD_MM", CAMERA_JUMP_THRESHOLD_MM],
        ["MAX_TIME_DIFFERENCE_SECONDS", MAX_TIME_DIFFERENCE_SECONDS],
        ["REFERENCE_X_M", reference_xyz_m[0]],
        ["REFERENCE_Y_M", reference_xyz_m[1]],
        ["REFERENCE_Z_M", reference_xyz_m[2]],
    ]
    parameters = pd.DataFrame(parameter_rows, columns=["参数名称", "参数值"])

    statistics = calculate_statistics(data)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
            data[existing_columns].rename(columns={"time":"时间","matched_camera_time":"相机时间(缺省采用GNSS时间)","time_difference_s":"时间差(秒)","time_sync_ok":"数量配对完成","camera_original":"目标一原始位移(mm)","camera_increment":"目标一位移增量(mm)","target2_original":"目标二原始位移(mm)","X":"GNSS原始X坐标(m)","Y":"GNSS原始Y坐标(m)","Z":"GNSS原始Z坐标(m)","X_clean":"GNSS清洗后X坐标(m)","Y_clean":"GNSS清洗后Y坐标(m)","Z_clean":"GNSS清洗后Z坐标(m)","dX_mm":"GNSS X方向位移(mm)","dY_mm":"GNSS Y方向位移(mm)","dZ_mm":"GNSS Z方向位移(mm)","dX_mm_clean":"GNSS清洗后X位移(mm)","dY_mm_clean":"GNSS清洗后Y位移(mm)","dZ_mm_clean":"GNSS清洗后Z位移(mm)","gnss_camera_x":"GNSS相机X方向位移(mm)","gnss_camera_y":"GNSS相机Y方向位移(mm)","gnss_camera_z":"GNSS相机Z方向位移(mm)","gnss_original":"GNSS原始投影位移(mm)","gnss_clean":"GNSS清洗后位移(mm)","gnss_mean":"GNSS滑动均值(mm)","gnss_median":"GNSS滑动中位数(mm)","gnss_smooth":"GNSS平滑位移(mm)","gnss_long_median":"GNSS长窗口中位数(mm)","gnss_long_window_samples":"GNSS长窗口样本数","gnss_ratio":"GNSS模糊度比值","ratio":"GNSS原始比值","Q":"GNSS解算质量代码","ns":"卫星数","gnss_ratio_bad":"GNSS比值异常","gnss_q_bad":"GNSS解算状态异常","gnss_ns_bad":"GNSS卫星数不足","gnss_quality_bad":"GNSS质量异常","gnss_jump_suspect":"GNSS疑似跳变","gnss_return_spike":"GNSS往返尖峰","gnss_mad_bad":"GNSS MAD异常","gnss_bad":"GNSS异常点","gnss_ratio_ok":"GNSS比值通过","gnss_solution_ok":"GNSS解算状态通过","gnss_jump_ok":"GNSS跳变通过","gnss_mad_ok":"GNSS MAD通过","gnss_local_volatility":"GNSS局部波动(mm)","gnss_volatility_ok":"GNSS局部波动通过","delta_camera_window":"窗口内相机增量(mm)","delta_gnss_window":"窗口内GNSS增量(mm)","camera_gnss_increment_error":"相机GNSS增量差(mm)","gnss_increment_consistency_ok":"增量一致性通过","gnss_reliable":"GNSS可靠","consistency_window_samples":"一致性窗口样本数","volatility_window_samples":"波动窗口样本数","camera_bad":"相机增量异常","prediction":"预测位移(mm)","fusion":"目标一融合位移(mm)","gnss_correction_used":"使用GNSS修正","common_error":"公共误差(mm)","target2_compensated":"目标二补偿后位移(mm)","camera_minus_gnss":"相机减GNSS位移(mm)","fusion_minus_gnss_long":"融合减GNSS基准(mm)"}).to_excel(writer, sheet_name="融合数据", index=False)
            parameters.to_excel(writer, sheet_name="参数配置", index=False)
            statistics.to_excel(writer, sheet_name="统计指标", index=False)
    except PermissionError as error:
        raise PermissionError(
            f"无法写入 {OUTPUT_FILE}，请先关闭正在打开的结果文件。"
        ) from error


def main():
    """完整 GNSS 检测和平滑后按相机条数抽取，再配对融合。"""
    validate_config()
    camera = load_camera_data(CAMERA_FILE)
    gnss = load_gnss_data(GNSS_FILE)
    gnss = apply_gnss_quality_screen(gnss)
    gnss = calculate_gnss_displacement(gnss)
    reference_xyz_m = gnss.attrs["reference_xyz_m"].copy()
    gnss = detect_gnss_outliers(gnss)
    gnss = transform_gnss_coordinate(gnss)
    gnss = smooth_gnss(gnss)
    gnss = evaluate_gnss_local_volatility(gnss)
    # 保留完整记录位置；异常点只做标记，不删除，以免改变抽取对应关系。
    gnss = select_gnss_by_camera_count(camera, gnss)
    print(f"相机{len(camera)}条，GNSS原始{gnss['gnss_total_count'].iloc[0]}条，检测平滑后抽取{len(gnss)}条")
    print(f"所选GNSS记录序号（前20条）: {gnss['gnss_source_row'].head(20).tolist()}")
    synchronized = synchronize_data(camera, gnss)
    synchronized = detect_camera_outliers(synchronized)
    synchronized = evaluate_gnss_reliability(synchronized)
    result = complementary_fusion(synchronized)
    save_results(result, reference_xyz_m)
    plot_results(result)

    print(f"Fusion result saved to: {OUTPUT_FILE}")
    print(f"Plots saved to: {PLOT_DIR}")
    print("Target 1 / GNSS displacement statistics:")
    print(calculate_statistics(result).iloc[:3].to_string(index=False))
    print(f"Rows: {len(result)}")
    print(f"GNSS bad: {int(result['gnss_bad'].sum())}")
    print(f"GNSS reliable: {int(result['gnss_reliable'].sum())}")
    print(f"Camera bad: {int(result['camera_bad'].sum())}")
    print(f"GNSS corrections used: {int(result['gnss_correction_used'].sum())}")
    print(
        "Target 2 camera standard deviation: "
        f"{result['target2_original'].std(ddof=STANDARD_DEVIATION_DDOF):.6f} mm"
    )
    print(
        "Clean GNSS standard deviation: "
        f"{result['gnss_clean'].std(ddof=STANDARD_DEVIATION_DDOF):.6f} mm"
    )
    print(
        "Target 2 compensated standard deviation: "
        f"{result['target2_compensated'].std(ddof=STANDARD_DEVIATION_DDOF):.6f} mm"
    )


if __name__ == "__main__":
    main()
