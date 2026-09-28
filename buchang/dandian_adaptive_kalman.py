from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# 配置区：后续实验优先只修改这里
# ============================================================
BASE_DIR = Path(__file__).resolve().parent

CAMERA_FILE = BASE_DIR / "camera_data" / "mubiao1_targets_2880.xlsx"
GNSS_FILE = Path(r"C:\Users\Administrator\Desktop\ohlo3000.pos")
OUTPUT_FILE = BASE_DIR / "adaptive_kalman_fusion_result.xlsx"
PLOT_DIR = BASE_DIR / "adaptive_kalman_fusion_plots"

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

# 自适应 Kalman 与相机异常检测。
# 状态只有 1 个：公共位移 x。Camera 增量作为控制输入，GNSS 作为观测。
KALMAN_INITIAL_VARIANCE_MM2 = 1
KALMAN_CAMERA_PROCESS_NOISE_MM2 = 0.01
KALMAN_CAMERA_MOTION_NOISE_FACTOR = 0.0
KALMAN_CAMERA_BAD_PROCESS_NOISE_MM2 =1
KALMAN_GNSS_BASE_R_MM2 = 4.0
KALMAN_GNSS_R_MIN_MM2 = 1
KALMAN_GNSS_R_MAX_MM2 = 100.0
KALMAN_GNSS_MEDIAN_WINDOW = 5
KALMAN_GNSS_NOISE_WINDOW = 7
KALMAN_GNSS_MIN_VALID = 3
CAMERA_JUMP_THRESHOLD_MM = 20.0
INITIALIZATION_DURATION_SECONDS = 360.0
KALMAN_GAIN_MAX = 0.02

# 时间同步。
TIME_SYNC_METHOD = "nearest"  # 当前实现 nearest；默认以 GNSS 时间轴为准。
MAX_TIME_DIFFERENCE_SECONDS = 0.5

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

SAVE_PLOTS = True
SHOW_PLOTS = True


def validate_config():
    """检查配置参数和旋转矩阵是否合法。"""
    if MAD_WINDOW < 3 or MAD_WINDOW % 2 == 0:
        raise ValueError("MAD_WINDOW 必须是大于等于 3 的奇数。")
    if GNSS_SMOOTH_WINDOW < 1:
        raise ValueError("GNSS_SMOOTH_WINDOW 必须大于 0。")
    if GNSS_LONG_MEDIAN_WINDOW_SECONDS <= 0:
        raise ValueError("GNSS_LONG_MEDIAN_WINDOW_SECONDS 必须大于 0。")
    if KALMAN_INITIAL_VARIANCE_MM2 <= 0:
        raise ValueError("KALMAN_INITIAL_VARIANCE_MM2 必须大于 0。")
    if KALMAN_CAMERA_PROCESS_NOISE_MM2 < 0:
        raise ValueError("KALMAN_CAMERA_PROCESS_NOISE_MM2 不能小于 0。")
    if KALMAN_GNSS_BASE_R_MM2 <= 0:
        raise ValueError("KALMAN_GNSS_BASE_R_MM2 必须大于 0。")
    if not 1 <= KALMAN_GNSS_MEDIAN_WINDOW:
        raise ValueError("KALMAN_GNSS_MEDIAN_WINDOW 必须大于 0。")
    if CONSISTENCY_WINDOW_SECONDS <= 0:
        raise ValueError("CONSISTENCY_WINDOW_SECONDS 必须大于 0。")
    if ROTATION_MATRIX.shape != (3, 3):
        raise ValueError("ROTATION_MATRIX 必须是 3x3 矩阵。")
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
    result["gnss_original"] = result["gnss_camera_y"]
    result["gnss_clean"] = result["gnss_original"].mask(result["gnss_bad"])
    for column in ["X", "Y", "Z", "dX_mm", "dY_mm", "dZ_mm"]:
        result[f"{column}_clean"] = result[column].mask(result["gnss_bad"])
    return result


def seconds_to_samples(time, duration_seconds):
    """根据实际时间间隔将秒数转换为至少 1 个采样点。"""
    interval = time.diff().dt.total_seconds().dropna()
    interval = interval[interval > 0]
    if interval.empty:
        raise ValueError("无法从 GNSS 时间列推断采样间隔。")
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
        min_periods=min(GNSS_LONG_MEDIAN_MIN_VALID, long_window),
    ).median()
    result["gnss_long_window_samples"] = long_window

    # Kalman 使用较短的鲁棒 GNSS 观测，避免 30 分钟 long median 带来的明显滞后。
    short_min = min(KALMAN_GNSS_MIN_VALID, KALMAN_GNSS_MEDIAN_WINDOW)
    result["gnss_kalman_measurement"] = result["gnss_clean"].rolling(
        window=KALMAN_GNSS_MEDIAN_WINDOW,
        min_periods=short_min,
    ).median()

    # 用 GNSS 相对短中值的局部残差估计当前观测噪声。
    measurement_residual = (
        result["gnss_clean"] - result["gnss_kalman_measurement"]
    )
    noise_min = min(KALMAN_GNSS_MIN_VALID, KALMAN_GNSS_NOISE_WINDOW)
    result["gnss_kalman_noise_std"] = measurement_residual.rolling(
        window=KALMAN_GNSS_NOISE_WINDOW,
        min_periods=noise_min,
    ).std()
    return result


def synchronize_data(camera, gnss):
    """以 GNSS 时间为基准，用最近时间匹配相机数据并限制最大时间误差。"""
    if TIME_SYNC_METHOD != "nearest":
        raise ValueError("第一版 TIME_SYNC_METHOD 仅支持 'nearest'。")

    camera = camera.copy()
    if camera["camera_time"].isna().all():
        camera["camera_time"] = gnss["time"].iloc[0] + pd.to_timedelta(
            np.arange(len(camera)) * CAMERA_SAMPLE_INTERVAL_SECONDS,
            unit="s",
        )

    camera = camera.sort_values("camera_time").reset_index(drop=True)
    gnss = gnss.sort_values("time").reset_index(drop=True)

    camera_for_merge = camera.rename(columns={"camera_time": "matched_camera_time"})
    synchronized = pd.merge_asof(
        gnss,
        camera_for_merge,
        left_on="time",
        right_on="matched_camera_time",
        direction="nearest",
        tolerance=pd.Timedelta(seconds=MAX_TIME_DIFFERENCE_SECONDS),
    )
    synchronized["time_difference_s"] = (
        synchronized["time"] - synchronized["matched_camera_time"]
    ).abs().dt.total_seconds()
    synchronized["time_sync_ok"] = synchronized["camera_original"].notna()
    return synchronized


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


def evaluate_gnss_reliability(data):
    """联合质量、突变、MAD、局部波动和增量一致性建立可靠性门控。"""
    result = data.copy()
    consistency_window = seconds_to_samples(
        result["time"],
        CONSISTENCY_WINDOW_SECONDS,
    )
    volatility_window = seconds_to_samples(
        result["time"],
        GNSS_LOCAL_VOLATILITY_WINDOW_SECONDS,
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

    gnss_increment = result["gnss_original"].diff()
    result["gnss_local_volatility"] = gnss_increment.rolling(
        window=volatility_window,
        min_periods=max(3, volatility_window // 2),
    ).std()

    result["gnss_ratio_ok"] = ~result["gnss_ratio_bad"]
    result["gnss_solution_ok"] = ~(
        result["gnss_q_bad"] | result["gnss_ns_bad"]
    )
    result["gnss_jump_ok"] = ~result["gnss_jump_suspect"]
    result["gnss_mad_ok"] = ~result["gnss_mad_bad"]
    result["gnss_volatility_ok"] = result["gnss_local_volatility"].le(
        GNSS_LOCAL_VOLATILITY_THRESHOLD_MM
    )
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
        & result["gnss_kalman_measurement"].notna()
    )
    result["consistency_window_samples"] = consistency_window
    result["volatility_window_samples"] = volatility_window
    return result


def adaptive_kalman_fusion(data):
    """Camera 增量预测 + 可靠 GNSS 自适应校正的一维 Kalman 融合。"""
    result = data.copy()
    count = len(result)

    prediction = np.full(count, np.nan, dtype=float)
    fusion = np.full(count, np.nan, dtype=float)
    prior_variance = np.full(count, np.nan, dtype=float)
    posterior_variance = np.full(count, np.nan, dtype=float)
    process_noise_q = np.full(count, np.nan, dtype=float)
    measurement_noise_r = np.full(count, np.nan, dtype=float)
    kalman_gain = np.zeros(count, dtype=float)
    innovation = np.full(count, np.nan, dtype=float)
    gnss_correction_used = np.zeros(count, dtype=bool)

    if count == 0:
        for name, values in {
            "prediction": prediction,
            "fusion": fusion,
            "kalman_prior_variance": prior_variance,
            "kalman_posterior_variance": posterior_variance,
            "kalman_process_noise_q": process_noise_q,
            "kalman_measurement_noise_r": measurement_noise_r,
            "kalman_gain": kalman_gain,
            "kalman_innovation": innovation,
            "gnss_correction_used": gnss_correction_used,
        }.items():
            result[name] = values
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
        initial_value = float(initialization_values.mean())
    elif result["gnss_kalman_measurement"].notna().any():
        initial_value = float(result["gnss_kalman_measurement"].dropna().iloc[0])
    elif pd.notna(result["camera_original"].iloc[0]):
        initial_value = float(result["camera_original"].iloc[0])
    else:
        raise ValueError("GNSS 和相机均无法提供融合初始值。")

    prediction[0] = initial_value
    fusion[0] = initial_value
    prior_variance[0] = KALMAN_INITIAL_VARIANCE_MM2
    posterior_variance[0] = KALMAN_INITIAL_VARIANCE_MM2
    process_noise_q[0] = 0.0

    for index in range(1, count):
        # 1) Camera 增量作为状态预测输入。
        if result["camera_bad"].iloc[index]:
            camera_increment = 0.0
            q_k = KALMAN_CAMERA_BAD_PROCESS_NOISE_MM2
        else:
            camera_increment = float(result["camera_increment"].iloc[index])
            q_k = (
                KALMAN_CAMERA_PROCESS_NOISE_MM2
                + KALMAN_CAMERA_MOTION_NOISE_FACTOR * camera_increment**2
            )

        prediction[index] = fusion[index - 1] + camera_increment
        process_noise_q[index] = q_k
        prior_variance[index] = posterior_variance[index - 1] + q_k

        # 2) 只有门控通过的 GNSS 才允许作为观测更新。
        measurement = result["gnss_kalman_measurement"].iloc[index]
        if result["gnss_reliable"].iloc[index] and pd.notna(measurement):
            local_std = result["gnss_kalman_noise_std"].iloc[index]
            if pd.isna(local_std):
                local_variance = 0.0
            else:
                local_variance = float(local_std) ** 2

            # 基础 R + 当前局部 GNSS 噪声；ratio 越接近门限，R 适度增大。
            r_k = KALMAN_GNSS_BASE_R_MM2 + local_variance
            ratio = result["gnss_ratio"].iloc[index]
            if pd.notna(ratio) and ratio > 0:
                ratio_penalty = np.clip(RATIO_THRESHOLD / float(ratio), 0.0, 1.0)
                r_k *= 1.0 + ratio_penalty

            r_k = float(np.clip(
                r_k,
                KALMAN_GNSS_R_MIN_MM2,
                KALMAN_GNSS_R_MAX_MM2,
            ))
            measurement_noise_r[index] = r_k

            innovation[index] = float(measurement) - prediction[index]
            k_k = prior_variance[index] / (prior_variance[index] + r_k)
            k_k = min(k_k, KALMAN_GAIN_MAX)
            kalman_gain[index] = k_k

            fusion[index] = prediction[index] + k_k * innovation[index]
            posterior_variance[index] = (1.0 - k_k) * prior_variance[index]
            gnss_correction_used[index] = True
        else:
            # GNSS 不可靠：只接受 Camera 预测，并让不确定性继续累积。
            fusion[index] = prediction[index]
            posterior_variance[index] = prior_variance[index]

    result["prediction"] = prediction
    result["fusion"] = fusion
    result["kalman_prior_variance"] = prior_variance
    result["kalman_posterior_variance"] = posterior_variance
    result["kalman_process_noise_q"] = process_noise_q
    result["kalman_measurement_noise_r"] = measurement_noise_r
    result["kalman_gain"] = kalman_gain
    result["kalman_innovation"] = innovation
    result["gnss_correction_used"] = gnss_correction_used

    result["camera_minus_gnss"] = (
        result["camera_original"] - result["gnss_original"]
    )
    result["fusion_minus_gnss_long"] = (
        result["fusion"] - result["gnss_long_median"]
    )
    result["fusion_minus_kalman_measurement"] = (
        result["fusion"] - result["gnss_kalman_measurement"]
    )

    # 目标二仍按“公共融合位移”进行补偿，并从首历元归零。
    result["fusion_for_compensation"] = (
        result["fusion"] - result["fusion"].iloc[0]
    )
    result["target2_compensated"] = (
        result["target2_original"] - result["fusion_for_compensation"]
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
        label="GNSS unreliable" if label else None,
    )


def plot_results(data):
    """生成融合诊断图，以及目标二补偿前后的对比图。"""
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    figures = []

    fig1, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(data["time"], data["camera_original"], label="Camera original")
    ax1.plot(data["time"], data["gnss_original"], label="GNSS original", alpha=0.75)
    ax1.set(title="Raw displacement", xlabel="Time", ylabel="Displacement / mm")
    ax1.grid(True)
    ax1.legend()
    fig1.tight_layout()
    figures.append((fig1, "01_raw_comparison.png"))

    fig2, ax2 = plt.subplots(figsize=(12, 5))
    ax2.plot(data["time"], data["gnss_original"], label="GNSS original", alpha=0.45)
    ax2.plot(data["time"], data["gnss_mean"], label="Rolling mean")
    ax2.plot(data["time"], data["gnss_median"], label="Rolling median")
    ax2.plot(
        data["time"],
        data["gnss_long_median"],
        label="GNSS long median",
        linewidth=2,
    )
    ax2.plot(
        data["time"],
        data["gnss_kalman_measurement"],
        label="GNSS Kalman measurement",
        linewidth=1.5,
    )
    ax2.set(title="GNSS smoothing", xlabel="Time", ylabel="Displacement / mm")
    ax2.grid(True)
    ax2.legend()
    fig2.tight_layout()
    figures.append((fig2, "02_gnss_smoothing.png"))

    fig3, ax3 = plt.subplots(figsize=(12, 5))
    ax3.plot(data["time"], data["camera_original"], label="Camera original", alpha=0.65)
    ax3.plot(data["time"], data["gnss_original"], label="GNSS original", alpha=0.35)
    ax3.plot(data["time"], data["fusion"], label="Adaptive Kalman fusion", linewidth=2)
    shade_unreliable(ax3, data)
    ax3.set(title="Adaptive Kalman fusion", xlabel="Time", ylabel="Displacement / mm")
    ax3.grid(True)
    ax3.legend()
    fig3.tight_layout()
    figures.append((fig3, "03_fusion.png"))

    fig4, ax4 = plt.subplots(figsize=(12, 5))
    ax4.plot(
        data["time"],
        data["camera_original"] - data["gnss_long_median"],
        label="Camera - GNSS long median",
    )
    ax4.plot(
        data["time"],
        data["fusion_minus_gnss_long"],
        label="Fusion - GNSS long median",
    )
    ax4.axhline(0.0, color="black", linewidth=1)
    ax4.set(title="Residuals", xlabel="Time", ylabel="Residual / mm")
    ax4.grid(True)
    ax4.legend()
    fig4.tight_layout()
    figures.append((fig4, "04_residuals.png"))

    fig5, ax5 = plt.subplots(figsize=(12, 5))
    ax5.plot(data["time"], data["gnss_original"], label="GNSS original")
    bad = data["gnss_bad"]
    ax5.scatter(
        data.loc[bad, "time"],
        data.loc[bad, "gnss_original"],
        color="red",
        s=18,
        label="GNSS bad",
        zorder=3,
    )
    shade_unreliable(ax5, data)
    ax5.set(title="GNSS outlier flags", xlabel="Time", ylabel="Displacement / mm")
    ax5.grid(True)
    ax5.legend()
    fig5.tight_layout()
    figures.append((fig5, "05_gnss_outliers.png"))

    fig6, ax6 = plt.subplots(figsize=(12, 5))
    ax6.plot(
        data["time"],
        data["target2_original"],
        label="Target 2 original",
        alpha=0.8,
    )
    ax6.plot(
        data["time"],
        data["fusion_for_compensation"],
        label="Fusion displacement used for compensation",
        alpha=0.7,
    )
    ax6.plot(
        data["time"],
        data["target2_compensated"],
        label="Target 2 compensated",
        linewidth=2,
    )
    ax6.axhline(0.0, color="black", linewidth=1)
    ax6.set(
        title="Target 2 compensation using fused displacement",
        xlabel="Time",
        ylabel="Displacement / mm",
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


def save_results(data, reference_xyz_m):
    """将融合明细和关键参数保存为新的 Excel 文件。"""
    output_columns = [
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
        "gnss_kalman_measurement",
        "gnss_kalman_noise_std",
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
        "kalman_prior_variance",
        "kalman_posterior_variance",
        "kalman_process_noise_q",
        "kalman_measurement_noise_r",
        "kalman_gain",
        "kalman_innovation",
        "gnss_correction_used",
        "fusion_for_compensation",
        "target2_compensated",
        "camera_minus_gnss",
        "fusion_minus_gnss_long",
        "fusion_minus_kalman_measurement",
    ]
    existing_columns = [column for column in output_columns if column in data.columns]

    parameter_rows = [
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
        ["KALMAN_INITIAL_VARIANCE_MM2", KALMAN_INITIAL_VARIANCE_MM2],
        ["KALMAN_CAMERA_PROCESS_NOISE_MM2", KALMAN_CAMERA_PROCESS_NOISE_MM2],
        ["KALMAN_CAMERA_MOTION_NOISE_FACTOR", KALMAN_CAMERA_MOTION_NOISE_FACTOR],
        ["KALMAN_CAMERA_BAD_PROCESS_NOISE_MM2", KALMAN_CAMERA_BAD_PROCESS_NOISE_MM2],
        ["KALMAN_GNSS_BASE_R_MM2", KALMAN_GNSS_BASE_R_MM2],
        ["KALMAN_GNSS_R_MIN_MM2", KALMAN_GNSS_R_MIN_MM2],
        ["KALMAN_GNSS_R_MAX_MM2", KALMAN_GNSS_R_MAX_MM2],
        ["KALMAN_GNSS_MEDIAN_WINDOW", KALMAN_GNSS_MEDIAN_WINDOW],
        ["KALMAN_GNSS_NOISE_WINDOW", KALMAN_GNSS_NOISE_WINDOW],
        ["GNSS_3D_JUMP_THRESHOLD_MM", GNSS_3D_JUMP_THRESHOLD_MM],
        ["CAMERA_JUMP_THRESHOLD_MM", CAMERA_JUMP_THRESHOLD_MM],
        ["MAX_TIME_DIFFERENCE_SECONDS", MAX_TIME_DIFFERENCE_SECONDS],
        ["REFERENCE_X_M", reference_xyz_m[0]],
        ["REFERENCE_Y_M", reference_xyz_m[1]],
        ["REFERENCE_Z_M", reference_xyz_m[2]],
    ]
    parameters = pd.DataFrame(parameter_rows, columns=["parameter", "value"])

    try:
        with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
            data[existing_columns].to_excel(writer, sheet_name="fusion_data", index=False)
            parameters.to_excel(writer, sheet_name="parameters", index=False)
    except PermissionError as error:
        raise PermissionError(
            f"无法写入 {OUTPUT_FILE}，请先关闭正在打开的结果文件。"
        ) from error


def main():
    """执行 GNSS 清洗、时间同步、自适应 Kalman 融合、保存和绘图。"""
    validate_config()
    camera = load_camera_data(CAMERA_FILE)
    gnss = load_gnss_data(GNSS_FILE)
    gnss = apply_gnss_quality_screen(gnss)
    gnss = calculate_gnss_displacement(gnss)
    reference_xyz_m = gnss.attrs["reference_xyz_m"].copy()
    gnss = detect_gnss_outliers(gnss)
    gnss = transform_gnss_coordinate(gnss)
    gnss = smooth_gnss(gnss)
    synchronized = synchronize_data(camera, gnss)
    synchronized = detect_camera_outliers(synchronized)
    synchronized = evaluate_gnss_reliability(synchronized)
    result = adaptive_kalman_fusion(synchronized)
    save_results(result, reference_xyz_m)
    plot_results(result)

    print(f"Fusion result saved to: {OUTPUT_FILE}")
    print(f"Plots saved to: {PLOT_DIR}")
    print(f"Rows: {len(result)}")
    print(f"GNSS bad: {int(result['gnss_bad'].sum())}")
    print(f"GNSS reliable: {int(result['gnss_reliable'].sum())}")
    print(f"Camera bad: {int(result['camera_bad'].sum())}")
    print(f"GNSS corrections used: {int(result['gnss_correction_used'].sum())}")


if __name__ == "__main__":
    main()
