from pathlib import Path
import math
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from static_rmse import static_rmse

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"
]
plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# 配置区：后续实验优先只修改这里
# ============================================================
BASE_DIR = Path(__file__).resolve().parent

CAMERA_FILE = BASE_DIR / "camera_data" / "mubiao1.xlsx"
GNSS_FILE = BASE_DIR / "gnss_data" / "ohlo3000.pos"
OUTPUT_FILE = BASE_DIR / "adaptive_kalman_fusion_result.xlsx"
METRICS_FILE = BASE_DIR / "adaptive_kalman_fusion_metrics.csv"
PLOT_DIR = BASE_DIR / "kalman_count_sampled_plots"

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
    "sdx": None,
    "sdy": None,
    "sdz": None,
    "sdxy": None,
    "sdyz": None,
    "sdzx": None,
}

# GNSS 质量筛选。
# ratio 仅保留原始记录，不参与筛选或 R 的计算。
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
GNSS_SMOOTH_MIN_VALID = 20
GNSS_SMOOTH_METHOD = "mean"  # "mean" 或 "median"
GNSS_LONG_MEDIAN_WINDOW_SECONDS = 1800.0  # 默认 30 分钟。
GNSS_LONG_MEDIAN_MIN_VALID = 10
USE_GNSS_LONG_MEDIAN_FOR_FUSION = False  # True: 30分钟中位数；False: 原始GNSS投影

# GNSS 可靠性门控。
CONSISTENCY_WINDOW_SECONDS = 300.0  # 相机/GNSS 增量一致性窗口，默认 5 分钟。
CAMERA_GNSS_INCREMENT_ERROR_THRESHOLD_MM = 10.0
GNSS_LOCAL_VOLATILITY_WINDOW_SECONDS = 300.0
GNSS_LOCAL_VOLATILITY_THRESHOLD_MM = 0.5

# 自适应 Kalman 与相机异常检测。
# 状态只有 1 个：公共位移 x。Camera 增量作为控制输入，GNSS 作为观测。
KALMAN_INITIAL_VARIANCE_MM2 = 1
# 目标一 30 秒间隔插值位移的相邻增量样本方差，作为基础 Q 试验值。
KALMAN_CAMERA_PROCESS_NOISE_MM2 = 0.1
# 当前实验：固定 Q、固定 R；R 使用静止 GNSS 经验方差。
# main 中根据目标一相邻位移增量的样本方差重新计算，滤波期间固定。
ORIGINAL_EXPERIMENT_Q_MM2 = 0.2544397105832515
ORIGINAL_EXPERIMENT_R_MM2 = 1.97251
LOCAL_R_ALPHA = 2.0
LOCAL_R_MIN_FACTOR = 1.0
LOCAL_R_MAX_FACTOR = 10.0
KALMAN_CAMERA_MOTION_NOISE_FACTOR = 0.0
KALMAN_CAMERA_BAD_PROCESS_NOISE_MM2 =1
ADAPTIVE_Q_ENABLED = False
ADAPTIVE_Q_WINDOW_SECONDS = 1800.0
ADAPTIVE_Q_MIN_SAMPLES = 20
ADAPTIVE_Q_ALPHA = 0.05
ADAPTIVE_Q_MIN_MM2 = KALMAN_CAMERA_PROCESS_NOISE_MM2 * 0.1
ADAPTIVE_Q_MAX_MM2 = KALMAN_CAMERA_PROCESS_NOISE_MM2 * 100.0
# 仅限制用于学习 Q 的新息平方，避免单点尖峰主导窗口；不改变位移更新。
ADAPTIVE_Q_NIS_CAP = 9.0
# ECEF 解算协方差投影用于有效性门控；当前 R 使用固定经验方差。
# 以下 R 边界参数保留，当前固定 R 更新不使用它们。
KALMAN_GNSS_R_MIN_MM2 = 1e-6
KALMAN_GNSS_R_MAX_MM2 = None
KALMAN_GNSS_MEDIAN_WINDOW = 5
KALMAN_GNSS_NOISE_WINDOW = 7
KALMAN_GNSS_MIN_VALID = 3
CAMERA_JUMP_THRESHOLD_MM = 20.0
INITIALIZATION_DURATION_SECONDS = 360.0
KALMAN_GAIN_MAX = 1

# 时间同步。
TIME_SYNC_METHOD = "nearest"  # 相机无时间列时先按全天24小时均匀生成时间。
MAX_TIME_DIFFERENCE_SECONDS = 15.0  # GNSS每30秒采样，允许最近邻相差15秒

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
SHOW_PLOTS = False
ADAPTIVE_R_ENABLED = False
ADAPTIVE_R_WINDOW_SECONDS = 600.0
ADAPTIVE_R_MIN_SAMPLES = 12
ADAPTIVE_R_ALPHA = 0.1
ADAPTIVE_R_MIN_FACTOR = 0.1
ADAPTIVE_R_MAX_FACTOR = 25.0


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
    if not (
        0 < ADAPTIVE_Q_ALPHA <= 1
        and math.isfinite(ADAPTIVE_Q_WINDOW_SECONDS) and ADAPTIVE_Q_WINDOW_SECONDS > 0
        and isinstance(ADAPTIVE_Q_MIN_SAMPLES, int) and ADAPTIVE_Q_MIN_SAMPLES >= 2
        and 0 < ADAPTIVE_Q_MIN_MM2 <= KALMAN_CAMERA_PROCESS_NOISE_MM2 <= ADAPTIVE_Q_MAX_MM2
        and math.isfinite(ADAPTIVE_Q_MAX_MM2)
        and math.isfinite(ADAPTIVE_Q_NIS_CAP) and ADAPTIVE_Q_NIS_CAP > 0
    ):
        raise ValueError("自适应 Q 的窗口、样本数、更新系数或方差范围无效。")
    if not math.isfinite(KALMAN_GNSS_R_MIN_MM2) or KALMAN_GNSS_R_MIN_MM2 <= 0:
        raise ValueError("GNSS 观测方差下限必须为有限正数。")
    if KALMAN_GNSS_R_MAX_MM2 is not None and (
        not math.isfinite(KALMAN_GNSS_R_MAX_MM2)
        or KALMAN_GNSS_R_MAX_MM2 < KALMAN_GNSS_R_MIN_MM2
    ):
        raise ValueError("GNSS 观测方差上限必须不小于下限，或设为 None。")
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
    # 与 POS 一致：必须是 ECEF XYZ 标准差及带符号协方差平方根，单位为米。
    for field in ("sdx", "sdy", "sdz", "sdxy", "sdyz", "sdzx"):
        column = _find_column(
            columns, GNSS_COLUMN_MAP[field], [field, f"{field}(m)"], required=True
        )
        result[field] = pd.to_numeric(data[column], errors="coerce")
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
    """根据解算状态和卫星数标记低质量点；ratio 仅记录，不参与判断。"""
    q_bad = pd.Series(False, index=gnss.index)
    ns_bad = pd.Series(False, index=gnss.index)

    if VALID_Q_VALUES is not None:
        if "Q" not in gnss:
            raise ValueError("启用质量筛选时，GNSS必须包含Q字段。")
        q_bad = ~gnss["Q"].isin(VALID_Q_VALUES)
    if MIN_SATELLITES is not None and "ns" in gnss and gnss["ns"].notna().any():
        ns_bad = gnss["ns"].notna() & (gnss["ns"] < MIN_SATELLITES)

    result = gnss.copy()
    result["gnss_q_bad"] = q_bad
    result["gnss_ns_bad"] = ns_bad
    result["gnss_quality_bad"] = q_bad | ns_bad
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
        good_xyz = gnss[["X", "Y", "Z"]]  # 不按质量筛选
        reference_samples = good_xyz.head(REFERENCE_SAMPLE_COUNT)
        if len(reference_samples) < 3:
            raise ValueError("GNSS 数据不足，无法计算参考坐标。")
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


def detect_gnss_mad(gnss):
    """在完整GNSS三轴位移上计算居中MAD，任一轴异常则拒绝该点校正。"""
    result = gnss.copy()
    flags = []
    for axis in ("dX_mm", "dY_mm", "dZ_mm"):
        bad, score = rolling_mad_filter(result[axis], MAD_WINDOW, MAD_THRESHOLD)
        result[f"{axis}_mad_z"] = score
        flags.append(bad)
    result["gnss_mad_bad"] = flags[0] | flags[1] | flags[2]
    return result


def detect_gnss_jumps(gnss):
    """仅检查完整GNSS相邻历元的ECEF位移变化，保留原记录位置。"""
    result = gnss.copy()
    step = result[["dX_mm", "dY_mm", "dZ_mm"]].diff()
    distance = np.sqrt((step ** 2).sum(axis=1))
    result["gnss_step_dx_mm"] = step["dX_mm"]
    result["gnss_step_dy_mm"] = step["dY_mm"]
    result["gnss_step_dz_mm"] = step["dZ_mm"]
    result["gnss_step_3d_mm"] = distance
    result["gnss_jump_suspect"] = (
        step["dX_mm"].abs().gt(GNSS_HORIZONTAL_AXIS_JUMP_THRESHOLD_MM)
        | step["dY_mm"].abs().gt(GNSS_HORIZONTAL_AXIS_JUMP_THRESHOLD_MM)
        | step["dZ_mm"].abs().gt(GNSS_VERTICAL_AXIS_JUMP_THRESHOLD_MM)
        | distance.gt(GNSS_3D_JUMP_THRESHOLD_MM)
    )
    return result


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


def project_gnss_solution_variance(sdx, sdy, sdz, sdxy, sdyz, sdzx, direction):
    """恢复 RTKLIB ECEF 协方差并投影，返回 mm²；无效输入返回 NaN。

    RTKLIB 交叉项是带符号的协方差平方根：cov = sd * abs(sd)。
    参考坐标在本滤波模型中作为固定零点；这里不估计其共同偏移不确定度。
    """
    fields = (sdx, sdy, sdz, sdxy, sdyz, sdzx)
    if len(direction) != 3 or not all(
        math.isfinite(float(value)) for value in (*fields, *direction)
    ):
        return math.nan
    if min(sdx, sdy, sdz) < 0:
        return math.nan
    diagonal = (sdx ** 2, sdy ** 2, sdz ** 2)
    cross = (sdxy * abs(sdxy), sdyz * abs(sdyz), sdzx * abs(sdzx))
    scale = max(diagonal)
    if scale <= 0.0 or not all(math.isfinite(v) for v in (*diagonal, *cross)):
        return math.nan

    # 用全部主子式检查对称 3x3 协方差是否半正定；仅容许浮点舍入误差。
    d0, d1, d2 = (v / scale for v in diagonal)
    xy, yz, zx = (v / scale for v in cross)
    tolerance = 1e-10
    if min(d0 * d1 - xy ** 2, d1 * d2 - yz ** 2, d2 * d0 - zx ** 2) < -tolerance:
        return math.nan
    determinant = (
        d0 * d1 * d2 + 2 * xy * yz * zx
        - d0 * yz ** 2 - d1 * zx ** 2 - d2 * xy ** 2
    )
    if determinant < -tolerance:
        return math.nan

    a0, a1, a2 = direction
    variance_mm2 = 1e6 * math.fsum((
        a0 * a0 * diagonal[0], a1 * a1 * diagonal[1], a2 * a2 * diagonal[2],
        2 * a0 * a1 * cross[0], 2 * a1 * a2 * cross[1], 2 * a2 * a0 * cross[2],
    ))
    return variance_mm2 if math.isfinite(variance_mm2) and variance_mm2 > 0 else math.nan


def transform_gnss_coordinate(gnss):
    """将 GNSS 三轴毫米位移转换到相机坐标系，并保留原始与清洗结果。"""
    displacement = gnss[["dX_mm", "dY_mm", "dZ_mm"]].to_numpy(dtype=float)
    camera_xyz = displacement @ ROTATION_MATRIX.T

    result = gnss.copy()
    result[["gnss_camera_x", "gnss_camera_y", "gnss_camera_z"]] = camera_xyz
    result["gnss_original"] = result["gnss_camera_y"]
    result["gnss_clean"] = result["gnss_original"].mask(result["gnss_bad"])
    covariance_fields = ["sdx", "sdy", "sdz", "sdxy", "sdyz", "sdzx"]
    missing = [column for column in covariance_fields if column not in result]
    if missing:
        raise ValueError(f"GNSS 缺少解算协方差字段：{missing}")
    direction = ROTATION_MATRIX[1]
    result["gnss_solution_variance_mm2"] = [
        project_gnss_solution_variance(*row, direction)
        for row in result[covariance_fields].to_numpy(dtype=float)
    ]
    result["gnss_covariance_ok"] = (
        np.isfinite(result["gnss_solution_variance_mm2"])
        & result["gnss_solution_variance_mm2"].gt(0)
    )
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
    """生成GNSS统计曲线，并按配置选择卡尔曼观测。"""
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

    # 与单点互补滤波相同，长窗口中位数使用居中窗口，仅适合离线实验。
    # 质量门控仍依据原始GNSS，避免平滑后掩盖异常点。
    result["gnss_kalman_measurement"] = (
        result["gnss_long_median"] if USE_GNSS_LONG_MEDIAN_FOR_FUSION
        else result["gnss_original"]
    ).copy()

    # 短窗口中位数仅作为噪声估计基准，不再作为卡尔曼观测。
    short_min = min(KALMAN_GNSS_MIN_VALID, KALMAN_GNSS_MEDIAN_WINDOW)
    noise_reference = result["gnss_clean"].rolling(
        window=KALMAN_GNSS_MEDIAN_WINDOW,
        min_periods=short_min,
    ).median()

    # 用 GNSS 相对短中值的局部残差估计当前观测噪声。
    measurement_residual = (
        result["gnss_clean"] - noise_reference
    )
    noise_min = min(KALMAN_GNSS_MIN_VALID, KALMAN_GNSS_NOISE_WINDOW)
    result["gnss_kalman_noise_std"] = measurement_residual.rolling(
        window=KALMAN_GNSS_NOISE_WINDOW,
        min_periods=noise_min,
    ).std()
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
    """无相机时间时均匀分布于GNSS首日[00:00,次日00:00)，最近邻匹配。"""
    camera = camera.copy().reset_index(drop=True)
    if camera.empty:
        raise ValueError("相机数据为空。")
    if camera['camera_time'].isna().all():
        start = gnss['time'].iloc[0].normalize()
        camera['camera_time'] = start + pd.to_timedelta(np.arange(len(camera))*86400/len(camera),unit='s')
    camera['camera_source_row'] = np.arange(1,len(camera)+1)
    source = gnss.copy().reset_index(drop=True)
    source['gnss_source_row'] = np.arange(1,len(source)+1)
    source['gnss_total_count'] = len(source)
    source = source.rename(columns={'time':'matched_gnss_time'})
    camera['camera_time'] = camera['camera_time'].astype('datetime64[ns]')
    source['matched_gnss_time'] = source['matched_gnss_time'].astype('datetime64[ns]')
    result = pd.merge_asof(camera.sort_values('camera_time'),source.sort_values('matched_gnss_time'),
        left_on='camera_time',right_on='matched_gnss_time',direction='nearest',
        tolerance=pd.Timedelta(seconds=MAX_TIME_DIFFERENCE_SECONDS))
    result['time'] = result['camera_time']
    result['matched_camera_time'] = result['camera_time']
    result['time_difference_s'] = (result['camera_time']-result['matched_gnss_time']).abs().dt.total_seconds()
    result['time_sync_ok'] = result['matched_gnss_time'].notna()
    result['gnss_quality_bad'] = result['gnss_quality_bad'].fillna(True).astype(bool)
    result['camera_total_count'] = len(camera)
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


def check_increment_consistency(data):
    """配对后比较同一窗口内的相机和原始GNSS投影增量；缺失增量不通过。"""
    result = data.copy()
    window = seconds_to_samples(result['time'], CONSISTENCY_WINDOW_SECONDS)
    result['delta_camera_window'] = result['camera_original'].diff(window)
    result['delta_gnss_window'] = result['gnss_original'].diff(window)
    result['camera_gnss_increment_error'] = (
        result['delta_camera_window'] - result['delta_gnss_window']
    ).abs()
    result['gnss_increment_consistency_ok'] = result['camera_gnss_increment_error'].le(
        CAMERA_GNSS_INCREMENT_ERROR_THRESHOLD_MM
    )
    result['consistency_window_samples'] = window
    result['consistency_actual_duration_s'] = result['time'].diff(window).dt.total_seconds()
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
        & result["gnss_solution_ok"]
        & result["gnss_covariance_ok"]
        & result["gnss_jump_ok"]
        & result["gnss_mad_ok"]
        & result["gnss_volatility_ok"]
        & result["gnss_increment_consistency_ok"]
        & result["gnss_kalman_measurement"].notna()
    )
    result["consistency_window_samples"] = consistency_window
    return result


class EffectiveQAdapter:
    """用过去窗口的新息进行有界协方差匹配；输出仅用于下一步预测。

    得到的是含模型误差的有效 Q，插值相关性及持续偏差使它不能解释为
    相机独立白噪声的精确方差。GNSS 不可靠时只清理过期记录，不学习。
    """

    def __init__(self, initial_q, window_seconds, min_samples, alpha, q_min, q_max, nis_cap):
        self.q = initial_q
        self.window_seconds = window_seconds
        self.min_samples = min_samples
        self.alpha = alpha
        self.q_min = q_min
        self.q_max = q_max
        self.nis_cap = nis_cap
        self.samples = deque()

    def observe(self, time_seconds, innovation, previous_p, r, predicted_p,
                eligible, extra_q=0.0):
        while self.samples and self.samples[0][0] < time_seconds - self.window_seconds:
            self.samples.popleft()
        candidate = math.nan
        updated = False
        clipped = False
        accepted = eligible and all(math.isfinite(v) for v in (
            innovation, previous_p, r, predicted_p, extra_q
        )) and r > 0 and previous_p >= 0 and predicted_p >= 0
        if accepted:
            squared = innovation ** 2
            limit = self.nis_cap * (predicted_p + r)
            clipped = squared > limit
            # 减去上一时刻后验方差，不能减已经含 Q 的本步预测方差。
            term = min(squared, limit) - previous_p - r - extra_q
            self.samples.append((time_seconds, term, innovation))
            if len(self.samples) >= self.min_samples:
                estimate = math.fsum(row[1] for row in self.samples) / len(self.samples)
                candidate = max(self.q_min, min(self.q_max, estimate))
                self.q = max(self.q_min, min(self.q_max,
                    (1 - self.alpha) * self.q + self.alpha * candidate))
                updated = True
        mean = (math.fsum(row[2] for row in self.samples) / len(self.samples)
                if self.samples else math.nan)
        return self.q, candidate, len(self.samples), updated, clipped, mean, bool(accepted)


def adaptive_kalman_fusion(data, adaptive_r=False):
    """固定Q，按局部增量方差调整R；保留质量、跳变、MAD和增量一致性门控。"""
    result = data.copy()
    if result.empty:
        raise ValueError("没有可融合的数据。")
    local_var = pd.to_numeric(result['gnss_local_volatility'], errors='coerce') ** 2
    # 窗口尚无有效方差时使用基础R，避免由NaN污染滤波状态。
    r_values = np.clip(
        ORIGINAL_EXPERIMENT_R_MM2 + LOCAL_R_ALPHA * local_var.fillna(0.0).to_numpy(),
        ORIGINAL_EXPERIMENT_R_MM2 * LOCAL_R_MIN_FACTOR,
        ORIGINAL_EXPERIMENT_R_MM2 * LOCAL_R_MAX_FACTOR,
    ) if adaptive_r else np.full(len(result), ORIGINAL_EXPERIMENT_R_MM2)
    result['adaptive_r_factor'] = r_values / ORIGINAL_EXPERIMENT_R_MM2
    result['local_r_variance_available'] = np.isfinite(local_var)

    initial_end = result['time'].iloc[0] + pd.Timedelta(seconds=INITIALIZATION_DURATION_SECONDS)
    eligible = (
        result['time_sync_ok'] & ~result['gnss_quality_bad']
        & ~result['gnss_jump_suspect'].fillna(True).astype(bool)
        & ~result['gnss_mad_bad'].fillna(True).astype(bool)
        & result['gnss_increment_consistency_ok'].fillna(False).astype(bool)
    )
    initial = result.loc[(result['time'] <= initial_end) & eligible, 'gnss_original']
    # 初始窗口无合格GNSS时，以首个相机位移初始化，避免使用不合格观测。
    x = float(initial.mean()) if not initial.empty else float(result['camera_original'].iloc[0])
    variance = KALMAN_INITIAL_VARIANCE_MM2
    result['camera_increment'] = result['camera_original'].diff().fillna(0.0)
    rows = [(x,x,variance,variance,0.0,r_values[0],0.0,np.nan,False)]
    for i in range(1,len(result)):
        prediction = x + float(result['camera_increment'].iloc[i])
        prior = variance + ORIGINAL_EXPERIMENT_Q_MM2
        r_k = float(r_values[i])
        used = bool(eligible.iloc[i])
        if used:
            innovation = float(result['gnss_original'].iloc[i]) - prediction
            gain = prior / (prior + r_k)
            x = prediction + gain * innovation
            variance = (1-gain)**2 * prior + gain**2 * r_k
        else:
            innovation, gain = np.nan, 0.0
            x, variance = prediction, prior
        rows.append((prediction,x,prior,variance,ORIGINAL_EXPERIMENT_Q_MM2,r_k,gain,innovation,used))
    columns = ['prediction','fusion','kalman_prior_variance','kalman_posterior_variance',
               'kalman_process_noise_q','kalman_measurement_noise_r','kalman_gain',
               'kalman_innovation','gnss_correction_used']
    for j,column in enumerate(columns):
        result[column] = [row[j] for row in rows]
    result['gnss_kalman_measurement'] = result['gnss_original']
    result['fusion_for_compensation'] = result['fusion']-result['fusion'].iloc[0]
    result['common_error_for_compensation'] = (result['camera_original']-result['camera_original'].iloc[0])-result['fusion_for_compensation']
    result['target2_compensated'] = result['target2_original']-result['common_error_for_compensation']
    result['camera_minus_gnss'] = result['camera_original']-result['gnss_original']
    result['fusion_minus_kalman_measurement'] = result['fusion']-result['gnss_original']
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
        label="GNSS 不可靠时段" if label else None,
    )


def plot_results(data):
    """仅绘制融合和目标二补偿结果。"""
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    for filename, title, curves in [
        ('03_fusion.png','目标一原始GNSS与卡尔曼融合', [('camera_original','相机目标一'),('gnss_original','原始GNSS投影'),('fusion_for_compensation','目标一融合后（首点归零）')]),
        ('06_target2_compensation.png','目标二补偿前后', [('target2_original','目标二原始'),('target2_compensated','目标二补偿后')]),
    ]:
        fig, ax = plt.subplots(figsize=(12,5))
        for column,label in curves:
            ax.plot(data['time'],data[column],label=label)
        ax.set(title=title,xlabel='时间',ylabel='位移 / mm')
        ax.grid(True); ax.legend(); fig.tight_layout()
        if SAVE_PLOTS:
            fig.savefig(PLOT_DIR / filename,dpi=160)
        if SHOW_PLOTS:
            plt.show()
        plt.close(fig)

def calculate_statistics(data):
    """统计完整输出序列；融合及补偿行的有效数量记录GNSS校正次数。"""
    correction_count = int(data["gnss_correction_used"].sum())
    rows = []
    for name, column in (
        ("目标一融合前", "camera_original"),
        ("目标一融合后", "fusion_for_compensation"),
        ("GNSS原始投影", "gnss_original"),
        ("目标二补偿后", "target2_compensated"),
    ):
        values = pd.to_numeric(data[column], errors="coerce")
        y = values[np.isfinite(values)]
        rows.append({
            "数据类型": name,
            "RMSE(mm)": static_rmse(data, column, ROTATION_MATRIX[1]),
            "标准差(mm)": y.std(ddof=1),
            "有效数量": (correction_count if column in (
                "fusion_for_compensation", "target2_compensated"
            ) else len(y)),
            "均值(mm)": y.mean(),
            "最小值(mm)": y.min(),
            "最大值(mm)": y.max(),
            "极差(mm)": y.max() - y.min(),
        })
    return pd.DataFrame(rows).round(2)


def save_results(data, reference_xyz_m, metrics=None, adaptive_r=False):
    """将融合明细和关键参数保存为新的 Excel 文件。"""
    output_columns = [
        "matched_gnss_time", "gnss_source_row", "camera_source_row", "gnss_total_count", "camera_total_count",
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
        "sdx",
        "sdy",
        "sdz",
        "sdxy",
        "sdyz",
        "sdzx",
        "gnss_solution_variance_mm2",
        "gnss_covariance_ok",
        "gnss_ratio",
        "ratio",
        "Q",
        "ns",
        "gnss_q_bad",
        "gnss_ns_bad",
        "gnss_quality_bad",
        "gnss_jump_suspect",
        "gnss_return_spike",
        "gnss_mad_bad",
        "gnss_bad",
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
        "adaptive_q_next",
        "adaptive_q_candidate",
        "adaptive_q_samples",
        "adaptive_q_updated",
        "adaptive_q_innovation_clipped",
        "adaptive_q_innovation_mean",
        "adaptive_q_sample_used",
        "kalman_measurement_noise_r",
        "adaptive_r_factor",
        "adaptive_r_candidate",
        "adaptive_r_samples",
        "kalman_gain",
        "kalman_innovation",
        "gnss_correction_used",
        "fusion_for_compensation",
        "common_error_for_compensation",
        "target2_compensated",
        "camera_minus_gnss",
        "fusion_minus_gnss_long",
        "fusion_minus_kalman_measurement",
    ]
    existing_columns = [column for column in output_columns if column in data.columns]
    # 只在导出副本中翻译名称和值，内部计算仍使用原字段。
    column_names = {
        "gnss_source_row": "GNSS排序去重后记录序号",
        "camera_source_row": "相机记录序号",
        "gnss_total_count": "GNSS抽取前总数",
        "camera_total_count": "相机总数",
        "time": "时间",
        "matched_camera_time": "相机时间（缺省采用GNSS时间）",
        "time_difference_s": "时间差（秒）",
        "time_sync_ok": "时间匹配是否完成",
        "camera_original": "相机原始位移（毫米）",
        "camera_increment": "相机位移增量（毫米）",
        "target2_original": "目标二原始位移（毫米）",
        "X": "GNSS原始X坐标（米）",
        "Y": "GNSS原始Y坐标（米）",
        "Z": "GNSS原始Z坐标（米）",
        "X_clean": "GNSS清洗后X坐标（米）",
        "Y_clean": "GNSS清洗后Y坐标（米）",
        "Z_clean": "GNSS清洗后Z坐标（米）",
        "dX_mm": "GNSS的X方向位移（毫米）",
        "dY_mm": "GNSS的Y方向位移（毫米）",
        "dZ_mm": "GNSS的Z方向位移（毫米）",
        "dX_mm_clean": "GNSS清洗后X方向位移（毫米）",
        "dY_mm_clean": "GNSS清洗后Y方向位移（毫米）",
        "dZ_mm_clean": "GNSS清洗后Z方向位移（毫米）",
        "gnss_camera_x": "GNSS在相机X方向的位移（毫米）",
        "gnss_camera_y": "GNSS在相机Y方向的位移（毫米）",
        "gnss_camera_z": "GNSS在相机Z方向的位移（毫米）",
        "gnss_original": "GNSS原始投影位移（毫米）",
        "gnss_clean": "GNSS清洗后投影位移（毫米）",
        "gnss_mean": "GNSS滑动均值（毫米）",
        "gnss_median": "GNSS滑动中位数（毫米）",
        "gnss_smooth": "GNSS平滑位移（毫米）",
        "gnss_long_median": "GNSS长窗口中位数（毫米）",
        "gnss_long_window_samples": "GNSS长窗口采样点数",
        "gnss_kalman_measurement": "GNSS卡尔曼观测位移（毫米）",
        "gnss_kalman_noise_std": "GNSS局部噪声标准差（毫米，仅诊断）",
        "sdx": "GNSS解算X标准差（米）",
        "sdy": "GNSS解算Y标准差（米）",
        "sdz": "GNSS解算Z标准差（米）",
        "sdxy": "GNSS的XY协方差带符号平方根（米）",
        "sdyz": "GNSS的YZ协方差带符号平方根（米）",
        "sdzx": "GNSS的ZX协方差带符号平方根（米）",
        "gnss_solution_variance_mm2": "GNSS解算投影方差（平方毫米）",
        "gnss_covariance_ok": "GNSS解算协方差是否有效",
        "gnss_ratio": "GNSS模糊度比值（仅记录）",
        "ratio": "GNSS原始模糊度比值",
        "Q": "GNSS解算质量代码",
        "ns": "卫星数量",
        "gnss_q_bad": "GNSS解算状态是否异常",
        "gnss_ns_bad": "GNSS卫星数量是否不足",
        "gnss_quality_bad": "GNSS质量是否异常",
        "gnss_jump_suspect": "GNSS是否疑似跳变",
        "gnss_return_spike": "GNSS是否为跳出返回尖峰",
        "gnss_mad_bad": "GNSS中位数绝对偏差检测是否异常",
        "gnss_bad": "GNSS是否为异常点",
        "gnss_solution_ok": "GNSS解算质量检查是否通过",
        "gnss_jump_ok": "GNSS跳变检查是否通过",
        "gnss_mad_ok": "GNSS中位数绝对偏差检查是否通过",
        "gnss_local_volatility": "GNSS局部增量波动标准差（毫米）",
        "gnss_volatility_ok": "GNSS局部波动检查是否通过",
        "delta_camera_window": "窗口内相机位移增量（毫米）",
        "delta_gnss_window": "窗口内GNSS位移增量（毫米）",
        "camera_gnss_increment_error": "相机与GNSS增量差绝对值（毫米）",
        "gnss_increment_consistency_ok": "相机与GNSS增量一致性是否通过",
        "gnss_reliable": "GNSS是否可靠",
        "consistency_window_samples": "增量一致性窗口采样点数",
        "volatility_window_samples": "局部波动窗口采样点数",
        "camera_bad": "相机增量是否异常",
        "prediction": "预测位移（毫米）",
        "fusion": "融合位移（毫米）",
        "kalman_prior_variance": "卡尔曼预测方差（平方毫米）",
        "kalman_posterior_variance": "卡尔曼更新后方差（平方毫米）",
        "kalman_process_noise_q": "过程噪声方差Q（平方毫米）",
        "adaptive_q_next": "下一步正常预测基础Q（平方毫米）",
        "adaptive_q_candidate": "窗口候选Q（平方毫米）",
        "adaptive_q_samples": "自适应Q窗口有效样本数",
        "adaptive_q_updated": "本步是否更新自适应Q",
        "adaptive_q_innovation_clipped": "Q学习用新息平方是否限幅",
        "adaptive_q_innovation_mean": "Q窗口新息均值（毫米）",
        "adaptive_q_sample_used": "本步是否纳入Q学习",
        "kalman_measurement_noise_r": "观测噪声方差R（平方毫米）",
        "adaptive_r_factor": "自适应R放大因子",
        "adaptive_r_candidate": "自适应R候选因子",
        "adaptive_r_samples": "自适应R窗口样本数",
        "kalman_gain": "卡尔曼增益K",
        "kalman_innovation": "卡尔曼新息（毫米）",
        "gnss_correction_used": "是否使用GNSS校正",
        "fusion_for_compensation": "目标一融合位移（首历元归零，毫米）",
        "common_error_for_compensation": "补偿量：目标一相机位移减融合位移（毫米）",
        "target2_compensated": "目标二融合补偿后位移（毫米）",
        "camera_minus_gnss": "相机与GNSS原始位移差（毫米）",
        "fusion_minus_gnss_long": "融合与GNSS长窗口中位数之差（毫米）",
        "fusion_minus_kalman_measurement": "融合与GNSS卡尔曼观测之差（毫米）",
    }
    export_data = data[existing_columns].copy()
    for column in export_data.select_dtypes(include=["bool", "boolean"]).columns:
        export_data[column] = export_data[column].map({True: "是", False: "否"})
    export_data = export_data.rename(columns=column_names)

    parameter_rows = [
        ["LOCAL_R_ALPHA", LOCAL_R_ALPHA],
        ["LOCAL_R_MIN_FACTOR", LOCAL_R_MIN_FACTOR],
        ["LOCAL_R_MAX_FACTOR", LOCAL_R_MAX_FACTOR],
        ["PAIRING_METHOD", "相机按24小时均匀赋时，最近邻匹配容差15秒；仅Q=1且通过跳变、MAD及增量一致性检查参与校正；局部波动仅调整R，不直接拒绝"],
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
        ["KALMAN_GNSS_R_MIN_MM2", KALMAN_GNSS_R_MIN_MM2],
        ["KALMAN_GNSS_R_MAX_MM2", (
            KALMAN_GNSS_R_MAX_MM2 if KALMAN_GNSS_R_MAX_MM2 is not None else "不设上限"
        )],
        ["KALMAN_GNSS_MEDIAN_WINDOW", KALMAN_GNSS_MEDIAN_WINDOW],
        ["KALMAN_GNSS_NOISE_WINDOW", KALMAN_GNSS_NOISE_WINDOW],
        ["GNSS_3D_JUMP_THRESHOLD_MM", GNSS_3D_JUMP_THRESHOLD_MM],
        ["CAMERA_JUMP_THRESHOLD_MM", CAMERA_JUMP_THRESHOLD_MM],
        ["MAX_TIME_DIFFERENCE_SECONDS", MAX_TIME_DIFFERENCE_SECONDS],
        ["REFERENCE_X_M", reference_xyz_m[0]],
        ["REFERENCE_Y_M", reference_xyz_m[1]],
        ["REFERENCE_Z_M", reference_xyz_m[2]],
    ]
    parameter_rows.extend([
        ["R_MODE", "固定Q、自适应R" if adaptive_r else "固定Q、固定R"],
        ["EXPERIMENT_Q", ORIGINAL_EXPERIMENT_Q_MM2],
        ["EXPERIMENT_R", (
            f"clip({ORIGINAL_EXPERIMENT_R_MM2}+{LOCAL_R_ALPHA}×局部增量方差, {ORIGINAL_EXPERIMENT_R_MM2 * LOCAL_R_MIN_FACTOR}, {ORIGINAL_EXPERIMENT_R_MM2 * LOCAL_R_MAX_FACTOR})"
            if adaptive_r else ORIGINAL_EXPERIMENT_R_MM2
        )],
        ["ADAPTIVE_R_WINDOW_SECONDS", ADAPTIVE_R_WINDOW_SECONDS],
        ["ADAPTIVE_R_MIN_SAMPLES", ADAPTIVE_R_MIN_SAMPLES],
        ["ADAPTIVE_R_ALPHA", ADAPTIVE_R_ALPHA],
        ["ADAPTIVE_R_MIN_FACTOR", ADAPTIVE_R_MIN_FACTOR],
        ["ADAPTIVE_R_MAX_FACTOR", ADAPTIVE_R_MAX_FACTOR],
        ["ADAPTIVE_Q_ENABLED", "启用" if ADAPTIVE_Q_ENABLED else "关闭"],
        ["ADAPTIVE_Q_WINDOW_SECONDS", ADAPTIVE_Q_WINDOW_SECONDS],
        ["ADAPTIVE_Q_MIN_SAMPLES", ADAPTIVE_Q_MIN_SAMPLES],
        ["ADAPTIVE_Q_ALPHA", ADAPTIVE_Q_ALPHA],
        ["ADAPTIVE_Q_MIN_MM2", ADAPTIVE_Q_MIN_MM2],
        ["ADAPTIVE_Q_MAX_MM2", ADAPTIVE_Q_MAX_MM2],
        ["ADAPTIVE_Q_NIS_CAP", ADAPTIVE_Q_NIS_CAP],
        ["KALMAN_GAIN_MAX", KALMAN_GAIN_MAX],
        ["INITIALIZATION_DURATION_SECONDS", INITIALIZATION_DURATION_SECONDS],
        ["GNSS_R_SOURCE", "静止GNSS经验方差；不使用ratio"],
        ["GNSS_MEASUREMENT_MODE", "30分钟居中长窗口中位数" if USE_GNSS_LONG_MEDIAN_FOR_FUSION else "原始GNSS投影位移"],
        ["RATIO_SCREENING", "未启用"],
    ])
    parameter_names = {
        "LOCAL_R_ALPHA": "局部增量方差权重α",
        "LOCAL_R_MIN_FACTOR": "局部自适应R下限倍数",
        "LOCAL_R_MAX_FACTOR": "局部自适应R上限倍数",
        "PAIRING_METHOD": "数据配对方式",
        "R_MODE": "观测方差R模式",
        "EXPERIMENT_Q": "本实验正常预测Q（平方毫米）",
        "EXPERIMENT_R": "本实验R（平方毫米或计算方式）",
        "ADAPTIVE_R_WINDOW_SECONDS": "自适应R窗口时长（秒）",
        "ADAPTIVE_R_MIN_SAMPLES": "自适应R最少样本数",
        "ADAPTIVE_R_ALPHA": "自适应R平滑系数",
        "ADAPTIVE_R_MIN_FACTOR": "自适应R最小因子",
        "ADAPTIVE_R_MAX_FACTOR": "自适应R最大放大因子",
        "ADAPTIVE_Q_ENABLED": "有效Q自适应开关",
        "ADAPTIVE_Q_WINDOW_SECONDS": "自适应Q窗口时长（秒）",
        "ADAPTIVE_Q_MIN_SAMPLES": "自适应Q最少有效历元数",
        "ADAPTIVE_Q_ALPHA": "自适应Q更新系数",
        "ADAPTIVE_Q_MIN_MM2": "正常预测基础Q下限（平方毫米）",
        "ADAPTIVE_Q_MAX_MM2": "正常预测基础Q上限（平方毫米）",
        "ADAPTIVE_Q_NIS_CAP": "Q学习用归一化新息平方上限",
        "MAD_WINDOW": "中位数绝对偏差检测窗口（点）",
        "MAD_THRESHOLD": "中位数绝对偏差检测阈值",
        "GNSS_SMOOTH_WINDOW": "GNSS平滑窗口（点）",
        "GNSS_SMOOTH_METHOD": "GNSS平滑方法",
        "GNSS_LONG_MEDIAN_WINDOW_SECONDS": "GNSS长窗口中位数时长（秒）",
        "CONSISTENCY_WINDOW_SECONDS": "增量一致性窗口时长（秒）",
        "CAMERA_GNSS_INCREMENT_ERROR_THRESHOLD_MM": "相机与GNSS增量差阈值（毫米）",
        "GNSS_LOCAL_VOLATILITY_WINDOW_SECONDS": "GNSS局部波动窗口时长（秒）",
        "GNSS_LOCAL_VOLATILITY_THRESHOLD_MM": "GNSS局部增量波动标准差阈值（毫米）",
        "KALMAN_INITIAL_VARIANCE_MM2": "卡尔曼初始方差（平方毫米）",
        "KALMAN_CAMERA_PROCESS_NOISE_MM2": "相机正常时过程噪声方差（平方毫米）",
        "KALMAN_CAMERA_MOTION_NOISE_FACTOR": "相机运动噪声系数",
        "KALMAN_CAMERA_BAD_PROCESS_NOISE_MM2": "相机异常时过程噪声方差（平方毫米）",
        "KALMAN_GNSS_R_MIN_MM2": "GNSS观测噪声方差下限（平方毫米）",
        "KALMAN_GNSS_R_MAX_MM2": "GNSS观测噪声方差上限（平方毫米）",
        "KALMAN_GNSS_MEDIAN_WINDOW": "GNSS诊断用中位数窗口（点，不用于观测平滑）",
        "KALMAN_GNSS_NOISE_WINDOW": "GNSS诊断用局部噪声窗口（点，不用于R）",
        "GNSS_3D_JUMP_THRESHOLD_MM": "GNSS三维跳变阈值（毫米）",
        "CAMERA_JUMP_THRESHOLD_MM": "相机位移增量跳变阈值（毫米）",
        "MAX_TIME_DIFFERENCE_SECONDS": "时间同步允许最大时间差（秒）",
        "REFERENCE_X_M": "GNSS参考X坐标（米）",
        "REFERENCE_Y_M": "GNSS参考Y坐标（米）",
        "REFERENCE_Z_M": "GNSS参考Z坐标（米）",
        "KALMAN_GAIN_MAX": "卡尔曼增益K上限",
        "INITIALIZATION_DURATION_SECONDS": "初始化时长（秒）",
        "GNSS_R_SOURCE": "观测噪声方差来源",
        "GNSS_MEASUREMENT_MODE": "GNSS卡尔曼观测模式",
        "RATIO_SCREENING": "模糊度比值筛选",
    }
    parameters = pd.DataFrame(parameter_rows, columns=["参数名称", "参数值"])
    parameters["参数名称"] = parameters["参数名称"].map(parameter_names)
    parameters["参数值"] = parameters["参数值"].replace(
        {"mean": "滑动均值", "median": "滑动中位数"}
    )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
            export_data.to_excel(writer, sheet_name="融合数据", index=False)
            parameters.to_excel(writer, sheet_name="参数配置", index=False)
            if metrics is None:
                metrics = calculate_statistics(data)
            metrics.to_excel(writer, sheet_name="统计指标", index=False)
            for worksheet in writer.sheets.values():
                worksheet.freeze_panes = "B2"
                worksheet.auto_filter.ref = worksheet.dimensions
                for cells in worksheet.columns:
                    # 中文字符按两个显示宽度估计，便于阅读完整表头。
                    text = str(cells[0].value or "")
                    width = sum(2 if ord(char) > 127 else 1 for char in text)
                    worksheet.column_dimensions[cells[0].column_letter].width = max(
                        18, min(64, width + 2)
                    )
                    header = str(cells[0].value or "")
                    for cell in cells[1:]:
                        if isinstance(cell.value, float):
                            # 仅控制显示精度，保留 Excel 内的原始数值。
                            if worksheet.title == "参数配置":
                                cell.number_format = (
                                    "0.00E+00" if 0 < abs(cell.value) < 0.01
                                    else "0.00"
                                )
                            elif worksheet.title == "融合数据" and (
                                "方差" in header or "基础Q" in header
                                or "候选Q" in header or "因子" in header
                            ):
                                cell.number_format = "0.00E+00"
                            else:
                                cell.number_format = "0.00"
    except PermissionError as error:
        raise PermissionError(
            f"无法写入 {OUTPUT_FILE}，请先关闭正在打开的结果文件。"
        ) from error


def main():
    global ORIGINAL_EXPERIMENT_Q_MM2
    """执行 GNSS 清洗、时间同步、自适应 Kalman 融合、保存和绘图。"""
    validate_config()
    camera = load_camera_data(CAMERA_FILE)
    increments = camera["camera_original"].diff().dropna()
    if len(increments) < 2 or not np.isfinite(increments).all():
        raise ValueError("计算相机Q至少需要3条有限位移数据。")
    ORIGINAL_EXPERIMENT_Q_MM2 = float(increments.var(ddof=1))
    print(f"相机增量样本数: {len(increments)}, 经验Q: {ORIGINAL_EXPERIMENT_Q_MM2:.12g} mm²")
    gnss = load_gnss_data(GNSS_FILE)
    gnss = apply_gnss_quality_screen(gnss)
    gnss = calculate_gnss_displacement(gnss)
    reference_xyz_m = gnss.attrs["reference_xyz_m"].copy()
    gnss = detect_gnss_jumps(gnss)
    gnss = detect_gnss_mad(gnss)
    displacement = gnss[["dX_mm", "dY_mm", "dZ_mm"]].to_numpy(float)
    gnss[["gnss_camera_x", "gnss_camera_y", "gnss_camera_z"]] = displacement @ ROTATION_MATRIX.T
    gnss["gnss_original"] = gnss["gnss_camera_y"]
    gnss = evaluate_gnss_local_volatility(gnss)
    synchronized = synchronize_data(camera, gnss)
    synchronized = check_increment_consistency(synchronized)
    print(f"相机{len(camera)}条，GNSS{len(gnss)}条，时间匹配{synchronized['time_sync_ok'].sum()}条，最大时间差{synchronized['time_difference_s'].max():.3f}秒")
    output_root = BASE_DIR / "kalman_count_sampled_plots"
    result = adaptive_kalman_fusion(synchronized, adaptive_r=True)
    metrics = calculate_statistics(result)
    global OUTPUT_FILE, METRICS_FILE, PLOT_DIR
    PLOT_DIR = output_root / "固定Q_局部波动自适应R_alpha5"
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE = PLOT_DIR / "融合结果.xlsx"
    METRICS_FILE = PLOT_DIR / "指标.csv"
    save_results(result, reference_xyz_m, metrics, adaptive_r=True)
    metrics.to_csv(METRICS_FILE, index=False, encoding="utf-8-sig",
                   float_format="%.2f")
    plot_results(result)
    fig, axes = plt.subplots(2, 1, figsize=(12,7), sharex=True)
    axes[0].plot(result['time'], result['kalman_measurement_noise_r'])
    axes[0].set_ylabel('R / mm²')
    axes[1].plot(result['time'], result['kalman_gain'])
    axes[1].set(xlabel='时间', ylabel='卡尔曼增益')
    for ax in axes:
        ax.grid(True)
    fig.tight_layout()
    if SAVE_PLOTS:
        fig.savefig(PLOT_DIR / '07_R与增益.png', dpi=160)
    plt.close(fig)
    print(f"R范围: {result['kalman_measurement_noise_r'].min():.6f}～{result['kalman_measurement_noise_r'].max():.6f} mm²")
    print("\n位移统计指标：")
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.2f}"))
    print(f"统计指标已保存：{METRICS_FILE}")
    print(f"Fusion result saved to: {OUTPUT_FILE}")
    print(f"Plots saved to: {PLOT_DIR}")
    print(f"Rows: {len(result)}")
    print(f"GNSS corrections used: {int(result['gnss_correction_used'].sum())}")


if __name__ == "__main__":
    main()
