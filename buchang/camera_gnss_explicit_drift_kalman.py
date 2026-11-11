# -*- coding: utf-8 -*-
"""固定R的相机公共漂移卡尔曼实验；不筛选、剔除或平滑GNSS。"""
from pathlib import Path
import math
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from static_rmse import static_rmse
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = Path(__file__).resolve().parent
CAMERA_FILE = BASE_DIR / 'camera_data' / 'mubiao1.xlsx'
GNSS_FILE = BASE_DIR / 'gnss_data' / 'ohlo3000.pos'
CAMERA_HAS_HEADER = False
CAMERA_TIME_COLUMN = None  # 有时间列时设列名/索引，如 0 或 '时间'
CAMERA_TARGET1_COLUMN = 0
CAMERA_TARGET2_COLUMN = 1
CAMERA_TARGET2_TRUTH_COLUMN = None
CAMERA_SAMPLE_INTERVAL_SECONDS = 30.0
CAMERA_START_TIME = None  # 推荐明确设置，例如 '2024-10-26 00:00:00'
CAMERA_TIME_SHIFT_SECONDS = 0.0  # 时标差修正，例如已知相机慢2秒，设为 +2
CAMERA_ZERO_AT_START = True
REFERENCE_XYZ_M = None  # 可以显式指定 ECEF 参考坐标，例如 (X0, Y0, Z0)
REFERENCE_SAMPLE_COUNT = 60  # 质量合格的前60个GNSS历元平均值作为参考坐标

ROTATION_MATRIX = np.array([
    [1 / np.sqrt(2), -1 / np.sqrt(2), 0],
    [1 / np.sqrt(3),  1 / np.sqrt(3), 1 / np.sqrt(3)],
    [-1 / np.sqrt(6), -1 / np.sqrt(6), 2 / np.sqrt(6)],
], dtype=float)

GNSS_MATCH_DIRECTION = 'nearest'  # 'nearest' 或 'backward'；backward适合实时因果处理
GNSS_MATCH_TOLERANCE_SECONDS = 2.0
DRIFT_RANDOM_WALK_VAR_PER_HOUR_MM2 = 0.1
INITIAL_BIAS_STD_MM = 3.0
INITIALIZATION_DURATION_SECONDS = 180.0
INITIALIZATION_MIN_SAMPLES = 3
TARGET2_STATIC_TRUE_MM = 0
SAVE_PLOTS = True
OUTPUT_ROOT = BASE_DIR / 'fixed_drift_experiment_plots'
FIXED_R_MM2 = 2.34

def get_column(data, column, display_name):
    if column in data.columns:
        return data[column]
    if isinstance(column, int) and 0 <= column < data.shape[1]:
        return data.iloc[:, column]
    raise KeyError(f'找不到{display_name}列 {column!r}，现有字段：{list(data.columns)}')

def load_camera(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f'找不到相机Excel文件：{path}')
    table = pd.read_excel(path, header=0 if CAMERA_HAS_HEADER else None)
    if table.empty:
        raise ValueError('相机Excel为空。')
    target1 = pd.to_numeric(get_column(table, CAMERA_TARGET1_COLUMN, '相机目标一'), errors='coerce')
    target2 = pd.to_numeric(get_column(table, CAMERA_TARGET2_COLUMN, '相机目标二'), errors='coerce')
    truth = None
    if CAMERA_TARGET2_TRUTH_COLUMN is not None:
        truth = pd.to_numeric(get_column(table, CAMERA_TARGET2_TRUTH_COLUMN, '目标二真值'), errors='coerce')
        if truth.isna().any():
            raise ValueError('目标二真值列存在空值或无法转为数值。')
    if target1.isna().any() or target2.isna().any():
        raise ValueError('相机目标一/目标二存在非数值或空白；应在上游明确处理，不能静默插值。')
    if CAMERA_TIME_COLUMN is not None:
        camera_time = pd.to_datetime(get_column(table, CAMERA_TIME_COLUMN, '相机时间'), errors='coerce')
        if camera_time.isna().any():
            raise ValueError('相机时间列有无法解析的值。')
    else:
        camera_time = pd.Series(pd.NaT, index=table.index, dtype='datetime64[ns]')
    c1, c2 = target1.to_numpy(float), target2.to_numpy(float)
    if CAMERA_ZERO_AT_START:
        c1 = c1 - c1[0]
        c2 = c2 - c2[0]
    out = pd.DataFrame({'time': camera_time.to_numpy(), 'camera_target1_mm': c1, 'camera_target2_mm': c2})
    if truth is not None:
        vals = truth.to_numpy(float)
        if CAMERA_ZERO_AT_START:
            vals = vals - vals[0]
        out['target2_truth_mm'] = vals
    return out

def _find_col(columns, alternatives, required=False):
    names = {str(c).strip().lower(): c for c in columns}
    for value in alternatives:
        if value.lower() in names:
            return names[value.lower()]
    if required:
        raise ValueError(f'GNSS Excel 缺少必需列，候选：{alternatives}')
    return None

def load_gnss(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f'找不到GNSS文件：{path}')
    if path.suffix.lower() == '.pos':
        # RTKLIB .pos也可能输出经纬高(LLH)。本程序必须读ECEF XYZ。
        with path.open('r', encoding='utf-8', errors='replace') as handle:
            header_lines = [handle.readline().lower() for _ in range(12)]
        header_text = ' '.join(line for line in header_lines if line.startswith('%'))
        if ('latitude' in header_text or 'longitude' in header_text) and 'x-ecef' not in header_text:
            raise ValueError('该POS文件疑似经纬高(LLH)格式；请在RTKPOST中输出ECEF XYZ格式。')
        raw = pd.read_csv(path, sep=r'\s+', comment='%', header=None)
        if raw.empty or raw.shape[1] < 15:
            raise ValueError('POS文件应至少有15列：日期 时间 XYZ Q ns sdx sdy sdz sdxy sdyz sdzx age ratio。')
        t = pd.to_datetime(raw.iloc[:, 0].astype(str) + ' ' + raw.iloc[:, 1].astype(str), errors='coerce')
        fields = dict(zip(['X', 'Y', 'Z', 'Q', 'ns', 'sdx', 'sdy', 'sdz', 'sdxy', 'sdyz', 'sdzx', 'age', 'ratio'], range(2, 15)))
        out = pd.DataFrame({'time': t})
        for key, index in fields.items():
            out[key] = pd.to_numeric(raw.iloc[:, index], errors='coerce').to_numpy()
    elif path.suffix.lower() in ('.xlsx', '.xls'):
        raw = pd.read_excel(path)
        names = list(raw.columns)
        mapping = {
            'time': ['time', '时间', 'datetime', 'timestamp', 'gpst'],
            'X': ['x', 'x(m)', 'x坐标(m)', 'x-ecef(m)'],
            'Y': ['y', 'y(m)', 'y坐标(m)', 'y-ecef(m)'],
            'Z': ['z', 'z(m)', 'z坐标(m)', 'z-ecef(m)'],
            'Q': ['q', 'quality'], 'ns': ['ns', '卫星数', 'satellites'],
            'sdx': ['sdx', 'sdx(m)'], 'sdy': ['sdy', 'sdy(m)'], 'sdz': ['sdz', 'sdz(m)'],
            'sdxy': ['sdxy', 'sdxy(m)'], 'sdyz': ['sdyz', 'sdyz(m)'], 'sdzx': ['sdzx', 'sdzx(m)'],
            'age': ['age'], 'ratio': ['ratio'],
        }
        out = pd.DataFrame(index=raw.index)
        for key, alternatives in mapping.items():
            column = _find_col(names, alternatives, required=key in ('time', 'X', 'Y', 'Z'))
            if column is None:
                out[key] = np.nan
            elif key == 'time':
                out[key] = pd.to_datetime(raw[column], errors='coerce')
            else:
                out[key] = pd.to_numeric(raw[column], errors='coerce')
        out = out.reset_index(drop=True)
    else:
        raise ValueError('GNSS仅支持RTKLIB .pos或 .xlsx/.xls。')
    if out[['time', 'X', 'Y', 'Z']].isna().any().any():
        raise ValueError('GNSS时间或ECEF XYZ包含缺失或不可解析的值。')
    out = out.sort_values('time').drop_duplicates('time').reset_index(drop=True)
    if len(out) < 3:
        raise ValueError('GNSS有效行数不足。')
    return out

def create_camera_times(camera, gnss):
    camera = camera.copy()
    if camera['time'].isna().all():
        if CAMERA_START_TIME is None:
            start = gnss['time'].iloc[0]
            warnings.warn('未提供相机时间列和 CAMERA_START_TIME：暂按相机首帧与GNSS首历元同步。'
                          ' 若真实起点不同，融合将产生错误结果，请设置 CAMERA_START_TIME。', stacklevel=2)
        else:
            start = pd.Timestamp(CAMERA_START_TIME)
        camera['time'] = start + pd.to_timedelta(np.arange(len(camera)) * CAMERA_SAMPLE_INTERVAL_SECONDS, unit='s')
    elif camera['time'].isna().any():
        raise ValueError('相机时间列不可部分缺失。')
    if CAMERA_TIME_SHIFT_SECONDS:
        camera['time'] += pd.to_timedelta(CAMERA_TIME_SHIFT_SECONDS, unit='s')
    if camera['time'].duplicated().any():
        raise ValueError('相机存在重复时间，无法逐帧融合。')
    camera = camera.sort_values('time').reset_index(drop=True)
    if camera['time'].diff().dropna().dt.total_seconds().le(0).any():
        raise ValueError('相机时间必须严格递增。')
    return camera

def metrics_for_series(time, series):  # 统计位移波动、分布和趋势
    """按有限数据统计样本标准差、均值、极值、极差和每小时趋势。"""
    y = pd.to_numeric(series, errors='coerce').to_numpy(float)
    hours = (time - time.iloc[0]).dt.total_seconds().to_numpy(float) / 3600
    valid = np.isfinite(y) & np.isfinite(hours)
    y, t = y[valid], hours[valid]
    return {
        '标准差(mm)': float(np.std(y, ddof=1)) if len(y) > 1 else np.nan,
        '有效数量': len(y),
        '均值(mm)': float(y.mean()) if len(y) else np.nan,
        '最小值(mm)': float(y.min()) if len(y) else np.nan,
        '最大值(mm)': float(y.max()) if len(y) else np.nan,
        '极差(mm)': float(np.ptp(y)) if len(y) else np.nan,
        '趋势斜率(mm/小时)': float(np.polyfit(t, y, 1)[0]) if len(y) > 1 and np.ptp(t) > 0 else np.nan,
    }


def make_experiment_plots(aligned, result, code, output_dir):
    """每组实验的图片只保存到对应文件夹，不弹出窗口。"""
    if not SAVE_PLOTS:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    t = aligned['time']
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(t, aligned['camera_target1_mm'], label='目标一融合前', lw=1)
    ax.plot(t, aligned['gnss_camera_y_mm'], label='GNSS投影', lw=.8, alpha=.6)
    ax.plot(t, result['target1_corrected_mm'], label='目标一融合后', lw=1.5)
    ax.set(xlabel='时间', ylabel='位移 / mm', title=f'{code} 目标一融合前后与GNSS位移')
    ax.grid(alpha=.3); ax.legend(); fig.tight_layout()
    fig.savefig(output_dir / '目标一位移对比.png', dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(t, aligned['camera_target2_mm'], label='目标二原始位移', lw=1)
    ax.plot(t, result['target2_corrected_mm'], label='目标二补偿后', lw=1.5)
    ax.set(xlabel='时间', ylabel='位移 / mm', title=f'{code} 目标二补偿前后位移')
    ax.grid(alpha=.3); ax.legend(); fig.tight_layout()
    fig.savefig(output_dir / '目标二位移对比.png', dpi=180); plt.close(fig)

def prepare_gnss(gnss):
    """只转换坐标，不依据Q、ratio、卫星数、尖峰或协方差筛选。"""
    result = gnss.copy()
    reference = (result[['X', 'Y', 'Z']].head(REFERENCE_SAMPLE_COUNT).mean().to_numpy(float)
                 if REFERENCE_XYZ_M is None else np.asarray(REFERENCE_XYZ_M, dtype=float))
    if reference.shape != (3,) or not np.isfinite(reference).all():
        raise ValueError('参考坐标必须为有限三维ECEF坐标。')
    displacement = (result[['X', 'Y', 'Z']].to_numpy(float)-reference)*1000
    result['gnss_camera_y_mm'] = displacement @ ROTATION_MATRIX[1]
    return result, reference


def align_by_count(camera, gnss):  # 按相机条数等间隔抽取GNSS并逐行配对
    """以零基索引 i*M//N 抽取；假设两组数据覆盖相同观测时段。"""
    n, m = len(camera), len(gnss)
    if n == 0 or m < n:
        raise ValueError(f'相机{n}条、GNSS{m}条，无法按数量无重复抽取。')
    indices = np.array([i*m//n for i in range(n)], dtype=np.int64)
    selected = gnss.iloc[indices].reset_index(drop=True)
    aligned = camera.reset_index(drop=True).copy()
    aligned['camera_original_time'] = aligned['time']
    # 使用对应GNSS的真实时间计算漂移过程方差，不再生成30秒相机时间。
    aligned['time'] = selected['time']
    aligned['gnss_time'] = selected['time']
    aligned['gnss_camera_y_mm'] = selected['gnss_camera_y_mm']
    aligned['gnss_source_row'] = indices+1
    aligned['camera_source_row'] = np.arange(1, n+1)
    aligned['gnss_matched'] = True  # 表示数量配对完成，未校验时间同步。
    aligned['camera_gnss_diff_mm'] = aligned['camera_target1_mm']-aligned['gnss_camera_y_mm']
    return aligned


def run_fixed_experiment(save=True):
    if FIXED_R_MM2 <= 0 or DRIFT_RANDOM_WALK_VAR_PER_HOUR_MM2 < 0:
        raise ValueError('R必须为正，漂移Q强度不能为负。')
    camera = load_camera(CAMERA_FILE)
    gnss, reference = prepare_gnss(load_gnss(GNSS_FILE))
    aligned = align_by_count(camera, gnss)
    init_end = aligned['time'].iloc[0]+pd.Timedelta(seconds=INITIALIZATION_DURATION_SECONDS)
    initial = aligned.loc[aligned['gnss_matched'] & (aligned['time']<=init_end),'camera_gnss_diff_mm']
    bias = float(initial.median() if len(initial)>=INITIALIZATION_MIN_SAMPLES else aligned.loc[aligned['gnss_matched'],'camera_gnss_diff_mm'].iloc[0])
    variance = INITIAL_BIAS_STD_MM**2
    records=[]
    for i,row in aligned.iterrows():
        q = 0.0 if i==0 else DRIFT_RANDOM_WALK_VAR_PER_HOUR_MM2*(row['time']-aligned['time'].iloc[i-1]).total_seconds()/3600
        variance += q
        gain=0.0; innovation=math.nan; used=False
        warmup=row['time']<=init_end
        if warmup:
            variance=INITIAL_BIAS_STD_MM**2
        elif row['gnss_matched']:
            innovation=float(row['camera_gnss_diff_mm'])-bias
            gain=variance/(variance+FIXED_R_MM2)
            bias+=gain*innovation
            variance=(1-gain)**2*variance+gain**2*FIXED_R_MM2
            used=True
        records.append((bias, math.sqrt(variance), q, FIXED_R_MM2, gain, innovation, used, warmup))
    result=aligned.join(pd.DataFrame(records,columns=['estimated_camera_bias_mm','estimated_camera_bias_std_mm',
        'process_Q_bias_mm2','observation_R_mm2','kalman_gain','camera_gnss_bias_innovation_mm',
        'gnss_kalman_update_used','bias_initialization_warmup']))
    result['estimated_camera_bias_relative_mm']=result['estimated_camera_bias_mm']-result['estimated_camera_bias_mm'].iloc[0]
    for number in (1,2):
        result[f'target{number}_corrected_mm']=result[f'camera_target{number}_mm']-result['estimated_camera_bias_relative_mm']
    summary = pd.DataFrame([
        {'数据类型': name, '数据列': column, **metrics_for_series(result['time'], result[column]),
         'RMSE(mm)': static_rmse(result, column, ROTATION_MATRIX[1]) if column != 'gnss_camera_y_mm' else float(np.sqrt(np.mean((result[column] + float((reference-np.array([648611.1274,-4755280.8146,4187111.9223])) @ ROTATION_MATRIX[1])*1000)**2)))}
        for name, column in [
            ('目标一融合前', 'camera_target1_mm'),
            ('目标一融合后', 'target1_corrected_mm'),
            ('GNSS原始投影', 'gnss_camera_y_mm'),
            ('目标二原始位移', 'camera_target2_mm'),
            ('目标二补偿后', 'target2_corrected_mm'),
        ]
    ])
    if save:
        OUTPUT_ROOT.mkdir(parents=True,exist_ok=True)
        with pd.ExcelWriter(OUTPUT_ROOT/'融合结果.xlsx',engine='openpyxl') as writer:
            result.to_excel(writer,sheet_name='固定R结果',index=False)
            summary.to_excel(writer,sheet_name='统计指标',index=False)
            pd.DataFrame({'参数':['固定R(mm²)','漂移Q强度(mm²/小时)','参考ECEF坐标(m)','GNSS处理'],
                '值':[FIXED_R_MM2,DRIFT_RANDOM_WALK_VAR_PER_HOUR_MM2,str(reference.tolist()),'坐标转换后按i*M//N抽取并逐行配对，无筛选、平滑和新息拒绝']}).to_excel(writer,sheet_name='参数',index=False)
        summary.to_csv(OUTPUT_ROOT/'指标.csv',index=False,encoding='utf-8-sig')
        make_experiment_plots(aligned,result,'固定R',OUTPUT_ROOT)
    return aligned,result,summary


def main():
    aligned,result,summary=run_fixed_experiment()
    print(f"相机历元: {len(aligned)}；GNSS匹配: {aligned['gnss_matched'].sum()}；更新: {result['gnss_kalman_update_used'].sum()}")
    print(summary.to_string(index=False))
    print(f'输出: {OUTPUT_ROOT}')


if __name__=='__main__':
    main()
