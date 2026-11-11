# -*- coding: utf-8 -*-
"""研究相机Y方向GNSS经验残差与POS报告协方差的关系，不进行融合。
使用用户提供的静态ECEF真值坐标，计算相机Y方向绝对误差。
去线性趋势残差仅作敏感性对比；相关性不等于因果或精度证明。
"""
from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
GNSS_FILE = BASE_DIR / 'gnss_data' / 'ohlo3000.pos'
OUTPUT_DIR = BASE_DIR / 'gnss_y_covariance_analysis'
# 默认与当前融合程序的相机Y方向相同；研究ECEF Y时改为[0,1,0]。
Y_DIRECTION = np.array([1., 1., 1.]) / np.sqrt(3)
TRUE_XYZ_M = np.array([648611.1274, -4755280.8146, 4187111.9223])
LOCAL_WINDOW = '30min'
LOCAL_MIN_SAMPLES = 10
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei','SimHei','DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

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


def load_data(path):
    """读取RTKLIB ECEF POS；保留所有解算状态，按时间排序去重。"""
    with path.open(encoding='utf-8',errors='replace') as f:
        header=' '.join(line.lower() for line in f if line.startswith('%'))
    if ('latitude' in header or 'longitude' in header) and 'x-ecef' not in header:
        raise ValueError('需要ECEF XYZ格式POS，不能直接使用经纬高。')
    raw=pd.read_csv(path,sep=r'\s+',comment='%',header=None)
    if raw.shape[1]<15:
        raise ValueError('POS至少需要15列。')
    data=pd.DataFrame({'time':pd.to_datetime(raw[0].astype(str)+' '+raw[1].astype(str),errors='coerce')})
    for name,i in zip(['X','Y','Z','Q','ns','sdx','sdy','sdz','sdxy','sdyz','sdzx','age','ratio'],range(2,15)):
        data[name]=pd.to_numeric(raw[i],errors='coerce')
    if data[['time','X','Y','Z']].isna().any().any():
        raise ValueError('时间或XYZ无效。')
    return data.sort_values('time').drop_duplicates('time').reset_index(drop=True)


def correlation(x,y):
    """返回有效样本上的Pearson与Spearman相关，不计算独立样本显著性。"""
    d=pd.DataFrame({'x':x,'y':y}).replace([np.inf,-np.inf],np.nan).dropna()
    if len(d)<3 or d.x.nunique()<2 or d.y.nunique()<2:
        return np.nan,np.nan
    return d.x.corr(d.y),d.x.rank().corr(d.y.rank())


def analyze(data):
    """投影完整协方差，计算残差、局部经验方差和标准化误差。"""
    a=np.asarray(Y_DIRECTION,dtype=float)
    if a.shape!=(3,) or not np.isfinite(a).all() or not np.isclose(np.linalg.norm(a),1):
        raise ValueError('Y_DIRECTION必须为三维单位向量。')
    d=data.copy()
    xyz=d[['X','Y','Z']].to_numpy(float)
    error_xyz=(xyz-TRUE_XYZ_M)*1000
    d[['error_X_mm','error_Y_ecef_mm','error_Z_mm']]=error_xyz
    y=error_xyz@a
    d['y_mm']=y
    d['reported_var_mm2']=[project_gnss_solution_variance(*row,a) for row in d[['sdx','sdy','sdz','sdxy','sdyz','sdzx']].to_numpy(float)]
    d['reported_std_mm']=np.sqrt(d.reported_var_mm2)
    d['error_demeaned_mm']=y-y.mean()
    d['error_true_mm']=y
    t=(d.time-d.time.iloc[0]).dt.total_seconds().to_numpy()/3600
    trend=np.polyval(np.polyfit(t,y,1),t)
    d['residual_detrended_mm']=y-trend
    d['local_empirical_var_mm2']=pd.Series(y,index=pd.DatetimeIndex(d.time)).rolling(LOCAL_WINDOW,min_periods=LOCAL_MIN_SAMPLES).var(ddof=1).to_numpy()
    d['local_reported_var_mm2']=pd.Series(d.reported_var_mm2.to_numpy(),index=pd.DatetimeIndex(d.time)).rolling(LOCAL_WINDOW,min_periods=LOCAL_MIN_SAMPLES).mean().to_numpy()
    d['normalized_residual']=d.error_true_mm/d.reported_std_mm
    return d


def summarize(d):
    """全部、固定解和浮点解分别统计，帮助识别解算类型混合造成的相关。"""
    rows=[]
    for label,g in [('全部',d),('Q=1固定解',d[d.Q.eq(1)]),('Q=2浮点解',d[d.Q.eq(2)])]:
        v=g[np.isfinite(g.reported_var_mm2)&g.reported_var_mm2.gt(0)]
        row={'分组':label,'总数':len(g),'有效协方差数量':len(v)}
        if len(v):
            row.update({'真值误差偏差(mm)':v.error_true_mm.mean(),
                '真值误差标准差(mm)':v.error_true_mm.std(ddof=1),
                '真值误差RMSE(mm)':np.sqrt(np.mean(v.error_true_mm**2)),
                '去趋势RMSE(mm)':np.sqrt(np.mean(v.residual_detrended_mm**2)),
                '报告方差均值(mm²)':v.reported_var_mm2.mean(),
                '残差均方/报告方差均值':np.mean(v.error_true_mm**2)/v.reported_var_mm2.mean(),
                '标准化残差均方':np.mean(v.normalized_residual**2),
                '1sigma覆盖率':np.mean(v.error_true_mm.abs()<=v.reported_std_mm),
                '2sigma覆盖率':np.mean(v.error_true_mm.abs()<=2*v.reported_std_mm),
                '3sigma覆盖率':np.mean(v.error_true_mm.abs()<=3*v.reported_std_mm)})
            for name,x,y in [('绝对真值误差与报告标准差',v.error_true_mm.abs(),v.reported_std_mm),
                ('真值误差平方与报告方差',v.error_true_mm**2,v.reported_var_mm2),
                ('去趋势真值误差平方与报告方差',v.residual_detrended_mm**2,v.reported_var_mm2),
                ('局部经验方差与局部报告方差',v.local_empirical_var_mm2,v.local_reported_var_mm2)]:
                row[name+'_Pearson'],row[name+'_Spearman']=correlation(x,y)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    d=analyze(load_data(GNSS_FILE)); summary=summarize(d)
    valid=d[d.reported_var_mm2.notna()].copy()
    if valid.empty:
        raise ValueError('没有有效的报告协方差，不能研究关系。')
    valid['variance_bin']=pd.qcut(valid.reported_var_mm2,q=8,duplicates='drop')
    bins=valid.groupby('variance_bin',observed=True).agg(count=('y_mm','size'),reported_variance=('reported_var_mm2','mean'),
        residual_mean_square=('error_true_mm',lambda x:np.mean(x**2)),detrended_mean_square=('residual_detrended_mm',lambda x:np.mean(x**2))).reset_index()
    bins['variance_bin']=bins.variance_bin.astype(str)
    OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
    with pd.ExcelWriter(OUTPUT_DIR/'GNSS_Y误差与协方差.xlsx',engine='openpyxl') as w:
        d.to_excel(w,sheet_name='逐历元',index=False)
        summary.to_excel(w,sheet_name='分组指标',index=False)
        bins.to_excel(w,sheet_name='报告方差分箱',index=False)
        pd.DataFrame({'参数':['方向','局部窗口','误差定义','参考限制'], '值':[str(Y_DIRECTION),LOCAL_WINDOW,f'坐标减真值{TRUE_XYZ_M.tolist()}后投影；另计算去均值及去趋势残差','静态真值需与POS使用相同坐标框架和历元；局部窗口重叠']}).to_excel(w,sheet_name='说明',index=False)
    summary.to_csv(OUTPUT_DIR/'分组指标.csv',index=False,encoding='utf-8-sig')
    fig,axs=plt.subplots(3,1,figsize=(13,9),sharex=True)
    axs[0].plot(d.time,d.error_true_mm,label='真值误差'); axs[0].plot(d.time,2*d.reported_std_mm,'--',label='±2σ报告范围'); axs[0].plot(d.time,-2*d.reported_std_mm,'--'); axs[0].set_ylabel('mm'); axs[0].legend()
    axs[1].plot(d.time,d.reported_var_mm2,label='POS投影方差'); axs[1].set_ylabel('mm²'); axs[1].legend()
    axs[2].plot(d.time,d.local_empirical_var_mm2,label='30分钟经验方差'); axs[2].plot(d.time,d.local_reported_var_mm2,label='30分钟平均报告方差'); axs[2].set_ylabel('mm²'); axs[2].legend()
    for ax in axs: ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUTPUT_DIR/'01_时间与方差.png',dpi=160); plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(12,5))
    for q,g in valid.groupby('Q'):
        axs[0].scatter(g.reported_var_mm2,g.error_true_mm**2,s=8,alpha=.4,label=f'Q={q}')
    axs[0].set(xlabel='报告方差 / mm²',ylabel='真值误差平方 / mm²'); axs[0].legend()
    axs[1].plot(bins.reported_variance,bins.residual_mean_square,'o-',label='分箱残差均方'); axs[1].plot(bins.reported_variance,bins.reported_variance,'--',label='理想1:1参考'); axs[1].set(xlabel='分箱报告方差均值 / mm²',ylabel='残差均方 / mm²'); axs[1].legend()
    for ax in axs: ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUTPUT_DIR/'02_误差与报告方差.png',dpi=160); plt.close(fig)
    print(summary.to_string(index=False,float_format=lambda x:f'{x:.4f}'))
    print(f'输出目录: {OUTPUT_DIR}')


if __name__=='__main__':
    main()
