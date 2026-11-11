"""静止目标RMSE：相机相对位移真值为0，GNSS使用已知ECEF真值。"""
import numpy as np
import pandas as pd
TRUE_XYZ_M = np.array([648611.1274, -4755280.8146, 4187111.9223])

def static_rmse(data, column, direction=None):
    values = pd.to_numeric(data[column], errors='coerce').to_numpy(float)
    if column.startswith('gnss'):
        if direction is None:
            direction = np.ones(3)/np.sqrt(3)
        xyz = data[['X','Y','Z']].to_numpy(float)
        errors = ((xyz-TRUE_XYZ_M)*1000) @ np.asarray(direction)
        errors = errors[np.isfinite(values)]
    elif column in ('fusion', 'fusion_for_compensation', 'target1_corrected_mm'):
        errors = values-values[0]
    else:
        errors = values
    errors = errors[np.isfinite(errors)]
    return float(np.sqrt(np.mean(errors**2))) if len(errors) else float('nan')
