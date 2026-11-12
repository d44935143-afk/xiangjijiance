"""Shared grayscale-gradient edge extraction and robust line fitting."""
import cv2
import numpy as np


def gradient_edge_points(gradient, rect, vertical=True):
    """Search the central 50% for strongest gradients; omit the central 10% crossing."""
    x, y, w, h = rect
    cx, cy = x + w / 2, y + h / 2
    length = gradient.shape[1] if vertical else gradient.shape[0]
    origin, extent = (cx, w) if vertical else (cy, h)
    lo = max(1, int(np.ceil(origin - 0.25 * extent)))
    hi = min(length - 2, int(np.floor(origin + 0.25 * extent)))
    points = []
    scan_start, scan_end = ((y + 0.15 * h, y + 0.85 * h) if vertical
                             else (x + 0.15 * w, x + 0.85 * w))
    cross, cross_size = (cy, h) if vertical else (cx, w)
    for scan in range(int(np.ceil(scan_start)), int(np.floor(scan_end))):
        if abs(scan - cross) < 0.05 * cross_size or lo > hi:
            continue
        profile = np.abs(gradient[scan, :] if vertical else gradient[:, scan])
        peak = lo + int(np.argmax(profile[lo:hi + 1]))
        if peak == lo or peak == hi or profile[peak] < 5.0:
            continue
        a, b, c = [float(v) for v in profile[peak - 1:peak + 2]]
        denominator = a - 2 * b + c
        offset = 0.5 * (a - c) / denominator if denominator < -1e-9 else 0.0
        position = peak + float(np.clip(offset, -0.5, 0.5))
        points.append([position, float(scan)] if vertical else [float(scan), position])
    return np.asarray(points, dtype=np.float32).reshape(-1, 2)


def robust_line(points):
    if len(points) < 6:
        return None
    vx, vy, x0, y0 = cv2.fitLine(points, cv2.DIST_HUBER, 0, 0.01, 0.01).reshape(-1)
    return ([float(x0 - vx * 10000), float(y0 - vy * 10000)],
            [float(x0 + vx * 10000), float(y0 + vy * 10000)])
