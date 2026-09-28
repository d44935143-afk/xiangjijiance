import argparse
import json
from pathlib import Path

import cv2
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_STATE_PATH = PROJECT_DIR / "targets_state.json"
DEFAULT_OUTPUT_PATH = PROJECT_DIR / "mubiaogenzong" / "center_debug.png"


def line_intersection(p1, p2, p3, p4):
    """Return the intersection point of two lines."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denominator) < 1e-9:
        return None

    px = (
        (x1 * y2 - y1 * x2) * (x3 - x4)
        - (x1 - x2) * (x3 * y4 - y3 * x4)
    ) / denominator
    py = (
        (x1 * y2 - y1 * x2) * (y3 - y4)
        - (y1 - y2) * (x3 * y4 - y3 * x4)
    ) / denominator
    return [float(px), float(py)]


def center_from_corners(corners):
    """Compute target center from four ordered corners."""
    if corners is None or len(corners) != 4:
        return None

    points = np.asarray(corners, dtype=np.float64)
    diagonal_center = line_intersection(points[0], points[2], points[1], points[3])
    if diagonal_center is not None:
        return diagonal_center

    center = points.mean(axis=0)
    return [float(center[0]), float(center[1])]


def center_from_homography_mm_to_pixel(
    homography_mm_to_pixel,
    target_size_mm,
):
    """Project the real target center back to image pixel coordinates."""
    if homography_mm_to_pixel is None or target_size_mm is None:
        return None

    homography = np.asarray(homography_mm_to_pixel, dtype=np.float64)
    if homography.shape != (3, 3):
        return None

    half_size = float(target_size_mm) / 2.0
    point_mm = np.array([[[half_size, half_size]]], dtype=np.float64)
    point_px = cv2.perspectiveTransform(point_mm, homography)
    x, y = point_px[0, 0]
    return [float(x), float(y)]


def cv_imread(path):
    """Read an image with support for Chinese paths."""
    path = Path(path)
    if not path.exists():
        print("File does not exist:", path)
        return None

    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def cv_imwrite(path, image):
    """Write an image with support for Chinese paths."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    success, buffer = cv2.imencode(path.suffix, image)
    if success:
        buffer.tofile(str(path))
    return success


def clip_rect(x, y, w, h, image_shape, margin=0):
    """Clip a rectangle to image bounds."""
    image_h, image_w = image_shape[:2]
    x1 = max(0, min(image_w, int(round(x - margin))))
    y1 = max(0, min(image_h, int(round(y - margin))))
    x2 = max(0, min(image_w, int(round(x + w + margin))))
    y2 = max(0, min(image_h, int(round(y + h + margin))))

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2 - x1, y2 - y1


def rect_from_corners(corners):
    """Build an axis-aligned bounding rectangle from four corners."""
    if corners is None or len(corners) != 4:
        return None

    points = np.asarray(corners, dtype=np.float64)
    x1 = float(points[:, 0].min())
    y1 = float(points[:, 1].min())
    x2 = float(points[:, 0].max())
    y2 = float(points[:, 1].max())
    return x1, y1, x2 - x1, y2 - y1


def adaptive_binary_from_roi(roi):
    """Normalize local brightness and binarize one target crop."""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    equalized = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(gray)
    binary = cv2.adaptiveThreshold(
        equalized,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        21,
        3,
    )
    return binary


def center_band_edge_points(binary, vertical=True, band_ratio=0.45):
    """Extract black-white transition points near the target center band."""
    height, width = binary.shape[:2]
    points = []

    if vertical:
        y1 = int(round(height * (0.5 - band_ratio / 2.0)))
        y2 = int(round(height * (0.5 + band_ratio / 2.0)))
        for y in range(max(0, y1), min(height, y2)):
            row = binary[y, :]
            transitions = np.where(row[:-1] != row[1:])[0]
            if transitions.size == 0:
                continue
            x = transitions[np.argmin(np.abs(transitions - width / 2.0))] + 0.5
            points.append([float(x), float(y)])
    else:
        x1 = int(round(width * (0.5 - band_ratio / 2.0)))
        x2 = int(round(width * (0.5 + band_ratio / 2.0)))
        for x in range(max(0, x1), min(width, x2)):
            col = binary[:, x]
            transitions = np.where(col[:-1] != col[1:])[0]
            if transitions.size == 0:
                continue
            y = transitions[np.argmin(np.abs(transitions - height / 2.0))] + 0.5
            points.append([float(x), float(y)])

    if len(points) < 2:
        return None
    return np.asarray(points, dtype=np.float32)


def fit_line_from_points(points):
    """Fit an OpenCV line and return two far points on it."""
    if points is None or len(points) < 2:
        return None

    vx, vy, x0, y0 = cv2.fitLine(
        points,
        cv2.DIST_L2,
        0,
        0.01,
        0.01,
    ).reshape(-1)

    length = 10000.0
    p1 = [float(x0 - vx * length), float(y0 - vy * length)]
    p2 = [float(x0 + vx * length), float(y0 + vy * length)]
    return p1, p2


def center_from_template_frame(
    frame,
    template_corners,
    margin_ratio=0.35,
):
    """Compute center by fitting two black-white center boundary lines."""
    rect = rect_from_corners(template_corners)
    if rect is None:
        return None, None

    x, y, w, h = rect
    margin = max(w, h) * float(margin_ratio)
    clipped = clip_rect(x, y, w, h, frame.shape, margin=margin)
    if clipped is None:
        return None, None

    rx, ry, rw, rh = clipped
    roi = frame[ry:ry + rh, rx:rx + rw]
    binary = adaptive_binary_from_roi(roi)

    vertical_points = center_band_edge_points(binary, vertical=True)
    horizontal_points = center_band_edge_points(binary, vertical=False)
    vertical_line = fit_line_from_points(vertical_points)
    horizontal_line = fit_line_from_points(horizontal_points)

    if vertical_line is None or horizontal_line is None:
        return None, None

    center_local = line_intersection(
        vertical_line[0],
        vertical_line[1],
        horizontal_line[0],
        horizontal_line[1],
    )
    if center_local is None:
        return None, None

    center = [center_local[0] + rx, center_local[1] + ry]
    debug = {
        "roi_rect": [rx, ry, rw, rh],
        "binary": binary,
        "vertical_line": vertical_line,
        "horizontal_line": horizontal_line,
        "vertical_points": vertical_points,
        "horizontal_points": horizontal_points,
    }
    return center, debug


def compute_precise_center(target, target_size_mm=None, frame=None):
    """Compute one target center, preferring image-fitted center lines."""
    if frame is not None:
        center, debug = center_from_template_frame(
            frame,
            target.get("template_corners"),
        )
        if center is not None:
            return center, "center_lines", debug

    center = center_from_homography_mm_to_pixel(
        target.get("homography_mm_to_pixel"),
        target_size_mm,
    )
    method = "homography_mm_to_pixel"

    if center is None:
        center = center_from_corners(target.get("template_corners"))
        method = "template_corners"

    if center is None:
        center = center_from_corners(target.get("corners"))
        method = "scale_corners"

    return center, method, None


def compute_displacement(initial_center, current_center):
    """Compute pixel displacement from initial center to current center."""
    if initial_center is None or current_center is None:
        return None

    return [
        float(current_center[0] - initial_center[0]),
        float(current_center[1] - initial_center[1]),
    ]


def update_target_center(target, target_size_mm=None, frame=None):
    """Update current center and displacement for one target dictionary."""
    center, method, _ = compute_precise_center(
        target,
        target_size_mm=target_size_mm,
        frame=frame,
    )
    if center is None:
        target["lost"] = True
        target["center_method"] = None
        return target

    if target.get("initial_center") is None:
        target["initial_center"] = center

    target["current_center"] = center
    target["displacement"] = compute_displacement(target["initial_center"], center)
    target["center_method"] = method
    target["lost"] = False
    return target


def update_state_centers(state, frame=None):
    """Update all target centers in a loaded state dictionary."""
    target_size_mm = state.get("target_size_mm")

    for target in state.get("targets", []):
        update_target_center(target, target_size_mm=target_size_mm, frame=frame)

    return state


def draw_center_debug(frame, state):
    """Draw fitted centers and stored template corners on the source frame."""
    output = frame.copy()

    for target in state.get("targets", []):
        target_id = target.get("id", "?")
        corners = target.get("template_corners")
        center = target.get("current_center")

        if corners is not None and len(corners) == 4:
            points = np.asarray(corners, dtype=np.int32)
            cv2.polylines(output, [points], True, (0, 255, 0), 2, cv2.LINE_AA)

        if center is None:
            continue

        cx, cy = [int(round(value)) for value in center]
        cv2.drawMarker(
            output,
            (cx, cy),
            (0, 0, 255),
            cv2.MARKER_CROSS,
            24,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            output,
            f"ID{target_id} {target.get('center_method')}",
            (cx + 8, max(24, cy - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    return output


def load_state(state_path):
    state_path = Path(state_path)
    if not state_path.exists():
        print("State file does not exist:", state_path)
        return None

    with open(state_path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_state(state_path, state):
    state_path = Path(state_path)
    with open(state_path, "w", encoding="utf-8") as file:
        json.dump(state, file, ensure_ascii=False, indent=2)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Compute precise target centers from saved corners or homography.",
    )
    parser.add_argument(
        "--state",
        default=str(DEFAULT_STATE_PATH),
        help=f"Target state JSON path. Default: {DEFAULT_STATE_PATH}",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Write updated centers back to the state JSON.",
    )
    parser.add_argument(
        "--image",
        default=None,
        help="Current frame image path. Default: source_image in state JSON.",
    )
    parser.add_argument(
        "--debug-output",
        default=str(DEFAULT_OUTPUT_PATH),
        help=f"Debug image path. Default: {DEFAULT_OUTPUT_PATH}",
    )
    return parser


def main():
    args = build_parser().parse_args()
    state = load_state(args.state)
    if state is None:
        return

    image_path = args.image or state.get("source_image")
    frame = cv_imread(image_path) if image_path is not None else None

    update_state_centers(state, frame=frame)

    for target in state.get("targets", []):
        print(
            f"ID{target.get('id')}: "
            f"center={target.get('current_center')}, "
            f"displacement={target.get('displacement')}, "
            f"method={target.get('center_method')}"
        )

    if args.save:
        save_state(args.state, state)
        print("State saved:", Path(args.state))

    if frame is not None and args.debug_output:
        debug_image = draw_center_debug(frame, state)
        if cv_imwrite(args.debug_output, debug_image):
            print("Debug image saved:", Path(args.debug_output))


if __name__ == "__main__":
    main()
