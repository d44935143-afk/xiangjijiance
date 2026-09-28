import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

import center_calculation


PROJECT_DIR = Path(__file__).resolve().parent.parent
RECOGNITION_DIR = PROJECT_DIR / "mubiaoshibie"
if str(RECOGNITION_DIR) not in sys.path:
    sys.path.insert(0, str(RECOGNITION_DIR))

import homography_scale_2x2


DEFAULT_STATE_PATH = PROJECT_DIR / "targets_state.json"
DEFAULT_DEBUG_DIR = PROJECT_DIR / "mubiaogenzong" / "tracking_debug"
IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}
DEFAULT_EXCLUDED_NAME_PARTS = ("overlay", "boxes", "debug", "tracking")
DEFAULT_TARGET_SIZE_MM = 60.0
DEFAULT_LOST_SCORE_THRESHOLD = 0.8
DEFAULT_LOCAL_MATCH_SCORE_THRESHOLD = 0.6
DEFAULT_REACQUIRE_MAX_CANDIDATES = 30
DEFAULT_OCCUPIED_IOU_THRESHOLD = 0.2
DEFAULT_SCALE_RECT_MAX_FACTOR = 4.0
DEFAULT_SCALE_RECT_STEP = 0.11
_INITIAL_TEMPLATE_CACHE = {}


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


def load_json(path):
    path = Path(path)
    if not path.exists():
        print("State file does not exist:", path)
        return None

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def list_image_frames(frames_dir, frame_pattern, exclude_generated=True):
    frames_dir = Path(frames_dir)
    if not frames_dir.exists():
        print("Frames folder does not exist:", frames_dir)
        return []

    frames = []
    for path in frames_dir.glob(frame_pattern):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if exclude_generated:
            lower_name = path.name.lower()
            if any(part in lower_name for part in DEFAULT_EXCLUDED_NAME_PARTS):
                continue
        frames.append(path)

    return sorted(frames, key=lambda path: path.name)


def rect_from_corners(corners):
    if corners is None or len(corners) != 4:
        return None

    points = np.asarray(corners, dtype=np.float64)
    x1 = float(points[:, 0].min())
    y1 = float(points[:, 1].min())
    x2 = float(points[:, 0].max())
    y2 = float(points[:, 1].max())
    return x1, y1, x2 - x1, y2 - y1


def clip_rect(rect, image_shape):
    x, y, w, h = rect
    image_h, image_w = image_shape[:2]
    x1 = max(0, min(image_w, int(round(x))))
    y1 = max(0, min(image_h, int(round(y))))
    x2 = max(0, min(image_w, int(round(x + w))))
    y2 = max(0, min(image_h, int(round(y + h))))

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2 - x1, y2 - y1


def expand_rect(rect, image_shape, factor):
    x, y, w, h = rect
    factor = max(1.0, float(factor))
    new_w = w * factor
    new_h = h * factor
    new_x = x + w / 2.0 - new_w / 2.0
    new_y = y + h / 2.0 - new_h / 2.0
    return clip_rect((new_x, new_y, new_w, new_h), image_shape)


def centered_rect(center, size, image_shape):
    """Build an image-clipped rectangle with a fixed size around a center."""
    center_x, center_y = [float(value) for value in center]
    width, height = [float(value) for value in size]
    return clip_rect(
        (
            center_x - width / 2.0,
            center_y - height / 2.0,
            width,
            height,
        ),
        image_shape,
    )


def build_expand_factors(max_factor, step):
    """Build gradual rectangle expansion factors, including the maximum."""
    max_factor = max(1.0, float(max_factor))
    step = max(0.01, float(step))
    factors = []
    current = 1.0

    while current < max_factor:
        factors.append(round(current, 6))
        current += step

    if not factors or abs(factors[-1] - max_factor) > 1e-6:
        factors.append(max_factor)
    return factors


def crop_rect(image, rect):
    clipped = clip_rect(rect, image.shape)
    if clipped is None:
        return None, None

    x, y, w, h = clipped
    return image[y:y + h, x:x + w].copy(), clipped


def rect_to_corners(rect):
    x, y, w, h = rect
    return [
        [float(x), float(y)],
        [float(x + w), float(y)],
        [float(x + w), float(y + h)],
        [float(x), float(y + h)],
    ]


def rect_iou(rect_a, rect_b):
    ax, ay, aw, ah = rect_a
    bx, by, bw, bh = rect_b

    ax2 = ax + aw
    ay2 = ay + ah
    bx2 = bx + bw
    by2 = by + bh

    ix1 = max(ax, bx)
    iy1 = max(ay, by)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    intersection = float((ix2 - ix1) * (iy2 - iy1))
    area_a = float(max(0.0, aw) * max(0.0, ah))
    area_b = float(max(0.0, bw) * max(0.0, bh))
    union = area_a + area_b - intersection
    if union <= 0:
        return 0.0

    return intersection / union


def point_in_rect(point, rect):
    x, y = point
    rx, ry, rw, rh = rect
    return rx <= x <= rx + rw and ry <= y <= ry + rh


def build_occupied_rects(targets, current_target):
    """First-frame manual ROI rects of other currently tracked targets."""
    current_id = current_target.get("id")
    occupied_rects = []

    for target in targets:
        if target.get("id") == current_id:
            continue
        if target.get("lost", False):
            continue

        rect = target.get("manual_roi_rect")
        if rect is not None:
            occupied_rects.append(tuple(rect))

    return occupied_rects


def candidate_hits_occupied_target(
    candidate,
    occupied_rects,
    iou_threshold=DEFAULT_OCCUPIED_IOU_THRESHOLD,
):
    candidate_rect = candidate.get("rect")
    candidate_center = candidate.get("center")
    if candidate_rect is None or candidate_center is None:
        return False

    for occupied_rect in occupied_rects:
        if rect_iou(candidate_rect, occupied_rect) > iou_threshold:
            return True
        if point_in_rect(candidate_center, occupied_rect):
            return True

    return False


def rotate_template(image, angle):
    """Rotate template by a right angle."""
    angle = int(angle) % 360
    if angle == 0:
        return image
    if angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if angle == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image


def build_initial_scaled_template(target, original_template_path=None):
    """Build the fixed first-frame template size saved for this target."""
    scaled_template = target.get("scaled_template", {})
    template_path = original_template_path or target.get("original_template")
    if template_path is None:
        return load_manual_roi_template(target)

    template_size = scaled_template.get("template_size")
    template_rotation = int(scaled_template.get("template_rotation", 0))
    template_scale = scaled_template.get("template_scale")
    cache_key = (
        str(Path(template_path)),
        tuple(template_size) if template_size is not None else None,
        template_rotation,
        None if template_scale is None else float(template_scale),
    )
    if cache_key in _INITIAL_TEMPLATE_CACHE:
        return _INITIAL_TEMPLATE_CACHE[cache_key]

    template_image = cv_imread(template_path)
    if template_image is None:
        return load_manual_roi_template(target)

    template_image = rotate_template(template_image, template_rotation)
    if template_size is not None and len(template_size) == 2:
        target_w, target_h = [int(round(value)) for value in template_size]
    elif template_scale is not None:
        target_w = int(round(template_image.shape[1] * float(template_scale)))
        target_h = int(round(template_image.shape[0] * float(template_scale)))
    else:
        target_w, target_h = template_image.shape[1], template_image.shape[0]

    if target_w <= 0 or target_h <= 0:
        return None

    if target_w != template_image.shape[1] or target_h != template_image.shape[0]:
        template_image = cv2.resize(
            template_image,
            (target_w, target_h),
            interpolation=cv2.INTER_AREA,
        )

    _INITIAL_TEMPLATE_CACHE[cache_key] = template_image
    return template_image


def match_template_in_search_rect(next_frame, template, search_rect):
    """Match one current-frame target template in the next-frame search area."""
    search_crop, clipped_search = crop_rect(next_frame, search_rect)
    if search_crop is None:
        return None

    template_h, template_w = template.shape[:2]
    search_h, search_w = search_crop.shape[:2]
    if template_w > search_w or template_h > search_h:
        return None

    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    search_gray = cv2.cvtColor(search_crop, cv2.COLOR_BGR2GRAY)
    result = cv2.matchTemplate(search_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    _, score, _, location = cv2.minMaxLoc(result)

    sx, sy, _, _ = clipped_search
    match_x = sx + location[0]
    match_y = sy + location[1]
    match_rect = [int(match_x), int(match_y), int(template_w), int(template_h)]
    return {
        "rect": match_rect,
        "corners": rect_to_corners(match_rect),
        "score": float(score),
        "search_rect": list(clipped_search),
    }


def match_template_candidates_in_search_rect(
    next_frame,
    template,
    search_rect,
    score_threshold,
    max_candidates=DEFAULT_REACQUIRE_MAX_CANDIDATES,
):
    """Return high-score template candidates in one search area."""
    search_crop, clipped_search = crop_rect(next_frame, search_rect)
    if search_crop is None:
        return []

    template_h, template_w = template.shape[:2]
    search_h, search_w = search_crop.shape[:2]
    if template_w > search_w or template_h > search_h:
        return []

    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    search_gray = cv2.cvtColor(search_crop, cv2.COLOR_BGR2GRAY)
    result = cv2.matchTemplate(search_gray, template_gray, cv2.TM_CCOEFF_NORMED)

    ys, xs = np.where(result >= float(score_threshold))
    if len(xs) == 0:
        return []

    sx, sy, _, _ = clipped_search
    candidates = []
    order = np.argsort(result[ys, xs])[::-1]
    min_distance = max(4.0, min(template_w, template_h) * 0.5)

    for index in order:
        x = int(xs[index])
        y = int(ys[index])
        score = float(result[y, x])
        match_x = sx + x
        match_y = sy + y
        center = np.array(
            [match_x + template_w / 2.0, match_y + template_h / 2.0],
            dtype=np.float64,
        )

        duplicate = False
        for candidate in candidates:
            candidate_center = np.asarray(candidate["center"], dtype=np.float64)
            if np.linalg.norm(center - candidate_center) < min_distance:
                duplicate = True
                break
        if duplicate:
            continue

        match_rect = [int(match_x), int(match_y), int(template_w), int(template_h)]
        candidates.append(
            {
                "rect": match_rect,
                "corners": rect_to_corners(match_rect),
                "center": [float(center[0]), float(center[1])],
                "score": score,
                "search_rect": list(clipped_search),
            }
        )
        if len(candidates) >= max_candidates:
            break

    return candidates


def distance_to_target_center(candidate, target):
    center = target.get("current_center") or target.get("initial_center")
    if center is None:
        return 0.0

    candidate_center = np.asarray(candidate["center"], dtype=np.float64)
    target_center = np.asarray(center, dtype=np.float64)
    return float(np.linalg.norm(candidate_center - target_center))


def load_manual_roi_template(target):
    """Load the first-frame manual ROI image saved during initialization."""
    template_path = target.get("manual_roi_template_path")
    if template_path is None:
        return None

    return cv_imread(template_path)


def update_scale_from_current_match(
    frame,
    target,
    match,
    target_size_mm=DEFAULT_TARGET_SIZE_MM,
    fallback_max_factor=DEFAULT_SCALE_RECT_MAX_FACTOR,
    fallback_step=DEFAULT_SCALE_RECT_STEP,
):
    """Refresh mm/px around the match center before displacement conversion."""
    match_rect = match.get("rect")
    manual_roi_rect = target.get("manual_roi_rect")
    if match_rect is None or manual_roi_rect is None or len(manual_roi_rect) != 4:
        target["scale_updated"] = False
        target["scale_update_reason"] = "missing_match_or_manual_roi_rect"
        return False

    match_x, match_y, match_w, match_h = [float(value) for value in match_rect]
    match_center = [match_x + match_w / 2.0, match_y + match_h / 2.0]
    initial_roi_size = [manual_roi_rect[2], manual_roi_rect[3]]
    scale_rect = centered_rect(match_center, initial_roi_size, frame.shape)

    scale_result = None
    if scale_rect is not None:
        scale_result = homography_scale_2x2.analyze_2x2_roi(
            image=frame,
            roi_rect=scale_rect,
            target_size_mm=target_size_mm,
        )
        if scale_result is not None:
            scale_result["source"] = "current_match_center_initial_roi_size"
            scale_result["rect"] = list(scale_rect)

    if scale_result is None:
        last_scale_result = None
        for factor in build_expand_factors(fallback_max_factor, fallback_step):
            expanded_rect = expand_rect(match_rect, frame.shape, factor)
            if expanded_rect is None:
                continue

            candidate = homography_scale_2x2.analyze_2x2_roi(
                image=frame,
                roi_rect=expanded_rect,
                target_size_mm=target_size_mm,
            )
            if candidate is None:
                continue

            candidate["source"] = "expanded_current_template_match_rect"
            candidate["rect"] = list(expanded_rect)
            candidate["expand_factor"] = float(factor)
            last_scale_result = candidate

        scale_result = last_scale_result

    if scale_result is None:
        target["scale_updated"] = False
        target["scale_update_reason"] = "corners_not_found"
        return False

    target["mm_per_pixel"] = float(scale_result["mm_per_pixel"])
    target["corners"] = homography_scale_2x2.corners_to_list(
        scale_result["corners"]
    )
    target["homography_pixel_to_mm"] = homography_scale_2x2.matrix_to_list(
        scale_result["homography_pixel_to_mm"]
    )
    target["homography_mm_to_pixel"] = homography_scale_2x2.matrix_to_list(
        scale_result["homography_mm_to_pixel"]
    )
    target["scale_source"] = scale_result["source"]
    target["scale_rect"] = scale_result["rect"]
    target["scale_expand_factor"] = scale_result.get("expand_factor")
    target["scale_updated"] = True
    target["scale_update_reason"] = None
    return True


def reacquire_target_full_frame(
    next_frame,
    target,
    score_threshold,
    original_template_path=None,
    occupied_rects=None,
):
    """Search the whole frame with the first-frame manual ROI template."""
    template = load_manual_roi_template(target)
    if template is None:
        return None

    image_h, image_w = next_frame.shape[:2]
    candidates = match_template_candidates_in_search_rect(
        next_frame,
        template,
        (0, 0, image_w, image_h),
        score_threshold,
    )
    if not candidates:
        return None

    if occupied_rects:
        candidates = [
            candidate
            for candidate in candidates
            if not candidate_hits_occupied_target(candidate, occupied_rects)
        ]
        if not candidates:
            return None

    best = min(candidates, key=lambda item: distance_to_target_center(item, target))
    best["distance_to_last_center_px"] = distance_to_target_center(best, target)
    return best


def apply_match_to_target(next_frame, target, match):
    """Update target position only after the match is accepted."""
    target["template_corners"] = match["corners"]
    target["current_center"] = center_calculation.center_from_corners(
        target["template_corners"]
    )
    center, method = update_center_from_frame(next_frame, target)
    if center is not None:
        target["current_center"] = center
        target["center_method"] = method
    else:
        target["center_method"] = "template_corners"

    target["displacement"] = center_calculation.compute_displacement(
        target.get("initial_center"),
        target.get("current_center"),
    )
    update_scale_from_current_match(
        frame=next_frame,
        target=target,
        match=match,
    )
    target["lost"] = False
    target["lost_reason"] = None
    target["tracking_score"] = match["score"]
    return target


def keep_last_position_as_lost(target, score=None, reason="low_tracking_score"):
    """Mark lost without overwriting the last valid position."""
    target["lost"] = True
    target["lost_reason"] = reason
    target["scale_updated"] = False
    target["scale_update_reason"] = "tracking_lost"
    if score is not None:
        target["tracking_score"] = float(score)
    return target


def get_mm_per_pixel(target, target_size_mm=DEFAULT_TARGET_SIZE_MM):
    """Return saved mm/px scale, or recompute it from 2x2 scale corners."""
    mm_per_pixel = target.get("mm_per_pixel")
    if mm_per_pixel is not None:
        return float(mm_per_pixel)

    corners = target.get("corners")
    if corners is None:
        return None

    points = np.asarray(corners, dtype=np.float32)
    top = float(np.linalg.norm(points[0] - points[1]))
    right = float(np.linalg.norm(points[1] - points[2]))
    bottom = float(np.linalg.norm(points[2] - points[3]))
    left = float(np.linalg.norm(points[3] - points[0]))
    average_side_px = (top + right + bottom + left) / 4.0
    if average_side_px <= 0:
        return None

    scale = float(target_size_mm) / average_side_px
    target["mm_per_pixel"] = scale
    return scale


def displacement_mm(target, target_size_mm=DEFAULT_TARGET_SIZE_MM):
    """Convert pixel displacement to millimeter displacement."""
    displacement = target.get("displacement")
    mm_per_pixel = get_mm_per_pixel(target, target_size_mm=target_size_mm)
    if displacement is None:
        return None

    if mm_per_pixel is None:
        if abs(displacement[0]) < 1e-9 and abs(displacement[1]) < 1e-9:
            return [0.0, 0.0]
        return None

    return [
        float(displacement[0] * mm_per_pixel),
        float(displacement[1] * mm_per_pixel),
    ]


def update_all_displacement_mm(state):
    """Update millimeter displacement for all targets in state."""
    target_size_mm = state.get("target_size_mm", DEFAULT_TARGET_SIZE_MM)
    for target in state.get("targets", []):
        target["displacement_mm"] = displacement_mm(
            target,
            target_size_mm=target_size_mm,
        )


def initialize_first_frame(state, frame):
    """Compute first-frame precise centers as fixed initial centers."""
    for target in state.get("targets", []):
        center, method = update_center_from_frame(frame, target)
        if center is None:
            continue

        target["initial_center"] = center
        target["current_center"] = center
        target["displacement"] = [0.0, 0.0]
        target["center_method"] = method
        target["lost"] = False

    update_all_displacement_mm(state)
    return state


def update_one_target(
    prev_frame,
    next_frame,
    target,
    search_expand_factor,
    lost_score_threshold=DEFAULT_LOST_SCORE_THRESHOLD,
    local_score_threshold=DEFAULT_LOCAL_MATCH_SCORE_THRESHOLD,
    original_template_path=None,
    occupied_rects=None,
):
    """Track one target with the fixed initial scaled template."""
    current_rect = rect_from_corners(target.get("template_corners"))
    if current_rect is None:
        reacquired = reacquire_target_full_frame(
            next_frame,
            target,
            lost_score_threshold,
            original_template_path,
            occupied_rects=occupied_rects,
        )
        if reacquired is not None:
            target["reacquire_score"] = reacquired["score"]
            target["reacquire_distance_px"] = reacquired.get(
                "distance_to_last_center_px"
            )
            return apply_match_to_target(next_frame, target, reacquired)
        return keep_last_position_as_lost(target, reason="missing_template_corners")

    clipped_current_rect = clip_rect(current_rect, next_frame.shape)
    template = build_initial_scaled_template(target, original_template_path)
    if template is None:
        reacquired = reacquire_target_full_frame(
            next_frame,
            target,
            lost_score_threshold,
            original_template_path,
            occupied_rects=occupied_rects,
        )
        if reacquired is not None:
            target["reacquire_score"] = reacquired["score"]
            target["reacquire_distance_px"] = reacquired.get(
                "distance_to_last_center_px"
            )
            return apply_match_to_target(next_frame, target, reacquired)
        return keep_last_position_as_lost(target, reason="missing_initial_template")
    if clipped_current_rect is None:
        reacquired = reacquire_target_full_frame(
            next_frame,
            target,
            lost_score_threshold,
            original_template_path,
            occupied_rects=occupied_rects,
        )
        if reacquired is not None:
            target["reacquire_score"] = reacquired["score"]
            target["reacquire_distance_px"] = reacquired.get(
                "distance_to_last_center_px"
            )
            return apply_match_to_target(next_frame, target, reacquired)
        return keep_last_position_as_lost(target, reason="current_rect_outside_frame")

    search_rect = expand_rect(
        clipped_current_rect,
        next_frame.shape,
        search_expand_factor,
    )
    if search_rect is None:
        reacquired = reacquire_target_full_frame(
            next_frame,
            target,
            lost_score_threshold,
            original_template_path,
            occupied_rects=occupied_rects,
        )
        if reacquired is not None:
            target["reacquire_score"] = reacquired["score"]
            target["reacquire_distance_px"] = reacquired.get(
                "distance_to_last_center_px"
            )
            return apply_match_to_target(next_frame, target, reacquired)
        return keep_last_position_as_lost(target, reason="invalid_search_rect")

    match = match_template_in_search_rect(next_frame, template, search_rect)
    if match is None:
        reacquired = reacquire_target_full_frame(
            next_frame,
            target,
            lost_score_threshold,
            original_template_path,
            occupied_rects=occupied_rects,
        )
        if reacquired is not None:
            target["reacquire_score"] = reacquired["score"]
            target["reacquire_distance_px"] = reacquired.get(
                "distance_to_last_center_px"
            )
            return apply_match_to_target(next_frame, target, reacquired)
        return keep_last_position_as_lost(target, reason="template_match_failed")

    if match["score"] < local_score_threshold:
        reacquired = reacquire_target_full_frame(
            next_frame,
            target,
            lost_score_threshold,
            original_template_path,
            occupied_rects=occupied_rects,
        )
        if reacquired is not None:
            target["reacquire_score"] = reacquired["score"]
            target["reacquire_distance_px"] = reacquired.get(
                "distance_to_last_center_px"
            )
            return apply_match_to_target(next_frame, target, reacquired)
        return keep_last_position_as_lost(
            target,
            score=match["score"],
            reason="low_tracking_score",
        )

    target["reacquire_score"] = None
    target["reacquire_distance_px"] = None
    return apply_match_to_target(next_frame, target, match)


def update_targets_between_frames(
    prev_frame,
    next_frame,
    state,
    search_expand_factor,
    frame_index=None,
    lost_score_threshold=DEFAULT_LOST_SCORE_THRESHOLD,
    local_score_threshold=DEFAULT_LOCAL_MATCH_SCORE_THRESHOLD,
):
    """Track all targets from previous frame to next frame and update state."""
    original_template_path = state.get("original_template")
    targets = state.get("targets", [])
    for target in targets:
        occupied_rects = build_occupied_rects(targets, target)
        update_one_target(
            prev_frame,
            next_frame,
            target,
            search_expand_factor,
            lost_score_threshold=lost_score_threshold,
            local_score_threshold=local_score_threshold,
            original_template_path=original_template_path,
            occupied_rects=occupied_rects,
        )

    update_all_displacement_mm(state)
    if frame_index is not None:
        state["frame_index"] = frame_index
    state["last_update_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    return state


def update_center_from_frame(frame, target):
    """Use center-line fitting in the current frame, with corner center fallback."""
    center, _ = center_calculation.center_from_template_frame(
        frame,
        target.get("template_corners"),
    )
    if center is not None:
        return center, "center_lines"

    center = center_calculation.center_from_corners(target.get("template_corners"))
    if center is not None:
        return center, "template_corners"

    return None, None


def draw_tracking(frame, targets, frame_name):
    output = frame.copy()

    for target in targets:
        color = (0, 255, 0) if not target.get("lost", False) else (0, 0, 255)
        corners = target.get("template_corners")
        center = target.get("current_center")
        target_id = target.get("id", "?")

        if corners is not None and len(corners) == 4:
            points = np.asarray(corners, dtype=np.int32)
            cv2.polylines(output, [points], True, color, 2, cv2.LINE_AA)

        if center is not None:
            cx, cy = [int(round(value)) for value in center]
            cv2.drawMarker(
                output,
                (cx, cy),
                color,
                cv2.MARKER_CROSS,
                20,
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                output,
                f"ID{target_id} S={target.get('tracking_score', 0):.2f}",
                (cx + 8, max(24, cy - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2,
                cv2.LINE_AA,
            )

    cv2.putText(
        output,
        frame_name,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return output


def track_frames(
    state,
    frames,
    search_expand_factor,
    debug_dir=None,
    lost_score_threshold=DEFAULT_LOST_SCORE_THRESHOLD,
    local_score_threshold=DEFAULT_LOCAL_MATCH_SCORE_THRESHOLD,
):
    if len(frames) < 2:
        print("Need at least two frames for tracking.")
        return state

    source_image = Path(state.get("source_image", ""))
    start_index = 0
    for index, frame_path in enumerate(frames):
        if frame_path.resolve() == source_image.resolve():
            start_index = index
            break

    prev_frame = cv_imread(frames[start_index])
    if prev_frame is None:
        return state

    initialize_first_frame(state, prev_frame)

    state["tracked_frames"] = [str(frames[start_index])]

    for frame_path in frames[start_index + 1:]:
        next_frame = cv_imread(frame_path)
        if next_frame is None:
            continue

        original_template_path = state.get("original_template")
        targets = state.get("targets", [])
        for target in targets:
            occupied_rects = build_occupied_rects(targets, target)
            update_one_target(
                prev_frame,
                next_frame,
                target,
                search_expand_factor,
                lost_score_threshold=lost_score_threshold,
                local_score_threshold=local_score_threshold,
                original_template_path=original_template_path,
                occupied_rects=occupied_rects,
            )
        update_all_displacement_mm(state)

        state["tracked_frames"].append(str(frame_path))
        print(f"Tracked frame: {frame_path}")

        if debug_dir is not None:
            debug_image = draw_tracking(next_frame, state.get("targets", []), frame_path.name)
            debug_path = Path(debug_dir) / f"{frame_path.stem}_tracking.png"
            cv_imwrite(debug_path, debug_image)

        prev_frame = next_frame

    return state


def build_parser():
    parser = argparse.ArgumentParser(
        description="Continuously track targets by template matching in expanded search areas.",
    )
    parser.add_argument(
        "--state",
        default=str(DEFAULT_STATE_PATH),
        help=f"Initial target state path. Default: {DEFAULT_STATE_PATH}",
    )
    parser.add_argument(
        "--frames-dir",
        default=None,
        help="Folder containing frame images. Default: folder of source_image.",
    )
    parser.add_argument(
        "--frame-pattern",
        default="*",
        help='Image filename pattern in frames-dir. Example: "frame_*.png".',
    )
    parser.add_argument(
        "--search-expand-factor",
        type=float,
        default=2.5,
        help="Expand current template range by this factor in the next frame.",
    )
    parser.add_argument(
        "--lost-score-threshold",
        type=float,
        default=DEFAULT_LOST_SCORE_THRESHOLD,
        help=(
            "Full-frame reacquisition score below this value is treated as lost. "
            f"Default: {DEFAULT_LOST_SCORE_THRESHOLD}"
        ),
    )
    parser.add_argument(
        "--local-score-threshold",
        type=float,
        default=DEFAULT_LOCAL_MATCH_SCORE_THRESHOLD,
        help=(
            "Local template match score below this value triggers reacquisition. "
            f"Default: {DEFAULT_LOCAL_MATCH_SCORE_THRESHOLD}"
        ),
    )
    parser.add_argument(
        "--debug-dir",
        default=str(DEFAULT_DEBUG_DIR),
        help=f"Folder for drawn tracking frames. Default: {DEFAULT_DEBUG_DIR}",
    )
    parser.add_argument(
        "--no-debug",
        action="store_true",
        help="Do not save drawn tracking frames.",
    )
    parser.add_argument(
        "--include-generated-images",
        action="store_true",
        help="Allow generated result images such as overlay/debug/boxes/tracking.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    state = load_json(args.state)
    if state is None:
        return

    frames_dir = args.frames_dir
    if frames_dir is None:
        source_image = state.get("source_image")
        if source_image is None:
            print("No frames folder and no source_image in state.")
            return
        frames_dir = Path(source_image).parent

    frames = list_image_frames(
        frames_dir,
        args.frame_pattern,
        exclude_generated=not args.include_generated_images,
    )
    if not frames:
        return

    debug_dir = None if args.no_debug else args.debug_dir
    track_frames(
        state=state,
        frames=frames,
        search_expand_factor=args.search_expand_factor,
        debug_dir=debug_dir,
        lost_score_threshold=args.lost_score_threshold,
        local_score_threshold=args.local_score_threshold,
    )
    save_json(args.state, state)
    print("Tracking state saved:", Path(args.state))


if __name__ == "__main__":
    main()
