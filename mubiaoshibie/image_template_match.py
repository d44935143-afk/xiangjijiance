import argparse
import json
from pathlib import Path

import cv2
import numpy as np

import homography_scale_2x2


BASE_DIR = Path(__file__).resolve().parent

# Initial frame image. If no image is passed on the command line, this is used.
DEFAULT_IMAGE_PATH = BASE_DIR / "1.jpg"

# Manual target-state JSON is saved next to test_main in the project root.
DEFAULT_STATE_PATH = BASE_DIR.parent / "targets_state.json"
DEFAULT_CORNERS_OUTPUT_PATH = BASE_DIR / "initial_corners_overlay.png"
DEFAULT_FIRST_FRAME_OUTPUT_PATH = BASE_DIR / "first_frame.png"
DEFAULT_INITIAL_ROI_TEMPLATE_DIR = BASE_DIR / "initial_roi_templates"
DEFAULT_TEMPLATE_PATH = BASE_DIR.parent / "moban" / "checkerboard_2x2_3cm.png"
DEFAULT_TEMPLATE_SCALES = "auto"
DEFAULT_TEMPLATE_ROTATIONS = "0,90"
DEFAULT_SCALE_COUNT = 25
DEFAULT_FALLBACK_SCALE_RECT_FACTOR = 4.0
DEFAULT_FALLBACK_SCALE_RECT_STEP = 0.11
MIN_TEMPLATE_SIZE_PX = 5
TARGET_SIZE_MM = homography_scale_2x2.TARGET_SIZE_MM
SQUARE_SIZE_MM = homography_scale_2x2.SQUARE_SIZE_MM


def cv_imread(path):
    """Read an image. This np.fromfile path also supports Chinese file paths."""
    path = Path(path)
    if not path.exists():
        print("File does not exist:", path)
        return None

    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def clip_rect(x, y, w, h, image_shape):
    """Keep one selected rectangle inside the image."""
    image_h, image_w = image_shape[:2]
    x1 = max(0, min(image_w, int(round(x))))
    y1 = max(0, min(image_h, int(round(y))))
    x2 = max(0, min(image_w, int(round(x + w))))
    y2 = max(0, min(image_h, int(round(y + h))))

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2 - x1, y2 - y1


def expand_rect(rect, image_shape, factor):
    """Expand one rectangle from its center and keep it inside the image."""
    x, y, w, h = rect
    factor = max(1.0, float(factor))
    new_w = w * factor
    new_h = h * factor
    new_x = x + w / 2.0 - new_w / 2.0
    new_y = y + h / 2.0 - new_h / 2.0
    return clip_rect(new_x, new_y, new_w, new_h, image_shape)


def build_expand_factors(max_factor, step):
    """Build gradual expansion factors and include the final max factor."""
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


def to_gray(image):
    """Convert BGR image to grayscale for template matching."""
    if len(image.shape) == 2:
        return image.copy()
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def parse_scales(text):
    """Parse template scales from command line text."""
    if text.strip().lower() == "auto":
        return None
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def parse_rotations(text):
    """Parse template rotation angles. Only 0, 90, 180 and 270 are supported."""
    rotations = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        angle = int(float(item)) % 360
        if angle not in (0, 90, 180, 270):
            raise ValueError("Only 0, 90, 180 and 270 degree rotations are supported.")
        rotations.append(angle)
    return rotations


def rotate_template(image, angle):
    """Rotate template by a right angle."""
    if angle == 0:
        return image
    if angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if angle == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError("Unsupported rotation angle.")


def build_auto_scales(rect, template_image, scale_count):
    """Build scale candidates for one ROI according to that ROI size."""
    _, _, roi_w, roi_h = rect
    template_h, template_w = template_image.shape[:2]

    min_scale = max(
        MIN_TEMPLATE_SIZE_PX / template_w,
        MIN_TEMPLATE_SIZE_PX / template_h,
    )
    max_scale = min(
        roi_w / template_w,
        roi_h / template_h,
    )

    if max_scale < min_scale:
        return []

    if scale_count <= 1:
        return [float(max_scale)]

    return [
        float(value)
        for value in np.linspace(min_scale, max_scale, scale_count)
    ]


def select_target_rects(image):
    """Manually select all target rectangles and keep confirmed boxes visible."""
    window_name = "Draw ROIs: drag mouse, Enter/Space save, u undo, Esc/q cancel"
    rects = []
    drawing = {
        "active": False,
        "start": None,
        "current": None,
        "cancelled": False,
    }

    def draw_view():
        view = image.copy()

        for index, rect in enumerate(rects, start=1):
            x, y, w, h = rect
            cv2.rectangle(view, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                view,
                f"ID{index}",
                (x, max(24, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        if drawing["active"] and drawing["start"] and drawing["current"]:
            x0, y0 = drawing["start"]
            x1, y1 = drawing["current"]
            cv2.rectangle(view, (x0, y0), (x1, y1), (0, 255, 255), 1)

        return view

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing["active"] = True
            drawing["start"] = (x, y)
            drawing["current"] = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and drawing["active"]:
            drawing["current"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and drawing["active"]:
            drawing["active"] = False
            x0, y0 = drawing["start"]
            x1, y1 = x, y

            left = min(x0, x1)
            top = min(y0, y1)
            width = abs(x1 - x0)
            height = abs(y1 - y0)

            clipped = clip_rect(left, top, width, height, image.shape)
            if clipped is not None:
                rects.append(clipped)

            drawing["start"] = None
            drawing["current"] = None

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1200, 700)
    cv2.setMouseCallback(window_name, on_mouse)

    while True:
        cv2.imshow(window_name, draw_view())
        key = cv2.waitKey(20) & 0xFF

        if key in (13, 32):
            break
        if key in (ord("q"), 27):
            drawing["cancelled"] = True
            rects.clear()
            break
        if key == ord("u") and rects:
            rects.pop()

    cv2.setMouseCallback(window_name, lambda *args: None)
    cv2.destroyWindow(window_name)

    return rects


def match_scaled_template_in_roi(
    image,
    rect,
    template_image,
    scales,
    scale_count,
    rotations,
):
    """Scale/rotate the original template and find its best match inside one ROI."""
    x, y, w, h = rect
    roi = image[y:y + h, x:x + w].copy()
    roi_gray = to_gray(roi)
    scale_candidates = scales

    if scale_candidates is None:
        scale_candidates = build_auto_scales(rect, template_image, scale_count)

    best = None
    for rotation in rotations:
        rotated_template_gray = to_gray(rotate_template(template_image, rotation))

        for scale in scale_candidates:
            template_w = int(round(rotated_template_gray.shape[1] * scale))
            template_h = int(round(rotated_template_gray.shape[0] * scale))

            if template_w < 5 or template_h < 5:
                continue
            if template_w > w or template_h > h:
                continue

            template_gray = cv2.resize(
                rotated_template_gray,
                (template_w, template_h),
                interpolation=cv2.INTER_AREA,
            )
            result = cv2.matchTemplate(
                roi_gray,
                template_gray,
                cv2.TM_CCOEFF_NORMED,
            )
            _, score, _, location = cv2.minMaxLoc(result)

            if best is None or score > best["match_score"]:
                match_x = x + location[0]
                match_y = y + location[1]
                best = {
                    "template_scale": float(scale),
                    "template_rotation": int(rotation),
                    "template_size": [int(template_w), int(template_h)],
                    "match_rect": [
                        int(match_x),
                        int(match_y),
                        int(template_w),
                        int(template_h),
                    ],
                    "match_center": [
                        match_x + template_w / 2.0,
                        match_y + template_h / 2.0,
                    ],
                    "match_score": float(score),
                }

    return best


def draw_corner_overlay(image, target_overlays):
    """Draw manual ROIs and detected 2x2 corners on the source image."""
    output = image.copy()

    for item in target_overlays:
        target_id = item["id"]
        roi_rect = item["roi_rect"]
        corners = item["corners"]

        x, y, w, h = roi_rect
        cv2.rectangle(output, (x, y), (x + w, y + h), (255, 180, 0), 1)
        cv2.putText(
            output,
            f"ROI{target_id}",
            (x, max(24, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 180, 0),
            2,
            cv2.LINE_AA,
        )

        if corners is None:
            cv2.putText(
                output,
                f"ID{target_id} corners not found",
                (x, y + h + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            continue

        corner_int = corners.astype(np.int32)
        cv2.polylines(output, [corner_int], True, (0, 255, 0), 2, cv2.LINE_AA)

        for index, point in enumerate(corners, start=1):
            px, py = int(round(point[0])), int(round(point[1]))
            cv2.circle(output, (px, py), 4, (0, 0, 255), -1)
            cv2.putText(
                output,
                str(index),
                (px + 5, py - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )

    return output


def cv_imwrite(path, image):
    """Write an image. This np.tofile path also supports Chinese file paths."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    success, buffer = cv2.imencode(path.suffix, image)
    if success:
        buffer.tofile(str(path))
    return success


def rect_to_corners(rect):
    """Convert an axis-aligned rectangle to four corner points."""
    x, y, w, h = rect
    return [
        [float(x), float(y)],
        [float(x + w), float(y)],
        [float(x + w), float(y + h)],
        [float(x), float(y + h)],
    ]


def corners_center(corners):
    """Compute the center point from four corners."""
    points = np.asarray(corners, dtype=np.float32)
    center = points.mean(axis=0)
    return [float(center[0]), float(center[1])]


def build_target_record(
    target_id,
    manual_roi_rect,
    manual_roi_template_path,
    template_match,
    scale_result,
):
    """Create one compact target record from the final matched center."""
    template_corners = rect_to_corners(template_match["match_rect"])
    center = corners_center(template_corners)
    scaled_template = {
        "template_scale": template_match["template_scale"],
        "template_rotation": template_match["template_rotation"],
        "template_size": template_match["template_size"],
        "match_score": template_match["match_score"],
    }
    mm_per_pixel = None
    corners = None
    homography_pixel_to_mm = None
    homography_mm_to_pixel = None
    scale_source = None

    if scale_result is not None:
        mm_per_pixel = scale_result["mm_per_pixel"]
        corners = scale_result["corners"]
        homography_pixel_to_mm = scale_result["homography_pixel_to_mm"]
        homography_mm_to_pixel = scale_result["homography_mm_to_pixel"]
        scale_source = scale_result.get("source")

    return {
        "id": target_id,
        "manual_roi_rect": [int(value) for value in manual_roi_rect],
        "manual_roi_template_path": str(Path(manual_roi_template_path)),
        "initial_center": center,
        "current_center": center,
        "template_corners": template_corners,
        "displacement": [0.0, 0.0],
        "mm_per_pixel": mm_per_pixel,
        "scale_source": scale_source,
        "corners": homography_scale_2x2.corners_to_list(corners),
        "homography_pixel_to_mm": homography_scale_2x2.matrix_to_list(
            homography_pixel_to_mm
        ),
        "homography_mm_to_pixel": homography_scale_2x2.matrix_to_list(
            homography_mm_to_pixel
        ),
        "lost": False,
        "scaled_template": scaled_template,
    }


def save_initial_targets(
    image,
    rects,
    state_path,
    source_image_path,
    template_path,
    template_image,
    scales,
    scale_count,
    rotations,
    corners_output_path,
    fallback_scale_rect_factor,
    fallback_scale_rect_step,
):
    """Write manually selected target boxes and initial target state."""
    state_path = Path(state_path)
    state_path.parent.mkdir(parents=True, exist_ok=True)

    targets = []
    target_overlays = []
    for target_id, rect in enumerate(rects, start=1):
        clipped = clip_rect(*rect, image.shape)
        if clipped is None:
            continue

        template_match = match_scaled_template_in_roi(
            image=image,
            rect=clipped,
            template_image=template_image,
            scales=scales,
            scale_count=scale_count,
            rotations=rotations,
        )
        if template_match is None:
            print(f"Target {target_id}: no valid template scale in ROI {clipped}")
            continue

        roi_x, roi_y, roi_w, roi_h = clipped
        manual_roi_template = image[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w].copy()
        manual_roi_template_path = (
            DEFAULT_INITIAL_ROI_TEMPLATE_DIR
            / f"target_{target_id:03d}_manual_roi.png"
        )
        cv_imwrite(manual_roi_template_path, manual_roi_template)

        scale_result = homography_scale_2x2.analyze_2x2_roi(
            image=image,
            roi_rect=clipped,
            target_size_mm=TARGET_SIZE_MM,
        )
        if scale_result is not None:
            scale_result["source"] = "manual_roi"
            scale_result["rect"] = list(clipped)
        else:
            match_rect = template_match["match_rect"]
            last_scale_result = None

            for factor in build_expand_factors(
                fallback_scale_rect_factor,
                fallback_scale_rect_step,
            ):
                expanded_match_rect = expand_rect(
                    match_rect,
                    image.shape,
                    factor,
                )
                if expanded_match_rect is None:
                    continue

                candidate_result = homography_scale_2x2.analyze_2x2_roi(
                    image=image,
                    roi_rect=expanded_match_rect,
                    target_size_mm=TARGET_SIZE_MM,
                )
                if candidate_result is None:
                    continue

                candidate_result["source"] = "expanded_template_match_rect"
                candidate_result["rect"] = list(expanded_match_rect)
                candidate_result["expand_factor"] = float(factor)
                last_scale_result = candidate_result

            scale_result = last_scale_result

        target_overlays.append(
            {
                "id": target_id,
                "roi_rect": (
                    clipped
                    if scale_result is None or scale_result.get("rect") is None
                    else tuple(scale_result["rect"])
                ),
                "corners": None if scale_result is None else scale_result["corners"],
            }
        )

        record = build_target_record(
            target_id,
            clipped,
            manual_roi_template_path,
            template_match,
            scale_result,
        )
        targets.append(record)

        mm_per_pixel = None if scale_result is None else scale_result["mm_per_pixel"]
        mm_text = "None" if mm_per_pixel is None else f"{mm_per_pixel:.6f}"
        print(
            f"Target {target_id}: "
            f"center=({record['initial_center'][0]:.2f}, "
            f"{record['initial_center'][1]:.2f}), "
            f"template_scale={template_match['template_scale']:.2f}, "
            f"template_rotation={template_match['template_rotation']}, "
            f"template_size={template_match['template_size']}, "
            f"match_score={template_match['match_score']:.4f}, "
            f"mm_per_pixel={mm_text}, "
            f"scale_source={record['scale_source']}, "
            f"expand_factor={None if scale_result is None else scale_result.get('expand_factor')}"
        )

    state = {
        "source_image": str(Path(source_image_path)),
        "original_template": str(Path(template_path)),
        "square_size_mm": SQUARE_SIZE_MM,
        "target_size_mm": TARGET_SIZE_MM,
        "target_count": len(targets),
        "targets": targets,
    }

    with open(state_path, "w", encoding="utf-8") as file:
        json.dump(state, file, ensure_ascii=False, indent=2)

    print("Initial target state saved:", state_path)

    corner_overlay = draw_corner_overlay(image, target_overlays)
    if cv_imwrite(corners_output_path, corner_overlay):
        print("Corner overlay saved:", Path(corners_output_path))

    return state


def initialize_targets_from_frame(
    frame,
    state_path=DEFAULT_STATE_PATH,
    template_path=DEFAULT_TEMPLATE_PATH,
    corners_output_path=DEFAULT_CORNERS_OUTPUT_PATH,
    scales_text=DEFAULT_TEMPLATE_SCALES,
    scale_count=DEFAULT_SCALE_COUNT,
    rotations_text=DEFAULT_TEMPLATE_ROTATIONS,
    fallback_scale_rect_factor=DEFAULT_FALLBACK_SCALE_RECT_FACTOR,
    fallback_scale_rect_step=DEFAULT_FALLBACK_SCALE_RECT_STEP,
    source_image_path="camera_first_frame",
    first_frame_output_path=DEFAULT_FIRST_FRAME_OUTPUT_PATH,
):
    """Initialize targets directly from one camera frame."""
    if first_frame_output_path is not None:
        first_frame_output_path = Path(first_frame_output_path)
        if cv_imwrite(first_frame_output_path, frame):
            source_image_path = str(first_frame_output_path)

    template_image = cv_imread(template_path)
    if template_image is None:
        return None

    scales = parse_scales(scales_text)
    if scales is not None and not scales:
        print("No template scales were provided.")
        return None

    try:
        rotations = parse_rotations(rotations_text)
    except ValueError as error:
        print(error)
        return None

    if not rotations:
        print("No template rotations were provided.")
        return None

    rects = select_target_rects(frame)
    if not rects:
        print("No target was selected.")
        return None

    return save_initial_targets(
        image=frame,
        rects=rects,
        state_path=state_path,
        source_image_path=source_image_path,
        template_path=template_path,
        template_image=template_image,
        scales=scales,
        scale_count=scale_count,
        rotations=rotations,
        corners_output_path=corners_output_path,
        fallback_scale_rect_factor=fallback_scale_rect_factor,
        fallback_scale_rect_step=fallback_scale_rect_step,
    )


def build_parser():
    parser = argparse.ArgumentParser(
        description="Manually select targets on the initial frame and save target state.",
    )
    parser.add_argument(
        "image",
        nargs="?",
        default=str(DEFAULT_IMAGE_PATH),
        help=f"Initial frame image path. Default: {DEFAULT_IMAGE_PATH}",
    )
    parser.add_argument(
        "--state",
        default=str(DEFAULT_STATE_PATH),
        help=f"Target state JSON path. Default: {DEFAULT_STATE_PATH}",
    )
    parser.add_argument(
        "--corners-output",
        default=str(DEFAULT_CORNERS_OUTPUT_PATH),
        help=f"Corner overlay image path. Default: {DEFAULT_CORNERS_OUTPUT_PATH}",
    )
    parser.add_argument(
        "--template",
        default=str(DEFAULT_TEMPLATE_PATH),
        help=f"Original template image path. Default: {DEFAULT_TEMPLATE_PATH}",
    )
    parser.add_argument(
        "--scales",
        default=DEFAULT_TEMPLATE_SCALES,
        help=(
            "Template scales to try, or auto to build scales per ROI. "
            f"Default: {DEFAULT_TEMPLATE_SCALES}"
        ),
    )
    parser.add_argument(
        "--scale-count",
        type=int,
        default=DEFAULT_SCALE_COUNT,
        help=f"Number of auto scale candidates per target. Default: {DEFAULT_SCALE_COUNT}",
    )
    parser.add_argument(
        "--rotations",
        default=DEFAULT_TEMPLATE_ROTATIONS,
        help=f"Template rotations to try. Default: {DEFAULT_TEMPLATE_ROTATIONS}",
    )
    parser.add_argument(
        "--fallback-scale-rect-factor",
        type=float,
        default=DEFAULT_FALLBACK_SCALE_RECT_FACTOR,
        help=(
            "When manual ROI corner detection fails, expand the template matched "
            "rectangle by this factor before detecting 2x2 corners. "
            f"Default: {DEFAULT_FALLBACK_SCALE_RECT_FACTOR}"
        ),
    )
    parser.add_argument(
        "--fallback-scale-rect-step",
        type=float,
        default=DEFAULT_FALLBACK_SCALE_RECT_STEP,
        help=(
            "Expansion step for fallback 2x2 corner detection. The last successful "
            "expanded result is used. "
            f"Default: {DEFAULT_FALLBACK_SCALE_RECT_STEP}"
        ),
    )
    return parser


def main():
    args = build_parser().parse_args()

    image = cv_imread(args.image)
    if image is None:
        return

    template_image = cv_imread(args.template)
    if template_image is None:
        return

    scales = parse_scales(args.scales)
    if scales is not None and not scales:
        print("No template scales were provided.")
        return

    try:
        rotations = parse_rotations(args.rotations)
    except ValueError as error:
        print(error)
        return

    if not rotations:
        print("No template rotations were provided.")
        return

    rects = select_target_rects(image)
    if not rects:
        print("No target was selected.")
        return

    save_initial_targets(
        image=image,
        rects=rects,
        state_path=args.state,
        source_image_path=args.image,
        template_path=args.template,
        template_image=template_image,
        scales=scales,
        scale_count=args.scale_count,
        rotations=rotations,
        corners_output_path=args.corners_output,
        fallback_scale_rect_factor=args.fallback_scale_rect_factor,
        fallback_scale_rect_step=args.fallback_scale_rect_step,
    )


if __name__ == "__main__":
    main()
