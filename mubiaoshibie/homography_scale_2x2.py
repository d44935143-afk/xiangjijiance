import cv2
import numpy as np


# One small checker square is 3 cm. The detected homography corners are the
# outer four corners of the full 2x2 target, so the real side length is 6 cm.
SQUARE_SIZE_MM = 30.0
TARGET_SIZE_MM = SQUARE_SIZE_MM * 2.0


def to_gray(image):
    """Convert BGR image to grayscale."""
    if len(image.shape) == 2:
        return image.copy()
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def order_corners(points):
    """Order four points as top-left, top-right, bottom-right, bottom-left."""
    points = np.asarray(points, dtype=np.float32)
    ordered = np.zeros((4, 2), dtype=np.float32)
    point_sum = points.sum(axis=1)
    point_diff = np.diff(points, axis=1).reshape(-1)

    ordered[0] = points[np.argmin(point_sum)]
    ordered[2] = points[np.argmax(point_sum)]
    ordered[1] = points[np.argmin(point_diff)]
    ordered[3] = points[np.argmax(point_diff)]
    return ordered


def normalize_vector(vector):
    """Return a unit vector, or None for a near-zero vector."""
    length = float(np.linalg.norm(vector))
    if length < 1e-6:
        return None
    return vector / length


def clip_rect(rect, image_shape, margin=0):
    """Clip one ROI rectangle to image bounds."""
    x, y, w, h = [float(value) for value in rect]
    image_h, image_w = image_shape[:2]

    x1 = max(0, min(image_w, int(round(x - margin))))
    y1 = max(0, min(image_h, int(round(y - margin))))
    x2 = max(0, min(image_w, int(round(x + w + margin))))
    y2 = max(0, min(image_h, int(round(y + h + margin))))

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2 - x1, y2 - y1


def black_square_candidates(binary, upscale):
    """Find separate black-square candidates in the selected ROI."""
    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    image_area = binary.shape[0] * binary.shape[1]
    candidates = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < image_area * 0.005:
            continue

        rect = cv2.minAreaRect(contour)
        (_, _), (width, height), _ = rect
        if width <= 1 or height <= 1:
            continue

        long_side = max(width, height)
        short_side = min(width, height)
        if long_side / short_side > 1.8:
            continue

        fill_ratio = area / max(1.0, width * height)
        if fill_ratio < 0.35:
            continue

        box = cv2.boxPoints(rect).astype(np.float32) / float(upscale)
        ordered_box = order_corners(box)
        center = ordered_box.mean(axis=0)
        side = float((width + height) * 0.5 / float(upscale))

        candidates.append(
            {
                "box": ordered_box,
                "center": center,
                "side": side,
                "area": float(area),
            }
        )

    candidates.sort(key=lambda item: item["area"], reverse=True)
    return candidates


def separated_black_square_candidates(binary, upscale):
    """Retry candidate extraction after a light erosion separates touching squares."""
    min_side = min(binary.shape[:2])
    kernel_size = 3
    if min_side >= 180:
        kernel_size = 5

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (kernel_size, kernel_size),
    )
    separated = cv2.erode(binary, kernel, iterations=1)
    return black_square_candidates(separated, upscale)


def infer_2x2_outer_corners(first, second):
    """Infer full 2x2 target outer corners from two diagonal black squares."""
    box = first["box"]
    u = normalize_vector(box[1] - box[0])
    v = normalize_vector(box[3] - box[0])
    if u is None or v is None:
        return None

    delta = second["center"] - first["center"]
    proj_u = float(np.dot(delta, u))
    proj_v = float(np.dot(delta, v))

    if abs(proj_u) < first["side"] * 0.35 or abs(proj_v) < first["side"] * 0.35:
        return None

    if proj_u < 0:
        u = -u
        proj_u = -proj_u
    if proj_v < 0:
        v = -v
        proj_v = -proj_v

    mean_side = (first["side"] + second["side"] + proj_u + proj_v) / 4.0
    if mean_side <= 1.0:
        return None

    center = (first["center"] + second["center"]) * 0.5
    corners = np.array(
        [
            center - u * mean_side - v * mean_side,
            center + u * mean_side - v * mean_side,
            center + u * mean_side + v * mean_side,
            center - u * mean_side + v * mean_side,
        ],
        dtype=np.float32,
    )
    return order_corners(corners)


def detect_2x2_corners_in_roi(image, roi_rect, roi_margin=0):
    """Detect the real 2x2 target outer corners inside a manual ROI."""
    clipped = clip_rect(roi_rect, image.shape, margin=roi_margin)
    if clipped is None:
        return None

    x, y, w, h = clipped
    roi = image[y:y + h, x:x + w].copy()
    gray = to_gray(roi)

    upscale = max(1, int(round(120 / max(1, min(w, h)))))
    if upscale > 1:
        gray = cv2.resize(
            gray,
            (w * upscale, h * upscale),
            interpolation=cv2.INTER_CUBIC,
        )

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )

    candidates = black_square_candidates(binary, upscale)
    if len(candidates) < 2:
        candidates = separated_black_square_candidates(binary, upscale)

    if len(candidates) < 2:
        return None

    best_corners = None
    best_score = None

    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            first = candidates[i]
            second = candidates[j]
            side_diff = abs(first["side"] - second["side"]) / max(
                1.0,
                (first["side"] + second["side"]) * 0.5,
            )
            if side_diff > 0.5:
                continue

            corners = infer_2x2_outer_corners(first, second)
            if corners is None:
                continue

            distance = float(np.linalg.norm(first["center"] - second["center"]))
            expected = ((first["side"] + second["side"]) * 0.5) * np.sqrt(2.0)
            score = side_diff + abs(distance - expected) / max(1.0, expected)

            if best_score is None or score < best_score:
                best_score = score
                best_corners = corners

    if best_corners is None:
        return None

    best_corners[:, 0] += x
    best_corners[:, 1] += y
    return best_corners


def compute_mm_per_pixel(corners, target_size_mm=TARGET_SIZE_MM):
    """Compute one average mm-per-pixel value from detected outer corners."""
    top = float(np.linalg.norm(corners[0] - corners[1]))
    right = float(np.linalg.norm(corners[1] - corners[2]))
    bottom = float(np.linalg.norm(corners[2] - corners[3]))
    left = float(np.linalg.norm(corners[3] - corners[0]))
    average_side_px = (top + right + bottom + left) / 4.0

    if average_side_px <= 0:
        return None

    return target_size_mm / average_side_px


def destination_mm_corners(target_size_mm=TARGET_SIZE_MM):
    """2x2 target destination corners in real millimeter coordinates."""
    size = float(target_size_mm)
    return np.array(
        [
            [0.0, 0.0],
            [size, 0.0],
            [size, size],
            [0.0, size],
        ],
        dtype=np.float32,
    )


def compute_homographies(corners, target_size_mm=TARGET_SIZE_MM):
    """Compute pixel-to-mm and mm-to-pixel homography matrices."""
    src_pixel = corners.astype(np.float32)
    dst_mm = destination_mm_corners(target_size_mm)
    homography_pixel_to_mm = cv2.getPerspectiveTransform(src_pixel, dst_mm)
    homography_mm_to_pixel = cv2.getPerspectiveTransform(dst_mm, src_pixel)
    return homography_pixel_to_mm, homography_mm_to_pixel


def analyze_2x2_roi(image, roi_rect, target_size_mm=TARGET_SIZE_MM, roi_margin=0):
    """Return corners, homographies and mm-per-pixel for a 2x2 target in one ROI."""
    corners = detect_2x2_corners_in_roi(
        image=image,
        roi_rect=roi_rect,
        roi_margin=roi_margin,
    )
    if corners is None:
        return None

    mm_per_pixel = compute_mm_per_pixel(corners, target_size_mm=target_size_mm)
    if mm_per_pixel is None:
        return None

    h_pixel_to_mm, h_mm_to_pixel = compute_homographies(
        corners,
        target_size_mm=target_size_mm,
    )

    return {
        "corners": corners,
        "mm_per_pixel": float(mm_per_pixel),
        "homography_pixel_to_mm": h_pixel_to_mm,
        "homography_mm_to_pixel": h_mm_to_pixel,
    }


def matrix_to_list(matrix):
    """Convert a numpy matrix to JSON-safe list."""
    if matrix is None:
        return None
    return matrix.astype(float).tolist()


def corners_to_list(corners):
    """Convert numpy corner points to JSON-safe list."""
    if corners is None:
        return None
    return [[float(x), float(y)] for x, y in corners]
