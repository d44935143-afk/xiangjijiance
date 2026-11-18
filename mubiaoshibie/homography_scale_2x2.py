import json
from pathlib import Path

import cv2
import numpy as np


# One small checker square is 1 cm. The detected homography corners are the
# outer four corners of the full 2x2 target, so the real side length is 2 cm.
SQUARE_SIZE_MM = 55.0
TARGET_SIZE_MM = SQUARE_SIZE_MM * 2.0
DEBUG_OUTPUT_DIR = None
_DEBUG_ATTEMPT = 0


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


def fit_square_edges(contour, coarse_box):
    """Fit contour sides robustly; return None when edge support is insufficient."""
    points = contour.reshape(-1, 2).astype(np.float32)
    box = order_corners(coarse_box)
    lines = []
    supports = []
    for i in range(4):
        a, b = box[i], box[(i + 1) % 4]
        direction = b - a
        length = float(np.linalg.norm(direction))
        if length < 6:
            return None
        direction /= length
        normal = np.array([-direction[1], direction[0]])
        offset = points - a
        along = offset @ direction
        distance = np.abs(offset @ normal)
        selected = points[(along > length * 0.12) & (along < length * 0.88)
                          & (distance < max(2.0, length * 0.25))]
        if len(selected) < 6:
            return None
        for _ in range(3):
            vx, vy, x0, y0 = cv2.fitLine(selected, cv2.DIST_HUBER, 0, .01, .01).reshape(-1)
            residual = np.abs((selected[:, 0] - x0) * vy - (selected[:, 1] - y0) * vx)
            threshold = max(1.0, 2.5 * float(np.median(residual)))
            keep = selected[residual <= threshold]
            if len(keep) < 6:
                return None
            selected = keep
        vx, vy, x0, y0 = cv2.fitLine(selected, cv2.DIST_HUBER, 0, .01, .01).reshape(-1)
        if abs(float(np.dot([vx, vy], direction))) < .94:
            return None
        if np.ptp(selected @ direction) < length * .4:
            return None
        lines.append((np.array([x0, y0]), np.array([vx, vy])))
        supports.append(int(len(selected)))
    corners = []
    for i in range(4):
        a, u = lines[(i - 1) % 4]
        b, v = lines[i]
        matrix = np.column_stack((u, -v))
        if abs(np.linalg.det(matrix)) < .1:
            return None
        point = a + np.linalg.solve(matrix, b - a)[0] * u
        if not np.isfinite(point).all():
            return None
        corners.append(point)
    fitted = np.asarray(corners, np.float32)
    if not cv2.isContourConvex(fitted.reshape(-1, 1, 2)):
        return None
    if np.max(np.linalg.norm(fitted - box, axis=1)) > np.max(np.linalg.norm(np.roll(box, -1, axis=0)-box,axis=1)) * .35:
        return None
    return fitted, supports


def refine_gray_corners(gray, box):
    """Refine each corner using two short grayscale edge segments beside it."""
    smooth = cv2.GaussianBlur(gray, (3, 3), 0).astype(np.float32)
    output = box.copy()
    diagnostics = []
    for i in range(4):
        origin = box[i]
        lines = []
        edge_points = []
        for neighbor in ((i - 1) % 4, (i + 1) % 4):
            vector = box[neighbor] - origin
            length = float(np.linalg.norm(vector))
            if length < 12:
                break
            u = vector / length
            n = np.array([-u[1], u[0]], np.float32)
            radius = max(3, min(10, int(round(length * .1))))
            offsets = np.arange(-radius, radius + 1, dtype=np.float32)
            points = []
            # Stay away from the corner itself and the midpoint/crossing.
            for along in np.linspace(length * .12, length * .48, max(8, int(length * .36))):
                base = origin + u * along
                samples = (base[None, :] + offsets[:, None] * n).astype(np.float32)
                if (samples[:, 0].min() < 1 or samples[:, 1].min() < 1 or
                        samples[:, 0].max() >= gray.shape[1]-1 or samples[:, 1].max() >= gray.shape[0]-1):
                    continue
                profile = cv2.remap(smooth, samples[:,0].reshape(1,-1),
                                    samples[:,1].reshape(1,-1), cv2.INTER_LINEAR).ravel()
                gradient = np.abs(np.gradient(profile))
                peak = int(np.argmax(gradient))
                if peak == 0 or peak == len(offsets)-1 or gradient[peak] < 5:
                    continue
                a,b,c = [float(v) for v in gradient[peak-1:peak+2]]
                denominator = a-2*b+c
                delta = np.clip(.5*(a-c)/denominator, -.5, .5) if denominator < -1e-8 else 0
                points.append(base + n * (offsets[peak] + delta))
            if len(points) < 6:
                break
            points = np.asarray(points, np.float32)
            vx,vy,x0,y0 = cv2.fitLine(points, cv2.DIST_HUBER, 0,.01,.01).ravel()
            residual = np.abs((points[:,0]-x0)*vy-(points[:,1]-y0)*vx)
            points = points[residual <= max(.75,2.5*float(np.median(residual)))]
            if len(points)<6 or np.ptp(points @ u)<length*.2:
                break
            vx,vy,x0,y0 = cv2.fitLine(points, cv2.DIST_HUBER, 0,.01,.01).ravel()
            if abs(float(np.dot([vx,vy],u)))<.97:
                break
            lines.append((np.array([x0,y0]),np.array([vx,vy])))
            edge_points.append(points.tolist())
        accepted = False
        if len(lines)==2:
            a,u=lines[0]; b,v=lines[1]; matrix=np.column_stack((u,-v))
            if abs(np.linalg.det(matrix))>.2:
                point=a+np.linalg.solve(matrix,b-a)[0]*u
                limit=.15*min(np.linalg.norm(box[(i-1)%4]-origin),np.linalg.norm(box[(i+1)%4]-origin))
                if np.isfinite(point).all() and np.linalg.norm(point-origin)<=limit:
                    output[i]=point
                    accepted=True
        diagnostics.append({'corner':i+1,'accepted':accepted,'edge_points_roi':edge_points})
    if not cv2.isContourConvex(output.reshape(-1,1,2)):
        return box.copy(), [{'accepted':False,'reason':'invalid_refined_polygon'}]
    return output, diagnostics


def black_square_candidates(binary, upscale):
    """Find separate black-square candidates in the selected ROI."""
    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
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
        coarse_box = order_corners(box)
        refined = fit_square_edges(contour.astype(np.float32) / float(upscale), coarse_box)
        if refined is None:
            continue  # Do not accept the protrusion-contaminated coarse rectangle.
        ordered_box = refined[0]
        center = ordered_box.mean(axis=0)
        side = float(np.mean(np.linalg.norm(np.roll(ordered_box, -1, axis=0) - ordered_box, axis=1)))

        candidates.append(
            {
                "box": ordered_box,
                "coarse_box": coarse_box,
                "edge_fit_success": refined is not None,
                "edge_support_counts": refined[1] if refined is not None else [],
                "center": center,
                "side": side,
                "area": float(area),
                "contour": contour.astype(np.float32) / float(upscale),
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

    raw_binary = binary.copy()
    # Remove narrow protrusions/connections before extracting the side contours.
    # Opening preserves long straight edges when the kernel fits inside the square.
    kernel_size = max(3, int(round(min(w, h) * upscale * 0.10)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    candidates = black_square_candidates(binary, upscale)
    if len(candidates) < 2:
        candidates = separated_black_square_candidates(binary, upscale)

    # Refine the two adjacent sides of every candidate corner in original grayscale.
    original_gray = to_gray(roi)
    for item in candidates:
        item['before_gray_refinement'] = item['box'].copy()
        item['box'], item['corner_refinement'] = refine_gray_corners(original_gray, item['box'])
        item['center'] = item['box'].mean(axis=0)
        item['side'] = float(np.mean(np.linalg.norm(np.roll(item['box'],-1,axis=0)-item['box'],axis=1)))

    def save_debug(pair=None, score=None):
        if DEBUG_OUTPUT_DIR is None:
            return
        global _DEBUG_ATTEMPT
        _DEBUG_ATTEMPT += 1
        folder = Path(DEBUG_OUTPUT_DIR) / f"attempt_{_DEBUG_ATTEMPT:04d}"
        folder.mkdir(parents=True, exist_ok=True)
        def write(name, picture):
            ok, data = cv2.imencode(".png", picture)
            if not ok:
                raise RuntimeError(f"Cannot encode debug image: {name}")
            data.tofile(str(folder / name))
        view = cv2.resize(roi, (binary.shape[1], binary.shape[0]))
        candidate_view = view.copy()
        for index, item in enumerate(candidates):
            contour = np.round(item['contour'] * upscale).astype(np.int32)
            coarse = np.round(item['coarse_box'] * upscale).astype(np.int32)
            cv2.polylines(candidate_view, [coarse], True, (255, 0, 0), 1)
            box = np.round(item['box'] * upscale).astype(np.int32)
            cv2.drawContours(candidate_view, [contour], -1, (0, 255, 0), 1)
            cv2.polylines(candidate_view, [box], True, (0, 255, 255), 1)
            at = tuple(np.round(item['center'] * upscale).astype(int))
            cv2.putText(candidate_view, str(index), at, cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 255), 1)
        selected_view = view.copy()
        if pair is not None:
            for index, color in zip(pair, ((0, 0, 255), (0, 255, 0))):
                box = np.round(candidates[index]['box'] * upscale).astype(np.int32)
                cv2.polylines(selected_view, [box], True, color, 1)
                at = tuple(box[0].astype(int))
                cv2.putText(selected_view, str(index), at, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        write('01_roi.png', view)
        write('02_raw_binary.png', raw_binary)
        write('02_binary.png', binary)
        write('02_removed_protrusions.png', cv2.subtract(raw_binary, binary))
        write('03_candidates.png', candidate_view)
        write('04_selected_pair.png', selected_view)
        refined_view = view.copy()
        for item in candidates:
            for key, color in (('before_gray_refinement', (255,0,0)), ('box', (0,255,255))):
                polygon = np.round(item[key]*upscale).astype(np.int32)
                cv2.polylines(refined_view, [polygon], True, color, 1)
        write('05_corner_refinement.png', refined_view)
        info = {'opening_kernel': kernel_size, 'roi_rect': [x,y,w,h], 'upscale': upscale, 'selected_indices': pair,
                'pair_score': score, 'candidates': [
                    {key: value.tolist() if isinstance(value, np.ndarray) else value
                     for key, value in item.items()} for item in candidates]}
        (folder / 'details.json').write_text(json.dumps(info, indent=2), encoding='utf-8')

    if len(candidates) < 2:
        save_debug()
        return None

    best_pair = None
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
                best_pair = [i, j]

    save_debug(best_pair, best_score)
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
