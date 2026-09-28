from pathlib import Path

import cv2
import numpy as np


DEFAULT_BOARD_COLS = 6
DEFAULT_BOARD_ROWS = 3
DEFAULT_SQUARE_SIZE_MM = 16.0
DEFAULT_DARK_THRESHOLD = 120


def cv_imread(path):
    """Recovered docstring."""
    path = Path(path)
    if not path.exists():
        print("File does not exist:", path)
        return None

    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def cv_imwrite(path, image):
    """Recovered docstring."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    success, buffer = cv2.imencode(path.suffix, image)
    if success:
        buffer.tofile(str(path))
    return success


def transform_points(matrix, points):
    """Recovered docstring."""
    points = np.asarray(points, dtype=np.float32)
    transformed = cv2.perspectiveTransform(points.reshape(-1, 1, 2), matrix)
    return transformed.reshape(-1, 2)


def contour_box(contour):
    """Recovered docstring."""
    rect = cv2.minAreaRect(contour)
    (cx, cy), (width, height), _ = rect

    if width <= 0 or height <= 0:
        return None

    long_side = max(width, height)
    short_side = min(width, height)
    ratio = long_side / short_side
    area = cv2.contourArea(contour)

    return {
        "contour": contour,
        "rect": rect,
        "center": np.array([cx, cy], dtype=np.float32),
        "side": float((width + height) * 0.5),
        "area": float(area),
        "ratio": float(ratio),
        "box": cv2.boxPoints(rect).astype(np.float32),
    }


def find_black_square_candidates(image, dark_threshold=DEFAULT_DARK_THRESHOLD):
    """Recovered docstring."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    _, binary = cv2.threshold(
        blur,
        dark_threshold,
        255,
        cv2.THRESH_BINARY_INV,
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    image_area = image.shape[0] * image.shape[1]
    min_area = image_area * 0.00015
    max_area = image_area * 0.0012

    candidates = []
    for contour in contours:
        box = contour_box(contour)
        if box is None:
            continue

        if box["area"] < min_area or box["area"] > max_area:
            continue
        if box["ratio"] > 1.55:
            continue

        fill = box["area"] / max(1.0, box["side"] * box["side"])
        if fill < 0.45:
            continue

        candidates.append(box)

    candidates.sort(key=lambda item: item["center"][0])
    return candidates


def build_components(candidates):
    """Recovered docstring."""
    if not candidates:
        return []

    median_side = float(np.median([item["side"] for item in candidates]))
    max_distance = median_side * 3.2
    count = len(candidates)

    neighbors = [[] for _ in range(count)]
    for i in range(count):
        for j in range(i + 1, count):
            distance = float(np.linalg.norm(candidates[i]["center"] - candidates[j]["center"]))
            area_ratio = max(candidates[i]["area"], candidates[j]["area"]) / max(
                1.0,
                min(candidates[i]["area"], candidates[j]["area"]),
            )
            if distance <= max_distance and area_ratio <= 2.4:
                neighbors[i].append(j)
                neighbors[j].append(i)

    seen = set()
    components = []

    for start in range(count):
        if start in seen:
            continue

        stack = [start]
        seen.add(start)
        component = []

        while stack:
            index = stack.pop()
            component.append(candidates[index])

            for nxt in neighbors[index]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)

        components.append(component)

    return components


def principal_axes(points):
    """Recovered docstring."""
    centered = points - points.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    u = vt[0].astype(np.float32)
    v = vt[1].astype(np.float32)

    if abs(u[0]) < abs(u[1]):
        u, v = v, u

    if u[0] < 0:
        u = -u

    if v[1] < 0:
        v = -v

    return u, v


def split_rows(squares, board_rows):
    """Recovered docstring."""
    centers = np.array([item["center"] for item in squares], dtype=np.float32)
    u, v = principal_axes(centers)

    projected_v = centers @ v
    order = np.argsort(projected_v)
    row_black_count = len(squares) // board_rows
    rows = []

    for row_index in range(board_rows):
        start = row_index * row_black_count
        end = start + row_black_count
        row = [squares[i] for i in order[start:end]]
        row.sort(key=lambda item: float(item["center"] @ u))
        rows.append(row)

    return rows


def black_square_centers_in_board(board_cols, board_rows, starts_black):
    """Recovered docstring."""
    points = []

    for row in range(board_rows):
        for col in range(board_cols):
            is_black = starts_black if (row + col) % 2 == 0 else not starts_black
            if is_black:
                points.append((col + 0.5, row + 0.5))

    return points


def solve_homography(canonical_points, image_points):
    """Recovered docstring."""
    canonical = np.asarray(canonical_points, dtype=np.float32)
    image = np.asarray(image_points, dtype=np.float32)

    matrix, _ = cv2.findHomography(canonical, image, method=0)
    if matrix is None:
        return None, float("inf")

    predicted = transform_points(matrix, canonical)
    error = float(np.mean(np.linalg.norm(predicted - image, axis=1)))
    return matrix, error


def homography_square_to_mm(homography_square_to_pixel, square_size_mm):
    """Recovered docstring."""
    scale = np.diag([1.0 / square_size_mm, 1.0 / square_size_mm, 1.0])
    return homography_square_to_pixel @ scale


def board_metrics(board, board_cols, board_rows, square_size_mm):
    """Recovered docstring."""
    top_left, top_right, bottom_right, bottom_left = board["outer_corners"]

    top_width = float(np.linalg.norm(top_left - top_right))
    bottom_width = float(np.linalg.norm(bottom_left - bottom_right))
    left_height = float(np.linalg.norm(top_left - bottom_left))
    right_height = float(np.linalg.norm(top_right - bottom_right))

    avg_width_px = (top_width + bottom_width) * 0.5
    avg_height_px = (left_height + right_height) * 0.5
    avg_px_per_square_x = avg_width_px / board_cols
    avg_px_per_square_y = avg_height_px / board_rows
    avg_px_per_mm_x = avg_px_per_square_x / square_size_mm
    avg_px_per_mm_y = avg_px_per_square_y / square_size_mm

    return {
        "avg_width_px": avg_width_px,
        "avg_height_px": avg_height_px,
        "top_width_px": top_width,
        "bottom_width_px": bottom_width,
        "left_height_px": left_height,
        "right_height_px": right_height,
        "width_diff_px": abs(top_width - bottom_width),
        "height_diff_px": abs(left_height - right_height),
        "avg_px_per_square_x": avg_px_per_square_x,
        "avg_px_per_square_y": avg_px_per_square_y,
        "avg_px_per_mm_x": avg_px_per_mm_x,
        "avg_px_per_mm_y": avg_px_per_mm_y,
        "avg_mm_per_px_x": 1.0 / avg_px_per_mm_x,
        "avg_mm_per_px_y": 1.0 / avg_px_per_mm_y,
    }


def fit_board_from_squares(
    squares,
    board_cols=DEFAULT_BOARD_COLS,
    board_rows=DEFAULT_BOARD_ROWS,
    square_size_mm=DEFAULT_SQUARE_SIZE_MM,
):
    """Recovered docstring."""
    black_square_count = (board_cols * board_rows + 1) // 2
    if len(squares) < black_square_count:
        return None

    squares = sorted(squares, key=lambda item: item["area"], reverse=True)[:black_square_count]
    rows = split_rows(squares, board_rows)
    image_points = [item["center"] for row in rows for item in row]

    patterns = [
        (black_square_centers_in_board(board_cols, board_rows, True), "top starts black"),
        (black_square_centers_in_board(board_cols, board_rows, False), "top starts white"),
    ]

    best_matrix = None
    best_error = None
    best_pattern = None

    for canonical_points, pattern_name in patterns:
        matrix, error = solve_homography(canonical_points, image_points)
        if best_error is None or error < best_error:
            best_matrix = matrix
            best_error = error
            best_pattern = pattern_name

    if best_matrix is None:
        return None

    inner_points = [
        (x, y)
        for y in range(1, board_rows)
        for x in range(1, board_cols)
    ]
    outer_points = [
        (0, 0),
        (board_cols, 0),
        (board_cols, board_rows),
        (0, board_rows),
    ]

    h_mm_to_pixel = homography_square_to_mm(best_matrix, square_size_mm)
    h_pixel_to_mm = np.linalg.inv(h_mm_to_pixel)

    board = {
        "found": True,
        "squares": squares,
        "pattern": best_pattern,
        "fit_error": best_error,
        "homography_square_to_pixel": best_matrix,
        "homography_mm_to_pixel": h_mm_to_pixel,
        "homography_pixel_to_mm": h_pixel_to_mm,
        "inner_corners": transform_points(best_matrix, inner_points),
        "outer_corners": transform_points(best_matrix, outer_points),
        "board_cols": board_cols,
        "board_rows": board_rows,
        "square_size_mm": square_size_mm,
    }
    board["center"] = board["outer_corners"].mean(axis=0)
    board["metrics"] = board_metrics(board, board_cols, board_rows, square_size_mm)

    return board


def detect_boards(
    image,
    board_cols=DEFAULT_BOARD_COLS,
    board_rows=DEFAULT_BOARD_ROWS,
    square_size_mm=DEFAULT_SQUARE_SIZE_MM,
    dark_threshold=DEFAULT_DARK_THRESHOLD,
):
    """Recovered docstring."""
    black_square_count = (board_cols * board_rows + 1) // 2
    candidates = find_black_square_candidates(image, dark_threshold=dark_threshold)
    components = build_components(candidates)

    boards = []
    for component in components:
        if len(component) < black_square_count:
            continue

        board = fit_board_from_squares(
            component,
            board_cols=board_cols,
            board_rows=board_rows,
            square_size_mm=square_size_mm,
        )
        if board is None:
            continue

        median_side = np.median([item["side"] for item in board["squares"]])
        max_error = max(18.0, median_side * 0.25)
        if board["fit_error"] > max_error:
            continue

        boards.append(board)

    boards.sort(key=lambda item: float(item["center"][1]))
    return candidates, boards


def detect_best_board(image, **kwargs):
    """Recovered docstring."""
    _, boards = detect_boards(image, **kwargs)
    if not boards:
        return None
    return min(boards, key=lambda item: item["fit_error"])


def pixel_to_real_mm(point, homography_pixel_to_mm):
    """Recovered docstring."""
    return transform_points(homography_pixel_to_mm, [point])[0]


def real_mm_to_pixel(point, homography_mm_to_pixel):
    """Recovered docstring."""
    return transform_points(homography_mm_to_pixel, [point])[0]


def warp_board_to_front_view(image, board, pixels_per_mm=4.0):
    """Recovered docstring."""
    board_cols = board["board_cols"]
    board_rows = board["board_rows"]
    square_size_mm = board["square_size_mm"]

    width_mm = board_cols * square_size_mm
    height_mm = board_rows * square_size_mm
    width_px = max(1, int(round(width_mm * pixels_per_mm)))
    height_px = max(1, int(round(height_mm * pixels_per_mm)))

    scale = np.array(
        [
            [pixels_per_mm, 0.0, 0.0],
            [0.0, pixels_per_mm, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    h_pixel_to_front = scale @ board["homography_pixel_to_mm"]
    warped = cv2.warpPerspective(image, h_pixel_to_front, (width_px, height_px))
    return warped, h_pixel_to_front


def warp_image_to_board_plane(image, board, pixels_per_mm=4.0, padding_mm=20.0):
    """Recovered docstring."""
    height, width = image.shape[:2]
    image_corners = np.array(
        [
            (0, 0),
            (width, 0),
            (width, height),
            (0, height),
        ],
        dtype=np.float32,
    )

    real_corners = transform_points(board["homography_pixel_to_mm"], image_corners)
    min_x = float(real_corners[:, 0].min() - padding_mm)
    min_y = float(real_corners[:, 1].min() - padding_mm)
    max_x = float(real_corners[:, 0].max() + padding_mm)
    max_y = float(real_corners[:, 1].max() + padding_mm)

    output_width = max(1, int(round((max_x - min_x) * pixels_per_mm)))
    output_height = max(1, int(round((max_y - min_y) * pixels_per_mm)))

    mm_to_front_pixel = np.array(
        [
            [pixels_per_mm, 0.0, -min_x * pixels_per_mm],
            [0.0, pixels_per_mm, -min_y * pixels_per_mm],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    h_pixel_to_front = mm_to_front_pixel @ board["homography_pixel_to_mm"]

    warped = cv2.warpPerspective(
        image,
        h_pixel_to_front,
        (output_width, output_height),
    )

    return warped, h_pixel_to_front


def draw_boards(image, candidates, boards):
    """Recovered docstring."""
    output = image.copy()

    for item in candidates:
        box = item["box"].astype(np.int32)
        cv2.polylines(output, [box], True, (80, 120, 255), 1, cv2.LINE_AA)

    for index, board in enumerate(boards, start=1):
        outer = board["outer_corners"].astype(np.int32)
        cv2.polylines(output, [outer], True, (0, 255, 255), 3, cv2.LINE_AA)

        for corner_index, point in enumerate(board["inner_corners"], start=1):
            x, y = point
            cv2.circle(output, (int(round(x)), int(round(y))), 8, (0, 0, 255), -1)
            cv2.putText(
                output,
                str(corner_index),
                (int(round(x)) + 10, int(round(y)) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        label_anchor = board["outer_corners"][0]
        cv2.putText(
            output,
            f"Board {index}",
            (int(label_anchor[0]), max(30, int(label_anchor[1]) - 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return output


def format_matrix(matrix):
    """Recovered docstring."""
    return np.array2string(
        matrix,
        precision=6,
        suppress_small=False,
        separator=", ",
    )


def print_board_summary(boards):
    """Recovered docstring."""
    print("Detected boards:", len(boards))

    for index, board in enumerate(boards, start=1):
        metrics = board["metrics"]
        print(
            f"Board {index}: "
            f"pattern={board['pattern']}, "
            f"fit_error={board['fit_error']:.2f}, "
            f"center={np.round(board['center'], 1).tolist()}, "
            f"px_per_mm={metrics['avg_px_per_mm_x']:.3f},"
            f"{metrics['avg_px_per_mm_y']:.3f}, "
            f"mm_per_px={metrics['avg_mm_per_px_x']:.4f},"
            f"{metrics['avg_mm_per_px_y']:.4f}"
        )
        print("  H mm_to_pixel:")
        print(format_matrix(board["homography_mm_to_pixel"]))
        print("  H pixel_to_mm:")
        print(format_matrix(board["homography_pixel_to_mm"]))


def main():
    """Recovered docstring."""
    base_dir = Path(__file__).resolve().parent
    image_path = base_dir / "test.jpg"
    output_path = base_dir / "checkerboard_calibration_result.png"

    image = cv_imread(image_path)
    if image is None:
        print("Failed to read image:", image_path)
        return

    candidates, boards = detect_boards(image)
    print("Black square candidates:", len(candidates))
    print_board_summary(boards)

    output = draw_boards(image, candidates, boards)
    if cv_imwrite(output_path, output):
        print("Result image saved:", output_path)


if __name__ == "__main__":
    main()
