import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_STATE_PATH = PROJECT_DIR / "targets_state.json"
DEFAULT_WINDOW_WIDTH = 900
DEFAULT_WINDOW_HEIGHT = 560
DEFAULT_MAX_SAMPLES = 1200

PLOT_MARGIN_LEFT = 74
PLOT_MARGIN_RIGHT = 34
PLOT_MARGIN_TOP = 86
PLOT_MARGIN_BOTTOM = 56

CURVE_COLORS = [
    (80, 210, 255),
    (80, 255, 130),
    (255, 190, 90),
    (180, 140, 255),
    (255, 110, 110),
    (110, 220, 220),
    (210, 210, 110),
    (220, 140, 190),
]


def load_state(state_path):
    state_path = Path(state_path)
    if not state_path.exists():
        return None

    try:
        with open(state_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except json.JSONDecodeError:
        return None


def displacement_mm_from_target(target):
    displacement_mm = target.get("displacement_mm")
    if displacement_mm is not None and len(displacement_mm) == 2:
        return displacement_mm

    displacement = target.get("displacement")
    mm_per_pixel = target.get("mm_per_pixel")
    if displacement is None or mm_per_pixel is None:
        return None

    return [
        float(displacement[0] * mm_per_pixel),
        float(displacement[1] * mm_per_pixel),
    ]


def _target_key(target):
    return str(target.get("id", "?"))


def collect_displacement_sample(state, frame_index=None):
    if frame_index is None and state is not None:
        frame_index = state.get("frame_index")
    if frame_index is None:
        frame_index = 0

    sample = {
        "frame_index": int(frame_index),
        "values": {},
    }
    targets = [] if state is None else state.get("targets", [])

    for target in targets:
        target_id = _target_key(target)
        displacement_mm = displacement_mm_from_target(target)
        lost = bool(target.get("lost", False))
        if displacement_mm is None or len(displacement_mm) != 2:
            sample["values"][target_id] = {
                "x": None,
                "y": None,
                "lost": lost,
            }
            continue

        sample["values"][target_id] = {
            "x": float(displacement_mm[0]),
            "y": float(displacement_mm[1]),
            "lost": lost,
        }

    return sample


def append_history(
    history,
    state,
    frame_index=None,
    history_seconds=None,
    max_samples=DEFAULT_MAX_SAMPLES,
):
    if frame_index is None and (state is None or state.get("frame_index") is None):
        if history:
            frame_index = history[-1]["frame_index"] + 1
        else:
            frame_index = 0

    sample = collect_displacement_sample(state, frame_index)
    history.append(sample)

    if len(history) > max_samples:
        del history[: len(history) - max_samples]

    return history


def _all_target_ids(history, state):
    target_ids = []
    targets = [] if state is None else state.get("targets", [])
    for target in targets:
        target_id = _target_key(target)
        if target_id not in target_ids:
            target_ids.append(target_id)

    for sample in history:
        for target_id in sample.get("values", {}):
            if target_id not in target_ids:
                target_ids.append(target_id)

    return target_ids


def _value_range(history, axis):
    values = []
    for sample in history:
        for target_data in sample.get("values", {}).values():
            value = target_data.get(axis)
            if value is not None:
                values.append(float(value))

    if not values:
        return -1.0, 1.0

    min_value = min(values)
    max_value = max(values)
    span = max_value - min_value
    if span < 1e-6:
        padding = max(0.5, abs(max_value) * 0.2)
        return min_value - padding, max_value + padding

    padding = max(0.2, span * 0.15)
    return min_value - padding, max_value + padding


def _frame_range(history):
    if len(history) < 2:
        frame = history[-1]["frame_index"] if history else 0
        return max(0, frame - 10), frame + 1

    start_frame = history[0]["frame_index"]
    end_frame = history[-1]["frame_index"]
    if end_frame <= start_frame:
        end_frame = start_frame + 1
    return start_frame, end_frame


def _map_point(frame_index, value, frame_min, frame_max, value_min, value_max, plot_rect):
    x0, y0, x1, y1 = plot_rect
    x_rate = (frame_index - frame_min) / max(1e-6, frame_max - frame_min)
    y_rate = (value - value_min) / max(1e-6, value_max - value_min)
    x = int(round(x0 + x_rate * (x1 - x0)))
    y = int(round(y1 - y_rate * (y1 - y0)))
    return x, y


def _draw_axes(image, plot_rect, frame_min, frame_max, value_min, value_max, title):
    x0, y0, x1, y1 = plot_rect
    axis_color = (145, 145, 145)
    grid_color = (54, 54, 54)
    text_color = (190, 190, 190)

    cv2.line(image, (x0, y1), (x1, y1), axis_color, 1, cv2.LINE_AA)
    cv2.line(image, (x0, y0), (x0, y1), axis_color, 1, cv2.LINE_AA)

    for index in range(5):
        rate = index / 4.0
        x = int(round(x0 + rate * (x1 - x0)))
        cv2.line(image, (x, y0), (x, y1), grid_color, 1, cv2.LINE_AA)
        frame = frame_min + rate * (frame_max - frame_min)
        cv2.putText(
            image,
            f"{frame:.0f}",
            (x - 18, y1 + 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            text_color,
            1,
            cv2.LINE_AA,
        )

    for index in range(5):
        rate = index / 4.0
        y = int(round(y1 - rate * (y1 - y0)))
        cv2.line(image, (x0, y), (x1, y), grid_color, 1, cv2.LINE_AA)
        value = value_min + rate * (value_max - value_min)
        cv2.putText(
            image,
            f"{value:.2f}",
            (10, y + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            text_color,
            1,
            cv2.LINE_AA,
        )

    cv2.putText(
        image,
        "Frame",
        ((x0 + x1) // 2 - 24, y1 + 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        title,
        (x0, y0 - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )


def _draw_curve(image, points, color):
    if len(points) < 2:
        for point in points:
            cv2.circle(image, point, 3, color, -1, cv2.LINE_AA)
        return

    for point_a, point_b in zip(points[:-1], points[1:]):
        cv2.line(image, point_a, point_b, color, 2, cv2.LINE_AA)


def draw_time_axis_dashboard(
    history,
    state,
    width=DEFAULT_WINDOW_WIDTH,
    height=DEFAULT_WINDOW_HEIGHT,
):
    image = np.full((height, width, 3), 26, dtype=np.uint8)
    plot_gap = 58
    plot_height = (height - PLOT_MARGIN_TOP - PLOT_MARGIN_BOTTOM - plot_gap) // 2
    x_plot_rect = (
        PLOT_MARGIN_LEFT,
        PLOT_MARGIN_TOP,
        width - PLOT_MARGIN_RIGHT,
        PLOT_MARGIN_TOP + plot_height,
    )
    y_plot_rect = (
        PLOT_MARGIN_LEFT,
        PLOT_MARGIN_TOP + plot_height + plot_gap,
        width - PLOT_MARGIN_RIGHT,
        PLOT_MARGIN_TOP + plot_height * 2 + plot_gap,
    )

    cv2.putText(
        image,
        "Realtime Displacement Curve by Frame",
        (32, 46),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.82,
        (240, 240, 240),
        2,
        cv2.LINE_AA,
    )

    update_text = "Waiting for target data"
    if state is not None:
        update_text = f"Last update: {state.get('last_update_time', '-')}"
    cv2.putText(
        image,
        update_text,
        (32, 76),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (170, 170, 170),
        1,
        cv2.LINE_AA,
    )

    if not history:
        _draw_axes(image, x_plot_rect, 0, 10, -1.0, 1.0, "X displacement (mm)")
        _draw_axes(image, y_plot_rect, 0, 10, -1.0, 1.0, "Y displacement (mm)")
        cv2.putText(
            image,
            "No displacement samples.",
            (PLOT_MARGIN_LEFT + 20, PLOT_MARGIN_TOP + 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (80, 80, 255),
            2,
            cv2.LINE_AA,
        )
        return image

    frame_min, frame_max = _frame_range(history)
    x_value_min, x_value_max = _value_range(history, "x")
    y_value_min, y_value_max = _value_range(history, "y")
    _draw_axes(
        image,
        x_plot_rect,
        frame_min,
        frame_max,
        x_value_min,
        x_value_max,
        "X displacement (mm)",
    )
    _draw_axes(
        image,
        y_plot_rect,
        frame_min,
        frame_max,
        y_value_min,
        y_value_max,
        "Y displacement (mm)",
    )

    target_ids = _all_target_ids(history, state)
    legend_x = width - 180
    legend_y = 38

    for target_index, target_id in enumerate(target_ids):
        color = CURVE_COLORS[target_index % len(CURVE_COLORS)]
        for axis, plot_rect, value_min, value_max in (
            ("x", x_plot_rect, x_value_min, x_value_max),
            ("y", y_plot_rect, y_value_min, y_value_max),
        ):
            points = []
            for sample in history:
                target_data = sample.get("values", {}).get(target_id)
                if not target_data or target_data.get("lost", False):
                    continue
                value = target_data.get(axis)
                if value is None:
                    continue
                points.append(
                    _map_point(
                        sample["frame_index"],
                        float(value),
                        frame_min,
                        frame_max,
                        value_min,
                        value_max,
                        plot_rect,
                    )
                )

            _draw_curve(image, points, color)

        legend_line_y = legend_y + target_index * 22
        cv2.line(
            image,
            (legend_x, legend_line_y - 4),
            (legend_x + 28, legend_line_y - 4),
            color,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            f"ID {target_id}",
            (legend_x + 36, legend_line_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

    latest_sample = history[-1]
    latest_text_parts = []
    for target_id in target_ids[:3]:
        target_data = latest_sample.get("values", {}).get(target_id)
        if not target_data or target_data.get("x") is None or target_data.get("y") is None:
            latest_text_parts.append(f"ID {target_id}: --")
        else:
            latest_text_parts.append(
                f"ID {target_id}: X {target_data['x']:.3f}, Y {target_data['y']:.3f} mm"
            )

    if latest_text_parts:
        cv2.putText(
            image,
            " | ".join(latest_text_parts),
            (PLOT_MARGIN_LEFT, height - 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )

    return image


def draw_dashboard(state, width=DEFAULT_WINDOW_WIDTH, height=DEFAULT_WINDOW_HEIGHT):
    image = np.full((height, width, 3), 28, dtype=np.uint8)
    targets = [] if state is None else state.get("targets", [])

    cv2.putText(
        image,
        "Realtime Target Displacement",
        (32, 54),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (240, 240, 240),
        2,
        cv2.LINE_AA,
    )

    update_text = "Waiting for targets_state.json"
    if state is not None:
        update_text = f"Last update: {state.get('last_update_time', '-')}"

    cv2.putText(
        image,
        update_text,
        (32, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )

    headers = ["ID", "X displacement (mm)", "Y displacement (mm)", "Status"]
    xs = [44, 150, 390, 650]
    y = 140
    for x, header in zip(xs, headers):
        cv2.putText(
            image,
            header,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (120, 210, 255),
            2,
            cv2.LINE_AA,
        )

    y += 34
    if not targets:
        cv2.putText(
            image,
            "No target data.",
            (44, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (80, 80, 255),
            2,
            cv2.LINE_AA,
        )
        return image

    for target in targets:
        target_id = target.get("id", "?")
        displacement_mm = displacement_mm_from_target(target)
        lost = target.get("lost", False)

        if displacement_mm is None:
            dx_text = "0.000"
            dy_text = "0.000"
        else:
            dx_text = f"{displacement_mm[0]:.3f}"
            dy_text = f"{displacement_mm[1]:.3f}"

        if lost:
            status = "LOST"
        elif displacement_mm is None:
            status = "NO_SCALE"
        else:
            status = "OK"

        color = (80, 80, 255) if lost else (80, 230, 120)
        values = [str(target_id), dx_text, dy_text, status]

        for x, value in zip(xs, values):
            cv2.putText(
                image,
                value,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                color,
                2,
                cv2.LINE_AA,
            )

        y += 36
        if y > height - 35:
            break

    return image


def build_parser():
    parser = argparse.ArgumentParser(
        description="Realtime display of target X/Y displacement in millimeters.",
    )
    parser.add_argument(
        "--state",
        default=str(DEFAULT_STATE_PATH),
        help=f"Tracking state JSON path. Default: {DEFAULT_STATE_PATH}",
    )
    parser.add_argument(
        "--refresh",
        type=float,
        default=0.2,
        help="Refresh interval in seconds. Default: 0.2",
    )
    return parser


def main():
    args = build_parser().parse_args()
    window_name = "Realtime Displacement mm"
    history = []

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 900, 560)

    try:
        while True:
            state = load_state(args.state)
            append_history(history, state)
            image = draw_time_axis_dashboard(history, state)
            cv2.imshow(window_name, image)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

            time.sleep(max(0.02, args.refresh))
    finally:
        cv2.destroyWindow(window_name)


if __name__ == "__main__":
    main()
