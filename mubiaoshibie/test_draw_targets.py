import argparse
import json
from pathlib import Path

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_STATE_PATH = BASE_DIR.parent / "targets_state.json"
DEFAULT_OUTPUT_PATH = BASE_DIR / "test_targets_boxes.png"


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


def load_target_state(state_path):
    state_path = Path(state_path)
    if not state_path.exists():
        print("State file does not exist:", state_path)
        return None

    with open(state_path, "r", encoding="utf-8") as file:
        return json.load(file)


def draw_targets(image, targets):
    output = image.copy()

    for target in targets:
        center = target.get("current_center") or target.get("initial_center")
        if center is None or len(center) != 2:
            continue

        center_x, center_y = center
        cx = int(round(center_x))
        cy = int(round(center_y))
        target_id = target.get("id", "?")
        score = target.get("scaled_template", {}).get("match_score")
        template_size = target.get("scaled_template", {}).get("template_size")
        template_corners = target.get("template_corners")

        color = (0, 255, 0) if not target.get("lost", False) else (0, 0, 255)
        cv2.drawMarker(
            output,
            (cx, cy),
            color,
            cv2.MARKER_CROSS,
            14,
            2,
            cv2.LINE_AA,
        )
        cv2.circle(output, (cx, cy), 4, color, -1)

        if template_corners is not None and len(template_corners) == 4:
            points = np.asarray(template_corners, dtype=np.int32)
            cv2.polylines(output, [points], True, color, 1, cv2.LINE_AA)
            x = int(points[:, 0].min())
            y = int(points[:, 1].min())
        elif template_size is not None and len(template_size) == 2:
            w, h = [int(round(value)) for value in template_size]
            x = cx - w // 2
            y = cy - h // 2
            cv2.rectangle(output, (x, y), (x + w, y + h), color, 1)
        else:
            x, y = cx, cy

        label = f"ID{target_id}"
        if score is not None:
            label += f" S={score:.2f}"

        cv2.putText(
            output,
            label,
            (x, max(24, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    return output


def build_parser():
    parser = argparse.ArgumentParser(
        description="Draw saved target rectangles on the original image.",
    )
    parser.add_argument(
        "--state",
        default=str(DEFAULT_STATE_PATH),
        help=f"Target state JSON path. Default: {DEFAULT_STATE_PATH}",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help=f"Output image path. Default: {DEFAULT_OUTPUT_PATH}",
    )
    return parser


def main():
    args = build_parser().parse_args()

    state = load_target_state(args.state)
    if state is None:
        return

    image = cv_imread(state.get("source_image"))
    if image is None:
        return

    targets = state.get("targets", [])
    output = draw_targets(image, targets)

    if cv_imwrite(args.output, output):
        print("Targets drawn:", len(targets))
        print("Result saved:", Path(args.output))
    else:
        print("Failed to save result:", args.output)


if __name__ == "__main__":
    main()
