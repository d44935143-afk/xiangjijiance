from pathlib import Path

import cv2

import target_recognition


COLORS = [
    (0, 0, 255),
    (0, 255, 0),
    (255, 0, 0),
    (0, 255, 255),
    (255, 0, 255),
    (255, 255, 0),
]


def draw_recognition_results(frame, results):
    """Recovered docstring."""
    output = frame.copy()

    for index, (region, number) in enumerate(results):
        color = COLORS[index % len(COLORS)]

        x, y, w, h = region.rect
        cv2.rectangle(
            output,
            (x, y),
            (x + w, y + h),
            color,
            2,
        )

        if region.ellipse is not None:
            (cx, cy), (axis_a, axis_b), angle = region.ellipse
            cv2.ellipse(
                output,
                (int(round(cx)), int(round(cy))),
                (int(round(axis_a / 2.0)), int(round(axis_b / 2.0))),
                angle,
                0,
                360,
                color,
                2,
            )

        if region.digit_bbox is not None:
            dx, dy, dw, dh = region.digit_bbox
            cv2.rectangle(
                output,
                (dx, dy),
                (dx + dw, dy + dh),
                (255, 255, 255),
                2,
            )

        # 逕ｻ荳ｭ蠢・せ
        if region.center is not None:
            center_x, center_y = region.center
            cv2.circle(
                output,
                (int(center_x), int(center_y)),
                5,
                color,
                -1,
            )

        # 蜀呵ｯ・悪譬・ｭｾ・壽ｨ｡譚ｿ郛門捷 + 謨ｰ蟄怜ｺ丞捷 + 蛻・焚
        label = (
            f"T{region.match_template_id} "
            f"ID={number} "
            f"S={region.digit_score:.2f}"
        )
        cv2.putText(
            output,
            label,
            (x, max(25, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )

    return output


def draw_first_frame_recognition(
    frame,
    recognizer=None,
    output_path=None,
    show=False,
):
    """Recovered docstring."""
    if recognizer is None:
        recognizer = target_recognition.TargetRecognizer()

    results = recognizer.recognize_targets(frame, draw_result=False)

    drawn_frame = draw_recognition_results(frame, results)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), drawn_frame)
        print("Draw result saved:", output_path)

    if show:
        cv2.imshow("First Frame Recognition", drawn_frame)
        cv2.waitKey(0)
        cv2.destroyWindow("First Frame Recognition")

    return drawn_frame, results


def draw_image_file(image_path, output_path=None, show=True):
    """Recovered docstring."""
    image_path = Path(image_path)
    base_dir = Path(__file__).resolve().parent

    recognizer = target_recognition.TargetRecognizer(base_dir=base_dir)
    frame = recognizer.moban.cv_imread(image_path)

    if frame is None:
        print("Failed to read image:", image_path)
        return None, []

    if output_path is None:
        output_path = base_dir / "first_frame_recognition_draw.png"

    return draw_first_frame_recognition(
        frame=frame,
        recognizer=recognizer,
        output_path=output_path,
        show=show,
    )


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    default_image = base_dir / "Image__2026-06-10__16-44-43.jpg"
    draw_image_file(default_image)
