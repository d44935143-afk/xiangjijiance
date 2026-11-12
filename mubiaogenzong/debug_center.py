"""Debug grayscale-gradient subpixel center localization without changing live tracking."""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

import center_calculation as center
from gradient_localization import gradient_edge_points, robust_line

PROJECT_DIR = Path(__file__).resolve().parent.parent
# Change this path to choose the image used when running without arguments.
DEFAULT_IMAGE_PATH = Path(r"C:\Users\Administrator\Desktop\xiangce\1.jpg")


def save_image(path, image):
    if not center.cv_imwrite(path, image):
        raise RuntimeError(f"Cannot save image: {path}")


def draw_line(image, line, color):
    height, width = image.shape[:2]
    p1, p2 = [tuple(int(round(v)) for v in point) for point in line]
    visible, a, b = cv2.clipLine((0, 0, width, height), p1, p2)
    if visible:
        cv2.line(image, a, b, color, 1, cv2.LINE_8)



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", nargs="?", default=str(DEFAULT_IMAGE_PATH),
                        help=f"Source image path. Default: {DEFAULT_IMAGE_PATH}")
    parser.add_argument("--roi", type=int, nargs=4, metavar=("X", "Y", "W", "H"),
                        help="Optional target rectangle; otherwise select with the mouse")
    parser.add_argument("--margin-ratio", type=float, default=0.35)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    if not np.isfinite(args.margin_ratio) or args.margin_ratio < 0:
        parser.error("--margin-ratio must be finite and nonnegative")
    image_path = Path(args.image).resolve()
    frame = center.cv_imread(image_path)
    if frame is None:
        parser.error(f"Cannot read image: {image_path}")
    if args.roi:
        x, y, w, h = args.roi
        height, width = frame.shape[:2]
        if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > width or y + h > height:
            parser.error("--roi must be a positive rectangle inside the source image")
        rects = [args.roi]
    else:
        sys.path.insert(0, str(PROJECT_DIR / "mubiaoshibie"))
        from image_template_match import select_target_rects
        rects = select_target_rects(frame)
    if not rects:
        print("No targets selected; nothing saved.")
        return
    output_dir = (args.output_dir or PROJECT_DIR / "data" /
                  datetime.now().strftime("center_debug_%Y-%m-%d_%H-%M-%S_%f")).resolve()
    # Use a fresh folder to protect source images and previous debug results.
    output_dir.mkdir(parents=True, exist_ok=False)
    save_image(output_dir / "00_source.png", frame)
    results = []
    for target_id, (x, y, w, h) in enumerate(rects, start=1):
        folder = output_dir / f"target_{target_id:03d}"
        folder.mkdir()
        files = []

        def save_step(name, image):
            save_image(folder / name, image)
            files.append(name)

        save_step("01_selected.png", frame[y:y + h, x:x + w])
        margin = max(w, h) * args.margin_ratio
        rx, ry, rw, rh = center.clip_rect(x, y, w, h, frame.shape, margin=margin)
        crop = frame[ry:ry + rh, rx:rx + rw].copy()
        save_step("02_roi.png", crop)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        # Sobel gradients are computed from the smoothed grayscale image.
        gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3, scale=1 / 8)
        gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3, scale=1 / 8)
        save_step("03_gray.png", gray)
        save_step("04_blurred.png", blurred)
        save_step("05_gradient_x.png", cv2.convertScaleAbs(gx))
        save_step("06_gradient_y.png", cv2.convertScaleAbs(gy))
        np.savez_compressed(folder / "gradients.npz", gx=gx, gy=gy)
        local_rect = (x - rx, y - ry, w, h)
        vertical = gradient_edge_points(gx, local_rect, vertical=True)
        horizontal = gradient_edge_points(gy, local_rect, vertical=False)
        points_image = crop.copy()
        for points, color, name in ((vertical, (0, 0, 255), "07_vertical_points.png"),
                                    (horizontal, (0, 255, 0), "08_horizontal_points.png")):
            separate = crop.copy()
            if points is not None:
                for px, py in points:
                    ix, iy = int(round(float(px))), int(round(float(py)))
                    if 0 <= ix < rw and 0 <= iy < rh:
                        separate[iy, ix] = color
                        points_image[iy, ix] = color
            save_step(name, separate)
        save_step("09_all_points.png", points_image)
        vertical_line = robust_line(vertical)
        horizontal_line = robust_line(horizontal)
        fitted = crop.copy()
        for line, color, name in ((vertical_line, (0, 0, 255), "10_vertical_line.png"),
                                  (horizontal_line, (0, 255, 0), "11_horizontal_line.png")):
            separate = crop.copy()
            if line is not None:
                draw_line(separate, line, color)
                draw_line(fitted, line, color)
            save_step(name, separate)
        save_step("12_both_lines.png", fitted)
        local = (center.line_intersection(*vertical_line, *horizontal_line)
                 if vertical_line is not None and horizontal_line is not None else None)
        location = [local[0] + rx, local[1] + ry] if local is not None else None
        final = fitted.copy()
        if local is not None:
            ix, iy = [int(round(v)) for v in local]
            if 0 <= ix < rw and 0 <= iy < rh:
                final[iy, ix] = (255, 0, 255)
        save_step("13_center.png", final)
        failure = ("insufficient_vertical_points" if vertical_line is None else
                   "insufficient_horizontal_points" if horizontal_line is None else
                   "parallel_lines" if local is None else None)
        result = {"id": target_id, "selected_rect": [x, y, w, h],
                  "roi_rect": [rx, ry, rw, rh], "center_px": location,
                  "center_roi_px": local, "success": local is not None,
                  "failure_reason": failure,
                  "vertical_line_roi": vertical_line,
                  "horizontal_line_roi": horizontal_line,
                  "vertical_points_roi": vertical.tolist() if vertical is not None else [],
                  "horizontal_points_roi": horizontal.tolist() if horizontal is not None else [],
                  "vertical_point_count": len(vertical) if vertical is not None else 0,
                  "horizontal_point_count": len(horizontal) if horizontal is not None else 0,
                  "step_files": files}
        with (folder / "results.json").open("w", encoding="utf-8") as file:
            json.dump(result, file, ensure_ascii=False, indent=2)
        results.append(result)
        print(f"ID{target_id}: center={location} px, failure={failure}")
    with (output_dir / "results.json").open("w", encoding="utf-8") as file:
        json.dump({"source_image": str(image_path), "margin_ratio": args.margin_ratio,
                   "algorithm": "grayscale_gradient_subpixel",
                   "parameters": {"gaussian_kernel": [3, 3], "sobel_kernel": 3,
                                  "sobel_scale": 0.125, "min_gradient": 5.0,
                                  "search_width_ratio": 0.5, "scan_length_ratio": 0.7,
                                  "crossing_exclusion_ratio": 0.1, "min_points": 6,
                                  "subpixel": "three-point parabolic interpolation",
                                  "fit_distance": "HUBER", "line_width_px": 1},
                   "targets": results}, file, ensure_ascii=False, indent=2)
    (output_dir / "README.txt").write_text(
        "00_source: original image\n"
        "01: selected target; 02: expanded ROI; 03: grayscale; 04: Gaussian blur; "
        "05: absolute X gradient; 06: absolute Y gradient; gradients.npz: signed float gradients\n"
        "07: vertical edge points (red); 08: horizontal edge points (green); "
        "09: all edge points\n"
        "10: vertical fitted line; 11: horizontal fitted line; 12: both lines; "
        "13: center marked by one magenta pixel\n"
        "All coordinates ending in _roi are local to 02_roi. center_px is in the source image.\n"
        "Failures still save all available stages; absent lines are not drawn. "
        "Decimal coordinates do not establish measurement accuracy.\n", encoding="utf-8")
    print(f"Debug results saved: {output_dir}")


if __name__ == "__main__":
    main()
