import argparse
import json
from pathlib import Path

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_IMAGES_DIR = Path(r"C:\Users\Administrator\Desktop\biaodingshuju")
DEFAULT_OUTPUT_PATH = BASE_DIR / "camera_intrinsics.json"
DEFAULT_DEBUG_DIR = BASE_DIR / "intrinsic_debug"
IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def cv_imread(path):
    path = Path(path)
    if not path.exists():
        print("File does not exist:", path)
        return None

    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def cv_imwrite(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    success, buffer = cv2.imencode(path.suffix, image)
    if success:
        buffer.tofile(str(path))
    return success


def list_images(images_dir):
    images_dir = Path(images_dir)
    if not images_dir.exists():
        print("Image folder does not exist:", images_dir)
        return []

    return sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def build_object_points(pattern_cols, pattern_rows, square_size_mm):
    points = np.zeros((pattern_cols * pattern_rows, 3), np.float32)
    points[:, :2] = np.mgrid[0:pattern_cols, 0:pattern_rows].T.reshape(-1, 2)
    points *= float(square_size_mm)
    return points


def find_chessboard_corners(image, pattern_cols, pattern_rows):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    pattern_size = (int(pattern_cols), int(pattern_rows))

    found, corners = cv2.findChessboardCorners(
        gray,
        pattern_size,
        cv2.CALIB_CB_ADAPTIVE_THRESH
        + cv2.CALIB_CB_NORMALIZE_IMAGE
        + cv2.CALIB_CB_FAST_CHECK,
    )
    if not found:
        return None

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return corners


def calibrate_camera(
    images_dir,
    pattern_cols,
    pattern_rows,
    square_size_mm,
    debug_dir=None,
):
    image_paths = list_images(images_dir)
    object_points = []
    image_points = []
    used_images = []
    image_size = None
    board_object_points = build_object_points(
        pattern_cols,
        pattern_rows,
        square_size_mm,
    )

    for image_path in image_paths:
        image = cv_imread(image_path)
        if image is None:
            continue

        image_size = (image.shape[1], image.shape[0])
        corners = find_chessboard_corners(image, pattern_cols, pattern_rows)
        if corners is None:
            print("Corners not found:", image_path)
            continue

        object_points.append(board_object_points.copy())
        image_points.append(corners)
        used_images.append(str(image_path))
        print("Corners found:", image_path)

        if debug_dir is not None:
            debug_image = image.copy()
            cv2.drawChessboardCorners(
                debug_image,
                (pattern_cols, pattern_rows),
                corners,
                True,
            )
            cv_imwrite(Path(debug_dir) / f"{image_path.stem}_corners.png", debug_image)

    if not object_points or image_size is None:
        print("No valid chessboard images.")
        return None

    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None,
    )

    return {
        "image_size": list(image_size),
        "pattern_cols": int(pattern_cols),
        "pattern_rows": int(pattern_rows),
        "square_size_mm": float(square_size_mm),
        "rms_reprojection_error": float(rms),
        "camera_matrix": camera_matrix.tolist(),
        "dist_coeffs": dist_coeffs.reshape(-1).tolist(),
        "used_images": used_images,
        "image_count": len(used_images),
        "rvecs": [rvec.reshape(-1).tolist() for rvec in rvecs],
        "tvecs": [tvec.reshape(-1).tolist() for tvec in tvecs],
    }


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Calibrate camera intrinsics from chessboard images.",
    )
    parser.add_argument(
        "--images-dir",
        default=str(DEFAULT_IMAGES_DIR),
        help=f"Folder containing calibration images. Default: {DEFAULT_IMAGES_DIR}",
    )
    parser.add_argument(
        "--pattern-cols",
        type=int,
        default=6,
        help="Number of inner corners per chessboard row.",
    )
    parser.add_argument(
        "--pattern-rows",
        type=int,
        default=9,
        help="Number of inner corners per chessboard column.",
    )
    parser.add_argument(
        "--square-size-mm",
        type=float,
        default=1.0,
        help=(
            "Chessboard square size. Use 1.0 when real size is unknown. "
            "Default: 1.0"
        ),
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help=f"Output intrinsics JSON path. Default: {DEFAULT_OUTPUT_PATH}",
    )
    parser.add_argument(
        "--debug-dir",
        default=str(DEFAULT_DEBUG_DIR),
        help=f"Corner debug image folder. Default: {DEFAULT_DEBUG_DIR}",
    )
    parser.add_argument(
        "--no-debug",
        action="store_true",
        help="Do not save corner debug images.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    result = calibrate_camera(
        images_dir=args.images_dir,
        pattern_cols=args.pattern_cols,
        pattern_rows=args.pattern_rows,
        square_size_mm=args.square_size_mm,
        debug_dir=None if args.no_debug else args.debug_dir,
    )
    if result is None:
        return

    save_json(args.output, result)
    print("Camera intrinsics saved:", Path(args.output))
    print("RMS reprojection error:", result["rms_reprojection_error"])
    print("Camera matrix:")
    print(np.asarray(result["camera_matrix"]))
    print("Distortion coefficients:")
    print(np.asarray(result["dist_coeffs"]))


if __name__ == "__main__":
    main()
