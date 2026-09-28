import argparse
import json
import math
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_PATH = BASE_DIR / "cgcs2000_transform_result.json"

CGCS2000_A = 6378137.0
CGCS2000_F = 1.0 / 298.257222101
CGCS2000_E2 = CGCS2000_F * (2.0 - CGCS2000_F)


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def cgcs2000_blh_to_xyz(lat_deg, lon_deg, height_m):
    lat = math.radians(float(lat_deg))
    lon = math.radians(float(lon_deg))
    height = float(height_m)

    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    sin_lon = math.sin(lon)
    cos_lon = math.cos(lon)

    n = CGCS2000_A / math.sqrt(1.0 - CGCS2000_E2 * sin_lat * sin_lat)
    x = (n + height) * cos_lat * cos_lon
    y = (n + height) * cos_lat * sin_lon
    z = (n * (1.0 - CGCS2000_E2) + height) * sin_lat

    return {
        "x_m": float(x),
        "y_m": float(y),
        "z_m": float(z),
    }


def cgcs2000_xyz_to_blh(x_m, y_m, z_m):
    x = float(x_m)
    y = float(y_m)
    z = float(z_m)

    b = CGCS2000_A * (1.0 - CGCS2000_F)
    ep2 = (CGCS2000_A * CGCS2000_A - b * b) / (b * b)
    p = math.sqrt(x * x + y * y)
    theta = math.atan2(z * CGCS2000_A, p * b)

    sin_theta = math.sin(theta)
    cos_theta = math.cos(theta)
    lat = math.atan2(
        z + ep2 * b * sin_theta ** 3,
        p - CGCS2000_E2 * CGCS2000_A * cos_theta ** 3,
    )
    lon = math.atan2(y, x)

    sin_lat = math.sin(lat)
    n = CGCS2000_A / math.sqrt(1.0 - CGCS2000_E2 * sin_lat * sin_lat)
    height = p / math.cos(lat) - n

    return {
        "lat_deg": float(math.degrees(lat)),
        "lon_deg": float(math.degrees(lon)),
        "height_m": float(height),
    }


def result_header():
    return {
        "coordinate_system": "CGCS2000 geodetic BLH <-> geocentric Cartesian XYZ",
        "ellipsoid": {
            "name": "CGCS2000",
            "semi_major_axis_m": CGCS2000_A,
            "flattening": CGCS2000_F,
            "first_eccentricity_squared": CGCS2000_E2,
        },
        "axis": {
            "x": "toward latitude 0 deg, longitude 0 deg",
            "y": "toward latitude 0 deg, longitude 90 deg east",
            "z": "toward north pole",
            "unit": "m",
        },
    }


def build_parser():
    parser = argparse.ArgumentParser(
        description="Convert CGCS2000 geodetic BLH and geocentric Cartesian XYZ coordinates.",
    )
    parser.add_argument(
        "--blh",
        nargs=3,
        type=float,
        metavar=("LAT_DEG", "LON_DEG", "HEIGHT_M"),
        help="Convert CGCS2000 latitude, longitude and height to XYZ.",
    )
    parser.add_argument(
        "--xyz",
        nargs=3,
        type=float,
        metavar=("X_M", "Y_M", "Z_M"),
        help="Convert CGCS2000 geocentric Cartesian XYZ to latitude, longitude and height.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help=f"Output JSON path. Default: {DEFAULT_OUTPUT_PATH}",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save conversion result to JSON.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    output = result_header()

    if args.blh is not None:
        output["input_type"] = "BLH"
        output["input"] = {
            "lat_deg": args.blh[0],
            "lon_deg": args.blh[1],
            "height_m": args.blh[2],
        }
        output["output_type"] = "XYZ"
        output["output"] = cgcs2000_blh_to_xyz(*args.blh)
        print("CGCS2000 BLH -> XYZ:", output["output"])

    if args.xyz is not None:
        output["input_type"] = "XYZ"
        output["input"] = {
            "x_m": args.xyz[0],
            "y_m": args.xyz[1],
            "z_m": args.xyz[2],
        }
        output["output_type"] = "BLH"
        output["output"] = cgcs2000_xyz_to_blh(*args.xyz)
        print("CGCS2000 XYZ -> BLH:", output["output"])

    if args.blh is None and args.xyz is None:
        print("Please provide --blh or --xyz.")
        return

    if args.save:
        save_json(args.output, output)
        print("Result saved:", Path(args.output))


if __name__ == "__main__":
    main()
