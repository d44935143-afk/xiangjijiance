import csv
import json
import statistics
import time
from datetime import datetime
from pathlib import Path


FIELDNAMES = [
    "window_start",
    "window_end",
    "frame_index",
    "target_id",
    "tracking_status",
    "sample_count",
    "valid_frame_count",
    "scale_valid_frame_count",
    "lost_frame_count",
    "min_tracking_score",
    "mm_per_pixel",
    "center_x_px",
    "center_y_px",
    "displacement_x_px",
    "displacement_y_px",
    "displacement_x_mm",
    "displacement_y_mm",
    "displacement_x_std_mm",
    "displacement_y_std_mm",
]


def pair_values(value):
    if value is None or len(value) < 2:
        return None, None
    return float(value[0]), float(value[1])


def median(values):
    values = [float(value) for value in values if value is not None]
    return "" if not values else float(statistics.median(values))


def standard_deviation(values):
    values = [float(value) for value in values if value is not None]
    if not values:
        return ""
    if len(values) == 1:
        return 0.0
    return float(statistics.pstdev(values))


def node_value(camera, names):
    for name in names:
        try:
            node = getattr(camera, name)
            if hasattr(node, "GetValue"):
                return node.GetValue()
            if hasattr(node, "Value"):
                return node.Value
        except Exception:
            continue
    return None


def read_camera_metadata(camera):
    device_info = camera.GetDeviceInfo()
    return {
        "camera_model": device_info.GetModelName(),
        "camera_serial": device_info.GetSerialNumber(),
        "camera_width_px": node_value(camera, ("Width",)),
        "camera_height_px": node_value(camera, ("Height",)),
        "camera_fps": node_value(
            camera,
            (
                "ResultingFrameRate",
                "ResultingFrameRateAbs",
                "AcquisitionFrameRate",
                "AcquisitionFrameRateAbs",
            ),
        ),
    }


def create_session_dir(data_root, started_at):
    data_root = Path(data_root)
    data_root.mkdir(parents=True, exist_ok=True)
    base_name = started_at.strftime("%Y-%m-%d_%H-%M-%S")
    session_dir = data_root / base_name
    suffix = 1
    while session_dir.exists():
        session_dir = data_root / f"{base_name}_{suffix:02d}"
        suffix += 1
    session_dir.mkdir(parents=True)
    return session_dir


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


class TrackingDataLogger:
    """Aggregate every frame and save one measurement per target per interval."""

    def __init__(self, data_root, output_interval_seconds=1.0):
        self.started_at = datetime.now()
        self.session_dir = create_session_dir(data_root, self.started_at)
        self.output_path = self.session_dir / "measurements.csv"
        self.output_interval_seconds = max(0.1, float(output_interval_seconds))
        self.samples = {}
        self.window_started_monotonic = time.monotonic()
        self.window_start_text = self.started_at.isoformat(
            sep=" ", timespec="milliseconds"
        )
        self.file = open(self.output_path, "w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=FIELDNAMES)
        self.writer.writeheader()
        self.file.flush()

    def initialize_session(self, state, camera, frame_shape, settings=None):
        initialized_at = datetime.now()
        self.samples.clear()
        self.window_started_monotonic = time.monotonic()
        self.window_start_text = initialized_at.isoformat(
            sep=" ", timespec="milliseconds"
        )
        metadata = read_camera_metadata(camera)
        metadata.update(
            {
                "session_started_at": self.started_at.isoformat(
                    sep=" ", timespec="milliseconds"
                ),
                "image_resolution": [int(frame_shape[1]), int(frame_shape[0])],
                "target_size_mm": state.get("target_size_mm"),
                "target_count": state.get(
                    "target_count", len(state.get("targets", []))
                ),
                "measurement_output_interval_seconds": (
                    self.output_interval_seconds
                ),
                "measurement_aggregation": "median",
            }
        )
        if settings:
            metadata["tracking_settings"] = settings

        save_json(self.session_dir / "metadata.json", metadata)
        save_json(self.session_dir / "targets_initial.json", state)

    def append_target_sample(
        self,
        target,
        frame_index,
        host_time,
    ):
        target_id = target.get("id", "")
        center_x, center_y = pair_values(target.get("current_center"))
        displacement_x_px, displacement_y_px = pair_values(
            target.get("displacement")
        )
        displacement_x_mm, displacement_y_mm = pair_values(
            target.get("displacement_mm")
        )
        self.samples.setdefault(target_id, []).append(
            {
                "host_time": host_time,
                "frame_index": int(frame_index),
                "lost": bool(target.get("lost", False)),
                "reacquired": target.get("reacquire_score") is not None,
                "tracking_score": target.get("tracking_score"),
                "scale_valid": bool(target.get("scale_updated", False)),
                "mm_per_pixel": target.get("mm_per_pixel"),
                "center_x_px": center_x,
                "center_y_px": center_y,
                "displacement_x_px": displacement_x_px,
                "displacement_y_px": displacement_y_px,
                "displacement_x_mm": displacement_x_mm,
                "displacement_y_mm": displacement_y_mm,
            }
        )

    def write_target_window(self, target_id, samples, window_end):
        valid_samples = [sample for sample in samples if not sample["lost"]]
        scale_samples = [
            sample
            for sample in valid_samples
            if sample["scale_valid"] and sample["mm_per_pixel"] is not None
        ]
        lost_count = len(samples) - len(valid_samples)
        if not valid_samples:
            status = "lost"
        elif lost_count:
            status = "partial"
        elif any(sample["reacquired"] for sample in samples):
            status = "reacquired"
        else:
            status = "tracked"

        latest = samples[-1]
        scores = [
            sample["tracking_score"]
            for sample in samples
            if sample["tracking_score"] is not None
        ]
        row = {
            "window_start": self.window_start_text,
            "window_end": window_end,
            "frame_index": latest["frame_index"],
            "target_id": target_id,
            "tracking_status": status,
            "sample_count": len(samples),
            "valid_frame_count": len(valid_samples),
            "scale_valid_frame_count": len(scale_samples),
            "lost_frame_count": lost_count,
            "min_tracking_score": "" if not scores else min(scores),
            "mm_per_pixel": median(
                [sample["mm_per_pixel"] for sample in scale_samples]
            ),
            "center_x_px": median(
                [sample["center_x_px"] for sample in valid_samples]
            ),
            "center_y_px": median(
                [sample["center_y_px"] for sample in valid_samples]
            ),
            "displacement_x_px": median(
                [sample["displacement_x_px"] for sample in valid_samples]
            ),
            "displacement_y_px": median(
                [sample["displacement_y_px"] for sample in valid_samples]
            ),
            "displacement_x_mm": median(
                [sample["displacement_x_mm"] for sample in valid_samples]
            ),
            "displacement_y_mm": median(
                [sample["displacement_y_mm"] for sample in valid_samples]
            ),
            "displacement_x_std_mm": standard_deviation(
                [sample["displacement_x_mm"] for sample in valid_samples]
            ),
            "displacement_y_std_mm": standard_deviation(
                [sample["displacement_y_mm"] for sample in valid_samples]
            ),
        }
        self.writer.writerow(row)

    def flush_window(self, window_end=None):
        if not self.samples:
            return
        if window_end is None:
            window_end = datetime.now().isoformat(sep=" ", timespec="milliseconds")

        for target_id, samples in self.samples.items():
            if samples:
                self.write_target_window(target_id, samples, window_end)

        self.file.flush()
        self.samples.clear()
        self.window_start_text = window_end
        self.window_started_monotonic = time.monotonic()

    def log_state(
        self,
        state,
        frame_index,
        timestamp=None,
    ):
        if timestamp is None:
            timestamp = datetime.now().isoformat(sep=" ", timespec="milliseconds")

        for target in state.get("targets", []):
            self.append_target_sample(
                target,
                frame_index,
                timestamp,
            )

        elapsed = time.monotonic() - self.window_started_monotonic
        if elapsed >= self.output_interval_seconds:
            self.flush_window(window_end=timestamp)

    def close(self):
        if not self.file.closed:
            self.flush_window()
            self.file.close()
