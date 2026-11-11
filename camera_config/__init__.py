"""Load camera settings after Open() and before StartGrabbing()."""
import json
from pathlib import Path

from pypylon import genicam

DEFAULT_CONFIG_PATH = Path(__file__).with_name("parameters.json")
NODE_NAMES = {
    "ExposureTime": ("ExposureTime", "ExposureTimeAbs"),
    "Gain": ("Gain", "GainRaw"),
    "AcquisitionFrameRate": ("AcquisitionFrameRate", "AcquisitionFrameRateAbs"),
    "Width": ("Width",),
    "Height": ("Height",),
}


def _set_node(camera, names, value):
    for name in names:
        node = camera.GetNodeMap().GetNode(name)
        if node is None or not genicam.IsWritable(node):
            continue
        try:
            node.SetValue(value)
        except Exception as exc:
            raise ValueError(f"Cannot set camera parameter {name}={value}: {exc}") from exc
        print(f"Camera setting: {name}={node.GetValue()}")
        return
    raise ValueError(f"Camera parameter is unavailable or not writable: {names}")


def apply_camera_config(camera, config_path=DEFAULT_CONFIG_PATH):
    """Apply non-null settings; reject invalid values instead of silently ignoring them."""
    config_path = Path(config_path)
    with config_path.open(encoding="utf-8-sig") as file:
        settings = json.load(file)
    if not isinstance(settings, dict):
        raise ValueError("Camera configuration must be a JSON object")
    unknown = set(settings) - set(NODE_NAMES)
    if unknown:
        raise ValueError(f"Unknown camera settings: {sorted(unknown)}")
    for name, value in settings.items():
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a number or null")
        if name in ("Width", "Height") and (not isinstance(value, int) or value <= 0):
            raise ValueError(f"{name} must be a positive integer")
    for name, names in NODE_NAMES.items():
        value = settings.get(name)
        if value is None:
            continue
        if name == "ExposureTime":
            _set_node(camera, ("ExposureAuto",), "Off")
        elif name == "Gain":
            _set_node(camera, ("GainAuto",), "Off")
        elif name == "AcquisitionFrameRate":
            _set_node(camera, ("AcquisitionFrameRateEnable",), True)
        _set_node(camera, names, value)
    print(f"Camera configuration: {config_path}")
    print(f"Camera resolution: {camera.Width.GetValue()} x {camera.Height.GetValue()}")
