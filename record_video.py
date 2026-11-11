"""Record the configured pypylon camera to an unannotated MP4 video."""
import argparse
import math
from datetime import datetime
from pathlib import Path
from contextlib import ExitStack

import cv2
from pypylon import genicam, pylon

from camera_config import DEFAULT_CONFIG_PATH, apply_camera_config

PROJECT_DIR = Path(__file__).resolve().parent
WINDOW_NAME = "Camera recording - Q / Esc to stop"


def camera_fps(camera):
    for name in ("ResultingFrameRate", "ResultingFrameRateAbs",
                 "AcquisitionFrameRate", "AcquisitionFrameRateAbs"):
        node = camera.GetNodeMap().GetNode(name)
        if node is not None and genicam.IsReadable(node):
            value = float(node.GetValue())
            if math.isfinite(value) and value > 0:
                return value
    raise RuntimeError("Cannot read camera frame rate; specify --fps explicitly")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None, help="Output MP4 path")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--fps", type=float, default=None,
                        help="Override saved video playback frame rate only")
    args = parser.parse_args()
    if args.fps is not None and (not math.isfinite(args.fps) or args.fps <= 0):
        parser.error("--fps must be positive and finite")
    output = (args.output or PROJECT_DIR / "data" / "recordings" /
              datetime.now().strftime("camera_%Y-%m-%d_%H-%M-%S_%f.mp4")).resolve()
    if output.suffix.lower() != ".mp4":
        parser.error("--output must use the .mp4 extension")
    if output.exists():
        parser.error(f"Output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    written_frames = 0
    writer = None
    try:
        with ExitStack() as cleanup:
            camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
            camera.Open()
            cleanup.callback(camera.Close)
            apply_camera_config(camera, args.config)
            fps = args.fps if args.fps is not None else camera_fps(camera)
            converter = pylon.ImageFormatConverter()
            converter.OutputPixelFormat = pylon.PixelType_BGR8packed
            converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            cleanup.callback(cv2.destroyAllWindows)
            camera.StartGrabbing(pylon.GrabStrategy_OneByOne)
            cleanup.callback(lambda: camera.StopGrabbing() if camera.IsGrabbing() else None)
            print(f"Recording to: {output}")
            print(f"Saved video frame rate: {fps:g} fps. Q / Esc or close window to stop.")
            while camera.IsGrabbing():
                # Short polling keeps window-close and keyboard events responsive.
                result = camera.RetrieveResult(100, pylon.TimeoutHandling_Return)
                # TimeoutHandling_Return may return a non-None but invalid wrapper.
                if result is not None and result.IsValid():
                    try:
                        if not result.GrabSucceeded():
                            raise RuntimeError(f"Camera grab failed: {result.GetErrorDescription()}")
                        frame = converter.Convert(result).GetArray()
                        if writer is None:
                            height, width = frame.shape[:2]
                            writer = cv2.VideoWriter(
                                str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                (width, height),
                            )
                            cleanup.callback(writer.release)
                            if not writer.isOpened():
                                raise RuntimeError(f"Cannot create video: {output}")
                            scale = min(1200 / width, 800 / height, 1.0)
                            cv2.resizeWindow(WINDOW_NAME, max(1, round(width * scale)),
                                             max(1, round(height * scale)))
                        # Save full-resolution source pixels; no tracking or annotations.
                        writer.write(frame)
                        written_frames += 1
                        cv2.imshow(WINDOW_NAME, frame)
                    finally:
                        result.Release()
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                try:
                    if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                        break
                except cv2.error:
                    break
    except KeyboardInterrupt:
        print("Recording stopped.")
    finally:
        if written_frames:
            print(f"Saved: {output} ({written_frames} frames)")
        else:
            print("No video frames recorded.")


if __name__ == "__main__":
    main()
