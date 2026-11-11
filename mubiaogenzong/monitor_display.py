import cv2

import continuous_tracking
import realtime_displacement_display


DEFAULT_MONITOR_WINDOW = "Target Monitoring"
DEFAULT_DISPLACEMENT_WINDOW = "Displacement mm"
DEFAULT_MONITOR_WINDOW_SIZE = (960, 540)


class MonitorDisplay:
    """Manage monitoring image windows."""

    def __init__(
        self,
        monitor_window=DEFAULT_MONITOR_WINDOW,
        displacement_window=DEFAULT_DISPLACEMENT_WINDOW,
        max_samples=realtime_displacement_display.DEFAULT_MAX_SAMPLES,
    ):
        self.monitor_window = monitor_window
        self.displacement_window = displacement_window
        self.displacement_history = []
        self.max_samples = max_samples
        self.monitor_frame_size = None
        self.zoom = 1.0
        self.view_x = 0.0
        self.view_y = 0.0

        cv2.namedWindow(self.monitor_window, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.setMouseCallback(self.monitor_window, self._on_monitor_mouse)
        cv2.namedWindow(self.displacement_window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(
            self.displacement_window,
            realtime_displacement_display.DEFAULT_WINDOW_WIDTH,
            realtime_displacement_display.DEFAULT_WINDOW_HEIGHT,
        )

    def _on_monitor_mouse(self, event, x, y, flags, param):
        if self.monitor_frame_size is None:
            return
        if event == cv2.EVENT_LBUTTONDBLCLK:
            self.zoom = 1.0
            self.view_x = self.view_y = 0.0
            return
        if event != cv2.EVENT_MOUSEWHEEL:
            return

        # HighGUI encodes the signed wheel delta in the upper 16 bits.
        delta = (flags >> 16) & 0xFFFF
        if delta >= 0x8000:
            delta -= 0x10000
        if delta == 0:
            return
        width, height = self.monitor_frame_size
        mouse_x = min(max(x / width, 0.0), 1.0)
        mouse_y = min(max(y / height, 0.0), 1.0)
        old_zoom = self.zoom
        new_zoom = min(16.0, max(1.0, old_zoom * (1.25 if delta > 0 else 0.8)))
        anchor_x = self.view_x + mouse_x * width / old_zoom
        anchor_y = self.view_y + mouse_y * height / old_zoom
        self.view_x = min(max(anchor_x - mouse_x * width / new_zoom, 0.0),
                          width - width / new_zoom)
        self.view_y = min(max(anchor_y - mouse_y * height / new_zoom, 0.0),
                          height - height / new_zoom)
        self.zoom = new_zoom

    def _zoom_monitor_frame(self, frame):
        if self.zoom == 1.0:
            return frame
        height, width = frame.shape[:2]
        crop_width = max(1, round(width / self.zoom))
        crop_height = max(1, round(height / self.zoom))
        left = min(max(0, round(self.view_x)), width - crop_width)
        top = min(max(0, round(self.view_y)), height - crop_height)
        crop = frame[top:top + crop_height, left:left + crop_width]
        return cv2.resize(crop, (width, height), interpolation=cv2.INTER_LINEAR)

    def show(self, frame, state, frame_name="monitor"):
        height, width = frame.shape[:2]
        if self.monitor_frame_size != (width, height):
            max_width, max_height = DEFAULT_MONITOR_WINDOW_SIZE
            scale = min(max_width / width, max_height / height, 1.0)
            cv2.resizeWindow(
                self.monitor_window,
                max(1, round(width * scale)),
                max(1, round(height * scale)),
            )
            self.monitor_frame_size = (width, height)
            self.zoom = 1.0
            self.view_x = self.view_y = 0.0
        monitor_frame = continuous_tracking.draw_tracking(
            frame,
            state.get("targets", []),
            frame_name,
        )
        realtime_displacement_display.append_history(
            self.displacement_history,
            state,
            max_samples=self.max_samples,
        )
        displacement_frame = realtime_displacement_display.draw_time_axis_dashboard(
            self.displacement_history,
            state,
        )

        cv2.imshow(self.monitor_window, self._zoom_monitor_frame(monitor_frame))
        cv2.imshow(self.displacement_window, displacement_frame)

    def should_exit(self):
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            return True
        for window_name in (self.monitor_window, self.displacement_window):
            try:
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    return True
            except cv2.error:
                return True
        return False

    def close(self):
        cv2.destroyAllWindows()
