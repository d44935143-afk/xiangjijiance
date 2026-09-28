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

        cv2.namedWindow(self.monitor_window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.monitor_window, *DEFAULT_MONITOR_WINDOW_SIZE)
        cv2.namedWindow(self.displacement_window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(
            self.displacement_window,
            realtime_displacement_display.DEFAULT_WINDOW_WIDTH,
            realtime_displacement_display.DEFAULT_WINDOW_HEIGHT,
        )

    def show(self, frame, state, frame_name="monitor"):
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

        cv2.imshow(self.monitor_window, monitor_frame)
        cv2.imshow(self.displacement_window, displacement_frame)

    def should_exit(self):
        key = cv2.waitKey(1) & 0xFF
        return key in (ord("q"), 27)

    def close(self):
        cv2.destroyAllWindows()
