"""High-level wrist camera abstraction for the SO-101 USB wrist camera.

Wraps cv2 capture, v4l2 manual-mode setup, an optional adaptive-brightness
controller, and ordered shutdown. The shutdown ordering (stop AE thread →
join → release cap → reset v4l2 to auto) matters: a naive sequence races
and can leave the next consumer with a stuck-manual camera.
"""
import os
import threading

import cv2
import numpy as np

from .wrist_v4l2 import set_v4l2, reset_wrist_to_auto


FLIP_CHOICES = ["v", "h", "180", "none"]
_FLIP_CODES = {"v": 0, "h": 1, "180": -1}


class WristCamera:
    def __init__(self, cam_id, flip="180", enable_ae=False):
        self.cam_id = cam_id
        self.flip_code = _FLIP_CODES.get(flip)  # None for "none"
        self._cap = None
        self._ae = None

        dev = f"/dev/video{cam_id}"
        if not os.path.exists(dev):
            present = sorted(p for p in os.listdir("/dev") if p.startswith("video"))
            raise FileNotFoundError(
                f"{dev} does not exist — wrist camera not connected. "
                f"Available video devices: {present}. "
                f"Check with `v4l2-ctl --list-devices`."
            )
        cap = cv2.VideoCapture(cam_id)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open wrist camera at index {cam_id}")
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        # Lock white balance to a fixed temperature — auto WB drift between
        # sessions can collapse predicted action deltas to near zero.
        set_v4l2(
            cam_id,
            white_balance_automatic=0,
            white_balance_temperature=4600,
            auto_exposure=1,
            brightness=32,
        )
        self._cap = cap
        if enable_ae:
            self._ae = _AEController(cam_id)

    def read(self):
        self._cap.grab()  # flush stale buffered frame
        ok, img = self._cap.read()
        if not ok:
            return None
        if self.flip_code is not None:
            img = cv2.flip(img, self.flip_code)
        return img

    def update_observations(self, *, wrist_bgr=None, scene_bgr=None):
        if self._ae is not None:
            self._ae.update_observations(wrist_bgr=wrist_bgr, scene_bgr=scene_bgr)

    def close(self):
        if self._cap is None:
            return
        if self._ae is not None:
            self._ae.stop()
            self._ae.join(timeout=2.0)
            self._ae = None
        self._cap.release()
        self._cap = None
        reset_wrist_to_auto(self.cam_id)


class _AEController:
    """Adaptive brightness controller for cameras whose exposure_time_absolute
    is non-functional. Nudges the `brightness` v4l2 control on a slow loop to
    track a target intensity, optionally matching a second camera's scene mean so
    both cameras stay matched as room lighting changes.
    """

    BRIGHTNESS_MIN, BRIGHTNESS_MAX = -32, 64
    TARGET_MIN, TARGET_MAX = 40.0, 100.0

    def __init__(self, cam_id, period_s=1.0):
        self.cam_id = cam_id
        self.period = period_s
        self.brightness = 32
        self.target = 70.0
        self._wrist_mean = None
        self._scene_mean = None
        self._lock = threading.Lock()
        self._stop_evt = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def update_observations(self, *, wrist_bgr=None, scene_bgr=None):
        with self._lock:
            if wrist_bgr is not None:
                self._wrist_mean = float(np.mean(wrist_bgr))
            if scene_bgr is not None:
                self._scene_mean = float(np.mean(scene_bgr))

    def _loop(self):
        while not self._stop_evt.wait(self.period):
            with self._lock:
                wrist_mean = self._wrist_mean
                scene_mean = self._scene_mean
            if wrist_mean is None:
                continue
            if scene_mean is not None:
                self.target = float(np.clip(scene_mean, self.TARGET_MIN, self.TARGET_MAX))
            err = self.target - wrist_mean
            if abs(err) < 4.0:
                continue
            step = int(np.clip(np.sign(err) * max(1, abs(err) / 4), -8, 8))
            new_b = int(np.clip(self.brightness + step,
                                self.BRIGHTNESS_MIN, self.BRIGHTNESS_MAX))
            if new_b != self.brightness:
                self.brightness = new_b
                set_v4l2(self.cam_id, brightness=new_b)
                print(f"[Wrist AE] mean={wrist_mean:.1f} target={self.target:.1f} "
                      f"→ brightness={new_b}")

    def stop(self):
        self._stop_evt.set()

    def join(self, timeout=2.0):
        self._thread.join(timeout=timeout)
