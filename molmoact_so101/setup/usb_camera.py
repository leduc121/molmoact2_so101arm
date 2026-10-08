"""Small, hardware-agnostic OpenCV capture layer for USB/V4L2 cameras."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Union

import cv2
import numpy as np

FLIP_CHOICES = ("none", "h", "v", "180")
_FLIPS = {"h": 1, "v": 0, "180": -1, "none": None}


def camera_source(value: str) -> Union[int, str]:
    """Accept an OpenCV numeric index or an explicit /dev path only."""
    if value.isdigit():
        return int(value)
    if value.startswith("/dev/"):
        return value
    raise ValueError("camera must be a numeric OpenCV index or an absolute /dev/... path")


@dataclass(frozen=True)
class CameraConfig:
    device: Union[int, str]
    width: int = 640
    height: int = 480
    fps: float = 15.0
    rotation: int = 0
    flip: str = "none"
    buffer_size: int = 1
    reconnect_attempts: int = 3

    def __post_init__(self):
        if self.width <= 0 or self.height <= 0 or self.fps <= 0:
            raise ValueError("camera width, height, and fps must be positive")
        if self.rotation not in (0, 90, 180, 270):
            raise ValueError("rotation must be one of 0, 90, 180, 270")
        if self.flip not in FLIP_CHOICES:
            raise ValueError(f"flip must be one of {FLIP_CHOICES}")


@dataclass(frozen=True)
class CameraFrame:
    bgr: np.ndarray
    captured_at: float


class UsbCamera:
    """Synchronous latest-frame capture; no queue means no intentional backlog."""
    def __init__(self, config: CameraConfig, name: str = "camera"):
        self.config, self.name, self._cap = config, name, None
        self._failures = 0
        self.open()

    def _source_exists(self) -> bool:
        return not isinstance(self.config.device, str) or os.path.exists(self.config.device)

    def open(self) -> None:
        if not self._source_exists():
            raise FileNotFoundError(f"{self.name} device not found: {self.config.device}")
        cap = cv2.VideoCapture(self.config.device, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"could not open {self.name} camera: {self.config.device}")
        cap.set(cv2.CAP_PROP_BUFFERSIZE, self.config.buffer_size)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        cap.set(cv2.CAP_PROP_FPS, self.config.fps)
        self._cap = cap

    def _transform(self, image: np.ndarray) -> np.ndarray:
        if self.config.rotation == 90:
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif self.config.rotation == 180:
            image = cv2.rotate(image, cv2.ROTATE_180)
        elif self.config.rotation == 270:
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        code = _FLIPS[self.config.flip]
        return cv2.flip(image, code) if code is not None else image

    def read(self) -> CameraFrame | None:
        if self._cap is None:
            return None
        self._cap.grab()  # discard one buffered frame when the backend supports it
        ok, image = self._cap.read()
        if ok and image is not None:
            self._failures = 0
            return CameraFrame(self._transform(image), time.monotonic())
        self._failures += 1
        if self._failures <= self.config.reconnect_attempts:
            self.close()
            try:
                self.open()
            except (OSError, RuntimeError):
                pass
        return None

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

