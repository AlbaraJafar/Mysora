from __future__ import annotations

import threading
from typing import Optional

import cv2


class CameraManager:
    """
    Single-process camera manager.
    - Keeps a capture open for stability/latency
    - Reopens on repeated read failures
    """

    def __init__(self, camera_index: int = 0) -> None:
        self._lock = threading.Lock()
        self._camera_index = int(camera_index)
        self._cap: Optional[cv2.VideoCapture] = None
        self._failures = 0

    def _ensure_open(self) -> None:
        if self._cap is not None and self._cap.isOpened():
            return
        self._cap = cv2.VideoCapture(self._camera_index)
        # Best-effort hints; may be ignored depending on backend
        try:
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        self._failures = 0

    def set_camera(self, camera_index: int) -> None:
        with self._lock:
            self._camera_index = int(camera_index)
            if self._cap is not None:
                try:
                    self._cap.release()
                finally:
                    self._cap = None

    def read(self):
        with self._lock:
            self._ensure_open()
            assert self._cap is not None
            ret, frame = self._cap.read()
            if not ret or frame is None:
                self._failures += 1
                if self._failures >= 5:
                    try:
                        self._cap.release()
                    finally:
                        self._cap = None
                return None
            self._failures = 0
            return frame

    def close(self) -> None:
        with self._lock:
            if self._cap is not None:
                try:
                    self._cap.release()
                finally:
                    self._cap = None

