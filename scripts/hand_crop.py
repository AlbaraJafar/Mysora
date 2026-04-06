"""
MediaPipe hand detection and crop pipeline for gesture recognition.
Uses MediaPipe Tasks (HandLandmarker). Crops hand with 20% margin, grayscale,
resize to 224x224. Fallback to full frame if no hand.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

# Lazy init
_landmarker = None

TARGET_SIZE = (224, 224)
MARGIN_FRAC = 0.20
HAND_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


def _get_model_path() -> str:
    """Return path to hand_landmarker.task, downloading if missing."""
    env_path = os.environ.get("HAND_LANDMARKER_MODEL")
    if env_path and os.path.isfile(env_path):
        return env_path
    script_dir = Path(__file__).resolve().parent
    model_path = script_dir / "hand_landmarker.task"
    if model_path.is_file():
        return str(model_path)
    try:
        import urllib.request
        urllib.request.urlretrieve(HAND_LANDMARKER_URL, model_path)
        return str(model_path)
    except Exception as e:
        raise FileNotFoundError(
            f"Hand landmarker model not found at {model_path} and download failed: {e}. "
            f"Download manually from {HAND_LANDMARKER_URL} and place at {model_path}"
        ) from e


def _get_landmarker():
    global _landmarker
    if _landmarker is None:
        from mediapipe.tasks.python.core import base_options
        from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
        from mediapipe.tasks.python.vision.core import vision_task_running_mode

        model_path = _get_model_path()
        base_opts = base_options.BaseOptions(model_asset_path=model_path)
        options = HandLandmarkerOptions(
            base_options=base_opts,
            running_mode=vision_task_running_mode.VisionTaskRunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.3,
        )
        _landmarker = HandLandmarker.create_from_options(options)
    return _landmarker


def _landmarks_to_bbox(
    landmarks: List,  # List of NormalizedLandmark (x, y in [0,1])
    image_width: int,
    image_height: int,
    margin_frac: float = MARGIN_FRAC,
) -> Optional[Tuple[int, int, int, int]]:
    """(x_min, y_min, x_max, y_max) in pixel coords with margin. Returns None if invalid."""
    xs = [lm.x * image_width for lm in landmarks if lm.x is not None]
    ys = [lm.y * image_height for lm in landmarks if lm.y is not None]
    if not xs or not ys:
        return None
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    w = x_max - x_min
    h = y_max - y_min
    if w <= 0 or h <= 0:
        return None
    margin_w = w * margin_frac
    margin_h = h * margin_frac
    x_min = max(0, int(x_min - margin_w))
    y_min = max(0, int(y_min - margin_h))
    x_max = min(image_width, int(x_max + margin_w))
    y_max = min(image_height, int(y_max + margin_h))
    if x_max <= x_min or y_max <= y_min:
        return None
    return (x_min, y_min, x_max, y_max)


def prepare_for_inference(frame: np.ndarray) -> Tuple[np.ndarray, bool]:
    """
    Detect hand with MediaPipe Tasks, crop with 20% margin, grayscale, resize to 224x224.
    If no hand is detected, use full frame (grayscale, resize to 224x224).
    Frame is BGR (OpenCV convention).
    Returns (grayscale 224×224 uint8, hand_detected).
    """
    if frame is None or frame.size == 0:
        raise ValueError("frame is empty")
    h, w = frame.shape[:2]
    if w == 0 or h == 0:
        raise ValueError("frame has zero width or height")

    from mediapipe.tasks.python.vision.core import image as mp_image

    landmarker = _get_landmarker()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = np.ascontiguousarray(rgb)
    mp_img = mp_image.Image(image_format=mp_image.ImageFormat.SRGB, data=rgb)

    result = landmarker.detect(mp_img)

    crop = None
    hand_detected = False
    if result.hand_landmarks and len(result.hand_landmarks) > 0:
        landmarks = result.hand_landmarks[0]
        bbox = _landmarks_to_bbox(landmarks, w, h, MARGIN_FRAC)
        if bbox is not None:
            x_min, y_min, x_max, y_max = bbox
            crop = frame[y_min:y_max, x_min:x_max]
            hand_detected = crop is not None and crop.size > 0

    if crop is None or crop.size == 0:
        crop = frame
        hand_detected = False

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    return resized, hand_detected
