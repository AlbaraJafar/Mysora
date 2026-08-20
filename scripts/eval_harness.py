"""
Automated evaluation harness for Mysora letter recognition.
Runs stored test clips through the model and reports per-letter
accuracy, confusion matrix, and latency stats.

Clip storage format expected:
  clips_dir/{label}/*.json
  Each JSON must have at minimum:
    - "frame_data": base64-encoded JPEG/PNG image string
    - "label": expected letter label (e.g. "Beh", "Alef")

Run from the project root:
  python scripts/eval_harness.py
"""
from __future__ import annotations

import base64
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

# Allow running from scripts/ or from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.hand_crop import prepare_for_inference
from scripts.inference import predict_proba


def _decode_frame(frame_data: str) -> np.ndarray:
    """Decode a base64 image string to a BGR numpy array."""
    raw = base64.b64decode(frame_data)
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image from frame_data")
    return img


def load_test_clips(clips_dir: Path) -> dict:
    """
    Load stored test clips grouped by label.
    Expects: clips_dir/{label}/*.json
    """
    test_data: dict = defaultdict(list)
    if not clips_dir.exists():
        return test_data

    for label_dir in clips_dir.iterdir():
        if label_dir.is_dir():
            label = label_dir.name
            for clip_file in label_dir.glob("*.json"):
                try:
                    with open(clip_file, encoding="utf-8") as f:
                        test_data[label].append(json.load(f))
                except Exception:
                    pass

    return test_data


def run_evaluation(clips_dir: Path | None = None) -> dict:
    """
    Run full evaluation across all available test clips.
    Returns per-letter accuracy, confusion matrix, and latency percentiles.
    """
    if clips_dir is None:
        import os
        clips_dir = Path(os.environ.get("DATA_DIR", "outputs")) / "clips"

    test_data = load_test_clips(clips_dir)

    if not test_data:
        return {
            "status": "no_data",
            "message": f"No test clips found in {clips_dir}",
        }

    per_letter: dict = {}
    confusion: dict = defaultdict(lambda: defaultdict(int))
    all_latencies: list = []

    for true_label, clips in test_data.items():
        correct = 0
        latencies: list = []

        for clip in clips:
            try:
                frame_data = clip.get("frame_data", "")
                if not frame_data:
                    continue
                frame = _decode_frame(frame_data)
            except Exception:
                continue

            try:
                start = time.perf_counter()
                gray224, _ = prepare_for_inference(frame)
                labels, probs = predict_proba(gray224)
                elapsed = time.perf_counter() - start
            except Exception:
                continue

            predicted_label = labels[int(probs.argmax())]
            latencies.append(elapsed)
            all_latencies.append(elapsed)
            confusion[true_label][predicted_label] += 1

            if predicted_label == true_label:
                correct += 1

        total = len(latencies)
        per_letter[true_label] = {
            "accuracy": correct / total if total > 0 else 0.0,
            "samples": total,
            "latency_p50_ms": float(np.percentile(latencies, 50) * 1000) if latencies else 0.0,
            "latency_p95_ms": float(np.percentile(latencies, 95) * 1000) if latencies else 0.0,
        }

    total_weighted = sum(r["accuracy"] * r["samples"] for r in per_letter.values())
    total_samples = sum(r["samples"] for r in per_letter.values())
    overall_accuracy = total_weighted / total_samples if total_samples > 0 else 0.0

    return {
        "status": "ok",
        "overall_accuracy": overall_accuracy,
        "per_letter": per_letter,
        "confusion_matrix": {k: dict(v) for k, v in confusion.items()},
        "latency_p50_ms": float(np.percentile(all_latencies, 50) * 1000) if all_latencies else 0.0,
        "latency_p95_ms": float(np.percentile(all_latencies, 95) * 1000) if all_latencies else 0.0,
        "latency_p99_ms": float(np.percentile(all_latencies, 99) * 1000) if all_latencies else 0.0,
        "total_samples": total_samples,
        "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


if __name__ == "__main__":
    results = run_evaluation()
    print(json.dumps(results, indent=2, ensure_ascii=False))
