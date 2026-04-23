from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional
import threading
import time

import numpy as np


@dataclass(frozen=True)
class StablePrediction:
    raw_label: str
    raw_confidence: float
    stable_label: Optional[str]
    stable_confidence: float
    margin: float


Tier = Literal["fast", "normal"]


class GestureStabilizer:
    """
    Fast, lightweight stabilization (~3 effective frames of EMA, no history buffer):
    - EMA over class probabilities with alpha ≈ 0.5 (short memory)
    - Raw confidence tiers:
        < 0.48  → never emit (ignore)
        0.48–0.72 → normal stable window + cooldown
        > 0.72  → shorter window + shorter cooldown
    - Release gate after emit to prevent duplicate letters on a held pose
    - EMA top-1 vs raw top-1 disagreement slows acceptance (reduces rapid wrong switching)
    """

    def __init__(
        self,
        *,
        num_classes: int,
        ema_alpha: float = 0.5,
        release_ms_fast: int = 140,
        release_ms_normal: int = 200,
    ) -> None:
        # alpha=0.5 → ~2–3 steps for EMA to track the signal (matches “max 3 frames” intent)
        self._lock = threading.Lock()
        self._ema = np.full((num_classes,), 1.0 / num_classes, dtype=np.float32)
        self._alpha = float(ema_alpha)

        self._release_ms_fast = int(release_ms_fast)
        self._release_ms_normal = int(release_ms_normal)

        self._last_emitted: Optional[str] = None
        self._cooldown_until = 0.0
        self._release_required = False
        self._release_ms_effective = self._release_ms_normal

        self._candidate_label: Optional[str] = None
        self._candidate_since = 0.0
        self._not_candidate_since = 0.0

        self._last_raw_idx: int = -1
        self._same_raw_streak: int = 0

    def reset(self) -> None:
        with self._lock:
            self._ema[:] = 1.0 / self._ema.size
            self._last_emitted = None
            self._cooldown_until = 0.0
            self._release_required = False
            self._release_ms_effective = self._release_ms_normal
            self._candidate_label = None
            self._candidate_since = 0.0
            self._not_candidate_since = 0.0
            self._last_raw_idx = -1
            self._same_raw_streak = 0

    @staticmethod
    def _confidence_tier(raw_conf: float) -> Optional[Tier]:
        if raw_conf < 0.48:
            return None
        if raw_conf > 0.72:
            return "fast"
        return "normal"

    def _tier_params(self, tier: Tier, raw_conf: float, margin: float, streak: int) -> tuple[int, int, float, float]:
        """stable_ms, cooldown_ms, min_margin, min_ema_top"""
        if tier == "fast":
            stable_ms = 95
            cooldown_ms = 155
            min_margin = 0.065
            min_ema = 0.48
        else:
            stable_ms = 190
            cooldown_ms = 250
            min_margin = 0.08
            min_ema = 0.47

        # Same raw class for 3+ inference steps → slightly faster (still capped for stability)
        if streak >= 3:
            stable_ms = max(70, int(stable_ms * 0.78))
            cooldown_ms = max(120, int(cooldown_ms * 0.82))

        # Very confident raw signal → shave a bit more off delay
        if raw_conf >= 0.88:
            stable_ms = max(65, int(stable_ms * 0.88))
            cooldown_ms = max(115, int(cooldown_ms * 0.88))

        if margin >= 0.22:
            stable_ms = max(60, int(stable_ms * 0.9))

        return stable_ms, cooldown_ms, min_margin, min_ema

    def update(self, labels: list[str], probs: np.ndarray) -> StablePrediction:
        probs = np.asarray(probs, dtype=np.float32)
        if probs.ndim != 1 or probs.size != self._ema.size:
            raise ValueError("probs must be a 1D array aligned with model classes")

        raw_idx = int(probs.argmax())
        raw_label = labels[raw_idx]
        raw_conf = float(probs[raw_idx])

        now = time.monotonic()

        with self._lock:
            if raw_idx == self._last_raw_idx:
                self._same_raw_streak += 1
            else:
                self._same_raw_streak = 1
                self._last_raw_idx = raw_idx

            self._ema = (1.0 - self._alpha) * self._ema + self._alpha * probs

            order = np.argsort(self._ema)
            top1 = int(order[-1])
            top2 = int(order[-2]) if order.size >= 2 else top1
            stable_label = labels[top1]
            stable_conf = float(self._ema[top1])
            margin = float(self._ema[top1] - self._ema[top2])

            tier = self._confidence_tier(raw_conf)

            # Below threshold: do not accumulate acceptance time (reduces junk emissions)
            if tier is None:
                self._candidate_since = now
                self._candidate_label = stable_label

            elif stable_label != self._candidate_label:
                self._candidate_label = stable_label
                self._candidate_since = now

            if self._release_required:
                if stable_label == self._last_emitted:
                    self._not_candidate_since = now
                else:
                    if self._not_candidate_since == 0.0:
                        self._not_candidate_since = now
                    if (now - self._not_candidate_since) * 1000.0 >= self._release_ms_effective:
                        self._release_required = False

            emit: Optional[str] = None

            if tier is not None:
                stable_ms, cooldown_ms, min_margin, min_ema = self._tier_params(
                    tier, raw_conf, margin, self._same_raw_streak
                )

                # EMA vs raw disagreement: wait longer to avoid rapid class flapping
                disagree = stable_label != raw_label
                if disagree:
                    stable_ms = int(stable_ms * 1.45)
                    min_margin = min_margin + 0.04

                # Normal tier: require at least 2 consistent raw frames before accepting
                min_streak = 1 if tier == "fast" else 2
                streak_ok = self._same_raw_streak >= min_streak

                stable_enough = ((now - self._candidate_since) * 1000.0) >= stable_ms
                allowed_by_cooldown = now >= self._cooldown_until

                if (
                    streak_ok
                    and stable_conf >= min_ema
                    and margin >= min_margin
                    and stable_enough
                    and allowed_by_cooldown
                    and not self._release_required
                ):
                    if stable_label != self._last_emitted:
                        emit = stable_label
                        self._last_emitted = stable_label
                        self._cooldown_until = now + (cooldown_ms / 1000.0)
                        self._release_required = True
                        self._not_candidate_since = 0.0
                        self._release_ms_effective = (
                            self._release_ms_fast if tier == "fast" else self._release_ms_normal
                        )

        return StablePrediction(
            raw_label=raw_label,
            raw_confidence=raw_conf,
            stable_label=emit,
            stable_confidence=stable_conf,
            margin=margin,
        )
