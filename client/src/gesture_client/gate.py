from __future__ import annotations

import time


class GestureGate:
    def __init__(self, threshold: float = 0.60, stable_frames: int = 4,
                 cooldown_seconds: float = 3.5):
        self.threshold = threshold
        self.stable_frames = stable_frames
        self.cooldown_seconds = cooldown_seconds
        self._label: str | None = None
        self._count = 0
        self._last_trigger = float("-inf")
        self._latched: str | None = None

    def reset(self) -> None:
        self._label, self._count, self._latched = None, 0, None

    def observe(self, label: str | None, confidence: float, arm_busy: bool,
                now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        if label is None or confidence < self.threshold:
            self.reset()
            return False
        if label != self._label:
            self._label, self._count, self._latched = label, 1, None
        else:
            self._count += 1
        if self._latched == label or self._count < self.stable_frames or arm_busy:
            return False
        if now - self._last_trigger < self.cooldown_seconds:
            return False
        self._last_trigger, self._latched = now, label
        return True

