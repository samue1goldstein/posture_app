"""
Hand gesture detection for posture v0.3 Easter eggs.

Gestures:
  - Open hand  (all fingers extended)   → freeze/unfreeze frame
  - Middle finger (only middle up)       → toggle hell theme
  - Timeout sign (T with two hands)      → pause alerts 30 min
"""

import math
import time
import urllib.request
from pathlib import Path
from typing import Optional, Callable

import mediapipe as mp

from config import APP_DIR

_HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)

# Hand connections for drawing (index pairs)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),           # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),           # index
    (0, 9), (9, 10), (10, 11), (11, 12),      # middle
    (0, 13), (13, 14), (14, 15), (15, 16),    # ring
    (0, 17), (17, 18), (18, 19), (19, 20),    # pinky
    (5, 9), (9, 13), (13, 17),                # palm
]


def ensure_hand_model() -> Path:
    p = APP_DIR / "hand_landmarker.task"
    if not p.exists():
        APP_DIR.mkdir(exist_ok=True)
        print("Downloading hand landmark model (~8 MB)...")
        urllib.request.urlretrieve(_HAND_MODEL_URL, p)
        print("Hand model ready.")
    return p


# ── Per-finger helpers ────────────────────────────────────────────────────────

def _extended(lm: list, tip: int, pip: int) -> bool:
    """True when fingertip is above the PIP joint (finger is extended)."""
    return lm[tip].y < lm[pip].y


# ── Single-hand gesture detectors ────────────────────────────────────────────

def detect_open_hand(lm: list) -> bool:
    """All four fingers extended (thumb ignored — unreliable from front-cam)."""
    return all(_extended(lm, t, p) for t, p in [(8, 6), (12, 10), (16, 14), (20, 18)])


def detect_middle_finger(lm: list) -> bool:
    """Only middle finger up; index, ring and pinky curled."""
    return (
        _extended(lm, 12, 10) and          # middle up
        not _extended(lm, 8,  6) and       # index down
        not _extended(lm, 16, 14) and      # ring down
        not _extended(lm, 20, 18)          # pinky down
    )


# ── Two-hand gesture detector ─────────────────────────────────────────────────

def _hand_angle(lm: list) -> float:
    """Degrees from wrist → middle-MCP.  90 = pointing up, 0 = pointing right."""
    wrist   = lm[0]
    mid_mcp = lm[9]
    dx =  (mid_mcp.x - wrist.x)
    dy = -(mid_mcp.y - wrist.y)   # invert y (screen coords)
    return math.degrees(math.atan2(dy, dx))


def detect_timeout_sign(lms_list: list) -> bool:
    """T sign: one hand roughly horizontal, the other roughly vertical."""
    if len(lms_list) < 2:
        return False
    a, b = _hand_angle(lms_list[0]), _hand_angle(lms_list[1])

    def horiz(x: float) -> bool:
        return abs(x) < 38 or abs(x) > 142

    def vert(x: float) -> bool:
        return 52 < abs(x) < 128

    return (horiz(a) and vert(b)) or (vert(a) and horiz(b))


# ── Hold-to-trigger tracker ───────────────────────────────────────────────────

class GestureTracker:
    """Fires `callback` once a gesture has been held for `hold_sec`.
    Enforces `cooldown_sec` between successive firings."""

    def __init__(
        self,
        hold_sec: float = 0.9,
        cooldown_sec: float = 3.0,
        callback: Optional[Callable] = None,
    ):
        self.hold_sec     = hold_sec
        self.cooldown_sec = cooldown_sec
        self.callback     = callback

        self._hold_start: Optional[float] = None
        self._last_fire   = 0.0
        self._fired_this_hold = False

    def update(self, detected: bool) -> None:
        now = time.time()
        if detected:
            if self._hold_start is None:
                self._hold_start = now
                self._fired_this_hold = False
            elif (
                not self._fired_this_hold
                and now - self._hold_start   >= self.hold_sec
                and now - self._last_fire    >= self.cooldown_sec
            ):
                self._last_fire = now
                self._fired_this_hold = True
                if self.callback:
                    self.callback()
        else:
            self._hold_start      = None
            self._fired_this_hold = False
