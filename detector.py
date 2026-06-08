import cv2
import mediapipe as mp
import numpy as np
import threading
import time
import urllib.request
from pathlib import Path
from typing import Optional, Callable

from config import APP_DIR
from gesture import (
    ensure_hand_model,
    detect_open_hand,
    detect_middle_finger,
    detect_timeout_sign,
    GestureTracker,
    HAND_CONNECTIONS,
)

# Tasks API aliases
_BaseOptions           = mp.tasks.BaseOptions
_PoseLandmarker        = mp.tasks.vision.PoseLandmarker
_PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
_HandLandmarker        = mp.tasks.vision.HandLandmarker
_HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
_RunningMode           = mp.tasks.vision.RunningMode
_PoseLandmark          = mp.tasks.vision.PoseLandmark
_POSE_CONNECTIONS      = mp.tasks.vision.PoseLandmarksConnections.POSE_LANDMARKS

POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)

_WEIGHTS = {
    "nose_height":    3.0,
    "shoulder_level": 1.0,
    "head_tilt":      1.0,
    "shoulder_width": 2.0,
    "nose_x_offset":  0.5,
}

# Low-light detection: mean grayscale brightness (0-255) below this, with no
# pose landmarks found, is treated as "the room is too dark to track you".
LOW_LIGHT_THRESHOLD     = 50
LOW_LIGHT_STREAK_FRAMES = 45       # ~3s @ 15fps of sustained darkness before alerting
LOW_LIGHT_COOLDOWN_SEC  = 5 * 60   # don't repeat the alert more than once per 5 minutes


def ensure_model() -> Path:
    p = APP_DIR / "pose_landmarker_lite.task"
    if not p.exists():
        APP_DIR.mkdir(exist_ok=True)
        print("Downloading pose model (~6 MB)...")
        urllib.request.urlretrieve(POSE_MODEL_URL, p)
        print("Pose model ready.")
    return p


# ── Pose helpers ──────────────────────────────────────────────────────────────

def _draw_pose(frame: np.ndarray, landmarks: list) -> None:
    h, w = frame.shape[:2]
    pts: dict = {}
    for idx, lm in enumerate(landmarks):
        vis = lm.visibility if lm.visibility is not None else 1.0
        if vis > 0.25:
            pts[idx] = (int(lm.x * w), int(lm.y * h))
            cv2.circle(frame, pts[idx], 4, (0, 255, 100), -1)
    for conn in _POSE_CONNECTIONS:
        a, b = conn.start, conn.end
        if a in pts and b in pts:
            cv2.line(frame, pts[a], pts[b], (0, 200, 255), 2)


def _draw_hands(frame: np.ndarray, hands_lm: list) -> None:
    h, w = frame.shape[:2]
    for lm_list in hands_lm:
        pts = {i: (int(lm_list[i].x * w), int(lm_list[i].y * h))
               for i in range(len(lm_list))}
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (255, 180, 0), 2)
        for pt in pts.values():
            cv2.circle(frame, pt, 3, (255, 220, 80), -1)


def extract_metrics(landmarks: list) -> Optional[dict]:
    lm         = landmarks
    nose       = lm[_PoseLandmark.NOSE]
    l_ear      = lm[_PoseLandmark.LEFT_EAR]
    r_ear      = lm[_PoseLandmark.RIGHT_EAR]
    l_shoulder = lm[_PoseLandmark.LEFT_SHOULDER]
    r_shoulder = lm[_PoseLandmark.RIGHT_SHOULDER]

    def vis(x) -> float:
        return x.visibility if x.visibility is not None else 1.0

    if vis(l_shoulder) < 0.4 or vis(r_shoulder) < 0.4:
        return None

    sw = abs(r_shoulder.x - l_shoulder.x)
    if sw < 0.05:
        return None

    mid_x = (l_shoulder.x + r_shoulder.x) / 2
    mid_y = (l_shoulder.y + r_shoulder.y) / 2

    return {
        "nose_height":    (mid_y - nose.y) / sw,
        "shoulder_level": (l_shoulder.y - r_shoulder.y) / sw,
        "head_tilt":      (l_ear.y - r_ear.y) / sw,
        "shoulder_width": sw,
        "nose_x_offset":  (nose.x - mid_x) / sw,
    }


def posture_score(current: dict, baseline: dict) -> float:
    total_w = total_d = 0.0
    for key, w in _WEIGHTS.items():
        if key in current and key in baseline:
            total_d += abs(current[key] - baseline[key]) * w
            total_w += w
    return total_d / total_w if total_w else 0.0


# ── Detector ──────────────────────────────────────────────────────────────────

class PostureDetector:
    CALIB_FRAMES = 30

    def __init__(
        self,
        on_bad_posture:    Optional[Callable] = None,
        on_frame:          Optional[Callable] = None,
        on_freeze_toggle:  Optional[Callable] = None,
        on_hell_toggle:    Optional[Callable] = None,
        on_timeout:        Optional[Callable] = None,
        on_low_light:      Optional[Callable] = None,
    ):
        self.on_bad_posture   = on_bad_posture   # (score: float)
        self.on_frame         = on_frame         # (frame_rgb, metrics)
        self.on_freeze_toggle = on_freeze_toggle # ()
        self.on_hell_toggle   = on_hell_toggle   # ()
        self.on_timeout       = on_timeout       # ()
        self.on_low_light     = on_low_light     # ()

        self.settings: dict              = {}
        self.active_profile: Optional[dict]  = None
        self.active_profile_name: Optional[str] = None

        self._running  = False
        self._thread: Optional[threading.Thread] = None

        self._last_notif  = 0.0
        self._last_check  = 0.0
        self._low_light_streak     = 0
        self._last_low_light_notif = 0.0
        self._timeout_until = 0.0        # epoch time when T-sign pause expires
        self.notifications_enabled = True # open-hand snooze toggle

        self._calib_active = False
        self._calib_buf: list = []
        self._calib_cb: Optional[Callable] = None

    # ── Public API ────────────────────────────────────────────────────────

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False

    def is_running(self) -> bool:
        return self._running

    def pause_for(self, seconds: float) -> None:
        """Suppress posture notifications for `seconds` seconds."""
        self._timeout_until = time.time() + seconds

    def is_paused(self) -> bool:
        return time.time() < self._timeout_until

    def pause_remaining(self) -> float:
        """Seconds remaining in current pause (0 if not paused)."""
        return max(0.0, self._timeout_until - time.time())

    def toggle_snooze(self) -> None:
        """Toggle open-hand notification snooze."""
        self.notifications_enabled = not self.notifications_enabled

    def resume(self) -> None:
        """Cancel any active pause and re-enable notifications."""
        self._timeout_until       = 0.0
        self.notifications_enabled = True

    def start_calibration(self, callback: Callable) -> None:
        self._calib_buf = []
        self._calib_cb  = callback
        self._calib_active = True

    def calibration_progress(self) -> float:
        return len(self._calib_buf) / self.CALIB_FRAMES

    def _check_low_light(self, frame: np.ndarray) -> None:
        """Track sustained darkness + no detected pose; alert the user once it
        looks like the room is too dark for tracking rather than a momentary
        occlusion."""
        gray       = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))

        if brightness >= LOW_LIGHT_THRESHOLD:
            self._low_light_streak = 0
            return

        self._low_light_streak += 1
        if self._low_light_streak < LOW_LIGHT_STREAK_FRAMES:
            return

        now = time.time()
        if (
            self.notifications_enabled
            and not self.is_paused()
            and now - self._last_low_light_notif >= LOW_LIGHT_COOLDOWN_SEC
        ):
            self._last_low_light_notif = now
            if self.on_low_light:
                self.on_low_light()

    # ── Detection loop ────────────────────────────────────────────────────

    def _loop(self) -> None:
        try:
            pose_path = ensure_model()
            hand_path = ensure_hand_model()
        except Exception as e:
            print(f"Model load failed: {e}")
            self._running = False
            return

        cam_idx = int(self.settings.get("camera_index", 0))
        cap = cv2.VideoCapture(cam_idx)
        if not cap.isOpened():
            self._running = False
            return

        # Build gesture trackers (callbacks set after window exists — via main.py)
        freeze_tracker  = GestureTracker(hold_sec=0.8, cooldown_sec=2.5,
                                         callback=self.on_freeze_toggle)
        hell_tracker    = GestureTracker(hold_sec=1.0, cooldown_sec=3.0,
                                         callback=self.on_hell_toggle)
        timeout_tracker = GestureTracker(hold_sec=1.5, cooldown_sec=35 * 60,
                                         callback=self.on_timeout)

        pose_opts = _PoseLandmarkerOptions(
            base_options=_BaseOptions(model_asset_path=str(pose_path)),
            running_mode=_RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        hand_opts = _HandLandmarkerOptions(
            base_options=_BaseOptions(model_asset_path=str(hand_path)),
            running_mode=_RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        start_ts   = time.time()
        frame_num  = 0

        with (_PoseLandmarker.create_from_options(pose_opts) as pose_lm,
              _HandLandmarker.create_from_options(hand_opts) as hand_lm):

            while self._running:
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.05)
                    continue

                frame_num += 1
                rgb          = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                ts_ms        = int((time.time() - start_ts) * 1000)
                mp_image     = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                # ── Pose detection (every frame) ─────────────────────────
                pose_result = pose_lm.detect_for_video(mp_image, ts_ms)
                annotated   = rgb.copy()
                metrics     = None

                if pose_result.pose_landmarks:
                    lm_list = pose_result.pose_landmarks[0]
                    _draw_pose(annotated, lm_list)
                    metrics = extract_metrics(lm_list)
                    self._low_light_streak = 0
                else:
                    self._check_low_light(frame)

                # ── Hand detection (every 3rd frame ≈ 5 fps) ────────────
                if frame_num % 3 == 0:
                    hand_result = hand_lm.detect_for_video(mp_image, ts_ms)
                    hands = hand_result.hand_landmarks if hand_result.hand_landmarks else []

                    if hands:
                        _draw_hands(annotated, hands)

                    # Update gesture trackers
                    open_detected   = any(detect_open_hand(h)     for h in hands)
                    middle_detected = any(detect_middle_finger(h)  for h in hands)
                    timeout_detected = detect_timeout_sign(hands)

                    freeze_tracker.callback  = self.on_freeze_toggle
                    hell_tracker.callback    = self.on_hell_toggle
                    timeout_tracker.callback = self.on_timeout

                    freeze_tracker.update(open_detected)
                    hell_tracker.update(middle_detected)
                    timeout_tracker.update(timeout_detected)

                # ── Calibration ──────────────────────────────────────────
                if self._calib_active and metrics is not None:
                    self._calib_buf.append(metrics)
                    if len(self._calib_buf) >= self.CALIB_FRAMES:
                        avg = {
                            k: float(np.mean([f[k] for f in self._calib_buf]))
                            for k in self._calib_buf[0]
                        }
                        self._calib_active = False
                        if self._calib_cb:
                            self._calib_cb(avg)

                # ── Frame callback ───────────────────────────────────────
                if self.on_frame:
                    self.on_frame(annotated, metrics)

                # ── Posture check (throttled, respects pause) ────────────
                now            = time.time()
                check_interval = self.settings.get("check_interval", 5)

                if (
                    not self._calib_active
                    and self.active_profile
                    and metrics
                    and not self.is_paused()
                    and self.notifications_enabled
                    and now - self._last_check >= check_interval
                ):
                    self._last_check = now
                    score       = posture_score(metrics, self.active_profile)
                    sensitivity = self.settings.get("sensitivity", 0.20)
                    cooldown    = self.settings.get("notification_cooldown", 300)

                    if score > sensitivity and now - self._last_notif >= cooldown:
                        self._last_notif = now
                        if self.on_bad_posture:
                            self.on_bad_posture(score)

                time.sleep(1 / 15)

        cap.release()
