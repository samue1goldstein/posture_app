import base64
import threading
import time
from pathlib import Path
from typing import Optional, Callable

import cv2
import numpy as np
import webview

from profiles import ProfileManager
from detector import PostureDetector, posture_score
from startup import is_startup_enabled, set_startup
import config


class PostureAPI:
    """Exposed to the webview as window.pywebview.api — all methods callable from JS."""

    def __init__(
        self,
        detector: PostureDetector,
        pm: ProfileManager,
        settings: dict,
        hell_mode_ref: list,
    ):
        self.detector        = detector
        self.pm              = pm
        self.settings        = settings
        self._hell_mode_ref  = hell_mode_ref

        self._hell_mode          = False
        self._frozen             = False
        self._latest_frame_b64: Optional[str] = None
        self._latest_metrics     = None

        self._calib_in_progress  = False
        self._calib_polls        = 0
        self._calib_done_name: Optional[str] = None

        self._win: Optional[webview.Window]                = None
        self.on_tray_state_change: Optional[Callable]      = None
        self._prev_paused_state: Optional[bool]            = None

        self.detector.on_frame  = self._on_frame
        self.detector.settings  = self.settings

        self._tick = threading.Thread(target=self._tick_loop, daemon=True)
        self._tick.start()

    def _set_window(self, win: webview.Window) -> None:
        self._win = win

    def _push(self, js: str) -> None:
        if self._win:
            try:
                self._win.evaluate_js(js)
            except Exception:
                pass

    # ── Frame ingestion (detector thread) ────────────────────────────────────

    def _on_frame(self, frame_rgb: np.ndarray, metrics) -> None:
        try:
            small = cv2.resize(frame_rgb, (460, 258))
            bgr   = cv2.cvtColor(small, cv2.COLOR_RGB2BGR)
            _, buf = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 75])
            self._latest_frame_b64 = (
                'data:image/jpeg;base64,' + base64.b64encode(buf).decode()
            )
        except Exception:
            pass
        self._latest_metrics = metrics

    # ── Polling API ───────────────────────────────────────────────────────────

    def get_frame(self) -> Optional[str]:
        return self._latest_frame_b64

    def get_status(self) -> dict:
        paused  = self.detector.is_paused()
        snoozed = not self.detector.notifications_enabled
        running = self.detector.is_running()
        metrics = self._latest_metrics

        quality = None
        if metrics and self.detector.active_profile:
            score   = posture_score(metrics, self.detector.active_profile)
            sens    = self.settings.get('sensitivity', 0.20)
            quality = max(0.0, min(1.0, 1.0 - score / sens))

        calib_done_name      = self._calib_done_name
        self._calib_done_name = None   # consume once

        return {
            'running':           running,
            'paused':            paused,
            'pause_remaining':   int(self.detector.pause_remaining()) if paused else 0,
            'snoozed':           snoozed,
            'pose_detected':     metrics is not None,
            'quality':           quality,
            'active_profile':    self.detector.active_profile_name,
            'calib_in_progress': self._calib_in_progress,
            'calib_progress':    self.detector.calibration_progress() if self._calib_in_progress else 0.0,
            'calib_done_name':   calib_done_name,
            'frozen':            self._frozen,
        }

    # ── Monitoring ────────────────────────────────────────────────────────────

    def toggle_monitoring(self) -> dict:
        if self.detector.is_running():
            self.detector.stop()
            return {'ok': True}
        if not self.detector.active_profile:
            return {'ok': False, 'error': 'no_profile'}
        self.detector.start()
        return {'ok': True}

    # ── Calibration ───────────────────────────────────────────────────────────

    def start_calibration(self, name: str) -> dict:
        name = name.strip()
        if not name:
            return {'ok': False, 'error': 'empty_name'}
        if not self.detector.is_running():
            self.detector.start()
        self._calib_in_progress = True
        self._calib_polls       = 0

        def on_done(metrics: dict) -> None:
            self.pm.add(name, metrics)
            self._calib_in_progress = False
            self._calib_done_name   = name

        self.detector.start_calibration(on_done)
        return {'ok': True}

    # ── Profile management ────────────────────────────────────────────────────

    def get_profiles(self) -> list:
        active = self.detector.active_profile_name
        return [{'name': n, 'active': n == active} for n in self.pm.names()]

    def check_profile_exists(self, name: str) -> bool:
        return name in self.pm.names()

    def set_active_profile(self, name: str) -> dict:
        metrics = self.pm.get(name)
        if not metrics:
            return {'ok': False, 'error': 'not_found'}
        self.detector.active_profile      = metrics
        self.detector.active_profile_name = name
        self.settings['active_profile']   = name
        config.save_settings(self.settings)
        return {'ok': True}

    def rename_profile(self, old_name: str, new_name: str) -> dict:
        new_name = new_name.strip()
        if not new_name:
            return {'ok': False, 'error': 'empty_name'}
        if new_name == old_name:
            return {'ok': True}
        if new_name in self.pm.names():
            return {'ok': False, 'error': 'duplicate'}
        if self.pm.rename(old_name, new_name):
            if self.detector.active_profile_name == old_name:
                self.detector.active_profile_name = new_name
                self.settings['active_profile']   = new_name
                config.save_settings(self.settings)
            return {'ok': True}
        return {'ok': False, 'error': 'rename_failed'}

    def delete_profile(self, name: str) -> dict:
        self.pm.delete(name)
        if self.detector.active_profile_name == name:
            self.detector.active_profile      = None
            self.detector.active_profile_name = None
            self.settings['active_profile']   = None
            config.save_settings(self.settings)
            self.detector.stop()
        return {'ok': True}

    # ── Settings ──────────────────────────────────────────────────────────────

    def get_settings(self) -> dict:
        return {
            'check_interval':        self.settings.get('check_interval', 5),
            'notification_cooldown': self.settings.get('notification_cooldown', 300) // 60,
            'sensitivity':           self.settings.get('sensitivity', 0.20),
            'camera_index':          self.settings.get('camera_index', 0),
            'startup_enabled':       is_startup_enabled(),
        }

    def save_settings_data(self, data: dict) -> dict:
        try:
            self.settings['check_interval']       = max(1, int(data.get('check_interval', 5)))
            self.settings['notification_cooldown'] = max(1, int(data.get('notification_cooldown', 5))) * 60
            self.settings['sensitivity']           = round(float(data.get('sensitivity', 0.20)), 2)
            self.settings['camera_index']          = max(0, int(data.get('camera_index', 0)))
            self.detector.settings                 = self.settings
            config.save_settings(self.settings)
            set_startup(bool(data.get('startup_enabled', False)))
            return {'ok': True}
        except Exception as e:
            return {'ok': False, 'error': str(e)}

    # ── Gesture-triggered actions (called from detector callbacks) ────────────

    def toggle_snooze(self) -> None:
        from notifier import send_notification
        self.detector.toggle_snooze()
        if self.detector.notifications_enabled:
            send_notification('posture v0.3', '🔔 Posture notifications resumed.')
        else:
            send_notification('posture v0.3',
                              '🔕 Notifications snoozed — show open hand to resume.')
        self._push("handleEvent('snooze')")

    def start_timeout(self) -> None:
        from notifier import send_notification
        self.detector.pause_for(30 * 60)
        self._frozen = True
        self._push("handleEvent('timeout')")
        send_notification('posture v0.3',
                          '⏸ Time Out! 30-minute break started. Camera frozen, alerts paused.')

    def resume_tracking(self) -> None:
        from notifier import send_notification
        self.detector.resume()
        self._frozen = False
        send_notification('posture v0.3', '▶ Posture tracking resumed.')

    def resume_tracking_js(self) -> None:
        """Called from the JS Resume button."""
        self.resume_tracking()

    def pause_js(self) -> dict:
        """Called from the JS Pause button — 30-second demo pause."""
        self.detector.pause_for(30)
        return {'ok': True}

    def toggle_hell_mode(self) -> None:
        from notifier import send_notification
        self._hell_mode         = not self._hell_mode
        self._hell_mode_ref[0]  = self._hell_mode
        val = 'true' if self._hell_mode else 'false'
        self._push(f"handleEvent('hell_toggle', {val})")
        if self._hell_mode:
            send_notification('fuck you too i guess?', 'you get red now for being an asshole')
        else:
            send_notification('posture v0.3', "Damn you dont let up huh?")

    def minimize_to_tray(self) -> None:
        if self._win:
            self._win.hide()

    # ── Background tick ───────────────────────────────────────────────────────

    def _tick_loop(self) -> None:
        while True:
            time.sleep(0.1)

            # Auto-unfreeze when the T-sign timeout expires naturally
            if self._frozen and not self.detector.is_paused():
                self._frozen = False
                self._push("handleEvent('unfreeze')")
                from notifier import send_notification
                send_notification('posture v0.3', '▶ Posture break ended. Tracking resumed.')

            # Calibration watchdog (~60 s = 600 polls × 100 ms)
            if self._calib_in_progress:
                self._calib_polls += 1
                if self._calib_polls > 600:
                    self.detector._calib_active = False
                    self._calib_in_progress     = False
                    self._push("handleEvent('calib_timeout')")

            # Tray icon colour — update only on state change
            paused = self.detector.is_paused() or not self.detector.notifications_enabled
            if paused != self._prev_paused_state:
                self._prev_paused_state = paused
                if self.on_tray_state_change:
                    self.on_tray_state_change(paused)


# ── Main window wrapper ───────────────────────────────────────────────────────

class MainWindow:
    """Wraps PostureAPI + a pywebview.Window; matches the interface main.py expects."""

    def __init__(
        self,
        detector: PostureDetector,
        pm: ProfileManager,
        settings: dict,
        hell_mode_ref: Optional[list] = None,
    ):
        self._hell_mode_ref = hell_mode_ref or [False]
        self.api = PostureAPI(detector, pm, settings, self._hell_mode_ref)
        self._win: Optional[webview.Window] = None

    # ── Tray callback property ────────────────────────────────────────────────

    @property
    def on_tray_state_change(self) -> Optional[Callable]:
        return self.api.on_tray_state_change

    @on_tray_state_change.setter
    def on_tray_state_change(self, cb: Optional[Callable]) -> None:
        self.api.on_tray_state_change = cb

    # ── Visibility ────────────────────────────────────────────────────────────

    def show(self) -> None:
        if self._win:
            self._win.show()

    def hide(self) -> None:
        if self._win:
            self._win.hide()

    def destroy(self) -> None:
        if self._win:
            self._win.destroy()

    # ── Gesture proxies (called from main.py detector callbacks) ─────────────

    def toggle_snooze(self)    -> None: self.api.toggle_snooze()
    def start_timeout(self)    -> None: self.api.start_timeout()
    def toggle_hell_mode(self) -> None: self.api.toggle_hell_mode()
    def resume_tracking(self)  -> None: self.api.resume_tracking()

    # ── Main event loop ───────────────────────────────────────────────────────

    def run(self, show_initially: bool = True) -> None:
        html_path = str(Path(__file__).parent / 'index.html')
        win = webview.create_window(
            'posture v0.3',
            html_path,
            js_api=self.api,
            width=740,
            height=610,
            resizable=False,
            hidden=not show_initially,
        )
        self._win = win
        self.api._set_window(win)

        def on_closing():
            win.hide()
            return False  # prevent window destruction; use tray Quit to exit

        win.events.closing += on_closing
        webview.start()
