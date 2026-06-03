# posture v0.3

A real-time posture monitoring app for Windows. Sits in the system tray, watches your webcam with MediaPipe, and sends a toast notification when you start to slouch.

---

## What it does

- Captures your webcam feed and runs MediaPipe pose + hand landmark detection on every frame
- Extracts a set of skeletal metrics (nose height, shoulder level, head tilt, shoulder width, lateral offset) and compares them against a saved calibration baseline
- Fires a Windows toast notification when the deviation score exceeds a configurable sensitivity threshold, with a cooldown to avoid spamming
- Supports multiple named profiles — calibrate once per setup (sitting, standing, gaming, etc.) and switch between them
- Draws the skeleton overlay live in the UI and shows a posture score ring updating in real time
- Runs quietly in the system tray; the window can be hidden and restored from the tray icon

---

## Gesture controls

Hold a gesture in front of the camera for ~1 second to trigger it. Detection runs on every third frame via the hand landmark model.

| Gesture | Effect |
|---|---|
| Open hand (all four fingers extended) | Toggle notification snooze on/off |
| T-sign (two hands forming a T shape) | Freeze the camera display and pause alerts for 30 minutes |

---

## How it works

```
Camera (OpenCV)
      |
      +---> MediaPipe Pose  --->  extract_metrics()  --->  posture_score()  --->  notification
      |
      +---> MediaPipe Hands --->  gesture detectors  --->  snooze / timeout callbacks
      |
      +--->  on_frame()  --->  base64 JPEG  --->  JS polls get_frame()  --->  <img>
```

The backend is plain Python. The frontend is HTML/CSS/JS running inside a native window via **pywebview**. A `PostureAPI` class is exposed to the webview so JavaScript can call methods like `toggle_monitoring()` or `set_active_profile()` as async promises. Python pushes one-shot events back via `evaluate_js()`.

---

## Project structure

```
main.py        entry point — wires detector, tray icon, and window
gui.py         pywebview MainWindow + PostureAPI (the JS-callable backend)
detector.py    camera loop, pose and hand detection, calibration
gesture.py     gesture recognisers and hold-to-trigger tracker
profiles.py    profile CRUD and JSON persistence
config.py      file paths and settings load/save
notifier.py    Windows toast notifications (winotify with plyer fallback)
startup.py     Windows registry run-on-startup helper
index.html     UI — Dashboard, Profiles, Settings tabs
style.css      dark theme with CSS custom properties
```
