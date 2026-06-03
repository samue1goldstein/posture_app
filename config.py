from pathlib import Path
import json

APP_DIR = Path.home() / ".posture_guardian"
PROFILES_FILE = APP_DIR / "profiles.json"
SETTINGS_FILE = APP_DIR / "settings.json"

DEFAULT_SETTINGS = {
    "check_interval": 5,
    "notification_cooldown": 300,
    "sensitivity": 0.20,
    "camera_index": 0,
    "active_profile": None,
}


def load_settings() -> dict:
    APP_DIR.mkdir(exist_ok=True)
    if SETTINGS_FILE.exists():
        with open(SETTINGS_FILE) as f:
            try:
                saved = json.load(f)
            except json.JSONDecodeError:
                saved = {}
        merged = dict(DEFAULT_SETTINGS)
        merged.update(saved)
        return merged
    return dict(DEFAULT_SETTINGS)


def save_settings(settings: dict) -> None:
    APP_DIR.mkdir(exist_ok=True)
    with open(SETTINGS_FILE, "w") as f:
        json.dump(settings, f, indent=2)
