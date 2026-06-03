import json
from typing import Optional
from config import APP_DIR, PROFILES_FILE


class ProfileManager:
    def __init__(self):
        APP_DIR.mkdir(exist_ok=True)
        self.profiles: dict = {}
        self.load()

    def load(self) -> None:
        if PROFILES_FILE.exists():
            with open(PROFILES_FILE) as f:
                try:
                    self.profiles = json.load(f)
                except json.JSONDecodeError:
                    self.profiles = {}

    def save(self) -> None:
        APP_DIR.mkdir(exist_ok=True)
        with open(PROFILES_FILE, "w") as f:
            json.dump(self.profiles, f, indent=2)

    def add(self, name: str, metrics: dict) -> None:
        self.profiles[name] = metrics
        self.save()

    def delete(self, name: str) -> None:
        self.profiles.pop(name, None)
        self.save()

    def rename(self, old: str, new: str) -> bool:
        if old in self.profiles and new not in self.profiles:
            self.profiles[new] = self.profiles.pop(old)
            self.save()
            return True
        return False

    def get(self, name: str) -> Optional[dict]:
        return self.profiles.get(name)

    def names(self) -> list:
        return list(self.profiles.keys())
