import sys
import os
import winreg

_APP_KEY  = "posture_v03"
_RUN_PATH = r"Software\Microsoft\Windows\CurrentVersion\Run"


def _command() -> str:
    exe    = sys.executable
    script = os.path.abspath(os.path.join(os.path.dirname(__file__), "main.py"))
    return f'"{exe}" "{script}"'


def is_startup_enabled() -> bool:
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, _RUN_PATH, 0, winreg.KEY_READ) as k:
            winreg.QueryValueEx(k, _APP_KEY)
            return True
    except OSError:
        return False


def set_startup(enable: bool) -> None:
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, _RUN_PATH, 0, winreg.KEY_SET_VALUE) as k:
        if enable:
            winreg.SetValueEx(k, _APP_KEY, 0, winreg.REG_SZ, _command())
        else:
            try:
                winreg.DeleteValue(k, _APP_KEY)
            except OSError:
                pass
