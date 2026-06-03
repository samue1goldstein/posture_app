APP_NAME = "posture v0.3"

try:
    from winotify import Notification, audio as _audio
    _BACKEND = "winotify"
except ImportError:
    try:
        from plyer import notification as _plyer
        _BACKEND = "plyer"
    except ImportError:
        _BACKEND = "print"


def send_notification(title: str, message: str, duration: int = 5) -> None:
    if _BACKEND == "winotify":
        n = Notification(app_id=APP_NAME, title=title, msg=message, duration="short")
        n.set_audio(_audio.Default, loop=False)
        n.show()
    elif _BACKEND == "plyer":
        _plyer.notify(title=title, message=message, app_name=APP_NAME, timeout=duration)
    else:
        print(f"[{APP_NAME}] {title}: {message}")
