import pystray
from PIL import Image, ImageDraw

from gui import MainWindow
from detector import PostureDetector
from profiles import ProfileManager
from notifier import send_notification
import config

APP_NAME = "posture v0.3"


def _make_tray_icon(paused: bool = False) -> Image.Image:
    """Purple when active, orange when paused/snoozed."""
    fill = "#fb923c" if paused else "#8b5cf6"
    img  = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    d    = ImageDraw.Draw(img)
    d.ellipse([0, 0, 63, 63], fill=fill)
    d.ellipse([24, 6, 40, 22],  fill="white")           # head
    d.line([32, 22, 32, 44],    fill="white", width=3)  # spine
    d.line([18, 30, 46, 30],    fill="white", width=3)  # shoulders
    d.line([32, 44, 20, 58],    fill="white", width=3)  # left leg
    d.line([32, 44, 44, 58],    fill="white", width=3)  # right leg
    return img


def main() -> None:
    settings   = config.load_settings()
    pm         = ProfileManager()
    hell_state = [False]

    def _on_bad_posture(score: float) -> None:
        if hell_state[0]:
            send_notification(
                "SIT UP STRAIGHT",
                f"Your posture score is: {score:.0%}",
            )
        else:
            send_notification(
                "Posture Alert",
                f"Time to straighten up!  Slouch score: {score:.0%}",
            )

    def _on_low_light() -> None:
        send_notification(
            "Room too dark",
            "Can't find you in frame — turn on a light for accurate tracking.",
        )

    detector          = PostureDetector(
        on_bad_posture=_on_bad_posture,
        on_low_light=_on_low_light,
    )
    detector.settings = settings

    saved_name = settings.get("active_profile")
    if saved_name:
        metrics = pm.get(saved_name)
        if metrics:
            detector.active_profile      = metrics
            detector.active_profile_name = saved_name

    window = MainWindow(detector, pm, settings, hell_mode_ref=hell_state)

    # Wire Easter egg callbacks (detector thread → thread-safe evaluate_js)
    detector.on_freeze_toggle = window.toggle_snooze
    detector.on_hell_toggle   = window.toggle_hell_mode
    detector.on_timeout       = window.start_timeout

    if detector.active_profile:
        detector.start()

    def _open(_icon, _item):
        window.show()

    def _quit(_icon, _item):
        detector.stop()
        tray.stop()
        window.destroy()

    tray = pystray.Icon(
        "posture_v03",
        _make_tray_icon(paused=False),
        APP_NAME,
        menu=pystray.Menu(
            pystray.MenuItem(f"Open {APP_NAME}", _open, default=True),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Quit", _quit),
        ),
    )
    tray.run_detached()

    window.on_tray_state_change = lambda paused: setattr(
        tray, "icon", _make_tray_icon(paused=paused)
    )

    window.run(show_initially=True)

    # Reached here after window.destroy() — clean up
    detector.stop()
    try:
        tray.stop()
    except Exception:
        pass


if __name__ == "__main__":
    main()
