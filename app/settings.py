"""User settings, kept in data/settings.json."""
import json
import threading

from . import storage

PATH = storage.ROOT / "settings.json"
DEFAULTS = {
    "theme": "dark",          # dark | light | system
    "framework": "pytorch",   # pytorch | tensorflow: used for new trainings
    "device": "auto",         # auto (GPU when usable) | cpu
    "threads": 0,             # CPU threads, 0 = framework default
}
CHOICES = {
    "theme": ("dark", "light", "system"),
    "framework": ("pytorch", "tensorflow"),
    "device": ("auto", "cpu"),
}
_lock = threading.Lock()


def get() -> dict:
    with _lock:
        try:
            data = json.loads(PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {}
    out = dict(DEFAULTS)
    out.update({k: v for k, v in data.items() if k in DEFAULTS})
    return out


def update(changes: dict) -> dict:
    cur = get()
    for k, v in changes.items():
        if k not in DEFAULTS:
            raise ValueError(f"Unknown setting: {k}")
        if k in CHOICES and v not in CHOICES[k]:
            raise ValueError(f"{k} must be one of {', '.join(CHOICES[k])}")
        if k == "threads" and (not isinstance(v, int) or not 0 <= v <= 256):
            raise ValueError("threads must be between 0 and 256")
        cur[k] = v
    with _lock:
        tmp = PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(cur, indent=2), encoding="utf-8")
        tmp.replace(PATH)
    return cur
