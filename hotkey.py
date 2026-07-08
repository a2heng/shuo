import json, threading
from pathlib import Path
from pynput import mouse, keyboard

CONFIG_DIR = Path.home() / ".shuo"
CONFIG_DIR.mkdir(exist_ok=True)
CONFIG_PATH = CONFIG_DIR / "config.json"
DEFAULT_HOTKEY = "f2"

_hotkey = DEFAULT_HOTKEY
_listener = None
_listener_lock = threading.RLock()  # 可重入锁，避免 start() 调用 stop() 时死锁
_on_down = None
_on_up = None
_hotkey_active = False

def load():
    global _hotkey
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, encoding="utf-8") as f:
                cfg = json.load(f)
                _hotkey = cfg.get("hotkey", DEFAULT_HOTKEY)
        except: pass

def save(hotkey=None):
    if hotkey:
        global _hotkey
        _hotkey = hotkey
    # 读取已有配置，合并后写回（避免覆盖其他字段）
    cfg = {}
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, encoding="utf-8") as f:
                cfg = json.load(f)
        except: pass
    cfg["hotkey"] = _hotkey
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

def get():
    return _hotkey

def start(on_down=None, on_up=None):
    global _listener, _on_down, _on_up, _hotkey_active
    _on_down = on_down
    _on_up = on_up
    _hotkey_active = False

    parts = _hotkey.split("+")
    key_name = parts[-1]
    is_mouse = key_name in ("xbutton1", "xbutton2")

    with _listener_lock:
        stop()

        if is_mouse:
            btn = mouse.Button.x2 if key_name == "xbutton2" else mouse.Button.x1
            def _on_click(x, y, b, pressed):
                global _hotkey_active
                if b != btn: return
                if pressed and not _hotkey_active:
                    _hotkey_active = True
                    if _on_down: _on_down()
                elif not pressed and _hotkey_active:
                    _hotkey_active = False
                    if _on_up: _on_up()
            ml = mouse.Listener(on_click=_on_click)
            ml.start()
            _listener = (ml,)
        else:
            needs_ctrl = "ctrl" in parts
            needs_shift = "shift" in parts
            needs_alt = "alt" in parts
            _pressed = set()

            def _on_press(key):
                global _hotkey_active
                if isinstance(key, keyboard.Key):
                    k = key.name.lower()
                else:
                    try: k = key.char.lower()
                    except: return True
                if k in ("ctrl", "ctrl_l", "ctrl_r"):
                    _pressed.add("ctrl")
                elif k in ("shift", "shift_l", "shift_r"):
                    _pressed.add("shift")
                elif k in ("alt", "alt_l", "alt_r"):
                    _pressed.add("alt")
                elif k == key_name and not _hotkey_active:
                    if (needs_ctrl and "ctrl" not in _pressed): return True
                    if (needs_shift and "shift" not in _pressed): return True
                    if (needs_alt and "alt" not in _pressed): return True
                    _hotkey_active = True
                    if _on_down: _on_down()
                return True

            def _on_release(key):
                global _hotkey_active
                if isinstance(key, keyboard.Key):
                    k = key.name.lower()
                else:
                    try: k = key.char.lower()
                    except: return True
                if k in ("ctrl", "ctrl_l", "ctrl_r"):
                    _pressed.discard("ctrl")
                elif k in ("shift", "shift_l", "shift_r"):
                    _pressed.discard("shift")
                elif k in ("alt", "alt_l", "alt_r"):
                    _pressed.discard("alt")
                elif k == key_name and _hotkey_active:
                    _hotkey_active = False
                    if _on_up: _on_up()
                return True

            kl = keyboard.Listener(on_press=_on_press, on_release=_on_release)
            kl.start()
            _listener = (kl,)

def stop():
    global _listener
    with _listener_lock:
        if _listener:
            for l in _listener:
                l.stop()
            _listener = None
