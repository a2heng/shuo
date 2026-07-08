import json, os
from pathlib import Path

_current = {}
_fallback = {}

def load(lang="en"):
    _current.clear()
    _fallback.clear()
    locale_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "locales")
    base = os.path.join(locale_dir, "en.json")
    if os.path.exists(base):
        with open(base, encoding="utf-8") as f:
            _fallback.update(json.load(f))
    path = os.path.join(locale_dir, f"{lang}.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            _current.update(json.load(f))

def tr(key):
    return _current.get(key, _fallback.get(key, key))

def list_locales():
    dirname = os.path.join(os.path.dirname(os.path.dirname(__file__)), "locales")
    locales = []
    for fname in os.listdir(dirname):
        if not fname.endswith(".json"):
            continue
        code = fname[:-5]
        path = os.path.join(dirname, fname)
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            name = data.get("lang.name", code)
        except Exception:
            name = code
        locales.append((code, name))
    locales.sort(key=lambda x: x[1])
    return locales
