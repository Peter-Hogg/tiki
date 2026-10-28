"""File > Open Recent: a short list of recently opened files."""

from __future__ import annotations

import json
from pathlib import Path

RECENT_FILE = Path.home() / '.config' / 'tiki' / 'recent.json'
MAX_RECENT = 10


def load() -> list[str]:
    try:
        items = json.loads(RECENT_FILE.read_text())
        return [str(p) for p in items][:MAX_RECENT]
    except (OSError, ValueError):
        return []


def add(path: str | Path) -> None:
    """Put ``path`` at the top of the list (never fails an Open because of it)."""
    path = str(Path(path).resolve())
    items = [p for p in load() if p != path]
    try:
        RECENT_FILE.parent.mkdir(parents=True, exist_ok=True)
        RECENT_FILE.write_text(json.dumps([path] + items[: MAX_RECENT - 1], indent=1))
    except OSError:
        pass


def clear() -> None:
    try:
        RECENT_FILE.unlink()
    except OSError:
        pass
