"""Utilities for optionally showing artifacts on the desktop."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Final

ENV_VAR: Final[str] = "ANALOGO_SHOW_VISUALS"
TRUE_VALUES = {"1", "true", "yes", "on"}


def should_show() -> bool:
    value = os.getenv(ENV_VAR)
    if not value:
        return False
    return value.strip().lower() in TRUE_VALUES


def show_file(path: Path) -> bool:
    """Attempt to open the file with the platform's default viewer."""

    if not should_show() or not path or not path.exists():
        return False

    try:
        if sys.platform.startswith("darwin"):
            cmd = ["open", str(path)]
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        if os.name == "nt":  # pragma: no cover - windows only
            os.startfile(path)  # type: ignore[attr-defined]
            return True
        cmd = ["xdg-open", str(path)]
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False


__all__ = ["show_file", "should_show"]
