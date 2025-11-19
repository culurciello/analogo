"""Analogo circuit-design agent."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

# Ensure matplotlib has a writable cache/config directory even in sandboxed environments.
cache_dir = Path(tempfile.gettempdir()) / "analogo-mplcache"
cache_dir.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))

__all__ = ["agent"]
