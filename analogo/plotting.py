"""Plotting helpers for waveforms."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional

from .spice import Waveform


def plot_waveform(waveform: Waveform, output_path: Path, title: Optional[str] = None) -> Optional[Path]:
    """Create a waveform plot using matplotlib."""

    if not waveform.x or not waveform.series:
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt = _get_matplotlib()

    plt.figure(figsize=(8, 4.5))
    for name, values in waveform.series.items():
        if len(values) != len(waveform.x):
            continue
        plt.plot(waveform.x, values, label=name)

    plt.xlabel(waveform.x_label)
    plt.ylabel("Value")
    plt.grid(True, linestyle="--", alpha=0.5)
    if title:
        plt.title(title)
    if waveform.series:
        plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path


def _get_matplotlib():
    if "MPLCONFIGDIR" not in os.environ:
        cache_dir = Path(tempfile.gettempdir()) / "analogo-mplcache"
        cache_dir.mkdir(exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(cache_dir)
        os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: WPS433

    return plt
