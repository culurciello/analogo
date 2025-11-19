"""NGSpice orchestration utilities."""
from __future__ import annotations

import csv
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class Waveform:
    """Represents waveform data exported from ngspice."""

    x_label: str
    x: List[float]
    series: Dict[str, List[float]]

    @property
    def signals(self) -> List[str]:
        return list(self.series.keys())


@dataclass
class SpiceRunResult:
    """Metadata about a single ngspice simulation."""

    netlist_path: Path
    log_path: Path
    raw_path: Path
    csv_path: Path
    return_code: int
    log_excerpt: str
    success: bool
    error: Optional[str]
    waveform: Optional[Waveform]


class SpiceRunner:
    """Handles writing netlists and launching ngspice."""

    def __init__(self, executable: str = "ngspice") -> None:
        self.executable = executable
        if shutil.which(executable) is None:
            raise RuntimeError(
                "ngspice executable not found. Install ngspice and ensure it is on your PATH."
            )

    def run(self, netlist_text: str, workdir: Path) -> SpiceRunResult:
        workdir.mkdir(parents=True, exist_ok=True)
        netlist_path = workdir / "circuit.sp"
        log_path = workdir / "ngspice.log"
        raw_path = workdir / "sim.raw"
        csv_path = workdir / "analogo_waveform.csv"

        netlist_path.write_text(netlist_text)

        cmd = [
            self.executable,
            "-b",
            "-o",
            str(log_path),
            "-r",
            str(raw_path),
            str(netlist_path),
        ]
        try:
            completed = subprocess.run(
                cmd,
                cwd=str(workdir),
                check=False,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as exc:  # pragma: no cover
            raise RuntimeError("Failed to launch ngspice") from exc

        log_excerpt = ""
        if log_path.exists():
            log_excerpt = _tail_file(log_path, max_chars=8000)
        else:
            log_excerpt = (completed.stdout or "") + (completed.stderr or "")

        waveform = _load_waveform(csv_path)
        success = completed.returncode == 0
        error = None if success else "ngspice returned non-zero exit status"

        return SpiceRunResult(
            netlist_path=netlist_path,
            log_path=log_path,
            raw_path=raw_path,
            csv_path=csv_path,
            return_code=completed.returncode,
            log_excerpt=log_excerpt,
            success=success,
            error=error,
            waveform=waveform,
        )


def _tail_file(path: Path, max_chars: int = 8000) -> str:
    text = path.read_text(errors="ignore")
    if len(text) <= max_chars:
        return text
    head = text[: max_chars // 2]
    tail = text[-(max_chars // 2) :]
    return f"{head}\n...\n{tail}"


def _load_waveform(path: Path) -> Optional[Waveform]:
    if not path.exists():
        return None

    with path.open("r", newline="") as handle:
        reader = csv.reader(handle)
        rows = [row for row in reader if row]

    if not rows:
        return None

    header = rows[0]
    if len(header) < 2:
        return None

    columns: List[List[float]] = [[] for _ in header]
    for row in rows[1:]:
        for idx, cell in enumerate(row):
            if idx >= len(columns):
                continue
            try:
                columns[idx].append(float(cell))
            except ValueError:
                # Skip values we cannot turn into floats.
                break

    x_label = header[0]
    series = {header[i + 1]: columns[i + 1] for i in range(len(header) - 1)}
    return Waveform(x_label=x_label, x=columns[0], series=series)
