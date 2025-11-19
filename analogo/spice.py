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
        log_path = workdir / "circuit.log"
        raw_path = workdir / "sim.raw"
        csv_path = workdir / "analogo_waveform.csv"

        netlist_path.write_text(netlist_text)

        cmd = [
            self.executable,
            "-b",
            "-o",
            log_path.name,
            "-r",
            raw_path.name,
            netlist_path.name,
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

    with path.open("r") as handle:
        # Read all rows, splitting on whitespace (handles NGSpice's space-delimited output)
        rows = []
        for line in handle:
            # Split on whitespace and filter empty strings
            parts = [p.strip() for p in line.split() if p.strip()]
            if parts:
                rows.append(parts)

    if not rows or len(rows) < 2:
        return None

    header = rows[0]
    if len(header) < 2:
        return None

    # For AC analysis, NGSpice outputs complex data with duplicate headers
    # Clean up duplicates and keep unique column names
    seen = set()
    clean_header = []
    col_indices = []

    for idx, name in enumerate(header):
        if name not in seen:
            seen.add(name)
            clean_header.append(name)
            col_indices.append(idx)

    # Initialize columns for clean headers
    columns: List[List[float]] = [[] for _ in clean_header]

    # Parse data rows, using only the selected column indices
    for row in rows[1:]:
        if len(row) < len(header):
            continue  # Skip malformed rows
        try:
            for col_idx, data_idx in enumerate(col_indices):
                if data_idx < len(row):
                    columns[col_idx].append(float(row[data_idx]))
        except (ValueError, IndexError):
            continue  # Skip rows we can't parse

    if not columns or not columns[0]:
        return None

    x_label = clean_header[0]
    series = {clean_header[i]: columns[i] for i in range(1, len(clean_header))}
    return Waveform(x_label=x_label, x=columns[0], series=series)
