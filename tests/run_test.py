#!/usr/bin/env python3
"""Run an ngspice netlist and plot both waveform and circuit diagram."""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import struct

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analogo.diagram import draw_diagram
from analogo.metadata import parse_metadata

ALLOWED_EXTENSIONS = {".sp", ".cir", ".ckt"}


def ensure_dependencies(netlist: Path) -> None:
    if shutil.which("ngspice") is None:
        sys.exit("ngspice is not installed or not on PATH.")
    if not netlist.exists():
        sys.exit(f"Unable to find {netlist}.")
    suffix = netlist.suffix.lower()
    if suffix and suffix not in ALLOWED_EXTENSIONS:
        print(
            f"Warning: unexpected netlist extension '{netlist.suffix}'. Proceeding anyway.",
            file=sys.stderr,
        )


def run_ngspice(netlist: Path) -> Tuple[Path, Path]:
    log_path = netlist.with_suffix(".log")
    with tempfile.NamedTemporaryFile(
        prefix=f"{netlist.stem}_", suffix=".raw", delete=False, dir=netlist.parent
    ) as tmp_raw:
        raw_path = Path(tmp_raw.name)
    cmd = ["ngspice", "-b", "-o", log_path.name, "-r", raw_path.name, netlist.name]
    result = subprocess.run(
        cmd,
        cwd=netlist.parent,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        log_excerpt = ""
        if log_path.exists():
            log_excerpt = _tail_text(log_path, max_chars=1200)
        else:
            log_excerpt = (result.stdout or "") + (result.stderr or "")
        try:
            raw_path.unlink()
        except FileNotFoundError:
            pass
        raise RuntimeError(
            f"ngspice failed for {netlist} (exit {result.returncode}).\n{log_excerpt.strip()}"
        )
    return log_path, raw_path


def _parse_value(token: str) -> float:
    token = token.strip()
    if "," in token:
        real, imag = token.split(",", 1)
        return math.hypot(float(real), float(imag))
    return float(token)


def _is_int_token(token: str) -> bool:
    if not token:
        return False
    return token.lstrip("+-").isdigit()


def _tokens_to_value(tokens: List[str]) -> str:
    if not tokens:
        raise ValueError("Cannot parse empty token list.")
    if len(tokens) == 1:
        return tokens[0]
    if len(tokens) == 2 and _is_int_token(tokens[0]):
        return tokens[1]
    if "," in tokens[-1]:
        return tokens[-1]
    if len(tokens) >= 2:
        return ",".join(tokens[-2:])
    return tokens[-1]


def parse_log(log_path: Path) -> Optional[Tuple[str, Dict[str, List[float]]]]:
    headers: List[str] = []
    columns: Dict[str, List[float]] = {}
    capture = False

    for raw_line in log_path.read_text().splitlines():
        line = raw_line.replace("\x0c", "").strip()
        if not line:
            continue
        if line.startswith("Index"):
            headers = line.split()
            if len(headers) < 2:
                raise RuntimeError("Unexpected header format in ngspice log.")
            columns = {h: [] for h in headers[1:]}
            capture = True
            continue
        if line.startswith("Note:"):
            if capture:
                break
            continue
        if line.startswith("---") or line.startswith("* "):
            continue
        if not capture:
            continue
        parts = line.split()
        if len(parts) < len(headers) or not parts[0].lstrip("-").isdigit():
            continue
        try:
            for key, token in zip(headers[1:], parts[1:]):
                columns[key].append(_parse_value(token))
        except ValueError:
            continue

    if not columns:
        return None
    x_key = headers[1]
    return x_key, columns


def parse_raw(raw_path: Path) -> Tuple[str, Dict[str, List[float]]]:
    if not raw_path.exists():
        raise RuntimeError(f"Missing raw data file: {raw_path}")

    header_lines: List[str] = []
    mode: Optional[str] = None
    data_offset: Optional[int] = None
    with raw_path.open("rb") as handle:
        while True:
            line = handle.readline()
            if not line:
                raise RuntimeError("Unexpected EOF while reading raw header.")
            decoded = line.decode("ascii", errors="ignore").rstrip("\r\n")
            header_lines.append(decoded)
            if decoded == "Binary:":
                mode = "binary"
                data_offset = handle.tell()
                break
            if decoded == "Values:":
                mode = "ascii"
                data_offset = handle.tell()
                break

    if mode not in {"binary", "ascii"} or data_offset is None:
        raise RuntimeError("Failed to determine raw file format.")

    num_vars = None
    num_points = None
    flags = ""
    var_names: List[str] = []

    reading_vars = False
    for line in header_lines:
        lower = line.lower()
        if lower.startswith("no. variables:"):
            num_vars = int(line.split(":", 1)[1].strip())
        elif lower.startswith("no. points:"):
            num_points = int(line.split(":", 1)[1].strip())
        elif lower.startswith("flags:"):
            flags = line.split(":", 1)[1].strip().lower()
        elif line.startswith("Variables:"):
            reading_vars = True
            continue
        elif reading_vars:
            if not line.startswith(("\t", " ")):
                reading_vars = False
                continue
            parts = line.strip().split()
            if len(parts) >= 3:
                var_names.append(parts[1])

    if not var_names or num_vars is None or num_points is None:
        raise RuntimeError("Failed to read header metadata from raw file.")
    if len(var_names) != num_vars:
        raise RuntimeError("Variable count mismatch in raw header.")

    is_complex = "complex" in flags
    value_size = 16 if is_complex else 8

    if mode == "binary":
        columns = _read_binary_raw(raw_path, data_offset, var_names, num_points, is_complex, value_size)
    else:
        columns = _read_ascii_raw(raw_path, data_offset, var_names, num_points)

    return var_names[0], columns


def _read_binary_raw(
    raw_path: Path,
    offset: int,
    var_names: List[str],
    num_points: int,
    is_complex: bool,
    value_size: int,
) -> Dict[str, List[float]]:
    columns: Dict[str, List[float]] = {name: [] for name in var_names}
    with raw_path.open("rb") as handle:
        handle.seek(offset)
        for _ in range(num_points):
            for name in var_names:
                chunk = handle.read(value_size)
                if len(chunk) != value_size:
                    raise RuntimeError("Raw file ended prematurely while reading data.")
                if is_complex:
                    real, imag = struct.unpack("<dd", chunk)
                    columns[name].append(math.hypot(real, imag))
                else:
                    (value,) = struct.unpack("<d", chunk)
                    columns[name].append(value)
    return columns


def _read_ascii_raw(
    raw_path: Path,
    offset: int,
    var_names: List[str],
    num_points: int,
) -> Dict[str, List[float]]:
    columns: Dict[str, List[float]] = {name: [] for name in var_names}
    with raw_path.open("r") as handle:
        handle.seek(offset)
        point_index = 0
        while point_index < num_points:
            line = handle.readline()
            if not line:
                break
            stripped = line.strip()
            if not stripped:
                continue
            if line.startswith("\t"):
                # Skip stray continuation lines until we hit a point header.
                continue
            tokens = stripped.split()
            if len(tokens) < 2:
                continue
            value_str = _tokens_to_value(tokens)
            columns[var_names[0]].append(_parse_value(value_str))
            for name in var_names[1:]:
                value_line = handle.readline()
                while value_line and not value_line.strip():
                    value_line = handle.readline()
                if not value_line:
                    raise RuntimeError("Raw file ended prematurely while reading ASCII data.")
                value_tokens = value_line.strip().split()
                value_token = _tokens_to_value(value_tokens)
                columns[name].append(_parse_value(value_token))
            point_index += 1

    base_len = len(columns[var_names[0]])
    if point_index != num_points or base_len == 0:
        raise RuntimeError("Failed to parse ASCII raw file data.")
    for name in var_names[1:]:
        if len(columns[name]) != base_len:
            raise RuntimeError("ASCII raw parsing produced inconsistent column lengths.")
    return columns


def _tail_text(path: Path, max_chars: int = 2000) -> str:
    try:
        text = path.read_text()
    except OSError:
        return ""
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def load_netlist_metadata(netlist: Path) -> Optional[Dict[str, Any]]:
    try:
        text = netlist.read_text()
    except OSError:
        return None
    return parse_metadata(text)


def write_diagram(metadata: Optional[Dict[str, Any]], netlist: Path) -> Optional[Path]:
    if not metadata:
        return None
    output_path = netlist.with_name(f"{netlist.stem}_diagram.png")
    return draw_diagram(metadata, output_path)


def select_series(
    x_key: str, data: Dict[str, List[float]], requested: Optional[List[str]]
) -> List[str]:
    available = {name.lower(): name for name in data if name != x_key}
    if not available:
        raise RuntimeError("No data columns available for plotting.")

    def resolve(name: str) -> str:
        key = name.lower()
        if key not in available:
            raise RuntimeError(f"Requested signal '{name}' not found in simulation data.")
        return available[key]

    if requested:
        seen: List[str] = []
        for item in requested:
            resolved = resolve(item)
            if resolved not in seen:
                seen.append(resolved)
        return seen

    preferred_pairs = [
        ("v_in", "v_out"),
        ("vin", "vout"),
    ]
    for vin_name, vout_name in preferred_pairs:
        if vin_name.lower() in available and vout_name.lower() in available:
            return [available[vin_name.lower()], available[vout_name.lower()]]

    voltage_cols = [
        available[name]
        for name in available
        if available[name].lower().startswith("v_") or available[name].lower().startswith("v")
    ]
    if voltage_cols:
        return voltage_cols

    return list(available.values())


def plot_waveforms(
    x_key: str,
    data: Dict[str, List[float]],
    title: str,
    output_path: Path,
    requested: Optional[List[str]],
) -> None:
    if x_key not in data:
        raise RuntimeError(f"Missing x-axis column '{x_key}' in parsed data.")
    x_data = data[x_key]
    series_names = select_series(x_key, data, requested)

    fig = plt.figure()
    for name in series_names:
        plt.plot(x_data, data[name], label=name, linewidth=1.2)
    plt.xlabel(x_key)
    plt.ylabel("Value (units)")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)

    # backend = plt.get_backend().lower()
    # if "agg" not in backend:
        # plt.show()
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run ngspice on a netlist and plot printed waveforms/diagram."
    )
    parser.add_argument(
        "netlist", help="Path to the netlist (.sp/.cir) to run."
    )
    parser.add_argument(
        "--signals",
        nargs="+",
        help="Optional list of vector names to plot (default auto-selects voltages such as v_in/v_out).",
    )
    args = parser.parse_args()

    netlist = Path(args.netlist).expanduser().resolve()
    ensure_dependencies(netlist)
    metadata = load_netlist_metadata(netlist)
    raw_path: Optional[Path] = None
    try:
        log_path, raw_path = run_ngspice(netlist)
        parsed = parse_log(log_path)
        if parsed is None:
            x_key, data = parse_raw(raw_path)
        else:
            x_key, data = parsed
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    finally:
        if raw_path:
            try:
                raw_path.unlink()
            except FileNotFoundError:
                pass
    output_path = netlist.with_suffix(".png")
    plot_waveforms(
        x_key,
        data,
        f"{netlist.stem} response",
        output_path,
        args.signals,
    )
    print(f"Saved plot to {output_path}")
    diagram_path = write_diagram(metadata, netlist)
    if diagram_path:
        print(f"Saved diagram to {diagram_path}")


if __name__ == "__main__":
    main()
