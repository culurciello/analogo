"""Circuit diagram plotting."""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

ROW_SPACING = 2.0
WIRE_LENGTH = 0.5


@dataclass(frozen=True)
class DiagramConnection:
    """Normalized connection description extracted from metadata."""

    source: str
    target: str
    component: str
    label: str


def draw_diagram(metadata: Optional[Dict[str, Any]], output_path: Path) -> Optional[Path]:
    """Render a simple schematic-style diagram with Schemdraw."""

    connections = _normalize_connections(metadata)
    if not connections:
        return None

    _configure_matplotlib()
    try:
        import schemdraw
        import schemdraw.elements as elm
        from schemdraw.transform import Point
    except ImportError:
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)

    drawing = schemdraw.Drawing(show=False)
    drawing.config(
        fontsize=11,
        unit=2.5,
        inches_per_unit=0.35,
        margin=0.25,
        bgcolor="#ffffff",
    )

    for index, connection in enumerate(connections):
        drawing.move_from(Point((0, -index * ROW_SPACING)))
        _add_connection_row(drawing, elm, connection)

    drawing.save(str(output_path))
    return output_path


def _add_connection_row(drawing, elm, connection: DiagramConnection) -> None:
    """Add a left-net/component/right-net trio to the drawing."""

    drawing.add(_endpoint(elm, connection.source, align="left"))
    drawing.add(elm.Line().right().length(WIRE_LENGTH))
    drawing.add(_component_element(elm, connection))
    drawing.add(elm.Line().right().length(WIRE_LENGTH))
    drawing.add(_endpoint(elm, connection.target, align="right"))


def _endpoint(elm, name: str, *, align: str) -> Any:
    """Create a labeled net endpoint."""

    label_loc = "left" if align == "left" else "right"
    label = name or "?"
    return elm.Dot().label(label, loc=label_loc)


def _component_element(elm, connection: DiagramConnection):
    """Select a Schemdraw element for the component and apply its label."""

    key = (connection.component or connection.label).strip().lower()
    element = _build_component(elm, key)
    return element.label(connection.label, loc="top")


def _build_component(elm, key: str):
    """Return a Schemdraw element instance best matching the component."""

    builders = {
        "r": lambda: elm.Resistor().right(),
        "c": lambda: elm.Capacitor().right(),
        "l": lambda: elm.Inductor().right(),
        "v": lambda: elm.SourceV().right(),
        "i": lambda: elm.SourceI().right(),
        "d": lambda: elm.Diode().right(),
        "q": lambda: elm.Bjt().right(),
        "m": lambda: elm.NMos().right(),
        "p": lambda: elm.PMos().right(),
        "u": lambda: elm.Opamp().right(),
    }
    builder = builders.get(key[:1])
    try:
        return builder() if builder else elm.Line().right().length(1.5)
    except AttributeError:
        return elm.Line().right().length(1.5)


def _normalize_connections(metadata: Optional[Dict[str, Any]]) -> List[DiagramConnection]:
    """Extract DiagramConnection objects from metadata."""

    if not metadata or not isinstance(metadata, dict):
        return []
    diag = metadata.get("diagram")
    if not isinstance(diag, dict):
        return []
    raw = diag.get("connections")
    if not isinstance(raw, list):
        return []

    normalized: List[DiagramConnection] = []
    for index, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            continue
        source = _clean_text(item.get("from") or item.get("start"))
        target = _clean_text(item.get("to") or item.get("end"))
        if not source or not target:
            continue
        component = _clean_text(
            item.get("component")
            or item.get("name")
            or item.get("type")
            or item.get("element")
        )
        value = _clean_text(item.get("value"))
        label_override = _clean_text(item.get("label"))
        label = label_override or _join_fields(component, value)
        if not label:
            label = f"Conn {index}"
        normalized.append(
            DiagramConnection(
                source=source,
                target=target,
                component=component or label,
                label=label,
            )
        )
    return normalized


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _join_fields(primary: str, secondary: str) -> str:
    parts = [part for part in (primary, secondary) if part]
    return " ".join(parts)


def _configure_matplotlib() -> None:
    if "MPLCONFIGDIR" not in os.environ:
        cache_dir = Path(tempfile.gettempdir()) / "analogo-mplcache"
        cache_dir.mkdir(exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(cache_dir)
        os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
    import matplotlib

    matplotlib.use("Agg")
