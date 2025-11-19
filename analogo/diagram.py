"""Circuit diagram plotting."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import networkx as nx


def draw_diagram(metadata: Optional[Dict[str, Any]], output_path: Path) -> Optional[Path]:
    if not metadata:
        return None
    diag = metadata.get("diagram") if isinstance(metadata, dict) else None
    if not isinstance(diag, dict):
        return None

    connections = diag.get("connections") or []
    if not connections:
        return None

    graph = nx.Graph()
    for conn in connections:
        src = conn.get("from") or conn.get("start")
        dst = conn.get("to") or conn.get("end")
        label = conn.get("label") or conn.get("component") or conn.get("name")
        if not src or not dst:
            continue
        graph.add_edge(src, dst, label=label)

    if graph.number_of_edges() == 0:
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt = _get_matplotlib()
    plt.figure(figsize=(6, 4.5))
    pos = nx.spring_layout(graph, seed=42)
    nx.draw_networkx_nodes(graph, pos, node_color="#f0f0f0", edgecolors="#333333")
    nx.draw_networkx_labels(graph, pos, font_size=9)
    nx.draw_networkx_edges(graph, pos)
    edge_labels = nx.get_edge_attributes(graph, "label")
    if edge_labels:
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=8)
    plt.axis("off")
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
