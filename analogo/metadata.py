"""Utilities for parsing structured metadata from netlists."""
from __future__ import annotations

import json
from typing import Any, Dict, Optional

MARKER = "* ANALOGO_METADATA"


def parse_metadata(netlist: str) -> Optional[Dict[str, Any]]:
    for line in netlist.splitlines():
        stripped = line.strip()
        if not stripped.startswith(MARKER):
            continue
        _, _, payload = stripped.partition(MARKER)
        payload = payload.strip(" :")
        if not payload:
            continue
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            continue
    return None
