"""Lightweight web research helpers for gathering SPICE references."""
from __future__ import annotations

import html
import json
import re
from typing import Dict, Iterable, List, Optional

USER_AGENT = "AnalogoResearchBot/1.0 (+https://github.com/euge/analogo)"
DDG_API_URL = "https://api.duckduckgo.com/"


def gather_spice_references(prompt: str, max_results: int = 5) -> str:
    """Return a short newline-separated summary of related SPICE resources.

    Uses DuckDuckGo's instant-answer API to fetch a few related links. When
    networking fails or the dependency is unavailable an empty string is
    returned so the agent can continue without blocking.
    """

    query = f"{prompt} SPICE netlist analog circuit"
    if max_results <= 0:
        return ""
    payload = _fetch_ddg_payload(query)
    if not payload:
        return ""
    entries = _extract_topics(payload)
    lines: List[str] = []
    for entry in entries:
        if len(lines) >= max_results:
            break
        title = entry.get("text")
        url = entry.get("url")
        if not title or not url:
            continue
        title = _clean_text(title)
        url = url.strip()
        lines.append(f"- {title} ({url})")
    return "\n".join(lines)


def _fetch_ddg_payload(query: str) -> Optional[Dict[str, object]]:
    try:
        import requests
    except ImportError:
        return None
    params = {
        "q": query,
        "format": "json",
        "no_html": 1,
        "skip_disambig": 1,
    }
    try:
        response = requests.get(
            DDG_API_URL,
            params=params,
            headers={"User-Agent": USER_AGENT},
            timeout=10,
        )
        response.raise_for_status()
    except Exception:
        return None
    try:
        return response.json()
    except json.JSONDecodeError:
        return None


def _extract_topics(payload: Dict[str, object]) -> Iterable[Dict[str, str]]:
    related = payload.get("RelatedTopics")
    if isinstance(related, list):
        for item in related:
            if isinstance(item, dict):
                topics = item.get("Topics")
                if isinstance(topics, list):
                    for nested in topics:
                        if isinstance(nested, dict):
                            text = nested.get("Text")
                            url = nested.get("FirstURL")
                            yield {"text": text or "", "url": url or ""}
                else:
                    text = item.get("Text")
                    url = item.get("FirstURL")
                    yield {"text": text or "", "url": url or ""}

    abstract = payload.get("AbstractText")
    abstract_url = payload.get("AbstractURL")
    if isinstance(abstract, str) and abstract and isinstance(abstract_url, str) and abstract_url:
        yield {"text": abstract, "url": abstract_url}


def _clean_text(value: str) -> str:
    text = html.unescape(value)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
