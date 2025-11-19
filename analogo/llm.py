"""LLM helper utilities."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence


@dataclass
class LLMOutput:
    """Represents a parsed response from the language model."""

    data: Dict[str, Any]
    raw: str


class LLMClient:
    """Thin wrapper around the OpenAI Chat Completions API."""

    def __init__(self, model: str = "gpt-5-mini-2025-08-07", temperature: float = 0.2, max_output_tokens: int = 2000) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable is required.")
        self._client = _load_openai_client()(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

    def complete(self, messages: Sequence[Dict[str, str]]) -> LLMOutput:
        """Call the Chat Completions API and return parsed JSON content."""

        response = self._client.chat.completions.create(
            model=self.model,
            messages=list(messages),
            temperature=self.temperature,
            max_tokens=self.max_output_tokens,
        )
        content = response.choices[0].message.content or ""
        data = self._extract_json(content)
        return LLMOutput(data=data, raw=content)

    @staticmethod
    def _extract_json(content: str) -> Dict[str, Any]:
        """Extract and parse JSON from the model content."""

        stripped = content.strip()
        if not stripped:
            raise ValueError("Empty response from language model")

        if stripped.startswith("```"):
            # Remove surrounding code fences.
            parts = stripped.split("```")
            # The pattern is usually ```json\n{...}\n```.
            if len(parts) >= 3:
                stripped = parts[1]

        stripped = stripped.strip()
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()

        # Attempt a full parse, otherwise try to locate the outermost braces.
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            start = stripped.find("{")
            end = stripped.rfind("}")
            if start != -1 and end != -1 and end > start:
                snippet = stripped[start : end + 1]
                return json.loads(snippet)
            raise


def _load_openai_client():
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "The openai package is required. Install dependencies via 'pip install -r requirements.txt'."
        ) from exc

    return OpenAI
