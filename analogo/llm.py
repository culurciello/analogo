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
    """Thin wrapper around LLM APIs supporting both Claude and OpenAI."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-5",
        temperature: float = 0.2,
        max_output_tokens: int = 8192,
        provider: str = "anthropic"
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.provider = provider.lower()

        if self.provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise RuntimeError("ANTHROPIC_API_KEY environment variable is required for Claude.")
            self._client = _load_anthropic_client()(api_key=api_key)
        elif self.provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY environment variable is required for OpenAI.")
            self._client = _load_openai_client()(api_key=api_key)
        else:
            raise ValueError(f"Unsupported provider: {provider}. Use 'anthropic' or 'openai'.")

    def complete(self, messages: Sequence[Dict[str, str]]) -> LLMOutput:
        """Call the Chat Completions API and return parsed JSON content."""

        if self.provider == "anthropic":
            # Claude API requires system message separate from messages
            system_msg = ""
            user_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    system_msg = msg["content"]
                else:
                    user_messages.append(msg)

            response = self._client.messages.create(
                model=self.model,
                max_tokens=self.max_output_tokens,
                temperature=self.temperature,
                system=system_msg,
                messages=user_messages,
            )
            if response.content:
                content = response.content[0].text
            else:
                # Debug: print stop reason if content is empty
                stop_reason = getattr(response, 'stop_reason', 'unknown')
                raise ValueError(
                    f"Claude returned empty content. Stop reason: {stop_reason}. "
                    f"This may indicate max_tokens was too low or content filtering occurred."
                )
        else:
            # OpenAI API
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
            raise ValueError(
                "Empty response from language model. This may indicate:\n"
                "1. Token limit exceeded (try increasing max_output_tokens)\n"
                "2. Content filtering blocked the response\n"
                "3. API rate limiting or timeout\n"
                "Original content length: 0"
            )

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


def _load_anthropic_client():
    try:
        from anthropic import Anthropic
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "The anthropic package is required. Install it via 'pip install anthropic'."
        ) from exc

    return Anthropic
