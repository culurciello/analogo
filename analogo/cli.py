"""Command-line entry point for Analogo."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

if "MPLCONFIGDIR" not in os.environ:
    cache_dir = Path(tempfile.gettempdir()) / "analogo-mplcache"
    cache_dir.mkdir(exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache_dir)
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))

from .agent import AnalogoAgent
from .research import gather_spice_references


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analogo circuit design agent")
    parser.add_argument("prompt", help="User request describing the desired circuit")
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Maximum number of design iterations (default: 5)",
    )
    parser.add_argument(
        "--provider",
        default="anthropic",
        choices=["anthropic", "openai"],
        help="LLM provider to use (default: anthropic for Claude)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model identifier. Defaults: claude-sonnet-4-5 (anthropic), gpt-4o (openai)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature for the LLM (default: 0.2)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs"),
        help="Directory where run artifacts will be written",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Set default models based on provider
    model = args.model
    if model is None:
        model = "claude-sonnet-4-5" if args.provider == "anthropic" else "gpt-4o"

    research_summary = gather_spice_references(args.prompt)
    agent = AnalogoAgent(
        prompt=args.prompt,
        max_iterations=args.iterations,
        model=model,
        provider=args.provider,
        temperature=args.temperature,
        output_root=args.output,
        research_notes=research_summary,
    )
    agent.run()
    print(json.dumps(agent.summary(), indent=2))


if __name__ == "__main__":
    main()
