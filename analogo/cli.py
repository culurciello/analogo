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
        "--model",
        default="gpt-4o-mini",
        help="OpenAI model identifier (default: gpt-4o-mini)",
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
    agent = AnalogoAgent(
        prompt=args.prompt,
        max_iterations=args.iterations,
        model=args.model,
        temperature=args.temperature,
        output_root=args.output,
    )
    agent.run()
    print(json.dumps(agent.summary(), indent=2))


if __name__ == "__main__":
    main()
