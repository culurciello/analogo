"""Core Analogo agent implementation."""
from __future__ import annotations

import datetime as dt
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .diagram import draw_diagram
from .display import show_file
from .llm import LLMClient, LLMOutput
from .metadata import parse_metadata
from .plotting import plot_waveform
from .spice import SpiceRunResult, SpiceRunner

SYSTEM_PROMPT = """
You are Analogo, an autonomous analog circuit design expert. Work step-by-step to satisfy the
user's specification. In every response you MUST return valid JSON with the following schema:
{
  "action": "design" | "revise" | "final_analysis",
  "status": "testing" | "final",
  "summary": "Short natural-language explanation",
  "netlist": "Complete NGSpice netlist string or empty when no new design is proposed",
  "validation_plan": "How you expect to validate the requirements",
  "assumptions": "Key simplifying assumptions you are making"
}

Design constraints:
* When providing a netlist you MUST include an `* ANALOGO_METADATA {...}` comment containing a JSON object
  with at least a `diagram.connections` array describing component names and the two nets they connect.
  Example: `* ANALOGO_METADATA {"waveform":{"x":"time","signals":["v(in)","v(out)"]},"diagram":{"connections":[{"component":"R1 10k","from":"vin","to":"vout"}]}}`
* Include a `.control` block that writes waveform data to `analogo_waveform.csv` using WRDATA or WRDATA-like commands.
  Ensure the CSV header starts with the x-axis label (for example `time, v(out)`) and that all referenced signals exist.
  Always include at least the primary input and output node voltages (for example `v(in)` and `v(out)`) so they can be plotted.
* For AC analyses, always use a strictly positive start frequency (>= 1 Hz) even if the request mentions 0 Hz, and record
  `frequency` instead of `time` as the x-axis in WRDATA/metadata for those sweeps.
* Always finish the netlist with `.end`.
* Prefer `.options filetype=ascii` to ensure deterministic raw output.
* When status is "final" do NOT provide a new netlist; instead explain how the last simulation meets the spec.
""".strip()

INITIAL_USER_TEMPLATE = (
    "Design an analog circuit to satisfy the following request:\\n"
    "\"{prompt}\"\\n\\n"
    "Follow an iterative plan: propose a circuit, explain how you will validate it, wait for "
    "simulation results, and refine until the requirements are satisfied. Keep component values realistic. "
    "Output strictly in JSON as described in the system instructions."
)

INITIAL_REFLECTION_PROMPT = (
    "Before providing a circuit, output a THINKING step: respond with the required JSON but leave the "
    "`netlist` field empty. Summarize the key requirements and outline your validation plan."
)

REFLECTION_TO_NETLIST_PROMPT = (
    "Great. Now provide the full SPICE netlist according to your plan. Ensure the response follows "
    "the JSON schema and includes the new circuit text."
)

FOLLOWUP_REFLECTION_PROMPT = (
    "Before proposing another circuit, output a THINKING step: respond with the required JSON but leave "
    "the `netlist` empty. Reflect on what the latest simulation taught you and how you'll change the design."
)

RUN_TEST_SCRIPT = Path(__file__).resolve().parents[1] / "tests" / "run_test.py"


@dataclass
class IterationArtifact:
    iteration: int
    response: Dict[str, Any]
    spice: SpiceRunResult
    metadata: Optional[Dict[str, Any]]
    waveform_plot: Optional[Path]
    diagram_path: Optional[Path]


class AnalogoAgent:
    """Coordinates the model, ngspice simulations, and artifact generation."""

    def __init__(
        self,
        prompt: str,
        *,
        max_iterations: int = 5,
        model: str = "gpt-5-mini-2025-08-07",
        temperature: float = 0.2,
        output_root: Optional[Path] = None,
    ) -> None:
        self.prompt = prompt
        self.max_iterations = max_iterations
        self.output_root = (output_root or Path("runs")) / dt.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.thinking_log_path = self.output_root / "agent_thinking.txt"
        header = (
            f"Analogo agent thinking log\nPrompt: {prompt}\nStarted: {dt.datetime.utcnow().isoformat()}Z\n\n"
        )
        self.thinking_log_path.write_text(header)
        self.llm = LLMClient(model=model, temperature=temperature)
        self.spice = SpiceRunner()
        self.messages: List[Dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": INITIAL_USER_TEMPLATE.format(prompt=prompt)},
        ]
        self.awaiting_reflection = True
        self.messages.append({"role": "user", "content": INITIAL_REFLECTION_PROMPT})
        self.artifacts: List[IterationArtifact] = []
        self.final_response: Optional[Dict[str, Any]] = None

    def run(self) -> None:
        """Execute the iterative design loop."""

        designs_tested = 0
        allow_new_designs = True
        while True:
            llm_output = self.llm.complete(self.messages)
            self.messages.append({"role": "assistant", "content": llm_output.raw})
            response = llm_output.data
            netlist = (response.get("netlist") or "").strip()
            status = (response.get("status") or "testing").lower()
            if self.awaiting_reflection:
                label = "Reflection response"
                self._record_thinking(label, llm_output.raw)
                self.awaiting_reflection = False
                self.messages.append({"role": "user", "content": REFLECTION_TO_NETLIST_PROMPT})
                continue

            is_final = (not netlist) or (not allow_new_designs)
            label = "Final response" if is_final else f"Iteration {designs_tested + 1} response"
            self._record_thinking(label, llm_output.raw)

            if is_final:
                self.final_response = response
                break

            designs_tested += 1
            iteration_dir = self.output_root / f"iter_{designs_tested:02d}"
            run_result = self.spice.run(netlist, iteration_dir)
            metadata = parse_metadata(netlist)

            waveform_plot = None
            if run_result.waveform:
                waveform_plot = plot_waveform(
                    run_result.waveform,
                    iteration_dir / "waveform.png",
                    title=f"Iteration {designs_tested} waveform",
                )
                if waveform_plot:
                    print(f"[Analogo] Saved waveform plot to {waveform_plot}")
                    netlist_plot = run_result.netlist_path.with_suffix(".png")
                    if netlist_plot != waveform_plot:
                        shutil.copy(waveform_plot, netlist_plot)

            diagram_path = draw_diagram(metadata, iteration_dir / "diagram.png")
            self._run_external_plotter(run_result.netlist_path)

            artifact = IterationArtifact(
                iteration=designs_tested,
                response=response,
                spice=run_result,
                metadata=metadata,
                waveform_plot=waveform_plot,
                diagram_path=diagram_path,
            )
            self.artifacts.append(artifact)

            limit_reached = designs_tested >= self.max_iterations
            feedback = self._build_feedback_message(
                artifact,
                request_final=limit_reached,
                expect_reflection=not limit_reached,
            )
            self.messages.append({"role": "user", "content": feedback})

            if limit_reached:
                allow_new_designs = False
                self.awaiting_reflection = False
                # Ask the model to finalize without proposing more designs.
                self.messages.append(
                    {
                        "role": "user",
                        "content": "Iteration limit reached. Provide final analysis without generating a new netlist.",
                    }
                )
            else:
                self.awaiting_reflection = True
                self.messages.append({"role": "user", "content": FOLLOWUP_REFLECTION_PROMPT})

        self._promote_final_artifacts()

    def _record_thinking(self, label: str, content: str) -> None:
        block = f"--- {label} ---\n{content.strip()}\n"
        print(f"\n{block}")
        with self.thinking_log_path.open("a", encoding="utf-8") as handle:
            handle.write(block + "\n")

    def _build_feedback_message(
        self,
        artifact: IterationArtifact,
        *,
        request_final: bool,
        expect_reflection: bool,
    ) -> str:
        result = artifact.spice
        waveform_summary = self._summarize_waveform(result)
        status_line = "SUCCESS" if result.success else f"ERROR code {result.return_code}"
        first_lines = result.log_excerpt[-1500:]
        if request_final:
            instructions = "Focus on explaining why the behavior meets the goals and confirm readiness to finish."
        else:
            instructions = (
                "Determine whether these results satisfy the requirements. If they do, respond with `status`=\"final\" "
                "and do not provide a new netlist. Otherwise, explain the needed changes and prepare to revise."
            )
        reflection = (
            ""
            if (request_final or not expect_reflection)
            else "\nBefore proposing another circuit, respond with a THINKING step (leave `netlist` empty) that explains what you'll change."
        )
        return (
            f"Simulation results for iteration {artifact.iteration} ({status_line}).\n"
            f"Waveform summary: {waveform_summary}\n"
            f"Ngspice log excerpt:\n{first_lines}\n"
            f"{instructions} Respond with valid JSON as specified.{reflection}"
        )

    @staticmethod
    def _summarize_waveform(result: SpiceRunResult) -> str:
        waveform = result.waveform
        if not waveform:
            return "No waveform data captured."
        parts = [f"x-axis={waveform.x_label} ({len(waveform.x)} samples)"]
        for name, values in waveform.series.items():
            if not values:
                continue
            minimum = min(values)
            maximum = max(values)
            parts.append(f"{name}: min={minimum:.4g}, max={maximum:.4g}")
        return ", ".join(parts)

    def _promote_final_artifacts(self) -> None:
        if not self.artifacts:
            return
        latest = self.artifacts[-1]
        final_dir = self.output_root / "final"
        final_dir.mkdir(exist_ok=True)

        shutil.copy(latest.spice.netlist_path, final_dir / "final_netlist.sp")
        if latest.spice.log_path.exists():
            shutil.copy(latest.spice.log_path, final_dir / "ngspice.log")
        if latest.waveform_plot and latest.waveform_plot.exists():
            shutil.copy(latest.waveform_plot, final_dir / "waveform.png")
        if latest.diagram_path and latest.diagram_path.exists():
            shutil.copy(latest.diagram_path, final_dir / "diagram.png")
        self._run_external_plotter(latest.spice.netlist_path)

    def _run_external_plotter(self, netlist_path: Path) -> None:
        """Invoke the regression plotting script for additional artifacts."""

        script = RUN_TEST_SCRIPT
        if not script.exists() or not netlist_path.exists():
            return

        cmd = [sys.executable, str(script), str(netlist_path)]
        print(f"[Analogo] Running plot helper: {' '.join(cmd)}")
        try:
            result = subprocess.run(
                cmd,
                cwd=str(script.parent.parent),
                check=False,
                stdout=sys.stdout,
                stderr=sys.stderr,
            )
            if result.returncode != 0:
                print(
                    f"[Analogo] Warning: run_test.py exited with status {result.returncode}; "
                    f"see {netlist_path.with_suffix('.log')} for details."
                )
        except Exception as exc:  # pragma: no cover
            print(f"[Analogo] Warning: Unable to run run_test.py ({exc})")

    def summary(self) -> Dict[str, Any]:
        last_artifact = self.artifacts[-1] if self.artifacts else None
        return {
            "output_root": str(self.output_root),
            "iterations": len(self.artifacts),
            "final_response": self.final_response,
            "latest_artifact": {
                "iteration": last_artifact.iteration if last_artifact else None,
                "netlist": str(last_artifact.spice.netlist_path) if last_artifact else None,
                "log": str(last_artifact.spice.log_path) if last_artifact else None,
                "waveform_plot": str(last_artifact.waveform_plot) if last_artifact else None,
                "diagram": str(last_artifact.diagram_path) if last_artifact else None,
            },
        }
