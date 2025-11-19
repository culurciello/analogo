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
You are Analogo, an expert analog circuit design assistant specializing in SPICE simulation and practical circuit design.

## OUTPUT FORMAT
Every response MUST be valid JSON with this exact schema:
{
  "action": "design" | "revise" | "final_analysis",
  "status": "testing" | "final",
  "thought_process": "Detailed reasoning about circuit behavior, component selection, and expected results",
  "netlist": "Complete NGSpice netlist (or empty string if not providing new design)",
  "validation_plan": "Specific metrics and measurements to verify requirements",
  "assumptions": "Key assumptions and design choices"
}

## CIRCUIT DESIGN PRINCIPLES

### Component Value Selection
1. **Resistors**: Use standard E12/E24 values (1.0k, 2.2k, 4.7k, 10k, 22k, 47k, 100k, etc.)
2. **Capacitors**: Common values (1pF, 10pF, 100pF, 1nF, 10nF, 100nF, 1uF, 10uF, 100uF)
3. **Calculate from specifications**: For RC filters, use f_c = 1/(2πRC) to derive component values
4. **Realistic ranges**:
   - Resistors: 100Ω to 1MΩ (prefer 1kΩ-100kΩ for general use)
   - Capacitors: 1pF to 100uF (avoid extreme values without justification)

### Validation Metrics
Be specific about what you will measure:
- **Cutoff frequency**: Measure -3dB point (0.707 × DC gain)
- **Passband**: Verify gain flatness within ±1dB
- **Stopband attenuation**: Check roll-off rate (dB/decade)
- **Time domain**: Rise time, settling time, overshoot
- **DC operating point**: Node voltages, currents

## SPICE NETLIST REQUIREMENTS

### Mandatory Elements
1. **Title line**: First line describes the circuit
2. **Ground node**: Must have node 0 (ground reference)
3. **Metadata comment**: Include circuit topology for visualization
4. **Control block**: Analysis commands and data export
5. **End statement**: `.end` as final line

### Metadata Format (REQUIRED)
```
* ANALOGO_METADATA {"waveform":{"x":"frequency","signals":["v_in","v_out"]},"diagram":{"connections":[{"component":"R1 10k","from":"v_in","to":"v_out"},{"component":"C1 10n","from":"v_out","to":"0"}]}}
```

### Analysis Types
**AC Analysis** (frequency response):
```
.ac dec 20 1 100k
.control
ac dec 20 1 100k
set wr_vecnames
option numdgt=6
wrdata analogo_waveform.csv frequency v_in v_out
.endc
```
- IMPORTANT: Use exactly 20 total points: `dec 20 fstart fstop` (20 points per decade) or `lin 20 fstart fstop`
- For filters, sweep at least 2 decades below and 2 decades above cutoff frequency
- NEVER start from 0 Hz (use 0.1 Hz or 1 Hz minimum)
- X-axis is "frequency" (not "time")

**Transient Analysis** (time response):
```
.tran 0.01m 10m
.control
run
set wr_vecnames
option numdgt=6
wrdata analogo_waveform.csv time v_in v_out
.endc
```
- Specify timestep and stop time: `.tran tstep tstop`
- X-axis is "time"

### Component Syntax
- Resistor: `R1 node1 node2 10k`
- Capacitor: `C1 node1 node2 10n`
- Voltage source (DC): `V1 node+ node- DC 5`
- Voltage source (AC): `V1 node+ node- DC 0 AC 1` (1V amplitude for AC analysis)
- Voltage source (Sine): `V1 node+ node- SIN(offset amplitude frequency)`

### Common Mistakes to AVOID
❌ Starting AC analysis at 0 Hz (use ≥ 1 Hz)
❌ Forgetting ground node (0)
❌ Missing .control block or data export
❌ Unrealistic component values (e.g., 1e-20F, 1e15Ω)
❌ Wrong x-axis label (use "frequency" for AC, "time" for transient)
❌ Not calculating component values from specifications
❌ Missing AC source amplitude in AC analysis

## EXAMPLE: 1 kHz RC LOWPASS FILTER

```
RC Lowpass Filter - 1kHz cutoff
* ANALOGO_METADATA {"waveform":{"x":"frequency","signals":["v_in","v_out"]},"diagram":{"connections":[{"component":"V1 1V","from":"v_in","to":"0"},{"component":"R1 10k","from":"v_in","to":"v_out"},{"component":"C1 15.9n","from":"v_out","to":"0"}]}}

V1 v_in 0 DC 0 AC 1
R1 v_in v_out 10k
C1 v_out 0 15.9n

.ac dec 20 10 100k
.control
ac dec 20 10 100k
set wr_vecnames
option numdgt=6
wrdata analogo_waveform.csv frequency v_in v_out
.endc
.options filetype=ascii
.end
```

**Validation Criteria** (MUST be checked to determine if design is acceptable):
- At 1 kHz: |V_out/V_in| should be 0.7 to 0.72 (approximately 0.707 = -3dB point)
- Below 100 Hz: Flat passband, gain ≈ 1.0 (0dB), variation < 5%
- Above 10 kHz: Roll-off at approximately -20dB/decade
- If ALL criteria are met within 10% tolerance, set status="final"

## ITERATION PROTOCOL
1. **First design**: Calculate component values from specifications, provide complete netlist
2. **Review results**: Compare simulation to requirements using specific metrics
3. **Revise if needed**: Adjust component values with clear reasoning
4. **Final**: When requirements are met, set status="final" and netlist=""

Always think through the circuit physics and expected behavior before providing a design.
""".strip()

INITIAL_VALIDATION_PROMPT = (
    "Before designing the circuit, first define the VALIDATION METRICS you will use to verify success.\\n\\n"
    "Request: \"{prompt}\"\\n\\n"
    "Respond with JSON containing:\\n"
    "- thought_process: What specific metrics will you measure? What values indicate success?\\n"
    "- validation_plan: Concrete pass/fail criteria with numerical thresholds\\n"
    "- netlist: (empty string for this planning step)\\n"
    "- status: \"testing\"\\n\\n"
    "For example, for an RC lowpass filter at 1kHz:\\n"
    "- At 1000 Hz: V_out/V_in should be 0.70-0.72 (within 2% of 0.707)\\n"
    "- Below 100 Hz: V_out/V_in should be 0.95-1.0 (flat passband)\\n"
    "- Above 10 kHz: Roll-off should be 18-22 dB/decade\\n"
    "Be specific with numbers and tolerances."
)

DESIGN_PROMPT = (
    "Now provide the complete SPICE netlist that implements your design. "
    "Calculate component values from the specifications and validation criteria you defined. "
    "Include the full netlist in the `netlist` field."
)

RUN_TEST_SCRIPT = Path(__file__).resolve().parents[1] / "tests" / "run_test.py"


@dataclass
class IterationArtifact:
    iteration: int
    response: Dict[str, Any]
    spice: SpiceRunResult
    metadata: Optional[Dict[str, Any]]
    waveform_plot: Optional[Path]
    waveform_data: Optional[Path]
    raw_data: Optional[Path]
    diagram_path: Optional[Path]


class AnalogoAgent:
    """Coordinates the model, ngspice simulations, and artifact generation."""

    def __init__(
        self,
        prompt: str,
        *,
        max_iterations: int = 5,
        model: str = "claude-sonnet-4-5",
        provider: str = "anthropic",
        temperature: float = 0.2,
        output_root: Optional[Path] = None,
        research_notes: Optional[str] = None,
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
        self.llm = LLMClient(model=model, temperature=temperature, provider=provider)
        self.spice = SpiceRunner()
        self.messages: List[Dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": INITIAL_VALIDATION_PROMPT.format(prompt=prompt)},
        ]
        if research_notes:
            summary = research_notes.strip()
            if summary:
                self.messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Pre-run web research on similar SPICE circuits:\n"
                            f"{summary}\n\n"
                            "Incorporate any relevant insights or component values into your validation plan and design."
                        ),
                    }
                )
        self.awaiting_validation_plan = True
        self.awaiting_goal_check = False
        self.goal_check_request_final = False
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

            # Handle validation planning step
            if self.awaiting_validation_plan:
                label = "Validation planning"
                self._record_thinking(label, llm_output.raw)
                self.awaiting_validation_plan = False
                # Now ask for the actual design
                self.messages.append({"role": "user", "content": DESIGN_PROMPT})
                continue

            if self.awaiting_goal_check:
                label = "Goal check response"
                self._record_thinking(label, llm_output.raw)
                self.awaiting_goal_check = False
                request_final = self.goal_check_request_final
                self.goal_check_request_final = False
                if status == "final":
                    self.final_response = response
                    break
                if request_final:
                    allow_new_designs = False
                    self.messages.append(
                        {
                            "role": "user",
                            "content": "Iteration limit reached. Provide final analysis without generating a new netlist.",
                        }
                    )
                    continue
                # If model wants to revise, ask for the revised netlist
                if not netlist:
                    self.messages.append(
                        {
                            "role": "user",
                            "content": "Provide the revised SPICE netlist based on your analysis. Include the complete netlist in the `netlist` field of your JSON response.",
                        }
                    )
                    continue
                # Model provided a netlist in the goal check response, fall through to process it
                # (though this is unusual - normally netlist should be empty during goal check)

            is_final = (not netlist) or (not allow_new_designs) or (status == "final")
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

            waveform_data = None
            if run_result.csv_path.exists():
                waveform_data = iteration_dir / f"waveform_data_iter_{designs_tested:02d}.csv"
                shutil.copy(run_result.csv_path, waveform_data)

            raw_data = None
            if run_result.raw_path.exists():
                raw_data = iteration_dir / f"raw_data_iter_{designs_tested:02d}.raw"
                shutil.copy(run_result.raw_path, raw_data)

            diagram_path = draw_diagram(metadata, iteration_dir / "diagram.png")
            # self._run_external_plotter(run_result.netlist_path)

            artifact = IterationArtifact(
                iteration=designs_tested,
                response=response,
                spice=run_result,
                metadata=metadata,
                waveform_plot=waveform_plot,
                waveform_data=waveform_data,
                raw_data=raw_data,
                diagram_path=diagram_path,
            )
            self.artifacts.append(artifact)

            limit_reached = designs_tested >= self.max_iterations
            self._queue_goal_check(artifact, request_final=limit_reached or (not allow_new_designs))
            self.awaiting_goal_check = True
            self.goal_check_request_final = limit_reached or (not allow_new_designs)
            if limit_reached:
                allow_new_designs = False

        self._promote_final_artifacts()

    def _record_thinking(self, label: str, content: str) -> None:
        block = f"--- {label} ---\n{content.strip()}\n"
        print(f"\n{block}")
        with self.thinking_log_path.open("a", encoding="utf-8") as handle:
            handle.write(block + "\n")

    def _queue_goal_check(
        self,
        artifact: IterationArtifact,
        *,
        request_final: bool,
    ) -> None:
        prompt = self._build_goal_message(artifact, request_final=request_final)
        self.messages.append({"role": "user", "content": prompt})

    def _build_goal_message(
        self,
        artifact: IterationArtifact,
        *,
        request_final: bool,
    ) -> str:
        result = artifact.spice
        waveform_summary = self._summarize_waveform(result)
        analysis_hints = self._generate_analysis_hints(result)
        status_line = "SUCCESS" if result.success else f"ERROR code {result.return_code}"
        log_excerpt = result.log_excerpt or "[no ngspice output captured]"
        if request_final:
            instructions = (
                "This is your final iteration. Analyze how well the design meets the requirements. "
                "Respond with `status`=\"final\" and set `netlist` to empty string."
            )
        else:
            instructions = (
                "Compare the simulation results to YOUR validation criteria from the planning phase.\n\n"
                "DECISION CRITERIA:\n"
                "✓ If ALL validation metrics are met within acceptable tolerance (typically ±10%):\n"
                "  → Set `status`=\"final\" and `netlist`=\"\"\n"
                "  → Explain how each metric was satisfied\n\n"
                "✗ If ANY validation metric fails:\n"
                "  → Set `status`=\"testing\" and `netlist`=\"\"\n"
                "  → Explain which metric(s) failed and why\n"
                "  → Describe specific component changes needed\n"
                "  → On next turn, provide the revised netlist\n\n"
                "Example for RC lowpass at 1kHz:\n"
                "- Target: V_out/V_in = 0.707 at 1kHz\n"
                "- Measured: 0.715 at 1kHz\n"
                "- Error: (0.715-0.707)/0.707 = 1.1% ✓ PASS (within 10%)\n"
                "- If all other metrics pass → status=\"final\""
            )
        return (
            f"## Simulation Results - Iteration {artifact.iteration} ({status_line})\n\n"
            f"### Waveform Data:\n{waveform_summary}\n\n"
            f"### Analysis Hints:\n{analysis_hints}\n\n"
            f"### NGSpice Log:\n{log_excerpt}\n\n"
            f"{instructions}"
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

    @staticmethod
    def _generate_analysis_hints(result: SpiceRunResult) -> str:
        """Provide quantitative analysis hints to help the model evaluate results."""
        import math

        waveform = result.waveform
        if not waveform or not waveform.series:
            return "Unable to generate analysis hints - no waveform data."

        hints = []
        x_data = waveform.x
        x_label = waveform.x_label.lower()

        # Check for AC analysis
        if "freq" in x_label and len(x_data) > 2:
            # Find input and output signals
            v_in = None
            v_out = None
            for key in waveform.series:
                key_lower = key.lower()
                if "v_in" in key_lower or key == "v(v_in)":
                    v_in = waveform.series[key]
                elif "v_out" in key_lower or key == "v(v_out)":
                    v_out = waveform.series[key]

            if v_in and v_out and len(v_in) == len(v_out):
                # Calculate gain at various points
                dc_gain = v_out[0] / v_in[0] if v_in[0] > 0 else 0
                dc_gain_db = 20 * math.log10(dc_gain) if dc_gain > 0 else float("-inf")
                hints.append(f"DC gain (lowest freq): {dc_gain:.4f} ({dc_gain_db:.2f} dB)")

                # Find -3dB point (0.707 of DC gain)
                target_gain = v_out[0] / math.sqrt(2) if v_in[0] > 0 else 0
                cutoff_freq = None
                cutoff_gain = None
                cutoff_ratio = None
                for i, (freq, vo, vi) in enumerate(zip(x_data, v_out, v_in)):
                    if vo <= target_gain:
                        cutoff_freq = freq
                        cutoff_gain = vo / vi if vi > 0 else 0
                        cutoff_ratio = cutoff_gain / dc_gain if dc_gain > 0 else 0
                        break
                if cutoff_freq:
                    hints.append(
                        f"-3dB cutoff frequency: ~{cutoff_freq:.1f} Hz "
                        f"(gain ratio = {cutoff_ratio:.3f}, target = 0.707)"
                    )

                # Check roll-off if we have enough data
                if len(x_data) > 10:
                    mid_idx = len(x_data) // 2
                    end_idx = len(x_data) - 1
                    if v_in[mid_idx] > 0 and v_in[end_idx] > 0:
                        gain_mid = 20 * math.log10(v_out[mid_idx] / v_in[mid_idx])
                        gain_end = 20 * math.log10(v_out[end_idx] / v_in[end_idx])
                        freq_ratio = x_data[end_idx] / x_data[mid_idx]
                        if freq_ratio > 1:
                            rolloff = (gain_end - gain_mid) / math.log10(freq_ratio)
                            hints.append(f"Roll-off rate (high freq): ~{rolloff:.1f} dB/decade")

        # Check for time domain analysis
        elif "time" in x_label and len(x_data) > 2:
            for name, values in waveform.series.items():
                if len(values) > 10:
                    avg_val = sum(values) / len(values)
                    hints.append(f"{name} average: {avg_val:.4g}")
                    # Check settling
                    final_val = values[-1]
                    hints.append(f"{name} final value: {final_val:.4g}")

        if not hints:
            hints.append("Review the waveform data to verify it matches your expectations.")

        return "\n".join(f"- {h}" for h in hints)

    def _promote_final_artifacts(self) -> None:
        if not self.artifacts:
            return
        latest = self.artifacts[-1]
        final_dir = self.output_root / "final"
        final_dir.mkdir(exist_ok=True)

        shutil.copy(latest.spice.netlist_path, final_dir / "final_netlist.sp")
        if latest.spice.log_path.exists():
            shutil.copy(latest.spice.log_path, final_dir / "circuit.log")
        if latest.waveform_plot and latest.waveform_plot.exists():
            shutil.copy(latest.waveform_plot, final_dir / "waveform.png")
        if latest.waveform_data and latest.waveform_data.exists():
            shutil.copy(latest.waveform_data, final_dir / "waveform_data.csv")
        if latest.raw_data and latest.raw_data.exists():
            shutil.copy(latest.raw_data, final_dir / "sim.raw")
        if latest.diagram_path and latest.diagram_path.exists():
            shutil.copy(latest.diagram_path, final_dir / "diagram.png")


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
                "waveform_data": str(last_artifact.waveform_data) if last_artifact else None,
                "raw_data": str(last_artifact.raw_data) if last_artifact else None,
                "diagram": str(last_artifact.diagram_path) if last_artifact else None,
            },
        }
