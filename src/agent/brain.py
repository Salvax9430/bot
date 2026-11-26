from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Protocol


class AsyncLLMClient(Protocol):
    """Minimal async interface any LLM backend can implement."""

    async def complete_json(self, system_prompt: str, user_prompt: str) -> str: ...


@dataclass
class ApproachOption:
    name: str
    strategy_type: str  # "programmatic" | "hybrid" | "gui"
    summary: str
    pros: list[str] = field(default_factory=list)
    cons: list[str] = field(default_factory=list)
    reliability_score: float = 0.5
    safety_score: float = 0.5
    speed_score: float = 0.5
    total_score: float = 0.0


@dataclass
class PlannedStep:
    step_num: int
    title: str
    tool: str
    args: dict[str, Any]
    verification_hint: str


@dataclass
class ExecutionPlan:
    goal: str
    options: list[ApproachOption]
    chosen: ApproachOption
    steps: list[PlannedStep]


def _extract_json_block(raw_text: str) -> dict[str, Any]:
    """Parse JSON cleanly even if wrapped in ```json fences or extra commentary."""
    cleaned = raw_text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL)
    if fenced:
        cleaned = fenced.group(1)
    else:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            cleaned = cleaned[start : end + 1]
    return json.loads(cleaned)


def _clamp01(value: Any, default: float = 0.5) -> float:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, num))


class BrainEngine:
    """Models human-style brainstorming, trade-off scoring, and step planning."""

    def __init__(self, llm: AsyncLLMClient) -> None:
        self.llm = llm

    async def build_plan(
        self,
        goal: str,
        context: dict[str, Any] | None = None,
        past_lessons: list[str] | None = None,
    ) -> ExecutionPlan:
        ctx = context or {}
        lessons = past_lessons or []

        system_prompt = (
            "You are a sharp, disciplined engineering & computer-use cognitive planner. "
            "Before acting, brainstorm multiple distinct ways to achieve the goal, evaluate "
            "trade-offs honestly, pick the strongest approach, and break it into concrete, "
            "verifiable steps. Prefer safe, reversible, programmatic bridges (files, scripts, "
            "APIs, tests) over fragile GUI clicks whenever possible. Respond ONLY with valid JSON."
        )

        user_prompt = json.dumps(
            {
                "goal": goal,
                "context": ctx,
                "past_lessons": lessons[-10:],
                "required_schema": {
                    "options": [
                        {
                            "name": "string",
                            "strategy_type": "programmatic | hybrid | gui",
                            "summary": "string",
                            "pros": ["string"],
                            "cons": ["string"],
                            "reliability_score": "0.0 to 1.0",
                            "safety_score": "0.0 to 1.0",
                            "speed_score": "0.0 to 1.0",
                        }
                    ],
                    "chosen_index": 0,
                    "steps": [
                        {
                            "step_num": 1,
                            "title": "string",
                            "tool": "string",
                            "args": {},
                            "verification_hint": "string",
                        }
                    ],
                },
            },
            indent=2,
            default=str,
        )

        raw = await self.llm.complete_json(system_prompt=system_prompt, user_prompt=user_prompt)
        data = _extract_json_block(raw)

        parsed_options: list[ApproachOption] = []
        for item in data.get("options", []):
            rel = _clamp01(item.get("reliability_score"), 0.6)
            saf = _clamp01(item.get("safety_score"), 0.7)
            spd = _clamp01(item.get("speed_score"), 0.6)
            # Weighted score: reliability and safety lead, speed supports
            weighted = round((0.45 * rel) + (0.35 * saf) + (0.20 * spd), 4)
            parsed_options.append(
                ApproachOption(
                    name=str(item.get("name", "Unnamed Approach")),
                    strategy_type=str(item.get("strategy_type", "programmatic")),
                    summary=str(item.get("summary", "")),
                    pros=[str(p) for p in item.get("pros", [])],
                    cons=[str(c) for c in item.get("cons", [])],
                    reliability_score=rel,
                    safety_score=saf,
                    speed_score=spd,
                    total_score=weighted,
                )
            )

        if not parsed_options:
            parsed_options = [
                ApproachOption(
                    name="Direct Inspect & Execute",
                    strategy_type="programmatic",
                    summary="Inspect workspace state first, execute a small safe step, and verify.",
                    pros=["Safe", "Observable"],
                    cons=["May require multiple iterations"],
                    reliability_score=0.7,
                    safety_score=0.85,
                    speed_score=0.65,
                    total_score=0.7425,
                )
            ]

        chosen_idx = data.get("chosen_index", 0)
        if not isinstance(chosen_idx, int) or not (0 <= chosen_idx < len(parsed_options)):
            chosen_idx = max(range(len(parsed_options)), key=lambda i: parsed_options[i].total_score)

        parsed_steps: list[PlannedStep] = []
        for idx, st in enumerate(data.get("steps", []), start=1):
            parsed_steps.append(
                PlannedStep(
                    step_num=int(st.get("step_num", idx)),
                    title=str(st.get("title", f"Step {idx}")),
                    tool=str(st.get("tool", "run_command")),
                    args=dict(st.get("args", {})) if isinstance(st.get("args"), dict) else {},
                    verification_hint=str(st.get("verification_hint", "Check output and state.")),
                )
            )

        return ExecutionPlan(
            goal=goal,
            options=parsed_options,
            chosen=parsed_options[chosen_idx],
            steps=parsed_steps,
        )