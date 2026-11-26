from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from agent.brain import AsyncLLMClient, PlannedStep, _clamp01, _extract_json_block


@dataclass
class CritiqueDecision:
    """Diagnosis and recovery choice after a failed or unexpected step."""

    root_cause: str
    recovery_action: str  # "retry_adjusted" | "rollback_and_retry" | "replan" | "abort"
    adjusted_args: dict[str, Any]
    alternative_tool: str | None
    lesson_learned: str
    confidence: float


class CriticEngine:
    """Diagnoses failed or stalled steps and chooses a concrete recovery path."""

    def __init__(self, llm: AsyncLLMClient, max_retries_per_step: int = 2) -> None:
        self.llm = llm
        self.max_retries_per_step = max(1, int(max_retries_per_step))

    async def diagnose_and_recover(
        self,
        goal: str,
        step: PlannedStep,
        observation: dict[str, Any],
        attempt_num: int,
        recent_history: list[dict[str, Any]] | None = None,
    ) -> CritiqueDecision:
        if attempt_num >= self.max_retries_per_step:
            return CritiqueDecision(
                root_cause=str(observation.get("error") or observation.get("stderr") or "Step failed repeatedly."),
                recovery_action="replan",
                adjusted_args={},
                alternative_tool=None,
                lesson_learned=(
                    f"Tool '{step.tool}' for '{step.title}' failed after {attempt_num} attempts; "
                    f"consider an alternative strategy or smaller sub-steps."
                ),
                confidence=0.85,
            )

        system_prompt = (
            "You are a sharp debugging and recovery critic for an autonomous computer agent. "
            "Given a goal, a planned step, the observed output/error, and recent history, "
            "diagnose the exact root cause, choose a recovery action "
            "('retry_adjusted', 'rollback_and_retry', 'replan', or 'abort'), propose adjusted "
            "arguments or an alternative tool if helpful, and write a concise reusable lesson. "
            "Respond ONLY with valid JSON."
        )

        user_prompt = json.dumps(
            {
                "goal": goal,
                "attempt_num": attempt_num,
                "max_retries_per_step": self.max_retries_per_step,
                "failed_step": {
                    "step_num": step.step_num,
                    "title": step.title,
                    "tool": step.tool,
                    "args": step.args,
                    "verification_hint": step.verification_hint,
                },
                "observation": observation,
                "recent_history": (recent_history or [])[-5:],
                "required_schema": {
                    "root_cause": "string",
                    "recovery_action": "retry_adjusted | rollback_and_retry | replan | abort",
                    "adjusted_args": {},
                    "alternative_tool": "string or null",
                    "lesson_learned": "string",
                    "confidence": "0.0 to 1.0",
                },
            },
            indent=2,
            default=str,
        )

        raw = await self.llm.complete_json(system_prompt=system_prompt, user_prompt=user_prompt)
        data = _extract_json_block(raw)

        valid_actions = {"retry_adjusted", "rollback_and_retry", "replan", "abort"}
        action = str(data.get("recovery_action", "retry_adjusted")).strip()
        if action not in valid_actions:
            action = "retry_adjusted"

        adj_args = data.get("adjusted_args", {})
        if not isinstance(adj_args, dict):
            adj_args = {}

        alt_tool = data.get("alternative_tool")
        if alt_tool is not None:
            alt_tool = str(alt_tool).strip() or None

        return CritiqueDecision(
            root_cause=str(data.get("root_cause", "Unclassified step failure.")),
            recovery_action=action,
            adjusted_args=adj_args,
            alternative_tool=alt_tool,
            lesson_learned=str(
                data.get(
                    "lesson_learned",
                    f"Avoid repeating exact args for '{step.title}' without checking preconditions.",
                )
            ),
            confidence=_clamp01(data.get("confidence"), 0.7),
        )