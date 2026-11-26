from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import replace
from typing import Any

from agent.brain import BrainEngine, ExecutionPlan, PlannedStep
from agent.critic import CriticEngine
from agent.guard import WorkspaceGuard
from agent.memory import ExperienceStore
from agent.skills import SkillStore

ToolHandler = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


class AgentLoop:
    """Runs BrainEngine plans, verifies tool results, and recovers via CriticEngine."""

    def __init__(
        self,
        brain: BrainEngine,
        critic: CriticEngine,
        guard: WorkspaceGuard,
        memory: ExperienceStore,
        extra_tools: dict[str, ToolHandler] | None = None,
        max_replans: int = 2,
        skill_store: SkillStore | None = None,
    ) -> None:
        self.brain = brain
        self.critic = critic
        self.guard = guard
        self.memory = memory
        self.extra_tools: dict[str, ToolHandler] = extra_tools or {}
        self.max_replans = max(0, int(max_replans))
        self.skill_store = skill_store
        self.learned_lessons: list[str] = []

    async def _dispatch_tool(self, tool: str, args: dict[str, Any]) -> dict[str, Any]:
        if tool == "read_text_file":
            path = args.get("path", "")
            if not path:
                return {"ok": False, "error": "Missing 'path' argument for read_text_file."}
            return self.guard.read_text_file(path)

        if tool == "write_text_file":
            path = args.get("path", "")
            content = args.get("content", "")
            if not path:
                return {"ok": False, "error": "Missing 'path' argument for write_text_file."}
            return self.guard.write_text_file(path, str(content))

        if tool == "run_command":
            command = args.get("command", "")
            timeout_sec = args.get("timeout_sec")
            if not command:
                return {"ok": False, "error": "Missing 'command' argument for run_command."}
            return self.guard.run_command(str(command), timeout_sec=timeout_sec)

        handler = self.extra_tools.get(tool)
        if handler is not None:
            return await handler(args)

        return {"ok": False, "error": f"Unsupported tool '{tool}'."}

    async def run_goal(
        self,
        goal: str,
        context: dict[str, Any] | None = None,
        stop_event: asyncio.Event | None = None,
    ) -> dict[str, Any]:
        run_id = self.memory.start_run(goal=goal)
        ctx = dict(context or {})
        if self.skill_store is not None and "relevant_skills" not in ctx:
            ctx["relevant_skills"] = [
                {
                    "name": s.name,
                    "domain": s.domain,
                    "description": s.description,
                    "steps": s.steps,
                    "tags": s.tags,
                }
                for s in self.skill_store.search_skills(query=goal, limit=5)
            ]
        recent_history: list[dict[str, Any]] = []
        replans_used = 0
        global_step_num = 0

        try:
            while True:
                if stop_event and stop_event.is_set():
                    summary = "Stopped by user stop signal."
                    self.memory.finish_run(run_id, status="stopped", final_summary=summary)
                    return {"run_id": run_id, "status": "stopped", "summary": summary, "lessons": self.learned_lessons}

                plan: ExecutionPlan = await self.brain.build_plan(
                    goal=goal,
                    context={**ctx, "recent_history": recent_history[-5:]},
                    past_lessons=self.learned_lessons,
                )
                self.memory.save_brainstorm(
                    run_id=run_id,
                    ideas=[
                        {
                            "name": opt.name,
                            "strategy_type": opt.strategy_type,
                            "summary": opt.summary,
                            "pros": opt.pros,
                            "cons": opt.cons,
                            "scores": {
                                "reliability": opt.reliability_score,
                                "safety": opt.safety_score,
                                "speed": opt.speed_score,
                                "total": opt.total_score,
                            },
                        }
                        for opt in plan.options
                    ],
                    chosen_idea=plan.chosen.name,
                )

                need_replan = False

                for step in plan.steps:
                    if stop_event and stop_event.is_set():
                        summary = "Stopped mid-plan by user stop signal."
                        self.memory.finish_run(run_id, status="stopped", final_summary=summary)
                        return {"run_id": run_id, "status": "stopped", "summary": summary, "lessons": self.learned_lessons}

                    current_step: PlannedStep = step
                    attempt = 0

                    while True:
                        attempt += 1
                        global_step_num += 1
                        obs = await self._dispatch_tool(current_step.tool, current_step.args)
                        ok = bool(obs.get("ok", False))

                        thought = (
                            f"[{plan.chosen.name}] Step {current_step.step_num} "
                            f"(attempt {attempt}): {current_step.title} "
                            f"(verify: {current_step.verification_hint})"
                        )
                        self.memory.log_step(
                            run_id=run_id,
                            step_num=global_step_num,
                            thought=thought,
                            action={"tool": current_step.tool, "args": current_step.args},
                            observation=obs,
                            verified_ok=ok,
                        )
                        recent_history.append(
                            {
                                "step_num": global_step_num,
                                "tool": current_step.tool,
                                "ok": ok,
                                "observation": obs,
                            }
                        )

                        if ok:
                            break

                        critique = await self.critic.diagnose_and_recover(
                            goal=goal,
                            step=current_step,
                            observation=obs,
                            attempt_num=attempt,
                            recent_history=recent_history,
                        )
                        if critique.lesson_learned and critique.lesson_learned not in self.learned_lessons:
                            self.learned_lessons.append(critique.lesson_learned)

                        if critique.recovery_action == "abort":
                            summary = f"Aborted after failure: {critique.root_cause}"
                            self.memory.finish_run(run_id, status="aborted", final_summary=summary)
                            return {
                                "run_id": run_id,
                                "status": "aborted",
                                "summary": summary,
                                "lessons": self.learned_lessons,
                            }

                        if critique.recovery_action == "replan":
                            need_replan = True
                            break

                        # retry_adjusted or rollback_and_retry
                        merged_args = {**current_step.args, **critique.adjusted_args}
                        next_tool = critique.alternative_tool or current_step.tool
                        current_step = replace(
                            current_step,
                            tool=next_tool,
                            args=merged_args,
                            title=f"{current_step.title} (recovered: {critique.root_cause[:60]})",
                        )

                    if need_replan:
                        break

                if not need_replan:
                    summary = f"Completed goal '{goal}' using '{plan.chosen.name}'."
                    self.memory.finish_run(run_id, status="completed", final_summary=summary)
                    return {
                        "run_id": run_id,
                        "status": "completed",
                        "summary": summary,
                        "lessons": self.learned_lessons,
                    }

                replans_used += 1
                if replans_used > self.max_replans:
                    summary = f"Goal incomplete after {replans_used} replans."
                    self.memory.finish_run(run_id, status="failed", final_summary=summary)
                    return {
                        "run_id": run_id,
                        "status": "failed",
                        "summary": summary,
                        "lessons": self.learned_lessons,
                    }

        except Exception as exc:
            summary = f"Unhandled exception in AgentLoop: {exc}"
            self.memory.finish_run(run_id, status="error", final_summary=summary)
            return {
                "run_id": run_id,
                "status": "error",
                "summary": summary,
                "lessons": self.learned_lessons,
            }