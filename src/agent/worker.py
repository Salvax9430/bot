from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent.loop import AgentLoop
from agent.skills import SkillStore


@dataclass
class WorkItem:
    goal: str
    context: dict[str, Any] = field(default_factory=dict)


class SilentBackgroundWorker:
    """Asynchronous queue worker that enriches goals with skills and runs AgentLoop."""

    def __init__(
        self,
        agent_loop: AgentLoop,
        skills: SkillStore,
        log_path: Path,
        poll_interval_sec: float = 1.5,
        stop_event: asyncio.Event | None = None,
    ) -> None:
        self.agent_loop = agent_loop
        self.skills = skills
        self.log_path = log_path.resolve()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.poll_interval_sec = max(0.2, float(poll_interval_sec))
        self.stop_event = stop_event if stop_event is not None else asyncio.Event()
        self._queue: asyncio.Queue[WorkItem] = asyncio.Queue()
        self._results: list[dict[str, Any]] = []

    def _write_log(self, event_type: str, payload: dict[str, Any]) -> None:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event_type,
            "payload": payload,
        }
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")

    def enqueue_goal(self, goal: str, context: dict[str, Any] | None = None) -> int:
        item = WorkItem(goal=goal.strip(), context=dict(context or {}))
        self._queue.put_nowait(item)
        q_size = self._queue.qsize()
        self._write_log("goal_enqueued", {"goal": item.goal, "queue_size": q_size})
        return q_size

    def _build_enriched_context(self, item: WorkItem) -> dict[str, Any]:
        ctx = dict(item.context)
        matches = self.skills.search_skills(query=item.goal, limit=5)
        ctx["relevant_skills"] = [
            {
                "name": rec.name,
                "domain": rec.domain,
                "description": rec.description,
                "steps": rec.steps,
                "tags": rec.tags,
            }
            for rec in matches
        ]
        return ctx

    async def process_next(self) -> dict[str, Any] | None:
        try:
            item = self._queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

        self._write_log("goal_started", {"goal": item.goal})
        enriched_ctx = self._build_enriched_context(item)
        try:
            result = await self.agent_loop.run_goal(
                goal=item.goal,
                context=enriched_ctx,
                stop_event=self.stop_event,
            )
            self._results.append(result)
            self._write_log("goal_finished", {"goal": item.goal, "result": result})
            return result
        except Exception as exc:
            err_res = {"status": "worker_error", "goal": item.goal, "error": str(exc)}
            self._results.append(err_res)
            self._write_log("goal_error", err_res)
            return err_res
        finally:
            self._queue.task_done()

    async def run_until_idle_or_stopped(self) -> list[dict[str, Any]]:
        processed_this_batch: list[dict[str, Any]] = []
        while not self.stop_event.is_set() and not self._queue.empty():
            res = await self.process_next()
            if res is not None:
                processed_this_batch.append(res)
        return processed_this_batch

    async def serve_forever(self) -> None:
        self._write_log("worker_started", {})
        while not self.stop_event.is_set():
            if self._queue.empty():
                try:
                    await asyncio.wait_for(self.stop_event.wait(), timeout=self.poll_interval_sec)
                except TimeoutError:
                    continue
            else:
                await self.process_next()
        self._write_log("worker_stopped", {})
