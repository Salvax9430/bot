from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any

from agent.skills import SkillStore


class SkillDistiller:
    """Distills successful runs from ExperienceStore into reusable SkillStore entries."""

    def __init__(self, memory_db_path: Path, skill_store: SkillStore) -> None:
        self.memory_db_path = memory_db_path.resolve()
        self.skill_store = skill_store

    @staticmethod
    def _slugify_skill_name(goal: str) -> str:
        cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", goal.strip().lower()).strip("_")
        if not cleaned:
            return "learned_workflow"
        return cleaned[:48]

    @staticmethod
    def _infer_domain(steps: list[dict[str, Any]], goal: str) -> str:
        joined_tools = " ".join(str(s.get("tool", "")) for s in steps).lower()
        lower_goal = goal.lower()
        if "excel" in joined_tools or "xlsx" in lower_goal or "spreadsheet" in lower_goal:
            return "excel"
        if "data_" in joined_tools or "csv" in lower_goal or "sql" in lower_goal:
            return "data"
        if "http" in joined_tools or "api" in lower_goal or "endpoint" in lower_goal:
            return "api"
        if "cad_" in joined_tools or "scad" in lower_goal or "stl" in lower_goal:
            return "cad"
        if "gui_" in joined_tools or "screen" in lower_goal or "click" in lower_goal:
            return "gui"
        if "code_" in joined_tools or "pytest" in lower_goal or "git" in lower_goal:
            return "code"
        return "general"

    def _fetch_run_steps(self, run_id: int | str) -> list[dict[str, Any]]:
        if not self.memory_db_path.exists():
            return []
        try:
            rid: int | str = int(run_id)
        except (TypeError, ValueError):
            rid = run_id

        with sqlite3.connect(self.memory_db_path) as conn:
            conn.row_factory = sqlite3.Row
            tables = {
                row["name"]
                for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            }
            if "steps" in tables:
                rows = conn.execute(
                    """
                    SELECT step_num, thought, action_json, observation_json, verified_ok
                    FROM steps
                    WHERE run_id = ?
                    ORDER BY step_num ASC
                    """,
                    (rid,),
                ).fetchall()
                parsed: list[dict[str, Any]] = []
                for r in rows:
                    item = dict(r)
                    try:
                        action = json.loads(item.get("action_json") or "{}")
                    except json.JSONDecodeError:
                        action = {}
                    parsed.append(
                        {
                            "step_num": item.get("step_num"),
                            "tool": str(action.get("tool", "")),
                            "args": action.get("args", {}),
                            "observation": str(item.get("observation_json", "")),
                            "verified_ok": bool(item.get("verified_ok")),
                        }
                    )
                return parsed

            if "step_logs" in tables:
                rows = conn.execute(
                    """
                    SELECT step_index, action_type, target, status, observation
                    FROM step_logs
                    WHERE run_id = ?
                    ORDER BY step_index ASC
                    """,
                    (str(rid),),
                ).fetchall()
                return [
                    {
                        "step_num": r["step_index"],
                        "tool": str(r["action_type"] or ""),
                        "args": {"target": r["target"]},
                        "observation": str(r["observation"] or ""),
                        "verified_ok": str(r["status"] or "").lower() in {"success", "ok", "completed"},
                    }
                    for r in rows
                ]
            return []

    def distill_run(
        self,
        run_id: int | str,
        goal: str,
        run_status: str,
        verifier_passed: bool = True,
        skill_name: str | None = None,
    ) -> int | None:
        """Save a reusable skill if the run completed and passed verification."""
        if run_status.lower() not in {"completed", "success", "ok"} or not verifier_passed:
            return None

        raw_steps = self._fetch_run_steps(run_id)
        successful_steps = [
            {
                "tool": str(s.get("tool", "")),
                "args": s.get("args", {}),
                "observation_preview": str(s.get("observation", ""))[:240],
            }
            for s in raw_steps
            if bool(s.get("verified_ok"))
        ]
        if not successful_steps:
            return None

        name = (skill_name or self._slugify_skill_name(goal)).strip()
        domain = self._infer_domain(successful_steps, goal)
        description = f"Auto-distilled workflow for goal: {goal.strip()}"
        tags = sorted({domain, "distilled", *(w.lower() for w in re.findall(r"[a-zA-Z]{3,}", goal)[:4])})

        return self.skill_store.save_skill(
            name=name,
            domain=domain,
            description=description,
            steps=successful_steps,
            tags=tags,
        )

    def record_injected_skill_outcomes(
        self,
        injected_skills: list[dict[str, Any]],
        run_succeeded: bool,
    ) -> list[dict[str, Any]]:
        """Update success/failure counters for skills that were injected into a run."""
        updates: list[dict[str, Any]] = []
        for sk in injected_skills:
            if not isinstance(sk, dict):
                continue
            name = str(sk.get("name", "")).strip()
            if not name:
                continue
            self.skill_store.record_outcome(name=name, succeeded=run_succeeded)
            updates.append({"name": name, "succeeded": run_succeeded})
        return updates

    def export_skills_snapshot(self, out_path: Path) -> Path:
        """Export all stored skills to a JSON file for inspection or version control."""
        resolved = out_path.resolve()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        matches = self.skill_store.search_skills(query="", limit=500)
        payload = [
            {
                "name": m.name,
                "domain": m.domain,
                "description": m.description,
                "steps": m.steps,
                "tags": m.tags,
                "success_count": m.success_count,
                "failure_count": m.failure_count,
            }
            for m in matches
        ]
        resolved.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
        return resolved
