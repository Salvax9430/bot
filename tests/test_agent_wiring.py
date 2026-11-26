from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import pytest

from agent import InboxTaskWatcher, LLMSettings, build_agent_stack


class _DeterministicPlannerLLM:
    """Stub LLM client returning valid plan/critic JSON for end-to-end wiring tests."""

    async def complete_json(self, system_prompt: str, user_prompt: str) -> str:
        payload = json.loads(user_prompt)
        if "failed_step" in payload:
            return json.dumps(
                {
                    "root_cause": "Retry step with corrected arguments.",
                    "recovery_action": "retry_adjusted",
                    "adjusted_args": {},
                    "alternative_tool": None,
                    "lesson_learned": "Verify file paths before writing.",
                    "confidence": 0.9,
                }
            )
        return json.dumps(
            {
                "options": [
                    {
                        "name": "Direct File & Artifact Pipeline",
                        "strategy_type": "programmatic",
                        "summary": "Write greeting module and output artifact directly.",
                        "pros": ["Deterministic", "Fast"],
                        "cons": ["Requires verification"],
                        "reliability_score": 0.95,
                        "safety_score": 0.95,
                        "speed_score": 0.9,
                    }
                ],
                "chosen_index": 0,
                "steps": [
                    {
                        "step_num": 1,
                        "title": "Write helper module",
                        "tool": "write_text_file",
                        "args": {
                            "path": "src/greeter.py",
                            "content": "def greet(name: str) -> str:\n    return f\"Hello, {name}!\"\n",
                        },
                        "verification_hint": "src/greeter.py exists",
                    },
                    {
                        "step_num": 2,
                        "title": "Write summary artifact",
                        "tool": "artifact_write_json",
                        "args": {
                            "path": "artifacts/result.json",
                            "payload": {"status": "wired_ok"},
                        },
                        "verification_hint": "artifacts/result.json created",
                    },
                ],
            }
        )


@pytest.mark.asyncio
async def test_full_stack_wiring_watcher_worker_verifier_and_distiller(tmp_path: Path) -> None:
    ws = tmp_path / "workspace"
    data_dir = tmp_path / "data"
    settings = LLMSettings(api_key="test-key", base_url="http://localhost:8000/v1", model="stub-model")

    worker, loop, components = build_agent_stack(
        workspace_root=ws,
        data_dir=data_dir,
        llm_settings=settings,
    )
    stub_llm = _DeterministicPlannerLLM()
    loop.brain.llm = stub_llm  # type: ignore[assignment]
    loop.critic.llm = stub_llm  # type: ignore[assignment]

    inbox_dir = ws / "inbox"
    inbox_dir.mkdir(parents=True, exist_ok=True)
    task_file = inbox_dir / "create_greeter.json"
    task_file.write_text(
        json.dumps(
            {
                "goal": "Create greeter module and result artifact",
                "context": {
                    "verification": {
                        "artifacts": ["src/greeter.py", "artifacts/result.json"],
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    watcher: InboxTaskWatcher = components["watcher"]
    ingested = watcher.scan_once()
    assert len(ingested) == 1
    assert ingested[0]["ok"] is True

    results = await worker.run_until_idle_or_stopped()
    assert len(results) == 1
    res = results[0]
    assert res["status"] == "completed"
    assert res["verified_success"] is True
    assert res["verification"]["passed"] is True
    assert (ws / "src/greeter.py").exists()
    assert (ws / "artifacts/result.json").exists()
    assert (data_dir / "worker.jsonl").exists()
    assert len(components["memory"].recent_runs(limit=5)) == 1
