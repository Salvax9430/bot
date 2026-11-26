from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any

from agent.brain import BrainEngine
from agent.bridges.api_bridge import ApiBridge
from agent.bridges.artifact_bridge import ArtifactBridge
from agent.bridges.code_bridge import CodeBridge
from agent.bridges.data_bridge import DataBridge
from agent.bridges.excel_bridge import ExcelBridge
from agent.bridges.gui_bridge import GuiInputBridge
from agent.critic import CriticEngine
from agent.guard import WorkspaceGuard
from agent.learner import SkillDistiller
from agent.llm_client import LLMSettings, OpenAICompatibleAsyncClient
from agent.loop import AgentLoop, ToolHandler
from agent.memory import ExperienceStore
from agent.perception import PerceptionEngine
from agent.skills import SkillStore
from agent.typing_model import TypingModel
from agent.verifier import TaskVerifier
from agent.watcher import InboxTaskWatcher
from agent.worker import SilentBackgroundWorker


def build_agent_stack(
    workspace_root: Path,
    data_dir: Path,
    llm_settings: LLMSettings | None = None,
    stop_event: asyncio.Event | None = None,
    allowed_api_hosts: set[str] | None = None,
) -> tuple[SilentBackgroundWorker, AgentLoop, dict[str, Any]]:
    """Assemble guard, memory, skills, distiller, verifier, bridges, brain, critic, loop, and worker."""
    shared_stop = stop_event if stop_event is not None else asyncio.Event()

    guard = WorkspaceGuard(workspace_root=workspace_root)
    memory_db = data_dir / "agent_memory.db"
    skills_db = data_dir / "agent_skills.db"

    memory = ExperienceStore(db_path=memory_db)
    skills = SkillStore(db_path=skills_db)
    distiller = SkillDistiller(memory_db_path=memory_db, skill_store=skills)

    perception = PerceptionEngine(guard=guard)
    typing_model = TypingModel(wpm=68.0, typo_prob=0.025)

    gui_bridge = GuiInputBridge(
        typing_model=typing_model,
        perception=perception,
        stop_event=shared_stop,
        fail_safe=True,
    )
    code_bridge = CodeBridge(guard=guard)
    excel_bridge = ExcelBridge(guard=guard)
    data_bridge = DataBridge(guard=guard)
    api_bridge = ApiBridge(allowed_hosts=allowed_api_hosts)
    artifact_bridge = ArtifactBridge(guard=guard)
    verifier = TaskVerifier(
        guard=guard,
        code_bridge=code_bridge,
        excel_bridge=excel_bridge,
        data_bridge=data_bridge,
        artifact_bridge=artifact_bridge,
    )

    tool_map: dict[str, ToolHandler] = {}
    for bridge in (
        code_bridge,
        excel_bridge,
        data_bridge,
        api_bridge,
        artifact_bridge,
        gui_bridge,
    ):
        tool_map.update(bridge.as_tool_map())

    async def _capture_diff(args: dict[str, Any]) -> dict[str, Any]:
        img = perception.capture()
        out_name = str(args.get("save_as", "captures/latest_screen.png"))
        return perception.save_snapshot(img, out_name)

    async def _verify_spec(args: dict[str, Any]) -> dict[str, Any]:
        spec = args.get("spec") if isinstance(args.get("spec"), dict) else args
        report = verifier.run_spec(spec)
        return report.to_dict()

    tool_map["capture_screenshot"] = _capture_diff
    tool_map["verify_task_spec"] = _verify_spec

    settings = llm_settings if llm_settings is not None else LLMSettings.from_env()
    llm_client = OpenAICompatibleAsyncClient(settings=settings)

    brain = BrainEngine(llm=llm_client)
    critic = CriticEngine(llm=llm_client, max_retries_per_step=2)

    agent_loop = AgentLoop(
        brain=brain,
        critic=critic,
        guard=guard,
        memory=memory,
        extra_tools=tool_map,
        max_replans=2,
    )

    worker = SilentBackgroundWorker(
        agent_loop=agent_loop,
        skills=skills,
        log_path=data_dir / "worker.jsonl",
        poll_interval_sec=1.5,
        stop_event=shared_stop,
    )

    # Wrap process_next so post-run verification and skill distillation happen automatically
    original_process_next = worker.process_next

    async def _process_next_with_learning() -> dict[str, Any] | None:
        try:
            item = worker._queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

        worker._write_log("goal_started", {"goal": item.goal})
        enriched_ctx = worker._build_enriched_context(item)
        injected_skills = list(enriched_ctx.get("relevant_skills", []))

        try:
            result = await agent_loop.run_goal(
                goal=item.goal,
                context=enriched_ctx,
                stop_event=shared_stop,
            )
            run_status = str(result.get("status", "")).lower()
            base_ok = run_status in {"completed", "success", "ok"}

            verification_dict: dict[str, Any] | None = None
            verification_passed = True
            ver_spec = item.context.get("verification")
            if isinstance(ver_spec, dict) and ver_spec:
                ver_report = verifier.run_spec(ver_spec)
                verification_dict = ver_report.to_dict()
                verification_passed = ver_report.passed
                result["verification"] = verification_dict

            overall_ok = base_ok and verification_passed
            result["verified_success"] = overall_ok

            distiller.record_injected_skill_outcomes(
                injected_skills=injected_skills,
                run_succeeded=overall_ok,
            )

            run_id = str(result.get("run_id", ""))
            if overall_ok and run_id:
                learned_id = distiller.distill_run(
                    run_id=run_id,
                    goal=item.goal,
                    run_status=run_status,
                    verifier_passed=verification_passed,
                )
                if learned_id is not None:
                    result["distilled_skill_id"] = learned_id

            worker._results.append(result)
            worker._write_log("goal_finished", {"goal": item.goal, "result": result})
            return result
        except Exception as exc:
            distiller.record_injected_skill_outcomes(
                injected_skills=injected_skills,
                run_succeeded=False,
            )
            err_res = {"status": "worker_error", "goal": item.goal, "error": str(exc)}
            worker._results.append(err_res)
            worker._write_log("goal_error", err_res)
            return err_res
        finally:
            worker._queue.task_done()

    worker.process_next = _process_next_with_learning  # type: ignore[method-assign]
    # Keep reference to original in case direct calls are needed
    worker._raw_process_next = original_process_next  # type: ignore[attr-defined]

    watcher = InboxTaskWatcher(
        worker=worker,
        inbox_dir=guard.workspace_root / "inbox",
        poll_interval_sec=2.0,
        stop_event=shared_stop,
    )

    components: dict[str, Any] = {
        "guard": guard,
        "memory": memory,
        "skills": skills,
        "distiller": distiller,
        "verifier": verifier,
        "watcher": watcher,
        "perception": perception,
        "gui_bridge": gui_bridge,
        "code_bridge": code_bridge,
        "excel_bridge": excel_bridge,
        "data_bridge": data_bridge,
        "api_bridge": api_bridge,
        "artifact_bridge": artifact_bridge,
        "brain": brain,
        "critic": critic,
        "stop_event": shared_stop,
    }
    return worker, agent_loop, components


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Autonomous Human-Paced Computer Agent")
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path(os.getenv("AGENT_WORKSPACE", "workspace")),
        help="Sandboxed working directory for files, code, data, and artifacts.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(os.getenv("AGENT_DATA_DIR", "data")),
        help="Directory for SQLite databases and JSONL logs.",
    )
    parser.add_argument(
        "--goal",
        type=str,
        default="",
        help="Optional goal to enqueue and run immediately.",
    )
    parser.add_argument(
        "--forever",
        action="store_true",
        help="Run background worker + inbox task watcher continuously.",
    )
    return parser.parse_args()


async def _async_main(args: argparse.Namespace) -> None:
    worker, _, components = build_agent_stack(
        workspace_root=args.workspace,
        data_dir=args.data_dir,
    )
    watcher: InboxTaskWatcher = components["watcher"]
    watcher.scan_once()

    if args.goal.strip():
        worker.enqueue_goal(args.goal.strip())

    if args.forever:
        await asyncio.gather(
            worker.serve_forever(),
            watcher.watch_forever(),
        )
    else:
        results = await worker.run_until_idle_or_stopped()
        print(json.dumps(results, indent=2, default=str))


def main() -> None:
    args = parse_args()
    asyncio.run(_async_main(args))


if __name__ == "__main__":
    main()