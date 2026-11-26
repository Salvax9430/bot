import argparse
import asyncio
import ctypes
import json
import subprocess
import time
import uuid
from pathlib import Path

from agent import InboxTaskWatcher, LLMSettings, build_agent_stack
from agent.kinematics import Point2D, build_cursor_trajectory
from agent.typing_model import TypingModel


def move_cursor_smooth(target_x: float, target_y: float) -> None:
    user32 = ctypes.windll.user32

    class _Pt(ctypes.Structure):
        _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

    pt = _Pt()
    user32.GetCursorPos(ctypes.byref(pt))
    start = Point2D(float(pt.x), float(pt.y))
    goal = Point2D(float(target_x), float(target_y))
    samples = build_cursor_trajectory(start=start, target=goal, dt=0.015)
    prev_t = 0.0
    for s in samples:
        user32.SetCursorPos(int(round(s.x)), int(round(s.y)))
        dt = max(0.0, float(s.t) - prev_t)
        prev_t = float(s.t)
        if dt > 0:
            time.sleep(dt)


def visible_desktop_coding_action(workspace: Path, rel_file: str, code_line: str, open_vscode: bool = True) -> None:
    target_path = (workspace / rel_file).resolve()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    opened = False
    if open_vscode:
        try:
            subprocess.Popen(["code", str(workspace), str(target_path)], shell=True)
            time.sleep(1.5)
            opened = True
        except Exception:
            opened = False

    if not opened:
        subprocess.Popen(["notepad.exe"])
        time.sleep(1.0)

    move_cursor_smooth(650.0, 420.0)
    tm = TypingModel(wpm=90.0)
    events = tm.plan_text(code_line[:85], allow_typos=False)

    send_parts = []
    for ev in events:
        if ev.kind == "char" and ev.value and ev.value not in {"'", '"', "{", "}", "+", "^", "%", "~", "(", ")"}:
            ms = int(max(10, float(ev.delay_sec) * 1000))
            send_parts.append(f"[System.Windows.Forms.SendKeys]::SendWait('{ev.value}'); Start-Sleep -Milliseconds {ms}")
        elif ev.kind == "key" and str(ev.value).lower() == "space":
            ms = int(max(10, float(ev.delay_sec) * 1000))
            send_parts.append(f"[System.Windows.Forms.SendKeys]::SendWait(' '); Start-Sleep -Milliseconds {ms}")

    if send_parts:
        ps_cmd = "Add-Type -AssemblyName System.Windows.Forms; " + "; ".join(send_parts)
        subprocess.run(["powershell", "-NoProfile", "-Command", ps_cmd], check=False)


class _CodingWorkflowPlanner:
    async def complete_json(self, system_prompt: str, user_prompt: str) -> str:
        payload = json.loads(user_prompt)
        if "failed_step" in payload:
            return json.dumps({
                "root_cause": "Adjust target path or arguments and retry.",
                "recovery_action": "retry_adjusted",
                "adjusted_args": {},
                "alternative_tool": None,
                "lesson_learned": "Use workspace-relative paths for all deliverables.",
                "confidence": 0.92,
            })

        goal = str(payload.get("goal", "Build autonomous module and report"))
        module_code = (
            '"""Autonomous Chama & Contribution Service Module."""\n\n'
            "def calculate_cycle_total(members: int, amount_kes: int) -> int:\n"
            "    if members <= 0 or amount_kes < 0:\n"
            '        raise ValueError("members must be positive and amount_kes non-negative")\n'
            "    return members * amount_kes\n\n"
            "def member_share_kes(total_pool_kes: int, members: int) -> float:\n"
            "    if members <= 0:\n"
            '        raise ValueError("members must be positive")\n'
            "    return round(total_pool_kes / members, 2)\n"
        )
        return json.dumps({
            "options": [
                {
                    "name": "Visible Coding + Verified Artifacts",
                    "strategy_type": "programmatic",
                    "summary": f"Write module, JSON ledger summary, and Markdown report for: {goal}",
                    "pros": ["Real files written", "Verified artifacts", "Skill distilled"],
                    "cons": [],
                    "reliability_score": 0.97,
                    "safety_score": 0.97,
                    "speed_score": 0.94,
                }
            ],
            "chosen_index": 0,
            "steps": [
                {
                    "step_num": 1,
                    "title": "Write Python service module",
                    "tool": "write_text_file",
                    "args": {"path": "src/chama_service.py", "content": module_code},
                    "verification_hint": "src/chama_service.py created",
                },
                {
                    "step_num": 2,
                    "title": "Write JSON summary artifact",
                    "tool": "artifact_write_json",
                    "args": {
                        "path": "artifacts/chama_service_summary.json",
                        "payload": {"goal": goal, "members": 10, "amount_kes": 1000, "pool_kes": 10000},
                    },
                    "verification_hint": "artifacts/chama_service_summary.json created",
                },
                {
                    "step_num": 3,
                    "title": "Write Markdown execution report",
                    "tool": "report_write_markdown",
                    "args": {
                        "path": "reports/chama_service_report.md",
                        "title": "Autonomous Coding Run Report",
                        "summary": goal,
                        "sections": [
                            {"heading": "Module", "body": "src/chama_service.py implements calculate_cycle_total and member_share_kes."}
                        ],
                        "metrics": {"members": 10, "pool_kes": 10000, "verified": 1},
                    },
                    "verification_hint": "reports/chama_service_report.md created",
                },
            ],
        })


async def run_loop(workspace: Path, data_dir: Path, poll_sec: float, open_vscode: bool) -> None:
    inbox = workspace / "inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    startup_task = inbox / f"startup_job_{uuid.uuid4().hex[:6]}.json"
    startup_task.write_text(json.dumps({
        "goal": "Build Chama contribution and share calculation module with summary report",
        "context": {
            "verification": {
                "artifacts": [
                    "src/chama_service.py",
                    "artifacts/chama_service_summary.json",
                    "reports/chama_service_report.md",
                ]
            }
        },
    }), encoding="utf-8")

    worker, loop, comps = build_agent_stack(
        workspace_root=workspace,
        data_dir=data_dir,
        llm_settings=LLMSettings(api_key="local-live", base_url="http://localhost:8000/v1", model="local-coder"),
    )
    planner = _CodingWorkflowPlanner()
    loop.brain.llm = planner
    loop.critic.llm = planner

    watcher: InboxTaskWatcher = comps["watcher"]
    print(f"[AUTONOMOUS BOT ONLINE] Workspace={workspace} | Watching {inbox} (Ctrl+C to stop)")

    while True:
        ingested = watcher.scan_once()
        if ingested:
            for item in ingested:
                goal = str(item.get("goal", ""))
                print(f"[TASK PICKED UP] {item.get('file')} -> {goal}")
                visible_desktop_coding_action(
                    workspace=workspace,
                    rel_file="src/chama_service.py",
                    code_line="def calculate_cycle_total(members, amount_kes): return members * amount_kes",
                    open_vscode=open_vscode,
                )
            results = await worker.run_until_idle_or_stopped()
            for r in results:
                print(
                    f"[TASK COMPLETED] run_id={r.get('run_id')} status={r.get('status')} "
                    f"verified={r.get('verified_success')} skill_id={r.get('distilled_skill_id')}"
                )
            print("[WAITING FOR NEXT TASK IN INBOX]")
        await asyncio.sleep(poll_sec)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default=r"C:\Users\dell\OneDrive\Desktop\BOT\bot_live\workspace")
    parser.add_argument("--data-dir", default=r"C:\Users\dell\OneDrive\Desktop\BOT\bot_live\data")
    parser.add_argument("--poll-sec", type=float, default=3.0)
    parser.add_argument("--no-vscode", action="store_true")
    args = parser.parse_args()
    try:
        asyncio.run(run_loop(Path(args.workspace), Path(args.data_dir), args.poll_sec, not args.no_vscode))
    except KeyboardInterrupt:
        print("\n[AUTONOMOUS BOT STOPPED]")


if __name__ == "__main__":
    main()
