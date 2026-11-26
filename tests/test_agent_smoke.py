from pathlib import Path
import pytest

from agent.guard import WorkspaceGuard
from agent.memory import ExperienceStore
from agent.skills import SkillStore
from agent.bridges.code_bridge import CodeBridge
from agent.bridges.excel_bridge import ExcelBridge
from agent.bridges.data_bridge import DataBridge
from agent.bridges.artifact_bridge import ArtifactBridge
from agent.verifier import TaskVerifier


def test_guard_path_containment_and_blocklist(tmp_path: Path) -> None:
    guard = WorkspaceGuard(workspace_root=tmp_path / "ws")

    safe_file = guard.resolve_path("notes/plan.txt")
    assert str(safe_file).startswith(str(guard.workspace_root))

    with pytest.raises(PermissionError):
        guard.resolve_path("../outside_secret.txt")

    reason = guard.is_command_blocked("rm -rf /")
    assert reason is not None and len(reason) > 0
    assert guard.is_command_blocked("echo hello") is None


def test_memory_and_skills_persistence(tmp_path: Path) -> None:
    mem = ExperienceStore(db_path=tmp_path / "agent_memory.db")
    run_id = mem.start_run("Clean messy CSV and generate report")
    mem.log_step(
        run_id=run_id,
        step_num=1,
        thought="Clean raw CSV into clean.csv",
        action={"tool": "data_clean", "target": "data/raw.csv"},
        observation={"ok": True, "rows": 10},
        verified_ok=True,
    )
    mem.finish_run(run_id=run_id, status="completed", final_summary="Cleaned 10 rows")

    steps = mem.get_run_steps(run_id=run_id)
    assert len(steps) == 1
    assert steps[0]["verified_ok"] == 1

    runs = mem.recent_runs(limit=5)
    assert len(runs) == 1
    assert runs[0]["status"] == "completed"

    skills = SkillStore(db_path=tmp_path / "agent_skills.db")
    skills.save_skill(
        name="csv_cleaner",
        domain="data",
        description="Normalize CSV columns and deduplicate rows",
        steps=[
            {"step": 1, "action": "Load CSV"},
            {"step": 2, "action": "Drop duplicates"},
            {"step": 3, "action": "Write output CSV"},
        ],
        tags=["csv", "clean", "etl"],
    )
    matches = skills.search_skills("clean csv")
    assert len(matches) == 1
    assert matches[0].name == "csv_cleaner"


def test_code_excel_data_artifact_and_verifier(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    guard = WorkspaceGuard(workspace_root=ws)
    code = CodeBridge(guard=guard)
    excel = ExcelBridge(guard=guard)
    data = DataBridge(guard=guard)
    artifacts = ArtifactBridge(guard=guard)
    verifier = TaskVerifier(
        guard=guard,
        code_bridge=code,
        excel_bridge=excel,
        data_bridge=data,
        artifact_bridge=artifacts,
    )

    guard.write_text_file("src/sample.py", "def add(a: int, b: int) -> int:\n    return a + b\n")
    guard.write_text_file("tests/test_sample.py", "from src.sample import add\n\ndef test_add():\n    assert add(2, 3) == 5\n")
    found = code.search_text("def add", subdir="src")
    assert found["ok"] is True

    test_check = verifier.verify_pytest(target="tests/test_sample.py")
    assert test_check.passed is True

    excel.create_workbook(
        "reports/summary.xlsx",
        sheets={"Summary": [["Metric", "Total"], ["Rows", 10]]},
    )
    excel.set_cells(
        "reports/summary.xlsx",
        "Summary",
        [{"cell": "B2", "value": 12}],
    )
    excel_check = verifier.verify_excel_clean("reports/summary.xlsx")
    assert excel_check.passed is True

    raw_csv = ws / "data/raw.csv"
    raw_csv.parent.mkdir(parents=True, exist_ok=True)
    raw_csv.write_text("id,name\n1,Alice\n1,Alice\n2,Bob\n", encoding="utf-8")
    clean_csv = ws / "data/clean.csv"
    clean_csv.write_text("id,name\n1,Alice\n2,Bob\n", encoding="utf-8")
    data_check = verifier.verify_cleaned_dataset(
        dataset_rel_path="data/clean.csv",
        required_cols=["id", "name"],
        min_rows=2,
        dedup_keys=["id"],
    )
    assert data_check.passed is True

    art = artifacts.write_json_artifact("artifacts/manifest.json", {"status": "ok", "rows": 2})
    assert art["ok"] is True
    art_check = verifier.verify_artifacts_exist(["artifacts/manifest.json"])
    assert art_check.passed is True
