from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agent.bridges.artifact_bridge import ArtifactBridge
from agent.bridges.code_bridge import CodeBridge
from agent.bridges.data_bridge import DataBridge
from agent.bridges.excel_bridge import ExcelBridge
from agent.guard import WorkspaceGuard


@dataclass
class VerificationCheck:
    name: str
    passed: bool
    detail: str
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class VerificationReport:
    passed: bool
    checks: list[VerificationCheck] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "total_checks": len(self.checks),
            "passed_checks": sum(1 for c in self.checks if c.passed),
            "failed_checks": [c.name for c in self.checks if not c.passed],
            "checks": [
                {
                    "name": c.name,
                    "passed": c.passed,
                    "detail": c.detail,
                    "meta": c.meta,
                }
                for c in self.checks
            ],
        }


class TaskVerifier:
    """Runs concrete, evidence-backed checks on files, tests, workbooks, and datasets."""

    def __init__(
        self,
        guard: WorkspaceGuard,
        code_bridge: CodeBridge | None = None,
        excel_bridge: ExcelBridge | None = None,
        data_bridge: DataBridge | None = None,
        artifact_bridge: ArtifactBridge | None = None,
    ) -> None:
        self.guard = guard
        self.code_bridge = code_bridge or CodeBridge(guard=guard)
        self.excel_bridge = excel_bridge or ExcelBridge(guard=guard)
        self.data_bridge = data_bridge or DataBridge(guard=guard)
        self.artifact_bridge = artifact_bridge or ArtifactBridge(guard=guard)

    def verify_artifacts_exist(self, relative_paths: list[str], min_bytes: int = 10) -> VerificationCheck:
        missing: list[str] = []
        sizes: dict[str, int] = {}
        for rel in relative_paths:
            res = self.artifact_bridge.verify_artifact_file(rel, min_bytes=min_bytes)
            if not res.get("ok"):
                missing.append(f"{rel} ({res.get('error', 'missing')})")
            else:
                sizes[rel] = int(res.get("size_bytes", 0))

        passed = len(missing) == 0
        detail = (
            f"All {len(relative_paths)} required artifact(s) present and >= {min_bytes} bytes."
            if passed
            else f"Missing or invalid artifacts: {', '.join(missing)}"
        )
        return VerificationCheck(
            name="artifacts_exist",
            passed=passed,
            detail=detail,
            meta={"sizes": sizes, "failures": missing},
        )

    def verify_pytest(self, target: str = ".") -> VerificationCheck:
        res = self.code_bridge.run_tests(test_target=target)
        passed = bool(res.get("ok")) and int(res.get("returncode", 1)) == 0
        detail = (
            f"pytest passed for '{target}'."
            if passed
            else f"pytest failed (rc={res.get('returncode')}): {res.get('stderr') or res.get('stdout') or res.get('error')}"
        )
        return VerificationCheck(
            name="pytest_suite",
            passed=passed,
            detail=detail,
            meta={"target": target, "return_code": res.get("returncode")},
        )

    def verify_excel_clean(self, workbook_rel_path: str) -> VerificationCheck:
        audit = self.excel_bridge.audit_workbook(workbook_rel_path)
        if not audit.get("ok"):
            return VerificationCheck(
                name="excel_formula_audit",
                passed=False,
                detail=f"Workbook audit failed: {audit.get('error')}",
                meta={"workbook": workbook_rel_path},
            )

        err_count = int(audit.get("issue_count", 0))
        passed = err_count == 0
        detail = (
            f"No Excel formula errors in '{workbook_rel_path}' across {len(audit.get('sheets', []))} sheet(s)."
            if passed
            else f"Found {err_count} formula error(s) in '{workbook_rel_path}'."
        )
        return VerificationCheck(
            name="excel_formula_audit",
            passed=passed,
            detail=detail,
            meta={"workbook": workbook_rel_path, "error_count": err_count, "errors": audit.get("issues", [])[:20]},
        )

    def verify_cleaned_dataset(
        self,
        dataset_rel_path: str,
        required_cols: list[str] | None = None,
        min_rows: int = 1,
        dedup_keys: list[str] | None = None,
    ) -> VerificationCheck:
        try:
            df = self.data_bridge._load_table(dataset_rel_path)
        except Exception as exc:
            return VerificationCheck(
                name="dataset_integrity",
                passed=False,
                detail=f"Failed to load dataset '{dataset_rel_path}': {exc}",
                meta={"dataset": dataset_rel_path},
            )

        issues: list[str] = []
        if len(df) < min_rows:
            issues.append(f"row count {len(df)} < minimum {min_rows}")

        for col in required_cols or []:
            if col not in df.columns:
                issues.append(f"missing column '{col}'")
            else:
                nulls = int(df[col].isna().sum())
                if nulls > 0:
                    issues.append(f"{nulls} null(s) in '{col}'")

        if dedup_keys:
            missing_k = [k for k in dedup_keys if k not in df.columns]
            if missing_k:
                issues.append(f"missing dedup keys {missing_k}")
            else:
                dups = int(df.duplicated(subset=dedup_keys).sum())
                if dups > 0:
                    issues.append(f"{dups} duplicate row(s) on {dedup_keys}")

        passed = len(issues) == 0
        detail = (
            f"Dataset '{dataset_rel_path}' verified ({len(df)} rows)."
            if passed
            else f"Dataset '{dataset_rel_path}' failed checks: {'; '.join(issues)}"
        )
        return VerificationCheck(
            name="dataset_integrity",
            passed=passed,
            detail=detail,
            meta={"dataset": dataset_rel_path, "rows": len(df), "issues": issues},
        )

    def run_spec(self, spec: dict[str, Any]) -> VerificationReport:
        """Execute a declarative verification spec dictionary and return a VerificationReport."""
        checks: list[VerificationCheck] = []

        req_files = list(spec.get("required_files") or []) + list(spec.get("artifacts") or [])
        if req_files:
            checks.append(
                self.verify_artifacts_exist(
                    relative_paths=req_files,
                    min_bytes=int(spec.get("min_bytes", 10)),
                )
            )

        if spec.get("pytest_target"):
            checks.append(self.verify_pytest(target=str(spec["pytest_target"])))

        for wb in spec.get("excel_workbooks", []):
            checks.append(self.verify_excel_clean(str(wb)))

        for ds in spec.get("datasets", []):
            if isinstance(ds, dict) and ds.get("path"):
                checks.append(
                    self.verify_cleaned_dataset(
                        dataset_rel_path=str(ds["path"]),
                        required_cols=ds.get("required_cols"),
                        min_rows=int(ds.get("min_rows", 1)),
                        dedup_keys=ds.get("dedup_keys"),
                    )
                )

        overall_passed = all(c.passed for c in checks) if checks else True
        return VerificationReport(passed=overall_passed, checks=checks)