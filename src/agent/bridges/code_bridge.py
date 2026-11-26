from __future__ import annotations

import sys

import os
import re
from pathlib import Path
from typing import Any

from agent.guard import WorkspaceGuard

_SKIP_DIRS = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        "node_modules",
        "dist",
        "build",
    }
)


class CodeBridge:
    """High-leverage code inspection, editing, test execution, and git helpers."""

    def __init__(self, guard: WorkspaceGuard) -> None:
        self.guard = guard

    def list_files(self, subdir: str = ".", max_files: int = 200) -> dict[str, Any]:
        """List files under a workspace subdirectory, skipping build/cache folders."""
        base = self.guard.resolve_path(subdir)
        if not base.exists():
            return {"ok": False, "error": f"Directory does not exist: {base}"}
        if not base.is_dir():
            return {"ok": False, "error": f"Not a directory: {base}"}

        found: list[str] = []
        root = self.guard.workspace_root

        for current_root, dirs, files in os.walk(base):
            dirs[:] = [d for d in sorted(dirs) if d not in _SKIP_DIRS]
            for fname in sorted(files):
                full_path = Path(current_root) / fname
                rel = full_path.relative_to(root).as_posix()
                found.append(rel)
                if len(found) >= max_files:
                    return {
                        "ok": True,
                        "truncated": True,
                        "count": len(found),
                        "files": found,
                    }

        return {"ok": True, "truncated": False, "count": len(found), "files": found}

    def search_text(
        self,
        pattern: str,
        subdir: str = ".",
        is_regex: bool = False,
        max_matches: int = 100,
    ) -> dict[str, Any]:
        """Search text or regex across workspace text files and return file/line matches."""
        base = self.guard.resolve_path(subdir)
        if not base.exists():
            return {"ok": False, "error": f"Directory does not exist: {base}"}

        try:
            compiled = re.compile(pattern if is_regex else re.escape(pattern))
        except re.error as exc:
            return {"ok": False, "error": f"Invalid regex pattern: {exc}"}

        matches: list[dict[str, Any]] = []
        root = self.guard.workspace_root

        for current_root, dirs, files in os.walk(base):
            dirs[:] = [d for d in sorted(dirs) if d not in _SKIP_DIRS]
            for fname in sorted(files):
                fpath = Path(current_root) / fname
                try:
                    raw = fpath.read_text(encoding="utf-8")
                except (UnicodeDecodeError, OSError):
                    continue

                for line_no, line in enumerate(raw.splitlines(), start=1):
                    if compiled.search(line):
                        matches.append(
                            {
                                "file": fpath.relative_to(root).as_posix(),
                                "line": line_no,
                                "text": line[:240],
                            }
                        )
                        if len(matches) >= max_matches:
                            return {
                                "ok": True,
                                "truncated": True,
                                "count": len(matches),
                                "matches": matches,
                            }

        return {"ok": True, "truncated": False, "count": len(matches), "matches": matches}

    def replace_exact(
        self,
        relative_path: str,
        old_text: str,
        new_text: str,
    ) -> dict[str, Any]:
        """Replace a unique exact substring inside a workspace file."""
        if not old_text:
            return {"ok": False, "error": "old_text must not be empty."}

        path = self.guard.resolve_path(relative_path)
        if not path.exists() or not path.is_file():
            return {"ok": False, "error": f"File not found: {path}"}

        original = path.read_text(encoding="utf-8")
        count = original.count(old_text)
        if count == 0:
            return {"ok": False, "error": f"Target snippet not found in {relative_path}."}
        if count > 1:
            return {
                "ok": False,
                "error": f"Target snippet appeared {count} times in {relative_path}; provide more context to make it unique.",
            }

        updated = original.replace(old_text, new_text, 1)
        path.write_text(updated, encoding="utf-8")
        return {
            "ok": True,
            "path": str(path),
            "replacements": 1,
        }

    def run_tests(self, test_target: str = ".", extra_args: str = "") -> dict[str, Any]:
        """Run pytest on a target path inside the workspace."""
        target = str(self.guard.resolve_path(test_target))
        cmd = f'"{sys.executable}" -m pytest "{target}" {extra_args}'.strip()
        return self.guard.run_command(cmd, timeout_sec=120)

    def git_status(self) -> dict[str, Any]:
        return self.guard.run_command("git status --short")

    def git_diff(self, path_filter: str = "") -> dict[str, Any]:
        safe_filter = f' -- "{path_filter}"' if path_filter.strip() else ""
        return self.guard.run_command(f"git diff{safe_filter}")

    def as_tool_map(self) -> dict[str, Any]:
        """Return callable tool functions ready to wire into AgentLoop.extra_tools."""
        async def _list_files(args: dict[str, Any]) -> dict[str, Any]:
            return self.list_files(
                subdir=str(args.get("subdir", ".")),
                max_files=int(args.get("max_files", 200)),
            )

        async def _search_text(args: dict[str, Any]) -> dict[str, Any]:
            return self.search_text(
                pattern=str(args.get("pattern", "")),
                subdir=str(args.get("subdir", ".")),
                is_regex=bool(args.get("is_regex", False)),
                max_matches=int(args.get("max_matches", 100)),
            )

        async def _replace_exact(args: dict[str, Any]) -> dict[str, Any]:
            return self.replace_exact(
                relative_path=str(args.get("path", "")),
                old_text=str(args.get("old_text", "")),
                new_text=str(args.get("new_text", "")),
            )

        async def _run_tests(args: dict[str, Any]) -> dict[str, Any]:
            return self.run_tests(
                test_target=str(args.get("test_target", ".")),
                extra_args=str(args.get("extra_args", "")),
            )

        async def _git_status(args: dict[str, Any]) -> dict[str, Any]:
            return self.git_status()

        async def _git_diff(args: dict[str, Any]) -> dict[str, Any]:
            return self.git_diff(path_filter=str(args.get("path_filter", "")))

        return {
            "list_files": _list_files,
            "search_text": _search_text,
            "replace_exact": _replace_exact,
            "run_tests": _run_tests,
            "git_status": _git_status,
            "git_diff": _git_diff,
        }