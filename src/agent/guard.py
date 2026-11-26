from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class WorkspaceGuard:
    """Enforces workspace path boundaries, command timeouts, and safety checks."""

    workspace_root: Path
    command_timeout_sec: int = 30
    max_output_chars: int = 5000
    blocked_patterns: list[str] = field(
        default_factory=lambda: [
            r"\brm\s+-rf\s+/",
            r"\bdel\s+/s\s+/q\s+c:\\",
            r"\bformat\s+[a-z]:",
            r"\bshutdown\b",
            r"\brestart-computer\b",
        ]
    )

    def __post_init__(self) -> None:
        self.workspace_root = self.workspace_root.resolve()
        self.workspace_root.mkdir(parents=True, exist_ok=True)

    def resolve_path(self, target_path: str | Path) -> Path:
        """Resolve a path and ensure it stays inside workspace_root."""
        candidate = Path(target_path)
        resolved = (
            candidate.resolve()
            if candidate.is_absolute()
            else (self.workspace_root / candidate).resolve()
        )
        if resolved != self.workspace_root and self.workspace_root not in resolved.parents:
            raise PermissionError(
                f"Access denied outside workspace ({self.workspace_root}): {resolved}"
            )
        return resolved

    def read_text_file(self, relative_path: str | Path) -> dict[str, Any]:
        """Safely read a UTF-8 text file inside the workspace."""
        path = self.resolve_path(relative_path)
        if not path.exists():
            return {"ok": False, "error": f"File does not exist: {path}"}
        if not path.is_file():
            return {"ok": False, "error": f"Not a regular file: {path}"}

        content = path.read_text(encoding="utf-8", errors="replace")
        truncated = len(content) > self.max_output_chars
        return {
            "ok": True,
            "path": str(path),
            "truncated": truncated,
            "content": content[-self.max_output_chars :] if truncated else content,
        }

    def write_text_file(self, relative_path: str | Path, content: str) -> dict[str, Any]:
        """Safely write a UTF-8 text file inside the workspace."""
        path = self.resolve_path(relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return {
            "ok": True,
            "path": str(path),
            "chars_written": len(content),
        }

    def is_command_blocked(self, command: str) -> str | None:
        """Return matching blocked pattern if command looks destructive, else None."""
        lowered = command.strip().lower()
        for pattern in self.blocked_patterns:
            if re.search(pattern, lowered):
                return pattern
        return None

    def run_command(self, command: str, timeout_sec: int | None = None) -> dict[str, Any]:
        """Run a shell command inside workspace_root with timeout and output capture."""
        if not command.strip():
            return {"ok": False, "error": "Empty command."}

        matched_rule = self.is_command_blocked(command)
        if matched_rule:
            return {
                "ok": False,
                "blocked": True,
                "error": f"Command blocked by safety rule: {matched_rule}",
            }

        effective_timeout = timeout_sec if timeout_sec is not None else self.command_timeout_sec
        try:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=str(self.workspace_root),
                capture_output=True,
                text=True,
                timeout=effective_timeout,
            )
            return {
                "ok": proc.returncode == 0,
                "returncode": proc.returncode,
                "stdout": proc.stdout[-self.max_output_chars :],
                "stderr": proc.stderr[-self.max_output_chars :],
            }
        except subprocess.TimeoutExpired as exc:
            return {
                "ok": False,
                "timed_out": True,
                "error": f"Command timed out after {effective_timeout}s",
                "stdout": (exc.stdout or "")[-self.max_output_chars :] if isinstance(exc.stdout, str) else "",
                "stderr": (exc.stderr or "")[-self.max_output_chars :] if isinstance(exc.stderr, str) else "",
            }