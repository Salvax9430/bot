from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent.worker import SilentBackgroundWorker


class InboxTaskWatcher:
    """Watches a folder for new JSON/TXT/MD task files and enqueues them into the worker."""

    SUPPORTED_SUFFIXES = {".json", ".txt", ".md"}

    def __init__(
        self,
        worker: SilentBackgroundWorker,
        inbox_dir: Path,
        processed_dir: Path | None = None,
        poll_interval_sec: float = 2.0,
        stop_event: asyncio.Event | None = None,
    ) -> None:
        self.worker = worker
        self.inbox_dir = inbox_dir.resolve()
        self.inbox_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir = (processed_dir or (self.inbox_dir / "processed")).resolve()
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.poll_interval_sec = max(0.5, float(poll_interval_sec))
        self.stop_event = stop_event if stop_event is not None else worker.stop_event

    @staticmethod
    def _parse_task_file(path: Path) -> tuple[str, dict[str, Any]]:
        raw = path.read_text(encoding="utf-8").strip()
        if not raw:
            raise ValueError(f"Task file '{path.name}' is empty.")

        if path.suffix.lower() == ".json":
            data = json.loads(raw)
            if not isinstance(data, dict):
                raise ValueError(f"JSON task file '{path.name}' must contain a top-level object.")
            goal = str(data.get("goal", "")).strip()
            if not goal:
                raise ValueError(f"JSON task file '{path.name}' is missing a non-empty 'goal' field.")
            context = data.get("context")
            ctx_dict = dict(context) if isinstance(context, dict) else {}
            ctx_dict.setdefault("source_task_file", path.name)
            return goal, ctx_dict

        # Plain text or Markdown spec: first non-empty line is the goal; full content is attached as context
        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        goal = lines[0].lstrip("#").strip()
        context = {
            "source_task_file": path.name,
            "task_spec_text": raw,
        }
        return goal, context

    def _archive_file(self, path: Path, status: str, detail: str | None = None) -> Path:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        dest = self.processed_dir / f"{stamp}_{status}_{path.name}"
        path.replace(dest)
        self.worker._write_log(
            "inbox_task_archived",
            {
                "original_name": path.name,
                "archived_to": str(dest),
                "status": status,
                "detail": detail,
            },
        )
        return dest

    def scan_once(self) -> list[dict[str, Any]]:
        """Scan inbox_dir once, enqueue valid task files, and archive processed files."""
        ingested: list[dict[str, Any]] = []
        candidates = sorted(
            p
            for p in self.inbox_dir.iterdir()
            if p.is_file() and p.suffix.lower() in self.SUPPORTED_SUFFIXES
        )

        for file_path in candidates:
            try:
                goal, context = self._parse_task_file(file_path)
                q_size = self.worker.enqueue_goal(goal=goal, context=context)
                archived_path = self._archive_file(file_path, status="enqueued")
                ingested.append(
                    {
                        "file": file_path.name,
                        "archived_to": str(archived_path),
                        "goal": goal,
                        "queue_size": q_size,
                        "ok": True,
                    }
                )
            except Exception as exc:
                archived_path = self._archive_file(file_path, status="invalid", detail=str(exc))
                ingested.append(
                    {
                        "file": file_path.name,
                        "archived_to": str(archived_path),
                        "error": str(exc),
                        "ok": False,
                    }
                )

        return ingested

    async def watch_forever(self) -> None:
        """Continuously poll inbox_dir until stop_event is set."""
        self.worker._write_log("inbox_watcher_started", {"inbox_dir": str(self.inbox_dir)})
        while not self.stop_event.is_set():
            self.scan_once()
            try:
                await asyncio.wait_for(self.stop_event.wait(), timeout=self.poll_interval_sec)
            except TimeoutError:
                continue
        self.worker._write_log("inbox_watcher_stopped", {"inbox_dir": str(self.inbox_dir)})