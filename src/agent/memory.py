from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ExperienceStore:
    """SQLite store for goals, brainstorm plans, step traces, and task outcomes."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path.resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    started_at TEXT NOT NULL,
                    ended_at TEXT,
                    goal TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'running',
                    ideas_json TEXT,
                    chosen_idea TEXT,
                    final_summary TEXT
                );

                CREATE TABLE IF NOT EXISTS steps (
                    step_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER NOT NULL,
                    step_num INTEGER NOT NULL,
                    recorded_at TEXT NOT NULL,
                    thought TEXT NOT NULL,
                    action_json TEXT NOT NULL,
                    observation_json TEXT NOT NULL,
                    verified_ok INTEGER NOT NULL DEFAULT 0,
                    FOREIGN KEY (run_id) REFERENCES runs(run_id)
                );
                """
            )

    def start_run(self, goal: str) -> int:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO runs (started_at, goal, status) VALUES (?, ?, 'running')",
                (now, goal),
            )
            return int(cur.lastrowid)

    def save_brainstorm(
        self,
        run_id: int,
        ideas: list[dict[str, Any]],
        chosen_idea: str,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE runs
                SET ideas_json = ?, chosen_idea = ?
                WHERE run_id = ?
                """,
                (json.dumps(ideas, default=str), chosen_idea, run_id),
            )

    def log_step(
        self,
        run_id: int,
        step_num: int,
        thought: str,
        action: dict[str, Any],
        observation: dict[str, Any],
        verified_ok: bool,
    ) -> int:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO steps (
                    run_id, step_num, recorded_at, thought, action_json, observation_json, verified_ok
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    step_num,
                    now,
                    thought,
                    json.dumps(action, default=str),
                    json.dumps(observation, default=str),
                    1 if verified_ok else 0,
                ),
            )
            return int(cur.lastrowid)

    def finish_run(self, run_id: int, status: str, final_summary: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE runs
                SET ended_at = ?, status = ?, final_summary = ?
                WHERE run_id = ?
                """,
                (now, status, final_summary, run_id),
            )

    def recent_runs(self, limit: int = 10) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM runs ORDER BY run_id DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_run_steps(self, run_id: int) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM steps WHERE run_id = ? ORDER BY step_num ASC, step_id ASC",
                (run_id,),
            ).fetchall()
            return [dict(r) for r in rows]