from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class SkillRecord:
    skill_id: int
    name: str
    domain: str
    description: str
    steps: list[dict[str, Any]]
    tags: list[str]
    success_count: int
    failure_count: int
    updated_at: str


class SkillStore:
    """Stores and retrieves verified reusable multi-step skills in SQLite."""

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
                CREATE TABLE IF NOT EXISTS skills (
                    skill_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    domain TEXT NOT NULL DEFAULT 'general',
                    description TEXT NOT NULL,
                    steps_json TEXT NOT NULL,
                    tags_json TEXT NOT NULL,
                    success_count INTEGER NOT NULL DEFAULT 1,
                    failure_count INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL
                );
                """
            )

    def save_skill(
        self,
        name: str,
        domain: str,
        description: str,
        steps: list[dict[str, Any]],
        tags: list[str] | None = None,
    ) -> int:
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("Skill name must not be empty.")
        now = datetime.now(timezone.utc).isoformat()
        clean_tags = sorted({t.strip().lower() for t in (tags or []) if t.strip()})

        with self._connect() as conn:
            existing = conn.execute(
                "SELECT skill_id, success_count FROM skills WHERE name = ?",
                (clean_name,),
            ).fetchone()

            if existing:
                skill_id = int(existing["skill_id"])
                next_success = int(existing["success_count"]) + 1
                conn.execute(
                    """
                    UPDATE skills
                    SET domain = ?, description = ?, steps_json = ?, tags_json = ?,
                        success_count = ?, updated_at = ?
                    WHERE skill_id = ?
                    """,
                    (
                        domain.strip() or "general",
                        description.strip(),
                        json.dumps(steps, default=str),
                        json.dumps(clean_tags),
                        next_success,
                        now,
                        skill_id,
                    ),
                )
                return skill_id

            cur = conn.execute(
                """
                INSERT INTO skills (
                    name, domain, description, steps_json, tags_json,
                    success_count, failure_count, updated_at
                )
                VALUES (?, ?, ?, ?, ?, 1, 0, ?)
                """,
                (
                    clean_name,
                    domain.strip() or "general",
                    description.strip(),
                    json.dumps(steps, default=str),
                    json.dumps(clean_tags),
                    now,
                ),
            )
            return int(cur.lastrowid)

    def record_outcome(self, name: str, succeeded: bool) -> None:
        now = datetime.now(timezone.utc).isoformat()
        col = "success_count" if succeeded else "failure_count"
        with self._connect() as conn:
            conn.execute(
                f"UPDATE skills SET {col} = {col} + 1, updated_at = ? WHERE name = ?",
                (now, name.strip()),
            )

    def search_skills(self, query: str, domain: str | None = None, limit: int = 5) -> list[SkillRecord]:
        terms = [w.lower() for w in query.split() if len(w) > 2]
        with self._connect() as conn:
            if domain:
                rows = conn.execute(
                    "SELECT * FROM skills WHERE LOWER(domain) = LOWER(?)",
                    (domain.strip(),),
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM skills").fetchall()

        scored: list[tuple[float, SkillRecord]] = []
        for r in rows:
            steps = json.loads(r["steps_json"])
            tags = json.loads(r["tags_json"])
            haystack = f"{r['name']} {r['domain']} {r['description']} {' '.join(tags)}".lower()

            overlap = sum(1 for t in terms if t in haystack)
            succ = int(r["success_count"])
            fail = int(r["failure_count"])
            reliability = (succ + 1.0) / (succ + fail + 2.0)
            score = (overlap * 2.0) + reliability

            if overlap > 0 or not terms:
                scored.append(
                    (
                        score,
                        SkillRecord(
                            skill_id=int(r["skill_id"]),
                            name=str(r["name"]),
                            domain=str(r["domain"]),
                            description=str(r["description"]),
                            steps=steps if isinstance(steps, list) else [],
                            tags=tags if isinstance(tags, list) else [],
                            success_count=succ,
                            failure_count=fail,
                            updated_at=str(r["updated_at"]),
                        ),
                    )
                )

        scored.sort(key=lambda item: item[0], reverse=True)
        return [rec for _, rec in scored[: max(1, int(limit))]]