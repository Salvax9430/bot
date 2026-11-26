# Autonomous Human-Paced Computer Agent

A modular Python agent designed to execute multi-step software engineering, spreadsheet auditing, data wrangling, REST API verification, CAD/3D artifact generation, and human-paced GUI tasks inside a guarded workspace.

## Project Layout

```text
src/agent/
├── guard.py               # Workspace path containment, command blocklist & timeouts
├── memory.py              # SQLite run & step audit ledger (agent_memory.db)
├── skills.py              # SQLite reusable skill store (agent_skills.db)
├── learner.py             # Auto-distills verified runs into skills & tracks outcomes
├── worker.py              # Async background queue runner with JSONL lifecycle logs
├── watcher.py             # Watches workspace/inbox for JSON/MD/TXT task specs
├── llm_client.py          # Async OpenAI-compatible JSON client with backoff retries
├── verifier.py            # Objective post-task verification checks & reports
├── kinematics.py          # Fitts's law, minimum-jerk, Bézier + OU drift & scroll impulse
├── typing_model.py        # Log-normal keystrokes, digraphs, fatigue & typo recovery
├── perception.py          # Screen/region capture & SSIM frame diffing
├── brain.py               # Option brainstorming, trade-off scoring & step planning
├── critic.py              # Failure diagnosis, recovery & lesson extraction
├── loop.py                # Main execution, self-healing & stop-signal loop
├── main.py                # Unified stack builder & CLI entrypoint
└── bridges/
    ├── code_bridge.py     # File search/edit, pytest & git helpers
    ├── excel_bridge.py    # Workbook creation, cell updates, error audit & reconciliation
    ├── data_bridge.py     # CSV/JSON/SQLite cleaning, IQR outliers & read-only SQL
    ├── api_bridge.py      # HTTP/REST probes, latency & JSON key assertions
    ├── artifact_bridge.py # Parametric OpenSCAD scripts, exports & Markdown/JSON reports
    └── gui_bridge.py      # Human-paced pointer, click, scroll & typing gated by stop_event

tests/
└── test_agent_smoke.py    # Isolated tmp_path smoke & verification tests