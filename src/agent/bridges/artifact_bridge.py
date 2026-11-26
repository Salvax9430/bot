from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent.guard import WorkspaceGuard


class ArtifactBridge:
    """Generates parametric CAD/3D scripts, validates export files, and builds reports."""

    def __init__(self, guard: WorkspaceGuard) -> None:
        self.guard = guard

    def write_openscad_part(
        self,
        relative_path: str,
        part_name: str,
        params: dict[str, float | int | str],
        body_code: str,
    ) -> dict[str, Any]:
        """Write a clean parametric OpenSCAD (.scad) script from explicit parameters."""
        path = self.guard.resolve_path(relative_path)
        if path.suffix.lower() != ".scad":
            path = path.with_suffix(".scad")

        lines: list[str] = [
            f"// Parametric part: {part_name}",
            f"// Generated: {datetime.now(timezone.utc).isoformat()}",
            "$fn = 64;",
            "",
        ]
        for key, val in params.items():
            clean_key = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in str(key))
            if not clean_key or clean_key[0].isdigit():
                clean_key = f"param_{clean_key}"
            if isinstance(val, bool):
                lines.append(f"{clean_key} = {'true' if val else 'false'};")
            elif isinstance(val, (int, float)):
                lines.append(f"{clean_key} = {val};")
            else:
                escaped = str(val).replace('"', '\\"')
                lines.append(f'{clean_key} = "{escaped}";')

        lines.extend(["", body_code.strip(), ""])
        script_text = "\n".join(lines)
        return self.guard.write_text_file(path, script_text)

    def render_openscad_export(
        self,
        scad_relative_path: str,
        export_relative_path: str,
        openscad_cmd: str = "openscad",
    ) -> dict[str, Any]:
        """Render a .scad file to an export format (e.g., .stl, .dxf, .off) and verify output."""
        scad_path = self.guard.resolve_path(scad_relative_path)
        out_path = self.guard.resolve_path(export_relative_path)

        if not scad_path.exists():
            return {"ok": False, "error": f"SCAD file not found: {scad_path}"}

        allowed_exts = {".stl", ".dxf", ".off", ".amf", ".3mf", ".csg", ".png"}
        if out_path.suffix.lower() not in allowed_exts:
            return {
                "ok": False,
                "error": f"Unsupported CAD export extension '{out_path.suffix}'. Allowed: {sorted(allowed_exts)}",
            }

        out_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = f'"{openscad_cmd}" -o "{out_path}" "{scad_path}"'
        res = self.guard.run_command(cmd, timeout_sec=60)
        if not res.get("ok"):
            return res

        return self.verify_artifact_file(export_relative_path)

    def verify_artifact_file(self, relative_path: str, min_bytes: int = 10) -> dict[str, Any]:
        """Check that a generated file (CAD export, report, dataset, etc.) exists and is non-empty."""
        path = self.guard.resolve_path(relative_path)
        if not path.exists():
            return {"ok": False, "path": str(path), "error": "Artifact file does not exist."}
        if not path.is_file():
            return {"ok": False, "path": str(path), "error": "Artifact path is not a file."}

        size_bytes = path.stat().st_size
        if size_bytes < max(1, int(min_bytes)):
            return {
                "ok": False,
                "path": str(path),
                "size_bytes": size_bytes,
                "error": f"Artifact smaller than minimum {min_bytes} bytes.",
            }

        return {
            "ok": True,
            "path": str(path),
            "extension": path.suffix.lower(),
            "size_bytes": size_bytes,
        }

    def build_markdown_report(
        self,
        relative_path: str,
        title: str,
        executive_summary: str,
        sections: list[dict[str, Any]],
        metrics: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Assemble a structured Markdown report with optional key metrics and sections."""
        parts: list[str] = [
            f"# {title.strip()}",
            "",
            f"*Generated (UTC): {datetime.now(timezone.utc).isoformat()}*",
            "",
            "## Executive Summary",
            "",
            executive_summary.strip(),
            "",
        ]

        if metrics:
            parts.extend(["## Key Metrics", "", "| Metric | Value |", "| :--- | :--- |"])
            for k, v in metrics.items():
                clean_v = str(v).replace("|", "\\|").replace("\n", " ")
                parts.append(f"| {str(k).replace('|', '\\|')} | {clean_v} |")
            parts.append("")

        for sec in sections:
            sec_title = str(sec.get("heading", "Details")).strip()
            sec_body = str(sec.get("body", "")).strip()
            parts.extend([f"## {sec_title}", "", sec_body, ""])

        report_text = "\n".join(parts).strip() + "\n"
        return self.guard.write_text_file(relative_path, report_text)

    def write_json_artifact(self, relative_path: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Write a formatted JSON artifact inside WorkspaceGuard."""
        serialized = json.dumps(payload, indent=2, default=str) + "\n"
        return self.guard.write_text_file(relative_path, serialized)

    def as_tool_map(self) -> dict[str, Any]:
        async def _write_scad(args: dict[str, Any]) -> dict[str, Any]:
            return self.write_openscad_part(
                relative_path=str(args.get("path", "cad/part.scad")),
                part_name=str(args.get("part_name", "ParametricPart")),
                params=dict(args.get("params", {})) if isinstance(args.get("params"), dict) else {},
                body_code=str(args.get("body_code", "cube([10, 10, 10]);")),
            )

        async def _render_scad(args: dict[str, Any]) -> dict[str, Any]:
            return self.render_openscad_export(
                scad_relative_path=str(args.get("scad_path", "")),
                export_relative_path=str(args.get("export_path", "")),
                openscad_cmd=str(args.get("openscad_cmd", "openscad")),
            )

        async def _verify_file(args: dict[str, Any]) -> dict[str, Any]:
            return self.verify_artifact_file(
                relative_path=str(args.get("path", "")),
                min_bytes=int(args.get("min_bytes", 10)),
            )

        async def _md_report(args: dict[str, Any]) -> dict[str, Any]:
            return self.build_markdown_report(
                relative_path=str(args.get("path", "reports/summary.md")),
                title=str(args.get("title", "Agent Report")),
                executive_summary=str(args.get("summary", "")),
                sections=list(args.get("sections", [])) if isinstance(args.get("sections"), list) else [],
                metrics=dict(args.get("metrics", {})) if isinstance(args.get("metrics"), dict) else None,
            )

        async def _json_artifact(args: dict[str, Any]) -> dict[str, Any]:
            return self.write_json_artifact(
                relative_path=str(args.get("path", "artifacts/output.json")),
                payload=dict(args.get("payload", {})) if isinstance(args.get("payload"), dict) else {},
            )

        return {
            "cad_write_scad": _write_scad,
            "cad_render_export": _render_scad,
            "artifact_verify": _verify_file,
            "report_write_markdown": _md_report,
            "artifact_write_json": _json_artifact,
        }