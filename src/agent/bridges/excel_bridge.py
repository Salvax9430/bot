from __future__ import annotations

from typing import Any

import openpyxl
import pandas as pd

from agent.guard import WorkspaceGuard

_FORMULA_ERRORS = frozenset(
    {"#REF!", "#DIV/0!", "#VALUE!", "#NAME?", "#N/A", "#NUM!", "#NULL!"}
)


class ExcelBridge:
    """Create, edit, audit, and reconcile Excel workbooks inside WorkspaceGuard."""

    def __init__(self, guard: WorkspaceGuard) -> None:
        self.guard = guard

    def create_workbook(
        self,
        relative_path: str,
        sheets: dict[str, list[list[Any]]],
    ) -> dict[str, Any]:
        """Create an .xlsx workbook from a mapping of {sheet_name: rows}."""
        path = self.guard.resolve_path(relative_path)
        if path.suffix.lower() != ".xlsx":
            return {"ok": False, "error": "Target file must end with .xlsx"}

        path.parent.mkdir(parents=True, exist_ok=True)
        wb = openpyxl.Workbook()
        # Remove default sheet once we create named sheets
        default_ws = wb.active
        created_sheets: list[str] = []

        for sheet_name, rows in sheets.items():
            ws = wb.create_sheet(title=str(sheet_name)[:31])
            created_sheets.append(ws.title)
            for r_idx, row in enumerate(rows, start=1):
                for c_idx, val in enumerate(row, start=1):
                    ws.cell(row=r_idx, column=c_idx, value=val)

        if default_ws is not None and created_sheets:
            wb.remove(default_ws)

        wb.save(path)
        return {
            "ok": True,
            "path": str(path),
            "sheets": created_sheets,
        }

    def set_cells(
        self,
        relative_path: str,
        sheet_name: str,
        updates: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Update specific cells (e.g. {'cell': 'D2', 'value': '=B2*C2'}) in a workbook."""
        path = self.guard.resolve_path(relative_path)
        if not path.exists():
            return {"ok": False, "error": f"Workbook not found: {path}"}

        wb = openpyxl.load_workbook(path)
        if sheet_name not in wb.sheetnames:
            return {
                "ok": False,
                "error": f"Sheet '{sheet_name}' not found. Available: {wb.sheetnames}",
            }

        ws = wb[sheet_name]
        applied = 0
        for item in updates:
            coord = str(item.get("cell", "")).strip()
            if not coord:
                continue
            ws[coord] = item.get("value")
            applied += 1

        wb.save(path)
        return {
            "ok": True,
            "path": str(path),
            "sheet": sheet_name,
            "cells_updated": applied,
        }

    def audit_workbook(self, relative_path: str) -> dict[str, Any]:
        """Inspect all sheets for formula cells, malformed formulas, and error tokens."""
        path = self.guard.resolve_path(relative_path)
        if not path.exists():
            return {"ok": False, "error": f"Workbook not found: {path}"}

        wb_formulas = openpyxl.load_workbook(path, data_only=False)
        wb_values = openpyxl.load_workbook(path, data_only=True)

        formula_count = 0
        issues: list[dict[str, Any]] = []

        for sname in wb_formulas.sheetnames:
            ws_f = wb_formulas[sname]
            ws_v = wb_values[sname]
            for r in range(1, ws_f.max_row + 1):
                for c in range(1, ws_f.max_column + 1):
                    raw_val = ws_f.cell(row=r, column=c).value
                    cached_val = ws_v.cell(row=r, column=c).value
                    coord = ws_f.cell(row=r, column=c).coordinate

                    if isinstance(raw_val, str) and raw_val.startswith("="):
                        formula_count += 1
                        if raw_val.strip() == "=":
                            issues.append(
                                {
                                    "sheet": sname,
                                    "cell": coord,
                                    "issue": "Empty formula '='",
                                }
                            )
                        if cached_val is None:
                            issues.append(
                                {
                                    "sheet": sname,
                                    "cell": coord,
                                    "formula": raw_val,
                                    "issue": "Uncached formula value (open/save in Excel or LibreOffice to evaluate)",
                                }
                            )

                    check_tokens = [raw_val, cached_val]
                    for tok in check_tokens:
                        if isinstance(tok, str) and any(err in tok for err in _FORMULA_ERRORS):
                            issues.append(
                                {
                                    "sheet": sname,
                                    "cell": coord,
                                    "value": tok,
                                    "issue": "Spreadsheet formula error token detected",
                                }
                            )
                            break

        return {
            "ok": True,
            "path": str(path),
            "sheets": wb_formulas.sheetnames,
            "formula_count": formula_count,
            "issue_count": len(issues),
            "issues": issues[:100],
        }

    def reconcile_sheets(
        self,
        relative_path: str,
        left_sheet: str,
        right_sheet: str,
        key_col: str,
        compare_cols: list[str],
        tolerance: float = 1e-6,
    ) -> dict[str, Any]:
        """Compare two sheets in an .xlsx file on a key column and report mismatches."""
        path = self.guard.resolve_path(relative_path)
        if not path.exists():
            return {"ok": False, "error": f"Workbook not found: {path}"}

        try:
            df_left = pd.read_excel(path, sheet_name=left_sheet)
            df_right = pd.read_excel(path, sheet_name=right_sheet)
        except Exception as exc:
            return {"ok": False, "error": f"Failed reading sheets: {exc}"}

        for col in [key_col, *compare_cols]:
            if col not in df_left.columns or col not in df_right.columns:
                return {
                    "ok": False,
                    "error": f"Column '{col}' missing in one of the sheets.",
                }

        left_map = {str(row[key_col]).strip(): row for _, row in df_left.iterrows()}
        right_map = {str(row[key_col]).strip(): row for _, row in df_right.iterrows()}

        only_in_left = sorted(k for k in left_map if k not in right_map)
        only_in_right = sorted(k for k in right_map if k not in left_map)
        value_mismatches: list[dict[str, Any]] = []

        for k in sorted(set(left_map) & set(right_map)):
            r_l = left_map[k]
            r_r = right_map[k]
            for col in compare_cols:
                v_l = r_l[col]
                v_r = r_r[col]
                if pd.isna(v_l) and pd.isna(v_r):
                    continue
                if isinstance(v_l, (int, float)) and isinstance(v_r, (int, float)):
                    if abs(float(v_l) - float(v_r)) > tolerance:
                        value_mismatches.append(
                            {"key": k, "column": col, "left": v_l, "right": v_r}
                        )
                elif str(v_l).strip() != str(v_r).strip():
                    value_mismatches.append(
                        {"key": k, "column": col, "left": str(v_l), "right": str(v_r)}
                    )

        return {
            "ok": True,
            "key_col": key_col,
            "only_in_left": only_in_left,
            "only_in_right": only_in_right,
            "mismatch_count": len(value_mismatches),
            "mismatches": value_mismatches[:100],
        }

    def as_tool_map(self) -> dict[str, Any]:
        """Return callable tool functions ready to wire into AgentLoop.extra_tools."""

        async def _create_workbook(args: dict[str, Any]) -> dict[str, Any]:
            return self.create_workbook(
                relative_path=str(args.get("path", "")),
                sheets=dict(args.get("sheets", {})),
            )

        async def _set_cells(args: dict[str, Any]) -> dict[str, Any]:
            return self.set_cells(
                relative_path=str(args.get("path", "")),
                sheet_name=str(args.get("sheet", "")),
                updates=list(args.get("updates", [])),
            )

        async def _audit_workbook(args: dict[str, Any]) -> dict[str, Any]:
            return self.audit_workbook(relative_path=str(args.get("path", "")))

        async def _reconcile_sheets(args: dict[str, Any]) -> dict[str, Any]:
            return self.reconcile_sheets(
                relative_path=str(args.get("path", "")),
                left_sheet=str(args.get("left_sheet", "")),
                right_sheet=str(args.get("right_sheet", "")),
                key_col=str(args.get("key_col", "")),
                compare_cols=list(args.get("compare_cols", [])),
                tolerance=float(args.get("tolerance", 1e-6)),
            )

        return {
            "excel_create_workbook": _create_workbook,
            "excel_set_cells": _set_cells,
            "excel_audit_workbook": _audit_workbook,
            "excel_reconcile_sheets": _reconcile_sheets,
        }