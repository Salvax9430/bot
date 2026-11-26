from __future__ import annotations

import json
import sqlite3
from typing import Any

import pandas as pd

from agent.guard import WorkspaceGuard


class DataBridge:
    """Cleans tabular/JSON datasets, queries SQLite safely, and detects anomalies."""

    def __init__(self, guard: WorkspaceGuard) -> None:
        self.guard = guard

    def _load_table(self, relative_path: str, sheet_or_table: str | None = None) -> pd.DataFrame:
        path = self.guard.resolve_path(relative_path)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")

        suffix = path.suffix.lower()
        if suffix == ".csv":
            return pd.read_csv(path)
        if suffix == ".json":
            return pd.read_json(path)
        if suffix in {".xlsx", ".xls"}:
            return pd.read_excel(path, sheet_name=sheet_or_table or 0)
        if suffix in {".db", ".sqlite", ".sqlite3"}:
            if not sheet_or_table:
                raise ValueError("Provide 'sheet_or_table' (table name) for SQLite files.")
            with sqlite3.connect(path) as conn:
                return pd.read_sql_query(f'SELECT * FROM "{sheet_or_table}"', conn)

        raise ValueError(f"Unsupported data file extension: {suffix}")

    def profile_and_clean(
        self,
        input_path: str,
        output_path: str,
        dedup_keys: list[str] | None = None,
        trim_strings: bool = True,
        numeric_cols: list[str] | None = None,
        required_cols: list[str] | None = None,
    ) -> dict[str, Any]:
        """Clean messy rows, normalize text/numbers, deduplicate, and save output."""
        try:
            df = self._load_table(input_path)
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

        out_file = self.guard.resolve_path(output_path)
        initial_rows = len(df)
        anomalies: list[dict[str, Any]] = []

        if trim_strings:
            for col in df.select_dtypes(include=["object", "string"]).columns:
                df[col] = df[col].map(lambda v: v.strip() if isinstance(v, str) else v)

        for col in numeric_cols or []:
            if col not in df.columns:
                return {"ok": False, "error": f"Numeric column '{col}' missing from dataset."}
            coerced = pd.to_numeric(df[col], errors="coerce")
            bad_mask = coerced.isna() & df[col].notna()
            for idx in df.index[bad_mask]:
                anomalies.append(
                    {
                        "row_index": int(idx),
                        "column": col,
                        "issue": "non_numeric_value",
                        "raw_value": str(df.loc[idx, col]),
                    }
                )
            df[col] = coerced

        for col in required_cols or []:
            if col not in df.columns:
                return {"ok": False, "error": f"Required column '{col}' missing from dataset."}
            missing_mask = df[col].isna() | (df[col].astype(str).str.strip() == "")
            for idx in df.index[missing_mask]:
                anomalies.append(
                    {
                        "row_index": int(idx),
                        "column": col,
                        "issue": "missing_required_value",
                    }
                )

        dedup_removed = 0
        if dedup_keys:
            missing_keys = [k for k in dedup_keys if k not in df.columns]
            if missing_keys:
                return {"ok": False, "error": f"Deduplication keys missing: {missing_keys}"}
            before_dedup = len(df)
            df = df.drop_duplicates(subset=dedup_keys, keep="first")
            dedup_removed = before_dedup - len(df)

        out_file.parent.mkdir(parents=True, exist_ok=True)
        if out_file.suffix.lower() == ".json":
            df.to_json(out_file, orient="records", indent=2)
        else:
            df.to_csv(out_file, index=False)

        return {
            "ok": True,
            "input_rows": initial_rows,
            "output_rows": len(df),
            "dedup_removed": dedup_removed,
            "anomaly_count": len(anomalies),
            "anomalies": anomalies[:100],
            "output_path": str(out_file),
        }

    def detect_numeric_outliers(
        self,
        input_path: str,
        column: str,
        iqr_multiplier: float = 1.5,
    ) -> dict[str, Any]:
        """Flag numeric outliers using Interquartile Range (IQR) bounds."""
        try:
            df = self._load_table(input_path)
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

        if column not in df.columns:
            return {"ok": False, "error": f"Column '{column}' not found."}

        series = pd.to_numeric(df[column], errors="coerce").dropna()
        if series.empty:
            return {"ok": True, "column": column, "outlier_count": 0, "outliers": []}

        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        lower = q1 - (iqr_multiplier * iqr)
        upper = q3 + (iqr_multiplier * iqr)

        outliers: list[dict[str, Any]] = []
        for idx, val in series.items():
            fval = float(val)
            if fval < lower or fval > upper:
                outliers.append({"row_index": int(idx), "value": fval})

        return {
            "ok": True,
            "column": column,
            "q1": round(q1, 4),
            "q3": round(q3, 4),
            "iqr": round(iqr, 4),
            "lower_bound": round(lower, 4),
            "upper_bound": round(upper, 4),
            "outlier_count": len(outliers),
            "outliers": outliers[:100],
        }

    def run_readonly_sql(self, db_relative_path: str, query: str) -> dict[str, Any]:
        """Execute a read-only SELECT/PRAGMA query against a SQLite database in workspace."""
        cleaned = query.strip()
        upper = cleaned.upper()
        if not (upper.startswith("SELECT ") or upper.startswith("WITH ") or upper.startswith("PRAGMA ")):
            return {
                "ok": False,
                "error": "Only read-only SELECT, WITH, or PRAGMA queries are allowed in run_readonly_sql.",
            }

        db_path = self.guard.resolve_path(db_relative_path)
        if not db_path.exists():
            return {"ok": False, "error": f"SQLite database not found: {db_path}"}

        try:
            with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(cleaned).fetchall()
                records = [dict(r) for r in rows[:200]]
                return {
                    "ok": True,
                    "row_count": len(rows),
                    "truncated": len(rows) > 200,
                    "rows": json.loads(json.dumps(records, default=str)),
                }
        except Exception as exc:
            return {"ok": False, "error": f"SQLite query error: {exc}"}

    def as_tool_map(self) -> dict[str, Any]:
        async def _clean_data(args: dict[str, Any]) -> dict[str, Any]:
            return self.profile_and_clean(
                input_path=str(args.get("input_path", "")),
                output_path=str(args.get("output_path", "data/cleaned.csv")),
                dedup_keys=args.get("dedup_keys"),
                trim_strings=bool(args.get("trim_strings", True)),
                numeric_cols=args.get("numeric_cols"),
                required_cols=args.get("required_cols"),
            )

        async def _detect_outliers(args: dict[str, Any]) -> dict[str, Any]:
            return self.detect_numeric_outliers(
                input_path=str(args.get("input_path", "")),
                column=str(args.get("column", "")),
                iqr_multiplier=float(args.get("iqr_multiplier", 1.5)),
            )

        async def _sql_readonly(args: dict[str, Any]) -> dict[str, Any]:
            return self.run_readonly_sql(
                db_relative_path=str(args.get("db_path", "")),
                query=str(args.get("query", "")),
            )

        return {
            "data_clean": _clean_data,
            "data_outliers": _detect_outliers,
            "data_sql_readonly": _sql_readonly,
        }