from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import duckdb
import pandas as pd


class FinancialTableStore:
    """DuckDB wrapper for storing and querying structured financial tables."""

    def __init__(self, db_path: str | Path = "data/financial_tables.duckdb") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = duckdb.connect(str(self.db_path))

    def register_dataframe(self, table_name: str, df: pd.DataFrame) -> None:
        """Persists a pandas DataFrame as a table inside DuckDB."""
        if df.empty:
            return
        self.conn.register("_temp_df", df)
        self.conn.execute(f'CREATE OR REPLACE TABLE "{table_name}" AS SELECT * FROM _temp_df')
        self.conn.unregister("_temp_df")

    def get_table_schema(self, table_name: str) -> str:
        """
        Returns a compact CREATE TABLE / column-type summary + sample rows
        to ground LLM Text-to-SQL prompts.
        """
        cols = self.conn.execute(f'DESCRIBE "{table_name}"').fetchall()
        col_defs = ", ".join(f"{col[0]} ({col[1]})" for col in cols)
        sample_df = self.conn.execute(f'SELECT * FROM "{table_name}" LIMIT 3').df()
        return (
            f"Table: {table_name}\n"
            f"Columns: {col_defs}\n"
            f"Sample Rows:\n{sample_df.to_string(index=False)}"
        )

    def execute_sql(self, sql_query: str) -> Dict[str, Any]:
        """
        Executes a read-only SELECT query and returns structured results or error details
        for the self-correction loop.
        """
        normalized = sql_query.strip().lstrip("(").upper()
        if not (normalized.startswith("SELECT") or normalized.startswith("WITH")):
            return {
                "success": False,
                "error": "Security/Safety violation: Only read-only SELECT or WITH queries are permitted.",
                "rows": [],
            }

        try:
            df = self.conn.execute(sql_query).df()
            return {
                "success": True,
                "error": None,
                "columns": list(df.columns),
                "rows": df.to_dict(orient="records"),
                "row_count": len(df),
            }
        except Exception as exc:
            return {
                "success": False,
                "error": str(exc),
                "rows": [],
            }

    def list_tables(self) -> List[str]:
        rows = self.conn.execute("SHOW TABLES").fetchall()
        return [r[0] for r in rows]

    def close(self) -> None:
        self.conn.close()