from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
from pydantic import BaseModel, Field


class FinQARecord(BaseModel):
    """Standardized representation of a single FinQA benchmark item."""
    doc_id: str
    table_name: str
    pre_text: str
    post_text: str
    full_narrative: str
    raw_table: List[List[str]]
    dataframe_records: List[Dict[str, Any]] = Field(default_factory=list)
    question: str
    program: Optional[str] = None
    gold_answer: Optional[float] = None
    gold_answer_str: Optional[str] = None


def _sanitize_identifier(name: str) -> str:
    """Sanitizes strings into valid SQL table and column identifiers."""
    clean = re.sub(r"[^a-zA-Z0-9_]", "_", name.strip().lower())
    clean = re.sub(r"_+", "_", clean).strip("_")
    if not clean or clean[0].isdigit():
        clean = f"col_{clean}"
    return clean


def _deduplicate_columns(headers: List[str]) -> List[str]:
    """Ensures unique column names when FinQA tables contain repeated or blank headers."""
    seen: Dict[str, int] = {}
    unique_cols: List[str] = []
    for idx, raw_header in enumerate(headers):
        base = _sanitize_identifier(raw_header) if raw_header.strip() else f"col_{idx}"
        if base in seen:
            seen[base] += 1
            unique_cols.append(f"{base}_{seen[base]}")
        else:
            seen[base] = 0
            unique_cols.append(base)
    return unique_cols


def _clean_numeric_cell(val: str) -> Any:
    """
    Converts financial string cells (e.g., '$ 1,234.5', '(45.2)', '12.5%')
    into floats where applicable while preserving descriptive labels.
    """
    if not isinstance(val, str):
        return val
    text = val.strip()
    if not text or text in {"-", "—", "N/A", "n/a"}:
        return None

    # Handle accounting negatives in parentheses: (123.4) -> -123.4
    is_negative = False
    if text.startswith("(") and text.endswith(")"):
        is_negative = True
        text = text[1:-1].strip()

    # Strip currency symbols, commas, and percentage signs for numeric parsing
    cleaned = text.replace("$", "").replace(",", "").replace("%", "").strip()
    try:
        num = float(cleaned)
        return -num if is_negative else num
    except ValueError:
        return val.strip()


def table_to_dataframe(raw_table: List[List[str]]) -> pd.DataFrame:
    """Converts a 2D FinQA table list into a clean pandas DataFrame for DuckDB."""
    if not raw_table or len(raw_table) < 2:
        return pd.DataFrame()

    headers = _deduplicate_columns(raw_table[0])
    rows = raw_table[1:]

    # Pad or trim rows to match header length
    normalized_rows = []
    for row in rows:
        padded = row[: len(headers)] + [""] * max(0, len(headers) - len(row))
        normalized_rows.append([_clean_numeric_cell(cell) for cell in padded])

    return pd.DataFrame(normalized_rows, columns=headers)


def parse_finqa_item(item: Dict[str, Any], index: int) -> FinQARecord:
    """Parses a single raw JSON entry from FinQA into a FinQARecord."""
    raw_id = item.get("id", f"finqa_{index}")
    safe_doc_id = _sanitize_identifier(raw_id)
    table_name = f"tbl_{safe_doc_id}"

    pre_lines = [line.strip() for line in item.get("pre_text", []) if line.strip()]
    post_lines = [line.strip() for line in item.get("post_text", []) if line.strip()]

    pre_text = "\n".join(pre_lines)
    post_text = "\n".join(post_lines)
    full_narrative = f"{pre_text}\n\n{post_text}".strip()

    raw_table = item.get("table", [])
    df = table_to_dataframe(raw_table)

    qa_block = item.get("qa", {})
    question = qa_block.get("question", "").strip()
    program = qa_block.get("program")
    exe_ans = qa_block.get("exe_ans")
    ans_str = str(qa_block.get("answer", "")).strip() or None

    gold_answer: Optional[float] = None
    if isinstance(exe_ans, (int, float)):
        gold_answer = float(exe_ans)

    return FinQARecord(
        doc_id=safe_doc_id,
        table_name=table_name,
        pre_text=pre_text,
        post_text=post_text,
        full_narrative=full_narrative,
        raw_table=raw_table,
        dataframe_records=df.to_dict(orient="records"),
        question=question,
        program=program,
        gold_answer=gold_answer,
        gold_answer_str=ans_str,
    )


def load_finqa_split(file_path: str | Path, limit: Optional[int] = None) -> List[FinQARecord]:
    """
    Loads a FinQA split file (train.json, dev.json, or test.json)
    and returns a list of parsed FinQARecord objects.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"FinQA split file not found at: {path}")
    with path.open("r", encoding="utf-8") as f:
        raw_data = json.load(f)
    if limit is not None:
        raw_data = raw_data[:limit]
    return [parse_finqa_item(entry, idx) for idx, entry in enumerate(raw_data)]


def table_to_markdown(raw_table: List[List[str]]) -> str:
    """
    Helper for the Plain RAG baseline: renders a raw 2D FinQA table
    as a Markdown string for naive text chunking.
    """
    if not raw_table:
        return ""
    header = "| " + " | ".join(str(cell).strip() for cell in raw_table[0]) + " |"
    separator = "| " + " | ".join("---" for _ in raw_table[0]) + " |"
    body = [
        "| " + " | ".join(str(cell).strip() for cell in row) + " |"
        for row in raw_table[1:]
    ]
    return "\n".join([header, separator, *body])


if __name__ == "__main__":
    # Quick smoke test when run directly
    sample_path = Path("data/finqa/dev.json")
    if sample_path.exists():
        records = load_finqa_split(sample_path, limit=3)
        for rec in records:
            print(f"Doc ID: {rec.doc_id} | Table: {rec.table_name}")
            print(f"Question: {rec.question}")
            print(f"Gold Answer: {rec.gold_answer} | Program: {rec.program}")
            print(f"Parsed Rows: {len(rec.dataframe_records)}\n{'-'*50}")
    else:
        print(f"Place FinQA JSON files in {sample_path.parent} to run the smoke test.")