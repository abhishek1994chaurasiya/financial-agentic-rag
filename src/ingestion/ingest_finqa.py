from __future__ import annotations

import argparse
from tqdm import tqdm
from src.ingestion.finqa_loader import (
    load_finqa_split,
    table_to_dataframe,
    table_to_markdown,
)
from src.stores.table_store import FinancialTableStore
from src.stores.vector_store import FinancialVectorStore


def run_ingestion(split_path: str, limit: int | None = None) -> None:
    records = load_finqa_split(split_path, limit=limit)
    print(f"Loaded {len(records)} records from {split_path}")

    baseline_vstore = FinancialVectorStore(collection_name="finqa_baseline")
    agentic_vstore = FinancialVectorStore(collection_name="finqa_narrative")
    table_store = FinancialTableStore()

    for rec in tqdm(records, desc="Ingesting into Dual Stores"):
        # 1. Plain RAG Baseline Store: Narrative + Flattened Markdown Table
        md_table = table_to_markdown(rec.raw_table)
        combined_text = f"{rec.pre_text}\n\n{md_table}\n\n{rec.post_text}".strip()
        baseline_vstore.add_document(
            doc_id=rec.doc_id,
            text=combined_text,
            metadata={"table_name": rec.table_name, "source": "finqa"},
        )

        # 2. Agentic Narrative Store: Narrative Text Only
        if rec.full_narrative:
            agentic_vstore.add_document(
                doc_id=rec.doc_id,
                text=rec.full_narrative,
                metadata={"table_name": rec.table_name, "source": "finqa"},
            )

        # 3. Agentic Structured Store: DuckDB Table
        df = table_to_dataframe(rec.raw_table)
        if not df.empty:
            table_store.register_dataframe(rec.table_name, df)

    print(f"Baseline Chroma chunks: {baseline_vstore.count()}")
    print(f"Agentic Narrative chunks: {agentic_vstore.count()}")
    print(f"DuckDB registered tables: {len(table_store.list_tables())}")
    table_store.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest FinQA split into ChromaDB and DuckDB.")
    parser.add_argument("--split", type=str, default="data/finqa/dev.json", help="Path to FinQA JSON split")
    parser.add_argument("--limit", type=int, default=200, help="Number of records to ingest (use -1 for all)")
    args = parser.parse_args()

    record_limit = None if args.limit == -1 else args.limit
    run_ingestion(split_path=args.split, limit=record_limit)