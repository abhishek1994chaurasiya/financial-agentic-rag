# Retrieves the top-$k$ chunks from the finqa_baseline Chroma collection. Note that in FinQA evaluation, each question belongs to a specific filing (doc_id_filter=rec.doc_id), mirroring single-document QA. You can also test open-corpus retrieval (doc_id_filter=None) later as an ablation.

from __future__ import annotations
from typing import Any, Dict, List, Optional
from src.stores.vector_store import FinancialVectorStore


class BaselineRetriever:
    """Dense vector retriever over flattened text + Markdown tables."""

    def __init__(
        self,
        persist_dir: str = "data/chroma_db",
        collection_name: str = "finqa_baseline",
        top_k: int = 5,
    ) -> None:
        self.store = FinancialVectorStore(
            persist_dir=persist_dir,
            collection_name=collection_name,
        )
        self.top_k = top_k

    def retrieve(self, question: str, doc_id: Optional[str] = None) -> List[Dict[str, Any]]:
        return self.store.query(
            query_text=question,
            top_k=self.top_k,
            doc_id_filter=doc_id,
        )