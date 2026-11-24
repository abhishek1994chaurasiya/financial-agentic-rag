from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import chromadb
from chromadb.utils import embedding_functions


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 150) -> List[str]:
    """Splits text into overlapping character chunks while respecting paragraph boundaries."""
    cleaned = text.strip()
    if not cleaned:
        return []
    if len(cleaned) <= chunk_size:
        return [cleaned]

    chunks: List[str] = []
    start = 0
    while start < len(cleaned):
        end = start + chunk_size
        if end < len(cleaned):
            # Try breaking at a newline or space to avoid mid-word splits
            break_idx = cleaned.rfind("\n", start + overlap, end)
            if break_idx == -1:
                break_idx = cleaned.rfind(" ", start + overlap, end)
            if break_idx != -1:
                end = break_idx

        chunk = cleaned[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = max(end - overlap, start + 1)

    return chunks


class FinancialVectorStore:
    """ChromaDB wrapper for local semantic retrieval over financial documents."""

    def __init__(
        self,
        persist_dir: str | Path = "data/chroma_db",
        collection_name: str = "finqa_narrative",
        embedding_model: str = "BAAI/bge-small-en-v1.5",
    ) -> None:
        self.persist_path = Path(persist_dir)
        self.persist_path.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(path=str(self.persist_path))
        self.embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embed_fn,
            metadata={"hnsw:space": "cosine"},
        )

    def add_document(
        self,
        doc_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        chunk_size: int = 1200,
        overlap: int = 150,
    ) -> int:
        """Chunks and indexes a document into ChromaDB. Returns number of chunks added."""
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        if not chunks:
            return 0

        base_meta = metadata.copy() if metadata else {}
        base_meta["doc_id"] = doc_id

        ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [{**base_meta, "chunk_index": i} for i in range(len(chunks))]

        self.collection.upsert(
            ids=ids,
            documents=chunks,
            metadatas=metadatas,
        )
        return len(chunks)

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        doc_id_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieves top-k relevant chunks, optionally scoped to a specific doc_id."""
        where_clause = {"doc_id": doc_id_filter} if doc_id_filter else None
        results = self.collection.query(
            query_texts=[query_text],
            n_results=top_k,
            where=where_clause,
        )

        hits: List[Dict[str, Any]] = []
        if not results["documents"] or not results["documents"][0]:
            return hits

        for idx, doc_text in enumerate(results["documents"][0]):
            hits.append(
                {
                    "id": results["ids"][0][idx],
                    "text": doc_text,
                    "metadata": results["metadatas"][0][idx] if results["metadatas"] else {},
                    "distance": results["distances"][0][idx] if results["distances"] else None,
                }
            )
        return hits

    def count(self) -> int:
        return self.collection.count()