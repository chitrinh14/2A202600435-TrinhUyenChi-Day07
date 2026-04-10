from __future__ import annotations
from typing import Any, Callable

from .chunking import _dot
from .embeddings import _mock_embed
from .models import Document

class EmbeddingStore:
    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._store: list[dict[str, Any]] = []
        self._next_index = 0
        
        # Tắt ChromaDB để chạy mượt trên Python 3.14
        self._use_chroma = False
        self._collection = None

    def _make_record(self, doc: Document) -> dict[str, Any]:
        # TRỌNG TÂM: Sử dụng "content" theo chuẩn của main.py
        content = getattr(doc, "content", getattr(doc, "text", str(doc)))
        metadata = getattr(doc, "metadata", {})
        doc_id = getattr(doc, "id", None) or f"chunk_{self._next_index}"
        self._next_index += 1

        return {
            "id": str(doc_id),
            "content": content, # Đồng nhất với main.py
            "metadata": metadata,
            "embedding": self._embedding_fn(content)
        }

    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        if not records:
            return []

        query_emb = self._embedding_fn(query)
        scored_records = []

        for record in records:
            try:
                score = _dot(query_emb, record["embedding"])
            except NameError:
                score = sum(a * b for a, b in zip(query_emb, record["embedding"]))
            scored_records.append((score, record))

        scored_records.sort(key=lambda x: x[0], reverse=True)

        final_results = []
        for score, rec in scored_records[:top_k]:
            result_dict = rec.copy()
            result_dict["score"] = score # Thêm score để main.py in ra
            final_results.append(result_dict)

        return final_results

    def add_documents(self, docs: list[Document]) -> None:
        if not docs:
            return
        records = [self._make_record(doc) for doc in docs]
        self._store.extend(records) # Chắc chắn dữ liệu được nạp vào kho

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        return self._search_records(query, self._store, top_k)

    def get_collection_size(self) -> int:
        return len(self._store)

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        metadata_filter = metadata_filter or {}
        filtered_records = []
        for record in self._store:
            match = True
            for key, value in metadata_filter.items():
                if record["metadata"].get(key) != value:
                    match = False
                    break
            if match:
                filtered_records.append(record)
        return self._search_records(query, filtered_records, top_k)

    def delete_document(self, doc_id: str) -> bool:
        initial_count = len(self._store)
        self._store = [r for r in self._store if r.get("metadata", {}).get("doc_id") != doc_id]
        return len(self._store) < initial_count