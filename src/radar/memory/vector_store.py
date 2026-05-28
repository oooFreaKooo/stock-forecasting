from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from radar.memory.regime_encoder import REGIME_VECTOR_COLUMNS, regime_vectors_as_matrix


def _date_to_ord(date_str: str) -> int:
    return int(date_str.replace("-", ""))


def _cosine_similarities(candidates: np.ndarray, query_vector: np.ndarray, chunk_size: int = 8) -> np.ndarray:
    """Stable batched cosine similarity (avoids BLAS overflow on macOS)."""
    candidates = np.asarray(candidates, dtype=np.float64)
    query = np.asarray(query_vector, dtype=np.float64).reshape(-1)
    candidate_norms = np.linalg.norm(candidates, axis=1, keepdims=True)
    candidate_norms = np.where(candidate_norms > 0, candidate_norms, 1.0)
    candidates = candidates / candidate_norms

    query_norm = np.linalg.norm(query)
    if query_norm > 0:
        query = query / query_norm
    else:
        query = np.zeros_like(query)

    if len(candidates) == 0:
        return np.array([], dtype=np.float64)

    sims: list[np.ndarray] = []
    for start in range(0, len(candidates), chunk_size):
        chunk = candidates[start:start + chunk_size]
        sims.append(chunk @ query)
    return np.concatenate(sims)


def _ord_to_date(date_ord: int) -> str:
    s = str(date_ord)
    return f"{s[:4]}-{s[4:6]}-{s[6:8]}"


@dataclass
class RegimeMatch:
    date: str
    similarity: float
    metadata: dict


class RegimeVectorStore:
    """ChromaDB-backed store for daily regime vectors."""

    COLLECTION_NAME = "regime_vectors"

    def __init__(self, store_dir: Union[str, Path]) -> None:
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self._client = None
        self._collection = None

    def _ensure_client(self):
        if self._client is None:
            import chromadb

            self._client = chromadb.PersistentClient(path=str(self.store_dir))
            self._collection = self._client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def _reset_collection(self, embedding_dim: int) -> None:
        import chromadb

        if self._client is None:
            self._client = chromadb.PersistentClient(path=str(self.store_dir))
        try:
            self._client.delete_collection(self.COLLECTION_NAME)
        except Exception:
            pass
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine", "embedding_dim": embedding_dim},
        )

    def _collection_embedding_dim(self, collection) -> Optional[int]:
        meta_dim = (collection.metadata or {}).get("embedding_dim")
        if meta_dim is not None:
            return int(meta_dim)
        if collection.count() == 0:
            return None
        sample = collection.get(limit=1, include=["embeddings"])
        embeddings = sample.get("embeddings")
        if embeddings is not None and len(embeddings) > 0 and embeddings[0] is not None:
            return len(embeddings[0])
        return None

    def upsert_regimes(self, regime_df: pd.DataFrame) -> int:
        """Insert or update regime vectors in ChromaDB."""
        vectors, date_ids = regime_vectors_as_matrix(regime_df)
        expected_dim = vectors.shape[1]

        collection = self._ensure_client()
        existing_dim = self._collection_embedding_dim(collection)
        if existing_dim is not None and existing_dim != expected_dim:
            self._reset_collection(expected_dim)
            collection = self._collection

        metadatas = []
        for _, row in regime_df.iterrows():
            date_str = row["date"].strftime("%Y-%m-%d")
            meta = {
                "date": date_str,
                "date_ord": _date_to_ord(date_str),
            }
            for col in REGIME_VECTOR_COLUMNS:
                val = row.get(col)
                if pd.notna(val):
                    meta[col] = float(val)
            metadatas.append(meta)

        collection.upsert(
            ids=date_ids,
            embeddings=vectors.tolist(),
            metadatas=metadatas,
        )
        return len(date_ids)

    def query_similar(
        self,
        query_vector: np.ndarray,
        before_date: str,
        top_k: int = 5,
    ) -> list[RegimeMatch]:
        """
        Query top-k similar regimes strictly before before_date.

        Uses metadata filter to prevent look-ahead.
        """
        collection = self._ensure_client()
        norm = np.linalg.norm(query_vector)
        if norm > 0:
            query_vector = query_vector / norm

        result = collection.query(
            query_embeddings=[query_vector.tolist()],
            n_results=min(top_k * 3, max(top_k, 1)),
            where={"date_ord": {"$lt": _date_to_ord(before_date)}},
        )

        matches: list[RegimeMatch] = []
        if not result["ids"] or not result["ids"][0]:
            return matches

        for doc_id, dist, meta in zip(
            result["ids"][0],
            result["distances"][0],
            result["metadatas"][0],
        ):
            similarity = 1.0 - dist
            matches.append(
                RegimeMatch(date=doc_id, similarity=float(similarity), metadata=meta or {})
            )
            if len(matches) >= top_k:
                break
        return matches

    def count(self) -> int:
        collection = self._ensure_client()
        return collection.count()


class InMemoryRegimeIndex:
    """Numpy fallback for tests and batch enrichment without ChromaDB."""

    def __init__(self) -> None:
        self._dates: list[str] = []
        self._vectors: Optional[np.ndarray] = None
        self._meta: list[dict] = []

    def build_from_frame(self, regime_df: pd.DataFrame) -> None:
        vectors, date_ids = regime_vectors_as_matrix(regime_df)
        self._vectors = vectors
        self._dates = date_ids
        self._meta = [
            {"date": d, **{c: float(row[c]) for c in REGIME_VECTOR_COLUMNS if c in row and pd.notna(row[c])}}
            for d, (_, row) in zip(date_ids, regime_df.iterrows())
        ]

    def query_similar(
        self,
        query_vector: np.ndarray,
        before_date: str,
        top_k: int = 5,
    ) -> list[RegimeMatch]:
        if self._vectors is None or len(self._dates) == 0:
            return []

        query_vector = np.asarray(query_vector, dtype=np.float64)

        mask = np.array([d < before_date for d in self._dates])
        if not mask.any():
            return []

        candidates = self._vectors[mask]
        candidate_dates = [d for d, m in zip(self._dates, mask) if m]
        candidate_meta = [m for m, keep in zip(self._meta, mask) if keep]

        sims = _cosine_similarities(candidates, query_vector)
        order = np.argsort(-sims)[:top_k]

        return [
            RegimeMatch(
                date=candidate_dates[i],
                similarity=float(sims[i]),
                metadata=candidate_meta[i],
            )
            for i in order
        ]
