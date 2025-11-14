from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class VectorFaissRetriever:
    data_dir: Path
    index_path: Path | None = None
    meta_path: Path | None = None

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        if self.index_path is None:
            self.index_path = self.data_dir / "index" / "faiss.index"
        if self.meta_path is None:
            self.meta_path = self.data_dir / "index" / "faiss_meta.pkl"
        self._index: faiss.Index | None = None
        self._documents: list[dict[str, str]] | None = None
        self._model_name: str | None = None
        self._embedder: SentenceTransformer | None = None

    def load(self) -> None:
        idx = self.index_path
        meta_path = self.meta_path
        assert idx is not None and meta_path is not None
        if not idx.exists() or not meta_path.exists():
            raise FileNotFoundError(f"FAISS index or meta not found at {idx} / {meta_path}")
        self._index = faiss.read_index(str(idx))
        with meta_path.open("rb") as f:
            meta_obj: dict[str, Any] = pickle.load(f)
        self._documents = meta_obj["documents"]
        model_name_val = meta_obj.get("model_name") or "sentence-transformers/all-MiniLM-L6-v2"
        self._model_name = str(model_name_val)
        try:
            local_only = os.getenv("LOCAL_EMBEDDINGS_ONLY", "1") == "1"
            self._embedder = SentenceTransformer(self._model_name, local_files_only=local_only)
        except Exception:
            self._embedder = None

    def _ensure_loaded(self) -> None:
        if self._index is None or self._documents is None:
            self.load()

    def retrieve(self, query: str, k: int = 8) -> list[dict[str, str]]:
        self._ensure_loaded()
        assert self._index is not None
        assert self._documents is not None
        if self._embedder is None:
            return []
        q_emb = self._embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        q_emb = q_emb.astype(np.float32)
        scores, indices = self._index.search(q_emb, k)
        out: list[dict[str, str]] = []
        for idx in indices[0]:
            if idx < 0 or idx >= len(self._documents):
                continue
            d = self._documents[idx]
            out.append(
                {
                    "text": d.get("text", ""),
                    "source": d.get("source", f"chunk-{idx}"),
                    "title": d.get("title"),
                    "desc": d.get("desc"),
                }
            )
        return out
