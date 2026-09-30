"""Local retrieval only: BM25 written here, dense vectors from a small local model (no API key)."""
from __future__ import annotations

import math
import os
import pickle
from collections import Counter
from functools import lru_cache

from .leak import words


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.toks = [words(d) for d in docs]
        self.k1, self.b = k1, b
        self.avg = sum(map(len, self.toks)) / max(1, len(self.toks))
        df = Counter(t for doc in self.toks for t in set(doc))
        n = len(self.toks)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.tf = [Counter(doc) for doc in self.toks]

    def scores(self, query: str) -> list[float]:
        q = words(query)
        out = []
        for tf, doc in zip(self.tf, self.toks):
            s = 0.0
            for t in q:
                if t in tf:
                    f = tf[t]
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * len(doc) / self.avg))
            out.append(s)
        return out


class Dense:
    """Cosine similarity over a local embedding model (fastembed, ONNX, CPU)."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", embedder=None, cache_file=None):
        self.model_name = model_name
        self._embedder = embedder
        self.cache_file = cache_file
        self._memo = {}
        if cache_file and os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                self._memo = pickle.load(f)

    def save(self):
        if self.cache_file:
            tmp = f"{self.cache_file}.tmp"
            with open(tmp, "wb") as f:
                pickle.dump(self._memo, f)
            os.replace(tmp, self.cache_file)

    def _model(self):
        if self._embedder is None:
            from fastembed import TextEmbedding
            self._embedder = TextEmbedding(self.model_name)
        return self._embedder

    def embed(self, texts: list[str]):
        import numpy as np
        todo = [t for t in dict.fromkeys(texts) if t not in self._memo]
        if todo:
            for t, v in zip(todo, self._model().embed(todo)):
                self._memo[t] = np.asarray(v, dtype="float32")
        vecs = np.array([self._memo[t] for t in texts], dtype="float32")
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.clip(norms, 1e-9, None)

    def scores(self, query: str, doc_vecs) -> list[float]:
        q = self.embed([query])[0]
        return (doc_vecs @ q).tolist()


def top_k(scores: list[float], k: int) -> list[int]:
    return sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:k]


def rrf(rankings: list[list[int]], k: int = 60) -> list[int]:
    """Reciprocal rank fusion of several ranked id lists."""
    total: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            total[idx] = total.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(total, key=lambda i: (-total[i], i))


def chunk_words(text: str, size: int = 200, overlap: int = 40) -> list[str]:
    """Fixed-size word windows, the default splitter of most RAG tutorials."""
    w = text.split()
    if len(w) <= size:
        return [" ".join(w)] if w else []
    step = size - overlap
    return [" ".join(w[i:i + size]) for i in range(0, len(w) - overlap, step)]
