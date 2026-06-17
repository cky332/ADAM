"""Sentence encoders z(.) : T -> R^d (Sec. 3).

The paper uses ``all-MiniLM-L6-v2`` and shows the attack is robust to the
choice of encoder (Fig. 3g). To keep the reproduction runnable fully offline
we ship a deterministic bag-of-words encoder as the default, and transparently
upgrade to a real Sentence-Transformer when one is installed/requested.

Both encoders return L2-normalised vectors so that cosine similarity is a plain
dot product.
"""
from __future__ import annotations

import hashlib
import re
from typing import List, Sequence

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Function words / boilerplate carry little topical signal. Real sentence
# encoders downweight them; our bag-of-words encoder simply drops them so that
# distinctive content words drive cosine similarity (and hence topic-selective
# retrieval). Pure digit tokens (IDs/dates) are also dropped as non-topical.
_STOPWORDS = {
    "the", "a", "an", "of", "to", "for", "and", "or", "is", "are", "was", "were",
    "with", "on", "in", "at", "since", "had", "has", "have", "that", "this",
    "what", "when", "who", "which", "where", "how", "find", "me", "please", "all",
    "any", "here", "you", "your", "my", "i", "do", "did", "can", "could", "would",
    "some", "they", "their", "there", "it", "as", "by", "from",
}


def _tokenize(text: str) -> List[str]:
    toks = _TOKEN_RE.findall(text.lower())
    return [t for t in toks if t not in _STOPWORDS and not t.isdigit()]


class HashingEncoder:
    """Deterministic, dependency-free encoder.

    Each token is hashed into a fixed-width vector via the hashing trick and
    accumulated; the result is L2-normalised. Texts that share words land close
    together in cosine space, which is exactly what the attack's clustering and
    k-center steps rely on. This makes the whole pipeline reproducible without a
    GPU or network access while preserving the semantic-overlap structure the
    paper exploits.
    """

    name = "hashing-256"

    def __init__(self, dim: int = 256):
        self.dim = dim

    def _embed_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float64)
        toks = _tokenize(text)
        if not toks:
            return vec
        for tok in toks:
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            idx = h % self.dim
            sign = 1.0 if (h // self.dim) % 2 == 0 else -1.0
            vec[idx] += sign
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        return np.vstack([self._embed_one(t) for t in texts])


class SentenceTransformerEncoder:
    """Wrapper around sentence-transformers (loaded lazily, optional dep)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # noqa: lazy import

        self.name = model_name
        self._model = SentenceTransformer(model_name)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        return np.asarray(
            self._model.encode(list(texts), normalize_embeddings=True),
            dtype=np.float64,
        )


def get_encoder(name: str = "hashing"):
    """Factory. ``name`` in {hashing, all-MiniLM-L6-v2, e5-large-v2, ...}.

    Falls back to the offline HashingEncoder if sentence-transformers (or the
    requested model) is unavailable, so experiments never hard-fail.
    """
    if name is None or name == "hashing":
        return HashingEncoder()
    if name.startswith("hashing-"):
        try:
            return HashingEncoder(dim=int(name.split("-")[1]))
        except ValueError:
            return HashingEncoder()
    try:
        return SentenceTransformerEncoder(name)
    except Exception as exc:  # pragma: no cover - depends on environment
        print(f"[embeddings] '{name}' unavailable ({exc}); using HashingEncoder.")
        return HashingEncoder()


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def score_matrix(query_vec: np.ndarray, mat: np.ndarray, scoring: str = "cosine") -> np.ndarray:
    """Similarity scores of one query vector against rows of ``mat`` (Fig. 3h)."""
    if scoring == "cosine":
        qn = query_vec / (np.linalg.norm(query_vec) + 1e-12)
        mn = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)
        return mn @ qn
    if scoring == "dot":
        return mat @ query_vec
    if scoring == "l2":
        return -np.linalg.norm(mat - query_vec, axis=1)  # negative dist -> larger=closer
    raise ValueError(f"unknown scoring '{scoring}'")
