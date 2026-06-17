"""Anchor extraction and the anchor pool T_t (Sec. 3).

At iteration t the attacker parses the agent response r_t and extracts keywords
/ topics ("anchors") in three steps: keyword detection (NER), normalization
(canonicalising PII tokens), and de-duplication. A new anchor ``a`` is admitted
to the pool only if it is sufficiently different from every existing anchor:

    add a  iff  max_{a' in T_{t-1}} sim(a, a') <= alpha
"""
from __future__ import annotations

import re
from typing import Dict, List, Sequence

import numpy as np

from .embeddings import cosine

# Tokens that are PII / non-topical and should be canonicalised away during
# normalisation rather than kept as anchors.
_STOP = {
    "the", "a", "an", "of", "to", "for", "and", "or", "is", "are", "was", "were",
    "with", "on", "in", "at", "since", "had", "has", "have", "that", "this",
    "what", "when", "who", "which", "find", "me", "please", "all", "any",
    "previous", "stored", "questions", "query", "queries", "here", "i", "you",
    "your", "my", "first", "last", "name", "list", "show", "return", "output",
    "patient", "customers", "recommend", "right", "now", "can", "help", "some",
    "they", "their", "do", "did", "get", "got",
}
_NUM_RE = re.compile(r"\b\d[\d/.\-:]*\b")
_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z\-]+")


def _normalize(text: str) -> str:
    """Canonicalise PII tokens (IDs, dates, numbers) -> <NUM> placeholder."""
    return _NUM_RE.sub(" <num> ", text.lower())


def extract_anchors(response_text: str, max_anchors: int = 12) -> List[str]:
    """Keyword detection + normalisation + de-duplication on a response.

    A lightweight NER substitute: keep content words (non-stop, alphabetic,
    length>=3), preserving order, de-duplicated.
    """
    norm = _normalize(response_text)
    seen: Dict[str, None] = {}
    for w in _WORD_RE.findall(norm):
        if len(w) < 3 or w in _STOP:
            continue
        seen.setdefault(w, None)
    return list(seen.keys())[:max_anchors]


class AnchorPool:
    """The growing set T_t of distinct anchors, with cached embeddings."""

    def __init__(self, encoder, seeds: Sequence[str], alpha: float = 0.5):
        self.encoder = encoder
        self.alpha = alpha
        self.anchors: List[str] = list(dict.fromkeys(seeds))
        self.emb: Dict[str, np.ndarray] = {
            a: encoder.encode([a])[0] for a in self.anchors
        }

    def __len__(self) -> int:
        return len(self.anchors)

    def __contains__(self, a: str) -> bool:
        return a in self.emb

    def vec(self, a: str) -> np.ndarray:
        if a not in self.emb:
            self.emb[a] = self.encoder.encode([a])[0]
        return self.emb[a]

    def update(self, candidates: Sequence[str]) -> List[str]:
        """Admit candidates that satisfy max sim to existing anchors <= alpha.

        Returns the list of newly-added anchors.
        """
        added: List[str] = []
        for a in candidates:
            if a in self.emb:
                continue
            av = self.encoder.encode([a])[0]
            if self.anchors:
                max_sim = max(cosine(av, self.emb[a2]) for a2 in self.anchors)
            else:
                max_sim = 0.0
            if max_sim <= self.alpha:
                self.anchors.append(a)
                self.emb[a] = av
                added.append(a)
        return added

    def matrix(self, subset: Sequence[str] | None = None) -> np.ndarray:
        keys = list(subset) if subset is not None else self.anchors
        if not keys:
            return np.zeros((0, 1))
        return np.vstack([self.vec(a) for a in keys])
