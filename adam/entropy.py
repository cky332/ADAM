"""Entropy-based query selection (Sec. 3).

For each selected anchor the generator produces a candidate query. We pick the
candidate with the highest topic-distribution entropy, i.e. the one most likely
to surface *new* memory content:

    phi(q) ⊆ T_t                      # topics q may retrieve from M
    H_t(q) = - sum_{a in phi(q)} P^_t(a) * log( P^_t(a) + eps )
    q_t    = argmax_q H_t(q)

High entropy <=> the predicted topics are spread out / under-explored
(Appendix G, Table 8).
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from .embeddings import cosine


def query_topics(query_vec: np.ndarray, pool, top: int = 3,
                 min_sim: float = 0.2) -> List[str]:
    """phi(q): the anchors (topics) a query is most likely to retrieve."""
    sims = [(a, cosine(query_vec, pool.vec(a))) for a in pool.anchors]
    sims.sort(key=lambda x: -x[1])
    chosen = [a for a, s in sims[:top] if s >= min_sim]
    return chosen or [sims[0][0]] if sims else []


def entropy(topics: Sequence[str], P: Dict[str, float], eps: float) -> float:
    if not topics:
        return 0.0
    p = np.array([P.get(a, 0.0) for a in topics], dtype=np.float64)
    s = p.sum()
    if s > 0:                      # normalise over the query's topic support
        p = p / s
    return float(-np.sum(p * np.log(p + eps)))


def select_query(candidates: List[dict], P: Dict[str, float], eps: float) -> dict:
    """Pick the candidate maximising H_t(q). Each candidate: {text, probe, vec, topics}."""
    best, best_h = candidates[0], -np.inf
    for c in candidates:
        h = entropy(c["topics"], P, eps)
        c["entropy"] = h
        if h > best_h:
            best_h, best = h, c
    return best
