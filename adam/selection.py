"""Weighted k-center anchor selection (Sec. 3, Eq. 1).

Adapts the k-center idea from active learning (Sener & Savarese, 2017):

  a*_1 = argmax_{a in T_t \ A_used}      P^_t(a)
  a*_j = argmax_{a in T_t \ A^{(j-1)}}   P^_t(a) * min_{a' in A^{(j-1)}} ||z(a)-z(a')||_2

The first pick is the most promising *unused* anchor; each subsequent pick is
both promising and far (in embedding space) from already-selected anchors,
yielding a diverse, high-probability batch.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Set

import numpy as np


def select_anchors(pool, P: Dict[str, float], k: int,
                   used: Set[str]) -> List[str]:
    candidates = list(pool.anchors)
    if not candidates:
        return []

    # a*_1 : most promising anchor not used before (fall back to all if needed)
    unused = [a for a in candidates if a not in used]
    pool_for_first = unused if unused else candidates
    a1 = max(pool_for_first, key=lambda a: P.get(a, 0.0))
    selected: List[str] = [a1]

    # a*_j : weighted k-center greedy step
    while len(selected) < min(k, len(candidates)):
        sel_mat = np.vstack([pool.vec(a) for a in selected])
        best_a, best_score = None, -np.inf
        for a in candidates:
            if a in selected:
                continue
            av = pool.vec(a)
            dist = float(np.min(np.linalg.norm(sel_mat - av, axis=1)))
            score = P.get(a, 0.0) * dist
            if score > best_score:
                best_score, best_a = score, a
        if best_a is None:
            break
        selected.append(best_a)
    return selected
