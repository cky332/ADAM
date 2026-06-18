"""Agent memory module M (Sec. 2.1).

M is a sequence of records (q_i, s_i). Each record also carries a latent topic
z_i (used only to define the *ground-truth* topic distribution P(D) for the
oracle test, Sec. 4.3 / Fig. 4) and a cached query embedding for retrieval.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .embeddings import score_matrix


@dataclass
class Record:
    qid: int
    query: str          # q_i : the private user query (the attack target)
    solution: str       # s_i : agent-generated solution
    topic: str          # z_i : latent ground-truth topic (oracle only)


class Memory:
    def __init__(self, records: List[Record], encoder):
        self.records = records
        self.encoder = encoder
        self._emb = encoder.encode([r.query for r in records]) if records else np.zeros((0, 1))

    def __len__(self) -> int:
        return len(self.records)

    def append(self, record: Record) -> None:
        """Append a new (q, s) record to M (paper Sec. 2.1, dynamic memory).

        The retrieval embedding matrix is grown by one row so subsequent
        retrievals can match against the new query immediately.
        """
        self.records.append(record)
        new_emb = self.encoder.encode([record.query])
        self._emb = (np.vstack([self._emb, new_emb]) if self._emb.size
                     else new_emb)

    def retrieve(self, query: str, k: int, threshold: float = 0.0,
                 scoring: str = "cosine") -> List[Record]:
        """Top-k records E(q, M) = {(q_i, s_i) | f(q, q_i) in top-k} (Sec. 2.1)."""
        if not self.records:
            return []
        qv = self.encoder.encode([query])[0]
        scores = score_matrix(qv, self._emb, scoring=scoring)
        order = np.argsort(-scores)
        out: List[Record] = []
        for idx in order:                       # scan all, collect up to k above threshold
            if scoring == "cosine" and scores[idx] < threshold:
                break                           # scores are sorted desc -> rest are lower
            out.append(self.records[idx])
            if len(out) >= k:
                break
        return out

    def topic_distribution(self) -> dict:
        """Ground-truth P(D) over latent topics (used for the oracle test)."""
        counts: dict = {}
        for r in self.records:
            counts[r.topic] = counts.get(r.topic, 0) + 1
        total = float(len(self.records)) or 1.0
        return {t: c / total for t, c in counts.items()}
