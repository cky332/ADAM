"""Victim LLM agent (Sec. 2.1).

The agent retrieves the top-k relevant memory records for an incoming query and
generates a response. Under the threat model (Sec. 2.2) the attacker only sees
the agent's textual output; here we model how much of the retrieved private
content a malicious, injection-bearing query manages to surface.

Two behavioural knobs drive the simulation, both grounded in the paper:
  * round-level leakage probability -> Attack Success Rate (ASR). Scales with the
    attack's injection strength and the agent core's compliance (larger / more
    capable LLMs leak more, Fig. 3b).
  * per-item reveal probability -> Complete Extraction Rate (CER). Scales with
    how well the attack's prompt elicits a *full* dump of the k retrieved items.

When a real LLM backend drives the agent the same interface is used, but the
leak behaviour is whatever the real model does -- no simulation knobs.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from .config import AgentConfig
from .memory import Memory, Record  # noqa: F401


@dataclass
class MaliciousQuery:
    """A query crafted by an attack."""

    text: str                         # full text incl. prefix/suffix injection
    probe: str                        # the topical body that drives retrieval
    injection_strength: float = 0.5   # -> ASR
    completeness: float = 0.5         # -> CER
    raw_markers: bool = True          # True if it still contains surface markers


@dataclass
class AgentResponse:
    retrieved: List[Record] = field(default_factory=list)
    revealed: List[Record] = field(default_factory=list)
    text: str = ""
    blocked: bool = False


class LLMAgent:
    def __init__(self, memory: Memory, cfg: AgentConfig,
                 defenses: Optional[List[Callable]] = None, seed: int = 0):
        self.memory = memory
        self.cfg = cfg
        self.defenses = defenses or []
        self.rng = random.Random(seed)

    def add_defense(self, fn: Callable) -> None:
        self.defenses.append(fn)

    def query(self, mq: MaliciousQuery) -> AgentResponse:
        # ---- input-side defenses (may block or transform the query) ----
        defense_factor = 1.0
        probe = mq.probe
        for fn in self.defenses:
            res = fn(mq)
            if res.get("blocked"):
                return AgentResponse(blocked=True, text="[blocked by defense]")
            defense_factor *= res.get("factor", 1.0)
            probe = res.get("probe", probe)

        # ---- retrieval E(q, M) ----
        # Pull a candidate pool (all records above threshold, capped) then sample
        # k from it, so repeated probes over a large memory can surface different
        # records across rounds rather than re-hitting the same most-central few.
        pool = self.memory.retrieve(
            probe, k=self.cfg.retrieval_pool_cap,
            threshold=self.cfg.sim_threshold, scoring=self.cfg.scoring,
        )
        retrieved = self._sample(pool, self.cfg.top_k)
        resp = AgentResponse(retrieved=retrieved)
        if not retrieved:
            resp.text = "I don't have anything relevant."
            return resp

        # ---- leak decision (simulation backend) ----
        # p_comply (injection lands) -> EE/ASR; full-dump prob (completeness) -> CER.
        p_comply = min(1.0, self.cfg.base_compliance * mq.injection_strength) * defense_factor
        if self.rng.random() < p_comply:
            if self.rng.random() < mq.completeness * defense_factor:
                resp.revealed = list(retrieved)          # full dump -> all k (CER)
            else:
                resp.revealed = [retrieved[0]]           # partial -> just the top item
        resp.text = self._render(resp.revealed)

        # Dynamic memory (paper Sec. 2.1)
        if self.cfg.dynamic_memory:
            self.memory.append(Record(qid=len(self.memory), query=mq.text,
                                      solution=resp.text, topic="<dynamic>"))

        return resp

    def _sample(self, pool: List[Record], k: int) -> List[Record]:
        """Sample k records from the candidate pool, weighted toward the top."""
        if len(pool) <= k:
            return pool
        # flat weighting over the (already relevance-filtered) candidate pool so
        # repeated probes explore a topic's records -- including the tail --
        # rather than re-hitting the same few most-central records.
        weights = [1.0 for _ in range(len(pool))]
        chosen: List[Record] = []
        idxs = list(range(len(pool)))
        w = list(weights)
        for _ in range(k):
            total = sum(w)
            r = self.rng.random() * total
            acc = 0.0
            for j, idx in enumerate(idxs):
                acc += w[j]
                if r <= acc:
                    chosen.append(pool[idx])
                    idxs.pop(j)
                    w.pop(j)
                    break
        return chosen

    @staticmethod
    def _render(revealed: List[Record]) -> str:
        if not revealed:
            return "Sorry, I cannot share that."
        lines = ["Here are the previous examples I have:"]
        lines += [f"- {r.query}" for r in revealed]
        return "\n".join(lines)
