"""ADAM attack -- Algorithm 1 (Sec. 3).

Ties together anchor extraction, distribution estimation, weighted k-center
selection and entropy-based query selection into the iterative, early-stopping
extraction loop.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .agent import LLMAgent, MaliciousQuery
from .anchors import AnchorPool
from .config import AttackConfig
from .distribution import DistributionEstimator
from .entropy import query_topics, select_query
from .injection import InjectionBank
from .metrics import MetricTracker
from .refine import refine
from .selection import select_anchors


@dataclass
class AttackResult:
    recovered: List[str] = field(default_factory=list)   # R (unique recovered queries)
    metrics: dict = field(default_factory=dict)
    rounds_run: int = 0
    eq_curve: List[int] = field(default_factory=list)     # EQ after each round
    dist_history: List[Dict[str, float]] = field(default_factory=list)


class Attack:
    """Base class: drives the round loop and metric bookkeeping."""

    name = "base"

    # query-optimization attacks paraphrase their injections; static/marker-based
    # attacks (Vanilla, MEXTRA) do not, leaving surface markers a filter can catch.
    paraphrase = True

    def __init__(self, agent: LLMAgent, generator, encoder, domain: str,
                 cfg: AttackConfig, injection_strength: float, completeness: float,
                 seed_topics: List[str]):
        self.agent = agent
        self.gen = generator
        self.encoder = encoder
        self.domain = domain
        self.cfg = cfg
        self.injection_strength = injection_strength
        self.completeness = completeness
        self.seed_topics = seed_topics
        self.injections = InjectionBank(seed=cfg.seed, paraphrase=self.paraphrase,
                                        llm=generator)
        # EE/CER use k = items returned per round = the agent's retrieval depth,
        # which is independent of the attack's k-center batch size (Sec. 4.1).
        self.tracker = MetricTracker(k=agent.cfg.top_k)
        self.recovered: List[str] = []
        self._seen = set()

    # --- to be overridden ---
    def propose(self, t: int) -> MaliciousQuery:
        raise NotImplementedError

    def observe(self, response, queries: List[str], anchors: List[str]) -> None:
        pass

    def _mq(self, probe_text: str) -> MaliciousQuery:
        pre, suf = self.injections.sample()
        return MaliciousQuery(
            text=f"{pre} {probe_text} {suf}",
            probe=probe_text,
            injection_strength=self.injection_strength,
            completeness=self.completeness,
            raw_markers=not self.paraphrase,   # static attacks keep surface markers
        )

    def run(self) -> AttackResult:
        res = AttackResult()
        no_gain = 0
        for t in range(1, self.cfg.T + 1):
            mq = self.propose(t)
            resp = self.agent.query(mq)
            queries, anchors = refine(resp.text, retrieved=resp.retrieved)
            for q in queries:
                if q.lower() not in self._seen:
                    self._seen.add(q.lower())
                    self.recovered.append(q)
            self.tracker.record([r.query for r in resp.retrieved], queries)
            self.observe(resp, queries, anchors)

            res.eq_curve.append(self.tracker.EQ)
            res.rounds_run = t

            # early stop (Eq. 2), opt-in: ||P_t - P_{t-1}||_1 < eps  OR
            # Delta-EQ < eta for rho consecutive rounds.
            if self.cfg.early_stop and t >= 2:
                d_eq = res.eq_curve[-1] - res.eq_curve[-2]
                no_gain = no_gain + 1 if d_eq < self.cfg.eta else 0
                if self._dist_converged(res) or no_gain >= self.cfg.rho:
                    break
            else:
                self._dist_converged(res)   # keep distribution history populated

        res.recovered = list(self.recovered)
        res.metrics = self.tracker.summary()
        return res

    def _dist_converged(self, res: AttackResult) -> bool:
        return False     # only ADAM tracks a distribution; baselines never converge


class ADAM(Attack):
    """The full ADAM attack (Algorithm 1)."""

    name = "ADAM"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pool = AnchorPool(self.encoder, self.seed_topics, alpha=self.cfg.alpha)
        self.dist = DistributionEstimator(self.cfg, self.pool)
        self.used: set = set()
        self._prev_P: Optional[Dict[str, float]] = None

    def propose(self, t: int) -> MaliciousQuery:
        P = self.dist.snapshot()
        # line 6: weighted k-center anchor selection
        anchors_t = select_anchors(self.pool, P, self.cfg.k, self.used)

        # line 7: one candidate query per selected anchor (G_aux)
        candidates = []
        for a in anchors_t:
            text = self.gen.generate(a, self.domain)
            vec = self.encoder.encode([text])[0]
            topics = query_topics(vec, self.pool)
            candidates.append({"text": text, "probe": text, "vec": vec,
                               "topics": topics, "anchor": a})
        if not candidates:                      # safety fallback
            a = self.seed_topics[t % len(self.seed_topics)]
            text = self.gen.generate(a, self.domain)
            vec = self.encoder.encode([text])[0]
            candidates = [{"text": text, "probe": text, "vec": vec,
                           "topics": query_topics(vec, self.pool), "anchor": a}]

        # line 8: entropy-based selection of q_t
        best = select_query(candidates, P, self.cfg.eps)
        self._chosen_anchor = best["anchor"]
        self.dist.note_selected([best["anchor"]])
        self.used.add(best["anchor"])
        self._last_candidates = candidates
        return self._mq(best["probe"])

    def observe(self, response, queries: List[str], anchors: List[str]) -> None:
        # line 12: grow anchor pool with sufficiently-novel anchors
        self.pool.update(anchors)
        # line 13: update estimated distribution from anchors observed in r_t
        self._prev_P = self.dist.snapshot()
        self.dist.update(anchors)

    def _dist_converged(self, res: AttackResult) -> bool:
        # line 14-16: stop if ||P_t - P_{t-1}||_1 < eps_stop
        if self._prev_P is None:
            return False
        d = self.dist.l1_to(self._prev_P)
        res.dist_history.append(self.dist.snapshot())
        return d < self.cfg.eps_stop
