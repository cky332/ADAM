"""Baseline attacks (Sec. 4.1 + Appendix).

For a fair comparison RAG-Thief and Pirate use the *same* prefix/suffix
injection as ADAM; only their query-generation strategy differs:

  * Vanilla   -- static prompt injection over the seed topics
                 (Zeng et al. 2024; Qi et al. 2024). No adaptation.
  * RAG-Thief -- greedily follows the most recent extracted anchor
                 (Jiang et al. 2024). Adaptive, but no distribution estimation.
  * Pirate    -- adaptively explores the anchor frontier with random,
                 non-distribution-aware selection (Di Maio et al. 2024).
  * MEXTRA    -- workflow-aligned injection with a *static* crafted prompt set
                 (Wang et al. 2025). Strong elicitation, limited topic coverage.
"""
from __future__ import annotations

import random
from typing import List

from .agent import MaliciousQuery
from .anchors import AnchorPool, extract_anchors
from .attack_adam import Attack


class Vanilla(Attack):
    name = "Vanilla"
    paraphrase = False        # static, surface-level prompt injection

    def propose(self, t: int) -> MaliciousQuery:
        topic = self.seed_topics[(t - 1) % len(self.seed_topics)]
        probe = self.gen.generate(topic, self.domain)
        return self._mq(probe, retrieval_hint=topic)


class RAGThief(Attack):
    name = "RAG-Thief"

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.frontier: List[str] = list(self.seed_topics)
        self.rng = random.Random(self.cfg.seed)

    def propose(self, t: int) -> MaliciousQuery:
        # greedily follow the most-recently-discovered anchor
        topic = self.frontier[-1] if self.frontier else self.seed_topics[0]
        probe = self.gen.generate(topic, self.domain)
        return self._mq(probe, retrieval_hint=topic)

    def observe(self, response, queries, anchors):
        for a in anchors:
            if a not in self.frontier:
                self.frontier.append(a)


class Pirate(Attack):
    name = "Pirate"

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.frontier: List[str] = list(self.seed_topics)
        self.used: set = set()
        self.rng = random.Random(self.cfg.seed)

    def propose(self, t: int) -> MaliciousQuery:
        unused = [a for a in self.frontier if a not in self.used]
        topic = self.rng.choice(unused) if unused else self.rng.choice(self.frontier)
        self.used.add(topic)
        probe = self.gen.generate(topic, self.domain)
        return self._mq(probe, retrieval_hint=topic)

    def observe(self, response, queries, anchors):
        for a in anchors:
            if a not in self.frontier:
                self.frontier.append(a)


class MEXTRA(Attack):
    name = "MEXTRA"
    paraphrase = False        # workflow-aligned but static, marker-based prompts

    def propose(self, t: int) -> MaliciousQuery:
        # cycles the *fixed* seed-topic set with workflow-aligned prompts. Unlike
        # ADAM it never discovers new topics from responses, so its coverage is
        # capped by the (possibly mismatched) seed topics -- the gap ADAM closes.
        topic = self.seed_topics[(t - 1) % len(self.seed_topics)]
        probe = self.gen.generate(topic, self.domain)
        return self._mq(probe, retrieval_hint=topic)


REGISTRY = {
    "vanilla": Vanilla,
    "rag-thief": RAGThief,
    "pirate": Pirate,
    "mextra": MEXTRA,
}
