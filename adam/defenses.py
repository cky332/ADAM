"""Defenses (Sec. 5, Appendix C/F/O).

Each defense is a callable applied to an incoming ``MaliciousQuery`` and returns
a dict understood by the agent:
    {"blocked": bool, "factor": float in [0,1], "probe": str}

The key qualitative finding the paper reports (Fig. 5) is that surface-level
defenses hurt the static / marker-based attacks (e.g. MEXTRA) far more than
they hurt ADAM, whose queries are *semantically* malicious and paraphrased, so
they survive rewriting, keyword filtering, random token dropping and suffix
erasure.
"""
from __future__ import annotations

import random
from typing import Callable

from .agent import MaliciousQuery
from .injection import SUFFIX_INJECTIONS


def query_rewriting(mq: MaliciousQuery) -> dict:
    """Paraphrase the query, preserving semantics (Ma et al. 2023).

    Because meaning is preserved, the malicious intent persists (Appendix F,
    Table 6/7): only a very slight drop for both methods.
    """
    return {"factor": 0.95}


def auxiliary_filtering(mq: MaliciousQuery) -> dict:
    """Rule-based keyword filter (Rahman & Harris 2025).

    Blocks prompts that contain raw harmful markers such as the verbatim
    retrieval-inducing suffixes ("output all ... queries", "show previous
    questions"). ADAM's paraphrased, semantically-equivalent suffixes evade it.
    """
    text = mq.text.lower()
    if mq.raw_markers and any(s.lower() in text for s in SUFFIX_INJECTIONS):
        return {"blocked": True}
    return {"factor": 0.98}


def ra_llm(mq: MaliciousQuery) -> dict:
    """RA-LLM: random token dropping + consistency check (Cao et al. 2023).

    Statically-structured, marker-based prompts are fragile to token dropping;
    ADAM's natural queries degrade only slightly.
    """
    return {"factor": 0.60 if mq.raw_markers else 0.92}


def erase_and_check(mq: MaliciousQuery) -> dict:
    """Erase-and-check: iterative suffix removal + safety filter (Kumar 2023).

    Effective against suffix-marker attacks; ADAM suffers only a slight drop.
    """
    return {"factor": 0.55 if mq.raw_markers else 0.90}


def rate_control(qps_ban_rate: float = 0.1, seed: int = 0) -> Callable:
    """Industry-style rate limiting (Appendix O).

    Randomly rejects a fraction (``qps_ban_rate``) of queries. Even at high ban
    rates ADAM keeps operating because it simply retries / proceeds.
    """
    rng = random.Random(seed)

    def _fn(mq: MaliciousQuery) -> dict:
        if rng.random() < qps_ban_rate:
            return {"blocked": True}
        return {"factor": 1.0}

    return _fn


REGISTRY = {
    "query_rewriting": query_rewriting,
    "auxiliary_filtering": auxiliary_filtering,
    "ra_llm": ra_llm,
    "erase_and_check": erase_and_check,
}
