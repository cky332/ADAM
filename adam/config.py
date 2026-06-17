"""Central configuration for the ADAM attack reproduction.

All default hyper-parameters follow the paper
"ADAM: A Systematic Data Extraction Attack on Agent Memory via Adaptive
Querying" (arXiv:2604.09747v1). Section/appendix references are given inline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class AttackConfig:
    """Hyper-parameters of the ADAM attack (Algorithm 1 + Appendix H)."""

    # ---- core loop ----
    T: int = 30                 # query budget (paper: "30 prompts", Sec. 4.1)
    k: int = 3                  # top-k retrieved chunks / k-center anchors (Sec. 4.1)

    # ---- anchor extraction ----
    alpha: float = 0.5          # novelty threshold for adding an anchor (Sec. 3).
                                # Paper recommends alpha in [0.4, 0.6] (Appendix H).

    # ---- distribution estimation ----
    lam: float = 0.9            # decay coefficient lambda for repeatedly-used anchors
                                # (Sec. 3, "we use lambda = 0.9", Appendix H).
    tau: float = 1.0            # softmax temperature (default 1.0, Appendix H).

    # ---- entropy selection ----
    eps: float = 1e-12          # numerical-stability constant in entropy (Sec. 3).
    n_candidates: int = 3       # candidate queries generated per anchor before the
                                # entropy selector picks one (Appendix M: "three queries").

    # ---- early stop (Eq. 2) ----
    # Off by default: the main table allocates a fixed budget of T=30 prompts to
    # every attack (Sec. 4.1). Turn on for the convergence study (Appendix K).
    early_stop: bool = False
    eps_stop: float = 1e-3      # ||P_t - P_{t-1}||_1 < eps_stop
    eta: float = 1.0            # Delta-EQ threshold
    rho: int = 5                # patience: Delta-EQ < eta for rho consecutive rounds

    # ---- clustering for distribution estimation ----
    cluster_method: str = "dbscan"   # one of: dbscan, kmeans, gmm, kde (Appendix I)
    dbscan_eps: float = 0.5
    dbscan_min_samples: int = 2

    # ---- misc ----
    seed: int = 0

    def validate(self) -> None:
        assert 0.0 < self.alpha < 1.0, "alpha must be in (0,1)"
        assert 0.0 < self.lam < 1.0, "lambda must be in (0,1)"
        assert self.tau > 0.0, "tau must be > 0"
        assert self.k >= 1 and self.T >= 1


@dataclass
class AgentConfig:
    """Configuration of the victim LLM agent (Sec. 2.1 / 4.1)."""

    memory_size: int = 300          # |M| (Sec. 4.1: "memory size of 300")
    top_k: int = 3                  # retrieval depth
    sim_threshold: float = 0.25     # retriever cosine-similarity cutoff (Fig. 3c)
    scoring: str = "cosine"         # cosine | dot | l2 (Fig. 3h)
    retrieval_pool_cap: int = 80    # the agent forms a candidate pool of all records
                                    # above threshold (up to this cap) and samples k
                                    # from it, so repeated probes reach a topic's tail
                                    # rather than re-hitting the same most-central few

    # LLM "compliance" -- how susceptible the agent core is to leaking retrieved
    # records once an injection lands. Larger / more capable models leak more
    # (Fig. 3b). Used only by the offline simulation backend.
    base_compliance: float = 0.5


# Seed topics per agent, ten anchors each (Appendix N, Table 13).
SEED_TOPICS = {
    "ehragent": [
        "appointment", "diagnosis", "insurance", "medicine", "prescription",
        "patient", "password", "doctor", "cancer", "medication",
    ],
    "react": [
        "history", "geography", "science", "biography", "politics",
        "literature", "sports", "culture", "discovery", "invention",
    ],
    "rap": [
        "product", "price", "discount", "shipping", "cart",
        "return", "review", "rating", "payment", "warranty",
    ],
}
