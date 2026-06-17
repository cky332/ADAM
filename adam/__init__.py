"""ADAM: A Systematic Data Extraction Attack on Agent Memory via Adaptive Querying.

Faithful reproduction of arXiv:2604.09747v1. See README.md for what is exactly
reproduced (algorithm, metrics, trends) vs. what requires the original gated
resources (real LLMs / MIMIC-III for the exact Table-1 numbers).
"""
from .config import AttackConfig, AgentConfig, SEED_TOPICS
from .attack_adam import ADAM, Attack, AttackResult
from .baselines import Vanilla, RAGThief, Pirate, MEXTRA, REGISTRY as ATTACKS
from .metrics import MetricTracker

__all__ = [
    "AttackConfig", "AgentConfig", "SEED_TOPICS",
    "ADAM", "Attack", "AttackResult",
    "Vanilla", "RAGThief", "Pirate", "MEXTRA", "ATTACKS",
    "MetricTracker",
]
