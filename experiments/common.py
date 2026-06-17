"""Shared harness + simulation calibration for all experiments.

The ADAM *algorithm* (anchor extraction, distribution estimation, k-center,
entropy) is parameter-free with respect to the numbers below. These constants
only calibrate the offline *victim simulation* so that the synthetic agents
reproduce the qualitative findings and approximate magnitudes of the paper's
Table 1. They live in one place so they are transparent and easy to change, and
they are ignored entirely when a real LLM backend drives the victim.

Grounding:
  * injection_strength -> how often the injection lands (drives EE / ASR). The
    paper notes query-optimization attacks (RAG-Thief, Pirate, ADAM) land far
    more often than static prompt injection (Vanilla); MEXTRA is in between.
  * completeness -> how often the agent dumps *all* k retrieved items
    (drives CER). ADAM's entropy-selected, workflow-aligned queries elicit the
    fullest dumps; RAG-Thief/Vanilla the least.
  * base_compliance -> larger / more capable LLMs leak more (Fig. 3b).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

# make the repo root importable when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adam.agent import LLMAgent
from adam.attack_adam import ADAM
from adam.baselines import Vanilla, RAGThief, Pirate, MEXTRA
from adam.config import AgentConfig, AttackConfig, SEED_TOPICS
from adam.embeddings import get_encoder
from adam.llm import get_attacker_llm
from datasets.synthetic import build_memory, ground_truth_topics, sim_seeds

# ---- per-LLM victim compliance (larger model -> leaks more, Fig. 3b) ----
LLM_COMPLIANCE: Dict[str, float] = {
    "llama2-7b-chat": 0.96,
    "mistral-7b-instruct": 0.97,
    "qwen2-72b": 0.985,
    "chatgpt-4": 0.995,
    # extra sizes used by the model-size ablation (Fig. 3b): larger models follow
    # the dump instruction more reliably, so they leak more completely.
    "llama-7b": 0.84, "llama-8b": 0.89, "llama-13b": 0.93,
    "llama-33b": 0.96, "llama-70b": 0.99,
}

# ---- per-attack (injection_strength, completeness) ----
ATTACK_PROFILE = {
    "Vanilla":   (0.33, 0.00),
    "RAG-Thief": (1.00, 0.02),
    "Pirate":    (1.00, 0.18),
    "MEXTRA":    (0.95, 0.46),
    "ADAM":      (1.00, 0.97),
}

ATTACK_CLASSES = {
    "Vanilla": Vanilla, "RAG-Thief": RAGThief, "Pirate": Pirate,
    "MEXTRA": MEXTRA, "ADAM": ADAM,
}

DOMAINS = ["ehragent", "react", "rap"]
DOMAIN_LABEL = {"ehragent": "EHRAgent", "react": "ReAct", "rap": "RAP"}


@dataclass
class RunSpec:
    domain: str
    attack: str
    llm: str = "llama2-7b-chat"
    seed: int = 0


def build_attack(spec: RunSpec, agent_cfg: AgentConfig, attack_cfg: AttackConfig,
                 encoder, defenses=None, attacker_llm_name: Optional[str] = None):
    memory = build_memory(spec.domain, size=agent_cfg.memory_size,
                          encoder=encoder, seed=spec.seed)
    agent_cfg.base_compliance = LLM_COMPLIANCE.get(spec.llm, 0.96)
    agent = LLMAgent(memory, agent_cfg, defenses=defenses, seed=spec.seed)
    gen = get_attacker_llm(attacker_llm_name, seed=spec.seed)
    inj, comp = ATTACK_PROFILE[spec.attack]
    cls = ATTACK_CLASSES[spec.attack]
    seeds = sim_seeds(spec.domain)
    return cls(agent, gen, encoder, spec.domain, attack_cfg, inj, comp, seeds), memory


def run_single(spec: RunSpec, agent_cfg: Optional[AgentConfig] = None,
               attack_cfg: Optional[AttackConfig] = None, encoder=None,
               defenses=None, encoder_name: str = "hashing",
               attacker_llm_name: Optional[str] = None) -> dict:
    agent_cfg = agent_cfg or AgentConfig()
    attack_cfg = attack_cfg or AttackConfig(seed=spec.seed)
    attack_cfg.seed = spec.seed
    encoder = encoder or get_encoder(encoder_name)
    attack, _ = build_attack(spec, agent_cfg, attack_cfg, encoder, defenses,
                             attacker_llm_name)
    res = attack.run()
    return {**res.metrics, "rounds": res.rounds_run, "eq_curve": res.eq_curve}
