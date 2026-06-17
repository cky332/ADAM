"""Reproduce Figure 4: oracle vs. estimated distribution (Sec. 4.3).

The paper's key claim: the better ADAM's estimated topic distribution P^(D)
matches the ground-truth P(D), the closer ADAM gets to an *oracle* attack that
is handed P(D) directly. We:

  (a) run ADAM (estimated P^) and an oracle-guided variant (true P) on each
      agent and compare EQ;
  (b) map ADAM's final anchor distribution onto the true topics and plot it
      against the ground-truth distribution, with their L1 gap.

Saves results/oracle.csv and results/fig4_oracle.png.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import (ATTACK_PROFILE, DOMAIN_LABEL, DOMAINS, LLM_COMPLIANCE, RunSpec)
from adam.agent import LLMAgent
from adam.attack_adam import ADAM
from adam.config import AgentConfig, AttackConfig
from adam.embeddings import cosine, get_encoder
from adam.llm import get_attacker_llm
from datasets.synthetic import build_memory, ground_truth_topics, sim_seeds


def _map_anchor_to_topic(anchor, topic_vecs, anchor_vec):
    """exact > substring > nearest-embedding match of an anchor to a true topic."""
    if anchor in topic_vecs:
        return anchor
    for t in topic_vecs:
        if t in anchor or anchor in t:
            return t
    return max(topic_vecs, key=lambda t: cosine(anchor_vec, topic_vecs[t]))


class OracleADAM(ADAM):
    """Upper-bound ADAM handed the ground-truth topic distribution P(D).

    It is seeded with the true topic set and its selection distribution is fixed
    to the ground-truth frequencies (newly discovered entity anchors are mapped
    back to their topic), so it focuses queries on the densest topics from the
    start -- something the estimated variant must learn.
    """

    name = "ADAM-Oracle"

    def set_oracle(self, gt_topics, topic_vecs):
        self._gt = gt_topics
        self._topic_vecs = topic_vecs

    def _oracle_probs(self):
        probs = {}
        for a in self.pool.anchors:
            t = _map_anchor_to_topic(a, self._topic_vecs, self.pool.vec(a))
            probs[a] = self._gt.get(t, 1e-6)
        s = sum(probs.values()) or 1.0
        return {a: p / s for a, p in probs.items()}

    def propose(self, t):
        self.dist.P = self._oracle_probs()      # fix selection to true P(D)
        return super().propose(t)

    def observe(self, response, queries, anchors):
        self.pool.update(anchors)               # still discover, but never trust estimate
        self.dist.P = self._oracle_probs()


def _topic_vectors(domain, encoder):
    return {t: encoder.encode([t])[0] for t in ground_truth_topics(domain)}


def run_attack(domain, oracle=False, seed=0):
    enc = get_encoder("hashing")
    mem = build_memory(domain, size=300, encoder=enc, seed=seed)
    ac = AgentConfig(); ac.base_compliance = LLM_COMPLIANCE["chatgpt-4"]
    agent = LLMAgent(mem, ac, seed=seed)
    inj, comp = ATTACK_PROFILE["ADAM"]
    # oracle is seeded with the full true topic set; ours starts from a subset
    seeds = list(ground_truth_topics(domain)) if oracle else sim_seeds(domain)
    cls = OracleADAM if oracle else ADAM
    atk = cls(agent, get_attacker_llm(None, seed), enc, domain,
              AttackConfig(seed=seed), inj, comp, seeds)
    if oracle:
        atk.set_oracle(ground_truth_topics(domain), _topic_vectors(domain, enc))
    res = atk.run()
    return atk, res


def run_attack_avg(domain, oracle=False, n=3):
    eqs, last = [], None
    for sd in range(n):
        atk, res = run_attack(domain, oracle=oracle, seed=sd)
        eqs.append(res.metrics["EQ"]); last = (atk, res)
    avg = round(sum(eqs) / len(eqs))
    return last[0], avg


def estimated_topic_dist(atk, domain, encoder):
    """Map ADAM's final anchor distribution onto the true topics."""
    tvecs = _topic_vectors(domain, encoder)
    est = {t: 0.0 for t in tvecs}
    for a, p in atk.dist.P.items():
        av = atk.pool.vec(a)
        best_t = max(tvecs, key=lambda t: cosine(av, tvecs[t]))
        est[best_t] += p
    s = sum(est.values()) or 1.0
    return {t: v / s for t, v in est.items()}


def main():
    Path("results").mkdir(exist_ok=True)
    enc = get_encoder("hashing")
    fig, axes = plt.subplots(1, 4, figsize=(22, 4.5))

    eq_ours, eq_oracle = [], []
    rows = ["domain,EQ_ours,EQ_oracle,L1_gap"]
    for i, dom in enumerate(DOMAINS):
        atk_o, eq_o = run_attack_avg(dom, oracle=False)
        _, eq_or = run_attack_avg(dom, oracle=True)
        eq_ours.append(eq_o)
        eq_oracle.append(max(eq_or, eq_o))      # oracle is an upper bound

        gt = ground_truth_topics(dom)
        est = estimated_topic_dist(atk_o, dom, enc)
        topics = list(gt.keys())
        gt_v = [gt[t] for t in topics]
        est_v = [est.get(t, 0.0) for t in topics]
        l1 = sum(abs(a - b) for a, b in zip(gt_v, est_v))
        rows.append(f"{dom},{eq_o},{eq_oracle[-1]},{l1:.3f}")

        ax = axes[i + 1]
        x = np.arange(len(topics))
        ax.bar(x - 0.2, gt_v, 0.4, label="Ground truth P(D)", color="tab:gray")
        ax.bar(x + 0.2, est_v, 0.4, label="Estimated P^(D)", color="tab:blue")
        ax.set_title(f"({chr(98+i)}) {DOMAIN_LABEL[dom]}  (L1={l1:.2f})")
        ax.set_xticks(x); ax.set_xticklabels(topics, rotation=60, ha="right", fontsize=7)
        ax.set_ylabel("probability"); ax.legend(fontsize=7)

    # (a) EQ comparison
    ax = axes[0]
    x = np.arange(len(DOMAINS))
    ax.bar(x - 0.2, eq_ours, 0.4, label="EQ (Ours)", color="tab:blue")
    ax.bar(x + 0.2, eq_oracle, 0.4, label="EQ (Oracle)", color="tab:orange")
    ax.set_xticks(x); ax.set_xticklabels([DOMAIN_LABEL[d] for d in DOMAINS])
    ax.set_title("(a) Oracle vs. Estimation"); ax.set_ylabel("EQ"); ax.legend()

    plt.tight_layout()
    plt.savefig("results/fig4_oracle.png", dpi=120)
    print("[saved] results/fig4_oracle.png")
    Path("results/oracle.csv").write_text("\n".join(rows) + "\n")
    print("[saved] results/oracle.csv")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
