"""Reproduce the convergence analyses (Sec. 3 / Appendix H, K, O).

  (a) Distribution estimation approaching ground truth: ||P^_t - P(D)||_1
      decreases over rounds (Fig. 6 / Appendix K).
  (b) EQ vs. round increases and plateaus; the early-stop criterion (Eq. 2)
      fires near convergence (Fig. 7).
  (c) EM view: cumulative extraction (a proxy for the data log-likelihood the
      paper proves is monotonically non-decreasing, Appendix H).
  (d) Rate control (Appendix O, Table 15): EQ is largely retained as the ban
      rate rises, so rate limiting is not a robust defense.

Saves results/convergence.csv and results/fig6_convergence.png.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import (ATTACK_PROFILE, LLM_COMPLIANCE, RunSpec, run_single)
from adam.agent import LLMAgent
from adam.attack_adam import ADAM
from adam.config import AgentConfig, AttackConfig
from adam.defenses import rate_control
from adam.embeddings import cosine, get_encoder
from adam.llm import get_attacker_llm
from datasets.synthetic import build_memory, ground_truth_topics, sim_seeds

DOM = "ehragent"


def run_with_history(domain=DOM, seed=0, early_stop=False, T=30, mem_size=300):
    """Run ADAM, recording per-round the estimated *data* distribution.

    The estimated data distribution at round t is the empirical topic frequency
    of all records observed so far (the attacker infers a record's topic from its
    extracted keywords). By the law of large numbers this converges to the true
    memory distribution P(D) -- which is what Fig. 6 / Appendix K plot.
    """
    enc = get_encoder("hashing")
    mem = build_memory(domain, size=mem_size, encoder=enc, seed=seed)
    ac = AgentConfig(memory_size=mem_size)
    ac.base_compliance = LLM_COMPLIANCE["chatgpt-4"]
    agent = LLMAgent(mem, ac, seed=seed)
    inj, comp = ATTACK_PROFILE["ADAM"]
    cfg = AttackConfig(seed=seed, early_stop=early_stop, T=T)
    atk = ADAM(agent, get_attacker_llm(None, seed), enc, domain, cfg, inj, comp,
               sim_seeds(domain))

    gt = ground_truth_topics(domain)
    seen_topics = {t: 0 for t in gt}
    l1_curve = []
    orig_query = agent.query

    def wrapped(mq):
        r = orig_query(mq)
        for rec in r.retrieved:            # observed records -> empirical topic freq
            seen_topics[rec.topic] = seen_topics.get(rec.topic, 0) + 1
        total = sum(seen_topics.values()) or 1
        est = {t: c / total for t, c in seen_topics.items()}
        l1_curve.append(sum(abs(gt[t] - est.get(t, 0.0)) for t in gt))
        return r

    agent.query = wrapped
    res = atk.run()
    return atk, res, enc, l1_curve


def main():
    Path("results").mkdir(exist_ok=True)
    atk, res, enc, l1 = run_with_history()
    eq_curve = res.eq_curve

    # early-stop run on a smaller, exhaustible memory + larger budget so the
    # criterion (Eq. 2) actually fires once coverage plateaus.
    atk_es, res_es, _, _ = run_with_history(early_stop=True, T=60, mem_size=120)

    # rate control (Appendix O): ban rates ~ Table 15 concurrency levels
    ban_rates = [0.0, 0.10, 0.25, 0.38, 0.59]
    rc_eq = []
    for br in ban_rates:
        vals = []
        for sd in range(3):
            defs = [rate_control(br, seed=sd)] if br > 0 else None
            m = run_single(RunSpec(DOM, "ADAM", "chatgpt-4", sd), AgentConfig(),
                           AttackConfig(seed=sd), defenses=defs)
            vals.append(m["EQ"])
        rc_eq.append(round(sum(vals) / len(vals)))

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    axes[0].plot(range(1, len(l1) + 1), l1, "o-", color="tab:purple")
    axes[0].set_title("(a) ||P^_t - P(D)||_1 vs. round")
    axes[0].set_xlabel("round"); axes[0].set_ylabel("L1 gap to ground truth")

    axes[1].plot(range(1, len(eq_curve) + 1), eq_curve, "o-", color="tab:blue",
                 label="EQ (|M|=300, T=30)")
    axes[1].plot(range(1, len(res_es.eq_curve) + 1), res_es.eq_curve, "^-",
                 color="tab:green", label="EQ (|M|=120, early-stop)")
    axes[1].axvline(res_es.rounds_run, color="tab:red", ls="--",
                    label=f"early-stop @ round {res_es.rounds_run}")
    axes[1].set_title("(b) EQ vs. round (EM monotonic non-decreasing)")
    axes[1].set_xlabel("round"); axes[1].set_ylabel("cumulative EQ"); axes[1].legend(fontsize=8)

    axes[2].plot(ban_rates, rc_eq, "s-", color="tab:green")
    axes[2].set_title("(c) Rate control (Appendix O)")
    axes[2].set_xlabel("ban rate"); axes[2].set_ylabel("EQ"); axes[2].set_ylim(0, max(rc_eq) * 1.2 + 1)

    plt.tight_layout()
    plt.savefig("results/fig6_convergence.png", dpi=120)
    print("[saved] results/fig6_convergence.png")

    rows = ["metric,round_or_banrate,value"]
    for i, v in enumerate(l1, 1):
        rows.append(f"L1_gap,{i},{v:.3f}")
    for i, v in enumerate(eq_curve, 1):
        rows.append(f"EQ,{i},{v}")
    for br, v in zip(ban_rates, rc_eq):
        rows.append(f"rate_control_EQ,{br},{v}")
    Path("results/convergence.csv").write_text("\n".join(rows) + "\n")
    print("[saved] results/convergence.csv")
    print(f"L1 gap: round1={l1[0]:.2f} -> round{len(l1)}={l1[-1]:.2f}")
    print(f"EQ: round1={eq_curve[0]} -> final={eq_curve[-1]}; early-stop@{res_es.rounds_run} EQ={res_es.metrics['EQ']}")
    print(f"rate-control EQ vs ban_rate {ban_rates}: {rc_eq}")


if __name__ == "__main__":
    main()
