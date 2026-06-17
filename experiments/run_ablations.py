"""Reproduce Figure 3: ablation studies on ADAM (EHRAgent, Llama-2-7b-chat).

  (a) top-k retrieved chunks      k in {1,3,5,7,9}            EQ,EE increase
  (b) model size                  {7B,8B,13B,33B,70B}        EQ,EE increase
  (c) retriever similarity thr.   {0.1,0.3,0.5,0.7,0.9}      EQ,EE decrease
  (d) #anchors in k-center        k in {1,3,5,7,9}            EQ,EE increase (modest)
  (e) memory size                 {100,...,500}              EQ,EE increase
  (f) domain knowledge            domain-aware vs agnostic    aware > agnostic
  (g) embedding model             hashing / MiniLM / ...      minor differences
  (h) scoring function            cosine / dot / l2           minor differences

Saves results/ablations.csv and results/fig3_ablations.png.
"""
from __future__ import annotations

import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import RunSpec, run_single
from adam.config import AgentConfig, AttackConfig

DOM, LLM = "ehragent", "llama2-7b-chat"
N_SEEDS = 3                      # average each point over this many random seeds


def _run_once(agent_cfg, attack_cfg, encoder_name, llm, seeds, sd):
    from common import build_attack, LLM_COMPLIANCE, ATTACK_PROFILE
    from adam.embeddings import get_encoder
    from adam.attack_adam import ADAM
    from adam.agent import LLMAgent
    from adam.llm import get_attacker_llm
    from datasets.synthetic import build_memory, sim_seeds
    import copy
    enc = get_encoder(encoder_name)
    ac = copy.copy(agent_cfg)
    atk_cfg = copy.copy(attack_cfg); atk_cfg.seed = sd
    mem = build_memory(DOM, size=ac.memory_size, encoder=enc, seed=sd)
    ac.base_compliance = LLM_COMPLIANCE.get(llm, 0.96)
    agent = LLMAgent(mem, ac, seed=sd)
    inj, comp = ATTACK_PROFILE["ADAM"]
    use_seeds = seeds if seeds is not None else sim_seeds(DOM)
    atk = ADAM(agent, get_attacker_llm(None, sd), enc, DOM, atk_cfg, inj, comp, use_seeds)
    res = atk.run()
    return res.metrics["EQ"], res.metrics["EE"]


def _run(agent_cfg, attack_cfg, encoder_name="hashing", llm=LLM, seeds=None):
    eqs, ees = [], []
    for sd in range(N_SEEDS):
        a, e = _run_once(agent_cfg, attack_cfg, encoder_name, llm, seeds, sd)
        eqs.append(a); ees.append(e)
    return round(sum(eqs) / len(eqs)), round(sum(ees) / len(ees), 2)


def ablation_topk():
    xs, eq, ee = [1, 3, 5, 7, 9], [], []
    for k in xs:
        a, e = _run(AgentConfig(top_k=k), AttackConfig(k=k, seed=0))
        eq.append(a); ee.append(e)
    return xs, eq, ee


def ablation_model_size():
    sizes = ["llama-7b", "llama-8b", "llama-13b", "llama-33b", "llama-70b"]
    eq, ee = [], []
    for s in sizes:
        a, e = _run(AgentConfig(), AttackConfig(seed=0), llm=s)
        eq.append(a); ee.append(e)
    return ["7B", "8B", "13B", "33B", "70B"], eq, ee


def ablation_threshold():
    xs, eq, ee = [0.1, 0.3, 0.5, 0.7, 0.9], [], []
    for thr in xs:
        a, e = _run(AgentConfig(sim_threshold=thr), AttackConfig(seed=0))
        eq.append(a); ee.append(e)
    return xs, eq, ee


def ablation_n_anchors():
    xs, eq, ee = [1, 3, 5, 7, 9], [], []
    for k in xs:
        a, e = _run(AgentConfig(top_k=3), AttackConfig(k=k, seed=0))   # only k-center size varies
        eq.append(a); ee.append(e)
    return xs, eq, ee


def ablation_memory_size():
    xs, eq, ee = [100, 200, 300, 400, 500], [], []
    for m in xs:
        a, e = _run(AgentConfig(memory_size=m), AttackConfig(seed=0))
        eq.append(a); ee.append(e)
    return xs, eq, ee


def ablation_domain_knowledge():
    from datasets.synthetic import sim_seeds
    aware = sim_seeds(DOM)
    rng = random.Random(0)
    # domain-agnostic: random out-of-domain words (Appendix D)
    ood = ["crusty", "flim", "dumbish", "boring", "sleep", "dryness"]
    eq_a, _ = _run(AgentConfig(), AttackConfig(seed=0), seeds=aware)
    eq_o, _ = _run(AgentConfig(), AttackConfig(seed=0), seeds=ood)
    return eq_a, eq_o


def ablation_embedding():
    # Try the real Sentence-Transformer encoders the paper uses; transparently
    # fall back to hashing-dimension variants when the optional dep is absent so
    # the robustness-to-encoder point (Fig. 3g) is still demonstrated.
    eqs = {}
    candidates = ["all-MiniLM-L6-v2", "e5-large-v2", "gte-large-en-v1.5"]
    try:
        import sentence_transformers  # noqa
        names = candidates
    except Exception:
        names = ["hashing-128", "hashing-256", "hashing-512"]
    for name in names:
        a, _ = _run(AgentConfig(), AttackConfig(seed=0), encoder_name=name)
        eqs[name] = a
    return eqs


def ablation_scoring():
    eqs = {}
    for s in ["cosine", "dot", "l2"]:
        a, _ = _run(AgentConfig(scoring=s), AttackConfig(seed=0))
        eqs[s] = a
    return eqs


def main():
    Path("results").mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))

    def twin_plot(ax, xs, eq, ee, title, xlabel):
        ax.plot(range(len(xs)), eq, "o-", color="tab:blue", label="EQ")
        ax.set_xticks(range(len(xs))); ax.set_xticklabels(xs)
        ax.set_ylabel("EQ", color="tab:blue"); ax.set_ylim(0, max(eq) * 1.3 + 1)
        ax.set_xlabel(xlabel); ax.set_title(title)
        ax2 = ax.twinx()
        ax2.plot(range(len(xs)), ee, "s--", color="tab:red", label="EE")
        ax2.set_ylabel("EE", color="tab:red"); ax2.set_ylim(0, 1)

    rows = []
    xs, eq, ee = ablation_topk(); twin_plot(axes[0, 0], xs, eq, ee, "(a) Top-k selection", "k")
    rows.append(("topk", xs, eq, ee))
    xs, eq, ee = ablation_model_size(); twin_plot(axes[0, 1], xs, eq, ee, "(b) Model size", "size")
    rows.append(("model_size", xs, eq, ee))
    xs, eq, ee = ablation_threshold(); twin_plot(axes[0, 2], xs, eq, ee, "(c) Similarity threshold", "thr")
    rows.append(("threshold", xs, eq, ee))
    xs, eq, ee = ablation_n_anchors(); twin_plot(axes[0, 3], xs, eq, ee, "(d) Number of anchors", "k")
    rows.append(("n_anchors", xs, eq, ee))
    xs, eq, ee = ablation_memory_size(); twin_plot(axes[1, 0], xs, eq, ee, "(e) Memory size", "|M|")
    rows.append(("memory_size", xs, eq, ee))

    eq_a, eq_o = ablation_domain_knowledge()
    axes[1, 1].bar(["domain", "w/o domain"], [eq_a, eq_o], color=["tab:blue", "tab:gray"])
    axes[1, 1].set_title("(f) Domain knowledge"); axes[1, 1].set_ylabel("EQ")
    rows.append(("domain", ["aware", "agnostic"], [eq_a, eq_o], []))

    emb = ablation_embedding()
    axes[1, 2].bar(list(emb.keys()), list(emb.values()), color="tab:green")
    axes[1, 2].set_title("(g) Embedding model"); axes[1, 2].set_ylabel("EQ")
    axes[1, 2].tick_params(axis="x", rotation=20)
    rows.append(("embedding", list(emb.keys()), list(emb.values()), []))

    sc = ablation_scoring()
    axes[1, 3].bar(list(sc.keys()), list(sc.values()), color="tab:purple")
    axes[1, 3].set_title("(h) Scoring function"); axes[1, 3].set_ylabel("EQ")
    rows.append(("scoring", list(sc.keys()), list(sc.values()), []))

    plt.tight_layout()
    out = "results/fig3_ablations.png"
    plt.savefig(out, dpi=120); print(f"[saved] {out}")

    with open("results/ablations.csv", "w") as f:
        f.write("ablation,x,EQ,EE\n")
        for name, xs, eq, ee in rows:
            for i, x in enumerate(xs):
                eev = ee[i] if ee else ""
                f.write(f"{name},{x},{eq[i]},{eev}\n")
    print("[saved] results/ablations.csv")

    # console summary
    for name, xs, eq, ee in rows:
        print(f"{name:14s} x={xs} EQ={eq}")


if __name__ == "__main__":
    main()
