"""Reproduce Figure 5: ADAM vs. four defenses (Sec. 5).

For each defense we measure EQ of ADAM (Ours) and the strongest baseline
(MEXTRA) with and without the defense on EHRAgent. The paper's finding: surface
-level defenses hurt the static / marker-based attack (MEXTRA) far more than
ADAM, whose semantically-malicious, paraphrased queries survive.

  (a) query rewriting     -- very slight drop for both
  (b) auxiliary filtering -- blocks MEXTRA's raw markers; ADAM barely affected
  (c) RA-LLM              -- MEXTRA fragile to token dropping; ADAM slight drop
  (d) erase-and-check     -- catches MEXTRA suffixes; ADAM slight drop

Saves results/defenses.csv and results/fig5_defenses.png.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import RunSpec, run_single
from adam import defenses as D
from adam.config import AgentConfig, AttackConfig

DEFENSES = [
    ("(a) Query rewriting", "query_rewriting", D.query_rewriting),
    ("(b) Auxiliary filtering", "auxiliary_filtering", D.auxiliary_filtering),
    ("(c) RA-LLM", "ra_llm", D.ra_llm),
    ("(d) Erase-and-check", "erase_and_check", D.erase_and_check),
]
ATTACKS = ["ADAM", "MEXTRA"]
DOM, LLM, N_SEEDS = "ehragent", "chatgpt-4", 3


def eq_avg(attack, defense_fn):
    vals = []
    for sd in range(N_SEEDS):
        defs = [defense_fn] if defense_fn else None
        m = run_single(RunSpec(DOM, attack, LLM, sd), AgentConfig(),
                       AttackConfig(seed=sd), defenses=defs)
        vals.append(m["EQ"])
    return round(sum(vals) / len(vals))


def main():
    Path("results").mkdir(exist_ok=True)
    base = {a: eq_avg(a, None) for a in ATTACKS}          # no-defense EQ
    rows = ["defense,attack,EQ_nodef,EQ_def,retained%"]
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))

    for ax, (title, key, fn) in zip(axes, DEFENSES):
        eq_def = {a: eq_avg(a, fn) for a in ATTACKS}
        x = np.arange(len(ATTACKS))
        ax.bar(x - 0.2, [base[a] for a in ATTACKS], 0.4, label="No defense",
               color=["tab:blue", "tab:orange"])
        ax.bar(x + 0.2, [eq_def[a] for a in ATTACKS], 0.4, label="With defense",
               color=["tab:blue", "tab:orange"], alpha=0.45)
        ax.set_xticks(x); ax.set_xticklabels(ATTACKS)
        ax.set_title(title); ax.set_ylabel("EQ"); ax.legend(fontsize=8)
        for a in ATTACKS:
            ret = 100.0 * eq_def[a] / base[a] if base[a] else 0.0
            rows.append(f"{key},{a},{base[a]},{eq_def[a]},{ret:.0f}")

    plt.tight_layout()
    plt.savefig("results/fig5_defenses.png", dpi=120)
    print("[saved] results/fig5_defenses.png")
    Path("results/defenses.csv").write_text("\n".join(rows) + "\n")
    print("[saved] results/defenses.csv")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
