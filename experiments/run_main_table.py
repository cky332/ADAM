"""Reproduce Table 1: five attacks x three agents x four LLMs (EQ/EE/CER/ASR).

Usage:
    python experiments/run_main_table.py [--encoder hashing|all-MiniLM-L6-v2]
                                         [--attacker-llm mock|gpt-4o-mini]
                                         [--out results/table1.csv]
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from common import (ATTACK_PROFILE, DOMAIN_LABEL, DOMAINS, LLM_COMPLIANCE,
                    RunSpec, run_single)
from adam.config import AgentConfig, AttackConfig

ATTACKS = ["Vanilla", "RAG-Thief", "Pirate", "MEXTRA", "ADAM"]
LLMS = ["llama2-7b-chat", "mistral-7b-instruct", "qwen2-72b", "chatgpt-4"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="hashing")
    ap.add_argument("--attacker-llm", default=None)
    ap.add_argument("--out", default="results/table1.csv")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = []
    header = f"{'Attack':<11}{'Model':<22}" + "".join(
        f"{DOMAIN_LABEL[d]+' '+m:<14}" for d in DOMAINS for m in ("EQ", "EE", "CER", "ASR")
    )
    print(header)
    print("-" * len(header))

    for attack in ATTACKS:
        for llm in LLMS:
            cells = []
            csv_row = {"attack": attack, "model": llm}
            for d in DOMAINS:
                spec = RunSpec(domain=d, attack=attack, llm=llm, seed=args.seed)
                m = run_single(spec, AgentConfig(), AttackConfig(seed=args.seed),
                               encoder_name=args.encoder,
                               attacker_llm_name=args.attacker_llm)
                for key in ("EQ", "EE", "CER", "ASR"):
                    cells.append(f"{m[key]:<14}")
                    csv_row[f"{d}_{key}"] = m[key]
            print(f"{attack:<11}{llm:<22}" + "".join(cells))
            rows.append(csv_row)
        print()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
