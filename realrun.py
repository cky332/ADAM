"""Run ADAM (and optional baselines) against a real LLM agent backed by
SiliconFlow's DeepSeek-V3.2-Exp (OpenAI-compatible API).

Usage
-----
  # set the key once (paste your sk-... in the env, don't commit it):
  export SILICONFLOW_API_KEY=sk-...

  # smoke test (1 attack x 1 domain x T=3 rounds; ~5 LLM calls)
  python realrun.py --smoke

  # default short demo (ADAM only, EHRAgent, T=5)
  python realrun.py

  # full real run (slow! ~hours on DeepSeek-V3.2-Exp)
  python realrun.py --attacks ADAM MEXTRA --domain ehragent --T 15 --memory 60

Responses are cached at .cache/siliconflow.json so reruns are free.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

# Configuration (lines 50-51 in the user spec) --------------------------------
MODEL = os.getenv("SF_MODEL", "Pro/deepseek-ai/DeepSeek-V3.2-Exp")
BASE_URL = "https://api.siliconflow.cn/v1"
# -----------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "experiments"))

from adam.attack_adam import ADAM
from adam.baselines import MEXTRA, Pirate, RAGThief, Vanilla
from adam.config import AgentConfig, AttackConfig
from adam.embeddings import get_encoder
from adam.llm import SiliconFlowLLM
from adam.real_agent import RealLLMAgent
from datasets.synthetic import build_memory, sim_seeds

ATTACK_CLASSES = {"Vanilla": Vanilla, "RAG-Thief": RAGThief, "Pirate": Pirate,
                  "MEXTRA": MEXTRA, "ADAM": ADAM}


def run_one(attack_name: str, domain: str, T: int, memory_size: int,
            llm: SiliconFlowLLM, encoder, seed: int = 0) -> dict:
    # Lower retrieval threshold + a generous pool: with a small memory, real
    # LLM-generated probes can still differ enough from stored queries that a
    # strict cosine cutoff would return nothing. Broader recall lets the attack
    # actually reach the records; the LLM (not the retriever) decides what
    # leaks.
    cfg_agent = AgentConfig(memory_size=memory_size, sim_threshold=0.10,
                            retrieval_pool_cap=memory_size)
    mem = build_memory(domain, size=memory_size, encoder=encoder, seed=seed)
    agent = RealLLMAgent(mem, cfg_agent, domain, llm=llm, seed=seed)

    cfg_atk = AttackConfig(T=T, seed=seed, eps=1e-12)
    seeds = sim_seeds(domain)
    cls = ATTACK_CLASSES[attack_name]
    # injection_strength / completeness are simulation knobs, unused by the
    # RealLLMAgent: it is the actual LLM that decides what to surface.
    atk = cls(agent, llm, encoder, domain, cfg_atk, 1.0, 1.0, seeds)
    # disable per-round paraphrase calls: they would triple LLM cost and add
    # little once the injection is already semantically aligned.
    atk.injections.paraphrase = False

    print(f"  -> {attack_name} / {domain} / T={T} / |M|={memory_size}", flush=True)
    print(f"     each round needs ~{cfg_atk.k + 1} LLM calls "
          f"(DeepSeek-V3.2-Exp is ~30-90s per call -- live progress below)\n",
          flush=True)
    t0 = time.time()
    res = atk.run()
    dt = time.time() - t0
    out = {**res.metrics, "rounds": res.rounds_run, "time_s": round(dt, 1),
           "calls": llm.calls, "cache_hits": llm.cached, "EQ_curve": res.eq_curve}
    print(f"\n     [done] {out}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--attacks", nargs="+", default=["ADAM"],
                    choices=list(ATTACK_CLASSES))
    ap.add_argument("--domain", default="ehragent",
                    choices=["ehragent", "react", "rap"])
    ap.add_argument("--T", type=int, default=5, help="rounds per attack")
    ap.add_argument("--memory", type=int, default=60, help="memory size")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true",
                    help="single round to verify connectivity")
    ap.add_argument("--diagnose", action="store_true",
                    help="run 3 trivial prompts against --model and report whether "
                         "the SiliconFlow endpoint is producing sane output")
    ap.add_argument("--model", default=MODEL,
                    help="SiliconFlow model id")
    ap.add_argument("--out", default="results/realrun.csv")
    args = ap.parse_args()

    if "SILICONFLOW_API_KEY" not in os.environ:
        print("set SILICONFLOW_API_KEY first."); sys.exit(2)

    if args.smoke:
        args.T = 3; args.memory = 30; args.attacks = ["ADAM"]

    print(f"model    : {args.model}")
    print(f"base_url : {BASE_URL}")

    if args.diagnose:
        from adam.llm import SiliconFlowLLM as _SL
        llm = _SL(model=args.model, seed=args.seed)
        tests = [("greeting", "Say hello in one sentence."),
                 ("math", "What is 2 + 2? Answer with one digit only."),
                 ("topic-query", "Produce one short user question about medication.")]
        bad = 0
        for name, prompt in tests:
            print(f"\n--- diagnose: {name} ---")
            out = llm._chat("You are a concise assistant.", prompt, max_tokens=120)
            print(f"  raw: {out[:160]!r}")
            if out.startswith("[error") or _SL._looks_degenerate(out):
                print(f"  >>> BAD ({'error' if out.startswith('[error') else 'degenerate'})")
                bad += 1
            else:
                print("  >>> OK")
        print(f"\n{len(tests) - bad}/{len(tests)} sane responses from {args.model}")
        if bad:
            print("This model's SiliconFlow endpoint is unhealthy / not enabled. Try:")
            print("  Qwen/Qwen2.5-72B-Instruct   (verified healthy)")
            print("  deepseek-ai/DeepSeek-V3     (verified healthy)")
            sys.exit(1)
        return

    print(f"settings : domain={args.domain} attacks={args.attacks} "
          f"T={args.T} |M|={args.memory} seed={args.seed}")

    llm = SiliconFlowLLM(model=args.model, seed=args.seed)
    encoder = get_encoder("hashing")

    rows = []
    for atk in args.attacks:
        m = run_one(atk, args.domain, args.T, args.memory, llm, encoder, args.seed)
        rows.append({"attack": atk, "domain": args.domain, **m})

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    import csv as _csv
    keys = ["attack", "domain", "EQ", "EE", "CER", "ASR", "rounds", "time_s",
            "calls", "cache_hits"]
    with out.open("w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    print(f"\n[saved] {out}")
    print(f"total LLM calls={llm.calls}  cache hits={llm.cached}")


if __name__ == "__main__":
    main()
