# ADAM — Reproduction

A faithful, runnable reproduction of

> **ADAM: A Systematic Data Extraction Attack on Agent Memory via Adaptive Querying**
> Lyu, He, Wang, Hu, Li, Chen, Li, Chen — arXiv:2604.09747v1 (2026)

ADAM is a black-box privacy attack that extracts the **private user queries
stored in an LLM agent's memory**. Its novelty over prior attacks (Vanilla,
RAG-Thief, Pirate, MEXTRA) is **data-distribution estimation** combined with an
**entropy-guided, k-center active-learning** query strategy, which lets it cover
the agent's memory far more efficiently and achieve much higher attack success.

This repo implements the **complete method (Algorithm 1)**, the **four baselines**,
the **four metrics**, the **four defenses**, and a harness that reproduces the
paper's **Table 1** and **Figures 3–6** end-to-end.

> ⚠️ **Ethics / intended use.** This is a *defensive* security research artifact
> reproducing a published academic paper, intended to help build privacy-preserving
> agent memory. It runs **only on synthetic data** — no real patient records or
> personal data. (MIMIC-III is credentialed data, and running a privacy attack on
> real patient records would itself be a privacy violation.) The repo also
> implements the paper's defenses. Do not point the real-LLM backends at systems
> you are not authorized to test.

---

## What is reproduced, and what needs gated resources

A *literal* reproduction requires Qwen2-72B / ChatGPT-4, the credentialed
**MIMIC-III** corpus (EHRAgent), **HotpotQA** (ReAct) and **WebShop** (RAP) — none
available in a sandbox. So the reproduction is split cleanly:

| Component | Status |
|---|---|
| ADAM algorithm: anchor extraction, distribution estimation (DBSCAN/KDE/GMM/k-means), weighted k-center, entropy selection, refine, early-stop (Eq. 2) | **Exact** implementation of the paper's equations |
| Baselines (Vanilla, RAG-Thief, Pirate, MEXTRA) | **Implemented** |
| Metrics (EQ, EE, CER, ASR) | **Exact** definitions (Sec. 4.1) |
| Defenses (query rewriting, auxiliary filtering, RA-LLM, erase-and-check, rate control) | **Implemented** |
| Injection templates, seed topics, EM convergence view | From the paper (Tables 13–14, Appendix H) |
| Victim LLM behavior + datasets | **Offline simulation** (synthetic agents with known latent topic distributions), **calibrated** to reproduce the paper's *trends, orderings and approximate magnitudes* |
| Exact Table-1 numbers | Require the real LLMs + gated datasets — plug them in via the optional backends |

**Pluggable real backends.** The same attack code runs against real resources:
set `OPENAI_API_KEY` and pass `--attacker-llm gpt-4o-mini`, and/or
`pip install sentence-transformers` and pass `--encoder all-MiniLM-L6-v2`. The
embedding/LLM/agent interfaces are identical; only the backend changes.

### Why a calibrated simulation is faithful

EQ is governed by **coverage** of the memory's topic distribution; ASR/CER by how
reliably the injection **lands** and elicits a **full dump**. The simulation makes
both first-class:

- The victim agent retrieves real top-k records (cosine over a sentence encoder)
  from a synthetic memory whose records are drawn from a **known, non-uniform
  (Zipfian) latent topic distribution** `P(D)` — this is the ground truth the
  oracle test (Fig. 4) compares against.
- Each attack carries an `(injection_strength, completeness)` profile encoding its
  *qualitative* strengths as reported in the paper (query-optimization attacks land
  more often than static injection; ADAM elicits the fullest dumps). These live in
  one transparent, tunable table (`experiments/common.py`) and are **ignored** when
  a real LLM drives the victim.
- **Coverage emerges from the actual algorithm**: ADAM's k-center + entropy +
  distribution decay genuinely probe more distinct topics than the baselines, so
  its higher EQ is produced by the real method, not hard-coded.

---

## Results (calibrated simulation)

### Table 1 — main results (excerpt: ChatGPT-4 victim)

| Attack | EHRAgent EQ/EE/CER/ASR | ReAct | RAP |
|---|---|---|---|
| Vanilla   | 7 / .08 / .00 / .23  | 2 / .02 / .00 / .07  | 7 / .08 / .00 / .23 |
| RAG-Thief | 27 / .30 / .00 / .90 | 20 / .22 / .23 / .57 | 21 / .23 / .00 / .70 |
| Pirate    | 36 / .40 / .17 / .97 | 23 / .26 / .20 / .70 | 24 / .27 / .03 / .80 |
| MEXTRA    | 40 / .44 / .23 / .97 | 24 / .27 / .20 / .57 | 28 / .31 / .13 / .73 |
| **ADAM**  | **63 / .70 / .97 / .93** | **42 / .47 / .57 / .77** | **41 / .46 / .93 / .70** |

The ordering and metric relationships match the paper:
**ADAM ≫ MEXTRA ≈ Pirate > RAG-Thief > Vanilla**, with ADAM ≈ 1.6–1.7× the best
baseline's EQ and far higher CER. EHRAgent magnitudes track the paper closely
(paper ADAM EHRAgent EQ = 77–83); ReAct/RAP absolute values are lower in
simulation but preserve every ordering (those exact numbers are LLM-specific).

### Figures (in `results/`)

- **`fig3_ablations.png`** — EQ/EE rise with top-k, model size, #anchors and memory
  size; fall with the retriever similarity threshold; are robust to the encoder and
  scoring function; domain knowledge gives a modest gain (Fig. 3, Appendix D).
- **`fig4_oracle.png`** — the oracle (handed `P(D)`) ≥ ADAM; the L1 gap between
  ADAM's estimated distribution and `P(D)` is smallest on EHRAgent (best estimation).
- **`fig5_defenses.png`** — ADAM retains **89–100%** of its EQ under every defense;
  MEXTRA is **blocked entirely** by keyword filtering and loses ~40% under RA-LLM /
  erase-and-check. Surface-level defenses fail against ADAM's semantic, paraphrased
  queries.
- **`fig6_convergence.png`** — the estimated data distribution converges to `P(D)`
  (L1 ↓), cumulative EQ is monotonically non-decreasing (the EM view, Appendix H),
  early-stop (Eq. 2) fires once coverage plateaus, and rate control only degrades
  EQ gracefully (Appendix O).

---

## Deployment (Linux + Anaconda)

```bash
# 1. get the code
git clone <repo-url> ADAM && cd ADAM
git checkout claude/charming-pascal-b4izsp        # the branch with this work

# 2a. create the env with conda (recommended)
conda env create -f environment.yml               # builds env "adam" (python 3.11)
conda activate adam

# 2b. or with plain pip / venv
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt                   # simulation only
pip install openai                                # also needed for realrun.py

# 3. verify the install
python tests/test_adam.py                          # expect "11/11 tests passed"

# 4. run the simulation (no API key, ~20s)  -- always run from the repo root
python run_all.py
```

## Quick start

```bash
pip install -r requirements.txt          # numpy, scikit-learn, scipy, matplotlib

python run_all.py                         # reproduce Table 1 + Figures 3–6 (~20s)
python -m pytest tests/ -q                # or: python tests/test_adam.py

# individual experiments
python experiments/run_main_table.py      # Table 1
python experiments/run_ablations.py       # Figure 3
python experiments/run_oracle.py          # Figure 4
python experiments/run_defenses.py        # Figure 5
python experiments/run_convergence.py     # Figure 6

# use real backends (identical attack code)
export OPENAI_API_KEY=...
python experiments/run_main_table.py --attacker-llm gpt-4o-mini --encoder all-MiniLM-L6-v2

# run against a real LLM agent via SiliconFlow (DeepSeek-V3.2-Exp)
pip install openai
export SILICONFLOW_API_KEY=sk-...
python realrun.py --smoke                            # 1 attack, T=3, |M|=30 (~few mins)
python realrun.py --attacks ADAM MEXTRA --T 10       # short comparison
```

---

## Repository layout

```
adam/
  config.py        Hyper-parameters (k, alpha, lambda, tau, T, eps...) + seed topics (Table 13)
  embeddings.py    z(.) encoders: offline HashingEncoder (default) + Sentence-Transformers
  llm.py           Attacker generator G_aux: MockLLM (offline) + OpenAI backend
  injection.py     Prefix/suffix injection templates (Table 14)
  memory.py        Agent memory M: records (q_i, s_i, latent topic) + top-k retrieval
  agent.py         Victim LLM agent: retrieval, defense hooks, leak model (ASR/CER)
  anchors.py       Anchor extraction (NER/normalize/dedup) + the alpha-novelty pool T_t
  distribution.py  Distribution estimation: clustering -> weights -> decayed softmax
  selection.py     Weighted k-center anchor selection (Eq. 1)
  entropy.py       Entropy-based query selection
  refine.py        Refine operator: parse, de-noise, self-consistency, dedup
  attack_adam.py   ADAM attack (Algorithm 1) + Attack base loop
  baselines.py     Vanilla, RAG-Thief, Pirate, MEXTRA
  defenses.py      query rewriting, auxiliary filtering, RA-LLM, erase-and-check, rate control
  metrics.py       EQ, EE, CER, ASR
datasets/
  synthetic.py     Synthetic EHRAgent / ReAct / RAP memories with known P(D) + seed subsets
experiments/       run_main_table / run_ablations / run_oracle / run_defenses / run_convergence
tests/             Unit tests for every core component (incl. the Table-8 entropy example)
results/           Generated CSVs and figures
```

## Mapping to the paper

| Paper | Code |
|---|---|
| Algorithm 1 (the loop) | `adam/attack_adam.py::ADAM.run` |
| Distribution est. `P~ -> P- -> softmax` | `adam/distribution.py::DistributionEstimator.update` |
| Weighted k-center (Eq. 1) | `adam/selection.py::select_anchors` |
| Entropy `H_t(q)` | `adam/entropy.py::entropy/select_query` |
| Anchor admit `max sim <= alpha` | `adam/anchors.py::AnchorPool.update` |
| Early stop (Eq. 2) | `adam/attack_adam.py` (`early_stop`) |
| EM convergence (Appendix H) | `experiments/run_convergence.py` |
| Metrics (Sec. 4.1) | `adam/metrics.py` |
| Clustering ablation (Appendix I) | `adam/distribution.py` (`cluster_method`) |
| OOD seeds (Appendix D) | `experiments/run_ablations.py::ablation_domain_knowledge` |

## Notes & limitations

- Absolute EQ/EE/CER/ASR from the simulation are **calibrated approximations**; the
  scientific claims reproduced are the **orderings, metric relationships, ablation
  trends, oracle behavior and defense robustness**. Plug in the real backends for
  the paper's exact numbers.
- The default `HashingEncoder` drops function words so distinctive topic words drive
  cosine similarity (mimicking how a real sentence encoder behaves); the paper shows
  the attack is robust to the encoder choice (Fig. 3g).
- Calibration constants (per-LLM compliance, per-attack injection/completeness) are
  isolated in `experiments/common.py` for full transparency.
