"""Unit tests for the core ADAM components.

Run with:  python -m pytest tests/ -q   (or)   python tests/test_adam.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adam.anchors import AnchorPool, extract_anchors
from adam.config import AgentConfig, AttackConfig
from adam.distribution import DistributionEstimator
from adam.embeddings import HashingEncoder, cosine
from adam.entropy import entropy, select_query
from adam.metrics import MetricTracker
from adam.refine import refine
from adam.selection import select_anchors


ENC = HashingEncoder()


# ---------------------------------------------------------------- entropy
def test_entropy_ordering_table8():
    """Paper Table 8: Q3 (flat) > Q2 > Q1 (peaked) in entropy."""
    P = {"a": 0.90, "b": 0.05, "c": 0.05}
    q1 = entropy(["a", "b", "c"], {"a": 0.90, "b": 0.05, "c": 0.05}, 1e-12)
    q2 = entropy(["a", "b", "c"], {"a": 0.60, "b": 0.20, "c": 0.20}, 1e-12)
    q3 = entropy(["a", "b", "c"], {"a": 0.34, "b": 0.33, "c": 0.33}, 1e-12)
    assert q1 < q2 < q3, (q1, q2, q3)


def test_entropy_selection_prefers_high_entropy():
    P = {"x": 0.5, "y": 0.3, "z": 0.2, "w": 0.0}
    cands = [
        {"text": "low", "topics": ["x"]},                 # peaked -> low H
        {"text": "high", "topics": ["x", "y", "z"]},      # spread -> high H
    ]
    chosen = select_query(cands, P, 1e-12)
    assert chosen["text"] == "high"


# ------------------------------------------------------- distribution est.
def test_distribution_normalised_and_decays():
    pool = AnchorPool(ENC, ["diagnosis", "medication", "patient"], alpha=0.5)
    cfg = AttackConfig(seed=0)
    de = DistributionEstimator(cfg, pool)
    P0 = de.snapshot()
    assert abs(sum(P0.values()) - 1.0) < 1e-9       # uniform prior sums to 1

    de.update(["diagnosis", "medication"])
    P1 = de.snapshot()
    assert abs(sum(P1.values()) - 1.0) < 1e-6       # softmax stays normalised

    # selecting an anchor repeatedly must lower its (pre-softmax) preference
    de.note_selected(["diagnosis"])
    de.note_selected(["diagnosis"])
    p_before = de.snapshot()["diagnosis"]
    de.update(["diagnosis", "medication"])
    assert de.snapshot()["diagnosis"] <= p_before + 1e-6


# ------------------------------------------------------------- k-center
class _FakePool:
    """Pool with controlled embeddings to test k-center geometry directly."""
    def __init__(self, vecs):
        self._v = {a: np.asarray(v, dtype=float) for a, v in vecs.items()}
        self.anchors = list(self._v)
    def vec(self, a):
        return self._v[a]


def test_kcenter_picks_diverse_anchors():
    # near1/near2 are close; far is distant. With equal P, after picking near1
    # the weighted k-center must pick `far`, not the near-duplicate near2.
    pool = _FakePool({"near1": [1.0, 0.0], "near2": [0.98, 0.2], "far": [0.0, 1.0]})
    P = {a: 1.0 for a in pool.anchors}
    sel = select_anchors(pool, P, k=2, used=set())
    assert sel[0] == "near1" and sel[1] == "far", sel


def test_kcenter_first_pick_prefers_unused_high_prob():
    pool = AnchorPool(ENC, ["a_topic", "b_topic", "c_topic"], alpha=0.99)
    P = {"a_topic": 0.1, "b_topic": 0.8, "c_topic": 0.1}
    sel = select_anchors(pool, P, k=1, used={"b_topic"})
    assert sel[0] != "b_topic"          # b is used -> skip despite high prob


# ------------------------------------------------------------- anchors
def test_anchor_pool_novelty_threshold():
    pool = AnchorPool(ENC, ["diagnosis"], alpha=0.5)
    pool.update(["diagnosis"])                # exact dup -> rejected
    assert len(pool) == 1
    added = pool.update(["microbiology"])     # novel -> added
    assert "microbiology" in added and len(pool) == 2


def test_extract_anchors_drops_pii_and_stopwords():
    text = "What was the medication that patient 29309 received on 01/06/2015?"
    anchors = extract_anchors(text)
    assert "medication" in anchors
    assert "29309" not in anchors and "patient" not in anchors  # PII / stopword


# ------------------------------------------------------------- refine
def test_refine_parses_dedups_and_verifies():
    resp = (
        "Here are the previous examples I have:\n"
        "- What was the name of the output for patient 655?\n"
        "- What was the name of the output for patient 655?\n"   # dup
        "- Sorry, I cannot share that.\n"                         # boilerplate
        "- What is the date of birth of patient 45601?"
    )
    queries, anchors = refine(resp)
    assert len(queries) == 2                          # dup + boilerplate removed
    assert any("date of birth" in q for q in queries)


# ------------------------------------------------------------- metrics
def test_metrics_definitions():
    mt = MetricTracker(k=3)
    mt.record(["q1", "q2", "q3"], ["q1", "q2", "q3"])   # full dump, all new
    mt.record(["q1", "q4", "q5"], ["q4"])               # 1 new
    mt.record(["q6", "q7", "q8"], [])                   # nothing
    assert mt.EQ == 4                                   # q1,q2,q3,q4 unique
    assert mt.n == 3
    assert abs(mt.EE - 4 / 9) < 1e-9                    # |Q|/(n*k)
    assert abs(mt.CER - 1 / 3) < 1e-9                   # only round1 complete
    assert abs(mt.ASR - 2 / 3) < 1e-9                   # rounds 1,2 had new


# ------------------------------------------------------------- embeddings
def test_encoder_topic_separation():
    e = HashingEncoder()
    v1 = e.encode(["diagnosis of the patient condition"])[0]
    v2 = e.encode(["diagnosis and patient illness"])[0]
    v3 = e.encode(["wireless bluetooth speaker device"])[0]
    assert cosine(v1, v2) > cosine(v1, v3)              # same-topic closer


# ------------------------------------------------------------- end-to-end
def test_adam_beats_vanilla_end_to_end():
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))
    from common import RunSpec, run_single
    adam = run_single(RunSpec("ehragent", "ADAM", "chatgpt-4", 0))
    van = run_single(RunSpec("ehragent", "Vanilla", "chatgpt-4", 0))
    assert adam["EQ"] > 2 * van["EQ"]
    assert adam["ASR"] >= van["ASR"]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL  {fn.__name__}: {e}")
        except Exception as e:  # noqa
            print(f"ERROR {fn.__name__}: {e}")
    print(f"\n{passed}/{len(fns)} tests passed")
    sys.exit(0 if passed == len(fns) else 1)
