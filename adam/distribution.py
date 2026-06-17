"""Data-distribution estimation over anchors (Sec. 3, Appendix I).

This is ADAM's key novelty. Given the anchor pool and the embeddings of anchors
seen in the latest response, we:

  1. cluster anchors (DBSCAN by default; KDE/GMM/k-means as alternatives) and
     read off each anchor's cluster size  c_t(a);
  2. form weights        w_t(a)  = c_t(a) / sum_{a'} c_t(a');
  3. accumulate          P~_t(a) = P^_{t-1}(a) + w_t(a);
  4. decay by re-use     P-_t(a) = P~_t(a) * lambda ** SelCount_{t-1}(a);
  5. normalise (softmax) P^_t(a) = softmax(P-_t(a) / tau).

This favours newly-discovered anchors while discouraging repeatedly-selected
ones, steering queries toward unexplored regions of the agent's memory.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity


def _cluster_sizes(emb: np.ndarray, method: str, cfg) -> np.ndarray:
    """Return, for each row of ``emb``, the size of the cluster it belongs to."""
    n = len(emb)
    if n == 0:
        return np.zeros(0)
    if n == 1:
        return np.ones(1)

    if method == "dbscan":
        labels = DBSCAN(eps=cfg.dbscan_eps, min_samples=cfg.dbscan_min_samples,
                        metric="cosine").fit_predict(emb)
        # DBSCAN noise points (label -1) form singleton "clusters".
        sizes = np.ones(n)
        for lab in set(labels):
            if lab == -1:
                continue
            idx = np.where(labels == lab)[0]
            sizes[idx] = len(idx)
        return sizes

    if method == "kmeans":
        k = max(1, min(n, int(np.sqrt(n))))
        labels = KMeans(n_clusters=k, n_init=5, random_state=cfg.seed).fit_predict(emb)
    elif method == "gmm":
        k = max(1, min(n, int(np.sqrt(n))))
        labels = GaussianMixture(n_components=k, random_state=cfg.seed).fit_predict(emb)
    elif method == "kde":
        kde = KernelDensity(bandwidth=0.5).fit(emb)
        dens = np.exp(kde.score_samples(emb))
        # discretise density into pseudo-clusters by rank bucketing
        ranks = np.argsort(np.argsort(dens))
        labels = (ranks * max(1, int(np.sqrt(n))) // n)
    else:
        raise ValueError(f"unknown clustering method '{method}'")

    sizes = np.ones(n)
    for lab in set(labels):
        idx = np.where(labels == lab)[0]
        sizes[idx] = len(idx)
    return sizes


class DistributionEstimator:
    """Maintains the estimated anchor distribution P^_t and selection counts."""

    def __init__(self, cfg, anchor_pool):
        self.cfg = cfg
        self.pool = anchor_pool
        # uniform prior P^_0(a) = 1/|T_0|  (Sec. 3 initialisation)
        m = max(1, len(anchor_pool))
        self.P: Dict[str, float] = {a: 1.0 / m for a in anchor_pool.anchors}
        self.sel_count: Dict[str, int] = {a: 0 for a in anchor_pool.anchors}

    def _ensure(self, anchors: Sequence[str]) -> None:
        for a in anchors:
            if a not in self.P:
                # new anchor gets a small prior before the next normalisation
                self.P[a] = 1.0 / max(1, len(self.P))
                self.sel_count.setdefault(a, 0)

    def note_selected(self, anchors: Sequence[str]) -> None:
        for a in anchors:
            self.sel_count[a] = self.sel_count.get(a, 0) + 1

    def update(self, observed_anchors: Sequence[str]) -> Dict[str, float]:
        """One UpdateDist step. ``observed_anchors`` = anchors present in r_t."""
        self._ensure(self.pool.anchors)
        keys = list(self.pool.anchors)
        emb = self.pool.matrix(keys)
        sizes = _cluster_sizes(emb, self.cfg.cluster_method, self.cfg)
        size_of = {a: float(s) for a, s in zip(keys, sizes)}

        # weights only from anchors observed in the current response r_t
        obs = set(observed_anchors) & set(keys)
        denom = sum(size_of[a] for a in obs) or 1.0
        w = {a: (size_of[a] / denom if a in obs else 0.0) for a in keys}

        # P~, then decay by re-use, then softmax
        p_tilde = {a: self.P.get(a, 0.0) + w[a] for a in keys}
        p_bar = {a: p_tilde[a] * (self.cfg.lam ** self.sel_count.get(a, 0)) for a in keys}

        vals = np.array([p_bar[a] for a in keys]) / self.cfg.tau
        vals -= vals.max()  # numerical stability
        ex = np.exp(vals)
        probs = ex / ex.sum()
        self.P = {a: float(p) for a, p in zip(keys, probs)}
        return self.P

    def l1_to(self, prev: Dict[str, float]) -> float:
        keys = set(self.P) | set(prev)
        return float(sum(abs(self.P.get(k, 0.0) - prev.get(k, 0.0)) for k in keys))

    def snapshot(self) -> Dict[str, float]:
        return dict(self.P)
