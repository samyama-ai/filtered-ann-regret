"""Synthetic predicate labels at a controlled selectivity s.

U (uncorrelated): each point eligible iid w.p. s  -> the order-statistics model holds exactly.
C (correlated):   eligibility correlated with distance-to-query rank (the adversarial case). We
                  implement a per-query mark process where eligibility probability varies with rank
                  via a coupling rho in [-1, 1]: rho>0 pushes eligible points EARLY (easy),
                  rho<0 pushes them LATE (the post-filter-killing adversary). rho=0 reduces to U.
"""
from __future__ import annotations

import numpy as np


def labels_uncorrelated(n: int, s: float, seed: int = 0) -> np.ndarray:
    """Global boolean eligibility vector, each point iid Bernoulli(s)."""
    rng = np.random.default_rng(seed)
    return rng.random(n) < s


def marks_correlated_for_ranking(rank_ids: np.ndarray, s: float, rho: float,
                                 seed: int = 0) -> np.ndarray:
    """Per-query eligibility marks along each query's ranking (nq x depth bool), correlated with rank.

    For query q with ranking r_0..r_{D-1} (nearest first), point at rank j is eligible with prob
        p_j = s * (1 + rho * (1 - 2*j/D)),   clipped to [0,1],
    so rho<0 lowers eligibility for near (small j) points -> eligible points pushed late (adversary).
    Mean over j stays ~ s. Returns a per-query mark matrix (not a global label vector, since the
    correlation is defined relative to each query's own ranking).
    """
    nq, D = rank_ids.shape
    j = np.arange(D)
    p = np.clip(s * (1.0 + rho * (1.0 - 2.0 * j / D)), 0.0, 1.0)        # (D,)
    rng = np.random.default_rng(seed)
    return rng.random((nq, D)) < p[None, :]
