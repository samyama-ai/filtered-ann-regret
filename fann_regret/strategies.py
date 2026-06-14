"""Empirical strategy objectives: recall@k under a fixed compute budget B, on real rankings.

Eligibility along a query's ranking is given as a boolean matrix `elig` (nq x depth): elig[q, j] =
is the point at rank j (j-th nearest to query q) eligible under predicate P. Build it from a global
label vector (uncorrelated) with `elig_from_labels`, or pass per-query correlated marks directly.

post-filter   -> needs `elig` (and under uncorrelated labels MUST match theory.post_recall_exact: H0)
pre-filter    -> analytic min(1, B/(s*n)); INDIFFERENT to query-predicate correlation (materializes
                 X_P exactly if the budget covers it). Same formula for U and C.
in-filter     -> requires the ANN graph (graph.py); percolation-limited. (Stage: stretch / H1b.)
"""
from __future__ import annotations

import numpy as np


def elig_from_labels(rank_ids: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """(nq x depth) bool: eligibility along each ranking, from a global per-point label vector."""
    return labels[rank_ids]


def post_recall_empirical(elig: np.ndarray, K: int, k: int = 10) -> np.ndarray:
    """Per-query recall@k of a post-filter examining the top-K of each ranking.

    Returns recall = min(k, E_q)/k where E_q = #eligible in the first K ranks of query q. This equals
    the closed form under uncorrelated labels and deviates under correlation (the content of C).
    """
    K = int(min(K, elig.shape[1]))
    E = elig[:, :K].sum(axis=1)
    return np.minimum(k, E) / k


def pre_recall(s: float, n: int, B: int, k: int = 10) -> float:
    """Pre-filter recall@k at budget B. Exact (=1) when the eligible set fits the budget (s*n<=B),
    else a random B-subset of X_P is scanned -> expected recall min(1, B/(s*n))."""
    sn = max(s * n, 1e-9)
    return float(min(1.0, B / sn))


def true_filtered_topk_depth(elig: np.ndarray, k: int = 10) -> np.ndarray:
    """Per-query ranking depth at which the k-th eligible point appears (np.inf if < k eligible in
    the ranking). Diagnostic for whether the ranking is deep enough at low selectivity."""
    nq, D = elig.shape
    out = np.full(nq, np.inf)
    cum = np.cumsum(elig, axis=1)
    for q in range(nq):
        idx = np.searchsorted(cum[q], k)
        if idx < D:
            out[q] = idx + 1
    return out


def objective_surface(rank_ids, labels_or_marks, s: float, n: int, B: int, k: int = 10,
                      correlated: bool = False):
    """Mean recall@k for {pre, post} at selectivity s, budget B. Returns dict of per-strategy means.

    labels_or_marks: global bool vector (correlated=False) or per-query mark matrix (correlated=True).
    """
    elig = labels_or_marks if correlated else elig_from_labels(rank_ids, labels_or_marks)
    post = float(post_recall_empirical(elig, B, k).mean())
    pre = pre_recall(s, n, B, k)
    return {"pre": pre, "post": post}
