"""Real approximate-ANN (hnswlib) post-filter, for the model-mismatch leak test.

The frozen H2 used a perfect-model threshold selector, so regret vanished off the boundary by
construction. The genuine question: when the planner's COST MODEL mis-locates s* -- because a real
approximate index's recall differs from the analytic order-statistics model -- does regret leak into
the phase interiors (a persistent miscalibration band that estimation-error robustness cannot fix)?
"""
from __future__ import annotations

import numpy as np


def build_hnsw(base: np.ndarray, M: int = 16, ef_construction: int = 200, seed: int = 100):
    import hnswlib
    n, d = base.shape
    idx = hnswlib.Index(space="l2", dim=d)
    idx.init_index(max_elements=n, ef_construction=ef_construction, M=M, random_seed=seed)
    idx.add_items(np.ascontiguousarray(base, dtype=np.float32), np.arange(n))
    return idx


def approx_post_recall(idx, queries, rank_ids, labels, B, k=10):
    """Per-query recall@k of a REAL approximate post-filter: pull the approx top-B (ef=B), drop
    violators, keep the k nearest eligible; score against the exact true filtered top-k."""
    idx.set_ef(int(max(B, k + 1)))
    cand, _ = idx.knn_query(np.ascontiguousarray(queries, dtype=np.float32), k=int(B))
    recalls = []
    for qi in range(queries.shape[0]):
        elig_along = labels[rank_ids[qi]]
        truth = rank_ids[qi][elig_along][:k]
        if truth.size < k:
            continue
        ce = cand[qi][labels[cand[qi]]][:k]            # approx eligible candidates, top-k by ANN order
        recalls.append(np.isin(truth, ce).mean())
    return float(np.mean(recalls)) if recalls else float("nan")
