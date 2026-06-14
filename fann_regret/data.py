"""Datasets + exact distance rankings for the filtered-ANN harness.

We need, per query, the exact distance ranking of base ids deep enough to (a) locate the true
filtered top-k at the lowest selectivity tested (depth ~ k / s_min) and (b) cover the largest
post-filter candidate list B. We compute it once with an exact flat index and cache it.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

BASE = Path(os.environ.get("FANN_BASE", Path(__file__).resolve().parent.parent / "data"))


# ---------------------------------------------------------------- synthetic (fast, controlled)
def synthetic_gaussian(n: int, d: int = 32, nq: int = 500, n_clusters: int = 50, seed: int = 0):
    """Clustered Gaussian vectors + queries drawn from the same mixture (realistic neighbor density)."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 10, size=(n_clusters, d)).astype(np.float32)
    assign = rng.integers(0, n_clusters, size=n)
    base = (centers[assign] + rng.normal(0, 1, size=(n, d))).astype(np.float32)
    qassign = rng.integers(0, n_clusters, size=nq)
    queries = (centers[qassign] + rng.normal(0, 1, size=(nq, d))).astype(np.float32)
    return base, queries


# ---------------------------------------------------------------- SIFT1M loader (.fvecs/.ivecs)
def _read_vecs(path: Path, dtype) -> np.ndarray:
    """Read .fvecs/.ivecs/.bvecs: each row = int32 dim followed by `dim` values of `dtype`."""
    raw = np.fromfile(path, dtype=np.int32)
    dim = raw[0]
    row = dim + 1
    return raw.reshape(-1, row)[:, 1:].view(dtype).astype(np.float32 if dtype != np.int32 else np.int32)


def load_sift1m(root: Path | None = None):
    """Load ANN_SIFT1M (base 1M x128, queries 10k). Download instructions in bench/fetch_data.sh."""
    root = Path(root) if root else BASE / "sift"
    base = _read_vecs(root / "sift_base.fvecs", np.float32)
    queries = _read_vecs(root / "sift_query.fvecs", np.float32)
    return base, queries


# ---------------------------------------------------------------- exact ranking (cached)
def exact_ranking(base: np.ndarray, queries: np.ndarray, depth: int, cache: Path | None = None,
                  tag: str = "") -> np.ndarray:
    """(nq x depth) int32 matrix of base ids sorted by ascending L2 distance to each query.

    Uses faiss IndexFlatL2 (exact). Cached to .npy keyed by tag (caller ensures tag uniqueness).
    """
    if cache is not None and (cache / f"rank_{tag}.npy").exists():
        return np.load(cache / f"rank_{tag}.npy")
    import faiss
    depth = int(min(depth, base.shape[0]))
    index = faiss.IndexFlatL2(base.shape[1])
    index.add(np.ascontiguousarray(base, dtype=np.float32))
    _, ids = index.search(np.ascontiguousarray(queries, dtype=np.float32), depth)
    ids = ids.astype(np.int32)
    if cache is not None:
        cache.mkdir(parents=True, exist_ok=True)
        np.save(cache / f"rank_{tag}.npy", ids)
    return ids
