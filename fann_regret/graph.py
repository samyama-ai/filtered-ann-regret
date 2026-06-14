"""In-filter strategy + the percolation phase boundary (H1b).

The third strategy: traverse a navigable graph but visit ONLY eligible nodes (greedy restricted to
the predicate-induced subgraph G[X_P]). This is what makes the graph cliff a *percolation* transition:
keeping an s-fraction of nodes is site percolation, and below a threshold s_c the eligible subgraph
loses its navigable backbone, so greedy search disconnects from the target and recall collapses.

Prediction (theory.percolation_sc): s_c ~ c * log(n) / M for graph out-degree M. We build an exact
M-NN graph (faiss), run restricted greedy with a budget, and locate s_c by recall collapse, then fit
the c * log n / M law across (n, M).
"""
from __future__ import annotations

import heapq

import numpy as np


def build_knn_graph(base: np.ndarray, M: int, long_range_frac: float = 0.25,
                    seed: int = 0) -> np.ndarray:
    """(n x M) int32 navigable small-world adjacency: out-degree M = (1-f)*M nearest neighbours +
    f*M random long-range edges. The long-range links make the graph a single connected component at
    s=1 (like HNSW), so percolation under node deletion reflects SELECTIVITY, not the cluster structure
    of a pure k-NN graph (which fragments into per-cluster cliques and confounds the transition)."""
    import faiss
    n = base.shape[0]
    n_long = max(1, int(round(long_range_frac * M)))
    n_near = M - n_long
    index = faiss.IndexFlatL2(base.shape[1])
    index.add(np.ascontiguousarray(base, dtype=np.float32))
    _, nn = index.search(np.ascontiguousarray(base, dtype=np.float32), n_near + 1)
    rng = np.random.default_rng(seed)
    adj = np.empty((n, M), dtype=np.int32)
    for i in range(n):
        near = nn[i][nn[i] != i][:n_near]
        if near.size < n_near:
            near = np.concatenate([near, nn[i][:n_near - near.size]])
        longr = rng.integers(0, n, size=n_long)              # random long-range targets
        adj[i] = np.concatenate([near[:n_near], longr])[:M]
    return adj


def filtered_greedy(adj, base, query, elig_mask, k=10, ef=64, n_seeds=8, rng=None):
    """Greedy best-first search restricted to eligible nodes; budget ef (size of the working set).

    Entry: the eligible node nearest to the query among n_seeds random eligible seeds. Expansion only
    follows eligible neighbours. Returns the k nearest eligible nodes found (ids). Recall collapses
    when G[X_P] is disconnected (percolation): greedy is trapped in the seed's local component.
    """
    elig_ids = np.flatnonzero(elig_mask)
    if elig_ids.size == 0:
        return np.empty(0, dtype=np.int64)
    rng = rng or np.random.default_rng(0)
    seeds = rng.choice(elig_ids, size=min(n_seeds, elig_ids.size), replace=False)
    d_seed = ((base[seeds] - query) ** 2).sum(1)
    entry = int(seeds[np.argmin(d_seed)])

    def dist(node):
        return float(((base[node] - query) ** 2).sum())

    visited = {entry}
    cand = [(dist(entry), entry)]                # min-heap by distance (candidates to expand)
    best = [(-dist(entry), entry)]               # max-heap (negated) of best-k found
    while cand:
        d, node = heapq.heappop(cand)
        if -best[0][0] < d and len(best) >= ef:  # no closer than current worst in a full working set
            break
        for nb in adj[node]:
            if nb in visited or not elig_mask[nb]:
                continue
            visited.add(nb)
            dnb = dist(nb)
            heapq.heappush(cand, (dnb, nb))
            heapq.heappush(best, (-dnb, nb))
            if len(best) > ef:
                heapq.heappop(best)
    found = sorted(((-nd, i) for nd, i in best))[:k]
    return np.array([i for _, i in found], dtype=np.int64)


def infilter_recall(adj, base, queries, rank_ids, elig_mask, k=10, ef=64, seed=0):
    """Mean recall@k of restricted filtered-greedy vs the true filtered top-k (first k eligible in the
    exact ranking) over the query set, for a single predicate realisation."""
    rng = np.random.default_rng(seed)
    recalls = []
    for qi in range(queries.shape[0]):
        # true filtered top-k = first k eligible ids along the exact ranking
        elig_along = elig_mask[rank_ids[qi]]
        truth = rank_ids[qi][elig_along][:k]
        if truth.size < k:
            continue
        got = filtered_greedy(adj, base, queries[qi], elig_mask, k=k, ef=ef, rng=rng)
        recalls.append(np.isin(truth, got).mean())
    return float(np.mean(recalls)) if recalls else float("nan")


def giant_component_fraction(adj: np.ndarray, elig_mask: np.ndarray) -> float:
    """Fraction of eligible nodes in the largest connected component of the induced subgraph G[X_P]
    (edges symmetrised). This is the site-percolation order parameter: keeping an s-fraction of nodes,
    does a giant navigable component survive? Union-find over eligible-eligible edges only."""
    elig_ids = np.flatnonzero(elig_mask)
    if elig_ids.size == 0:
        return 0.0
    parent = {int(i): int(i) for i in elig_ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in elig_ids:
        ii = int(i)
        for nb in adj[ii]:
            nb = int(nb)
            if elig_mask[nb]:                    # edge kept only if both endpoints eligible
                union(ii, nb)
    sizes = {}
    for i in elig_ids:
        r = find(int(i))
        sizes[r] = sizes.get(r, 0) + 1
    return max(sizes.values()) / elig_ids.size


def percolation_threshold(adj, n, s_grid=None, n_pred=3, rho_target=0.5):
    """Locate s_c: selectivity where the giant-component fraction of G[X_P] crosses rho_target.

    Tests the percolation phase transition directly (connectivity, no search-quality confound). The
    `base/queries/rank` args are no longer needed -- s_c is a structural property of the graph.
    """
    from . import labels
    if s_grid is None:
        s_grid = np.geomspace(0.003, 0.6, 22)
    frac = []
    for s in s_grid:
        vals = [giant_component_fraction(adj, labels.labels_uncorrelated(n, s, seed=200 + j))
                for j in range(n_pred)]
        frac.append(float(np.mean(vals)))
    frac = np.array(frac)
    above = np.where(frac >= rho_target)[0]
    if above.size == 0:
        return float("nan"), s_grid, frac
    i = above[0]
    if i == 0:
        return float(s_grid[0]), s_grid, frac
    x0, x1, y0, y1 = np.log(s_grid[i - 1]), np.log(s_grid[i]), frac[i - 1], frac[i]
    sc = float(np.exp(x0 + (rho_target - y0) * (x1 - x0) / (y1 - y0 + 1e-12)))
    return sc, s_grid, frac
