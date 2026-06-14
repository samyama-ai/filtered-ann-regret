"""Correctness for the percolation / in-filter module (graph.py)."""
import numpy as np

from fann_regret import data, graph as G


def test_giant_component_fraction_known():
    # adjacency of two disjoint triangles {0,1,2} and {3,4,5}; all eligible -> giant frac = 3/6.
    adj = np.array([[1, 2], [0, 2], [0, 1], [4, 5], [3, 5], [3, 4]], dtype=np.int32)
    mask = np.ones(6, bool)
    assert abs(G.giant_component_fraction(adj, mask) - 0.5) < 1e-9
    # drop node 5 -> components {0,1,2}(3) and {3,4}(2) of 5 eligible -> 3/5
    mask[5] = False
    assert abs(G.giant_component_fraction(adj, mask) - 0.6) < 1e-9


def test_percolation_sc_decreases_with_degree():
    """Site percolation: higher degree M => lower selectivity needed to keep a giant component."""
    base, _ = data.synthetic_gaussian(n=8000, d=16, nq=2, n_clusters=15, seed=1)
    sc = {}
    for M in (8, 24):
        adj = G.build_knn_graph(base, M, seed=0)
        sc[M], _, _ = G.percolation_threshold(adj, n=8000, n_pred=1)
    assert np.isfinite(sc[8]) and np.isfinite(sc[24])
    assert sc[24] < sc[8]                       # s_c ~ 1/M
