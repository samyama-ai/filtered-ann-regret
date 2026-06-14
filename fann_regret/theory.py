"""Closed-form predictions for filtered-ANN strategy selection (the first-principles core).

Everything here is pure math — no data, no indexes — so it is deterministic and unit-testable.
The empirical harness (strategies.py) must *reproduce* these curves on real data; where it
deviates is where the geometry (approximate ANN, correlated predicates, graph percolation) adds
content beyond the idealized model.

Notation
--------
n  : corpus size
s  : true selectivity = |X_P| / n  (fraction passing predicate P)
k  : neighbors requested (recall@k); default 10
K  : post-filter candidate-list size (unfiltered ANN top-K examined before dropping violators)
B  : compute budget in distance evaluations (hardware-independent)
M  : ANN graph degree (HNSW/Vamana out-degree)

The objective M(strategy, s) is recall@k achieved under a fixed budget B.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import binom


# --------------------------------------------------------------------------------------
# (B) Order statistics -> the post-filter recall cliff, exact under uncorrelated labels
# --------------------------------------------------------------------------------------
def post_recall_exact(K: int, s: float, k: int = 10) -> float:
    """Expected recall@k of a post-filter examining the unfiltered top-K, uncorrelated predicate.

    Derivation (see FIRST-PRINCIPLES sec. B). The global distance ranking is a prefix-nested
    sequence; the unfiltered top-K is its length-K prefix. The *true filtered top-k* are the first
    k eligible points in the global ranking. Post-filter returns the first min(k, E) eligible points
    within the top-K, where E = #eligible in top-K. Because top-K is a prefix, those first min(k,E)
    eligible points ARE the first min(k,E) of the true filtered ranking -> the intersection has size
    exactly min(k, E). Under an uncorrelated predicate each of the K points is eligible iid w.p. s,
    so E ~ Binomial(K, s) and

        recall@k(K, s) = E[min(k, Binom(K, s))] / k.

    This is exact (not an approximation) for the uncorrelated-label model, independent of geometry.
    """
    if K <= 0:
        return 0.0
    s = float(np.clip(s, 0.0, 1.0))
    js = np.arange(0, k + 1)                     # E = 0..k contribute min(k,E)=E; E>k contributes k
    pmf = binom.pmf(js, K, s)                     # P(E = j) for j=0..k
    tail = 1.0 - binom.cdf(k, K, s) + binom.pmf(k, K, s)  # P(E >= k) ... handled below
    # E[min(k,E)] = sum_{j=0}^{k-1} j*P(E=j) + k*P(E>=k)
    exp_min = float(np.dot(js[:k], pmf[:k])) + k * (1.0 - binom.cdf(k - 1, K, s))
    return exp_min / k


def post_recall_curve(K: int, s_grid, k: int = 10) -> np.ndarray:
    return np.array([post_recall_exact(K, s, k) for s in s_grid])


def post_knee_selectivity(K: int, k: int = 10) -> float:
    """Selectivity at the post-filter cliff knee. Mean eligible Ks = k  ->  s = k/K."""
    return k / float(K)


# --------------------------------------------------------------------------------------
# (A) Percolation -> the graph / in-filter recall cliff
# --------------------------------------------------------------------------------------
def percolation_sc(n: int, M: int, c: float = 1.0) -> float:
    """Site-percolation navigability threshold for filtered greedy search on a degree-M graph.

    Keeping an s-fraction of nodes (those passing P) is site percolation; the navigable backbone
    shatters below s_c ~ c * log(n) / M (catalog's M >~ log n / s read as a critical line). Below
    s_c, filtered greedy search disconnects from the target neighborhood and in-filter recall
    collapses. c is an O(1) graph-dependent constant.

    NOTE (empirical, run_percolation.py): for the *giant-component* (navigability) threshold the data
    pick the n-INDEPENDENT law s_c ~ 0.83 / M (slope_lnM=-0.91, slope_ln(ln n)=-0.20, R2=0.998) over
    this full-connectivity log n / M form. Use percolation_sc_giant for the fitted law.
    """
    return c * np.log(n) / float(M)


def percolation_sc_giant(M: int, c: float = 0.83) -> float:
    """Empirically-fit in-filter giant-component percolation threshold: s_c ~ c / M (n-independent),
    the navigability-governing threshold (run_percolation.py: R2=0.998, c~0.83)."""
    return c / float(M)


# --------------------------------------------------------------------------------------
# Strategy objectives under a fixed distance-evaluation budget B
# --------------------------------------------------------------------------------------
def pre_recall_budget(s: float, n: int, B: int, k: int = 10) -> float:
    """Pre-filter: materialize X_P (cost ~ s*n) then exact search within it.

    If s*n <= B the budget covers the whole eligible set -> exact -> recall 1. Otherwise only a
    random B-fraction of X_P can be touched, and a given true-filtered neighbor is reached w.p.
    B/(s*n). So recall ~ min(1, B/(s*n)). Monotonically DECREASING in s (the pre-filter phase lives
    at low s).
    """
    sn = max(s * n, 1e-9)
    return float(min(1.0, B / sn))


def post_recall_budget(s: float, B: int, k: int = 10) -> float:
    """Post-filter under budget B: identify budget with the candidate-list size, K = B.

    A graph ANN visits ~K nodes to assemble a K-candidate list, so B distance-evals -> K ~ B. Then
    apply the exact order-statistics recall. Monotonically INCREASING in s (post phase lives high).
    """
    return post_recall_exact(int(B), s, k)


def value_gap(s: float, n: int, B: int, k: int = 10) -> float:
    """V(s) = M(pre,s) - M(post,s). Sign tells which phase wins; V(s*) = 0 is the boundary."""
    return pre_recall_budget(s, n, B, k) - post_recall_budget(s, B, k)


def crossover_selectivity(n: int, B: int, k: int = 10,
                          lo: float = 1e-5, hi: float = 1.0, iters: int = 80) -> float | None:
    """Bisection for s* where V(s*) = 0 (pre/post crossover). None if no sign change on [lo,hi]."""
    f_lo, f_hi = value_gap(lo, n, B, k), value_gap(hi, n, B, k)
    if np.sign(f_lo) == np.sign(f_hi):
        return None
    a, b = lo, hi
    for _ in range(iters):
        m = np.sqrt(a * b)                        # geometric bisection (s is a log-scale quantity)
        fm = value_gap(m, n, B, k)
        if np.sign(fm) == np.sign(f_lo):
            a, f_lo = m, fm
        else:
            b = m
    return float(np.sqrt(a * b))


# --------------------------------------------------------------------------------------
# (C) Decision theory + linear response -> the flip-margin law
# --------------------------------------------------------------------------------------
def value_gap_slope(s_star: float, n: int, B: int, k: int = 10, h: float = 1e-3) -> float:
    """|dV/d ln s| at the boundary. Sharpness of the phase transition in log-selectivity."""
    lo, hi = s_star * np.exp(-h), s_star * np.exp(h)
    return abs((value_gap(hi, n, B, k) - value_gap(lo, n, B, k)) / (2 * h))


def flip_margin(s_star: float, n: int, B: int, k: int = 10) -> float:
    """Half-width (in ln s) of the band where a unit ln-estimation-error is dangerous.

    Regret of a wrong pick ~ |V'(s*)| * |ln s - ln s*| to first order; equivalently a ln-estimation
    error eps flips the choice (and costs the cliff) only within |ln s - ln s*| <~ eps. The margin
    1/|V'(s*)| is the Problem-#1 condition number kappa_flip reappearing as the LOCAL theory of this
    phase boundary. Larger margin = sharper boundary = more dangerous (a small eps suffices to flip).
    """
    slope = value_gap_slope(s_star, n, B, k)
    return float("inf") if slope == 0 else 1.0 / slope


def danger_band_halfwidth(s_star: float, eps: float, n: int, B: int, k: int = 10) -> float:
    """Predicted danger-band half-width in ln s for an estimation error eps = |ln Q|.

    A point at true s is mis-classified by an eps-error selector iff ln s and ln(s*Q) straddle ln s*,
    i.e. |ln s - ln s*| < eps. So to first order the band half-width is simply eps (independent of
    the slope); the slope sets the *height* (regret) of the band, the eps sets its *width*. This is
    the prediction H2c/H3 test against.
    """
    return float(eps)
