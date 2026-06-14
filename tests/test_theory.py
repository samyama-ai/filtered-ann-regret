"""Correctness tests for the closed-form core (theory.py). No data, fully deterministic.

These are the Stage-3 'correctness + invariant' layer for the analytical predictions, plus a direct
Monte-Carlo simulation of the post-filter PROCESS that independently validates the order-statistics
derivation (this is H0 at the theory level: the closed form must equal a from-scratch simulation).
"""
import numpy as np
import pytest

from fann_regret import theory as T


# ---- (B) order statistics: closed form == direct simulation of the post-filter process ----
@pytest.mark.parametrize("K,s,k", [(200, 0.05, 10), (200, 0.005, 10), (1000, 0.02, 10),
                                   (500, 0.5, 10), (100, 0.9, 10), (2000, 0.01, 5)])
def test_post_recall_matches_simulation(K, s, k):
    rng = np.random.default_rng(0)
    trials = 40000
    # Direct process: rank is a prefix; mark each of the first K points iid Bernoulli(s).
    # True filtered top-k = first k marks globally; within top-K we recover min(k, #marks in top-K).
    marks = rng.random((trials, K)) < s
    E = marks.sum(axis=1)                 # eligible count in top-K
    recall = np.minimum(k, E) / k         # = intersection / k  (derivation in theory.post_recall_exact)
    sim = recall.mean()
    closed = T.post_recall_exact(K, s, k)
    assert abs(sim - closed) < 0.01, f"closed {closed:.4f} vs sim {sim:.4f}"


def test_post_recall_monotone_and_limits():
    s_grid = np.linspace(0.001, 0.999, 50)
    r = T.post_recall_curve(200, s_grid, k=10)
    assert np.all(np.diff(r) >= -1e-9)            # non-decreasing in s
    assert r[-1] > 0.99                            # ~1 at high s
    assert T.post_recall_exact(200, 0.0001, 10) < 0.1   # collapses far below knee


def test_post_knee_near_k_over_K():
    K, k = 1000, 10
    knee_pred = T.post_knee_selectivity(K, k)      # 0.01
    # numeric steepest-descent of recall(s) on a log grid should sit within [0.5,2]x predicted
    s_grid = np.geomspace(1e-4, 1.0, 400)
    r = T.post_recall_curve(K, s_grid, k)
    d = np.gradient(r, np.log(s_grid))
    knee_emp = s_grid[np.argmax(d)]
    assert 0.5 * knee_pred <= knee_emp <= 2.0 * knee_pred, (knee_emp, knee_pred)


# ---- pre-filter budget model ----
def test_pre_recall_budget():
    n, B = 1_000_000, 20_000
    assert T.pre_recall_budget(0.01, n, B) == 1.0          # s*n = 10k <= B -> exact
    assert T.pre_recall_budget(0.5, n, B) == pytest.approx(B / (0.5 * n))  # 20k/500k
    # monotonically non-increasing in s
    ss = np.geomspace(1e-4, 1.0, 30)
    pr = [T.pre_recall_budget(s, n, B) for s in ss]
    assert np.all(np.diff(pr) <= 1e-9)


# ---- crossover + value gap: pre wins low-s, post wins high-s, boundary in between ----
def test_crossover_and_value_gap_sign():
    # Constrained-budget regime B < sqrt(k*n) so pre/post genuinely trade off (no free-lunch overlap).
    n, B, k = 1_000_000, 2_000, 10        # sqrt(k*n) = 3162 > B  -> constrained
    s_star = T.crossover_selectivity(n, B, k)
    assert s_star is not None and 1e-4 < s_star < 1.0
    assert T.value_gap(s_star * 0.3, n, B, k) > 0          # below boundary: pre > post
    assert T.value_gap(min(s_star * 3, 0.99), n, B, k) < 0 # above boundary: post > pre


def test_free_lunch_vs_constrained_regime():
    """Criticality requires a constrained budget: B < sqrt(k*n). Else pre and post both saturate
    to ~1 over an overlap band [B/n, k/B] and strategy choice (hence estimation error) is harmless."""
    n, k = 1_000_000, 10
    thresh = np.sqrt(k * n)                                  # ~3162
    # Generous budget -> overlap band where BOTH strategies are ~1.
    B_free = 20_000
    s_mid = np.sqrt((B_free / n) * (k / B_free))            # geo-mean of the two knees
    assert T.pre_recall_budget(s_mid, n, B_free, k) > 0.99
    assert T.post_recall_budget(s_mid, B_free, k) > 0.99
    # Constrained budget -> a gap where neither is good (the literature's 'middle regime').
    B_tight = 1_500                                          # < thresh
    s_gap = np.sqrt((B_tight / n) * (k / B_tight))
    assert min(T.pre_recall_budget(s_gap, n, B_tight, k),
               T.post_recall_budget(s_gap, B_tight, k)) < 0.95


# ---- (C) flip margin finite and positive at a real boundary ----
def test_flip_margin_positive_finite():
    n, B = 1_000_000, 2_000
    s_star = T.crossover_selectivity(n, B)
    fm = T.flip_margin(s_star, n, B)
    assert 0 < fm < np.inf


# ---- (A) percolation threshold scaling: s_c * M / log(n) is constant in (n, M) ----
def test_percolation_scaling_constant():
    c = 1.3
    vals = [T.percolation_sc(n, M, c) * M / np.log(n)
            for n in (1e5, 1e6, 1e7) for M in (8, 16, 32)]
    assert np.allclose(vals, c, rtol=1e-9)
