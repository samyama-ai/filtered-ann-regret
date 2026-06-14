"""Empirical harness tests (Stage-3 reproduction gate + invariants), fast synthetic data.

H0 at the empirical level: post-filter recall computed on a REAL exact ranking under uncorrelated
labels must match theory.post_recall_exact across the selectivity grid. If it doesn't, the harness
(ranking, eligibility bookkeeping, or recall accounting) is wrong and nothing downstream is trusted.
"""
import numpy as np
import pytest

from fann_regret import data, labels, strategies as S, theory as T

K_DEPTH = 6000


@pytest.fixture(scope="module")
def rank():
    base, queries = data.synthetic_gaussian(n=20000, d=32, nq=400, n_clusters=40, seed=1)
    return data.exact_ranking(base, queries, depth=K_DEPTH, tag="")  # no cache in tests


@pytest.mark.parametrize("s", [0.01, 0.02, 0.05, 0.1, 0.3])
@pytest.mark.parametrize("K", [200, 1000])
def test_H0_post_recall_matches_closed_form(rank, s, K):
    # 'Uncorrelated predicate' baseline = EXPECTED over random predicates: Monte-Carlo over label
    # draws (a single draw is overdispersed vs Binom on clustered data -- a real effect, not a bug).
    vals = [float(S.post_recall_empirical(
                S.elig_from_labels(rank, labels.labels_uncorrelated(20000, s, seed=100 + i)), K, k=10
            ).mean()) for i in range(30)]
    emp = float(np.mean(vals))
    closed = T.post_recall_exact(K, s, k=10)
    assert abs(emp - closed) < 0.03, f"s={s} K={K}: emp {emp:.3f} vs closed {closed:.3f}"


def test_single_predicate_overdispersion(rank):
    """Documented effect: a SINGLE random predicate on clustered data is overdispersed vs Binom, so
    its post-recall sits at/below the predicate-averaged closed form (concavity of min(k,.))."""
    s, K = 0.01, 1000
    single = float(S.post_recall_empirical(
        S.elig_from_labels(rank, labels.labels_uncorrelated(20000, s, seed=2)), K).mean())
    closed = T.post_recall_exact(K, s, 10)
    assert single <= closed + 0.01                 # single draw does not exceed the expectation


def test_post_recall_increases_with_selectivity(rank):
    K = 500
    rec = []
    for s in [0.005, 0.02, 0.05, 0.2, 0.5]:
        lab = labels.labels_uncorrelated(20000, s, seed=3)
        rec.append(float(S.post_recall_empirical(S.elig_from_labels(rank, lab), K).mean()))
    assert np.all(np.diff(rec) >= -0.02)           # monotone up to MC noise


def test_correlated_labels_hurt_post(rank):
    """NC2 direction check: pushing eligible points late (rho<0) lowers post recall vs uncorrelated."""
    s, K = 0.05, 500
    elig_U = S.elig_from_labels(rank, labels.labels_uncorrelated(20000, s, seed=4))
    elig_C = labels.marks_correlated_for_ranking(rank, s, rho=-0.9, seed=4)
    r_U = float(S.post_recall_empirical(elig_U, K).mean())
    r_C = float(S.post_recall_empirical(elig_C, K).mean())
    assert r_C < r_U, f"correlated should hurt: U {r_U:.3f} vs C {r_C:.3f}"


def test_empirical_crossover_exists(rank):
    """A pre/post crossover s* exists in the constrained-budget regime on the empirical surface."""
    n, B, k = 20000, 120, 10                       # B < sqrt(k*n)=447 -> constrained
    ss = np.geomspace(2e-3, 0.6, 25)
    gaps = []
    for s in ss:
        surf = S.objective_surface(rank, labels.labels_uncorrelated(20000, s, seed=5), s, n, B, k)
        gaps.append(surf["pre"] - surf["post"])
    gaps = np.array(gaps)
    assert gaps[0] > 0 and gaps[-1] < 0            # pre wins low s, post wins high s
    assert np.any(np.diff(np.sign(gaps)) != 0)     # sign change -> crossover
