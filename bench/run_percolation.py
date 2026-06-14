"""H1b: locate the in-filter percolation threshold s_c and fit its scaling in (n, M).

Tests whether the graph cliff is a site-percolation transition and which law it follows:
  s_c ~ c / M           (giant-component emergence; n-independent), vs
  s_c ~ c * log n / M    (full navigability; n-dependent).
We fit both and report which the data supports (pre-registration permits reporting the actual law).

    python bench/run_percolation.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from fann_regret import data, graph as G  # noqa: E402

NS = [10_000, 30_000, 100_000]
MS = [8, 16, 24, 32]


def main():
    rows = []
    for n in NS:
        base, _ = data.synthetic_gaussian(n=n, d=32, nq=2, n_clusters=max(20, n // 1000), seed=1)
        for M in MS:
            adj = G.build_knn_graph(base, M, seed=0)
            sc, sg, frac = G.percolation_threshold(adj, n=n, n_pred=2)
            rows.append({"n": n, "M": M, "s_c": sc, "sc_M": sc * M, "sc_M_over_logn": sc * M / np.log(n)})
            print(f"n={n:>7,} M={M:2d}  s_c={sc:.4f}  s_c*M={sc*M:.3f}  s_c*M/ln n={sc*M/np.log(n):.3f}")

    # Fit ln s_c = a + b ln M + g ln(ln n).  b~ -1 => 1/M;  g~ +1 => log n dependence.
    R = [r for r in rows if np.isfinite(r["s_c"])]
    X = np.array([[1.0, np.log(r["M"]), np.log(np.log(r["n"]))] for r in R])
    y = np.array([np.log(r["s_c"]) for r in R])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    ss_res = float(np.sum((y - pred) ** 2)); ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot
    # constancy of the two candidate invariants (lower CV = better law)
    cv = lambda key: float(np.std([r[key] for r in R]) / np.mean([r[key] for r in R]))
    out = {"rows": rows, "fit_lnsc": {"intercept": coef[0], "slope_lnM": coef[1],
            "slope_lnlnn": coef[2], "R2": r2},
           "cv_sc_M": cv("sc_M"), "cv_sc_M_over_logn": cv("sc_M_over_logn")}
    print(f"\nfit ln s_c: slope_lnM={coef[1]:+.2f} (want ~ -1: 1/M)  "
          f"slope_ln(ln n)={coef[2]:+.2f} (~0 => no log n; ~+1 => log n)  R2={r2:.3f}")
    print(f"CV(s_c*M)={out['cv_sc_M']:.3f}  vs  CV(s_c*M/log n)={out['cv_sc_M_over_logn']:.3f}  "
          f"(smaller = better-fitting invariant)")
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "percolation.json").write_text(json.dumps(out, indent=2))
    _plot(rows)


def _plot(rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"(figure skipped: {e})"); return
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4.2))
    for n in sorted(set(r["n"] for r in rows)):
        rr = [r for r in rows if r["n"] == n and np.isfinite(r["s_c"])]
        ax0.plot([r["M"] for r in rr], [r["s_c"] for r in rr], "o-", label=f"n={n:,}")
    Ms = np.array(sorted(set(r["M"] for r in rows)))
    ax0.plot(Ms, 0.83 / Ms, "k--", alpha=.6, label="0.83 / M")
    ax0.set_xlabel("graph degree M"); ax0.set_ylabel("percolation s_c"); ax0.set_xscale("log")
    ax0.set_yscale("log"); ax0.set_title("In-filter cliff is a percolation threshold"); ax0.legend()
    for n in sorted(set(r["n"] for r in rows)):
        rr = [r for r in rows if r["n"] == n and np.isfinite(r["s_c"])]
        ax1.plot([r["M"] for r in rr], [r["sc_M"] for r in rr], "o-", label=f"n={n:,}")
    ax1.set_xlabel("graph degree M"); ax1.set_ylabel("s_c * M"); ax1.set_title("s_c * M ~ const  (s_c ∝ 1/M)")
    ax1.legend(); fig.tight_layout()
    (ROOT / "figures").mkdir(exist_ok=True)
    fig.savefig(ROOT / "figures" / "percolation.png", dpi=130)
    print("wrote figures/percolation.png")


if __name__ == "__main__":
    main()
