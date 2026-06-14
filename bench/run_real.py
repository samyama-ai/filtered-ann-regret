"""Real / large-scale validation: H2 criticality at scale + H3 finite-size scaling collapse.

Design: the n-sweep is a CONTROLLED family (synthetic Gaussian, only n changes) so the collapse test
is clean; SIFT1M is the real-geometry anchor at n=1e6. Runs unattended; caches exact rankings.

    python bench/run_real.py                 # default sweep n in {1e5,1e6,1e7} + SIFT1M if present
    FANN_NS=100000,1000000 python bench/run_real.py     # override n list (skip the heavy 1e7)

Writes results/real_<n>.npz, results/collapse.json, figures/collapse.png. The 1e7 exact ranking is
the compute that wants a bigger box (>=16 GB RAM); 1e5/1e6 run on a laptop.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from fann_regret import data, experiment as E, theory as T  # noqa: E402

K = 10
DEFAULT_NS = [100_000, 1_000_000, 10_000_000]
CACHE = ROOT / "data" / "rankings"


def budget_for(n):
    """Constrained-regime budget: comfortably below sqrt(k*n) so the phases contend."""
    return max(K + 5, int(0.4 * np.sqrt(K * n)))


def depth_for(n, B, s_min=0.002):
    return int(min(n, max(2 * B, K / s_min, 8000)))


def run_one(base, queries, n, tag, n_pred=10):
    B = budget_for(n)
    depth = depth_for(n, B)
    # Centre the selectivity grid on this n's crossover s* (pure-theory, no data) so the danger band
    # is well-resolved at every scale -- avoids s* landing on the grid edge at large n.
    s_star0 = T.crossover_selectivity(n, B, K) or np.sqrt(K / n)
    s_grid = np.geomspace(max(1e-6, s_star0 / 30), min(0.85, s_star0 * 30), 90)
    # depth need only cover the budget B (post examines the top-B); B << depth_for already.
    t0 = time.time()
    rank = data.exact_ranking(base, queries, depth=depth, cache=CACHE, tag=tag)
    res = E.run_H2(rank, n=n, B=B, k=K, n_pred=n_pred, s_grid=s_grid)
    res["_meta"] = {"n": n, "B": B, "depth": depth, "tag": tag,
                    "rank_secs": round(time.time() - t0, 1)}
    print(f"[{tag}] n={n:,} B={B} depth={depth}  s*={res['s_star']:.4f}  "
          f"H2a={res['interior_mean']:.4f}  H2b={res['band_over_interior']:.0f}x  "
          f"H2c_i={res['width_slope_vs_eps']:.3f}  H2c_ii={res['rho_peak_vs_Vp_eps']:.3f}  "
          f"({res['_meta']['rank_secs']}s)")
    return res


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    (ROOT / "results").mkdir(exist_ok=True)
    ns = [int(x) for x in os.environ.get("FANN_NS", "").split(",") if x] or DEFAULT_NS

    results_by_n = []
    for n in ns:
        nq = 500 if n >= 1_000_000 else 800
        base, queries = data.synthetic_gaussian(n=n, d=32, nq=nq, n_clusters=max(40, n // 20000),
                                                seed=1)
        res = run_one(base, queries, n, tag=f"syn{n}")
        results_by_n.append(res)
        np.savez(ROOT / "results" / f"real_{n}.npz",
                 s_grid=res["s_grid"], pre=res["pre"], post=res["post"], s_star=res["s_star"],
                 Vprime=res["Vprime"], **{f"regret_Q{r['Q']}": r["regret"] for r in res["rows"]})

    # H3 collapse across the controlled n-sweep
    if len(results_by_n) >= 2:
        col = E.collapse_quality(results_by_n)
        col["ns"] = ns
        print(f"[H3 collapse] rmse={col['collapse_rmse']:.4f} (want<=0.05)  "
              f"spread_reduction={col['spread_reduction']:.1f}x (want>=5)  pts={col['n_points']}")
        (ROOT / "results" / "collapse.json").write_text(json.dumps(col, indent=2))
        _plot_collapse(results_by_n)

    # SIFT1M real-geometry anchor (if downloaded)
    sift_dir = data.BASE / "sift"
    if (sift_dir / "sift_base.fvecs").exists():
        base, queries = data.load_sift1m()
        q = queries[:500]
        res = run_one(base, q, n=base.shape[0], tag="sift1m")
        np.savez(ROOT / "results" / "real_sift1m.npz",
                 s_grid=res["s_grid"], pre=res["pre"], post=res["post"],
                 s_star=res["s_star"], Vprime=res["Vprime"],
                 **{f"regret_Q{r['Q']}": r["regret"] for r in res["rows"]})
    else:
        print("(SIFT1M not present -- run bench/fetch_data.sh to add the real-geometry anchor)")


def _plot_collapse(results_by_n):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"(collapse figure skipped: {e})"); return
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4.2))
    for res in results_by_n:
        n = res["_meta"]["n"]
        for r in res["rows"]:
            if r["Q"] == 1.0:
                continue
            ax0.plot((np.log(res["s_grid"]) - np.log(res["s_star"])), r["regret"], alpha=.5)
        x, y = E.collapse_points(res)
        ax1.scatter(x, y, s=6, alpha=.4, label=f"n={n:,}")
    ax0.set_xlabel("ln s - ln s*"); ax0.set_ylabel("ΔR"); ax0.set_title("Raw (un-scaled) regret")
    ax0.set_xlim(-3, 3)
    ax1.set_xlabel("(ln s - ln s*) / ε"); ax1.set_ylabel("ΔR / (|V'| ε)")
    ax1.set_title("Finite-size scaling collapse"); ax1.set_xlim(-3, 3); ax1.legend(fontsize=8)
    (ROOT / "figures").mkdir(exist_ok=True)
    fig.tight_layout(); fig.savefig(ROOT / "figures" / "collapse.png", dpi=130)
    print("wrote figures/collapse.png")


if __name__ == "__main__":
    main()
