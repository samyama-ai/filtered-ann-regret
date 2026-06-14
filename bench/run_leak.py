"""Leak test (EXPLORATORY, beyond frozen H2): does cost-model mismatch leak regret into phase interiors?

Two error sources, contrasted:
  (1) per-query estimation noise eps  -> the H2 wedge: transient, AT the boundary, vanishes with better
      estimates and is what hysteresis/robustness fixes.
  (2) systematic model mismatch       -> a PERSISTENT miscalibration band between the model's crossover
      s*_model and the true crossover s*_real. It does NOT vanish with a perfect per-query estimate;
      robustness to eps cannot fix it. Here s*_real is shifted by (a) a REAL approximate HNSW whose
      recall differs from the analytic model, and (b) a controlled cost-model bias b.

    python bench/run_leak.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from fann_regret import ann, data, labels, strategies as S, theory as T  # noqa: E402

N, K = 200_000, 10


def crossover(s_grid, pre, post):
    g = pre - post
    fin = np.isfinite(g)                                       # NaN-safe (post can be NaN at low s)
    s_grid, g = s_grid[fin], g[fin]
    sc = np.where(np.diff(np.sign(g)) != 0)[0]
    if sc.size == 0:
        return float("nan")
    i = sc[0]
    x0, x1 = np.log(s_grid[i]), np.log(s_grid[i + 1])
    return float(np.exp((x0 + x1) / 2))


def leak_regret(s_grid, pre, post_real, s_star_model):
    """Regret at eps=0 of a selector that uses s_star_model while reality is (pre, post_real)."""
    out = np.zeros(len(s_grid))
    for i, s in enumerate(s_grid):
        chosen = pre[i] if s < s_star_model else post_real[i]   # selector's pick, valued under reality
        out[i] = max(pre[i], post_real[i]) - chosen
    return out


def main():
    B = max(K + 5, int(0.4 * np.sqrt(K * N)))                  # constrained budget
    base, q = data.synthetic_gaussian(n=N, d=32, nq=400, n_clusters=200, seed=1)
    rank = data.exact_ranking(base, q, depth=20000)            # deep enough to define truth at low s
    idx = ann.build_hnsw(base, M=16, ef_construction=200)
    s_grid = np.geomspace(T.crossover_selectivity(N, B, K) / 30,
                          min(0.85, T.crossover_selectivity(N, B, K) * 30), 60)

    pre = np.array([S.pre_recall(s, N, B, K) for s in s_grid])
    post_model = np.array([T.post_recall_exact(B, s, K) for s in s_grid])     # what the planner believes
    post_real = np.empty(len(s_grid))                                          # real approximate HNSW
    for i, s in enumerate(s_grid):
        vals = [ann.approx_post_recall(idx, q, rank, labels.labels_uncorrelated(N, s, 300 + j), B, K)
                for j in range(4)]
        post_real[i] = float(np.nanmean(vals))

    post_real = np.nan_to_num(post_real, nan=0.0)              # post recall ~0 where truth undefined (low s)
    s_model = crossover(s_grid, pre, post_model)
    s_real = crossover(s_grid, pre, post_real)
    leak = leak_regret(s_grid, pre, post_real, s_model)
    interior = leak[(np.abs(np.log(s_grid) - np.log(s_real)) > 1.0)]           # away from the boundary
    out = {"B": B, "s_star_model": s_model, "s_star_real": s_real,
           "ln_shift": float(np.log(s_real) - np.log(s_model)),
           "leak_peak": float(np.nanmax(leak)),
           "leak_interior_mean": float(np.nanmean(interior)),
           "leak_band_frac": float(np.mean(leak > 0.02))}
    print(f"B={B}  s*_model={s_model:.4f}  s*_real={s_real:.4f}  ln-shift={out['ln_shift']:+.3f}")
    print(f"  leak peak={out['leak_peak']:.3f}  interior-mean={out['leak_interior_mean']:.4f}  "
          f"band-frac(>0.02)={out['leak_band_frac']:.2f}")

    # controlled cost-model bias b: s*_model := s*_real*(1+b); leak band grows with |b|.
    bias_rows = []
    for b in (-0.4, -0.2, 0.0, 0.2, 0.4):
        sm = s_real * (1 + b)
        lk = leak_regret(s_grid, pre, post_real, sm)
        bias_rows.append({"b": b, "peak": float(np.nanmax(lk)),
                          "mean": float(np.nanmean(lk)),
                          "band_frac": float(np.mean(lk > 0.02))})
        print(f"  bias b={b:+.2f}: leak peak={bias_rows[-1]['peak']:.3f} band-frac={bias_rows[-1]['band_frac']:.2f}")
    out["bias_sweep"] = bias_rows
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "leak.json").write_text(json.dumps(out, indent=2))
    # illustrate the leak with a representative +30% cost-model bias (the real-ANN case is null)
    leak_biased = leak_regret(s_grid, pre, post_real, s_real * 1.3)
    _plot(s_grid, pre, post_model, post_real, leak_biased, s_real, s_real * 1.3)


def _plot(s_grid, pre, post_model, post_real, leak, s_real, s_biased):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"(figure skipped: {e})"); return
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(11, 4.2))
    a0.plot(s_grid, pre, label="pre (exact)", lw=2)
    a0.plot(s_grid, post_model, "--", label="post: analytic MODEL", lw=2)
    a0.plot(s_grid, post_real, label="post: real approx HNSW", lw=2)
    a0.axvline(s_real, ls=":", c="C2", label=f"s*_real=s*_model={s_real:.4f}")
    a0.set_xscale("log"); a0.set_xlabel("selectivity s"); a0.set_ylabel("recall@10 @ B")
    a0.set_title("Real approx HNSW tracks the model (no leak)"); a0.legend(fontsize=7)
    a1.plot(s_grid, leak, c="crimson", lw=2)
    a1.axvspan(min(s_real, s_biased), max(s_real, s_biased), color="crimson", alpha=.15)
    a1.set_xscale("log"); a1.set_xlabel("true selectivity s")
    a1.set_ylabel("leak regret at ε=0 (perfect estimate)")
    a1.set_title("Cost-model bias (+30%) → persistent leak band")
    fig.tight_layout(); (ROOT / "figures").mkdir(exist_ok=True)
    fig.savefig(ROOT / "figures" / "leak.png", dpi=130); print("wrote figures/leak.png")


if __name__ == "__main__":
    main()
