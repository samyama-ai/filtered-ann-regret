"""Regenerate the preliminary H2 criticality result + figure on synthetic data (one command).

    python bench/run_h2_synthetic.py

Writes results/h2_synthetic.npz and figures/criticality_synthetic.png. Fast (~seconds); no external
data. The SIFT1M / real-ANN version lives in bench/run_h2_sift.py (Stage: real-data validation).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fann_regret import data, experiment as E  # noqa: E402

N, B, K_DEPTH = 20000, 120, 6000


def main():
    base, q = data.synthetic_gaussian(n=N, d=32, nq=400, n_clusters=40, seed=1)
    rank = data.exact_ranking(base, q, depth=K_DEPTH)
    res = E.run_H2(rank, n=N, B=B, k=10, n_pred=20)

    out = Path(__file__).resolve().parent.parent
    (out / "results").mkdir(exist_ok=True)
    np.savez(out / "results" / "h2_synthetic.npz",
             s_grid=res["s_grid"], pre=res["pre"], post=res["post"], s_star=res["s_star"],
             **{f"regret_Q{r['Q']}": r["regret"] for r in res["rows"]})

    summary = (f"s*={res['s_star']:.4f}  |V'|={res['Vprime']:.3f}  "
               f"H2a={res['interior_mean']:.4f}  H2b={res['band_over_interior']:.0f}x  "
               f"H2c_i={res['width_slope_vs_eps']:.3f}  H2c_ii={res['rho_peak_vs_Vp_eps']:.3f}")
    print(summary)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4.2))
        sg = res["s_grid"]
        ax0.plot(sg, res["pre"], label="pre-filter", lw=2)
        ax0.plot(sg, res["post"], label="post-filter", lw=2)
        ax0.axvline(res["s_star"], ls="--", c="k", alpha=.6, label=f"s*={res['s_star']:.3f}")
        ax0.set_xscale("log"); ax0.set_xlabel("selectivity s"); ax0.set_ylabel("recall@10 @ budget B")
        ax0.set_title("Strategy objectives (the two phases)"); ax0.legend()
        for r in res["rows"]:
            if r["Q"] == 1.0:
                continue
            ax1.plot(sg, r["regret"], label=f"Q={r['Q']:.2f} (ε={r['eps']:.2f})")
        ax1.axvline(res["s_star"], ls="--", c="k", alpha=.6)
        ax1.set_xscale("log"); ax1.set_xlabel("true selectivity s")
        ax1.set_ylabel("selection regret ΔR (recall@10 lost)")
        ax1.set_title("Regret is a critical-region phenomenon"); ax1.legend(fontsize=8)
        fig.tight_layout()
        (out / "figures").mkdir(exist_ok=True)
        fig.savefig(out / "figures" / "criticality_synthetic.png", dpi=130)
        print("wrote figures/criticality_synthetic.png")
    except Exception as e:  # noqa: BLE001
        print(f"(figure skipped: {e})")


if __name__ == "__main__":
    main()
