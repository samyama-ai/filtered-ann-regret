"""H2 experiment: is strategy-selection regret a critical-region phenomenon?

Selector model: a planner with a cost model picks argmax over {pre, post} evaluated at its estimate
ŝ = clip(s*Q). Because model_pre is decreasing and model_post increasing in s (crossing at s*), this
is equivalently 'pick pre iff ŝ < s*'. The oracle picks at the true s. They disagree iff s and ŝ
straddle s*, and the regret is then the empirical value gap |M_emp(pre,s) - M_emp(post,s)| at the
true s. We measure where (in s) regret concentrates, and how the band scales with ε = |ln Q|.
"""
from __future__ import annotations

import numpy as np

from . import labels, strategies as S, theory as T


def empirical_surfaces(rank_ids, n, s_grid, B, k=10, n_pred=20, seed0=100):
    """Predicate-averaged empirical recall for pre and post over the selectivity grid."""
    pre = np.array([S.pre_recall(s, n, B, k) for s in s_grid])
    post = np.empty(len(s_grid))
    for i, s in enumerate(s_grid):
        vals = [S.post_recall_empirical(
                    S.elig_from_labels(rank_ids, labels.labels_uncorrelated(n, s, seed0 + j)), B, k
                ).mean() for j in range(n_pred)]
        post[i] = float(np.mean(vals))
    return pre, post


def regret_curve(s_grid, pre, post, s_star, Q):
    """ΔR(s; Q): the value gap |pre-post| where the selector's estimate straddles s*, else 0."""
    out = np.zeros(len(s_grid))
    for i, s in enumerate(s_grid):
        s_hat = float(np.clip(s * Q, 0.0, 1.0))
        a_star_pre = s < s_star
        a_sel_pre = s_hat < s_star
        if a_star_pre != a_sel_pre:                 # straddle -> wrong pick
            out[i] = abs(pre[i] - post[i])
    return out


def run_H2(rank_ids, n, B, k=10, n_pred=20,
           s_grid=None, Q_grid=(1.0, 1.25, 1.5, 2.0, 3.0), tau_safe=0.02):
    """Returns the criticality summary: s*, per-Q regret curves, danger-band stats, scaling fits."""
    if s_grid is None:
        s_grid = np.geomspace(1e-3, 0.8, 90)
    s_star = T.crossover_selectivity(n, B, k)
    pre, post = empirical_surfaces(rank_ids, n, s_grid, B, k, n_pred)
    Vp = T.value_gap_slope(s_star, n, B, k)         # |dV/d ln s| at boundary (model)

    rows, widths1, peaks, eps_list = [], [], [], []
    for Q in Q_grid:
        if Q == 1.0:
            dR = np.zeros(len(s_grid))               # oracle estimate -> zero regret (NC1)
            dR1 = dR
        else:
            dR1 = regret_curve(s_grid, pre, post, s_star, Q)        # one-sided (over-estimate)
            # symmetric over/under-estimation for the mean stats: average the two directions
            dR = 0.5 * (dR1 + regret_curve(s_grid, pre, post, s_star, 1.0 / Q))
        eps = abs(np.log(Q))
        band2 = s_grid[dR > tau_safe]
        width2 = (np.log(band2.max()) - np.log(band2.min())) if band2.size >= 2 else 0.0
        band1 = s_grid[dR1 > tau_safe]
        width1 = (np.log(band1.max()) - np.log(band1.min())) if band1.size >= 2 else 0.0
        rows.append({"Q": Q, "eps": eps, "regret": dR, "band_width": width2,
                     "band_width_1sided": width1, "peak": float(dR.max())})
        if Q != 1.0:
            widths1.append(width1); peaks.append(float(dR.max())); eps_list.append(eps)

    interior_mask = np.ones(len(s_grid), bool)
    for r in rows:                                   # interior = never in any danger band
        interior_mask &= (r["regret"] <= tau_safe)
    interior_mean = float(np.mean([r["regret"][interior_mask].mean() if interior_mask.any() else 0
                                   for r in rows if r["Q"] != 1.0]))
    band_mean = float(np.mean([r["regret"][r["regret"] > tau_safe].mean()
                               for r in rows if r["Q"] != 1.0 and (r["regret"] > tau_safe).any()]))

    eps_arr, w_arr, p_arr = np.array(eps_list), np.array(widths1), np.array(peaks)
    width_slope = float(np.polyfit(eps_arr, w_arr, 1)[0]) if len(eps_arr) >= 2 else float("nan")  # one-sided
    # peak vs |V'|*eps correlation (Spearman)
    from scipy.stats import spearmanr
    rho_peak = float(spearmanr(p_arr, Vp * eps_arr).correlation) if len(eps_arr) >= 2 else float("nan")

    return {
        "s_star": s_star, "Vprime": Vp, "s_grid": s_grid, "pre": pre, "post": post, "rows": rows,
        "interior_mean": interior_mean, "band_mean": band_mean,
        "band_over_interior": (band_mean / interior_mean) if interior_mean > 0 else float("inf"),
        "width_slope_vs_eps": width_slope, "rho_peak_vs_Vp_eps": rho_peak,
    }


def collapse_points(res):
    """Rescale a run's regret curves to the universal coordinates (Amendment-1 prediction):
    x = (ln s - ln s*) / eps,  y = ΔR / (|V'(s*)| * eps).  Returns stacked (x, y) over all Q!=1."""
    s_star, Vp = res["s_star"], res["Vprime"]
    xs, ys = [], []
    for r in res["rows"]:
        if r["Q"] == 1.0 or r["eps"] == 0:
            continue
        x = (np.log(res["s_grid"]) - np.log(s_star)) / r["eps"]
        y = r["regret"] / (Vp * r["eps"] + 1e-12)
        xs.append(x); ys.append(y)
    return np.concatenate(xs), np.concatenate(ys)


def _binned_scatter(X, Y, x_bins):
    """Normalized within-bin scatter of Y(X) about its per-bin mean: RMSE_resid / std(bin-means).
    ~0 => points lie on one tight curve; ~1 => no curve structure. Also returns the raw resid RMSE."""
    m = np.isfinite(X) & np.isfinite(Y) & (X >= x_bins[0]) & (X <= x_bins[-1])
    X, Y = X[m], Y[m]
    idx = np.digitize(X, x_bins)
    resid, means = [], []
    for b in np.unique(idx):
        yb = Y[idx == b]
        if yb.size >= 3:
            means.append(float(np.mean(yb))); resid.append(yb - means[-1])
    if not resid:
        return float("nan"), float("nan"), 0
    rr = np.concatenate(resid)
    rmse = float(np.sqrt(np.mean(rr ** 2)))
    signal = float(np.std(means)) if len(means) > 1 else float("nan")
    return rmse, (rmse / signal if signal else float("nan")), int(X.size)


def collapse_quality(results_by_n, x_bins=None):
    """H3 test: do per-n, per-Q regret curves collapse onto ONE universal g(x)?

    Compares the normalized within-bin scatter (scatter/signal) of the curves BEFORE rescaling
    (x = ln s - ln s*, y = ΔR) versus AFTER (x = (ln s-ln s*)/ε, y = ΔR/(|V'|ε)). A good collapse
    makes the after-scatter small. Frozen rule: confirmed if collapse_rmse <= 0.05 (scaled-y units,
    signal ~ O(1)) AND spread_reduction = unscaled_norm_scatter / scaled_norm_scatter >= 5.
    """
    if x_bins is None:
        x_bins = np.linspace(-3, 3, 31)
    # scaled (collapsed) coordinates
    sx, sy = [], []
    for res in results_by_n:
        x, y = collapse_points(res); sx.append(x); sy.append(y)
    SX, SY = np.concatenate(sx), np.concatenate(sy)
    rmse_s, norm_s, npts = _binned_scatter(SX, SY, x_bins)
    # unscaled coordinates: x = raw ln-distance to boundary, y = raw ΔR
    ux, uy = [], []
    for res in results_by_n:
        ls = np.log(res["s_grid"]) - np.log(res["s_star"])
        for r in res["rows"]:
            if r["Q"] == 1.0:
                continue
            ux.append(ls); uy.append(r["regret"])
    UX, UY = np.concatenate(ux), np.concatenate(uy)
    _, norm_u, _ = _binned_scatter(UX, UY, x_bins)
    return {"collapse_rmse": rmse_s, "scaled_norm_scatter": norm_s,
            "unscaled_norm_scatter": norm_u,
            "spread_reduction": (norm_u / norm_s) if norm_s else float("nan"),
            "n_points": npts}
