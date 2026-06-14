# HYPOTHESIS — Problem #2 (PRE-REGISTERED, FROZEN)

> **Frozen: 2026-06-14**, before any results are computed. No hypothesis, threshold, or decision
> rule below may be edited after data exists — only annotated in RESULTS. (Same discipline as
> Problem #1.) Derives from [FIRST-PRINCIPLES.md](./FIRST-PRINCIPLES.md); grounded in
> [BRIEF.md](./BRIEF.md). Lever (user-chosen): **selectivity-estimation-*error* robustness of
> filter-strategy selection, framed as criticality of a phase-transition system** — lean in, scaling
> collapse included.

## AMENDMENT 1 (2026-06-14, **pre-data**, theory-driven — no results computed yet)

While deriving `theory.py` (before any experiment ran) the first-order analysis sharpened H2c/H3.
A threshold selector picks `pre` iff `ŝ = s·Q < s*`, so it mis-picks iff `ln s` and `ln s + ln Q`
straddle `ln s*` ⇒ the **flip-region ln-width is exactly `ε = |ln Q|`** (slope-independent); the
value-gap slope `|V′(s*)|` sets the regret **height**, not the width. The original H2c (`w ≈ ε/|V′|`)
conflated width and height. Corrected H2c/H3 below; **original text struck, not deleted** (integrity
trail). This is legitimate pre-registration practice: the change is *pre-data* and derived from the
model, not chosen to fit results. The collapse prediction is now *sharper* (a named exponent α=1).

## Thesis under test

Filtered-ANN strategy selection is a **phase-transition system**: selectivity `s` is the order
parameter; `{pre, post, in}` are phases; selectivity-estimation error causes plan regret **only in
the critical regions around the phase boundaries**, whose locations follow from **percolation** (graph
cliff) and **order statistics** (post cliff), whose regret obeys a **flip-margin law**
(`= 1/|value-gap slope|`, the Problem-#1 condition number reappearing locally), and whose danger-band
width obeys a **finite-size scaling collapse**.

## Frozen experimental setup

- **Corpora:** SIFT1M (1M × 128-d, ℓ₂) = controlled core. Finite-size sweep `n ∈ {10⁵, 10⁶, 10⁷}`
  via SIFT subsample (100K), full SIFT1M, and a 10M set (Deep10M *or* big-ann YFCC-10M slice).
  **External validity:** big-ann YFCC-10M with its *real* tags/labels.
- **Synthetic predicate labels (controlled):** target selectivity grid
  `s ∈ {.001,.002,.005,.01,.02,.05,.1,.2,.35,.5,.7,.9,.95}`. Two correlation models:
  **U** = each point eligible iid w.p. `s`; **C** = eligibility correlated with distance-to-query rank
  via tunable coupling (adversary pushes eligible points later in the ranking — the OT-adversary at
  fixed `s`).
- **Strategies:** **pre** (exact within `X_P`; recall 1, cost ∝ `s·n`), **post** (unfiltered ANN
  top-`K`, drop violators), **in** (filter-aware graph traversal: hnswlib filter callback / Qdrant).
- **Primary objective `M`:** **recall@10 at a fixed compute budget `B`**, `B` = distance-evaluation
  budget (hardware-independent) set so *unfiltered* HNSW reaches recall@10 ≈ 0.90; `B` recorded once at
  setup then **fixed**. Secondary (real-data only): **QPS at recall@10 ≥ 0.90** on YFCC-10M.
- **Estimation-error model:** `ŝ = clip(s · Q, 0, 1)`, `Q` = **multiplicative** error (a *q-error*,
  deliberately the Problem-#1 metric) on a frozen grid `Q ∈ {1, 1.25, 1.5, 2, 3}` applied both
  directions (over/under-estimate). `ε ≡ |ln Q|`.
- **Selector:** `a_ŝ = argmax_a M̂(a, ŝ)` from the closed-form/model surface `M̂`; **oracle** uses true
  `s`. **Regret (primary):** additive recall gap `ΔR = M(a*(s),s) − M(a_ŝ(ŝ),s) ∈ [0,1]` (bounded,
  no divide-by-zero). Secondary multiplicative regret on the QPS objective.
- **Frozen knobs:** `k = 10`; HNSW degree `M ∈ {8,16,32}` (for percolation scaling); post-filter `K`
  grid spanning the predicted knee; ≥ **1000 queries per cell**; bootstrap **95%** CIs over queries;
  **50/50 held-out query split** for any fitted rule.

---

## H0 — Reproduction gate (NOT novel; must pass or we stop)

The post-filter recall curve on SIFT1M+U matches the order-statistics closed form
`recall@10(K,s) = P(Binomial(K,s) ≥ 10)` with **mean abs error ≤ 0.05** across the `s` grid; and the
unfiltered baseline hits recall@10 ∈ [0.88, 0.92] at budget `B`.
**Decision:** fail ⇒ harness broken ⇒ **Gate B stop / fix before proceeding** (no downstream claims).

## H1 — Two phase boundaries have closed-form locations

- **H1a (post cliff, order statistics):** the recall@10(s) knee (steepest-descent `s`) lies within
  `[0.5, 2] × (k/K)`.
- **H1b (graph cliff, percolation):** in-filter recall collapses at `s_c`, and `s_c` follows
  `s_c ≈ c · (log n)/M`: fitting `s_c` across the `(n, M)` grid, the law `s_c·M/log n = const` holds
  with **R² ≥ 0.8**, and a competing constant-`s_c` (n,M-independent) model is rejected (non-overlapping
  fit).
**Decision:** H1a and H1b each independently confirmed/refused on the above thresholds.

## H2 — Regret is a critical-region phenomenon (CORE BET)

Let `s*` = crossover where the optimal strategy flips (pre↔post, and/or post↔in). Define the measured
**danger band** = `{s : ΔR(s) > τ_safe}`, `τ_safe = 0.02`. Pre-registered claims:
- **H2a (safe interiors):** mean `ΔR` over phase interiors (`|ln s − ln s*| > 2w`) `< τ_safe`, with
  bootstrap 95% CI **upper** bound `< τ_safe`.
- **H2b (peak at boundary):** mean `ΔR` inside the danger band `> 5×` the interior mean, CIs separated.
- ~~**H2c (flip-margin law):** the predicted band half-width `w(Q) ≈ ε / |dV/d ln s|(s*)` ... correlates
  with the measured danger-band half-width across the `Q` grid at Spearman ρ ≥ 0.8.~~ **[struck — Amendment 1]**
- **H2c′ (flip-margin law, corrected):** with value gap `V = M(pre,·) − M(post,·)` and `s*` its zero,
  (i) the measured **flip-region ln-width grows linearly in `ε = |ln Q|` with slope ≈ 1** (regress
  width on `ε`: fitted slope ∈ [0.7, 1.3], R² ≥ 0.8); and (ii) the **peak regret height scales as
  `|V′(s*)|·ε`** — across the `Q` grid, measured peak `ΔR` vs `|V′(s*)|·ε` has Spearman **ρ ≥ 0.8**.
**Decision:** H2 **confirmed** iff H2a ∧ H2b ∧ H2c all hold. This is the primary scientific claim.

## H3 — Finite-size scaling collapse (HIGH-RISK HEADLINE; **stretch, null is honest**)

Rescaling **abscissa** `x = (ln s − ln s*) / w(Q,n)` **and ordinate** `ΔR / (|V′(s*)|·ε)` collapses
`ΔR(s; Q, n)` across all `(Q,n)` cells onto a single universal wedge `g(x)`. Pre-registered abscissa
form `w(Q,n) = A · ε^α · (log n)^{−β}`; **Amendment 1 sharpens the prediction to `α = 1` (abscissa
scale ≈ ε) and the n-dependence carried by the ordinate scale `|V′(s*;n)|`**. Fit `(A,α,β)` on a
**training** subset of `(Q,n)` cells, evaluate on **held-out** cells. (A confirmed `α ≈ 1` is a
stronger result than a free-exponent fit.)
**Decision:** **confirmed** iff held-out cells collapse with `g`-fit RMSE ≤ 0.05 **and** the inter-cell
curve spread shrinks ≥ 5× vs unscaled. **Pre-registered as STRETCH:** a null (no clean collapse) is a
legitimate, reportable finding and does **not** sink the result.

## H4 — Mode-(a) upside: hysteresis robust rule beats naive-ŝ

A **hysteresis** selector (catastrophe-theoretic optimal control of the fold: stay on current strategy;
switch only when `ŝ` crosses `s* ± h`, `h` derived from cliff heights + `ε`) achieves **lower mean
`ΔR` than naive `argmax M̂(·,ŝ)`** in the danger band, at `≤ τ_cost = 0.01` mean recall cost in
interiors.
**Decision:** **mode-(a) win** iff danger-band mean `ΔR` reduced by **≥ Δ = 0.03** with bootstrap 95%
CI excluding 0 on the **held-out** split; else report **negative** (rule didn't beat naive) — still a
complete, shippable result.

## Negative controls (leak detectors — pre-registered)

- **NC1 (oracle):** with `Q = 1` (`ŝ = s`), every selector achieves `ΔR ≈ 0` (mean < 0.005)
  everywhere. Violation ⇒ harness leak ⇒ stop.
- **NC2 (correlation direction):** correlated labels **C** must *worsen* the post cliff / shift `s*` and
  *increase* danger-band regret vs **U** (effect in the predicted direction). Opposite ⇒ model wrong.
- **NC3 (information use):** the model selector must beat a random-strategy selector in interiors; if a
  random selector "wins," the harness is leaking.

## Statistics & honesty

- ≥1000 queries/cell; bootstrap 95% CIs over queries; 50/50 held-out split for all fitted rules
  (H3 `(A,α,β)`, H4 `h`). Thresholds `τ_safe=0.02`, 5×, `ρ≥0.8`, RMSE≤0.05, `Δ=0.03`, `τ_cost=0.01`
  are **frozen here**.
- **Mode (frozen):** primary **(b)-with-teeth** = harness + H0/H1/H2 structural characterization is the
  shippable floor. **(a)** = H4 (+ H3 collapse as bonus). A confirmed H2 with null H3/H4 still ships.

## Day-1 critical path vs stretch (scope-down rule: cut hypotheses, not rigor)

- **Day-1 solid core (must):** H0 gate · H1a (post cliff) · **H2** (criticality on SIFT1M, single `n`,
  U-labels) · NC1.
- **Stretch / subsequent days:** H1b (percolation `n,M` scaling) · **H3** (collapse, full `n`-sweep,
  needs mini compute) · H4 (hysteresis) · NC2/C-labels · YFCC-10M external validity & QPS secondary.

## Novelty TODO (resolve in NOVELTY.md before any "new" claim)

Check the *phase-diagram + criticality-of-regret + flip-margin-for-strategy-selection + scaling-collapse*
framing against: ACORN, Filtered-DiskANN, SeRF, UNIFY, **arXiv 2602.17914** (full text), AlloyDB
adaptive filtering, and the percolation-on-ANN-graphs literature. Percolation *connectivity* threshold
is known folklore — our candidate-novel pieces are (i) using it + order-stats to **locate strategy
boundaries**, (ii) **regret = criticality** with the flip-margin law, (iii) the **scaling collapse**.
