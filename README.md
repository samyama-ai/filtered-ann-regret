# filtered-ann-regret

**When does selectivity-estimation error cause plan regret in filtered ANN?**
A reproducible, pre-registered study framing **filtered-ANN strategy selection as a phase-transition
system** — and characterizing *where, how badly, and how universally* a wrong selectivity estimate
hurts. This is a **measurement + characterization** (an honest baseline), not a new index or planner.

> Problem #2 of an open daily-research program. Companion to
> [`ce-metric-eval`](https://github.com/samyama-ai/ce-metric-eval) (Problem #1), whose
> estimation-error → decision-regret lens this work transfers to vector search.

## The one-paragraph claim
A filtered-ANN query `(q, P, k)` returns the `k` nearest vectors among those passing predicate `P`,
of selectivity `s`. The optimal execution strategy — **pre-filter / post-filter / in-filter** — flips
with `s`, so the system must *estimate* `s` and choose. We model this as an `argmax` over a landscape
with **phases** (regions where each strategy wins) separated by **boundaries**, and show that
selectivity-estimation error produces plan regret **only in the critical regions around those
boundaries** — a wedge of ln-width = the estimation error `ε` and height = the local cliff `|V′(s*)|·ε`.
The boundary locations follow from independent physics (**order statistics** for the post cliff
`s≈k/K`; **site percolation** for the in-filter cliff `s_c≈0.83/M`), the regret obeys a
**flip-margin law** (`1/|V′(s*)|`, the condition number from Problem #1 reappearing locally), and the
regret curves obey a **finite-size scaling collapse** across two decades of corpus size. We validate on
synthetic sweeps and real **SIFT1M**, all under pre-registered decision rules.

## Results (all pre-registered; `PREREGISTRATION.md`)
| # | claim | outcome |
|---|---|---|
| **H0** | post-filter recall = order-statistics closed form `E[min(k,Binom(K,s))]/k` | reproduced on real rankings ✓ |
| **H2** | regret is a **critical-region** phenomenon | interior ΔR=0.0005, **~290×** concentration at boundary, width∝ε, peak∝`\|V′\|ε` — at n∈{10⁵,10⁶,10⁷} **and real SIFT1M** ✓ |
| **H3** | **finite-size scaling collapse** of selection regret | one universal V-wedge across 2 decades of n (rmse 0.015, 15.7× spread reduction) ✓ |
| **H1b** | in-filter cliff is a **percolation** transition | **`s_c ≈ 0.83/M`, n-independent** (R²=0.998) — refines the pre-registered `log n/M` ✓ |
| leak | does cost-model mismatch leak regret into interiors? | **real approx HNSW: no leak**; **cost-model *bias*: persistent band** robustness can't fix (two failure modes) |

Emergent findings: criticality requires a **constrained budget `B<√(k·n)`** (else a free-lunch overlap
makes estimation error harmless); a single random predicate on clustered data is **overdispersed** vs
Binomial.

Figures: `figures/criticality_synthetic.png` · `collapse.png` · `percolation.png` · `leak.png`.

## Reproduce (one command each)
```bash
pip install -e . && python -m pytest -q          # 29 tests: closed forms, H0 gate, percolation
python bench/run_h2_synthetic.py                 # H2 criticality + figure
python bench/run_real.py                         # H2 at n∈{1e5,1e6,1e7} + H3 collapse + SIFT anchor
python bench/run_percolation.py                  # H1b percolation s_c ≈ 0.83/M
python bench/run_leak.py                         # model-mismatch leak (approx HNSW + cost-model bias)
bash   bench/fetch_data.sh                        # (optional) download SIFT1M real anchor
```
Full env/provenance in `REPRODUCIBILITY.md`. The n=10⁷ point used a 16-vCPU AWS spot box; everything
else runs on a laptop in seconds–minutes.

## What's prior art, and what's ours (`NOVELTY.md`)
**Prior art (credited, not claimed):** the three strategies and the hard "middle regime" (ACORN,
Filtered-DiskANN, SeRF, UNIFY, big-ann/IfF benchmarks); selectivity-threshold strategy selection,
including hysteresis (Vespa's two-parameter scheme, AlloyDB adaptive filtering, Milvus, Gan & Wang's
learned planner, arXiv:2602.17914); the `M≳log n/s` connectivity folklore.
**Our delta (modest, conceptual):** the *quantitative criticality* + flip-margin law; the **scaling
collapse** (selection regret is scale-invariant — new to vector search); the **constrained-budget law**;
and the model-mismatch result separating the transient ε-wedge from a persistent calibration band.

## Honest limitations
The H2 regret uses a threshold/cost-based selector (realistic; the wedge *support* is then partly
definitional — see the leak test, which shows the wedge is real on approximate ANN and that the larger
danger is model *calibration*). Correlated-predicate sweep (NC2) and the hysteresis rule (H4) are
characterized only partially. This is a characterization of *when accuracy tracks plan quality*, not a
new system.

## License & cite
Apache-2.0 (`LICENSE`). Please cite via `CITATION.cff` and the preprint (link when posted).
Authors: Madhulatha Mandarapu, Sandeep Kunkunuru (VaidhyaMegha Private Limited).
