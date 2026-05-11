# Phase C Test Plan — Pre-flight + Production-Realism Diagnostics

**Purpose.** Before burning ~15 hours of cloud A100 time (3 variants × 14 OOS years × ~5h base × seed ensemble = ~75 GPU-hr), define exactly what we'll measure, what counts as "passing," and what we do if a test fails. Everything here is grounded in Chincarini-Kim *QEPM* (Ch 4, 8, 11, 12) and Lopez de Prado *AFML* (Ch 7, 13, 14, 16) — these are the standard PM-side and ML-side defensibility frameworks.

The output of Phase C is **not just a Sharpe number** — it's a defensibility dossier we can hand to a multistrat PM.

---

## 0. Pre-flight (CPU, before any GPU spend)

| # | Test | What | Pass criterion |
|---|---|---|---|
| 0.1 | `py_compile` all 5 scripts | already done in Phase B6 | ✓ all clean |
| 0.2 | `--smoke` end-to-end on a single year, 2 epochs, top-200 | exercises data load → train → eval → CSV write path | runs to completion in <10 min on CPU |
| 0.3 | Tensor channel slicing | run `tcnn_attention.py --in-ch 3` on a 7-channel tensor; verify shape `(N, 3, 252)` is what hits the encoder | first conv input is 3 channels |
| 0.4 | Determinism check | run smoke twice with `--seed 0`; compare `all_daily_returns.csv` byte-by-byte | files identical |
| 0.5 | Bug-fix sanity | with patched `data_download.py`, recompute `ret_norm` on a tiny synthetic series — confirm it uses `vol_ewma_lag` (BUG-1 fix) | new `ret_norm[t]` matches `ret[t]/vol_ewma[t-1]` |

**Rule:** if any 0.x fails, **do not provision the cloud GPU.** Fix locally first.

---

## 1. Reproduction validation (does the rerun match paper Table 1?)

The paper reports 0.53 / 0.61 / 0.65 Sharpe for fixed pool / single-head attn / multi-head attn over OOS 2010-2023. After our bug fixes, we expect the numbers to **shift modestly**:
- BUG-1 (EWMA leak fix) → Sharpe likely down a few hundredths
- BUG-2 (delisting returns) → Sharpe likely down 0.02 to 0.05
- Net: **0.55 ± 0.05 for multi-head attn** is the realistic range

| # | Test | What | Pass criterion |
|---|---|---|---|
| 1.1 | Single-head attention rerun | `tcnn_attention.py --start-year 2010 --end-year 2023 --in-ch 3` | Sharpe in [0.55, 0.65] |
| 1.2 | Fixed pool rerun | `tcnn_fixed_pool.py` | Sharpe in [0.45, 0.55] |
| 1.3 | Multi-head attention rerun | `tcnn_multihead_attn.py --num-heads 4` | Sharpe in [0.58, 0.70]; **must beat single-head by ≥0.02** |
| 1.4 | Variant ordering preserved | fixed < single-head < multi-head | strict inequality |
| 1.5 | Per-year Sharpe matches paper Figure 1 | each year's Sharpe within 0.15 of paper visual | qualitative; flag any year > 0.30 off |

**If 1.4 fails** (e.g., multi-head doesn't beat single-head): debug the K-head implementation (`tcnn_multihead_attn.py`). The paper claims monotonic improvement; if we don't see it, our multi-head is wrong.

---

## 2. Robustness diagnostics

### 2.1 Seed ensemble (5 seeds)
Run multi-head with `--seed {0,1,2,3,4}`. Average daily P&L across seeds.
- **Per-seed Sharpe std**: target < 0.05. Higher → training is unstable, results are noise.
- **Ensemble Sharpe vs best single seed**: target +0.03 to +0.10. Standard ML-finance lift.

### 2.2 Train-window length sensitivity
Rerun multi-head with `--train-years {6, 8, 10, 12}`. Plot Sharpe vs train years.
- **Pass:** Sharpe monotone-ish or flat in train_years. Strong increase = under-trained; strong decrease = regime drift.

### 2.3 Hyperparameter ablation (one factor at a time)
| Knob | Settings to compare |
|---|---|
| weight_decay | {0.05, 0.10, 0.15, 0.30} — paper = 0.15 |
| dropout | {0.05, 0.15, 0.30} — paper = 0.15 |
| dilations | {(1,2,4,8,16), (1,2,4), (1,2,4,8,16,32)} |
| num_heads | {1, 2, 4, 8} |

Limit to *one fold's worth* (single test year) for tractability. Pick best; verify on full OOS.

### 2.4 Year-by-year drawdown stability
For multi-head: per-year Sharpe and max DD over 2010-2023.
- **Pass:** ≥ 11/14 years positive Sharpe; max single-year DD < -25%; no two consecutive years with Sharpe < 0.

---

## 3. Backtest honesty (AFML Ch 7, 13, 14)

### 3.1 Purged K-fold + Embargo (AFML Ch 7)
Our rolling 8+2yr is already a form of purged validation, but make it explicit:
- **Embargo window:** 21 calendar days at the train→val and val→test boundaries.
- **Test:** rerun with embargo enforced; ratio `Sharpe_embargoed / Sharpe_paper`.
- **Pass:** ratio ≥ 0.95. < 0.95 indicates microstructural leakage we hadn't caught.

### 3.2 Deflated Sharpe Ratio (DSR; AFML Ch 14)
Our Sharpe of ~0.65 is one of *three trials* (3 pooling variants), so multiple-testing deflation matters:

```
DSR = Φ( ( SR_obs * √(T-1) - E[max_SR] ) / √(1 - γ_3 * E[max_SR] + (γ_4-1)/4 * E[max_SR]²) )
E[max_SR] ≈ √(2 * ln(N_trials)) * (1 - γ + γ/√(2 ln N_trials))   ; γ = Euler-Mascheroni ≈ 0.5772
```
With `N_trials=3`, T=14yr×12mo=168 monthly observations, SR_obs=0.65:
- E[max_SR] ≈ √(2 ln 3) × 0.7 ≈ 1.04 (annualized noise floor)
- Adjust scale: DSR target should land around **0.55 - 0.60 after deflation**.
- **Pass:** DSR > 0.50 (i.e., > median random strategy after multiple-testing correction).

### 3.3 Probability of Backtest Overfitting (PBO; AFML Ch 14)
Combinatorial split: partition OOS 2010-2023 into K=4 splits of 3.5 years each. For each of `C(4,2)=6` (train, test) split-pairs, retrain and record Sharpe. Compute the rank of the chosen variant's Sharpe in IS vs OOS across pairs.
- **Pass:** PBO ≤ 25% (less than 1-in-4 chance our backtest is overfit by random selection).
- **Red flag:** PBO ≥ 50% → the result is inseparable from luck.

### 3.4 Minimum Track Record Length (MinTRL; AFML Ch 15)
For SR=0.65, `MinTRL ≈ (1 + 0.5×SR² - γ_3×SR + (γ_4-1)/4×SR²) × (z_α/SR)²`. At α=0.05 this is roughly 2 years. With **14 years** of OOS we comfortably exceed; t-statistic ≈ 0.65 × √14 ≈ 2.43, p < 0.01.
- **Pass:** OOS ≥ MinTRL by ≥ 5x. (We're at ~7×.)

### 3.5 IID-violation diagnostic
Lopez de Prado's concern: financial returns aren't IID. **For monthly non-overlapping rebalance, this concern largely doesn't apply** — observations are nearly independent (correlation between consecutive monthly portfolio returns is empirically ~0.0-0.1). **Document this** rather than apply sample-weight reweighting.

---

## 4. Alpha quality (QEPM Ch 4)

### 4.1 Information Coefficient (IC) by month
Per-month cross-sectional Spearman correlation between TCNN scores `u_i,t` and forward returns `r_i,t+1`.
- **Pass:** mean IC ≥ 0.015 monthly (FLAM-implied for IR=0.65 at N=2000)
- **Pass:** IC > 0 in ≥ 60% of months

### 4.2 Decile anatomy
Sort stocks by `u_i,t`, form 10 deciles, plot D10 - D1 mean monthly return.
- **Pass:** monotone increasing across deciles (or near-monotone)
- **Pass:** D10 - D1 spread ≥ 2% per month
- **Diagnostic:** check for "all alpha in tails" pattern (D10 alone, or short side dominates) — important for capacity discussion.

### 4.3 IC across regimes (Figure 2 in paper)
IC by realized-vol regime (low/mid/high 21-day):
- **Pass:** IC > 0 in all 3 regimes (paper claim).

### 4.4 Risk-adjusted (purified) IC
Regress monthly TCNN score on FF5 + momentum factor exposures *cross-sectionally*; compute IC of the residual scores.
- **Pass:** purified IC ≥ 80% of raw IC. If purified IC collapses to ≈0, the alpha is just factor exposure (not novel).

---

## 5. Portfolio mechanics (QEPM Ch 8, 11)

### 5.1 Annual turnover
Define `T_annual = 12 × E_t[ Σ_i |w_{i,t} - w_{i,t-1}^{drift}| ] / 2`.
- **Pass:** annual turnover < 200%. Higher → expect TC drag of ~5-15% of gross IR.
- **Diagnostic:** plot turnover by month; spikes in volatile periods (2020-03) are expected.

### 5.2 Forecast autocorrelation ρ_f
Compute `ρ_f = corr(u_{i,t}, u_{i,t-1})` per stock, then average. Low ρ_f = signal flips fast = high turnover. (QEPM Eq 8.20.)
- **Diagnostic, no pass criterion** — just document for the "where does turnover come from?" PM question.

### 5.3 Long/short ratio
Per month: |Σ w_long| / |Σ w_short|.
- **Pass:** mean ≈ 1.0 (dollar-neutral by construction); std < 0.1.

### 5.4 Position concentration
Effective number of names = `1 / Σ w_i²`.
- **Diagnostic:** typical ENN should be in [200, 1000] for top-2000 universe. Lower → concentrated, higher capacity issues.

### 5.5 Transfer coefficient under constraints (QEPM Ch 11)
Run a constrained version: max position = 1%, sector-neutral via sign of FF industry. Report Sharpe with vs without constraints.
- **Pass:** TC = `Sharpe_constrained / Sharpe_unconstrained ≥ 0.85`. Lower → strategy depends on extreme positions.

---

## 6. Factor neutrality (interview defensibility)

### 6.1 FF5 + momentum regression
Daily TCNN returns regressed on Fama-French 5 + UMD (momentum):
```
r_TCNN_t = α + β_MKT · MKT_t + β_SMB · SMB_t + β_HML · HML_t + β_RMW · RMW_t + β_CMA · CMA_t + β_UMD · UMD_t + ε_t
```
- **Headline:** report `α (annualized)` and t-stat. Pass if `α > 0` with `t > 2`.
- **Decomposition:** report what fraction of TCNN return variance is explained by factors. Lower R² → more idiosyncratic alpha.

### 6.2 Sector exposure
Rolling 6-month average sector weights of long and short legs. Document max sector exposure.
- **Pass:** no sector consistently > 25% of long or short leg.

### 6.3 Beta to market
Daily TCNN return regressed on excess market return.
- **Pass:** |β_MKT| < 0.10 (it's dollar-neutral; we expect near-zero).

---

## 7. Transaction-cost + capacity (QEPM Ch 12) — sets up Phase D

This is **the biggest interview-credibility lever**. Paper has no TC. We deliver it.

### 7.1 Linear-quadratic cost model
Cost per dollar traded: `c(Δw) = θ |Δw| + ψ Δw²`. Parameter calibration for top-2000 US equities (Almgren-Chriss-style):
- **θ** (commission + half-spread): ~3-4 bps (1-2 bps commission, 4-6 bps spread for top-2000)
- **ψ** (market impact, per dollar of ADV): ~10-50 bps; depends on participation rate
- **Sweep**: report net Sharpe at θ ∈ {0, 5, 10, 20} bps for fixed ψ=20 bps.

### 7.2 Net-of-TC Sharpe table
| Cost (bps round-trip) | Net Sharpe | Δ from gross |
|---|---|---|
| 0 | 0.65 | — |
| 5 | ? | ~-0.05 |
| 10 | ? | ~-0.10 |
| 20 | ? | ~-0.20 |
- **Pass:** net Sharpe at 10 bps ≥ 0.40 (PM-defensible "satellite alpha").

### 7.3 Capacity estimate
At what AUM does Sharpe degrade by 50% (multi-head 0.65 → 0.32) under realistic market impact?
```
ψ_AUM ≈ ψ_base × (AUM / AUM_base)^0.5     # square-root impact law
Sharpe_net(AUM) = Sharpe_gross - turnover × (θ + ψ_AUM × turnover_share_of_ADV)
```
With monthly rebal, turnover ≈ 80-150% per month, top-2000 ADV ≈ $50M-$5B per name:
- **Pass:** capacity > $200M before Sharpe halves. Below that = hobby strategy.

### 7.4 Turnover-IR efficient frontier
From QEPM Ch 8: plot `gross IR vs net IR` at turnover levels. The "kink" tells us where to throttle re-trades.

---

## 8. Baseline comparisons

### 8.1 vs JKP momentum + reversal (paper baselines)
Already in paper (Sharpe 0.21 / 0.30). Reproduce in our run as sanity check.

### 8.2 vs HRP/NCO baseline (AFML Ch 16) — "is it alpha or sizing?"
Build Hierarchical Risk Parity (HRP) portfolio on top-2000 with **random scoring** (replace TCNN scores with iid noise) and compare Sharpe.
- **Pass:** TCNN Sharpe ≥ 1.2 × HRP-on-noise Sharpe. If similar → TCNN is just clever weighting, not real alpha.

### 8.3 vs simple linear baseline
Replace TCNN encoder with: `score = mean(ret[t-252:t-22]) / vol_ewma`. Same dual-softmax weights.
- **Pass:** TCNN Sharpe ≥ 1.5× simple-mean baseline. This is the "did the network learn anything?" test.

---

## Summary pass-rate table (the resume-defensibility scorecard)

| Section | Test | Pass threshold | Likely outcome | What to do if it fails |
|---|---|---|---|---|
| 1.3 | Multi-head Sharpe | ≥ 0.58 | likely 0.55-0.65 post-fixes | reframe headline as "0.55 gross" |
| 2.1 | Seed ensemble Sharpe std | < 0.05 | usually 0.02-0.05 | flag instability in pitch |
| 3.2 | Deflated Sharpe | > 0.50 | likely 0.50-0.55 | acknowledge multiple-testing |
| 3.3 | PBO | ≤ 25% | likely 15-30% | redo CV split if > 30% |
| 4.1 | Mean monthly IC | ≥ 0.015 | likely 0.02-0.04 | strong signal — fine |
| 4.4 | Purified IC / raw IC | ≥ 80% | unknown | concerning if low — alpha is factor-expo |
| 5.1 | Annual turnover | < 200% | likely 150-250% | report honestly |
| 5.5 | Transfer coefficient | ≥ 0.85 | likely 0.85-0.95 | concerning if low |
| 6.1 | FF5+UMD α t-stat | > 2 | likely 2-4 | strong signal |
| 7.2 | Net Sharpe @ 10 bps | ≥ 0.40 | likely 0.40-0.55 | the resume number |
| 8.2 | TCNN / HRP-noise Sharpe | ≥ 1.2 | unknown | concerning if close |
| 8.3 | TCNN / simple-mean Sharpe | ≥ 1.5 | likely 2-3× | strong signal |

If we hit at least **9 of 12** thresholds, the paper is resume-defensible. **Critical fails to watch:** 4.4 (purified IC), 7.2 (net Sharpe), 8.2 (HRP-on-noise). Those three each kill the pitch independently if they fail.

---

## What to deliver from Phase C

1. `code/results_paper_repro/all_daily_returns.csv` — fresh OOS daily P&L for each of 3 variants
2. `code/notebooks/07_diagnostics.ipynb` — sections 3.1, 4, 5, 6 above (reusable)
3. `code/notebooks/08_capacity_tc_sweep.ipynb` — section 7 (Phase D deliverable)
4. `prep/results_summary.md` — populated Sharpe table with gross *and* net (Phase E)
5. **Decision memo:** if multiple critical tests fail, do we (a) redo with different config, (b) reframe the resume claim, or (c) drop the resume bullet and pivot to a smaller claim?

---

## Order of execution (the actual GPU recipe)

```bash
# ON LAMBDA / PAPERSPACE A100:
# 1. Setup (5 min)
git clone <repo>; cd code
pip install -r requirements.txt

# 2. Build tensors (~10 min)
python data/prep_tensor.py \
    --parquet data/raw/daily_data_sf_v2.parquet \
    --out-dir precomputed_tensors_3ch \
    --num-channels 3 --top-n 2000

# 3. Smoke run (15 min) — Section 0
python train/tcnn_attention.py \
    --tensor-dir precomputed_tensors_3ch \
    --parquet data/raw/daily_data_sf_v2.parquet \
    --cache-dir results/smoke \
    --smoke

# 4. Single-head attention (~5 hr) — Section 1.1
python train/tcnn_attention.py \
    --tensor-dir precomputed_tensors_3ch \
    --parquet data/raw/daily_data_sf_v2.parquet \
    --cache-dir results/attn

# 5. Fixed pool (~5 hr) — Section 1.2
python train/tcnn_fixed_pool.py ... --cache-dir results/fixed

# 6. Multi-head attention (~5 hr) — Section 1.3
python train/tcnn_multihead_attn.py ... --cache-dir results/mhead

# 7. Seed ensemble (~25 hr) — Section 2.1, multi-head with --seed 1..4

# 8. Diagnostics (notebooks) — Sections 3-6 [CPU OK after artifacts saved]
```

Total wall-clock: ~50 GPU-hr at $1-2/hr depending on cloud → **$50-100 for full Phase C**, plus diagnostics CPU-time.

If 1.3 fails (multi-head doesn't beat single-head), stop and debug before doing 7. No point doing capacity analysis on a non-paper-matching variant.
