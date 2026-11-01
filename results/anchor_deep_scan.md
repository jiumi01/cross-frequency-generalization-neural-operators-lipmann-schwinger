# Analytic head depth: what the anchor's plateau is made of

Author: Longgang Wang. All numbers below are produced by the scripts named
beside them and are recomputed by `code/check_claims.py`; none is transcribed
by hand.

Protocol for everything in this file: `n = 200` media per cell, seeds
`60000+vi` per test frequency and `70000+vi` in band (the manuscript's own
evaluation seeds), 64x64 grid, loss tangent `gamma = 0.05`, test frequencies
`{1.5, 2.5, 3.5, 4, 4.25, 4.5, 4.75, 5, 6, ..., 12}`. The plateau
`epsilon_inf` is the mean over `f >= 6`. The band edge ratio is
`r = epsilon(5)/epsilon(4)`.

---

## 1. The plateau is Born truncation error (code/anchor_higher_order.py)

The manuscript's anchor is `u_hat = a B1 + b B2`, four real numbers fitted on a
512-sample in-band prefix. Fitting `M` complex coefficients on `B1..B_M`
instead — still FFT-only, still label-free, still fitted on the same in-band
prefix — drives the out-of-band plateau down geometrically:

| `v_max` | M=1 | M=2 | M=3 | M=4 | M=5 | M=6 | M=7 | M=8 |
|---|---|---|---|---|---|---|---|---|
| 0.05 | 0.148 | 0.023 | 0.004 | 0.001 | 0.000 | 0.000 | 0.000 | 0.000 |
| 0.10 | 0.268 | 0.077 | 0.022 | 0.007 | 0.003 | 0.001 | 0.000 | 0.000 |
| 0.25 | 0.517 | **0.300** | 0.165 | 0.104 | 0.064 | 0.044 | 0.029 | **0.022** |
| 0.50 | 0.690 | 0.579 | 0.424 | 0.358 | 0.280 | 0.245 | 0.206 | 0.187 |
| 1.00 | 0.722 | **0.703** | 0.587 | 0.582 | **0.516** | 0.517 | 0.495 | refused |

Bold: M=2 is the manuscript's existing anchor; M=5 at `v=1.00` is the turning
point. The M=8 fit at `v=1.00` is **refused**, not reported: the Gram matrix
condition number reaches `8.9e12`, past the `1e12` ceiling at which a
least-squares solve is a rounding artefact rather than a measurement.

Two honest caveats, both material:

- **The improvement is not unbounded.** At `v=1.00` the curve turns between
  M=5 (0.516) and M=6 (0.517). This is expected and is discussed in §2: the
  Born contraction factor exceeds one there, so the fit is an asymptotic
  summation on a formally divergent series, and asymptotic summations have an
  optimal truncation point.
- **Coefficients stay moderate.** At M=8 the fitted `|c_j|` are of order 0.04
  to 3.2 at every contrast where the fit is accepted, so the improvement is
  not catastrophic cancellation in an ill-conditioned basis.

## 2. The mechanism: a measured contraction factor (code/anchor_contraction.py)

The truncation error is governed by the contraction factor of the Neumann
operator `T psi = kappa^2 G (v psi)`, measured directly as the geometric mean
over samples of `||B_{j+1}|| / ||B_j||`:

| `v_max` | rho (level) | rho spread over f | rho(4) | rho(12) |
|---|---|---|---|---|
| 0.05 | 0.234 | 2.43x | 0.231 | 0.298 |
| 0.10 | 0.455 | 2.39x | 0.435 | 0.581 |
| 0.25 | 1.120 | 2.40x | 1.099 | 1.434 |
| 0.50 | 2.257 | 2.37x | 2.185 | 2.871 |
| 1.00 | 4.512 | 2.40x | 4.401 | 5.760 |

Two readings, one of which **corrects** a clause of the manuscript:

- **Contrast sets the level, essentially linearly**: `rho / v_max` is
  4.68, 4.55, 4.48, 4.51, 4.51. The series converges (`rho < 1`) only below
  `v_max ~ 0.22`, which is exactly where the anchor alone beats the trained
  operator in band.
- **Frequency does move rho, by 2.40x, uniformly across contrasts.** The
  manuscript's factorisation reading says the residual is "near-frequency-
  independent". That clause is now measured and is *not* strictly true: rho
  drifts by a factor of 2.4 across `f in [1.5, 12]`. The correct, weaker,
  and still sufficient statement is that the drift is **smooth and monotone**
  (`rho(12)/rho(4) = 1.29` at every contrast) and **three times smaller than
  the cliff** the trained operators carry (`r = 7.19--8.30`). A smooth 2.4x
  drift cannot manufacture a 7--8x discontinuity at the band edge.

## 3. The depth is selectable in band (code/anchor_order_selection.py)

If choosing `M` required out-of-band labels, the construction would stop being
zero-shot and the manuscript's central distinction would collapse. It does not:
fit on the first 512 in-band cache samples, score on the **disjoint** next 512
in-band samples, take the argmin.

| `v_max` | `M_sel` (in-band rule) | `M_oracle` (peeking out of band) | plateau at `M_sel` | plateau at M=2 |
|---|---|---|---|---|
| 0.05 | 8 | 8 | 0.000 | 0.023 |
| 0.10 | 8 | 8 | 0.000 | 0.077 |
| 0.25 | 8 | 8 | 0.022 | 0.300 |
| 0.50 | 8 | 8 | 0.187 | 0.579 |
| 1.00 | 7 | 7 | 0.495 | 0.703 |

The in-band rule picks the out-of-band-optimal order at **five out of five**
contrasts. No out-of-band information enters the choice.

## 4. What this does to the manuscript's existing Table 7

Against the trained FNO (three seeds, each arm trained and scored at its own
contrast):

| `v_max` | FNO in-band | FNO plateau | anchor(`M_sel`) in-band | anchor(`M_sel`) plateau | plateau ratio |
|---|---|---|---|---|---|
| 0.05 | 0.0159 | 0.222 | 0.000 | 0.000 | > 100x |
| 0.10 | 0.0336 | 0.434 | 0.000 | 0.000 | > 100x |
| 0.25 | 0.0854 | 0.963 | 0.025 | 0.022 | 44x |
| 0.50 | 0.1872 | 1.430 | 0.183 | 0.187 | 7.6x |
| 1.00 | 0.3319 | 1.869 | 0.499 | 0.495 | 3.8x |

**This reverses one claim the manuscript currently makes.** Table 7 as it
stands reports "precise complementarity": the trained operator wins in band by
4.0x, the anchor wins out of band by 3.2x, and the two objects fail in mirror
image. With the in-band-selected depth that is no longer what happens. The
analytic head, which costs **no training at all**, now

- beats the trained operator out of band at every contrast, by 3.8x to 44x;
- **also matches or beats it in band** for `v_max <= 0.50`
  (0.000 vs 0.016, 0.000 vs 0.034, 0.025 vs 0.085, 0.183 vs 0.187);
- loses in band only at `v_max = 1.00` (0.499 vs 0.332), which is precisely
  where `rho = 4.5` and the Born series is being summed outside its
  convergence disk.

The crossover in band sits between `v_max = 0.50` and `v_max = 1.00`, i.e.
just above the `rho = 1` boundary at `v_max ~ 0.22` once several terms are
summed.

## 5. Learning the residual on top of the head (code/run_ba1.py, run_ba2.py)

`u_hat = sum_j c_j B_j + r_theta`, coefficients initialised by the in-band fit
and fine-tuned on the band, `r_theta` the baseline FNO at identical width,
modes, depth, budget, schedule and seeds. All numbers at `v_max = 0.25`, three
seeds, against `main_fno_a025` on the same media.

| arm | in-band | e(4) | e(5) | plateau | r |
|---|---|---|---|---|---|
| trained FNO | 0.0854 | 0.1493 | 1.1165 | 0.9631 | 7.48 |
| head alone, M=2 | 0.338 | 0.282 | 0.280 | 0.300 | 0.99 |
| BA1 = M=2 head + residual | 0.0275 | 0.0706 | 0.3668 | 0.4829 | 5.20 |
| head alone, M=8 | 0.0253 | 0.092 | 0.096 | 0.0218 | 1.04 |
| **BA2 = M=8 head + residual** | **0.0054** | **0.0129** | **0.0264** | **0.0259** | **2.05** |

Pre-registered criteria for the deep arm: in-band at or below the trained
FNO's, and plateau at or below half of it. Both hold, by 15.8x and 37.1x, so
BA2 reads **PASS**. Against the head alone the residual buys 4.7x in band and
costs 1.19x on the plateau.

BA1 reads as a negative result and is reported as one: its plateau 0.483 is
worse than the M=2 head's own 0.300, so the learned correction is adding
out-of-band error on top of a phase-correct head.

One implementation note that is part of the result. Feeding `B_1..B_M` to the
network as input channels works while `rho <~ 1`; at `v_max = 1.00`, where
`rho = 4.5` and `B_7` is about `8e3` times `B_1`, it diverged (val = 19 at
epoch 10). The `a100` arm therefore sums seven terms in the head but hands the
network only `B_1, B_2` as channels. The analytic content is unchanged, since
the coefficients are applied outside the network.

## 6. Scope of the claim

- Everything here is on 64x64 periodic synthetic media at one loss tangent.
  The manuscript's resolution ladder (64--256, first-order decay, 128
  retrain, pointwise resolution invariance) covers the grid question, not the
  geometry question.
- No measured data. The analytic head has **not** been run on the Marmousi
  windows, so the external-validity claim for the *construction* is not yet
  made; only the boundary and jump claims have Level-3 support.
- The construction still needs the medium `v` and the incident field, and it
  still needs the in-band prefix to fit `2M` numbers. It is zero-shot with
  respect to frequency, **not** with respect to contrast, and the failure is
  not graceful: the `v_max = 0.25` head applied at `v_max = 1.00` gives
  relative errors of order `1e4` (plateau `7.3e4` for the M=8 arm, `12.8` for
  the M=2 arm) against `1.69` for the trained operator evaluated the same way.
  Deeper heads are worse there, not better.
