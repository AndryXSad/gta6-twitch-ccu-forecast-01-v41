# DESIGN.md — gta6-twitch-ccu-forecast-01-v41

## What this program does

Reads one input case (a launch's hour-1 Twitch viewers, its pre-release
7-day mean viewers, and a library of comparable past launches) and writes
the 5/25/50/75/95% quantiles of concurrent viewers for hours 2–24 of
release day. Offline; no network or model calls. Python 3.13, standard
library only.

## Members (five predictors, all validated to 0.000000 vs cached rows)

Each member is a per-hour (log-median, log-spread) pair, built from two
kinds of view over the 38-launch comparable library:

- **OLS view**: per-hour OLS on features of the target
  (`[lnV1, lnPREM, g12]` or a per-member subset), over comparables
  within ±2.5 log units of the target, with an extrapolation guard.
  The OLS prediction is recentered by the median of the comparables'
  log-viewers (recenter = median; recenter = none was tested and is
  wrong by ~0.13 in mu). Residual sd is the spread.
- **Shape view**: a median log-ratio curve (hour / hour 1) over a pool,
  then per-launch level `c` + amplitude `s` regressed on features;
  forecast = `lnV1 + c + s · template[h]`; spread = per-hour residual
  sd. Pool window (windowed vs whole-library), sd convention, fit-hour
  minimum, and early-access filter differ per member.

The conventions that reproduce the cached rows exactly (verified to
0.000000 max |diff| on 92 practice rows and 851 LOO rows):

- OLS sd: **population** for the v4 family (y5, z0, y1, mv),
  **sample** for clone. Shape sd: **population** for the v4 family,
  **sample** for clone. (An earlier prose note had these backwards;
  the empirical test settles it — sample sd anywhere in the v4
  family costs 0.006–0.012 in sigma.)
- fit_hours / min_fits: **(10, 6)** for the v4 members,
  **(20, 5)** for clone.
- The clone sub-member inside mv must use clone's own sd conventions
  (sample), not mv's — using mv's conventions breaks mv.
- Edge cases: the OLS fallback for the largest targets and the
  windowed shape-pool handling when the pool falls below min_fits for
  the smallest targets are what v39 got wrong (mu RMS 0.27–0.31 in
  LOO); the implementation here follows the harness member algorithm
  and reproduces the cached rows exactly, including those edges.
- The linear solver is a full Gauss-Jordan with partial pivoting,
  tested on known 2×2 and 3×3 systems before member validation
  (v37 lesson: a forward-elimination-only solver is silently wrong
  for p ≥ 2).

## Combine (v41 cell)

Per hour-band (early h2–6, mid h7–14, late h15–24), with
`frac` = within-band progress (span-based, `(h − lo)/(hi − lo)`) and
`size = max(0, lnV1 − 11.5)`:

- Median: early **Y1**; mid **median(Y5, clone, mv)**; late
  **median(Y5, Z0, Y1)**.
- Disagreement sd: **sample sd** of the five member medians in early
  and late, **range/2** in mid.
- Location shift:
  `shift = k_dis · (base + slope · size ** size_exp) · sd · dis`
  `+ ramp · frac ** ramp_exp + sshift · size`.
  The ramp is a **direct log offset** (not sd-scaled) — this is the
  structural correction vs the v38 prose, which read as if the ramp
  sat inside the sd-scaled term; scaling it by sd costs +0.008 on
  practice.
- Spread: `factor · (0.45·sigY5 + 0.45·sigZ0 + 0.05·sigClone
  + 0.05·sigMV)`, `factor` = spread_factor (early/mid) and ramping
  spread_factor → late_spread_lo with `frac ** late_spread_exp` in
  late; then `spread *= 1 + w5 · size · sd`;
  `sig_lo = min(spread · alo · (1 + w1 · size), cap)`,
  `sig_hi = min(spread · ahi, cap)`.

The v41 cell (work/sweep41b-best.json):

| band | base | slope | ramp | ramp_exp | dis | alo | ahi | sshift |
|---|---|---|---|---|---|---|---|---|
| early | 0.475514 | 1.288883 | +0.20836 | 2.046789 | 1.064788 | 1.163068 | 0.823456 | −0.190328 |
| mid | 0.306707 | 0.864176 | −0.177019 | 1.224932 | 1.394191 | 1.1507 | 1.006655 | −0.076861 |
| late | 0.329018 | 0.794653 | +0.079743 | 0.283456 | 1.143692 | 3.54135 | 0.747014 | −0.093587 |

Globals: k_dis 0.646755, size_exp 2.298076, spread_factor 1.194096,
late_spread_lo 0.642486, late_spread_exp 1.427597, w5 −0.476073,
w1 0.01014, cap 0.324745.

## How the cell was found

Three search stages on cached rows (no member rebuild; scoring is
exact pinball vs the cached practice answers and LOO rows):

1. **Coordinate descent + constrained random search** (work/sweep41.py,
   4,000 trials) from the v40 cell, ranked by
   practice + LOO-large + LOO-mega with a 10× penalty on any gate
   violation. Landed at practice 0.114343 / LOO 0.200048 / 0.201102 —
   but the practice margin over v40 was only 1e-6, a tie in any honest
   sense, with the gain bought entirely by LOO.
2. **Balanced re-search** (work/sweep41b.py, 4,000 trials) ranked by
   the WEAKEST gate margin, with a soft guard on LOO-all. This is the
   packaged cell: practice 0.113229 / LOO-large 0.201343 / LOO-mega
   0.202190 / LOO-all 0.485595 — margins +0.001115 / +0.001115 /
   +0.001712 over the v40 bar on all three gates.
3. **Regularised sanity check** (work/sweep41c.py, 4,000 trials):
   same balanced objective plus a penalty on normalised squared
   distance from the v40 cell. It found a much closer cell
   (dist² 0.16 vs 8.37) that also passes all gates, but with smaller
   margins (+0.00018): practice 0.114162 / LOO 0.202275 / 0.203701 /
   LOO-all 0.485404 (better than v40 on LOO-all too). The packaged
   cell was chosen over it because it wins the honest LOO check by
   the widest margin; the regularised cell is recorded here as the
   conservative alternative.

## Scores (exact RULES scorer, "Average viewers")

| metric | v41 | v40 |
|---|---|---|
| practice pinball loss | **0.113229** | 0.114344 |
| LOO-large (lnV1 ≥ 11.0) | **0.201343** | 0.202458 |
| LOO-mega (lnV1 ≥ 11.5) | **0.202190** | 0.203902 |
| median-only practice | 0.171543 | 0.171033 |
| LOO-all | 0.485595 | 0.485548 |
| median-only LOO-large | **0.311421** | 0.312754 |

Per-case practice: p00 0.100338 / p01 0.131791 / p02 0.075499 /
p03 0.145286 (v40: 0.103210 / 0.135415 / 0.074424 / 0.144328).
Direct grade of this file matches the cached-row score exactly
(0.113229), because the members are exact.

## Honest caveats

- The practice leaderboard is only a proxy: the top of the board is
  dominated by replay/memorisation entries, and this program is built
  to generalise to GTA VI's real launch day, not to memorise the four
  practice launches. The LOO gates are the honest check, and they
  improved at least as much as practice — but LOO is still past
  launches, not GTA VI, and all three search stages tuned on the same
  LOO rows, so the LOO gains are partly tuned-on-LOO gains.
- **The practice gain is spread-driven, not median-driven.**
  Median-only practice is slightly WORSE than v40 (0.171543 vs
  0.171033) while the full practice score improved by 0.0011. The
  cell wins by calibrating the spread (and the disagreement shift),
  not by finding a better median. Median-only LOO-large did improve
  (0.311421 vs 0.312754), so the median is better on large launches
  in LOO — but the practice median is not.
- **The cell extrapolates harder than v40 at GTA VI scale.**
  size_exp 2.30 (v40: 1.80), early slope 1.29 (v40: 0.65): at
  size 2.5 the early shift coefficient is ~2.4× v40's, and the
  mid/late coefficients ~1.5–1.6×. LOO-mega (largest launches in the
  library, size up to 2.18) supports the steeper shift, but GTA VI
  may be bigger than anything in the library, and no past launch can
  prove the extrapolation is right.
- **Two parameters sit in degenerate corners.** Late `alo` 3.54
  pushes `sig_lo` onto the cap (0.325) in the late band almost
  always, so its exact value is unidentified — any value above ~2
  would give nearly the same forecasts. `w5` flipped negative
  (−0.476 vs v40's +0.841), so the spread now SHRINKS with
  size × disagreement instead of widening. Both are what the data
  preferred, but both are the kind of corner a random search finds
  by fitting quirks, and both deserve a second look in the next
  round.
- LOO-all is essentially tied with v40 (0.485595 vs 0.485548): the
  gain is concentrated in large launches; small launches are
  ~neutral. That is the right trade for a mega launch, but it means
  the cell is not uniformly better.
- The v40 cell as written from its DESIGN.md prose reproduces to
  0.114337 / 0.202464 / 0.203905 in this combine family, not to
  v40's reported 0.114344 / 0.202458 / 0.203902 (a ±7e-6 gap, most
  likely prose rounding of the packaged parameters). The structure
  was confirmed independently: the `frac` definition
  `(h − lo)/(hi − lo)` matches the reported anchor to 7e-6 on all
  four metrics, while the two alternative definitions miss practice
  by +0.0005 and +0.0026. The v41 cell beats v40's reported bar
  under either baseline, so the improvement does not depend on
  resolving that gap — but the gap is recorded here rather than
  hidden.
- The median is still the bottleneck: median-only practice 0.171543
  against a per-hour OLS residual sd of 0.35–0.45 log. A practice
  score near 0.100 is not reachable honestly with this family; gains
  now come in ~0.001 steps and increasingly fit the four practice
  cases.
- No submission has been made from this repo. Per the owner's rule,
  it is submitted only if a candidate reaches the practice top 10.
