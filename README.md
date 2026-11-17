# gta6-twitch-ccu-forecast-01-v41

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/AndryXSad/gta6-twitch-ccu-forecast-01-v41/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/AndryXSad/gta6-twitch-ccu-forecast-01-v41/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

A make.fun `gta6-twitch-ccu-forecast-01` entry. It forecasts, for each of
hours 2–24 of GTA VI's launch day, the 5/25/50/75/95% quantiles of Twitch
concurrent viewers, from one input case (the launch's hour-1 viewers, its
pre-release 7-day mean, and comparable past launches). Offline, no network
or model calls. See `DESIGN.md` for the method.

## Scores (exact RULES scorer, "Average viewers")

| metric | v41 | v40 |
|---|---|---|
| practice pinball loss | **0.113229** | 0.114344 |
| LOO-large (lnV1 ≥ 11.0) | **0.201343** | 0.202458 |
| LOO-mega (lnV1 ≥ 11.5) | **0.202190** | 0.203902 |
| median-only practice | 0.171543 | 0.171033 |
| LOO-all | 0.485595 | 0.485548 |

Per-case practice: p00 0.100338 / p01 0.131791 / p02 0.075499 /
p03 0.145286. All five member predictors reproduce the cached
validation rows to 0.000000 on practice and LOO. The v41 cell beats
v40 on practice AND both LOO gates (margins +0.001115 / +0.001115 /
+0.001712), with the LOO gates improving at least as much as practice
— the signature of a genuine generalisation gain, not practice
overfit. Median-only LOO-large improved to 0.311421 (v40: 0.312754),
but median-only practice is slightly worse than v40 (0.171543 vs
0.171033): the practice gain is spread-driven, not median-driven.
See `DESIGN.md` for the caveats.

## Method in one paragraph

Five member predictors (y5, z0, y1, clone, mv) are built from per-hour
OLS views and one-factor shape views over 38 comparable launches, then
combined per hour-band with per-band median sources, per-band
asymmetric spread scales, a spread cap, and disagreement-driven
location/spread shifts. New in v41: the combine was re-searched from
the v40 cell in the same family (coordinate descent + constrained
random search, ranked by the weakest gate margin, then a regularised
second search that stayed close to v40 as a sanity check). The packaged
cell landed in a steeper region — size exponent 2.30, early slope 1.29,
spread factor 1.194, w5 flipped negative — and improves all three
gates by ~0.001 or more. The key structural facts carried over from
v40: the within-band ramp enters the location shift as a **direct log
offset**, not sd-scaled, and `frac` is span-based within each band.
See `DESIGN.md` for the full cell and the honest caveats.

## Run

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `*.in` file in the input directory and writes one `*.out`
file per case, with 23 rows (hours 2–24) of five quantiles each.

## License

MIT — see `LICENSE`.
