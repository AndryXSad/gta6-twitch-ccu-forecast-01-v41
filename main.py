#!/usr/bin/env python3
"""Forecast a game launch's Twitch viewers, hour by hour (hours 2-24).

Reads one launch case per *.in file from $MAKEFUN_INPUT_DIR (or /input):
hour-1 viewers, the 7-day pre-release hourly history, and a library of
comparable past launches. Writes one *.out file per case to
$MAKEFUN_OUTPUT_DIR (or /output) with the 5/25/50/75/95% quantiles of
concurrent viewers for hours 2-24 of release day.

Method
------
Five per-hour (log-median, log-spread) members are built from the
comparable library, then combined per hour-band:

* y5 / z0 / y1 -- "v4 family": per hour, a weighted blend of an OLS view
  (features of the target over comparables within +/-2.5 log units of
  its hour-1 size, residual sd as spread, prediction recentered by the
  median residual) and two shape views (median log-ratio curve over a
  windowed pool, per-launch level + amplitude regressed on features).
  The three differ in the early-access filter and the shape-view
  feature set.
* clone -- same views with whole-library shape pools, a count-after
  OLS pool, and sample (not population) sds.
* mv -- per-hour median of the member medians (OLS, console-shape,
  all-shape, clone) and mean of their spreads.

Combine (per band: early hours 2-6, mid 7-14, late 15-24):
median from one member (early) or the median of three members
(mid/late); a location shift proportional to the disagreement between
the five member medians, a within-band ramp that is a direct log
offset, and a size term; spread as a weighted blend of member spreads
with a late-band ramp, widened by size x disagreement, then split
into lower/upper sigmas with per-band asymmetry and a cap.

Standard library only; offline; no network or model calls.

Packaged cell: v41 (practice 0.113229, LOO-large 0.201343, LOO-mega
0.202190) — beats the v40 bar (0.114344 / 0.202458 / 0.203902) on all
three gates. See DESIGN.md for the honest caveats: the gain is
spread-driven, the median-only practice score is slightly worse than
v40, and the cell extrapolates steeper than v40 at GTA VI scale.
"""
import json
import math
import os
import sys
from datetime import datetime, timedelta

LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
Z_VALUES = (-1.6448536269514722, -0.6744897501960817, 0.0,
            0.6744897501960817, 1.6448536269514722)
HOURS = tuple(range(2, 25))
WINDOW = 2.5


# ---------------------------------------------------------------- numerics
def _median(vals):
    ordered = sorted(vals)
    n = len(ordered)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def _mean(vals):
    return sum(vals) / len(vals) if vals else 0.0


def _pop_sd(vals):
    if len(vals) < 2:
        return 0.0
    m = _mean(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))


def _sample_sd(vals):
    if len(vals) < 2:
        return 0.0
    m = _mean(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))


def _sd(vals, population):
    return _pop_sd(vals) if population else _sample_sd(vals)


def _clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def _solve(a, b):
    """Full Gauss-Jordan elimination with partial pivoting.

    Returns the solution vector, or None if the system is singular.
    """
    n = len(b)
    aug = [list(a[i]) + [b[i]] for i in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[piv][col]) < 1e-11:
            return None
        aug[col], aug[piv] = aug[piv], aug[col]
        for r in range(n):
            if r == col:
                continue
            factor = aug[r][col] / aug[col][col]
            if factor == 0.0:
                continue
            for k in range(col, n + 1):
                aug[r][k] -= factor * aug[col][k]
    return [aug[i][n] / aug[i][i] for i in range(n)]


def _ols(rows, ys, ridge=1e-9):
    p = len(rows[0])
    gram = [[sum(r[i] * r[j] for r in rows) + (ridge if i == j else 0.0)
             for j in range(p)] for i in range(p)]
    vec = [sum(r[i] * y for r, y in zip(rows, ys)) for i in range(p)]
    return _solve(gram, vec)


def _dot(beta, row):
    return sum(b * x for b, x in zip(beta, row))


# -------------------------------------------------------------------- data
def _parse_ts(text):
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ")


def _series(points):
    out = {}
    for pt in points:
        val = pt.get("viewers")
        if val is None or val <= 0:
            continue
        out[_parse_ts(pt["from"])] = float(val)
    return out


def _pre_rough(vmap, day_start):
    start = _parse_ts(day_start)
    pts = sorted((t, v) for t, v in vmap.items()
                 if start - timedelta(days=7) <= t < start)
    diffs = []
    for (t0, v0), (t1, v1) in zip(pts, pts[1:]):
        if (t1 - t0).total_seconds() == 3600:
            diffs.append(math.log(v1) - math.log(v0))
    return _pop_sd(diffs) if len(diffs) >= 10 else None


def _features(vmap, day_start, hour1):
    start = _parse_ts(day_start)
    prior = [v for t, v in vmap.items()
             if start - timedelta(days=7) <= t < start]
    if not prior:
        return None
    prem = math.log(_mean(prior))
    want = start - timedelta(hours=12)
    best = None
    for t, v in vmap.items():
        gap = abs((t - want).total_seconds())
        if gap <= 3 * 3600 and (best is None or gap < best[0]):
            best = (gap, v)
    g12 = None if best is None else math.log(hour1) - math.log(best[1])
    return {"lnV1": math.log(hour1), "lnPREM": prem, "g12": g12,
            "pre_rough": _pre_rough(vmap, day_start)}


def _tier(note):
    text = (note or "").lower()
    if ("early-access-contaminated" in text
            or "early-access contaminated" in text):
        return "ea"
    if "tier: weak" in text:
        return "weak"
    return "comparable"


def _parse_launch(comp):
    vmap = _series(comp["hourly"])
    start = _parse_ts(comp["day_start"])
    v1 = vmap.get(start)
    if v1 is None:
        return None
    curve = {}
    for h in range(1, 25):
        t = start + timedelta(hours=h - 1)
        if t in vmap:
            curve[h] = vmap[t]
    if sum(1 for h in curve if h >= 2) < 20:
        return None
    feats = _features(vmap, comp["day_start"], v1)
    if feats is None:
        return None
    feats["curve"] = curve
    feats["name"] = comp["name"]
    feats["console"] = "PC" not in comp.get("platforms", "")
    note = comp.get("unlock", "") or ""
    feats["tier"] = _tier(note)
    feats["space_ea"] = "early access" in note.lower()
    feats["hyphen_ea"] = "early-access" in note.lower()
    return feats


def _parse_target(case):
    vmap = _series(case["history_hourly"])
    v1 = max(1.0, float(next(k for k in case["known"]
                            if k["hour"] == 1)["viewers"]))
    feats = _features(vmap, case["day_start"], v1)
    if feats is None:
        feats = {"lnV1": math.log(v1), "lnPREM": math.log(v1),
                 "g12": 0.0, "pre_rough": None}
    feats["curve"] = {}
    return feats


def _load_case(case):
    tgt = _parse_target(case)
    lib = []
    for comp in case["comparables"]:
        if "before release" in comp["name"]:
            continue
        parsed = _parse_launch(comp)
        if parsed is not None:
            lib.append(parsed)
    for key in ("g12", "pre_rough"):
        vals = sorted(f[key] for f in lib if f[key] is not None)
        if not vals:
            continue
        fill = vals[len(vals) // 2]
        if tgt[key] is None:
            tgt[key] = fill
        for f in lib:
            if f[key] is None:
                f[key] = fill
    return tgt, lib


# -------------------------------------------------------------------- views
def _span(lib):
    return (min(f["lnV1"] for f in lib), max(f["lnV1"] for f in lib))


def _design_row(feats, lo, hi, cols):
    size = _clamp(feats["lnV1"], lo, hi + 0.5)
    vals = {"lnV1": size, "lnPREM": feats["lnPREM"], "g12": feats["g12"]}
    return [1.0] + [vals[c] for c in cols]


def _near(lib, lnv1, half=WINDOW):
    close = [f for f in lib if abs(f["lnV1"] - lnv1) <= half]
    return close if len(close) >= 10 else list(lib)


def _ols_view(lib, tgt, hour, population, count_after):
    if count_after:
        pool = [f for f in lib if hour in f["curve"]
                and abs(f["lnV1"] - tgt["lnV1"]) <= WINDOW]
        if len(pool) < 10:
            pool = [f for f in lib if hour in f["curve"]]
    else:
        pool = [f for f in _near(lib, tgt["lnV1"]) if hour in f["curve"]]
    if len(pool) < 8:
        return None
    lo, hi = _span(lib)
    cols = ("lnV1", "lnPREM", "g12")
    design = [_design_row(f, lo, hi, cols) for f in pool]
    ys = [math.log(f["curve"][hour]) for f in pool]
    beta = _ols(design, ys)
    if beta is None:
        return None
    resid = [y - _dot(beta, r) for r, y in zip(design, ys)]
    sd = _sd(resid, population)
    mu = _dot(beta, _design_row(tgt, lo, hi, cols))
    mu += _median(resid)
    return mu, sd * 0.8


def _ea_filter(base, kind):
    if kind == "tier":
        return [f for f in base if f["tier"] != "ea"]
    if kind == "naive":
        return [f for f in base if not f["space_ea"]]
    return [f for f in base if not f["hyphen_ea"]]


def _shape_view(lib, tgt, console_only, feats, population, windowed,
                ea_kind, fit_hours, min_fits):
    base = _near(lib, tgt["lnV1"]) if windowed else list(lib)
    pool = _ea_filter(base, ea_kind)
    if console_only:
        pool = [f for f in pool if f["console"]]
    if len(pool) < 6:
        pool = _ea_filter(base, ea_kind)
    if len(pool) < 6:
        return None
    template = {}
    for h in HOURS:
        vals = [math.log(f["curve"][h] / f["curve"][1]) for f in pool
                if h in f["curve"] and 1 in f["curve"]]
        template[h] = _median(vals) if vals else 0.0
    fits = []
    for f in pool:
        xs = [template[h] for h in HOURS if h in f["curve"]]
        ys = [math.log(f["curve"][h] / f["curve"][1]) for h in HOURS
              if h in f["curve"]]
        if len(xs) < fit_hours:
            continue
        mx, my = _mean(xs), _mean(ys)
        den = sum((x - mx) ** 2 for x in xs)
        slope = (sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
                 if den else 0.0)
        fits.append((f, my - slope * mx, slope))
    if len(fits) < min_fits:
        return None
    lo, hi = _span(lib)
    design = [_design_row(f, lo, hi, feats) for f, _, _ in fits]
    beta_c = _ols(design, [c for _, c, _ in fits])
    beta_s = _ols(design, [s for _, _, s in fits])
    if beta_c is None or beta_s is None:
        return None
    xt = _design_row(tgt, lo, hi, feats)
    c_star = _dot(beta_c, xt)
    s_star = _clamp(_dot(beta_s, xt), 0.0, 2.0)
    out = {}
    for h in HOURS:
        res = [math.log(f["curve"][h] / f["curve"][1])
               - (c + s * template[h])
               for f, c, s in fits if h in f["curve"]]
        sd = _sd(res, population)
        out[h] = (tgt["lnV1"] + c_star + s_star * template[h], sd)
    return out


# ------------------------------------------------------------------ members
def _family_member(lib, tgt, ea_kind, feats, shape_pop, fit_hours,
                   min_fits, ols_pop):
    con = _shape_view(lib, tgt, True, feats, shape_pop, True, ea_kind,
                      fit_hours, min_fits)
    all_ = _shape_view(lib, tgt, False, feats, shape_pop, True, ea_kind,
                       fit_hours, min_fits)
    out = {}
    for h in HOURS:
        views = []
        ov = _ols_view(lib, tgt, h, ols_pop, False)
        if ov is not None:
            views.append((0.5, ov))
        else:
            views.append((0.5, (tgt["lnV1"], 0.5)))
        if con is not None and h in con:
            views.append((0.2, con[h]))
        if all_ is not None and h in all_:
            views.append((0.3, all_[h]))
        tot = sum(w for w, _ in views)
        mu = sum(w * v[0] for w, v in views) / tot
        sig = math.sqrt(max(0.02, sum(w * v[1] ** 2 for w, v in views)
                            / tot))
        out[h] = (mu, sig)
    return out


def _clone_member(lib, tgt):
    con = _shape_view(lib, tgt, True, ("lnV1", "lnPREM", "g12"), False,
                      False, "hyphen", 20, 5)
    all_ = _shape_view(lib, tgt, False, ("lnV1", "lnPREM", "g12"), False,
                       False, "hyphen", 20, 5)
    out = {}
    for h in HOURS:
        views = []
        ov = _ols_view(lib, tgt, h, False, True)
        if ov is not None:
            views.append((0.6, ov))
        if con is not None and h in con:
            views.append((0.2, con[h]))
        if all_ is not None and h in all_:
            views.append((0.2, all_[h]))
        if not views:
            out[h] = (tgt["lnV1"], 0.5)
            continue
        tot = sum(w for w, _ in views)
        mu = sum(w * v[0] for w, v in views) / tot
        sig = 0.81 * sum(w * v[1] for w, v in views) / tot
        out[h] = (mu, sig)
    return out


def _median_member(lib, tgt):
    con = _shape_view(lib, tgt, True, ("lnV1", "lnPREM", "g12"), True,
                      True, "hyphen", 10, 6)
    all_ = _shape_view(lib, tgt, False, ("lnV1", "lnPREM", "g12"), True,
                       True, "hyphen", 10, 6)
    cl = _clone_member(lib, tgt)
    out = {}
    for h in HOURS:
        mus, sigs = [], []
        ov = _ols_view(lib, tgt, h, True, False)
        if ov is not None:
            mus.append(ov[0])
            sigs.append(ov[1])
        if con is not None and h in con:
            mus.append(con[h][0])
            sigs.append(con[h][1])
        if all_ is not None and h in all_:
            mus.append(all_[h][0])
            sigs.append(all_[h][1])
        mus.append(cl[h][0])
        sigs.append(cl[h][1])
        out[h] = (_median(mus), _mean(sigs))
    return out


def _build_members(lib, tgt):
    """{kind: {hour: (mu, sig)}} for the five members."""
    members = {}
    members["y5"] = _family_member(lib, tgt, "tier", ("lnV1", "g12"),
                                   True, 10, 6, True)
    members["z0"] = _family_member(lib, tgt, "naive", ("lnV1", "g12"),
                                   True, 10, 6, True)
    members["y1"] = _family_member(lib, tgt, "naive", ("lnV1", "lnPREM"),
                                   True, 10, 6, True)
    members["clone"] = _clone_member(lib, tgt)
    members["mv"] = _median_member(lib, tgt)
    return members


# ------------------------------------------------------------------ combine
# v41 packaged cell (verified: practice 0.113229, LOO-large 0.201343,
# LOO-mega 0.202190 — beats the v40 bar on all three gates).
# Found by coordinate descent + constrained random search from the v40
# cell, ranked by the weakest gate margin (see work/sweep41b.py).
# A closer-to-v40 regularised cell (work/sweep41c-best.json) also
# passes the gates (+0.00018) but with smaller margins; this cell is
# packaged because it wins the honest LOO check by the widest margin.
# Caveats (see DESIGN.md): steeper size extrapolation than v40, w5
# flipped negative, late alo saturated at the cap, and the practice
# gain is spread-driven (median-only practice slightly worse).
CELL = {
    "size_exp": 2.298076,
    "spread_factor": 1.194096,
    "k_dis": 0.646755,
    "late_spread_lo": 0.642486,
    "late_spread_exp": 1.427597,
    "w5": -0.476073,
    "w1": 0.01014,
    "cap": 0.324745,
    "early": {"base": 0.475514, "slope": 1.288883, "ramp": 0.20836,
              "ramp_exp": 2.046789, "dis": 1.064788, "alo": 1.163068,
              "ahi": 0.823456, "sshift": -0.190328},
    "mid": {"base": 0.306707, "slope": 0.864176, "ramp": -0.177019,
            "ramp_exp": 1.224932, "dis": 1.394191, "alo": 1.1507,
            "ahi": 1.006655, "sshift": -0.076861},
    "late": {"base": 0.329018, "slope": 0.794653, "ramp": 0.079743,
             "ramp_exp": 0.283456, "dis": 1.143692, "alo": 3.54135,
             "ahi": 0.747014, "sshift": -0.093587},
}
BANDS = (("early", 2, 6), ("mid", 7, 14), ("late", 15, 24))


def _band_of(hour):
    for name, lo, hi in BANDS:
        if lo <= hour <= hi:
            return name, lo, hi
    return "late", 15, 24


def _combine(members, lnv1, cell=CELL):
    size = max(0.0, lnv1 - 11.5)
    out = {}
    for h in HOURS:
        name, lo, hi = _band_of(h)
        band = cell[name]
        frac = (h - lo) / (hi - lo)
        mus = [members[k][h][0] for k in
               ("y5", "z0", "y1", "clone", "mv")]
        if name == "early":
            mu = members["y1"][h][0]
            dis_sd = _sample_sd(mus)
        elif name == "mid":
            mu = _median([members["y5"][h][0], members["clone"][h][0],
                          members["mv"][h][0]])
            dis_sd = 0.5 * (max(mus) - min(mus))
        else:
            mu = _median([members["y5"][h][0], members["z0"][h][0],
                          members["y1"][h][0]])
            dis_sd = _sample_sd(mus)
        shift = (cell["k_dis"]
                 * (band["base"] + band["slope"] * size ** cell["size_exp"])
                 * dis_sd * band["dis"]
                 + band["ramp"] * frac ** band["ramp_exp"]
                 + band["sshift"] * size)
        if name == "late":
            factor = (cell["spread_factor"]
                      + (cell["late_spread_lo"] - cell["spread_factor"])
                      * frac ** cell["late_spread_exp"])
        else:
            factor = cell["spread_factor"]
        spread = factor * (0.45 * members["y5"][h][1]
                           + 0.45 * members["z0"][h][1]
                           + 0.05 * members["clone"][h][1]
                           + 0.05 * members["mv"][h][1])
        spread *= 1.0 + cell["w5"] * size * dis_sd
        sig_lo = min(spread * band["alo"] * (1.0 + cell["w1"] * size),
                     cell["cap"])
        sig_hi = min(spread * band["ahi"], cell["cap"])
        centre = mu + shift
        qs = []
        for z in Z_VALUES:
            sig = sig_lo if z < 0 else sig_hi
            qs.append(max(1.0, math.exp(centre + z * sig)))
        out[h] = qs
    return out


# --------------------------------------------------------------------- main
def _forecast(case):
    tgt, lib = _load_case(case)
    if not lib:
        flat = {h: [max(1.0, math.exp(tgt["lnV1"]))] * 5
                for h in HOURS}
        return {"hours": [{"hour": h, "q": flat[h]} for h in HOURS]}
    members = _build_members(lib, tgt)
    combined = _combine(members, tgt["lnV1"])
    return {"hours": [{"hour": h, "q": combined[h]} for h in HOURS]}


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(out_dir, exist_ok=True)
    for name in sorted(os.listdir(in_dir)):
        if not name.endswith(".in"):
            continue
        with open(os.path.join(in_dir, name)) as fh:
            case = json.load(fh)
        result = _forecast(case)
        stem = name[:-3]
        with open(os.path.join(out_dir, stem + ".out"), "w") as fh:
            json.dump(result, fh)
    return 0


if __name__ == "__main__":
    sys.exit(main())
