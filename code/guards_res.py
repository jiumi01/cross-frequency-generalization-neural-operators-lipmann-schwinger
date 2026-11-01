"""Guards for the resolution study (contribution 1's discretisation bound and
contribution 4's refined Marmousi rung).

The claim these guards defend
-----------------------------
"Frequency and per-wavelength resolution co-vary by construction, so the
benchmark cannot separate them; what bounds the consequence is that refining the
mesh does not move the measured boundary, the jump or the Marmousi contrast."

That sentence is only worth printing if the refinement was real and if the
invariance is measured with instruments whose resolution is finer than the
invariance.  So the guards check, in order:

  1. the two arms are *different models* (different checkpoint digests) -- the
     failure this rules out is silently scoring one model twice;
  2. the stored locators are reproducible from the stored curves;
  3. the boundary does not move by more than one step of the test grid that
     resolves it, at the coarse grid's value, for every seed of the refined arm;
  4. the jump factor moves by no more than the seed spread of the two arms;
  5. the paired-continuum control -- the same media on both meshes -- agrees,
     which is what removes "two different draws" as an explanation;
  6. the Marmousi rung's *coarse* side is reproduced by an independent script
     (`make_realgeo.py`), arm by arm and window by window;
  7. the synthetic *coarse* side is reproduced by the main evaluator
     (`evaluate.py`), frequency by frequency;
  8. the out-of-band energy fraction of the Marmousi windows is a property of
     the 3 km window and not of the sampling, so the two grids must agree on it;
  9. the macros the prose quotes carry the table's values.

Points 6 and 7 are the ones that make the rest non-circular: two code paths that
share no aggregation code must produce the same numbers, or one of them is
stale.  The probes in `test_gate_falsification.py` mutate the archive and
require each assertion to fire.
"""
import os
import sys
import json
import hashlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

FAIL = []
EDGE = 4.0
MULT = 6.0
NODE = 0.25          # spacing of the densified test grid around the band edge


def ok(msg):
    print("  [ok]   %s" % msg)


def bad(msg):
    print("  [FAIL] %s" % msg)
    FAIL.append(msg)


def note(msg):
    print("  [note] %s" % msg)


# ------------------------------------------------------- second implementation
def _three(f, e, edge=EDGE):
    """The three locators, re-derived here without importing locators.py."""
    f = np.asarray(f, float)
    e = np.asarray(e, float)
    cand = f[f <= edge + 1e-9]
    if not len(cand):
        return None, None, None
    ref = float(cand.max())
    i0 = int(np.where(f == ref)[0][0])
    fstar = None
    for i in range(i0 + 1, len(f)):
        if e[i] > MULT * e[i0]:
            fstar = float(f[i])
            break
    out = f > edge + 1.0
    onset = None
    if out.sum() >= 2:
        plat = float(e[out].mean())
        for i in range(len(f)):
            if e[i] > 0.5 * plat:
                onset = float(f[i])
                break
    step = None
    if i0 < len(f) - 1:
        d = np.diff(e)
        j = i0 + int(np.argmax(d[i0:]))
        step = (round(float(f[j]), 2), round(float(f[j + 1]), 2))
    return fstar, onset, step


def _md5(p):
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _load(res):
    p = os.path.join(res, "res_study.json")
    if not os.path.exists(p):
        return None
    return json.load(open(p, encoding="utf-8"))


def _jump_range(lb):
    js = lb.get("jump_seeds") or [lb["jump"]]
    return max(js) - min(js)


# ------------------------------------------------------------------- the guard
def res_guard(res=None, root=None):
    res = res or C.RES
    root = root or C.ROOT
    d = _load(res)
    if d is None:
        note("res_study: no results/res_study.json, resolution claims not asserted")
        return
    A = d.get("part_a", {}).get("grids", {})
    B = d.get("part_b", {}).get("grids", {})
    Pc = d.get("part_c", {}).get("grids", {})
    if "64" not in A or "128" not in A:
        bad("res_study: the synthetic part is missing one of the two grids")
        return
    a, b = A["64"], A["128"]

    # 1) two different models, or the invariance is vacuous
    dg, missing = [], []
    for s in range(3):
        p1 = os.path.join(root, "runs", "main_fno_a025_s%d" % s, "checkpoint.pt")
        p2 = os.path.join(root, "runs", "main_fno_a025_128_s%d" % s, "checkpoint.pt")
        if not (os.path.exists(p1) and os.path.exists(p2)):
            missing.append(s)
            continue
        if _md5(p1) == _md5(p2):
            dg.append(("identical", s))
    if dg:
        bad("res_study: the refined arm is not a distinct set of checkpoints (%s)"
            % dg)
    elif missing:
        note("res_study: %d checkpoint pairs are not present under this root, so "
             "the distinctness of the arms cannot be asserted here" % len(missing))
    else:
        ok("res_study: all three refined checkpoints differ from their coarse "
           "counterparts by digest, so the comparison scores different models")

    # 2) stored locators reproduced from stored curves
    drift = []

    def _cmp(where, f, e, lb):
        fs, on, st = _three(f, e)
        for key, mine, stored in (("f*", fs, lb["fstar"]),
                                  ("onset", on, lb["onset"]),
                                  ("step", st, lb["step"])):
            if mine is None and stored is None:
                continue
            if mine is None or stored is None:
                drift.append((where, key, mine, stored))
            elif isinstance(mine, tuple):
                if tuple(round(x, 2) for x in mine) != tuple(round(x, 2) for x in stored):
                    drift.append((where, key, mine, stored))
            elif abs(float(mine) - float(stored)) > 0.011:
                drift.append((where, key, mine, stored))

    for g, rows in d.get("part_a", {}).get("curves", {}).items():
        f = [r["f"] for r in rows]
        e = [float(np.mean([r["s%d" % s]["mean"] for s in range(3) if "s%d" % s in r]))
             for r in rows]
        _cmp("a/%s" % g, f, e, A[g])
    brows = d.get("part_b", {}).get("rows") or []
    if brows:
        for g in ("64", "128"):
            keys = sorted({k for r in brows for k in r
                           if k.startswith("g%s_s" % g)})
            if not keys:
                continue
            f = [r["f"] for r in brows]
            e = [float(np.mean([r[k]["mean"] for k in keys])) for r in brows]
            _cmp("b/%s" % g, f, e, B[g])
    for g, geos in d.get("part_c", {}).get("curves", {}).items():
        for gname, rows in geos.items():
            f = [r["f"] for r in rows]
            e = [float(np.mean([r["s%d" % s]["mean"] for s in range(3)
                                if "s%d" % s in r])) for r in rows]
            _cmp("c/%s/%s" % (g, gname), f, e, Pc[g]["per_geometry"][gname])
    if drift:
        bad("res_study: %d stored locator values are not reproducible from the "
            "stored curves: %s" % (len(drift), drift[:3]))
    else:
        ok("res_study: every stored locator in all three parts is reproduced "
           "from the stored per-frequency curves by an independent implementation")

    # 3) the boundary does not move beyond one node of the test grid
    if a["fstar"] is None or b["fstar"] is None:
        bad("res_study: a locator returned n/a on the reference arm, so the "
            "refinement claim cannot be read")
    else:
        shift = abs(b["fstar"] - a["fstar"])
        seeds = [v for v in (b.get("fstar_seeds") or []) if v is not None]
        worst = max([abs(v - a["fstar"]) for v in seeds], default=0.0)
        if shift > NODE + 1e-9:
            bad("res_study: f* moves by %.2f from the coarse to the refined grid, "
                "more than the %.2f spacing of the nodes that resolve it"
                % (shift, NODE))
        elif seeds and worst > NODE + 1e-9:
            bad("res_study: f* is stable on the seed mean but one refined seed "
                "reads %.2f, i.e. %.2f from the coarse value: the invariance is "
                "not seed-robust" % (min(seeds, key=lambda v: abs(v - a["fstar"])), worst))
        else:
            ok("res_study: f* reads %.2f at 64^2 and %.2f at 128^2 (shift %.2f, "
               "one node = %.2f); every refined seed lies within one node"
               % (a["fstar"], b["fstar"], shift, NODE))

    # 4) the jump factor moves by no more than the seed spread of the two arms
    tol = _jump_range(a) + _jump_range(b)
    dj = abs(b["jump"] - a["jump"])
    if dj > tol + 1e-9:
        bad("res_study: the jump factor moves from %.2f to %.2f (%.2f), which "
            "exceeds the combined seed spread %.2f" % (a["jump"], b["jump"], dj, tol))
    else:
        ok("res_study: the jump factor moves from %.2f to %.2f (%.2f), inside "
           "the combined seed spread of %.2f" % (a["jump"], b["jump"], dj, tol))

    # 5) the paired-continuum control: same media, two meshes
    if "64" in B and "128" in B:
        pb, fb = B["64"], B["128"]
        if pb["fstar"] is not None and fb["fstar"] is not None:
            if abs(fb["fstar"] - pb["fstar"]) > NODE + 1e-9:
                bad("res_study: with the media held fixed, f* still moves from "
                    "%.2f to %.2f, so the coarse/fine agreement in (a) is a "
                    "property of the two draws, not of the mesh"
                    % (pb["fstar"], fb["fstar"]))
            else:
                ok("res_study: with one draw of media discretised on both grids, "
                   "f* reads %.2f and %.2f -- the agreement survives holding the "
                   "media fixed" % (pb["fstar"], fb["fstar"]))
        else:
            note("res_study: the paired-continuum control returned n/a on a grid")

    # 6) the coarse Marmousi side, reproduced by make_realgeo.py
    rg = os.path.join(res, "realgeo.json")
    if "64" in Pc and os.path.exists(rg):
        r = json.load(open(rg, encoding="utf-8"))
        dev = []
        for g, lb in Pc["64"]["per_geometry"].items():
            ref = r["summary"]["per_geometry"].get(g, {}).get("0.25")
            if ref is None:
                dev.append((g, "absent from realgeo.json"))
                continue
            if lb["fstar"] is None or ref["f_star"] is None:
                dev.append((g, "f* n/a in one path"))
                continue
            if abs(lb["fstar"] - ref["f_star"]) > 1e-9:
                dev.append((g, "f* %.3f vs %.3f" % (lb["fstar"], ref["f_star"])))
            if abs(lb["jump"] - ref["r5"]) > 1e-6:
                dev.append((g, "r5 %.6f vs %.6f" % (lb["jump"], ref["r5"])))
        if dev:
            bad("res_study: the coarse Marmousi side does not reproduce "
                "realgeo.json: %s" % dev[:3])
        else:
            ok("res_study: the coarse Marmousi side reproduces realgeo.json "
               "f* and r5 window by window, from a separate script")

    # 7) the coarse synthetic side, reproduced by evaluate.py
    ev = os.path.join(res, "eval_main_fno_a025_s0.json")
    rows = d.get("part_a", {}).get("curves", {}).get("64")
    if rows and os.path.exists(ev):
        e0 = json.load(open(ev, encoding="utf-8"))["sweep"]["0.25"]
        worst = 0.0
        for r0, r1 in zip(e0, rows):
            if abs(r0["f"] - r1["f"]) > 1e-9:
                worst = 9.9
                break
            worst = max(worst, abs(r0["mean"] - r1["s0"]["mean"]))
        if worst > 1e-3:
            bad("res_study: the coarse synthetic side deviates from "
                "eval_main_fno_a025_s0.json by %.2e per frequency, so the two "
                "code paths disagree" % worst)
        else:
            ok("res_study: the coarse synthetic curve reproduces "
               "eval_main_fno_a025_s0.json to %.1e per frequency" % worst)

    # 8) the windows' out-of-band fraction belongs to the survey, not the grid
    if "64" in Pc and "128" in Pc:
        g64 = [Pc["64"]["per_geometry"]["geo_%s" % w]["bands"] for w in ("west", "centre", "east")]
        g128 = [Pc["128"]["per_geometry"]["geo_%s" % w]["bands"] for w in ("west", "centre", "east")]
        modes = sorted(set(g64[0]) & set(g128[0]))
        worst = 0.0
        for M in modes:
            v64 = np.mean([b[M] for b in g64])
            v128 = np.mean([b[M] for b in g128])
            worst = max(worst, abs(v128 - v64))
        if worst > 0.01:
            bad("res_study: the out-of-band energy fraction of the same 3 km "
                "windows differs between grids by %.3f, so the contrast between "
                "real and synthetic geology is grid-dependent" % worst)
        else:
            ok("res_study: the windows' out-of-band fraction agrees across the "
               "two grids to %.4f at every mode (%d modes), so the real-vs-"
               "synthetic contrast is a property of the survey" % (worst, len(modes)))

    # 9) the macros must carry the table's values
    gm = os.path.join(root, "paper", "en", "gen_macros.tex")
    if os.path.exists(gm) and "128" in Pc:
        raw = open(gm, encoding="utf-8").read()
        want = {
            "ResFstarCoarse": "%.2f" % a["fstar"],
            "ResFstarFine": "%.2f" % b["fstar"],
            "ResJumpCoarse": "%.2f" % a["jump"],
            "ResJumpFine": "%.2f" % b["jump"],
            "ResGeoFstarCoarse": "%.2f" % Pc["64"]["family"]["real"]["fstar"],
            "ResGeoFstarFine": "%.2f" % Pc["128"]["family"]["real"]["fstar"],
            "ResGeoSynFstarFine": "%.2f" % Pc["128"]["family"]["synthetic"]["fstar"],
        }
        miss = [(k, v) for k, v in want.items()
                if ("\\newcommand{\\%s}{%s}" % (k, v)) not in raw]
        if miss:
            bad("res_study: gen_macros does not carry the table's values: %s" % miss)
        else:
            ok("res_study: the %d resolution macros match the table exactly"
               % len(want))


# ------------------------------------------------ the qualitative verdicts
# These are *not* read from the archive.  They are the sentences the paper
# prints, written down as booleans so that a re-run which changes one of them
# breaks the gate instead of quietly contradicting the prose.  Each entry was
# set by reading the paragraph it belongs to:
#
#   jump_rises             Sec. gridstudy  "The jump factor ... does move"
#   jump_ranges_disjoint   Sec. gridstudy  "the two arms' per-seed ranges do
#                                           not overlap"
#   locators_identical     Sec. gridstudy  "All three locators land where they
#                                           were"
#   geo_jump_order_flip    Sec. realgeo    "The jump factor instead reverses"
#   geo_plateau_sep_coarse Sec. realgeo    "does not separate ... at 64^2"
#   geo_plateau_sep_fine   Sec. realgeo    "The plateau does: at 128^2"
CLAIM = dict(
    locators_identical=True,
    jump_rises=True,
    jump_ranges_disjoint=True,
    geo_jump_order_flip=True,
    geo_plateau_sep_coarse=False,
    geo_plateau_sep_fine=True,
)
WHERE = dict(
    locators_identical="the three locators do not move",
    jump_rises="the jump factor rises with refinement",
    jump_ranges_disjoint="the two arms' per-seed jump ranges do not overlap",
    geo_jump_order_flip="the jump factor reverses the two families' order",
    geo_plateau_sep_coarse="the plateau separates the families at 64^2",
    geo_plateau_sep_fine="the plateau separates the families at 128^2",
)


def _mean_curve(rows):
    ks = [k for k in rows[0] if k.startswith("s")]
    f = np.asarray([r["f"] for r in rows], float)
    e = np.asarray([np.mean([r[k]["mean"] for k in ks]) for r in rows], float)
    return f, e


def _jump_of(f, e):
    i4 = int(np.where(np.isclose(f, 4.0))[0][0])
    i5 = int(np.where(np.isclose(f, 5.0))[0][0])
    return float(e[i5] / e[i4])


def _plat5(f, e):
    return float(e[f >= 5.0].mean())


def claim_guard(res=None, claims=None, verbose=False):
    """Fail when the archive no longer supports the sentences in the paper."""
    res = res or C.RES
    claims = CLAIM if claims is None else claims
    d = _load(res)
    if d is None:
        note("res_claim: no res_study.json, qualitative claims not asserted")
        return
    got = {}

    # (a) locators, recomputed from the seed-mean curve on each grid
    loc = {}
    for g in ("64", "128"):
        rows = d["part_a"]["curves"][g]
        f, e = _mean_curve(rows)
        loc[g] = _three(f, e)
    got["locators_identical"] = loc["64"] == loc["128"]

    # (b) the jump factor, per seed and per grid
    jj = {}
    for g in ("64", "128"):
        rows = d["part_a"]["curves"][g]
        ks = [k for k in rows[0] if k.startswith("s")]
        jj[g] = [_jump_of(*(np.asarray([r["f"] for r in rows], float),
                            np.asarray([r[k]["mean"] for r in rows], float)))
                 for k in ks]
    got["jump_rises"] = float(np.mean(jj["128"])) > float(np.mean(jj["64"]))
    got["jump_ranges_disjoint"] = min(jj["128"]) > max(jj["64"])

    # (c) Marmousi: family jump order and plateau separation, per grid
    flip, sep = {}, {}
    for g in ("64", "128"):
        cur = d["part_c"]["curves"][g]
        fam = {}
        for famname, pre in (("real", "geo_"), ("syn", "syn_")):
            names = [k for k in cur if k.startswith(pre)]
            js = [_jump_of(*_mean_curve(cur[k])) for k in names]
            ps = [_plat5(*_mean_curve(cur[k])) for k in names]
            fam[famname] = (float(np.mean(js)), min(ps), max(ps))
        flip[g] = fam["real"][0] > fam["syn"][0]
        sep[g] = fam["syn"][1] > fam["real"][2]
    got["geo_jump_order_flip"] = flip["64"] != flip["128"]
    got["geo_plateau_sep_coarse"] = sep["64"]
    got["geo_plateau_sep_fine"] = sep["128"]

    if verbose:
        print("      measured: %s" % {k: got[k] for k in sorted(got)})
        print("      jumps 64=%s 128=%s"
              % (["%.2f" % v for v in jj["64"]], ["%.2f" % v for v in jj["128"]]))

    drift = [(k, claims[k], got[k]) for k in got if got[k] != claims[k]]
    if drift:
        for k, want, have in drift:
            bad("res_claim: the paper asserts %r (%s) but the archive gives %r; "
                "re-read the paragraph before printing it"
                % (WHERE[k], want, have))
    else:
        ok("res_claim: all %d qualitative claims still hold (%s)"
           % (len(got), ", ".join(sorted(got))))
