"""Guards for the three boundary locators and the ladder table.

The paper uses three quantities for "where is the boundary", and Section 5 now
recomputes all three on every rung of every control ladder (`locators.json`).
That table supports four claims, and each of them can fail silently:

  1. the three locators disagree on the reference arm, so the paper's admission
     that they are not the same number is grounded rather than decorative;
  2. capacity and training-set size leave every locator unchanged, which is the
     content of "these buy in-band accuracy, not out-of-band reach";
  3. the mode budget leaves the largest step on one interval across all eight
     rungs, which is the precise form of the negative result;
  4. widening the band moves all three, and the widest band has no out-of-band
     node at all, so the n/a in that row is an absence and not a boundary at the
     last node.

Every assertion below is checked against a **second, independent read** of the
per-seed `eval_*.json` files, written here rather than imported from
`locators.py`.  If the archived curves and the stored table ever diverge, one of
the two paths is stale and the guard says so; if both drifted together the guard
would stay quiet, which is why the probes in `test_gate_falsification.py` mutate
the stored table and require the comparison to fire.
"""
import os
import sys
import json

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

FAIL = []
_SEEDS = (0, 1, 2)
_MULT = 6.0
_CONTRAST = "0.25"


def ok(msg):
    print("  [ok]   %s" % msg)


def bad(msg):
    print("  [FAIL] %s" % msg)
    FAIL.append(msg)


def note(msg):
    print("  [note] %s" % msg)


# ------------------------------------------------------------- second path
def _curve(prefix, res):
    """Seed-mean curve, recomputed from the per-seed eval files.

    Deliberately not sharing code with locators.py: the aggregation is written
    again here, so a bug in one implementation is not invisible to both.
    """
    per_seed = []
    for s in _SEEDS:
        p = os.path.join(res, "eval_%s_s%d.json" % (prefix, s))
        if not os.path.exists(p):
            return None
        row = json.load(open(p, encoding="utf-8")).get("sweep", {}).get(_CONTRAST)
        if not row:
            return None
        per_seed.append(np.array([x["mean"] for x in row], float))
    f = np.array([x["f"] for x in row], float)
    n = min(len(c) for c in per_seed)
    return f[:n], np.array([c[:n] for c in per_seed]).mean(axis=0)


def _edge_node(f, edge):
    c = f[f <= edge + 1e-9]
    return float(c.max()) if len(c) else None


def _three(f, e, edge):
    ref = _edge_node(f, edge)
    if ref is None:
        return None, None, None
    i0 = int(np.where(f == ref)[0][0])
    fstar = None
    for i in range(i0 + 1, len(f)):
        if e[i] > _MULT * e[i0]:
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


# ------------------------------------------------------------------ guard
def locator_guard(res=None, root=None, eval_res=None):
    """The stored ladder table must be reproducible from the per-seed curves."""
    res = res or C.RES
    root = root or C.ROOT
    # `eval_res` lets a probe point at a mutated table while the curves
    # stay real; in production the two are the same directory.
    eval_res = eval_res or res
    p = os.path.join(res, "locators.json")
    if not os.path.exists(p):
        note("locators: no results/locators.json, table not asserted")
        return
    d = json.load(open(p, encoding="utf-8"))
    L = d["ladders"]

    # 1) independent recomputation, arm by arm
    drift = []
    for name, lad in L.items():
        for r in lad["rows"]:
            got = _curve(r["prefix"], eval_res)
            if got is None:
                drift.append((name, r["label"], "missing eval files"))
                continue
            f, e = got
            fs, on, st = _three(f, e, r["edge"])
            for key, mine, stored in (("f*", fs, r["fstar"]),
                                      ("onset", on, r["onset"]),
                                      ("step", st, r["step"])):
                if mine is None and stored is None:
                    continue
                if mine is None or stored is None:
                    drift.append((name, r["label"], "%s %s vs %s"
                                  % (key, mine, stored)))
                    continue
                if isinstance(mine, tuple):
                    if tuple(round(x, 2) for x in mine) != tuple(stored):
                        drift.append((name, r["label"], "%s %s vs %s"
                                      % (key, mine, stored)))
                elif abs(float(mine) - float(stored)) > 0.011:
                    drift.append((name, r["label"], "%s %s vs %s"
                                  % (key, mine, stored)))
    if drift:
        bad("locators: %d of the stored values are not reproducible from the "
            "per-seed curves, e.g. %s" % (len(drift), drift[:3]))
    else:
        ok("locators: every stored value in all %d rungs is reproduced by an "
           "independent read of the per-seed curves"
           % sum(len(v["rows"]) for v in L.values()))

    # 2) the three locators disagree on the reference arm -- otherwise the
    #    paper's "they are not the same number" would be an empty admission
    ref = [r for r in L["band"]["rows"] if r["label"].startswith("[1,4]")]
    if not ref:
        bad("locators: the reference rung is missing from the band ladder")
    else:
        r = ref[0]
        vals = (r["fstar"], r["onset"], tuple(r["step"]))
        if len(set(vals)) != 3:
            bad("locators: the reference arm returns %s for the three "
                "locators, so they are not three distinct diagnostics" % (vals,))
        else:
            ok("locators: on the reference arm the three locators give three "
               "different answers (%.2f, %.2f, %.2f-%.2f), which is what the "
               "text admits" % (r["fstar"], r["onset"], r["step"][0], r["step"][1]))

    # 3) capacity and data ladders: no locator moves
    for k, tag in (("cap", "capacity"), ("ndata", "training-set size")):
        rows = L[k]["rows"]
        fs = {r["fstar"] for r in rows if r["fstar"] is not None}
        on = {r["onset"] for r in rows if r["onset"] is not None}
        st = {(round(s[0], 2), round(s[1], 2)) for s in (r["step"] for r in rows) if s}
        if len(fs) > 1 or len(on) > 1 or len(st) > 1:
            bad("locators: the %s ladder moves a locator (f*=%s, onset=%s, "
                "steps=%s)" % (tag, sorted(fs), sorted(on), sorted(st)))
        else:
            ok("locators: across the %s ladder (%d rungs) no locator returns "
               "more than one value; the step interval is fixed at %.2f-%.2f"
               % (tag, len(rows), list(st)[0][0], list(st)[0][1]))

    # 4) the mode ladder: one interval for the step, and f* silent at low budget
    rows = L["mode"]["rows"]
    st = {(round(s[0], 2), round(s[1], 2)) for s in (r["step"] for r in rows) if s}
    silent = [r for r in rows if r["fstar"] is None]
    if len(st) != 1:
        bad("locators: the mode ladder's largest step takes %d different "
            "intervals, so 'the budget does not move the boundary' is not what "
            "this table shows" % len(st))
    elif not silent:
        bad("locators: the f* rule fires on every mode budget, but the text "
            "says it declines at low budget; one of the two is wrong")
    else:
        ok("locators: across all %d mode budgets the largest step occupies "
           "%.2f-%.2f, while f* declines to fire on %d of them"
           % (len(rows), list(st)[0][0], list(st)[0][1], len(silent)))
    # the onset, by contrast, does move -- and the text says so, so a silent
    # move here would be an undetected edit of the paper's one admitted caveat
    on = {r["onset"] for r in rows if r["onset"] is not None}
    if len(on) < 2:
        bad("locators: the half-plateau onset is constant across the mode "
            "ladder, but the text reports it reading one node early at the "
            "smallest budget")
    else:
        ok("locators: the half-plateau onset does move on the mode ladder "
           "(%s), matching the caveat in the text"
           % "-".join("%.2f" % v for v in sorted(on)))

    # 5) the band ladder is the positive control, and its widest rung has no
    #    locator at all: an absence, not a boundary at the last node
    rows = L["band"]["rows"]
    fs = [r["fstar"] for r in rows if r["fstar"] is not None]
    if len(fs) < 3 or (max(fs) - min(fs)) < 3.0:
        bad("locators: widening the band moves f* over only %s, so the positive "
            "control does not show movement" % (sorted(fs),))
    else:
        ok("locators: widening the band moves f* from %.2f to %.2f, a spread of "
           "%.2f, and the onset with it" % (min(fs), max(fs), max(fs) - min(fs)))
    off = [r for r in rows
           if r["fstar"] is None and r["onset"] is None and r["step"] is None]
    if len(off) != 1:
        bad("locators: %d rungs have no locator; exactly the widest band "
            "should, because its grid lies inside the trained band" % len(off))
    else:
        ok("locators: %s returns n/a under all three locators -- every test "
           "node lies inside the trained band, which is reported as an absence "
           "rather than as a boundary at the last node" % off[0]["label"])

    # 6) the printed macros must agree with the table
    gm = os.path.join(root, "paper", "en", "gen_macros.tex")
    if os.path.exists(gm):
        raw = open(gm, encoding="utf-8").read()
        want = {}
        if ref:
            want["LocRefFstar"] = "%.2f" % ref[0]["fstar"]
            want["LocRefOnset"] = "%.2f" % ref[0]["onset"]
        hit = [r for r in L["mode"]["rows"] if r["fstar"] is not None]
        if hit:
            want["LocModeFstar"] = "%.2f" % hit[0]["fstar"]
        want["LocModeSilent"] = str(len(silent))
        # the two-effect sentence quotes one ratio per ladder, computed from
        # the same in-band column the table prints
        for k, tag in (("ndata", "Data"), ("mode", "Mode"), ("cap", "Cap")):
            ins = [r["inband"] for r in L[k]["rows"] if r["inband"]]
            if len(ins) > 1:
                if tag == "Cap":
                    want["LocCapInLo"] = "%.3f" % min(ins)
                    want["LocCapInHi"] = "%.3f" % max(ins)
                else:
                    want["Loc%sInRatio" % tag] = "%.1f" % (max(ins) / min(ins))
        miss = []
        for k, v in want.items():
            if ("\\newcommand{\\%s}{%s}" % (k, v)) not in raw:
                miss.append((k, v))
        if miss:
            bad("locators: gen_macros does not carry the table's values: %s"
                % miss)
        else:
            ok("locators: the %d macros the prose quotes match the table "
               "exactly" % len(want))
