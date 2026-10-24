"""Three boundary locators, applied to every rung of every control ladder.

Why this file exists
--------------------
Section 5 reports three different quantities that all answer "where is the
boundary":

  f*         first test node whose error exceeds 6x the band-edge node
             (stated rule, used for the mode-budget sweep)
  onset      first node reaching 50% of the arm's own plateau
             (curve statistic, used for the interval table)
  step       the adjacent-node interval carrying the largest increase

Each was defined at its point of use, and the prose compares a f*-shift for one
control against a step-shift for another.  A referee is entitled to ask whether
that is the same quantity twice.  Rather than assert that it is, this script
recomputes all three on every rung of every ladder from the archived per-seed
evaluation files, so the paper can report the *locator* spread as a measured
quantity and say plainly where the three disagree.

The answer it produces is not the flattering one:
  * on the reference arm the three locators return 4.50, 4.25 and [4.00,4.25]
    respectively -- they are not the same number;
  * but across the capacity and data ladders each of them is constant, so the
    conclusion "capacity and data do not move the boundary" holds under all
    three rather than under the one that was quoted;
  * across the band-width ladder all three move with the band edge, which is
    the positive control.

Conventions
-----------
All means are taken over the paired media first (already stored in `sweep`)
and then over the three seeds, so a locator is read off a seed-mean curve.
The band edge for an arm is the largest test node not exceeding that arm's
training edge B; for B in {2,3} the edge itself is not a node, which is why
the paper brackets those onsets between adjacent nodes.

Only the v_max=0.25 row is used: it is the arm every ladder shares, and
mixing contrasts would confound the ladder with the contrast.

Usage
-----
    python code/locators.py            # prints the table
    python code/locators.py --write    # also writes results/locators.json
"""
import os
import sys
import json
import glob

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
CONTRAST = "0.25"
F_BAND = 4.0            # training band [1,4] for every ladder except the width one
MULT = 6.0              # stated threshold of the f* rule


# ---------------------------------------------------------------- loading
def load_curve(prefix, seeds=(0, 1, 2), res=RES):
    """Seed-mean error curve for one arm, from its per-seed eval_*.json.

    Returns (f, e, meta) or None.  `e` is the mean over seeds of the per-media
    mean error, i.e. the same aggregation the tables use.
    """
    curves, meta = [], None
    for s in seeds:
        p = os.path.join(res, "eval_%s_s%d.json" % (prefix, s))
        if not os.path.exists(p):
            return None
        d = json.load(open(p, encoding="utf-8"))
        row = d.get("sweep", {}).get(CONTRAST)
        if not row:
            return None
        f = np.array([x["f"] for x in row], float)
        e = np.array([x["mean"] for x in row], float)
        curves.append(e)
        if meta is None:
            meta = dict(params=d.get("params"), modes=d.get("modes"),
                        train_data=os.path.basename(str(d.get("train_data", ""))),
                        n=int(d.get("n", 0)))
    n = min(len(c) for c in curves)
    return f[:n], np.mean([c[:n] for c in curves], axis=0), meta


# --------------------------------------------------------------- locators
def loc_fstar(f, e, edge):
    """First node beyond the band whose error exceeds MULT x the edge node."""
    ref = _edge_node(f, edge)
    if ref is None:
        return None
    i0 = int(np.where(f == ref)[0][0])
    thr = MULT * e[i0]
    for i in range(i0 + 1, len(f)):
        if e[i] > thr:
            return float(f[i])
    return None


def loc_onset(f, e, edge):
    """First node reaching 50% of the arm's own plateau (nodes > edge + 1)."""
    out = f > edge + 1.0
    if out.sum() < 2:
        return None
    plat = float(e[out].mean())
    for i in range(len(f)):
        if e[i] > 0.5 * plat:
            return float(f[i])
    return None


def loc_step(f, e, edge):
    """Adjacent-node interval beyond the band carrying the largest increase.

    Restricted to intervals whose left node is at or beyond the band-edge node.
    Without that restriction the locator is undefined where the grid lies
    entirely inside the trained band: on [1,12] the largest rise is at
    2.5-3.5, which is ordinary in-band learning, not a boundary.  Returning
    that number would manufacture a boundary where the experiment says the
    boundary has moved off the grid.
    """
    ref = _edge_node(f, edge)
    if ref is None:
        return None
    i0 = int(np.where(f == ref)[0][0])
    if i0 >= len(f) - 1:
        return None
    d = np.diff(e)
    i = i0 + int(np.argmax(d[i0:]))
    return (float(f[i]), float(f[i + 1]))


def _edge_node(f, edge):
    """Largest test node not exceeding the arm's training edge."""
    cand = f[f <= edge + 1e-9]
    return float(cand.max()) if len(cand) else None


def locators(prefix, edge=F_BAND, res=RES):
    got = load_curve(prefix, res=res)
    if got is None:
        return None
    f, e, meta = got
    st = loc_step(f, e, edge)
    return dict(prefix=prefix, edge=edge,
                fstar=loc_fstar(f, e, edge),
                onset=loc_onset(f, e, edge),
                step=st,
                plateau=float(e[f > edge + 1.0].mean()) if (f > edge + 1).sum() else None,
                inband=float(e[f < edge].mean()) if (f < edge).sum() else None,
                params=meta["params"], modes=meta["modes"],
                train=meta["train_data"])


# ---------------------------------------------------------------- ladders
# (ladder, [(label, prefix, edge), ...]) -- the reference arm is repeated in
# each ladder so that the rung it occupies is visible rather than implied.
LADDERS = {
    "mode": [(str(m), "mode_m%02d" % m, F_BAND)
             for m in (1, 2, 3, 4, 8, 12, 24, 32)],
    "cap": [("w32 (ref)", "main_fno_a025", F_BAND),
            ("w64", "cap_w64_a025", F_BAND)],
    "ndata": [("1536", "ndata_n01536", F_BAND),
              ("3072", "ndata_n03072", F_BAND),
              ("6144", "ndata_n06144", F_BAND),
              ("12288 (ref)", "main_fno_a025", F_BAND)],
    "band": [("[1,2]", "band_b2", 2.0),
             ("[1,3]", "band_b3", 3.0),
             ("[1,4] (ref)", "main_fno_a025", 4.0),
             ("[1,6]", "band_b6", 6.0),
             ("[1,12]", "band_b12", 12.0)],
}


def _fmt(v):
    if v is None:
        return "n/a"
    if isinstance(v, tuple):
        return "%.2f-%.2f" % v
    return "%.2f" % v


def main():
    write = "--write" in sys.argv
    out = {"contrast": CONTRAST, "mult": MULT, "ladders": {}}
    print("v_max = %s, seed-mean curves, three locators per rung\n" % CONTRAST)
    for name, rows in LADDERS.items():
        print("-- %s --" % name)
        print("  %-12s %8s %8s %6s %8s %10s %8s"
              % ("rung", "f*", "onset", "step", "in-band", "plateau", "params"))
        recs = []
        for label, prefix, edge in rows:
            r = locators(prefix, edge=edge)
            if r is None:
                print("  %-12s MISSING (no eval_%s_s0.json)" % (label, prefix))
                continue
            r["label"] = label
            recs.append(r)
            print("  %-12s %8s %8s %6s %8s %8s %8d"
                  % (label, _fmt(r["fstar"]), _fmt(r["onset"]),
                     _fmt(r["step"]),
                     "n/a" if r["inband"] is None else "%.4f" % r["inband"],
                     "n/a" if r["plateau"] is None else "%.4f" % r["plateau"],
                     r["params"] or 0))
        # spread of each locator across the rungs, which is the quantity the
        # control experiments are actually about
        sp = {}
        for k in ("fstar", "onset"):
            vals = [r[k] for r in recs if r[k] is not None]
            sp[k] = (max(vals) - min(vals)) if vals else None
        steps = [r["step"] for r in recs if r["step"]]
        sp["step_lo"] = min(s[0] for s in steps) if steps else None
        sp["step_hi"] = max(s[1] for s in steps) if steps else None
        sp["n"] = len(recs)
        out["ladders"][name] = dict(rows=recs, spread=sp)
        print("  spread: f* %s  onset %s  step [%s, %s]  over %d rungs\n"
              % ("n/a" if sp["fstar"] is None else "%.2f" % sp["fstar"],
                 "n/a" if sp["onset"] is None else "%.2f" % sp["onset"],
                 "n/a" if sp["step_lo"] is None else "%.2f" % sp["step_lo"],
                 "n/a" if sp["step_hi"] is None else "%.2f" % sp["step_hi"],
                 sp["n"]))
    if write:
        p = os.path.join(RES, "locators.json")
        json.dump(out, open(p, "w", encoding="utf-8"), indent=1)
        print("wrote %s" % p)


if __name__ == "__main__":
    main()
