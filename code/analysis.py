# Author: Longgang Wang <longgangwang@163.com>
# Affiliation: Luoyang Institute of Science and Technology, Luoyang 471000, Henan, China

"""Aggregate evaluation JSONs into the paper's tables and fit the scaling law."""
import os
import sys
import glob
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C


def load_all(pattern=os.path.join(C.RES, "eval_*.json")):
    out = {}
    for p in sorted(glob.glob(pattern)):
        d = C.load_json(p)
        out[os.path.basename(p)] = d
    return out


def curve(d, vmax, key="sweep"):
    rows = d[key][str(vmax)] if str(vmax) in d[key] else d[key]["%.2f" % vmax]
    return np.array([r["f"] for r in rows]), np.array([r["mean"] for r in rows])


def group_by_arm(runs):
    """runs: dict name -> eval json.  Returns arm -> list of (name, json)."""
    arms = {}
    for n, d in runs.items():
        tag = d.get("arm") or d.get("run")
        arms.setdefault(tag, []).append((n, d))
    return arms


def fit_power(f, e, f0):
    """Fit e ~ A * (f/f0)^alpha on points with f > f0."""
    m = f > f0
    if m.sum() < 3:
        return None
    x = np.log(f[m] / f0)
    y = np.log(e[m])
    A = np.vstack([x, np.ones_like(x)]).T
    coef, res, *_ = np.linalg.lstsq(A, y, rcond=None)
    yhat = A @ coef
    ss = 1 - ((y - yhat) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-30)
    return dict(alpha=float(coef[0]), A=float(np.exp(coef[1])), R2=float(ss),
                n=int(m.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default=os.path.join(C.RES, "eval_*.json"))
    ap.add_argument("--out", default=os.path.join(C.RES, "summary.json"))
    a = ap.parse_args()
    runs = load_all(a.pattern)
    print("loaded %d evaluation files" % len(runs))
    for n in runs:
        print("   ", n, runs[n].get("tag", ""), "epoch", runs[n].get("ckpt_epoch"))
    C.save_json({k: dict(run=v.get("run"), arm=v.get("arm"), tag=v.get("tag"),
                         ckpt_epoch=v.get("ckpt_epoch"), ckpt_val=v.get("ckpt_val"))
                 for k, v in runs.items()}, a.out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
