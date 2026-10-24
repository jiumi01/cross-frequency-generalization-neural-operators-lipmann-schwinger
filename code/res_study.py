# Author: Longgang Wang <longgangwang@163.com>
# Affiliation: Luoyang Institute of Science and Technology, Luoyang 471000, Henan, China

"""Resolution study: is the boundary a property of the problem or of the mesh?

Why this file exists
--------------------
Two of the four contributions carry an unclosed obligation of the same kind.

  Contribution 1 asks for the discretisation to be controlled.  The benchmark
  fixes N=64 while the wavelength shrinks as 1/f, so "frequency" and "points per
  wavelength" co-vary by construction and cannot be separated by protocol
  design alone.  The honest way to bound that confound is to refine the mesh and
  ask whether the measured quantities move.

  Contribution 4 repeats the sweep on the Marmousi velocity field, but that rung
  was only ever run on a 64^2 grid, so the third level inherits the same open
  question as the second.

Both are answered by one experiment: retrain the reference arm with nothing
changed but the grid (128^2 instead of 64^2, same 55 epochs, same 12288
training media, same width/modes/params) and re-measure.

  Part A  native protocol on both grids.  The same seed, the same contrast arm,
          the same frequency grid, the same 200 paired media *count* -- but each
          model is fed media drawn natively at its own grid, which is what it
          was trained on.  Reported: the three locators of ``locators.py``,
          the jump factor eps(5)/eps(4), the plateau and the in-band level.

  Part B  paired continuum control.  A referee can object that Part A compares
          two different draws of media, not two meshes.  Part B removes that:
          the media are drawn ONCE at 128^2 and block-mean downsampled to 64^2,
          so both models see the same continuum object, and each is scored
          against the reference solution computed at its own grid.  Media are
          then identical by construction and only the mesh differs.

  Part C  Marmousi at both grids.  Same three windows (same rule, unchanged),
          each area-averaged to the grid in question; synthetic control fields
          drawn at the same grid; identical design otherwise.  The 64^2 side is
          recomputed here by the same code path, so the archived 64^2 rung can be
          cross-checked number by number rather than trusted.

Nothing is averaged over frequencies before a locator is read: every locator is
computed per arm on the seed-mean curve, exactly as ``locators.py`` does, so the
numbers in this file are comparable to Table 6 line by line.

Usage
-----
    python code/res_study.py --part a          # synthetic, both grids
    python code/res_study.py --part b          # paired continuum control
    python code/res_study.py --part c          # Marmousi, both grids
    python code/res_study.py --part all        # all three, one process
Results are merged into results/res_study.json, so the parts can be run one at a
time without overwriting each other.
"""
import os
import sys
import json
import time
import argparse

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import physics as P
import data as D
import evaluate as E
import locators as L
import make_realgeo as G

ARM = 0.25                    # the reference arm: every headline number lives here
VI = {"0.05": 0, "0.10": 1, "0.25": 2, "0.50": 3, "1.00": 4}["%.2f" % ARM]
EDGE = 4.0                    # training band [1,4] for this arm
GRIDS = (64, 128)
SEEDS = (0, 1, 2)
OUT = os.path.join(C.RES, "res_study.json")


# --------------------------------------------------------------------- utils
def ckpt(grid, seed):
    if grid == 64:
        return os.path.join(C.RUNS, "main_fno_a025_s%d" % seed, "checkpoint.pt")
    return os.path.join(C.RUNS, "main_fno_a025_128_s%d" % seed, "checkpoint.pt")


def load(grid, seed, dev):
    p = ckpt(grid, seed)
    if not os.path.exists(p):
        return None, None
    return E.load_model(p, dev)


def rel_err(model, grid, k, theta, v, psi_ref, dev):
    """Per-medium relative L2 error, same definition as evaluate.eval_cell."""
    x, inc = E.build(grid, k, theta, v, dev)
    with torch.no_grad():
        o = model(x)
    psi_hat = inc + torch.complex(o[:, 0], o[:, 1])
    num = torch.linalg.norm((psi_hat - psi_ref).reshape(v.shape[0], -1), dim=1)
    den = torch.linalg.norm(psi_ref.reshape(v.shape[0], -1), dim=1)
    return (num / den).cpu().numpy()


def block_mean(x, f=2):
    """(B, fN, fN) -> (B, N, N) by block averaging (antialias by construction)."""
    B, H, W = x.shape
    return x.reshape(B, H // f, f, W // f, f).mean(dim=(2, 4))


def loc_block(f, per_seed):
    """Locators on the seed-mean curve, plus the per-seed spread of each."""
    f = np.asarray(f, float)
    per_seed = [np.asarray(c, float) for c in per_seed]
    mean = np.mean(np.stack(per_seed, axis=0), axis=0)
    out = dict(
        fstar=L.loc_fstar(f, mean, EDGE),
        onset=L.loc_onset(f, mean, EDGE),
        step=L.loc_step(f, mean, EDGE),
        in_band=float(mean[f < EDGE].mean()),
        plateau=float(mean[f > EDGE + 1.0].mean()),
        # The Marmousi rung predates the locator convention and averages its
        # plateau over f >= 5 (not f > 5).  Both are stored, because the new
        # table has to quote the same number for the same arm as the table it
        # sits next to, and a silent switch of averaging window between two
        # tables of one paper is exactly the kind of drift this archive forbids.
        plateau5=float(mean[f >= 5.0].mean()),
        curve=[float(x) for x in mean],
        seeds=[float(np.mean(c)) for c in per_seed],
    )
    i4 = int(np.where(f == 4.0)[0][0])
    i5 = int(np.where(f == 5.0)[0][0])
    out["jump"] = float(mean[i5] / mean[i4])
    out["jump_seeds"] = [float(c[i5] / c[i4]) for c in per_seed]
    # per-seed locators, so a reader can see whether a shift exceeds seed noise
    out["fstar_seeds"] = [L.loc_fstar(f, c, EDGE) for c in per_seed]
    out["onset_seeds"] = [L.loc_onset(f, c, EDGE) for c in per_seed]
    out["step_seeds"] = [L.loc_step(f, c, EDGE) for c in per_seed]
    return out


def merge(block):
    """Merge one part into the archive without clobbering the other parts."""
    d = {}
    if os.path.exists(OUT):
        try:
            d = json.load(open(OUT, encoding="utf-8"))
        except Exception:
            d = {}
    d.update(block)
    d["n"] = block.get("n", d.get("n"))
    json.dump(d, open(OUT, "w", encoding="utf-8"), indent=1)
    return OUT


# ------------------------------------------------------------------ Part A
def part_a(n, dev):
    """Native protocol on 64^2 and 128^2: same arm, same f grid, same n."""
    res = dict(part_a=dict(arm=ARM, edge=EDGE, n=n, grids={}, curves={}))
    for grid in GRIDS:
        models = {}
        for s in SEEDS:
            m, ck = load(grid, s, dev)
            if m is None:
                print("  MISSING checkpoint for grid=%d seed=%d" % (grid, s), flush=True)
                continue
            models[s] = (m, ck)
        if not models:
            continue
        print("\n[A] grid=%d  models=%d" % (grid, len(models)), flush=True)
        rows, per_seed = [], {s: [] for s in models}
        for f in C.F_TEST:
            d = D.sample_batch(n, 60000 + VI, dev, vmax_lo=ARM, vmax_hi=ARM,
                               f_fixed=f, n_grid=grid, return_info=True)
            row = dict(f=f, ref_res_max=float(d["res"].max()))
            for s, (m, ck) in models.items():
                e = rel_err(m, grid, d["k"], d["theta"], d["v"], d["psi"], dev)
                per_seed[s].append(float(e.mean()))
                row["s%d" % s] = dict(mean=float(e.mean()), std=float(e.std()),
                                      median=float(np.median(e)),
                                      q90=float(np.quantile(e, 0.9)))
            rows.append(row)
            print("   f=%5.2f  " % f
                  + "  ".join("s%d=%.4f" % (s, row["s%d" % s]["mean"])
                              for s in sorted(models)), flush=True)
        f = [r["f"] for r in rows]
        lb = loc_block(f, [per_seed[s] for s in sorted(models)])
        lb["params"] = int(models[sorted(models)[0]][1]["params"])
        lb["epochs"] = int(models[sorted(models)[0]][1]["epoch"])
        lb["val"] = [float(models[s][1]["val"]) for s in sorted(models)]
        lb["ref_res_max"] = max(r["ref_res_max"] for r in rows)
        res["part_a"]["grids"][str(grid)] = lb
        res["part_a"]["curves"][str(grid)] = rows
    merge(res)
    return res


# ------------------------------------------------------------------ Part B
def part_b(n, dev):
    """Paired continuum control: one draw at 128^2, block-mean to 64^2."""
    res = dict(part_b=dict(arm=ARM, edge=EDGE, n=n, grids={}))
    models = {}
    for grid in GRIDS:
        for s in SEEDS:
            m, ck = load(grid, s, dev)
            if m is not None:
                models[(grid, s)] = (m, ck)
    if not models:
        return res
    per_seed = {k: [] for k in models}
    rows = []
    print("\n[B] paired continuum: media drawn at 128, block-mean to 64", flush=True)
    for f in C.F_TEST:
        d128 = D.sample_batch(n, 60000 + VI, dev, vmax_lo=ARM, vmax_hi=ARM,
                              f_fixed=f, n_grid=128, return_info=True)
        v64 = block_mean(d128["v"].double(), 2)
        g = torch.Generator(device=dev).manual_seed(int(60000 + VI) + 7777)
        # the 64^2 reference is solved from the downsampled field, not reused:
        # scoring a 64^2 prediction against a 128^2 target would mix the two
        # effects this part exists to separate.
        psi64, r64 = P.solve_ls(v64, d128["k"], d128["theta"], C.GAMMA, 64,
                                tol=1e-9, maxiter=900, device=dev,
                                return_info=True)
        row = dict(f=f, ref_res_max_64=float(r64.max()),
                   ref_res_max_128=float(d128["res"].max()))
        for (grid, s), (m, ck) in models.items():
            if grid == 128:
                e = rel_err(m, 128, d128["k"], d128["theta"], d128["v"],
                            d128["psi"], dev)
            else:
                e = rel_err(m, 64, d128["k"], d128["theta"], v64.float(),
                            psi64, dev)
            per_seed[(grid, s)].append(float(e.mean()))
            row["g%d_s%d" % (grid, s)] = dict(mean=float(e.mean()),
                                              std=float(e.std()))
        rows.append(row)
        print("   f=%5.2f  64=%.4f  128=%.4f"
              % (f, np.mean([row["g64_s%d" % s]["mean"] for s in SEEDS
                             if (64, s) in models]),
                 np.mean([row["g128_s%d" % s]["mean"] for s in SEEDS
                          if (128, s) in models])), flush=True)
    f = [r["f"] for r in rows]
    for grid in GRIDS:
        keys = [(grid, s) for s in SEEDS if (grid, s) in models]
        if not keys:
            continue
        lb = loc_block(f, [per_seed[k] for k in keys])
        lb["ref_res_max"] = max(r["ref_res_max_%d" % grid] for r in rows)
        res["part_b"]["grids"][str(grid)] = lb
    res["part_b"]["rows"] = rows
    merge(res)
    return res


# ------------------------------------------------------------------ Part C
def _geo_bank(grid, c, x, z, wins):
    geo = {}
    for k in ("west", "centre", "east"):
        geo["geo_" + k] = torch.from_numpy(
            G.unit01(G.resample(G.crop(c, x, z, wins[k]), grid))).double()
    for s in range(3):
        g = torch.Generator(device="cpu").manual_seed(70000 + 7777 + s)
        fld = P.make_contrast(grid, 1, 1.0, C.MM_REF, g, "cpu", kind="smooth")[0]
        geo["syn_%d" % s] = torch.from_numpy(G.unit01(fld.numpy())).double()
    return geo


def part_c(n, dev):
    """Marmousi and its synthetic control, recomputed at 64^2 and 128^2."""
    c, x, z = G.load_marmousi()
    wins, rough = G.pick_windows(c, x, z)
    res = dict(part_c=dict(arm=ARM, edge=EDGE, n=n, n_modes=G.MODES,
                           windows={k: list(wins[k]) for k in ("west", "centre", "east")},
                           window_roughness=rough, grids={}, curves={}))
    for grid in GRIDS:
        models = {}
        for s in SEEDS:
            m, ck = load(grid, s, dev)
            if m is not None:
                models[s] = (m, ck)
        if not models:
            continue
        geo = _geo_bank(grid, c, x, z, wins)
        fields = {k: v.numpy() for k, v in geo.items()}
        bands = {k: {str(M): G.band_frac_unit(fields[k], M) for M in G.MODES}
                 for k in fields}
        print("\n[C] grid=%d  windows resampled to %d^2" % (grid, grid), flush=True)
        print("    out-of-band energy fraction (mode %d): real %.4f  syn %.4f"
              % (20, np.mean([bands["geo_" + k]["20"] for k in ("west", "centre", "east")]),
                 np.mean([bands["syn_%d" % s]["20"] for s in range(3)])), flush=True)
        curves, summary = {}, {}
        t0 = time.time()
        for gname in geo:
            vunit = geo[gname]
            rows, per_seed = [], {s: [] for s in models}
            for f in C.F_TEST:
                _, _, theta = D.sample_params(n, None, None, ARM, ARM,
                                              60000 + VI, dev, f_fixed=f)
                k = torch.full((n,), 2 * np.pi * f, device=dev)
                v = (vunit.to(dev).double() * ARM).unsqueeze(0).repeat(n, 1, 1)
                psi_ref, rres = P.solve_ls(v, k, theta, C.GAMMA, grid, tol=1e-9,
                                           maxiter=900, device=dev, return_info=True)
                row = dict(f=f, ref_res_max=float(rres.max()))
                for s, (m, ck) in models.items():
                    e = rel_err(m, grid, k, theta, v, psi_ref, dev)
                    per_seed[s].append(float(e.mean()))
                    row["s%d" % s] = dict(mean=float(e.mean()), std=float(e.std()),
                                          median=float(np.median(e)))
                rows.append(row)
            curves[gname] = rows
            lb = loc_block([r["f"] for r in rows], [per_seed[s] for s in sorted(models)])
            lb["bands"] = bands[gname]
            summary[gname] = lb
            print("    %-11s f*=%.2f jump=%.2f plateau=%.3f  (%.0fs)"
                  % (gname, -1 if lb["fstar"] is None else lb["fstar"],
                     lb["jump"], lb["plateau"], time.time() - t0), flush=True)
        fam = {}
        for name, names in (("real", ["geo_west", "geo_centre", "geo_east"]),
                            ("synthetic", ["syn_0", "syn_1", "syn_2"])):
            fs = [summary[g]["fstar"] for g in names if summary[g]["fstar"] is not None]
            fam[name] = dict(
                fstar=float(np.mean(fs)) if fs else None,
                fstar_lo=float(min(fs)) if fs else None,
                fstar_hi=float(max(fs)) if fs else None,
                jump=float(np.mean([summary[g]["jump"] for g in names])),
                jump_lo=float(min(summary[g]["jump"] for g in names)),
                jump_hi=float(max(summary[g]["jump"] for g in names)),
                plateau=float(np.mean([summary[g]["plateau"] for g in names])),
                plateau_lo=float(min(summary[g]["plateau"] for g in names)),
                plateau_hi=float(max(summary[g]["plateau"] for g in names)),
                plateau5=float(np.mean([summary[g]["plateau5"] for g in names])),
                plateau5_lo=float(min(summary[g]["plateau5"] for g in names)),
                plateau5_hi=float(max(summary[g]["plateau5"] for g in names)),
                in_band=float(np.mean([summary[g]["in_band"] for g in names])))
        res["part_c"]["grids"][str(grid)] = dict(per_geometry=summary, family=fam)
        res["part_c"]["curves"][str(grid)] = curves
    merge(res)
    return res


# ------------------------------------------------------------------ report
def report():
    d = json.load(open(OUT, encoding="utf-8"))
    if "part_a" in d:
        print("\n== Part A: native protocol, 64^2 vs 128^2 (arm %.2f) ==" % ARM)
        print("  %-6s %6s %6s %10s %7s %8s %9s"
              % ("grid", "f*", "onset", "step", "jump", "in-band", "plateau"))
        for g, b in d["part_a"]["grids"].items():
            st = b["step"]
            print("  %-6s %6s %6s %10s %7.2f %8.4f %9.4f"
                  % (g, b["fstar"], b["onset"],
                     "n/a" if st is None else "%.2f-%.2f" % tuple(st),
                     b["jump"], b["in_band"], b["plateau"]))
        for g, b in d["part_a"]["grids"].items():
            print("  grid %s  f* per seed %s  jump per seed %s  val %s"
                  % (g, b["fstar_seeds"], ["%.2f" % j for j in b["jump_seeds"]],
                     ["%.4f" % v for v in b["val"]]))
    if "part_b" in d and d["part_b"].get("grids"):
        print("\n== Part B: paired continuum (same media, two meshes) ==")
        for g, b in d["part_b"]["grids"].items():
            print("  grid %-4s f*=%-5s onset=%-5s jump=%.2f  in-band=%.4f  plateau=%.4f"
                  % (g, b["fstar"], b["onset"], b["jump"], b["in_band"], b["plateau"]))
    if "part_c" in d and d["part_c"].get("grids"):
        print("\n== Part C: Marmousi, 64^2 vs 128^2 ==")
        for g, blk in d["part_c"]["grids"].items():
            fam = blk["family"]
            print("  grid %-4s  f* real %s / syn %s   jump real %.2f / syn %.2f"
                  % (g, fam["real"]["fstar"], fam["synthetic"]["fstar"],
                     fam["real"]["jump"], fam["synthetic"]["jump"]))
            print("             plateau real [%.3f, %.3f] / syn [%.3f, %.3f]"
                  % (fam["real"]["plateau_lo"], fam["real"]["plateau_hi"],
                     fam["synthetic"]["plateau_lo"], fam["synthetic"]["plateau_hi"]))


def main():
    global GRIDS
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all", choices=["a", "b", "c", "all"])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--grids", default="64,128",
                    help="comma-separated subset, mainly for smoke tests")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report:
        report()
        return
    GRIDS = tuple(int(g) for g in a.grids.split(",") if g.strip())
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device=%s  n=%d  arm=%.2f  grids=%s" % (dev, a.n, ARM, GRIDS), flush=True)
    t0 = time.time()
    if a.part in ("a", "all"):
        part_a(a.n, dev)
        print("[A] %.0fs" % (time.time() - t0), flush=True)
    if a.part in ("b", "all"):
        part_b(a.n, dev)
        print("[B] %.0fs" % (time.time() - t0), flush=True)
    if a.part in ("c", "all"):
        part_c(a.n, dev)
        print("[C] %.0fs" % (time.time() - t0), flush=True)
    print("\nwrote %s" % OUT)
    report()


if __name__ == "__main__":
    main()
