# Author: Longgang Wang <longgangwang@163.com>
# Affiliation: Luoyang Institute of Science and Technology, Luoyang 471000, Henan, China

"""Third evidence rung: real geological structure (Marmousi).

The first two rungs establish the result on analytic theory and on a synthetic
numerical benchmark.  This rung substitutes the synthetic Gaussian-correlated
geometry by the velocity field of the Marmousi model -- a model whose horizons
are interpreted picks from a real survey rather than draws from a random
process -- and changes NOTHING else: the same 64^2 grid, the same loss tangent,
the same frequency grid, the same incident angles (paired, inherited from the
main protocol), the same reference solver and the same checkpoints.  The only
variable that moves is the geometry.

Pre-registered before any learning error was inspected, and enforced by
``check_claims.realgeo_guard``:

  * the window rule below is fixed by the velocity model alone, never by an
    error curve;
  * per-window min-max normalisation, so every window contributes the same
    contrast amplitude and amplitude cannot act as a confound;
  * the synthetic control is re-measured inside this script under an identical
    design (three fixed geometries x angles), so the two arms differ only in
    geometry -- not in the number of scattering media per cell.

What this rung is NOT: the fields are real geological *structure*, not measured
field data.  Nothing here replaces a physical experiment, and the manuscript
says so in the same sentence that reports the result.
"""
import os
import sys
import time
import json
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import physics as P
import data as D
import evaluate as E
from diag_spectrum import band_energy

REAL = os.path.join(C.ROOT, "data", "real")
MODES = [4, 8, 12, 16, 20, 24, 32]
WIN_M = 3000.0          # 3.0 km square window, matching the 9.2 x 3.0 km model depth
SPACING = 10.0          # m, mlamoureux copy


# ---------------------------------------------------------------------------
# windows -- selection rule fixed by the model, error-blind
# ---------------------------------------------------------------------------
def load_marmousi():
    from scipy.io import loadmat
    m = loadmat(os.path.join(REAL, "marmousi_dz10.mat"))
    c = np.asarray(m["vel"], dtype=np.float64)          # (301, 941) z x x
    x = np.asarray(m["x"]).ravel().astype(np.float64)
    z = np.asarray(m["z"]).ravel().astype(np.float64)
    return c, x, z


def _load_js(path):
    """The primary copy also ships its axes as plain JS assignments
    (`x_ = [...]`), which is not strict JSON; the prefix is stripped here."""
    import re
    with open(path, encoding="utf-8") as fh:
        txt = fh.read()
    txt = re.sub(r"^\s*[A-Za-z_][A-Za-z0-9_]*\s*=\s*", "", txt)
    return np.asarray(json.loads(txt.rstrip().rstrip(";")), dtype=np.float64)


def pick_windows(c, x, z):
    """Return {name: (x0, x1, z0, z1)} in metres, plus the roughness of each.

    Rule: three EQUALLY SPACED, NON-OVERLAPPING 3 km windows spanning the
    survey.  There is no selection freedom at all -- no metric is maximised
    over candidate positions -- so the choice cannot be steered by an error
    curve even in principle.  The mean |grad c| of each window is reported
    afterwards as a descriptive statistic, not as a selection criterion.
    """
    x_lo, x_hi = float(x.min()), float(x.max())
    z_lo, z_hi = float(z.min()), float(z.max())
    assert z_hi - z_lo <= WIN_M + 1e-9, "window is deeper than the model"
    assert 3 * WIN_M <= (x_hi - x_lo) + 1e-9, "three windows do not fit"
    starts = (x_lo, x_lo + (x_hi - x_lo - WIN_M) / 2.0, x_hi - WIN_M)
    names = ("west", "centre", "east")
    w = {n: (s, s + WIN_M, z_lo, z_hi) for n, s in zip(names, starts)}
    gy, gx = np.gradient(c, SPACING, SPACING)
    grad = np.hypot(gx, gy)
    rough = {}
    for n in names:
        i0 = int(round((w[n][0] - x_lo) / SPACING))
        i1 = i0 + int(round(WIN_M / SPACING))
        rough[n] = float(grad[:, i0:i1].mean())
    return w, rough


def crop(c, x, z, box):
    x0, x1, z0, z1 = box
    i0 = int(round((x0 - x.min()) / SPACING))
    i1 = int(round((x1 - x.min()) / SPACING))
    return c[:, i0:i1]


def resample(a, n=64):
    """Area-average (anti-aliased) downsample of a 2D array to n x n."""
    ny, nx = a.shape
    yi = np.rint(np.linspace(0, ny, n + 1)).astype(int)
    xi = np.rint(np.linspace(0, nx, n + 1)).astype(int)
    out = np.empty((n, n))
    for r in range(n):
        for cc in range(n):
            out[r, cc] = a[yi[r]:max(yi[r] + 1, yi[r + 1]),
                           xi[cc]:max(xi[cc] + 1, xi[cc + 1])].mean()
    return out


def unit01(a):
    return (a - a.min()) / max(a.max() - a.min(), 1e-30)


def band_frac_unit(v, M):
    """Fraction of the field's L2 energy outside |mode| <= M (unit-scale field)."""
    F = np.fft.fft2(v)
    p2 = F.real ** 2 + F.imag ** 2
    idx = np.abs(np.fft.fftfreq(v.shape[0], d=1.0 / v.shape[0]))
    m = np.maximum(idx[:, None], idx[None, :])
    return float(p2[m > M].sum() / max(p2.sum(), 1e-30))


# ---------------------------------------------------------------------------
def arm_tag(vmax):
    return "a%.3d" % int(round(vmax * 100))


def models_for(vmax):
    """Checkpoints evaluated at this contrast arm, as (label, run_name)."""
    tag = arm_tag(vmax)
    out = [("FNO", "main_fno_%s_s%d" % (tag, s)) for s in (0, 1, 2)]
    if tag in ("a025", "a100"):
        for pre, lab in (("base_unet", "U-Net (base)"),
                         ("pmatch_unet", "U-Net (matched)"),
                         ("base_don", "DeepONet")):
            for s in (0, 1, 2):
                out.append((lab, "%s_%s_s%d" % (pre, tag, s)))
    return out


def summarise(res):
    """Reduce the curves to the few numbers the manuscript quotes.

    Computed here rather than by hand so that every number in the text has a
    single machine-readable source, and so that ``check_claims.realgeo_guard``
    can re-derive it and compare.
    """
    F = res["f_test"]
    order = res["order"]
    geo_names = [g for g in order if g.startswith("geo_")]
    syn_names = [g for g in order if g.startswith("syn_")]
    out = dict(per_geometry={}, family={}, bands=res["bands"])
    for g in order:
        per = {}
        for vmax in res["vmax_list"]:
            rows = res["curves"][g]["%.2f" % vmax]
            fno = []
            for row in rows:
                ks = [k for k in row if k.startswith("FNO|")]
                fno.append(float(np.mean([row[k]["mean"] for k in ks])))
            ib = float(np.mean([m for m, f in zip(fno, F) if f <= 4.0]))
            pl = float(np.mean([m for m, f in zip(fno, F) if f >= 5.0]))
            e4 = fno[F.index(4.0)]
            e425 = fno[F.index(4.25)]
            e5 = fno[F.index(5.0)]
            fstar = float(min(f for f, m in zip(F, fno) if m > 6.0 * e4))
            per["%.2f" % vmax] = dict(in_band=ib, plateau=pl, r5=e5 / e4,
                                      rise_lo=e425 / e4, f_star=fstar, curve=fno)
        out["per_geometry"][g] = per
    for fam, names in (("real", geo_names), ("synthetic", syn_names)):
        famd = {}
        for vmax in res["vmax_list"]:
            key = "%.2f" % vmax
            ibs = [out["per_geometry"][g][key]["in_band"] for g in names]
            pls = [out["per_geometry"][g][key]["plateau"] for g in names]
            rs = [out["per_geometry"][g][key]["r5"] for g in names]
            fs = [out["per_geometry"][g][key]["f_star"] for g in names]
            famd[key] = dict(
                in_band=float(np.mean(ibs)), in_band_lo=float(min(ibs)), in_band_hi=float(max(ibs)),
                plateau=float(np.mean(pls)), plateau_lo=float(min(pls)), plateau_hi=float(max(pls)),
                r5=float(np.mean(rs)), r5_lo=float(min(rs)), r5_hi=float(max(rs)),
                f_star=float(np.mean(fs)))
        out["family"][fam] = famd
    # the two statements the third rung is allowed to make
    key = "0.25"
    out["verdict"] = dict(
        f_star_real=out["family"]["real"][key]["f_star"],
        f_star_syn=out["family"]["synthetic"][key]["f_star"],
        r5_real=out["family"]["real"][key]["r5"],
        r5_syn=out["family"]["synthetic"][key]["r5"],
        plateau_real=[out["family"]["real"][key]["plateau_lo"],
                      out["family"]["real"][key]["plateau_hi"]],
        plateau_syn=[out["family"]["synthetic"][key]["plateau_lo"],
                     out["family"]["synthetic"][key]["plateau_hi"]],
    )
    return out


def mirror_correlation(M, A):
    """Best whole-sample correlation between the copy read and a mirror.

    Only an integer lag, a transposition and axis flips are searched: the
    alignment has no continuous free parameter, so the number is a *floor* on
    the agreement rather than a fitted maximum.  A resampling-aware fit reaches
    about 0.99, but that fit carries a free spacing parameter and therefore
    cannot be quoted as evidence without also quoting the parameter.  Both are
    recorded; the paper quotes neither, it only says the copies agree on the
    overlapping structure.
    """
    best = None
    for tr in (0, 1):
        B = A.T if tr else A
        for fz in (0, 1):
            for fx in (0, 1):
                C = B[::-1] if fz else B
                C = C[:, ::-1] if fx else C
                for sy in (1, 2):
                    for sx in (1, 2):
                        nz = (C.shape[0] - 1) * sy + 1
                        nx = (C.shape[1] - 1) * sx + 1
                        if nz > M.shape[0] or nx > M.shape[1]:
                            continue
                        for i0 in range(0, M.shape[0] - nz + 1, sy):
                            for j0 in range(0, M.shape[1] - nx + 1, sx):
                                sub = M[i0:i0 + nz:sy, j0:j0 + nx:sx]
                                if sub.shape != C.shape:
                                    continue
                                r = float(np.corrcoef(sub.ravel(), C.ravel())[0, 1])
                                if best is None or r > best:
                                    best = r
    return best


def source_provenance(c, x, z, wins, rough):
    """Record what was actually read off disk, so the rung is re-checkable.

    The primary copy's two axis files are parsed and their extents re-stated,
    which is what pins the field to the Marmousi survey: the shipped axes run
    from -200 to 9200 m at 10 m spacing and from 0 to 3000 m at 10 m spacing,
    which is the survey's 9.4 km length and 3.0 km depth.  Two independently
    hosted mirrors are then correlated against it so a reader can see the field
    is the real model and not a renamed synthetic surrogate.
    """
    files = ["marmousi_dz10.mat", "marmousi_mirror_300x298.npy",
             "marmousi3d_VP_201x201x70.raw"]
    out = dict(used="marmousi_dz10.mat (var 'vel')",
               used_shape=list(c.shape),
               used_range_m_s=[float(c.min()), float(c.max())],
               spacing_m=SPACING,
               x_m=[float(x.min()), float(x.max())],
               z_m=[float(z.min()), float(z.max())],
               windows={k: list(wins[k]) for k in wins},
               window_roughness=rough,
               files={}, cross_check={})

    # axis files shipped with the primary copy: these, not the file name, are
    # what says the grid really is the 9.4 km x 3.0 km survey at 10 m.
    for nm, key in (("marmousi_x.json", "x_m"),
                    ("marmousi_z.json", "z_m")):
        p = os.path.join(REAL, nm)
        if os.path.exists(p):
            a = np.asarray(_load_js(p), dtype=np.float64)
            out[key + "_axis"] = dict(n=int(a.size), first=float(a[0]),
                                      last=float(a[-1]),
                                      step=float((a[-1] - a[0]) / max(a.size - 1, 1)))

    M = c
    p = os.path.join(REAL, "marmousi_mirror_300x298.npy")
    if os.path.exists(p):
        A2 = np.load(p).astype(np.float64)
        out["cross_check"]["marmousi_mirror_300x298"] = dict(
            shape=list(A2.shape), whole_sample_r=mirror_correlation(M, A2))
    p = os.path.join(REAL, "marmousi3d_VP_201x201x70.raw")
    if os.path.exists(p):
        raw = np.fromfile(p, dtype="<f4")
        if raw.size == 201 * 201 * 70:
            R = raw.reshape(201, 201, 70).astype(np.float64)
            # same survey province, different sampling, so the two copies are
            # compared section by section rather than by one global correlation:
            # a horizontal depth slice and a vertical section taken mid-survey.
            ref = unit01(resample(M, 64))
            out["cross_check"]["marmousi3d_VP_201x201x70"] = dict(
                shape=[201, 201, 70],
                horizontal_slice_r=float(np.corrcoef(
                    unit01(resample(R[:, :, 35], 64)).ravel(), ref.ravel())[0, 1]),
                vertical_slice_r=float(np.corrcoef(
                    unit01(resample(R[100, :, :], 64)).ravel(), ref.ravel())[0, 1]),
                note="different sampling; section-wise comparison")

    for nm in files:
        p = os.path.join(REAL, nm)
        if os.path.exists(p):
            out["files"][nm] = dict(bytes=os.path.getsize(p), md5=C.md5(p))
    return out


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", default=os.path.join(C.RES, "realgeo.json"))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--summarise", action="store_true",
                    help="reduce an existing realgeo.json in place, no GPU")
    ap.add_argument("--reprovenance", action="store_true",
                    help="refresh only the source/cross-check block of an "
                         "existing realgeo.json, no GPU: used when a data file "
                         "is renamed or an extra cross-check is added, so the "
                         "measured curves are never recomputed to change what "
                         "is only a bookkeeping field")
    a = ap.parse_args()
    if a.summarise:
        d = C.load_json(a.out)
        d["summary"] = summarise(d)
        C.save_json(d, a.out)
        _report(d)
        print("summarised %s" % a.out)
        return
    if a.reprovenance:
        d = C.load_json(a.out)
        c, x, z = load_marmousi()
        wins, rough = pick_windows(c, x, z)
        d["source"] = source_provenance(c, x, z, wins, rough)
        C.save_json(d, a.out)
        print("refreshed provenance of %s (curves untouched)" % a.out)
        return
    if a.quick:
        a.n = 40
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device=%s  n=%d" % (dev, a.n), flush=True)

    c, x, z = load_marmousi()
    wins, rough = pick_windows(c, x, z)
    print("windows (metres), equally spaced, non-overlapping:")
    for k in ("west", "centre", "east"):
        print("  %-8s x=[%.0f, %.0f]  z=[%.0f, %.0f]  mean|grad c|=%.1f m/s/m"
              % (k, wins[k][0], wins[k][1], wins[k][2], wins[k][3], rough[k]), flush=True)

    # --- geometry bank: 3 real windows + 3 synthetic fields ------------------
    geo = {}
    for k in ("west", "centre", "east"):
        geo["geo_" + k] = torch.from_numpy(
            unit01(resample(crop(c, x, z, wins[k]), 64))).double()
    # synthetic control: three FIXED geometries drawn from the same generator
    # the benchmark uses, so the two arms are structurally comparable.
    for s in range(3):
        g = torch.Generator(device="cpu").manual_seed(70000 + 0 + 7777 + s)
        fld = P.make_contrast(64, 1, 1.0, C.MM_REF, g, "cpu", kind="smooth")[0]
        geo["syn_%d" % s] = torch.from_numpy(unit01(fld.numpy())).double()

    fields = {k: v.numpy() for k, v in geo.items()}
    bands = {k: {str(M): band_frac_unit(fields[k], M) for M in MODES} for k in fields}
    print("\ninput field: fraction of energy outside mode M")
    print("  field      " + "".join("%9s" % M for M in MODES))
    for k in fields:
        print("  %-10s " % k + "".join("%9.5f" % bands[k][str(M)] for M in MODES), flush=True)

    # --- models --------------------------------------------------------------
    cache = {}
    for vmax in C.VMAX_LIST:
        for lab, run in models_for(vmax):
            p = os.path.join(C.RUNS, run, "checkpoint.pt")
            if not os.path.exists(p):
                print("  MISSING %s" % p, flush=True)
                continue
            cache[(lab, run)] = E.load_model(p, dev)[0]
    print("\nloaded %d checkpoints" % len(cache), flush=True)

    res = dict(n=a.n, gamma=C.GAMMA, f_test=C.F_TEST, vmax_list=C.VMAX_LIST,
               modes=MODES,
               windows={k: list(wins[k]) for k in ("west", "centre", "east")},
               window_roughness=rough,
               order=list(geo.keys()), bands=bands,
               models={"%s|%s" % k: k[0] for k in cache})
    curves = {}
    t0 = time.time()
    for gname in geo.keys():
        curves[gname] = {}
        vunit = geo[gname]
        for vi, vmax in enumerate(C.VMAX_LIST):
            rows = []
            for f in C.F_TEST:
                # Paired angles: the SAME generator seed and the SAME draw order
                # as data.sample_batch, so the incident directions coincide with
                # the synthetic benchmark's at this (vmax arm, frequency).
                _, _, theta = D.sample_params(a.n, None, None, vmax, vmax,
                                              60000 + vi, dev, f_fixed=f)
                # k is float32 here on purpose: data.sample_batch builds
                # k = 2*pi*f from a float32 draw, and the checkpoints were
                # trained and evaluated on that operand.  Promoting it to
                # float64 would change the network's input dtype and make the
                # two arms differ by something other than the geometry.
                k = torch.full((a.n,), 2 * np.pi * f, device=dev)
                v = (vunit.to(dev).double() * vmax).unsqueeze(0).repeat(a.n, 1, 1)
                psi_ref, rres = P.solve_ls(v, k, theta, C.GAMMA, C.N, tol=1e-9,
                                           maxiter=900, device=dev, return_info=True)
                row = dict(f=f, ref_res_max=float(rres.max()),
                           delta={str(M): float(np.mean(band_energy(psi_ref, M, C.N)))
                                  for M in MODES})
                for lab, run in models_for(vmax):
                    m = cache.get((lab, run))
                    if m is None:
                        continue
                    xx, psi_inc = E.build(C.N, k, theta, v, dev)
                    o = m(xx)
                    psi_hat = psi_inc + torch.complex(o[:, 0], o[:, 1])
                    num = torch.linalg.norm((psi_hat - psi_ref).reshape(a.n, -1), dim=1)
                    den = torch.linalg.norm(psi_ref.reshape(a.n, -1), dim=1)
                    e = (num / den).cpu().numpy()
                    row["%s|%s" % (lab, run)] = dict(
                        mean=float(e.mean()), std=float(e.std()),
                        median=float(np.median(e)),
                        q90=float(np.quantile(e, 0.9)))
                rows.append(row)
            # seed span is per FREQUENCY (rows are frequencies, not seeds): the
            # span of the three seed means at this f, not the span over the sweep.
            labels = sorted({lab for lab, _ in models_for(vmax)})
            for r in rows:
                r["seedspan"] = {}
                for lab in labels:
                    ms = [r["%s|%s" % (lab, run)]["mean"]
                          for l2, run in models_for(vmax)
                          if l2 == lab and "%s|%s" % (lab, run) in r]
                    if ms:
                        r["seedspan"][lab] = [float(min(ms)), float(max(ms))]
            curves[gname]["%.2f" % vmax] = rows
        print("  %s done  (%.0f s elapsed)" % (gname, time.time() - t0), flush=True)

    res["curves"] = curves
    res["source"] = source_provenance(c, x, z, wins, rough)
    res["summary"] = summarise(res)
    C.save_json(res, a.out)
    print("\nwrote %s" % a.out)
    _report(res)


def _report(res):
    F = res["f_test"]
    print("\nFNO mean relL2 error (rows: geometry, cols: f)")
    print("  %-11s " % "" + "".join("%7.1f" % f for f in F))
    for g in res["order"]:
        c = res["summary"]["per_geometry"][g]["0.25"]["curve"]
        print("  %-11s " % g + "".join("%7.3f" % x for x in c))
    s = res["summary"]
    print("\n  f* (first f with eps > 6 x eps(4)) : real %.2f   synthetic %.2f"
          % (s["verdict"]["f_star_real"], s["verdict"]["f_star_syn"]))
    print("  jump factor eps(5)/eps(4)          : real %.2f   synthetic %.2f"
          % (s["verdict"]["r5_real"], s["verdict"]["r5_syn"]))
    print("  plateau range (vmax=0.25)          : real %s   synthetic %s"
          % (["%.3f" % v for v in s["verdict"]["plateau_real"]],
             ["%.3f" % v for v in s["verdict"]["plateau_syn"]]))


if __name__ == "__main__":
    main()
