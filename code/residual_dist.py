# Author: Longgang Wang <longgangwang@163.com>
# Affiliation: Luoyang Institute of Science and Technology, Luoyang 471000, Henan, China

"""Residual diagnostics for the solver, answering a referee request directly.

A referee asked for three things that the paper previously did not give in full:

  (1) the distribution of the backward-substitution residual over the whole
      generated corpus, not only its maximum;
  (2) an independent recomputation of that residual from the *shipped* fields,
      i.e. an instrument self-test rather than a re-quote of the generator's own
      number;
  (3) the smallest label corruption the gate is actually able to detect --
      a usable threshold instead of an assertion.

Three parts, in increasing cost:

  part 1 (CPU only)  read the ``res`` member out of every production cache chunk
                     and report percentiles per arm and overall.  Uncompressed
                     members are read directly, so this touches ~150 MB, not the
                     9 GB of fields.
  part 2 (one matvec per sample, plus one refit per arm) measure BOTH the
                     recomputation from the shipped fields AND the instrument's
                     own resolution.  A residual formed from the shipped fields
                     is limited by the single precision in which they are stored
                     (psi is complex64), so it cannot resolve the 1e-9 gate; the
                     part reports that floor explicitly, and separately reports a
                     fresh working-precision solve from the shipped inputs, which
                     is the quantity that can be compared with the gate.
  part 3 (small)     two controlled measurements: (a) the same heterogeneous
                     batch solved with the naive and the frozen recurrences,
                     partitioned by the TRUE residual of each solution, to
                     measure how many labels the naive recurrence silently
                     ruins -- and to establish what that damage *looks like*.
                     It is not a large finite residual: the ruined solutions are
                     non-finite, so there is no finite gap to threshold against,
                     which is why an in-loop residual test cannot see them; (b) a
                     synthetic label perturbation swept over decades, to find the
                     perturbation size at which the 1e-9 gate fires.

Run:  conda run -n wirenet python code/residual_dist.py [--no-gpu]
"""
import argparse
import glob
import json
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
except Exception:
    pass

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

GATE = 1e-9          # the acceptance gate used when the corpus was generated
BAD = 1e-6           # the corruption flag used throughout the paper

# g02/g10 arms were generated with a different loss tangent; everything else
# uses the default.  Reading the wrong gamma would make the recomputed residual
# disagree with the stored one for a reason that has nothing to do with honesty.
def gamma_of(folder):
    b = os.path.basename(folder)
    if b.endswith("g02"):
        return 0.02
    if b.endswith("g10"):
        return 0.10
    return C.GAMMA


def pct(a, qs=(50, 90, 99, 99.9, 100)):
    a = np.asarray(a, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return None
    out = {("p%g" % q): float(np.percentile(a, q)) for q in qs}
    out["n"] = int(a.size)
    return out


# --------------------------------------------------------------- part 1
def corpus_residuals():
    """Percentiles of the recorded residual over every generated sample."""
    per_arm, all_res = {}, []
    for folder in sorted(glob.glob(os.path.join(C.DATA, "*"))):
        if not os.path.isdir(folder):
            continue
        chunks = sorted(glob.glob(os.path.join(folder, "chunk_*.npz")))
        if not chunks:
            continue
        rs = []
        for p in chunks:
            try:
                with np.load(p) as z:
                    if "res" in z.files:
                        rs.append(z["res"].astype(np.float64))
            except Exception as e:                       # pragma: no cover
                print("  !! unreadable %s (%s)" % (p, e))
        if not rs:
            continue
        rs = np.concatenate(rs)
        per_arm[os.path.basename(folder)] = dict(
            chunks=len(chunks), **pct(rs), max=float(np.nanmax(rs)),
            n_over_gate=int((rs > GATE).sum()), n_over_bad=int((rs > BAD).sum()))
        all_res.append(rs)
        print("  %-14s n=%6d  p50=%.2e  p99=%.2e  max=%.2e  >1e-9: %d"
              % (os.path.basename(folder), rs.size, np.percentile(rs, 50),
                 np.percentile(rs, 99), np.nanmax(rs), int((rs > GATE).sum())))
    allr = np.concatenate(all_res) if all_res else np.array([])
    return per_arm, dict(**pct(allr), max=float(np.nanmax(allr)) if allr.size else None,
                         n_over_gate=int((allr > GATE).sum()) if allr.size else None,
                         n_over_bad=int((allr > BAD).sum()) if allr.size else None)


# --------------------------------------------------------------- part 2
def _relres(psi, v, k, th, gam, dev):
    """||b - A psi|| / ||b|| for the shipped operator, one matvec per sample."""
    import torch
    import physics as P
    v = v.to(torch.complex128)
    Ghat = P.green_symbol(C.N, k, gam, dev)
    k2 = (k.to(torch.complex128) * (1.0 + 1j * gam)) ** 2
    b = P.incident_plane_wave(C.N, k, th, dev).to(torch.complex128)
    Apsi = psi - k2[:, None, None] * torch.fft.ifft2(Ghat * torch.fft.fft2(v * psi))
    return (torch.linalg.norm((b - Apsi).reshape(len(k), -1), dim=1)
            / torch.linalg.norm(b.reshape(len(k), -1), dim=1))


def _stats(a):
    a = np.asarray(a, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return dict(n=0)
    return dict(**pct(a), min=float(a.min()), max=float(a.max()))


def recheck(n_arm=1, per_chunk=128, arms=None):
    """Independent recomputation, with the instrument's own floor measured.

    The first version of this check compared a residual recomputed from the
    shipped fields against the value the generator recorded, and called a
    mismatch a defect.  It reported a 30-300x mismatch, which looked like the
    recorded residual was dishonest.

    It was not.  The archive stores the fields in SINGLE precision (psi as
    complex64 -- see data.py), so psi already carries a relative perturbation of
    ~eps32 ~ 1e-7 before any arithmetic happens, and ||b - A psi|| formed from it
    cannot be sharper than ~1e-8.  The check was asking a float32 artifact for a
    float64 answer, and the 'disagreement' was the check's resolution, not the
    solver's honesty.

    So this part now measures the floor instead of assuming it, on every arm:
      shipped   -- residual of the shipped complex64 psi
      roundtrip -- residual after pushing a working-precision solution through
                   the same complex64 round-trip (isolates storage precision)
      refined   -- residual of a fresh working-precision solve from the SHIPPED
                   inputs (v, k, theta, gamma only).  Nothing from the iteration
                   that produced the labels is reused, and this is the quantity
                   that can actually be compared with the 1e-9 gate.
    """
    import torch
    import physics as P
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rows, all_rt, all_ref, all_pert, all_ship = [], [], [], [], []
    for folder in sorted(glob.glob(os.path.join(C.DATA, "*"))):
        if not os.path.isdir(folder):
            continue
        arm = os.path.basename(folder)
        if arms and arm not in arms:
            continue
        chunks = sorted(glob.glob(os.path.join(folder, "chunk_*.npz")))[:n_arm]
        if not chunks:
            continue
        gam = gamma_of(folder)
        rec, sto, rtp, ref, pert = [], [], [], [], []
        for p in chunks:
            with np.load(p) as z:
                v = torch.from_numpy(z["v"][:per_chunk].astype(np.float64)).to(dev)
                psi = torch.from_numpy(
                    z["psi"][:per_chunk].astype(np.complex128)).to(dev)
                k = torch.from_numpy(z["k"][:per_chunk].astype(np.float64)).to(dev)
                th = torch.from_numpy(
                    z["theta"][:per_chunk].astype(np.float64)).to(dev)
                sto.append(z["res"][:per_chunk].astype(np.float64))
            s = _relres(psi, v, k, th, gam, dev).detach().cpu().numpy()
            psi_ref, _ = P.solve_ls(v, k, th, gam, C.N, tol=1e-12, maxiter=900,
                                    device=dev, return_info=True)
            w = _relres(psi_ref, v, k, th, gam, dev).detach().cpu().numpy()
            psi_rt = torch.from_numpy(
                psi_ref.detach().cpu().numpy().astype(np.complex64)
                .astype(np.complex128)).to(dev)
            t = _relres(psi_rt, v, k, th, gam, dev).detach().cpu().numpy()
            d = (torch.linalg.norm((psi - psi_ref).reshape(len(k), -1), dim=1)
                 / torch.linalg.norm(psi_ref.reshape(len(k), -1), dim=1))
            rec.append(s)
            rtp.append(t)
            ref.append(w)
            pert.append(d.detach().cpu().numpy())
        rec = np.concatenate(rec)
        sto = np.concatenate(sto)
        rtp = np.concatenate(rtp)
        ref = np.concatenate(ref)
        pert = np.concatenate(pert)
        d = np.abs(rec - sto) / np.maximum(sto, 1e-30)
        rows.append(dict(arm=arm, gamma=gam, n=int(rec.size),
                         shipped=_stats(rec), stored=_stats(sto),
                         roundtrip=_stats(rtp),
                         refined=_stats(ref),
                         psi_storage_perturbation=_stats(pert),
                         agree_median_rel=float(np.median(d)),
                         agree_max_rel=float(d.max())))
        all_ship.append(rec)
        all_rt.append(rtp)
        all_ref.append(ref)
        all_pert.append(pert)
        print("  %-14s n=%5d  shipped=%.2e  roundtrip=%.2e  refined=%.2e  "
              "psi pert=%.2e" % (arm, rec.size, rec.max(), rtp.max(),
                                 ref.max(), pert.max()))
    if not rows:
        return dict(per_arm=[], floor=dict(n=0))
    floor = dict(
        n=int(np.concatenate(all_ref).size),
        shipped=_stats(np.concatenate(all_ship)),
        roundtrip=_stats(np.concatenate(all_rt)),
        refined=_stats(np.concatenate(all_ref)),
        psi_storage_perturbation=_stats(np.concatenate(all_pert)),
        gate=GATE)
    return dict(per_arm=rows, floor=floor)


# --------------------------------------------------------------- part 3
def _true_res(mv, b, x):
    import torch
    return (torch.linalg.norm((b - mv(x)).reshape(len(x), -1), dim=1)
            / torch.linalg.norm(b.reshape(len(b), -1), dim=1))


def naive_vs_frozen(nb=2, B=512, seed=4242, tol=GATE, maxiter=600):
    """Split the two populations on one heterogeneous batch, and find the gate.

    The populations must be defined by the TRUE residual of the naive solution,
    not by the residual the naive loop reports -- the whole point of the failure
    mode is that the loop reports success.  An earlier version partitioned on
    ``res_n`` (the loop residual), found no sample above the corruption flag,
    and crashed on an empty array, because by construction the loop residual is
    the quantity that stays small.

    A second version partitioned the TRUE residual into "clean" and "corrupt" by
    a finite corruption flag, and reported an EMPTY corrupt set while the silent
    count was 21.6%: the ruined solutions are not large-but-finite, they are
    non-finite (NaN/Inf), so ``isfinite`` -- not the flag -- was doing the
    partitioning and the two definitions disagreed.  The honest structure is
    therefore reported explicitly: how many labels are finite-and-inside (clean),
    how many are finite-and-large (a genuinely corrupted finite remainder, which
    on this corpus is empty), and how many are non-finite.  A residual gate can
    never separate the third class, because there is no residual left to compare;
    that is the finding, and it is why the frozen recurrence is required.
    """
    import torch
    import physics as P
    import data as D
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    clean, corrupt, frozen_true, silent, n_tot = [], [], [], 0, 0
    n_nonfinite = 0
    inj = []
    for i in range(nb):
        d = D.sample_batch(B, seed + 13 * i, dev, f_lo=1.0, f_hi=4.0,
                           vmax_lo=0.05, vmax_hi=1.0)
        Ghat = P.green_symbol(C.N, d["k"], C.GAMMA, dev)
        k2 = (d["k"].to(torch.complex128) * (1.0 + 1j * C.GAMMA)) ** 2
        vc = d["v"].to(torch.complex128)
        b = P.incident_plane_wave(C.N, d["k"], d["theta"], dev).to(torch.complex128)

        def mv(z):
            return z - k2[:, None, None] * torch.fft.ifft2(Ghat * torch.fft.fft2(vc * z))

        x_n, res_n, loop_n = P.bicgstab_naive(mv, b, tol=tol, maxiter=maxiter,
                                              return_loop_res=True)
        x_f, res_f = P.bicgstab(mv, b, tol=tol, maxiter=maxiter)
        tn = _true_res(mv, b, x_n).detach().cpu().numpy()
        tf = _true_res(mv, b, x_f).detach().cpu().numpy()
        lp = loop_n.detach().cpu().numpy()
        n_tot += len(tn)
        finite_ok = np.isfinite(tn)
        large = finite_ok & (tn > BAD)          # finite, but still wrong
        nonfinite = ~finite_ok                  # the actual failure signature
        n_nonfinite += int(nonfinite.sum())
        clean.append(tn[finite_ok & ~large])
        corrupt.append(tn[large])
        frozen_true.append(tf)
        # the silent failure: the recurrence's own bookkeeping says converged
        # while the honest residual says the label is ruined
        silent += int((np.isfinite(lp) & (lp < tol) & (nonfinite | large)).sum())
        # synthetic perturbation of a converged (frozen) solution: how large does
        # a label error have to be before the gate notices?
        g = torch.Generator(device=dev).manual_seed(11 + i)
        noise = torch.randn(x_f.shape, generator=g, device=dev,
                            dtype=torch.complex128)
        for eta in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]:
            step = eta * torch.linalg.norm(x_f.reshape(len(x_f), -1), dim=1)
            step = step / torch.linalg.norm(noise.reshape(len(x_f), -1), dim=1)
            xp = x_f + noise * step[:, None, None]
            rp = _true_res(mv, b, xp).detach().cpu().numpy()
            inj.append(dict(eta=eta, batch=i,
                            frac_flagged=float((rp > tol).mean()),
                            median=float(np.median(rp))))
    clean = np.concatenate(clean) if clean else np.array([])
    corrupt = np.concatenate(corrupt) if corrupt else np.array([])
    frozen_true = np.concatenate(frozen_true) if frozen_true else np.array([])
    return dict(
        n_total=int(n_tot), n_batches=int(nb), batch_size=int(B),
        fired_silently=int(silent),
        silent_fraction=float(silent / n_tot) if n_tot else None,
        n_nonfinite=int(n_nonfinite),
        nonfinite_fraction=float(n_nonfinite / n_tot) if n_tot else None,
        n_finite_corrupt=int(corrupt.size),
        clean=_stats(clean), corrupt=_stats(corrupt),
        frozen_true=_stats(frozen_true),
        separation_orders=(float(np.log10(corrupt.min() / clean.max()))
                           if clean.size and corrupt.size else None),
        injection=inj)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or os.path.join(C.RES, "residual_dist.json")
    res = {}
    print("part 1: recorded residual over the whole corpus")
    per_arm, overall = corpus_residuals()
    res["corpus"] = dict(per_arm=per_arm, overall=overall, gate=GATE, bad=BAD)
    print("part 2: independent recomputation from the shipped fields")
    if a.no_gpu:
        res["recheck"] = None
    else:
        res["recheck"] = recheck()
        print("part 3: population separation and detectable threshold")
        res["naive"] = naive_vs_frozen()
    C.save_json(res, out)
    print("\nwrote", out)
    print("corpus n=%d  p50=%.2e  p99=%.2e  max=%.2e  over gate=%s"
          % (overall["n"], overall["p50"], overall["p99"], overall["max"],
             overall["n_over_gate"]))
    nv = res.get("naive")
    if nv:
        print("naive batch n=%d  silent=%.1f%%  non-finite=%.1f%%  "
              "finite-corrupt=%d  clean max=%.2e  frozen max=%.2e"
              % (nv["n_total"], 100 * nv["silent_fraction"],
                 100 * nv["nonfinite_fraction"], nv["n_finite_corrupt"],
                 nv["clean"]["max"], nv["frozen_true"]["max"]))


if __name__ == "__main__":
    main()
