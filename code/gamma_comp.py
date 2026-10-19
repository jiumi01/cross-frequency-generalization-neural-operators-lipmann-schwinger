# Author: Longgang Wang <longgangwang@163.com>
# Affiliation: Luoyang Institute of Science and Technology, Luoyang 471000, Henan, China

"""Per-frequency loss tangent that flattens the DISCRETE operator norm.

Motivation (external review, round 4, item 2): the manuscript's headline
mechanism -- the error step at the training-band edge -- could in principle be
a conditioning artefact rather than a frequency-generalisation boundary.  The
attained discrete operator norm max|kappa^2 Ghat| varies by 2.2x across the
test grid at fixed gamma = 0.05 (4.53 at f = 2.5 to 10.02 from f = 4 on; see
code/diag_propnorm.py), so the wavenumber is not, on the grid, the only thing
that changes with f: the forward operator's conditioning changes too.

The discriminating experiment is to remove that confound: choose gamma(f) so
that the attained norm is the SAME at every test frequency, retrain, and ask
whether the cliff is still at the band edge.  If it is, the step is a property
of the frequency boundary and not of the conditioning; if it moves or vanishes,
the conditioning explanation wins.

The continuum sup (1+gamma^2)/(2 gamma) is k-independent, so the frequency
dependence that has to be cancelled is purely the grid-sampling one: on an
N x N grid the symbol is sampled at |xi|^2 = 4 pi^2 |n|^2 and the attained
maximum is max over the representable modes.  attained(f, gamma) is monotone
decreasing in gamma on (0, 1) (the branch containing the manuscript's
gamma = 0.05), so a bisection per frequency is exact and needs no derivative.

The target is chosen inside the COMMON attainable range, not at the baseline
band-edge value.  Two hard bounds constrain it:

  * gamma must stay in [GAMMA_MIN, GAMMA_MAX] = [0.02, 0.20].  Below 0.02 the
    batched Bi-CGSTAB stops converging at the 1e-9 acceptance gate (measured:
    worst residual 7.8e-1 at gamma = 0.001, 1.6e-9 at 0.005, 1e-9 only from
    0.02 up), because the Green resonance narrows past what a Krylov method can
    resolve on this grid; above 0.20 the regularisation stops being the one the
    manuscript's gamma-robustness table brackets.
  * attained(f, gamma) is monotone decreasing in gamma on (0, 1), so each
    frequency can reach any value in [attained(f, GAMMA_MAX), attained(f,
    GAMMA_MIN)].  The common range over the whole test grid is non-empty
    (max_f attained(f, 0.20) = 2.60 <= min_f attained(f, 0.02) = 3.00), so a
    single target attains EVERY frequency: the baseline's 2.2x spread on the
    test grid (4.53 .. 10.02 at gamma = 0.05) collapses to a constant, and the
    target itself sits 3.4x below the baseline's band-edge norm, i.e. the
    compensated arm is the better-conditioned one everywhere.  The target is
    placed at 97% of the upper edge, i.e. inside the range with margin on both
    sides, so no frequency is solved at a gamma bound.

Read-only w.r.t. the archive: writes results/gamma_comp.json and prints a
table.  No GPU (a 64x64 symbol needs none) and no training data is touched.
"""
import os
import sys
import json
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import physics as P

GAMMA_MIN = 0.02      # solver convergence bound (measured, see module docstring)
GAMMA_MAX = 0.20      # stay inside the range the gamma table brackets + margin


def attained_norm(f, gamma, n):
    """max |kappa^2 Ghat| on the n x n grid at frequency f, loss tangent gamma."""
    k = torch.tensor([2.0 * np.pi * float(f)], dtype=torch.float64)
    G = P.green_symbol(n, k, float(gamma), "cpu")
    kap2 = (k.to(torch.complex128) * (1.0 + 1j * float(gamma))) ** 2
    return float((kap2[:, None, None] * G).abs().max())


def solve_gamma(f, target, n, lo=GAMMA_MIN, hi=GAMMA_MAX, iters=80):
    """Bisect gamma in [GAMMA_MIN, GAMMA_MAX] for attained_norm(f, gamma) == target.

    attained is monotone decreasing in gamma on this branch, so the bisection is
    well-defined; 80 halvings drive the bracket below 1e-24, far past float64.
    Returns None if the target is outside the attainable range at this f.
    """
    flo, fhi = attained_norm(f, lo, n), attained_norm(f, hi, n)
    if not (fhi <= target <= flo):
        return None                      # target outside the attainable range
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        fm = attained_norm(f, mid, n)
        if fm > target:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-15:
            break
    return 0.5 * (lo + hi)


def common_target(n, f_lo=1.0, f_hi=12.0, step=0.01, margin=0.97):
    """Largest target every frequency in [f_lo, f_hi] can attain, times margin."""
    fs = np.arange(f_lo, f_hi + 1e-9, step)
    upper = min(attained_norm(f, GAMMA_MIN, n) for f in fs)
    lower = max(attained_norm(f, GAMMA_MAX, n) for f in fs)
    if lower > upper:
        raise SystemExit("no common target: attainable ranges do not overlap")
    return margin * upper


def build(n=None, f_lo=1.0, f_hi=12.0, step=0.01, target=None):
    n = n or C.N
    if target is None:
        target = common_target(n)
    fs = [round(x, 4) for x in np.arange(f_lo, f_hi + 1e-9, step)]
    rows, missing = [], []
    for f in fs:
        g = solve_gamma(f, target, n)
        if g is None:
            missing.append(f)
        else:
            rows.append((f, g))
    if missing:
        raise SystemExit("target %.4f unreachable at %d frequencies, first few: %s"
                         % (target, len(missing), missing[:8]))
    fg = [r[0] for r in rows]
    gg = [r[1] for r in rows]
    out = dict(n=n, gamma_baseline=C.GAMMA, target=float(target),
               target_rule="0.97 * min_f attained(f, 0.02); common range "
                           "[max_f attained(f,0.20), min_f attained(f,0.02)]",
               gamma_min=GAMMA_MIN, gamma_max=GAMMA_MAX,
               f=fg, gamma=[float(x) for x in gg],
               attained=[attained_norm(f, g, n) for f, g in rows],
               f_min=float(min(fg)), f_max=float(max(fg)),
               gamma_lo=float(min(gg)), gamma_hi=float(max(gg)))
    C.save_json(out, os.path.join(C.RES, "gamma_comp.json"))
    return out


_TABLE = None


def comp_gamma(f, table_path=None):
    """gamma(f) from the table; accepts a scalar or a torch/numpy batch.

    Loaded lazily and cached so the training and evaluation processes share one
    table.  Outside the tabulated range the end values are held constant, which
    is the conservative choice: the table spans [1, 12], the whole test grid.
    """
    global _TABLE
    if _TABLE is None:
        p = table_path or os.path.join(C.RES, "gamma_comp.json")
        _TABLE = C.load_json(p)
    if isinstance(f, torch.Tensor):
        fn = f.detach().cpu().numpy().astype(np.float64)
        g = np.interp(fn, _TABLE["f"], _TABLE["gamma"])
        return torch.tensor(g, dtype=torch.float64, device=f.device)
    fn = float(f)
    return float(np.interp(fn, _TABLE["f"], _TABLE["gamma"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--f-lo", type=float, default=1.0)
    ap.add_argument("--f-hi", type=float, default=12.0)
    ap.add_argument("--step", type=float, default=0.01)
    ap.add_argument("--target", type=float, default=None)
    a = ap.parse_args()
    t = build(f_lo=a.f_lo, f_hi=a.f_hi, step=a.step, target=a.target)
    print("target attained norm = %.4f (%s)" % (t["target"], t["target_rule"]))
    print("gamma(f) range: %.4f .. %.4f over f in [%.2f, %.2f]"
          % (t["gamma_lo"], t["gamma_hi"], t["f_min"], t["f_max"]))
    print("   f     gamma(f)    attained")
    for f, g, at in list(zip(t["f"], t["gamma"], t["attained"]))[::4]:
        print("%6.2f   %8.4f   %8.4f" % (f, g, at))
    # sanity: the tabulated gamma must actually reproduce the target
    worst = max(abs(at - t["target"]) for at in t["attained"])
    print("max |attained - target| over table = %.2e (must be ~0)" % worst)
    # sanity: interpolation must stay monotone-sane at the test frequencies
    for f in C.F_TEST:
        g = comp_gamma(f)
        print("  f=%5.2f -> gamma=%.4f, attained=%.4f"
              % (f, g, attained_norm(f, g, t["n"])))
