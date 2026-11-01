# Author: Longgang Wang <longgangwang@163.com>
# Affiliation: Luoyang Institute of Science and Technology, Luoyang 471000, Henan, China

"""Discrete operator norm of the Lippmann-Schwinger propagator, per test frequency.

The benchmark's design argument was that with kappa = k(1 + i*gamma) the
propagator norm obeys sup_xi |kappa^2 G| = (1 + gamma^2)/(2 gamma) *independently
of k*, which removes the frequency dependence of the forward map's conditioning
and leaves the frequency of the solution as the only thing varying.

The exact constant matters, and an earlier revision of this script got it wrong.
With |kappa^2| = k^2 (1 + gamma^2) and the denominator minimised over real |xi|
at |xi| = k sqrt(1 - gamma^2), where |xi^2 - kappa^2| = 2 gamma k^2, the sup is
(1 + gamma^2)/(2 gamma) -- NOT 1/(2 gamma), which is only its gamma -> 0 limit and
is 0.25% low at gamma = 0.05.  The manuscript quoted 1/(2 gamma) as an identity
and then compared it with a table that printed 10.02, so the text said 10 where
its own table said 10.02.

That identity is a statement about the CONTINUUM symbol.  On an N x N grid the
symbol is sampled at |xi|^2 = 4*pi^2*|n|^2, which takes only discrete values, so
the attained maximum is max over the representable modes and is frequency
dependent.  It reaches the continuum value only where a grid mode happens to land
on the circle |xi| = k sqrt(1 - gamma^2) -- a property of the grid, not of the
frequency, and not the same condition as "k lands exactly on a mode (integer f)".
At integer f the attained norm is (1 + gamma^2)/(gamma sqrt(gamma^2 + 4)), which
is 0.03% *below* the continuum value at gamma = 0.05.  Both facts are recorded
below so the manuscript can state them instead of inheriting the continuum
identity.

Read-only: it touches no archive, only results/propnorm.json.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=C.N)
    ap.add_argument("--gamma", type=float, default=0.05)
    ap.add_argument("--out", default=os.path.join(C.RES, "propnorm.json"))
    a = ap.parse_args()

    dev = "cpu"          # a 64x64 symbol needs no GPU, and this must be runnable
    freqs = list(C.F_TEST)
    k = 2.0 * np.pi * np.array(freqs, float)
    G = P.green_symbol(a.n, torch.tensor(k), a.gamma, dev)      # (B, N, N)
    # kappa^2 * Ghat, with the same kappa = k(1 + i gamma) the solver uses.
    kap2 = (torch.tensor(k, dtype=torch.complex128)
            * (1.0 + 1j * a.gamma)) ** 2
    KG = kap2[:, None, None] * G
    norm = KG.abs().reshape(len(freqs), -1).max(dim=1).values.numpy()

    # Two constants, kept apart on purpose: the exact continuum sup and its
    # gamma -> 0 limit.  Anything the manuscript compares against a table must
    # use the exact one.
    nominal_exact = (1.0 + a.gamma ** 2) / (2.0 * a.gamma)
    nominal_leading = 1.0 / (2.0 * a.gamma)
    imax = int(np.argmax(norm))
    integ = [(f, v) for f, v in zip(freqs, norm) if abs(f - round(f)) < 1e-9]
    out = dict(n=a.n, gamma=a.gamma,
               nominal=float(nominal_exact),
               nominal_leading=float(nominal_leading),
               nominal_formula="(1+gamma^2)/(2*gamma)",
               f_argmax=float(freqs[imax]), norm_argmax=float(norm[imax]),
               norm_on_integer=max(v for _, v in integ) if integ else None,
               onmode_formula="(1+gamma^2)/(gamma*sqrt(gamma^2+4))",
               f=freqs, norm=[float(v) for v in norm],
               ratio=[float(v / nominal_exact) for v in norm])
    C.save_json(out, a.out)

    print("grid %dx%d, gamma=%.3f" % (a.n, a.n, a.gamma))
    print("continuum sup (1+gamma^2)/(2 gamma) = %.6f" % nominal_exact)
    print("leading order 1/(2 gamma)          = %.6f  (%.2f%% low)"
          % (nominal_leading, 100 * a.gamma ** 2))
    print("mode-aligned (1+g^2)/(g sqrt(g^2+4)) = %.6f"
          % (a.gamma and (1.0 + a.gamma ** 2)
             / (a.gamma * np.sqrt(a.gamma ** 2 + 4.0))))
    print("   f      max|kappa^2 Ghat|    ratio to exact sup")
    for f, v in zip(freqs, norm):
        print("%6.2f      %12.6f      %10.4f" % (f, v, v / nominal_exact))
    print("min over test grid = %.4f at f = %.2f"
          % (float(norm.min()), freqs[int(np.argmin(norm))]))
    print("max over test grid = %.4f at f = %.2f"
          % (float(norm.max()), freqs[int(np.argmax(norm))]))


if __name__ == "__main__":
    main()
