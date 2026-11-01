# Author: Longgang Wang <longgangwang@163.com>
# Affiliation: Luoyang Institute of Science and Technology, Luoyang 471000, Henan, China

"""Is the part-2 'disagreement' a solver defect, or a precision floor of the check?

Part 2 recomputed b - A psi from the *shipped* fields and got ~1e-8, i.e. 30-300x
the recorded ~1e-9.  Before concluding anything about the solver, the instrument
has to be shown capable of the measurement at all.

The shipped fields are stored single precision (psi complex64, v/k/theta float32;
see data.py lines 78-85).  A residual formed from a complex64 psi cannot be
sharper than ~eps32 * ||A psi|| / ||b||, so if the check is precision-limited the
same solve evaluated (i) at working precision and (ii) after a complex64
round-trip must straddle the recorded value.

That is a two-line experiment on ONE shipped chunk, entirely on CPU:

  ref  : float64 solve from the shipped (float32) v, k, theta
         -> residual at working precision
  ship : the shipped complex64 psi itself
         -> residual after the storage round-trip

If ref reproduces the recorded ~1e-9 and ship lands at ~1e-8, the discrepancy is
the round-trip and nothing else.
"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
except Exception:
    pass


def relres(psi, v, k, theta, gam, dev="cpu"):
    import torch
    import physics as P
    v = v.to(torch.complex128)
    Ghat = P.green_symbol(C.N, k, gam, dev)
    k2 = (k.to(torch.complex128) * (1.0 + 1j * gam)) ** 2
    b = P.incident_plane_wave(C.N, k, theta, dev).to(torch.complex128)
    Apsi = psi - k2[:, None, None] * torch.fft.ifft2(Ghat * torch.fft.fft2(v * psi))
    return (torch.linalg.norm((b - Apsi).reshape(len(k), -1), dim=1)
            / torch.linalg.norm(b.reshape(len(k), -1), dim=1)).numpy()


def main():
    import torch
    p = sorted(glob.glob(os.path.join(C.DATA, "train_a025", "chunk_*.npz")))[0]
    with np.load(p) as z:
        n = 32
        v32 = z["v"][:n]
        psi32 = z["psi"][:n]
        k32 = z["k"][:n]
        th32 = z["theta"][:n]
        stored = z["res"][:n].astype(np.float64)
    print("chunk:", os.path.basename(p), "  n =", n)

    v = torch.from_numpy(v32.astype(np.float64))
    k = torch.from_numpy(k32.astype(np.float64))
    th = torch.from_numpy(th32.astype(np.float64))
    psi_ship = torch.from_numpy(psi32.astype(np.complex128))

    # (i) working-precision solve from the SHIPPED inputs
    psi_ref, info = __import__("physics").solve_ls(
        v, k, th, C.GAMMA, C.N, tol=1e-12, maxiter=900, device="cpu",
        return_info=True)
    r_ref = relres(psi_ref, v, k, th, C.GAMMA)

    # (ii) the same solution pushed through the complex64 storage round-trip
    psi_rt = torch.from_numpy(psi_ref.numpy().astype(np.complex64).astype(np.complex128))
    r_rt = relres(psi_rt, v, k, th, C.GAMMA)

    # (iii) the shipped fields, as the verifier read them
    r_ship = relres(psi_ship, v, k, th, C.GAMMA)

    # how far the storage round-trip moves the solution itself
    dpsi = (torch.linalg.norm((psi_rt - psi_ref).reshape(n, -1), dim=1)
            / torch.linalg.norm(psi_ref.reshape(n, -1), dim=1)).numpy()
    dship = (torch.linalg.norm((psi_ship - psi_ref).reshape(n, -1), dim=1)
             / torch.linalg.norm(psi_ref.reshape(n, -1), dim=1)).numpy()

    print("\n%-34s %-12s %-12s" % ("quantity", "median", "max"))
    for name, a in [("recorded residual (generator)", stored),
                    ("residual of float64 solve", r_ref),
                    ("residual after complex64 round-trip", r_rt),
                    ("residual of shipped complex64 psi", r_ship),
                    ("rel. ||psi_ship - psi_ref||", dship),
                    ("rel. ||round-trip - psi_ref||", dpsi)]:
        print("%-34s %-12.3e %-12.3e" % (name, np.median(a), np.max(a)))

    print("\nfloat64 solve reproduces the recorded value: %s"
          % (np.max(np.abs(r_ref - stored)) < 1e-4 * np.max(stored) + 1e-11))
    print("round-trip alone lifts the floor to ~%.1e" % np.median(r_rt))
    print("psi relative perturbation from storage: ~%.1e (complex64 eps ~ %.1e)"
          % (np.median(dship), np.finfo(np.complex64).eps))


if __name__ == "__main__":
    main()
