# Author: Longgang Wang <longgangwang@163.com>
# Affiliation: Luoyang Institute of Science and Technology, Luoyang 471000, Henan, China

"""Validation of the Lippmann-Schwinger solver (run before any training)."""
import sys, os, time, json
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")
except Exception:
    pass
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import physics as P

torch.manual_seed(0)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEV)


def dense_solve(v, k_arr, theta_arr, gamma, N):
    """Reference: form the dense operator by matvec probing and solve directly.

    NOTE: the operator is built per-sample (the Green symbol / kappa^2 must not
    carry a batch dimension here, otherwise broadcasting silently mixes samples).
    """
    B = v.shape[0]
    n2 = N * N
    out = []
    for b in range(B):
        kb = k_arr[b:b + 1]
        Ghat = P.green_symbol(N, kb, gamma, DEV)                       # (1,N,N)
        kappa2 = ((kb.to(torch.complex128) * (1.0 + 1j * gamma)) ** 2)[:, None, None]
        vb = v[b:b + 1].to(torch.complex128)

        def matvec(psi):
            return psi - kappa2 * torch.fft.ifft2(Ghat * torch.fft.fft2(vb * psi))

        A = torch.zeros((n2, n2), dtype=torch.complex128, device=DEV)
        e = torch.zeros((1, N, N), dtype=torch.complex128, device=DEV)
        for j in range(n2):
            e.zero_()
            e[0].reshape(-1)[j] = 1.0
            A[:, j] = matvec(e)[0].reshape(-1)
        rhs = P.incident_plane_wave(N, kb, theta_arr[b:b + 1], DEV)[0].reshape(-1)
        psi = torch.linalg.solve(A, rhs)
        out.append(psi.reshape(N, N))
    return torch.stack(out)


print("\n=== Test 1: BiCGSTAB vs dense direct solve (N=10) ===")
N = 10
B = 3
gen = torch.Generator(device=DEV).manual_seed(1)
v = P.make_contrast(N, B, 0.4, 0.15, gen, DEV, kind="smooth")
k_arr = torch.tensor([2 * np.pi * 1.5, 2 * np.pi * 3.0, 2 * np.pi * 5.0], device=DEV)
th = torch.tensor([0.3, 1.2, 2.7], device=DEV)
psi_ref = dense_solve(v, k_arr, th, 0.05, N)
psi_bi, res = P.solve_ls(v, k_arr, th, 0.05, N, tol=1e-12, maxiter=2000,
                         device=DEV, return_info=True)
err = torch.linalg.norm((psi_bi - psi_ref).reshape(B, -1), dim=1) / \
      torch.linalg.norm(psi_ref.reshape(B, -1), dim=1)
print("  rel diff BiCGSTAB vs dense:", [float(x) for x in err])
print("  bi-cgstab residual:", [float(x) for x in res])

print("\n=== Test 2: residual & iteration cost across f and v_max (N=64) ===")
N = 64
rows = []
for f in [1.0, 2.0, 4.0, 6.0, 8.0, 12.0]:
    for vmax in [0.1, 0.5, 1.0]:
        B = 8
        gen = torch.Generator(device=DEV).manual_seed(11)
        v = P.make_contrast(N, B, vmax, 0.12, gen, DEV, kind="smooth")
        k_arr = torch.full((B,), 2 * np.pi * f, device=DEV)
        th = torch.rand(B, generator=gen, device=DEV) * 2 * np.pi
        t0 = time.time()
        psi, res = P.solve_ls(v, k_arr, th, 0.05, N, tol=1e-9, maxiter=1000,
                              device=DEV, return_info=True)
        dt = time.time() - t0
        rows.append(dict(f=f, vmax=vmax, max_res=float(res.max()),
                         mean_res=float(res.mean()), sec=round(dt, 2),
                         amp=float(psi.abs().mean())))
        print("  f=%5.1f vmax=%.1f  max_res=%.2e  amp=%.3f  %.2fs" %
              (f, vmax, float(res.max()), float(psi.abs().mean()), dt))

print("\n=== Test 3: Born-series behaviour vs (f, v_max)  [exact reference] ===")
N = 64
B = 4
rows3 = []
for f in [1.0, 2.0, 4.0, 8.0]:
    for vmax in [0.05, 0.2, 0.5, 1.0]:
        gen = torch.Generator(device=DEV).manual_seed(5)
        v = P.make_contrast(N, B, vmax, 0.12, gen, DEV, kind="smooth")
        k_arr = torch.full((B,), 2 * np.pi * f, device=DEV)
        th = torch.rand(B, generator=gen, device=DEV) * 2 * np.pi
        psi_exact = P.solve_ls(v, k_arr, th, 0.05, N, tol=1e-10, device=DEV)
        e1 = float((torch.linalg.norm((P.born_approx(v, k_arr, th, 0.05, N, 1, DEV)
                                       - psi_exact).reshape(B, -1), dim=1) /
                    torch.linalg.norm(psi_exact.reshape(B, -1), dim=1)).mean())
        e5 = float((torch.linalg.norm((P.born_approx(v, k_arr, th, 0.05, N, 5, DEV)
                                       - psi_exact).reshape(B, -1), dim=1) /
                    torch.linalg.norm(psi_exact.reshape(B, -1), dim=1)).mean())
        # truncation-vs-exact ratio tells whether the Neumann series contracts
        rows3.append(dict(f=f, vmax=vmax, born1=round(e1, 4), born5=round(e5, 4)))
        print("  f=%4.1f vmax=%.2f  relL2(Born order1)=%9.3e  order5=%9.3e" % (f, vmax, e1, e5))

json.dump(rows, open(os.path.join(os.path.dirname(__file__), "..", "results",
                                  "solver_validation.json"), "w"), indent=1)
print("\nDONE")
