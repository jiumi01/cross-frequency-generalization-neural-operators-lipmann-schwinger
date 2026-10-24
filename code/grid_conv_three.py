# Author: Longgang Wang
"""Systematic three-grid convergence study: 64^2 / 128^2 / 256^2.

Review item 1 (highest priority).  The previous study had one 256^2 point at
f = 12 and n = 8 media, which supports a trend but not a convergence claim.
This one makes the claim measurable:

  Stage A  reference-solution convergence.  One continuous medium is drawn at
           256^2 and block-mean downsampled to 128^2 and 64^2, so the three
           grids discretise the *same* continuum problem.  For every f in
           {4,5,6,8,10,12} we report

               d64v128  = mean || psi64  - down(psi128) || / ||down(psi128)||
               d128v256 = mean || psi128 - down(psi256) || / ||down(psi256)||
               p        = log2( d64v128 / d128v256 )          (observed order)

           and the trivial incident-field baseline on each of the three grids,
           so that the quantity the plateau is compared against is measured at
           the same resolutions as the plateau itself.

  Stage B  plateau height against a refined target.  The trained FNO is held
           at its native 64^2 input.  Its prediction is scored three times,
           against the 64^2 reference, against down(psi128) and against
           down(down(psi256)).  If the three agree, the reported plateau is
           not an artefact of the 64^2 target.  The trivial baseline is scored
           the same way, so the plateau-vs-baseline ratio is compared at a
           matched resolution rather than mixed across resolutions.

  Control  input-path check.  The model is trained on media drawn natively at
           64^2, while stage B feeds it a medium downsampled from 256^2.  The
           control scores the same model on both input paths against the same
           64^2 reference, so that input path can be separated from the
           target-resolution effect.

Nothing here is averaged over frequencies: every quantity is reported per f.
"""
import os, sys, json, time, argparse
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import physics as P
import data as D
from evaluate import load_model

dev = "cuda" if torch.cuda.is_available() else "cpu"
FREQS = [4, 5, 6, 8, 10, 12]


def downsample(x, f=2):
    """Block-mean downsample (B, fN, fN) -> (B, N, N)."""
    B = x.shape[0]
    N = x.shape[-1] // f
    return x.reshape(B, N, f, N, f).mean(dim=(2, 4))


def rel_l2(pred, target):
    num = torch.linalg.norm((pred - target).reshape(pred.shape[0], -1), dim=1)
    den = torch.linalg.norm(target.reshape(target.shape[0], -1), dim=1).clamp_min(1e-12)
    return num / den


def make_media(n, seed, vmax, n_grid, device):
    g = torch.Generator(device=device).manual_seed(int(seed) + 7777)
    return P.make_contrast(n_grid, n, 1.0, C.MM_REF, g, device) * vmax


def solve_at(v, f, theta, n_grid, device):
    n_b = v.shape[0]
    fv = torch.full((n_b,), float(f), dtype=torch.float64, device=device)
    k = 2 * np.pi * fv
    return P.solve_ls(v, k, theta, C.GAMMA, n_grid, tol=1e-9, maxiter=900,
                      device=device)


# ------------------------------------------------------------------ Stage A
def stage_a(n, seed, vmax, device):
    g = torch.Generator(device=device).manual_seed(seed)
    theta = torch.rand(n, generator=g, device=device) * 2 * np.pi
    v256 = make_media(n, seed, vmax, 256, device)
    v128 = downsample(v256)
    v64 = downsample(v128)

    rows = []
    t0 = time.time()
    for f in FREQS:
        psi256 = solve_at(v256, f, theta, 256, device)
        psi128 = solve_at(v128, f, theta, 128, device)
        psi64 = solve_at(v64, f, theta, 64, device)

        d128v256 = rel_l2(downsample(psi256), psi128).mean().item()
        d64v128 = rel_l2(psi64, downsample(psi128)).mean().item()
        d64v256 = rel_l2(psi64, downsample(downsample(psi256))).mean().item()
        p = float(np.log2(d64v128 / max(d128v256, 1e-12)))

        # trivial incident-field baseline on each native grid
        kv = 2 * np.pi * torch.full((n,), float(f), dtype=torch.float64,
                                    device=device)
        triv = {}
        for tag, N, psi in (("64", 64, psi64), ("128", 128, psi128),
                            ("256", 256, psi256)):
            inc = P.incident_plane_wave(N, kv, theta, device)
            triv[tag] = rel_l2(psi, inc).mean().item()

        rows.append(dict(f=f, d64v128=d64v128, d128v256=d128v256,
                         d64v256=d64v256, p=p,
                         triv64=triv["64"], triv128=triv["128"],
                         triv256=triv["256"]))
        print("[A] vmax=%-5s f=%-4g d64v128=%.4f d128v256=%.4f p=%.2f "
              "triv=%.3f/%.3f/%.3f  (%.0fs)"
              % (vmax, f, d64v128, d128v256, p, triv["64"], triv["128"],
                 triv["256"], time.time() - t0), flush=True)
        del psi256
        if device == "cuda":
            torch.cuda.empty_cache()
    return rows


# ------------------------------------------------------------------ Stage B
def stage_b(n, seed, vmax, device, ckpt):
    g = torch.Generator(device=device).manual_seed(seed)
    theta = torch.rand(n, generator=g, device=device) * 2 * np.pi
    v256 = make_media(n, seed, vmax, 256, device)
    v128 = downsample(v256)
    v64 = downsample(v128)

    model, ck = load_model(ckpt, device)
    model.eval()

    @torch.no_grad()
    def predict(v_in, N):
        """Network output on an N-grid input, in complex128 on the N grid."""
        kv = 2 * np.pi * torch.full((n,), float(f_cur[0]), dtype=torch.float64,
                                    device=device)
        inc = P.incident_plane_wave(N, kv, theta, device).to(torch.complex64)
        fv = torch.full((n,), float(f_cur[0]), device=device)
        o = model(D.build_inputs(v_in.float(), inc, fv, emb="none"))
        u = torch.complex(o[:, 0], o[:, 1])
        return (inc + u).to(torch.complex128)

    rows = []
    f_cur = [0.0]
    t0 = time.time()
    for f in FREQS:
        f_cur[0] = float(f)
        kv64 = 2 * np.pi * torch.full((n,), float(f), dtype=torch.float64,
                                      device=device)
        inc64 = P.incident_plane_wave(64, kv64, theta, device)

        psi256 = solve_at(v256, f, theta, 256, device)
        psi128 = solve_at(v128, f, theta, 128, device)
        psi64 = solve_at(v64, f, theta, 64, device)
        t64, t128, t256 = psi64, downsample(psi128), downsample(downsample(psi256))

        pred = predict(v64, 64)

        e64 = rel_l2(pred, t64).mean().item()
        e128 = rel_l2(pred, t128).mean().item()
        e256 = rel_l2(pred, t256).mean().item()
        b64 = rel_l2(inc64, t64).mean().item()
        b128 = rel_l2(inc64, t128).mean().item()
        b256 = rel_l2(inc64, t256).mean().item()

        # control: does the input path (native 64 draw vs downsampled) matter?
        v64nat = make_media(n, seed, vmax, 64, device)
        psinat = solve_at(v64nat, f, theta, 64, device)
        prednat = predict(v64nat, 64)
        enat = rel_l2(prednat, psinat).mean().item()

        rows.append(dict(f=f, e64=e64, e128=e128, e256=e256,
                         b64=b64, b128=b128, b256=b256, e_native=enat))
        print("[B] vmax=%-5s f=%-4g eps=%.4f/%.4f/%.4f  triv=%.4f/%.4f/%.4f "
              "native64=%.4f  (%.0fs)"
              % (vmax, f, e64, e128, e256, b64, b128, b256, enat,
                 time.time() - t0), flush=True)
        del psi256
        if device == "cuda":
            torch.cuda.empty_cache()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-soft", type=int, default=64)
    ap.add_argument("--n-strong", type=int, default=32)
    ap.add_argument("--only", default="")
    ap.add_argument("--go", action="store_true")
    a = ap.parse_args()
    if not a.go:
        print("refusing to run without --go (pre-registration: the frequency "
              "list and the sample counts are fixed before the run)")
        return

    ck_soft = os.path.join(C.RUNS, "main_fno_a025_s0", "checkpoint.pt")
    ck_strong = os.path.join(C.RUNS, "main_fno_a100_s0", "checkpoint.pt")
    out = dict(stage="three-grid", freqs=FREQS, gamma=C.GAMMA,
               corr=C.MM_REF, device=dev)
    t0 = time.time()

    if a.only in ("", "A"):
        for tag, vmax, n, seed in (("0.25", 0.25, a.n_soft, 61001),
                                   ("1.00", 1.00, a.n_strong, 61002)):
            out["ref_%s" % tag] = dict(vmax=vmax, n=n, seed=seed,
                                       rows=stage_a(n, seed, vmax, dev))

    if a.only in ("", "B"):
        for tag, vmax, n, seed, ck in (("0.25", 0.25, a.n_soft, 61001, ck_soft),
                                       ("1.00", 1.00, a.n_strong, 61002, ck_strong)):
            if not os.path.exists(ck):
                print("missing checkpoint", ck)
                continue
            out["model_%s" % tag] = dict(vmax=vmax, n=n, seed=seed,
                                         rows=stage_b(n, seed, vmax, dev, ck))

    out["total_s"] = time.time() - t0
    C.save_json(out, os.path.join(C.RES, "grid_conv_three.json"))
    print("saved results/grid_conv_three.json  total %.0fs" % (time.time() - t0),
          flush=True)


if __name__ == "__main__":
    main()
