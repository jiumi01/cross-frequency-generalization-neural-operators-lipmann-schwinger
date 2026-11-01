# Author: Longgang Wang
"""Multi-resolution FNO training for the frequency x resolution cross design.

The cross design needs ONE model that can be evaluated at every (f, N) cell.
Evaluating the 64^2-trained model at N=192 would confound the frequency
extrapolation with a resolution extrapolation the arm never trained for, so
the model is trained on all four grids used at test time.

Budget is matched to the reference arm step for step: same batch (64), same
number of optimiser steps (55 epochs x 192 steps), same width/modes/layers,
same learning-rate schedule.  Each step draws a grid in rotation, so every
grid sees exactly 1/4 of the steps and a quarter of the gradient signal; that
is inherent to multi-grid training and is why the arm also reports its in-band
error -- if multi-grid training by itself damaged in-band accuracy, the
out-of-band comparison would be against a different operating point.

Caches are memory-mapped: the four grids together are ~11 GB of float32 /
complex64, which does not fit beside the model and the CUDA context on a
32 GB machine, and random access through the OS page cache is fast enough that
the loader is never the bottleneck.

Usage
-----
    python code/train_multigrid.py --name cross_fno_a025_s0 --seed 0
"""
import os
import sys
import csv
import time
import argparse

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import data as D
import models as M
from train import rel_l2

GRIDS = (64, 96, 128)
N_TRAIN = 12288
N_VAL = 1228


class GridStore:
    """Multi-grid cache backed by one memory-mapped .npy per field.

    An .npz is a zip archive, so ``mmap_mode`` cannot map its members and every
    element access re-reads the whole member; a first version that indexed the
    chunks per sample did not finish an epoch in five minutes.  The flattened
    layout (code/flatten_cross_cache.py) maps cleanly, so a random batch pages
    in exactly the rows it needs.
    """

    def __init__(self, grids=GRIDS, root=None, mem_limit_gb=8.0):
        root = root or C.DATA
        self.grids = list(grids)
        self.a = {}
        total = 0.0
        for N in self.grids:
            folder = os.path.join(root, "cross_N%03d" % N)
            total += (os.path.getsize(os.path.join(folder, "v.npy")) +
                      os.path.getsize(os.path.join(folder, "psi.npy"))) / 2**30
        # Random rows out of a memory map are single small reads at scattered
        # offsets; on this machine that costs ~10 ms each, i.e. ~0.6 s per
        # batch, which is six times the arithmetic.  If the whole set fits,
        # hold it in RAM and the same batch becomes a memcpy.
        self.inram = total <= mem_limit_gb
        for N in self.grids:
            folder = os.path.join(root, "cross_N%03d" % N)
            self.a[N] = {k: np.load(os.path.join(folder, k + ".npy"),
                                    mmap_mode=None if self.inram else "r")
                         for k in ("v", "psi", "k", "f", "theta")}
            n = self.a[N]["v"].shape[0]
            assert n >= N_TRAIN + N_VAL, (N, n)
        print("[store] %d grids, %.1f GB held %s"
              % (len(self.grids), total, "in RAM" if self.inram else "as mmap"),
              flush=True)

    def batch(self, N, idx, device):
        idx = np.asarray(idx)
        a = self.a[N]
        v = torch.from_numpy(np.asarray(a["v"][idx])).to(device).float()
        psi = torch.from_numpy(np.asarray(a["psi"][idx])).to(device)
        k = torch.from_numpy(np.asarray(a["k"][idx], dtype=np.float32)).to(device)
        f = torch.from_numpy(np.asarray(a["f"][idx], dtype=np.float32)).to(device)
        th = torch.from_numpy(np.asarray(a["theta"][idx], dtype=np.float32)).to(device)
        return v, psi, k, f, th


def forward_loss(model, v, psi, k, f, th):
    import physics as P
    psi_inc = P.incident_plane_wave(v.shape[-1], k, th, v.device).to(torch.complex64)
    x = D.build_inputs(v, psi_inc, f)
    o = model(x)
    u = torch.complex(o[:, 0], o[:, 1])
    return rel_l2(psi_inc + u, psi).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=55)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--width", type=int, default=40)
    ap.add_argument("--modes", type=int, default=20)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--proj", type=int, default=128)
    ap.add_argument("--norm", default="layer")
    ap.add_argument("--val-per-grid", type=int, default=192)
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = os.path.join(C.RUNS, a.name)
    if os.path.exists(os.path.join(out, "checkpoint.pt")):
        print("run %s already complete; refusing to overwrite" % a.name)
        return
    os.makedirs(out, exist_ok=True)

    def log(msg):
        print("[%s] %s" % (C.now(), msg), flush=True)

    store = GridStore()
    log("grids=%s train/grid=%d val/grid=%d" % (store.grids, N_TRAIN, N_VAL))

    model = M.build_model("fno", in_ch=D.in_channels("none", "none"),
                          out_ch=2, width=a.width, modes=a.modes,
                          n_layers=a.layers, proj=a.proj, norm=a.norm).to(dev)
    nparam = C.count_params(model)
    log("params=%.3fM" % (nparam / 1e6))

    cfg = dict(name=a.name, model="fno", grids=list(store.grids),
               epochs=a.epochs, batch=a.batch, lr=a.lr, width=a.width,
               modes=a.modes, layers=a.layers, proj=a.proj, norm=a.norm,
               n_train=N_TRAIN, n_val=N_VAL, seed=a.seed, params=int(nparam),
               device=dev, torch=torch.__version__,
               design="frequency x resolution cross (rev. item 4, path 3)")
    C.save_json(cfg, os.path.join(out, "config.json"))

    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs,
                                                       eta_min=a.lr * 0.01)
    steps = N_TRAIN // a.batch
    val_idx = torch.arange(N_TRAIN, N_TRAIN + a.val_per_grid)
    csv_f = open(os.path.join(out, "train_log.csv"), "w", newline="")
    w = csv.writer(csv_f)
    w.writerow(["epoch", "train_relL2"] + ["val_%d" % N for N in store.grids]
               + ["val_mean", "lr", "sec"])
    best = 1e9
    t0 = time.time()

    for ep in range(1, a.epochs + 1):
        model.train()
        perm = {N: torch.randperm(N_TRAIN, generator=torch.Generator().manual_seed(
            a.seed * 100003 + ep * 31 + N)) for N in store.grids}
        tot, cnt = 0.0, 0
        trace = os.environ.get("TRACE_STEPS") == "1" and ep == 1
        for s in range(steps):
            N = store.grids[s % len(store.grids)]
            sel = perm[N][s // len(store.grids) * a.batch:
                          (s // len(store.grids) + 1) * a.batch]
            t_load = time.time()
            v, psi, k, f, th = store.batch(N, sel.numpy(), dev)
            if trace:
                print("   step %3d N=%3d load=%.2fs" % (s, N, time.time() - t_load),
                      flush=True)
            loss = forward_loss(model, v, psi, k, f, th)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss) * sel.numel()
            cnt += sel.numel()
        sched.step()

        model.eval()
        vals = {}
        with torch.no_grad():
            for N in store.grids:
                # 192 validation samples in ONE batch at 192^2 asks for ~25 GB
                # on a 12 GB card; WDDM silently backs it with system memory
                # and the step slows by an order of magnitude rather than
                # failing, which is how the first two schedules were lost.
                tot_v, cnt_v = 0.0, 0
                for s0 in range(0, val_idx.numel(), 32):
                    sub = val_idx[s0:s0 + 32]
                    v, psi, k, f, th = store.batch(N, sub.numpy(), dev)
                    l = forward_loss(model, v, psi, k, f, th)
                    tot_v += float(l) * sub.numel()
                    cnt_v += sub.numel()
                vals[N] = tot_v / cnt_v
        vm = float(np.mean(list(vals.values())))
        w.writerow([ep, tot / cnt] + ["%.6e" % vals[N] for N in store.grids]
                   + ["%.6e" % vm, "%.3e" % sched.get_last_lr()[0],
                      "%.0f" % (time.time() - t0)])
        csv_f.flush()
        log("ep %3d train=%.4e %s mean=%.4e %.0fs"
            % (ep, tot / cnt, " ".join("v%d=%.3e" % (N, vals[N])
                                       for N in store.grids), vm,
               time.time() - t0))
        if vm < best:
            best = vm
            torch.save(dict(model=model.state_dict(), epoch=ep, val=vm,
                            cfg=cfg), os.path.join(out, "checkpoint.pt"))
    csv_f.close()
    log("DONE best_val=%.4e (%.0fs)" % (best, time.time() - t0))


if __name__ == "__main__":
    main()
