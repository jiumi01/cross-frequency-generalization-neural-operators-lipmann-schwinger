# Author: Longgang Wang
"""Timing probe for one multi-grid training step, per grid and per stage.

Run before setting a schedule: the first two attempts at the multi-grid trainer
were set from an estimate, and the estimate was wrong by an order of magnitude.
Splits the step into batch load, forward, backward and reports each.
"""
import os, sys, time
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import data as D
import models as M
import physics as P
from train import rel_l2
from train_multigrid import GridStore

dev = "cuda"
model = M.build_model("fno", in_ch=4, out_ch=2, width=40, modes=20,
                      n_layers=4, proj=128, norm="layer").to(dev)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
store = GridStore()
for N in (64, 96, 128, 192):
    idx = torch.randperm(4096)[:64].numpy()
    for rep in range(3):
        t0 = time.time(); v, psi, k, f, th = store.batch(N, idx, dev)
        torch.cuda.synchronize(); t1 = time.time()
        inc = P.incident_plane_wave(N, k, th, dev).to(torch.complex64)
        x = D.build_inputs(v, inc, f)
        o = model(x)
        u = torch.complex(o[:, 0], o[:, 1])
        loss = rel_l2(inc + u, psi).mean()
        torch.cuda.synchronize(); t2 = time.time()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        torch.cuda.synchronize(); t3 = time.time()
        if rep == 2:
            print("N=%3d  load=%.3fs  fwd(+loss)=%.3fs  bwd=%.3fs  total=%.3fs  peakMB=%.0f"
                  % (N, t1 - t0, t2 - t1, t3 - t2, t3 - t0,
                     torch.cuda.max_memory_allocated() / 2**20), flush=True)
