# Author: Longgang Wang
"""Flatten the chunked cross-design caches into one memory-mappable array set.

Why this exists
---------------
An .npz is a zip archive: ``np.load(..., mmap_mode="r")`` cannot map a member,
so every element access re-reads the whole member.  Training then spends its
time in disk I/O -- the first attempt did not finish an epoch in five minutes,
against 19 s/epoch for the same model on the 64^2 arm.  One .npy per field per
grid maps cleanly, so a random batch pages in exactly the rows it needs.

The .npz chunks are kept: they are the archived form, this is a derived
training layout.

Usage
-----
    python code/flatten_cross_cache.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

FIELDS = [("v", np.float32), ("psi", np.complex64), ("k", np.float32),
          ("f", np.float32), ("theta", np.float32), ("res", np.float32)]


def flatten(N):
    folder = os.path.join(C.DATA, "cross_N%03d" % N)
    fs = sorted(os.path.join(folder, f) for f in os.listdir(folder)
                if f.endswith(".npz"))
    if not fs:
        print("[N=%3d] no chunks" % N)
        return
    out = {}
    for name, dt in FIELDS:
        arrs = [np.load(f)[name].astype(dt) for f in fs]
        out[name] = np.concatenate(arrs, axis=0)
        arrs = None
    n = out["v"].shape[0]
    for name, _ in FIELDS:
        np.save(os.path.join(folder, name + ".npy"), out[name])
    with open(os.path.join(folder, "flat.txt"), "w") as fh:
        fh.write("%d\n" % n)
    print("[N=%3d] %d samples -> v.npy (%.0f MB), psi.npy (%.0f MB)"
          % (N, n, out["v"].nbytes / 1e6, out["psi"].nbytes / 1e6), flush=True)


if __name__ == "__main__":
    targets = [int(x) for x in sys.argv[1:]] or [64, 96, 128, 192]
    for N in targets:
        flatten(N)
