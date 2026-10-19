# Author: Longgang Wang <longgangwang@163.com>
# Affiliation: Luoyang Institute of Science and Technology, Luoyang 471000, Henan, China

"""Verification: which U-Net attains the lowest out-of-band plateau in Fig. 4?

Read-only diagnostic. Recomputes the plateau and in-band floor per architecture
arm from the evaluation records with the project's own figure code, so the
caption can be checked against the same computation that draws the figure.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figures as G

ev = G.load_evals()
groups = [("main_fno_%s_", "FNO"), ("base_unet_%s_", "U-Net (base)"),
          ("pmatch_unet_%s_", "U-Net (matched)"), ("base_don_%s_", "DeepONet")]

for vmax, tag in ((0.25, "0.25"), (1.00, "1.00")):
    print("=== vmax = %.2f ===" % vmax)
    key = "a025" if abs(vmax - 0.25) < 1e-9 else "a100"
    for pref, lab in groups:
        names = sorted(n for n in ev if n.startswith(pref % key))
        if not names:
            print("  %-16s NO DATA" % lab)
            continue
        F, M, S = G.mean_curve(ev, names, vmax)
        plateau = float(M[F >= 6].mean())
        inband = float(M[F <= 4].mean())
        params = ev[names[0]]["params"] / 1e6
        print("  %-16s plateau=%.3f  in-band=%.3f  params=%.2fM  seeds=%d"
              % (lab, plateau, inband, params, len(names)))
    print()
