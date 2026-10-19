# Author: Longgang Wang <longgangwang@163.com>
# Affiliation: Luoyang Institute of Science and Technology, Luoyang 471000, Henan, China

"""Verification: which cells of Fig. 6 (mode/frequency heatmap) lack annotations?

Read-only diagnostic -- it recomputes the figure's data matrix with the project's
own figure code so the count comes from the same computation that draws the
figure, not from a recollection.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import figures as G

ev = G.load_evals()
modes = [2, 4, 8, 12, 20, 24]
F = C.F_TEST
Mat = np.full((len(modes), len(F)), np.nan)
for i, m in enumerate(modes):
    cand = ["main_fno_a025_s0"] if m == 20 else ["mode_m%02d_s0" % m]
    cand = [n for n in cand if n in ev]
    if not cand:
        print("M=%d NO DATA" % m)
        continue
    f_, m_, _ = G.curve(ev[cand[0]], 0.25)
    for j, fv in enumerate(F):
        k = np.where(np.isclose(f_, fv))[0]
        if k.size:
            Mat[i, j] = m_[k[0]]

print("f_test =", list(F))
rows_missing = 0
for i, m in enumerate(modes):
    row = Mat[i]
    miss = ["%.2f" % F[j] for j in range(len(F)) if np.isnan(row[j])]
    if miss:
        rows_missing += 1
    print("M=%2d  annotated=%2d/%-2d  missing at f=%s"
          % (m, int((~np.isnan(row)).sum()), len(F), miss or "-"))
print()
print("total NaN cells:", int(np.isnan(Mat).sum()))
print("rows with at least one missing annotation:", rows_missing)
