# Author: Longgang Wang <longgangwang@163.com>
# Affiliation: Luoyang Institute of Science and Technology, Luoyang 471000, Henan, China

import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
p = sys.argv[1] if len(sys.argv) > 1 else "results/probe_fno2.json"
d = json.load(open(p, encoding="utf-8"))
print("in-band generalisation (held-out media, f~U[1,4]):")
for r in d["in_band"]:
    print("  vmax=%.2f mean=%.3f std=%.3f q90=%.3f" % (r["vmax"], r["mean"], r["std"], r["q90"]))
print()
print("mean/median by frequency, vmax=0.05 :")
for r in d["sweep"]["0.05"]:
    print("   f=%5.1f mean=%.3f median=%.3f q90=%.3f" % (r["f"], r["mean"], r["median"], r["q90"]))
print("mean/median by frequency, vmax=1.0 :")
for r in d["sweep"]["1.0"]:
    print("   f=%5.1f mean=%.3f median=%.3f q90=%.3f" % (r["f"], r["mean"], r["median"], r["q90"]))
for k in ("born_order1", "born_order5"):
    if k in d:
        print(k, "at f=1.5:",
              ["%.3f" % d[k][v][0]["mean"] for v in ["0.05", "0.1", "0.25", "0.5", "1.0"]])
        print(k, "at f=4.0:",
              ["%.3f" % d[k][v][3]["mean"] for v in ["0.05", "0.1", "0.25", "0.5", "1.0"]])
