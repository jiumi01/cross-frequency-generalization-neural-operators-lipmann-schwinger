# Author: Longgang Wang <longgangwang@163.com>
# Affiliation: Luoyang Institute of Science and Technology, Luoyang 471000, Henan, China

import numpy as np, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
a = np.load(os.path.join(C.DATA, "train_b1_2", "chunk_0000.npz"))
b = np.load(os.path.join(C.DATA, "train_b1_4", "chunk_0000.npz"))
c = np.load(os.path.join(C.DATA, "train_b1_12", "chunk_0000.npz"))
print("v   identical (b1_2 vs b1_4):", bool(np.allclose(a["v"], b["v"])))
print("vmax identical              :", bool(np.allclose(a["vmax"], b["vmax"])))
print("theta identical             :", bool(np.allclose(a["theta"], b["theta"])))
print("f   differs                 :", bool(not np.allclose(a["f"], b["f"])))
print("v   identical (b1_2 vs b1_12):", bool(np.allclose(a["v"], c["v"])))
print("f ranges: b1_2 [%.2f,%.2f]  b1_4 [%.2f,%.2f]  b1_12 [%.2f,%.2f]"
      % (a["f"].min(), a["f"].max(), b["f"].min(), b["f"].max(),
         c["f"].min(), c["f"].max()))
print("psi differs (b1_2 vs b1_4)  :", bool(not np.allclose(a["psi"], b["psi"])))
print("max res over caches:",
      max(float(np.max(np.load(os.path.join(C.DATA, d, f))["res"]))
          for d in sorted(os.listdir(C.DATA)) if d.startswith("train_")
          for f in os.listdir(os.path.join(C.DATA, d))))
