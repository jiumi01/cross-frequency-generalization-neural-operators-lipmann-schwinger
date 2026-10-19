# Author: Longgang Wang <longgangwang@163.com>
# Affiliation: Luoyang Institute of Science and Technology, Luoyang 471000, Henan, China

import numpy as np, glob, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
roots = sorted(glob.glob(os.path.join(C.DATA, "train_b*"))) + \
        sorted(glob.glob(os.path.join(C.DATA, "train_c*"))) + \
        sorted(glob.glob(os.path.join(C.DATA, "train_smoke")))
for d in roots:
    rs, fs, vs = [], [], []
    for p in sorted(glob.glob(os.path.join(d, "*.npz"))):
        z = np.load(p)
        rs.append(z["res"]); fs.append(z["f"]); vs.append(z["vmax"])
    if not rs:
        print("%-16s EMPTY" % os.path.basename(d)); continue
    r = np.concatenate(rs); f = np.concatenate(fs); v = np.concatenate(vs)
    m = (~np.isfinite(r)) | (r > 1e-6)
    print("%-16s n=%5d bad=%4d (%.2f%%)  f in [%.2f,%.2f]  vmax in [%.2f,%.2f]"
          % (os.path.basename(d), r.size, int(m.sum()), 100 * m.mean(), f.min(), f.max(),
             v.min(), v.max()))
    if m.sum():
        print("     bad f in [%.2f,%.2f]  bad vmax in [%.2f,%.2f]"
              % (f[m].min(), f[m].max(), v[m].min(), v[m].max()))
