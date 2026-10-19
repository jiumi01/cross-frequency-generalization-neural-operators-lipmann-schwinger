# Author: Longgang Wang <longgangwang@163.com>
# Affiliation: Luoyang Institute of Science and Technology, Luoyang 471000, Henan, China

"""Shared utilities: seeding, logging, run directories, config IO."""
import os
import sys
import json
import time
import hashlib
import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(ROOT, "data")
RUNS = os.path.join(ROOT, "runs")
RES = os.environ.get("LS_RES") or os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
for _d in (DATA, RUNS, RES, FIG):
    os.makedirs(_d, exist_ok=True)

# ---- global physics configuration -----------------------------------------
N = 64                 # grid points per dimension
GAMMA = 0.05           # loss tangent of the propagator (kappa = k (1 + i gamma))
F_TRAIN = (1.0, 4.0)   # training frequency band (wavelengths per domain)
# Test frequencies.  The band edge at f = 4 is resolved to 0.25 because the
# headline claim is that the rise is a *step*, not a decay: with only the nodes
# 4.0 / 4.5 / 5.0 the width of the rise would be set by the grid rather than
# measured, and a referee is entitled to ask whether the step is real or an
# artefact of coarse sampling. 4.25 and 4.75 are the densification that answers
# that question directly.
F_TEST = [1.5, 2.5, 3.5, 4.0, 4.25, 4.5, 4.75, 5.0,
          6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
F_MAX = 12.0
VMAX_LIST = [0.05, 0.10, 0.25, 0.50, 1.00]
MM_REF = 0.12           # contrast correlation length (domain units)

# The architecture families compared in Table III and Fig. 4.  The prefix takes
# the contrast key (a025 / a100).  Defined in exactly ONE place on purpose: an
# earlier revision labelled the base U-Net "U-Net (small)" in the figure and
# "U-Net" in the table, and the figure caption then credited the lowest plateau
# to the wrong member of the pair -- the matched U-Net has the lowest plateau,
# the base one is the smaller model.  One source of truth, plus a gate that
# compares the emitted table and the manuscript text against it, makes that
# drift unreachable rather than merely fixed.
ARCH_GROUPS = [("main_fno_%s", "FNO"),
               ("base_unet_%s", "U-Net (base)"),
               ("pmatch_unet_%s", "U-Net (matched)"),
               ("base_don_%s", "DeepONet")]


def set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


class Logger(object):
    """ASCII-safe stdout + UTF-8 file log (Windows/conda safe)."""

    def __init__(self, path):
        self.path = path
        self.f = open(path, "a", encoding="utf-8")

    def __call__(self, msg):
        line = "[%s] %s" % (now(), msg)
        try:
            print(line.encode("ascii", "backslashreplace").decode("ascii"), flush=True)
        except Exception:
            pass
        self.f.write(line + "\n")
        self.f.flush()


def run_dir(name, create=True):
    d = os.path.join(RUNS, name)
    if create:
        os.makedirs(d, exist_ok=True)
    return d


def save_json(obj, path):
    tmp = path + ".part"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False, default=_default)
    os.replace(tmp, path)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().tolist()
    raise TypeError(str(type(o)))


def md5(path, blocksize=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(blocksize)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
