# Author: Longgang Wang <longgangwang@163.com>
# Affiliation: Luoyang Institute of Science and Technology, Luoyang 471000, Henan, China
"""Few-shot out-of-band fine-tuning (supplementary experiment 1).

The zero-shot protocol says out-of-band extrapolation is not purchasable by
capacity, data inside the band, encoding, loss or transfer.  The question
here is the deployment-relevant one: is it purchasable by a SMALL number of
out-of-band labels?  The baseline FNO (trained on f in [1,4]) is
fine-tuned on n_train in {64, 256, 1024} samples drawn from f in [5,8] at
v_max = 0.25, and scored zero-shot at f = 5, 8 (inside the fine-tune band
for the largest n_train) and f = 12 (still beyond it).  The resulting
sample-count vs out-of-band-error curve is the 'purchase price' of
extrapolation.
"""
import os
import sys
import subprocess
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import data as D
import evaluate as E

CACHE = os.path.join(C.DATA, "fewshot_a025_o58")
N_CACHE = 2048
F_FT = (5.0, 8.0)          # fine-tune band
F_EVAL = [5.0, 8.0, 12.0]
BASE = "main_fno_a025_s0"
N_TRAIN = [64, 256, 1024]


def ensure_cache():
    have = len([f for f in os.listdir(CACHE) if f.endswith(".npz")]) \
        if os.path.isdir(CACHE) else 0
    need = int(np.ceil(N_CACHE / 512))
    if have >= need:
        print("cache present (%d chunks)" % have, flush=True)
        return
    D.build_train_cache(N_CACHE, 512, seed0=4100, out_dir=CACHE, device="cuda",
                        f_lo=F_FT[0], f_hi=F_FT[1], vmax_lo=0.25, vmax_hi=0.25,
                        budget=240)
    print("cache built", flush=True)


def fine_tune(n_tr):
    name = "fs_o58_n%d_s0" % n_tr
    out = os.path.join(C.RUNS, name)
    if os.path.exists(os.path.join(out, "result.json")):
        print("already trained", name, flush=True)
        return name
    ck = torch.load(os.path.join(C.RUNS, BASE, "checkpoint.pt"),
                    map_location="cpu", weights_only=False)
    cfg = ck.get("cfg", {})
    cmd = [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "train.py"),
           "--name", name, "--model", "fno", "--data", CACHE,
           "--n-train", str(n_tr), "--epochs", "60", "--lr", "1e-4",
           "--width", str(cfg.get("width", 32)),
           "--modes", str(cfg.get("modes", 16)),
           "--layers", str(cfg.get("layers", 4)),
           "--proj", str(cfg.get("proj", 128)),
           "--norm", cfg.get("norm", "none"),
           "--init", os.path.join(C.RUNS, BASE, "checkpoint.pt"),
           "--seed", "0", "--notes", "few-shot oob f=[5,8] vmax=0.25"]
    print("running:", " ".join(cmd[2:]), flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("TRAIN FAIL", name, r.stderr[-2000:], flush=True)
        raise SystemExit(1)
    return name


@torch.no_grad()
def score(ckpt, f, n=200, device="cuda"):
    model, ck = E.load_model(ckpt, device)
    emb = ck["cfg"].get("f_emb", "none")
    anchor = ck["cfg"].get("anchor", "none")
    ab = ck.get("anchor_ab", None)
    e, _ = E.eval_cell(model, f, 0.25, n=n, device=device, seed=60000 + 2,
                       gamma=C.GAMMA, emb=emb, anchor=anchor, ab=ab)
    return float(e.mean())


def main():
    ensure_cache()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rows = {}
    ck_base = os.path.join(C.RUNS, BASE, "checkpoint.pt")
    rows["0"] = {f: score(ck_base, f, device=dev) for f in F_EVAL}
    print("baseline: " + "  ".join("f=%g:%.4f" % (f, v) for f, v in rows["0"].items()),
          flush=True)
    for n_tr in N_TRAIN:
        name = fine_tune(n_tr)
        ck = os.path.join(C.RUNS, name, "checkpoint.pt")
        rows[str(n_tr)] = {f: score(ck, f, device=dev) for f in F_EVAL}
        print("n_train=%d: " % n_tr + "  ".join(
            "f=%g:%.4f" % (f, v) for f, v in rows[str(n_tr)].items()), flush=True)
    res = dict(cache=CACHE, n_cache=N_CACHE, ft_band=list(F_FT),
               eval_f=F_EVAL, base=BASE, vmax=0.25, rows=rows)
    C.save_json(res, os.path.join(C.RES, "fewshot_purchase.json"))
    print("saved fewshot_purchase.json", flush=True)


if __name__ == "__main__":
    main()
