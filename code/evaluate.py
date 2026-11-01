# Author: Longgang Wang <longgangwang@163.com>
# Affiliation: Luoyang Institute of Science and Technology, Luoyang 471000, Henan, China

"""Evaluation of trained models across the (frequency x contrast) grid.

Test samples are PAIRED: for a fixed v_max arm the same (v, theta) set is reused
at every frequency, so every error-vs-frequency curve is a paired measurement in
which f is the only variable.  Physics baselines (truncated Born/Neumann) are
evaluated on exactly the same media.
"""
import os
import sys
import glob
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import physics as P
import data as D
import models as M


def build(n_grid, k, theta, v, device, emb="none", anchor="none", order=2,
          channels=None):
    """Build one evaluation cell.

    HEAD SEMANTICS (the fit-depth/sum-depth defect, fixed here): the analytic
    head must sum every term it was fitted against.  `channels` truncates the
    Born terms handed to the NETWORK as input channels only; the returned
    third value is the FULL term list, and anchored_u applies all `order` of
    them -- exactly what run_ba2.py documents ("the head still sums all seven
    terms -- the coefficients are applied outside the network").  Every
    a050/a100 arm evaluated before this fix summed only the first `channels`
    terms of an `order`-term fit, which is a DIFFERENT and much worse head
    (plateau 98 vs 0.39 at v=0.50 under the formal protocol, see
    results/ls_head_alone.json); the head-alone archives
    (results/ls_head_alone.json, results/anchor_freeze_check.json) record
    both readings under the names "truncated"/"shallow" vs "full".
    """
    psi_inc = P.incident_plane_wave(n_grid, k, theta, device).to(torch.complex64)
    vf = v.float()
    if anchor == "none":
        return D.build_inputs(vf, psi_inc, (k / (2 * np.pi)), emb=emb), psi_inc
    born = D.born_series(vf, k, theta, order=order)
    x = D.build_inputs(vf, psi_inc, (k / (2 * np.pi)), emb=emb,
                       anchor=anchor, born=born[:channels])
    return x, psi_inc, born


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ck["cfg"]
    name = cfg["model"]
    # The input channel count follows the run's own frequency embedding, so an
    # ff run loads without special-casing at every call site.
    # The analytic head's depth changes the input width, so it has to be read
    # back from the run's own config: defaulting it to 2 would rebuild an
    # order-8 head with four channels and score the wrong model.
    kw = dict(in_ch=D.in_channels(cfg.get("f_emb", "none"),
                                  cfg.get("anchor", "none"),
                                  cfg.get("anchor_order", 2),
                                  cfg.get("anchor_channels")), out_ch=2)
    if name in ("fno", "fno-film"):
        kw.update(width=cfg["width"], modes=cfg["modes"], n_layers=cfg["layers"],
                  proj=cfg["proj"], norm=cfg.get("norm", "none"))
    elif name == "deeponet":
        kw.update(width=cfg["width"], p=cfg["p"], n_grid=C.N)
    elif name == "unet":
        kw.update(base=cfg["base"], depth=3)
    model = M.build_model(name, **kw).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck


def parse_name(name):
    """Derive (arch, arm, seed, tag) from a run folder name."""
    toks = name.split("_")
    arch = "fno"
    if "unet" in toks:
        arch = "unet"
    elif "don" in toks or "deeponet" in toks:
        arch = "deeponet"
    arm = "mixed"
    for t in toks:
        if len(t) == 4 and t[0] == "a" and t[1:].isdigit():
            arm = t
    seed = 0
    for t in toks:
        if len(t) == 2 and t[0] == "s" and t[1].isdigit():
            seed = int(t[1:])
    tag = ""
    if name.startswith("band_"):
        arm, tag = "a025", toks[1]
    elif name.startswith("mode_"):
        arm, tag = "a025", toks[1]
    elif name.startswith("gamma_"):
        arm, tag = "a025", toks[1]
    elif name.startswith("gc_"):
        arm, tag = "a025", "gc"
    elif name.startswith("ff_"):
        arm, tag = "a025", "ff"
    elif name.startswith("ba_"):
        arm, tag = "a025", "ba"
    elif name.startswith("mixed_"):
        arm = "mixed"
    return arch, arm, seed, tag


@torch.no_grad()
def eval_cell(model, f, vmax, n=200, device="cuda", seed=50000, f_lo=None, f_hi=None,
              gamma=None, emb="none", anchor="none", ab=None, order=2,
              channels=None):
    gamma = C.GAMMA if gamma is None else gamma
    d = D.sample_batch(n, seed, device, f_lo=f_lo, f_hi=f_hi, vmax_lo=vmax,
                       vmax_hi=vmax, f_fixed=(None if f is None else f),
                       return_info=True, gamma=gamma)
    if anchor == "none":
        x, psi_inc = build(C.N, d["k"], d["theta"], d["v"], device, emb=emb)
    else:
        x, psi_inc, born = build(C.N, d["k"], d["theta"], d["v"], device,
                                 emb=emb, anchor=anchor, order=order,
                                 channels=channels)
    o = model(x)
    u = torch.complex(o[:, 0], o[:, 1])
    if anchor == "born":
        # the anchor is applied exactly as in training: same Born terms, same
        # coefficients, both shipped with the checkpoint
        u = D.anchored_u(u, born, ab.to(device))
    psi_hat = psi_inc + u
    e = (torch.linalg.norm((psi_hat - d["psi"]).reshape(n, -1), dim=1) /
         torch.linalg.norm(d["psi"].reshape(n, -1), dim=1))
    return e.cpu().numpy(), d


def born_cell(f, vmax, n=200, device="cuda", seed=50000, order=1, gamma=None):
    gamma = C.GAMMA if gamma is None else gamma
    # A per-frequency gamma (callable) resolves to a scalar here because the
    # Born cell is evaluated at one fixed f; born_approx takes a scalar.
    if callable(gamma):
        gamma = float(gamma(float(f)))
    d = D.sample_batch(n, seed, device, vmax_lo=vmax, vmax_hi=vmax, f_fixed=f,
                       gamma=gamma)
    pb = P.born_approx(d["v"], d["k"], d["theta"], gamma, C.N, order=order,
                       device=device)
    e = (torch.linalg.norm((pb - d["psi"]).reshape(n, -1), dim=1) /
         torch.linalg.norm(d["psi"].reshape(n, -1), dim=1))
    return e.cpu().numpy()


def evaluate_run(name, ckpt, arm, arch, seed, tag, n, gamma, device, out_dir):
    model, ck = load_model(ckpt, device)
    emb = ck["cfg"].get("f_emb", "none")
    anchor = ck["cfg"].get("anchor", "none")
    aorder = ck["cfg"].get("anchor_order", 2)
    achannels = ck["cfg"].get("anchor_channels")
    ab = ck.get("anchor_ab", None)
    if anchor == "born" and ab is None:
        raise SystemExit("%s: anchor=born but the checkpoint carries no "
                         "anchor_ab -- the analytic term would be silently "
                         "dropped and the arm scored as a bare network" % name)
    print("[%s] %s arm=%s epoch=%s val=%.4f f_emb=%s anchor=%s order=%d"
          % (name, arch, arm, ck.get("epoch"), ck.get("val"), emb, anchor,
             aorder), flush=True)
    # A callable gamma (per-frequency compensation) is recorded as a protocol
    # key, not as a number: _fresh compares this field as a string so that a
    # change of compensation table forces re-evaluation.
    gkey = ("compensated" if callable(gamma) else float(gamma))
    # Protocol key for the analytic head's semantics.  Archives scored before
    # the fit-depth/sum-depth fix lack this field; _fresh() treats an anchored
    # archive without "full" as stale, so no truncated-scored number can sit
    # beside a full-scored one in any table assembled from results/.
    res = dict(name=name, run=name, arch=arch, arm=arm, seed=seed, tag=tag,
               f_emb=emb, anchor=anchor, anchor_order=aorder,
               anchor_channels=achannels, head_semantics="full",
               params=int(ck["cfg"]["params"]), ckpt_epoch=ck.get("epoch"),
               ckpt_val=ck.get("val"), f_test=C.F_TEST, n=n, gamma=gkey,
               model=ck["cfg"]["model"], modes=ck["cfg"].get("modes"),
               train_data=ck["cfg"].get("data"))
    res["in_band"] = []
    # Per-sample errors are kept, not just their moments.  The design is paired
    # (the same media at every frequency and for every architecture), so the
    # only honest way to ask "is the U-Net's lower plateau significant?" is a
    # paired test on the identical samples -- which is impossible from means
    # and standard deviations alone (a two-sample test would throw the pairing
    # away and be needlessly conservative).  See code/paired.py.
    per_ib = np.zeros((len(C.VMAX_LIST), n), dtype=np.float32)
    for vi, vmax in enumerate(C.VMAX_LIST):
        e, _ = eval_cell(model, None, vmax, n=n, device=device, seed=70000 + vi,
                         f_lo=1.0, f_hi=4.0, gamma=gamma, emb=emb,
                         anchor=anchor, ab=ab, order=aorder, channels=achannels)
        per_ib[vi] = e
        res["in_band"].append(dict(vmax=vmax, mean=float(e.mean()),
                                   std=float(e.std()), median=float(np.median(e)),
                                   q90=float(np.quantile(e, 0.9))))
    sweep = {}
    per = np.zeros((len(C.VMAX_LIST), len(C.F_TEST), n), dtype=np.float32)
    for vi, vmax in enumerate(C.VMAX_LIST):
        rows = []
        for fi, f in enumerate(C.F_TEST):
            e, _ = eval_cell(model, f, vmax, n=n, device=device, seed=60000 + vi,
                             gamma=gamma, emb=emb, anchor=anchor, ab=ab,
                             order=aorder, channels=achannels)
            per[vi, fi] = e
            rows.append(dict(f=f, mean=float(e.mean()), std=float(e.std()),
                             sem=float(e.std() / np.sqrt(len(e))),
                             median=float(np.median(e)),
                             q90=float(np.quantile(e, 0.9))))
        sweep["%.2f" % vmax] = rows
    res["sweep"] = sweep
    ppath = os.path.join(out_dir, "percell_%s.npz" % name)
    np.savez_compressed(ppath, per=per, per_ib=per_ib,
                        vmax=np.array(C.VMAX_LIST, dtype=np.float64),
                        f_test=np.array(C.F_TEST, dtype=np.float64), n=n)
    res["percell"] = os.path.basename(ppath)
    for order in [1, 5]:
        bl = {}
        for vi, vmax in enumerate(C.VMAX_LIST):
            rows = []
            for f in C.F_TEST:
                e = born_cell(f, vmax, n=max(64, n // 2), device=device,
                              seed=60000 + vi, order=order, gamma=gamma)
                rows.append(dict(f=f, mean=float(np.nanmean(e))))
            bl["%.2f" % vmax] = rows
        res["born_order%d" % order] = bl
    out = os.path.join(out_dir, "eval_%s.json" % name)
    C.save_json(res, out)
    return out, res


def console(res):
    print("  mean relL2  (rows: v_max, cols: f)")
    print("  vmax  " + "".join("%7.1f" % f for f in C.F_TEST))
    for vmax in C.VMAX_LIST:
        print("  %.2f  " % vmax +
              "".join("%7.3f" % r["mean"] for r in res["sweep"]["%.2f" % vmax]))


def _fresh(path, n, gamma):
    """True iff an existing eval file was produced under the CURRENT protocol.

    Skipping on bare existence silently freezes an eval file against any later
    change to the test grid: adding a test frequency would leave every run
    scored on the old grid, and nothing downstream would notice because the
    file is present.  Compare the recorded grid (and sample count / gamma
    protocol key) instead, so a protocol change forces a re-evaluation.  The
    gamma field is a string protocol key: a float for constant-gamma runs, and
    "compensated" for the per-frequency arm, so the comparison is exact.
    """
    if not os.path.exists(path):
        return False
    try:
        d = C.load_json(path)
    except Exception:
        return False
    got = list(d.get("f_test") or [])
    if got != list(C.F_TEST):
        return False
    if int(d.get("n", -1)) != int(n):
        return False
    want = ("compensated" if callable(gamma) else float(gamma))
    g = d.get("gamma", None)
    if g is None:
        return False
    if isinstance(g, str) or isinstance(want, str):
        if str(g) != str(want):
            return False
    elif abs(float(g) - float(want)) > 1e-12:
        return False
    # An anchored archive scored before the fit-depth/sum-depth fix has no
    # head-semantics key; it was produced by a DIFFERENT operator (the head
    # summed only the first `channels` terms of an `order`-term fit), so it is
    # stale even though its grid and gamma match.
    if str(d.get("anchor")) == "born" and d.get("head_semantics") != "full":
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--pattern", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--gamma", type=float, default=None)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", default=C.RES)
    ap.add_argument("--shard", default=None,
                    help="i/N -- process every N-th run (parallel evaluation)")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    names = []
    if a.all:
        pat = a.pattern or os.path.join(C.RUNS, "*", "checkpoint.pt")
        for ck in sorted(glob.glob(pat)):
            nm = os.path.basename(os.path.dirname(ck))
            if nm.startswith("_") or nm.startswith("probe"):
                continue
            # A run is FINISHED iff result.json exists -- the same rule
            # run_grid.py uses.  train.py overwrites checkpoint.pt every epoch,
            # so without this guard an evaluator running alongside training
            # would silently score a half-trained model and, because this loop
            # skips names whose eval_*.json already exists, that wrong number
            # would then be treated as final for the rest of the campaign.
            if not os.path.exists(os.path.join(os.path.dirname(ck), "result.json")):
                print("skip (still training)", nm, flush=True)
                continue
            names.append((nm, ck))
    else:
        names.append((a.name or "custom", a.ckpt))

    # Sharding lets the (embarrassingly parallel, run-independent) evaluation
    # be spread over several processes.  Each run owns its own files, so the
    # shards never touch the same output; C.save_json is atomic regardless.
    if a.shard:
        si, sn = (int(x) for x in a.shard.split("/"))
        names = [t for k, t in enumerate(names) if k % sn == si]
        print("shard %d/%d -> %d runs" % (si, sn, len(names)), flush=True)

    for nm, ck in names:
        out = os.path.join(a.out, "eval_%s.json" % nm)
        # The freshness check needs this run's own gamma (the g02/g10 arms train
        # with a different lossy-part), so derive it BEFORE deciding to skip.
        # Resolve the default here rather than leaving it as the raw --gamma:
        # smuggling a None into the record makes _fresh() compare against None,
        # which is never equal to anything, so the "re-evaluate on protocol
        # change" guard silently degenerates into "re-evaluate unconditionally"
        # and burns a full evaluation pass on every rebuild.
        arch, arm, seed, tag = parse_name(nm)
        # The gamma-compensation arm is scored with the same per-frequency
        # gamma(f) it was trained on: the evaluation regenerates its own data
        # (eval_cell re-solves the LS system), so a fixed 0.05 here would score
        # the model on a different operator than it was trained on.
        if tag == "gc":
            import gamma_comp
            gamma = gamma_comp.comp_gamma
        else:
            gamma = (0.02 if tag == "g02" else
                     0.10 if tag == "g10" else
                     C.GAMMA if a.gamma is None else float(a.gamma))
        if _fresh(out, a.n, gamma):
            print("skip", nm, flush=True)
            continue
        if os.path.exists(out):
            print("re-eval (protocol changed)", nm, flush=True)
        _, res = evaluate_run(nm, ck, arm, arch, seed, tag, a.n, gamma, dev, a.out)
        console(res)


if __name__ == "__main__":
    main()
