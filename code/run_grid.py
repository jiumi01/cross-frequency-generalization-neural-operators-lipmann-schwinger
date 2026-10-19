# Author: Longgang Wang <longgangwang@163.com>
# Affiliation: Luoyang Institute of Science and Technology, Luoyang 471000, Henan, China

"""Resumable experiment grid driver.

Each spec runs train.py into runs/<name>/.  A run is considered done when
runs/<name>/result.json exists, so the driver can be interrupted and restarted
without redoing work.
"""
import os
import sys
import time
import json
import argparse
import subprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

PY = sys.executable

# --- common training flags --------------------------------------------------
BASE = ["--batch", "64", "--layers", "4"]
FNO = ["--model", "fno", "--width", "40", "--modes", "20", "--norm", "layer"]
UNET = ["--model", "unet", "--base", "32"]
DON = ["--model", "deeponet", "--width", "32", "--p", "128"]
EP = ["--epochs", "55", "--n-train", "12288"]


def spec(name, data, arch, seed=0, extra=None):
    return dict(name=name, data=os.path.join(C.DATA, data), arch=arch, seed=seed,
                extra=extra or [])


def grid():
    G = []
    # ---- A. main per-contrast arms (FNO) -----------------------------------
    # Three seeds for EVERY arm, so the error bars in Fig. 1 are the
    # between-seed spread throughout and the seeding protocol is uniform.
    for arm in ["a005", "a010", "a025", "a050", "a100"]:
        for s in [0, 1, 2]:
            G.append(spec("main_fno_%s_s%d" % (arm, s), "train_" + arm, FNO, s))
    # ---- B. architecture baselines -----------------------------------------
    for arm in ["a025", "a100"]:
        G.append(spec("base_unet_%s_s0" % arm, "train_" + arm, UNET, 0))
        G.append(spec("base_don_%s_s0" % arm, "train_" + arm, DON, 0))
    # ---- B1. parameter-MATCHED U-Net ---------------------------------------
    # The base-32 U-Net has 1.93 M parameters against the FNO's 10.25 M, i.e. it
    # is 5.3x smaller, so on its own it cannot carry a cross-architecture claim.
    # base=76 gives 10.86 M, the FNO's parameter count to within 6%.  If a
    # slightly *larger* convolutional model still fails to beat the FNO, the
    # "the binding constraint is the training band, not the representation
    # budget" reading is supported rather than merely asserted.
    for arm in ["a025", "a100"]:
        G.append(spec("pmatch_unet_%s_s0" % arm, "train_" + arm,
                      ["--model", "unet", "--base", "76"], 0))
    # ---- B2. capacity ablation: same recipe, 2.6x the parameters -----------
    # Without this the "capacity is not the bottleneck" claim would rest only on
    # the mode sweep (which barely moves the parameter count) and on comparing
    # across architectures, neither of which is a matched capacity ablation.
    for s in [0, 1]:
        G.append(spec("cap_w64_a025_s%d" % s, "train_a025",
                      ["--model", "fno", "--width", "64", "--modes", "20",
                       "--norm", "layer"], s))
    # ---- C. training-band-width ablation at v_max = 0.25 -------------------
    for tag, folder in [("b2", "train_b025_2"), ("b3", "train_b025_3"),
                        ("b6", "train_b025_6"), ("b12", "train_b025_12")]:
        G.append(spec("band_%s_s0" % tag, folder, FNO, 0))
    # ---- D. Fourier-mode budget sweep at v_max = 0.25 ----------------------
    # m=2 is the M < f_train case: it is what separates the training-band-limited
    # from the mode-budget-limited regime, i.e. it is the evidence for
    # f* = min(f_train, M).
    for m in [2, 4, 8, 12, 24, 32]:
        G.append(spec("mode_m%02d_s0" % m, "train_a025",
                      ["--model", "fno", "--width", "40", "--modes", str(m),
                       "--norm", "layer"], 0))
    # ---- E. mixed-contrast model (deployment-style) ------------------------
    G.append(spec("mixed_fno_s0", "train_b1_4", FNO, 0))
    # ---- F. propagator-loss robustness at v_max = 0.25 ---------------------
    G.append(spec("gamma_g02_s0", "train_g02", FNO, 0))
    G.append(spec("gamma_g10_s0", "train_g10", FNO, 0))
    return G


def extensions():
    """Reviewer-driven extensions.

    Three gaps a referee can see without leaving the paper's own logic:

      * the paper asserts "more training data does not help", but no spec in
        ``grid()`` ever varied ``--n-train``: every run takes the 12288 samples
        hard-wired in ``EP``.  The assertion therefore had no ablation behind it;
      * ``f* = min(f_train, M)`` rests on a single mode budget below f_train
        (M = 2), and that arm rises monotonically rather than stepping;
      * "widening the training band is the only intervention that moves the
        cliff" was verified on the FNO only, while the architecture table shows
        the U-Net does not even have a step.

    Everything is at v_max = 0.25 so the new arms extend the existing ones like
    for like.  Kept in a separate function, selected by ``--ext``, so the
    original grid stays byte-for-byte what it was.
    """
    G = []
    # (1) data-volume ladder at fixed band [1,4].  12288 already exists as
    #     main_fno_a025_s0/s1/s2, so these are the three lower rungs.
    for n in (1536, 3072, 6144):
        for s in (0, 1, 2):
            G.append(spec("ndata_n%05d_s%d" % (n, s), "train_a025", FNO, s,
                          ["--n-train", str(n)]))
    # (2) mode budgets below f_train = 4 (M = 2 already exists).
    for m in (1, 3):
        for s in (0, 1, 2):
            G.append(spec("mode_m%02d_s%d" % (m, s), "train_a025",
                          ["--model", "fno", "--width", "40", "--modes", str(m),
                           "--norm", "layer"], s))
    # (3) band-width ablation for a non-spectral family.  [1,4] already exists
    #     as base_unet_a025_s0/s1/s2; this adds the wider band [1,6].
    for s in (0, 1, 2):
        G.append(spec("uband_b6_s%d" % s, "train_b025_6", UNET, s))
    return G


def replicates():
    """Second and third seeds for every ablation / baseline arm.

    The seed protocol was uniform (3 seeds) only for the five main contrast
    arms; every ablation ran a single seed.  A referee is entitled to say that
    the bar heights in different figures then mean different things, and that a
    3-seed headline result is not like-for-like with a 1-seed ablation.  This
    pass closes that gap rather than documenting it as a caveat.  Runs that
    already exist are skipped by main(), so re-invoking it is idempotent.
    """
    G = []
    for s in (1, 2):
        for arm in ["a025", "a100"]:
            G.append(spec("base_unet_%s_s%d" % (arm, s), "train_" + arm,
                          UNET, s))
            G.append(spec("base_don_%s_s%d" % (arm, s), "train_" + arm,
                          DON, s))
            G.append(spec("pmatch_unet_%s_s%d" % (arm, s), "train_" + arm,
                          ["--model", "unet", "--base", "76"], s))
        G.append(spec("cap_w64_a025_s%d" % s, "train_a025",
                      ["--model", "fno", "--width", "64", "--modes", "20",
                       "--norm", "layer"], s))
        for tag, folder in [("b2", "train_b025_2"), ("b3", "train_b025_3"),
                            ("b6", "train_b025_6"), ("b12", "train_b025_12")]:
            G.append(spec("band_%s_s%d" % (tag, s), folder, FNO, s))
        for m in [2, 4, 8, 12, 24, 32]:
            G.append(spec("mode_m%02d_s%d" % (m, s), "train_a025",
                          ["--model", "fno", "--width", "40", "--modes",
                           str(m), "--norm", "layer"], s))
        G.append(spec("mixed_fno_s%d" % s, "train_b1_4", FNO, s))
        G.append(spec("gamma_g02_s%d" % s, "train_g02", FNO, s))
        G.append(spec("gamma_g10_s%d" % s, "train_g10", FNO, s))
    return G


# Per-spec wall-clock ceiling.  The heaviest arm is the width-matched U-Net
# (base=76, 10.86 M) which needs ~1700 s on an idle GPU; base=32 U-Net ~1460 s.
# A 3000 s ceiling was observed to kill a base=76 run once evaluation happened to
# be sharing the same GPU, which silently cost one of the three seeds.  The
# ceiling exists only to stop a wedged run, so it is set well clear of the
# uncontended cost and can be overridden per campaign.
PER_SPEC_TIMEOUT = float(os.environ.get("LS_SPEC_TIMEOUT", "7200"))


def all_specs():
    """The full spec universe: the main grid plus every replication seed."""
    seen = {}
    for s in grid() + replicates():
        seen[s["name"]] = s
    return list(seen.values())


def missing_specs():
    """Specs with no result.json, derived from the SPEC LIST rather than the disk.

    Enumerating ``runs/*/`` cannot see a spec whose directory was removed (for
    instance after a per-spec timeout), so a deleted arm is silently treated as
    "nothing missing".  One base=76 U-Net seed was lost exactly that way, and it
    is the one arm whose three-seed uniformity the replication pass exists to
    buy.  Deriving the missing set from the declared universe closes the class.
    """
    out = []
    for s in all_specs():
        rd = C.run_dir(s["name"], create=False)
        if not os.path.exists(os.path.join(rd, "result.json")):
            out.append(s["name"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=1e9)
    ap.add_argument("--only", default=None, help="comma separated substrings")
    ap.add_argument("--replicates", action="store_true",
                    help="run the extra seeds for the ablation arms instead")
    ap.add_argument("--all", action="store_true",
                    help="consider the union of the grid and the replication seeds")
    ap.add_argument("--ext", action="store_true",
                    help="run the reviewer-driven extensions instead")
    ap.add_argument("--list-missing", action="store_true",
                    help="print specs with no result.json and exit")
    a = ap.parse_args()
    if a.list_missing:
        print(" ".join(missing_specs()))
        return
    todo = (extensions() if a.ext else
            replicates() if a.replicates else
            (all_specs() if a.all else grid()))
    if a.only:
        keys = [k.strip() for k in a.only.split(",")]
        todo = [s for s in todo if any(k in s["name"] for k in keys)]
    log = C.Logger(os.path.join(C.RES, "grid_progress.log"))
    t0 = time.time()
    log("grid: %d specs to consider" % len(todo))
    n_done = 0
    for s in todo:
        rd = C.run_dir(s["name"], create=False)
        if os.path.exists(os.path.join(rd, "result.json")):
            n_done += 1
            continue
        if time.time() - t0 > a.budget:
            log("budget reached before %s" % s["name"])
            break
        cmd = [PY, os.path.join(os.path.dirname(os.path.abspath(__file__)), "train.py"),
               "--name", s["name"], "--data", s["data"], "--seed", str(s["seed"])] + \
              s["arch"] + BASE + EP + s["extra"]
        log("START %s" % s["name"])
        t1 = time.time()
        try:
            p = subprocess.run(cmd, capture_output=True, text=True,
                               timeout=PER_SPEC_TIMEOUT,
                               encoding="utf-8", errors="backslashreplace")
            ok = p.returncode == 0
        except Exception as e:  # pragma: no cover
            ok = False
            log("  EXC %s" % e)
        log("DONE  %s  ok=%s  %.0fs" % (s["name"], ok, time.time() - t1))
    log("grid pass finished (%d already done, %.0fs)" % (n_done, time.time() - t0))


if __name__ == "__main__":
    main()
