# Author: Longgang Wang <longgangwang@163.com>
# Affiliation: Luoyang Institute of Science and Technology, Luoyang 471000, Henan, China

"""Paper figures (IEEE: vector PDF, Okabe-Ito, >=7pt)."""
import os
import re
import sys
import glob
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, NullFormatter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

OKABE = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00",
         "#CC79A7", "#999999"]
MK = ["o", "s", "^", "v", "D", "P", "X", "*"]


def set_style(width="single"):
    # The A4 single-column block is 21 - 2*2.2 = 16.6 cm = 6.53 in wide, so a
    # figure declared "wide" must not exceed that or \includegraphics[width=
    # \textwidth] silently scales the type down to ~7 pt and the labels stop
    # being legible at print size.  A half-width single panel stays at 3.5 in.
    w = 3.5 if width == "single" else 6.5
    plt.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9, "axes.labelsize": 10, "axes.titlesize": 10,
        "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.linewidth": 0.8, "lines.linewidth": 1.5, "lines.markersize": 4.5,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "xtick.major.size": 3.5, "ytick.major.size": 3.5,
        "legend.frameon": False, "savefig.dpi": 600, "figure.dpi": 150,
    })
    return w


def save(fig, path):
    fig.savefig(path + ".pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(path + ".png", dpi=400, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    # Guard against a degenerate axis (e.g. a log scale fed an exact zero):
    # matplotlib happily emits a PDF thousands of points tall, and the failure
    # only surfaces much later as "Dimension too large" inside LaTeX.
    warn = ""
    try:
        raw = open(path + ".pdf", "rb").read()
        m = re.search(rb"/MediaBox\s*\[\s*([-\d.]+)\s+([-\d.]+)\s+"
                      rb"([-\d.]+)\s+([-\d.]+)", raw)
        if m:
            wp = float(m.group(3)) - float(m.group(1))
            hp = float(m.group(4)) - float(m.group(2))
            if wp > 1000 or hp > 1000:
                warn = "   !! SUSPECT page size %.0f x %.0f pt" % (wp, hp)
    except Exception:
        pass
    print("wrote", path + warn)


def panel_label(ax, letter, text="", dy=-16, fontsize=8.5, weight="bold"):
    """Put the sub-figure identifier (a)(b)... BELOW the panel (scientific
    convention) instead of as a top title, so it never sits on top of the data
    curves.  When the panel carries an x-axis label, the identifier is sunk
    BELOW the whole xlabel band (tick labels + label text, ~14 pt extra), so
    the two never share one horizontal line.  The decision is re-evaluated at
    DRAW time, so the call order of panel_label()/set_xlabel() does not matter.
    """
    s = "(%s)" % letter if not text else "(%s) %s" % (letter, text)
    ann = ax.annotate(s, xy=(0.5, 0.0), xycoords="axes fraction",
                      xytext=(0, dy), textcoords="offset points",
                      ha="center", va="top", fontsize=fontsize,
                      fontweight=weight)

    def _sink(_evt=None):
        off = dy - 14 if ax.get_xlabel().strip() else dy
        ann.set_position((0, off))

    _sink()
    ax.figure.canvas.mpl_connect("draw_event", _sink)
    return ann


def figure_sizes(outs):
    """Return {basename: (w_pt, h_pt)} for every PDF in the given folders."""
    out = {}
    for d in outs:
        for p in glob.glob(os.path.join(d, "*.pdf")):
            raw = open(p, "rb").read()
            m = re.search(rb"/MediaBox\s*\[\s*([-\d.]+)\s+([-\d.]+)\s+"
                          rb"([-\d.]+)\s+([-\d.]+)", raw)
            if m:
                out[os.path.basename(p)] = (float(m.group(3)) - float(m.group(1)),
                                            float(m.group(4)) - float(m.group(2)))
    return out


def load_evals(res_dir=None):
    res_dir = res_dir or C.RES
    out = {}
    for p in sorted(glob.glob(os.path.join(res_dir, "eval_*.json"))):
        d = json.load(open(p, encoding="utf-8"))
        out[d["name"]] = d
    return out


def curve(d, vmax):
    rows = d["sweep"]["%.2f" % vmax]
    return (np.array([r["f"] for r in rows]), np.array([r["mean"] for r in rows]),
            np.array([r["sem"] for r in rows]))


def mean_curve(evals, names, vmax):
    """Average curves over replicate runs (seeds)."""
    fs, ms, ss = [], [], []
    for n in names:
        f, m, s = curve(evals[n], vmax)
        fs.append(f); ms.append(m); ss.append(s)
    F = fs[0]
    M = np.mean(np.stack(ms), axis=0)
    S = np.std(np.stack(ms), axis=0, ddof=0) if len(ms) > 1 else np.stack(ss[0:1])[0]
    return F, M, S


# ---------------------------------------------------------------------------
def fig_main(evals, out):
    """Error vs frequency, one panel per contrast arm (FNO).

    The y axis is shared.  The point of the strip is that the cliff sits at the
    same frequency for every contrast while the plateau height moves by a factor
    of nine, and a separate axis per panel would rescale that difference away.
    """
    w = set_style("double")
    arms = C.VMAX_LIST
    fig, axes = plt.subplots(1, len(arms), figsize=(w, w * 0.30), sharey=True)
    for k, (ax, vmax) in enumerate(zip(axes, arms)):
        tag = "a%03d" % int(round(vmax * 100))
        names = [n for n in evals if n.startswith("main_fno_%s_" % tag)]
        if not names:
            continue
        F, M, S = mean_curve(evals, sorted(names), vmax)
        ax.errorbar(F, M, yerr=S, color=OKABE[4], marker=MK[0], capsize=2,
                    label="FNO", clip_on=False)
        b1 = np.array([r["mean"] for r in evals[sorted(names)[0]]["born_order1"]["%.2f" % vmax]])
        ax.plot(F, b1, color=OKABE[1], marker=MK[1], ls="--", lw=1.2,
                label="Born $n{=}1$")
        ax.axvspan(1.0, 4.0, color="#888888", alpha=0.13, lw=0)
        ax.axvline(4.0, color="k", lw=0.8, ls=":")
        ax.set_yscale("log")
        ax.set_ylim(0.008, 4.5)
        # Narrow five-panel strip: matplotlib's default log locator labels every
        # 1..9 mantissa, and in a panel this narrow the labels collide.  Decade
        # ticks only.
        ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=3))
        ax.yaxis.set_minor_formatter(NullFormatter())
        panel_label(ax, "abcde"[k], r"$v_{\max}=%.2f$" % vmax)
        ax.set_xlabel("frequency $f$")
        ax.set_xticks([2, 4, 6, 8, 10, 12])
        ax.set_xlim(1, 12.2)
        ax.spines[["top", "right"]].set_visible(False)
        if k == 0:
            ax.set_ylabel(r"relative $L_2$ error")
        else:
            # Setting the major locator on a shared axis re-enables its tick
            # labels; they are turned off again here so the strip keeps a
            # single y scale label on the left.
            ax.tick_params(labelleft=False)
    # One legend above the strip: inside panel 0 it collides with the Born
    # curve, which is exactly the thing panel 0 is there to show.
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=len(handles),
                   frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.005))
    save(fig, os.path.join(out, "fig_main_cliff"))


def fig_arch(evals, out):
    """Architecture comparison at both contrast levels, plus the normalised view.

    Three panels rather than two.  The first two are the raw error curves.  The
    third divides each curve by its OWN plateau, which is the only reading that
    separates two claims the raw panels cannot separate: "this family reaches a
    lower plateau" and "this family does not step".  A family with a lower
    plateau looks flat on the raw panel even if it stepped, so the plateau
    height alone cannot establish either claim.  Normalised, the reference value
    is 1 by construction, so a family that STEPS rises past 1 and a family that
    decays smoothly never reaches it -- and that is a statement about shape that
    survives the plateau being lower.
    """
    # Labels come from common.ARCH_GROUPS, the same single definition that
    # fills Table III, so the legend and the table rows cannot disagree.
    # The two U-Nets used to be orange and vermilion, which is a pair a reader
    # cannot separate at 8 pt; base and matched are now orange and green.
    styles = [(OKABE[4], MK[0], "-"), (OKABE[1], MK[1], "-"),
              (OKABE[3], MK[2], "-"), (OKABE[6], MK[3], "-")]
    series = [(p % "a025", p % "a100", lab, col, mk, ls)
              for (p, lab), (col, mk, ls) in zip(C.ARCH_GROUPS, styles)]
    w = set_style("double")
    fig, axes = plt.subplots(1, 3, figsize=(w, w * 0.30), sharey=False)
    for k, (ax, vmax) in enumerate(zip(axes[:2], (0.25, 1.00))):
        for pref0, pref1, lab, col, mk, ls in series:
            pref = pref0 if abs(vmax - 0.25) < 1e-9 else pref1
            names = sorted(n for n in evals if n.startswith(pref))
            if not names:
                continue
            F, M, S = mean_curve(evals, names, vmax)
            ax.errorbar(F, M, yerr=S, color=col, marker=mk, ls=ls, capsize=2,
                        label=lab, clip_on=False)
        ax.axvspan(1, 4, color="#888888", alpha=0.13, lw=0)
        ax.axvline(4, color="k", lw=0.8, ls=":")
        ax.set_yscale("log")
        # Same y range as Fig. 2, so the plateau heights of the two figures can
        # be read against each other without a mental rescaling.  The lower
        # limit catches the 1e-3 plateaus; the upper one catches the Born curve
        # in the strong-contrast panel, which is the topmost point drawn here.
        ax.set_ylim(0.008, 4.5)
        ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=3))
        ax.yaxis.set_minor_formatter(NullFormatter())
        panel_label(ax, "ab"[k], r"$v_{\max}=%.2f$" % vmax)
        ax.set_xlabel("frequency $f$")
        ax.set_xticks([2, 4, 6, 8, 10, 12])
        ax.set_xlim(1, 12.2)
        ax.spines[["top", "right"]].set_visible(False)
        if k == 0:
            ax.set_ylabel(r"relative $L_2$ error")
        else:
            # sharey hides the labels by construction, but re-setting the major
            # locator per axis brings them back; turn them off explicitly.
            ax.tick_params(labelleft=False)

    # ---- panel (c): each curve divided by its own plateau ------------------
    axn = axes[2]
    for pref0, pref1, lab, col, mk, ls in series:
        names = sorted(n for n in evals if n.startswith(pref0))
        if not names:
            continue
        F, M, _ = mean_curve(evals, names, 0.25)
        pl = M[F >= 6.0]
        if not pl.size or pl.mean() <= 0:
            continue
        axn.plot(F, M / pl.mean(), color=col, marker=mk, ls=ls, ms=3,
                 lw=1.2, label=lab, clip_on=False)
    axn.axhline(1.0, color="k", lw=0.8, ls=":")
    axn.axvspan(1, 4, color="#888888", alpha=0.13, lw=0)
    axn.axvline(4, color="k", lw=0.8, ls=":")
    axn.set_ylim(0.0, 1.7)
    panel_label(axn, "c", r"$v_{\max}=0.25$, own plateau")
    axn.set_xlabel("frequency $f$")
    axn.set_ylabel(r"$\varepsilon(f)\,/\,\varepsilon_{\infty}$")
    axn.set_xticks([2, 4, 6, 8, 10, 12])
    axn.set_xlim(1, 12.2)
    axn.spines[["top", "right"]].set_visible(False)
    # Four long family names do not fit inside a 3.25 in panel, so the legend
    # goes under the strip, where it also cannot be mistaken for a curve label.
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=len(handles),
                   frameon=False, fontsize=8, handlelength=2.0,
                   columnspacing=1.6, bbox_to_anchor=(0.5, 1.02))
    save(fig, os.path.join(out, "fig_arch"))


def fig_band(evals, out):
    w = set_style("single")
    fig, ax = plt.subplots(figsize=(w, w * 0.72))
    specs = [("A025", "band_a025", 4.0, OKABE[0]), ("A025_b2", "band_b2", 2.0, OKABE[1]),
             ("A025_b3", "band_b3", 3.0, OKABE[2]), ("A025_b6", "band_b6", 6.0, OKABE[3]),
             ("A025_b12", "band_b12", 12.0, OKABE[5])]
    for lab, pref, fedge, col in specs:
        if pref == "band_a025":
            names = sorted(n for n in evals if n.startswith("main_fno_a025"))
        else:
            names = sorted(n for n in evals if n.startswith(pref))
        if not names:
            continue
        F, M, S = mean_curve(evals, names, 0.25)
        ax.errorbar(F, M, yerr=S, color=col, marker=MK[0], capsize=2,
                    label=r"train band $[1,%.0f]$" % fedge, clip_on=False)
        # Each arm has its own training edge, so a single shared vertical line
        # would mark only one of the five.  Marking the edge with a ring on the
        # arm's own curve locates it without the five full-height dotted rules
        # that used to cut the panel into six stripes.
        k = int(np.argmin(np.abs(F - fedge)))
        ax.plot([F[k]], [M[k]], marker="o", ms=9.5, mfc="none", mec=col,
                mew=1.3, clip_on=False)
    ax.set_yscale("log")
    ax.set_ylim(0.008, 4.5)
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=3))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("frequency $f$")
    ax.set_ylabel(r"relative $L_2$ error")
    ax.set_xticks([2, 4, 6, 8, 10, 12])
    ax.set_xlim(1, 12.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.16),
              handlelength=1.6, columnspacing=1.2, labelspacing=0.3)
    ax.text(0.985, 0.03, "ring: train band edge", transform=ax.transAxes,
            fontsize=6.5, ha="right", va="bottom", color="#444444")
    save(fig, os.path.join(out, "fig_bandwidth"))


def fig_mode_phase(evals, out, band_tag, train_edge, fname):
    """Heatmap of test error over (mode budget M, frequency f)."""
    w = set_style("single")
    # M = 32 is deliberately excluded: on the 64x64 grid it spans every
    # representable mode, so its band-limitation bound is vacuous.
    modes = [2, 4, 8, 12, 20, 24]
    F = C.F_TEST
    Mat = np.full((len(modes), len(F)), np.nan)
    for i, m in enumerate(modes):
        # Every budget is averaged over the same three seeds.  The M = 20 row is
        # the main contrast sweep and the others are the mode sweep, but they are
        # trained under one protocol; taking seed 0 for one row and three seeds
        # for another put two different estimators inside a single heat map.
        pref = "main_fno_a025_" if m == 20 else "mode_m%02d_" % m
        names = sorted(n for n in evals if n.startswith(pref))
        if not names:
            continue
        f_, m_, _ = mean_curve(evals, names, 0.25)
        for j, fv in enumerate(F):
            k = np.where(np.isclose(f_, fv))[0]
            if k.size:
                Mat[i, j] = m_[k[0]]
    if np.all(np.isnan(Mat)):
        print("no mode sweep data"); return
    fig, ax = plt.subplots(figsize=(w, w * 0.72))
    vmax_c = np.nanmax(Mat)
    im = ax.imshow(Mat, cmap="viridis", aspect="auto", origin="lower",
                   vmin=np.nanmin(Mat), vmax=vmax_c)
    ax.set_xticks(range(len(F)))
    ax.set_xticklabels(["%.1f" % f for f in F], rotation=60, ha="right", fontsize=6.5)
    ax.set_yticks(range(len(modes)))
    ax.set_yticklabels([str(m) for m in modes])
    ax.set_xlabel("frequency $f$"), ax.set_ylabel("retained modes $M$")
    xb = np.searchsorted(F, train_edge - 1e-9)
    ax.axvline(xb - 0.5, color="w", lw=1.2, ls="--")
    # Above the axes, not inside: rotated inside, the label sat on top of the
    # cell annotations it is there to help read.
    ax.annotate("training edge", xy=(xb - 0.5, 1.0),
                xycoords=("data", "axes fraction"),
                xytext=(0, 3), textcoords="offset points",
                ha="center", va="bottom", fontsize=6.5, color="#333333",
                annotation_clip=False)
    for i, m in enumerate(modes):
        for j in range(len(F)):
            if np.isnan(Mat[i, j]):
                continue
            ax.text(j, i, "%.2f" % Mat[i, j], ha="center", va="center", fontsize=5.6,
                    color="k" if Mat[i, j] > 0.5 * vmax_c else "w")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.outline.set_linewidth(0.5)
    cb.ax.tick_params(labelsize=7)
    save(fig, os.path.join(out, fname))


def fig_scaling(evals, out):
    """Where the rise starts, and how far past the plateau it goes.

    Dividing each arm by its own plateau removes the contrast dependence of the
    height by construction, so the collapse that survives is a statement about
    the *onset*: the rise begins at the same relative frequency for every
    contrast.  It is deliberately not drawn as a collapse onto one step -- the
    high-contrast arms overshoot their plateau at f/f_train ~ 1.25 and relax back
    by ~1.75, and an axis that stops just above 1.0 (as this one used to) renders
    the overshoot outside the frame, where it reads as noise rather than as the
    non-monotonicity it is.
    """
    w = set_style("single")
    fig, ax = plt.subplots(figsize=(w, w * 0.76))
    stats = {}
    for i, vmax in enumerate(C.VMAX_LIST):
        tag = "a%03d" % int(round(vmax * 100))
        names = sorted(n for n in evals if n.startswith("main_fno_%s_" % tag))
        if not names:
            continue
        F, M, S = mean_curve(evals, names, vmax)
        plateau = float(M[F >= 6].mean())
        e4 = float(M[np.isclose(F, 4.0)][0])
        m = F >= 2.0
        top = float(np.max(M[m] / plateau))
        ax.plot(F[m] / 4.0, M[m] / plateau, color=OKABE[i], marker=MK[i],
                ms=3.2, lw=1.4, label=r"$v_{\max}=%.2f$" % vmax, clip_on=False)
        stats[vmax] = dict(edge_ratio=e4 / plateau, plateau=plateau,
                           overshoot=top)
    ax.axvspan(0.25, 1.0, color="#888888", alpha=0.13, lw=0)
    ax.axvline(1.0, color="k", lw=0.8, ls=":")
    ax.axhline(1.0, color="#666666", lw=0.8, ls="-", zorder=0)
    ax.set_xscale("log")
    ax.set_xlim(0.4, 3.2)
    # Headroom for the overshoot.  The previous cap of 1.25 sat below the peak of
    # three of the five arms.
    ax.set_ylim(0.0, 1.7)
    ax.set_xlabel(r"relative frequency $f/f_{\mathrm{train}}$")
    ax.set_ylabel(r"$\varepsilon(f)/\varepsilon_{\infty}$")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=6.5, ncol=2, loc="lower right", handlelength=1.5,
              columnspacing=0.9, labelspacing=0.25)
    save(fig, os.path.join(out, "fig_scaling"))
    return stats


def fig_spectrum(out):
    p = os.path.join(C.RES, "spectrum.json")
    if not os.path.exists(p):
        print("no spectrum.json"); return
    d = json.load(open(p, encoding="utf-8"))
    modes = d["modes"]; F = d["f"]
    w = set_style("single")
    fig, ax = plt.subplots(figsize=(w, w * 0.72))
    # M = 32 is excluded for the same reason as in fig_mode_phase: on the 64x64
    # grid it spans every representable mode, so delta_32 == 0 identically and a
    # log axis has nothing to display.  The floor below is a rendering guard
    # only -- it never bites for a non-vacuous budget.
    shown = [2, 4, 8, 12, 20, 24]
    for i, M in enumerate(shown):
        y = np.maximum(np.array([d["data"]["0.25"]["f=%g" % f]["psi"][str(M)]
                                 for f in F], dtype=float), 1e-6)
        ax.semilogy(F, y, color=OKABE[i], marker=MK[i], label=r"$M=%d$" % M,
                    clip_on=False)
    ax.set_xlabel("frequency $f$")
    ax.set_ylabel("out-of-band energy fraction $\\delta_M$")
    ax.axvspan(1.0, 4.0, color="#888888", alpha=0.13, lw=0)
    ax.axvline(4.0, color="k", lw=0.8, ls=":")
    ax.set_xticks([2, 4, 6, 8, 10, 12])
    ax.set_xlim(1, 12.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.17),
              handlelength=1.6, columnspacing=1.1, labelspacing=0.3)
    save(fig, os.path.join(out, "fig_spectrum"))


def fig_fields(out, device=None):
    """Reference, prediction, and the two corrections, in and out of band.

    The bottom two rows exist to decide a mechanism rather than to illustrate
    one.  A plateau at the incident-field baseline can come from a correction
    that has collapsed toward zero or from one that is present but
    phase-incoherent; comparing |u_hat| against |u| on a SHARED colour scale
    separates them by eye, which the pointwise error panel it replaced could not.
    """
    import torch
    import physics as P
    import data as D
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = os.path.join(C.RUNS, "main_fno_a025_s0", "checkpoint.pt")
    if not os.path.exists(ck):
        print("no ckpt for fields"); return
    from evaluate import load_model
    model, _ = load_model(ck, device)
    w = set_style("double")
    freqs = [1.5, 4.0, 6.0, 12.0]
    n = 4

    # Pass 1: the fields themselves.  Ground truth and prediction are the same
    # physical quantity and must therefore share one colour scale; giving each
    # panel its own scale would make a field that is uniformly wrong look like
    # a field that is uniformly right.
    got = []
    for f in freqs:
        d = D.sample_batch(n, 60000, device, vmax_lo=0.25, vmax_hi=0.25, f_fixed=f,
                           return_info=True)
        psi = d["psi"][0].cpu().numpy()
        x, psi_inc = build_in(C.N, d["k"], d["theta"], d["v"], device)
        with torch.no_grad():
            o = model(x)
        pred = (psi_inc + torch.complex(o[:, 0], o[:, 1]))[0]
        inc = psi_inc[0].cpu().numpy()
        ph = pred.cpu().numpy()
        uh = (pred - psi_inc[0]).cpu().numpy()
        u = psi - inc
        nu = float(np.linalg.norm(u))
        got.append((f, psi, ph, uh, u, inc,
                    float(np.linalg.norm(ph - psi) / np.linalg.norm(psi)),
                    (float(np.linalg.norm(uh)) / nu) if nu > 0 else np.nan))

    v_field = max(float(np.abs(p).max()) for _, p, _, _, _, _, _, _ in got)
    # Shared scale for the two correction rows, so a small |u_hat| cannot be
    # rescaled into looking like a match.
    v_corr = max(max(float(np.abs(g[3]).max()), float(np.abs(g[4]).max()))
                 for g in got)

    # Four square rows over the full text width.  The panel aspect is fixed by
    # imshow, so the figure has to be tall enough for four of them; the previous
    # three-row height would have shrunk the panels until the labels were
    # unreadable at print size.
    fig, axes = plt.subplots(4, len(freqs), figsize=(w, w * 0.80))
    # Tighten the gaps between the four frequency columns and the four rows:
    # the earlier default spacing left the middle sub-plots too far apart.
    fig.subplots_adjust(wspace=0.05, hspace=0.05, right=0.86)
    rows = [("reference $\\psi$", "RdBu_r", v_field, -v_field, lambda g: g[1].real),
            ("FNO $\\hat\\psi$", "RdBu_r", v_field, -v_field, lambda g: g[2].real),
            ("pred. corr. $|\\hat u|$", "magma", v_corr, 0.0,
             lambda g: np.abs(g[3])),
            ("true corr. $|u|$", "magma", v_corr, 0.0, lambda g: np.abs(g[4]))]
    for j, g in enumerate(got):
        f = g[0]
        band = "in band" if f <= 4.0 else "out of band"
        for r, (ttl, cmap, vmax_c, vmin_c, pick) in enumerate(rows):
            ax = axes[r, j]
            ax.imshow(pick(g), cmap=cmap, vmin=vmin_c, vmax=vmax_c, origin="lower")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_linewidth(0.6)
            if r == 0:
                ax.set_title(r"$f=%.1f$ (%s)" % (f, band), fontsize=8)
            if j == 0:
                ax.set_ylabel(ttl, fontsize=8)
            # Per-panel metric stamp in the top-right corner, the way the
            # full-wave EM paper stamps its per-panel MSE.  Putting the two
            # decisive numbers in the panel instead of a caption line under it
            # keeps the eye on the mechanism: the L2 on the prediction row,
            # the correction energy ratio on the predicted-correction row.
            # The translucent plate keeps the stamp legible over both the
            # dark and the light ends of every colour map.
            if r == 1:
                stamp = "rel. $L_2=%.3f$" % g[6]
            elif r == 2:
                stamp = r"$\|\hat u\|/\|u\|=%.2f$" % g[7]
            else:
                stamp = None
            if stamp:
                ax.text(0.97, 0.97, stamp, transform=ax.transAxes,
                        ha="right", va="top", fontsize=6.5,
                        bbox=dict(facecolor="white", alpha=0.75,
                                  edgecolor="none", pad=1.2))
    # One colourbar per physical quantity, not per panel and not per row: the
    # two field rows are the same quantity on one scale, the two correction
    # rows are the same quantity on one scale, and the two pairs are different
    # quantities.  Four bars (one per row) made the panel strip look like a
    # dashboard; two bars grouped by quantity read as a legend.
    for rr, lab in ((slice(0, 2), r"Re $\psi$"), (slice(2, 4), r"$|u|$")):
        cb = fig.colorbar(axes[rr.stop - 1, -1].images[0],
                          ax=list(axes[rr, :].ravel()),
                          fraction=0.030, pad=0.012)
        cb.outline.set_linewidth(0.5)
        cb.ax.tick_params(labelsize=7)
        cb.set_label(lab, fontsize=7.5, labelpad=2)
    save(fig, os.path.join(out, "fig_fields"))


def fig_bound(evals, out):
    """Measured error against two reference scales, and why only one applies.

    The theorem lower-bounds the error of a predictor whose *output* is
    band-limited to M modes.  No model benchmarked here satisfies that premise:
    each returns psi_inc + u_theta with psi_inc supplied in full, and the FNO
    additionally carries a pointwise branch in every layer, so its output is not
    band-limited either.  The dashed curves are therefore a property of the data
    -- the out-of-band energy fraction of the reference solution -- and not a
    bound on the measured error.

    Panel (b) plots the quantity the mode budget does constrain: the
    out-of-band energy of the *correction*, relative to the total field.  A
    predictor of the form psi_inc + P_M u_theta can never beat it, because
    (P_M u_theta - P_M u) is in band and (P_M u - u) is out of band, so the two
    are orthogonal and the residual is at least as large as the second.
    """
    p = os.path.join(C.RES, "spectrum.json")
    if not os.path.exists(p):
        print("no spectrum.json"); return
    d = json.load(open(p, encoding="utf-8"))
    F = np.array(d["f"], float)
    spec = d["data"]["0.25"]
    rows = [spec["f=%g" % f] for f in F]
    have_u = "u_out_rel" in rows[0]
    cols = ["#666666", OKABE[1], OKABE[2], OKABE[3], OKABE[4], OKABE[5]]
    budgets = [2, 4, 8, 12, 20, 24]
    w = set_style("double")
    fig, axes = plt.subplots(1, 2, figsize=(w, w * 0.40), sharex=True)
    any_ = False
    for col, Mb in zip(cols, budgets):
        key = str(Mb)
        # Three seeds for every budget, the M = 20 row included: the main
        # contrast sweep and the mode sweep are trained under one protocol, so
        # mixing seed 0 with a three-seed mean inside one figure is not
        # defensible.
        pref = "main_fno_a025_" if Mb == 20 else "mode_m%02d_" % Mb
        names = sorted(n for n in evals if n.startswith(pref))
        bnd = np.array([r["psi_sqrt_mean"][key] for r in rows], float)
        axes[0].plot(F, bnd, color=col, ls="--", lw=1.1, clip_on=False)
        if have_u:
            bu = np.array([r["u_out_rel"][key] for r in rows], float)
            axes[1].plot(F, bu, color=col, ls=":", lw=1.5, clip_on=False)
        if names:
            Fv, Mv, Sv = mean_curve(evals, names, 0.25)
            for ax in axes:
                ax.errorbar(Fv, Mv, yerr=Sv, color=col, marker=MK[0], ls="-",
                            ms=2.8, lw=1.0, capsize=1.2, clip_on=False)
            any_ = True
    for ax in axes:
        ax.axvspan(1.0, 4.0, color="#888888", alpha=0.13, lw=0)
        ax.axvline(4.0, color="k", lw=0.8, ls=":")
        ax.set_yscale("log")
        ax.set_xlim(1, 12.3)
        ax.set_xticks([2, 4, 6, 8, 10, 12])
        ax.set_xlabel("frequency $f$")
        ax.set_ylabel(r"relative $L_2$ error", fontsize=8.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=4))
        ax.yaxis.set_minor_formatter(NullFormatter())
    axes[0].set_ylim(0.03, 5.0)
    panel_label(axes[0], "a", fontsize=8)
    axes[1].set_ylim(1e-5, 3.0)
    panel_label(axes[1], "b", fontsize=8)
    if any_:
        h, l = [], []
        for col, Mb in zip(cols, budgets):
            h.append(Line2D([], [], color=col, marker=MK[0], ls="-", lw=1.0,
                            ms=2.8))
            l.append(r"$M=%d$" % Mb)
        h.append(Line2D([], [], color="k", ls="-", lw=1.3))
        l.append("measured error")
        # Seven entries in one row.  Legend entries for the dashed and dotted
        # reference curves are deliberately absent: each belongs to one panel
        # only, so it is named in that panel's title instead of in a legend that
        # would imply it appears in both.
        # The family/reference legend sits at the TOP: the (a)/(b) descriptors
        # now live below the panels, so a bottom legend would collide with them.
        fig.legend(h, l, fontsize=7, ncol=7, loc="upper center",
                   bbox_to_anchor=(0.5, 1.02), handlelength=1.6,
                   columnspacing=1.0, labelspacing=0.3)
    fig.tight_layout()
    save(fig, os.path.join(out, "fig_bound"))


def fig_solver(out):
    """Solver validation: dense cross-check and the freezing ablation."""
    p = os.path.join(C.RES, "solver_fig.json")
    if not os.path.exists(p):
        print("no solver_fig.json"); return
    d = json.load(open(p, encoding="utf-8"))
    w = set_style("double")
    fig, axes = plt.subplots(1, 2, figsize=(w, w * 0.32))

    # (a) dense direct solve vs batched Bi-CGSTAB on a 10x10 grid
    ax = axes[0]
    diffs = np.array(d["dense"]["diffs"])
    x = np.arange(1, len(diffs) + 1)
    ax.bar(x, diffs, color=OKABE[4], width=0.62)
    ax.axhline(1e-12, color=OKABE[5], ls="--", lw=1.0)
    ax.set_yscale("log")
    ax.set_ylim(max(1e-15, diffs.min() / 3), max(1e-11, diffs.max() * 3))
    ax.set_xticks(x)
    ax.set_xlabel("test medium")
    ax.set_ylabel(r"rel. difference vs dense solve")
    v = d["dense"]["max_diff"]
    e = int(np.floor(np.log10(v)))
    m = v / (10.0 ** e)
    ax.text(0.97, 0.93, r"max $=%.1f\times10^{%d}$" % (m, e),
            transform=ax.transAxes, ha="right", va="top", fontsize=8)
    ax.text(0.97, 0.80, r"$10^{-12}$ ref.", transform=ax.transAxes,
            ha="right", va="top", fontsize=7, color=OKABE[5])
    ax.set_title("(a) $10\\times10$ grid, dense cross-check", fontsize=8.5)
    ax.spines[["top", "right"]].set_visible(False)

    # (b) freezing ablation: three implementations of the shared loop
    ax = axes[1]
    conds = d["freeze"]["conds"]
    idx = np.arange(len(conds))
    naive = np.array([100.0 * c["naive"] / c["n"] for c in conds])
    zmask = np.array([100.0 * c["zeromask"] / c["n"] for c in conds])
    frozen = np.array([100.0 * c["frozen"] / c["n"] for c in conds])
    sil = np.array([100.0 * c["silent"] / c["n"] for c in conds])
    # "silent" is a subset of "naive" (the corrupted labels the in-loop check
    # cannot see), so it is drawn as a narrower bar on top of the plain-loop
    # bar instead of beside it.  Both used to be shades of red, which made the
    # containment -- the whole point of the panel -- invisible.
    ax.bar(idx, naive, width=0.66, color=OKABE[5], alpha=0.45,
           label="unfrozen (plain loop)")
    ax.bar(idx, sil, width=0.32, color=OKABE[6],
           label="of which silent to the in-loop check")
    ax.axhline(0.0, color=OKABE[3], lw=2.2)
    ax.plot([], [], color=OKABE[3], lw=2.2, label="frozen (both variants)")
    ax.set_xticks(idx)
    ax.set_xticklabels(["%d" % (c["batch"] + 1) for c in conds], fontsize=7)
    ax.set_xlabel("batch index")
    ax.set_ylabel("% corrupted labels")
    # The frozen bars are identically zero, so they are stated rather than
    # drawn: an invisible bar would leave the reader guessing whether the
    # condition was missing or measured at zero.  The wording gives the maximum
    # over batches, which stays true if a batch ever becomes non-zero -- an
    # "is zero" claim would silently go stale.
    ax.text(0.5, 0.965,
            "frozen variants: max %.1f%% and %.1f%% over all batches"
            % (float(frozen.max()), float(zmask.max())),
            transform=ax.transAxes, ha="center", va="top", fontsize=6.6,
            color=OKABE[3])
    ax.set_ylim(0, max(naive.max() * 1.28, 1.0))
    ax.legend(fontsize=6.4, loc="upper center", bbox_to_anchor=(0.5, -0.20),
              ncol=1, handlelength=1.5, labelspacing=0.28)
    ax.set_title("(b) corrupted labels, %d heterogeneous batches"
                 % len(conds), fontsize=8.5)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, os.path.join(out, "fig_solver"))


def fig_training(out):
    """In-band validation error vs epoch, one curve per contrast arm.

    Read straight from runs/*/train_log.csv, so the figure cannot drift from the
    logs.  The point it makes is the contrast-sensitivity of the in-band floor:
    the ordering of the curves at the last epoch is the same as the ordering of
    the plateau in Fig. 1.
    """
    w = set_style("single")
    fig, ax = plt.subplots(figsize=(w, w * 0.74))
    any_ = False
    for i, vmax in enumerate(C.VMAX_LIST):
        tag = "a%03d" % int(round(vmax * 100))
        curves, seeds = [], []
        for d in sorted(glob.glob(os.path.join(C.RUNS, "main_fno_%s_s*" % tag))):
            p = os.path.join(d, "train_log.csv")
            if not os.path.exists(p):
                continue
            rows = [l.split(",") for l in
                    open(p, encoding="utf-8").read().strip().splitlines()[1:]]
            rows = [r for r in rows if len(r) >= 3]
            if len(rows) < 2:
                continue
            curves.append(np.array([float(r[2]) for r in rows]))
            seeds.append(os.path.basename(d).rsplit("_", 1)[1])
        if not curves:
            continue
        n = min(len(c) for c in curves)
        A = np.array([c[:n] for c in curves])
        ep = np.arange(1, n + 1)
        if A.shape[0] > 1:
            ax.fill_between(ep, A.min(axis=0), A.max(axis=0),
                            color=OKABE[i], alpha=0.16, lw=0)
        ax.semilogy(ep, A.mean(axis=0), color=OKABE[i], marker=MK[i],
                    markevery=max(1, n // 8), ms=3.0, lw=1.3,
                    label=r"$v_{\max}=%.2f$ ($n{=}%d$)" % (vmax, len(seeds)))
        any_ = True
    if not any_:
        print("no training logs"); return
    ax.set_xlabel("epoch")
    ax.set_ylabel("in-band validation relative $L_2$")
    # The span of this axis is under two decades, so the default log locator
    # labels every mantissa and the tick column becomes unreadable.  Round
    # values only, spanning the actual data range.
    ax.set_ylim(0.015, 0.8)
    ax.set_yticks([0.02, 0.05, 0.1, 0.2, 0.4])
    ax.set_yticklabels(["0.02", "0.05", "0.10", "0.20", "0.40"])
    # semilogy labels the minor ticks as well when the axis spans fewer than
    # two decades, which interleaved "6 x 10^-2" between the round values.
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.spines[["top", "right"]].set_visible(False)
    # Seven entries in two columns used to sit on top of the descending curves,
    # and the "shaded" note was placed over the v_max = 0.05 curve.  The span is
    # now a legend entry of its own and the whole legend is below the axis.
    h, l = ax.get_legend_handles_labels()
    h += [Patch(facecolor="0.5", alpha=0.25, lw=0)]
    l += ["seed span"]
    ax.legend(h, l, fontsize=6.5, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.17), handlelength=1.8,
              columnspacing=1.0, labelspacing=0.35)
    save(fig, os.path.join(out, "fig_training"))


def fig_crosscontrast(evals, out):
    """Why a cliff ratio must be read arm-matched.

    Each trained model already carries a full sweep over all five evaluation
    contrasts, so the cross-contrast reading costs nothing extra.  It shows that
    the cliff is a property of a model measured *at the contrast it was trained
    on*: evaluate the weak-contrast model at strong contrast and the error is
    already at plateau level inside the training band, so the band edge leaves no
    signature at all.
    """
    pairs = [("main_fno_a005_s0", 0.05, r"trained at $v_{\max}=0.05$"),
             ("main_fno_a100_s0", 1.00, r"trained at $v_{\max}=1.00$")]
    pairs = [(n, v, t) for n, v, t in pairs if n in evals]
    if not pairs:
        print("no cross-contrast models available"); return
    w = set_style("double")
    fig, axes = plt.subplots(1, len(pairs), figsize=(w, w * 0.32),
                             sharex=True, sharey=True)
    if len(pairs) == 1:
        axes = [axes]
    for k, (ax, (name, own, ttl)) in enumerate(zip(axes, pairs)):
        d = evals[name]
        for i, vmax in enumerate(C.VMAX_LIST):
            F, M, S = curve(d, vmax)
            style = dict(color=OKABE[i], marker=MK[i], ms=3.0, lw=1.3)
            if abs(vmax - own) < 1e-9:
                style["lw"] = 2.4
            ax.semilogy(F, M, label=r"eval $v_{\max}=%.2f$" % vmax,
                        clip_on=False, **style)
        ax.axvspan(1.0, 4.0, color="#888888", alpha=0.13, lw=0)
        ax.axvline(4.0, color="k", lw=0.8, ls=":")
        panel_label(ax, "ab"[k], ttl, fontsize=8.5)
        ax.set_xlabel("frequency $f$")
        ax.set_xticks([2, 4, 6, 8, 10, 12])
        ax.set_xlim(1, 12.2)
        ax.set_ylim(0.012, 5.0)
        ax.spines[["top", "right"]].set_visible(False)
        if k:
            ax.tick_params(labelleft=False)
    axes[0].set_ylabel(r"relative $L_2$ error")
    # The eval-contrast legend must appear on BOTH panels: panel (a) previously
    # carried it and panel (b) did not, which is the asymmetry the review flagged.
    for ax0 in axes:
        ax0.legend(fontsize=6.0, ncol=1, loc="lower right", handlelength=1.4,
                   labelspacing=0.22)
    save(fig, os.path.join(out, "fig_crosscontrast"))


def fig_c1(out):
    """Conditioning-confound discriminating experiment (Section sec:c1).

    Two panels, because the experiment manipulates one quantity and measures
    another, and overlaying them on one axis would hide which is which:
      (a) the measured curves -- seed-mean relative L2 error vs f for the
          baseline (gamma = 0.05) and the compensated arm (gamma(f) holding
          the attained norm constant), error bars one seed SD, training band
          shaded, cliff-onset interval marked.  Both arms step inside the
          same grid interval.
      (b) the manipulated quantity -- the attained operator norm at the same
          test frequencies.  The baseline's non-monotone spread collapses to a
          constant that sits below the baseline's band-edge value, i.e. the
          compensated arm is the better-conditioned one at every frequency.
    Everything is read from results/c1_verdict.json (written by
    code/analyze_c1.py), so the figure and the table cannot drift apart.
    No GPU: the curves are evaluation records plus a 64x64 symbol norm.
    """
    p = os.path.join(C.RES, "c1_verdict.json")
    if not os.path.exists(p):
        print("no c1_verdict.json -- run code/analyze_c1.py"); return
    d = json.load(open(p, encoding="utf-8"))
    v = d["verdict"]
    w = set_style("double")
    fig, axes = plt.subplots(1, 2, figsize=(w, w * 0.38))
    specs = [("baseline", OKABE[4], MK[0],
              r"baseline, $\gamma=0.05$ ($%.2f$--$%.2f$)"
              % (v["attained_baseline_lo"], v["attained_baseline_hi"])),
             ("compensated", OKABE[1], MK[1],
              r"compensated, $\gamma(f)$ ($%.2f$ const.)"
              % v["attained_compensated"])]
    # ---- (a) measured error curves
    ax = axes[0]

    def _fl(x):
        # 4.0 must not print as "4": the whole argument rests on the 0.25
        # grid spacing at the band edge.
        return "%.1f" % x if abs(x - round(x)) < 1e-9 else "%.2f" % x

    for key, col, mk, lab in specs:
        arm = d["arms"][key]
        ax.errorbar(arm["f"], arm["mean"], yerr=arm["sd"], color=col,
                    marker=mk, capsize=2, lw=1.3, label=lab, clip_on=False)
    ax.axvspan(1.0, 4.0, color="#888888", alpha=0.13, lw=0)
    lo, hi = d["arms"]["baseline"]["onset"]
    ax.axvspan(lo, hi, color="#CC79A7", alpha=0.30, lw=0)
    ax.axvline(4.0, color="k", lw=0.8, ls=":")
    ax.set_yscale("log")
    ax.set_ylim(0.008, 4.5)
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=3))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("frequency $f$")
    ax.set_ylabel(r"relative $L_2$ error")
    ax.set_xticks([2, 4, 6, 8, 10, 12])
    ax.set_xlim(1, 12.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=6.5, loc="upper left", frameon=False, handlelength=1.6,
              labelspacing=0.25)
    ax.text(0.985, 0.045, "onset $[%s,%s]$\n$\\pm1$ SD, %d seeds"
            % (_fl(lo), _fl(hi), d["arms"]["baseline"]["seeds"]),
            transform=ax.transAxes, fontsize=6.5, ha="right", va="bottom",
            color="#444444")
    panel_label(ax, "a")
    # ---- (b) the manipulated quantity: attained operator norm
    ax = axes[1]
    F = d["arms"]["baseline"]["f"]
    for key, col, mk, lab in specs:
        ax.plot(F, d["arms"][key]["attained"], color=col, marker=mk, ls="-",
                lw=1.3, clip_on=False)
    ax.axvspan(1.0, 4.0, color="#888888", alpha=0.13, lw=0)
    ax.axvline(4.0, color="k", lw=0.8, ls=":")
    ax.set_ylim(2.0, 11.0)
    ax.set_xlabel("frequency $f$")
    ax.set_ylabel(r"attained $\max|\kappa^{2}\widehat{G}_{\kappa}|$")
    ax.set_xticks([2, 4, 6, 8, 10, 12])
    ax.set_xlim(1, 12.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(0.985, 0.05, "spread $%.1f\\times$ $\\rightarrow$ const."
            % v["attained_baseline_spread"], transform=ax.transAxes,
            fontsize=6.5, ha="right", va="bottom", color="#444444")
    panel_label(ax, "b")
    save(fig, os.path.join(out, "fig_c1"))


def fig_anchor(out):
    """The analytic anchor against the trained operator (Section sec:anchor).

    Two panels, because the claim is not "the anchor is more accurate" but "the
    anchor has no cliff", and a single accuracy curve cannot show that:
      (a) the four predictors at v_max = 0.25: the trained FNO (seed mean,
          one seed SD, from the evaluation records), the anchor (two complex
          coefficients fitted on the training cache), the raw first-order Born
          series, and the trivial baseline that returns the incident field.
          The trained curve steps at the band edge; the anchor does not.
      (b) the plateau against contrast for the same three learned/analytic
          objects, which is where the complementarity lives: the trained
          operator wins inside the band and the anchor wins outside it.
    Read from results/anchor_baseline.json and the evaluation records, so the
    figure and Table tab:anchor cannot drift apart.  No GPU.
    """
    p = os.path.join(C.RES, "anchor_baseline.json")
    if not os.path.exists(p):
        print("no anchor_baseline.json -- run code/anchor_baseline.py"); return
    d = json.load(open(p, encoding="utf-8"))
    ev = load_evals()
    w = set_style("double")
    fig, axes = plt.subplots(1, 2, figsize=(w, w * 0.38))
    F = np.array(d["f_test"])

    # ---- (a) v_max = 0.25
    ax = axes[0]
    a = d["arms"]["0.25"]
    tag = "a025"
    names = [n for n in sorted(ev) if n.startswith("main_fno_%s_" % tag)]
    if names:
        Fm, M, S = mean_curve(ev, names, 0.25)
        ax.errorbar(Fm, M, yerr=S, color=OKABE[5], marker=MK[0], capsize=2,
                    lw=1.3, label="trained FNO", clip_on=False)
    ax.plot(F, [r["mean"] for r in a["sweep"]], color=OKABE[2], marker=MK[1],
            lw=1.3, label="anchor, $aB_1+bB_2$", clip_on=False)
    ax.plot(F, a["born1"], color=OKABE[0], marker=MK[2], lw=1.0, ls="--",
            label="Born series, order 1", clip_on=False)
    ax.plot(F, a["trivial"], color="#888888", marker=MK[3], lw=1.0, ls=":",
            label="trivial, $\\hat u\\equiv0$", clip_on=False)
    ax.axvspan(1.0, 4.0, color="#888888", alpha=0.13, lw=0)
    ax.axvline(4.0, color="k", lw=0.8, ls=":")
    ax.set_yscale("log")
    ax.set_xlabel("frequency $f$")
    ax.set_ylabel(r"relative $L_2$ error")
    ax.set_xticks([2, 4, 6, 8, 10, 12])
    ax.set_xlim(1, 12.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=6.5, loc="lower right", frameon=False, handlelength=1.6,
              labelspacing=0.25)
    ax.text(0.03, 0.97, "$v_{\\max}=0.25$", transform=ax.transAxes,
            fontsize=6.5, ha="left", va="top", color="#444444")
    panel_label(ax, "a")

    # ---- (b) plateau against contrast
    ax = axes[1]
    vm = np.array([float(k) for k in sorted(d["arms"])])
    anch = np.array([d["arms"]["%.2f" % v]["plateau"] for v in vm])
    triv = np.array([d["arms"]["%.2f" % v]["trivial_plateau"] for v in vm])
    ax.plot(vm, anch, color=OKABE[2], marker=MK[1], lw=1.3,
            label="anchor", clip_on=False)
    ax.plot(vm, triv, color="#888888", marker=MK[3], lw=1.0, ls=":",
            label="trivial, $\\hat u\\equiv0$", clip_on=False)
    fno, keep = [], []
    for v in vm:
        tag = "a%03d" % int(round(v * 100))
        ns = [n for n in sorted(ev) if n.startswith("main_fno_%s_" % tag)]
        if not ns:
            continue
        _, M, _ = mean_curve(ev, ns, float(v))
        fno.append(float(np.mean(M[np.array(d["f_test"]) >= 6])))
        keep.append(float(v))
    if fno:
        ax.plot(np.array(keep), np.array(fno), color=OKABE[5], marker=MK[0],
                lw=1.3, label="trained FNO", clip_on=False)
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel("$v_{\\max}$")
    ax.set_ylabel(r"plateau $\varepsilon_{\infty}$")
    ax.set_xticks([0.05, 0.1, 0.25, 0.5, 1.0])
    ax.set_xticklabels(["0.05", "0.10", "0.25", "0.50", "1.00"])
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=6.5, loc="upper left", frameon=False, handlelength=1.6,
              labelspacing=0.25)
    panel_label(ax, "b")
    save(fig, os.path.join(out, "fig_anchor"))


def fig_fewshot(out):
    """Few-shot purchase curve (supplementary experiment 1, Section sec:interv).

    The baseline FNO (trained on f in [1,4]) is fine-tuned on n in
    {64, 256, 1024} samples drawn from f in [5,8] at v_max = 0.25 and scored
    zero-shot at f = 5, 8 (inside the fine-tune band) and f = 12 (beyond it).
    The covered band is purchased monotonically while the uncovered frequency
    is flat -- labels buy coverage, not extrapolation.
    """
    p = os.path.join(C.RES, "fewshot_purchase.json")
    if not os.path.exists(p):
        print("no fewshot_purchase.json -- run code/run_fewshot.py"); return
    d = C.load_json(p)
    ns = [0, 64, 256, 1024]
    w = set_style("single")
    fig, ax = plt.subplots(figsize=(w, w * 0.78))
    specs = [(5.0, OKABE[0], "f = 5 (fine-tune band)"),
             (8.0, OKABE[1], "f = 8 (fine-tune band)"),
             (12.0, OKABE[2], "f = 12 (uncovered)")]
    for f, col, lab in specs:
        ys = [d["rows"][str(n)][str(float(f))] for n in ["0", "64", "256", "1024"]]
        ax.plot(ns, ys, color=col, marker="o", ls="-", lw=1.4, ms=4.5,
                label=lab, clip_on=False)
    ax.axvspan(64, 1024, color="#888888", alpha=0.10, lw=0)
    ax.set_xlabel("out-of-band labels $n$")
    ax.set_ylabel(r"relative $L_2$ error")
    ax.set_xticks(ns)
    ax.set_ylim(0.3, 1.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=6.5, loc="upper right", frameon=False, handlelength=1.6,
              labelspacing=0.25)
    ax.text(0.03, 0.97, "$v_{\\max}=0.25$", transform=ax.transAxes,
            fontsize=6.5, ha="left", va="top", color="#444444")
    save(fig, os.path.join(out, "fig_fewshot"))


def fig_c3probe(out):
    """Loss-shaping discriminators (C3/C3b): energy probe of Section sec:disc.

    Two panels, both read from results/c3_verdict.json (written by
    code/analyze_c3.py, which recomputes the probe from the checkpoints over
    freshly drawn paired media -- an instrument independent of the eval JSONs):
      (a) the correction-energy ratio |u_hat| / |u| and
      (b) the phase alignment |<u_hat, u>| / (|u_hat| |u|),
    each vs f for the baseline, the in-band energy penalty (C3) and the
    out-of-band Born-anchored penalty (C3b).  The out-of-band arm recovers the
    ENERGY (that is what its penalty targets) while its ALIGNMENT drops to
    ~0.01 and its error gets WORSE (Table tab:disc): the pessimistic-collapse
    hypothesis is excluded -- what the model lacks out of band is phase
    information, not output energy.
    """
    p = os.path.join(C.RES, "c3_verdict.json")
    if not os.path.exists(p):
        print("no c3_verdict.json -- run code/analyze_c3.py"); return
    d = json.load(open(p, encoding="utf-8"))
    w = set_style("double")
    fig, axes = plt.subplots(1, 2, figsize=(w, w * 0.38))
    specs = [("baseline", OKABE[4], MK[0], "baseline"),
             ("energy", OKABE[1], MK[1], "in-band penalty (C3)"),
             ("energy_ob", OKABE[6], MK[2], "out-of-band anchor (C3b)")]
    for pi, (field, ylab) in enumerate(
            [("energy_ratio", r"$\|\widehat{u}\|/\|u\|$"),
             ("alignment", r"$|\langle\widehat{u},u\rangle|/( \|\widehat{u}\|\|u\| )$")]):
        ax = axes[pi]
        for key, col, mk, lab in specs:
            arm = d["arms"][key]
            fs = sorted(float(k) for k in arm["energy"])
            ys = [arm["energy"]["%.2f" % f][field] for f in fs]
            ax.plot(fs, ys, color=col, marker=mk, ls="-", lw=1.3, label=lab,
                    clip_on=False)
        ax.axvspan(1.0, 4.0, color="#888888", alpha=0.13, lw=0)
        ax.axvline(4.0, color="k", lw=0.8, ls=":")
        ax.set_xlabel("frequency $f$")
        ax.set_ylabel(ylab)
        ax.set_xticks([4, 5, 6, 8, 10, 12])
        ax.set_xlim(3.6, 12.4)
        ax.set_ylim(0.0, 1.75)
        ax.spines[["top", "right"]].set_visible(False)
        panel_label(ax, "ab"[pi])
        if pi == 0:
            ax.legend(fontsize=6.0, loc="upper left", frameon=False,
                      handlelength=1.6, labelspacing=0.25)
    axes[0].text(0.03, 0.06, "training band", transform=axes[0].transAxes,
                 fontsize=6.5, ha="left", va="bottom", color="#444444")
    save(fig, os.path.join(out, "fig_c3probe"))


def build_in(n_grid, k, theta, v, device):
    import torch
    import physics as P
    import data as D
    psi_inc = P.incident_plane_wave(n_grid, k, theta, device).to(torch.complex64)
    return D.build_inputs(v.float(), psi_inc, (k / (2 * np.pi))), psi_inc


def fig_realgeo(out):
    """Third evidence rung: the same boundary on real geological structure.

    (a) the Marmousi velocity model with the three equally spaced windows
        marked, (b) the error sweep for the real windows against the synthetic
        control under an identical design, (c) the one quantity that tells the
        two families apart before any network is involved: the out-of-band
        energy of the contrast field itself.
    """
    import make_realgeo as RG
    p = os.path.join(C.RES, "realgeo.json")
    if not os.path.exists(p):
        print("no realgeo.json"); return
    d = json.load(open(p, encoding="utf-8"))
    c, x, z = RG.load_marmousi()
    wins, rough = RG.pick_windows(c, x, z)

    w = set_style("double")
    fig = plt.figure(figsize=(w, w * 0.30))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.7, 1.0, 1.0], wspace=0.34)

    # ---- (a) the model and where the windows were taken --------------------
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(c, cmap="turbo", aspect="auto",
                   extent=[x.min(), x.max(), z.max(), z.min()])
    # Three 3 km windows cover 96% of the 9.4 km survey, so shading them would
    # tint the whole panel and distinguish nothing.  Mark the extents with a
    # three-colour ruler along the top edge instead.
    RULER = ["#FFFFFF", "#00E5FF", "#FFD400"]
    for i, k in enumerate(("west", "centre", "east")):
        x0, x1, z0, z1 = wins[k]
        ax.add_patch(Rectangle((x0, 0), x1 - x0, 150, fc=RULER[i], ec="none"))
        ax.axvline(x0, color="w", lw=0.9, ls=["-", "--", ":"][i])
        ax.text(0.5 * (x0 + x1), 75, ["i", "ii", "iii"][i], color="k",
                ha="center", va="center", fontsize=8, fontweight="bold")
    ax.set_xlabel("horizontal distance (km)")
    ax.set_ylabel("depth (km)")
    ax.set_xticks([0, 2000, 4000, 6000, 8000])
    ax.set_xticklabels(["0", "2", "4", "6", "8"])
    ax.set_yticks([0, 1000, 2000, 3000])
    ax.set_yticklabels(["0", "1", "2", "3"])
    panel_label(ax, "a")
    # Colourbar OUTSIDE panel (a): an in-panel bar occludes the velocity image,
    # and an axis-label collision with panel (b) was the earlier failure mode.
    # Placing it to the right of panel (a) keeps the image intact and the gap
    # to panel (b) clear.
    cb = fig.colorbar(im, ax=ax, location="right", pad=0.03, shrink=0.85,
                      aspect=20)
    cb.set_label("$c$ (m/s)", fontsize=6.5, labelpad=1)
    cb.set_ticks([2000, 3500, 5000])
    cb.ax.tick_params(labelsize=6.5, length=1.6)
    cb.outline.set_linewidth(0.5)

    # ---- (b) the sweep, real against synthetic ----------------------------
    VMAX = 0.25
    key = "%.2f" % VMAX
    F = d["f_test"]
    ax = fig.add_subplot(gs[0, 1])
    for fam, names, col, ls, lab in (
            ("real", [g for g in d["order"] if g.startswith("geo_")], OKABE[5], "-", "real geology"),
            ("syn", [g for g in d["order"] if g.startswith("syn_")], OKABE[4], "--", "synthetic")):
        # Collapse the three windows of each family to a mean curve with a
        # min--max band over windows (and their seed spans).  This reduces the
        # six raw curves to the two families the legend names: the review noted
        # six curves but only two legend entries.
        Mm, Lo, Hi = [], [], []
        for j in range(len(F)):
            vals = []
            for g in names:
                row = d["curves"][g][key][j]
                ks = [kk for kk in row if kk.startswith("FNO|")]
                m = float(np.mean([row[kk]["mean"] for kk in ks]))
                lo, hi = row["seedspan"]["FNO"]
                vals.append((m, float(lo), float(hi)))
            ms = [v[0] for v in vals]
            Mm.append(float(np.mean(ms)))
            Lo.append(min(v[1] for v in vals))
            Hi.append(max(v[2] for v in vals))
        Mm, Lo, Hi = np.array(Mm), np.array(Lo), np.array(Hi)
        ax.plot(F, Mm, ls, color=col, lw=1.6, label=lab)
        ax.fill_between(F, Lo, Hi, color=col, alpha=0.15, lw=0)
    ax.axvspan(1.0, 4.0, color="#888888", alpha=0.13, lw=0)
    ax.axvline(4.0, color="k", lw=0.8, ls=":")
    ax.set_yscale("log")
    ax.set_ylim(0.02, 2.2)
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=3))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(1, 12.2)
    ax.set_xticks([2, 4, 6, 8, 10, 12])
    ax.set_xlabel("frequency $f$")
    ax.set_ylabel(r"relative $L_2$ error")
    panel_label(ax, "b", r"$v_{\max}=%.2f$, FNO" % VMAX)
    ax.legend(loc="lower right", fontsize=7.5)
    ax.spines[["top", "right"]].set_visible(False)

    # ---- (c) the input fields themselves ---------------------------------
    ax = fig.add_subplot(gs[0, 2])
    # M = 32 is omitted: on a 64-point grid the largest mode index IS 32, so the
    # fraction there is zero by construction and a log axis would render the
    # last segment as a spurious cliff.
    Md = [M for M in d["modes"] if M < 32]
    for fam, names, col, ls in (
            ("real", [g for g in d["order"] if g.startswith("geo_")], OKABE[5], "-"),
            ("syn", [g for g in d["order"] if g.startswith("syn_")], OKABE[4], "--")):
        for g in names:
            y = [d["bands"][g][str(M)] for M in Md]
            ax.plot(Md, y, ls, color=col, lw=1.3)
    ax.legend([Line2D([], [], color=OKABE[5], ls="-", lw=1.3),
               Line2D([], [], color=OKABE[4], ls="--", lw=1.3)],
              ["real geology", "synthetic"], loc="lower left",
              fontsize=7, frameon=False)
    ax.set_yscale("log")
    ax.set_ylim(1e-6, 1.0)
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=8))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("mode index $M$")
    ax.set_ylabel("field energy outside $M$")
    panel_label(ax, "c", "contrast field")
    ax.set_xticks([4, 8, 12, 16, 20, 24])
    ax.spines[["top", "right"]].set_visible(False)
    b4g = max(d["bands"][g]["4"] for g in d["order"] if g.startswith("geo_"))
    b4s = max(d["bands"][g]["4"] for g in d["order"] if g.startswith("syn_"))
    # The gap is a single vertical distance at M = 4, so draw it as one: an
    # arrow floating elsewhere leaves the reader to guess which two curves it
    # refers to.
    ax.annotate("", xy=(4, b4g), xytext=(4, b4s),
                arrowprops=dict(arrowstyle="<->", lw=0.8, color="#333333"))
    ax.text(5.4, np.sqrt(b4g * b4s), "%.0f$\\times$" % (b4g / b4s),
            fontsize=8, va="center")
    save(fig, os.path.join(out, "fig_realgeo"))


def fig_banddown(evals, out):
    """Two-sided boundary: extend the FNO error curves below the training band.

    Below-band points come from results/banddown.json, a fresh evaluation of
    the same checkpoints on paired media at f in {0.25, 0.5, 0.75, 1.0}; the
    above-band part reuses the eval sweeps.  The trivial baseline is drawn for
    each arm in the same colour, so the contrast between the below-band
    elevation (model several times above baseline) and the above-band
    saturation-at-baseline is visible in one panel.
    """
    p = os.path.join(C.RES, "banddown.json")
    if not os.path.exists(p):
        print("no banddown.json"); return
    d = json.load(open(p, encoding="utf-8"))
    spec = json.load(open(os.path.join(C.RES, "spectrum.json"),
                          encoding="utf-8"))
    w = set_style("single")
    fig, ax = plt.subplots(figsize=(w, w * 0.62))
    ax.axvspan(1.0, 4.0, color="#888888", alpha=0.13, lw=0)
    ax.axvline(1.0, color="k", lw=0.8, ls=":")
    ax.axvline(4.0, color="k", lw=0.8, ls=":")
    for k, (vk, vmax) in enumerate(zip(("a005", "a010", "a025", "a050", "a100"),
                                       C.VMAX_LIST)):
        col = OKABE[k + 1]
        key = "%.2f" % vmax
        Fb = np.array(d["f_test"])
        rows = np.mean([[d["runs"]["main_fno_%s_s%d" % (vk, s)]
                         ["sweep"][key][i]["mean"]
                         for i in range(len(Fb))] for s in range(3)], axis=0)
        triv = np.mean([[d["runs"]["main_fno_%s_s%d" % (vk, s)]
                         ["sweep"][key][i]["triv_mean"]
                         for i in range(len(Fb))] for s in range(3)], axis=0)
        ax.plot(Fb, rows, "-", color=col, lw=1.5,
                label="FNO, $v_{\\max}=%.2f$" % vmax)
        ax.plot(Fb, triv, "--", color=col, lw=1.0, alpha=0.65)
        # above band from the eval sweeps (seed means of the same checkpoints)
        Fa, Ma = [], []
        for s in range(3):
            dk = evals.get("main_fno_%s_s%d" % (vk, s))
            if dk is None:
                continue
            F_, M_, _ = curve(dk, vmax)
            Fa.append(F_); Ma.append(M_)
        if Fa:
            Fa = np.mean(Fa, axis=0)
            Ma = np.mean(Ma, axis=0)
            ax.plot(Fa, Ma, "-", color=col, lw=1.5)
            Ta = np.array([spec["data"][str(vmax)]["f=%g" % f]["triv"]
                           for f in C.F_TEST])
            ax.plot(Fa, Ta, "--", color=col, lw=1.0, alpha=0.65)
    ax.plot([], [], "-", color="#444444", lw=1.5, label="FNO")
    ax.plot([], [], "--", color="#444444", lw=1.0, alpha=0.65,
            label="trivial baseline")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.22, 13.0)
    ax.set_ylim(0.01, 3.0)
    ax.set_xticks([0.25, 0.5, 1, 2, 4, 8, 12])
    ax.set_xticklabels(["0.25", "0.5", "1", "2", "4", "8", "12"])
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=4))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("frequency $f$")
    ax.set_ylabel(r"relative $L_2$ error")
    ax.legend(loc="upper left", fontsize=7, ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, os.path.join(out, "fig_banddown"))


def fig_plateau_coupling(evals, out):
    """Out-of-band plateau against the same-arm trivial baseline, vs coupling.

    The plateau is the seed-mean error over f >= 6; the baseline comes from
    spectrum.json; the coupling is v_max/(2 gamma).  One log-log panel shows the
    model tracking the baseline across the 20x coupling range, with the excess
    (model over baseline) growing slowly with coupling.
    """
    spec = json.load(open(os.path.join(C.RES, "spectrum.json"),
                          encoding="utf-8"))
    w = set_style("single")
    fig, ax = plt.subplots(figsize=(w, w * 0.72))
    coups, plats, trivs = [], [], []
    for vk, vmax in zip(("a005", "a010", "a025", "a050", "a100"), C.VMAX_LIST):
        plateau = []
        for s in range(3):
            dk = evals.get("main_fno_%s_s%d" % (vk, s))
            if dk is None:
                continue
            F, M, _ = curve(dk, vmax)
            plateau.append(M[F >= 6])
        if not plateau:
            continue
        plats.append(float(np.mean(plateau)))
        trivs.append(float(np.mean(
            [spec["data"][str(vmax)]["f=%g" % f]["triv"]
             for f in C.F_TEST if f >= 6])))
        coups.append(vmax / (2 * C.GAMMA))
    coups, plats, trivs = np.array(coups), np.array(plats), np.array(trivs)
    ax.plot(coups, plats, "-o", color=OKABE[4], lw=1.5, ms=5,
            label="FNO plateau")
    ax.plot(coups, trivs, "--s", color=OKABE[2], lw=1.5, ms=4,
            label="trivial baseline")
    ax.annotate("$\\times{%.2f}$" % (plats[0] / trivs[0]),
                xy=(coups[0], plats[0]), xytext=(1.4, 0.30),
                fontsize=7.5, va="center",
                arrowprops=dict(arrowstyle="->", lw=0.7, color="#333333"))
    ax.annotate("$\\times{%.2f}$" % (plats[-1] / trivs[-1]),
                xy=(coups[-1], plats[-1]), xytext=(4.2, 2.35),
                fontsize=7.5, va="center",
                arrowprops=dict(arrowstyle="->", lw=0.7, color="#333333"))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.4, 12)
    ax.set_ylim(0.15, 3.2)
    ax.set_xticks([0.5, 1, 2.5, 5, 10])
    ax.set_xticklabels(["0.5", "1", "2.5", "5", "10"])
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=4))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(r"coupling $v_{\max}/(2\gamma)$")
    ax.set_ylabel(r"mean relative $L_2$ error, $f \geq 6$")
    ax.legend(loc="upper left", fontsize=7.5)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, os.path.join(out, "fig_plateau_coupling"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=C.FIG)
    ap.add_argument("--only", default="all")
    ap.add_argument("--res", default=None,
                    help="folder holding eval_*.json (default results/)")
    a = ap.parse_args()
    ev = load_evals(a.res)
    print("loaded %d evals" % len(ev))
    if a.only in ("all", "main"):
        fig_main(ev, a.out)
    if a.only in ("all", "training"):
        fig_training(a.out)
    if a.only in ("all", "arch"):
        fig_arch(ev, a.out)
    if a.only in ("all", "cross"):
        fig_crosscontrast(ev, a.out)
    if a.only in ("all", "band"):
        fig_band(ev, a.out)
    if a.only in ("all", "mode"):
        fig_mode_phase(ev, a.out, "b1_4", 4.0, "fig_mode_phase_b4")
    if a.only in ("all", "scaling"):
        print(fig_scaling(ev, a.out))
    if a.only in ("all", "banddown"):
        fig_banddown(ev, a.out)
    if a.only in ("all", "plateau"):
        fig_plateau_coupling(ev, a.out)
    if a.only in ("all", "spectrum"):
        fig_spectrum(a.out)
    if a.only in ("all", "bound"):
        fig_bound(ev, a.out)
    if a.only in ("all", "solver"):
        fig_solver(a.out)
    if a.only in ("all", "fields"):
        fig_fields(a.out)
    if a.only in ("all", "realgeo"):
        fig_realgeo(a.out)
    if a.only in ("all", "c1"):
        fig_c1(a.out)
    if a.only in ("all", "c3probe"):
        fig_c3probe(a.out)
    if a.only in ("all", "anchor"):
        fig_anchor(a.out)
    if a.only in ("all", "fewshot"):
        fig_fewshot(a.out)
