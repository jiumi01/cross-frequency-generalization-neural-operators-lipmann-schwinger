# Author: Longgang Wang <longgangwang@163.com>
# Affiliation: Luoyang Institute of Science and Technology, Luoyang 471000, Henan, China

"""Paper figures -- design system v2.

Rules every figure in the manuscript follows, and why each exists:

1. TWO natural widths only.  FULL is the MDPI text block (16.6 cm = 470 pt);
   HALF is a half column (8.1 cm).  A figure is emitted at its natural width
   and included at \\textwidth (FULL) or 0.49\\textwidth (HALF) in BOTH
   language versions, so the effective type size is identical in EN and CN
   and equals the declared rcParams size.  The previous generation emitted
   widths between 228 and 454 pt and included them at 0.52--1.0 \\textwidth,
   which scaled the EN type down to ~6.5 pt (below print legibility) and the
   CN type up to ~120% -- the two versions of one figure did not even agree
   with each other.  code/fig_widths.py re-derives the inclusion widths from
   the emitted MediaBoxes and asserts EN == CN, so this cannot drift again.

2. ONE style registry (SERIES) per physical object.  A curve keeps its
   colour and marker in every figure it appears in.  The earlier figures let
   the same model be blue in one panel and vermilion in the next.

3. DIRECT end labels.  Series are named at the right edge of the axes at the
   height of their own curve.  In-panel legends are used only where end
   labels would collide; they used to float over the data region and, in two
   panels, over the band-edge rule they were meant to explain.

4. ONE band convention.  The training band is shaded, its edge is a dotted
   rule, and the shading is drawn by band() everywhere -- earlier figures
   used a dotted line in some panels, a shaded span in others, and a pink
   span in one.

5. Decade ticks on log axes with the minor formatter suppressed; explicit
   round ticks when the span is under two decades.

6. Every curve is read from the JSON archives that also feed the tables and
   the manuscript macros -- nothing is re-derived here, so a figure cannot
   disagree with the number printed next to it.
"""
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
from matplotlib.ticker import LogLocator, NullFormatter, FixedLocator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

CM = 1.0 / 2.54
FULL_W = 16.6 * CM          # MDPI text block, included at \textwidth
HALF_W = 8.1 * CM           # half column, included at 0.49\textwidth

OKABE = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00",
         "#CC79A7", "#999999"]
K, ORANGE, SKY, GREEN, BLUE, VERMILION, PINK, GREY = OKABE
INK = "#222222"             # analytic references
FAINT = "#8c8c8c"           # trivial baseline

# ---- one style per physical object, shared by every figure -----------------
SERIES = {
    "FNO":              dict(c=BLUE,      m="o", ls="-"),
    "U-Net (base)":     dict(c=ORANGE,    m="s", ls="-"),
    "U-Net (matched)":  dict(c=GREEN,     m="^", ls="-"),
    "DeepONet":         dict(c=PINK,      m="v", ls="-"),
    "anchor":           dict(c=VERMILION, m="D", ls="-"),
    "head":             dict(c=VERMILION, m="D", ls="-"),
    "Born":             dict(c=INK,       m="",  ls="--"),
    "trivial":          dict(c=FAINT,     m="",  ls=":"),
}
# local palettes for objects that live in a single figure
ARMS5 = [ORANGE, SKY, GREEN, BLUE, VERMILION]           # contrast arms
MODES6 = [GREY, ORANGE, SKY, GREEN, BLUE, VERMILION]    # retained-mode budgets


def set_style(scale=1.0):
    """Global style.  `scale` exists for the two figures a referee found too
    small: their natural width was half a column, so at the 1:1 inclusion rule
    `fig_widths.py` enforces the labels printed at their nominal size inside a
    8 cm panel.  Enlarging the panel without enlarging the type would leave the
    same complaint, so both move together here and the numeric arguments in
    those two functions are multiplied by the same factor."""
    plt.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9 * scale, "axes.labelsize": 9.5 * scale,
        "axes.titlesize": 9.5 * scale,
        "legend.fontsize": 8 * scale, "xtick.labelsize": 8.5 * scale,
        "ytick.labelsize": 8.5 * scale,
        "axes.linewidth": 0.8 * scale, "lines.linewidth": 1.4 * scale,
        "lines.markersize": 4.2 * scale,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.major.width": 0.8 * scale, "ytick.major.width": 0.8 * scale,
        "xtick.major.size": 3.2 * scale, "ytick.major.size": 3.2 * scale,
        "legend.frameon": False, "savefig.dpi": 600, "figure.dpi": 150,
        "axes.axisbelow": True, "axes.labelpad": 3.0 * scale,
    })


def data_inside_axes(fig, tol=1.0):
    """Plot points that fall outside their axes, in display pixels.

    Every panel draws with ``clip_on=False`` so that a marker sitting on the
    frame is not halved.  The price is that a hand-set limit which is too
    tight does not crop the data, it paints it over the tick labels and the
    neighbouring panel.  A limit is a statement about the data, so it is
    checked here rather than trusted.

    The comparison is made in display coordinates after applying each
    artist's OWN transform: axvline/axhline and legend handles mix data and
    axes coordinates, and comparing raw x/y against get_xlim() reports them
    as violations.
    """
    bad = []
    for k, ax in enumerate(fig.axes):
        # Opt-in per panel.  Most panels carry a deliberately divergent
        # reference curve (the truncated Born series leaves the top of the
        # frame by design), so a figure-wide rule would be all false
        # positives; a panel whose argument depends on the full range of its
        # data declares that with require_inside().
        if not getattr(ax, "_must_contain", False):
            continue
        bb = ax.bbox
        artists = []
        for ln in ax.lines:
            if not ln.get_visible():
                continue
            x = np.asarray(ln.get_xdata(orig=False), dtype=float)
            y = np.asarray(ln.get_ydata(orig=False), dtype=float)
            if x.size == 0:
                continue
            artists.append((x, y, ln.get_transform(),
                            "panel %d line %r" % (k, ln.get_label())))
        for co in ax.collections:
            if not co.get_visible():
                continue
            # errorbar's caps and bars are LineCollections drawn from
            # segments and carry a single placeholder offset at (0, 0);
            # reading that offset reports a phantom point at the origin.
            segs = getattr(co, "get_segments", None)
            if segs is not None:
                sg = [s for s in segs() if len(s)]
                if sg:
                    arr = np.concatenate([np.asarray(s, dtype=float) for s in sg])
                    artists.append((arr[:, 0], arr[:, 1], co.get_transform(),
                                    "panel %d segments" % k))
                    continue
            off = co.get_offsets()
            if off is None or len(off) == 0:
                continue
            arr = np.asarray(off, dtype=float)
            if arr.shape == (1, 2) and not arr.any():
                continue
            artists.append((arr[:, 0], arr[:, 1], co.get_offset_transform(),
                            "panel %d collection" % k))
        for x, y, tr, what in artists:
            if x.size != y.size:
                y = np.resize(y, x.size)
            keep = np.isfinite(x) & np.isfinite(y)
            if int((~keep).sum()):
                bad.append("%s: %d non-finite point(s)"
                           % (what, int((~keep).sum())))
            if not keep.any():
                continue
            disp = tr.transform(np.column_stack([x[keep], y[keep]]))
            out = ((disp[:, 0] < bb.x0 - tol) | (disp[:, 0] > bb.x1 + tol) |
                   (disp[:, 1] < bb.y0 - tol) | (disp[:, 1] > bb.y1 + tol))
            if out.any():
                bad.append("%s: %d of %d point(s) outside the axes"
                           % (what, int(out.sum()), int(keep.sum())))
    return bad


_CONTAINED = []


def require_inside(ax):
    """Declare that every plotted point in this panel must be inside it."""
    ax._must_contain = True
    _CONTAINED.append(ax)


def save(fig, path):
    escaped = data_inside_axes(fig)
    if escaped:
        for msg in escaped[:8]:
            print("   !! data outside axes -- " + msg)
        raise SystemExit("%s: plotted data escapes its axes; fix the limit, "
                         "do not widen clip_on" % os.path.basename(path))
    fig.savefig(path + ".pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(path + ".png", dpi=400, bbox_inches="tight", pad_inches=0.03)
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


def panel_label(ax, letter, text="", dy=-13, fontsize=8.5, weight="bold"):
    """Put the sub-figure identifier (a)(b)... BELOW the panel (scientific
    convention) instead of as a top title, so it never sits on top of the data
    curves.  When the panel carries an x-axis label, the identifier is sunk
    BELOW the whole xlabel band, so the two never share one horizontal line.
    The decision is re-evaluated at DRAW time, so the call order of
    panel_label()/set_xlabel() does not matter.
    """
    s = "(%s)" % letter if not text else "(%s) %s" % (letter, text)
    ann = ax.annotate(s, xy=(0.5, 0.0), xycoords="axes fraction",
                      xytext=(0, dy), textcoords="offset points",
                      ha="center", va="top", fontsize=fontsize,
                      fontweight=weight)

    def _sink(_evt=None):
        off = dy - 13 if ax.get_xlabel().strip() else dy
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
# drawing primitives shared by every figure
# ---------------------------------------------------------------------------
def band(ax, lo=1.0, hi=4.0, shade=True, rule=True):
    """The training band, drawn the same way in every figure."""
    if shade:
        ax.axvspan(lo, hi, color="#7a828c", alpha=0.12, lw=0, zorder=0)
    if rule:
        ax.axvline(hi, color=INK, lw=0.7, ls=(0, (1.1, 1.5)), zorder=1)


def decades(ax, axis="y", numticks=4):
    """Decade ticks only; minor labels suppressed (they collide in narrow
    panels and interleave '6 x 10^-2' between round values)."""
    a = ax.yaxis if axis == "y" else ax.xaxis
    a.set_major_locator(LogLocator(base=10.0, numticks=numticks))
    a.set_minor_formatter(NullFormatter())


def round_ticks(ax, axis, ticks, labels=None):
    a = ax.yaxis if axis == "y" else ax.xaxis
    a.set_major_locator(FixedLocator(ticks))
    if labels is not None:
        a.set_ticklabels(labels)


def despine(ax):
    ax.spines[["top", "right"]].set_visible(False)


def sweep_axis(ax, xlabel="frequency $f$"):
    """Frequency axis common to every sweep panel."""
    ax.set_xlabel(xlabel)
    ax.set_xticks([2, 4, 6, 8, 10, 12])
    ax.set_xlim(1, 12.2)
    despine(ax)


def end_labels(ax, entries, fs=7.4, dx=3.5, min_sep=9.5, bold=True,
               side="right"):
    """Direct series labels just outside an edge of the axes.

    entries: [(y_data, text, colour), ...].  Labels are ordered by height and
    nudged apart in DISPLAY points, so two curves that converge out of band
    still get two readable labels.  The nudge moves the text, never the data.
    side="left" places them at the left edge, which is the right choice when
    the curves are separated where they enter the panel (a purchase curve)
    and coincident where they leave it.
    """
    if not entries:
        return
    entries = sorted(entries, key=lambda e: e[0])
    fig = ax.figure
    fig.canvas.draw()
    tr = ax.transData
    inv = tr.inverted()
    x0 = 0.0 if side == "left" else 1.0
    sgn = -1.0 if side == "left" else 1.0
    ys = [tr.transform((0.0, y))[1] for y, _, _ in entries]
    # min_sep is specified in points (it must stay legible against the fs of
    # the labels), but transData works in display pixels at the canvas dpi --
    # comparing the two directly made every nudge a factor dpi/72 too small,
    # which is how the stacked labels at a converging right edge kept
    # overprinting each other.
    sep_px = min_sep * fig.dpi / 72.0
    for i in range(1, len(ys)):
        if ys[i] - ys[i - 1] < sep_px:
            ys[i] = ys[i - 1] + sep_px
    for (y, text, col), yd in zip(entries, ys):
        back = inv.transform((0.0, yd))[1]
        ax.annotate(text, xy=(x0, back), xycoords=("axes fraction", "data"),
                    xytext=(sgn * dx, 0), textcoords="offset points",
                    fontsize=fs, color=col, va="center",
                    ha="right" if side == "left" else "left",
                    fontweight="bold" if bold else "normal",
                    annotation_clip=False)


def badge(ax, x, y, text, fs=7.0, ha="left", va="top", color="#333333"):
    """Small note that stays legible over both ends of a colour map."""
    ax.text(x, y, text, transform=ax.transAxes, fontsize=fs, ha=ha, va=va,
            color=color, bbox=dict(facecolor="white", alpha=0.78,
                                   edgecolor="none", pad=1.4))


def ratio_arrow(ax, x, y0, y1, text, fs=7.6, color="#333333", side="right",
                lw=0.8):
    """Vertical double arrow between two levels with the ratio beside it."""
    ax.annotate("", xy=(x, y0), xytext=(x, y1),
                arrowprops=dict(arrowstyle="<->", lw=lw, color=color))
    ymid = np.sqrt(y0 * y1) if (y0 > 0 and y1 > 0) else 0.5 * (y0 + y1)
    dx = 4 if side == "right" else -4
    ax.annotate(text, xy=(x, ymid), xytext=(dx, 0),
                textcoords="offset points", fontsize=fs, va="center",
                ha="left" if side == "right" else "right", color=color)


def arm_names(ev, tag):
    return sorted(n for n in ev if n.startswith("main_fno_%s_" % tag))


def born_order1(d, vmax):
    return np.array([r["mean"] for r in d["born_order1"]["%.2f" % vmax]])


# ---------------------------------------------------------------------------
# Figure 1 -- the paired benchmark: the cliff at five contrasts
# ---------------------------------------------------------------------------
def fig_main(evals, out):
    """Error vs frequency, one panel per contrast arm (FNO).

    Laid out 3 + 2 rather than as one strip of five.  Five panels across a
    17 cm column leaves 3.2 cm each, at which the tick labels and the error
    bars interleave; 3 + 2 leaves 5.2 cm and keeps one shared y axis, so the
    comparison the strip exists for -- one cliff, five plateau heights -- is
    still made at a single scale.
    """
    set_style()
    arms = C.VMAX_LIST
    fig = plt.figure(figsize=(FULL_W, FULL_W * 0.62))
    gs = fig.add_gridspec(2, 6, hspace=0.66, wspace=0.20)
    cells = [gs[0, 0:2], gs[0, 2:4], gs[0, 4:6], gs[1, 1:3], gs[1, 3:5]]
    axes, first = [], None
    for k, vmax in enumerate(arms):
        ax = fig.add_subplot(cells[k], sharey=first)
        if first is None:
            first = ax
        axes.append(ax)
        tag = "a%03d" % int(round(vmax * 100))
        names = arm_names(evals, tag)
        if not names:
            continue
        F, M, S = mean_curve(evals, names, vmax)
        b1 = born_order1(evals[names[0]], vmax)
        ax.errorbar(F, M, yerr=S, color=BLUE, marker="o", ms=3.4, lw=1.2,
                    capsize=1.6, clip_on=False, zorder=3)
        ax.plot(F, b1, color=INK, lw=1.1, ls=(0, (4, 2)), clip_on=False,
                zorder=2)
        band(ax)
        # The open ring means one thing across every figure in the paper: the
        # grid node at THIS curve's training-band edge.  The band-width figure
        # rings five different edges, the cross-contrast figure rings the
        # matched contrast's edge and the depth figure rings the residual's;
        # this panel used to ring the node one step PAST the edge instead,
        # which made the same glyph mean "edge" in three figures and "first
        # node past the edge" in the fourth.  The step itself is the arrow in
        # panel (a), and the arrow starts on the ring, so nothing is lost.
        k4 = int(np.argmin(np.abs(np.asarray(F, float) - 4.0)))
        ax.plot([F[k4]], [M[k4]], marker="o", ms=9.0, mfc="none",
                mec=BLUE, mew=1.1, clip_on=False, zorder=4)
        beyond = np.where(np.asarray(F, float) > 4.0 + 1e-9)[0]
        if len(beyond):
            i_out = int(beyond[0])
            # One local mark for the step itself, on the first panel only: an
            # arrow from the edge node to the first node past it, i.e. from the
            # level the operator holds to the level it falls to.
            # No numeral is printed here.  The step is quoted as a range over
            # the five arms and lives in the macro \\CliffLo--\\CliffHi; a
            # figure-local factor would be a second, ungated number for the
            # same quantity.
            if k == 0 and i_out > k4:
                ratio_arrow(ax, float(F[i_out]) + 0.55,
                            float(M[i_out]), float(M[k4]), "step",
                            fs=7.2, color=VERMILION)
        ax.set_yscale("log")
        ax.set_ylim(0.008, 4.5)
        decades(ax, numticks=3)
        sweep_axis(ax)
        panel_label(ax, "abcde"[k], r"$v_{\max}=%.2f$" % vmax)
        if k % 3 == 0:
            ax.set_ylabel(r"relative $L_2$ error")
        else:
            ax.tick_params(labelleft=False)
    end_labels(axes[2], [(float(M[-1]), "FNO", BLUE),
                         (float(b1[-1]), r"Born $n{=}1$", INK)])
    save(fig, os.path.join(out, "fig_main_cliff"))


# ---------------------------------------------------------------------------
# Figure -- architecture families at matched budget
# ---------------------------------------------------------------------------
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
    set_style()
    # Labels come from common.ARCH_GROUPS, the same single definition that
    # fills the architecture table, so the end labels and the table rows
    # cannot disagree.
    styles = [(BLUE, "o"), (ORANGE, "s"), (GREEN, "^"), (PINK, "v")]
    series = [(p % "a025", p % "a100", lab, col, mk)
              for (p, lab), (col, mk) in zip(C.ARCH_GROUPS, styles)]
    # DeepONet gets its own dash pattern: three of the four families are drawn
    # with the same solid line, so in a greyscale reproduction (and for a
    # red-green colour-blind reader, PINK against GREEN) the only cue left was
    # the marker.  The line style is a second, colour-independent channel.
    def line_of(lab):
        return (0, (5.0, 1.6, 1.2, 1.6)) if "DeepONet" in lab else "-"
    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, FULL_W * 0.40),
                             layout="constrained")
    for k, (ax, vmax) in enumerate(zip(axes[:2], (0.25, 1.00))):
        for pref0, pref1, lab, col, mk in series:
            pref = pref0 if abs(vmax - 0.25) < 1e-9 else pref1
            names = sorted(n for n in evals if n.startswith(pref))
            if not names:
                continue
            F, M, S = mean_curve(evals, names, vmax)
            ax.errorbar(F, M, yerr=S, color=col, marker=mk, ms=3.0, lw=1.2,
                        ls=line_of(lab), capsize=1.6, clip_on=False, zorder=3)
        band(ax)
        ax.set_yscale("log")
        # Same y range as Fig. 1, so the plateau heights of the two figures can
        # be read against each other without a mental rescaling.  The upper
        # limit catches the strongest-contrast Born reference, which is the
        # topmost point drawn here; an earlier cap clipped it out of frame.
        ax.set_ylim(0.008, 9.0)
        decades(ax, numticks=3)
        sweep_axis(ax)
        panel_label(ax, "ab"[k], r"$v_{\max}=%.2f$" % vmax)
        if k == 0:
            ax.set_ylabel(r"relative $L_2$ error")
        else:
            ax.tick_params(labelleft=False)

    # ---- panel (c): each curve divided by its own plateau ------------------
    axn = axes[2]
    for pref0, pref1, lab, col, mk in series:
        names = sorted(n for n in evals if n.startswith(pref0))
        if not names:
            continue
        F, M, S = mean_curve(evals, names, 0.25)
        pl = M[F >= 6.0]
        if not pl.size or pl.mean() <= 0:
            continue
        # Each curve is divided by ITS OWN plateau, so the seed spread has to
        # be divided by the same number; carrying the raw spread here would
        # make the band wider for the family with the higher plateau, which is
        # the opposite of what the normalised panel compares.
        axn.errorbar(F, M / pl.mean(), yerr=S / pl.mean(), color=col,
                     marker=mk, ms=2.6, lw=1.2, ls=line_of(lab), capsize=1.4,
                     clip_on=False, zorder=3)
    axn.axhline(1.0, color="#666666", lw=0.8, ls=(0, (1.1, 1.5)), zorder=1)
    # Upper left: at the right edge of (c) the four curves converge on the
    # rule, so a label placed there printed on top of them.
    axn.annotate("own plateau $=1$", xy=(2.05, 1.62), fontsize=7.0,
                 color="#333333", va="top", ha="left")
    band(axn)
    axn.set_ylim(0.0, 1.75)
    panel_label(axn, "c", r"$v_{\max}=0.25$, own plateau")
    sweep_axis(axn)
    axn.set_ylabel(r"$\varepsilon(f)\,/\,\varepsilon_{\infty}$")
    # The four families are separated at the right edge of every panel, so the
    # legend row above the strip is replaced by direct labels -- one per panel
    # would repeat four names three times, and one panel's labels would not
    # name the curves drawn in the other two.
    handles = [Line2D([], [], color=col, marker=mk, ls=line_of(lab), lw=1.2,
                      ms=3.0)
               for _, _, lab, col, mk in series]
    labels = [lab for _, _, lab, _, _ in series]
    fig.legend(handles, labels, loc="lower center", ncol=len(handles),
               frameon=False, fontsize=8, handlelength=2.0,
               columnspacing=1.8, bbox_to_anchor=(0.5, 1.01))
    save(fig, os.path.join(out, "fig_arch"))


# ---------------------------------------------------------------------------
# Figure -- training-band width: the gain sits just beyond the new edge
# ---------------------------------------------------------------------------
def fig_band(evals, out):
    # 0.78 of the text block at 1:1 (fig_widths derives the fraction from the
    # MediaBox), with the type 1.2x: five curves with end labels were crowded
    # in a half-column panel at nominal size.
    Z = 1.2
    set_style(Z)
    fig, ax = plt.subplots(figsize=(FULL_W * 0.78, FULL_W * 0.47),
                           layout="constrained")
    specs = [("A025", "band_a025", 4.0, ARMS5[0]),
             ("A025_b2", "band_b2", 2.0, ARMS5[1]),
             ("A025_b3", "band_b3", 3.0, ARMS5[2]),
             ("A025_b6", "band_b6", 6.0, ARMS5[3]),
             ("A025_b12", "band_b12", 12.0, ARMS5[4])]
    ends = []
    for lab, pref, fedge, col in specs:
        if pref == "band_a025":
            names = arm_names(evals, "a025")
        else:
            names = sorted(n for n in evals if n.startswith(pref))
        if not names:
            continue
        F, M, S = mean_curve(evals, names, 0.25)
        ax.errorbar(F, M, yerr=S, color=col, marker="o", ms=3.2, lw=1.3,
                    capsize=1.6, clip_on=False, zorder=3)
        # Each arm has its own training edge, so a single shared vertical rule
        # would mark only one of the five.  Marking the edge with a ring on
        # the arm's own curve locates it without five full-height rules
        # cutting the panel into stripes.  The edge frequency is printed next
        # to the ring: five rings at five heights are otherwise the same
        # glyph, and the reader has to count rings against the legend.
        k = int(np.argmin(np.abs(F - fedge)))
        ax.plot([F[k]], [M[k]], marker="o", ms=9.0, mfc="none", mec=col,
                mew=1.3, clip_on=False, zorder=4)
        # No number beside the ring: the right-edge label already reads
        # "train [1,N]", so a second copy of N inside the plot only collided
        # with the five curves that cross at f = 4 (measured: the "4" glyph sat
        # on three of them).  The ring marks WHERE the edge is, the end label
        # says WHICH edge it is.
        ends.append((float(M[-1]), r"train $[1,%.0f]$" % fedge, col))
    band(ax)
    # The shading is the REFERENCE band, not the band of every curve: the
    # figure's whole argument is that each curve has a different one.  Saying
    # so on the patch itself keeps the grey from being read as "the band".
    ax.annotate(r"reference band $[1,4]$", xy=(1.05, 0.0105), fontsize=7.0 * Z,
                color="#555555", ha="left", va="bottom",
                bbox=dict(boxstyle="round,pad=0.10", fc="white", ec="none",
                          alpha=0.75))
    ax.set_yscale("log")
    ax.set_ylim(0.008, 4.5)
    decades(ax, numticks=3)
    sweep_axis(ax)
    ax.set_ylabel(r"relative $L_2$ error")
    end_labels(ax, ends, fs=7.0 * Z, min_sep=10.5 * Z)
    badge(ax, 0.985, 0.035, r"ring: train band edge", ha="right", va="bottom",
          fs=7.4 * Z)
    save(fig, os.path.join(out, "fig_bandwidth"))


# ---------------------------------------------------------------------------
# Figure -- heat map over (retained modes M, frequency f)
# ---------------------------------------------------------------------------
def fig_mode_phase(evals, out, band_tag, train_edge, fname):
    set_style()
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
    # Half a column is too narrow for a 6 x 15 matrix with the values printed
    # in every cell: at the old 8.1 cm natural width a 5.8 pt value sat inside
    # an ~10 pt cell.  The figure is emitted wider instead, and fig_widths
    # derives the larger inclusion fraction from the MediaBox, so the 1:1
    # type-size rule still holds.
    # Full text width: 15 frequency columns of a four-character value need
    # ~31 pt per cell; at the previous 10.4 cm the cells were 19 pt wide and
    # neighbouring values printed into each other ("0.860.83").
    fig, ax = plt.subplots(figsize=(FULL_W, FULL_W * 0.56),
                           layout="constrained")
    vmax_c = np.nanmax(Mat)
    im = ax.imshow(Mat, cmap="viridis", aspect="auto", origin="lower",
                   vmin=np.nanmin(Mat), vmax=vmax_c)
    # Cell borders: without them adjacent cells of similar value merge into
    # one blotch and the M direction of the map stops being readable.
    ax.set_xticks(np.arange(-0.5, len(F), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(modes), 1), minor=True)
    ax.grid(which="minor", color="w", lw=0.5)
    ax.tick_params(which="minor", length=0)
    ax.set_xticks(range(len(F)))
    ax.set_xticklabels(["%.2g" % f for f in F], rotation=45, ha="right",
                       fontsize=8)
    ax.set_yticks(range(len(modes)))
    ax.set_yticklabels([str(m) for m in modes])
    ax.set_xlabel("frequency $f$")
    ax.set_ylabel("retained modes $M$")
    xb = np.searchsorted(F, train_edge - 1e-9)
    # The edge rule is drawn twice: a white dash on the viridis fill (visible
    # over the dark low-error cells) and a thin black hairline underneath it
    # (visible over the bright high-error cells).  A single white rule
    # disappeared in the top-right corner of the map, which is exactly where
    # the reader checks whether the step is at f=4 or beyond it.
    ax.axvline(xb - 0.5, color="k", lw=2.0, ls=(0, (4, 2)), zorder=4)
    ax.axvline(xb - 0.5, color="w", lw=1.0, ls=(0, (4, 2)), zorder=5)
    # Above the axes, not inside: rotated inside, the label sat on top of the
    # cell annotations it is there to help read.  The gap is 7 pt rather than
    # 3 pt so that the 8 pt label clears the top row of cells at print size.
    ax.annotate(r"training-band edge $f=%g$" % train_edge,
                xy=(xb - 0.5, 1.0),
                xycoords=("data", "axes fraction"),
                xytext=(0, 7), textcoords="offset points",
                ha="center", va="bottom", fontsize=8, color="#333333",
                annotation_clip=False)
    for i, m in enumerate(modes):
        for j in range(len(F)):
            if np.isnan(Mat[i, j]):
                continue
            ax.text(j, i, "%.2f" % Mat[i, j], ha="center", va="center",
                    fontsize=9.4,
                    color="k" if Mat[i, j] > 0.5 * vmax_c else "w")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.outline.set_linewidth(0.5)
    cb.ax.tick_params(labelsize=7.5)
    # The bar's own maximum is printed as a tick: an auto-located top tick sat
    # below the largest cell (1.11) and a reader checking the scale against the
    # hottest cell had no tick to check it against.
    _lo, _hi = float(np.nanmin(Mat)), float(vmax_c)
    _ticks = sorted({round(_lo + (_hi - _lo) * k / 4.0, 2) for k in range(5)}
                    | {round(_hi, 2)})
    cb.set_ticks(_ticks)
    cb.set_ticklabels(["%.2f" % t for t in _ticks])
    cb.set_label(r"relative $L_2$ error", fontsize=8)
    save(fig, os.path.join(out, fname))


# ---------------------------------------------------------------------------
# Figure -- onset collapse: the rise begins at the same relative frequency
# ---------------------------------------------------------------------------
def fig_scaling(evals, out):
    """Where the rise starts, and how far past the plateau it goes.

    Dividing each arm by its own plateau removes the contrast dependence of
    the height by construction, so the collapse that survives is a statement
    about the *onset*: the rise begins at the same relative frequency for
    every contrast.  It is deliberately not drawn as a collapse onto one step
    -- the high-contrast arms overshoot their plateau at f/f_train ~ 1.25 and
    relax back by ~1.75, and an axis that stops just above 1.0 (as this one
    used to) renders the overshoot outside the frame, where it reads as noise
    rather than as the non-monotonicity it is.
    """
    set_style()
    fig, ax = plt.subplots(figsize=(HALF_W, HALF_W * 0.95), layout="constrained")
    for i, vmax in enumerate(C.VMAX_LIST):
        tag = "a%03d" % int(round(vmax * 100))
        names = arm_names(evals, tag)
        if not names:
            continue
        F, M, S = mean_curve(evals, names, vmax)
        plateau = float(M[F >= 6].mean())
        # Light seed band, same colours as Figures 3 and 4: five curves on one
        # panel need a spread, but a full errorbar at every node hid the
        # overshoot the figure exists to show.
        ax.fill_between(F / 4.0, (M - S) / plateau, (M + S) / plateau,
                        color=ARMS5[i], alpha=0.16, lw=0, zorder=1)
        ax.plot(F / 4.0, M / plateau, color=ARMS5[i], marker="o", ms=3.0,
                lw=1.3, clip_on=False, zorder=3,
                label=r"$v_{\max}=%.2f$" % vmax)
    ax.axvspan(0.25, 1.0, color="#7a828c", alpha=0.12, lw=0, zorder=0)
    ax.axvline(1.0, color=INK, lw=0.7, ls=(0, (1.1, 1.5)), zorder=1)
    ax.axhline(1.0, color="#666666", lw=0.8, ls=(0, (1.1, 1.5)), zorder=1)
    ax.set_xscale("log")
    ax.set_xlim(0.4, 3.2)
    # Plain tick labels: the default log formatter printed "4 x 10^-1" and
    # "2 x 10^0" under an axis that spans barely one decade.
    ax.set_xticks([0.5, 1, 1.5, 2, 3])
    ax.set_xticklabels(["0.5", "1", "1.5", "2", "3"])
    # The minor locator still labels 4e-1 and 6e-1 in mantissa notation on
    # top of the plain ticks unless it is silenced explicitly.
    ax.xaxis.set_minor_formatter(NullFormatter())
    # Headroom for the overshoot.  The previous cap of 1.25 sat below the peak
    # of three of the five arms.
    ax.set_ylim(0.0, 1.7)
    ax.set_xlabel(r"relative frequency $f/f_{\mathrm{train}}$")
    ax.set_ylabel(r"$\varepsilon(f)/\varepsilon_{\infty}$")
    despine(ax)
    # All five curves converge to the same plateau, so right-edge direct
    # labels stacked on one point collided; the region below the plateau on
    # the right is empty, which is where the legend goes instead.
    ax.legend(fontsize=8, loc="lower right", frameon=False,
              handlelength=1.6, labelspacing=0.32)
    save(fig, os.path.join(out, "fig_scaling"))
    return {}


# ---------------------------------------------------------------------------
# Figure -- out-of-band energy fraction per retained-mode budget
# ---------------------------------------------------------------------------
def fig_spectrum(out):
    set_style()
    p = os.path.join(C.RES, "spectrum.json")
    if not os.path.exists(p):
        print("no spectrum.json"); return
    d = json.load(open(p, encoding="utf-8"))
    modes = d["modes"]; F = d["f"]
    fig, ax = plt.subplots(figsize=(HALF_W, HALF_W * 0.92), layout="constrained")
    # M = 32 is excluded for the same reason as in the mode heat map: on the
    # 64x64 grid it spans every representable mode, so delta_32 == 0
    # identically and a log axis has nothing to display.  The floor below is
    # a rendering guard only -- it never bites for a non-vacuous budget.
    shown = [2, 4, 8, 12, 20, 24]
    ends = []
    for i, M in enumerate(shown):
        y = np.maximum(np.array([d["data"]["0.25"]["f=%g" % f]["psi"][str(M)]
                                 for f in F], dtype=float), 1e-6)
        ax.semilogy(F, y, color=MODES6[i], marker="o", ms=2.8, lw=1.2,
                    clip_on=False, zorder=3)
        ends.append((float(y[-1]), r"$M=%d$" % M, MODES6[i]))
    band(ax)
    sweep_axis(ax)
    ax.set_ylabel(r"out-of-band energy fraction $\delta_M$")
    end_labels(ax, ends, fs=7.0, min_sep=10.5)
    save(fig, os.path.join(out, "fig_spectrum"))


# ---------------------------------------------------------------------------
# Figure (SI) -- field panels: reference, prediction, corrections
# ---------------------------------------------------------------------------
def fig_fields(out, device=None):
    """Reference, prediction, and the two corrections, in and out of band.

    The bottom two rows exist to decide a mechanism rather than to illustrate
    one.  A plateau at the incident-field baseline can come from a correction
    that has collapsed toward zero or from one that is present but
    phase-incoherent; comparing |u_hat| against |u| on a SHARED colour scale
    separates them by eye, which the pointwise error panel it replaced could
    not.

    Forced onto the CPU: the figure needs four forward passes of a 64x64 FNO,
    which is seconds of CPU work, and allocating a CUDA context here would
    evict a concurrent training job from the only GPU.
    """
    import torch
    import physics as P
    import data as D
    device = "cpu"
    ck = os.path.join(C.RUNS, "main_fno_a025_s0", "checkpoint.pt")
    if not os.path.exists(ck):
        print("no ckpt for fields"); return
    from evaluate import load_model
    model, _ = load_model(ck, device)
    set_style()
    freqs = [1.5, 4.0, 6.0, 12.0]
    n = 4

    # Pass 1: the fields themselves.  Ground truth and prediction are the same
    # physical quantity and must therefore share one colour scale; giving each
    # panel its own scale would make a field that is uniformly wrong look like
    # a field that is uniformly right.
    got = []
    for f in freqs:
        d = D.sample_batch(n, 60000, device, vmax_lo=0.25, vmax_hi=0.25,
                           f_fixed=f, return_info=True)
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
    fig, axes = plt.subplots(4, len(freqs), figsize=(FULL_W, FULL_W * 0.80))
    fig.subplots_adjust(wspace=0.05, hspace=0.05, right=0.87)
    rows = [("reference $\\psi$", "RdBu_r", v_field, -v_field, lambda g: g[1].real),
            ("FNO $\\hat\\psi$", "RdBu_r", v_field, -v_field, lambda g: g[2].real),
            ("pred. corr. $|\\hat u|$", "magma", v_corr, 0.0,
             lambda g: np.abs(g[3])),
            ("true corr. $|u|$", "magma", v_corr, 0.0, lambda g: np.abs(g[4]))]
    for j, g in enumerate(got):
        f = g[0]
        band_tag = "in band" if f <= 4.0 else "out of band"
        for r, (ttl, cmap, vmax_c, vmin_c, pick) in enumerate(rows):
            ax = axes[r, j]
            ax.imshow(pick(g), cmap=cmap, vmin=vmin_c, vmax=vmax_c,
                      origin="lower")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_linewidth(0.6)
            if r == 0:
                ax.set_title(r"$f=%.1f$ (%s)" % (f, band_tag), fontsize=8)
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
                        ha="right", va="top", fontsize=8,
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
        cb.ax.tick_params(labelsize=7.5)
        cb.set_label(lab, fontsize=8, labelpad=2)
    save(fig, os.path.join(out, "fig_fields"))


# ---------------------------------------------------------------------------
# Figure -- measured error against the two reference scales
# ---------------------------------------------------------------------------
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
    set_style()
    p = os.path.join(C.RES, "spectrum.json")
    if not os.path.exists(p):
        print("no spectrum.json"); return
    d = json.load(open(p, encoding="utf-8"))
    F = np.array(d["f"], float)
    spec = d["data"]["0.25"]
    rows = [spec["f=%g" % f] for f in F]
    have_u = "u_out_rel" in rows[0]
    cols = MODES6
    budgets = [2, 4, 8, 12, 20, 24]
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, FULL_W * 0.42),
                             sharex=True, layout="constrained")
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
                ax.errorbar(Fv, Mv, yerr=Sv, color=col, marker="o", ms=2.8,
                            lw=1.1, capsize=1.2, clip_on=False, zorder=3)
            any_ = True
    for ax in axes:
        band(ax)
        ax.set_yscale("log")
        ax.set_xlim(1, 12.3)
        sweep_axis(ax)
        ax.set_ylabel(r"relative $L_2$ error")
        decades(ax, numticks=4)
    axes[0].set_ylim(0.03, 5.0)
    panel_label(axes[0], "a", fontsize=8)
    axes[1].set_ylim(1e-5, 3.0)
    panel_label(axes[1], "b", fontsize=8)
    if any_:
        # Two-level legend: colour carries the mode budget, line style carries
        # the physical quantity.  The single row of seven entries this replaces
        # forced the reader to decode both mappings from one string.
        hM, lM = [], []
        for col, Mb in zip(cols, budgets):
            hM.append(Line2D([], [], color=col, lw=2.2))
            lM.append(r"$M=%d$" % Mb)
        hQ = [Line2D([], [], color="k", ls="-", lw=1.3),
              Line2D([], [], color="k", ls="--", lw=1.1),
              Line2D([], [], color="k", ls=":", lw=1.6)]
        lQ = ["measured model error",
              "reference out-of-band fraction",
              "correction out-of-band energy"]
        fig.legend(hM, lM, fontsize=8, ncol=6, loc="lower center",
                   bbox_to_anchor=(0.5, 1.075), handlelength=1.6,
                   columnspacing=1.0, frameon=False,
                   title="retained modes", title_fontsize=8)
        fig.legend(hQ, lQ, fontsize=8, ncol=3, loc="lower center",
                   bbox_to_anchor=(0.5, 1.005), handlelength=2.6,
                   columnspacing=1.4, frameon=False)
    save(fig, os.path.join(out, "fig_bound"))


# ---------------------------------------------------------------------------
# Figure -- solver validation: dense cross-check and the freezing ablation
# ---------------------------------------------------------------------------
def fig_solver(out):
    set_style()
    p = os.path.join(C.RES, "solver_fig.json")
    if not os.path.exists(p):
        print("no solver_fig.json"); return
    d = json.load(open(p, encoding="utf-8"))
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, FULL_W * 0.42),
                             layout="constrained")

    # (a) dense direct solve vs batched Bi-CGSTAB on a 10x10 grid
    ax = axes[0]
    diffs = np.array(d["dense"]["diffs"])
    x = np.arange(1, len(diffs) + 1)
    ax.bar(x, diffs, color=BLUE, width=0.62, zorder=3)
    ax.axhline(1e-12, color=VERMILION, ls="--", lw=1.0, zorder=2)
    ax.set_yscale("log")
    ax.set_ylim(max(1e-15, diffs.min() / 3), max(1e-11, diffs.max() * 3))
    ax.set_xticks(x)
    ax.set_xlabel("test medium")
    ax.set_ylabel(r"rel. difference vs dense solve")
    decades(ax, numticks=6)
    v = d["dense"]["max_diff"]
    e = int(np.floor(np.log10(v)))
    m = v / (10.0 ** e)
    badge(ax, 0.97, 0.94, r"max $=%.1f\times10^{%d}$" % (m, e), ha="right")
    badge(ax, 0.97, 0.80, r"$10^{-12}$ ref.", ha="right", color=VERMILION)
    # The comparison is only a cross-check if the two solves share no code
    # path; the dense solve is a direct LU factorisation, stated in the panel
    # title rather than as a badge -- a badge in this corner collided with the
    # "max" annotation, which is the number the panel exists to report.
    panel_label(ax, "a", r"$10\times10$ grid, dense cross-check (direct LU)")
    despine(ax)

    # (b) freezing ablation: three implementations of the shared loop
    ax = axes[1]
    conds = d["freeze"]["conds"]
    idx = np.arange(len(conds))
    naive = np.array([100.0 * c["naive"] / c["n"] for c in conds])
    zmask = np.array([100.0 * c["zeromask"] / c["n"] for c in conds])
    frozen = np.array([100.0 * c["frozen"] / c["n"] for c in conds])
    sil = np.array([100.0 * c["silent"] / c["n"] for c in conds])
    # STACKED in principle, but the "detected" segment is measured at exactly
    # zero on every batch: the in-loop check found none of the corrupted
    # labels.  Drawing a zero-height second segment would look like an
    # omission, so the bar is the silent corruption and the undetected part is
    # stated in words.  The containment (silent <= total, equal here) is the
    # finding, not a plotting choice.
    det = np.clip(naive - sil, 0.0, None)
    ax.bar(idx, sil, width=0.66, color=PINK, zorder=3,
           label="silent corrupted labels")
    ax.axhline(0.0, color=GREEN, lw=2.2, zorder=4)
    # Both frozen variants measure identically zero; the labels carry the
    # value so the entry cannot be read as "condition missing".
    ax.plot([], [], color=GREEN, lw=2.2,
            label="frozen variants: max %.1f%% and %.1f%%"
                  % (float(frozen.max()), float(zmask.max())))
    ax.set_xticks(idx)
    ax.set_xticklabels(["%d" % (c["batch"] + 1) for c in conds])
    ax.set_xlabel("batch index")
    ax.set_ylabel("% corrupted labels")
    ax.set_ylim(0, max(naive.max() * 1.30, 1.0))
    # Both zero categories are stated rather than drawn: an invisible bar
    # leaves the reader guessing whether the condition was missing or measured
    # at zero.  The wording gives the maximum over batches, which stays true if
    # a batch ever becomes non-zero -- an "is zero" claim would go stale.
    badge(ax, 0.03, 0.955,
          "in-loop check detected %.1f%%" % float(det.max()),
          ha="left", fs=6.9, color=GREEN)
    # Legend row ABOVE the panel: below the axes it used to sit on top of the
    # "(b)" panel identifier, which shares that band.  Two columns, not three:
    # at 8 pt the three labels in one row were wider than the text block, and
    # because the PDF is emitted with bbox_inches="tight" that showed up as a
    # figure wider than \textwidth rather than as an overlong legend.
    ax.legend(fontsize=7.8, loc="lower center", bbox_to_anchor=(0.5, 1.02),
              ncol=2, handlelength=1.4, columnspacing=1.2, labelspacing=0.3)
    panel_label(ax, "b", r"%d heterogeneous batches" % len(conds))
    despine(ax)
    save(fig, os.path.join(out, "fig_solver"))


# ---------------------------------------------------------------------------
# Figure (SI) -- training curves: the in-band floor is contrast-sensitive
# ---------------------------------------------------------------------------
def fig_training(out):
    set_style()
    fig, ax = plt.subplots(figsize=(HALF_W, HALF_W * 0.95), layout="constrained")
    any_ = False
    ends = []
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
                            color=ARMS5[i], alpha=0.16, lw=0, zorder=1)
        ax.semilogy(ep, A.mean(axis=0), color=ARMS5[i], marker="o",
                    markevery=max(1, n // 8), ms=3.0, lw=1.3, zorder=3,
                    label=r"$v_{\max}=%.2f$ ($n{=}%d$)" % (vmax, len(seeds)))
        ends.append((float(A.mean(axis=0)[-1]),
                     r"$v_{\max}=%.2f$" % vmax, ARMS5[i]))
        any_ = True
    if not any_:
        print("no training logs"); return
    ax.set_xlabel("epoch")
    ax.set_ylabel("in-band validation rel. $L_2$")
    # The span of this axis is under two decades, so the default log locator
    # labels every mantissa and the tick column becomes unreadable.  Round
    # values only, spanning the actual data range.
    ax.set_ylim(0.015, 0.8)
    round_ticks(ax, "y", [0.02, 0.05, 0.1, 0.2, 0.4],
                ["0.02", "0.05", "0.10", "0.20", "0.40"])
    ax.yaxis.set_minor_formatter(NullFormatter())
    despine(ax)
    end_labels(ax, ends, fs=7.0)
    badge(ax, 0.03, 0.05, "band: seed span", ha="left", va="bottom")
    save(fig, os.path.join(out, "fig_training"))


# ---------------------------------------------------------------------------
# Figure -- cross-contrast evaluation: the cliff is read arm-matched
# ---------------------------------------------------------------------------
def fig_crosscontrast(evals, out):
    """Why a cliff ratio must be read arm-matched.

    Each trained model already carries a full sweep over all five evaluation
    contrasts, so the cross-contrast reading costs nothing extra.  It shows
    that the cliff is a property of a model measured *at the contrast it was
    trained on*: evaluate the weak-contrast model at strong contrast and the
    error is already at plateau level inside the training band, so the band
    edge leaves no signature at all.
    """
    set_style()
    pairs = [("main_fno_a005_s0", 0.05, r"trained at $v_{\max}=0.05$"),
             ("main_fno_a100_s0", 1.00, r"trained at $v_{\max}=1.00$")]
    pairs = [(n, v, t) for n, v, t in pairs if n in evals]
    if not pairs:
        print("no cross-contrast models available"); return
    fig, axes = plt.subplots(1, len(pairs), figsize=(FULL_W, FULL_W * 0.40),
                             sharex=True, sharey=True, layout="constrained")
    if len(pairs) == 1:
        axes = [axes]
    for k, (ax, (name, own, ttl)) in enumerate(zip(axes, pairs)):
        d = evals[name]
        for i, vmax in enumerate(C.VMAX_LIST):
            F, M, S = curve(d, vmax)
            own_curve = abs(vmax - own) < 1e-9
            ax.errorbar(F, M, color=ARMS5[i], marker="o", ms=3.0,
                        lw=2.4 if own_curve else 1.2,
                        zorder=4 if own_curve else 3, clip_on=False,
                        label=r"eval $v_{\max}=%.2f$" % vmax)
            if own_curve:
                # The arm's own contrast is the one the band edge is read on;
                # ring its band-edge point so the reader finds it without a
                # second legend entry.
                k4 = int(np.argmin(np.abs(F - 4.0)))
                ax.plot([F[k4]], [M[k4]], marker="o", ms=10, mfc="none",
                        mec=ARMS5[i], mew=1.4, zorder=5, clip_on=False)
        band(ax)
        ax.set_yscale("log")
        ax.set_ylim(0.012, 5.0)
        decades(ax, numticks=3)
        sweep_axis(ax)
        panel_label(ax, "ab"[k], ttl, fontsize=8.5)
        if k:
            ax.tick_params(labelleft=False)
    axes[0].set_ylabel(r"relative $L_2$ error")
    # One legend row for the strip: the eval contrast is the same set in both
    # panels, and the previous per-panel legends repeated five entries twice
    # inside the data region.  Short labels on purpose -- the long form made
    # the strip wider than the text block, which forced a down-scale on
    # inclusion and put the type below print size.
    h = [Line2D([], [], color=ARMS5[i], marker="o", ls="-", lw=1.2, ms=3.0)
         for i in range(len(C.VMAX_LIST))]
    l = [r"$v_{\max}=%.2f$" % v for v in C.VMAX_LIST]
    h.append(Line2D([], [], color="#555555", marker="o", ls="-", lw=2.4,
                    ms=3.0))
    l.append("train-matched")
    h.append(Line2D([], [], color="none", marker="o", ms=8, mfc="none",
                    mec="#555555", mew=1.3))
    l.append("band edge")
    fig.legend(h, l, fontsize=7.6, ncol=len(h), loc="lower center",
               bbox_to_anchor=(0.5, 1.005), handlelength=1.4,
               columnspacing=0.8, labelspacing=0.3,
               title="eval contrast", title_fontsize=7.6)
    save(fig, os.path.join(out, "fig_crosscontrast"))


# ---------------------------------------------------------------------------
# Figure -- conditioning confound: the cliff survives re-conditioning
# ---------------------------------------------------------------------------
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
          test frequencies.  The baseline's non-monotone spread collapses to
          a constant that sits below the baseline's band-edge value, i.e. the
          compensated arm is the better-conditioned one at every frequency.
    Everything is read from results/c1_verdict.json (written by
    code/analyze_c1.py), so the figure and the table cannot drift apart.
    No GPU: the curves are evaluation records plus a 64x64 symbol norm.
    """
    set_style()
    p = os.path.join(C.RES, "c1_verdict.json")
    if not os.path.exists(p):
        print("no c1_verdict.json -- run code/analyze_c1.py"); return
    d = json.load(open(p, encoding="utf-8"))
    v = d["verdict"]
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, FULL_W * 0.42),
                             layout="constrained")
    specs = [("baseline", BLUE, "o",
              r"baseline, $\gamma=0.05$ ($%.2f$--$%.2f$)"
              % (v["attained_baseline_lo"], v["attained_baseline_hi"])),
             ("compensated", ORANGE, "s",
              r"compensated, $\gamma(f)$ ($%.2f$ const.)"
              % v["attained_compensated"])]

    def _fl(x):
        # 4.0 must not print as "4": the whole argument rests on the 0.25
        # grid spacing at the band edge.
        return "%.1f" % x if abs(x - round(x)) < 1e-9 else "%.2f" % x

    # ---- (a) measured error curves
    ax = axes[0]
    for key, col, mk, lab in specs:
        arm = d["arms"][key]
        ax.errorbar(arm["f"], arm["mean"], yerr=arm["sd"], color=col,
                    marker=mk, ms=3.4, lw=1.3, capsize=2, clip_on=False,
                    zorder=3)
    lo, hi = d["arms"]["baseline"]["onset"]
    # The onset interval is a property of the grid, so it is drawn where it
    # lives -- at the band edge -- and not parked in a corner of the panel.
    ax.axvspan(lo, hi, color=PINK, alpha=0.30, lw=0, zorder=0)
    band(ax)
    ax.set_yscale("log")
    ax.set_ylim(0.008, 4.5)
    decades(ax, numticks=3)
    sweep_axis(ax)
    ax.set_ylabel(r"relative $L_2$ error")
    end_labels(ax, [(float(d["arms"]["baseline"]["mean"][-1]),
                     "baseline", BLUE),
                    (float(d["arms"]["compensated"]["mean"][-1]),
                     "compensated", ORANGE)], fs=7.2)
    badge(ax, 0.5, 0.955,
          r"onset $[%s,%s]$, $\pm1$ SD, %d seeds -- identical in both arms"
          % (_fl(lo), _fl(hi), d["arms"]["baseline"]["seeds"]),
          ha="center", fs=6.8)
    panel_label(ax, "a")

    # ---- (b) the manipulated quantity: attained operator norm
    ax = axes[1]
    F = d["arms"]["baseline"]["f"]
    for key, col, mk, lab in specs:
        ax.plot(F, d["arms"][key]["attained"], color=col, marker=mk, ls="-",
                lw=1.3, clip_on=False, zorder=3)
    band(ax)
    ax.set_ylim(2.0, 11.0)
    sweep_axis(ax)
    ax.set_ylabel(r"attained $\max|\kappa^{2}\widehat{G}_{\kappa}|$")
    end_labels(ax, [(float(d["arms"]["baseline"]["attained"][-1]),
                     "baseline", BLUE),
                    (float(d["arms"]["compensated"]["attained"][-1]),
                     "compensated", ORANGE)], fs=7.2)
    # The spread->constant statement lives in the caption: inside the panel it
    # collided with the compensated arm, which is a flat line at the only
    # height the note could occupy.
    # The two panels share the colour code; (b) states it in a legend because
    # its two series are the point of the panel (what the compensation did to
    # the norm) and a reader should not have to carry the code across panels.
    ax.legend([Line2D([], [], color=BLUE, marker="o", ls="-", lw=1.3),
               Line2D([], [], color=ORANGE, marker="s", ls="-", lw=1.3)],
              ["baseline", "compensated"], fontsize=7.4, loc="upper left",
              frameon=False, handlelength=1.8, labelspacing=0.3)
    panel_label(ax, "b")
    save(fig, os.path.join(out, "fig_c1"))


# ---------------------------------------------------------------------------
# Figure -- the analytic anchor: no cliff out of band, at every contrast
# ---------------------------------------------------------------------------
def fig_anchor(out):
    """The analytic anchor against the trained operator (Section sec:anchor).

    Two panels, because the claim is not "the anchor is more accurate" but
    "the anchor has no cliff", and a single accuracy curve cannot show that:
      (a) the four predictors at v_max = 0.25: the trained FNO (seed mean,
          one seed SD, from the evaluation records), the anchor (two complex
          coefficients fitted on the training cache), the raw first-order Born
          series, and the trivial baseline that returns the incident field.
          The trained curve steps at the band edge; the anchor does not.
      (b) the plateau against contrast for the same three learned/analytic
          objects, which is where the complementarity lives: the trained
          operator wins inside the band and the anchor wins outside it.
    Read from results/anchor_baseline.json and the evaluation records, so the
    figure and the anchor table cannot drift apart.  No GPU.
    """
    set_style()
    p = os.path.join(C.RES, "anchor_baseline.json")
    if not os.path.exists(p):
        print("no anchor_baseline.json -- run code/anchor_baseline.py"); return
    d = json.load(open(p, encoding="utf-8"))
    ev = load_evals()
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, FULL_W * 0.42),
                             layout="constrained")
    F = np.array(d["f_test"])

    # ---- (a) v_max = 0.25
    ax = axes[0]
    a = d["arms"]["0.25"]
    names = arm_names(ev, "a025")
    if names:
        Fm, M, S = mean_curve(ev, names, 0.25)
        ax.errorbar(Fm, M, yerr=S, color=BLUE, marker="o", ms=3.4, lw=1.3,
                    capsize=2, clip_on=False, zorder=3,
                    label="Trained FNO")
    anch = np.array([r["mean"] for r in a["sweep"]])
    # The anchor is deterministic given its two fitted coefficients, so it has no
    # seed spread; what it does have is a spread over the 200 media, and the bar
    # drawn here is the standard error of their mean (n = 200), not a seed SD.
    # Mixing the two would put a bar on one curve that is 40% of its own value
    # beside a bar on the other that is 2%, and invite the reader to compare
    # them.  The caption states which is which.
    asem = np.array([r["sem"] for r in a["sweep"]])
    ax.errorbar(F, anch, yerr=asem, color=VERMILION, marker="D", ms=3.4, lw=1.3,
                capsize=2.2, capthick=1.1, elinewidth=1.1, clip_on=False,
                zorder=3, label="anchor, $aB_1+bB_2$")
    ax.plot(F, a["born1"], color=INK, lw=1.0, ls=(0, (4, 2)), clip_on=False,
            zorder=2, label="Born series, order 1")
    ax.plot(F, a["trivial"], color=FAINT, lw=1.0, ls=(0, (1.5, 1.6)),
            clip_on=False, zorder=2, label="trivial, $\\hat u\\equiv0$")
    band(ax)
    ax.set_yscale("log")
    ax.set_ylim(0.008, 4.5)
    decades(ax, numticks=3)
    sweep_axis(ax)
    ax.set_ylabel(r"relative $L_2$ error")
    end_labels(ax, [(float(M[-1]), "trained FNO", BLUE),
                    (float(anch[-1]), "anchor", VERMILION),
                    (float(np.asarray(a["born1"])[-1]), "Born $n{=}1$", INK),
                    (float(np.asarray(a["trivial"])[-1]), "trivial", FAINT)],
                fs=7.2)
    # The decisive reading of the panel: the anchor's plateau sits below the
    # trained operator's by a constant factor, so the two never cross.
    pl_a = float(np.mean(anch[F >= 6.0]))
    pl_f = float(np.mean(M[F >= 6.0]))
    if pl_a > 0 and pl_f > pl_a:
        # Drawn at the right edge, vertical, with the ratio beside it: the
        # panel's claim is a constant vertical gap, so the arrow spans the two
        # plateaus rather than pointing at one of them.  Sized up from
        # 8.8 pt/1.1 pt after review: at print size the 1.1 pt shaft was a
        # hairline in the same grey as the axis rules.
        ratio_arrow(ax, 11.55, pl_a, pl_f,
                    r"$%.1f\times$" % (pl_f / pl_a), fs=9.6, lw=1.5)
    panel_label(ax, "a", r"$v_{\max}=0.25$")

    # ---- (b) plateau against contrast
    ax = axes[1]
    vm = np.array([float(k) for k in sorted(d["arms"])])
    anchp = np.array([d["arms"]["%.2f" % v]["plateau"] for v in vm])
    trivp = np.array([d["arms"]["%.2f" % v]["trivial_plateau"] for v in vm])
    ax.plot(vm, anchp, color=VERMILION, marker="D", ms=4.0, lw=1.3,
            clip_on=False, zorder=3, label="anchor")
    ax.plot(vm, trivp, color=FAINT, lw=1.0, ls=(0, (1.5, 1.6)),
            clip_on=False, zorder=2, label="trivial, $\\hat u\\equiv0$")
    fno, keep, fsd = [], [], []
    for v in vm:
        ns = arm_names(ev, "a%03d" % int(round(v * 100)))
        if not ns:
            continue
        # Per-seed plateau means, so panel (b) can carry a vertical spread:
        # the panel compares the learned plateau with two analytic curves, and
        # without a spread a one-seed wobble reads as a contrast trend.
        per = []
        for n in ns:
            Fn, Mn, _ = curve(ev[n], float(v))
            per.append(float(np.mean(Mn[Fn >= 6])))
        fno.append(float(np.mean(per)))
        fsd.append(float(np.std(per, ddof=1)) if len(per) > 1 else 0.0)
        keep.append(float(v))
    if fno:
        ax.errorbar(np.array(keep), np.array(fno), yerr=np.array(fsd),
                    color=BLUE, marker="o", ms=4.0, lw=1.3, capsize=1.8,
                    elinewidth=0.8, clip_on=False, zorder=3,
                    label="Trained FNO")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel(r"contrast $v_{\max}$")
    ax.set_ylabel(r"plateau $\varepsilon_{\infty}$")
    ax.set_xticks([0.05, 0.1, 0.25, 0.5, 1.0])
    ax.set_xticklabels(["0.05", "0.10", "0.25", "0.50", "1.00"])
    ax.set_xlim(0.04, 1.25)
    despine(ax)
    end_labels(ax, [(float(fno[-1]), "trained FNO", BLUE),
                    (float(anchp[-1]), "anchor", VERMILION),
                    (float(trivp[-1]), "trivial", FAINT)], fs=7.2)
    # The anchor stays below the trivial baseline across the whole contrast
    # range: two fitted coefficients carry real out-of-band information at
    # every contrast, which is the claim the trained operator cannot make.
    if anchp[-1] < trivp[-1]:
        # Upper left, two lines.  On one line the note ended at x ~ 0.3, which
        # is exactly where the rising blue curve crosses the row it occupies;
        # wrapping it halves the width and keeps the box above the crossing.
        badge(ax, 0.03, 0.995,
              "anchor below the trivial baseline\nat every contrast",
              ha="left", va="top", fs=6.4)
    panel_label(ax, "b", r"out-of-band plateau")
    require_inside(ax)
    save(fig, os.path.join(out, "fig_anchor"))


# ---------------------------------------------------------------------------
# Figure -- head depth and the residual on top of it
# ---------------------------------------------------------------------------
def fig_deep(out):
    """The analytic head's depth and the residual on top of it (sec:deep).

    Two panels, matching the two claims that are easy to overstate:
      (a) plateau against the number of Born corrections summed, at the
          contrast where the series converges and at the one where it does
          not.  The point of plotting both is the turning point: at
          v_max = 1.00 the curve is not monotone, so "deeper is better" is
          false and the manuscript does not say it.  Orders whose fit was
          refused on conditioning grounds are not plotted at all -- a hole in
          the curve is honest, a point invented from a singular solve is not.
      (b) the four objects at v_max = 0.25 on the frequency axis: the trained
          operator, the head alone, the shallow head with a learned residual,
          and the deep head with one.  This is the panel that carries the
          method claim, and it is drawn from the same evaluation records as
          the tables so the figure cannot drift from the numbers.
    No GPU.
    """
    set_style()
    ph = os.path.join(C.RES, "anchor_higher_order.json")
    ps = os.path.join(C.RES, "anchor_order_selection.json")
    if not (os.path.exists(ph) and os.path.exists(ps)):
        print("no anchor_higher_order/order_selection json -- run "
              "code/anchor_higher_order.py and code/anchor_order_selection.py")
        return
    ho = json.load(open(ph, encoding="utf-8"))
    sel = json.load(open(ps, encoding="utf-8"))
    ev = load_evals()
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, FULL_W * 0.42),
                             layout="constrained")

    # ---- (a) plateau against head depth
    ax = axes[0]
    ends_a = []
    for vk, col, mk, lab in (("0.25", VERMILION, "D", r"$v_{\max}=0.25$"),
                             ("1.00", INK, "^", r"$v_{\max}=1.00$")):
        ms, ps_ = [], []
        for M in range(1, 9):
            o = ho["arms"][vk]["orders"].get(str(M), {})
            if not o.get("fitted"):
                continue
            ms.append(M)
            ps_.append(o["plateau"])
        if not ms:
            continue
        ax.plot(ms, ps_, color=col, marker=mk, ms=3.6, lw=1.3,
                clip_on=False, zorder=3, label=lab)
        msel = sel["arms"][vk]["M_sel"]
        if msel in ms:
            # Open SQUARE, not the open ring: the ring is the band-edge glyph
            # in every frequency panel, and M_sel is a hyper-parameter on a
            # panel whose abscissa is not frequency at all.  Reusing the ring
            # here is what made the same symbol mean two things in one figure.
            ax.plot([msel], [ps_[ms.index(msel)]], color=col, marker="s",
                    markersize=6.4, mfc="none", mew=1.2, clip_on=False, zorder=4)
        ends_a.append((float(ps_[-1]), lab, col))
    ax.plot([], [], color="#444444", marker="s", markersize=6.4, mfc="none",
            mew=1.2, ls="none", label="$M_{\\mathrm{sel}}$, chosen in band")
    ax.set_yscale("log")
    # Under two decades: round ticks, not the default mantissa ladder that
    # used to print "6 x 10^-1" style labels into each other.
    ax.set_ylim(0.015, 1.0)
    round_ticks(ax, "y", [0.02, 0.05, 0.1, 0.2, 0.5],
                ["0.02", "0.05", "0.10", "0.20", "0.50"])
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Born corrections summed, $M$")
    ax.set_ylabel(r"plateau $\varepsilon_{\infty}$")
    ax.set_xticks(range(1, 9))
    despine(ax)
    # Lower left, not upper right: the upper-right corner is where the
    # v_max = 1.00 curve runs, and the legend box used to sit on the curve.
    ax.legend(fontsize=8, loc="lower left", frameon=False,
              handlelength=1.6, labelspacing=0.25)
    panel_label(ax, "a")

    # ---- (b) the objects at v_max = 0.25
    ax = axes[1]
    F = np.array(sel.get("f_test") or [1.5, 2.5, 3.5, 4.0, 4.25, 4.5, 4.75,
                                       5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0,
                                       12.0])
    ns = arm_names(ev, "a025")
    ends_b = []
    vals_b = []
    if ns:
        Fm, M, S = mean_curve(ev, ns, 0.25)
        ax.errorbar(Fm, M, yerr=S, color=BLUE, marker="o", ms=3.2, lw=1.3,
                    capsize=1.8, clip_on=False, zorder=3, label="Trained FNO")
        ends_b.append((float(M[-1]), "trained FNO", BLUE))
        vals_b.append(np.ravel(M))
    # the head curve comes from the order scan, which is the instrument that
    # records a sweep per (contrast, order); the order-selection record only
    # keeps the plateau it selected on.
    msel = sel["arms"]["0.25"]["M_sel"]
    sweep = ho["arms"]["0.25"]["orders"].get(str(msel), {}).get("sweep") or []
    if sweep:
        Fh = np.array([r["f"] for r in sweep])
        hv = np.array([r["mean"] for r in sweep])
        ax.plot(Fh, hv, color=VERMILION, marker="D", ms=3.2, lw=1.3,
                clip_on=False, zorder=3, label="Analytic head")
        ends_b.append((float(hv[-1]), "Analytic head", VERMILION))
        vals_b.append(np.ravel(hv))
    for pref, col, mk, lab in (("ba_fno_a025_m8_s", GREEN, "P",
                                "Residual on deep head, $M=8$"),
                               ("ba_fno_a025_s", ORANGE, "X",
                                "Residual on shallow head, $M=2$")):
        ns = [n for n in sorted(ev) if n.startswith(pref)]
        if not ns:
            continue
        Fm, M, S = mean_curve(ev, ns, 0.25)
        ax.errorbar(Fm, M, yerr=S, color=col, marker=mk, ms=3.2, lw=1.3,
                    capsize=1.8, clip_on=False, zorder=3, label=lab)
        ends_b.append((float(M[-1]),
                       "deep head $+$ res." if "deep" in lab
                       else "shallow head $+$ res.", col))
        vals_b.append(np.ravel(M))
        if "deep" in lab:
            # The residual does not remove the step, and the panel says so with
            # the same edge rule the other figures use -- no arrow.  An arrow
            # between e(4) and e(5) on this arm was 12 px long: it landed on the
            # green curve and on the analytic-head curve, both of which sit at
            # the plateau by f = 5.  The ratio itself is in the caption and in
            # the macro the caption is built from, so a hand-placed number here
            # would be a second, unchecked copy of it.
            k4 = int(np.argmin(np.abs(Fm - 4.0)))
            ax.plot([Fm[k4]], [M[k4]], marker="o", ms=7.5, mfc="none",
                    mec=GREEN, mew=1.2, clip_on=False, zorder=5)
    band(ax)
    ax.set_yscale("log")
    # The four objects span three decades here (the head alone is an order of
    # magnitude below the trained operator inside the band), so the limit is
    # taken from the data with one decade of headroom instead of being written
    # in by hand: 0.008 was tighter than the head curve, and clip_on=False
    # painted its low-frequency markers over the tick labels.
    val = np.concatenate(vals_b)
    val = val[np.isfinite(val) & (val > 0)]
    ax.set_ylim(val.min() / 1.5, val.max() * 1.5)
    require_inside(ax)
    decades(ax, numticks=3)
    sweep_axis(ax)
    ax.set_ylabel(r"relative $L_2$ error")
    end_labels(ax, ends_b, fs=7.0)
    panel_label(ax, "b", r"$v_{\max}=0.25$")
    save(fig, os.path.join(out, "fig_deep"))


# ---------------------------------------------------------------------------
# Figure -- few-shot purchase curve
# ---------------------------------------------------------------------------
def fig_fewshot(out):
    """Few-shot purchase curve (supplementary experiment, Section sec:interv).

    The baseline FNO (trained on f in [1,4]) is fine-tuned on n in
    {64, 256, 1024} samples drawn from f in [5,8] at v_max = 0.25 and scored
    zero-shot at f = 5, 8 (inside the fine-tune band) and f = 12 (beyond it).
    The covered band is purchased monotonically while the uncovered frequency
    is flat -- labels buy coverage, not extrapolation.
    """
    set_style()
    p = os.path.join(C.RES, "fewshot_purchase.json")
    if not os.path.exists(p):
        print("no fewshot_purchase.json -- run code/run_fewshot.py"); return
    d = C.load_json(p)
    ns = [0, 64, 256, 1024]
    # Categorical x axis: the rungs are 0, 64, 256, 1024, so a linear axis
    # places 0, 64 and 256 inside the first tenth of the panel and asks the
    # reader to infer a spacing that carries no meaning.  The four rungs are
    # four conditions, not four quantities.
    xs = np.arange(len(ns), dtype=float)
    fig, ax = plt.subplots(figsize=(HALF_W, HALF_W * 0.95), layout="constrained")
    specs = [(5.0, ARMS5[0], "f = 5 (fine-tune band)"),
             (8.0, ARMS5[1], "f = 8 (fine-tune band)"),
             (12.0, ARMS5[2], "f = 12 (uncovered)")]
    for f, col, lab in specs:
        ys = [d["rows"][str(n)][str(float(f))] for n in
              ["0", "64", "256", "1024"]]
        ax.plot(xs, ys, color=col, marker="o", ls="-", lw=1.4, ms=4.2,
                clip_on=False, zorder=3, label=lab)
    # The grey span covers the fine-tuned rungs only; the baseline is outside
    # it, because it has seen no out-of-band label at all.
    ax.axvspan(0.5, len(ns) - 0.5, color="#7a828c", alpha=0.10, lw=0, zorder=0)
    ax.annotate("fine-tuned", xy=(2.0, 0.995), xycoords=("data", "axes fraction"),
                ha="center", va="bottom", fontsize=7.4, color="#555555")
    ax.set_xlabel("out-of-band labels $n$")
    ax.set_ylabel(r"relative $L_2$ error")
    ax.set_xticks(xs)
    ax.set_xticklabels(["Baseline", "64", "256", "1024"])
    ax.set_xlim(-0.45, len(ns) - 0.55)
    ax.set_ylim(0.3, 1.25)
    despine(ax)
    # A legend, not end labels: the two fine-tune-band curves CONVERGE at
    # n = 1024, and a label stack in the left margin ran into the y-axis label.
    # The lower left of the panel is empty (all three curves start at 0.9 or
    # above there).
    ax.legend(fontsize=7.4, loc="lower left", frameon=False,
              handlelength=1.8, labelspacing=0.3)
    badge(ax, 0.97, 0.955, r"$v_{\max}=0.25$", ha="right")
    save(fig, os.path.join(out, "fig_fewshot"))


# ---------------------------------------------------------------------------
# Figure -- loss-shaping discriminators (C3/C3b): energy probe
# ---------------------------------------------------------------------------
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
    ~0.01 and its error gets WORSE (the discriminator table): the
    pessimistic-collapse hypothesis is excluded -- what the model lacks out of
    band is phase information, not output energy.
    """
    set_style()
    p = os.path.join(C.RES, "c3_verdict.json")
    if not os.path.exists(p):
        print("no c3_verdict.json -- run code/analyze_c3.py"); return
    d = json.load(open(p, encoding="utf-8"))
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, FULL_W * 0.42),
                             layout="constrained")
    specs = [("baseline", BLUE, "o", "baseline"),
             ("energy", ORANGE, "s", "in-band penalty (C3)"),
             ("energy_ob", PINK, "^", "out-of-band anchor (C3b)")]
    for pi, (field, ylab) in enumerate(
            [("energy_ratio", r"$\|\widehat{u}\|/\|u\|$"),
             ("alignment",
              r"$|\langle\widehat{u},u\rangle|/(\|\widehat{u}\|\|u\|)$")]):
        ax = axes[pi]
        # y = 1 in both panels is the "does nothing useful" level: the ratio
        # equals 1 when the correction has the energy of the target, and the
        # alignment equals 1 when it points along it.  Without the rule the
        # reader has to hold the two scales in mind at once.
        ax.axhline(1.0, color=INK, lw=0.7, ls=(0, (2.0, 2.0)), zorder=1)
        for key, col, mk, lab in specs:
            arm = d["arms"][key]
            fs = sorted(float(k) for k in arm["energy"])
            ys = [arm["energy"]["%.2f" % f][field] for f in fs]
            ax.plot(fs, ys, color=col, marker=mk, ls="-", lw=1.3,
                    clip_on=False, zorder=3)
        band(ax, lo=1.0, hi=4.0)
        ax.set_xlabel("frequency $f$")
        ax.set_ylabel(ylab)
        ax.set_xticks([4, 5, 6, 8, 10, 12])
        ax.set_xlim(3.6, 12.4)
        ax.set_ylim(-0.45 if pi else 0.0, 1.75)
        despine(ax)
        panel_label(ax, "ab"[pi])
    # The signed correlation, on the arm where it was measured.  |<u,u_hat>|
    # cannot separate "aligned but small" from "antiparallel", so the absolute
    # value alone cannot support the sentence in the text that the emitted
    # correction is on average OPPOSED to the target.  The signed curve is
    # drawn from results/corr_energy.json (the same three-seed probe, written
    # by code/diag_corr.py), and only for the baseline arm -- the other two
    # arms were probed by code/analyze_c3.py, which stores the absolute value
    # only.  Drawing the sign for one arm and the magnitude for three keeps
    # the figure inside the data that exists.
    cp = os.path.join(C.RES, "corr_energy.json")
    if os.path.exists(cp):
        cd = json.load(open(cp, encoding="utf-8"))
        fq = [float(x) for x in cd["f"]]
        ax = axes[1]
        ax.plot(fq, [cd["signed"]["%g" % f] for f in fq], color=BLUE,
                marker="o", mfc="none", ms=4.0, lw=1.2, ls=(0, (4, 1.6)),
                clip_on=False, zorder=4)
        ax.axhline(0.0, color=INK, lw=0.6, zorder=1)
    # No end labels here: the whole point of the right panel is that the three
    # arms CONVERGE (alignment collapses to the same floor for all of them),
    # so a right-edge label stack would print three names at one height.  One
    # legend row for the strip, with the short arm names the tables use.
    h = [Line2D([], [], color=col, marker=mk, ls="-", lw=1.3, ms=3.4)
         for _, col, mk, _ in specs]
    h.append(Line2D([], [], color=BLUE, marker="o", mfc="none", ms=4.0,
                    ls=(0, (4, 1.6)), lw=1.2))
    fig.legend(h, [lab for _, _, _, lab in specs] +
               ["baseline, signed correlation"],
               fontsize=8, ncol=4, loc="lower center",
               bbox_to_anchor=(0.5, 1.005), handlelength=1.8,
               columnspacing=1.4)
    badge(axes[0], 0.03, 0.05, "training band", ha="left", va="bottom")
    save(fig, os.path.join(out, "fig_c3probe"))


def build_in(n_grid, k, theta, v, device):
    import torch
    import physics as P
    import data as D
    psi_inc = P.incident_plane_wave(n_grid, k, theta, device).to(torch.complex64)
    return D.build_inputs(v.float(), psi_inc, (k / (2 * np.pi))), psi_inc


# ---------------------------------------------------------------------------
# Figure -- third evidence rung: real geological structure
# ---------------------------------------------------------------------------
def fig_realgeo(out):
    """Third evidence rung: the same boundary on real geological structure.

    (a) the Marmousi velocity model with the three equally spaced windows
        marked, (b) the error sweep for the real windows against the synthetic
        control under an identical design, (c) the one quantity that tells the
        two families apart before any network is involved: the out-of-band
        energy of the contrast field itself.
    """
    set_style()
    import make_realgeo as RG
    p = os.path.join(C.RES, "realgeo.json")
    if not os.path.exists(p):
        print("no realgeo.json"); return
    d = json.load(open(p, encoding="utf-8"))
    c, x, z = RG.load_marmousi()
    wins, rough = RG.pick_windows(c, x, z)

    fig = plt.figure(figsize=(FULL_W, FULL_W * 0.40))
    # wspace wide enough that panel (b)'s y-axis label clears the colourbar
    # of panel (a); at 0.30 the two printed on top of each other.
    gs = fig.add_gridspec(1, 3, width_ratios=[1.55, 1.0, 1.0], wspace=0.52)

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
        ax.text(0.5 * (x0 + x1), 75, ["I", "II", "III"][i], color="k",
                ha="center", va="center", fontsize=8, fontweight="bold")
    ax.set_xlabel("horizontal distance (km)")
    ax.set_ylabel("depth (km)")
    ax.set_xticks([0, 2000, 4000, 6000, 8000])
    ax.set_xticklabels(["0", "2", "4", "6", "8"])
    ax.set_yticks([0, 1000, 2000, 3000])
    ax.set_yticklabels(["0", "1", "2", "3"])
    panel_label(ax, "a", r"Marmousi velocity model, windows I--III")
    # Colourbar OUTSIDE panel (a): an in-panel bar occludes the velocity image,
    # and an axis-label collision with panel (b) was the earlier failure mode.
    cb = fig.colorbar(im, ax=ax, location="right", pad=0.03, shrink=0.85,
                      aspect=20)
    # The colourbar label goes ABOVE the bar, not beside it: placed beside, it
    # shares a column with panel (b)'s y-axis label and the two print as one
    # string ("c (m/s)relative L2 error").
    cb.set_label("$c$ (km/s)", fontsize=7.5, labelpad=2)
    cb.set_ticks([2000, 3500, 5000])
    cb.set_ticklabels(["2", "3.5", "5"])
    cb.set_ticks([2000, 3500, 5000])
    cb.ax.tick_params(labelsize=6.8, length=1.6)
    cb.outline.set_linewidth(0.5)

    # ---- (b) the sweep, real against synthetic ----------------------------
    VMAX = 0.25
    key = "%.2f" % VMAX
    F = d["f_test"]
    ax = fig.add_subplot(gs[0, 1])
    for fam, names, col, ls, lab in (
            ("real", [g for g in d["order"] if g.startswith("geo_")],
             VERMILION, "-", "Marmousi"),
            ("syn", [g for g in d["order"] if g.startswith("syn_")],
             BLUE, "--", "synthetic")):
        # Collapse the three windows of each family to a mean curve with a
        # min--max band over windows (and their seed spans).  This reduces the
        # six raw curves to the two families the labels name.
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
        ax.plot(F, Mm, ls, color=col, lw=1.6, clip_on=False, zorder=3,
                label=lab)
        ax.fill_between(F, Lo, Hi, color=col, alpha=0.15, lw=0, zorder=1)
    band(ax)
    ax.set_yscale("log")
    ax.set_ylim(0.02, 2.2)
    decades(ax, numticks=3)
    ax.set_xlim(1, 12.2)
    ax.set_xticks([2, 4, 6, 8, 10, 12])
    ax.set_xlabel("frequency $f$")
    ax.set_ylabel(r"relative $L_2$ error")
    # In-panel legend instead of right-edge end labels: the labels used to
    # land in the inter-panel gap and on top of panel (c)'s axis label.  The
    # region below the plateau on the right is empty.
    # The label text must fit INSIDE the panel: at "synthetic control
    # (unmatched)" the legend box was wider than the axes, so matplotlib kept
    # its right edge on the right and spilled the left edge over panel (a)'s
    # colourbar, printing the two strings on top of each other.  The caption
    # already says the control is unmatched, so the entry is shortened rather
    # than the panel widened.
    ax.legend(fontsize=7.6, loc="lower right", frameon=False,
              handlelength=2.2, labelspacing=0.3, borderaxespad=0.3)
    panel_label(ax, "b", r"$v_{\max}=%.2f$, FNO" % VMAX)
    despine(ax)

    # ---- (c) the input fields themselves ---------------------------------
    ax = fig.add_subplot(gs[0, 2])
    # M = 32 is omitted: on a 64-point grid the largest mode index IS 32, so the
    # fraction there is zero by construction and a log axis would render the
    # last segment as a spurious cliff.
    Md = [M for M in d["modes"] if M < 32]
    ends = []
    for fam, names, col, ls in (
            ("real", [g for g in d["order"] if g.startswith("geo_")],
             VERMILION, "-"),
            ("syn", [g for g in d["order"] if g.startswith("syn_")],
             BLUE, "--")):
        for g in names:
            y = [d["bands"][g][str(M)] for M in Md]
            ax.plot(Md, y, ls, color=col, lw=1.3, clip_on=False, zorder=2)
        y_end = float(np.mean([d["bands"][g][str(Md[-1])] for g in names]))
        ends.append((min(max(y_end, 2e-14), 0.5),
                     "Marmousi" if fam == "real" else "synthetic", col))
    ax.set_yscale("log")
    # The synthetic control's energy falls to ~1e-13 by mode 24 (it is
    # band-limited by construction) and the real windows stop at ~3e-3; a
    # 1e-6 floor cut seven decades off the one curve that shows the contrast
    # and made the cut look like an artefact of the model.  The full range is
    # kept: the separation between the two families IS the panel.
    ax.set_ylim(2e-14, 1.0)
    decades(ax, numticks=5)
    require_inside(ax)
    ax.set_xlabel("mode index $M$")
    ax.set_ylabel(r"out-of-band energy fraction of $v$")
    panel_label(ax, "c", "input contrast field")
    ax.set_xticks([4, 8, 12, 16, 20, 24])
    despine(ax)
    end_labels(ax, ends, fs=7.2)
    b4g = max(d["bands"][g]["4"] for g in d["order"] if g.startswith("geo_"))
    b4s = max(d["bands"][g]["4"] for g in d["order"] if g.startswith("syn_"))
    # The gap is a single vertical distance at M = 4, so draw it as one: an
    # arrow floating elsewhere leaves the reader to guess which two curves it
    # refers to.
    ax.annotate("", xy=(4, b4g), xytext=(4, b4s),
                arrowprops=dict(arrowstyle="<->", lw=0.8, color="#333333"))
    ax.text(4.8, np.sqrt(b4g * b4s), "%.0f$\\times$" % (b4g / b4s),
            fontsize=8, va="center")
    save(fig, os.path.join(out, "fig_realgeo"))


# ---------------------------------------------------------------------------
# Figure -- two-sided boundary: the curves below the training band
# ---------------------------------------------------------------------------
def fig_banddown(evals, out):
    """Two-sided boundary: extend the FNO error curves below the training band.

    Below-band points come from results/banddown.json, a fresh evaluation of
    the same checkpoints on paired media at f in {0.25, 0.5, 0.75, 1.0}; the
    above-band part reuses the eval sweeps.  The trivial baseline is drawn for
    each arm in the same colour, so the contrast between the below-band
    elevation (model several times above baseline) and the above-band
    saturation-at-baseline is visible in one panel.
    """
    set_style()
    p = os.path.join(C.RES, "banddown.json")
    if not os.path.exists(p):
        print("no banddown.json"); return
    d = json.load(open(p, encoding="utf-8"))
    spec = json.load(open(os.path.join(C.RES, "spectrum.json"),
                          encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(HALF_W, HALF_W * 0.88), layout="constrained")
    ax.axvspan(1.0, 4.0, color="#7a828c", alpha=0.12, lw=0, zorder=0)
    ax.axvline(1.0, color=INK, lw=0.7, ls=(0, (1.1, 1.5)), zorder=1)
    ax.axvline(4.0, color=INK, lw=0.7, ls=(0, (1.1, 1.5)), zorder=1)
    ends = []
    for k, (vk, vmax) in enumerate(zip(("a005", "a010", "a025", "a050", "a100"),
                                       C.VMAX_LIST)):
        col = ARMS5[k]
        key = "%.2f" % vmax
        Fb = np.array(d["f_test"])
        rows = np.mean([[d["runs"]["main_fno_%s_s%d" % (vk, s)]
                         ["sweep"][key][i]["mean"]
                         for i in range(len(Fb))] for s in range(3)], axis=0)
        triv = np.mean([[d["runs"]["main_fno_%s_s%d" % (vk, s)]
                         ["sweep"][key][i]["triv_mean"]
                         for i in range(len(Fb))] for s in range(3)], axis=0)
        ax.plot(Fb, rows, "-", color=col, lw=1.5, clip_on=False, zorder=3)
        ax.plot(Fb, triv, "--", color=col, lw=1.0, alpha=0.65, zorder=2)
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
            ax.plot(Fa, Ma, "-", color=col, lw=1.5, clip_on=False, zorder=3)
            Ta = np.array([spec["data"][str(vmax)]["f=%g" % f]["triv"]
                           for f in C.F_TEST])
            ax.plot(Fa, Ta, "--", color=col, lw=1.0, alpha=0.65, zorder=2)
            ends.append((float(Ma[-1]), r"$v_{\max}=%.2f$" % vmax, col))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.22, 13.0)
    ax.set_ylim(0.01, 3.0)
    ax.set_xticks([0.25, 0.5, 1, 2, 4, 8, 12])
    ax.set_xticklabels(["0.25", "0.5", "1", "2", "4", "8", "12"])
    decades(ax, numticks=4)
    ax.set_xlabel("frequency $f$")
    ax.set_ylabel(r"relative $L_2$ error")
    despine(ax)
    end_labels(ax, ends, fs=7.0)
    # A legend inside the axes (below), not a badge: the "solid/dashed" gloss
    # used to be a free-text annotation, so the one encoding the panel rests on
    # -- solid FNO against dashed baseline -- was the only one not stated as a
    # legend entry.  Both band edges are already drawn by the shading and the
    # two dotted rules; the patch names them.
    h = [Line2D([], [], color="#3a3a3a", ls="-", lw=1.5),
         Line2D([], [], color="#3a3a3a", ls="--", lw=1.0, alpha=0.65),
         Patch(facecolor="#7a828c", alpha=0.12, edgecolor="none")]
    # Above the axes, two columns: inside, the only free space sits between
    # the solid and dashed curves at f<1 and the box printed over both.
    ax.legend(h, ["Trained FNO", "Trivial incident-field baseline",
                  "Training band $[1,4]$"],
              fontsize=7.2, loc="lower center", bbox_to_anchor=(0.5, 1.01),
              frameon=False, ncol=2, handlelength=2.2, labelspacing=0.3,
              columnspacing=1.2)
    save(fig, os.path.join(out, "fig_banddown"))


# ---------------------------------------------------------------------------
# Figure -- the plateau tracks the trivial baseline across the coupling range
# ---------------------------------------------------------------------------
def fig_plateau_coupling(evals, out):
    """Out-of-band plateau against the same-arm trivial baseline, vs coupling.

    The plateau is the seed-mean error over f >= 6; the baseline comes from
    spectrum.json; the coupling is v_max/(2 gamma).  One log-log panel shows
    the model tracking the baseline across the 20x coupling range, with the
    excess (model over baseline) growing slowly with coupling.
    """
    # Single five-point panel: at half a column it left more white space than
    # information.  Emitted at 0.72 of the text block, 1.25x type, so the
    # points, the printed values and the y=1 rule all print larger.
    Z = 1.25
    set_style(Z)
    spec = json.load(open(os.path.join(C.RES, "spectrum.json"),
                          encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(FULL_W * 0.72, FULL_W * 0.50),
                           layout="constrained")
    coups, plats, trivs, sds = [], [], [], []
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
        # Per-seed plateau means first, then the mean and the spread over
        # seeds: the panel is the only place the seed-to-seed spread of the
        # plateau is visible, and a bare mean curve let a reader take the
        # 1.4x excess for a point estimate rather than for a mean over three
        # seeds.  ddof=1 because the three seeds are a sample of the training
        # variance, not its population.
        per_seed = np.array([float(np.mean(p)) for p in plateau])
        plats.append(float(per_seed.mean()))
        sds.append(float(per_seed.std(ddof=1)) if len(per_seed) > 1 else 0.0)
        trivs.append(float(np.mean(
            [spec["data"][str(vmax)]["f=%g" % f]["triv"]
             for f in C.F_TEST if f >= 6])))
        coups.append(vmax / (2 * C.GAMMA))
    coups, plats, trivs = np.array(coups), np.array(plats), np.array(trivs)
    sds = np.array(sds)
    # y = 1 is the "predict zero" level of the metric: a curve above it is
    # worse than returning the incident field alone, which is what makes the
    # plateau-versus-baseline comparison legible without a second axis.
    ax.axhline(1.0, color=INK, lw=0.7, ls=(0, (2.0, 2.0)), zorder=1)
    ax.annotate(r"$y=1$ (predict zero)", xy=(0.52, 1.02), fontsize=7.0 * Z,
                color="#333333", va="bottom")
    ax.errorbar(coups, plats, yerr=sds, fmt="-o", color=BLUE, lw=1.5, ms=4.6,
                capsize=2.0, elinewidth=0.8,
                clip_on=False, zorder=3, label="FNO plateau")
    ax.plot(coups, trivs, "--s", color=VERMILION, lw=1.5, ms=4.2,
            clip_on=False, zorder=3, label="trivial baseline")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.4, 12)
    ax.set_ylim(0.15, 3.2)
    ax.set_xticks([0.5, 1, 2.5, 5, 10])
    ax.set_xticklabels(["0.5", "1", "2.5", "5", "10"])
    decades(ax, numticks=4)
    ax.set_xlabel(r"coupling $v_{\max}/(2\gamma)$")
    ax.set_ylabel(r"mean relative $L_2$ error, $f \geq 6$")
    despine(ax)
    # The value of every point is printed: the figure carries the plateau
    # numbers the text quotes, and a log axis without printed values makes
    # the reader measure distances to guess them.
    for x, y in zip(coups, plats):
        ax.annotate("%.2f" % y, xy=(x, y), xytext=(0, 5.5 * Z),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=8 * Z, color=BLUE)
    for x, y in zip(coups, trivs):
        ax.annotate("%.2f" % y, xy=(x, y), xytext=(0, -6.5 * Z),
                    textcoords="offset points", ha="center", va="top",
                    fontsize=8 * Z, color=VERMILION)
    end_labels(ax, [(float(plats[-1]), "FNO", BLUE),
                    (float(trivs[-1]), "trivial", VERMILION)], fs=7.4 * Z)
    # The excess is the quantity of interest and is stated twice: per point,
    # in the printed values above, and summarised for the whole range in the
    # badge.  The old ratio arrow sat at the low-coupling end, where the two
    # curves coincide, and rendered as a degenerate zero-length glyph.
    badge(ax, 0.03, 0.955,
          r"excess over baseline: $%.2f\times$ $\rightarrow$ $%.2f\times$ "
          r"across the range" % (plats[0] / trivs[0], plats[-1] / trivs[-1]),
          ha="left", fs=6.9 * Z)
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
    _before = len(_CONTAINED)
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
        fig_scaling(ev, a.out)
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
    if a.only in ("all", "deep"):
        fig_deep(a.out)
    if a.only in ("all", "fewshot"):
        fig_fewshot(a.out)
    # A declaration that no longer fires is worse than none: it reads as a
    # check in the log.  At least one panel must have been tested.
    if len(_CONTAINED) == _before:
        raise SystemExit("no panel declared require_inside() on this run -- "
                         "the containment check was vacuous")
    print("containment: %d panel(s) checked" % (len(_CONTAINED) - _before))
