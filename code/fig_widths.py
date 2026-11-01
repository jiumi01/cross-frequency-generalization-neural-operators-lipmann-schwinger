# Author: Longgang Wang <longgangwang@163.com>
# Affiliation: Luoyang Institute of Science and Technology, Luoyang 471000, Henan, China

"""Figure inclusion widths: derive them from the emitted PDFs, then guard.

The failure this closes.  figures.py emits a figure at a natural width set by
its own figsize; the manuscripts then include it at a hand-written fraction of
\\textwidth.  Nothing tied the two together, so over two revisions the EN body
included figures at 0.52--0.86 \\textwidth while the CN body included the same
PDFs at 0.52--1.0 \\textwidth: the same figure rendered at 72% in one language
and 117% in the other, with an effective type size down to ~6.5 pt -- below
print legibility -- in the EN version.  A figure can also grow past the text
block when a legend row is added (\\textwidth inclusion then silently
down-scales it), which is how the cross-contrast strip reached 1.06 \\textwidth.

What this script does.

  1. measures every emitted figure's MediaBox width,
  2. derives the inclusion fraction w/\\textwidth (\\textwidth = 470.6 pt for
     the MDPI class both manuscripts use),
  3. rewrites every \\includegraphics[width=...]{fig_*.pdf} in the EN body,
     the CN body and the SI to that fraction, and
  4. in --check mode asserts, for every figure named in either manuscript:
     the file exists, the fraction equals the measured one in BOTH bodies
     (EN and CN cannot disagree), and the fraction never exceeds 1.

    python code/fig_widths.py            # rewrite
    python code/fig_widths.py --check    # verify only, exit 1 on drift
"""
import os
import re
import sys
import glob

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    ".."))
CANVAS_PT = 470.6             # the 16.6 cm canvas figures.py draws on
REAL_TEXTWIDTH_PT = 392.9       # measured: caption right edge on the built PDF
# The mdpi class at a4paper sets left=5.87 cm (1.27 + 0.51 + 4.10 marginpar)
# and right=1.27 cm, so the real \textwidth is 21 - 5.87 - 1.27 = 13.86 cm
# = 392.9 pt, not the canvas width.  Every figure is therefore placed at
# 392.9/470.6 = 0.835 of its natural size -- uniformly, which is what keeps
# EN and CN identical, but it is NOT 1:1: the 9 pt / 8.5 pt nominal type
# renders at 7.5 pt / 7.1 pt.  Named for what it is rather than for what
# the layout would need.
DOCS = [
    os.path.join(ROOT, "submission", "paper_en", "main.tex"),
    os.path.join(ROOT, "submission", "paper_cn", "main.tex"),
    os.path.join(ROOT, "submission", "paper_en", "SI_main.tex"),
]
FIGDIRS = [
    os.path.join(ROOT, "paper", "en", "figs"),
    os.path.join(ROOT, "paper", "cn", "figs"),
]

INC = re.compile(r"\\includegraphics\s*\[([^\]]*)\]\{([^}]+)\}")
WOPT = re.compile(r"width\s*=\s*([^\],]+)")


def mediabox_width(path):
    raw = open(path, "rb").read()
    m = re.search(rb"/MediaBox\s*\[\s*([-\d.]+)\s+([-\d.]+)\s+"
                  rb"([-\d.]+)\s+([-\d.]+)", raw)
    if not m:
        return None
    return float(m.group(3)) - float(m.group(1))


def measure():
    out = {}
    for d in FIGDIRS:
        for p in sorted(glob.glob(os.path.join(d, "*.pdf"))):
            w = mediabox_width(p)
            if w:
                out[os.path.basename(p)] = w
    return out


def frac_of(w_pt):
    """Inclusion fraction: 1.0 (i.e. \\textwidth) at or above 0.995, else two
    decimals.  A figure wider than the text block is an error, not a number
    the manuscripts should compensate for."""
    f = w_pt / CANVAS_PT
    if f > 1.0 + 1e-9:
        return None, f
    if f >= 0.995:
        return "\\textwidth", 1.0
    return "%.2f\\textwidth" % round(f + 1e-9, 2), round(f + 1e-9, 2)


def scan(path):
    """-> {figure basename: current width option} for one document."""
    txt = open(path, encoding="utf-8").read()
    out = {}
    for m in INC.finditer(txt):
        opts, fig = m.group(1), m.group(2).strip()
        wm = WOPT.search(opts)
        out[os.path.basename(fig)] = wm.group(1).strip() if wm else None
    return out


def rewrite(path, widths):
    txt = open(path, encoding="utf-8").read()
    n = [0]

    def _sub(m):
        opts, fig = m.group(1), m.group(2).strip()
        base = os.path.basename(fig)
        if base not in widths:
            return m.group(0)
        new_w, _ = frac_of(widths[base])
        if new_w is None:
            return m.group(0)
        wm = WOPT.search(opts)
        if wm:
            new_opts = opts[:wm.start(1)] + new_w + opts[wm.end(1):]
        else:
            new_opts = "width=" + new_w + (", " + opts if opts else "")
        if new_opts != opts:
            n[0] += 1
        return "\\includegraphics[%s]{%s}" % (new_opts, m.group(2))

    txt = INC.sub(_sub, txt)
    if n[0]:
        open(path, "w", encoding="utf-8").write(txt)
    return n[0]


def check(widths):
    bad = 0
    seen = {}
    for doc in DOCS:
        if not os.path.exists(doc):
            print("[skip] %s missing" % doc)
            continue
        cur = scan(doc)
        for base, wopt in sorted(cur.items()):
            if base not in widths:
                print("[FAIL] %s: %s has no emitted PDF in paper/*/figs"
                      % (os.path.basename(doc), base))
                bad += 1
                continue
            want, f = frac_of(widths[base])
            if want is None:
                print("[FAIL] %s: %s is %.1f pt wide, past the %.1f pt text "
                      "block" % (os.path.basename(doc), base, widths[base],
                                 CANVAS_PT))
                bad += 1
                continue
            if wopt != want:
                print("[FAIL] %s: %s included at width=%s, measured natural "
                      "width wants %s" % (os.path.basename(doc), base, wopt,
                                          want))
                bad += 1
            if base in seen and seen[base] != wopt:
                print("[FAIL] %s: %s included at width=%s, but another "
                      "manuscript uses %s -- the two language versions of "
                      "one figure must render at the same size"
                      % (os.path.basename(doc), base, wopt, seen[base]))
                bad += 1
            seen[base] = wopt
    if not bad:
        scale = REAL_TEXTWIDTH_PT / CANVAS_PT
        print("[ok]   %d figures: placed at the same fraction of their natural "
              "width in both manuscripts (uniform %.3f scale), EN and CN agree"
              % (len(seen), scale))
        print("[note] canvas %.1f pt vs class text block %.1f pt -> nominal 9/8.5 "
              "pt type renders at %.1f/%.1f pt"
              % (CANVAS_PT, REAL_TEXTWIDTH_PT, 9 * scale, 8.5 * scale))
    return bad


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "--rewrite"
    widths = measure()
    if not widths:
        raise SystemExit("no figure PDFs found under paper/*/figs")
    if mode == "--check":
        raise SystemExit(1 if check(widths) else 0)
    if mode not in ("--rewrite", ""):
        raise SystemExit("usage: fig_widths.py [--check]")
    total = 0
    for doc in DOCS:
        if os.path.exists(doc):
            k = rewrite(doc, widths)
            total += k
            print("rewrote %d includegraphics widths in %s"
                  % (k, os.path.relpath(doc, ROOT)))
    bad = check(widths)
    print("measured %d figures" % len(widths))
    raise SystemExit(1 if bad else 0)
