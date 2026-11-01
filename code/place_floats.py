# Author: Longgang Wang <longgangwang@163.com>
# Affiliation: Luoyang Institute of Science and Technology, Luoyang 471000, Henan, China

"""Move every float to its first reference, one table file per table.

Why this exists
---------------
Both manuscripts used to hold all of their floats in two places: every table
arrived through a single ``\\input{gen_results}`` at the top of the Experiments
section, and all twelve figure environments sat in one run just before the
Discussion.  LaTeX has no choice in that arrangement -- it defers each float
until it finds room, and the only room left is at the end of the document, so
the counters spilled past the body into the reference list.  In the compiled
English PDF the References heading falls on page 18 while Figures 7-12 land on
pages 18-23, interleaved with bibliography entries, and the body pages keep a
quarter of each page empty.

The fix is to emit the tables as one file each (make_results_tex.py) and put
every float *after the paragraph that first cites it*.  Doing that by hand is
how the previous version of this file drifted; this script derives the position
from the ``\\ref`` that is already in the text, so it is repeatable and it
fails loudly on a float that nothing cites.

Usage:  python place_floats.py paper/en/main.tex paper/cn/main.tex
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Single source of truth.  This file used to carry its own copy of the tuple and
# it drifted silently: by the time the propnorm, paired and cost tables were
# added, the local copy still listed the original seven, so a re-run would have
# (a) left the three new tables wherever they happened to be and (b) inserted a
# SECOND \input for each of the seven it did know about, because the step that
# strips the old placement only removes ``\input{gen_results}``.  Importing the
# generator's tuple closes both halves of that class at once.
import make_results_tex as MR

TAB_ORDER = MR.TAB_ORDER


def chunks(lines):
    """Split lines into blocks of consecutive non-blank lines, blank lines alone."""
    out, cur = [], []
    for ln in lines:
        if ln.strip() == "":
            if cur:
                out.append(cur)
                cur = []
            out.append([ln])
        else:
            cur.append(ln)
    if cur:
        out.append(cur)
    return out


def process(path, tab_order=TAB_ORDER):
    src = open(path, encoding="utf-8").read()
    lines = src.split("\n")
    head = os.path.basename(path)

    # --- 1. the composite table input disappears --------------------------
    n_before = len(lines)
    lines = [ln for ln in lines
             if ln.strip() != "\\input{gen_results}"]
    if len(lines) != n_before:
        print("  %s: removed \\input{gen_results}" % head)

    # --- 2. lift every figure environment out of the text ------------------
    # ``figure`` and ``figure*`` both occur -- the wide multi-panel plots are
    # starred -- and an earlier revision of this regex matched only the
    # unstarred form, silently leaving five floats where they were while the
    # run reported success on the rest.  The backreference keeps each
    # \begin paired with its own \end.
    text = "\n".join(lines)
    pat = re.compile(r"\\begin\{(figure\*?)\}.*?\\end\{\1\}\n", re.S)
    items = []            # (label, body, kind)
    for m in pat.finditer(text):
        blk = m.group(0)
        lab_m = re.search(r"\\label\{(fig:[^}]+)\}", blk)
        if not lab_m:
            print("  %s: !! figure with no label" % head)
            continue
        items.append([lab_m.group(1), blk])
    text = pat.sub("", text)
    for k in tab_order:
        # Tables that are already \input at their first reference must not get a
        # second copy: this script adds a placement, it does not own one.  The
        # figures above are safe because they were removed from ``text``; the
        # table inputs were not, so the guard has to be explicit.
        if "\\input{gen_tab_%s}" % k in text:
            print("  %s: tab:%s already placed, left alone" % (head, k))
            continue
        items.append(["tab:%s" % k, "\\input{gen_tab_%s}\n" % k])

    # --- 3. place each one after the block that first cites it -------------
    ch = chunks(text.split("\n"))
    place = {}            # chunk index -> list of bodies
    unresolved = []
    for lab, body in items:
        needle = "\\ref{%s}" % lab
        at = None
        for i, c in enumerate(ch):
            if needle in "\n".join(c):
                at = i
                break
        if at is None:
            unresolved.append(lab)
            continue
        j = at
        while j + 1 < len(ch) and all(x.strip() == "" for x in ch[j + 1]):
            j += 1
        place.setdefault(j, []).append((lab, body))

    out = []
    for i, c in enumerate(ch):
        out.extend(c)
        if i in place:
            # tables before figures within one landing spot, then by label, so
            # repeated runs produce byte-identical output
            for lab, body in sorted(place[i], key=lambda t: (t[0].startswith("fig:"), t[0])):
                if not body.endswith("\n"):
                    body += "\n"
                out.extend(body.rstrip("\n").split("\n"))
                out.append("")

    new = "\n".join(out)
    new = re.sub(r"\n{4,}", "\n\n\n", new)
    if not new.endswith("\n"):
        new += "\n"
    open(path, "w", encoding="utf-8",
         newline="\n").write(new)

    print("  %s: placed %d floats, %d unresolved %s"
          % (head, len(items) - len(unresolved), len(unresolved),
             unresolved if unresolved else ""))
    return unresolved


def main():
    bad = 0
    for p in sys.argv[1:]:
        if process(p):
            bad += 1
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
