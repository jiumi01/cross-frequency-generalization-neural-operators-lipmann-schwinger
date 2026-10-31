"""Distance between a float's first citation and the page it is printed on.

A reader who meets "Table 4" on page 23 and finds the table on page 30 has to
hold the argument in their head while turning pages.  The rule enforced here is
that a figure or table is printed on the page where it is first cited, or on
the next one.

The rule has one exemption, and it is a property of the page grid rather than a
weakening of the standard.  A two-column float (``figure*``/``table*``) occupies
the whole top block of a page, so two of them cannot share that block unless
their combined height stays under ``\\dbltopfraction``; when two are declared
within a page of each other the second is pushed one page beyond the block the
first did not take.  One page of slack is therefore structural for that class,
and two is the tolerance applied to it.  Single-column floats keep one page.
The class is read from the source, not guessed from the PDF, so a single-column
float never inherits the exemption.

Two traps, both measured rather than assumed:

* The MDPI class prints a line number in the right margin of every line.  With
  ``pdftotext -layout`` those numbers land *inside* the running text, so
  "Supplementary Table 4" extracts as "Supplementary 632 Table 4" and a
  look-behind that excludes the supplementary prefix stops matching.  The
  first version of this measurement therefore reported Table 4 as eight pages
  away from its citation when it was on the same page.  Line numbers are
  stripped per line before anything is matched.
* Chinese captions are "图 1. " and Chinese citations are "图 1 给出" / "图 1c）",
  so the caption is identified by the full stop and the citation by its
  absence.
"""

import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAX_GAP = {"single": 1, "dbl": 2}


def pages(path):
    raw = subprocess.run(["pdftotext", "-layout", path, "-"],
                         capture_output=True)
    text = raw.stdout.decode("utf-8", "replace")
    out = []
    for page in text.split("\f"):
        lines = [re.sub(r"\s+\d{3,4}\s*$", "", ln) for ln in page.split("\n")]
        out.append(" ".join(" ".join(lines).split()))
    return out


def float_classes(lang):
    """Map (kind, number) to "single" or "dbl", in source declaration order.

    LaTeX numbers a float when \\begin{...} is processed and places it later, so
    declaration order is numbering order.  Tables pulled in with
    \\input{gen_tab_*} are classified from the included file.
    """
    src = os.path.join(ROOT, "submission", "paper_%s" % lang, "main.tex")
    if not os.path.exists(src):
        return {}
    with open(src, encoding="utf-8") as fh:
        text = fh.read()
    kinds, classes = {"figure": 0, "table": 0}, {}
    pat = re.compile(r"\\begin\{(figure\*?|table\*?)\}"
                     r"|\\input\{(gen_tab_[A-Za-z0-9_]+)\}")
    for m in pat.finditer(text):
        if m.group(1):
            base = m.group(1).rstrip("*")
            kinds[base] += 1
            classes[(base.capitalize(), kinds[base])] = (
                "dbl" if m.group(1).endswith("*") else "single")
        else:
            inc = os.path.join(os.path.dirname(src), m.group(2) + ".tex")
            if not os.path.exists(inc):
                continue
            with open(inc, encoding="utf-8") as fh:
                head = fh.read(400)
            star = re.search(r"\\begin\{table\*?\}", head)
            kinds["table"] += 1
            classes[("Table", kinds["table"])] = (
                "dbl" if star and star.group(0).startswith("\\begin{table*}")
                else "single")
    return classes


def measure(path, lang):
    P = pages(path)
    cap, ref = {}, {}
    if lang == "en":
        cap_re = re.compile(r"\b(Figure|Table)\s+(\d+)\.")
        fig_re = re.compile(r"(?:Fig\.|Figure)\s*(\d+)(?!\d)")
        tab_re = re.compile(r"(?<!Supplementary )Table\s+(\d+)(?!\d)")
    else:
        cap_re = re.compile(r"([图表])\s*(\d+)\.")
        fig_re = re.compile(r"图\s*(\d+)(?!\d)(?!\.)")
        tab_re = re.compile(r"(?<!补充材料)(?<!材料)表\s*(\d+)(?!\d)(?!\.)")
    for i, pg in enumerate(P, 1):
        for m in cap_re.finditer(pg):
            kind = "Figure" if m.group(1) in ("Figure", "图") else "Table"
            cap.setdefault((kind, int(m.group(2))), i)
    for i, pg in enumerate(P, 1):
        for m in fig_re.finditer(pg):
            ref.setdefault(("Figure", int(m.group(1))), i)
        for m in tab_re.finditer(pg):
            ref.setdefault(("Table", int(m.group(1))), i)
    return cap, ref, len(P)


def judge(cap, ref, classes):
    """Return the offending floats as (kind, num, class, first, printed, gap).

    Kept separate from the PDF reading so the rule itself can be fed a known-bad
    case in test_gate_falsification.py.
    """
    bad = []
    for k in sorted(cap):
        cls = classes.get(k, "single")
        r = ref.get(k)
        if r is None:
            bad.append((k[0], k[1], cls, None, cap[k], None))
            continue
        gap = cap[k] - r
        if abs(gap) > MAX_GAP[cls]:
            bad.append((k[0], k[1], cls, r, cap[k], gap))
    return bad


def main():
    problems = 0
    for lang, name in (("en", "paper/en/main.pdf"), ("cn", "paper/cn/main.pdf")):
        path = os.path.join(ROOT, name)
        if not os.path.exists(path):
            print("[FAIL] %s missing" % name)
            problems += 1
            continue
        cap, ref, npages = measure(path, lang)
        classes = float_classes(lang)
        print("=== %s: %d pages, %d floats" % (name, npages, len(cap)))
        print("  %-7s %-4s %-6s %-9s %-8s %s"
              % ("kind", "num", "class", "firstRef", "printed", "gap"))
        offenders = {(o[0], o[1]): o for o in judge(cap, ref, classes)}
        for k in sorted(cap, key=lambda x: (x[0], x[1])):
            r = ref.get(k)
            cls = classes.get(k, "single")
            lim = MAX_GAP[cls]
            if r is None:
                print("  %-7s %-4d %-6s %-9s %-8d %s"
                      % (k[0], k[1], cls, "none", cap[k], "never cited"))
                continue
            gap = cap[k] - r
            flag = "   <-- beyond %d page(s)" % lim if (k[0], k[1]) in offenders else ""
            print("  %-7s %-4d %-6s %-9d %-8d %+d%s"
                  % (k[0], k[1], cls, r, cap[k], gap, flag))
        problems += len(offenders)
    if problems:
        print("\nFloat placement gate: %d problem(s)" % problems)
        return 1
    print("\nFloat placement gate: every float is cited within one page of "
          "where it is printed (two for two-column floats)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
