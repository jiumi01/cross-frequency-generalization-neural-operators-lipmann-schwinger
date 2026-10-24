# Author: Longgang Wang <longgangwang@163.com>
# Affiliation: Luoyang Institute of Science and Technology, Luoyang 471000, Henan, China

"""Guard the hard-coded Supplementary Table numbers in the main text.

The SI is a separate LaTeX document, so its tables cannot be reached with
\\ref from the main text; the prose therefore quotes them by number
("Supplementary Table~12").  A number is only correct for the order the
tables happen to appear in the SI, and reordering the SI would silently
break every one of those quotes.  This gate recomputes the numbering from
the SI source and checks that each quoted number still points at a table
whose caption shares vocabulary with the sentence that quotes it.

What it asserts, per quoted number N:
  1. N is within 1..(number of tables in the SI);
  2. the sentence quoting N and the caption of table N share at least one
     significant token (>= 5 letters, prefix-matched so that "compare"
     matches "comparison").

Exit code 0 iff every quote passes.
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
EN = os.path.join(ROOT, "submission", "paper_en")


def read(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def si_tables():
    """Return {number: caption} for the SI, resolving \\input in place."""
    src = read(os.path.join(EN, "SI_main.tex"))
    # inline the inputs at the position they are pulled in
    while True:
        m = re.search(r"\\input\{([^}]+)\}", src)
        if not m:
            break
        name = m.group(1)
        if not name.endswith(".tex"):
            name += ".tex"
        path = os.path.join(ROOT, "paper", "en", name)
        if not os.path.exists(path):
            raise SystemExit("SI input missing from paper/en/: %s" % name)
        src = src[:m.start()] + read(path) + src[m.end():]
    out = {}
    n = 0
    for tm in re.finditer(r"\\begin\{table\*?\}(.*?)\\end\{table\*?\}",
                          src, re.S):
        cm = re.search(r"\\caption\{(.*?)\}\s*\n", tm.group(1), re.S)
        if not cm:
            cm = re.search(r"\\caption\{(.*?)\}", tm.group(1), re.S)
        n += 1
        out[n] = " ".join(cm.group(1).split()) if cm else ""
    return out


def stems(text):
    """Lower-cased >=5 letter tokens, hyphen/underscore split."""
    text = re.sub(r"\\[a-zA-Z]+\*?", " ", text)
    text = re.sub(r"[^A-Za-z ]", " ", text)
    return {w.lower() for w in text.split() if len(w) >= 5}


def shared(a, b):
    """Count distinct stems of a sharing a >=5 letter prefix with a stem of b."""
    n = 0
    for w in a:
        for c in b:
            k = 0
            while k < len(w) and k < len(c) and w[k] == c[k]:
                k += 1
            if k >= 5:
                n += 1
                break
    return n


def sentences(masked, start, end):
    """Sentence containing [start,end) plus the next one, as one token set.

    The next sentence is included because several quotes are continued there
    ("Both column groups of that table ..."), and it is the only place the
    shared vocabulary appears.
    """
    before = [m for m in re.finditer(r"\.(?:\s|$)", masked[:start])]
    a = before[-1].end() if before else 0
    after = [m for m in re.finditer(r"\.(?:\s|$)", masked[end:])]
    b = end + after[0].start() + 1 if after else len(masked)
    return stems(masked[a:b])


def mask_math(s):
    """Blank out $...$ spans, preserving length so indices stay valid."""
    out = list(s)
    i = 0
    while i < len(s):
        if s[i] == "$":
            j = s.find("$", i + 1)
            if j < 0:
                break
            for k in range(i, j + 1):
                if out[k] != "\n":
                    out[k] = " "
            i = j + 1
        else:
            i += 1
    return "".join(out)


def same_stem(w, d):
    """True when w and d share a common prefix of >= 5 letters."""
    k = 0
    while k < len(w) and k < len(d) and w[k] == d[k]:
        k += 1
    return k >= 5


def main():
    tabs = si_tables()
    if not tabs:
        print("FAIL: no tables found in the SI")
        return 1
    print("SI tables: %d (numbered 1..%d)" % (len(tabs), max(tabs)))
    src = read(os.path.join(EN, "main.tex"))
    masked = mask_math(src)
    quotes = list(re.finditer(r"Supplementary Table~(\d+)", src))
    if not quotes:
        print("FAIL: no 'Supplementary Table~N' quotes found -- "
              "did the phrasing change?")
        return 1
    print("main-text quotes: %d" % len(quotes))
    cap_stems = {c: stems(cap) for c, cap in tabs.items()}
    bad = 0
    seen = {}
    for m in quotes:
        n = int(m.group(1))
        seen[n] = seen.get(n, 0) + 1
        if n not in tabs:
            print("FAIL  Table~%d: the SI only has %d tables" % (n, len(tabs)))
            bad += 1
            continue
        # The quoting sentence (plus its continuation) must share a stem with
        # the caption of the table it names that no other caption contains:
        # a renumbering of the SI would move captions relative to numbers,
        # and a generic word such as "relative" or "paired" appears in too
        # many captions to catch that on its own.
        ctx = sentences(masked, m.start(), m.end())
        mine = cap_stems.get(n, set())
        others = set().union(*[cap_stems[c] for c in cap_stems if c != n]) \
            if len(cap_stems) > 1 else set()
        # a caption stem is distinctive when no other caption carries it
        # verbatim; "retained" stays distinctive against a caption that says
        # "retains", while "paired" or "relative" do not
        distinctive = mine - others
        hit = sorted(w for w in ctx if any(same_stem(w, d)
                                           for d in distinctive))
        if not hit:
            print("FAIL  Table~%d: the sentence shares no distinctive "
                  "vocabulary with the caption" % n)
            print("      sentence stems: %s" % sorted(ctx))
            print("      distinctive caption stems: %s" % sorted(distinctive))
            print("      caption: %s" % tabs[n][:120])
            bad += 1
    print("quoted numbers: %s" % ", ".join(
        "%d(x%d)" % (k, v) for k, v in sorted(seen.items())))
    unquoted = [n for n in tabs if n not in seen]
    if unquoted:
        print("note: tables never quoted from the main text: %s" % unquoted)
    if bad:
        print("RESULT: %d misaligned quote(s)" % bad)
        return 1
    print("RESULT: every quoted number points at a caption that matches "
          "its sentence")
    return 0


if __name__ == "__main__":
    sys.exit(main())
