# -*- coding: utf-8 -*-
"""Every numeric macro that expands to math must be used inside math.

The class of defect: `gen_macros.tex` defines a scale of numbers, and a few of
them (`\\ResBandSynCoarsePct` is `2.1\\times10^{-9}`) carry math-only tokens.
Writing one of those in prose compiles as far as the first `^` and then dies with
"Missing $ inserted" -- a message that names the macro's argument rather than the
macro, and which cost two build cycles when this file's rule was enforced by
memory alone.  The check is mechanical, so it is a gate rather than a habit.
"""
import io
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MATHY = re.compile(r"\\(times|cdot|div|pm|mp|approx|le|ge|neq|ll|gg|frac|sqrt|"
                   r"mathrm|mathcal|langle|rangle|sum|int|infty|partial|alpha|"
                   r"beta|gamma|delta|varepsilon|psi|kappa|pi|sigma|omega|\^|_)")


def macros(path):
    t = io.open(path, encoding="utf-8", newline="").read()
    out = {}
    for m in re.finditer(r"\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}", t):
        name, body = m.group(1), m.group(2)
        # A macro that carries its own $...$ is self-delimiting: it is written
        # bare in prose on purpose and must not be reported.  Without this the
        # gate fires on 11 correct call sites and is worthless (the same
        # false-positive signature the archive gate had).
        if "$" in body:
            continue
        if MATHY.search(body):
            out[name] = body
    return out


def strip_comments(t):
    return re.sub(r"(?<!\\)%.*", "", t)


def math_spans(t):
    """(start, end) index pairs of inline/display math, comments removed first."""
    spans = []
    i, n = 0, len(t)
    while i < n:
        if t[i] == "\\" and t[i + 1:i + 2] == "[":
            j = t.find("\\]", i)
            if j < 0:
                break
            spans.append((i, j + 2))
            i = j + 2
        elif t[i] == "$":
            if t[i + 1:i + 2] == "$":
                j = t.find("$$", i + 2)
                if j < 0:
                    break
                spans.append((i, j + 2))
                i = j + 2
            else:
                j = i + 1
                while j < n:
                    if t[j] == "\\":
                        j += 2
                        continue
                    if t[j] == "$":
                        break
                    j += 1
                spans.append((i, j + 1))
                i = j + 1
        else:
            i += 1
    for env in ("equation", "equation*", "align", "align*", "gather", "gather*"):
        for m in re.finditer(r"\\begin\{%s\}(.*?)\\end\{%s\}" % (env, env), t, re.S):
            spans.append((m.start(), m.end()))
    return sorted(spans)


def inside(pos, spans):
    for a, b in spans:
        if a <= pos < b:
            return True
        if a > pos:
            break
    return False


def check(path, mm):
    raw = io.open(path, encoding="utf-8", newline="").read().replace("\r\n", "\n")
    t = strip_comments(raw)
    spans = math_spans(t)
    bad = []
    for name in mm:
        for m in re.finditer(r"\\%s(?![A-Za-z])" % name, t):
            if not inside(m.start(), spans):
                line = t[:m.start()].count("\n") + 1
                bad.append((line, name, mm[name]))
    return bad


def main():
    mm = macros(os.path.join(ROOT, "paper", "en", "gen_macros.tex"))
    bad = []
    for rel in ("submission/paper_en/main.tex", "submission/paper_cn/main.tex"):
        for line, name, body in check(os.path.join(ROOT, rel), mm):
            bad.append("%s:%d %s  ->  %s" % (os.path.basename(os.path.dirname(rel))
                                             + "/" + os.path.basename(rel),
                                             line, name, body[:40]))
    if bad:
        print("[FAIL] math macro used outside math in %d place(s):" % len(bad))
        for b in bad:
            print("   -", b)
        return 1
    print("[ok]   every math-valued macro is inside math (%d macros checked)" % len(mm))
    return 0


if __name__ == "__main__":
    sys.exit(main())
