# -*- coding: utf-8 -*-
"""Assert that every `$` in the manuscripts is paired.

Why this exists
---------------
A restructure script that slices text between two anchors cut one sentence
mid-macro and left a lone `$` at the end of a line:

    ...the analytic head and the media that are reused across $.

`-interaction=nonstopmode` without `-halt-on-error` reports ``Missing $ inserted``
and then still writes a PDF, so a hand-run build looked healthy while
`make_paper.py` (which does pass `-halt-on-error`) refused to compile.  Reading
the page count therefore did not detect the damage; the only reliable test is a
delimiter census.

What is checked
---------------
For each file, walk the source with a small state machine and count `$` that
actually open or close inline maths.  Skipped: `\\$` (literal dollar), comments
(`%` to end of line, but not `\\%`), and the contents of verbatim-like
environments where `$` carries no meaning.  At EOF the machine must be in text
mode; any imbalance is reported with the line where the count first went odd,
which is where the offending `$` sits.

This is deliberately independent of what the file *says* -- it is a lexical
check, so it cannot be fooled by a sentence that reads fine but typesets badly.
"""
import io
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FILES = [
    "submission/paper_en/main.tex",
    "submission/paper_cn/main.tex",
    "submission/paper_en/SI_main.tex",
]

# `$` inside these carries no meaning: verbatim reproduces source characters.
VERBATIM = ("verbatim", "Verbatim", "lstlisting", "minted", "alltt")


def scan(text):
    """Return (delta_after_file, [(line, kind)]) for one source string.

    ``delta`` is the number of unclosed `$` at EOF; ``odd_lines`` lists the
    lines after which the running count became odd, i.e. candidate culprits.
    """
    delta = 0
    odd = []
    line = 1
    i = 0
    n = len(text)
    verbatim = None          # name of the open verbatim-like environment
    while i < n:
        ch = text[i]
        if ch == "\n":
            line += 1
            i += 1
            continue
        # --- verbatim environments: copy through untouched -----------------
        if verbatim is not None:
            end = "\\end{%s}" % verbatim
            j = text.find(end, i)
            if j < 0:
                return delta, odd
            line += text.count("\n", i, j)
            i = j + len(end)
            verbatim = None
            continue
        if ch == "\\":
            m = re.match(r"\\begin\{([A-Za-z*]+)\}", text[i:])
            if m and m.group(1) in VERBATIM:
                verbatim = m.group(1)
                line += text.count("\n", i, i + m.end())
                i += m.end()
                continue
            # any other control sequence: skip the backslash and the token, so
            # that `\$` and `\%` never count as delimiters or as comments
            i += 2 if i + 1 < n else 1
            continue
        if ch == "%":
            j = text.find("\n", i)
            if j < 0:
                break
            line += 1
            i = j + 1
            continue
        if ch == "$":
            delta = 1 - delta
            if delta == 1:
                odd.append(line)
            i += 1
            continue
        i += 1
    return delta, odd


SELF_TEST = [
    # (name, source, expected unclosed count)
    ("lone trailing $", "the media that are reused across $.\n", 1),
    ("paired", "the media reused across $f$.\n", 0),
    ("escaped dollar", "costs 5 \\$ and 6 \\$.\n", 0),
    ("comment hides", "text % a $ b\n", 0),
    ("verbatim hides", "\\begin{verbatim}$ \\end{verbatim}\n", 0),
    ("two balanced lines", "a $x$ b\nc $y$ d\n", 0),
    ("unclosed inline", "a $x b\n", 1),
]


def self_test():
    bad = 0
    for name, src, want in SELF_TEST:
        got = 1 if scan(src)[0] else 0
        if got != want:
            print("FAIL  probe %-20s want %d, got %d" % (name, want, got))
            bad += 1
    if bad:
        print("RESULT: %d probe(s) failed" % bad)
        return 1
    print("RESULT: %d probes agree (the checker fails where it must and is "
          "silent where it must)" % len(SELF_TEST))
    return 0


def main():
    if "--self-test" in sys.argv:
        return self_test()
    problems = []
    for rel in FILES:
        path = os.path.join(ROOT, rel)
        if not os.path.exists(path):
            problems.append("%s: missing" % rel)
            continue
        text = io.open(path, encoding="utf-8", newline="").read()
        delta, odd = scan(text)
        if delta:
            where = "line %d" % odd[-1] if odd else "start of file"
            problems.append("%s: %d unclosed `$` (count first goes odd after %s)"
                            % (rel, delta, where))
        else:
            print("[ok] %s: $ delimiters balanced" % rel)
    if problems:
        for p in problems:
            print("FAIL  " + p)
        print("RESULT: %d problem(s)" % len(problems))
        return 1
    print("RESULT: all files balanced")
    return 0


if __name__ == "__main__":
    sys.exit(main())
