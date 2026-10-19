# -*- coding: utf-8 -*-
"""One-shot structural reorganisation of Section 5 (Experiments) of the EN
manuscript, following the review's mainline recommendation:
characterise the error curve / quantify boundary and plateau / probe mechanisms
by controlled comparison.  Negative-result arms stay in the section but are
grouped under "Excluded mechanisms and supporting checks" (the text of the
excluded arms is compressed separately, with the full argument in Discussion
and the tables in the SI).  The script only moves blocks and demotes the 17
flat subsections to subsubsections under three new subsections; it does not
touch any prose.

Usage: python reorg_s5.py <path-to-main.tex>
"""
import io
import sys

def main():
    path = sys.argv[1]
    with io.open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # line numbers are 1-based; slice boundaries are 0-based half-open
    def sl(a, b):  # a,b 1-based inclusive -> list of lines
        return lines[a-1:b]

    # current blocks (from Grep of \subsection in main.tex, EN)
    head        = sl(922, 962)     # \section{Experiments} + overview table
    cliff       = sl(963, 1081)    # sec:cliff
    gridconv    = sl(1082, 1126)   # sec:gridconv
    scaling     = sl(1127, 1180)   # sec:scaling
    banddown    = sl(1181, 1223)   # sec:banddown
    arch        = sl(1224, 1303)   # sec:arch
    paired      = sl(1304, 1345)   # sec:paired
    mode        = sl(1346, 1404)   # sec:mode
    data        = sl(1405, 1447)   # sec:data
    band        = sl(1448, 1496)   # sec:band
    contrast    = sl(1497, 1545)   # sec:contrast
    spectrum    = sl(1546, 1644)   # sec:spectrum
    gamma       = sl(1645, 1668)   # sec:gamma
    c1          = sl(1669, 1715)   # sec:c1
    disc        = sl(1716, 1779)   # sec:disc
    interv      = sl(1780, 1851)   # sec:interv
    anchor      = sl(1852, 1915)   # sec:anchor
    qualitative = sl(1916, 1929)   # (no label)
    rest        = lines[1929:]     # Discussion onward (index 1929 = line 1930)

    def demote(block):
        out = []
        for ln in block:
            if ln.startswith("\\subsection{"):
                ln = ln.replace("\\subsection{", "\\subsubsection{", 1)
            out.append(ln)
        return out

    b5_1 = [
        "\\subsection{Characterising the error curve, its boundary and its plateau}\n",
        "The first block is the primary result: on the strictly delimited periodic\n",
        "Lippmann--Schwinger benchmark of Section~\\ref{sec:protocol} the relative $L_2$\n",
        "error curve is flat in band, steps at the training edge, and plateaus at the\n",
        "trivial baseline; the block quantifies the boundary and the plateau and\n",
        "separates the plateau from a band-limitation artefact.\n",
    ]
    b5_2 = [
        "\\subsection{Mechanism probes by controlled comparison}\n",
        "The second block tests which ingredient sets the boundary: architecture and\n",
        "parameter budget, mode budget, training data, and the analytic Born anchor.\n",
        "Each arm is paired with the baseline and differs in a single controlled\n",
        "ingredient, so a claim is read off the arm that excludes its alternative.\n",
    ]
    b5_3 = [
        "\\subsection{Excluded mechanisms and supporting checks}\n",
        "The third block reports the discriminating arms that failed to move the\n",
        "boundary, each compressed to its conclusion with the full argument in\n",
        "Discussion, and the two supporting checks that the phenomena are not\n",
        "artefacts of the grid or of the synthetic medium.\n",
    ]

    new = []
    new += head
    new += b5_1
    new += demote(cliff)
    new += demote(scaling)
    new += demote(banddown)
    new += demote(contrast)
    new += demote(spectrum)
    new += b5_2
    new += demote(arch)
    new += demote(paired)
    new += demote(mode)
    new += demote(data)
    new += demote(band)
    new += demote(anchor)
    new += b5_3
    new += demote(gamma)
    new += demote(c1)
    new += demote(disc)
    new += demote(interv)
    new += demote(gridconv)
    new += demote(qualitative)
    new += rest

    with io.open(path, "w", encoding="utf-8", newline="\n") as f:
        f.writelines(new)
    print("reorganised: %d lines -> %d lines" % (len(lines), len(new)))

if __name__ == "__main__":
    main()
