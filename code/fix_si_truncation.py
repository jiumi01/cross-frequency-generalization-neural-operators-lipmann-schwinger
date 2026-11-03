# -*- coding: utf-8 -*-
"""Move the SI bibliography block to the real end of the document.

Both SI sources had \\end{document} immediately after S3, with S4--S25 after it,
so 22 of the 25 sections the manuscript promises were never typeset and never
delivered.  The block is cut out and re-appended at the end.
"""
import io
import sys

FILES = ["submission/paper_en/SI_main.tex", "submission/paper_cn/SI_main.tex"]

for p in FILES:
    s = io.open(p, encoding="utf-8").read()
    b0 = s.find("\\begin{thebibliography}")
    e0 = s.find("\\end{document}")
    assert b0 > 0 and e0 > b0, p
    blk = s[b0:e0 + len("\\end{document}")]
    rest = s[:b0].rstrip() + "\n" + s[e0 + len("\\end{document}"):].lstrip("\n")
    # The tail must still carry all 25 sections, and the block must occur once.
    assert rest.count("\\begin{thebibliography}") == 0, p
    assert rest.count("\\end{document}") == 0, p
    assert rest.count("\\section*{S") == 25, (p, rest.count("\\section*{S"))
    out = rest.rstrip() + "\n\n" + blk.strip() + "\n"
    io.open(p, "w", encoding="utf-8", newline="\n").write(out)
    print("moved bibliography to end of", p,
          "| sections:", out.count("\\section*{S"))
