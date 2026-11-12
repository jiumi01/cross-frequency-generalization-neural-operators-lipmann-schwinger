#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""One-off: compile paper/<lang>/SI_main.tex to submission/paper_<lang>/SI_main.pdf.

make_paper's build stage only compiles the main manuscript and delivers its PDF;
the SI PDF is never produced by the pipeline, so it drifts whenever the SI source
is edited.  This mirrors compile_paper: stage in an ASCII temp dir (the project
path is non-ASCII), run the engine 3x (cross-references need the passes), and
deliver the PDF back to submission/.
"""
import os, sys, shutil, subprocess, tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LANGS = [("en", "pdflatex"), ("cn", "xelatex")]

for lang, engine in LANGS:
    src = os.path.join(ROOT, "paper", lang)
    if not os.path.exists(os.path.join(src, "SI_main.tex")):
        print("[si:%s] no SI_main.tex, skip" % lang)
        continue
    # Structural gate, added after S4--S25 were found sitting *after*
    # \end{document}: LaTeX compiles that silently, so the delivered PDF was
    # three pages long while the manuscript promised twenty-five sections, and
    # no existing gate looked at the SI body.  Anything non-blank after the
    # terminator is dead text, and dead text here means missing sections.
    body = open(os.path.join(src, "SI_main.tex"), encoding="utf-8").read()
    tail_at = body.find("\\end{document}")
    if tail_at < 0:
        print("[si:%s] no \\end{document}" % lang)
        sys.exit(1)
    dead = body[tail_at + len("\\end{document}"):].strip()
    if dead:
        print("[si:%s] %d characters after \\end{document} are never typeset:"
              % (lang, len(dead)))
        print("   |", dead.splitlines()[0][:90])
        sys.exit(1)

    tmp = tempfile.mkdtemp(prefix="si_%s_" % lang)
    work = os.path.join(tmp, "p")
    shutil.copytree(src, work)
    cmds = [[engine, "-interaction=nonstopmode", "-halt-on-error",
             "-file-line-error", "SI_main.tex"]] * 3
    for i, cmd in enumerate(cmds):
        p = subprocess.run(cmd, cwd=work, capture_output=True, text=True,
                           encoding="utf-8", errors="backslashreplace")
        log = os.path.join(work, "SI_main.log")
        if p.returncode != 0 and os.path.exists(log):
            txt = open(log, encoding="utf-8", errors="backslashreplace").read()
            errs = [l for l in txt.splitlines() if l.startswith("!")]
            print("[si:%s] step %d failed" % (lang, i))
            for e in errs[:15]:
                print("   |", e)
            if not errs:
                for l in txt.splitlines()[-25:]:
                    print("   |", l)
            sys.exit(1)
    pdf = os.path.join(work, "SI_main.pdf")
    if not os.path.exists(pdf):
        print("[si:%s] no PDF produced" % lang)
        sys.exit(1)
    # Keep the build output next to its sources first.  That path is never held
    # open by the desktop preview, so it always gets the fresh file; the
    # delivered copy is attempted afterwards.  Without this, a preview holding
    # submission/paper_<lang>/SI_main.pdf made the script exit on the
    # PermissionError and the *other* language was never compiled at all --
    # one locked file silently costing a whole document.
    keep = os.path.join(src, "SI_main.pdf")
    shutil.copy2(pdf, keep)
    # The .aux is the only independent record of the numbers LaTeX actually
    # assigned, and `check_si_refs.py` cross-checks the generated symbol map
    # against it.  That instrument was reading a stale aux, because only the
    # PDF was ever delivered; a map that disagreed with the render therefore
    # passed by comparing itself with an older render.
    aux = os.path.join(work, "SI_main.aux")
    if os.path.exists(aux):
        shutil.copy2(aux, os.path.join(src, "SI_main.aux"))
        shutil.copy2(aux, os.path.join(ROOT, "submission", "paper_%s" % lang,
                                       "SI_main.aux"))
    dst = os.path.join(ROOT, "submission", "paper_%s" % lang, "SI_main.pdf")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        shutil.copy2(pdf, dst)
        print("[si:%s] wrote %s (%d bytes)" % (lang, dst, os.path.getsize(pdf)))
    except PermissionError:
        print("[si:%s] could not replace %s: the file is open in another "
              "program.  Close it and copy %s over it." % (lang, dst, keep))
    shutil.rmtree(tmp, ignore_errors=True)
print("SI PDFs compiled and delivered.")
