# Author: Longgang Wang <longgangwang@163.com>
# Affiliation: Luoyang Institute of Science and Technology, Luoyang 471000, Henan, China

"""Deliverable state readout: the numbers the response document quotes.

Every figure here is measured from the shipped artefacts rather than
transcribed, so a statement in the response document can be re-derived by
re-running this script instead of by trusting a hand-copied number.
"""
import os
import re
import sys
import glob
import subprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

try:
    import matplotlib  # noqa: F401
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False


def pdf_text(path):
    for exe in ("pdftotext",):
        try:
            p = subprocess.run([exe, "-layout", path, "-"], capture_output=True,
                               text=True, encoding="utf-8", errors="replace")
            if p.returncode == 0 and p.stdout:
                return p.stdout
        except FileNotFoundError:
            return None
    return None


def pdf_pages(path):
    p = subprocess.run(["pdfinfo", path], capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    m = re.search(r"^Pages:\s+(\d+)", p.stdout, re.M)
    return int(m.group(1)) if m else None


def main():
    print("=== deliverable state ===")
    for lang in ("en", "cn"):
        pdf = os.path.join(C.ROOT, "paper", lang, "main.pdf")
        if not os.path.exists(pdf):
            print("%s: no PDF" % lang); continue
        txt = pdf_text(pdf)
        n_dash = len(re.findall(r"(?<!-)--(?!-)", txt)) if txt else None
        n_endash = len(re.findall("\u2013", txt)) if txt else None
        n_chars = len(txt) if txt else None
        figs = len(glob.glob(os.path.join(C.ROOT, "paper", lang, "figs", "*.pdf")))
        print("%s: pages=%s chars=%s  literal '--'=%s  en-dash=%s  figures=%d"
              % (lang, pdf_pages(pdf), n_chars, n_dash, n_endash, figs))

    print("\n=== sources ===")
    for lang in ("en", "cn"):
        main_tex = os.path.join(C.ROOT, "paper", lang, "main.tex")
        s = open(main_tex, encoding="utf-8").read()
        caps = re.findall(r"\\caption\{", s)
        print("%s: main.tex %d lines, %d captions in main, cites=%d"
              % (lang, s.count("\n") + 1, len(caps),
                 len(re.findall(r"\\cite\{", s))))
    bib = os.path.join(C.ROOT, "paper", "en", "refs.bib")
    print("refs.bib: %d entries" % len(re.findall(r"^@", open(bib, encoding="utf-8").read(),
                                                  re.M)))


if __name__ == "__main__":
    main()
