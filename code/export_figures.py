# Author: Longgang Wang <longgangwang@163.com>

"""Export the eight main-text figures as standalone files.

The figures have to leave as individual files -- vector PDF plus a 600 dpi
raster -- rather than as crops of the compiled paper.  Cropping the PDF
gives whatever the page happened to scale the figure to; these are rendered
from the figure's own PDF at its natural size, so the vector version carries the
original coordinates and the raster is sampled from those coordinates rather
than from a down-scaled placement.

Output: figures/export/Figure<n>_<name>.{pdf,png,tif}
  pdf  copy of the vector original
  png  600 dpi, RGB
  tif  600 dpi, LZW-compressed (journals that ask for TIFF want a lossless one)

Run: python code/export_figures.py
"""
import os
import re
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = os.path.join(ROOT, "figures")
OUTDIR = os.path.join(FIGDIR, "export")
MANUSCRIPT = os.path.join(ROOT, "submission", "paper_en", "main.tex")

# Order is the order of first citation in the manuscript, so the file numbers
# match the printed figure numbers.  Keep this list in step with main.tex: the
# check below asserts that the source list is exactly the \includegraphics
# sequence of the delivery copy, in order.  A figure added to the paper but not
# here would be exported under someone else's number; a figure dropped from the
# paper would still ship an orphan file, and an exported Figure 7 would then
# be a different figure from the paper's Figure 7.
FIGURES = [
    (1, "fig_main_cliff"),
    (2, "fig_crosscontrast"),
    (3, "fig_plateau_coupling"),
    (4, "fig_arch"),
    (5, "fig_mode_phase_b4"),
    (6, "fig_bandwidth"),
    (7, "fig_anchor"),
    (8, "fig_deep"),
]
DPI = 600


def manuscript_figures(path):
    r"""The \includegraphics sequence of the manuscript, in source order."""
    with open(path, encoding="utf-8") as fh:
        body = fh.read()
    inc = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", body)
    return [os.path.splitext(os.path.basename(x))[0] for x in inc]


def main():
    names = manuscript_figures(MANUSCRIPT)
    want = [name for _, name in FIGURES]
    if names != want:
        print("[FAIL] FIGURES does not match the manuscript's "
              "\\includegraphics order")
        print("       manuscript: %s" % names)
        print("       export    : %s" % want)
        extra = [n for n in names if n not in want]
        miss = [n for n in want if n not in names]
        if extra:
            print("       not exported: %s" % extra)
        if miss:
            print("       not in the paper: %s" % miss)
        return 1
    if [n for n, _ in FIGURES] != list(range(1, len(FIGURES) + 1)):
        print("[FAIL] figure numbers are not 1..%d in order" % len(FIGURES))
        return 1

    os.makedirs(OUTDIR, exist_ok=True)
    made = []
    for n, name in FIGURES:
        src = os.path.join(FIGDIR, name + ".pdf")
        if not os.path.exists(src):
            print("[FAIL] missing %s" % src)
            return 1
        stem = os.path.join(OUTDIR, "Figure%d_%s" % (n, name))
        shutil.copy2(src, stem + ".pdf")
        # pdftoppm wrote the raster at the figure's own size; -r sets the
        # sampling density, so 600 dpi here means 600 px per inch of the
        # figure, not of the page it was placed on.
        r = subprocess.run(["pdftoppm", "-r", str(DPI), "-png", "-singlefile",
                            src, stem], capture_output=True)
        if r.returncode != 0:
            print("[FAIL] pdftoppm %s: %s" % (name, r.stderr.decode()[:200]))
            return 1
        from PIL import Image
        im = Image.open(stem + ".png").convert("RGB")
        im.save(stem + ".tif", compression="tiff_lzw", dpi=(DPI, DPI))
        made.append((n, name, im.size))
        print("Figure %d  %-24s %5d x %5d px  pdf %6.1f kB"
              % (n, name, im.size[0], im.size[1],
                 os.path.getsize(stem + ".pdf") / 1024.0))

    # The raster must be big enough to be a 600 dpi rendering and not a
    # re-sampled thumbnail: 600 dpi over the 47 cm canvas the figures use is
    # ~11000 px wide, so anything under 2000 px means the density flag was
    # ignored.
    thin = [m for m in made if m[2][0] < 2000]
    if thin:
        print("[FAIL] raster too small, dpi flag ignored: %s" % thin)
        return 1
    print("\n%d figures exported to %s" % (len(made), OUTDIR))
    return 0


if __name__ == "__main__":
    sys.exit(main())
