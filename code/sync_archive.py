# -*- coding: utf-8 -*-
"""Refresh submission/supplementary from the working tree, then rewrite MANIFEST.

The archive is what the Data Availability Statement points at, so a reader must
be able to rebuild the delivered PDFs from it.  It drifted twice without anyone
noticing, because the drift is silent in both directions:

  * paper/en lacked gen_si_refs.tex, so the archived main.tex inputs a file that
    is not there and cannot be compiled at all;
  * code/ kept an old check_si_refs.py and had no gen_si_labels.py, so nothing
    in the archive could regenerate the very file that was missing.

The MANIFEST described the tree on disk correctly the whole time.  A manifest
cannot detect that the disk disagrees with the repository, so this script ends by
comparing the delivered manuscripts against the archived copies byte for byte.

Copy only.  Archive-only material (seeds 3-5 evaluation records, the cached
Marmousi inputs) is never deleted.
"""
import hashlib
import os
import shutil
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARCHIVE = os.path.join(ROOT, "submission", "supplementary")

SKIP_DIRS = {"__pycache__", ".ipynb_checkpoints"}
SKIP_SUFFIX = (".pyc", ".log", ".aux", ".out", ".toc", ".spl", ".fls",
               ".fdb_latexmk", ".synctex.gz", ".bbl", ".blg")
SKIP_PREFIX = ("_tmp_", "_scan_", ".")

# the six files a referee is handed, checked after the copy
DELIVERED = [
    ("paper/en/main.tex", "paper_en/main.tex"),
    ("paper/en/main.pdf", "paper_en/main.pdf"),
    ("paper/en/SI_main.tex", "paper_en/SI_main.tex"),
    ("paper/en/SI_main.pdf", "paper_en/SI_main.pdf"),
    ("paper/cn/main.tex", "paper_cn/main.tex"),
    ("paper/cn/main.pdf", "paper_cn/main.pdf"),
]


def skip(name):
    if name.startswith(SKIP_PREFIX) and name not in (".gitignore",):
        return True
    return name.endswith(SKIP_SUFFIX)


def sync_tree(src_sub):
    """Copy src_sub from the repository into the archive; add and overwrite."""
    src = os.path.join(ROOT, src_sub)
    dst = os.path.join(ARCHIVE, src_sub)
    copied = added = 0
    for dirpath, dirnames, filenames in os.walk(src):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            if skip(fn):
                continue
            s = os.path.join(dirpath, fn)
            rel = os.path.relpath(s, src)
            d = os.path.join(dst, rel)
            os.makedirs(os.path.dirname(d), exist_ok=True)
            if os.path.exists(d):
                if os.path.getsize(d) == os.path.getsize(s) and \
                        md5(d) == md5(s):
                    continue
                copied += 1
            else:
                added += 1
            shutil.copy2(s, d)
    print("[archive] %-10s %3d overwritten, %3d new" % (src_sub, copied, added))
    return copied, added


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest():
    rows = []
    for dirpath, dirnames, filenames in os.walk(ARCHIVE):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            p = os.path.join(dirpath, fn)
            rel = os.path.relpath(p, ARCHIVE).replace(os.sep, "/")
            if rel == "MANIFEST.md":
                continue
            rows.append((rel, os.path.getsize(p), md5(p)))
    rows.sort()
    total = sum(r[1] for r in rows)
    out = ["# MANIFEST - md5 checksums", "",
           "Generated: %s" % time.strftime("%Y-%m-%d %H:%M:%S"),
           "", "Files: %d" % len(rows),
           "Bytes: %d (%.1f MB)" % (total, total / 1048576.0), "", "```"]
    for rel, size, dig in rows:
        out.append("%s  %10d  %s" % (dig, size, rel))
    out += ["```", ""]
    path = os.path.join(ARCHIVE, "MANIFEST.md")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(out))
    print("[archive] MANIFEST.md: %d files, %.1f MB" % (len(rows),
                                                       total / 1048576.0))


def verify():
    bad = []
    for arch_rel, live_rel in DELIVERED:
        a = os.path.join(ARCHIVE, arch_rel)
        b = os.path.join(ROOT, "submission", live_rel)
        if not os.path.exists(a):
            bad.append("%s missing from the archive" % arch_rel)
            continue
        if md5(a) != md5(b):
            bad.append("%s differs from the delivered copy" % arch_rel)
    # the manuscript cannot be compiled without the generated cross-ref map
    for lang in ("en", "cn"):
        need = os.path.join(ARCHIVE, "paper", lang, "gen_si_refs.tex")
        src = os.path.join(ROOT, "paper", lang, "gen_si_refs.tex")
        if os.path.exists(src) and not os.path.exists(need):
            bad.append("paper/%s/gen_si_refs.tex missing (main.tex inputs it)"
                       % lang)
    if bad:
        for b in bad:
            print("[FAIL] %s" % b)
        return 1
    print("[ok] the six delivered files match the archive byte for byte, and "
          "the generated SI map is present")
    return 0


if __name__ == "__main__":
    for sub in ("code", "paper/en", "paper/cn"):
        sync_tree(sub)
    write_manifest()
    sys.exit(verify())
