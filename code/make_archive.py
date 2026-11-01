# Author: Longgang Wang <longgangwang@163.com>
# Affiliation: Luoyang Institute of Science and Technology, Luoyang 471000, Henan, China

"""Freeze an immutable revision snapshot under versions/.

A snapshot is copy-only: it never writes into the working tree.  What it copies
is the set of artefacts a reader needs to reproduce the delivered PDFs --
code/, paper/, results/ -- plus a CHANGELOG and an md5 manifest.  Internal
administrative documents are not part of that set; pass --with-admin to add
them to a local rollback copy.  Earlier snapshots are left untouched so
revisions can be diffed and rolled back.

Usage:  python code/make_archive.py [--with-admin] v03_2026-10-12_mdpi_single_column
"""
import glob
import os
import sys
import shutil
import hashlib
import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))

SKIP_DIRS = {"__pycache__", ".ipynb_checkpoints"}
SKIP_FILE_SUFFIX = (".aux", ".log", ".out", ".bbl", ".blg", ".part", ".synctex.gz")
# Driver transcripts and build stamps.  make_release.py has always refused
# these, and the snapshot is described in the Data Availability statement as a
# superset of that release -- so shipping them here put the two doors back out
# of step one layer down.  They record which stage ran on this disk, they are
# evidence for nothing, and the build stamp made the archive's own md5 manifest
# differ between two rebuilds of the same sources.
SKIP_STATE_FILES = {".finish_state", ".lastmile_state", ".pipeline_done",
                    ".residual_state", ".residual2_state"}


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def copy_tree(src, dst, base, rows):
    for name in sorted(os.listdir(src)):
        if name in SKIP_DIRS:
            continue
        s = os.path.join(src, name)
        if os.path.isdir(s):
            copy_tree(s, os.path.join(dst, name), base, rows)
            continue
        if name.endswith(SKIP_FILE_SUFFIX) or name in SKIP_STATE_FILES:
            continue
        os.makedirs(dst, exist_ok=True)
        shutil.copy2(s, os.path.join(dst, name))
        rows.append((md5(s), os.path.getsize(s), os.path.relpath(s, base)
                     .replace(os.sep, "/")))


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        print("usage: make_archive.py [--with-admin] <version-label>"); return 1
    label = args[0]
    dest = os.path.join(ROOT, "versions", label)
    if os.path.exists(dest):
        # A directory that already carries the copied trees is a finished
        # snapshot and must never be rewritten.  A directory holding only a
        # hand-written CHANGELOG is not: the changelog is normally drafted
        # first, so the copy pass still has to run into it.
        stale = [s for s in ("code", "paper", "results", "data/real")
                 if os.path.exists(os.path.join(dest, s))]
        if stale:
            print("refusing to overwrite populated snapshot:", dest)
            return 1
    os.makedirs(dest, exist_ok=True)
    rows = []
    # code/paper/results are the artefacts a reader needs.  data/real is added
    # on top of that list for a specific reason: it holds *downloaded inputs*,
    # not generated ones.  The cached fields elsewhere under data/ are a
    # deterministic function of a seed, so the archive ships their generator
    # instead; the Marmousi copies cannot be regenerated that way, and without
    # them the third evidence level's provenance record is checkable in form
    # only.  They are ~19 MB, which is affordable.
    for sub in ("code", "paper", "results", "data/real"):
        src = os.path.join(ROOT, sub)
        if os.path.isdir(src):
            copy_tree(src, os.path.join(dest, sub), ROOT, rows)
    # Administrative documents of this class are internal.  The snapshot named
    # in the Data Availability statement is handed to a reader, so it carries
    # the reproduction artefacts and nothing else; --with-admin is for a local
    # rollback copy that also wants the disposition trail.
    if "--with-admin" in sys.argv:
        docs = sorted(glob.glob(os.path.join(ROOT, "REVIEW_*.md")))
        if os.path.exists(os.path.join(ROOT, "REVIEW_RESPONSE.md")):
            docs.insert(0, os.path.join(ROOT, "REVIEW_RESPONSE.md"))
        seen = set()
        for src in docs:
            doc = os.path.basename(src)
            if doc in seen:
                continue
            seen.add(doc)
            shutil.copy2(src, os.path.join(dest, doc))
            rows.append((md5(src), os.path.getsize(src), doc))
    # The four delivered PDFs come from submission/, not from paper/.
    # paper/<lang>/main.pdf is a build intermediate and it is the file a desktop
    # viewer holds open; when it is locked make_paper falls back to main_new.pdf
    # and the intermediate stays behind the delivery.  Copying the intermediate
    # would make the snapshot cite an older render than the one a reader is
    # handed -- and check_das compares exactly these files against the delivery.
    for lang in ("en", "cn"):
        for name in ("main.pdf", "SI_main.pdf"):
            src = os.path.join(ROOT, "submission", "paper_%s" % lang, name)
            rel = "paper/%s/%s" % (lang, name)
            dst = os.path.join(dest, "paper", lang, name)
            if not os.path.exists(src):
                continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
            rows = [r for r in rows if r[2] != rel]
            rows.append((md5(src), os.path.getsize(src), rel))
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # A file drafted into the snapshot before the copy pass (the CHANGELOG) is
    # part of the snapshot like any other, so account for it rather than ship a
    # manifest that silently under-reports the contents.
    accounted = set(p for _, _, p in rows)
    for name in sorted(os.listdir(dest)):
        full = os.path.join(dest, name)
        if os.path.isfile(full) and name not in accounted and name != "MANIFEST.md":
            rows.append((md5(full), os.path.getsize(full), name))
    rows.sort(key=lambda r: r[2])
    with open(os.path.join(dest, "MANIFEST.md"), "w", encoding="utf-8") as fh:
        fh.write("# MANIFEST — md5 checksums (%s)\n\n" % label)
        fh.write("Generated: %s\n\nFiles: %d\n\n```\n" % (stamp, len(rows)))
        for m, n, p in rows:
            fh.write("%s %9d  %s\n" % (m, n, p))
        fh.write("```\n")
    print("archived %d files -> %s" % (len(rows), dest))
    return 0


if __name__ == "__main__":
    sys.exit(main())
