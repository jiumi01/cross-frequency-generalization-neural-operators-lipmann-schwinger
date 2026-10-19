# Author: Longgang Wang <longgangwang@163.com>
# Affiliation: Luoyang Institute of Science and Technology, Luoyang 471000, Henan, China

"""Build the public release tree that backs the Data Availability statement.

This is the *public* counterpart of make_archive.py.  make_archive.py freezes an
internal revision snapshot for diffing and rollback; it deliberately carries
REVIEW_RESPONSE.md, because that document is the audit trail of how each round
of review was dispositioned.  None of that belongs in a repository a reader
clones, so the two tools copy different sets and this one is stricter about it.

What ships (and why):

  code/        every entry point named in the Data Availability statement, plus
               the gates and their falsification probes.  612 KB.
  paper/       LaTeX sources of both manuscripts, the bibliography, the style
               file and the generated tables/macros/figures, plus the compiled
               PDFs.  Excludes LaTeX build debris (.aux/.log/.out/.bbl/.blg).
  results/     the raw evaluation records every table and figure is derived
               from.  ~9 MB.  These are the evidence base, not a cache.
  data/real/   the *downloaded* Marmousi inputs of the third evidence level.
               ~19 MB.  Unlike the cached fields under data/, these cannot be
               regenerated from a seed, so without them that level's provenance
               is checkable in form only.

What does not ship, and the reason for each:

  data/train_* 9.1 GB of cached fields.  A fixed seed per cache chunk makes them
               bit-identical to the ones used here, so the generator ships
               instead of the caches -- exactly what the statement promises.
  runs/        2.9 GB of checkpoints and per-run logs.
  versions/    internal revision snapshots.
  reviews/     the three expert review reports and the reviewer scratch files.
  .workbuddy/  session-local working state (excluded from release).
  *scratch*    throwaway analysis scripts at the project root.
  REVIEW_RESPONSE.md  the internal review-disposition document.

Usage:  python code/make_release.py [dest]
        (default dest: <project root>/release)
"""
import os
import sys
import json
import shutil
import hashlib
import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))

SKIP_DIRS = {"__pycache__", ".ipynb_checkpoints", ".git", "figs_tmp"}
SKIP_SUFFIX = (".aux", ".log", ".out", ".bbl", ".blg", ".part",
               ".synctex.gz", ".fls", ".fdb_latexmk", ".toc", ".idx",
               ".ilg", ".ind", ".nav", ".snm", ".vrb", ".pyc")
# Files that exist at the project root and must never reach a public tree.
SKIP_ROOT_FILES = {"REVIEW_RESPONSE.md"}
SKIP_NAME_PATTERNS = ("scratch", "_reviewer", "reviewer", "bak")
# Pipeline bookkeeping under results/: records which stages ran on THIS
# machine, and some of it carries absolute paths.  It is not evidence and a
# reader reproducing the corpus does not need it, so it does not ship.  Note
# this is a name list, not a "dot-files" rule -- release_docs/.gitignore is a
# dot-file that very much has to ship.
SKIP_STATE_FILES = {".finish_state", ".lastmile_state", ".pipeline_done",
                    ".residual_state", ".residual2_state"}

# The curated copy set: (source relative to ROOT, whether to recurse).
COPY_TREE = ("code", "paper", "results", "data/real")
# Release metadata, copied flat into the destination (README.md, LICENSE,
# CITATION.cff, .gitignore, environment.yml, requirements.txt, docs/).
COPY_DOCS = "release_docs"


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def blocked(name):
    low = name.lower()
    if name in SKIP_ROOT_FILES or name in SKIP_STATE_FILES:
        return True
    return any(p in low for p in SKIP_NAME_PATTERNS)


def copy_tree(src, dst, rows, rel_base=""):
    n = 0
    for name in sorted(os.listdir(src)):
        if name in SKIP_DIRS or blocked(name):
            continue
        s = os.path.join(src, name)
        rel = os.path.join(rel_base, name).replace(os.sep, "/")
        if os.path.isdir(s):
            n += copy_tree(s, os.path.join(dst, name), rows, rel)
            continue
        if name.endswith(SKIP_SUFFIX):
            continue
        os.makedirs(dst, exist_ok=True)
        shutil.copy2(s, os.path.join(dst, name))
        rows.append((md5(s), os.path.getsize(s), rel))
        n += 1
    return n


def _fix_str(v, root_fwd, hit):
    """Replace one string's machine-specific prefix with a portable token."""
    if not isinstance(v, str):
        return v
    s = v.replace("\\", "/")
    if root_fwd in s:
        hit.append(1)
        return "<PROJECT_ROOT>" + s.split(root_fwd, 1)[1]
    return v


def _walk(obj, root_fwd, hit):
    if isinstance(obj, dict):
        return {k: _walk(v, root_fwd, hit) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_walk(v, root_fwd, hit) for v in obj]
    return _fix_str(obj, root_fwd, hit)


def sanitize_text(rows):
    """Rewrite machine-specific paths inside the copied tree.

    ``evaluate.py`` records the training cache directory it read from, so every
    ``eval_*.json`` in results/ carries this machine's absolute project path in
    its ``train_data`` field.  That is not a credential, but it is still not the
    reader's business: it leaks where the project sits on a private disk, it is
    meaningless on any other machine, and it would make two clones of the same
    release compare unequal for no reason.  The field is provenance metadata and
    no gate reads it, so it is rewritten to a token rather than dropped -- a
    reader can still tell WHICH cache was used, just not where it lived.
    """
    root_fwd = ROOT.replace("\\", "/")
    n = 0
    for _, _, rel in rows:
        p = os.path.join(DEST, rel)
        if rel.endswith(".json"):
            try:
                obj = json.load(open(p, encoding="utf-8"))
            except (ValueError, OSError):
                continue
            hit = []
            obj2 = _walk(obj, root_fwd, hit)
            if hit:
                with open(p, "w", encoding="utf-8") as fh:
                    json.dump(obj2, fh, ensure_ascii=False, indent=1)
                n += 1
            continue
        if not rel.endswith((".py", ".tex", ".bib", ".sty", ".sh", ".md",
                             ".yml", ".txt", ".cff")):
            continue
        try:
            t = open(p, encoding="utf-8").read()
        except (UnicodeDecodeError, OSError):
            continue
        t2 = t
        for form in (root_fwd, root_fwd.replace("/", "\\"),
                     root_fwd.replace("/", "\\\\"),
                     root_fwd.replace("/", "\\\\\\\\")):
            t2 = t2.replace(form, "<PROJECT_ROOT>")
        if t2 != t:
            with open(p, "w", encoding="utf-8") as fh:
                fh.write(t2)
            n += 1
    return n


def audit_text(rows):
    """Second-pass audit on the *copied* files, not on the sources.

    The point of auditing the copy is that a leak introduced by the copy step
    itself (a stale .log that still carries an absolute path, say) is caught
    here and not in the source tree, where it would look clean.
    """
    dest = DEST
    bad = []
    for _, _, rel in rows:
        if not rel.endswith((".py", ".tex", ".bib", ".sty", ".sh", ".md",
                             ".yml", ".txt", ".cff", ".json")):
            continue
        # The packager's own source necessarily spells out the very tokens it
        # scans for, so including it would match by construction and report a
        # leak that exists only in the instrument.  It is also the one file the
        # sanitiser must never rewrite.
        if os.path.basename(rel) == "make_release.py":
            continue
        p = os.path.join(dest, rel)
        try:
            txt = open(p, encoding="utf-8").read()
        except (UnicodeDecodeError, OSError):
            continue
        for i, line in enumerate(txt.splitlines(), 1):
            if "workbudd" in line or "C:\\Users" in line or "C:/Users" in line:
                bad.append("%s:%d absolute path: %s" % (rel, i, line.strip()[:80]))
            low = line.lower()
            for kw in ("api_key", "apikey", "password", "passwd",
                       "secret_key", "bearer ", "ghp_", "github_pat"):
                if kw in low:
                    bad.append("%s:%d credential-like token %r"
                               % (rel, i, kw))
    return bad


def main():
    global DEST
    dest = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "release")
    DEST = os.path.abspath(dest)
    if os.path.exists(DEST):
        # Refuse to rewrite a populated release: the manifest would no longer
        # describe what is on disk.
        print("refusing to overwrite existing release tree:", DEST)
        print("delete it first, or pass a fresh destination")
        return 1
    rows = []
    total = 0
    for sub in COPY_TREE:
        src = os.path.join(ROOT, sub)
        if not os.path.isdir(src):
            print("  [warn] missing %s -- skipped" % sub)
            continue
        total += copy_tree(src, os.path.join(DEST, sub), rows, sub)
    docs = os.path.join(ROOT, COPY_DOCS)
    if os.path.isdir(docs):
        total += copy_tree(docs, DEST, rows, "")
    else:
        print("  [warn] missing %s -- release tree will have no README/LICENSE"
              % COPY_DOCS)
    print("copied %d files -> %s" % (total, DEST))

    nfix = sanitize_text(rows)
    print("sanitised %d copied file(s) that carried a machine-specific path"
          % nfix)
    # Re-hash after sanitising: the manifest has to describe what is on disk,
    # and a checksum taken before the rewrite would describe a file that was
    # delivered only in its pre-sanitised form.
    rows = [(md5(os.path.join(DEST, r)),
             os.path.getsize(os.path.join(DEST, r)), r)
            for _, _, r in rows]

    bad = audit_text(rows)
    if bad:
        print("\n[AUDIT FAILED] %d issue(s) in the copied tree:" % len(bad))
        for b in bad[:20]:
            print("   ", b)
        print("\nrelease tree left in place for inspection; NOT shippable")
        return 2
    print("audit: no absolute paths, no credential-like tokens")

    rows.sort(key=lambda r: r[2])
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    nbytes = sum(n for _, n, _ in rows)
    with open(os.path.join(DEST, "MANIFEST.md"), "w", encoding="utf-8") as fh:
        fh.write("# MANIFEST - md5 checksums\n\n")
        fh.write("Generated: %s\n\n" % stamp)
        fh.write("Files: %d\nBytes: %d (%.1f MB)\n\n```\n"
                 % (len(rows), nbytes, nbytes / 1e6))
        for m, n, p in rows:
            fh.write("%s %9d  %s\n" % (m, n, p))
        fh.write("```\n")
    print("manifest: %d files, %.1f MB" % (len(rows), nbytes / 1e6))
    print("\nnext: write README.md / LICENSE / CITATION.cff / .gitignore,")
    print("      environment.yml, requirements.txt, docs/REPRODUCE.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
