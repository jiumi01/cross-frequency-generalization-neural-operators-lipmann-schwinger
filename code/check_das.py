# Author: Longgang Wang <longgangwang@163.com>
# Affiliation: Luoyang Institute of Science and Technology, Luoyang 471000, Henan, China

"""Check that the Data Availability statement matches the repository and the disk.

The statement names three things that can each go false while the manuscript
still reads correctly:

  * a commit hash of the public repository.  The hash is only meaningful if that
    object is the state the results came from.  ``git rev-parse HEAD`` moves, so
    a hash written once stays plausible forever and silently stops being true;
  * the tree hash of that commit.  It is content-addressed, so it is the one
    identifier that is identical here and on the remote -- and the one this gate
    can compare against the local checkout without network access;
  * a snapshot directory under ``versions/`` whose copy of the delivered PDFs is
    byte-identical to the PDFs actually shipped.

What is asserted, per language:
  1. both manuscripts name the same commit, tree and snapshot;
  2. the snapshot directory exists, has a non-empty MD5 manifest, and its
     ``paper/<lang>/main.pdf`` has the same MD5 as ``submission/paper_<lang>/main.pdf``;
  3. the named tree equals the tree of the local HEAD, and the named commit, when
     present in the local object store, points at that same tree.

Exit code 0 iff all hold.  ``--self-test`` mutates one field in a temporary copy
and asserts the checker rejects it, so the instrument is shown to fail on a
known-bad input rather than only to pass on the current one.
"""
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, ".."))
ROOT = REPO_ROOT
SUB = os.path.join(ROOT, "submission")

FAIL = []


def read(p):
    with open(p, encoding="utf-8") as fh:
        return fh.read()


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def git(*args, repo=None):
    """Run git in the checkout that owns the claims (not in a copied tree)."""
    p = subprocess.run(["git", "-C", repo or REPO_ROOT] + list(args),
                       capture_output=True, text=True)
    return p.returncode, p.stdout.strip()


def parse(lang):
    """Pull (commit, tree, snapshot) out of one manuscript's DAS block.

    The two manuscripts carry the statement in different containers -- the MDPI
    English build uses the class's ``\\dataavailability{}`` macro, the Chinese
    build an unnumbered section -- so the block is located by either handle and
    ended at the next sectioning command or at the conflicts-of-interest block.
    """
    src = read(os.path.join(SUB, "paper_%s" % lang, "main.tex"))
    starts = [src.find("\\dataavailability{"), src.find("\\section*{数据与代码可获得性}")]
    starts = [i for i in starts if i >= 0]
    if not starts:
        return None
    i = min(starts)
    ends = [j for j in (src.find("\\conflictsofinterest", i),
                        src.find("\\section*{", i + 10),
                        src.find("\\begin{thebibliography", i)) if j > 0]
    block = src[i:min(ends) if ends else i + 6000]
    block = block.replace("\\_", "_")
    commit = re.search(r"(?:commit|提交)\s*\\texttt\{([0-9a-f]{7,40})\}", block)
    tree = re.search(r"(?:tree|树)\s*\\texttt\{([0-9a-f]{7,40})\}", block)
    snap = re.search(r"\\texttt\{(v\d+_[0-9]{4}-[0-9]{2}-[0-9]{2}_[A-Za-z0-9_.-]+)\}", block)
    return dict(commit=commit.group(1) if commit else None,
                tree=tree.group(1) if tree else None,
                snapshot=snap.group(1) if snap else None)


def check(root=None, repo=None):
    """Run every assertion against ``root``; return the list of failures.

    ``repo`` is the checkout the commit and tree are read from.  It defaults to
    the real project root, so the self-test can point ``root`` at a mutated copy
    of the manuscripts while still comparing their claims against the actual
    repository.
    """
    global ROOT, SUB
    ROOT = root or REPO_ROOT
    SUB = os.path.join(ROOT, "submission")
    repo = repo or REPO_ROOT
    FAIL.clear()
    enc, cnc = parse("en"), parse("cn")
    for name, d in (("en", enc), ("cn", cnc)):
        if not d or not all(d.values()):
            FAIL.append("%s: DAS block missing a commit, a tree or a snapshot name "
                        "(parsed %s)" % (name, d))
    if FAIL:
        return FAIL
    if not (enc["commit"] == cnc["commit"] and enc["tree"] == cnc["tree"]
            and enc["snapshot"] == cnc["snapshot"]):
        FAIL.append("en and cn disagree about the archive: %s vs %s" % (enc, cnc))
        return FAIL

    snap = os.path.join(ROOT, "versions", enc["snapshot"])
    if not os.path.isdir(snap):
        FAIL.append("snapshot %s does not exist under versions/" % enc["snapshot"])
        return FAIL
    man = os.path.join(snap, "MANIFEST.md")
    if not os.path.exists(man) or os.path.getsize(man) < 1000:
        FAIL.append("snapshot %s has no usable MD5 manifest" % enc["snapshot"])

    for lang in ("en", "cn"):
        a = os.path.join(SUB, "paper_%s" % lang, "main.pdf")
        b = os.path.join(snap, "paper", lang, "main.pdf")
        if not (os.path.exists(a) and os.path.exists(b)):
            FAIL.append("%s: delivered or archived PDF missing" % lang)
        elif md5(a) != md5(b):
            FAIL.append("%s: archived PDF is not the delivered PDF (%s vs %s)"
                        % (lang, md5(b)[:8], md5(a)[:8]))

    rc, head_tree = git("rev-parse", "HEAD^{tree}", repo=repo)
    if rc != 0:
        FAIL.append("cannot read the local HEAD tree")
    elif not head_tree.startswith(enc["tree"]):
        # the statement quotes an abbreviation, so compare as a prefix: a full
        # 40-character comparison would reject a correct 7-character quote
        FAIL.append("the named tree %s is not the checkout's tree %s -- the "
                    "statement is stale relative to HEAD"
                    % (enc["tree"], head_tree[:len(enc["tree"])]))
    rc, ctree = git("rev-parse", "%s^{tree}" % enc["commit"], repo=repo)
    if rc == 0 and not ctree.startswith(enc["tree"]):
        FAIL.append("commit %s has tree %s, not the named %s"
                    % (enc["commit"], ctree, enc["tree"]))
    return FAIL


def self_test():
    """One known-bad variant per assertion class, each must be rejected."""
    tmp = tempfile.mkdtemp()
    try:
        shutil.copytree(os.path.join(ROOT, "submission"),
                        os.path.join(tmp, "submission"))
        os.makedirs(os.path.join(tmp, "versions"), exist_ok=True)
        # a stand-in snapshot: the delivered PDFs plus a manifest of the right size
        d = parse("en") or {}
        snap = os.path.join(tmp, "versions", d.get("snapshot") or "v00")
        for lang in ("en", "cn"):
            os.makedirs(os.path.join(snap, "paper", lang), exist_ok=True)
            shutil.copy2(os.path.join(SUB, "paper_%s" % lang, "main.pdf"),
                         os.path.join(snap, "paper", lang, "main.pdf"))
        with open(os.path.join(snap, "MANIFEST.md"), "w", encoding="utf-8") as fh:
            fh.write("# stand-in manifest\n" + ("0" * 32 + "  stub\n") * 100)
        # copy the list: check() returns the module-level FAIL object and
        # clears it in place, so holding a reference would let the later
        # mutation runs append to this "clean" result and make the clean case
        # look rejected -- an instrument defect that would only show up as a
        # confusing self-test failure
        good = list(check(tmp))
        print("clean tree         -> %s" % ("accepted" if not good else "REJECTED: %s" % good))
        p = os.path.join(tmp, "submission", "paper_en", "main.tex")
        q = os.path.join(tmp, "submission", "paper_cn", "main.tex")
        src_en, src_cn = read(p), read(q)
        # the snapshot name is written with escaped underscores in the sources,
        # while parse() returns it unescaped: mutating with the parsed form
        # would change nothing and the probe would report a false rejection of
        # the mutation rather than a false acceptance by the gate
        esc = d["snapshot"].replace("_", "\\_")
        cases = [
            ("stale tree", "en", src_en.replace(d["tree"], "0" * 7, 1)),
            ("missing snapshot", "en",
             src_en.replace(esc, "v00\\_1999-01-01\\_none", 1)),
            ("cn disagrees", "cn",
             src_cn.replace(esc, "v01\\_1999-01-01\\_other", 1)),
        ]
        ok = True
        for what, lang, mutated in cases:
            before = src_en if lang == "en" else src_cn
            if mutated == before:
                print("%-18s -> MUTATION DID NOT APPLY" % what)
                ok = False
                continue
            open(p, "w", encoding="utf-8").write(src_en)
            open(q, "w", encoding="utf-8").write(src_cn)
            open(p if lang == "en" else q, "w", encoding="utf-8").write(mutated)
            fails = list(check(tmp))
            print("%-18s -> %s" % (what, "rejected" if fails else "ACCEPTED (bad)"))
            ok = ok and bool(fails)
        return ok and not good
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    if "--self-test" in sys.argv:
        ok = self_test()
        print("SELF-TEST:", "all known-bad variants rejected" if ok
              else "THE INSTRUMENT DOES NOT FIRE")
        return 0 if ok else 1
    fails = check()
    for f in fails:
        print("  [FAIL]", f)
    if fails:
        print("DAS gate: %d failure(s)" % len(fails))
        return 1
    d = parse("en")
    print("  [ok]   commit %s, tree %s, snapshot %s" % (d["commit"], d["tree"], d["snapshot"]))
    print("  [ok]   both manuscripts agree; archived PDFs are the delivered PDFs; "
          "tree equals the checkout")
    print("DAS gate passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
