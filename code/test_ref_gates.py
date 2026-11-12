# Author: Longgang Wang <longgangwang@163.com>
# Affiliation: Luoyang Institute of Science and Technology, Luoyang 471000, Henan, China

"""Falsification probes for the two reference gates in audit_paper.py.

A gate that has never failed is indistinguishable from a gate that cannot
fail.  Each case below takes the real, passing artefacts, breaks exactly one
thing, and asserts that the gate notices -- and the first case asserts that it
stays quiet when nothing is broken, because a gate that always shouts gets
switched off.

    python code/test_ref_gates.py      # exits non-zero if any probe is wrong
"""
import os
import shutil
import sys
import tempfile
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import audit_paper as A

SRC = os.path.join(C.ROOT, "paper", "en")


def stage():
    """A throwaway folder holding the real refs.bib and gen_bib.tex."""
    d = tempfile.mkdtemp(prefix="refgate_")
    for f in ("refs.bib", "gen_bib.tex"):
        shutil.copy2(os.path.join(SRC, f), os.path.join(d, f))
    return d


def fails(fn, folder):
    A.FAIL.clear()
    fn(folder)
    return list(A.FAIL)


def rewrite(path, old, new):
    txt = open(path, encoding="utf-8").read()
    assert old in txt, "probe target %r not in %s" % (old, path)
    open(path, "w", encoding="utf-8").write(txt.replace(old, new, 1))


def main():
    bad = 0

    # 1. the clean artefacts must pass, or every later case proves nothing
    d = stage()
    got = fails(A.bib_publication_kind, d)
    print("  [%s] clean list is accepted%s"
          % ("ok" if not got else "FAIL", "" if not got else ": " + str(got)))
    bad += bool(got)
    shutil.rmtree(d, ignore_errors=True)

    # 2. a preprint that stops saying it is a preprint
    d = stage()
    rewrite(os.path.join(d, "gen_bib.tex"), "\\emph{Preprint} \\textbf{2026}",
            "\\textbf{2026}")
    got = fails(A.bib_publication_kind, d)
    print("  [%s] unlabelled preprint is caught (%d finding(s))"
          % ("ok" if got else "FAIL", len(got)))
    bad += not got
    shutil.rmtree(d, ignore_errors=True)

    # 3. a .bib note that never reaches the page
    d = stage()
    rewrite(os.path.join(d, "gen_bib.tex"), " Oral presentation; no proceedings "
            "volume.", "")
    got = fails(A.bib_publication_kind, d)
    print("  [%s] dropped note is caught (%d finding(s))"
          % ("ok" if got else "FAIL", len(got)))
    bad += not got
    shutil.rmtree(d, ignore_errors=True)

    # 4. a preprint that has been withdrawn since the record was written
    d = stage()
    rec = json.load(open(A.RECORD, encoding="utf-8"))
    for r in rec["entries"]:
        if r["key"] == "apex2026":
            r["status"] = "withdrawn"
    tmp = os.path.join(d, "ref_status.json")
    json.dump(rec, open(tmp, "w", encoding="utf-8"))
    keep, A.RECORD = A.RECORD, tmp
    got = fails(A.bib_status_record, d)
    A.RECORD = keep
    print("  [%s] withdrawn preprint is caught (%d finding(s))"
          % ("ok" if got else "FAIL", len(got)))
    bad += not got
    shutil.rmtree(d, ignore_errors=True)

    # 5. and the clean record must still pass, so case 4 was about the status
    got = fails(A.bib_status_record, SRC)
    print("  [%s] live record is accepted%s"
          % ("ok" if not got else "FAIL", "" if not got else ": " + str(got)))
    bad += bool(got)

    print("\n  %d probe(s) wrong" % bad)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
