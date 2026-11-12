# Author: Longgang Wang <longgangwang@163.com>
# Affiliation: Luoyang Institute of Science and Technology, Luoyang 471000, Henan, China

"""Refresh the status record of every preprint this manuscript cites.

Why this is a separate step
---------------------------
A build gate must not depend on the network: a compile that fails because
arXiv is slow is a false alarm, and a gate that is switched off is worse than
none.  So the network question -- *is this preprint still citable?* -- is asked
here, on demand, and answered into a dated record (`code/ref_status.json`).
The audit gate then asserts against the record, offline.

What the record catches
-----------------------
Withdrawal.  On 2026-11-03 the reference list carried
`arXiv:2312.15575` for the Neural Born Series Operator; the authors had
withdrawn it on 2026-08-03, leaving no PDF, and its replacement
(arXiv:2508.12226) is a different paper.  Nothing in the .bib, in the renderer
or in any gate disclosed that, because none of them ever re-asked.  A
withdrawn preprint is not a weak reference -- it is an unrunnable one: the
reader who clicks it is told the work "no longer reflects the authors' final
results".

The title match is checked as well, because an arXiv number that resolves to
a different paper is the same failure with a quieter signature.

Usage
-----
    python code/check_arxiv_status.py            # refresh the record
    python code/check_arxiv_status.py --dry-run  # print, write nothing
"""
import argparse
import datetime
import json
import os
import re
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from make_bib_tex import parse_bib

RECORD = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "ref_status.json")
API = "https://export.arxiv.org/api/query?id_list=%s&max_results=%d"


def norm(s):
    """Compare titles without caring about case, punctuation or braces."""
    s = re.sub(r"\\[a-z]+", " ", s.lower())
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(s.split())


def fetch(ids):
    """Return {arxiv_id: {title, comment, updated}} from the arXiv API."""
    out = {}
    for i in range(0, len(ids), 25):
        chunk = ids[i:i + 25]
        url = API % (",".join(chunk), len(chunk))
        with urllib.request.urlopen(url, timeout=60) as fh:
            xml = fh.read().decode("utf-8", "replace")
        for ent in re.findall(r"<entry>(.*?)</entry>", xml, re.S):
            m = re.search(r"<id>https?://arxiv\.org/abs/([^<v]+)", ent)
            if not m:
                continue
            aid = m.group(1).strip()
            title = re.search(r"<title>(.*?)</title>", ent, re.S)
            comment = re.search(r"<arxiv:comment>(.*?)</arxiv:comment>",
                                ent, re.S)
            updated = re.search(r"<updated>(.*?)</updated>", ent)
            out[aid] = dict(
                title=" ".join(title.group(1).split()) if title else "",
                comment=" ".join(comment.group(1).split()) if comment else "",
                updated=updated.group(1)[:10] if updated else "",
            )
    return out


def classify(rec):
    """'withdrawn' dominates; a mention of a replacement id is 'superseded'."""
    c = rec.get("comment", "").lower()
    if "withdrawn" in c or "withdrew" in c:
        return "withdrawn"
    if re.search(r"arxiv:\s*\d{4}\.\d{4,5}", c) and "replace" in c:
        return "superseded"
    return "live"


def arxiv_id(e):
    """The arXiv number of an entry, wherever the .bib happens to store it.

    Two spellings are in use and both must be covered: an `eprint` field
    (conference entries with an arXiv locator) and a `journal` field written
    as "arXiv preprint arXiv:XXXX.XXXXX" (preprint-only entries).  Reading
    only `eprint` would have skipped every 2025-2026 preprint in this list,
    which is exactly the set a withdrawal would hit first.
    """
    if e.get("eprint"):
        return e["eprint"].strip()
    m = re.search(r"arXiv:\s*(\d{4}\.\d{4,5})", e.get("journal", ""))
    return m.group(1) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    bib = parse_bib(open(os.path.join(C.ROOT, "paper", "en", "refs.bib"),
                         encoding="utf-8").read())
    preprints = [(e["key"], arxiv_id(e)) for e in bib if arxiv_id(e)]
    if not preprints:
        print("no preprint entries in refs.bib")
        return 0

    recs = fetch([i for _, i in preprints])
    today = datetime.date.today().isoformat()
    rows, bad = [], []
    for key, aid in preprints:
        r = recs.get(aid)
        title = next(e["title"] for e in bib if e["key"] == key)
        if r is None:
            row = dict(key=key, arxiv=aid, status="unresolved", checked=today,
                       title_match=False, arxiv_title="", comment="")
        else:
            na, nb = norm(r["title"]), norm(title)
            row = dict(key=key, arxiv=aid, status=classify(r), checked=today,
                       # A published version extends the arXiv title (JMLR
                       # reprints add a subtitle), so a prefix relation counts
                       # as the same paper; an unrelated title does not.
                       title_match=na == nb or na.startswith(nb)
                       or nb.startswith(na),
                       arxiv_title=r["title"], comment=r["comment"])
        rows.append(row)
        if row["status"] != "live" or not row["title_match"]:
            bad.append(row)

    width = max(len(r["key"]) for r in rows) + 2
    for r in rows:
        print("  %-*s %-12s %-11s %s"
              % (width, r["key"], r["arxiv"], r["status"],
                 "title ok" if r["title_match"] else "TITLE MISMATCH"))
    print("\n  %d preprints, %d needing attention" % (len(rows), len(bad)))
    for r in bad:
        if r["status"] == "withdrawn":
            print("  [WITHDRAWN] %s  %s" % (r["key"], r["comment"][:110]))
        elif r["status"] == "unresolved":
            print("  [MISSING]   %s  arXiv:%s does not resolve" %
                  (r["key"], r["arxiv"]))
        elif not r["title_match"]:
            print("  [TITLE]     %s  arXiv:%s is %r" %
                  (r["key"], r["arxiv"], r["arxiv_title"][:80]))

    if a.dry_run:
        print("\n  dry run: %s not written" % RECORD)
        return 1 if bad else 0

    json.dump(dict(checked=today, entries=rows), open(RECORD, "w",
              encoding="utf-8"), indent=2)
    print("\n  wrote %s" % RECORD)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
