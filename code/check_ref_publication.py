# Author: Longgang Wang <longgangwang@163.com>
# Affiliation: Luoyang Institute of Science and Technology, Luoyang 471000, Henan, China

"""Verify that every reference is cited in a form that can actually be checked.

The three rules this encodes were stated by the author on 2026-11-03:

  1. a work that has appeared must be cited in its published version, with the
     journal, volume, pages and DOI of that version -- not as the preprint it
     once was;
  2. a withdrawn work is not cited at all;
  3. a work that cannot be located exactly -- no DOI, no arXiv number, no
     volume, no page, no persistent URL, or a meeting abstract whose only
     record is a session listing -- is not cited at all.

Rule 1 is the one a hand-maintained .bib always drifts on: an entry is written
when the paper is a preprint and never revisited, so `li2022pfno` sat in this
list for two years as arXiv:2209.12340 after it had become IEEE TGRS 61:1-18,
and `chen2024nsno` as arXiv:2401.13494 after it had become J. Syst. Sci.
Complex. 37:413-440.  Both were found on 2026-11-03, not by reading the .bib
but by asking Crossref.

What is asked here
------------------
* every entry that carries a DOI: does the DOI resolve, and does the record it
  resolves to have this title?
* every entry that does not: does a published version exist under this title?
* every entry that has neither a DOI nor an arXiv number nor a URL: it has no
  locator at all, and rule 3 removes it.

The network step writes a dated record (`code/ref_publication.json`); the audit
gate asserts against that record offline, for the same reason the arXiv record
is separate: a compile must not fail because Crossref is slow.

Usage
-----
    python code/check_ref_publication.py            # refresh the record
    python code/check_ref_publication.py --dry-run  # print, write nothing
"""
import argparse
import datetime
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from make_bib_tex import parse_bib

RECORD = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "ref_publication.json")
WORKS = "https://api.crossref.org/works/%s"
SEARCH = ("https://api.crossref.org/works?query.bibliographic=%s&rows=5"
          "&select=title,container-title,volume,page,article-number,issued,"
          "DOI,type")

PUBLISHED_TYPES = ("journal-article", "proceedings-article", "book-chapter",
                   "book", "monograph", "reference-entry")


def norm(s):
    """Compare titles without caring about case, punctuation or braces."""
    s = re.sub(r"\\[a-z]+", " ", s.lower())
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(s.split())


def same_title(a, b):
    """Equality, or a prefix relation (published versions add subtitles)."""
    na, nb = norm(a), norm(b)
    if not na or not nb:
        return False
    return na == nb or na.startswith(nb) or nb.startswith(na)


def get(url, tries=3):
    for i in range(tries):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "ref-audit/1.0 (mailto:%s)"
                              % "longgangwang@163.com"})
            with urllib.request.urlopen(req, timeout=45) as fh:
                return json.loads(fh.read().decode("utf-8", "replace"))
        except Exception as exc:                      # noqa: BLE001
            if i == tries - 1:
                return dict(error=str(exc)[:160])
            time.sleep(2 + 2 * i)
    return dict(error="unreachable")


def kind_of(e):
    """The bibliographic category the entry claims to be."""
    t = e["type"]
    j = (e.get("journal") or "").lower()
    bt = (e.get("booktitle") or "").lower()
    if "arxiv" in j:
        return "preprint"
    if t in ("inproceedings", "conference"):
        if "abstract" in bt and "meeting" in bt:
            return "meeting-abstract"
        return "proceedings"
    if t in ("misc",):
        return "repository" if e.get("url") else "preprint"
    if t in ("book",):
        return "book"
    return "journal"


def crossref_doi(doi):
    """Resolve one DOI through Crossref; return the record fields we compare."""
    r = get(WORKS % urllib.parse.quote(doi, safe="/"))
    if "message" not in r:
        return dict(resolved=False, error=r.get("error", "no message"))
    m = r["message"]
    issued = (m.get("issued", {}).get("date-parts") or [[None]])[0]
    return dict(
        resolved=True,
        title=(m.get("title") or [""])[0],
        container=(m.get("container-title") or [""])[0],
        volume=m.get("volume", ""),
        page=m.get("page", "") or m.get("article-number", ""),
        year=issued[0] if issued else None,
        type=m.get("type", ""),
    )


def doi_handle(doi):
    """Does the DOI exist in *any* registration agency?

    Crossref answers for most of this list but not for all of it: the arXiv
    DOIs (10.48550) are DataCite, and the one Chinese journal in the list
    (10.13443) is registered with CNKI.  Asking Crossref alone would have
    called both of them broken DOIs, which is the instrument's error, not the
    bibliography's.
    """
    url = "https://doi.org/api/handles/" + urllib.parse.quote(doi, safe="/")
    r = get(url)
    if r.get("responseCode") != 1:
        return None
    for v in r.get("values", []):
        if v.get("type") == "URL":
            return v.get("data", {}).get("value", "")
    return ""


def crossref_title(title):
    """Ask Crossref whether a published version exists under this title."""
    r = get(SEARCH % urllib.parse.quote(title))
    if "message" not in r:
        return None
    for it in r["message"].get("items", []):
        t = (it.get("title") or [""])[0]
        if not same_title(t, title):
            continue
        if it.get("type") not in PUBLISHED_TYPES:
            continue
        issued = (it.get("issued", {}).get("date-parts") or [[None]])[0]
        return dict(doi=it.get("DOI", ""),
                    container=(it.get("container-title") or [""])[0],
                    volume=it.get("volume", ""),
                    page=it.get("page", "") or it.get("article-number", ""),
                    year=issued[0] if issued else None,
                    type=it.get("type", ""),
                    title=t)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    path = os.path.join(C.ROOT, "submission", "paper_en", "refs.bib")
    bib = parse_bib(open(path, encoding="utf-8").read())
    today = datetime.date.today().isoformat()

    rows, flagged = [], []
    for e in bib:
        key, kind = e["key"], kind_of(e)
        # The .bib stores LaTeX-escaped surface forms (the IEEEtran build needs
        # \_ for _ in a typeset DOI); resolving needs the semantic string.
        doi = e.get("doi", "").replace("\\_", "_")
        row = dict(key=key, kind=kind, doi=doi,
                   arxiv="", url=e.get("url", ""), checked=today,
                   verdict="ok", detail="")

        if doi:
            rec = crossref_doi(doi)
            row["doi_record"] = rec
            if not rec["resolved"]:
                # Not in Crossref is not the same as not registered: fall back
                # to the handle system before calling the DOI broken.
                landing = doi_handle(doi)
                if landing is None:
                    row["verdict"] = "doi-unresolved"
                    row["detail"] = rec.get("error", "")[:70]
                else:
                    row["verdict"] = "ok"
                    row["detail"] = "registered outside Crossref: %s" % \
                        landing.split("/")[2] if landing else "registered"
            elif not same_title(rec["title"], e["title"]):
                row["verdict"] = "doi-title-mismatch"
                row["detail"] = rec["title"][:90]
            elif rec["year"] and e.get("year") and \
                    abs(int(re.sub(r"\D", "", e["year"]) or 0) - rec["year"]) > 1:
                row["verdict"] = "doi-year-mismatch"
                row["detail"] = "bib %s vs crossref %s" % (e["year"], rec["year"])
            else:
                row["verdict"] = "ok"
                row["detail"] = "%s %s:%s (%s)" % (
                    rec["container"][:34], rec["volume"], rec["page"], rec["year"])
        else:
            m = re.search(r"arXiv:\s*(\d{4}\.\d{4,5})", e.get("journal", ""))
            if m:
                row["arxiv"] = m.group(1)
            if kind == "meeting-abstract":
                row["verdict"] = "no-formal-record"
                row["detail"] = (e.get("booktitle", "")[:50] +
                                 "; abstract only, no DOI")
            elif kind == "repository":
                row["verdict"] = "ok" if e.get("url") else "no-locator"
                row["detail"] = e.get("url", "")[:60]
            elif kind == "journal" and e.get("volume") and e.get("pages"):
                # JMLR and a few other journals issue no DOIs at all; volume and
                # page range still locate the article exactly, so rule 3 does
                # not remove it.
                row["verdict"] = "ok"
                row["detail"] = "no DOI (journal issues none); %s %s:%s" % (
                    e.get("journal", "")[:30], e["volume"], e["pages"])
            else:
                # no DOI: is there a published version now?
                hit = crossref_title(e["title"])
                if hit:
                    row["published_version"] = hit
                    row["verdict"] = "published-version-exists"
                    row["detail"] = "%s %s:%s (%s) doi:%s" % (
                        hit["container"][:30], hit["volume"], hit["page"],
                        hit["year"], hit["doi"])
                elif row["arxiv"] or e.get("eprint"):
                    row["arxiv"] = row["arxiv"] or e["eprint"]
                    row["verdict"] = "ok"
                    row["detail"] = "preprint only, arXiv:%s" % row["arxiv"]
                else:
                    row["verdict"] = "no-locator"
        rows.append(row)
        if row["verdict"] != "ok":
            flagged.append(row)

    width = max(len(r["key"]) for r in rows) + 2
    for r in rows:
        print("  %-*s %-11s %-24s %s"
              % (width, r["key"], r["kind"], r["verdict"], r["detail"]))
    print("\n  %d entries, %d flagged" % (len(rows), len(flagged)))
    for r in flagged:
        print("  [%s] %s  %s" % (r["verdict"].upper(), r["key"], r["detail"]))

    if a.dry_run:
        print("\n  dry run: %s not written" % RECORD)
        return 1 if flagged else 0

    json.dump(dict(checked=today, entries=rows),
              open(RECORD, "w", encoding="utf-8"), indent=2,
              ensure_ascii=False)
    print("\n  wrote %s" % RECORD)
    return 1 if flagged else 0


if __name__ == "__main__":
    sys.exit(main())
