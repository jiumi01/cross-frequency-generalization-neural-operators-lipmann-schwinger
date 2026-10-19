# Author: Longgang Wang <longgangwang@163.com>
# Affiliation: Luoyang Institute of Science and Technology, Luoyang 471000, Henan, China

"""Emit an MDPI-style reference list from refs.bib.

The official MDPI bibliography style (`mdpi.bst`) ships with the MDPI LaTeX
class, which cannot be redistributed inside this project.  Rather than settle
for `IEEEtran.bst` -- whose entry shape is visibly different ("Z. Li, N.
Kovachki, ... in Proc. ICLR, 2021") -- the list is generated here so that the
rendered entries match MDPI's convention:

    Author, A.B.; Author, C.D. Title of the article in sentence case. *Journal
    Abbrev.* **Year**, *Volume*, Pages. https://doi.org/...

and MDPI numbers the entries in order of first citation, not alphabetically.
`refs.bib` stays the single source of truth: this script only formats it, so a
new reference is added in one place.

Writes paper/en/gen_bib.tex and paper/cn/gen_bib.tex (identical content; the
section title is `\\refname`, which each manuscript sets for itself).
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

# Journal abbreviations used by MDPI (ISO-4 style).  Anything not listed keeps
# its full name, which is what MDPI does for journals it does not abbreviate.
JABBR = {
    "nature machine intelligence": "Nat. Mach. Intell.",
    "journal of machine learning research": "J. Mach. Learn. Res.",
    "ieee transactions on neural networks": "IEEE Trans. Neural Netw.",
    "journal of computational physics": "J. Comput. Phys.",
    "geophysical journal international": "Geophys. J. Int.",
    "applied sciences": "Appl. Sci.",
    "siam journal on scientific and statistical computing":
        "SIAM J. Sci. Stat. Comput.",
    "archives of computational methods in engineering":
        "Arch. Comput. Methods Eng.",
    "numerische mathematik": "Numer. Math.",
    "communications in computational physics": "Commun. Comput. Phys.",
    "computer methods in applied mechanics and engineering":
        "Comput. Methods Appl. Mech. Eng.",
    "acm/ims journal of data science": "ACM/IMS J. Data Sci.",
    "jasa express letters": "JASA Express Lett.",
    "scientific reports": "Sci. Rep.",
    "ieee transactions on geoscience and remote sensing":
        "IEEE Trans. Geosci. Remote Sens.",
    "ieee transactions on antennas and propagation":
        "IEEE Trans. Antennas Propag.",
    "ieee antennas and wireless propagation letters":
        "IEEE Antennas Wirel. Propag. Lett.",
    "ieee geoscience and remote sensing letters":
        "IEEE Geosci. Remote Sens. Lett.",
    "nature reviews physics": "Nat. Rev. Phys.",
    "the leading edge": "Lead. Edge",
    "电波科学学报": "Chin. J. Radio Sci.",
}

# Proper nouns and acronyms that must survive sentence-casing of a title.
KEEP = {
    "fourier", "helmholtz", "lippmann", "schwinger", "born", "green", "neumann",
    "poisson", "banach", "sobolev", "jacobi", "laplace", "laplacian", "krylov",
    "wasserstein", "fno", "mscalefno", "deeponet", "deepgreen", "deepnis",
    "pinn", "pinns", "pde", "pdes", "fft", "gmres", "bi-cgstab", "bi-cg",
    "u-net", "gaussian", "adam", "gelu", "relu", "cnn", "mse", "ssim",
    "matlab", "python", "pytorch", "born", "newton", "euler",
}


def parse_bib(text):
    """Minimal .bib parser; the file is small and uniformly formatted."""
    entries = []
    for m in re.finditer(r"@(\w+)\s*\{\s*([^,]+),", text):
        etype, key = m.group(1).lower(), m.group(2).strip()
        # value body runs to the matching closing brace of the entry
        i, depth = m.end(), 1
        while i < len(text) and depth:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        body = text[m.end():i - 1]
        fields = {}
        for fm in re.finditer(r"(\w+)\s*=\s*(\{(?:[^{}]|\{[^{}]*\})*\}|"
                              r'"[^"]*"|[^,\n]+)', body, re.S):
            name = fm.group(1).lower()
            val = fm.group(2).strip()
            if val.startswith("{"):
                val = val[1:-1] if val.endswith("}") else val[1:]
            elif val.startswith('"') and val.endswith('"'):
                val = val[1:-1]
            val = re.sub(r"\s+", " ", val).strip().rstrip(",").strip()
            fields[name] = val
        entries.append(dict(type=etype, key=key, **fields))
    return entries


def initials(given):
    """'Zongyi' -> 'Z.'; 'Henk A.' -> 'H.A.'; 'José Antonio' -> 'J.A.'"""
    out = []
    for tok in re.split(r"[\s\-]+", given):
        tok = tok.strip(".")
        if not tok:
            continue
        m = re.match(r"\{?\\?[a-zA-Z]?", tok)
        if m and m.group(0):
            out.append(m.group(0).upper() + ".")
    return "".join(out)


def fmt_authors(s):
    """'Li, Zongyi and Kovachki, Nikola' -> 'Li, Z.; Kovachki, N.'

    A name in braces is an organisation ('{Devito Codes}') and is emitted
    verbatim: inverting it would produce 'Codes, D.'.
    """
    names = re.split(r"\s+and\s+", s)
    out = []
    for n in names:
        n = n.strip().rstrip(".")
        if not n:
            continue
        if n.startswith("{") and n.endswith("}"):
            out.append(n[1:-1])
            continue
        if "," in n:
            sur, giv = n.split(",", 1)
        else:
            parts = n.split()
            sur, giv = parts[-1], " ".join(parts[:-1])
        ini = initials(giv)
        out.append("%s, %s" % (sur.strip(), ini) if ini else sur.strip())
    return "; ".join(out)


def sentence(t):
    """Sentence-case a title, protecting brace groups and known proper nouns.

    A brace group is opaque, so a title that *starts* with one (``{DeepNIS}: ...``)
    still has to capitalise the next real word -- hence `first` is cleared only
    when a word is actually emitted.  Hyphenated compounds are cased part by
    part, so ``Auto-Regressive`` becomes ``Auto-regressive`` rather than
    ``auto-Regressive``.
    """
    def lower_word(w):
        m = re.match(r"^([\(\[]*)(.*?)([\)\]\.,;:]*)$", w)
        pre, core, post = m.groups()
        if not core:
            return w
        low = core.lower()
        if (low in KEEP or low.rstrip("s") in KEEP
                or any(c.isdigit() for c in core) or core.isupper()):
            return w
        return pre + core[0].lower() + core[1:] + post

    parts = re.split(r"(\{[^{}]*\})", t)
    res, first = [], True
    for p in parts:
        if p.startswith("{"):
            res.append(p)
            continue
        for w in re.split(r"(\s+)", p):
            if not w.strip():
                res.append(w)
                continue
            if first:
                res.append(w[0].upper() + w[1:])
                first = False
            else:
                res.append("-".join(lower_word(x) for x in w.split("-")))
    return "".join(res)


def journal(j):
    return JABBR.get(j.strip().lower(), j.strip())


def url(s):
    """Escape a URL for text mode.

    A repository path can contain an underscore, and a bare ``_`` in text mode
    is a math subscript: ``https://github.com/shenyaojin/reverse_time_migration``
    reached a real build as a hard "Missing $ inserted", which is why the
    escaping lives here rather than in the .bib file -- the .bib stays valid
    BibTeX and the LaTeX escaping is the renderer's job.
    """
    return s.replace("\\", "\\textbackslash{}").replace("_", "\\_") \
            .replace("&", "\\&").replace("%", "\\%").replace("#", "\\#")


def render(e):
    au = fmt_authors(e.get("author", ""))
    # the author list already ends in a full stop ("..., Anandkumar, A."), so
    # the separator is a space rather than ". " -- otherwise every entry reads
    # "Anandkumar, A.."
    if au and not au.endswith("."):
        au += "."
    ti = sentence(e.get("title", "")).rstrip(".")
    y = e.get("year", "")
    doi = e.get("doi")
    doi_s = " https://doi.org/%s" % doi if doi else ""
    t = e["type"]

    if t == "article":
        j = e.get("journal", "")
        if j.lower().startswith("arxiv"):
            ep = e.get("eprint") or j.split()[-1]
            return "%s %s. \\emph{Preprint} \\textbf{%s}, %s.%s" % (au, ti, y, ep, doi_s)
        vol, pages = e.get("volume"), e.get("pages")
        s = "%s %s. \\emph{%s} \\textbf{%s}" % (au, ti, journal(j), y)
        if vol and pages:
            s += ", \\emph{%s}, %s." % (vol, pages)
        elif vol:
            s += ", \\emph{%s}." % vol
        elif pages:
            s += ", %s." % pages
        else:
            s += "."
        return s + doi_s

    if t in ("inproceedings", "conference"):
        bt = e.get("booktitle", "")
        s = "%s %s. In \\emph{%s}" % (au, ti, bt)
        vol, pages = e.get("volume"), e.get("pages")
        if vol:
            s += ", vol. %s" % vol
        if pages:
            s += ", pp. %s" % pages
        return s + "; %s.%s" % (y, doi_s)

    if t == "book":
        ed = e.get("edition")
        ed_s = ", %s ed." % ed if ed else "."
        return "%s \\emph{%s}%s %s, %s.%s" % (au, ti, ed_s,
                                             e.get("publisher", ""), y, doi_s)

    if t == "misc":
        # an online resource or data set: MDPI gives the hosting venue, the
        # URL and an access date rather than a journal.
        hp = e.get("howpublished", "")
        s = "%s %s." % (au, ti)
        if hp:
            s += " %s," % hp
        s += " %s." % y
        link = e.get("url", "")
        note = e.get("note", "")
        if link:
            s += " Available online: %s" % url(link)
        if note:
            s += " (%s)." % note
        return s + doi_s

    return "%s %s. %s.%s" % (au, ti, y, doi_s)


def main():
    for lang in ("en", "cn"):
        folder = os.path.join(C.ROOT, "paper", lang)
        bib = parse_bib(open(os.path.join(folder, "refs.bib"), encoding="utf-8").read())
        by_key = {e["key"]: e for e in bib}

        tex = open(os.path.join(folder, "main.tex"), encoding="utf-8").read()
        tex = "\n".join(re.sub(r"(?<!\\)%.*$", "", l) for l in tex.splitlines())
        order, seen = [], set()
        for m in re.finditer(r"\\cite[a-z]*\*?(?:\[[^\]]*\])*\{([^}]*)\}", tex):
            for k in (x.strip() for x in m.group(1).split(",")):
                if k and k not in seen:
                    seen.add(k)
                    order.append(k)

        missing = [k for k in order if k not in by_key]
        if missing:
            print("  !! cited but not in refs.bib:", missing)

        lines = ["% Auto-generated by code/make_bib_tex.py -- do not edit.",
                 "\\begin{thebibliography}{99}", ""]
        for k in order:
            lines.append("\\bibitem{%s} %s" % (k, render(by_key[k])))
            lines.append("")
        lines.append("\\end{thebibliography}")
        out = os.path.join(folder, "gen_bib.tex")
        with open(out, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
        print("  [%s] %d cited entries -> %s" % (lang, len(order), out))
        unused = sorted(set(by_key) - seen)
        if unused:
            print("  [%s] %d bib entries never cited: %s" % (lang, len(unused), unused))


if __name__ == "__main__":
    main()
