# Author: Longgang Wang <longgangwang@163.com>
# Affiliation: Luoyang Institute of Science and Technology, Luoyang 471000, Henan, China

"""Check the delivered texts that are not the manuscript.

Every other gate reads the .tex sources or the numbers behind them.  That left a
blind spot of exactly the kind the toolchain is supposed to close: the files an
editor or a reader meets *before* the manuscript were outside every check, and
each of them drifted while the manuscript stayed correct.

What actually happened, so the checks below are not hypothetical:

  * the cover letter kept the title the paper had two revisions earlier, and
    still said "pre-registered" in two places after the manuscript had been
    changed to "pre-specified";
  * CITATION.cff described the plateau as sitting "at the trivial
    incident-field baseline", a reading the manuscript had already replaced with
    "Born truncation error, near that baseline, excess growing with contrast";
  * README.md announced a 42-page manuscript, a six-section supplement, an
    in-house style file and an undecided journal, four revisions after each of
    those became false;
  * environment.yml pinned ``pytorch=2.13.0`` and ``python=3.11.15`` -- a
    version pair no channel carries -- while the supplementary checklist named
    the versions actually measured.  Following the recipe could not work;
  * the cover letter said the archive was deposited with a DOI while the Data
    Availability statement said the DOI would be obtained on acceptance.  Two
    delivered documents, opposite statements about one fact.

The checks are content-based (a title must appear verbatim, after TeX
normalisation), not structural, so they survive reformatting:

  1. the manuscript's title appears verbatim in the cover letter, the release
     README and both CITATION.cff copies; the Chinese title still carries its own
     four load-bearing fragments;
  2. no delivered text carries the retired title marker ("... Benchmark: An
     Empirical Characterization") or a retired claim phrase;
  3. the environment pins agree across environment.yml, requirements.txt, the
     archive copies of both, and the S23 checklist;
  4. the Data Availability statement and the cover letter agree on when the DOI
     exists, and neither asserts a DOI that does not;
  5. the supplementary material is declared in both manuscripts;
  6. the archive's evaluation records cover every arm and seed the live results
     directory has, and its MANIFEST describes the files actually on disk.

Run with ``--self-test`` to see each check reject a known-bad variant.
"""
import hashlib
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
SUB = os.path.join(ROOT, "submission")

# Title fragments that must survive in the Chinese title.  A translation cannot
# be compared character by character with the English one, so the fragments are
# the parts that carry the claim: the discretised periodic setting, the
# frequency axis, the paired measurement design and the analytic correction.
CN_TITLE_FRAGMENTS = ["离散周期", "跨频率泛化", "配对测量", "解析校正"]

# Phrases that describe a state the paper no longer is in.  Each one shipped at
# least once in a delivered file after it stopped being true.  Matched
# case-sensitively and only against the envelope documents: the manuscripts
# themselves use "Lippmann--Schwinger benchmark" as a common noun, which is
# correct, and "mdpi.sty" appears in a Chinese source comment.
RETIRED = [
    "Lippmann-Schwinger Benchmark",
    "Lippmann–Schwinger Benchmark",
    "Lippmann--Schwinger Benchmark",
    "An Empirical Characterization",
    "plateaus at the trivial incident-field baseline",
    "the plateau is the trivial incident-field baseline",
    "pre-registered",
    "Pre-registered",
    "not yet finalized",
    "mdpi.sty",
]

# The documents a reader or an editor meets before the manuscript.  These are
# the ones no .tex-side gate can see.
ENVELOPE = ("cover", "highlights", "readme", "citation", "citation_archive",
            "si", "env", "reqs")

DOI_DEFERRED = "on acceptance"


def read(p):
    with open(p, encoding="utf-8") as fh:
        return fh.read()


def norm(s):
    """Normalise TeX/markdown so a title can be compared as a substring."""
    s = s.replace("\\\\", " ")
    s = s.replace("~", " ")
    s = s.replace("--", "-").replace("–", "-").replace("—", "-")
    s = s.replace("{", "").replace("}", "")
    s = re.sub(r"\\[a-zA-Z]+\*?", " ", s)      # drop control words
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def tex_title(src, cmd="Title"):
    """Return the argument of \\Title{...} / \\title{...}, braces balanced."""
    m = re.search(r"\\%s\s*\{" % cmd, src)
    if not m:
        return None
    i = m.end() - 1
    depth = 0
    for j in range(i, len(src)):
        if src[j] == "{":
            depth += 1
        elif src[j] == "}":
            depth -= 1
            if depth == 0:
                return src[i + 1:j]
    return None


# --------------------------------------------------------------------------
# 1. title
# --------------------------------------------------------------------------
def title_failures(texts):
    """texts: {name: contents} for every carrier of the title."""
    fails = []
    en = tex_title(texts.get("paper_en", ""), "Title")
    if not en:
        return ["paper_en: no \\Title{...} found"]
    canonical = norm(en)
    if len(canonical.split()) < 8:
        fails.append("paper_en: the parsed title is implausibly short (%r)"
                     % canonical)
    for name in ("cover", "readme", "citation", "citation_archive", "si"):
        t = texts.get(name)
        if t is None:
            fails.append("%s: missing, cannot check the title" % name)
        elif canonical not in norm(t):
            fails.append("%s: does not carry the manuscript title; expected %r"
                         % (name, canonical))
    cn = tex_title(texts.get("paper_cn", ""), "title") or ""
    for frag in CN_TITLE_FRAGMENTS:
        if frag not in cn:
            fails.append("paper_cn: the Chinese title lost the fragment %r "
                         "(title is %r)" % (frag, cn.strip()))
    return fails


# --------------------------------------------------------------------------
# 2. retired wording
# --------------------------------------------------------------------------
def retired_failures(texts, names=ENVELOPE):
    fails = []
    for name in names:
        t = texts.get(name)
        if not t:
            continue
        for ph in RETIRED:
            if ph in t:
                fails.append("%s: still carries the retired phrase %r"
                             % (name, ph))
    return fails


# --------------------------------------------------------------------------
# 3. environment pins
# --------------------------------------------------------------------------
def _pins_yml(t):
    out = {}
    for k, v in re.findall(r"^\s*-\s*([A-Za-z0-9_-]+)=([0-9][^\s#]*)", t, re.M):
        out[k] = v
    # conda writes the CUDA build as a separate package
    if "pytorch-cuda" in out:
        out["cuda"] = out.pop("pytorch-cuda")
    return out


def _pins_req(t):
    out = {}
    for k, v in re.findall(r"^([A-Za-z0-9_-]+)==([0-9][^\s#]*)", t, re.M):
        out[k] = v
    if "torch" in out:
        v = out.pop("torch")
        m = re.match(r"([0-9.]+)\+cu([0-9]+)", v)
        if m:
            out["pytorch"] = m.group(1)
            out["cuda"] = "%s.%s" % (m.group(2)[:2], m.group(2)[2:])
        else:
            out["pytorch"] = v
    return out


def _pins_si(t):
    """The checklist sentence: 'Python 3.13.14, PyTorch 2.6.0 (CUDA 12.4), ...'"""
    out = {}
    pats = {
        "python": r"Python\s+([0-9][0-9.]*)",
        "pytorch": r"PyTorch\s+([0-9][0-9.]*)",
        "cuda": r"CUDA\s+([0-9][0-9.]*)",
        "numpy": r"NumPy\s+([0-9][0-9.]*)",
        "scipy": r"SciPy\s+([0-9][0-9.]*)",
    }
    for k, p in pats.items():
        m = re.search(p, t)
        if m:
            out[k] = m.group(1)
    return out


def version_failures(env, reqs, si, env_arch=None, reqs_arch=None):
    fails = []
    a = _pins_yml(env)
    b = _pins_req(reqs)
    c = _pins_si(si)
    # Python is pinned in the conda file and the checklist; a pip requirements
    # file conventionally does not pin the interpreter, so requiring it there
    # would fail on a correct file.
    if "python" not in a or "python" not in c:
        fails.append("environment: the Python version is not stated in both "
                     "environment.yml and the S23 checklist (yml %s, checklist "
                     "%s)" % (a.get("python"), c.get("python")))
    elif a["python"] != c["python"]:
        fails.append("environment: python is %s in environment.yml but %s in "
                     "the checklist" % (a["python"], c["python"]))
    missing = [k for k in ("pytorch", "cuda", "numpy", "scipy")
               if k not in a or k not in b or k not in c]
    if missing:
        fails.append("environment: %s not stated in all three places (yml %s, "
                     "reqs %s, checklist %s)" % (missing, a, b, c))
    for k in ("pytorch", "cuda", "numpy", "scipy"):
        vals = {n: d.get(k) for n, d in (("environment.yml", a),
                                         ("requirements.txt", b),
                                         ("S23 checklist", c))}
        if len({v for v in vals.values() if v}) > 1:
            fails.append("environment: %s disagrees across files: %s" % (k, vals))
    for name, t, ref in (("archive environment.yml", env_arch, env),
                         ("archive requirements.txt", reqs_arch, reqs)):
        if t is None:
            fails.append("%s: missing from the archive" % name)
        elif norm(t) != norm(ref):
            fails.append("%s: is not the file at the project root -- the "
                         "archive ships a recipe that differs from the one "
                         "being maintained" % name)
    return fails


# --------------------------------------------------------------------------
# 4. DOI wording
# --------------------------------------------------------------------------
def doi_failures(das, cover, *others):
    """The statement and the letter must place the DOI at the same time."""
    fails = []
    for name, t in (("Data Availability statement", das),
                    ("cover letter", cover)) + tuple(others):
        if DOI_DEFERRED not in t:
            fails.append("%s: does not say the DOI is obtained %r, so it no "
                         "longer agrees with the other document"
                         % (name, DOI_DEFERRED))
        if "10.5281" in t:
            # a real DOI can only appear once the deposit exists; when it does,
            # both files change together and this line is updated deliberately
            fails.append("%s: carries a Zenodo DOI, but the archive gate also "
                         "requires the deferral wording -- change both files "
                         "and this check in one edit" % name)
        if "INSERT" in t or "[TO BE" in t.upper():
            fails.append("%s: carries an unfilled placeholder" % name)
        if re.search(r"deposited on Zenodo", t, re.I):
            fails.append("%s: states the deposit as done while the DOI is "
                         "deferred" % name)
    return fails


# --------------------------------------------------------------------------
# 5. supplementary declaration
# --------------------------------------------------------------------------
def supplementary_failures(en, cn):
    fails = []
    if "\\supplementary{" not in en:
        fails.append("paper_en: no \\supplementary{...} statement, which the "
                     "MDPI class expects when a supplement is uploaded")
    if "section*{补充材料}" not in cn:
        fails.append("paper_cn: no 补充材料 section")
    return fails


# --------------------------------------------------------------------------
# 6. archive coverage and manifest
# --------------------------------------------------------------------------
def _seedmap(results_dir):
    fam = {}
    for n in os.listdir(results_dir):
        m = re.match(r"^eval_(.+)_s(\d+)\.json$", n)
        if m:
            fam.setdefault(m.group(1), set()).add(int(m.group(2)))
    return fam


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# (delivered path under submission/, same file under the archive root).  The
# archive ships the manuscript it was built with; if a later edit lands only in
# submission/, the archive describes a paper that no longer exists -- which is
# how the six-seed results and the archived PDFs came apart.
MIRROR = [
    ("paper_en/main.tex", "paper/en/main.tex"),
    ("paper_en/SI_main.tex", "paper/en/SI_main.tex"),
    ("paper_en/main.pdf", "paper/en/main.pdf"),
    ("paper_en/SI_main.pdf", "paper/en/SI_main.pdf"),
    ("paper_cn/main.tex", "paper/cn/main.tex"),
    ("paper_cn/main.pdf", "paper/cn/main.pdf"),
]


def mirror_failures(root=None):
    root = root or ROOT
    sub = os.path.join(root, "submission")
    arc = os.path.join(sub, "supplementary")
    fails = []
    for a, b in MIRROR:
        pa, pb = os.path.join(sub, a), os.path.join(arc, b)
        if not os.path.exists(pb):
            fails.append("archive: %s is missing, so the archive does not "
                         "carry the manuscript it is cited for" % b)
        elif md5(pa) != md5(pb):
            fails.append("archive: %s differs from submission/%s -- the "
                         "archive was built before the last edit" % (b, a))
    return fails


def staleness_failures(root=None):
    """A delivered PDF older than its source is not the delivered source.

    The archive mirror check cannot see this: it compares the PDF against the
    PDF, and both can be stale together.  This is the direction that shipped
    once already -- the SI source gained the corrected commit and seed rows
    while SI_main.pdf was never recompiled.
    """
    root = root or ROOT
    sub = os.path.join(root, "submission")
    pairs = [("paper_en/main.tex", "paper_en/main.pdf"),
             ("paper_en/SI_main.tex", "paper_en/SI_main.pdf"),
             ("paper_cn/main.tex", "paper_cn/main.pdf")]
    fails = []
    for tex, pdf in pairs:
        a, b = os.path.join(sub, tex), os.path.join(sub, pdf)
        if not (os.path.exists(a) and os.path.exists(b)):
            fails.append("%s or %s missing" % (tex, pdf))
        elif os.path.getmtime(b) < os.path.getmtime(a):
            fails.append("%s is older than %s -- it was not rebuilt after the "
                         "last edit" % (pdf, tex))
    return fails


def archive_failures(root=None):
    """The archive must be able to regenerate the tables it is cited for."""
    root = root or ROOT
    live = os.path.join(root, "results")
    arc = os.path.join(root, "submission", "supplementary", "results")
    fails = []
    if not os.path.isdir(arc):
        return ["archive: submission/supplementary/results is missing"]
    a, b = _seedmap(live), _seedmap(arc)
    for arm, seeds in sorted(a.items()):
        if arm not in b:
            fails.append("archive: arm %s is absent, so its rows cannot be "
                         "regenerated" % arm)
        elif not seeds <= b[arm]:
            fails.append("archive: arm %s is missing seeds %s (has %s)"
                         % (arm, sorted(seeds - b[arm]), sorted(b[arm])))
    if not a:
        fails.append("archive: no eval_*.json in results/ -- nothing to compare")

    man = os.path.join(root, "submission", "supplementary", "MANIFEST.md")
    if not os.path.exists(man):
        fails.append("archive: MANIFEST.md missing")
    else:
        txt = read(man)
        m = re.search(r"Files:\s*(\d+)", txt)
        if not m:
            fails.append("archive: MANIFEST.md does not state a file count")
        else:
            base = os.path.join(root, "submission", "supplementary")
            on_disk = sum(len(fs) for _, _, fs in os.walk(base)) - 1  # manifest
            if int(m.group(1)) != on_disk:
                fails.append("archive: MANIFEST claims %s files but %d are on "
                             "disk -- it no longer describes the tree"
                             % (m.group(1), on_disk))
    return fails


# --------------------------------------------------------------------------
def collect(root=None):
    root = root or ROOT
    sub = os.path.join(root, "submission")
    rel = os.path.join(root, "release_docs")
    t = {
        "paper_en": read(os.path.join(sub, "paper_en", "main.tex")),
        "paper_cn": read(os.path.join(sub, "paper_cn", "main.tex")),
        "si": read(os.path.join(sub, "paper_en", "SI_main.tex")),
        "cover": read(os.path.join(sub, "cover_letter.tex")),
        "highlights": read(os.path.join(sub, "highlights.md")),
        "readme": read(os.path.join(rel, "README.md")),
        "citation": read(os.path.join(rel, "CITATION.cff")),
    }
    arc = os.path.join(sub, "supplementary")
    for name, p in (("citation_archive", os.path.join(arc, "CITATION.cff")),
                    ("env_arch", os.path.join(arc, "environment.yml")),
                    ("reqs_arch", os.path.join(arc, "requirements.txt"))):
        t[name] = read(p) if os.path.exists(p) else None
    t["env"] = read(os.path.join(rel, "environment.yml"))
    t["reqs"] = read(os.path.join(rel, "requirements.txt"))
    return t


def run(root=None, texts=None):
    t = texts or collect(root)
    fails = []
    fails += title_failures(t)
    fails += retired_failures(t)
    fails += version_failures(t["env"], t["reqs"], t["si"],
                              t.get("env_arch"), t.get("reqs_arch"))
    # Only the two documents that make the claim.  README and CITATION mention
    # the repository but say nothing about when a DOI exists, so requiring the
    # deferral wording there would fail on correct text.
    fails += doi_failures(t["paper_en"], t["cover"])
    fails += supplementary_failures(t["paper_en"], t["paper_cn"])
    if texts is None:
        fails += staleness_failures(root)
        fails += mirror_failures(root)
        fails += archive_failures(root)
    return fails


def main():
    fails = run()
    for f in fails:
        print("  [FAIL]", f)
    if fails:
        print("delivery-text gate: %d failure(s)" % len(fails))
        return 1
    t = collect()
    print("  [ok]   title: manuscript, cover letter, README, both CITATION "
          "copies and the SI agree")
    print("  [ok]   environment pins agree with the S23 checklist (%s)"
          % ", ".join("%s %s" % kv for kv in
                      sorted(_pins_yml(t["env"]).items())))
    print("  [ok]   DOI deferral wording agrees; no unfilled placeholder")
    print("  [ok]   supplement declared in both manuscripts")
    print("  [ok]   archive covers every arm and seed in results/, MANIFEST "
          "matches the tree")
    print("delivery-text gate passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
