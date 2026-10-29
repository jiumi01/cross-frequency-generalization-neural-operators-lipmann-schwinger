# Author: Longgang Wang <longgangwang@163.com>
# Affiliation: Luoyang Institute of Science and Technology, Luoyang 471000, Henan, China

"""Guard every cross-reference from the main text into the supplement.

The SI is a separate LaTeX document, so \\ref cannot reach it.  The main text
therefore cites SI floats through a generated map, gen_si_refs.tex, keyed by
the SI's own \\label:

    Supplementary Table~\\siTab{tab:si_born}
    Supplementary Figure~\\siFig{figS:bound}

Before that map existed the prose carried literal numbers.  Literals are only
correct for one particular order of the SI: inserting one figure ahead of the
others shifted every figure after it by one, and the main text went on citing
"Supplementary Figure~S3" for a float that had become S4.  The old version of
this gate checked tables only, and table numbers happened to still be right, so
the drift reached a submitted draft with a clean report.

What it asserts
  1. gen_si_refs.tex is byte-identical to what the SI source generates now
     (catches a map that was not rebuilt after the SI changed);
  2. the map agrees with the numbers LaTeX itself wrote into SI_main.aux --
     an instrument independent of the parser in gen_si_labels.py;
  3. every \\siTab{}/\\siFig{} key used in the main text is defined (an
     undefined key typesets ?? in the PDF, which is invisible to a text audit);
  4. no literal "Supplementary Table~N" / "Supplementary Figure~SN" survives
     in the main text or in any generated table;
  5. the numbering form matches the render: figures S-prefixed in the SI,
     tables not;
  6. each citation sits in a sentence that shares vocabulary with the caption
     of the float it names -- a key can exist and still name the wrong float;
  7. no caption carries a stale self-citation ("as Supplementary Figure~S3").

Exit code 0 iff every assertion holds.
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
# The falsification probe runs this gate against a mutated copy of the
# manuscript; see the note in the older version of this file.
if "--root" in sys.argv:
    ROOT = os.path.abspath(sys.argv[sys.argv.index("--root") + 1])
EN = os.path.join(ROOT, "submission", "paper_en")
PEN = os.path.join(ROOT, "paper", "en")

FAIL = []


def read(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def bad(msg):
    FAIL.append(msg)


def strip_comments(src):
    out = []
    for line in src.split("\n"):
        k = 0
        keep = line
        while True:
            j = line.find("%", k)
            if j < 0:
                break
            b = 0
            p = j - 1
            while p >= 0 and line[p] == "\\":
                b += 1
                p -= 1
            if b % 2 == 1:
                k = j + 1
                continue
            keep = line[:j]
            break
        out.append(keep)
    return "\n".join(out)


def inline_inputs(src, base):
    seen = set()
    for _ in range(64):
        m = re.search(r"\\input\{([^}]+)\}", src)
        if not m:
            break
        name = m.group(1).strip()
        if not name.endswith(".tex"):
            name += ".tex"
        path = os.path.join(base, name)
        if not os.path.exists(path) or path in seen:
            break
        seen.add(path)
        src = src[:m.start()] + read(path) + src[m.end():]
    return src


def braced(text, start):
    """Return the group that starts at text[start] == '{' (balanced)."""
    assert text[start] == "{"
    depth = 0
    i = start
    while i < len(text):
        c = text[i]
        if c == "\\":
            i += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1:i]
        i += 1
    return ""


def floats(src):
    """{label: {'kind':..., 'n':..., 'caption':...}} in SI source order."""
    out = {}
    for kind in ("table", "figure"):
        pat = r"\\begin\{%s\*?\}(.*?)\\end\{%s\*?\}" % (kind, kind)
        for n, m in enumerate(re.finditer(pat, src, re.S), 1):
            body = m.group(1)
            lab = re.search(r"\\label\{([^}]+)\}", body)
            if not lab:
                continue
            cap = ""
            cm = re.search(r"\\caption\s*\{", body)
            if cm:
                cap = " ".join(braced(body, cm.end() - 1).split())
            out[lab.group(1)] = {"kind": kind, "n": n, "caption": cap}
    return out


def expected_map(fl):
    """The text gen_si_refs.tex must contain, for the SI as it stands now."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "gen_si_labels", os.path.join(ROOT, "code", "gen_si_labels.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    text, tables, figs = mod.build(ROOT)
    return text, tables, figs


def aux_numbers():
    """{label: raw value string} from the aux that built the delivered PDF."""
    path = os.path.join(EN, "SI_main.aux")
    if not os.path.exists(path):
        return None
    out = {}
    for m in re.finditer(r"\\newlabel\{([^}]+)\}\{\{([^}]*)\}", read(path)):
        out[m.group(1)] = m.group(2)
    return out


def stems_of(text):
    text = re.sub(r"\\[a-zA-Z]+\*?", " ", text)
    text = re.sub(r"[^A-Za-z ]", " ", text)
    return {w.lower() for w in text.split() if len(w) >= 5}


# Discourse connectives and function words.  They can be "distinctive" across
# captions -- "therefore" appears in the cost caption and nowhere else -- while
# carrying no subject matter, so a caption match on one of them is not evidence
# that the sentence is about that float.  Measured case: redirecting a Born
# citation at the cost table left "therefore" as the only overlap and the check
# passed.  This is a class filter (words that denote no subject), not a list of
# the collisions seen so far.
CONNECTIVES = {
    "therefore", "although", "though", "however", "whereas", "because",
    "between", "within", "without", "rather", "against", "through", "which",
    "while", "where", "whose", "there", "their", "these", "those", "being",
    "would", "could", "should", "might", "must", "about", "above", "after",
    "before", "during", "since", "until", "under", "over", "thus", "hence",
    "also", "only", "both", "each", "more", "most", "much", "many", "some",
    "such", "when", "with", "from", "into", "onto", "upon", "across", "along",
    "among", "around", "than", "then", "this", "that", "have", "been", "does",
}


def same_stem(a, b):
    k = 0
    while k < len(a) and k < len(b) and a[k] == b[k]:
        k += 1
    return k >= 5


def sentence_of(masked, start, end):
    before = [m for m in re.finditer(r"\.(?:\s|$)", masked[:start])]
    a = before[-1].end() if before else 0
    after = [m for m in re.finditer(r"\.(?:\s|$)", masked[end:])]
    b = end + after[0].start() + 1 if after else len(masked)
    return stems_of(masked[a:b])


def mask_math(s):
    out = list(s)
    i = 0
    while i < len(s):
        if s[i] == "$":
            j = s.find("$", i + 1)
            if j < 0:
                break
            for k in range(i, j + 1):
                if out[k] != "\n":
                    out[k] = " "
            i = j + 1
        else:
            i += 1
    return "".join(out)


def main():
    si_path = os.path.join(EN, "SI_main.tex")
    if not os.path.exists(si_path):
        print("FAIL: no SI source at %s" % si_path)
        return 1
    si_raw = read(si_path)
    fl = floats(inline_inputs(strip_comments(si_raw), EN))
    n_tab = sum(1 for v in fl.values() if v["kind"] == "table")
    n_fig = sum(1 for v in fl.values() if v["kind"] == "figure")
    print("SI floats: %d tables, %d figures, %d labelled"
          % (n_tab, n_fig, len(fl)))

    # 1. the generated map is current -------------------------------------
    want, tab_list, fig_list = expected_map(fl)
    gen_path = os.path.join(PEN, "gen_si_refs.tex")
    if not os.path.exists(gen_path):
        bad("paper/en/gen_si_refs.tex is missing -- run `--steps tex`")
        have = ""
    else:
        have = read(gen_path)
        if have != want:
            bad("gen_si_refs.tex is stale: it does not match the SI source "
                "(run `--steps tex`)")
    defined_tabs = dict(re.findall(r"sitab@([^\\]+)\\endcsname\{(\d+)\}", have))
    defined_figs = dict(re.findall(r"sifig@([^\\]+)\\endcsname\{(\d+)\}", have))
    print("map: %d table keys, %d figure keys"
          % (len(defined_tabs), len(defined_figs)))

    # 2. cross-check against LaTeX's own counters --------------------------
    aux = aux_numbers()
    if aux is None:
        print("note: no SI_main.aux -- counter cross-check skipped")
    else:
        checked = 0
        for lab, val in list(defined_tabs.items()) + list(defined_figs.items()):
            if lab not in aux:
                bad("%s is in the map but LaTeX recorded no such label in "
                    "SI_main.aux" % lab)
                continue
            digits = re.sub(r"[^0-9]", "", aux[lab])
            if digits and int(digits) != int(val):
                bad("%s: map says %s, SI_main.aux says %s"
                    % (lab, val, aux[lab]))
            checked += 1
        print("counter cross-check: %d labels agree with SI_main.aux" % checked)

    # 3. numbering form matches the render --------------------------------
    if r"\renewcommand{\thefigure}{S\arabic{figure}}" not in si_raw:
        bad("the SI does not number figures with an S prefix, but the main "
            "text cites them as S<n>")
    if re.search(r"\\renewcommand\{\\thetable\}", si_raw):
        bad("the SI redefines \\thetable, but the map emits a bare number")

    # 4./5. every key used in the main text resolves -----------------------
    main_path = os.path.join(EN, "main.tex")
    src = read(main_path)
    masked = mask_math(src)
    used_tab = re.findall(r"\\siTab\{([^}]+)\}", src)
    used_fig = re.findall(r"\\siFig\{([^}]+)\}", src)
    if not used_tab:
        bad("no \\siTab{...} in the main text -- did the phrasing change?")
    for key in used_tab:
        if key not in defined_tabs:
            bad("\\siTab{%s}: no such SI table label (typesets ??)" % key)
    for key in used_fig:
        if key not in defined_figs:
            bad("\\siFig{%s}: no such SI figure label (typesets ??)" % key)
    print("main-text keys: %d table citations over %d labels, "
          "%d figure citations over %d labels"
          % (len(used_tab), len(set(used_tab)),
             len(used_fig), len(set(used_fig))))

    # 6. no literals survive ----------------------------------------------
    targets = [main_path] + [os.path.join(PEN, f) for f in
                             sorted(os.listdir(PEN))
                             if f.startswith("gen_") and f.endswith(".tex")]
    lit = re.compile(r"Supplementary\s+(?:Table~\d|Figure~S?\d)")
    for path in targets:
        hits = lit.findall(read(path))
        if hits:
            bad("%s still carries a literal SI float number: %s"
                % (os.path.relpath(path, ROOT), hits[:4]))

    # 7. the sentence must be about the float it names ---------------------
    # Evaluated per key, not per citation: a key is cited once in a sentence
    # that carries its argument ("the dashed curves of ... are a property of
    # the data") and again in a terse "are plotted in ..." follow-up.  The
    # follow-up shares no vocabulary with any caption and is not evidence of a
    # mispointed key; requiring every citation to match would fail on prose
    # style rather than on substance.
    stems = {k: stems_of(v["caption"]) - CONNECTIVES for k, v in fl.items()}
    for kind, macro, defined in (("table", "siTab", defined_tabs),
                                 ("figure", "siFig", defined_figs)):
        contexts = {}
        for m in re.finditer(r"\\%s\{([^}]+)\}" % macro, src):
            key = m.group(1)
            if key in fl:
                contexts.setdefault(key, []).append(
                    sentence_of(masked, m.start(), m.end()))
        for key, ctxs in sorted(contexts.items()):
            mine = stems[key]
            others = set().union(*[stems[k] for k in stems
                                   if k != key and fl[k]["kind"] == kind]) \
                if len(stems) > 1 else set()
            distinctive = mine - others
            hit, shared = False, set()
            for ctx in ctxs:
                content = ctx - CONNECTIVES
                if any(same_stem(w, d) for w in content for d in distinctive):
                    hit = True
                shared |= {w for w in content
                           if any(same_stem(w, d) for d in mine)}
            okk = hit if kind == "table" else (hit or len(shared) >= 2)
            if not okk:
                bad("Supplementary %s \\%s{%s}: no citing sentence shares "
                    "vocabulary with the caption" % (kind.capitalize(), macro, key))
                print("      sentence stems: %s" % sorted(set().union(*ctxs)))
                print("      caption: %s" % fl[key]["caption"][:110])

    # 8. no stale self-citation in a caption ------------------------------
    stale = re.findall(r"as Supplementary (?:Table|Figure)~S?\d", si_raw)
    if stale:
        bad("an SI caption still names its own number by hand: %s" % stale[:3])

    if FAIL:
        for msg in FAIL:
            print("FAIL  " + msg)
        print("RESULT: %d problem(s)" % len(FAIL))
        return 1
    print("RESULT: every SI cross-reference resolves and matches its caption")
    return 0


if __name__ == "__main__":
    sys.exit(main())
