# Author: Longgang Wang <longgangwang@163.com>
# Affiliation: Luoyang Institute of Science and Technology, Luoyang 471000, Henan, China

"""Class-level style gate for the two manuscripts.

`audit_paper.py` checks that the *numbers* are right and `check_claims.py` checks
that the *sentences* about them are true.  Neither looks at how the sentences are
written, which is where the remaining defects of this round live:

  * em-dashes used as a parenthetical aside ("the stronger---and decisive---claim")
  * explanatory parentheses that repeat what the clause already said
    ("(the same scatterers at every frequency)")
  * a caption or table title long enough to be a paragraph
  * "Sec.~" next to "Section~" for the same object
  * rhetorical openers that carry no information ("It is worth noting that")
  * an en-dash pair used as an aside ("no residual to threshold against -- no
    gap in which to place a gate -- which is why")

These are counted per file and printed with line numbers, so a violation is
locatable rather than merely reported.  Budgets are explicit: a rule with a
budget of 0 must be empty for the gate to pass.

Three properties of this gate are deliberate and were each added after the gate
itself was found to be blind:

  * Rules are per language.  The first version applied the English rules to the
    Chinese manuscript, where most of them cannot match, so the Chinese file was
    reported "clean" no matter how it was written.
  * Caption length is measured in reading units, not in whitespace-separated
    words.  Chinese has no inter-word spaces, so `len(text.split())` returns 1
    for an arbitrarily long Chinese caption and the caption budget was vacuous.
  * Comments are stripped before scanning.  The rules are regular expressions
    over raw source, so a generator's banner comment ("-- do not edit") was
    scored as prose, and adding that rule without this one produced two
    false positives on the first run.
"""
import os
import re
import glob
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

FAIL = []

# (name, regex, budget, description)
#
# The aside rules deliberately allow a line break between the two dashes.  An
# earlier revision wrote the inner class as `[^-\n]`, which meant a wrapped
# aside -- `... the level above it --- measured\nfield data --- is named ...` --
# was invisible to the gate: the source is hard-wrapped at 79 columns, so
# whether an aside was caught depended on where the wrap happened to fall.  Five
# such asides were live in the English manuscript while the rule reported clean.
RULES_EN = [
    ("em-dash-aside", r"---(?!original draft|review and editing)[^-]{1,140}---", 0,
     "em-dash pair used as a parenthetical aside"),
    # `--` is an en-dash in LaTeX and is correct inside a numeric range
    # ("0.05--1.00"), where it is never surrounded by spaces.  A spaced pair is
    # therefore an aside written with the wrong dash, which the `---` rule above
    # cannot see.
    #
    # The pairing has to be real, though.  A tabular row that marks "not
    # applicable" writes two *independent* `--`, one per cell --
    # `Bi-CGSTAB & -- & -- & 1.31` -- and a flat scan of the line pairs them
    # into an "aside" that does not exist.  Measured 2026-10-12: that is the
    # only line the rule flagged in either language, i.e. the rule's whole
    # output was this one false positive.  A bare `&` can only be a column
    # separator (prose must escape it as `\&`), so forbidding it inside the
    # aside body keeps the rule on running text and drops the cell markers.
    # What this gives up: an aside that spans a tabular alignment point -- the
    # `---` rule still catches one written with an em dash.
    ("en-dash-aside", r"[ \t]--[ \t][^-&]{1,140}[ \t]--[ \t]", 0,
     "en-dash pair used as a parenthetical aside; use `---` or restructure"),
    ("explanatory-paren",
     r"\((?:i\.e\.|e\.g\.|which|that is|the same|so that|not |see |but )",
     0, "parenthesis that explains rather than names"),
    ("sec-abbrev", r"\bSec\.~?\s*\\ref", 0,
     "abbreviated cross-reference; use 'Section'"),
    ("rhetorical-opener",
     r"(?:It is worth (?:noting|being explicit)|Note that|Importantly,|"
     r"Crucially,|Notably,|In other words|Interestingly,)",
     0, "rhetorical opener carrying no information"),
    ("citation-after-space", r"[a-zA-Z0-9]\s+\\cite\{", 0,
     "citation separated from its subject by a space without a tie"),
    ("double-space-source", r"[a-zA-Z,;](?<!\\)[ \t]{2,}[a-zA-Z]", 0,
     "double space inside a sentence"),
    ("dangling-open-paren", r"\([^)\n]{40,}\n", 0,
     "parenthesis left open across a line break"),
]

RULES_CN = [
    # The Chinese manuscript is not exempt from the English rules that can
    # legitimately match it (em-dash asides, citations glued to their subject),
    # so those are carried over rather than re-declared.
    ("em-dash-aside", r"---(?!original draft|review and editing)[^-]{1,140}---", 0,
     "em-dash pair used as a parenthetical aside"),
    ("en-dash-aside", r"[ \t]--[ \t][^-&]{1,140}[ \t]--[ \t]", 0,
     "en-dash pair used as a parenthetical aside; use `---` or restructure"),
    ("citation-after-space", r"[a-zA-Z0-9\u4e00-\u9fff]\s+\\cite\{", 0,
     "citation separated from its subject by a space without a tie"),
    ("cn-em-dash-aside", r"——[^—]{1,60}——", 0,
     "中文破折号插入语"),
    ("cn-explanatory-paren",
     r"（(?:即|亦即|也就是|也就是说|换言之|换句话说|因为|由于|之所以|"
     r"而不是|而非|注意|见|参见|若|如果)",
     0, "中文括号内是解释而非命名"),
    ("cn-rhetorical-opener",
     r"(?:值得注意的是|值得一提的是|需要指出的是|应当指出的是|必须指出的是|"
     r"众所周知|不难看出|不言而喻|显而易见)",
     0, "中文说教性开头，不携带信息"),
    ("cn-halfwidth-paren-with-cjk",
     r"\([^)\n]*[\u4e00-\u9fff][^)\n]*\)", 0,
     "中文括号内容误用半角圆括号"),
    ("cn-halfwidth-comma-in-cjk",
     r"[\u4e00-\u9fff],[^ \n]", 0,
     "中文语境误用半角逗号"),
]

# A caption is a label, not a paragraph.  Latin captions are metered in words;
# CJK captions in reading units where one Latin word counts as two characters,
# which is the usual width ratio between the two scripts at the same point size.
CAPTION_WORD_BUDGET = 42
CAPTION_CJK_CHAR_BUDGET = 75


def strip_comments(text):
    """Blank out LaTeX comments, preserving line structure.

    Rules are regexes over the raw source, so without this a generator's banner
    ("-- do not edit") is scored as prose.  A `%` starts a comment unless it is
    escaped; the backslash count decides, since `\\\\%` is a line break followed
    by a comment while `\\%` is a literal percent sign.  Newlines are kept so the
    reported line numbers still point at the original file.
    """
    out = []
    for line in text.split("\n"):
        cut = None
        for j, ch in enumerate(line):
            if ch != "%":
                continue
            nb = 0
            k = j - 1
            while k >= 0 and line[k] == "\\":
                nb += 1
                k -= 1
            if nb % 2 == 0:
                cut = j
                break
        out.append(line if cut is None else line[:cut])
    return "\n".join(out)


def reading_load(body):
    """(cjk_chars, latin_words, units) for a caption body."""
    cjk = len(re.findall(r"[\u4e00-\u9fff]", body))
    latin = len(re.findall(r"[A-Za-z][A-Za-z0-9'\-]*", body))
    return cjk, latin, cjk + 2 * latin


def _strip_tex(s):
    s = re.sub(r"\\[a-zA-Z]+\*?\{?", " ", s)
    return re.sub(r"[$~\\{}]", " ", s)


def scan(path, text, rules):
    text = strip_comments(text)
    for name, rx, budget, desc in rules:
        hits = []
        for m in re.finditer(rx, text):
            ln = text[:m.start()].count("\n") + 1
            hits.append((ln, re.sub(r"\s+", " ", m.group(0))[:70]))
        if len(hits) > budget:
            FAIL.append("%s: %d x %s (budget %d) at lines %s"
                        % (os.path.basename(path), len(hits), name, budget,
                           [h[0] for h in hits][:8]))
            for ln, s in hits[:6]:
                print("      L%-5d %s" % (ln, s))
        elif hits:
            print("  [note] %s: %d x %s within budget"
                  % (os.path.basename(path), len(hits), name))


def captions(path, text):
    """Report every caption longer than its language budget."""
    n_over = 0
    text = strip_comments(text)
    for m in re.finditer(r"\\caption\{(.*?)\}\s*\n", text, re.S):
        body = _strip_tex(m.group(1))
        cjk, latin, units = reading_load(body)
        ln = text[:m.start()].count("\n") + 1
        if cjk > 4:
            over = units > CAPTION_CJK_CHAR_BUDGET
            why = "%d chars + %d words = %d units > %d" % (
                cjk, latin, units, CAPTION_CJK_CHAR_BUDGET)
        else:
            over = latin > CAPTION_WORD_BUDGET
            why = "%d words > %d" % (latin, CAPTION_WORD_BUDGET)
        if over:
            n_over += 1
            print("      L%-5d caption %s: %s..."
                  % (ln, why, re.sub(r"\s+", " ", m.group(1))[:80]))
    if n_over:
        FAIL.append("%s: %d caption(s) over budget" % (os.path.basename(path), n_over))
    else:
        print("  [ok]   %s: every caption inside its budget" % os.path.basename(path))


def run_over_root(root):
    """Scan both manuscripts under ``root``; return the number of files read.

    Split out of ``main`` so the falsifiability probe can drive it against a
    synthetic -- and deliberately empty -- tree without tripping ``sys.exit``.
    """
    scanned = 0
    for lang in ("en", "cn"):
        folder = os.path.join(root, lang)
        rules = RULES_EN if lang == "en" else RULES_CN
        print("\n=== %s (%d rules) ===" % (lang, len(rules)))
        found = 0
        # The tables were split one file per label and the concatenated
        # gen_results.tex was deleted; this list kept naming it, so generated
        # TABLE captions stopped being style-checked from the split onwards --
        # "every caption inside its budget" was then a statement about the
        # figure captions in main.tex alone.  Derive the list instead.
        gen = sorted(glob.glob(os.path.join(folder, "gen_tab_*.tex")))
        for name in ["main.tex", "gen_macros.tex"] + [os.path.basename(g) for g in gen]:
            p = os.path.join(folder, name)
            if not os.path.exists(p):
                continue
            found += 1
            text = open(p, encoding="utf-8").read()
            scan(p, text, rules)
            captions(p, text)
        if not found:
            # An empty scan is not a pass.  This gate used to skip missing files
            # silently, so pointing it at the wrong directory printed "style
            # gate passed" over zero manuscripts -- a green light manufactured
            # out of nothing, which is worse than having no gate at all, because
            # it also hides that nothing was measured.
            FAIL.append("[%s] nothing scanned under %s: a gate that reads no "
                        "files cannot report a pass" % (lang, folder))
        scanned += found
    return scanned


def main():
    # LS_PAPER_ROOT points the same instrument at an archived revision, which
    # is how the before/after counts in the response document are produced:
    # identical rules, identical metering, two snapshots.  Both spellings are
    # accepted -- the paper/ directory itself, or the snapshot root containing
    # it -- and the resolved path is always echoed, so a run that scanned the
    # wrong tree is visible in its own log.
    env = os.environ.get("LS_PAPER_ROOT")
    root = env or os.path.join(C.ROOT, "paper")
    if env:
        if not os.path.isdir(os.path.join(root, "en")):
            nested = os.path.join(root, "paper")
            if os.path.isdir(os.path.join(nested, "en")):
                root = nested
        print("scanning %s" % root)
    run_over_root(root)
    print("\n=== summary ===")
    if FAIL:
        print("  %d style violations:" % len(FAIL))
        for f in FAIL:
            print("   - " + f)
        sys.exit(1)
    print("  style gate passed")


if __name__ == "__main__":
    main()
