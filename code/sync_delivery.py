# Author: Longgang Wang
"""Keep the delivered manuscript and the audited manuscript identical.

The gates (check_claims, lint_style, audit_paper) read paper/<lang>/main.tex,
while the file that is actually submitted lives in submission/paper_<lang>/.
Those two trees drifted once already: on 2026-10-22 paper/en/main.tex was four
days stale and still lacked the two subsections added in submission/paper_en,
so every "gate passed" was a statement about a manuscript nobody would send.
That is the failure class this file closes, and check_claims asserts it rather
than trusting the operator to remember.

Direction, stated once because it is not symmetric:

  body       submission/paper_<lang>/main.tex  ->  paper/<lang>/main.tex
             The submission copy is the manuscript of record.  Edit there.

  generated  paper/<lang>/gen_*.tex, figs/*    ->  submission/paper_<lang>/
             make_results_tex.py and figures.py write into paper/, so those
             artefacts travel the other way.  Hand-editing them is forbidden
             anyway (they carry a do-not-edit header).

Run with --check to report without writing; that is what the gate does.
"""
import os, sys, shutil, hashlib, glob, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

LANGS = ("en", "cn")


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def pairs(lang):
    sub = os.path.join(C.ROOT, "submission", "paper_%s" % lang)
    pap = os.path.join(C.ROOT, "paper", lang)
    return sub, pap


def generated_names(pap):
    """Relative paths the generator owns: gen_*.tex, then figs/<file>."""
    out = [os.path.basename(p)
           for p in sorted(glob.glob(os.path.join(pap, "gen_*.tex")))]
    out += [os.path.join("figs", os.path.basename(p))
            for p in sorted(glob.glob(os.path.join(pap, "figs", "*")))
            if os.path.isfile(p)]
    return out


def check():
    bad = 0
    for lang in LANGS:
        sub, pap = pairs(lang)
        a = os.path.join(sub, "main.tex")
        b = os.path.join(pap, "main.tex")
        if not (os.path.exists(a) and os.path.exists(b)):
            print("[skip] %s: missing main.tex in one of the two trees" % lang)
            continue
        if md5(a) != md5(b):
            print("[FAIL] %s: paper/%s/main.tex is not the delivered manuscript "
                  "(audited %d bytes, delivered %d bytes)"
                  % (lang, lang, os.path.getsize(b), os.path.getsize(a)))
            bad += 1
        else:
            print("[ok]   %s: the audited body is the delivered body (%s)"
                  % (lang, md5(a)[:8]))
        # The SI is hand-edited in submission/ and compiled from paper/, so it
        # drifts by exactly the mechanism the body used to: on 2026-10-22 the
        # SI cross-references were corrected in submission/ only, and the PDF
        # that got delivered was compiled from the stale paper/en copy.  Same
        # direction as the body: the submission copy is the manuscript of
        # record.
        si_a = os.path.join(sub, "SI_main.tex")
        si_b = os.path.join(pap, "SI_main.tex")
        if os.path.exists(si_a) and os.path.exists(si_b):
            if md5(si_a) != md5(si_b):
                print("[FAIL] %s: paper/%s/SI_main.tex is not the delivered SI "
                      "(audited %d bytes, delivered %d bytes)"
                      % (lang, lang, os.path.getsize(si_b), os.path.getsize(si_a)))
                bad += 1
            else:
                print("[ok]   %s: the audited SI is the delivered SI (%s)"
                      % (lang, md5(si_a)[:8]))
    return bad


def sync():
    for lang in LANGS:
        sub, pap = pairs(lang)
        src = os.path.join(sub, "main.tex")
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(pap, "main.tex"))
            print("body      %s -> %s" % (src, os.path.join(pap, "main.tex")))
        si_src = os.path.join(sub, "SI_main.tex")
        if os.path.exists(si_src):
            shutil.copy2(si_src, os.path.join(pap, "SI_main.tex"))
            print("body      %s -> %s" % (si_src, os.path.join(pap, "SI_main.tex")))
        for name in generated_names(pap):
            s = os.path.join(pap, name)
            d = os.path.join(sub, name)
            os.makedirs(os.path.dirname(d), exist_ok=True)
            shutil.copy2(s, d)
        print("generated %d artefacts %s -> %s" % (len(generated_names(pap)), pap, sub))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.check:
        sys.exit(1 if check() else 0)
    sync()
    sys.exit(1 if check() else 0)
