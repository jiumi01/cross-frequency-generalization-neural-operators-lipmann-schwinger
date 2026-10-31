# Author: Longgang Wang <longgangwang@163.com>
# Affiliation: Luoyang Institute of Science and Technology, Luoyang 471000, Henan, China

"""End-to-end driver: from trained checkpoints to the two compiled manuscripts.

Stages (each idempotent and individually selectable with --steps):

  spec     measure the out-of-band energy fraction           -> results/spectrum.json
  solver   solver hygiene measurements (dense + freezing)    -> results/solver_fig.json
  eval     evaluate every run over the (f x v_max) grid      -> results/eval_*.json
  tex      emit gen_macros.tex + gen_tab_*.tex for en & cn
  figs     render every paper figure                         -> paper/<lang>/figs/*.pdf
  build    compile the English (pdflatex) and Chinese (xelatex) PDFs

LaTeX is compiled in an ASCII staging directory because the project path contains
non-ASCII characters, which MiKTeX cannot always handle through the Windows API.
"""
import os
import re
import sys
import glob
import shutil
import argparse
import subprocess
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import make_results_tex as MR

PY = sys.executable
HERE = os.path.dirname(os.path.abspath(__file__))

# Every generated file the tex stage emits and the build stage consumes.
# Derived from make_results_tex.TAB_ORDER rather than spelled out, so adding a
# table cannot leave a stale copy unchecked -- an out-of-sync list is exactly
# how a composite gen_results.tex would have survived the split unnoticed.
GEN_FILES = (["gen_bib.tex", "gen_macros.tex", "gen_si_refs.tex",
              "gen_cross.tex", "gen_tab_cross.tex", "gen_tab_cross_q.tex"]
             + ["gen_tab_%s.tex" % k for k in MR.TAB_ORDER])


def run(cmd, cwd=None, check=True, quiet=False):
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                       encoding="utf-8", errors="backslashreplace")
    if not quiet:
        tail = (p.stdout or "") + (p.stderr or "")
        tail = tail.strip().splitlines()
        for ln in tail[-25:]:
            print("   |", ln)
    if check and p.returncode != 0:
        raise RuntimeError("command failed (%d): %s" % (p.returncode, " ".join(cmd)))
    return p


def have(exe):
    return shutil.which(exe) is not None


# ---------------------------------------------------------------------------
def stage_spec():
    if os.path.exists(os.path.join(C.RES, "spectrum.json")):
        print("[spec] cached"); return
    run([PY, os.path.join(HERE, "diag_spectrum.py")])


def stage_solver():
    if os.path.exists(os.path.join(C.RES, "solver_fig.json")):
        print("[solver] cached"); return
    run([PY, os.path.join(HERE, "diag_solver_fig.py")])


def stage_eval(jobs=None):
    """Evaluate every finished run over the (f x v_max) grid.

    Runs are independent -- each owns its own output file and every record is
    written atomically -- so the work is sharded over a few processes.  A
    full re-evaluation is triggered whenever the test grid changes (evaluate.py
    compares the recorded f_test against the current one), and on ~36 runs that
    is the difference between roughly an hour and roughly twenty minutes.
    """
    if jobs is None:
        jobs = int(os.environ.get("LS_EVAL_JOBS", "3"))
    if jobs <= 1:
        run([PY, os.path.join(HERE, "evaluate.py"), "--all"])
        return
    procs = []
    for i in range(jobs):
        p = subprocess.Popen(
            [PY, os.path.join(HERE, "evaluate.py"), "--all",
             "--shard", "%d/%d" % (i, jobs)],
            cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="backslashreplace")
        procs.append(p)
    bad = []
    for i, p in enumerate(procs):
        out, _ = p.communicate()
        for ln in (out or "").strip().splitlines()[-6:]:
            print("   | [shard %d] %s" % (i, ln))
        if p.returncode != 0:
            bad.append(i)
    if bad:
        raise RuntimeError("eval shards failed: %s" % bad)


def sync_shared():
    """Mirror the files both manuscripts compile against from en to cn.

    Neither file is produced by a script, so without this mirror an edit to the
    English copy leaves the Chinese build silently compiling against the old
    layout or the old reference list -- a difference that shows up only as a
    layout change in one PDF.
    """
    for name in ("mdpi.sty", "refs.bib"):
        src = os.path.join(C.ROOT, "paper", "en", name)
        dst = os.path.join(C.ROOT, "paper", "cn", name)
        if not os.path.exists(src):
            continue
        same = (os.path.exists(dst)
                and open(src, "rb").read() == open(dst, "rb").read())
        if not same:
            shutil.copy2(src, dst)
            print("[tex] mirrored paper/en/%s -> paper/cn/" % name)


def stage_tex():
    run([PY, os.path.join(HERE, "make_results_tex.py")])
    # The frequency x mesh cross has its own single source of truth
    # (results/cross_analysis.json); every macro and both tables the
    # manuscripts quote come from here.  Must run before gen_si_labels.py,
    # which inlines the SI's \input{gen_tab_cross_q} to count its floats.
    run([PY, os.path.join(HERE, "make_cross_tex.py")])
    # The reference list is emitted from refs.bib rather than run through
    # bibtex: the MDPI entry shape is produced directly, which would otherwise
    # require mdpi.bst (not redistributable here).
    run([PY, os.path.join(HERE, "make_bib_tex.py")])
    # SI float numbers, keyed by the SI's own \label: the main text cannot use
    # \ref across documents, and a hand-written number drifts the moment a
    # float is inserted (which is exactly how "Supplementary Figure~S3" came to
    # name the fourth figure).
    run([PY, os.path.join(HERE, "gen_si_labels.py")])
    sync_shared()


def stage_figs():
    for lang in ("en", "cn"):
        out = os.path.join(C.ROOT, "paper", lang, "figs")
        os.makedirs(out, exist_ok=True)
        run([PY, os.path.join(HERE, "figures.py"), "--out", out])
    # the two manuscript folders share identical figure files
    src = os.path.join(C.ROOT, "paper", "en", "figs")
    dst = os.path.join(C.ROOT, "paper", "cn", "figs")
    for f in glob.glob(os.path.join(src, "*")):
        shutil.copy2(f, os.path.join(dst, os.path.basename(f)))


# ---------------------------------------------------------------------------
def _pt(line):
    """Overfull size in points from a LaTeX log line (0 if unparseable)."""
    m = re.search(r"\(([\d.]+)pt too wide\)", line)
    return float(m.group(1)) if m else 0.0


def _pt_txt(line):
    m = re.search(r"\(([\d.]+)pt too wide\).*?lines (\d+)--(\d+)", line)
    return ("%.1fpt at lines %s-%s" % (float(m.group(1)), m.group(2), m.group(3))
            if m else line.strip()[:80])


def compile_paper(lang, engine):
    """Compile paper/<lang>/main.tex in an ASCII staging dir and copy back."""
    src = os.path.join(C.ROOT, "paper", lang)
    if not os.path.exists(os.path.join(src, "main.tex")):
        print("[build] missing main.tex for", lang); return None
    if not have(engine):
        print("[build] %s not found on PATH" % engine); return None

    tmp = tempfile.mkdtemp(prefix="ls_%s_" % lang)
    try:
        work = os.path.join(tmp, "p")
        shutil.copytree(src, work)
        env_opts = ["-interaction=nonstopmode", "-halt-on-error",
                    "-file-line-error"]
        # bibtex runs only when the manuscript actually asks for it.  The MDPI
        # manuscripts pull in a pre-formatted thebibliography via \input, and
        # bibtex exits non-zero when there is no \bibdata, which would abort
        # the build for no reason.
        body = open(os.path.join(work, "main.tex"), encoding="utf-8").read()
        cmds = [[engine] + env_opts + ["main.tex"]]
        if "\\bibliography{" in body:
            cmds.append(["bibtex", "main"])
        cmds += [[engine] + env_opts + ["main.tex"]] * 2
        for i, cmd in enumerate(cmds):
            p = subprocess.run(cmd, cwd=work, capture_output=True, text=True,
                               encoding="utf-8", errors="backslashreplace")
            log = os.path.join(work, "main.log")
            if p.returncode != 0 and os.path.exists(log):
                txt = open(log, encoding="utf-8", errors="backslashreplace").read()
                errs = [l for l in txt.splitlines() if l.startswith("!")]
                print("[build:%s] step %d failed" % (lang, i))
                for e in errs[:12]:
                    print("   |", e)
                if not errs:
                    for l in txt.splitlines()[-25:]:
                        print("   |", l)
                return None

        pdf = os.path.join(work, "main.pdf")
        if not os.path.exists(pdf):
            print("[build:%s] no PDF produced" % lang); return None
        dest = os.path.join(src, "main.pdf")
        try:
            shutil.copy2(pdf, dest)
        except PermissionError:
            # Windows: the canonical main.pdf is routinely held open by the PDF
            # viewer or the built-in preview pane, and an open handle blocks the
            # overwrite.  Aborting the whole build over an environment condition
            # would silently leave the compiled artefact in the deleted staging
            # dir, so fall back to a sibling name and say so -- the canonical
            # copy refreshes on the next build once the file is closed.  (No
            # unlink is attempted: it is blocked too and a trash move would only
            # raise a second, unrelated error.)
            alt = dest[:-4] + "_new.pdf"
            shutil.copy2(pdf, alt)
            print("[build:%s] %s locked by an open viewer; wrote %s instead"
                  % (lang, os.path.basename(dest), os.path.basename(alt)))
            dest = alt
        # Copy the log back too.  Leaving it behind in the staging directory
        # (which is deleted) meant paper/<lang>/main.log stayed at whatever the
        # last hand-run left there, so a claim like "0 Overfull / 6 Underfull"
        # could be read off a log from a different, older build -- and a fatal
        # preamble error from a months-old revision could look like a live one.
        log = os.path.join(work, "main.log")
        if os.path.exists(log):
            shutil.copy2(log, os.path.join(src, "main.log"))
        warn = ""
        if os.path.exists(log):
            txt = open(log, encoding="utf-8", errors="backslashreplace").read()
            # Only count genuine dangling \ref / \cite.  LaTeX font-shape
            # warnings also contain the word "undefined" and are benign.
            und = [l for l in txt.splitlines()
                   if "undefined" in l.lower()
                   and ("reference" in l.lower() or "citation" in l.lower())]
            warn = " (%d undefined refs/cites)" % len(und) if und else ""
            if und:
                for l in und[:8]:
                    print("   |", l.strip())
            # Overfull boxes are not fatal but they are visible in print, so
            # surface the worst offenders instead of leaving them in a log that
            # gets deleted with the staging directory.
            ob = [l for l in txt.splitlines() if l.startswith("Overfull \\hbox")]
            if ob:
                worst = sorted(ob, key=lambda s: -_pt(s))[:3]
                print("   | %d overfull hbox(es); worst: %s"
                      % (len(ob), "; ".join(_pt_txt(s) for s in worst)))
        print("[build:%s] wrote %s%s" % (lang, dest, warn))
        # Deliver it here rather than leaving the copy to a human step.
        # audit_paper reads submission/, so a stale copy there is audited as if
        # it were this build; the body sync closed that for main.tex and left
        # the PDF open.  A locked target file is reported, not swallowed:
        # delivery_guard compares the two text layers and fails loudly.
        sub = os.path.join(C.ROOT, "submission", "paper_%s" % lang)
        if os.path.isdir(sub):
            try:
                shutil.copy2(pdf, os.path.join(sub, "main.pdf"))
                print("[build:%s] delivered submission/paper_%s/main.pdf" % (lang, lang))
            except PermissionError:
                print("[build:%s] WARNING submission/paper_%s/main.pdf is locked; "
                      "delivery is now stale and the parity gate will fail"
                      % (lang, lang))
        return dest
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def check_generated_fresh(folder, label=None):
    """Refuse to compile a manuscript whose generated files are older than it.

    gen_bib.tex, gen_results.tex and gen_macros.tex are derived from main.tex,
    refs.bib and the evaluation JSONs.  Running `--steps build` after editing a
    manuscript but before the tex stage compiles against the previous
    generation, and the only symptom is a handful of "citation undefined"
    warnings plus a reference list numbered against the old citation order --
    which is easy to read as cosmetic.  Make it fatal instead.

    ``folder`` is passed in rather than derived so that the falsifiability probe
    can exercise this on a scratch directory.
    """
    label = label or folder
    src = os.path.join(folder, "main.tex")
    if not os.path.exists(src):
        return
    newest_input = os.path.getmtime(src)
    for extra in ("refs.bib",):
        p = os.path.join(folder, extra)
        if os.path.exists(p):
            newest_input = max(newest_input, os.path.getmtime(p))
    stale = [g for g in GEN_FILES
             if os.path.exists(os.path.join(folder, g))
             and os.path.getmtime(os.path.join(folder, g)) < newest_input]
    if stale:
        raise RuntimeError(
            "paper/%s: %s older than main.tex/refs.bib -- run `--steps tex` first"
            % (label, ", ".join(stale)))


def stage_build():
    for lang in ("en", "cn"):
        check_generated_fresh(os.path.join(C.ROOT, "paper", lang), lang)
    # A `None` return means the engine exited non-zero or produced no PDF.  That
    # used to be printed and then dropped, so on 2026-10-22 the English build
    # died on a missing figure while the driver reported success and a Chinese
    # PDF was written as a whole -- "the build passed" was true of one
    # manuscript and false of the other.  The return value is the verdict, so
    # the verdict is what decides.
    built = {}
    for lang, engine in (("en", "pdflatex"), ("cn", "xelatex")):
        built[lang] = compile_paper(lang, engine)
    missing = [k for k, v in built.items() if v is None]
    if missing:
        raise RuntimeError("build failed for: " + ", ".join(missing))


def stage_audit():
    # The gates are run in sequence and their exit codes are collected rather
    # than discarded: an audit that failed silently would let a pipeline report
    # success on numbers that no longer match the text.
    gates = ["audit_paper", "check_claims", "lint_style", "check_math_macros",
             "check_math_delims", "check_prose"]
    failed = []
    for g in gates:
        p = run([PY, os.path.join(HERE, g + ".py")], check=False)
        if p.returncode != 0:
            failed.append(g)
    # audit_paper checks that the generated NUMBERS match a recomputation,
    # check_claims checks that the SENTENCES about them are still true on the
    # final data, and lint_style checks how those sentences are written and how
    # long the captions are.  Three disjoint instruments on one artefact.
    if failed:
        raise RuntimeError("gate failure: " + ", ".join(failed))


STAGES = dict(spec=stage_spec, solver=stage_solver, eval=stage_eval,
              tex=stage_tex, figs=stage_figs, build=stage_build, audit=stage_audit)
ORDER = ["spec", "solver", "eval", "tex", "figs", "build", "audit"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default="all")
    a = ap.parse_args()
    steps = ORDER if a.steps == "all" else [s.strip() for s in a.steps.split(",")]
    failed = []
    for s in steps:
        if s not in STAGES:
            print("unknown step:", s); continue
        print("\n=== stage: %s ===" % s)
        try:
            STAGES[s]()
        except Exception as e:
            # A failing stage must not silently look like a finished pipeline:
            # the shell drivers downstream key off the completion marker, and
            # an exception here used to be the only signal.
            print("[stage %s] FAILED: %s" % (s, e))
            failed.append(s)
    # Completion marker, written even on failure so that a waiting driver can
    # decide for itself instead of hanging.  Records which stages failed.
    with open(os.path.join(C.RES, ".pipeline_done"), "w", encoding="utf-8") as fh:
        fh.write("%s steps=%s failed=%s\n"
                 % (C.now(), ",".join(steps), ",".join(failed) or "none"))
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
