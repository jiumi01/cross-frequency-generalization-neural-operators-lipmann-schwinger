"""Assert that every error bar's cap is wider than the marker it sits on.

The defect this guards against is invisible in source review: `capsize` is a
plausible-looking number, the call succeeds, the figure renders, and the caps
are nevertheless nowhere on the page.  A cap is a horizontal segment of
length `capsize` centred on the bar's endpoint, i.e. on the marker centre, so
any `capsize <= ms` is drawn entirely inside a filled marker and disappears.
This project shipped such figures and a reader reported "the error bars are
not obvious" while every call did have a capsize.

The rule is `capsize >= ms + 1.0` (0.5 pt of cap visible each side).  The
file currently uses `ms + 2.0`; the gate only enforces the minimum, so
tightening the margins further does not need a gate edit, but shrinking them
below the marker does.

Calls without `yerr` draw no bar and are skipped.  A call has no marker to
hide behind only when it says so twice -- neither a `marker=` kwarg nor a
marker character in `fmt`.  Most calls here pass `fmt="-o"` instead of
`marker="o"`, so reading only the kwarg would quietly exempt the very bars
the rule exists for.  When the marker comes from `fmt` and `ms` is absent,
matplotlib's default of 6.0 is used.

Exit code 0 = all caps visible, 1 = at least one hidden.
"""
import ast
import io
import os
import sys

PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures.py")
MIN_CLEARANCE = 1.0
DEFAULT_MS = 6.0          # matplotlib's scatter/line marker default
LINE_ONLY = set("-: ")    # fmt characters that draw no marker


def _marker_symbol(fmt):
    """Return the marker character encoded in a format string, or None."""
    if not isinstance(fmt, str):
        return None
    for ch in fmt:
        if ch not in LINE_ONLY and ch not in ".,":
            return ch
    return None


def main(path=PATH):
    src = io.open(path, encoding="utf-8").read()
    tree = ast.parse(src)
    calls, bad = 0, []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "attr", getattr(node.func, "id", ""))
        if name != "errorbar":
            continue
        kw = {k.arg: k.value for k in node.keywords if k.arg}

        def num(key):
            if key not in kw:
                return None
            try:
                return ast.literal_eval(kw[key])
            except Exception:
                return None

        if "yerr" not in kw:
            continue
        marker = num("marker")
        if marker in (None, "None", "", " "):
            # No marker kwarg: fall back to the format string before deciding
            # that there is no disc to hide the cap behind.
            marker = _marker_symbol(num("fmt"))
        if marker is None:
            continue
        ms, cs = num("ms"), num("capsize")
        if ms is None:
            ms = DEFAULT_MS
        if cs is None:
            bad.append((node.lineno, "missing capsize", ms, cs))
            continue
        calls += 1
        if cs < ms + MIN_CLEARANCE - 1e-9:
            bad.append((node.lineno, "cap inside marker", ms, cs))

    for ln, why, ms, cs in bad:
        print("  [FAIL] figures.py:%d  %s  (ms=%s, capsize=%s, need >= %s)"
              % (ln, why, ms, cs, None if ms is None else ms + MIN_CLEARANCE))

    print("  %d errorbar call(s) with yerr checked, %d hidden cap(s)" % (calls, len(bad)))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
