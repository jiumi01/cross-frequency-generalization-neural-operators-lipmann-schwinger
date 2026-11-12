# Author: Longgang Wang <longgangwang@163.com>
"""Falsification probe for the figure containment gate.

`figures.data_inside_axes` reports plotted points that fall outside their axes.
The gate is opt-in (a panel declares `require_inside()`), and until v79 only
three of nineteen figures had declared it, so the panels that needed it most
were the ones not covered.

A guard is only worth its place if it can be shown to fail.  This probe runs the
two directions:

  * a point outside the axes must be REPORTED      (the gate is live)
  * a point inside must stay SILENT                (the gate is not noise)

The second direction matters: a gate that reports everything is switched off
within a day, which is worse than no gate at all.

Run: python code/test_fig_containment.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figures as F


def _panel(xs, lim):
    fig, ax = plt.subplots()
    ax.plot(xs, [1.0] * len(xs))
    ax.set_xlim(*lim)
    F.require_inside(ax)
    return fig


def main():
    bad = F.data_inside_axes(_panel([1.0, 3.0, 5.0], (2.0, 4.0)))
    good = F.data_inside_axes(_panel([2.5, 3.0, 3.5], (2.0, 4.0)))
    print("point outside the axes ->", bad or "NOT REPORTED")
    print("points inside the axes ->", good or "silent")
    if not bad:
        print("FAIL: a point one unit outside the axis was not reported")
        return 1
    if good:
        print("FAIL: the gate fired on data inside the axis")
        return 1
    print("PASS: fires on the bad case, silent on the good one")
    return 0


if __name__ == "__main__":
    sys.exit(main())
