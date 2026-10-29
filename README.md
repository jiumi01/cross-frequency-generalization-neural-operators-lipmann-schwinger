# Cross-Frequency Generalization of Neural Operators for a Periodic Lippmann-Schwinger Benchmark

An empirical characterisation of how neural operators generalise across the
excitation frequency when solving the periodic Lippmann–Schwinger integral
equation. On a strictly delimited numerical benchmark (one equation family,
fixed-loss-tangent propagator, contrasts up to \(v_{\max}=1\)), the paper
measures the cross-frequency error curve under a paired protocol, locates its
band-edge boundary, reads the plateau against the trivial incident-field
baseline, and tests which interventions do and do not move that boundary. The
boundary is separated from the sampling density it is usually confounded with
by evaluating one multi-grid model over the product of frequency and mesh.

## What this repository contains

This is the **code and data** companion of the manuscript (the manuscript
itself and its submission artefacts are not part of this repository).

| Path | Contents |
|---|---|
| `code/` | The full reproduction pipeline: data generation, solver, training, evaluation, figure and table generation, and the claims/audit gates used during revision. |
| `results/` | All experiment outputs as structured JSON (per-arm errors, convergence, spectral, conditioning, anchor, Marmousi and intervention results), i.e. the numbers behind every figure and table. |

## Reproduction

The heavy corpora are **not redistributed**:

- `data/` (~13 GB of training caches and the third-party Marmousi models) and
  `runs/` (~4.6 GB of training runs) are regenerated or re-downloaded via the
  scripts in `code/`; see `code/` entry points (e.g. `common.py`, the dataset
  builders and `make_paper.py --steps tex,build,audit` for the final artefact
  pipeline).
- The Marmousi models are third-party data; obtain them under their original
  licenses before re-running the real-geology level.

`results/` contains every number reported in the manuscript, keyed by the same
names used in the paper (e.g. `grid_conv_256.json`, `spectrum.json`,
`realgeo.json`, `prescreen_anchor.json`).

The frequency x mesh cross reported in Section 3.4 has its own chain, in this
order: `cross_design.py` (media drawn at \(384^2\) and spectrally restricted to
each grid) -> `gen_cross_cache.py` -> `flatten_cross_cache.py` ->
`train_multigrid.py` -> `cross_eval.py` -> `cross_analyze.py` (writes
`results/cross_analysis.json`, including the pre-registered verdict) ->
`make_cross_tex.py` (emits every macro and both tables the manuscripts quote,
and refuses to run without a verdict).

## Manuscript status

**Not submitted to any journal.** This repository has never been under
review, and the manuscript it accompanies is not simultaneously under
consideration elsewhere. The exact state the paper's numbers were produced
from is identified by the commit hash and tree hash quoted in its Data
Availability Statement; that commit's tree is this repository's content, so
`git checkout <commit>` reproduces it.

An archival deposit with a DOI (Zenodo) is made **on acceptance**, per journal
policy; until then the permanent identifier is the versioned snapshot named in
the Data Availability Statement.
