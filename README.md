# Cross-Frequency Generalization of Neural Operators on a Periodic Lippmann–Schwinger Benchmark

An empirical characterisation of how neural operators generalise across the
excitation frequency when solving the periodic Lippmann–Schwinger integral
equation. On a strictly delimited numerical benchmark (one equation family,
fixed-loss-tangent propagator, contrasts up to \(v_{\max}=1\)), the paper
systematically characterises the cross-frequency error curve, quantifies its
boundary and plateau, and probes candidate mechanisms with controlled
comparisons.

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

## Manuscript status

Under review. Data availability (DOI/code archive) will be published on
acceptance, per journal policy.
