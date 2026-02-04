# `scripts/` — entry points, experiments, and figures

All commands below should be run from the root directory of the project.

## Start here: tutorial

```bash
python -m scripts.tutorial
```

This is the intended entry point. It demonstrates:
- selecting an algorithm and target by commenting/uncommenting lines,
- setting parameters explicitly,
- constructing the corresponding run configuration (`RWRunConfig`),
- running the method, and
- producing plots (convergence diagnostic and final particles).

## Paper sweeps (write to `runs/`)

```bash
python -m scripts.convergence
python -m scripts.quantization
python -m scripts.bandwidths
```

These scripts create run folders under `runs/` by default.

## Figure generation (reads from `runs/`)

```bash
python -m scripts.convergence_fig
python -m scripts.quantization_fig
python -m scripts.bandwidths_fig
```
