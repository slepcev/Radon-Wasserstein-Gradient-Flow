# `src/` — core code

This folder contains the core implementation used by `scripts/`.

Most users should start with the tutorial. Run the following from the root directory of the project:

```bash
python -m scripts.tutorial
```

## Modules
- `algorithms.py`: main algorithm code
  - Radon-Wasserstein gradient descent for interacting particles.
  					In particular: KDRW/RRW and FFT-accelerated variants.
  - Stein Variational Gradient Descent (SVGD) of Liu and Wang, for comparison.
- `targets.py`: target distributions (standard Gaussian, GeneralGaussian, banana, etc.)
  - A target is "Transformable" if there exists an explicit bijective transport map from the distribution to the standard Gaussian, as well as an inverse. These transformations are used when comparing the samples produced to the target measure. Such transforms are not needed to execute the algorithms, but are used to evaluate the quality of the samples. 
- `tools.py`: utilities and run configurations
  - `RWRunConfig` for KDRW/RRW runs (bandwidth rules via `get_bw_kdrw()` / `get_bw_rrw()`)
  - `SVGDRunConfig` for SVGD runs
  - maximum mean discrepancy to a round gaussian: `mmd_torch`
  - initialization helpers (e.g., `StandardGaussianInitData`, `RoundGaussianInitData`)
