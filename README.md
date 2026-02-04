# Radon--Wasserstein Gradient Flows

**Repository for "Radon--Wasserstein Gradient Flows for Interacting-Particle Sampling in High Dimensions"** (arXiv link forthcoming)  

by [Elias Hess-Childs](https://www.math.cmu.edu/~ehesschi/) (CMU), [Dejan Slepcev](https://www.math.cmu.edu/~slepcev/) (CMU), and Lantian Xu (CMU).  


## Quickstart 

The script tutorial.py is intended as the main entry point. It allows one to run the main algorithms with several different target distributions.
In particular, one can
- select an algorithm (KDRW/RRW, FFT/non-FFT, and SVGD),
- select a target (banana, standard Gaussian, or general Gaussian), and
- set parameters.
  
The code runs the method and plots a convergence diagnostic and final particle positions (in 2D). To run from Terminal, use the following command from the root directory of the project:

```bash
python -m scripts.tutorial
```


## Reproducing paper experiments

All scripts save their data under `runs/`, by default. 

- The script bandwidths.py runs the experiments that generate the data for Figure 4.2, which studies how the output depends on the selected bandwidth.
- The script convergence.py runs the experiments that generate the data for Figure 4.4, which compares the convergence over time of the algorithms for banana (and Gaussian) targets. To run the experiment for a specific panel of the figure, uncomment the appropriate parameter block (chosen by target, dimension, and number of particles).
- The script quantization.py runs the experiments that generate the data for Figures 4.5 and 4.6, which study the approximation error for different particle numbers. To run the experiment for a specific panel of the figure, uncomment the appropriate parameter block (chosen by dimension).

Once the data have been generated using the scripts above, one can generate the figures by running
- convergence_fig.py
- quantization_fig.py
- bandwidths_fig.py


## Repository layout
- `src/` — core implementations (algorithms, targets, utilities)
- `scripts/` — runnable entry points (tutorial, sweeps, figure generation)

## License

MIT License. See `LICENSE`.

## Citation

If you found this repository useful or the associated paper interesting, please consider citing our (forthcoming) paper.
