"""
This file is intended as an easy-to-follow tutorial for running Radon--Wasserstein algorithms:
  1) Pick an algorithm
  2) Pick parameters
  3) Pick a target
--------
  4) Sets up run configuration
  5) Runs
  6) Plots final configuration and MMD^2 convergence curve

Run from repo root:
    python -m scripts.tutorial

Notes
-----
- The diagnostic uses MMD^2 against N(0,I). For banana and general Gaussian targets,
  we first transform samples via target.transform_to_gaussian before computing MMD^2.
"""

from __future__ import annotations
import math
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.algorithms import RRW, KDRW, SVGD, RRW_fft, KDRW_fft
from src.targets import (
    BananaParams,
    StandardGaussianTarget,
    TransformableBananaTarget,
    GeneralGaussianParams,
    GeneralGaussianTarget,
    TransformableGeneralGaussianTarget,
    is_transformable,
)
from src.tools import (
    RoundGaussianInitData,
    RWRunConfig,
    StandardGaussianInitData,
    SVGDRunConfig,
    mmd_torch,
)

def transform_to_normal(x: torch.Tensor, target) -> torch.Tensor:
    """For banana (transformable), compute MMD in Gaussian coords; else identity."""
    if is_transformable(target):
        return target.transform_to_gaussian(x)
    return x


def main() -> None:

    # =========================
    # 1) Algorithm (uncomment ONE)
    # =========================

    alg = "RRW_fft"
    # alg = "KDRW_fft"
    # alg = "RRW"
    # alg = "KDRW"
    # alg = "SVGD"

    # =========================
    # 2) Parameters
    # =========================

    dim = 2
    n_samples = 512
    seed = 1  # random seed

    if dim < 40:   # smaller step sizes for low dimensions
        step_size = 0.01
    elif dim < 200:
        step_size = 0.1
    else:
        step_size = 0.5
    total_steps = 20000

    # RW kernel / bandwidth / FFT params
    kernel = "gaussian"  # other options for RW: "laplacian"
    epsilon = 0.02 / n_samples # epsilon from the KDRW and RRW equations
    L = 5.0  # used by *_fft
    M = 8  # used by *_fft
    bw_kdrw_method, bw_kdrw_const = (
        "adaptive",
        1.0,
    )  # defines bandwidth to scale like 2*(standard deviation of projection)*n^{-1/5}
    """other options for bw_kdrw_method: "constant", "scaled_with_n". See tools.py for details """
    bw_rrw_method, bw_rrw_const = (
        "adaptive",
        1.0,
    )  # defines bandwidth to scale like (standard deviation of projection)*n^{-1/5}
    """other options for bw_rrw_method: "constant", "scaled_with_n" """
    bw_svgd_method, bw_svgd_const = (
        "median",
        1.0,
    )  # defines bandwidth to scale like (standard deviation of projection)*n^{-1/5}
    """other options for bw_svgd_method: "constant", "scaled_with_n", "median_trick" """


    # Convergence curve checkpoints
    n_checkpoints = 35

    # =========================
    # 3) Target (uncomment ONE block)
    # =========================

    ### --- Banana target ---
    """d-dimensional banana (Rosenbrock-type) target with potential U(x) = 0.5 * ( ||x_r||^2 + (u^2 / sigma^2) ) where:
       - x = [x_1, ..., x_d],
       - x_r = [x_1, ..., x_{d-1}],
       - u = (x_d - mu) - (c / sqrt(d)) * ( ||x_r||^2 - (d - 1) ).
    for larger c and smaller sigma the target is more challenging and hence the time steps should be reduced """
    """"""
    bparams = BananaParams(c=0.5, sigma=0.25, mu=0.0)
    target = TransformableBananaTarget(params=bparams)
    init_data = RoundGaussianInitData(std=0.5, shift=1.0)

    ### --- Standard Gaussian ---
    # target = StandardGaussianTarget()
    # init_data = RoundGaussianInitData(std=0.5, shift=1.0)

    ### --- General Gaussian with diagonal Precision matrix ---
    # gauss_mu = torch.zeros(dim)
    # gauss_prec_diag = torch.ones(dim)
    # gauss_prec_diag[0] = 16.0  
    # gauss_prec_diag[1] = 1.0
    # gauss_Prec = torch.diag(gauss_prec_diag)
    # gauss_L= torch.diag(torch.sqrt(gauss_prec_diag))  # Cholesky factor of Precision matrix
    # gauss_params = GeneralGaussianParams(mu=gauss_mu, Prec=gauss_Prec, L=gauss_L)
    # target = TransformableGeneralGaussianTarget(gauss_params)
    # init_data = RoundGaussianInitData(std=0.5, shift=1.0)

    # =========================
    # 4) Set up run configuration
    # =========================

    rw_config = RWRunConfig(
        target=target,
        initial_data=init_data,
        random_seed=seed,
        is_kdrw=alg in {"KDRW", "KDRW_fft"},
        is_rrw=alg in {"RRW", "RRW_fft"},
        dim=dim,
        n_samples=n_samples,
        step=step_size,
        max_iter=total_steps,
        kernel=kernel,
        bw_kdrw_method=bw_kdrw_method,
        bw_kdrw_const=bw_kdrw_const,
        bw_rrw_method=bw_rrw_method,
        bw_rrw_const=bw_rrw_const,
        eps_kdrw=epsilon,
        L=L,
        M=M,
    )
    svgd_config = SVGDRunConfig(
        target=target,
        initial_data=init_data,
        random_seed=seed,
        dim=dim,
        n_samples=n_samples,
        step=step_size,
        max_iter=total_steps,
        bw_SVGD_method=bw_svgd_method,
        bw_SVGD_const=bw_svgd_const,
        kernel=kernel,
    )

    # Reproducibility
    torch.manual_seed(rw_config.random_seed)
    np.random.seed(rw_config.random_seed)

    # =========================
    # 5) Run
    # =========================
    
    # Setup hardware: Note for low dimensions or low n_sample CPU may be faster for FFT algs.
    # In particular on 2024 MacBook PRO CPU was faster than MPS for RRW_fft precisely when dim*n_samples <= 2^22.
    # We use that as the cutoff here, but you should modify based on your hardware.
    if torch.cuda.is_available() and ((alg=="RRW" or alg=='KDRW' or alg=="SVGD") or rw_config.dim * rw_config.n_samples >= 2**20):
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and ((alg=="RRW" or alg=='KDRW' or alg=="SVGD") or rw_config.dim * rw_config.n_samples >= 2**22):
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    dtype = torch.float32

    # print("Is transformable target: ", is_transformable(rw_config.target))

    # Set initial data
    x = rw_config.initial_data.init_data(rw_config.dim, rw_config.n_samples, device=device, dtype=dtype)

    # step functions for each algorithm
    if alg == "RRW_fft":
        step_fn = lambda x_in, iters: RRW_fft(
            x_in.detach(),
            rw_config.target.score,
            step=rw_config.step,
            max_iter=iters,
            bw=rw_config.get_bw_rrw(),
            eps=rw_config.eps_kdrw,
            L=rw_config.L,
            M=rw_config.M,
            kernel=rw_config.kernel,
        )
    elif alg == "KDRW_fft":
        step_fn = lambda x_in, iters: KDRW_fft(
            x_in.detach(),
            rw_config.target.score,
            step=rw_config.step,
            max_iter=iters,
            bw=rw_config.get_bw_kdrw(),
            eps=rw_config.eps_kdrw,
            L=rw_config.L,
            M=rw_config.M,
            kernel=rw_config.kernel,
        )
    elif alg == "RRW":
        step_fn = lambda x_in, iters: RRW(
            x_in,
            rw_config.target.score,
            step=rw_config.step,
            max_iter=iters,
            bw=rw_config.get_bw_rrw(),
            eps=rw_config.eps_kdrw,
            kernel=rw_config.kernel,
        )
    elif alg == "KDRW":
        step_fn = lambda x_in, iters: KDRW(
            x_in,
            rw_config.target.score,
            step=rw_config.step,
            max_iter=iters,
            bw=rw_config.get_bw_kdrw(),
            eps=rw_config.eps_kdrw,
            kernel=rw_config.kernel,
        )
    elif alg == "SVGD":
        step_fn = lambda x_in, iters: SVGD(
            x_in,
            svgd_config.target.score,
            step=svgd_config.step,
            max_iter=iters,
            bw=svgd_config.get_bw_SVGD(),
            alpha=0.9,
            ada = False,
        )
    else:
        raise ValueError(f"Unknown alg={alg!r}")

    # Run in chunks to get a convergence curve
    cum_steps = np.unique(
        np.round(np.logspace(0, math.log10(total_steps), n_checkpoints)).astype(int)
    )
    cum_steps[0] = 0
    times = [0.0]
    cum_steps = np.unique(cum_steps)

    mmd_vals = [
        float(
            mmd_torch(
                transform_to_normal(x.detach().cpu(), rw_config.target),
                bw=1.0,
                var=1.0,
                device=torch.device("cpu"),
            )
        )
    ]
    for k in range(1, len(cum_steps)):
        delta = int(cum_steps[k] - cum_steps[k - 1])
        if delta <= 0:
            continue
        # The next line calls the selected algorithm's step function and executes 'delta' iterations
        x = step_fn(x, delta)
        times.append(times[-1] + rw_config.step * delta)
        # Compute MMD^2 against N(0,I) after transforming the samples if needed
        mmd_vals.append(
            float(
                mmd_torch(
                    transform_to_normal(x.detach().cpu(), rw_config.target),
                    bw=1.0,
                    var=1.0,
                    device=torch.device("cpu"),
                )
            )
        )

    times = np.asarray(times, dtype=float)
    mmd_vals = np.asarray(mmd_vals, dtype=float)
    x_iid = torch.randn(rw_config.n_samples, rw_config.dim, device=torch.device("cpu"))
    # for comparison compute MMD^2 for IID samples from N(0,I)
    mmd_iid = mmd_torch(x_iid, bw=1.0, var=1.0, device=torch.device("cpu"))

    # ============================
    # 6) Plot
    # ============================

    plt.figure(figsize=(6.5, 5.5))
    plt.plot(times, mmd_vals, label=alg)
    plt.axhline(y=mmd_iid, linestyle='--', color='black', linewidth=1, label="IID")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("time")
    plt.ylabel(r"MMD$^2$")
    plt.tight_layout()
    plt.legend()
    if dim == 2:
        x_vis = x.detach().cpu().numpy()
        plt.figure(figsize=(6.5, 5.5))
        plt.scatter(x_vis[:, 0], x_vis[:, 1], s=6)

        plt.axis("equal")
        plt.tight_layout()
    elif dim == 3:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        x_vis = x.detach().cpu().numpy()
        fig = plt.figure(figsize=(6.5, 5.5))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(x_vis[:, 0], x_vis[:, 1], x_vis[:, 2], s=6)
        plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
