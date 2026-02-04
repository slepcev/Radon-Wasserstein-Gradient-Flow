import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts._paths import runs_root
from src.algorithms import SVGD, RRW_fft, KDRW_fft
from src.targets import StandardGaussianTarget
from src.tools import RWRunConfig, StandardGaussianInitData, SVGDRunConfig, mmd_torch


def collect_env_metadata(run_device: torch.device) -> dict:
    """Collect environment metadata for reproducibility and debugging."""
    meta = {
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "device": str(run_device),
    }

    if run_device.type == "cuda":
        meta.update(
            {
                "cuda_version": torch.version.cuda,
                "cudnn_version": torch.backends.cudnn.version(),
                "gpu_name": torch.cuda.get_device_name(0),
            }
        )

    if run_device.type == "mps":
        meta["mps_available"] = bool(torch.backends.mps.is_available())

    return meta


if __name__ == "__main__":

    # Setup hardware
    # Right now GPU is used when available. 
    # For n_samples*dim <= 2^22, CPU tends to be faster than GPU for FFT-based methods
    if torch.cuda.is_available():
        run_device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        run_device = torch.device("mps")
    else:
        run_device = torch.device("cpu")

    # ===============================================
    # Comment in/out below for appropriate experiment
    # ===============================================

    dim = 2
    step_RW = 0.1
    step_SVGD = 0.2
    steps_RW = 100000
    steps_SVGD = 50000
    #run_num = 10
    #n_values = [2 ** (6 + k) for k in range(6)]
    
    run_num = 1
    n_values = [2 ** (6 + k) for k in range(3)]

    # dim = 32
    # step_RW = 0.2
    # step_SVGD = 0.4
    # steps_RW = 50000
    # steps_SVGD = 25000
    # run_num = 10
    # n_values = [2 ** (5 + k) for k in range(7)]

    # dim = 256
    # step_RW = 0.5
    # step_SVGD = 0.5
    # steps_RW = 20000
    # steps_SVGD = 20000
    # run_num = 10
    # n_values = [2 ** (6 + k) for k in range(7)]

    # dim = 2048
    # step_RW = 1.0
    # step_SVGD = 1.0
    # steps_RW = 10000
    # steps_SVGD = 10000
    # run_num = 5
    # n_values = [2 ** (6 + k) for k in range(8)]

    k_to_n = {k: n_values[k] for k in range(len(n_values))}

    # Target and initialization
    standard_gaussian = StandardGaussianTarget()
    stnd_gaussian_init_data = StandardGaussianInitData()

    # Manifest
    root = runs_root() / "quantization"
    dim_root = root / "gaussian" / f"dim_{dim}"
    dim_root.mkdir(parents=True, exist_ok=True)

    dim_params = {
        "experiment_type": "quantization",
        "target": getattr(standard_gaussian, "name", "standard_gaussian"),
        "dim": int(dim),
        "n_trials": int(run_num),
        "n_values": [int(n) for n in n_values],
        "device": str(run_device),
        "rw": {
            "kernel": "gaussian",
            "bw_kdrw_method": "scaled_with_n",
            "bw_kdrw_const": 1.0,
            "bw_rrw_method": "scaled_with_n",
            "bw_rrw_const": 1.0,
            "eps_rule": "0.02 / n",
            "L": 5.0,
            "M": 8,
            "step": float(step_RW),
            "max_iter": int(steps_RW),
        },
        "svgd": {
            "kernel": "gaussian",
            "bw_SVGD_method": "scaled_with_n",
            "bw_SVGD_const": 2.0,
            "step": float(step_SVGD),
            "max_iter": int(steps_SVGD),
            "alpha": 0.9,
            "ada": False,
        },
        "mmd_eval": {"bw": 1.0, "var": 1.0},
        "env": collect_env_metadata(run_device),
    }

    with open(dim_root / "params.json", "w") as f:
        json.dump(dim_params, f, indent=2)

    # Run sweep
    for k in range(len(n_values)):
        n_samples = k_to_n[k]
        print(f"\nNumber of samples: {n_samples}")

        for r in range(1, run_num + 1):
            print(f"\n[Gaussian] Round {r}")

            # Reproducibility: seed once per trial
            seed = 1000 * k + r
            torch.manual_seed(seed)
            np.random.seed(seed)
            if run_device.type == "cuda":
                torch.cuda.manual_seed_all(seed)

            rw_config = RWRunConfig(
                standard_gaussian,
                stnd_gaussian_init_data,
                random_seed=seed,
                is_kdrw=True,
                is_rrw=True,
                dim=dim,
                n_samples=n_samples,
                step=step_RW,
                max_iter=steps_RW,
                kernel="gaussian",
                bw_kdrw_method="scaled_with_n",
                bw_kdrw_const=1.0,
                bw_rrw_method="scaled_with_n",
                bw_rrw_const=1.0,
                eps_kdrw=0.02 / n_samples,
                L=5.0,
                M=8,
            )

            svgd_config = SVGDRunConfig(
                rw_config.target,
                rw_config.initial_data,
                random_seed=seed,
                dim=rw_config.dim,
                n_samples=rw_config.n_samples,
                step=step_SVGD,
                max_iter=steps_SVGD,
                kernel="gaussian",
                bw_SVGD_method="scaled_with_n",
                bw_SVGD_const=2.0,
            )

            # Initial particles and IID baseline
            x0 = rw_config.initial_data.init_data(
                rw_config.dim, rw_config.n_samples
            ).to(run_device)
            x_iid = torch.randn(n_samples, dim, device=run_device)
            mmd_iid = mmd_torch(x_iid.detach(), bw=1.0, var=1.0, device=run_device)
            print(f"IID         : MMD^2={mmd_iid:.6e}")

            # KDRW_fft
            t0 = time.time()
            x_rw = KDRW_fft(
                x0,
                rw_config.target.score,
                step=rw_config.step,
                max_iter=int(rw_config.max_iter * 998 / 1000),
                bw=rw_config.get_bw_kdrw(),
                eps=rw_config.eps_kdrw,
                L=rw_config.L,
                M=rw_config.M,
                kernel=rw_config.kernel,
            )
            x_rw = KDRW_fft(
                x_rw,
                rw_config.target.score,
                step=rw_config.step / 20,
                max_iter=int(rw_config.max_iter * 40 / 1000),
                bw=rw_config.get_bw_kdrw(),
                eps=rw_config.eps_kdrw,
                L=rw_config.L,
                M=rw_config.M,
                kernel=rw_config.kernel,
            )
            t_rw = time.time() - t0
            mmd_rw = mmd_torch(x_rw.detach(), bw=1.0, var=1.0, device=run_device)
            print(f"KDRW_fft      : time={t_rw:.3f}s   MMD^2={mmd_rw:.6e}")

            # RRW_fft
            t0 = time.time()
            x_rrw = RRW_fft(
                x0,
                rw_config.target.score,
                step=rw_config.step,
                max_iter=int(rw_config.max_iter * 998 / 1000),
                bw=rw_config.get_bw_rrw(),
                eps=rw_config.eps_kdrw,
                L=rw_config.L,
                M=rw_config.M,
                kernel=rw_config.kernel,
            )
            x_rrw = RRW_fft(
                x_rrw,
                rw_config.target.score,
                step=rw_config.step / 20,
                max_iter=int(rw_config.max_iter * 40 / 1000),
                bw=rw_config.get_bw_rrw(),
                eps=rw_config.eps_kdrw,
                L=rw_config.L,
                M=rw_config.M,
                kernel=rw_config.kernel,
            )
            t_rrw = time.time() - t0
            mmd_rrw = mmd_torch(x_rrw.detach(), bw=1.0, var=1.0, device=run_device)
            print(f"RRW_fft     : time={t_rrw:.3f}s   MMD^2={mmd_rrw:.6e}")

            # SVGD
            t0 = time.time()
            x_svgd = SVGD(
                x0,
                svgd_config.target.score,
                step=svgd_config.step,
                max_iter=int(svgd_config.max_iter * 998 / 1000),
                bw=svgd_config.get_bw_SVGD(),
                alpha=0.9,
                ada=False,
            )
            x_svgd = SVGD(
                x_svgd,
                svgd_config.target.score,
                step=svgd_config.step / 20,
                max_iter=int(svgd_config.max_iter * 40 / 1000),
                bw=svgd_config.get_bw_SVGD(),
                alpha=0.9,
                ada=False,
            )
            t_svgd = time.time() - t0
            mmd_svgd = mmd_torch(x_svgd.detach(), bw=1.0, var=1.0, device=run_device)
            print(f"SVGD         : time={t_svgd:.3f}s   MMD^2={mmd_svgd:.6e}")

            # Save experiments
            trial_path = dim_root / f"n_{n_samples:06d}" / f"trial_{r:02d}"
            trial_path.mkdir(parents=True, exist_ok=True)

            np.save(trial_path / "final_KDRW_fft.npy", np.array(x_rw.detach().cpu()))
            np.save(trial_path / "final_RRW_fft.npy", np.array(x_rrw.detach().cpu()))
            np.save(trial_path / "final_SVGD.npy", np.array(x_svgd.detach().cpu()))
            np.save(trial_path / "x_iid.npy", np.array(x_iid.detach().cpu()))

            np.save(trial_path / "mmd_KDRW_fft.npy", np.array(mmd_rw))
            np.save(trial_path / "mmd_RRW_fft.npy", np.array(mmd_rrw))
            np.save(trial_path / "mmd_SVGD.npy", np.array(mmd_svgd))
            np.save(trial_path / "mmd_iid.npy", np.array(mmd_iid))

            params = {
                "experiment_type": "quantization",
                "target": getattr(standard_gaussian, "name", "standard_gaussian"),
                "dim": int(dim),
                "n_samples": int(n_samples),
                "seed": int(seed),
                "device": str(run_device),
                "rw_config": {
                    "random_seed": int(rw_config.random_seed),
                    "is_kdrw": bool(rw_config.is_kdrw),
                    "is_rrw": bool(rw_config.is_rrw),
                    "n_samples": int(rw_config.n_samples),
                    "step": float(rw_config.step),
                    "max_iter": int(rw_config.max_iter),
                    "kernel": str(rw_config.kernel),
                    "bw_kdrw_method": str(rw_config.bw_kdrw_method),
                    "bw_kdrw_const": float(rw_config.bw_kdrw_const),
                    "bw_rrw_method": str(rw_config.bw_rrw_method),
                    "bw_rrw_const": float(rw_config.bw_rrw_const),
                    "eps_kdrw": float(rw_config.eps_kdrw),
                    "L": float(rw_config.L),
                    "M": int(rw_config.M),
                },
                "svgd_config": {
                    "random_seed": int(svgd_config.random_seed),
                    "n_samples": int(svgd_config.n_samples),
                    "step": float(svgd_config.step),
                    "max_iter": int(svgd_config.max_iter),
                    "kernel": str(svgd_config.kernel),
                    "bw_SVGD_method": str(svgd_config.bw_SVGD_method),
                    "bw_SVGD_const": float(svgd_config.bw_SVGD_const),
                },
                "mmd_eval": {"bw": 1.0, "var": 1.0},
                "env": collect_env_metadata(run_device),
            }

            with open(trial_path / "params.json", "w") as f:
                json.dump(params, f, indent=2)
