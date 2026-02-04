import json
import platform
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts._paths import runs_root
from src.algorithms import RRW, KDRW, RRW_fft, KDRW_fft
from src.targets import (
    BananaParams,
    StandardGaussianTarget,
    TransformableBananaTarget,
    is_transformable,
)
from src.tools import RoundGaussianInitData, RWRunConfig, mmd_torch


# Collect meta data
def collect_env_metadata(run_device: torch.device) -> dict:
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
    if torch.cuda.is_available():
        run_device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        run_device = torch.device("mps")
    else:
        run_device = torch.device("cpu")

    # Parameters
    n_samples = 1024
    dims = [2, 256]
    max_iter = 50000
    eps_val = 1e-2 / n_samples
    L_val = 5.0
    M_val = 8
    n_trials = 8
    iid_random_seed = 42
    n_iid_trials = 50

    bandwidths = 2.0 ** np.linspace(-6.0, 3.0, 10)

    # Target distributions
    targets = [
        StandardGaussianTarget(),
        TransformableBananaTarget(params=BananaParams(c=0.5, sigma=0.25, mu=0.0)),
    ]

    root = runs_root() / "bandwidths"
    root.mkdir(parents=True, exist_ok=True)

    for target in targets:
        target_folder_name = (
            "gaussian" if target.name == "standard_gaussian" else target.name
        )
        print(f"\n>>> TARGET: {target_folder_name.upper()} <<<")

        # Dimension dependent step sizes
        for dim in dims:
            if target.name == "banana" and dim < 2:
                continue
            if dim == 2:
                step = 1e-2
            else:
                step = 1e-1

            print(f"--- Dimension: {dim} ---")
            dim_root = root / target_folder_name / f"dim_{dim:04d}"
            dim_root.mkdir(parents=True, exist_ok=True)

            dim_params = {
                "experiment_type": "bandwidths",
                "target": target_folder_name,
                "dim": int(dim),
                "n_samples": int(n_samples),
                "device": str(run_device),
                "max_iter": int(max_iter),
                "step": float(step),
                "eps_kdrw": float(eps_val),
                "L": float(L_val),
                "M": int(M_val),
                "n_trials": int(n_trials),
                "bandwidths": [float(b) for b in bandwidths.tolist()],
                "mmd_eval": {"bw": 1.0, "var": 1.0},
                "env": collect_env_metadata(run_device),
            }

            with open(dim_root / "params.json", "w") as f:
                json.dump(dim_params, f, indent=2)

            # Prepare summaries
            results_mean = {alg: [] for alg in ["KDRW", "RRW", "KDRW_fft", "RRW_fft"]}
            results_std = {alg: [] for alg in ["KDRW", "RRW", "KDRW_fft", "RRW_fft"]}

            # Main experiments
            for bw_val in bandwidths:
                bw_root = dim_root / f"bw_{bw_val:.6f}"
                bw_root.mkdir(parents=True, exist_ok=True)

                rw_config = RWRunConfig(
                    target=target,
                    initial_data=RoundGaussianInitData(std=1.0, shift=2.0),
                    random_seed=42,
                    is_kdrw=True,
                    is_rrw=True,
                    dim=dim,
                    n_samples=n_samples,
                    step=step,
                    max_iter=max_iter,
                    kernel="gaussian",
                    bw_kdrw_method="constant",
                    bw_kdrw_const=bw_val,
                    bw_rrw_method="constant",
                    bw_rrw_const=bw_val,
                    eps_kdrw=eps_val,
                    L=L_val,
                    M=M_val,
                )

                trial_params = {
                    "experiment_type": "bandwidths",
                    "target": target_folder_name,
                    "dim": int(dim),
                    "n_samples": int(n_samples),
                    "seed": int(rw_config.random_seed),
                    "device": str(run_device),
                    "bandwidth": float(bw_val),
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
                    "mmd_eval": {"bw": 1.0, "var": 1.0},
                    "env": collect_env_metadata(run_device),
                }

                with open(bw_root / "params.json", "w") as f:
                    json.dump(trial_params, f, indent=2)

                trial_mmds = {alg: [] for alg in results_mean.keys()}

                for r in range(1, n_trials + 1):
                    trial_path = bw_root / f"trial_{r:02d}"
                    trial_path.mkdir(parents=True, exist_ok=True)

                    if run_device.type == "cuda":
                        torch.backends.cuda.matmul.allow_tf32 = True

                    torch.manual_seed(rw_config.random_seed + r)

                    x0 = rw_config.initial_data.init_data(dim, n_samples).to(run_device)

                    x_kdrw = KDRW(
                        x0.clone(),
                        target.score,
                        step=rw_config.step,
                        max_iter=rw_config.max_iter,
                        bw=rw_config.get_bw_kdrw(),
                        eps=rw_config.eps_kdrw,
                        kernel=rw_config.kernel,
                    )

                    x_rrw = RRW(
                        x0.clone(),
                        target.score,
                        step=rw_config.step,
                        max_iter=rw_config.max_iter,
                        bw=rw_config.get_bw_rrw(),
                        eps=rw_config.eps_kdrw,
                        kernel=rw_config.kernel,
                    )

                    x0_cpu = x0.cpu()
                    x_kdrw_f = KDRW_fft(
                        x0_cpu.clone(),
                        target.score,
                        step=rw_config.step,
                        max_iter=rw_config.max_iter,
                        bw=rw_config.get_bw_kdrw(),
                        eps=rw_config.eps_kdrw,
                        L=rw_config.L,
                        M=rw_config.M,
                        kernel=rw_config.kernel,
                    )

                    x_rrw_f = RRW_fft(
                        x0_cpu.clone(),
                        target.score,
                        step=rw_config.step,
                        max_iter=rw_config.max_iter,
                        bw=rw_config.get_bw_rrw(),
                        eps=rw_config.eps_kdrw,
                        L=rw_config.L,
                        M=rw_config.M,
                        kernel=rw_config.kernel,
                    )

                    outputs = [x_kdrw, x_rrw, x_kdrw_f, x_rrw_f]
                    names = ["KDRW", "RRW", "KDRW_fft", "RRW_fft"]

                    for name, final_x in zip(names, outputs):

                        if is_transformable(target):
                            eval_x = target.transform_to_gaussian(final_x)
                        else:
                            eval_x = final_x

                        m_val = mmd_torch(eval_x, bw=1.0, var=1.0, device=run_device)
                        trial_mmds[name].append(m_val)

                        np.save(
                            trial_path / f"final_{name}.npy",
                            final_x.detach().cpu().numpy(),
                        )
                        np.save(trial_path / f"mmd_{name}.npy", np.array(m_val))

                for alg in results_mean.keys():
                    avg_mmd = np.mean(trial_mmds[alg])
                    results_mean[alg].append(avg_mmd)

                    std_mmd = np.std(trial_mmds[alg])
                    results_std[alg].append(std_mmd)

                kdrw_fft_mean = results_mean["KDRW_fft"][-1]
                kdrw_fft_std = results_std["KDRW_fft"][-1]
                print(
                    f"  BW: {bw_val:.4f} | KDRW_fft MMD: {kdrw_fft_mean:.6e} +/- {kdrw_fft_std:.6e}"
                )

            for alg in results_mean.keys():
                np.save(dim_root / f"mmd_{alg}_avg.npy", np.array(results_mean[alg]))
                np.save(dim_root / f"mmd_{alg}_std.npy", np.array(results_std[alg]))

            np.save(dim_root / "bandwidths.npy", np.array(bandwidths))

            # I.I.D. runs
            print(f"  Running I.I.D. Baseline for {target.name}...")

            iid_root = dim_root / "iid"
            iid_root.mkdir(parents=True, exist_ok=True)

            iid_params = {
                "experiment_type": "iid_baseline",
                "target": getattr(target, "name", str(target)),
                "initial_data": {
                    "name": "standard_gaussian",
                    "note": "inverse_transformed_if_applicable",
                },
                "random_seed": iid_random_seed,
                "dim": dim,
                "n_samples": n_samples,
                "n_trials": n_iid_trials,
                "device": str(run_device),
            }

            with open(iid_root / "params.json", "w") as f:
                json.dump(iid_params, f, indent=2)

            iid_values = []

            for i in range(1, n_iid_trials + 1):
                trial_path = iid_root / f"trial_{i:02d}"
                trial_path.mkdir(parents=True, exist_ok=True)

                torch.manual_seed(iid_random_seed + i)
                z_iid = torch.randn(n_samples, dim, device=run_device)

                if is_transformable(target):
                    x_iid = target.inverse_transform(z_iid)
                    eval_x = target.transform_to_gaussian(x_iid)
                else:
                    x_iid = z_iid
                    eval_x = x_iid

                val = mmd_torch(eval_x, bw=1.0, var=1.0, device=run_device)
                iid_values.append(val)

                np.save(trial_path / "final_iid.npy", x_iid.detach().cpu().numpy())
                np.save(trial_path / "mmd_iid.npy", np.array(val))

            iid_avg = np.mean(iid_values)
            iid_std = np.std(iid_values)

            np.save(dim_root / "mmd_iid_avg.npy", np.array(iid_avg))
            np.save(dim_root / "mmd_iid_std.npy", np.array(iid_std))

            print(f"  I.I.D. Baseline: {iid_avg:.4e} +/- {iid_std:.4e}")

    print("\nSimulations complete.")
