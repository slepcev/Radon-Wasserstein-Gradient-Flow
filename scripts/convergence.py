import json
import platform
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts._paths import runs_root

from src.algorithms import SVGD, RRW_fft, KDRW_fft
from src.targets import (
    BananaParams,
    StandardGaussianTarget,
    TransformableBananaTarget,
    is_transformable,
)
from src.tools import (
    RoundGaussianInitData,
    RWRunConfig,
    StandardGaussianInitData,
    SVGDRunConfig,
    mmd_torch,
)


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


def default_run_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    # if torch.backends.mps.is_available():
    #     return torch.device("mps")
    return torch.device("cpu")


def canonical_target_folder(target) -> str:
    """Canonical folder name for output paths."""
    
    name = getattr(target, "name", None)
    if name in {"standard_gaussian", "gaussian"}:
        return "gaussian"
    if name in {"banana"}:
        return "banana"
    return str(name) if name is not None else target.__class__.__name__.lower()


def output_path(
    *, target_name: str, dim: int, n_samples: int, root: Optional[Path] = None
) -> Path:
    base = runs_root() if root is None else root
    return base / "convergence" / target_name / f"dim_{dim}" / f"n_{n_samples}"


def mmd_against_gaussian(
    x: torch.Tensor, *, target, mmd_bw: float, mmd_var: float, run_device: torch.device
) -> float:
    """Compute MMD^2 against a standard Gaussian reference (using target transform when available)."""
    
    if getattr(target, "name", None) == "standard_gaussian":
        return mmd_torch(x, bw=mmd_bw, var=mmd_var, device=run_device)
    if is_transformable(target):
        x_t = target.transform_to_gaussian(x)
        return mmd_torch(x_t, bw=mmd_bw, var=mmd_var, device=run_device)
    raise ValueError("Target does not support Gaussian-referenced MMD evaluation.")


def rate_of_convergence_run(
    rw_config: RWRunConfig,
    svgd_config: SVGDRunConfig,
    *,
    is_svgd: bool,
    time_intervals: int,
    intervals_to_double: float,
    short_initial_steps: bool,
    ratio_of_step_numbers: float,
    ratio_of_steps: float,
    save_data: bool,
    run_name: str,
    mmd_bw: float = 1.0,
    mmd_var: float = 1.0,
    run_device: torch.device | None = None,
) -> None:
    """Run convergence experiment and save results."""

    if run_device is None:
        run_device = default_run_device()

    target_folder = canonical_target_folder(rw_config.target)
    print(
        target_folder,
        f" target, dimension= {rw_config.dim},  number of samples= {rw_config.n_samples}",
    )

    torch.manual_seed(rw_config.random_seed)
    np.random.seed(rw_config.random_seed)
    if run_device.type == "cuda":
        torch.cuda.manual_seed_all(rw_config.random_seed)

    x0 = rw_config.initial_data.init_data(rw_config.dim, rw_config.n_samples).to(
        run_device
    )

    # Per-method state
    x_rw_state = x0
    x_rrw_state = x0
    x_svgd_state = x0

    x_iid = torch.randn(rw_config.n_samples, rw_config.dim, device=run_device)

    mmd_iid = mmd_torch(x_iid, bw=mmd_bw, var=mmd_var, device=run_device)
    mmd_0 = mmd_against_gaussian(
        x0,
        target=rw_config.target,
        mmd_bw=mmd_bw,
        mmd_var=mmd_var,
        run_device=run_device,
    )

    print(f"IID     :  MMD^2 ={mmd_iid:.6e}")
    print(f"    :  MMD^2(x0) ={mmd_0:.6e}")

    physical_time_kdrw = np.zeros(time_intervals, dtype=float)
    physical_time_rrw = np.zeros(time_intervals, dtype=float)
    physical_time_svgd = np.zeros(time_intervals, dtype=float)

    times = np.zeros(time_intervals, dtype=float)
    mmd_rw = np.zeros(time_intervals, dtype=float)
    mmd_rrw = np.zeros(time_intervals, dtype=float)
    mmd_svgd = np.zeros(time_intervals, dtype=float)

    # Shift first time point so it appears on log-log plots
    if short_initial_steps:
        times[0] = rw_config.step / 20.0
    else:
        times[0] = rw_config.step * rw_config.max_iter / 4.0

    mmd_rw[0] = mmd_0
    mmd_rrw[0] = mmd_0
    mmd_svgd[0] = mmd_0

    for time_interval in range(1, time_intervals):
        if short_initial_steps:
            if time_interval < 10:
                step_RW = rw_config.step / 100.0
                step_SVGD = ratio_of_steps * step_RW
                steps_SVGD = 5
                steps_RW = int(round(ratio_of_step_numbers * steps_SVGD))
            elif time_interval < 20:
                step_RW = rw_config.step / 10.0
                step_SVGD = ratio_of_steps * step_RW
                steps_SVGD = 5
                steps_RW = int(round(ratio_of_step_numbers * steps_SVGD))
            else:
                step_RW = rw_config.step
                step_SVGD = ratio_of_steps * step_RW
                steps_SVGD = int(
                    round(
                        rw_config.max_iter
                        * 2 ** ((time_interval - 20) / intervals_to_double)
                    )
                )
                steps_RW = int(round(ratio_of_step_numbers * steps_SVGD))
        else:
            step_RW = rw_config.step
            step_SVGD = ratio_of_steps * step_RW
            steps_SVGD = int(
                round(rw_config.max_iter * 2 ** (time_interval / intervals_to_double))
            )
            steps_RW = int(round(ratio_of_step_numbers * steps_SVGD))

        print(f"\n--- Time interval {time_interval + 1}/{time_intervals} ---")
        times[time_interval] = times[time_interval - 1] + step_RW * steps_RW

        # KDRW_fft
        if rw_config.is_kdrw:
            t0 = time.time()
            x_rw = KDRW_fft(
                x_rw_state,
                rw_config.target.score,
                step=step_RW,
                max_iter=steps_RW,
                bw=rw_config.get_bw_kdrw(),
                eps=rw_config.eps_kdrw,
                L=rw_config.L,
                M=rw_config.M,
                kernel=rw_config.kernel,
            )
            physical_time_kdrw[time_interval] = time.time() - t0
            mmd_rw_current = mmd_against_gaussian(
                x_rw,
                target=rw_config.target,
                mmd_bw=mmd_bw,
                mmd_var=mmd_var,
                run_device=run_device,
            )
            print(
                f"KDRW_fft:  time={physical_time_kdrw[time_interval]:.3f}s   MMD^2 ={mmd_rw_current:.6e}"
            )
            mmd_rw[time_interval] = mmd_rw_current
            x_rw_state = x_rw

        # RRW_fft
        if rw_config.is_rrw:
            t0 = time.time()
            x_rrw = RRW_fft(
                x_rrw_state,
                rw_config.target.score,
                step=step_RW,
                max_iter=steps_RW,
                bw=rw_config.get_bw_rrw(),
                eps=rw_config.eps_kdrw,
                L=rw_config.L,
                M=rw_config.M,
                kernel=rw_config.kernel,
            )
            physical_time_rrw[time_interval] = time.time() - t0
            mmd_rrw_current = mmd_against_gaussian(
                x_rrw,
                target=rw_config.target,
                mmd_bw=mmd_bw,
                mmd_var=mmd_var,
                run_device=run_device,
            )
            print(
                f"RRW_fft: time={physical_time_rrw[time_interval]:.3f}s   MMD^2 ={mmd_rrw_current:.6e}"
            )
            mmd_rrw[time_interval] = mmd_rrw_current
            x_rrw_state = x_rrw

        # SVGD
        if is_svgd:
            t0 = time.time()
            x_svgd = SVGD(
                x_svgd_state,
                svgd_config.target.score,
                step=step_SVGD,
                max_iter=steps_SVGD,
                bw=svgd_config.get_bw_SVGD(),
                alpha=0.9,
                ada=False,
            )
            physical_time_svgd[time_interval] = time.time() - t0
            mmd_svgd_current = mmd_against_gaussian(
                x_svgd,
                target=rw_config.target,
                mmd_bw=mmd_bw,
                mmd_var=mmd_var,
                run_device=run_device,
            )
            print(
                f"SVGD:    time={physical_time_svgd[time_interval]:.3f}s   MMD^2 ={mmd_svgd_current:.6e}"
            )
            mmd_svgd[time_interval] = mmd_svgd_current
            x_svgd_state = x_svgd

    # Save data
    if not save_data:
        return

    run_path = output_path(
        target_name=target_folder, dim=rw_config.dim, n_samples=rw_config.n_samples
    )
    run_path.mkdir(parents=True, exist_ok=True)

    con_params = {
        "is_svgd": bool(is_svgd),
        "time_intervals": int(time_intervals),
        "intervals_to_double": float(intervals_to_double),
        "short_initial_steps": bool(short_initial_steps),
        "ratio_of_step_numbers": float(ratio_of_step_numbers),
        "ratio_of_steps": float(ratio_of_steps),
        "save_data": bool(save_data),
        "run_name": str(run_name),
        "mmd_bw": float(mmd_bw),
        "mmd_var": float(mmd_var),
    }

    params = {
        "experiment_type": "convergence",
        "target": target_folder,
        "dim": int(rw_config.dim),
        "n_samples": int(rw_config.n_samples),
        "seed": int(rw_config.random_seed),
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
        "convergence": con_params,
        "mmd_eval": {"bw": float(mmd_bw), "var": float(mmd_var)},
        "env": collect_env_metadata(run_device),
    }
    with open(run_path / "params.json", "w") as f:
        json.dump(params, f, indent=2)

    np.save(run_path / "times.npy", times)
    np.save(run_path / "mmd_iid.npy", np.array(mmd_iid, dtype=float))

    if rw_config.is_kdrw:
        np.save(run_path / "mmd_KDRW_fft.npy", mmd_rw)
        np.save(run_path / "physical_time_KDRW_fft.npy", physical_time_kdrw)
        np.save(run_path / "final_KDRW_fft.npy", np.array(x_rw.detach().cpu()))
    if rw_config.is_rrw:
        np.save(run_path / "mmd_RRW_fft.npy", mmd_rrw)
        np.save(run_path / "physical_time_RRW_fft.npy", physical_time_rrw)
        np.save(run_path / "final_RRW_fft.npy", np.array(x_rrw.detach().cpu()))
    if is_svgd:
        np.save(run_path / "mmd_SVGD.npy", mmd_svgd)
        np.save(run_path / "physical_time_SVGD.npy", physical_time_svgd)
        np.save(run_path / "final_SVGD.npy", np.array(x_svgd.detach().cpu()))

    np.save(run_path / "x_iid.npy", np.array(x_iid.detach().cpu()))
    np.save(run_path / "x0.npy", np.array(x0.detach().cpu()))

    mmd_data = {"times": times.tolist(), "mmd_iid": float(mmd_iid)}
    if rw_config.is_kdrw:
        mmd_data.update(
            {"mmd_rw": mmd_rw.tolist(), "physical_time_kdrw": physical_time_kdrw.tolist()}
        )
    if rw_config.is_rrw:
        mmd_data.update(
            {
                "mmd_rrw": mmd_rrw.tolist(),
                "physical_time_rrw": physical_time_rrw.tolist(),
            }
        )
    if is_svgd:
        mmd_data.update(
            {
                "mmd_svgd": mmd_svgd.tolist(),
                "physical_time_svgd": physical_time_svgd.tolist(),
            }
        )
    with open(run_path / "mmd_data.json", "w") as f:
        json.dump(mmd_data, f, indent=2)


if __name__ == "__main__":

    run_device = default_run_device()

    bparams = BananaParams(c=0.5, sigma=0.25, mu=0.0)
    banana = TransformableBananaTarget(params=bparams)
    standard_gaussian = StandardGaussianTarget()

    stnd_gaussian_init_data = StandardGaussianInitData()
    shifted_round_gaussian_init_data = RoundGaussianInitData(std=0.5, shift=1.0)

    # Experments 1-4: Initial data is i.i.d. standard Gaussian. Target is standard Gaussian.
    # Experiments 5-10: Initial data is round Gaussian shifted in horizontal direction. Target is Banana.

    # ---------------------------------------------------------------------
    # Experiment 1. gaussian, 2D, 256 samples
    # ---------------------------------------------------------------------

    # step_RW = 0.04
    # step_SVGD = 1.0
    # rw_config = RWRunConfig(
    #     standard_gaussian,
    #     stnd_gaussian_init_data,
    #     random_seed=1,
    #     is_kdrw=True,
    #     is_rrw=True,
    #     dim=2,
    #     n_samples=256,
    #     step=step_RW,
    #     max_iter=10,
    #     kernel="gaussian",
    #     bw_kdrw_method="scaled_with_n",
    #     bw_kdrw_const=1.0,
    #     bw_rrw_method="scaled_with_n",
    #     bw_rrw_const=1.0,
    #     eps_kdrw=0.02 / 256,
    #     L=5.0,
    #     M=10,
    # )
    # svgd_config = SVGDRunConfig(
    #     rw_config.target,
    #     rw_config.initial_data,
    #     random_seed=rw_config.random_seed,
    #     dim=rw_config.dim,
    #     n_samples=rw_config.n_samples,
    #     step=step_SVGD,
    #     max_iter=10,
    #     kernel="gaussian",
    #     bw_SVGD_method="scaled_with_n",
    #     bw_SVGD_const=1.0,
    # )

    # is_svgd = True
    # time_intervals = 112
    # intervals_to_double = 8.0
    # short_initial_steps = True
    # ratio_of_step_numbers = 25
    # ratio_of_steps = step_SVGD / step_RW
    # save_data = True
    # run_name = "run_gaussian_iid_"

    #    # ---------------------------------------------------------------------
    #    # Experiment 2. gaussian, 32D, 256 samples
    #    # ---------------------------------------------------------------------

    # step_RW = 0.04
    # step_SVGD = 1.0
    # rw_config = RWRunConfig(
    #     standard_gaussian,
    #     stnd_gaussian_init_data,
    #     random_seed=1,
    #     is_kdrw=True, is_rrw=True,
    #     dim=32, n_samples=256,
    #     step=step_RW, max_iter=10,
    #     kernel="gaussian",
    #     bw_kdrw_method="scaled_with_n", bw_kdrw_const=1.0,
    #     bw_rrw_method="scaled_with_n", bw_rrw_const=1.0,
    #     eps_kdrw=0.02/256, L=5.0, M=10
    # )
    # svgd_config = SVGDRunConfig(
    #     rw_config.target, rw_config.initial_data,
    #     random_seed=rw_config.random_seed,
    #     dim=rw_config.dim, n_samples=rw_config.n_samples,
    #     step=step_SVGD, max_iter=10,
    #     kernel="gaussian",
    #     bw_SVGD_method="scaled_with_n", bw_SVGD_const=1.0
    # )
    # is_svgd = True
    # time_intervals = 96
    # intervals_to_double = 8.0
    # short_initial_steps = True
    # ratio_of_step_numbers = 25
    # ratio_of_steps = step_SVGD / step_RW
    # save_data = True
    # save_plots = False
    # run_name = "run_gaussian_iid_"

    #    # ---------------------------------------------------------------------
    #    # Experiment 3. gaussian, 256D, 256 samples
    #    # ---------------------------------------------------------------------

    #    step_RW = 0.25
    #    step_SVGD = 2.0
    #    rw_config = RWRunConfig(
    #         standard_gaussian,
    #         stnd_gaussian_init_data,
    #         random_seed=1,
    #         is_kdrw=True, is_rrw=True,
    #         dim=256, n_samples=256,
    #         step=step_RW, max_iter=10,
    #         kernel="gaussian",
    #         bw_kdrw_method="scaled_with_n", bw_kdrw_const=1.0,
    #         bw_rrw_method="scaled_with_n", bw_rrw_const=1.0,
    #         eps_kdrw=0.02/256, L=5.0, M=10
    #    )
    #   svgd_config = SVGDRunConfig(
    #         rw_config.target, rw_config.initial_data,
    #         random_seed=rw_config.random_seed,
    #         dim=rw_config.dim, n_samples=rw_config.n_samples,
    #         step=step_SVGD, max_iter=10,
    #         kernel="gaussian",
    #         bw_SVGD_method="scaled_with_n", bw_SVGD_const=1.0
    #    )
    #    is_svgd = True
    #    time_intervals = 74
    #    intervals_to_double = 8.0
    #    short_initial_steps = True
    #    ratio_of_step_numbers = 8
    #    ratio_of_steps = step_SVGD / step_RW
    #    save_data = True
    #    save_plots = False
    #    run_name = "run_gaussian_iid_"

    #    # ---------------------------------------------------------------------
    #    # Experiment 4. gaussian, 256D, 1024 samples
    #    # ---------------------------------------------------------------------

    #    step_RW = 0.25
    #    step_SVGD = 2.0
    #    rw_config = RWRunConfig(
    #         standard_gaussian,
    #         stnd_gaussian_init_data,
    #         random_seed=1,
    #         is_kdrw=True, is_rrw=True,
    #         dim=256, n_samples=1024,
    #         step=step_RW, max_iter=10,
    #         kernel="gaussian",
    #         bw_kdrw_method="scaled_with_n", bw_kdrw_const=1.0,
    #         bw_rrw_method="scaled_with_n", bw_rrw_const=1.0,
    #         eps_kdrw=0.02/1024, L=5.0, M=10
    #    )
    #   svgd_config = SVGDRunConfig(
    #         rw_config.target, rw_config.initial_data,
    #         random_seed=rw_config.random_seed,
    #         dim=rw_config.dim, n_samples=rw_config.n_samples,
    #         step=step_SVGD, max_iter=10,
    #         kernel="gaussian",
    #         bw_SVGD_method="scaled_with_n", bw_SVGD_const=1.0
    #    )
    #    is_svgd = True
    #    time_intervals = 74
    #    intervals_to_double = 8.0
    #    short_initial_steps = True
    #    ratio_of_step_numbers = 8
    #    ratio_of_steps = step_SVGD / step_RW
    #    save_data = True
    #    save_plots = False
    #    run_name = "run_gaussian_iid_"

    #    # ---------------------------------------------------------------------
    #    # Experiment 5. banana, 2D, 256 samples
    #    # ---------------------------------------------------------------------
    # step_RW = 0.005
    # step_SVGD = 0.1
    # rw_config = RWRunConfig(
    #     banana,
    #     shifted_round_gaussian_init_data,
    #     random_seed=1,
    #     is_kdrw=True, is_rrw=True,
    #     dim=2, n_samples=256,
    #     step=step_RW, max_iter=10,
    #     kernel="gaussian",
    #     bw_kdrw_method="adaptive", bw_kdrw_const=1.0,
    #     bw_rrw_method="adaptive", bw_rrw_const=1.0,
    #     eps_kdrw=0.02/256, L=5.0, M=10
    # )
    # svgd_config = SVGDRunConfig(
    #     rw_config.target, rw_config.initial_data,
    #     random_seed=rw_config.random_seed,
    #     dim=rw_config.dim, n_samples=rw_config.n_samples,
    #     step=step_SVGD, max_iter=10,
    #     kernel="gaussian",
    #     bw_SVGD_method="median", bw_SVGD_const=1.0
    # )
    # is_svgd = True
    # time_intervals = 114
    # intervals_to_double = 8.0
    # short_initial_steps = True
    # ratio_of_step_numbers = 20
    # ratio_of_steps = step_SVGD / step_RW
    # save_data = True
    # save_plots = False
    # run_name = "run_banana_adaptive_med_"

    #    # ---------------------------------------------------------------------
    #    # Experiment 6. banana, 32D, 256 samples
    #    # ---------------------------------------------------------------------

    #    step_RW = 0.1
    #    step_SVGD = 0.2
    #    rw_config = RWRunConfig(
    #         banana,
    #         shifted_round_gaussian_init_data,
    #         random_seed=1,
    #         is_kdrw=True, is_rrw=True,
    #         dim=32, n_samples=256,
    #         step=step_RW, max_iter=10,
    #         kernel="gaussian",
    #         bw_kdrw_method="adaptive", bw_kdrw_const=1.0,
    #         bw_rrw_method="adaptive", bw_rrw_const=1.0,
    #         eps_kdrw=0.02/256, L=5.0, M=10
    #    )
    #    svgd_config = SVGDRunConfig(
    #         rw_config.target, rw_config.initial_data,
    #         random_seed=rw_config.random_seed,
    #         dim=rw_config.dim, n_samples=rw_config.n_samples,
    #         step=step_SVGD, max_iter=10,
    #         kernel="gaussian",
    #         bw_SVGD_method="median", bw_SVGD_const=1.0
    #    )
    #    is_svgd = True
    #    time_intervals = 90
    #    intervals_to_double = 8.0
    #    short_initial_steps = True
    #    ratio_of_step_numbers = 2
    #    ratio_of_steps = step_SVGD / step_RW
    #    save_data = True
    #    save_plots = False
    #    run_name = "run_banana_adaptive_median_"

    #    # ---------------------------------------------------------------------
    #    # Experiment 7. banana, 256D, 256 samples
    #    # ---------------------------------------------------------------------

    #    step_RW = 0.2
    #    step_SVGD = 0.2
    #    rw_config = RWRunConfig(
    #         banana,
    #         shifted_round_gaussian_init_data,
    #         random_seed=1,
    #         is_kdrw=True, is_rrw=True,
    #         dim=256, n_samples=256,
    #         step=step_RW, max_iter=10,
    #         kernel="gaussian",
    #         bw_kdrw_method="adaptive", bw_kdrw_const=1.0,
    #         bw_rrw_method="adaptive", bw_rrw_const=1.0,
    #         eps_kdrw=0.02/256, L=5.0, M=10
    #    )
    #    svgd_config = SVGDRunConfig(
    #         rw_config.target, rw_config.initial_data,
    #         random_seed=rw_config.random_seed,
    #         dim=rw_config.dim, n_samples=rw_config.n_samples,
    #         step=step_SVGD, max_iter=10,
    #         kernel="gaussian",
    #         bw_SVGD_method="median", bw_SVGD_const=1.0
    #    )
    #    is_svgd = True
    #    time_intervals = 90
    #    intervals_to_double = 8.0
    #    short_initial_steps = True
    #    ratio_of_step_numbers = 1
    #    ratio_of_steps = step_SVGD / step_RW
    #    save_data = True
    #    save_plots = False
    #    run_name = "run_banana_adaptive_median_"

    #    # ---------------------------------------------------------------------
    #    # Experiment 8. banana, 256D, 1024 samples
    #    # ---------------------------------------------------------------------

    #    step_RW = 0.2
    #    step_SVGD = 0.2
    #    rw_config = RWRunConfig(
    #         banana,
    #         shifted_round_gaussian_init_data,
    #         random_seed=1,
    #         is_kdrw=True, is_rrw=True,
    #         dim=256, n_samples=1024,
    #         step=step_RW, max_iter=10,
    #         kernel="gaussian",
    #         bw_kdrw_method="adaptive", bw_kdrw_const=1.0,
    #         bw_rrw_method="adaptive", bw_rrw_const=1.0,
    #         eps_kdrw=0.02/1024, L=5.0, M=10
    #    )
    #    svgd_config = SVGDRunConfig(
    #    rw_config.target, rw_config.initial_data,
    #    random_seed=rw_config.random_seed,
    #    dim=rw_config.dim, n_samples=rw_config.n_samples,
    #    step=step_SVGD, max_iter=10,
    #    kernel="gaussian",
    #    bw_SVGD_method="median", bw_SVGD_const=1.0
    #    )
    #    is_svgd = True
    #    time_intervals = 90
    #    intervals_to_double = 8.0
    #    short_initial_steps = True
    #    ratio_of_step_numbers = 1
    #    ratio_of_steps = step_SVGD / step_RW
    #    save_data = True
    #    save_plots = False
    #    run_name = "run_banana_adaptive_median_"

    #    # ---------------------------------------------------------------------
    #    # Experiment 9. banana, 2048D, 256 samples
    #    # ---------------------------------------------------------------------
    #    """step_RW = 1.0 would be fine, but since conrun does not allow for non-integer ratio of steps lengths, we use 0.1 here"""
    #    step_RW = 0.1
    #    step_SVGD = 0.1
    #    rw_config = RWRunConfig(
    #         banana,
    #         shifted_round_gaussian_init_data,
    #         random_seed=1,
    #         is_kdrw=True, is_rrw=True,
    #         dim=2048, n_samples=256,
    #         step=step_RW, max_iter=10,
    #         kernel="gaussian",
    #         bw_kdrw_method="adaptive", bw_kdrw_const=1.0,
    #         bw_rrw_method="adaptive", bw_rrw_const=1.0,
    #         eps_kdrw=0.02/256, L=5.0, M=10
    #    )
    #    svgd_config = SVGDRunConfig(
    #         rw_config.target, rw_config.initial_data,
    #         random_seed=rw_config.random_seed,
    #         dim=rw_config.dim, n_samples=rw_config.n_samples,
    #         step=step_SVGD, max_iter=10,
    #         kernel="gaussian",
    #         bw_SVGD_method="median", bw_SVGD_const=1.0
    #    )
    #    is_svgd = True
    #    time_intervals = 94
    #    intervals_to_double = 8.0
    #    short_initial_steps = True
    #    ratio_of_step_numbers = 1
    #    ratio_of_steps = step_SVGD / step_RW
    #    save_data = True
    #    save_plots = False
    #    run_name = "run_banana_adaptive_median_"

    #    # ---------------------------------------------------------------------
    #    # Experiment 10. banana, 2048D, 2048 samples
    #    # ---------------------------------------------------------------------

    # step_RW = 1.0 would be fine, but since conrun does not allow for non-integer ratio of steps lengths, we use 0.1 here
    step_RW = 0.2
    step_SVGD = 0.2
    rw_config = RWRunConfig(
        banana,
        shifted_round_gaussian_init_data,
        random_seed=1,
        is_kdrw=True, is_rrw=True,
        dim=2048, n_samples=2048,
        step=step_RW, max_iter=10,
        kernel="gaussian",
        bw_kdrw_method="adaptive", bw_kdrw_const=1.0,
        bw_rrw_method="adaptive", bw_rrw_const=1.0,
        eps_kdrw=0.02/2048, L=5.0, M=10
    )
    svgd_config = SVGDRunConfig(
    rw_config.target, rw_config.initial_data,
    random_seed=rw_config.random_seed,
    dim=rw_config.dim, n_samples=rw_config.n_samples,
    step=step_SVGD, max_iter=10,
    kernel="gaussian",
    bw_SVGD_method="median", bw_SVGD_const=1.0
    )
    is_svgd = True
    time_intervals = 94
    intervals_to_double = 8.0
    short_initial_steps = True
    ratio_of_step_numbers = 1
    ratio_of_steps = step_SVGD / step_RW
    save_data = True
    save_plots = False
    run_name = "run_banana_adaptive_median_"

    rate_of_convergence_run(
        rw_config,
        svgd_config,
        is_svgd=is_svgd,
        time_intervals=time_intervals,
        intervals_to_double=intervals_to_double,
        short_initial_steps=short_initial_steps,
        ratio_of_step_numbers=ratio_of_step_numbers,
        ratio_of_steps=ratio_of_steps,
        save_data=save_data,
        run_name=run_name,
        run_device=run_device,
    )
