import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Sequence

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts._paths import runs_root

DEFAULT_DPI = 600


def _load_json(path: Path) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def _discover_run_dirs(root: Path) -> Sequence[Path]:
    """Discover directories"""
    
    if not root.exists():
        return []
    run_dirs = []
    for params_path in root.glob("*/*/n_*/params.json"):
        run_dirs.append(params_path.parent)
    return sorted(run_dirs)


def _plot_one_run(run_dir: Path, dpi: int = DEFAULT_DPI) -> None:
    params_path = run_dir / "params.json"
    if not params_path.exists():
        return

    params = _load_json(params_path)
    # Prefer explicit target name from metadata; fall back to folder name.
    target = str(params.get("target", run_dir.parent.name))
    dim = int(params.get("dim"))
    n_samples = int(params.get("n_samples"))
    con = params.get("convergence", {})

    ratio_of_steps = float(con.get("ratio_of_steps", 1.0))
    ratio_of_step_numbers = float(con.get("ratio_of_step_numbers", 1.0))
    time_ratio = (
        ratio_of_steps / ratio_of_step_numbers if ratio_of_step_numbers != 0 else 1.0
    )

    times = np.load(run_dir / "times.npy")
    mmd_iid = float(np.load(run_dir / "mmd_iid.npy"))

    # Method curves (optional)
    mmd_kdrw = (
        np.load(run_dir / "mmd_KDRW_fft.npy")
        if (run_dir / "mmd_KDRW_fft.npy").exists()
        else None
    )
    mmd_rrw = (
        np.load(run_dir / "mmd_RRW_fft.npy")
        if (run_dir / "mmd_RRW_fft.npy").exists()
        else None
    )
    mmd_svgd = (
        np.load(run_dir / "mmd_SVGD.npy")
        if (run_dir / "mmd_SVGD.npy").exists()
        else None
    )

    # MMD figure
    plt.figure()
    plt.axhline(y=mmd_iid, linestyle="--", color="black", linewidth=1, label="IID")
    if mmd_svgd is not None:
        plt.plot(time_ratio * times, mmd_svgd, color="blue", label="SVGD")
    if mmd_kdrw is not None:
        plt.plot(times, mmd_kdrw, color="green", label="KDRW_fft")
    if mmd_rrw is not None:
        plt.plot(times, mmd_rrw, color="crimson", label="RRW_fft")

    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel("time")
    plt.ylabel(r"MMD$^2$")
    plt.legend()
    plt.tight_layout()

    out_mmd = run_dir / f"convergence_{target}_d{dim}_n{n_samples}.pdf"
    plt.savefig(out_mmd, dpi=DEFAULT_DPI)
    plt.close()
    print(f"Generated: {out_mmd}")

    # Sample scatter
    if dim == 2:
        x_svgd = (
            np.load(run_dir / "final_SVGD.npy")
            if (run_dir / "final_SVGD.npy").exists()
            else None
        )
        x_kdrw = (
            np.load(run_dir / "final_KDRW_fft.npy")
            if (run_dir / "final_KDRW_fft.npy").exists()
            else None
        )
        x_rrw = (
            np.load(run_dir / "final_RRW_fft.npy")
            if (run_dir / "final_RRW_fft.npy").exists()
            else None
        )

        plt.figure(figsize=(12.8, 9.6))
        ax = plt.gca()
        plt.xlim(-4.5, 4.5)
        plt.ylim(-4.5, 4.5)

        if x_svgd is not None:
            ax.scatter(x_svgd[:, 0], x_svgd[:, 1], s=6, label="SVGD")
        if x_kdrw is not None:
            ax.scatter(x_kdrw[:, 0], x_kdrw[:, 1], s=6, label="KDRW_fft")
        if x_rrw is not None:
            ax.scatter(x_rrw[:, 0], x_rrw[:, 1], s=6, label="RRW_fft")

        ax.legend()
        plt.tight_layout()
        out_scatter = run_dir / f"convergence_samples_{target}_d{dim}_n{n_samples}.pdf"
        plt.savefig(out_scatter, dpi=DEFAULT_DPI)
        plt.close()
        print(f"Generated: {out_scatter}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run_dir",
        type=str,
        default="",
        help="Optional path to a single run directory (scripts/convergence/<target>/dim_<d>/n_<n_samples>/).",
    )
    ap.add_argument(
        "--root",
        type=str,
        default=str(runs_root() / "convergence"),
        help="Root folder under which run directories are searched.",
    )
    ap.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_DPI,
        help="Figure DPI (default matches original script output).",
    )
    args = ap.parse_args()

    if args.run_dir:
        _plot_one_run(Path(args.run_dir), dpi=args.dpi)
        return

    root = Path(args.root)
    run_dirs = _discover_run_dirs(root)
    for d in run_dirs:
        _plot_one_run(d, dpi=args.dpi)


if __name__ == "__main__":
    main()
