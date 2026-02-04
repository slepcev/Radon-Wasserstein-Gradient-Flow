import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts._paths import runs_root


def format_base2(x, pos):
    exponent = np.log2(x)
    if np.isclose(exponent, np.round(exponent)):
        return f"$2^{{{int(np.round(exponent))}}}$"
    return f"{x:.2g}"


def load_dim(dim_root: Path):
    """Load curves for one dimension"""
    if not dim_root.exists():
        return None

    # Discover folders
    n_dirs = []
    for p in dim_root.iterdir():
        if p.is_dir() and p.name.startswith("n_"):
            try:
                n_val = int(p.name.split("_", 1)[1])
                n_dirs.append((n_val, p))
            except Exception:
                pass
    n_dirs.sort(key=lambda t: t[0])
    if len(n_dirs) == 0:
        return None

    n_vals = np.array([n for n, _ in n_dirs], dtype=int)

    methods = ["iid", "SVGD", "KDRW_fft", "RRW_fft"]
    mmd = {m: np.full(len(n_vals), np.nan) for m in methods}
    meanerr = {m: np.full(len(n_vals), np.nan) for m in methods}

    # Filenames
    mmd_file = {
        "iid": "mmd_iid.npy",
        "SVGD": "mmd_SVGD.npy",
        "KDRW_fft": "mmd_KDRW_fft.npy",
        "RRW_fft": "mmd_RRW_fft.npy",
    }
    x_file = {
        "iid": "x_iid.npy",
        "SVGD": "final_SVGD.npy",
        "KDRW_fft": "final_KDRW_fft.npy",
        "RRW_fft": "final_RRW_fft.npy",
    }

    for k, (_n, n_dir) in enumerate(n_dirs):
        trial_dirs = sorted(
            [p for p in n_dir.iterdir() if p.is_dir() and p.name.startswith("trial_")]
        )
        if len(trial_dirs) == 0:
            continue

        for method in methods:
            mmd_trials = []
            meanerr_trials = []

            for tdir in trial_dirs:
                mp = tdir / mmd_file[method]
                xp = tdir / x_file[method]

                if mp.exists():
                    mmd_trials.append(float(np.load(mp)))

                if xp.exists():
                    x = np.load(xp)
                    meanerr_trials.append(float(np.mean(np.abs(x.mean(axis=0)))))

            if len(mmd_trials) > 0:
                mmd[method][k] = float(np.mean(mmd_trials))
            if len(meanerr_trials) > 0:
                meanerr[method][k] = float(np.mean(meanerr_trials))

    return n_vals, mmd, meanerr


def main(root_dir=None, target="gaussian", save=True, dpi=300):
    if root_dir is None:
        root_dir = str(runs_root() / "quantization")
    root = Path(root_dir) / target
    if not root.exists():
        print(f"Error: target directory not found: {root}")
        return

    dim_dirs = []
    for p in root.iterdir():
        if p.is_dir() and p.name.startswith("dim_"):
            try:
                d = int(p.name.split("_", 1)[1])
                dim_dirs.append((d, p))
            except Exception:
                pass
    dim_dirs.sort(key=lambda t: t[0])

    if len(dim_dirs) == 0:
        print(f"Error: no dim_* directories found under {root}")
        return

    styles = {
        "iid": dict(color="gray", marker="o", label="IID"),
        "SVGD": dict(color="blue", marker="s", label="SVGD"),
        "KDRW_fft": dict(color="green", marker="d", label="KDRW_fft"),
        "RRW_fft": dict(color="crimson", marker="^", label="RRW_fft"),
    }
    methods = ["iid", "SVGD", "KDRW_fft", "RRW_fft"]

    for dim, dim_root in dim_dirs:
        res = load_dim(dim_root)
        if res is None:
            print(f"Skipping d={dim}: no data under {dim_root}")
            continue

        n_vals, mmd, meanerr = res

        out_dir = dim_root / "figures"
        out_dir.mkdir(parents=True, exist_ok=True)

        # MMD^2 figure
        fig, ax = plt.subplots(figsize=(8, 6))
        for method in methods:
            ax.plot(n_vals, mmd[method], linewidth=1.5, **styles[method])

        ax.set_xscale("log", base=2)
        ax.set_yscale("log")

        ax.set_xticks(n_vals)
        ax.xaxis.set_major_formatter(FuncFormatter(format_base2))

        ax.set_xlabel("$n$", fontsize=12)
        ax.set_ylabel("MMD$^2$", fontsize=12)

        ax.grid(False, which="both")
        ax.grid(True, which="major", linestyle=":", alpha=0.4)
        ax.grid(False, which="minor")

        ax.legend(loc="best", fontsize=9)

        if save:
            fig.savefig(
                out_dir / f"quantization_dim_{dim}.png", dpi=dpi, bbox_inches="tight"
            )
            fig.savefig(out_dir / f"quantization_dim_{dim}.pdf", bbox_inches="tight")
        plt.close(fig)

        # Mean Error figure
        fig, ax = plt.subplots(figsize=(8, 6))
        for method in methods:
            ax.plot(n_vals, meanerr[method], linewidth=1.5, **styles[method])

        ax.set_xscale("log", base=2)
        ax.set_yscale("log")

        ax.set_xticks(n_vals)
        ax.xaxis.set_major_formatter(FuncFormatter(format_base2))

        ax.set_xlabel("$n$", fontsize=12)
        ax.set_ylabel("Mean Error", fontsize=12)

        ax.grid(False, which="both")
        ax.grid(True, which="major", linestyle=":", alpha=0.4)
        ax.grid(False, which="minor")

        ax.legend(loc="best", fontsize=9)

        if save:
            fig.savefig(
                out_dir / f"mean_mean_err_dim_{dim}.png", dpi=dpi, bbox_inches="tight"
            )
            fig.savefig(out_dir / f"mean_mean_err_dim_{dim}.pdf", bbox_inches="tight")
        plt.close(fig)

        if save:
            print(f"Generated: {out_dir / f'quantization_dim_{dim}.png'} and .pdf")
            print(f"Generated: {out_dir / f'mean_mean_err_dim_{dim}.png'} and .pdf")


def _parse_args():

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=str(runs_root() / "quantization"))
    ap.add_argument("--target", type=str, default="gaussian")
    ap.add_argument(
        "--no_save", action="store_true", help="Compute plots but do not write files."
    )
    ap.add_argument("--dpi", type=int, default=300)
    return ap.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(root_dir=args.root, target=args.target, save=(not args.no_save), dpi=args.dpi)
