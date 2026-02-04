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


def plot_bandwidth_experiments(root_dir=None, out_dir=None, dpi=300):
    if root_dir is None:
        root_dir = str(runs_root() / "bandwidths")
    root = Path(root_dir)

    if not root.exists():
        print(f"Error: Directory '{root_dir}' not found.")
        return

    out_dir_path = Path(out_dir) if out_dir is not None else (root / "figures")
    out_dir_path.mkdir(parents=True, exist_ok=True)

    # Style settings
    styles = {
        "KDRW": {"color": "blue", "marker": "s", "linestyle": "--", "label": "KDRW"},
        "RRW": {"color": "orange", "marker": "o", "linestyle": "--", "label": "RRW"},
        "KDRW_fft": {
            "color": "green",
            "marker": "d",
            "linestyle": "-",
            "label": "KDRW_fft",
        },
        "RRW_fft": {
            "color": "crimson",
            "marker": "^",
            "linestyle": "-",
            "label": "RRW_fft",
        },
    }

    target_dirs = sorted(
        [d for d in root.iterdir() if d.is_dir()], key=lambda p: p.name
    )

    for target_dir in target_dirs:
        target_name = target_dir.name

        dim_dirs = sorted(
            [
                d
                for d in target_dir.iterdir()
                if d.is_dir() and d.name.startswith("dim_")
            ],
            key=lambda p: p.name,
        )

        for dim_dir in dim_dirs:
            try:
                dim_val = int(dim_dir.name.split("_")[1])
            except (IndexError, ValueError):
                dim_val = dim_dir.name

            # Load bandwidths
            bw_path = dim_dir / "bandwidths.npy"
            if not bw_path.exists():
                print(f"Skipping {dim_dir}: bandwidths.npy not found")
                continue

            x_vals = np.load(bw_path)

            # Load i.i.d.
            iid_path = dim_dir / "mmd_iid_avg.npy"
            iid_val = None
            if iid_path.exists():
                iid_val = float(np.load(iid_path))

            # Plot algorithms
            fig, ax = plt.subplots(figsize=(8, 6))

            has_data = False
            for method, style in styles.items():
                data_path = dim_dir / f"mmd_{method}_avg.npy"
                if data_path.exists():
                    y_vals = np.load(data_path)

                    # Ensure dimensions match
                    min_len = min(len(x_vals), len(y_vals))
                    ax.plot(x_vals[:min_len], y_vals[:min_len], **style)
                    has_data = True

            if not has_data:
                plt.close(fig)
                continue

            # Plot baseline
            if iid_val is not None:
                ax.axhline(
                    y=iid_val,
                    color="#444444",
                    linestyle="--",
                    linewidth=1.5,
                    label="i.i.d. baseline",
                )

            # Formatting
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")

            ax.set_xticks(x_vals)
            ax.xaxis.set_major_formatter(FuncFormatter(format_base2))

            ax.set_xlabel("$b$", fontsize=12)
            ax.set_ylabel("MMD$^2$", fontsize=12)

            ax.grid(False, which="both")
            ax.grid(True, which="major", linestyle=":", alpha=0.4)
            ax.grid(False, which="minor")

            ax.legend(fontsize=10)

            # Save Figure
            stem = f"plot_{target_name}_d{dim_val}"
            fig.savefig(out_dir_path / f"{stem}.png", dpi=dpi, bbox_inches="tight")
            fig.savefig(out_dir_path / f"{stem}.pdf", bbox_inches="tight")
            print(f"Generated: {out_dir_path / (stem + '.png')} and .pdf")

            plt.close(fig)


def _parse_args():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=str(runs_root() / "bandwidths"))
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--dpi", type=int, default=300)
    return ap.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    out_dir = args.out if args.out else None
    plot_bandwidth_experiments(root_dir=args.root, out_dir=out_dir, dpi=args.dpi)
