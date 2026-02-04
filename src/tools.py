from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import torch
from scipy.spatial.distance import cdist

from src.targets import Target


@runtime_checkable
class InitialData(Protocol):
    """Protocol for initial particle distributions."""

    name: str

    def init_data(
        self, dim: int, n_samples: int, *, device=None, dtype=None
    ) -> torch.Tensor: ...


@dataclass
class StandardGaussianInitData:
    """Standard normal initialization."""

    name: str = "standard_gaussian_init_data"

    def init_data(
        self, dim: int, n_samples: int, *, device=None, dtype=None
    ) -> torch.Tensor:

        return torch.randn(n_samples, dim, device=device, dtype=dtype)


@dataclass
class RoundGaussianInitData:
    """Isotropic Gaussian initialization with optional mean shift."""

    std: float
    shift: float
    name: str = "round_gaussian_init_data"

    def init_data(
        self, dim: int, n_samples: int, *, device=None, dtype=None
    ) -> torch.Tensor:
        shift_vector = torch.zeros(dim, device=device, dtype=dtype)
        shift_vector[0] = self.shift
        return (
            torch.randn(n_samples, dim, device=device, dtype=dtype) * self.std
            + shift_vector
        )


@dataclass
class RWRunConfig:
    """Runner configuration for KDRW and RRW algorithms."""

    target: Target
    initial_data: InitialData
    random_seed: int
    is_kdrw: bool
    is_rrw: bool
    dim: int
    n_samples: int
    step: float
    max_iter: int
    bw_kdrw_method: str
    bw_kdrw_const: float
    bw_rrw_method: str
    bw_rrw_const: float
    eps_kdrw: float
    L: float
    M: int
    kernel: str

    def get_bw_kdrw(self):
        methods = {
            "constant": self.bw_kdrw_fixed,
            "adaptive": self.bw_kdrw_adaptive,
            "scaled_with_n": self.bw_kdrw_scale_with_n,
        }
        return methods[self.bw_kdrw_method]

    def get_bw_rrw(self):
        methods = {
            "constant": self.bw_rrw_fixed,
            "adaptive": self.bw_rrw_adaptive,
            "scaled_with_n": self.bw_rrw_scale_with_n,
        }
        return methods[self.bw_rrw_method]

    def bw_kdrw_fixed(self, x_theta: torch.Tensor):
        return self.bw_kdrw_const

    def bw_kdrw_scale_with_n(self, x_theta: torch.Tensor):
        # Scale as 2 * const * n^(-1/5)
        return 2 * self.bw_kdrw_const / (x_theta.shape[0] ** 0.2)

    def bw_kdrw_adaptive(self, x_theta: torch.Tensor):
        # Scale as 2 * const * std * n^(-1/5)
        return (
            2
            * self.bw_kdrw_const
            * x_theta.std(unbiased=False).item()
            / (x_theta.shape[0] ** 0.2)
        )

    def bw_rrw_fixed(self, x_theta: torch.Tensor):
        return self.bw_rrw_const

    def bw_rrw_scale_with_n(self, x_theta: torch.Tensor):
        # Scale as const * n^(-1/5)
        return self.bw_rrw_const / (x_theta.shape[0] ** 0.2)

    def bw_rrw_adaptive(self, x_theta: torch.Tensor):
        # Scale as const * std * n^(-1/5)
        return (
            self.bw_rrw_const
            * x_theta.std(unbiased=False).item()
            / (x_theta.shape[0] ** 0.2)
        )


@dataclass
class SVGDRunConfig:
    """Runner configuration for SVGD."""

    target: Target
    initial_data: InitialData
    random_seed: int
    dim: int
    n_samples: int
    step: float
    max_iter: int
    bw_SVGD_method: str
    bw_SVGD_const: float
    kernel: str

    def get_bw_SVGD(self):
        methods = {
            "constant": self.bw_SVGD_fixed,
            "scaled_with_n": self.bw_SVGD_scaled_with_n,
            "median": self.bw_SVGD_median,
            "median_trick": self.bw_SVGD_median_trick,
        }
        return methods[self.bw_SVGD_method]

    def bw_SVGD_fixed(self, x: torch.Tensor):
        return self.bw_SVGD_const

    def bw_SVGD_scaled_with_n(self, x: torch.Tensor):
        # Scale as const * sqrt(2 * dim).
        return self.bw_SVGD_const * np.sqrt(2 * x.shape[1])

    def bw_SVGD_median(self, x: torch.Tensor):
        # Scale as Median of pairwise distances

        x_norm_sq = (x * x).sum(dim=1, keepdim=True)
        dists = torch.sqrt((x_norm_sq + x_norm_sq.T - 2.0 * (x @ x.T)).clamp_min(0.0))

        triu_indices = torch.triu_indices(
            x.shape[0], x.shape[0], offset=1, device=x.device
        )

        return (
            self.bw_SVGD_const
            * torch.median(dists[triu_indices[0], triu_indices[1]]).item()
        )

    def bw_SVGD_median_trick(self, x: torch.Tensor):
        # Scale as median / sqrt(log(n))
        return self.bw_SVGD_median(x) / np.sqrt(np.log(x.shape[0]))


@torch.no_grad()
def mmd_torch(
    x,
    *,
    bw=1.0,
    var=1.0,
    device=None,
    dtype=torch.float32,
    accumulate_dtype=torch.float64,
    disable_tf32=True,
):
    """MMD^2 with GPU support."""

    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    acc_dtype = accumulate_dtype
    if device.type == "mps":
        dtype = torch.float32
        acc_dtype = torch.float32

    old_tf32 = None
    if device.type == "cuda" and disable_tf32:
        old_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False

    try:
        x = x.to(device=device, dtype=dtype)
        n, d = x.shape

        bw_t = torch.tensor(bw, device=device, dtype=dtype)
        var_t = torch.tensor(var, device=device, dtype=dtype)
        d_t = torch.tensor(d, device=device, dtype=dtype)

        sig_mmd = bw_t * d_t

        x_norm_sq = (x * x).sum(dim=1, keepdim=True)
        dists_xx = x_norm_sq + x_norm_sq.T - 2.0 * (x @ x.T)
        dists_xx.clamp_(min=0.0)

        K_xx = torch.exp(-dists_xx / (2.0 * sig_mmd))

        PP = K_xx.sum(dtype=acc_dtype) / (n * n)

        QQ = (sig_mmd / (sig_mmd + 2.0 * var_t)) ** (d_t / 2.0)
        QQ = QQ.to(dtype=acc_dtype)

        factor = (sig_mmd / (sig_mmd + var_t)) ** (d_t / 2.0)
        norm_x = (x * x).sum(dim=1)
        exps = torch.exp(-norm_x / (2.0 * (sig_mmd + var_t)))
        PQ = (factor * exps).sum(dtype=acc_dtype) / n

        mmd_val = PP + QQ - 2.0 * PQ
        return float(mmd_val.detach().cpu().item())

    finally:
        if old_tf32 is not None:
            torch.backends.cuda.matmul.allow_tf32 = old_tf32


def mmd(x, *, bw=1.0, var=1.0):
    """Simpler MMD^2 computation."""

    x = np.asarray(x, dtype=float)
    n, d = x.shape
    sig_mmd = bw * d

    dists_xx = cdist(x, x, metric="sqeuclidean")
    K_xx = np.exp(-dists_xx / (2.0 * sig_mmd))

    PP = K_xx.mean()
    QQ = (sig_mmd / (sig_mmd + 2.0 * var)) ** (d / 2.0)
    norm_x = (x**2).sum(axis=-1)

    exps = np.exp(-norm_x / (2.0 * (sig_mmd + var)))
    PQ = (((sig_mmd / (sig_mmd + var)) ** (d / 2.0)) * exps).mean()

    return PP + QQ - 2.0 * PQ
