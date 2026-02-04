from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class Target(Protocol):
    """Structural protocol for target probability distributions."""

    name: str

    def U(self, x: torch.Tensor) -> torch.Tensor: ...

    def score(self, x: torch.Tensor) -> torch.Tensor: ...


@runtime_checkable
class TransformableTarget(Target, Protocol):
    """Protocol for targets that have a transport map to/from a standard Gaussian."""

    def transform_to_gaussian(self, x: torch.Tensor) -> torch.Tensor: ...

    def inverse_transform(self, y: torch.Tensor) -> torch.Tensor: ...


def is_transformable(target: Target) -> bool:
    """Return True if target implements TransformableTarget."""
    return isinstance(target, TransformableTarget)


@dataclass(frozen=True)
class BananaParams:
    """Parameters for BananaTarget."""
    c: float = 1.0
    sigma: float = 1.0
    mu: float = 0.0


@dataclass
class BananaTarget:
    """
    d-dimensional banana (Rosenbrock-type) target with potential U(x) = 0.5 * ( ||x_r||^2 + (u^2 / sigma^2) ) where:
        - x = [x_1, ..., x_d],
        - x_r = [x_1, ..., x_{d-1}],
        - u = (x_d - mu) - (c / sqrt(d)) * ( ||x_r||^2 - (d - 1) ).
    """

    params: BananaParams
    name: str = "banana"

    def U(self, x: torch.Tensor) -> torch.Tensor:

        xr, xd = x[..., :-1], x[..., -1]
        d = x.shape[-1]
        sqrt_d = torch.sqrt(torch.tensor(float(d), dtype=x.dtype, device=x.device))

        quad = xr.square().sum(dim=-1)
        u = (xd - self.params.mu) - (self.params.c / sqrt_d) * (quad - (d - 1))
        return 0.5 * (quad + (u**2) / (self.params.sigma**2))

    def score(self, x: torch.Tensor) -> torch.Tensor:

        xr, xd = x[..., :-1], x[..., -1]
        d = x.shape[-1]
        sqrt_d = torch.sqrt(torch.tensor(float(d), dtype=x.dtype, device=x.device))

        inv_sigma2 = 1.0 / (self.params.sigma**2)
        u = (xd - self.params.mu) - (self.params.c / sqrt_d) * (
            xr.square().sum(dim=-1) - (d - 1)
        )

        y = torch.empty_like(x)
        y[..., :-1] = (
            (2.0 * self.params.c * inv_sigma2 / sqrt_d) * u.unsqueeze(-1) - 1.0
        ) * xr
        y[..., -1] = -(u * inv_sigma2)
        return y


@dataclass
class TransformableBananaTarget(BananaTarget):
    """Extends BananaTarget with transport map to/from a standard Gaussian."""

    def transform_to_gaussian(self, x: torch.Tensor) -> torch.Tensor:

        xr, xd = x[..., :-1], x[..., -1]
        d = x.shape[-1]
        sqrt_d = torch.sqrt(torch.tensor(float(d), dtype=x.dtype, device=x.device))

        centered = xr.square().sum(dim=-1) - (d - 1)
        y_last = (
            xd - self.params.mu - (self.params.c / sqrt_d) * centered
        ) / self.params.sigma
        return torch.cat([xr, y_last.unsqueeze(-1)], dim=-1)

    def inverse_transform(self, y: torch.Tensor) -> torch.Tensor:

        yr, yd = y[..., :-1], y[..., -1]
        d = y.shape[-1]
        sqrt_d = torch.sqrt(torch.tensor(float(d), dtype=y.dtype, device=y.device))

        centered = yr.square().sum(dim=-1) - (d - 1)
        x_last = (
            self.params.mu
            + (self.params.c / sqrt_d) * centered
            + self.params.sigma * yd
        )
        return torch.cat([yr, x_last.unsqueeze(-1)], dim=-1)


@dataclass
class StandardGaussianTarget:
    """d-dimensional standard Gaussian: U(x) = 0.5*||x||^2"""

    name: str = "standard_gaussian"

    def U(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x.square().sum(dim=-1)

    def score(self, x: torch.Tensor) -> torch.Tensor:
        return -x


@dataclass
class GeneralGaussianParams:
    """Parameters for GeneralGaussianTarget."""
    mu: torch.Tensor
    Prec: torch.Tensor
    L: torch.Tensor  # Optional parameter used to transform to standard Gaussian. Lower triangular matrix so that Prec = L @ L.T


@dataclass
class GeneralGaussianTarget:
    """d-dimensional general Gaussian target with potential U(x) = 0.5 (x-mu)^T Prec (x-mu)"""
    
    name: str = "general_gaussian"

    def __init__(self, params: GeneralGaussianParams):
        self.params = params

        # Optional sanity checks (can comment out for speed)
        mu, Prec = params.mu, params.Prec
        assert mu.ndim == 1, f"mu must be (d,), got {mu.shape}"
        assert Prec.ndim == 2 and Prec.shape[0] == Prec.shape[1], f"Prec must be (d,d), got {Prec.shape}"
        assert Prec.shape[0] == mu.shape[0], f"Prec dim {Prec.shape[0]} must match mu dim {mu.shape[0]}"

    def U(self, x: torch.Tensor) -> torch.Tensor:
        
        mu = self.params.mu.to(x.device)
        Prec = self.params.Prec.to(x.device)
        dx = x - mu
        q = torch.einsum("...i,ij,...j->...", dx, Prec, dx)
        return 0.5 * q

    def score(self, x: torch.Tensor) -> torch.Tensor:

        mu = self.params.mu.to(x.device)
        Prec = self.params.Prec.to(x.device)
        dx = x - mu
        return -torch.einsum("ij,...j->...i", Prec, dx)

class TransformableGeneralGaussianTarget(GeneralGaussianTarget):
    """
    Adds transforms between x ~ N(mu, Prec^{-1}) and y ~ N(0, I),
    assuming Prec = L @ L.T with L lower-triangular.
    """

    def transform_to_gaussian(self, x: torch.Tensor) -> torch.Tensor:

        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.get_default_dtype())

        mu = self.params.mu.to(x.device)
        L = self.params.L.to(x.device)
        dx = x - mu
        y = dx @ L
        return y

    def inverse_transform(self, y: torch.Tensor) -> torch.Tensor:

        if not torch.is_tensor(y):
            y = torch.tensor(y, dtype=torch.get_default_dtype())

        mu = self.params.mu.to(y.device)
        L = self.params.L.to(y.device)

        v = torch.linalg.solve_triangular(
            L.transpose(-1, -2),
            y.unsqueeze(-1),
            upper=True,
            left=True
        ).squeeze(-1)

        x = mu + v
        return x
