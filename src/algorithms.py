import math
from typing import Callable, Union

import torch
from tqdm import tqdm


def sample_hypersphere(n: int, d: int, *, device=None, dtype=None) -> torch.Tensor:
    """Sample n random unit vectors in R^d."""

    z = torch.randn(n, d, device=device, dtype=dtype)
    z = z / torch.linalg.norm(z, dim=-1, keepdim=True).clamp_min(1e-12)
    return z


@torch.no_grad()
def KDRW(
    x0: torch.Tensor,
    score,
    step: float = 1e-3,
    max_iter: int = 1000,
    bw: Union[float, Callable[[torch.Tensor], float]] = 1.0,
    eps: float = 1e-4,
    kernel: str = "Gaussian",
):
    """
    Kernel-Density Radon--Wasserstein (KDRW) gradient flow.

    Args:
        x0: Initial particles of shape (n, d).
        score: Target score function ∇log p(x).
        step: Integration step size.
        max_iter: Number of iterations.
        bw: Kernel bandwidth (float or adaptive callable).
        eps: Regularization parameter for the 1D kernel.
        kernel: Type of kernel to use ('Gaussian' or 'Laplace').
    """

    kl = kernel.lower()
    is_gauss = kl.startswith("gauss")
    is_laplace = kl.startswith("lap")

    if not (is_gauss or is_laplace):
        raise NotImplementedError("KDRW supports 'Gaussian' or 'Laplace' kernels only.")

    x = x0.detach().clone()
    device = x.device
    n, d = x.shape

    # Random directions
    thetas = sample_hypersphere(max_iter, d, device=device, dtype=x.dtype)

    pbar = tqdm(range(max_iter), desc="KDRW", mininterval=0.5)
    for it in pbar:
        theta = thetas[it]
        score_x = score(x)
        p = x @ theta  # (n,)
        s = score_x @ theta  # (n,)

        # Bandwidth
        if isinstance(bw, float):
            bw_val = bw
        else:
            bw_val = bw(p)
        if bw_val <= 0:
            raise ValueError("bw must be positive.")

        # Kernel constants
        if is_gauss:
            inv2bw = 1.0 / (2.0 * bw_val * bw_val)
            invbw = 1.0 / (bw_val * bw_val)
        else:
            invbw = 1.0 / bw_val

        d1 = p[:, None] - p[None, :]  # (n, n)

        if is_gauss:
            k = torch.exp(-(d1 * d1) * inv2bw)  # (n, n)
            k_der = (d1 * invbw) * k  # (n, n)
        else:
            ad = d1.abs()
            k = torch.exp(-ad * invbw)  # (n, n)
            k_der = d1.sign() * invbw * k  # (n, n)

        k_score = k @ s  # (n,)
        k_der = k_der.sum(dim=0)  # (n,)
        k_sum = k.sum(dim=0)  # (n,)

        v = (k_der - k_score) / (k_sum + n * eps)  # (n,)
        x -= step * torch.outer(v, theta)

    return x


@torch.no_grad()
def KDRW_fft(
    x0: torch.Tensor,
    score,
    step: float = 1e-3,
    max_iter: int = 1000,
    bw: Union[float, Callable[[torch.Tensor], float]] = 1.0,
    eps: float = 1e-4,
    L: float = 5.0,
    M: int = 8,
    kernel: str = "Gaussian",
):
    """
    FFT-accelerated KDRW.

    Args:
        x0, score, step, max_iter, bw, eps, kernel: See KDRW.
        L: Threshold for kernel truncation to |x| <= L * bw.
        M: Resolution for the 1D convolution grid.
    """

    kl = kernel.lower()
    is_gauss = kl.startswith("gauss")
    is_laplace = kl.startswith("lap")

    if not (is_gauss or is_laplace):
        raise NotImplementedError(
            "KDRW_fft supports 'Gaussian' or 'Laplace' kernels only."
        )

    x = x0.detach().clone()
    device = x.device
    n, d = x.shape

    # Random directions
    thetas = sample_hypersphere(max_iter, d, device=device, dtype=x.dtype)

    pbar = tqdm(range(max_iter), desc="KDRW_fft", mininterval=0.5)
    for it in pbar:
        theta = thetas[it]
        score_x = score(x)
        p = x @ theta  # (n,)
        s = score_x @ theta  # (n,)

        # Bandwidth
        if isinstance(bw, float):
            bw_val = bw
        else:
            bw_val = bw(p)
        if bw_val <= 0:
            raise ValueError("bw must be positive.")

        # Kernel constants
        if is_gauss:
            inv2bw = 1.0 / (2.0 * bw_val * bw_val)
            invbw = 1.0 / (bw_val * bw_val)
        else:
            invbw = 1.0 / bw_val

        # Grid span with buffer R
        p_min = p.min()
        p_max = p.max()

        R = L * bw_val

        a = (p_min - R).item()
        c = (p_max + R).item()
        W = c - a
        if not math.isfinite(W) or W <= 0.0:
            continue

        # Grid size (varies per iteration)
        h = bw_val / M
        G_raw = int(math.ceil(W / h))

        # Scalar (power of 2)
        G = max(16, 1 << (G_raw - 1).bit_length())
        h = W / G

        # Linear interpolation from particles to grid
        t = ((p - a) / h).clamp(0.0, float(G - 1.001))  # (n,)
        i0 = torch.floor(t).to(torch.long)  # (n,)
        frac = t - i0.to(t.dtype)  # (n,)
        i1 = i0 + 1  # (n,)
        w1 = 1.0 - frac  # (n,)
        w2 = frac  # (n,)

        rho = torch.zeros(G, device=device, dtype=x.dtype)  # (G,)
        rho.index_add_(0, i0, w1)
        rho.index_add_(0, i1, w2)

        rho_s = torch.zeros(G, device=device, dtype=x.dtype)  # (G,)
        rho_s.index_add_(0, i0, w1 * s)
        rho_s.index_add_(0, i1, w2 * s)

        # Centered kernel on grid (truncated to |x| <= R)
        m_idx = torch.arange(G, device=device, dtype=x.dtype)  # (G,)
        x_centered = (m_idx - G // 2) * h  # (G,)

        if is_gauss:
            k_centered = torch.exp(-(x_centered * x_centered) * inv2bw)  # (G,)
        else:
            k_centered = torch.exp(-x_centered.abs() * invbw)  # (G,)

        k_centered = torch.where(
            (x_centered.abs() <= R),
            k_centered,
            torch.zeros(1, dtype=x.dtype, device=device),
        )  # (G,)
        k_grid = torch.roll(k_centered, shifts=-G // 2)  # (G,)

        # FFTs and convolutions
        K_hat = torch.fft.fft(k_grid)  # (G,)
        rho_hat = torch.fft.fft(rho)  # (G,)
        rho_s_hat = torch.fft.fft(rho_s)  # (G,)

        phi_hat = K_hat * rho_hat  # (G,)
        phi_s_hat = K_hat * rho_s_hat  # (G,)

        phi = torch.fft.ifft(phi_hat).real  # (G,)
        phi_s = torch.fft.ifft(phi_s_hat).real  # (G,)

        # Derivative in Fourier space
        c_unit = torch.tensor(
            1j,
            device=device,
            dtype=torch.complex128 if x.dtype == torch.float64 else torch.complex64,
        )
        freqs = torch.fft.fftfreq(G, d=h, device=device)  # (G,)
        omega = 2.0 * math.pi * freqs  # (G,)
        k_der_hat = (c_unit * omega) * phi_hat  # (G,)
        k_der_grid = torch.fft.ifft(k_der_hat).real  # (G,)

        # Interpolate back to particles
        k_sum = w1 * phi[i0] + w2 * phi[i1]  # (n,)
        k_score = w1 * phi_s[i0] + w2 * phi_s[i1]  # (n,)
        k_der = w1 * k_der_grid[i0] + w2 * k_der_grid[i1]  # (n,)

        v = (k_der - k_score) / (k_sum + n * eps)  # (n,)
        x -= step * torch.outer(v, theta)

    return x


@torch.no_grad()
def RRW(
    x0: torch.Tensor,
    score,
    step: float = 1e-3,
    max_iter: int = 1000,
    bw: Union[float, Callable[[torch.Tensor], float]] = 1.0,
    eps: float = 1e-4,
    kernel: str = "Gaussian",
):
    """
    Regularized Radon--Wasserstein (RRW) gradient flow.

    Args:
        x0: Initial particles of shape (n, d).
        score: Target score function ∇log p(x).
        step: Integration step size.
        max_iter: Number of iterations.
        bw: Kernel bandwidth (float or adaptive callable).
        eps: Regularization parameter (used in denominator for stability).
        kernel: Type of kernel to use ('Gaussian' or 'Laplace').
    """

    kl = kernel.lower()
    is_gauss = kl.startswith("gauss")
    is_laplace = kl.startswith("lap")

    if not (is_gauss or is_laplace):
        raise NotImplementedError("RRW supports 'Gaussian' or 'Laplace' kernels only.")

    x = x0.detach().clone()
    device = x.device
    n, d = x.shape

    # Random directions
    thetas = sample_hypersphere(max_iter, d, device=device, dtype=x.dtype)

    pbar = tqdm(range(max_iter), desc="RRW", mininterval=0.5)
    for it in pbar:
        theta = thetas[it]
        score_x = score(x)
        p = x @ theta  # (n,)
        s = score_x @ theta  # (n,)

        # Bandwidth
        if isinstance(bw, float):
            bw_val = bw
        else:
            bw_val = bw(p)
        if bw_val <= 0:
            raise ValueError("bw must be positive.")

        # Kernel constants
        if is_gauss:
            inv2bw = 1.0 / (2.0 * bw_val * bw_val)
            invbw = 1.0 / (bw_val * bw_val)
        else:
            invbw = 1.0 / bw_val

        d1 = p[:, None] - p[None, :]  # (n, n)

        if is_gauss:
            k = torch.exp(-(d1 * d1) * inv2bw)  # (n, n)
            k_der = (d1 * invbw) * k  # (n, n)
        else:
            ad = d1.abs()
            k = torch.exp(-ad * invbw)  # (n, n)
            k_der = d1.sign() * invbw * k  # (n, n)

        k_score = k @ s  # (n,)
        k_der = k_der.sum(dim=0)  # (n,)
        k_sum = k.sum(dim=0)  # (n,)

        v1 = (k_der - k_score) / (k_sum + n * eps)  # (n,)
        v2 = (k @ v1) / k_sum  # (n,)
        x -= step * torch.outer(v2, theta)

    return x


@torch.no_grad()
def RRW_fft(
    x0: torch.Tensor,
    score,
    step: float = 1e-3,
    max_iter: int = 1000,
    bw: Union[float, Callable[[torch.Tensor], float]] = 1.0,
    eps: float = 1e-4,
    L: float = 5.0,
    M: int = 8,
    kernel: str = "Gaussian",
):
    """
    FFT-accelerated RRW.

    Args:
        x0, score, step, max_iter, bw, eps, kernel: See RRW.
        L: Threshold for kernel truncation to |x| <= L * bw.
        M: Resolution for the 1D convolution grid.
    """

    kl = kernel.lower()
    is_gauss = kl.startswith("gauss")
    is_laplace = kl.startswith("lap")

    if not (is_gauss or is_laplace):
        raise NotImplementedError(
            "RRW_fft supports 'Gaussian' or 'Laplace' kernels only."
        )

    x = x0.detach().clone()
    device = x.device
    n, d = x.shape

    # Random directions
    thetas = sample_hypersphere(max_iter, d, device=device, dtype=x.dtype)

    pbar = tqdm(range(max_iter), desc="RRW_fft", mininterval=0.5)
    for it in pbar:
        theta = thetas[it]
        score_x = score(x)
        p = x @ theta  # (n,)
        s = score_x @ theta  # (n,)

        # Bandwidth
        if isinstance(bw, float):
            bw_val = bw
        else:
            bw_val = bw(p)
        if bw_val <= 0:
            raise ValueError("bw must be positive.")

        # Kernel constants
        if is_gauss:
            inv2bw = 1.0 / (2.0 * bw_val * bw_val)
            invbw = 1.0 / (bw_val * bw_val)
        else:
            invbw = 1.0 / bw_val

        # Grid span with buffer 2R
        p_min = p.min()
        p_max = p.max()

        R = L * bw_val

        a = (p_min - 2.0 * R).item()
        c = (p_max + 2.0 * R).item()
        W = c - a
        if not math.isfinite(W) or W <= 0.0:
            continue

        # Grid size (varies per iteration)
        h = bw_val / M
        G_raw = int(math.ceil(W / h))

        # Scalar (power of 2)
        G = max(16, 1 << (G_raw - 1).bit_length())
        h = W / G

        t = ((p - a) / h).clamp(0.0, float(G - 1.001))  # (n,)
        i0 = torch.floor(t).to(torch.long)  # (n,)
        frac = t - i0.to(t.dtype)  # (n,)
        i1 = i0 + 1  # (n,)
        w1 = 1.0 - frac  # (n,)
        w2 = frac  # (n,)

        rho = torch.zeros(G, device=device, dtype=x.dtype)  # (G,)
        rho.index_add_(0, i0, w1)
        rho.index_add_(0, i1, w2)

        rho_s = torch.zeros(G, device=device, dtype=x.dtype)  # (G,)
        rho_s.index_add_(0, i0, w1 * s)
        rho_s.index_add_(0, i1, w2 * s)

        # Centered kernel on grid (truncated to |x| <= R)
        m_idx = torch.arange(G, device=device, dtype=x.dtype)  # (G,)
        x_centered = (m_idx - G // 2) * h  # (G,)

        if is_gauss:
            k_centered = torch.exp(-(x_centered * x_centered) * inv2bw)  # (G,)
        else:
            k_centered = torch.exp(-x_centered.abs() * invbw)  # (G,)

        k_centered = torch.where(
            (x_centered.abs() <= R),
            k_centered,
            torch.zeros(1, dtype=x.dtype, device=device),
        )  # (G,)
        k_mass = k_centered.sum().clamp_min(1e-12)
        k_grid = torch.roll(k_centered, shifts=-G // 2)  # (G,)

        # FFTs
        K_hat = torch.fft.fft(k_grid)  # (G,)
        rho_hat = torch.fft.fft(rho)  # (G,)
        rho_s_hat = torch.fft.fft(rho_s)  # (G,)

        # Stage 1: compute v1
        phi_hat = K_hat * rho_hat  # (G,)
        phi_s_hat = K_hat * rho_s_hat  # (G,)

        phi = torch.fft.ifft(phi_hat).real  # (G,)
        phi_s = torch.fft.ifft(phi_s_hat).real  # (G,)

        freqs = torch.fft.fftfreq(G, d=h, device=device)  # (G,)
        omega = 2.0 * math.pi * freqs  # (G,)

        c_unit = torch.tensor(
            1j,
            device=device,
            dtype=torch.complex128 if x.dtype == torch.float64 else torch.complex64,
        )
        k_der_hat = (c_unit * omega) * phi_hat  # (G,)
        k_der_grid = torch.fft.ifft(k_der_hat).real  # (G,)

        v1_grid = (k_der_grid - phi_s) / (phi + n * eps)  # (G,)

        # Stage 2: compute v2=k*v1
        v1_hat = torch.fft.fft(v1_grid)  # (G,)
        conv_v1 = torch.fft.ifft(K_hat * v1_hat).real  # (G,)
        v2_grid = conv_v1 / k_mass  # (G,)

        # Interpolate back to particles
        v2 = w1 * v2_grid[i0] + w2 * v2_grid[i1]  # (n,)
        x -= step * torch.outer(v2, theta)

    return x


@torch.no_grad()
def SVGD(
    x0: torch.Tensor,
    score,
    step: float = 1e-3,
    max_iter: int = 1000,
    bw: Union[float, Callable[[torch.Tensor], float]] = 1.0,
    alpha: float = 0.9,
    ada: bool = True,
):
    """
    Stein Variational Gradient Descent (Liu & Wang, 2016).

    Args:
        x0: Initial particles of shape (n, d).
        score: Target score function ∇log p(x).
        step: Integration step size (learning rate).
        max_iter: Number of iterations.
        bw: Kernel bandwidth (float or adaptive callable).
        alpha: Momentum decay factor for the AdaGrad optimizer (used when ada=True).
        ada: Boolean indicating whether to use AdaGrad (True) or standard SGD (False).
    """

    x = x0.detach().clone()
    device = x.device
    n, d = x.shape

    fudge_factor = 1e-6

    if ada:
        historical_grad = torch.zeros_like(x, device=device)  # (n, d)

    pbar = tqdm(range(max_iter), desc="SVGD", mininterval=0.5)
    for it in pbar:

        s = score(x)  # (n, d)
        if s.shape != x.shape:
            raise ValueError(f"score(x) must return shape {x.shape}, got {s.shape}")

        # Squared pairwise distances
        x2 = (x * x).sum(dim=1, keepdim=True)  # (n, 1)
        d2 = x2 + x2.T - 2.0 * (x @ x.T)  # (n, n)
        d2.clamp_min_(0.0)

        if isinstance(bw, float):
            bw_val = bw
        else:
            bw_val = bw(x)

        if bw_val < 0:
            raise ValueError("bandwidth must be nonnegative")

        h2 = bw_val * bw_val

        inv2h2 = 1.0 / (2.0 * h2)
        invh2 = 1.0 / h2

        K = torch.exp(-d2 * inv2h2)  # (n, n)
        k_score = K @ s  # (n, d)

        Ksum = K.sum(dim=1, keepdim=True)  # (n, 1)
        Kx = K @ x  # (n, d)
        grad_K = (x * Ksum - Kx) * invh2  # (n, d)

        direction = (k_score + grad_K) / n  # (n, d)

        # AdaGrad update
        if ada:
            if it == 0:
                historical_grad = historical_grad + direction**2
            else:
                historical_grad = alpha * historical_grad + (1 - alpha) * (direction**2)

            adj_direction = direction / (
                torch.sqrt(historical_grad) + fudge_factor
            )  # (n, d)
            x = x + step * adj_direction
        else:
            x = x + step * direction

    return x
