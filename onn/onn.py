"""A small photonic neural network: an N x N Clements mesh of MZIs, trained by gradient descent on its phases.

Each MZI = splitter, phase shifter (theta, between the splitters), splitter, phase shifter (phi, on one output).
A splitter keeps a fraction t of the power (insertion loss) and sends a fraction eta to the cross port.
Inputs are encoded as light amplitudes; outputs are read by photodetectors (power |E|^2) with additive detector noise.
Loss that is the same on every path only dims the light, so at a fixed laser power it shows up as lower
signal-to-noise; splitting-ratio and phase errors change the matrix itself.
"""
import numpy as np
import torch

torch.set_default_dtype(torch.float64)


def splitter(t=1.0, eta=0.5):
    """2x2 field transfer matrix of a lossy directional splitter."""
    a, b = np.sqrt(t * (1 - eta)), np.sqrt(t * eta)
    return torch.tensor([[a, 1j * b], [1j * b, a]], dtype=torch.complex128)


class Mesh(torch.nn.Module):
    """Clements rectangular mesh: N columns of MZIs on alternating port pairs (0-1, 2-3, ...) and (1-2, 3-4, ...)."""

    def __init__(self, n, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.n = n
        self.pairs = [[(i, i + 1) for i in range(c % 2, n - 1, 2)] for c in range(n)]
        m = sum(len(p) for p in self.pairs)
        self.theta = torch.nn.Parameter(2 * np.pi * torch.rand(m, generator=g))
        self.phi = torch.nn.Parameter(2 * np.pi * torch.rand(m, generator=g))

    def forward(self, x, t=1.0, eta=0.5, phase_sigma=0.0, eta_sigma=0.0, rng=None):
        """x: (batch, n) complex amplitudes. Optional static fabrication errors: phase_sigma (rad), eta_sigma."""
        B = splitter(t, eta)
        k = 0
        for col in self.pairs:
            for i, j in col:
                th, ph = self.theta[k], self.phi[k]
                if phase_sigma:
                    th = th + phase_sigma * rng.standard_normal()
                    ph = ph + phase_sigma * rng.standard_normal()
                B1 = B2 = B
                if eta_sigma:
                    B1 = splitter(t, float(np.clip(eta + eta_sigma * rng.standard_normal(), 0, 1)))
                    B2 = splitter(t, float(np.clip(eta + eta_sigma * rng.standard_normal(), 0, 1)))
                P = torch.diag(torch.stack([torch.exp(1j * th), torch.ones((), dtype=torch.complex128)]))
                F = torch.diag(torch.stack([torch.exp(1j * ph), torch.ones((), dtype=torch.complex128)]))
                U = F @ B2 @ P @ B1
                v = torch.stack([x[:, i], x[:, j]], 1) @ U.T
                x = x.clone()
                x[:, i], x[:, j] = v[:, 0], v[:, 1]
                k += 1
        return x
