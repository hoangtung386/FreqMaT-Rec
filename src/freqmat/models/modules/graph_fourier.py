"""Time-aware graph Fourier basis over an item sequence."""

import torch

SECONDS_PER_DAY = 86400.0
PAD_PENALTY = 1e3


@torch.no_grad()
def time_graph_basis(
    timestamps: torch.Tensor, mask: torch.Tensor, n_modes: int, gamma: float = 0.5, pad_penalty: float = PAD_PENALTY
) -> torch.Tensor:
    """Lowest ``n_modes`` graph-Fourier modes of a path graph with time-aware edge weights.

    Edge ``i -- i+1`` has weight ``(1 + gap_days) ** -gamma``: neighbours close in time are strongly
    connected, distant ones weakly. The ring-graph DFT of BSARec is the special case of equal weights.
    Padding nodes get a large diagonal penalty so their modes sit at the *high*-frequency end and never
    mix with the real (connected) component. Eigenvectors are constants (no gradient through ``eigh``).

    Args:
        timestamps: ``(B, N)`` seconds.
        mask: ``(B, N)`` bool, True for real items.
    Returns:
        ``(B, N, n_modes)`` orthonormal columns.
    """
    timestamps, mask = timestamps.float(), mask.bool()
    batch, length = timestamps.shape
    gap_days = (timestamps[:, 1:] - timestamps[:, :-1]).clamp_min(0) / SECONDS_PER_DAY
    weight = (1.0 + gap_days).pow(-gamma) * (mask[:, 1:] & mask[:, :-1]).float()
    degree = timestamps.new_zeros(batch, length)
    degree[:, :-1] += weight
    degree[:, 1:] += weight
    laplacian = torch.diag_embed(degree + (~mask).float() * pad_penalty)
    idx = torch.arange(length - 1, device=timestamps.device)
    laplacian[:, idx, idx + 1] -= weight
    laplacian[:, idx + 1, idx] -= weight
    _, eigvecs = torch.linalg.eigh(laplacian)  # ascending eigenvalues
    return eigvecs[:, :, :n_modes]
