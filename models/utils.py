"""Utility functions shared by the 3D DiT model."""

import collections.abc
import math
from itertools import repeat
from typing import Optional

import torch


def _ntuple(n: int):
    """Return a parser that converts scalars or iterables into ``n``-tuples."""

    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply AdaLN-style scale and shift conditioning to token features."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def get_normalized_3d_pos_enc(
    grid_size: int,
    embed_dim: int,
    num_frequencies: Optional[int] = None,
    dtype: torch.dtype = torch.float32,
    scale_freqs: float = 1.0,
) -> torch.Tensor:
    """Create normalized sinusoidal position encodings for a cubic 3D grid.

    Args:
        grid_size: Number of tokens along each spatial axis.
        embed_dim: Output embedding dimension. Must be divisible by 6 when
            ``num_frequencies`` is not provided.
        num_frequencies: Optional number of frequency bands per axis.
        dtype: Floating point dtype for the returned tensor.
        scale_freqs: Optional frequency multiplier.

    Returns:
        Tensor of shape ``[grid_size ** 3, embed_dim]``.
    """
    if num_frequencies is None:
        if embed_dim % 6 != 0:
            raise ValueError("embed_dim must be divisible by 6 or num_frequencies must be provided")
        num_frequencies = embed_dim // 6

    coords = (torch.arange(grid_size, dtype=dtype) + 0.5) / grid_size
    zz, yy, xx = torch.meshgrid(coords, coords, coords, indexing="ij")
    pos = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)

    safety_factor = 0.95
    f_min = 1.0
    f_max = safety_factor * (grid_size / 2.0)
    t = torch.linspace(0.0, 1.0, num_frequencies, dtype=dtype)
    freqs = f_min * (f_max / f_min) ** t * 2.0 * math.pi * scale_freqs

    pos_expanded = pos.unsqueeze(-1) * freqs.unsqueeze(0)
    sin_enc = torch.sin(pos_expanded)
    cos_enc = torch.cos(pos_expanded)

    encodings = []
    for dim in range(3):
        encodings.append(sin_enc[:, dim])
        encodings.append(cos_enc[:, dim])

    pos_enc = torch.cat(encodings, dim=-1)
    if pos_enc.shape[-1] != embed_dim:
        raise ValueError(f"Encoding dimension mismatch: got {pos_enc.shape[-1]}, expected {embed_dim}")

    return pos_enc
