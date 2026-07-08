"""Core layers used by the 3D DiT sampler."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """Multi-head self-attention with an optional PyTorch SDPA fast path.

    Args:
        dim: Token embedding dimension.
        num_heads: Number of attention heads.
        qkv_bias: Whether to use bias in the QKV projection.
        qk_norm: Whether to apply per-head normalization to queries and keys.
        attn_drop: Attention dropout probability.
        proj_drop: Output projection dropout probability.
        norm_layer: Normalization layer used when ``qk_norm`` is enabled.
        use_flash_attention: Use ``scaled_dot_product_attention`` when true.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        use_flash_attention: bool = False,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.use_flash_attention = use_flash_attention

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply self-attention to a token sequence of shape ``[B, N, C]``."""
        batch, num_tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch, num_tokens, 3, self.num_heads, self.head_dim)

        if self.use_flash_attention:
            qkv = qkv.permute(0, 3, 1, 2, 4)
            q = self.q_norm(qkv[..., 0, :])
            k = self.k_norm(qkv[..., 1, :])
            v = qkv[..., 2, :]
            with torch.amp.autocast(device_type=x.device.type, dtype=torch.bfloat16):
                x = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    dropout_p=self.attn_drop.p if self.training else 0.0,
                    is_causal=False,
                ).transpose(1, 2)
            if x.dtype != torch.float32:
                x = x.to(torch.float32)
        else:
            q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
            q = self.q_norm(q) * self.scale
            k = self.k_norm(k)
            attn = (q @ k.transpose(-2, -1)).softmax(dim=-1)
            x = (self.attn_drop(attn) @ v).transpose(1, 2)

        x = x.reshape(batch, num_tokens, channels)
        return self.proj_drop(self.proj(x))


class RMSNorm(nn.Module):
    """Root-mean-square normalization over the final tensor dimension."""

    def __init__(self, normalized_shape: int, eps: float = 1e-6, elementwise_affine: bool = False):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize ``x`` by its RMS value and optional affine parameters."""
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        x = x * rms
        if self.weight is not None:
            x = x * self.weight + self.bias
        return x
