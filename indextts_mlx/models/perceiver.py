"""
Perceiver Resampler implementation in MLX.

Used in UnifiedVoice to compress variable-length conditioning
into a fixed number of latent tokens via cross-attention.
"""

import math
from typing import Optional
import mlx.core as mx
import mlx.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int):
        super().__init__()
        self.scale = math.sqrt(dim)
        self.gamma = mx.ones(dim)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: (B, T, dim)
        Returns:
            normalized: (B, T, dim)
        """
        # Normalize to unit norm
        norm = mx.sqrt(mx.sum(x**2, axis=-1, keepdims=True) + 1e-8)
        x_norm = x / norm

        # Scale
        return x_norm * self.scale * self.gamma


class GEGLU(nn.Module):
    """Gated GLU with GELU activation."""

    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: (B, T, 2*dim) input
        Returns:
            output: (B, T, dim)
        """
        x, gate = mx.split(x, 2, axis=-1)
        return x * nn.gelu(gate)


class FeedForward(nn.Module):
    """Feed-forward network with GEGLU activation."""

    def __init__(self, dim: int, mult: float = 4.0):
        super().__init__()
        dim_inner = int(dim * mult * 2 / 3)

        self.net = [
            nn.Linear(dim, dim_inner * 2),  # *2 for GEGLU
            GEGLU(),
            nn.Linear(dim_inner, dim),
        ]

    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: (B, T, dim)
        Returns:
            output: (B, T, dim)
        """
        for layer in self.net:
            x = layer(x)
        return x


class CrossAttention(nn.Module):
    """Cross-attention layer for Perceiver.

    Queries come from latents, keys/values from context.
    """

    def __init__(
        self, dim: int, dim_head: int = 64, heads: int = 8, cross_attn_include_queries: bool = False
    ):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.cross_attn_include_queries = cross_attn_include_queries

        dim_inner = dim_head * heads

        self.to_q = nn.Linear(dim, dim_inner, bias=False)
        self.to_kv = nn.Linear(dim, dim_inner * 2, bias=False)
        self.to_out = nn.Linear(dim_inner, dim, bias=False)

    def __call__(self, x: mx.array, context: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        """
        Args:
            x: (B, N_latents, dim) queries (latents)
            context: (B, T_context, dim) keys/values (context)
            mask: (B, T_context) attention mask (1 = attend, 0 = ignore)
        Returns:
            output: (B, N_latents, dim)
        """
        B, N, _ = x.shape
        _, T, _ = context.shape

        # If cross_attn_include_queries, concat latents to context
        # This allows latents to attend to themselves as well
        if self.cross_attn_include_queries:
            context = mx.concatenate([x, context], axis=1)
            if mask is not None:
                # Extend mask for latents (always attend to latents)
                latent_mask = mx.ones((B, N), dtype=mask.dtype)
                mask = mx.concatenate([latent_mask, mask], axis=1)

        # Project to Q, K, V
        q = self.to_q(x)  # (B, N, dim_inner)
        kv = self.to_kv(context)  # (B, T, 2*dim_inner)
        k, v = mx.split(kv, 2, axis=-1)

        # Reshape for multi-head: (B, T, dim_inner) -> (B, heads, T, dim_head)
        q = q.reshape(B, N, self.heads, self.dim_head).transpose(0, 2, 1, 3)
        k = k.reshape(B, -1, self.heads, self.dim_head).transpose(0, 2, 1, 3)
        v = v.reshape(B, -1, self.heads, self.dim_head).transpose(0, 2, 1, 3)

        # Compute attention scores
        scores = mx.matmul(q, k.transpose(0, 1, 3, 2)) / math.sqrt(self.dim_head)

        # Apply mask if provided
        if mask is not None:
            # mask: (B, T') -> (B, 1, 1, T')
            mask_expanded = mask[:, None, None, :]
            scores = mx.where(mask_expanded == 0, -1e9, scores)

        # Softmax and weighted sum
        attn_weights = mx.softmax(scores, axis=-1)
        attn_output = mx.matmul(attn_weights, v)  # (B, heads, N, dim_head)

        # Reshape back: (B, heads, N, dim_head) -> (B, N, dim_inner)
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(B, N, -1)

        # Output projection
        return self.to_out(attn_output)


class PerceiverResampler(nn.Module):
    """Perceiver Resampler.

    Compresses variable-length context into fixed number of latent tokens
    using cross-attention.

    Architecture:
    - Learnable latent queries: (num_latents, dim)
    - depth layers of:
        - Cross-attention: latents attend to context
        - Feed-forward with GEGLU
    - Final RMSNorm
    """

    def __init__(
        self,
        dim: int,
        depth: int = 2,
        dim_context: Optional[int] = None,
        num_latents: int = 32,
        dim_head: int = 64,
        heads: int = 8,
        ff_mult: float = 4.0,
    ):
        super().__init__()
        dim_context = dim_context or dim

        # Project context to latent dimension if needed
        self.proj_context = nn.Linear(dim_context, dim) if dim_context != dim else None

        # Learnable latent queries
        self.latents = mx.random.normal((num_latents, dim)) * 0.02

        # Transformer layers
        self.layers = [
            (
                CrossAttention(dim, dim_head, heads, cross_attn_include_queries=True),
                FeedForward(dim, ff_mult),
            )
            for _ in range(depth)
        ]

        # Final norm
        self.norm = RMSNorm(dim)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        """
        Args:
            x: (B, T, dim_context) context
            mask: (B, T) attention mask
        Returns:
            latents: (B, num_latents, dim) compressed representation
        """
        B = x.shape[0]

        # Project context if needed
        if self.proj_context is not None:
            x = self.proj_context(x)

        # Expand latents for batch
        latents = mx.broadcast_to(
            self.latents[None, :, :], (B, self.latents.shape[0], self.latents.shape[1])
        )

        # Apply transformer layers
        for attn, ff in self.layers:
            latents = attn(latents, x, mask) + latents
            latents = ff(latents) + latents

        # Final norm
        return self.norm(latents)


def create_perceiver_resampler(
    dim: int = 1280,
    depth: int = 2,
    dim_context: int = 512,
    num_latents: int = 32,
    dim_head: int = 64,
    heads: int = 8,
    ff_mult: float = 4.0,
) -> PerceiverResampler:
    """Create a Perceiver resampler with UnifiedVoice default parameters.

    Default config:
    - Compresses 512-dim Conformer output to 1280-dim latents
    - 32 latent tokens
    - 2 layers of cross-attention + FF
    """
    return PerceiverResampler(dim, depth, dim_context, num_latents, dim_head, heads, ff_mult)
