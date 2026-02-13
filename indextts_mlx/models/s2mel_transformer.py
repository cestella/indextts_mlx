"""
MLX Transformer Implementation for DiT

Implements GPT-style transformer with AdaLN conditioning for diffusion.
"""

import mlx.core as mx
import mlx.nn as nn
import math
from typing import Optional, Tuple
from .s2mel_layers import AdaptiveLayerNorm, modulate


class TransformerBlock(nn.Module):
    """Transformer block with AdaLN conditioning.

    Implements:
    1. AdaLN-modulated self-attention
    2. AdaLN-modulated feedforward
    3. Residual connections
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        """Initialize transformer block.

        Args:
            hidden_size: Hidden dimension
            num_heads: Number of attention heads
            mlp_ratio: Ratio of MLP hidden dim to hidden_size
            dropout: Dropout probability (not used in inference)
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # AdaLN for attention
        self.norm1 = nn.LayerNorm(hidden_size, affine=False, eps=1e-6)
        self.attn_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size)
        )

        # Multi-head self-attention
        self.attn = nn.MultiHeadAttention(
            hidden_size,
            num_heads,
            bias=True
        )

        # AdaLN for feedforward
        self.norm2 = nn.LayerNorm(hidden_size, affine=False, eps=1e-6)
        self.mlp_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size)
        )

        # Feedforward MLP
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        # Round to nearest multiple of 256 (like PyTorch version)
        mlp_hidden_dim = ((mlp_hidden_dim + 255) // 256) * 256

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.SiLU(),
            nn.Linear(mlp_hidden_dim, hidden_size)
        )

    def __call__(
        self,
        x: mx.array,
        conditioning: mx.array,
        mask: Optional[mx.array] = None
    ) -> mx.array:
        """Apply transformer block.

        Args:
            x: Input array of shape (batch, seq_len, hidden_size)
            conditioning: Conditioning of shape (batch, hidden_size)
            mask: Attention mask of shape (batch, seq_len, seq_len)

        Returns:
            Output array of shape (batch, seq_len, hidden_size)
        """
        # Self-attention with AdaLN
        modulation = self.attn_modulation(conditioning)
        shift_attn, scale_attn = mx.split(modulation, 2, axis=-1)

        x_norm = self.norm1(x)
        x_mod = modulate(x_norm, shift_attn, scale_attn)

        # Apply attention
        attn_out = self.attn(x_mod, x_mod, x_mod, mask=mask)

        # Residual connection
        x = x + attn_out

        # Feedforward with AdaLN
        modulation = self.mlp_modulation(conditioning)
        shift_mlp, scale_mlp = mx.split(modulation, 2, axis=-1)

        x_norm = self.norm2(x)
        x_mod = modulate(x_norm, shift_mlp, scale_mlp)

        # Apply MLP
        mlp_out = self.mlp(x_mod)

        # Residual connection
        x = x + mlp_out

        return x


class Transformer(nn.Module):
    """GPT-style transformer with AdaLN conditioning for diffusion.

    Used in DiT for processing noisy mel spectrograms conditioned on
    timestep and semantic features.
    """

    def __init__(
        self,
        dim: int,
        n_layers: int,
        n_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        """Initialize transformer.

        Args:
            dim: Hidden dimension
            n_layers: Number of transformer blocks
            n_heads: Number of attention heads
            mlp_ratio: Ratio of MLP hidden dim to dim
            dropout: Dropout probability
        """
        super().__init__()
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads

        # Transformer blocks
        self.blocks = [
            TransformerBlock(dim, n_heads, mlp_ratio, dropout)
            for _ in range(n_layers)
        ]

    def __call__(
        self,
        x: mx.array,
        conditioning: mx.array,
        mask: Optional[mx.array] = None
    ) -> mx.array:
        """Apply transformer.

        Args:
            x: Input array of shape (batch, seq_len, dim)
            conditioning: Timestep conditioning of shape (batch, dim)
            mask: Attention mask of shape (batch, seq_len, seq_len)

        Returns:
            Output array of shape (batch, seq_len, dim)
        """
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, conditioning, mask)

        return x


def create_attention_mask(
    seq_len: int,
    is_causal: bool = False,
    padding_mask: Optional[mx.array] = None,
    num_heads: Optional[int] = None
) -> Optional[mx.array]:
    """Create attention mask.

    Args:
        seq_len: Sequence length
        is_causal: If True, create causal (autoregressive) mask
        padding_mask: Padding mask of shape (batch, seq_len)
        num_heads: Number of attention heads (for 4D mask)

    Returns:
        Attention mask of shape (batch, heads, seq_len, seq_len) or None
        Note: MLX MultiHeadAttention expects 4D mask or None
    """
    if not is_causal and padding_mask is None:
        return None

    # Start with full attention
    if is_causal:
        # Causal mask: lower triangular
        mask = mx.tril(mx.ones((seq_len, seq_len), dtype=mx.bool_))
        mask = mx.expand_dims(mask, 0)  # (1, seq_len, seq_len)
    else:
        mask = None

    if padding_mask is not None:
        # padding_mask: (batch, seq_len) - True for valid positions
        batch_size = padding_mask.shape[0]

        # Create mask (batch, seq_len, seq_len)
        # Where padding_mask[b, i] and padding_mask[b, j] are both True
        padding_mask_expanded = mx.expand_dims(padding_mask, 1)  # (batch, 1, seq_len)
        padding_mask_tiled = mx.expand_dims(padding_mask, 2)  # (batch, seq_len, 1)

        padding_attention_mask = mx.logical_and(
            padding_mask_tiled,
            padding_mask_expanded
        )

        if mask is not None:
            # Combine causal and padding masks
            mask = mx.logical_and(mask, padding_attention_mask)
        else:
            mask = padding_attention_mask

    # MLX MultiHeadAttention expects 4D mask: (batch, heads, seq, seq)
    # Or None (for full attention)
    # For now, return None since we're not using padding masks yet
    # TODO: Properly format 4D mask when needed
    return None
