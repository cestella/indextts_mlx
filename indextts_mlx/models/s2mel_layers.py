"""
MLX Custom Layers for s2mel

Custom layer implementations needed for DiT that aren't in MLX standard library.
"""

import mlx.core as mx
import mlx.nn as nn
import math
from typing import Optional


class Mish(nn.Module):
    """Mish activation function.

    Mish(x) = x * tanh(softplus(x))
    """

    def __call__(self, x: mx.array) -> mx.array:
        """Apply Mish activation.

        Args:
            x: Input array

        Returns:
            Activated array
        """
        return x * mx.tanh(nn.softplus(x))


class AdaptiveLayerNorm(nn.Module):
    """Adaptive Layer Normalization (AdaLN).

    LayerNorm with learned scale/shift modulation from conditioning.
    Used in DiT for timestep conditioning.
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-6):
        """Initialize AdaLN.

        Args:
            normalized_shape: Size of the normalized dimension
            eps: Epsilon for numerical stability
        """
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps

        # Standard LayerNorm without affine parameters
        self.norm = nn.LayerNorm(normalized_shape, affine=False, eps=eps)

        # Projection for modulation parameters
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(normalized_shape, 2 * normalized_shape)
        )

    def __call__(
        self,
        x: mx.array,
        conditioning: Optional[mx.array] = None
    ) -> mx.array:
        """Apply adaptive layer normalization.

        Args:
            x: Input array of shape (batch, seq_len, features)
            conditioning: Conditioning array of shape (batch, features)
                         If None, applies standard LayerNorm

        Returns:
            Normalized and modulated array
        """
        # Standard LayerNorm
        x_norm = self.norm(x)

        if conditioning is None:
            return x_norm

        # Get modulation parameters
        modulation = self.modulation(conditioning)  # (batch, 2*features)
        scale, shift = mx.split(modulation, 2, axis=-1)  # Each (batch, features)

        # Apply modulation: x_norm * (1 + scale) + shift
        # Expand scale and shift to match x_norm shape
        scale = mx.expand_dims(scale, 1)  # (batch, 1, features)
        shift = mx.expand_dims(shift, 1)  # (batch, 1, features)

        return x_norm * (1 + scale) + shift


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations.

    Uses sinusoidal embeddings followed by MLP, similar to Transformers
    positional encodings.
    """

    def __init__(
        self,
        hidden_size: int,
        frequency_embedding_size: int = 256,
        max_period: int = 10000,
        scale: float = 1000.0
    ):
        """Initialize timestep embedder.

        Args:
            hidden_size: Output dimension
            frequency_embedding_size: Dimension of sinusoidal embeddings
            max_period: Maximum period for sinusoidal embeddings
            scale: Scale factor for timesteps
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.frequency_embedding_size = frequency_embedding_size
        self.max_period = max_period
        self.scale = scale

        # MLP to project embeddings to hidden size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )

        # Precompute frequency bands
        half = frequency_embedding_size // 2
        freqs = mx.exp(
            -math.log(max_period) * mx.arange(0, half, dtype=mx.float32) / half
        )
        self.freqs = freqs

    def timestep_embedding(self, t: mx.array) -> mx.array:
        """Create sinusoidal timestep embeddings.

        Args:
            t: Timesteps of shape (batch,)

        Returns:
            Embeddings of shape (batch, frequency_embedding_size)
        """
        # Scale timesteps
        t_scaled = self.scale * t  # (batch,)

        # Compute sinusoidal embeddings
        args = mx.expand_dims(t_scaled, -1) * mx.expand_dims(self.freqs, 0)
        embedding = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)

        # Handle odd frequency_embedding_size
        if self.frequency_embedding_size % 2:
            embedding = mx.concatenate(
                [embedding, mx.zeros_like(embedding[:, :1])],
                axis=-1
            )

        return embedding

    def __call__(self, t: mx.array) -> mx.array:
        """Embed timesteps.

        Args:
            t: Timesteps of shape (batch,)

        Returns:
            Embeddings of shape (batch, hidden_size)
        """
        t_freq = self.timestep_embedding(t)
        t_emb = self.mlp(t_freq)
        return t_emb


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    RMSNorm normalizes using RMS instead of mean and variance like LayerNorm.
    It has a learnable scale parameter but no bias.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # Initialize weight as a parameter (not just an array)
        self.weight = mx.ones((dim,))
        # Note: MLX nn.Module automatically treats arrays as parameters when set as attributes

    def __call__(self, x: mx.array) -> mx.array:
        """Apply RMS normalization.

        Args:
            x: Input of shape (... , dim)

        Returns:
            Normalized output of same shape as input
        """
        # Compute RMS: sqrt(mean(x^2))
        rms = mx.sqrt(mx.mean(x * x, axis=-1, keepdims=True) + self.eps)

        # Normalize and scale
        return (x / rms) * self.weight


def modulate(x: mx.array, shift: mx.array, scale: mx.array) -> mx.array:
    """Apply modulation to input.

    Args:
        x: Input array of shape (batch, seq_len, features)
        shift: Shift array of shape (batch, features) or (batch, seq_len, features)
        scale: Scale array of shape (batch, features) or (batch, seq_len, features)

    Returns:
        Modulated array (scale * x + shift)
    """
    # Expand shift and scale for broadcasting if they don't have sequence dimension
    if len(shift.shape) == 2:
        shift = mx.expand_dims(shift, 1)  # (batch, 1, features)
        scale = mx.expand_dims(scale, 1)  # (batch, 1, features)

    return scale * x + shift


def sequence_mask(length: mx.array, max_length: Optional[int] = None) -> mx.array:
    """Create sequence mask from lengths.

    Args:
        length: Tensor of sequence lengths, shape (batch,)
        max_length: Maximum sequence length. If None, uses max(length)

    Returns:
        Boolean mask of shape (batch, max_length)
    """
    if max_length is None:
        max_length = int(mx.max(length).item())

    # Create range tensor
    batch_size = length.shape[0]
    range_tensor = mx.arange(max_length)  # (max_length,)
    range_tensor = mx.expand_dims(range_tensor, 0)  # (1, max_length)
    range_tensor = mx.broadcast_to(range_tensor, (batch_size, max_length))

    # Create mask
    length_expanded = mx.expand_dims(length, 1)  # (batch, 1)
    mask = range_tensor < length_expanded  # (batch, max_length)

    return mask


class FinalLayer(nn.Module):
    """Final layer of DiT with AdaLN modulation."""

    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        """Initialize final layer.

        Args:
            hidden_size: Hidden dimension
            patch_size: Patch size (usually 1 for s2mel)
            out_channels: Output channels (80 for mel)
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.out_channels = out_channels

        self.norm_final = nn.LayerNorm(hidden_size, affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size)
        )

    def __call__(self, x: mx.array, c: mx.array) -> mx.array:
        """Apply final layer.

        Args:
            x: Input array of shape (batch, seq_len, hidden_size)
            c: Conditioning of shape (batch, hidden_size)

        Returns:
            Output array of shape (batch, seq_len, out_channels)
        """
        # Get modulation parameters
        modulation = self.adaLN_modulation(c)  # (batch, 2*hidden_size)
        shift, scale = mx.split(modulation, 2, axis=-1)

        # Apply modulated normalization
        x_norm = self.norm_final(x)
        x_mod = modulate(x_norm, shift, scale)

        # Project to output
        x_out = self.linear(x_mod)

        return x_out


def precompute_freqs_cis(
    seq_len: int,
    n_elem: int,
    base: int = 10000
) -> mx.array:
    """Precompute frequency tensor for rotary positional embeddings.

    Args:
        seq_len: Maximum sequence length
        n_elem: Dimension of embeddings (head_dim)
        base: Base for frequency computation

    Returns:
        Frequency tensor of shape (seq_len, n_elem // 2, 2) containing [cos, sin] pairs
    """
    # Compute frequencies
    freqs = 1.0 / (base ** (mx.arange(0, n_elem, 2, dtype=mx.float32) / n_elem))
    t = mx.arange(seq_len, dtype=mx.float32)
    freqs = mx.outer(t, freqs)  # (seq_len, n_elem // 2)

    # Compute cos and sin
    freqs_cos = mx.cos(freqs)
    freqs_sin = mx.sin(freqs)

    # Stack as (seq_len, n_elem // 2, 2)
    cache = mx.stack([freqs_cos, freqs_sin], axis=-1)

    return cache


def apply_rotary_emb(x: mx.array, freqs_cis: mx.array) -> mx.array:
    """Apply rotary positional embeddings to input tensor.

    Args:
        x: Input tensor of shape (batch, seq_len, n_heads, head_dim)
        freqs_cis: Frequency tensor of shape (seq_len, head_dim // 2, 2)

    Returns:
        Tensor with rotary embeddings applied, same shape as input
    """
    # Reshape x to separate real/imag components
    # (batch, seq_len, n_heads, head_dim) -> (batch, seq_len, n_heads, head_dim // 2, 2)
    xshaped = x.reshape(*x.shape[:-1], -1, 2)

    # Reshape freqs_cis for broadcasting
    # (seq_len, head_dim // 2, 2) -> (1, seq_len, 1, head_dim // 2, 2)
    freqs_cis = mx.expand_dims(freqs_cis, axis=[0, 2])

    # Apply rotation:
    # real' = real * cos - imag * sin
    # imag' = imag * cos + real * sin
    x_out = mx.stack([
        xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
        xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
    ], axis=-1)

    # Flatten back to original shape
    x_out = x_out.reshape(*x.shape)

    return x_out


def reflect_pad1d(x: mx.array, padding: tuple[int, int]) -> mx.array:
    """Apply reflect padding to 1D input.

    Args:
        x: Input tensor of shape (batch, seq_len, channels) - channels-last format
        padding: Tuple of (left_pad, right_pad)

    Returns:
        Padded tensor of shape (batch, seq_len + left_pad + right_pad, channels)
    """
    pad_left, pad_right = padding

    if pad_left == 0 and pad_right == 0:
        return x

    # Extract boundary values for reflection
    # For reflect mode, we mirror the interior elements (not including the boundary itself)
    # Example: [a, b, c, d] with pad_left=2 -> [c, b, a, b, c, d]
    parts = [x]

    if pad_left > 0:
        # Reflect left: take elements [1:pad_left+1] and reverse
        left_reflect = x[:, 1:pad_left+1, :]  # (batch, pad_left, channels)
        # Reverse using array indexing: [:, ::-1, :]
        left_reflect = left_reflect[:, ::-1, :]
        parts.insert(0, left_reflect)

    if pad_right > 0:
        # Reflect right: take elements [-pad_right-1:-1] and reverse
        right_reflect = x[:, -pad_right-1:-1, :]  # (batch, pad_right, channels)
        # Reverse using array indexing: [:, ::-1, :]
        right_reflect = right_reflect[:, ::-1, :]
        parts.append(right_reflect)

    return mx.concatenate(parts, axis=1)
