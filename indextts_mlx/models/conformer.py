"""
Conformer Encoder for conditioning in UnifiedVoice (MLX).

Similar to W2V-BERT Conformer but with different dimensions:
- Input: mel spectrograms (80-100 channels)
- Hidden dim: 512 (vs 1024 in W2V-BERT)
- Layers: 6 (vs 24 in W2V-BERT)
- Conv subsampling + relative position attention
"""

import math
from typing import Optional, Tuple
import mlx.core as mx
import mlx.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding with scaling.

    PyTorch applies x * sqrt(d_model) before adding positional encoding.
    """

    def __init__(self, dim: int, max_len: int = 5000):
        super().__init__()
        self.dim = dim
        self.xscale = math.sqrt(dim)  # CRITICAL: scale input by sqrt(dim)

        # Precompute positional encodings
        pe = mx.zeros((max_len, dim))
        position = mx.arange(0, max_len).reshape(-1, 1)
        div_term = mx.exp(mx.arange(0, dim, 2) * (-math.log(10000.0) / dim))

        pe_sin = mx.sin(position * div_term)
        pe_cos = mx.cos(position * div_term)

        # Interleave sin and cos
        pe_np = mx.zeros((max_len, dim))
        pe = mx.concatenate([pe_sin[:, :, None], pe_cos[:, :, None]], axis=2).reshape(max_len, -1)

        self.pe = pe[None, :, :]  # (1, max_len, dim)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: (B, T, dim)
        Returns:
            x * sqrt(dim) + pe: (B, T, dim)
        """
        T = x.shape[1]
        # CRITICAL: scale by sqrt(dim) before adding positional encoding
        return x * self.xscale + self.pe[:, :T, :]


class Conv2dSubsampling(nn.Module):
    """Conv2d subsampling layer for Conformer.

    Reduces sequence length by factor of 4 using two conv layers.
    """

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        # Conv2dSubsampling2: ONE conv layer with stride=2
        self.conv = [
            nn.Conv2d(1, output_dim, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
        ]
        # After conv with stride=2, kernel=3, padding=0:
        # output_size = (input_size - 3) / 2 + 1 = (input_size - 1) / 2
        reduced_spatial = (input_dim - 1) // 2
        self.out_proj = nn.Linear(output_dim * reduced_spatial, output_dim)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: (B, T, input_dim) input features
        Returns:
            output: (B, T//2, output_dim)
        """
        B, T, C = x.shape

        # PyTorch uses (B, 1, T, F) layout where H=T, W=F
        # MLX Conv2d expects (B, H, W, C_in) format
        # So we need: (B, T, input_dim) -> (B, T, input_dim, 1)
        x = x[:, :, :, None]  # (B, T, input_dim, 1)

        # Apply conv layers
        for layer in self.conv:
            x = layer(x)

        # x is now (B, T', F', C_out) where T' is reduced time, F' is reduced freq
        B, T_new, F_new, C_new = x.shape

        # PyTorch flattens as (B, T', C*F') so we need to transpose to match
        # (B, T', F', C) -> (B, T', C, F')
        x = x.transpose(0, 1, 3, 2)  # (B, T', C_out, F')

        # Flatten channel and spatial dimensions: (B, T', C_out*F')
        x = x.reshape(B, T_new, C_new * F_new)

        # Project to output_dim
        x = self.out_proj(x)

        return x


class RelPositionMultiHeadedAttention(nn.Module):
    """Multi-head attention with relative positional encoding.

    Uses pos_bias_u and pos_bias_v for relative position bias.
    """

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # Q, K, V projections
        self.linear_q = nn.Linear(dim, dim)
        self.linear_k = nn.Linear(dim, dim)
        self.linear_v = nn.Linear(dim, dim)
        self.linear_out = nn.Linear(dim, dim)

        # Relative position projection
        self.linear_pos = nn.Linear(dim, dim, bias=False)

        # Learnable position biases
        self.pos_bias_u = mx.zeros((num_heads, self.head_dim))
        self.pos_bias_v = mx.zeros((num_heads, self.head_dim))

    def __call__(self, x: mx.array, pos_emb: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        """
        Args:
            x: (B, T, dim)
            pos_emb: (B, T, dim) relative positional embeddings
            mask: (B, 1, T) attention mask
        Returns:
            output: (B, T, dim)
        """
        B, T, C = x.shape

        # Project Q, K, V
        q = self.linear_q(x).reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.linear_k(x).reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.linear_v(x).reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Project positional embeddings
        p = self.linear_pos(pos_emb).reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Compute attention with position bias
        # q_with_bias_u = q + pos_bias_u
        q_with_bias_u = q + self.pos_bias_u[None, :, None, :]

        # Content-based attention: (q + u)K^T
        content_scores = mx.matmul(q_with_bias_u, k.transpose(0, 1, 3, 2))

        # q_with_bias_v = q + pos_bias_v
        q_with_bias_v = q + self.pos_bias_v[None, :, None, :]

        # Position-based attention: (q + v)P^T
        pos_scores = mx.matmul(q_with_bias_v, p.transpose(0, 1, 3, 2))

        # Combined scores
        scores = (content_scores + pos_scores) / math.sqrt(self.head_dim)

        # Apply mask if provided
        if mask is not None:
            # mask: (B, 1, T) -> (B, 1, 1, T)
            scores = mx.where(mask[:, :, None, :] == 0, -1e9, scores)

        # Softmax and weighted sum
        attn_weights = mx.softmax(scores, axis=-1)
        attn_output = mx.matmul(attn_weights, v)

        # Reshape and project
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(B, T, C)
        return self.linear_out(attn_output)


class ConvolutionModule(nn.Module):
    """Convolution module in Conformer."""

    def __init__(self, dim: int, kernel_size: int = 15):
        super().__init__()
        assert (kernel_size - 1) % 2 == 0, "kernel_size must be odd"

        self.pointwise_conv1 = nn.Conv1d(dim, 2 * dim, kernel_size=1)
        self.depthwise_conv = nn.Conv1d(
            dim, dim, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, groups=dim
        )
        self.norm = nn.LayerNorm(dim)
        self.pointwise_conv2 = nn.Conv1d(dim, dim, kernel_size=1)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: (B, T, dim)
        Returns:
            output: (B, T, dim)
        """
        # Pointwise conv 1 (with GLU)
        x_conv = self.pointwise_conv1(x)  # (B, T, 2*dim)
        x_conv, gate = mx.split(x_conv, 2, axis=-1)
        x_conv = x_conv * mx.sigmoid(gate)

        # Depthwise conv
        x_conv = self.depthwise_conv(x_conv)  # (B, T, dim)

        # Layer norm
        x_conv = self.norm(x_conv)

        # Swish activation
        x_conv = x_conv * mx.sigmoid(x_conv)

        # Pointwise conv 2
        x_conv = self.pointwise_conv2(x_conv)

        return x_conv


class FeedForwardModule(nn.Module):
    """Feed-forward module in Conformer."""

    def __init__(self, dim: int, intermediate_dim: int, dropout: float = 0.0):
        super().__init__()
        self.w_1 = nn.Linear(dim, intermediate_dim)
        self.w_2 = nn.Linear(intermediate_dim, dim)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: (B, T, dim)
        Returns:
            output: (B, T, dim)
        """
        x = self.w_1(x)
        x = x * mx.sigmoid(x)  # SiLU (Swish) activation
        x = self.w_2(x)
        return x


class ConformerEncoderLayer(nn.Module):
    """Single Conformer encoder layer.

    Architecture (NON-macaron style):
    - Multi-head self-attention (with relative position)
    - Conv module
    - Feed-forward (scale 1.0)
    - Final LayerNorm

    Note: This checkpoint does NOT use macaron-style (no feed_forward_macaron).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        intermediate_dim: int,
        kernel_size: int = 15,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm_ff = nn.LayerNorm(dim)
        self.feed_forward = FeedForwardModule(dim, intermediate_dim, dropout)

        self.norm_mha = nn.LayerNorm(dim)
        self.self_attn = RelPositionMultiHeadedAttention(dim, num_heads, dropout)

        self.norm_conv = nn.LayerNorm(dim)
        self.conv_module = ConvolutionModule(dim, kernel_size)

        self.norm_final = nn.LayerNorm(dim)

    def __call__(self, x: mx.array, pos_emb: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        """
        Args:
            x: (B, T, dim)
            pos_emb: (B, T, dim) positional embeddings
            mask: (B, 1, T) attention mask
        Returns:
            output: (B, T, dim)
        """
        # Multi-head self-attention
        x = x + self.self_attn(self.norm_mha(x), pos_emb, mask)

        # Conv module
        x = x + self.conv_module(self.norm_conv(x))

        # Feed-forward (scale 1.0, not 0.5 - no macaron style)
        x = x + self.feed_forward(self.norm_ff(x))

        # Final layer norm (applied AFTER all operations)
        x = self.norm_final(x)

        return x


class ConformerEncoder(nn.Module):
    """Conformer encoder for mel conditioning.

    Architecture:
    - Conv2d subsampling (4x downsampling)
    - Positional encoding
    - N Conformer layers
    - Final layer norm
    """

    def __init__(
        self,
        input_dim: int = 100,
        output_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        intermediate_dim: int = 2048,
        kernel_size: int = 15,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.output_dim = output_dim

        # Conv subsampling
        self.embed = Conv2dSubsampling(input_dim, output_dim)

        # Positional encoding
        self.pos_enc = PositionalEncoding(output_dim)

        # Conformer layers
        self.encoders = [
            ConformerEncoderLayer(output_dim, num_heads, intermediate_dim, kernel_size, dropout)
            for _ in range(num_layers)
        ]

        # Final norm
        self.after_norm = nn.LayerNorm(output_dim)

    def __call__(self, x: mx.array, lengths: Optional[mx.array] = None) -> Tuple[mx.array, Optional[mx.array]]:
        """
        Args:
            x: (B, T, input_dim) mel spectrogram
            lengths: (B,) sequence lengths (before subsampling)
        Returns:
            output: (B, T', output_dim) encoded features
            mask: (B, 1, T') attention mask
        """
        # Conv subsampling
        x = self.embed(x)

        # Add positional encoding
        x = self.pos_enc(x)

        # Create attention mask if lengths provided
        mask = None
        if lengths is not None:
            # After 2x subsampling (Conv2dSubsampling2 with stride=2)
            lengths_subsampled = (lengths - 1) // 2
            B, T = x.shape[0], x.shape[1]
            mask = mx.arange(T)[None, :] < lengths_subsampled[:, None]
            mask = mask[:, None, :]  # (B, 1, T)

        # Positional embeddings (same as input for relative position)
        pos_emb = self.pos_enc.pe[:, :x.shape[1], :]
        pos_emb = mx.broadcast_to(pos_emb, (x.shape[0], x.shape[1], x.shape[2]))

        # Apply Conformer layers
        for encoder in self.encoders:
            x = encoder(x, pos_emb, mask)

        # Final norm
        x = self.after_norm(x)

        return x, mask


def create_conditioning_encoder(
    input_dim: int = 100,
    output_dim: int = 512,
    num_layers: int = 6,
    num_heads: int = 8,
    intermediate_dim: int = 2048,
) -> ConformerEncoder:
    """Create Conformer encoder for mel conditioning.

    Default config matches UnifiedVoice checkpoint.
    """
    return ConformerEncoder(input_dim, output_dim, num_layers, num_heads, intermediate_dim)
