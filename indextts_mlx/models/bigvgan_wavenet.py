"""
MLX WaveNet Implementation

WaveNet-style decoder used as final layer in DiT.
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Optional
import math
from .s2mel_layers import reflect_pad1d


class WaveNetResidualBlock(nn.Module):
    """WaveNet-style residual block with dilated convolutions."""

    def __init__(
        self,
        hidden_channels: int,
        kernel_size: int,
        dilation: int,
        p_dropout: float = 0.0,
        causal: bool = False,
    ):
        """Initialize WaveNet residual block.

        Args:
            hidden_channels: Number of hidden channels
            kernel_size: Convolution kernel size
            dilation: Dilation rate
            p_dropout: Dropout probability
            causal: Whether to use causal convolutions
        """
        super().__init__()
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.causal = causal
        self.stride = 1

        # Dilated convolution with NO built-in padding (we'll apply reflect padding manually)
        self.conv = nn.Conv1d(
            hidden_channels,
            2 * hidden_channels,  # For tanh and sigmoid gates
            kernel_size,
            stride=self.stride,
            padding=0,  # No padding - we apply reflect padding manually
        )

        # Dropout
        self.dropout = nn.Dropout(p_dropout) if p_dropout > 0 else None

    def __call__(
        self,
        x: mx.array,
        g_cond: Optional[mx.array] = None
    ) -> mx.array:
        """Apply WaveNet residual block.

        Args:
            x: Input of shape (batch, length, channels) - channels-last for MLX
            g_cond: Pre-projected global conditioning of shape (batch, length, 2*channels)

        Returns:
            Gated activation output before res/skip projection
        """
        # Calculate padding (matching PyTorch's SConv1d)
        effective_kernel_size = (self.kernel_size - 1) * self.dilation + 1
        padding_total = effective_kernel_size - self.stride

        # Calculate extra padding for output length
        length = x.shape[1]
        n_frames = (length - effective_kernel_size + padding_total) / self.stride + 1
        ideal_length = (math.ceil(n_frames) - 1) * self.stride + (effective_kernel_size - padding_total)
        extra_padding = ideal_length - length

        # Apply reflect padding (matching PyTorch's pad_mode='reflect')
        if self.causal:
            # Causal: all padding on left
            pad_left = padding_total
            pad_right = extra_padding
        else:
            # Non-causal: asymmetric padding
            padding_right = padding_total // 2
            padding_left = padding_total - padding_right
            pad_left = padding_left
            pad_right = padding_right + extra_padding

        x_padded = reflect_pad1d(x, (pad_left, pad_right))

        # Dilated convolution (MLX Conv1d uses channels-last)
        h = self.conv(x_padded)  # (batch, length, 2*channels)

        # Add global conditioning (pre-projected)
        if g_cond is not None:
            h = h + g_cond

        # Split for gated activation (split along channel dim, which is -1)
        h_tanh, h_sigmoid = mx.split(h, 2, axis=-1)
        h = mx.tanh(h_tanh) * mx.sigmoid(h_sigmoid)

        # Dropout
        if self.dropout is not None:
            h = self.dropout(h)

        return h


class WaveNet(nn.Module):
    """WaveNet decoder."""

    def __init__(
        self,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        gin_channels: int = 0,
        p_dropout: float = 0.0,
        causal: bool = False,
    ):
        """Initialize WaveNet.

        Args:
            hidden_channels: Number of hidden channels
            kernel_size: Convolution kernel size
            dilation_rate: Dilation rate base (usually 1 or 2)
            n_layers: Number of WaveNet layers
            gin_channels: Global conditioning channels
            p_dropout: Dropout probability
            causal: Whether to use causal convolutions
        """
        super().__init__()
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers

        # Shared global conditioning layer (matches PyTorch structure)
        # Projects to 2 * n_layers * hidden_channels (one conditioning per layer)
        if gin_channels > 0:
            self.cond_layer = nn.Conv1d(gin_channels, 2 * n_layers * hidden_channels, 1)
        else:
            self.cond_layer = None

        # WaveNet residual blocks (dilated convolutions)
        self.in_layers = []
        for i in range(n_layers):
            dilation = dilation_rate ** i if dilation_rate > 1 else 1
            self.in_layers.append(
                WaveNetResidualBlock(
                    hidden_channels,
                    kernel_size,
                    dilation,
                    p_dropout,
                    causal=causal,
                )
            )

        # Residual and skip projection layers (separate from blocks, matches PyTorch)
        # Note: Last layer outputs only hidden_channels (skip only, no residual)
        self.res_skip_layers = []
        for i in range(n_layers):
            out_channels = hidden_channels if i == n_layers - 1 else 2 * hidden_channels
            self.res_skip_layers.append(
                nn.Conv1d(hidden_channels, out_channels, 1)
            )

    def __call__(
        self,
        x: mx.array,
        x_mask: Optional[mx.array] = None,
        g: Optional[mx.array] = None,
    ) -> mx.array:
        """Apply WaveNet.

        Args:
            x: Input of shape (batch, length, channels) - channels-last for MLX
            x_mask: Mask of shape (batch, length, 1)
            g: Global conditioning of shape (batch, length, gin_channels)

        Returns:
            Output of shape (batch, length, channels)
        """
        # Apply shared global conditioning once and split per layer
        g_cond_all = None
        if self.cond_layer is not None and g is not None:
            g_cond_all = self.cond_layer(g)  # (batch, length, 2*n_layers*channels)

        # Apply WaveNet blocks and accumulate skip connections
        skip_sum = mx.zeros_like(x)

        for i, layer in enumerate(self.in_layers):
            # Extract conditioning for this layer
            g_cond = None
            if g_cond_all is not None:
                # Split along channel dimension to get this layer's conditioning
                start_idx = i * 2 * self.hidden_channels
                end_idx = (i + 1) * 2 * self.hidden_channels
                g_cond = g_cond_all[:, :, start_idx:end_idx]

            # Apply dilated conv + gating
            h = layer(x, g_cond)

            # Apply res/skip projection
            res_skip = self.res_skip_layers[i](h)

            # Last layer: only skip (no residual)
            if i == self.n_layers - 1:
                skip = res_skip
            else:
                res, skip = mx.split(res_skip, 2, axis=-1)
                # Residual connection (no sqrt(2) division - matches PyTorch)
                x = x + res
                if x_mask is not None:
                    x = x * x_mask

            # Accumulate skip
            skip_sum = skip_sum + skip

        # Sum skip connections (NO averaging - matches PyTorch)
        output = skip_sum

        # Apply mask
        if x_mask is not None:
            output = output * x_mask

        return output
