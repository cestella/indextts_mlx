"""
CAMPPlus Speaker Encoder - Pure MLX Implementation

Port of the CAMPPlus (Context-Aware Masked PLDA) speaker encoder to MLX.
Achieves perfect parity with PyTorch implementation.

Architecture:
- FCM (Front-end Conv Module): Conv2d + ResNet blocks
- Dense TDNN blocks with CAM (Context-Aware Modulation) layers
- Statistics pooling (mean + std)
- Final dense layer to embedding (192 or 512 dim)
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from typing import List, Tuple


def statistics_pooling(x: mx.array, axis: int = -1) -> mx.array:
    """Compute mean and std statistics over time dimension.

    Args:
        x: Input tensor in MLX format (B, T, C) when axis=1
        axis: Axis to pool over (default: -1 for time)

    Returns:
        stats: Concatenated [mean, std] (B, 2*C)
    """
    mean = mx.mean(x, axis=axis)
    # Unbiased std: divide by (N-1) to match PyTorch's std(unbiased=True)
    # MLX var has ddof parameter (delta degrees of freedom)
    std = mx.sqrt(mx.var(x, axis=axis, ddof=1))
    stats = mx.concatenate([mean, std], axis=-1)
    return stats


class StatsPool(nn.Module):
    """Statistics pooling layer (mean + std concatenation).

    Pools along the time dimension (axis=1) for input shape (B, T, C).
    """

    def __call__(self, x: mx.array) -> mx.array:
        # Input: (B, T, C), pool along time axis to get (B, 2*C)
        return statistics_pooling(x, axis=1)


class BasicResBlock(nn.Module):
    """Basic residual block for Conv2d."""

    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()

        # PyTorch uses (stride, 1) for frequency downsampling, time unchanged
        # MLX expects (H_stride, W_stride), so stride in H (freq), 1 in W (time)
        stride_2d = (stride, 1)

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride_2d, padding=1)
        self.bn1 = nn.BatchNorm(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm(planes)

        # Shortcut connection
        self.shortcut = None
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = [
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride_2d),
                nn.BatchNorm(self.expansion * planes),
            ]

    def __call__(self, x: mx.array) -> mx.array:
        out = nn.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.shortcut is not None:
            shortcut = x
            for layer in self.shortcut:
                shortcut = layer(shortcut)
            out = out + shortcut
        else:
            out = out + x

        out = nn.relu(out)
        return out


class FCM(nn.Module):
    """Front-end Convolutional Module with ResNet blocks."""

    def __init__(
        self,
        block=BasicResBlock,
        num_blocks: List[int] = [2, 2],
        m_channels: int = 32,
        feat_dim: int = 80,
    ):
        super().__init__()

        self.in_planes = m_channels
        self.conv1 = nn.Conv2d(1, m_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm(m_channels)

        # ResNet layers
        self.layer1 = self._make_layer(block, m_channels, num_blocks[0], stride=2)
        self.layer2 = self._make_layer(block, m_channels, num_blocks[1], stride=2)

        # Final conv with stride (2, 1) - downsamples frequency, not time
        self.conv2 = nn.Conv2d(m_channels, m_channels, kernel_size=3, stride=(2, 1), padding=1)
        self.bn2 = nn.BatchNorm(m_channels)

        self.out_channels = m_channels * (feat_dim // 8)

    def _make_layer(self, block, planes: int, num_blocks: int, stride: int):
        """Create a sequence of residual blocks."""
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return layers

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, F, T) -> MLX expects (B, H, W, C) for Conv2d
        # Treat F as height, T as width
        # (B, F, T) -> (B, F, T, 1) for single channel
        x = mx.expand_dims(x, axis=-1)  # Add channel dim at end

        out = nn.relu(self.bn1(self.conv1(x)))

        # Apply ResNet blocks
        for layer in self.layer1:
            out = layer(out)
        for layer in self.layer2:
            out = layer(out)

        out = nn.relu(self.bn2(self.conv2(out)))

        # Reshape to match PyTorch: (B, C*H, W)
        # PyTorch: (B, C, H, W) -> (B, C*H, W) where index i = c*H + h (C-major order)
        # MLX: (B, H, W, C) -> (B, W, C*H) where index i = c*H + h
        shape = out.shape  # (B, H, W, C) = (B, 10, 300, 32)
        # Transpose to (B, W, C, H) then reshape to (B, W, C*H) for C-major order
        out = mx.transpose(out, (0, 2, 3, 1))  # (B, H, W, C) -> (B, W, C, H)
        out = mx.reshape(
            out, (shape[0], shape[2], shape[3] * shape[1])
        )  # (B, W, C*H) = (B, 300, 320)
        return out


class TDNNLayer(nn.Module):
    """Time-Delay Neural Network layer (1D convolution)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = False,
        config_str: str = "batchnorm-relu",
    ):
        super().__init__()

        # Auto padding for "same" convolution
        if padding < 0:
            assert (
                kernel_size % 2 == 1
            ), f"Expect odd kernel size for auto padding, got {kernel_size}"
            padding = (kernel_size - 1) // 2 * dilation

        self.linear = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias
        )

        # Parse config string for normalization and activation
        self.nonlinear = self._make_nonlinear(config_str, out_channels)

    def _make_nonlinear(self, config_str: str, channels: int):
        layers = []
        for name in config_str.split("-"):
            if name == "relu":
                layers.append(nn.ReLU())
            elif name == "batchnorm":
                layers.append(nn.BatchNorm(channels))
            elif name == "batchnorm_":
                layers.append(nn.BatchNorm(channels, affine=False))
            # Note: prelu not commonly used, skip for now
        return layers

    def __call__(self, x: mx.array) -> mx.array:
        x = self.linear(x)
        for layer in self.nonlinear:
            x = layer(x)
        return x


class CAMLayer(nn.Module):
    """Context-Aware Modulation layer with attention."""

    def __init__(
        self,
        bn_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        dilation: int,
        bias: bool,
        reduction: int = 2,
    ):
        super().__init__()

        # Local convolution
        self.linear_local = nn.Conv1d(
            bn_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

        # Attention mechanism
        self.linear1 = nn.Conv1d(bn_channels, bn_channels // reduction, kernel_size=1)
        self.linear2 = nn.Conv1d(bn_channels // reduction, out_channels, kernel_size=1)

    def seg_pooling(self, x: mx.array, seg_len: int = 100) -> mx.array:
        """Segmented average pooling with expansion (matches PyTorch avg_pool1d with ceil_mode=True).

        Args:
            x: Input tensor in MLX format (B, T, C)
        Returns:
            Pooled tensor in MLX format (B, T, C)
        """
        # MLX format: (B, T, C)
        B, T, C = x.shape

        # Calculate output length with ceil
        out_len = (T + seg_len - 1) // seg_len

        # Compute segment averages (handling partial last segment)
        segments = []
        for i in range(out_len):
            start = i * seg_len
            end = min(start + seg_len, T)
            segment = mx.mean(x[:, start:end, :], axis=1)  # (B, C)
            segments.append(segment)

        seg = mx.stack(segments, axis=1)  # (B, out_len, C)

        # Expand back: (B, out_len, C) -> (B, out_len, 1, C) -> (B, out_len, seg_len, C)
        seg = mx.expand_dims(seg, axis=2)  # (B, out_len, 1, C)
        seg = mx.tile(seg, [1, 1, seg_len, 1])  # (B, out_len, seg_len, C)
        seg = mx.reshape(seg, (B, -1, C))  # (B, out_len*seg_len, C)

        # Trim to original length
        seg = seg[:, :T, :]
        return seg

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass with MLX format (B, T, C)."""
        y = self.linear_local(x)

        # Context: global mean + segmented pooling
        # Global mean across time dimension, keep (B, 1, C) shape
        context = mx.mean(x, axis=1, keepdims=True) + self.seg_pooling(x)
        context = nn.relu(self.linear1(context))
        m = nn.sigmoid(self.linear2(context))

        return y * m


class CAMDenseTDNNLayer(nn.Module):
    """Dense TDNN layer with CAM attention."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bn_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        bias: bool = False,
        config_str: str = "batchnorm-relu",
    ):
        super().__init__()

        assert kernel_size % 2 == 1, f"Expect odd kernel size, got {kernel_size}"
        padding = (kernel_size - 1) // 2 * dilation

        self.nonlinear1 = self._make_nonlinear(config_str, in_channels)
        self.linear1 = nn.Conv1d(in_channels, bn_channels, kernel_size=1, bias=False)
        self.nonlinear2 = self._make_nonlinear(config_str, bn_channels)
        self.cam_layer = CAMLayer(
            bn_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def _make_nonlinear(self, config_str: str, channels: int):
        layers = []
        for name in config_str.split("-"):
            if name == "relu":
                layers.append(nn.ReLU())
            elif name == "batchnorm":
                layers.append(nn.BatchNorm(channels))
            elif name == "batchnorm_":
                layers.append(nn.BatchNorm(channels, affine=False))
        return layers

    def __call__(self, x: mx.array) -> mx.array:
        # Bottleneck
        for layer in self.nonlinear1:
            x = layer(x)
        x = self.linear1(x)

        # CAM layer
        for layer in self.nonlinear2:
            x = layer(x)
        x = self.cam_layer(x)

        return x


class CAMDenseTDNNBlock(nn.Module):
    """Dense TDNN block with multiple layers and dense connections."""

    def __init__(
        self,
        num_layers: int,
        in_channels: int,
        out_channels: int,
        bn_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        bias: bool = False,
        config_str: str = "batchnorm-relu",
    ):
        super().__init__()

        self.layers = []
        for i in range(num_layers):
            layer = CAMDenseTDNNLayer(
                in_channels=in_channels + i * out_channels,
                out_channels=out_channels,
                bn_channels=bn_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                bias=bias,
                config_str=config_str,
            )
            self.layers.append(layer)

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.layers:
            # Dense connection: concatenate input with output
            # MLX format is (B, T, C), so concatenate along channel axis (axis=-1)
            out = layer(x)
            x = mx.concatenate([x, out], axis=-1)
        return x


class TransitLayer(nn.Module):
    """Transition layer to reduce channels."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        config_str: str = "batchnorm-relu",
    ):
        super().__init__()

        self.nonlinear = self._make_nonlinear(config_str, in_channels)
        self.linear = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=bias)

    def _make_nonlinear(self, config_str: str, channels: int):
        layers = []
        for name in config_str.split("-"):
            if name == "relu":
                layers.append(nn.ReLU())
            elif name == "batchnorm":
                layers.append(nn.BatchNorm(channels))
        return layers

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.nonlinear:
            x = layer(x)
        x = self.linear(x)
        return x


class DenseLayer(nn.Module):
    """Final dense layer for embedding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = False,
        config_str: str = "batchnorm_",
    ):
        super().__init__()

        self.linear = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.nonlinear = self._make_nonlinear(config_str, out_channels)

    def _make_nonlinear(self, config_str: str, channels: int):
        layers = []
        for name in config_str.split("-"):
            if name == "batchnorm":
                layers.append(nn.BatchNorm(channels))
            elif name == "batchnorm_":
                layers.append(nn.BatchNorm(channels, affine=False))
        return layers

    def __call__(self, x: mx.array) -> mx.array:
        # Handle 2D input (B, C) -> (B, 1, C) for MLX Conv1d
        # MLX Conv1d expects (B, L, C) format
        if len(x.shape) == 2:
            x = mx.expand_dims(x, axis=1)  # (B, C) -> (B, 1, C)
            x = self.linear(x)
            x = mx.squeeze(x, axis=1)  # (B, 1, C') -> (B, C')
        else:
            x = self.linear(x)

        for layer in self.nonlinear:
            x = layer(x)

        return x


class CAMPPlus(nn.Module):
    """CAMPPlus speaker encoder.

    Extracts speaker embeddings from FBANK features.

    Args:
        feat_dim: Input feature dimension (default: 80 for FBANK)
        embedding_size: Output embedding dimension (default: 192)
        growth_rate: Growth rate for dense blocks (default: 32)
        bn_size: Bottleneck size multiplier (default: 4)
        init_channels: Initial channels after FCM (default: 128)
        config_str: Configuration for normalization/activation
    """

    def __init__(
        self,
        feat_dim: int = 80,
        embedding_size: int = 192,
        growth_rate: int = 32,
        bn_size: int = 4,
        init_channels: int = 128,
        config_str: str = "batchnorm-relu",
    ):
        super().__init__()

        # Front-end convolutional module
        self.head = FCM(feat_dim=feat_dim)
        channels = self.head.out_channels

        # Initial TDNN layer
        self.tdnn = TDNNLayer(
            channels,
            init_channels,
            kernel_size=5,
            stride=2,
            dilation=1,
            padding=-1,
            config_str=config_str,
        )
        channels = init_channels

        # Dense blocks
        self.blocks = []
        self.transits = []

        for i, (num_layers, kernel_size, dilation) in enumerate(
            zip((12, 24, 16), (3, 3, 3), (1, 2, 2))
        ):
            # Dense block
            block = CAMDenseTDNNBlock(
                num_layers=num_layers,
                in_channels=channels,
                out_channels=growth_rate,
                bn_channels=bn_size * growth_rate,
                kernel_size=kernel_size,
                dilation=dilation,
                config_str=config_str,
            )
            self.blocks.append(block)
            channels = channels + num_layers * growth_rate

            # Transition layer
            transit = TransitLayer(channels, channels // 2, bias=False, config_str=config_str)
            self.transits.append(transit)
            channels //= 2

        # Final layers
        self.out_nonlinear = self._make_nonlinear(config_str, channels)
        self.stats = StatsPool()
        self.dense = DenseLayer(channels * 2, embedding_size, config_str="batchnorm_")

    def _make_nonlinear(self, config_str: str, channels: int):
        layers = []
        for name in config_str.split("-"):
            if name == "relu":
                layers.append(nn.ReLU())
            elif name == "batchnorm":
                layers.append(nn.BatchNorm(channels))
        return layers

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass.

        Args:
            x: Input FBANK features (B, T, F) where F=80

        Returns:
            embeddings: Speaker embeddings (B, embedding_size)
        """
        # Transpose: (B, T, F) -> (B, F, T)
        x = mx.transpose(x, (0, 2, 1))

        # FCM
        x = self.head(x)

        # Initial TDNN
        x = self.tdnn(x)

        # Dense blocks with transitions
        for block, transit in zip(self.blocks, self.transits):
            x = block(x)
            x = transit(x)

        # Final processing
        for layer in self.out_nonlinear:
            x = layer(x)

        # Statistics pooling
        x = self.stats(x)  # (B, 2*C)

        # Dense layer
        x = self.dense(x)  # (B, embedding_size)

        return x


def create_campplus(feat_dim: int = 80, embedding_size: int = 192) -> CAMPPlus:
    """Create CAMPPlus model with standard configuration.

    Args:
        feat_dim: Input feature dimension (80 for FBANK)
        embedding_size: Output embedding size (192 or 512)

    Returns:
        CAMPPlus model
    """
    return CAMPPlus(feat_dim=feat_dim, embedding_size=embedding_size)
