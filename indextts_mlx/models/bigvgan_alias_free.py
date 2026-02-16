"""
Alias-free activation implementation in MLX.

Adapted from https://github.com/junjun3518/alias-free-torch under the Apache License 2.0
"""

import math
import mlx.core as mx
import mlx.nn as nn


def sinc(x: mx.array) -> mx.array:
    """
    Implementation of sinc, i.e. sin(pi * x) / (pi * x)
    """
    return mx.where(
        x == 0,
        mx.ones_like(x),
        mx.sin(math.pi * x) / (math.pi * x),
    )


def kaiser_sinc_filter1d(cutoff: float, half_width: float, kernel_size: int) -> mx.array:
    """
    Create a Kaiser-windowed sinc filter.

    Args:
        cutoff: Cutoff frequency (0 to 0.5)
        half_width: Half-width of transition band
        kernel_size: Size of the filter kernel

    Returns:
        Filter of shape (1, 1, kernel_size)
    """
    even = kernel_size % 2 == 0
    half_size = kernel_size // 2

    # Kaiser window parameters
    delta_f = 4 * half_width
    A = 2.285 * (half_size - 1) * math.pi * delta_f + 7.95
    if A > 50.0:
        beta = 0.1102 * (A - 8.7)
    elif A >= 21.0:
        beta = 0.5842 * (A - 21) ** 0.4 + 0.07886 * (A - 21.0)
    else:
        beta = 0.0

    # Create Kaiser window
    # MLX doesn't have kaiser_window, so we'll use PyTorch's formula
    n = mx.arange(kernel_size, dtype=mx.float32)
    alpha = (kernel_size - 1) / 2.0
    window = mx.i0(beta * mx.sqrt(1 - ((n - alpha) / alpha) ** 2)) / mx.i0(mx.array(beta))

    # Create time vector
    if even:
        time = mx.arange(-half_size, half_size, dtype=mx.float32) + 0.5
    else:
        time = mx.arange(kernel_size, dtype=mx.float32) - half_size

    # Create sinc filter
    if cutoff == 0:
        filter_ = mx.zeros_like(time)
    else:
        filter_ = 2 * cutoff * window * sinc(2 * cutoff * time)
        # Normalize
        filter_ = filter_ / mx.sum(filter_)

    return filter_.reshape(1, 1, kernel_size)


class LowPassFilter1d(nn.Module):
    """
    Low-pass filter using windowed sinc filter.

    Note: Filter is loaded from checkpoint rather than computed,
    ensuring exact parity with PyTorch.
    """

    def __init__(
        self,
        cutoff: float = 0.5,
        half_width: float = 0.6,
        stride: int = 1,
        padding: bool = True,
        kernel_size: int = 12,
    ):
        super().__init__()
        if cutoff < 0.0:
            raise ValueError("Minimum cutoff must be larger than zero.")
        if cutoff > 0.5:
            raise ValueError("A cutoff above 0.5 does not make sense.")

        self.kernel_size = kernel_size
        self.even = kernel_size % 2 == 0
        self.pad_left = kernel_size // 2 - int(self.even)
        self.pad_right = kernel_size // 2
        self.stride = stride
        self.padding = padding

        # Filter will be loaded from checkpoint
        # Initialize with placeholder (will be replaced during weight loading)
        self.filter = mx.zeros((1, 1, kernel_size))

    def __call__(self, x: mx.array) -> mx.array:
        """
        Apply lowpass filter.

        Args:
            x: Input (B, T, C) - MLX format

        Returns:
            Filtered output (B, T', C)
        """
        B, T, C = x.shape

        # (B, T, C) → (B, C, T) for padding
        x = x.transpose(0, 2, 1)

        if self.padding:
            x = mx.pad(x, [(0, 0), (0, 0), (self.pad_left, self.pad_right)], mode="edge")

        # Fold channels into batch: (B, C, T+pad) → (B*C, T+pad, 1)
        # This lets a single conv1d kernel handle all channels at once instead
        # of dispatching one kernel per channel (MLX has no groups= parameter).
        x = x.reshape(B * C, -1, 1)
        out = mx.conv1d(x, self.filter.transpose(0, 2, 1), stride=self.stride)  # (B*C, T', 1)

        T_out = out.shape[1]
        return out.reshape(B, C, T_out).transpose(0, 2, 1)  # (B, T', C)


class UpSample1d(nn.Module):
    """
    Upsample using transposed convolution with sinc filter.

    Note: Filter is loaded from checkpoint rather than computed.
    """

    def __init__(self, ratio: int = 2, kernel_size: int = None):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        self.stride = ratio
        self.pad = self.kernel_size // ratio - 1
        self.pad_left = self.pad * self.stride + (self.kernel_size - self.stride) // 2
        self.pad_right = self.pad * self.stride + (self.kernel_size - self.stride + 1) // 2

        # Filter will be loaded from checkpoint
        self.filter = mx.zeros((1, 1, self.kernel_size))

    def __call__(self, x: mx.array) -> mx.array:
        """
        Upsample input.

        Args:
            x: Input (B, T, C) - MLX format

        Returns:
            Upsampled output (B, T*ratio, C)
        """
        B, T, C = x.shape

        # (B, T, C) → (B, C, T), pad, fold into batch → (B*C, T+pad, 1)
        x = x.transpose(0, 2, 1)
        x = mx.pad(x, [(0, 0), (0, 0), (self.pad, self.pad)], mode="edge")
        x = x.reshape(B * C, -1, 1)

        # Single conv_transpose1d dispatch instead of one per channel.
        out = mx.conv_transpose1d(
            x, self.filter.transpose(0, 2, 1), stride=self.stride
        )  # (B*C, T_up, 1)

        T_up = out.shape[1]
        result = out.reshape(B, C, T_up).transpose(0, 2, 1)  # (B, T_up, C)

        # Trim edges
        if self.pad_right > 0:
            result = result[:, self.pad_left : -self.pad_right, :]
        else:
            result = result[:, self.pad_left :, :]

        # Scale by ratio
        result = self.ratio * result

        return result


class DownSample1d(nn.Module):
    """
    Downsample using lowpass filter.
    """

    def __init__(self, ratio: int = 2, kernel_size: int = None):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        self.lowpass = LowPassFilter1d(
            cutoff=0.5 / ratio,
            half_width=0.6 / ratio,
            stride=ratio,
            kernel_size=self.kernel_size,
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.lowpass(x)


class Activation1d(nn.Module):
    """
    Alias-free activation with anti-aliasing via upsampling and downsampling.
    """

    def __init__(
        self,
        activation,
        up_ratio: int = 2,
        down_ratio: int = 2,
        up_kernel_size: int = 12,
        down_kernel_size: int = 12,
    ):
        super().__init__()
        self.up_ratio = up_ratio
        self.down_ratio = down_ratio
        self.act = activation
        self.upsample = UpSample1d(up_ratio, up_kernel_size)
        self.downsample = DownSample1d(down_ratio, down_kernel_size)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Apply alias-free activation.

        Args:
            x: Input (B, T, C)

        Returns:
            Activated output (B, T, C)
        """
        x = self.upsample(x)
        x = self.act(x)
        x = self.downsample(x)
        return x
