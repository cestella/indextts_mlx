"""
MLX InterpolateRegulator Implementation

Length regulator that upsamples semantic tokens to mel frame rate.
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Optional, Tuple
from dataclasses import dataclass

from .s2mel_layers import Mish, sequence_mask


@dataclass
class InterpolateRegulatorConfig:
    """Configuration for InterpolateRegulator."""

    channels: int = 512
    in_channels: Optional[int] = None  # For continuous input
    out_channels: Optional[int] = None
    groups: int = 1  # For GroupNorm


class InterpolateRegulator(nn.Module):
    """Interpolate regulator for upsampling semantic features.

    Upsamples semantic token sequences to match mel frame rate using
    nearest-neighbor interpolation + Conv1d processing.
    """

    def __init__(self, config: InterpolateRegulatorConfig):
        """Initialize regulator.

        Args:
            config: Regulator configuration
        """
        super().__init__()
        self.config = config

        channels = config.channels
        in_channels = config.in_channels or channels
        out_channels = config.out_channels or channels
        groups = config.groups

        # Input projection for continuous features
        self.content_in_proj = nn.Linear(in_channels, channels)

        # Processing layers (simple version without sampling_ratios)
        # In the full implementation, would have multiple Conv1d + GroupNorm + Mish
        # MLX Conv1d signature: Conv1d(in_channels, out_channels, kernel_size, ...)
        self.conv1 = nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.act1 = Mish()

        self.conv2 = nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.act2 = Mish()

        # Output projection
        self.out_proj = nn.Conv1d(in_channels=channels, out_channels=out_channels, kernel_size=1, stride=1)

    def __call__(
        self,
        x: mx.array,
        ylens: mx.array,
        f0: Optional[mx.array] = None,
    ) -> Tuple[mx.array, mx.array]:
        """Forward pass.

        Args:
            x: Input semantic features, shape (batch, T_in, in_channels)
            ylens: Target output lengths, shape (batch,)
            f0: F0 contours (not used in simple version)

        Returns:
            Tuple of (output, output_lengths):
            - output: Upsampled features, shape (batch, T_out, out_channels)
            - output_lengths: Output lengths (same as ylens)
        """
        # Project input
        x = self.content_in_proj(x)  # (batch, T_in, channels)

        # MLX Conv1d expects channels-last format: (batch, length, channels)
        # So NO transpose needed!

        # Get max output length
        max_len = int(mx.max(ylens).item())

        # Interpolate to target length using nearest neighbor
        batch_size, T_in, channels = x.shape

        # Create index mapping for nearest neighbor interpolation
        # For each output position, find nearest input position
        out_indices = mx.arange(max_len, dtype=mx.float32)
        scale = T_in / max_len
        in_indices = (out_indices * scale).astype(mx.int32)
        in_indices = mx.clip(in_indices, 0, T_in - 1)

        # Gather values along time axis
        x_interp = mx.take(x, in_indices, axis=1)  # (batch, T_out, channels)

        # Apply processing layers (all in channels-last format)
        x = self.conv1(x_interp)
        x = self.norm1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act2(x)

        # Output projection
        x = self.out_proj(x)  # (batch, T_out, out_channels)

        # Apply length mask
        mask = sequence_mask(ylens, max_length=max_len)  # (batch, T_out)
        mask = mx.expand_dims(mask, -1)  # (batch, T_out, 1)
        x = x * mask

        return x, ylens


def create_regulator_from_pytorch_config(pytorch_args) -> InterpolateRegulator:
    """Create MLX regulator from PyTorch config.

    Args:
        pytorch_args: PyTorch configuration object

    Returns:
        InterpolateRegulator model
    """
    config = InterpolateRegulatorConfig(
        channels=pytorch_args.length_regulator.channels,
        in_channels=getattr(pytorch_args.length_regulator, 'in_channels', None),
        out_channels=pytorch_args.length_regulator.channels,  # Typically same as channels
        groups=1,  # Standard value
    )

    return InterpolateRegulator(config)
