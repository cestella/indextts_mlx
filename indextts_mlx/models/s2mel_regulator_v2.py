"""
MLX InterpolateRegulator v2 - Matching Checkpoint Structure

Length regulator that upsamples semantic tokens to mel frame rate.
Matches the actual PyTorch checkpoint structure exactly.
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Optional, Tuple, List

from .s2mel_layers import Mish


class InterpolateRegulator(nn.Module):
    """Interpolate regulator for upsampling semantic features.

    Upsamples semantic token sequences to match mel frame rate using
    nearest-neighbor interpolation + Conv1d processing.

    Architecture (from checkpoint):
    - content_in_proj: Linear(in_channels, channels)
    - model: Sequential of Conv1d + GroupNorm + Mish blocks
      - 4x: Conv1d(3x3) + GroupNorm + Mish
      - 1x: Conv1d(1x1) final output
    """

    def __init__(
        self,
        in_channels: int,
        channels: int,
        sampling_ratios: List[int],
        groups: int = 1,
    ):
        """Initialize regulator.

        Args:
            in_channels: Input feature dimension
            channels: Hidden dimension
            sampling_ratios: Upsampling ratios (not used in interpolate version)
            groups: Number of groups for GroupNorm
        """
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.sampling_ratios = sampling_ratios
        self.groups = groups

        # Input projection for continuous features
        self.content_in_proj = nn.Linear(in_channels, channels)

        # Embedding for discrete content (if needed)
        # Note: We'll create this but may not load weights
        self.embedding = None  # Will be created if needed

        # Mask token
        self.mask_token = None  # Will be loaded if present

        # Processing layers matching checkpoint structure
        # Pattern: Conv1d → GroupNorm → Mish (x4) → Conv1d(1x1)
        self.model = []

        # Layer 0: Conv1d(3x3)
        self.model.append(nn.Conv1d(channels, channels, kernel_size=3, padding=1))
        # Layer 1: GroupNorm
        self.model.append(nn.GroupNorm(groups, channels))
        # Layer 2: Mish
        self.model.append(Mish())

        # Layer 3: Conv1d(3x3)
        self.model.append(nn.Conv1d(channels, channels, kernel_size=3, padding=1))
        # Layer 4: GroupNorm
        self.model.append(nn.GroupNorm(groups, channels))
        # Layer 5: Mish
        self.model.append(Mish())

        # Layer 6: Conv1d(3x3)
        self.model.append(nn.Conv1d(channels, channels, kernel_size=3, padding=1))
        # Layer 7: GroupNorm
        self.model.append(nn.GroupNorm(groups, channels))
        # Layer 8: Mish
        self.model.append(Mish())

        # Layer 9: Conv1d(3x3)
        self.model.append(nn.Conv1d(channels, channels, kernel_size=3, padding=1))
        # Layer 10: GroupNorm
        self.model.append(nn.GroupNorm(groups, channels))
        # Layer 11: Mish
        self.model.append(Mish())

        # Layer 12: Conv1d(1x1) - final output
        self.model.append(nn.Conv1d(channels, channels, kernel_size=1))

    def __call__(
        self, x: mx.array, ylens: mx.array, f0: Optional[mx.array] = None
    ) -> Tuple[mx.array, mx.array]:
        """Apply regulator.

        Args:
            x: Input features of shape (batch, T_in, in_channels)
            ylens: Target lengths of shape (batch,)
            f0: Optional F0 conditioning (not used)

        Returns:
            Tuple of:
                - Upsampled features of shape (batch, T_out, channels)
                - Output lengths of shape (batch,)
        """
        batch_size, T_in, _ = x.shape

        # Project input
        x = self.content_in_proj(x)  # (batch, T_in, channels)

        # Upsample to target length using nearest-neighbor interpolation
        # MLX doesn't have built-in interpolate, so we'll use indexing
        max_len = int(mx.max(ylens).item())

        # Create indices for nearest-neighbor upsampling
        # For each output position, find the nearest input position
        out_positions = mx.arange(max_len, dtype=mx.float32)  # [0, 1, 2, ..., max_len-1]

        # Compute input positions for each sample
        x_upsampled = []
        for i in range(batch_size):
            target_len = int(ylens[i].item())
            # Compute ratio
            ratio = T_in / target_len
            # Compute input indices (nearest neighbor)
            in_indices = mx.clip(
                mx.floor(out_positions[:target_len] * ratio).astype(mx.int32), 0, T_in - 1
            )
            # Gather features
            x_sample = mx.take(x[i], in_indices, axis=0)  # (target_len, channels)

            # Pad to max_len if needed
            if target_len < max_len:
                pad_len = max_len - target_len
                padding = mx.zeros((pad_len, self.channels))
                x_sample = mx.concatenate([x_sample, padding], axis=0)

            x_upsampled.append(x_sample)

        x = mx.stack(x_upsampled, axis=0)  # (batch, max_len, channels)

        # Apply processing layers (Conv1d expects channels-last for MLX)
        for layer in self.model:
            x = layer(x)

        return x, ylens


def create_regulator_from_config(config_dict: dict) -> InterpolateRegulator:
    """Create regulator from config dictionary.

    Args:
        config_dict: Configuration dictionary from YAML

    Returns:
        InterpolateRegulator instance
    """
    reg_config = config_dict

    regulator = InterpolateRegulator(
        in_channels=reg_config["in_channels"],
        channels=reg_config["channels"],
        sampling_ratios=reg_config["sampling_ratios"],
        groups=1,  # Default to 1 group
    )

    return regulator
