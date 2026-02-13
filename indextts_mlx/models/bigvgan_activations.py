"""
Snake and SnakeBeta activation functions for BigVGAN.

Implementation adapted from https://github.com/EdwardDixon/snake under the MIT license.
"""

import mlx.core as mx
import mlx.nn as nn


class Snake(nn.Module):
    """
    Sine-based periodic activation function.

    Snake(x) = x + (1/α) * sin²(xα)

    Args:
        in_features: Number of input features (channels)
        alpha: Initial value for alpha parameter
        alpha_trainable: Whether alpha should be trainable
        alpha_logscale: Whether to use log-scale parameterization

    Shape:
        - Input: (B, C, T) or (B, T, C)
        - Output: Same shape as input

    Reference:
        https://arxiv.org/abs/2006.08195
    """

    def __init__(
        self,
        in_features: int,
        alpha: float = 1.0,
        alpha_trainable: bool = True,
        alpha_logscale: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.alpha_logscale = alpha_logscale

        # Initialize alpha
        if alpha_logscale:
            # Log scale: initialized to zeros
            self.alpha = mx.zeros(in_features) * alpha
        else:
            # Linear scale: initialized to ones * alpha
            self.alpha = mx.ones(in_features) * alpha

        self.alpha_trainable = alpha_trainable
        self.no_div_by_zero = 1e-9

    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass.

        Args:
            x: Input tensor (B, T, C) - MLX Conv1d format

        Returns:
            Activated tensor with same shape as input
        """
        # Reshape alpha to match input dimensions
        # For (B, T, C), alpha should be (1, 1, C)
        if x.ndim == 3:
            # Channels are in the last dimension
            alpha = self.alpha.reshape(1, 1, -1)
        else:
            alpha = self.alpha

        if self.alpha_logscale:
            alpha = mx.exp(alpha)

        # Snake: x + (1/α) * sin²(xα)
        sin_term = mx.sin(x * alpha)
        result = x + (1.0 / (alpha + self.no_div_by_zero)) * (sin_term ** 2)

        return result


class SnakeBeta(nn.Module):
    """
    Modified Snake function with separate parameters for frequency (alpha) and magnitude (beta).

    SnakeBeta(x) = x + (1/β) * sin²(xα)

    Args:
        in_features: Number of input features (channels)
        alpha: Initial value for alpha and beta parameters
        alpha_trainable: Whether alpha and beta should be trainable
        alpha_logscale: Whether to use log-scale parameterization

    Shape:
        - Input: (B, C, T) or (B, T, C)
        - Output: Same shape as input

    Reference:
        Modified version based on https://arxiv.org/abs/2006.08195
    """

    def __init__(
        self,
        in_features: int,
        alpha: float = 1.0,
        alpha_trainable: bool = True,
        alpha_logscale: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.alpha_logscale = alpha_logscale

        # Initialize alpha and beta
        if alpha_logscale:
            # Log scale: initialized to zeros
            self.alpha = mx.zeros(in_features) * alpha
            self.beta = mx.zeros(in_features) * alpha
        else:
            # Linear scale: initialized to ones * alpha
            self.alpha = mx.ones(in_features) * alpha
            self.beta = mx.ones(in_features) * alpha

        self.alpha_trainable = alpha_trainable
        self.no_div_by_zero = 1e-9

    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass.

        Args:
            x: Input tensor (B, T, C) - MLX Conv1d format

        Returns:
            Activated tensor with same shape as input
        """
        # Reshape alpha and beta to match input dimensions
        # For (B, T, C), alpha/beta should be (1, 1, C)
        if x.ndim == 3:
            # Channels are in the last dimension
            alpha = self.alpha.reshape(1, 1, -1)
            beta = self.beta.reshape(1, 1, -1)
        else:
            alpha = self.alpha
            beta = self.beta

        if self.alpha_logscale:
            alpha = mx.exp(alpha)
            beta = mx.exp(beta)

        # SnakeBeta: x + (1/β) * sin²(xα)
        sin_term = mx.sin(x * alpha)
        result = x + (1.0 / (beta + self.no_div_by_zero)) * (sin_term ** 2)

        return result


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    """Calculate padding for 'same' convolution."""
    return int((kernel_size * dilation - dilation) / 2)
