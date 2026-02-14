"""
BigVGAN vocoder implementation in MLX.

Based on NVIDIA's BigVGAN: https://github.com/NVIDIA/BigVGAN
"""

import mlx.core as mx
import mlx.nn as nn
from typing import List, Tuple

from .bigvgan_activations import Snake, SnakeBeta, get_padding
from .bigvgan_alias_free import Activation1d


class AMPBlock1(nn.Module):
    """
    Anti-aliased Multi-Periodicity (AMP) residual block.

    AMPBlock1 has two parallel convolution paths:
    - convs1: Dilated convolutions with varying dilation rates
    - convs2: Additional convolutions with dilation=1

    Args:
        channels: Number of channels
        kernel_size: Convolution kernel size
        dilation: Tuple of dilation rates for convs1
        activation: Activation type ('snake' or 'snakebeta')
        snake_logscale: Whether to use log-scale for activation parameters
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: Tuple[int, ...] = (1, 3, 5),
        activation: str = "snakebeta",
        snake_logscale: bool = True,
        use_anti_aliasing: bool = True,
    ):
        super().__init__()
        self.channels = channels
        self.num_layers = len(dilation) * 2  # convs1 + convs2
        self.use_anti_aliasing = use_anti_aliasing

        # Dilated convolutions
        self.convs1 = []
        for d in dilation:
            padding = get_padding(kernel_size, d)
            self.convs1.append(
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding,
                    dilation=d,
                )
            )

        # Fixed dilation=1 convolutions
        self.convs2 = []
        padding = get_padding(kernel_size, 1)
        for _ in range(len(dilation)):
            self.convs2.append(
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding,
                )
            )

        # Activation functions (one per conv layer)
        # Optionally wrapped in Activation1d for anti-aliasing
        if activation == "snake":
            base_activations = [
                Snake(channels, alpha_logscale=snake_logscale) for _ in range(self.num_layers)
            ]
        elif activation == "snakebeta":
            base_activations = [
                SnakeBeta(channels, alpha_logscale=snake_logscale) for _ in range(self.num_layers)
            ]
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Wrap in anti-aliasing if enabled
        if use_anti_aliasing:
            self.activations = [Activation1d(activation=act) for act in base_activations]
        else:
            self.activations = base_activations

    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass.

        Args:
            x: Input tensor (B, C, T)

        Returns:
            Output tensor (B, C, T)
        """
        acts1 = self.activations[::2]  # Even indices
        acts2 = self.activations[1::2]  # Odd indices

        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, acts1, acts2):
            # Apply activation -> conv -> activation -> conv -> residual
            xt = a1(x)
            xt = c1(xt)
            xt = a2(xt)
            xt = c2(xt)
            x = xt + x  # Residual connection

        return x


class AMPBlock2(nn.Module):
    """
    Simpler AMP block without the extra convs2 layers.

    Args:
        channels: Number of channels
        kernel_size: Convolution kernel size
        dilation: Tuple of dilation rates
        activation: Activation type ('snake' or 'snakebeta')
        snake_logscale: Whether to use log-scale for activation parameters
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: Tuple[int, ...] = (1, 3, 5),
        activation: str = "snakebeta",
        snake_logscale: bool = True,
        use_anti_aliasing: bool = True,
    ):
        super().__init__()
        self.channels = channels
        self.num_layers = len(dilation)
        self.use_anti_aliasing = use_anti_aliasing

        # Dilated convolutions
        self.convs = []
        for d in dilation:
            padding = get_padding(kernel_size, d)
            self.convs.append(
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding,
                    dilation=d,
                )
            )

        # Activation functions
        if activation == "snake":
            base_activations = [
                Snake(channels, alpha_logscale=snake_logscale) for _ in range(self.num_layers)
            ]
        elif activation == "snakebeta":
            base_activations = [
                SnakeBeta(channels, alpha_logscale=snake_logscale) for _ in range(self.num_layers)
            ]
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Wrap in anti-aliasing if enabled
        if use_anti_aliasing:
            self.activations = [Activation1d(activation=act) for act in base_activations]
        else:
            self.activations = base_activations

    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass.

        Args:
            x: Input tensor (B, C, T)

        Returns:
            Output tensor (B, C, T)
        """
        for c, a in zip(self.convs, self.activations):
            # Apply activation -> conv -> residual
            xt = a(x)
            xt = c(xt)
            x = xt + x  # Residual connection

        return x


class BigVGAN(nn.Module):
    """
    BigVGAN neural vocoder for mel-to-waveform generation.

    Args:
        num_mels: Number of mel bands (default: 80)
        upsample_rates: Upsampling rates for each stage
        upsample_initial_channel: Initial channel count
        upsample_kernel_sizes: Kernel sizes for upsampling
        resblock_kernel_sizes: Kernel sizes for residual blocks
        resblock_dilation_sizes: Dilation sizes for residual blocks
        resblock: Residual block type ('1' or '2')
        activation: Activation function ('snake' or 'snakebeta')
        snake_logscale: Whether to use log-scale for Snake activations
        use_tanh_at_final: Whether to apply tanh to final output
    """

    def __init__(
        self,
        num_mels: int = 80,
        upsample_rates: List[int] = [4, 4, 2, 2, 2, 2],
        upsample_initial_channel: int = 1536,
        upsample_kernel_sizes: List[int] = [8, 8, 4, 4, 4, 4],
        resblock_kernel_sizes: List[int] = [3, 7, 11],
        resblock_dilation_sizes: List[List[int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        resblock: str = "1",
        activation: str = "snakebeta",
        snake_logscale: bool = True,
        use_tanh_at_final: bool = False,
        use_anti_aliasing: bool = True,
    ):
        super().__init__()

        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.use_anti_aliasing = use_anti_aliasing

        # Pre-convolution: mel -> initial_channel
        self.conv_pre = nn.Conv1d(
            num_mels,
            upsample_initial_channel,
            kernel_size=7,
            stride=1,
            padding=3,
        )

        # Select residual block type
        if resblock == "1":
            resblock_class = AMPBlock1
        elif resblock == "2":
            resblock_class = AMPBlock2
        else:
            raise ValueError(f"Unknown resblock type: {resblock}")

        # Upsampling layers
        self.ups = []
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            in_channels = upsample_initial_channel // (2**i)
            out_channels = upsample_initial_channel // (2 ** (i + 1))
            padding = (k - u) // 2

            self.ups.append(
                nn.ConvTranspose1d(
                    in_channels,
                    out_channels,
                    kernel_size=k,
                    stride=u,
                    padding=padding,
                )
            )

        # Residual blocks
        self.resblocks = []
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(
                    resblock_class(
                        ch,
                        kernel_size=k,
                        dilation=tuple(d),
                        activation=activation,
                        snake_logscale=snake_logscale,
                        use_anti_aliasing=use_anti_aliasing,
                    )
                )

        # Post-convolution activation and conv
        final_ch = upsample_initial_channel // (2**self.num_upsamples)

        if activation == "snake":
            base_activation_post = Snake(final_ch, alpha_logscale=snake_logscale)
        elif activation == "snakebeta":
            base_activation_post = SnakeBeta(final_ch, alpha_logscale=snake_logscale)
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Wrap in anti-aliasing if enabled
        if use_anti_aliasing:
            self.activation_post = Activation1d(activation=base_activation_post)
        else:
            self.activation_post = base_activation_post

        self.conv_post = nn.Conv1d(
            final_ch,
            1,
            kernel_size=7,
            stride=1,
            padding=3,
        )

        self.use_tanh_at_final = use_tanh_at_final

    def __call__(self, mel: mx.array) -> mx.array:
        """
        Generate waveform from mel spectrogram.

        Args:
            mel: Mel spectrogram (B, num_mels, T) - PyTorch format

        Returns:
            Waveform (B, 1, T * total_upsample_rate) - PyTorch format
        """
        # Convert from (B, C, T) to (B, T, C) for MLX
        x = mel.transpose(0, 2, 1)

        # Pre-conv
        x = self.conv_pre(x)

        # Upsampling + residual blocks
        for i in range(self.num_upsamples):
            # Upsample
            x = self.ups[i](x)

            # Apply residual blocks (average outputs)
            xs = None
            for j in range(self.num_kernels):
                block_idx = i * self.num_kernels + j
                if xs is None:
                    xs = self.resblocks[block_idx](x)
                else:
                    xs = xs + self.resblocks[block_idx](x)

            x = xs / self.num_kernels

        # Post-conv
        x = self.activation_post(x)
        x = self.conv_post(x)

        # Final activation
        if self.use_tanh_at_final:
            x = mx.tanh(x)
        else:
            x = mx.clip(x, -1.0, 1.0)

        # Convert back to (B, C, T) for PyTorch compatibility
        x = x.transpose(0, 2, 1)

        return x


def create_bigvgan(config_path: str = None, use_anti_aliasing: bool = True) -> BigVGAN:
    """
    Create BigVGAN model with default or custom config.

    Args:
        config_path: Optional path to config JSON file
        use_anti_aliasing: Whether to use anti-aliasing in activations (default: True)
                          Set to False for faster inference with slight quality reduction

    Returns:
        BigVGAN model
    """
    if config_path is not None:
        import json

        with open(config_path) as f:
            config = json.load(f)

        return BigVGAN(
            num_mels=config.get("num_mels", 80),
            upsample_rates=config["upsample_rates"],
            upsample_initial_channel=config["upsample_initial_channel"],
            upsample_kernel_sizes=config["upsample_kernel_sizes"],
            resblock_kernel_sizes=config["resblock_kernel_sizes"],
            resblock_dilation_sizes=config["resblock_dilation_sizes"],
            resblock=config.get("resblock", "1"),
            activation=config.get("activation", "snakebeta"),
            snake_logscale=config.get("snake_logscale", True),
            use_tanh_at_final=config.get("use_tanh_at_final", False),
            use_anti_aliasing=use_anti_aliasing,
        )
    else:
        # Default BigVGAN-v2 22kHz config
        return BigVGAN(
            num_mels=80,
            upsample_rates=[4, 4, 2, 2, 2, 2],
            upsample_initial_channel=1536,
            upsample_kernel_sizes=[8, 8, 4, 4, 4, 4],
            resblock_kernel_sizes=[3, 7, 11],
            resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            resblock="1",
            activation="snakebeta",
            snake_logscale=True,
            use_tanh_at_final=False,
            use_anti_aliasing=use_anti_aliasing,
        )
