"""
MLX Conditional Flow Matching (CFM) Implementation

ODE-based diffusion wrapper around DiT for semantic-to-mel generation.
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Optional, Union
from dataclasses import dataclass

from .s2mel_dit_v2 import DiTV2


@dataclass
class CFMConfig:
    """Configuration for CFM."""

    sigma_min: float = 1e-6
    n_timesteps: int = 25
    temperature: float = 1.0
    inference_cfg_rate: float = 0.0  # Classifier-free guidance rate


class CFM(nn.Module):
    """Conditional Flow Matching with Euler ODE solver.

    Implements flow-based diffusion for mel spectrogram generation.
    """

    def __init__(self, estimator: DiTV2, config: CFMConfig):
        """Initialize CFM.

        Args:
            estimator: DiTV2 model for velocity estimation
            config: CFM configuration
        """
        super().__init__()
        self.estimator = estimator
        self.config = config
        self.sigma_min = config.sigma_min
        self.in_channels = estimator.config.in_channels

    def inference(
        self,
        mu: mx.array,
        x_lens: mx.array,
        prompt: mx.array,
        style: mx.array,
        f0: Optional[mx.array] = None,
        n_timesteps: Optional[int] = None,
        temperature: Optional[float] = None,
        inference_cfg_rate: Optional[float] = None,
        z: Optional[mx.array] = None,
    ) -> mx.array:
        """Forward diffusion inference.

        Args:
            mu: Semantic conditioning, shape (batch, T, content_dim)
            x_lens: Sequence lengths, shape (batch,)
            prompt: Prompt mel, shape (batch, in_channels, prompt_len)
            style: Style embeddings, shape (batch, style_dim)
            f0: F0 contours (not used in current implementation)
            n_timesteps: Number of ODE steps (default: config.n_timesteps)
            temperature: Noise temperature (default: config.temperature)
            inference_cfg_rate: CFG rate (default: config.inference_cfg_rate)
            z: Initial noise (optional), shape (batch, in_channels, T). If None, samples from N(0, temperature^2)

        Returns:
            Generated mel spectrogram, shape (batch, in_channels, T)
        """
        if n_timesteps is None:
            n_timesteps = self.config.n_timesteps
        if temperature is None:
            temperature = self.config.temperature
        if inference_cfg_rate is None:
            inference_cfg_rate = self.config.inference_cfg_rate

        batch_size, T, _ = mu.shape

        # Sample initial noise (or use provided noise for testing)
        if z is None:
            z = mx.random.normal((batch_size, self.in_channels, T)) * temperature

        # Create timestep schedule
        t_span = mx.linspace(0, 1, n_timesteps + 1)

        # Solve ODE
        return self.solve_euler(z, x_lens, prompt, mu, style, f0, t_span, inference_cfg_rate)

    def solve_euler(
        self,
        x: mx.array,
        x_lens: mx.array,
        prompt: mx.array,
        mu: mx.array,
        style: mx.array,
        f0: Optional[mx.array],
        t_span: mx.array,
        inference_cfg_rate: float = 0.0,
    ) -> mx.array:
        """Euler ODE solver for flow matching.

        Args:
            x: Initial noise, shape (batch, in_channels, T)
            x_lens: Sequence lengths, shape (batch,)
            prompt: Prompt mel, shape (batch, in_channels, prompt_len)
            mu: Semantic conditioning, shape (batch, T, content_dim)
            style: Style embeddings, shape (batch, style_dim)
            f0: F0 contours (not used)
            t_span: Timestep schedule, shape (n_timesteps + 1,)
            inference_cfg_rate: CFG rate

        Returns:
            Final generated mel, shape (batch, in_channels, T)
        """
        batch_size, in_channels, T = x.shape

        # Prepare prompt
        prompt_len = min(prompt.shape[-1], T)
        prompt_x = mx.zeros_like(x)
        if prompt_len > 0:
            prompt_x[:, :, :prompt_len] = prompt[:, :, :prompt_len]

        # Zero out prompt region in x
        if prompt_len > 0:
            x[:, :, :prompt_len] = 0

        # Initialize timestep - matching PyTorch
        t = t_span[0]

        # Euler integration loop
        for step in range(1, len(t_span)):
            dt = t_span[step] - t_span[step - 1]

            if inference_cfg_rate > 0:
                # Classifier-free guidance: run model twice (with and without conditioning)
                # Stack inputs for batched processing
                stacked_x = mx.concatenate([x, x], axis=0)
                stacked_prompt_x = mx.concatenate([prompt_x, mx.zeros_like(prompt_x)], axis=0)
                stacked_style = mx.concatenate([style, mx.zeros_like(style)], axis=0)
                stacked_mu = mx.concatenate([mu, mx.zeros_like(mu)], axis=0)
                # Create timestep tensor - matching PyTorch which uses t.unsqueeze(0) twice
                t_tensor = mx.array([float(t)])
                stacked_t = mx.concatenate([t_tensor, t_tensor], axis=0)

                # Forward pass - PyTorch doesn't stack x_lens
                stacked_dphi_dt = self.estimator(
                    stacked_x,
                    stacked_prompt_x,
                    x_lens,  # Don't stack x_lens - PyTorch only passes original
                    stacked_t,
                    stacked_style,
                    stacked_mu,
                )

                # Split and apply CFG - matching PyTorch's chunk
                dphi_dt = stacked_dphi_dt[:batch_size]
                cfg_dphi_dt = stacked_dphi_dt[batch_size:]
                dphi_dt = (1.0 + inference_cfg_rate) * dphi_dt - inference_cfg_rate * cfg_dphi_dt
            else:
                # Standard forward pass - matching PyTorch which uses t.unsqueeze(0)
                t_batch = mx.array([float(t)])
                dphi_dt = self.estimator(x, prompt_x, x_lens, t_batch, style, mu)

            # Euler step: x = x + dt * dphi_dt
            x = x + dt * dphi_dt

            # Update timestep - matching PyTorch which does t = t + dt
            t = t + dt

            # Recalculate dt for next step to avoid float accumulation errors
            # Matching PyTorch line 112
            if step < len(t_span) - 1:
                dt = float(t_span[step + 1] - t)

            # Keep prompt region zeroed - matching PyTorch line 113
            if prompt_len > 0:
                x[:, :, :prompt_len] = 0

        return x


def create_cfm_from_dit(dit: DiTV2, n_timesteps: int = 25) -> CFM:
    """Create CFM wrapper around DiTV2 model.

    Args:
        dit: DiTV2 model
        n_timesteps: Number of ODE steps for inference

    Returns:
        CFM model
    """
    config = CFMConfig(
        sigma_min=1e-6,
        n_timesteps=n_timesteps,
        temperature=1.0,
        inference_cfg_rate=0.0,
    )

    return CFM(dit, config)
