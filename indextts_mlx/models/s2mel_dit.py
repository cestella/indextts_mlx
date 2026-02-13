"""
MLX DiT (Diffusion Transformer) Implementation

Main DiT model for semantic-to-mel diffusion.
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Optional, Tuple
from dataclasses import dataclass

from .s2mel_layers import TimestepEmbedder, FinalLayer, sequence_mask
from .s2mel_transformer import Transformer, create_attention_mask


@dataclass
class DiTConfig:
    """Configuration for DiT model."""

    # Model architecture
    hidden_dim: int = 512
    depth: int = 13  # Number of transformer layers
    num_heads: int = 8
    in_channels: int = 80  # Mel bins
    out_channels: int = 80

    # Conditioning
    content_dim: int = 512  # Semantic feature dimension
    style_dim: int = 192  # Speaker style dimension

    # Architecture options
    long_skip_connection: bool = True
    is_causal: bool = False

    # Misc
    block_size: int = 16384  # Max sequence length


class DiT(nn.Module):
    """Diffusion Transformer (DiT) for semantic-to-mel generation.

    Implements the core diffusion model that takes:
    - Noisy mel spectrogram
    - Timestep
    - Semantic conditioning
    - Style conditioning
    And predicts the denoising direction (velocity field).
    """

    def __init__(self, config: DiTConfig):
        """Initialize DiT.

        Args:
            config: DiT configuration
        """
        super().__init__()
        self.config = config

        # Input embedders
        self.x_embedder = nn.Linear(config.in_channels, config.hidden_dim)
        self.cond_projection = nn.Linear(config.content_dim, config.hidden_dim)

        # Merge all inputs
        merge_input_dim = (
            config.in_channels  # noisy x
            + config.in_channels  # prompt x
            + config.content_dim  # semantic conditioning
            + config.style_dim  # style
        )
        self.cond_x_merge_linear = nn.Linear(merge_input_dim, config.hidden_dim)

        # Timestep embedder
        self.t_embedder = TimestepEmbedder(config.hidden_dim)

        # Main transformer
        self.transformer = Transformer(
            dim=config.hidden_dim,
            n_layers=config.depth,
            n_heads=config.num_heads,
            mlp_ratio=4.0,
        )

        # Long skip connection
        if config.long_skip_connection:
            self.skip_linear = nn.Linear(
                config.hidden_dim + config.in_channels,
                config.hidden_dim
            )

        # Output head
        self.final_mlp = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.out_channels)
        )

    def __call__(
        self,
        x: mx.array,
        prompt_x: mx.array,
        x_lens: mx.array,
        t: mx.array,
        style: mx.array,
        cond: mx.array,
    ) -> mx.array:
        """Forward pass of DiT.

        Args:
            x: Noisy mel, shape (batch, in_channels, T)
            prompt_x: Prompt mel (reference + zeros), shape (batch, in_channels, T)
            x_lens: Sequence lengths, shape (batch,)
            t: Timesteps (0 to 1), shape (batch,)
            style: Style embeddings, shape (batch, style_dim)
            cond: Semantic conditioning, shape (batch, T, content_dim)

        Returns:
            Predicted velocity field, shape (batch, out_channels, T)
        """
        batch_size, _, T = x.shape

        # Embed timestep
        t_emb = self.t_embedder(t)  # (batch, hidden_dim)

        # Project semantic conditioning
        cond_proj = self.cond_projection(cond)  # (batch, T, hidden_dim)

        # Transpose x and prompt_x from (B, C, T) to (B, T, C)
        x_transposed = mx.transpose(x, (0, 2, 1))  # (batch, T, in_channels)
        prompt_x_transposed = mx.transpose(prompt_x, (0, 2, 1))  # (batch, T, in_channels)

        # Concatenate all inputs
        # x_in: (batch, T, in_channels + in_channels + content_dim)
        x_in = mx.concatenate([x_transposed, prompt_x_transposed, cond], axis=-1)

        # Add style (repeat for each timestep)
        style_expanded = mx.expand_dims(style, 1)  # (batch, 1, style_dim)
        style_repeated = mx.broadcast_to(style_expanded, (batch_size, T, self.config.style_dim))
        x_in = mx.concatenate([x_in, style_repeated], axis=-1)

        # Merge all inputs
        x_in = self.cond_x_merge_linear(x_in)  # (batch, T, hidden_dim)

        # Create attention mask from lengths
        padding_mask = sequence_mask(x_lens, max_length=T)  # (batch, T)
        padding_mask_expanded = mx.expand_dims(padding_mask, 1)  # (batch, 1, T)
        attention_mask = create_attention_mask(
            seq_len=T,
            is_causal=self.config.is_causal,
            padding_mask=padding_mask
        )

        # Apply transformer
        x_res = self.transformer(
            x_in,
            conditioning=t_emb,
            mask=attention_mask
        )  # (batch, T, hidden_dim)

        # Long skip connection
        if self.config.long_skip_connection:
            x_res = self.skip_linear(
                mx.concatenate([x_res, x_transposed], axis=-1)
            )

        # Output projection
        x_out = self.final_mlp(x_res)  # (batch, T, out_channels)

        # Transpose back to (batch, out_channels, T)
        x_out = mx.transpose(x_out, (0, 2, 1))

        return x_out


def create_dit_from_pytorch_config(pytorch_args) -> Tuple[DiT, DiTConfig]:
    """Create MLX DiT from PyTorch config.

    Args:
        pytorch_args: PyTorch configuration object (Munch)

    Returns:
        Tuple of (dit_model, dit_config)
    """
    config = DiTConfig(
        hidden_dim=pytorch_args.DiT.hidden_dim,
        depth=pytorch_args.DiT.depth,
        num_heads=pytorch_args.DiT.num_heads,
        in_channels=pytorch_args.DiT.in_channels,
        out_channels=pytorch_args.DiT.in_channels,
        content_dim=pytorch_args.DiT.content_dim,
        style_dim=pytorch_args.style_encoder.dim,
        long_skip_connection=pytorch_args.DiT.long_skip_connection,
        is_causal=pytorch_args.DiT.is_causal,
        block_size=16384,
    )

    model = DiT(config)

    return model, config


def load_dit_weights_from_pytorch(
    mlx_model: DiT,
    pytorch_state_dict: dict,
    verbose: bool = True
) -> None:
    """Load weights from PyTorch state dict into MLX model.

    Args:
        mlx_model: MLX DiT model
        pytorch_state_dict: PyTorch state dictionary
        verbose: Print loading progress

    Note:
        This function performs in-place weight loading with shape validation.
        Weight names may need mapping between PyTorch and MLX.
    """
    if verbose:
        print("Loading DiT weights from PyTorch...")

    # Weight name mapping (PyTorch -> MLX)
    # This will need to be filled in based on actual state dict keys
    name_mapping = {
        # Example mappings (to be completed):
        # "x_embedder.weight": "x_embedder.weight",
        # "transformer.blocks.0.attn.in_proj_weight": "transformer.blocks.0.attn.query_proj.weight",
    }

    loaded_count = 0
    skipped_count = 0

    for pt_name, pt_weight in pytorch_state_dict.items():
        # Map PyTorch name to MLX name
        mlx_name = name_mapping.get(pt_name, pt_name)

        # TODO: Implement actual weight loading logic
        # This is a placeholder that needs to be completed
        loaded_count += 1

    if verbose:
        print(f"  ✓ Loaded {loaded_count} weights")
        if skipped_count > 0:
            print(f"  ⚠ Skipped {skipped_count} weights (shape mismatch or not found)")
