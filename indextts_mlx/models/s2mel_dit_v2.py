"""
MLX DiT v2 - Matching Actual IndexTTS2 Architecture

Uses:
- gpt-fast transformer with wqkv + SwiGLU
- U-ViT skip connections
- WaveNet final layer
- Weight normalization support
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Optional
from dataclasses import dataclass

from .s2mel_layers import TimestepEmbedder, sequence_mask, modulate
from .s2mel_gptfast_transformer import GPTFastTransformer
from .bigvgan_wavenet import WaveNet


@dataclass
class DiTConfigV2:
    """Configuration for DiT model matching actual architecture."""

    # Model architecture
    hidden_dim: int = 512
    depth: int = 13
    num_heads: int = 8
    in_channels: int = 80  # Mel bins
    out_channels: int = 80

    # Conditioning
    content_dim: int = 512
    style_dim: int = 192

    # Architecture options
    long_skip_connection: bool = True
    uvit_skip_connection: bool = True
    is_causal: bool = False
    final_layer_type: str = "wavenet"  # "wavenet" or "mlp"

    # WaveNet config (if final_layer_type == "wavenet")
    wavenet_hidden_dim: int = 512
    wavenet_kernel_size: int = 5
    wavenet_dilation_rate: int = 1
    wavenet_num_layers: int = 8
    wavenet_dropout: float = 0.2

    # FFN config
    ffn_hidden_dim: int = 1536  # ~3x hidden_dim

    # Misc
    block_size: int = 8192


class FinalLayer(nn.Module):
    """Final layer with AdaLN modulation."""

    def __init__(self, hidden_size: int, out_channels: int):
        """Initialize final layer.

        Args:
            hidden_size: Hidden dimension
            out_channels: Output channels
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.out_channels = out_channels

        self.norm_final = nn.LayerNorm(hidden_size, affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size)
        )

    def __call__(self, x: mx.array, c: mx.array) -> mx.array:
        """Apply final layer.

        Args:
            x: Input of shape (batch, seq_len, hidden_size)
            c: Conditioning of shape (batch, hidden_size)

        Returns:
            Output of shape (batch, seq_len, out_channels)
        """
        modulation = self.adaLN_modulation(c)
        # FinalLayer uses different convention: (shift, scale) and x * (1 + scale) + shift
        shift, scale = mx.split(modulation, 2, axis=-1)

        x_norm = self.norm_final(x)
        # Expand for broadcasting
        shift = mx.expand_dims(shift, 1)  # (batch, 1, hidden)
        scale = mx.expand_dims(scale, 1)  # (batch, 1, hidden)
        # FinalLayer modulate: x * (1 + scale) + shift
        x_mod = x_norm * (1 + scale) + shift

        return self.linear(x_mod)


class DiTV2(nn.Module):
    """DiT v2 matching actual IndexTTS2 architecture."""

    def __init__(self, config: DiTConfigV2):
        """Initialize DiT v2.

        Args:
            config: DiT configuration
        """
        super().__init__()
        self.config = config

        # Input embedders (with weight norm in original)
        self.x_embedder = nn.Linear(config.in_channels, config.hidden_dim)
        self.cond_projection = nn.Linear(config.content_dim, config.hidden_dim)

        # Merge all inputs
        # Input dims: x(80) + prompt_x(80) + cond(512) + style(192) = 864
        merge_input_dim = (
            config.in_channels
            + config.in_channels
            + config.content_dim
            + config.style_dim
        )
        self.cond_x_merge_linear = nn.Linear(merge_input_dim, config.hidden_dim)

        # Timestep embedders
        self.t_embedder = TimestepEmbedder(config.hidden_dim)

        # Main transformer with U-ViT
        self.transformer = GPTFastTransformer(
            dim=config.hidden_dim,
            n_layers=config.depth,
            n_heads=config.num_heads,
            hidden_dim=config.ffn_hidden_dim,
            use_uvit_skip=config.uvit_skip_connection,
            block_size=config.block_size,
            rope_base=10000,  # Standard RoPE base
        )

        # Long skip connection
        if config.long_skip_connection:
            self.skip_linear = nn.Linear(
                config.hidden_dim + config.in_channels,
                config.hidden_dim
            )

        # Final layer
        if config.final_layer_type == "wavenet":
            # WaveNet final layer
            self.t_embedder2 = TimestepEmbedder(config.wavenet_hidden_dim)

            self.conv1 = nn.Linear(config.hidden_dim, config.wavenet_hidden_dim)

            self.wavenet = WaveNet(
                hidden_channels=config.wavenet_hidden_dim,
                kernel_size=config.wavenet_kernel_size,
                dilation_rate=config.wavenet_dilation_rate,
                n_layers=config.wavenet_num_layers,
                gin_channels=config.wavenet_hidden_dim,
                p_dropout=config.wavenet_dropout,
                causal=False,
            )

            self.final_layer = FinalLayer(config.wavenet_hidden_dim, config.wavenet_hidden_dim)

            self.res_projection = nn.Linear(config.hidden_dim, config.wavenet_hidden_dim)

            self.conv2 = nn.Conv1d(in_channels=config.wavenet_hidden_dim, out_channels=config.in_channels, kernel_size=1)
        else:
            # Simple MLP final layer
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
        """Forward pass of DiT v2.

        Args:
            x: Noisy mel, shape (batch, in_channels, T)
            prompt_x: Prompt mel, shape (batch, in_channels, T)
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
        x_transposed = mx.transpose(x, (0, 2, 1))
        prompt_x_transposed = mx.transpose(prompt_x, (0, 2, 1))

        # Concatenate all inputs (use cond_proj, NOT cond!)
        x_in = mx.concatenate([x_transposed, prompt_x_transposed, cond_proj], axis=-1)

        # Add style
        style_expanded = mx.expand_dims(style, 1)
        style_repeated = mx.broadcast_to(style_expanded, (batch_size, T, self.config.style_dim))
        x_in = mx.concatenate([x_in, style_repeated], axis=-1)

        # Merge inputs
        x_in = self.cond_x_merge_linear(x_in)  # (batch, T, hidden_dim)

        # Apply transformer
        #DEBUG# print(f"[DEBUG] Before transformer: x_in.shape={x_in.shape}, t_emb.shape={t_emb.shape}")
        x_res = self.transformer(x_in, conditioning=t_emb, mask=None)
        #DEBUG# print(f"[DEBUG] After transformer: x_res.shape={x_res.shape}")

        # Long skip connection
        if self.config.long_skip_connection:
            x_res = self.skip_linear(
                mx.concatenate([x_res, x_transposed], axis=-1)
            )

        # Final layer
        if self.config.final_layer_type == "wavenet":
            # WaveNet path
            x_out = self.conv1(x_res)  # (batch, T, wavenet_hidden_dim)

            # Global conditioning from timestep
            # PyTorch passes g as (B, C, 1) to WaveNet, we transpose to (B, 1, C) for MLX
            t2_emb = self.t_embedder2(t)  # (batch, wavenet_hidden_dim)
            g = mx.expand_dims(t2_emb, 1)  # (batch, 1, wavenet_hidden_dim)

            # WaveNet (expects channels-last, so g is (B, 1, C))
            x_out = self.wavenet(x_out, x_mask=None, g=g)  # (batch, T, wavenet_hidden_dim)

            # Add residual projection from transformer output
            res_proj = self.res_projection(x_res)  # (batch, T, wavenet_hidden_dim)
            x_out = x_out + res_proj

            # Final layer with AdaLN
            x_out = self.final_layer(x_out, t_emb)  # (batch, T, wavenet_hidden_dim)

            # MLX Conv1d is channels-last, so input (B, T, C) → output (B, T, out_channels)
            x_out = self.conv2(x_out)  # (batch, T, out_channels)

            # Transpose to match PyTorch output format (B, C, T)
            x_out = mx.transpose(x_out, (0, 2, 1))  # (batch, out_channels, T)
        else:
            # Simple MLP path
            x_out = self.final_mlp(x_res)
            x_out = mx.transpose(x_out, (0, 2, 1))

        return x_out


def create_dit_v2_from_config(config_dict: dict) -> tuple[DiTV2, DiTConfigV2]:
    """Create DiT v2 from config dictionary.

    Args:
        config_dict: Configuration dictionary from YAML

    Returns:
        Tuple of (dit_model, dit_config)
    """
    dit_cfg = config_dict['DiT']
    wavenet_cfg = config_dict.get('wavenet', {})

    config = DiTConfigV2(
        hidden_dim=dit_cfg['hidden_dim'],
        depth=dit_cfg['depth'],
        num_heads=dit_cfg['num_heads'],
        in_channels=dit_cfg['in_channels'],
        out_channels=dit_cfg['in_channels'],
        content_dim=dit_cfg['content_dim'],
        style_dim=config_dict['style_encoder']['dim'],
        long_skip_connection=dit_cfg['long_skip_connection'],
        uvit_skip_connection=dit_cfg.get('uvit_skip_connection', True),
        is_causal=dit_cfg['is_causal'],
        final_layer_type=dit_cfg.get('final_layer_type', 'wavenet'),
        wavenet_hidden_dim=wavenet_cfg.get('hidden_dim', 512),
        wavenet_kernel_size=wavenet_cfg.get('kernel_size', 5),
        wavenet_dilation_rate=wavenet_cfg.get('dilation_rate', 1),
        wavenet_num_layers=wavenet_cfg.get('num_layers', 8),
        wavenet_dropout=wavenet_cfg.get('p_dropout', 0.2),
        ffn_hidden_dim=1536,  # ~3x hidden_dim, from checkpoint analysis
        block_size=dit_cfg.get('block_size', 8192),
    )

    model = DiTV2(config)

    return model, config
