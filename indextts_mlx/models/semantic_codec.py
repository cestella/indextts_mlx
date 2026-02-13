"""
Semantic Codec (RepCodec) implementation in MLX.

RepCodec quantizes W2V-BERT semantic embeddings into discrete codes using:
1. VocosBackbone encoder (ConvNeXt-based transformer)
2. ResidualVQ with Factorized Vector Quantization
"""

import mlx.core as mx
import mlx.nn as nn
import math
from typing import Optional, Tuple


class ConvNeXtBlock(nn.Module):
    """ConvNeXt block adapted for 1D audio features.

    Structure:
    - Depthwise Conv1d (kernel=7, groups=dim)
    - LayerNorm
    - Pointwise Linear (expand to intermediate_dim)
    - GELU
    - Pointwise Linear (project back to dim)
    - Layer scaling (gamma parameter)
    - Residual connection
    """

    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        layer_scale_init_value: float,
    ):
        super().__init__()

        # Depthwise convolution (groups=dim means depthwise)
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)

        # Layer norm
        self.norm = nn.LayerNorm(dim, eps=1e-6)

        # Pointwise convolutions (implemented as linear layers)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.pwconv2 = nn.Linear(intermediate_dim, dim)

        # Layer scaling parameter
        if layer_scale_init_value > 0:
            self.gamma = mx.ones(dim) * layer_scale_init_value
        else:
            self.gamma = None

    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: (B, T, C) in MLX convention
        Returns:
            output: (B, T, C)
        """
        residual = x

        # Depthwise conv expects (B, T, C)
        x = self.dwconv(x)

        # Layer norm
        x = self.norm(x)

        # Pointwise convs with GELU
        x = self.pwconv1(x)
        x = nn.gelu(x)
        x = self.pwconv2(x)

        # Layer scaling
        if self.gamma is not None:
            x = self.gamma * x

        # Residual connection
        x = residual + x

        return x


class VocosBackbone(nn.Module):
    """Vocos backbone with ConvNeXt blocks.

    Used as encoder/decoder in RepCodec.
    """

    def __init__(
        self,
        input_channels: int,
        dim: int,
        intermediate_dim: int,
        num_layers: int,
        layer_scale_init_value: Optional[float] = None,
    ):
        super().__init__()

        self.input_channels = input_channels

        # Initial embedding
        self.embed = nn.Conv1d(input_channels, dim, kernel_size=7, padding=3)

        # Initial norm
        self.norm = nn.LayerNorm(dim, eps=1e-6)

        # ConvNeXt blocks
        layer_scale_init_value = layer_scale_init_value or 1 / num_layers
        self.convnext = [
            ConvNeXtBlock(
                dim=dim,
                intermediate_dim=intermediate_dim,
                layer_scale_init_value=layer_scale_init_value,
            )
            for _ in range(num_layers)
        ]

        # Final layer norm
        self.final_layer_norm = nn.LayerNorm(dim, eps=1e-6)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: (B, T, C)
        Returns:
            output: (B, T, dim)
        """
        # Embed
        x = self.embed(x)

        # Initial norm
        x = self.norm(x)

        # ConvNeXt blocks
        for block in self.convnext:
            x = block(x)

        # Final norm
        x = self.final_layer_norm(x)

        return x


class FactorizedVectorQuantize(nn.Module):
    """Factorized Vector Quantization with L2 normalization.

    Projects input to codebook_dim (if different), quantizes using
    L2-normalized Euclidean distance, then projects back.

    IMPORTANT: PyTorch version expects (B, D, T) format!
    """

    def __init__(
        self,
        input_dim: int,
        codebook_size: int,
        codebook_dim: int,
        use_l2_normlize: bool = True,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.use_l2_normlize = use_l2_normlize

        # Projection layers (with weight norm in PyTorch, but MLX doesn't have it)
        # We'll just use regular Conv1d
        if input_dim != codebook_dim:
            self.in_project = nn.Conv1d(input_dim, codebook_dim, kernel_size=1)
            self.out_project = nn.Conv1d(codebook_dim, input_dim, kernel_size=1)
        else:
            self.in_project = None
            self.out_project = None

        # Codebook
        self.codebook = mx.zeros((codebook_size, codebook_dim))

    def __call__(self, z: mx.array) -> Tuple[mx.array, mx.array]:
        """
        Args:
            z: (B, D, T) input (PyTorch channels format)
        Returns:
            z_q: (B, D, T) quantized output
            indices: (B, T) codebook indices
        """
        # Project to codebook dimension
        # MLX Conv1d expects (B, T, C), so transpose
        if self.in_project is not None:
            z_e = self.in_project(z.transpose(0, 2, 1))  # (B, T, D) -> (B, T, codebook_dim)
            z_e = z_e.transpose(0, 2, 1)  # Back to (B, codebook_dim, T)
        else:
            z_e = z

        # Decode latents to get quantized output
        z_q, indices = self.decode_latents(z_e)

        # Straight-through estimator
        z_q = z_e + mx.stop_gradient(z_q - z_e)

        # Project back to input dimension
        if self.out_project is not None:
            z_q = self.out_project(z_q.transpose(0, 2, 1))  # (B, T, D)
            z_q = z_q.transpose(0, 2, 1)  # Back to (B, D, T)

        return z_q, indices

    def decode_latents(self, latents: mx.array) -> Tuple[mx.array, mx.array]:
        """Quantize latents using codebook."""
        # latents: (B, D, T)
        B, D, T = latents.shape

        # Transpose to (B, T, D) then reshape to (B*T, D)
        encodings = latents.transpose(0, 2, 1).reshape(-1, D)
        codebook = self.codebook

        # L2 normalize if enabled
        if self.use_l2_normlize:
            # Normalize encodings
            encodings = encodings / (mx.linalg.norm(encodings, axis=-1, keepdims=True) + 1e-8)
            # Normalize codebook
            codebook = codebook / (mx.linalg.norm(codebook, axis=-1, keepdims=True) + 1e-8)

        # Compute Euclidean distance: ||e - c||^2 = ||e||^2 - 2*e·c + ||c||^2
        # When L2 normalized, this is proportional to cosine distance
        encodings_sq = mx.sum(encodings ** 2, axis=-1, keepdims=True)  # (B*T, 1)
        codebook_sq = mx.sum(codebook ** 2, axis=-1, keepdims=True).T  # (1, K)
        dot_product = mx.matmul(encodings, codebook.T)  # (B*T, K)

        dist = encodings_sq - 2 * dot_product + codebook_sq  # (B*T, K)

        # Find nearest codebook entry (minimize distance = maximize negative distance)
        indices = mx.argmax(-dist, axis=-1)  # (B*T,)
        indices = indices.reshape(B, T)

        # Decode codes: (B, T, D) then transpose to (B, D, T)
        z_q = self.decode_code(indices).transpose(0, 2, 1)

        return z_q, indices

    def decode_code(self, indices: mx.array) -> mx.array:
        """Convert indices to embeddings."""
        # indices: (B, T)
        # codebook: (K, D)
        # output: (B, T, D)
        return self.codebook[indices]

    def vq2emb(self, indices: mx.array, out_proj: bool = True) -> mx.array:
        """Convert VQ codes to embeddings (for GPT inference)."""
        emb = self.decode_code(indices)  # Returns (B, T, D)

        if out_proj and self.out_project is not None:
            # emb is (B, T, D) - MLX Conv1d expects (B, L, C_in) format
            # No need to transpose - pass directly
            emb = self.out_project(emb)  # (B, T, codebook_dim) -> (B, T, input_dim)

        return emb


class ResidualVQ(nn.Module):
    """Residual Vector Quantization.

    Applies multiple quantizers in sequence, where each quantizer
    encodes the residual from the previous quantizers.

    IMPORTANT: Expects (B, D, T) format to match PyTorch!
    """

    def __init__(
        self,
        input_dim: int,
        num_quantizers: int,
        codebook_size: int,
        codebook_dim: int,
        use_l2_normlize: bool = True,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim

        # Create quantizers
        self.quantizers = [
            FactorizedVectorQuantize(
                input_dim=input_dim,
                codebook_size=codebook_size,
                codebook_dim=codebook_dim,
                use_l2_normlize=use_l2_normlize,
            )
            for _ in range(num_quantizers)
        ]

    def __call__(self, z: mx.array) -> Tuple[mx.array, mx.array]:
        """
        Args:
            z: (B, D, T) input
        Returns:
            quantized_out: (B, D, T) quantized output
            all_indices: (num_quantizers, B, T) indices for all quantizers
        """
        quantized_out = mx.zeros_like(z)
        residual = z

        all_indices = []

        for quantizer in self.quantizers:
            z_q, indices = quantizer(residual)

            quantized_out = quantized_out + z_q
            residual = residual - z_q

            all_indices.append(indices)

        # Stack indices: (num_quantizers, B, T)
        all_indices = mx.stack(all_indices, axis=0)

        return quantized_out, all_indices

    def vq2emb(self, vq: mx.array) -> mx.array:
        """Convert VQ codes to embeddings.

        Args:
            vq: (num_quantizers, B, T) or (1, B, T) indices
        Returns:
            embeddings: (B, T, D)
        """
        quantized_out = None

        for idx, quantizer in enumerate(self.quantizers):
            if idx >= vq.shape[0]:
                break

            emb = quantizer.vq2emb(vq[idx], out_proj=True)

            if quantized_out is None:
                quantized_out = emb
            else:
                quantized_out = quantized_out + emb

        return quantized_out


class RepCodec(nn.Module):
    """RepCodec: Representation Codec for semantic features.

    Encodes W2V-BERT semantic embeddings into discrete codes using:
    1. VocosBackbone encoder (12 ConvNeXt layers)
    2. ResidualVQ (1 quantizer with 8192 codebook size)
    """

    def __init__(
        self,
        codebook_size: int = 8192,
        hidden_size: int = 1024,
        codebook_dim: int = 8,
        vocos_dim: int = 384,
        vocos_intermediate_dim: int = 2048,
        vocos_num_layers: int = 12,
        num_quantizers: int = 1,
    ):
        super().__init__()

        self.codebook_size = codebook_size
        self.hidden_size = hidden_size
        self.codebook_dim = codebook_dim
        self.vocos_dim = vocos_dim
        self.vocos_intermediate_dim = vocos_intermediate_dim
        self.vocos_num_layers = vocos_num_layers
        self.num_quantizers = num_quantizers

        # Encoder
        self.encoder = [
            VocosBackbone(
                input_channels=hidden_size,
                dim=vocos_dim,
                intermediate_dim=vocos_intermediate_dim,
                num_layers=vocos_num_layers,
            ),
            nn.Linear(vocos_dim, hidden_size),
        ]

        # Quantizer
        self.quantizer = ResidualVQ(
            input_dim=hidden_size,
            num_quantizers=num_quantizers,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            use_l2_normlize=True,
        )

    def quantize(self, x: mx.array) -> Tuple[mx.array, mx.array]:
        """Quantize semantic features to discrete codes.

        Args:
            x: (B, T, 1024) semantic features from W2V-BERT layer 17
        Returns:
            all_indices: (1, B, T) or (B, T) discrete codes (if num_quantizers=1, squeeze)
            quantized_out: (B, T, 1024) reconstructed features
        """
        # Encoder expects (B, T, C), outputs (B, T, C)
        for layer in self.encoder:
            x = layer(x)

        # Transpose to (B, C, T) for quantizer (to match PyTorch)
        x = x.transpose(0, 2, 1)

        # Quantize (expects (B, C, T))
        quantized_out, all_indices = self.quantizer(x)

        # Transpose quantized output back to (B, T, C)
        quantized_out = quantized_out.transpose(0, 2, 1)

        # If only one quantizer, squeeze the first dimension
        if self.num_quantizers == 1:
            return all_indices.squeeze(0), quantized_out

        return all_indices, quantized_out

    def vq2emb(self, codes: mx.array) -> mx.array:
        """Convert discrete codes back to embeddings.

        Args:
            codes: (B, T) or (1, B, T) discrete codes
        Returns:
            embeddings: (B, T, 1024) reconstructed semantic embeddings
        """
        # Ensure codes are (num_quantizers, B, T)
        if codes.ndim == 2:
            codes = mx.expand_dims(codes, 0)  # (1, B, T)

        # Decode through quantizer
        return self.quantizer.vq2emb(codes)


def create_semantic_codec(
    codebook_size: int = 8192,
    hidden_size: int = 1024,
    codebook_dim: int = 8,
    vocos_dim: int = 384,
    vocos_intermediate_dim: int = 2048,
    vocos_num_layers: int = 12,
    num_quantizers: int = 1,
) -> RepCodec:
    """Create RepCodec semantic codec with default parameters."""
    return RepCodec(
        codebook_size=codebook_size,
        hidden_size=hidden_size,
        codebook_dim=codebook_dim,
        vocos_dim=vocos_dim,
        vocos_intermediate_dim=vocos_intermediate_dim,
        vocos_num_layers=vocos_num_layers,
        num_quantizers=num_quantizers,
    )
