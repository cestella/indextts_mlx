"""
Wav2Vec2-BERT (W2V-BERT) implementation in MLX.

A 24-layer Conformer-based transformer for semantic feature extraction from audio.
"""

import mlx.core as mx
import mlx.nn as nn
import math
from typing import Optional, Tuple


class Wav2Vec2BertRelativePositionBias(nn.Module):
    """Relative position bias for attention."""

    def __init__(self, num_heads: int, left_max: int = 64, right_max: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.left_max = left_max
        self.right_max = right_max

        # Total positions: left_max + 1 + right_max
        num_positions = left_max + 1 + right_max
        self.embeddings = mx.zeros((num_positions, num_heads))

    def __call__(self, query_length: int, key_length: int) -> mx.array:
        """Compute relative position bias.

        Returns:
            bias: (num_heads, query_length, key_length)
        """
        # Simplified: return zeros for now (TODO: implement proper relative pos)
        return mx.zeros((self.num_heads, query_length, key_length))


class Wav2Vec2BertSelfAttention(nn.Module):
    """Multi-head self-attention with relative position bias."""

    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # HF uses linear_q, linear_k, linear_v, linear_out (all with bias)
        self.linear_q = nn.Linear(hidden_size, hidden_size, bias=True)
        self.linear_k = nn.Linear(hidden_size, hidden_size, bias=True)
        self.linear_v = nn.Linear(hidden_size, hidden_size, bias=True)
        self.linear_out = nn.Linear(hidden_size, hidden_size, bias=True)

        # Relative position embedding (73 positions, 64 per head)
        # 73 = left_max(64) + 1 + right_max(8)
        self.distance_embedding = mx.zeros((73, num_heads))

        self.dropout = dropout

    def __call__(self, x: mx.array, attention_mask: Optional[mx.array] = None) -> mx.array:
        """
        Args:
            x: (B, T, C)
            attention_mask: (B, T) - 1 for real, 0 for padding
        Returns:
            output: (B, T, C)
        """
        B, T, C = x.shape

        # Project to Q, K, V
        Q = self.linear_q(x).reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        K = self.linear_k(x).reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        V = self.linear_v(x).reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Scaled dot-product attention
        scores = mx.matmul(Q, K.transpose(0, 1, 3, 2)) / math.sqrt(self.head_dim)

        # Add relative position bias (relative_key method)
        # Position IDs for query and key
        query_length, key_length = T, T
        position_ids_l = mx.arange(query_length).reshape(-1, 1)  # (query_len, 1)
        position_ids_r = mx.arange(key_length).reshape(1, -1)  # (1, key_len)

        # Compute distance and clamp to valid range
        distance = position_ids_r - position_ids_l  # (query_len, key_len)
        left_max = 64
        right_max = 8
        distance = mx.clip(distance, -left_max, right_max)

        # Shift by left_max to get indices [0, 72]
        distance_indices = distance + left_max  # (query_len, key_len)

        # Lookup positional embeddings
        # distance_embedding: (73, head_dim) -> index with distance_indices
        # Result: (query_len, key_len, head_dim)
        positional_embedding = self.distance_embedding[distance_indices]

        # Compute relative position attention weights
        # Q: (B, H, T, D), positional_embedding: (T, T, D)
        # einsum "bhld,lrd->bhlr"
        relative_position_attn_weights = mx.sum(
            Q[:, :, :, None, :] * positional_embedding[None, None, :, :, :], axis=-1
        )  # (B, H, query_len, key_len)

        # Add to scores (with scaling)
        scores = scores + (relative_position_attn_weights / math.sqrt(self.head_dim))

        # Apply attention mask if provided
        if attention_mask is not None:
            # attention_mask: (B, T) -> (B, 1, 1, T)
            mask = mx.expand_dims(mx.expand_dims(attention_mask, 1), 1)
            scores = mx.where(mask, scores, -1e9)

        attn_weights = mx.softmax(scores, axis=-1)

        # Apply attention to values
        attn_output = mx.matmul(attn_weights, V)  # (B, H, T, D)

        # Reshape and project
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(B, T, C)
        output = self.linear_out(attn_output)

        return output


class Wav2Vec2BertConformerConvolution(nn.Module):
    """Conformer convolution module (depthwise separable convolution)."""

    def __init__(self, hidden_size: int, kernel_size: int = 31, dropout: float = 0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.kernel_size = kernel_size

        # Pointwise conv (expand) - GLU doubles channels
        self.pointwise_conv1 = nn.Conv1d(hidden_size, 2 * hidden_size, kernel_size=1, bias=False)

        # Depthwise conv (groups=hidden_size for depthwise)
        # No padding - we'll do manual left padding for causal convolution
        self.depthwise_conv = nn.Conv1d(
            hidden_size,
            hidden_size,
            kernel_size=kernel_size,
            padding=0,
            groups=hidden_size,
            bias=False,
        )
        self.depthwise_layer_norm = nn.LayerNorm(hidden_size)

        # Pointwise conv (compress)
        self.pointwise_conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=1, bias=False)

        self.dropout = dropout

    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: (B, T, C)
        Returns:
            output: (B, T, C)
        """
        # Layer norm
        x = self.layer_norm(x)

        # MLX Conv1d expects (B, T, C) input and produces (B, T, C_out) output
        # Pointwise conv 1 (with GLU)
        x = self.pointwise_conv1(x)  # (B, T, 2*C)
        # GLU: split and gate
        x, gate = mx.split(x, 2, axis=-1)
        x = x * mx.sigmoid(gate)  # (B, T, C)

        # Causal padding (left padding only)
        # Pad (kernel_size - 1) zeros on the LEFT (time dimension)
        padding_size = self.kernel_size - 1
        x = mx.pad(x, [(0, 0), (padding_size, 0), (0, 0)])  # (B, T+padding, C)

        # Depthwise conv
        x = self.depthwise_conv(x)  # (B, T, C)

        # LayerNorm (already in B, T, C format)
        x = self.depthwise_layer_norm(x)

        # Swish activation
        x = x * mx.sigmoid(x)

        # Pointwise conv 2
        x = self.pointwise_conv2(x)  # (B, T, C)

        return x


class Wav2Vec2BertFeedForward(nn.Module):
    """Feed-forward network."""

    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float = 0.0):
        super().__init__()
        self.intermediate_dense = nn.Linear(hidden_size, intermediate_size, bias=True)
        self.output_dense = nn.Linear(intermediate_size, hidden_size, bias=True)
        self.dropout = dropout

    def __call__(self, x: mx.array) -> mx.array:
        x = self.intermediate_dense(x)
        x = x * mx.sigmoid(x)  # Swish activation: x * sigmoid(x)
        x = self.output_dense(x)
        return x


class Wav2Vec2BertConformerLayer(nn.Module):
    """Conformer encoder layer: FFN + Attention + Conv + FFN."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        conv_kernel_size: int = 31,
        dropout: float = 0.0,
    ):
        super().__init__()

        # First FFN (half-step)
        self.ffn1 = Wav2Vec2BertFeedForward(hidden_size, intermediate_size, dropout)
        self.ffn1_layer_norm = nn.LayerNorm(hidden_size)

        # Self-attention
        self.self_attn = Wav2Vec2BertSelfAttention(hidden_size, num_heads, dropout)
        self.self_attn_layer_norm = nn.LayerNorm(hidden_size)

        # Convolution
        self.conv_module = Wav2Vec2BertConformerConvolution(hidden_size, conv_kernel_size, dropout)

        # Second FFN (half-step)
        self.ffn2 = Wav2Vec2BertFeedForward(hidden_size, intermediate_size, dropout)
        self.ffn2_layer_norm = nn.LayerNorm(hidden_size)

        # Final layer norm
        self.final_layer_norm = nn.LayerNorm(hidden_size)

    def __call__(self, x: mx.array, attention_mask: Optional[mx.array] = None) -> mx.array:
        """
        Args:
            x: (B, T, C)
            attention_mask: (B, T)
        Returns:
            output: (B, T, C)
        """
        # FFN1 (half-step: scaled by 0.5)
        residual = x
        x = self.ffn1_layer_norm(x)
        x = self.ffn1(x)
        x = residual + 0.5 * x

        # Self-attention
        residual = x
        x = self.self_attn_layer_norm(x)
        x = self.self_attn(x, attention_mask)
        x = residual + x

        # Convolution
        residual = x
        x = self.conv_module(x)
        x = residual + x

        # FFN2 (half-step: scaled by 0.5)
        residual = x
        x = self.ffn2_layer_norm(x)
        x = self.ffn2(x)
        x = residual + 0.5 * x

        # Final layer norm
        x = self.final_layer_norm(x)

        return x


class Wav2Vec2BertEncoder(nn.Module):
    """Wav2Vec2-BERT encoder (24 Conformer layers)."""

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.layers = [
            Wav2Vec2BertConformerLayer(
                hidden_size=config["hidden_size"],
                num_heads=config["num_attention_heads"],
                intermediate_size=config["intermediate_size"],
                conv_kernel_size=config["conv_depthwise_kernel_size"],
                dropout=config.get("hidden_dropout", 0.0),
            )
            for _ in range(config["num_hidden_layers"])
        ]

    def __call__(
        self,
        x: mx.array,
        attention_mask: Optional[mx.array] = None,
        output_hidden_states: bool = False,
    ) -> Tuple:
        """
        Args:
            x: (B, T, C)
            attention_mask: (B, T)
            output_hidden_states: If True, return all layer outputs
        Returns:
            last_hidden_state: (B, T, C)
            hidden_states: List of (B, T, C) if output_hidden_states=True
        """
        hidden_states = [] if output_hidden_states else None

        for layer in self.layers:
            if output_hidden_states:
                hidden_states.append(x)

            x = layer(x, attention_mask)

        if output_hidden_states:
            hidden_states.append(x)
            return x, hidden_states

        return x, None


class FeatureProjection(nn.Module):
    """Feature projection wrapper to match HF structure."""

    def __init__(self, input_dim: int, hidden_size: int):
        super().__init__()
        self.layer_norm = nn.LayerNorm(input_dim)
        self.projection = nn.Linear(input_dim, hidden_size, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.layer_norm(x)
        x = self.projection(x)
        return x


class Wav2Vec2BertModel(nn.Module):
    """Complete Wav2Vec2-BERT model."""

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        # Feature projection: 160 -> 1024
        self.feature_projection = FeatureProjection(
            config["feature_projection_input_dim"], config["hidden_size"]
        )

        # Position embeddings (not used in base model, relative pos in attention)
        # self.pos_conv_embed = nn.Conv1d(
        #     config['hidden_size'], config['hidden_size'],
        #     kernel_size=128, padding=64, groups=16
        # )

        # Encoder
        self.encoder = Wav2Vec2BertEncoder(config)

    def __call__(
        self,
        input_features: mx.array,
        attention_mask: Optional[mx.array] = None,
        output_hidden_states: bool = False,
    ):
        """
        Args:
            input_features: (B, T, 160) - FBANK features from SeamlessM4T
            attention_mask: (B, T) - 1 for real, 0 for padding
            output_hidden_states: If True, return all layer outputs
        Returns:
            Namespace with:
                last_hidden_state: (B, T, 1024)
                hidden_states: List of layer outputs
        """
        # Project features
        x = self.feature_projection(input_features)

        # Encode
        last_hidden_state, hidden_states = self.encoder(x, attention_mask, output_hidden_states)

        # Return in HuggingFace-like format
        class Output:
            pass

        output = Output()
        output.last_hidden_state = last_hidden_state
        output.hidden_states = hidden_states

        return output


def create_w2vbert_model(config_path: Optional[str] = None):
    """Create W2V-BERT model with default config."""
    config = {
        "feature_projection_input_dim": 160,
        "hidden_size": 1024,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "intermediate_size": 4096,
        "conv_depthwise_kernel_size": 31,
        "hidden_dropout": 0.0,
        "attention_dropout": 0.0,
        "feat_proj_dropout": 0.0,
    }

    return Wav2Vec2BertModel(config)
