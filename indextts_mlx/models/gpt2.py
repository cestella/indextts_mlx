"""
GPT-2 Transformer implementation in MLX.

Based on HuggingFace transformers GPT2Model.
Used in UnifiedVoice for text-to-semantic-code generation.
"""

import math
from typing import Optional, Tuple
import mlx.core as mx
import mlx.nn as nn


def gelu_new(x):
    """GELU activation with tanh approximation (HuggingFace 'gelu_new').

    This is the activation function used in GPT-2 and GPT-Neo.
    Formula: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    """
    return 0.5 * x * (1.0 + mx.tanh(0.797884560802865 * (x + 0.044715 * x * x * x)))


class GPT2MLP(nn.Module):
    """GPT-2 MLP (feed-forward) block.

    Standard transformer FFN with GELU activation.
    """

    def __init__(self, model_dim: int, intermediate_dim: int):
        super().__init__()
        self.c_fc = nn.Linear(model_dim, intermediate_dim)
        self.c_proj = nn.Linear(intermediate_dim, model_dim)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: (B, T, model_dim)
        Returns:
            output: (B, T, model_dim)
        """
        x = self.c_fc(x)
        x = gelu_new(x)  # Use gelu_new (tanh approximation) like HuggingFace
        x = self.c_proj(x)
        return x


class GPT2Attention(nn.Module):
    """GPT-2 Multi-head causal self-attention.

    Uses fused QKV projection and causal masking.
    Supports KV-cache for efficient autoregressive generation.
    """

    def __init__(self, model_dim: int, num_heads: int):
        super().__init__()
        assert model_dim % num_heads == 0, f"model_dim ({model_dim}) must be divisible by num_heads ({num_heads})"

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads

        # Fused QKV projection
        self.c_attn = nn.Linear(model_dim, 3 * model_dim)

        # Output projection
        self.c_proj = nn.Linear(model_dim, model_dim)

    def __call__(
        self,
        x: mx.array,
        attention_mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> Tuple[mx.array, Optional[Tuple[mx.array, mx.array]]]:
        """
        Args:
            x: (B, T, model_dim) input
            attention_mask: (B, T) mask (1 = attend, 0 = ignore)
            cache: Optional (K, V) cache from previous step
        Returns:
            output: (B, T, model_dim)
            new_cache: Optional (K, V) cache for next step
        """
        B, T, C = x.shape

        # Fused QKV projection
        qkv = self.c_attn(x)  # (B, T, 3 * model_dim)

        # Split into Q, K, V
        q, k, v = mx.split(qkv, 3, axis=-1)

        # Reshape for multi-head attention: (B, T, C) -> (B, num_heads, T, head_dim)
        q = q.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Handle KV-cache
        if cache is not None:
            k_cache, v_cache = cache
            k = mx.concatenate([k_cache, k], axis=2)  # Concat along time dimension
            v = mx.concatenate([v_cache, v], axis=2)

        # Always return cache to enable KV caching for autoregressive generation
        new_cache = (k, v)

        # Compute attention scores
        scores = mx.matmul(q, k.transpose(0, 1, 3, 2)) / math.sqrt(self.head_dim)

        # Apply causal mask (lower triangular)
        # For cached inference, we only compute attention for the new token
        if cache is None:
            # Full causal mask for training/initial pass
            causal_mask = mx.tril(mx.ones((T, T)))
            causal_mask = causal_mask.reshape(1, 1, T, T)
            # Use same mask value as PyTorch: finfo(float32).min
            # PyTorch: torch.finfo(torch.float32).min = -3.4028234663852886e+38
            mask_value = -3.4028235e+38
            scores = mx.where(causal_mask == 0, mask_value, scores)

        # Apply attention mask if provided
        if attention_mask is not None:
            # attention_mask: (B, T_total) where T_total = T + cache_len
            # Reshape to (B, 1, 1, T_total) for broadcasting
            T_total = k.shape[2]
            attn_mask = attention_mask[:, None, None, :T_total]
            scores = mx.where(attn_mask == 0, -1e9, scores)

        # Softmax and weighted sum
        attn_weights = mx.softmax(scores, axis=-1)
        attn_output = mx.matmul(attn_weights, v)  # (B, num_heads, T, head_dim)

        # Reshape back: (B, num_heads, T, head_dim) -> (B, T, model_dim)
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(B, T, C)

        # Output projection
        output = self.c_proj(attn_output)

        return output, new_cache


class GPT2Block(nn.Module):
    """GPT-2 transformer block.

    Pre-norm architecture:
    - LayerNorm → Attention → Residual
    - LayerNorm → FFN → Residual
    """

    def __init__(self, model_dim: int, num_heads: int, intermediate_dim: int):
        super().__init__()
        self.ln_1 = nn.LayerNorm(model_dim)
        self.attn = GPT2Attention(model_dim, num_heads)
        self.ln_2 = nn.LayerNorm(model_dim)
        self.mlp = GPT2MLP(model_dim, intermediate_dim)

    def __call__(
        self,
        x: mx.array,
        attention_mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> Tuple[mx.array, Optional[Tuple[mx.array, mx.array]]]:
        """
        Args:
            x: (B, T, model_dim)
            attention_mask: (B, T_total) mask
            cache: Optional (K, V) cache
        Returns:
            output: (B, T, model_dim)
            new_cache: Optional (K, V) cache
        """
        # Pre-norm attention
        attn_out, new_cache = self.attn(self.ln_1(x), attention_mask, cache)
        x = x + attn_out

        # Pre-norm FFN
        x = x + self.mlp(self.ln_2(x))

        return x, new_cache


class GPT2Model(nn.Module):
    """GPT-2 Transformer model.

    Standard GPT-2 architecture with:
    - Multiple transformer blocks
    - Final layer norm
    - Support for KV-caching
    """

    def __init__(
        self,
        num_layers: int,
        model_dim: int,
        num_heads: int,
        intermediate_dim: Optional[int] = None,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.intermediate_dim = intermediate_dim or (4 * model_dim)

        # Transformer blocks
        self.h = [
            GPT2Block(model_dim, num_heads, self.intermediate_dim)
            for _ in range(num_layers)
        ]

        # Final layer norm
        self.ln_f = nn.LayerNorm(model_dim)

    def __call__(
        self,
        x: mx.array,
        attention_mask: Optional[mx.array] = None,
        cache: Optional[list] = None,
    ) -> Tuple[mx.array, list]:
        """
        Args:
            x: (B, T, model_dim) input embeddings
            attention_mask: (B, T_total) attention mask
            cache: Optional list of (K, V) caches for each layer
        Returns:
            output: (B, T, model_dim) final hidden states
            new_cache: list of (K, V) caches (always returned for generation)
        """
        # Always create cache for generation
        new_cache = []

        # Pass through transformer blocks
        for i, block in enumerate(self.h):
            layer_cache = cache[i] if cache is not None else None
            x, layer_new_cache = block(x, attention_mask, layer_cache)
            new_cache.append(layer_new_cache)

        # Final layer norm
        x = self.ln_f(x)

        return x, new_cache


def create_gpt2(
    num_layers: int = 24,
    model_dim: int = 1280,
    num_heads: int = 8,
    intermediate_dim: Optional[int] = None,
) -> GPT2Model:
    """Create a GPT-2 model with default UnifiedVoice parameters.

    Default config matches the pretrained checkpoint:
    - 24 layers
    - 1280 model dimension
    - 8 attention heads (head_dim = 160)
    - 5120 intermediate dimension (4x)
    """
    return GPT2Model(num_layers, model_dim, num_heads, intermediate_dim)
