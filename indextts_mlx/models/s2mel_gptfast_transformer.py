"""
MLX gpt-fast Style Transformer Implementation

Matches the actual IndexTTS2 architecture with:
- Combined wqkv projections
- SwiGLU feedforward
- U-ViT skip connections
- AdaptiveLayerNorm
"""

import mlx.core as mx
import mlx.nn as nn
import math
from typing import Optional

from .s2mel_layers import modulate, RMSNorm


class SwiGLU(nn.Module):
    """SwiGLU activation function.

    SwiGLU(x) = (W1*x * silu(W3*x)) @ W2
    """

    def __init__(self, dim: int, hidden_dim: int):
        """Initialize SwiGLU.

        Args:
            dim: Input/output dimension
            hidden_dim: Hidden dimension (typically ~2.67x dim)
        """
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        """Apply SwiGLU.

        Args:
            x: Input of shape (batch, seq_len, dim)

        Returns:
            Output of shape (batch, seq_len, dim)
        """
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))


class Attention(nn.Module):
    """Multi-head attention with combined wqkv projection."""

    def __init__(self, dim: int, n_heads: int):
        """Initialize attention.

        Args:
            dim: Model dimension
            n_heads: Number of attention heads
        """
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.dim = dim

        # Combined QKV projection (gpt-fast style)
        self.wqkv = nn.Linear(dim, 3 * dim, bias=False)

        # Output projection
        self.wo = nn.Linear(dim, dim, bias=False)

    def __call__(
        self,
        x: mx.array,
        freqs_cis: mx.array,
        mask: Optional[mx.array] = None
    ) -> mx.array:
        """Apply multi-head attention with RoPE.

        Args:
            x: Input of shape (batch, seq_len, dim)
            freqs_cis: RoPE frequency tensor of shape (seq_len, head_dim // 2, 2)
            mask: Optional attention mask

        Returns:
            Output of shape (batch, seq_len, dim)
        """
        from .s2mel_layers import apply_rotary_emb

        batch_size, seq_len, _ = x.shape

        # Combined QKV projection
        qkv = self.wqkv(x)  # (batch, seq_len, 3*dim)

        # Split into Q, K, V
        q, k, v = mx.split(qkv, 3, axis=-1)  # Each (batch, seq_len, dim)

        # Reshape for multi-head attention
        # (batch, seq_len, dim) -> (batch, seq_len, n_heads, head_dim)
        q = q.reshape(batch_size, seq_len, self.n_heads, self.head_dim)
        k = k.reshape(batch_size, seq_len, self.n_heads, self.head_dim)
        v = v.reshape(batch_size, seq_len, self.n_heads, self.head_dim)

        # Apply RoPE to Q and K
        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        # Transpose to (batch, n_heads, seq_len, head_dim) for attention
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = (q @ k.transpose(0, 1, 3, 2)) * scale  # (batch, n_heads, seq_len, seq_len)

        if mask is not None:
            # Mask should be broadcastable to (batch, n_heads, seq_len, seq_len)
            scores = scores + mask

        attn = mx.softmax(scores, axis=-1)
        output = attn @ v  # (batch, n_heads, seq_len, head_dim)

        # Reshape back
        output = output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.dim)

        # Output projection
        output = self.wo(output)

        return output


class GPTFastTransformerBlock(nn.Module):
    """Transformer block matching gpt-fast architecture with U-ViT skip connections."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        hidden_dim: int,
        use_uvit_skip: bool = True,
    ):
        """Initialize transformer block.

        Args:
            dim: Model dimension
            n_heads: Number of attention heads
            hidden_dim: FFN hidden dimension
            use_uvit_skip: Whether to use U-ViT skip connections
        """
        super().__init__()
        self.dim = dim
        self.use_uvit_skip = use_uvit_skip

        # Attention
        self.attention = Attention(dim, n_heads)

        # AdaLN for attention
        self.attention_norm = RMSNorm(dim, eps=1e-6)
        self.attention_norm_modulation = nn.Linear(dim, 2 * dim)

        # SwiGLU feedforward
        self.feed_forward = SwiGLU(dim, hidden_dim)

        # AdaLN for feedforward
        self.ffn_norm = RMSNorm(dim, eps=1e-6)
        self.ffn_norm_modulation = nn.Linear(dim, 2 * dim)

        # U-ViT skip connection
        if use_uvit_skip:
            # Skip connection from encoder to decoder (cross-attention style)
            # For now, simple linear projection
            self.skip_in_linear = nn.Linear(2 * dim, dim)

    def __call__(
        self,
        x: mx.array,
        conditioning: mx.array,
        freqs_cis: mx.array,
        skip_input: Optional[mx.array] = None,
        mask: Optional[mx.array] = None
    ) -> tuple[mx.array, mx.array]:
        """Apply transformer block.

        Args:
            x: Input of shape (batch, seq_len, dim)
            conditioning: Timestep conditioning of shape (batch, dim)
            freqs_cis: RoPE frequency tensor
            skip_input: Skip connection input from encoder (for U-ViT)
            mask: Optional attention mask

        Returns:
            Tuple of (output, skip_output) for U-ViT connections
        """
        # U-ViT skip connection from encoder
        if self.use_uvit_skip and skip_input is not None:
            x = self.skip_in_linear(mx.concatenate([x, skip_input], axis=-1))

        # Self-attention with AdaLN
        attn_mod = self.attention_norm_modulation(conditioning)
        # PyTorch splits as (weight, bias) = (scale, shift), so we need scale first
        scale_attn, shift_attn = mx.split(attn_mod, 2, axis=-1)

        x_norm = self.attention_norm(x)
        x_mod = modulate(x_norm, shift_attn, scale_attn)

        attn_out = self.attention(x_mod, freqs_cis, mask=mask)
        x = x + attn_out  # Residual

        # Feedforward with AdaLN
        ffn_mod = self.ffn_norm_modulation(conditioning)
        scale_ffn, shift_ffn = mx.split(ffn_mod, 2, axis=-1)

        x_norm = self.ffn_norm(x)
        x_mod = modulate(x_norm, shift_ffn, scale_ffn)

        ffn_out = self.feed_forward(x_mod)
        x = x + ffn_out  # Residual

        # PyTorch returns the OUTPUT as skip connection, not the input
        return x, x


class GPTFastTransformer(nn.Module):
    """gpt-fast style transformer with U-ViT skip connections."""

    def __init__(
        self,
        dim: int,
        n_layers: int,
        n_heads: int,
        hidden_dim: int,
        use_uvit_skip: bool = True,
        block_size: int = 8192,
        rope_base: float = 10000,
    ):
        """Initialize transformer.

        Args:
            dim: Model dimension
            n_layers: Number of transformer blocks
            n_heads: Number of attention heads
            hidden_dim: FFN hidden dimension
            use_uvit_skip: Whether to use U-ViT skip connections
            block_size: Maximum sequence length
            rope_base: Base for RoPE frequency computation
        """
        super().__init__()
        self.dim = dim
        self.n_layers = n_layers
        self.use_uvit_skip = use_uvit_skip
        self.block_size = block_size

        # Transformer blocks
        self.layers = [
            GPTFastTransformerBlock(dim, n_heads, hidden_dim, use_uvit_skip)
            for _ in range(n_layers)
        ]

        # Final normalization with AdaLN modulation
        self.norm = RMSNorm(dim, eps=1e-6)
        self.norm_modulation = nn.Linear(dim, 2 * dim)

        # Precompute RoPE frequencies
        from .s2mel_layers import precompute_freqs_cis
        head_dim = dim // n_heads
        self.freqs_cis = precompute_freqs_cis(block_size, head_dim, int(rope_base))

    def __call__(
        self,
        x: mx.array,
        conditioning: mx.array,
        mask: Optional[mx.array] = None
    ) -> mx.array:
        """Apply transformer.

        Args:
            x: Input of shape (batch, seq_len, dim)
            conditioning: Timestep conditioning of shape (batch, dim)
            mask: Optional attention mask

        Returns:
            Output of shape (batch, seq_len, dim)
        """
        # Get sequence length and slice freqs_cis
        seq_len = x.shape[1]
        freqs_cis = self.freqs_cis[:seq_len]  # (seq_len, head_dim // 2, 2)

        if self.use_uvit_skip:
            # U-ViT: encoder, middle layer, then decoder
            # Matches PyTorch: layers_emit_skip = [i for i < n_layer // 2]
            #                  layers_receive_skip = [i for i > n_layer // 2]
            #                  Middle layer (i == n_layer // 2) has no skip
            n_encoder_layers = self.n_layers // 2

            # Encoder pass: collect skip connections (layers 0 to n//2-1)
            skip_connections = []
            for i in range(n_encoder_layers):
                x, skip = self.layers[i](x, conditioning, freqs_cis, skip_input=None, mask=mask)
                skip_connections.append(skip)

            # Middle layer (layer n//2): no skip connection
            x, _ = self.layers[n_encoder_layers](x, conditioning, freqs_cis, skip_input=None, mask=mask)

            # Decoder pass: use skip connections in reverse order (layers n//2+1 to n-1)
            for i in range(n_encoder_layers + 1, self.n_layers):
                # Map decoder layer to encoder layer:
                # i=n//2+1 -> skip from layer n//2-1
                # i=n//2+2 -> skip from layer n//2-2, etc.
                skip_idx = self.n_layers - i - 1
                skip_input = skip_connections[skip_idx]
                x, _ = self.layers[i](x, conditioning, freqs_cis, skip_input=skip_input, mask=mask)
        else:
            # Standard transformer without U-ViT
            for layer in self.layers:
                x, _ = layer(x, conditioning, freqs_cis, skip_input=None, mask=mask)

        # Final normalization with AdaLN modulation
        norm_mod = self.norm_modulation(conditioning)
        scale_norm, shift_norm = mx.split(norm_mod, 2, axis=-1)

        x_norm = self.norm(x)
        x = modulate(x_norm, shift_norm, scale_norm)

        return x
