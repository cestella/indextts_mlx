"""
Load UnifiedVoice GPT weights from NPZ format into MLX model.
"""

import mlx.core as mx
import numpy as np
from pathlib import Path
from typing import Dict

from indextts_mlx.models.gpt import UnifiedVoice


def load_gpt_weights(weights_path: str) -> Dict[str, mx.array]:
    """Load GPT weights from NPZ file.

    Args:
        weights_path: Path to .npz file containing weights
    Returns:
        Dictionary mapping weight names to MLX arrays
    """
    print(f"Loading GPT weights from {weights_path}...")

    # Load numpy weights
    weights_np = np.load(weights_path)
    weights = {}

    for name in weights_np.keys():
        weights[name] = mx.array(weights_np[name])

    print(f"✓ Loaded {len(weights)} parameters")

    return weights


def _load_conformer_encoder(encoder, flat_weights: Dict, prefix: str):
    """Load conformer encoder weights (reused for conditioning and emo_conditioning)."""
    encoder.embed.conv[0].weight = flat_weights[f'{prefix}.embed.conv.0.weight']
    encoder.embed.conv[0].bias = flat_weights[f'{prefix}.embed.conv.0.bias']
    encoder.embed.out_proj.weight = flat_weights[f'{prefix}.embed.out.0.weight']
    encoder.embed.out_proj.bias = flat_weights[f'{prefix}.embed.out.0.bias']
    encoder.pos_enc.pe = flat_weights[f'{prefix}.embed.pos_enc.pe']
    encoder.after_norm.weight = flat_weights[f'{prefix}.after_norm.weight']
    encoder.after_norm.bias = flat_weights[f'{prefix}.after_norm.bias']

    for layer_idx in range(len(encoder.encoders)):
        layer = encoder.encoders[layer_idx]
        lp = f'{prefix}.encoders.{layer_idx}'
        layer.self_attn.linear_q.weight = flat_weights[f'{lp}.self_attn.linear_q.weight']
        layer.self_attn.linear_q.bias = flat_weights[f'{lp}.self_attn.linear_q.bias']
        layer.self_attn.linear_k.weight = flat_weights[f'{lp}.self_attn.linear_k.weight']
        layer.self_attn.linear_k.bias = flat_weights[f'{lp}.self_attn.linear_k.bias']
        layer.self_attn.linear_v.weight = flat_weights[f'{lp}.self_attn.linear_v.weight']
        layer.self_attn.linear_v.bias = flat_weights[f'{lp}.self_attn.linear_v.bias']
        layer.self_attn.linear_out.weight = flat_weights[f'{lp}.self_attn.linear_out.weight']
        layer.self_attn.linear_out.bias = flat_weights[f'{lp}.self_attn.linear_out.bias']
        layer.self_attn.linear_pos.weight = flat_weights[f'{lp}.self_attn.linear_pos.weight']
        layer.self_attn.pos_bias_u = flat_weights[f'{lp}.self_attn.pos_bias_u']
        layer.self_attn.pos_bias_v = flat_weights[f'{lp}.self_attn.pos_bias_v']
        layer.feed_forward.w_1.weight = flat_weights[f'{lp}.feed_forward.w_1.weight']
        layer.feed_forward.w_1.bias = flat_weights[f'{lp}.feed_forward.w_1.bias']
        layer.feed_forward.w_2.weight = flat_weights[f'{lp}.feed_forward.w_2.weight']
        layer.feed_forward.w_2.bias = flat_weights[f'{lp}.feed_forward.w_2.bias']
        layer.conv_module.pointwise_conv1.weight = flat_weights[f'{lp}.conv_module.pointwise_conv1.weight']
        layer.conv_module.pointwise_conv1.bias = flat_weights[f'{lp}.conv_module.pointwise_conv1.bias']
        layer.conv_module.depthwise_conv.weight = flat_weights[f'{lp}.conv_module.depthwise_conv.weight']
        layer.conv_module.depthwise_conv.bias = flat_weights[f'{lp}.conv_module.depthwise_conv.bias']
        layer.conv_module.norm.weight = flat_weights[f'{lp}.conv_module.norm.weight']
        layer.conv_module.norm.bias = flat_weights[f'{lp}.conv_module.norm.bias']
        layer.conv_module.pointwise_conv2.weight = flat_weights[f'{lp}.conv_module.pointwise_conv2.weight']
        layer.conv_module.pointwise_conv2.bias = flat_weights[f'{lp}.conv_module.pointwise_conv2.bias']
        layer.norm_ff.weight = flat_weights[f'{lp}.norm_ff.weight']
        layer.norm_ff.bias = flat_weights[f'{lp}.norm_ff.bias']
        layer.norm_mha.weight = flat_weights[f'{lp}.norm_mha.weight']
        layer.norm_mha.bias = flat_weights[f'{lp}.norm_mha.bias']
        layer.norm_conv.weight = flat_weights[f'{lp}.norm_conv.weight']
        layer.norm_conv.bias = flat_weights[f'{lp}.norm_conv.bias']
        layer.norm_final.weight = flat_weights[f'{lp}.norm_final.weight']
        layer.norm_final.bias = flat_weights[f'{lp}.norm_final.bias']


def _load_perceiver_encoder(perceiver, flat_weights: Dict, prefix: str):
    """Load perceiver resampler weights (reused for perceiver and emo_perceiver)."""
    perceiver.latents = flat_weights[f'{prefix}.latents']
    if perceiver.proj_context is not None:
        perceiver.proj_context.weight = flat_weights[f'{prefix}.proj_context.weight']
        perceiver.proj_context.bias = flat_weights[f'{prefix}.proj_context.bias']
    for layer_idx in range(len(perceiver.layers)):
        attn, ff = perceiver.layers[layer_idx]
        lp = f'{prefix}.layers.{layer_idx}'
        attn.to_q.weight = flat_weights[f'{lp}.0.to_q.weight']
        attn.to_kv.weight = flat_weights[f'{lp}.0.to_kv.weight']
        attn.to_out.weight = flat_weights[f'{lp}.0.to_out.weight']
        ff.net[0].weight = flat_weights[f'{lp}.1.0.weight']
        ff.net[0].bias = flat_weights[f'{lp}.1.0.bias']
        ff.net[2].weight = flat_weights[f'{lp}.1.2.weight']
        ff.net[2].bias = flat_weights[f'{lp}.1.2.bias']
    perceiver.norm.gamma = flat_weights[f'{prefix}.norm.gamma']


def load_gpt_model(model: UnifiedVoice, weights_path: str) -> UnifiedVoice:
    """Load pretrained weights into UnifiedVoice GPT model.

    Args:
        model: MLX UnifiedVoice model instance
        weights_path: Path to .npz weights file
    Returns:
        Model with loaded weights
    """
    print("Loading UnifiedVoice pretrained weights...")

    # Load flat weights
    flat_weights = load_gpt_weights(weights_path)

    print("Loading weights into model...")

    # --- Embeddings ---
    print("  Loading embeddings...")
    model.text_embedding.weight = flat_weights['text_embedding.weight']
    model.mel_embedding.weight = flat_weights['mel_embedding.weight']

    # Position embeddings
    model.text_pos_embedding.emb.weight = flat_weights['text_pos_embedding.emb.weight']
    model.mel_pos_embedding.emb.weight = flat_weights['mel_pos_embedding.emb.weight']

    # --- Conditioning Encoder (Conformer) ---
    print("  Loading conditioning encoder...")
    _load_conformer_encoder(model.conditioning_encoder, flat_weights, 'conditioning_encoder')

    # --- Emotion Conditioning Encoder (same structure) ---
    print("  Loading emotion conditioning encoder...")
    _load_conformer_encoder(model.emo_conditioning_encoder, flat_weights, 'emo_conditioning_encoder')

    # --- Emotion Perceiver Encoder ---
    print("  Loading emotion perceiver encoder...")
    _load_perceiver_encoder(model.emo_perceiver_encoder, flat_weights, 'emo_perceiver_encoder')

    # --- Emotion vector layers + speed embeddings ---
    model.emovec_layer.weight = flat_weights['emovec_layer.weight']
    model.emovec_layer.bias = flat_weights['emovec_layer.bias']
    model.emo_layer.weight = flat_weights['emo_layer.weight']
    model.emo_layer.bias = flat_weights['emo_layer.bias']
    model.speed_emb.weight = flat_weights['speed_emb.weight']

    # --- Perceiver Encoder ---
    print("  Loading perceiver encoder...")
    _load_perceiver_encoder(model.perceiver_encoder, flat_weights, 'perceiver_encoder')

    # --- GPT2 ---
    print("  Loading GPT2 transformer...")
    prefix = 'gpt'

    num_gpt_layers = len(model.gpt.h)
    for layer_idx in range(num_gpt_layers):
        block = model.gpt.h[layer_idx]
        block_prefix = f'{prefix}.h.{layer_idx}'

        # Layer norms
        block.ln_1.weight = flat_weights[f'{block_prefix}.ln_1.weight']
        block.ln_1.bias = flat_weights[f'{block_prefix}.ln_1.bias']
        block.ln_2.weight = flat_weights[f'{block_prefix}.ln_2.weight']
        block.ln_2.bias = flat_weights[f'{block_prefix}.ln_2.bias']

        # Attention (fused QKV)
        # HuggingFace stores as (in, out), MLX needs (out, in)
        block.attn.c_attn.weight = flat_weights[f'{block_prefix}.attn.c_attn.weight'].T
        block.attn.c_attn.bias = flat_weights[f'{block_prefix}.attn.c_attn.bias']
        block.attn.c_proj.weight = flat_weights[f'{block_prefix}.attn.c_proj.weight'].T
        block.attn.c_proj.bias = flat_weights[f'{block_prefix}.attn.c_proj.bias']

        # MLP
        block.mlp.c_fc.weight = flat_weights[f'{block_prefix}.mlp.c_fc.weight'].T
        block.mlp.c_fc.bias = flat_weights[f'{block_prefix}.mlp.c_fc.bias']
        block.mlp.c_proj.weight = flat_weights[f'{block_prefix}.mlp.c_proj.weight'].T
        block.mlp.c_proj.bias = flat_weights[f'{block_prefix}.mlp.c_proj.bias']

    # Final layer norm
    model.gpt.ln_f.weight = flat_weights[f'{prefix}.ln_f.weight']
    model.gpt.ln_f.bias = flat_weights[f'{prefix}.ln_f.bias']

    # --- Output head ---
    print("  Loading output head...")
    model.final_norm.weight = flat_weights['final_norm.weight']
    model.final_norm.bias = flat_weights['final_norm.bias']
    # mel_head weight is already in correct format (out, in) = (8194, 1280)
    model.mel_head.weight = flat_weights['mel_head.weight']  # No transpose needed!
    model.mel_head.bias = flat_weights['mel_head.bias']

    print("✓ Weights loaded successfully")

    return model
