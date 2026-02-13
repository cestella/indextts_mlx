"""
Load semantic codec weights from NPZ format into MLX RepCodec model.
"""

import mlx.core as mx
import numpy as np
from pathlib import Path
from typing import Dict

from indextts_mlx.models.semantic_codec import RepCodec


def load_semantic_codec_weights(weights_path: str) -> Dict[str, mx.array]:
    """Load semantic codec weights from NPZ file.

    Args:
        weights_path: Path to .npz file containing weights

    Returns:
        Dictionary mapping weight names to MLX arrays
    """
    print(f"Loading semantic codec weights from {weights_path}...")

    # Load numpy weights
    weights_np = np.load(weights_path)
    weights = {}

    # Reconstruct weights from weight_norm (weight_g and weight_v)
    weight_norm_pairs = {}
    for name in weights_np.keys():
        if '.weight_v' in name:
            base_name = name.replace('.weight_v', '')
            if base_name not in weight_norm_pairs:
                weight_norm_pairs[base_name] = {}
            weight_norm_pairs[base_name]['v'] = weights_np[name]
        elif '.weight_g' in name:
            base_name = name.replace('.weight_g', '')
            if base_name not in weight_norm_pairs:
                weight_norm_pairs[base_name] = {}
            weight_norm_pairs[base_name]['g'] = weights_np[name]
        else:
            weights[name] = mx.array(weights_np[name])

    # Reconstruct actual weights: weight = weight_g * (weight_v / ||weight_v||)
    for base_name, pair in weight_norm_pairs.items():
        if 'v' in pair and 'g' in pair:
            weight_v = pair['v']
            weight_g = pair['g']

            # Normalize weight_v along dimension 1 (out_features dimension for Conv1d)
            # For Conv1d: weight is (out, in, kernel)
            norm_v = np.linalg.norm(weight_v, axis=(1, 2), keepdims=True)
            weight = weight_g * (weight_v / norm_v)

            weights[base_name + '.weight'] = mx.array(weight)
            print(f"  Reconstructed {base_name}.weight from weight_norm")

    print(f"✓ Loaded {len(weights)} parameters")

    return weights


def load_semantic_codec_model(model: RepCodec, weights_path: str) -> RepCodec:
    """Load pretrained weights into semantic codec model.

    Args:
        model: MLX RepCodec model instance
        weights_path: Path to .npz weights file

    Returns:
        Model with loaded weights
    """
    print("Loading semantic codec pretrained weights...")

    # Load flat weights
    flat_weights = load_semantic_codec_weights(weights_path)

    # Filter to encoder and quantizer only (no decoder for inference)
    encoder_quantizer_weights = {
        k: v for k, v in flat_weights.items()
        if k.startswith('encoder.') or k.startswith('quantizer.')
    }

    print(f"✓ Filtered to {len(encoder_quantizer_weights)} parameters for inference")

    # Load weights manually
    print("Loading weights into model...")

    # Encoder: VocosBackbone
    prefix = 'encoder.0'  # encoder.0 is VocosBackbone

    # Embedding conv
    model.encoder[0].embed.weight = encoder_quantizer_weights[f'{prefix}.embed.weight']
    model.encoder[0].embed.bias = encoder_quantizer_weights[f'{prefix}.embed.bias']

    # Initial norm
    model.encoder[0].norm.weight = encoder_quantizer_weights[f'{prefix}.norm.weight']
    model.encoder[0].norm.bias = encoder_quantizer_weights[f'{prefix}.norm.bias']

    # ConvNeXt blocks
    num_layers = len(model.encoder[0].convnext)
    for layer_idx in range(num_layers):
        block = model.encoder[0].convnext[layer_idx]
        block_prefix = f'{prefix}.convnext.{layer_idx}'

        # Depthwise conv
        block.dwconv.weight = encoder_quantizer_weights[f'{block_prefix}.dwconv.weight']
        block.dwconv.bias = encoder_quantizer_weights[f'{block_prefix}.dwconv.bias']

        # Norm
        block.norm.weight = encoder_quantizer_weights[f'{block_prefix}.norm.weight']
        block.norm.bias = encoder_quantizer_weights[f'{block_prefix}.norm.bias']

        # Pointwise convs
        block.pwconv1.weight = encoder_quantizer_weights[f'{block_prefix}.pwconv1.weight']
        block.pwconv1.bias = encoder_quantizer_weights[f'{block_prefix}.pwconv1.bias']
        block.pwconv2.weight = encoder_quantizer_weights[f'{block_prefix}.pwconv2.weight']
        block.pwconv2.bias = encoder_quantizer_weights[f'{block_prefix}.pwconv2.bias']

        # Gamma (layer scaling)
        if f'{block_prefix}.gamma' in encoder_quantizer_weights:
            block.gamma = encoder_quantizer_weights[f'{block_prefix}.gamma']

    # Final layer norm
    model.encoder[0].final_layer_norm.weight = encoder_quantizer_weights[f'{prefix}.final_layer_norm.weight']
    model.encoder[0].final_layer_norm.bias = encoder_quantizer_weights[f'{prefix}.final_layer_norm.bias']

    # Encoder linear projection (encoder.1)
    model.encoder[1].weight = encoder_quantizer_weights['encoder.1.weight']
    model.encoder[1].bias = encoder_quantizer_weights['encoder.1.bias']

    # Quantizer
    num_quantizers = model.quantizer.num_quantizers
    for q_idx in range(num_quantizers):
        quantizer = model.quantizer.quantizers[q_idx]
        q_prefix = f'quantizer.quantizers.{q_idx}'

        # Codebook
        quantizer.codebook = encoder_quantizer_weights[f'{q_prefix}.codebook.weight']

        # Optional projection layers
        if f'{q_prefix}.in_project.weight' in encoder_quantizer_weights:
            quantizer.in_project.weight = encoder_quantizer_weights[f'{q_prefix}.in_project.weight']
            quantizer.in_project.bias = encoder_quantizer_weights[f'{q_prefix}.in_project.bias']

        if f'{q_prefix}.out_project.weight' in encoder_quantizer_weights:
            quantizer.out_project.weight = encoder_quantizer_weights[f'{q_prefix}.out_project.weight']
            quantizer.out_project.bias = encoder_quantizer_weights[f'{q_prefix}.out_project.bias']

    print("✓ Weights loaded successfully")

    return model
