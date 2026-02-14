"""
Load BigVGAN weights from NPZ format into MLX model.
"""

import mlx.core as mx
import numpy as np
from pathlib import Path
from typing import Dict

from indextts_mlx.models.bigvgan import BigVGAN, create_bigvgan


def load_bigvgan_weights(weights_path: str) -> Dict[str, mx.array]:
    """Load BigVGAN weights from NPZ file.

    Args:
        weights_path: Path to .npz file containing weights
    Returns:
        Dictionary mapping weight names to MLX arrays
    """
    print(f"Loading BigVGAN weights from {weights_path}...")

    # Load numpy weights
    weights_np = np.load(weights_path)
    weights = {}

    for name in weights_np.keys():
        weights[name] = mx.array(weights_np[name])

    print(f"✓ Loaded {len(weights)} parameters")

    return weights


def load_bigvgan_model(model: BigVGAN, weights_path: str) -> BigVGAN:
    """Load pretrained weights into BigVGAN model.

    Args:
        model: MLX BigVGAN model instance
        weights_path: Path to .npz weights file
    Returns:
        Model with loaded weights
    """
    print("Loading BigVGAN pretrained weights...")

    # Load flat weights
    flat_weights = load_bigvgan_weights(weights_path)

    print("Loading weights into model...")

    # --- Pre-conv ---
    print("  Loading pre-conv...")
    model.conv_pre.weight = flat_weights["conv_pre.weight"]
    model.conv_pre.bias = flat_weights["conv_pre.bias"]

    # --- Upsampling layers ---
    print("  Loading upsampling layers...")
    for i in range(model.num_upsamples):
        model.ups[i].weight = flat_weights[f"ups.{i}.0.weight"]
        model.ups[i].bias = flat_weights[f"ups.{i}.0.bias"]

    # --- Residual blocks ---
    print("  Loading residual blocks...")
    num_resblocks = len(model.resblocks)
    for i in range(num_resblocks):
        block = model.resblocks[i]
        prefix = f"resblocks.{i}"

        # Check if this is AMPBlock1 or AMPBlock2
        if hasattr(block, "convs1"):
            # AMPBlock1
            num_dilations = len(block.convs1)

            # Load convs1
            for j in range(num_dilations):
                block.convs1[j].weight = flat_weights[f"{prefix}.convs1.{j}.weight"]
                if f"{prefix}.convs1.{j}.bias" in flat_weights:
                    block.convs1[j].bias = flat_weights[f"{prefix}.convs1.{j}.bias"]

            # Load convs2
            for j in range(num_dilations):
                block.convs2[j].weight = flat_weights[f"{prefix}.convs2.{j}.weight"]
                if f"{prefix}.convs2.{j}.bias" in flat_weights:
                    block.convs2[j].bias = flat_weights[f"{prefix}.convs2.{j}.bias"]

            # Load activations (Snake/SnakeBeta parameters and anti-aliasing filters)
            for j in range(block.num_layers):
                activation_module = block.activations[j]

                # Check if anti-aliasing is enabled
                if block.use_anti_aliasing:
                    # Activation is wrapped in Activation1d
                    act = activation_module.act  # Snake/SnakeBeta
                    act.alpha = flat_weights[f"{prefix}.activations.{j}.act.alpha"]
                    if hasattr(act, "beta"):
                        act.beta = flat_weights[f"{prefix}.activations.{j}.act.beta"]

                    # Load anti-aliasing filters
                    activation_module.upsample.filter = flat_weights[
                        f"{prefix}.activations.{j}.upsample.filter"
                    ]
                    activation_module.downsample.lowpass.filter = flat_weights[
                        f"{prefix}.activations.{j}.downsample.lowpass.filter"
                    ]
                else:
                    # Activation is raw Snake/SnakeBeta
                    activation_module.alpha = flat_weights[f"{prefix}.activations.{j}.act.alpha"]
                    if hasattr(activation_module, "beta"):
                        activation_module.beta = flat_weights[f"{prefix}.activations.{j}.act.beta"]
        else:
            # AMPBlock2
            num_dilations = len(block.convs)

            # Load convs
            for j in range(num_dilations):
                block.convs[j].weight = flat_weights[f"{prefix}.convs.{j}.weight"]
                if f"{prefix}.convs.{j}.bias" in flat_weights:
                    block.convs[j].bias = flat_weights[f"{prefix}.convs.{j}.bias"]

            # Load activations
            for j in range(block.num_layers):
                activation_module = block.activations[j]

                # Check if anti-aliasing is enabled
                if block.use_anti_aliasing:
                    act = activation_module.act
                    act.alpha = flat_weights[f"{prefix}.activations.{j}.act.alpha"]
                    if hasattr(act, "beta"):
                        act.beta = flat_weights[f"{prefix}.activations.{j}.act.beta"]

                    # Load anti-aliasing filters
                    activation_module.upsample.filter = flat_weights[
                        f"{prefix}.activations.{j}.upsample.filter"
                    ]
                    activation_module.downsample.lowpass.filter = flat_weights[
                        f"{prefix}.activations.{j}.downsample.lowpass.filter"
                    ]
                else:
                    activation_module.alpha = flat_weights[f"{prefix}.activations.{j}.act.alpha"]
                    if hasattr(activation_module, "beta"):
                        activation_module.beta = flat_weights[f"{prefix}.activations.{j}.act.beta"]

    # --- Post-conv ---
    print("  Loading post-conv...")
    # Check if anti-aliasing is enabled for post-activation
    if model.use_anti_aliasing:
        # Post-activation is wrapped in Activation1d
        act_post = model.activation_post.act
        act_post.alpha = flat_weights["activation_post.act.alpha"]
        if hasattr(act_post, "beta"):
            act_post.beta = flat_weights["activation_post.act.beta"]

        # Load post-activation filters
        model.activation_post.upsample.filter = flat_weights["activation_post.upsample.filter"]
        model.activation_post.downsample.lowpass.filter = flat_weights[
            "activation_post.downsample.lowpass.filter"
        ]
    else:
        # Raw activation
        model.activation_post.alpha = flat_weights["activation_post.act.alpha"]
        if hasattr(model.activation_post, "beta"):
            model.activation_post.beta = flat_weights["activation_post.act.beta"]

    model.conv_post.weight = flat_weights["conv_post.weight"]
    # Note: use_bias_at_final=False in BigVGAN v2
    if "conv_post.bias" in flat_weights:
        model.conv_post.bias = flat_weights["conv_post.bias"]

    print("✓ Weights loaded successfully")

    return model
