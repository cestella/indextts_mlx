"""
Load W2V-BERT weights from NumPy format into MLX model.
"""

import mlx.core as mx
import numpy as np
from typing import Dict
from pathlib import Path


def load_w2vbert_weights(weights_path: str) -> Dict[str, mx.array]:
    """Load W2V-BERT weights from NPZ file.

    Args:
        weights_path: Path to .npz file containing weights

    Returns:
        Dictionary mapping weight names to MLX arrays
    """
    print(f"Loading W2V-BERT weights from {weights_path}...")

    # Load numpy weights
    weights_np = np.load(weights_path)
    weights = {}

    num_transposed = 0
    for name in weights_np.keys():
        weight = weights_np[name]

        # Transpose Conv1d weights: (out, in, kernel) -> (out, kernel, in)
        if "conv" in name and ".weight" in name and weight.ndim == 3:
            weight = np.transpose(weight, (0, 2, 1))
            num_transposed += 1

        weights[name] = mx.array(weight)

    print(f"✓ Loaded {len(weights)} parameters")
    print(f"✓ Transposed {num_transposed} Conv1d weights")

    return weights


def nested_dict_from_flat(flat_dict: Dict[str, mx.array]) -> Dict:
    """Convert flat dictionary with dot-separated keys to nested structure.

    Args:
        flat_dict: Dict with keys like 'encoder.layers.0.ffn1.weight'

    Returns:
        Nested dict matching MLX model structure
    """
    result = {}

    for key, value in flat_dict.items():
        parts = key.split(".")
        current = result

        i = 0
        while i < len(parts) - 1:
            part = parts[i]

            # Check if the CURRENT part already exists
            if part not in current:
                # Check if NEXT part is a digit to decide list vs dict
                if i + 1 < len(parts) and parts[i + 1].isdigit():
                    current[part] = []
                else:
                    current[part] = {}

            # If current container is a list, get/create the indexed element
            if isinstance(current[part], list):
                idx = int(parts[i + 1])
                # Extend list if needed
                while len(current[part]) <= idx:
                    current[part].append({})
                current = current[part][idx]
                i += 2  # Skip both the list name and the index
            else:
                current = current[part]
                i += 1

        # Set the final value
        current[parts[-1]] = value

    return result


def load_w2vbert_model(model, weights_path: str):
    """Load pretrained weights into W2V-BERT model.

    Args:
        model: MLX W2V-BERT model instance
        weights_path: Path to .npz weights file

    Returns:
        Model with loaded weights
    """
    print("Loading W2V-BERT pretrained weights...")

    # Load flat weights
    flat_weights = load_w2vbert_weights(weights_path)

    # Filter out weights not needed for inference
    skip_keys = ["masked_spec_embed"]
    flat_weights = {
        k: v for k, v in flat_weights.items() if not any(skip in k for skip in skip_keys)
    }

    print(f"✓ Filtered to {len(flat_weights)} parameters for inference")

    # Manually load weights (simpler and more reliable than nested dict approach)
    print("Loading weights into model...")

    # Feature projection
    model.feature_projection.layer_norm.weight = flat_weights[
        "feature_projection.layer_norm.weight"
    ]
    model.feature_projection.layer_norm.bias = flat_weights["feature_projection.layer_norm.bias"]
    model.feature_projection.projection.weight = flat_weights[
        "feature_projection.projection.weight"
    ]
    model.feature_projection.projection.bias = flat_weights["feature_projection.projection.bias"]

    # Encoder layers
    for layer_idx in range(24):
        layer = model.encoder.layers[layer_idx]
        prefix = f"encoder.layers.{layer_idx}"

        # FFN1
        layer.ffn1.intermediate_dense.weight = flat_weights[
            f"{prefix}.ffn1.intermediate_dense.weight"
        ]
        layer.ffn1.intermediate_dense.bias = flat_weights[f"{prefix}.ffn1.intermediate_dense.bias"]
        layer.ffn1.output_dense.weight = flat_weights[f"{prefix}.ffn1.output_dense.weight"]
        layer.ffn1.output_dense.bias = flat_weights[f"{prefix}.ffn1.output_dense.bias"]
        layer.ffn1_layer_norm.weight = flat_weights[f"{prefix}.ffn1_layer_norm.weight"]
        layer.ffn1_layer_norm.bias = flat_weights[f"{prefix}.ffn1_layer_norm.bias"]

        # Self attention
        layer.self_attn.linear_q.weight = flat_weights[f"{prefix}.self_attn.linear_q.weight"]
        layer.self_attn.linear_q.bias = flat_weights[f"{prefix}.self_attn.linear_q.bias"]
        layer.self_attn.linear_k.weight = flat_weights[f"{prefix}.self_attn.linear_k.weight"]
        layer.self_attn.linear_k.bias = flat_weights[f"{prefix}.self_attn.linear_k.bias"]
        layer.self_attn.linear_v.weight = flat_weights[f"{prefix}.self_attn.linear_v.weight"]
        layer.self_attn.linear_v.bias = flat_weights[f"{prefix}.self_attn.linear_v.bias"]
        layer.self_attn.linear_out.weight = flat_weights[f"{prefix}.self_attn.linear_out.weight"]
        layer.self_attn.linear_out.bias = flat_weights[f"{prefix}.self_attn.linear_out.bias"]
        layer.self_attn.distance_embedding = flat_weights[
            f"{prefix}.self_attn.distance_embedding.weight"
        ]
        layer.self_attn_layer_norm.weight = flat_weights[f"{prefix}.self_attn_layer_norm.weight"]
        layer.self_attn_layer_norm.bias = flat_weights[f"{prefix}.self_attn_layer_norm.bias"]

        # Conv module
        layer.conv_module.layer_norm.weight = flat_weights[
            f"{prefix}.conv_module.layer_norm.weight"
        ]
        layer.conv_module.layer_norm.bias = flat_weights[f"{prefix}.conv_module.layer_norm.bias"]
        layer.conv_module.pointwise_conv1.weight = flat_weights[
            f"{prefix}.conv_module.pointwise_conv1.weight"
        ]
        layer.conv_module.depthwise_conv.weight = flat_weights[
            f"{prefix}.conv_module.depthwise_conv.weight"
        ]
        layer.conv_module.depthwise_layer_norm.weight = flat_weights[
            f"{prefix}.conv_module.depthwise_layer_norm.weight"
        ]
        layer.conv_module.depthwise_layer_norm.bias = flat_weights[
            f"{prefix}.conv_module.depthwise_layer_norm.bias"
        ]
        layer.conv_module.pointwise_conv2.weight = flat_weights[
            f"{prefix}.conv_module.pointwise_conv2.weight"
        ]

        # FFN2
        layer.ffn2.intermediate_dense.weight = flat_weights[
            f"{prefix}.ffn2.intermediate_dense.weight"
        ]
        layer.ffn2.intermediate_dense.bias = flat_weights[f"{prefix}.ffn2.intermediate_dense.bias"]
        layer.ffn2.output_dense.weight = flat_weights[f"{prefix}.ffn2.output_dense.weight"]
        layer.ffn2.output_dense.bias = flat_weights[f"{prefix}.ffn2.output_dense.bias"]
        layer.ffn2_layer_norm.weight = flat_weights[f"{prefix}.ffn2_layer_norm.weight"]
        layer.ffn2_layer_norm.bias = flat_weights[f"{prefix}.ffn2_layer_norm.bias"]

        # Final layer norm
        layer.final_layer_norm.weight = flat_weights[f"{prefix}.final_layer_norm.weight"]
        layer.final_layer_norm.bias = flat_weights[f"{prefix}.final_layer_norm.bias"]

    print("✓ Weights loaded successfully")

    return model
