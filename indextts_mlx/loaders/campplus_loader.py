"""
CAMPPlus Weight Loader

Loads PyTorch CAMPPlus weights into MLX format with proper transposition.
"""

import mlx.core as mx
import numpy as np
from pathlib import Path
from typing import Dict


def transpose_conv2d_weight(weight: np.ndarray) -> np.ndarray:
    """Transpose Conv2d weight from PyTorch to MLX format.

    PyTorch: (out_channels, in_channels, kernel_h, kernel_w)
    MLX: (out_channels, kernel_h, kernel_w, in_channels)
    """
    return np.transpose(weight, (0, 2, 3, 1))


def transpose_conv1d_weight(weight: np.ndarray) -> np.ndarray:
    """Transpose Conv1d weight from PyTorch to MLX format.

    PyTorch: (out_channels, in_channels, kernel_size)
    MLX: (out_channels, kernel_size, in_channels)
    """
    return np.transpose(weight, (0, 2, 1))


def load_campplus_weights(weights_path: Path) -> Dict[str, mx.array]:
    """Load and transpose CAMPPlus weights for MLX.

    Args:
        weights_path: Path to campplus.npz file

    Returns:
        Dictionary of MLX arrays with proper format
    """
    print(f"Loading weights from {weights_path}...")
    pt_weights = np.load(weights_path)

    mlx_weights = {}

    # Process each weight
    for name, weight in pt_weights.items():
        # Skip PyTorch-specific parameters that don't exist in MLX
        if "num_batches_tracked" in name:
            continue

        # Remove 'xvector.' prefix since MLX model doesn't have that container
        if name.startswith("xvector."):
            name = name[8:]  # Remove 'xvector.'

        # Map PyTorch Sequential submodule names to MLX list indices
        # In PyTorch: nonlinear.batchnorm.weight
        # In MLX: nonlinear.0.weight (batchnorm is first in list for 'batchnorm-relu')
        name = name.replace(".nonlinear.batchnorm.", ".nonlinear.0.")
        name = name.replace(".nonlinear.relu.", ".nonlinear.1.")

        # Handle nonlinear1 and nonlinear2 in DenseTDNNLayer
        name = name.replace(".nonlinear1.batchnorm.", ".nonlinear1.0.")
        name = name.replace(".nonlinear2.batchnorm.", ".nonlinear2.0.")

        # Handle out_nonlinear (at top level, no dot prefix)
        if name.startswith("out_nonlinear.batchnorm."):
            name = name.replace("out_nonlinear.batchnorm.", "out_nonlinear.0.")
        elif name.startswith("out_nonlinear.relu."):
            name = name.replace("out_nonlinear.relu.", "out_nonlinear.1.")

        # Determine if it's a conv weight by checking name and shape
        # Conv2d can be named 'conv' or be in a shortcut (indexed as shortcut.0)
        is_conv2d = (
            ".weight" in name
            and weight.ndim == 4
            and ("conv" in name.lower() or ".0.weight" in name)
        )
        # Conv1d weights have 3 dimensions and contain 'linear' in the name
        # This includes TDNN layers (linear), CAM layers (linear1, linear2, linear_local),
        # and transition/dense layers (linear)
        is_conv1d = "linear" in name.lower() and ".weight" in name and weight.ndim == 3

        if is_conv2d:
            # Conv2d: transpose to MLX format
            weight = transpose_conv2d_weight(weight)
        elif is_conv1d:
            # Conv1d: transpose to MLX format
            weight = transpose_conv1d_weight(weight)

        mlx_weights[name] = mx.array(weight)

    print(f"✓ Loaded and transposed {len(mlx_weights)} parameters")
    return mlx_weights


def set_module_weights(module, weights: Dict[str, mx.array], prefix: str = ""):
    """Recursively set weights in a module tree.

    Args:
        module: MLX module to set weights on
        weights: Dictionary of weights
        prefix: Current prefix for nested modules
    """
    # Handle list of layers (like layer1, layer2)
    if isinstance(module, list):
        for i, submodule in enumerate(module):
            set_module_weights(submodule, weights, f"{prefix}.{i}" if prefix else str(i))
        return

    # Set direct weights
    for attr_name in ["weight", "bias", "running_mean", "running_var"]:
        full_name = f"{prefix}.{attr_name}" if prefix else attr_name

        if full_name in weights:
            if hasattr(module, attr_name):
                setattr(module, attr_name, weights[full_name])

    # Recursively handle child modules
    for attr_name in dir(module):
        if attr_name.startswith("_"):
            continue

        attr = getattr(module, attr_name)

        # Skip if it's a method or built-in
        if callable(attr) and not hasattr(attr, "__self__"):
            continue

        # Handle nested modules
        if hasattr(attr, "__dict__") or isinstance(attr, list):
            child_prefix = f"{prefix}.{attr_name}" if prefix else attr_name
            set_module_weights(attr, weights, child_prefix)


def nested_dict_from_flat(flat_dict: Dict[str, mx.array]) -> Dict:
    """Convert flat dictionary with dotted keys to nested dictionary.

    Args:
        flat_dict: Dictionary with keys like 'head.conv1.weight'

    Returns:
        Nested dictionary structure matching MLX module parameters
    """
    nested = {}

    for key, value in flat_dict.items():
        parts = key.split(".")
        current = nested
        parent = None
        parent_key = None

        # Navigate/create nested structure
        for i, part in enumerate(parts[:-1]):
            parent = current
            parent_key = part

            if part.isdigit():
                # Handle list indices
                idx = int(part)

                # If parent isn't a list yet, convert it
                if not isinstance(current, list):
                    # Get the previous key to update in parent
                    if i > 0:
                        prev_key = parts[i - 1]
                        # Check what type parent[prev_key] is
                        if prev_key in parent and not isinstance(parent[prev_key], list):
                            parent[prev_key] = []
                        current = parent[prev_key] if prev_key in parent else []
                    else:
                        current = []

                # Extend list if needed
                while len(current) <= idx:
                    current.append({})
                current = current[idx]
            else:
                # Handle dict keys
                if part not in current:
                    # Check if next part is a digit - if so, make this a list
                    if i + 1 < len(parts) - 1 and parts[i + 1].isdigit():
                        current[part] = []
                    else:
                        current[part] = {}
                current = current[part]

        # Set the value
        current[parts[-1]] = value

    # Convert block1, block2, block3 -> blocks: [...]
    if "block1" in nested:
        blocks = []
        i = 1
        while f"block{i}" in nested:
            block = nested.pop(f"block{i}")

            # Within each block, convert tdnnd1, tdnnd2, ... -> layers: [...]
            if "tdnnd1" in block:
                layers = []
                j = 1
                while f"tdnnd{j}" in block:
                    layers.append(block.pop(f"tdnnd{j}"))
                    j += 1
                block["layers"] = layers

            blocks.append(block)
            i += 1
        nested["blocks"] = blocks

    # Convert transit1, transit2, transit3 -> transits: [...]
    if "transit1" in nested:
        transits = []
        i = 1
        while f"transit{i}" in nested:
            transits.append(nested.pop(f"transit{i}"))
            i += 1
        nested["transits"] = transits

    return nested


def merge_nested_dicts(base: Dict, updates: Dict) -> Dict:
    """Recursively merge updates into base dictionary.

    Args:
        base: Base dictionary (from model.parameters())
        updates: Updates dictionary (from checkpoint)

    Returns:
        Merged dictionary
    """
    result = dict(base)

    for key, value in updates.items():
        if key in result:
            if isinstance(result[key], dict) and isinstance(value, dict):
                # Recursively merge dicts
                result[key] = merge_nested_dicts(result[key], value)
            elif isinstance(result[key], list) and isinstance(value, list):
                # Merge lists element-wise
                merged_list = []
                for i in range(max(len(result[key]), len(value))):
                    if i < len(value) and i < len(result[key]):
                        if isinstance(result[key][i], dict) and isinstance(value[i], dict):
                            merged_list.append(merge_nested_dicts(result[key][i], value[i]))
                        else:
                            merged_list.append(value[i])
                    elif i < len(value):
                        merged_list.append(value[i])
                    else:
                        merged_list.append(result[key][i])
                result[key] = merged_list
            else:
                # Replace value
                result[key] = value
        else:
            result[key] = value

    return result


def load_campplus_model(model, weights_path: Path):
    """Load weights into a CAMPPlus model using MLX's update() method.

    Args:
        model: CAMPPlus MLX model
        weights_path: Path to campplus.npz file
    """
    # Load flat weights
    flat_weights = load_campplus_weights(weights_path)

    print("Converting flat weights to nested structure...")

    # Convert to nested structure
    nested_weights = nested_dict_from_flat(flat_weights)

    # Get current model parameters
    current_params = model.parameters()

    print("Merging with model parameters...")

    # Merge loaded weights with current params
    updated_params = merge_nested_dicts(current_params, nested_weights)

    print("Updating model...")

    # Update the model
    model.update(updated_params)

    print("✓ Weights loaded successfully")
    return model
