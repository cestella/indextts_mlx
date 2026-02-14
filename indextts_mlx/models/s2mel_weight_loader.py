"""
PyTorch to MLX Weight Loader

Handles loading weights from PyTorch checkpoint with weight normalization
into MLX model.
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Dict, Optional, Tuple, List
import re

from indextts_mlx.models.s2mel_dit_v2 import DiTV2
from indextts_mlx.models.s2mel_regulator import InterpolateRegulator


def denormalize_weight_norm(weight_g, weight_v):
    """De-normalize weight_norm parameters to get actual weights.

    PyTorch weight_norm stores:
        weight_g: magnitude (out_features, 1) or (out_features,)
        weight_v: direction (out_features, in_features)
        actual_weight = weight_g * (weight_v / ||weight_v||)

    Args:
        weight_g: Weight magnitude tensor
        weight_v: Weight direction tensor

    Returns:
        De-normalized weight tensor
    """
    # Compute norm along input dimension
    if weight_v.dim() == 2:
        # Linear layer: (out_features, in_features)
        norm = torch.norm(weight_v, dim=1, keepdim=True)
    elif weight_v.dim() == 3:
        # Conv1d: (out_features, in_features, kernel_size)
        norm = torch.norm(weight_v.reshape(weight_v.size(0), -1), dim=1, keepdim=True)
        norm = norm.unsqueeze(-1)  # Add kernel_size dim
    else:
        raise ValueError(f"Unexpected weight_v dims: {weight_v.shape}")

    # Ensure weight_g has correct shape for broadcasting
    if weight_g.dim() == 1:
        weight_g = weight_g.unsqueeze(1)
    if weight_v.dim() == 3 and weight_g.dim() == 2:
        weight_g = weight_g.unsqueeze(-1)

    # De-normalize: weight_g * (weight_v / norm)
    weight = weight_g * (weight_v / (norm + 1e-8))

    return weight


def torch_to_mlx(tensor) -> mx.array:
    """Convert PyTorch tensor to MLX array.

    Args:
        tensor: PyTorch tensor

    Returns:
        MLX array
    """
    return mx.array(tensor.detach().cpu().numpy())


class WeightMapper:
    """Maps PyTorch checkpoint parameter names to MLX model paths."""

    def __init__(self):
        """Initialize weight mapper with name mappings."""
        self.mappings: List[Tuple[str, str]] = []

    def add_mapping(self, pt_pattern: str, mlx_path: str):
        """Add a parameter name mapping.

        Args:
            pt_pattern: PyTorch parameter pattern (can include {i} for layer index)
            mlx_path: MLX model path
        """
        self.mappings.append((pt_pattern, mlx_path))

    def map_name(self, pt_name: str) -> Optional[str]:
        """Map PyTorch parameter name to MLX path.

        Args:
            pt_name: PyTorch parameter name from checkpoint

        Returns:
            MLX model path, or None if no mapping found
        """
        for pt_pattern, mlx_path in self.mappings:
            # Convert pattern to regex, escaping dots
            regex_pattern = pt_pattern.replace(".", r"\.")
            regex_pattern = regex_pattern.replace("{i}", r"(\d+)")

            # Match full string
            match = re.fullmatch(regex_pattern, pt_name)
            if match:
                result = mlx_path
                # Replace {i} with captured layer index
                if "{i}" in result and len(match.groups()) > 0:
                    result = result.replace("{i}", match.group(1))
                # Replace $1, $2 etc with captured groups
                for i, group in enumerate(match.groups(), start=1):
                    result = result.replace(f"${i}", group)
                return result

        return None


def create_dit_weight_mapper() -> WeightMapper:
    """Create weight mapper for DiT model.

    Returns:
        WeightMapper configured for DiT
    """
    mapper = WeightMapper()

    # Input embedders
    mapper.add_mapping("estimator.x_embedder.(weight_v|weight_g|weight|bias)", "x_embedder.$1")
    mapper.add_mapping("estimator.cond_projection.(weight|bias)", "cond_projection.$1")
    mapper.add_mapping("estimator.cond_x_merge_linear.(weight|bias)", "cond_x_merge_linear.$1")

    # Timestep embedders
    mapper.add_mapping("estimator.t_embedder.freqs", "t_embedder.freqs")
    mapper.add_mapping("estimator.t_embedder.mlp.0.(weight|bias)", "t_embedder.mlp.layers.0.$1")
    mapper.add_mapping("estimator.t_embedder.mlp.2.(weight|bias)", "t_embedder.mlp.layers.2.$1")

    mapper.add_mapping("estimator.t_embedder2.freqs", "t_embedder2.freqs")
    mapper.add_mapping("estimator.t_embedder2.mlp.0.(weight|bias)", "t_embedder2.mlp.layers.0.$1")
    mapper.add_mapping("estimator.t_embedder2.mlp.2.(weight|bias)", "t_embedder2.mlp.layers.2.$1")

    # Transformer layers
    mapper.add_mapping(
        "estimator.transformer.layers.{i}.attention.wqkv.weight",
        "transformer.layers.{i}.attention.wqkv.weight",
    )
    mapper.add_mapping(
        "estimator.transformer.layers.{i}.attention.wo.weight",
        "transformer.layers.{i}.attention.wo.weight",
    )

    mapper.add_mapping(
        "estimator.transformer.layers.{i}.feed_forward.w1.weight",
        "transformer.layers.{i}.feed_forward.w1.weight",
    )
    mapper.add_mapping(
        "estimator.transformer.layers.{i}.feed_forward.w2.weight",
        "transformer.layers.{i}.feed_forward.w2.weight",
    )
    mapper.add_mapping(
        "estimator.transformer.layers.{i}.feed_forward.w3.weight",
        "transformer.layers.{i}.feed_forward.w3.weight",
    )

    # RMSNorm weights (now using RMSNorm instead of LayerNorm with affine=False)
    mapper.add_mapping(
        "estimator.transformer.layers.{i}.attention_norm.norm.weight",
        "transformer.layers.{i}.attention_norm.weight",
    )
    mapper.add_mapping(
        "estimator.transformer.layers.{i}.attention_norm.project_layer.(weight|bias)",
        "transformer.layers.{i}.attention_norm_modulation.$2",
    )

    mapper.add_mapping(
        "estimator.transformer.layers.{i}.ffn_norm.norm.weight",
        "transformer.layers.{i}.ffn_norm.weight",
    )
    mapper.add_mapping(
        "estimator.transformer.layers.{i}.ffn_norm.project_layer.(weight|bias)",
        "transformer.layers.{i}.ffn_norm_modulation.$2",
    )

    mapper.add_mapping(
        "estimator.transformer.layers.{i}.skip_in_linear.(weight|bias)",
        "transformer.layers.{i}.skip_in_linear.$2",
    )

    # Transformer final norm with AdaLN modulation
    mapper.add_mapping("estimator.transformer.norm.norm.weight", "transformer.norm.weight")
    mapper.add_mapping(
        "estimator.transformer.norm.project_layer.(weight|bias)", "transformer.norm_modulation.$1"
    )

    # Skip and residual projections
    mapper.add_mapping("estimator.skip_linear.(weight|bias)", "skip_linear.$1")
    mapper.add_mapping("estimator.res_projection.(weight|bias)", "res_projection.$1")

    # WaveNet layers (note: .conv.conv due to weight_norm wrapping)
    # Shared global conditioning layer (now properly handles 8192 channels = 2*n_layers*hidden)
    mapper.add_mapping(
        "estimator.wavenet.cond_layer.conv.conv.(weight_v|weight_g|weight|bias)",
        "wavenet.cond_layer.$1",
    )

    # Dilated convolution layers (per-block)
    mapper.add_mapping(
        "estimator.wavenet.in_layers.{i}.conv.conv.(weight_v|weight_g|weight|bias)",
        "wavenet.in_layers.{i}.conv.$2",
    )

    # Residual/skip projection layers (separate from blocks, matches PyTorch)
    mapper.add_mapping(
        "estimator.wavenet.res_skip_layers.{i}.conv.conv.(weight_v|weight_g|weight|bias)",
        "wavenet.res_skip_layers.{i}.$2",
    )

    # Conv layers
    mapper.add_mapping("estimator.conv1.(weight|bias)", "conv1.$1")
    mapper.add_mapping("estimator.conv2.(weight|bias)", "conv2.$1")

    # Final layer
    mapper.add_mapping("estimator.final_layer.norm_final.weight", "final_layer.norm_final.weight")
    mapper.add_mapping(
        "estimator.final_layer.linear.(weight_v|weight_g|weight|bias)", "final_layer.linear.$1"
    )
    mapper.add_mapping(
        "estimator.final_layer.adaLN_modulation.1.(weight|bias)",
        "final_layer.adaLN_modulation.layers.1.$1",
    )

    return mapper


def create_regulator_weight_mapper() -> WeightMapper:
    """Create weight mapper for InterpolateRegulator.

    Returns:
        WeightMapper configured for regulator
    """
    mapper = WeightMapper()

    mapper.add_mapping("content_in_proj.(weight|bias)", "content_in_proj.$1")
    mapper.add_mapping("embedding.weight", "embedding.weight")
    mapper.add_mapping("mask_token", "mask_token")
    mapper.add_mapping("model.{i}.(weight|bias)", "model.{i}.$2")  # Numbered layers

    return mapper


def load_regulator_weights(
    mlx_model, pytorch_checkpoint: Dict, verbose: bool = True
) -> Tuple[int, int, List[str]]:
    """Load weights from PyTorch checkpoint into MLX regulator model.

    Args:
        mlx_model: MLX InterpolateRegulator model
        pytorch_checkpoint: PyTorch state dict (checkpoint['net']['length_regulator'])
        verbose: Print loading progress

    Returns:
        Tuple of (loaded_count, total_pytorch_params, unmapped_keys)
    """
    if verbose:
        print("=" * 80)
        print("LOADING REGULATOR WEIGHTS FROM PYTORCH")
        print("=" * 80)

    mapper = create_regulator_weight_mapper()

    loaded_count = 0
    unmapped_keys = []

    for pt_name, pt_tensor in pytorch_checkpoint.items():
        # Map parameter name
        mlx_path = mapper.map_name(pt_name)
        if mlx_path is None:
            unmapped_keys.append(pt_name)
            continue

        # Convert and load
        mlx_value = torch_to_mlx(pt_tensor)

        # Transpose Conv1d weights: PyTorch (out, in, kernel) -> MLX (out, kernel, in)
        if mlx_value.ndim == 3 and ".weight" in mlx_path:
            mlx_value = mx.transpose(mlx_value, (0, 2, 1))

        # Set parameter in model
        try:
            set_nested_attr(mlx_model, mlx_path, mlx_value)
            loaded_count += 1
            if verbose and loaded_count % 5 == 0:
                print(f"  Loaded {loaded_count} parameters...")
        except Exception as e:
            if verbose:
                print(f"  ⚠️  Failed to load {mlx_path}: {e}")

    if verbose:
        print(f"\n✓ Loaded {loaded_count} parameters")
        if unmapped_keys:
            print(f"\n⚠️  {len(unmapped_keys)} parameters not mapped:")
            for key in unmapped_keys:
                print(f"    {key}")

    return loaded_count, len(pytorch_checkpoint), unmapped_keys


def load_dit_weights(
    mlx_model: DiTV2, pytorch_checkpoint: Dict, verbose: bool = True
) -> Tuple[int, int, List[str]]:
    """Load weights from PyTorch checkpoint into MLX DiT model.

    Args:
        mlx_model: MLX DiT v2 model
        pytorch_checkpoint: PyTorch state dict (checkpoint['net']['cfm'])
        verbose: Print loading progress

    Returns:
        Tuple of (loaded_count, total_pytorch_params, unmapped_keys)
    """
    if verbose:
        print("=" * 80)
        print("LOADING DiT WEIGHTS FROM PYTORCH")
        print("=" * 80)

    mapper = create_dit_weight_mapper()

    loaded_count = 0
    unmapped_keys = []
    weight_norm_pairs = {}  # Collect weight_v/weight_g pairs

    # First pass: collect all parameters
    for pt_name, pt_tensor in pytorch_checkpoint.items():
        if not pt_name.startswith("estimator."):
            continue

        # Handle weight_norm: collect pairs
        if ".weight_v" in pt_name or ".weight_g" in pt_name:
            base_name = pt_name.replace(".weight_v", "").replace(".weight_g", "")
            if base_name not in weight_norm_pairs:
                weight_norm_pairs[base_name] = {}

            if ".weight_v" in pt_name:
                weight_norm_pairs[base_name]["weight_v"] = pt_tensor
            else:
                weight_norm_pairs[base_name]["weight_g"] = pt_tensor
            continue

        # Map parameter name
        mlx_path = mapper.map_name(pt_name)
        if mlx_path is None:
            unmapped_keys.append(pt_name)
            continue

        # Convert and load
        mlx_value = torch_to_mlx(pt_tensor)

        # Transpose Conv1d weights: PyTorch (out, in, kernel) -> MLX (out, kernel, in)
        if (
            mlx_value.ndim == 3
            and ".weight" in mlx_path
            and ("conv" in mlx_path.lower() or "cond_layer" in mlx_path or "res_skip" in mlx_path)
        ):
            mlx_value = mx.transpose(mlx_value, (0, 2, 1))

        # Set parameter in model
        try:
            set_nested_attr(mlx_model, mlx_path, mlx_value)
            loaded_count += 1
            if verbose and loaded_count % 20 == 0:
                print(f"  Loaded {loaded_count} parameters...")
        except Exception as e:
            if verbose:
                print(f"  ⚠️  Failed to load {mlx_path}: {e}")

    # Second pass: handle weight_norm pairs
    for base_name, pair in weight_norm_pairs.items():
        if "weight_v" not in pair or "weight_g" not in pair:
            if verbose:
                print(f"  ⚠️  Incomplete weight_norm pair for {base_name}")
            continue

        # De-normalize
        weight = denormalize_weight_norm(pair["weight_g"], pair["weight_v"])

        # Map name (use .weight suffix)
        pt_name = base_name + ".weight"
        mlx_path = mapper.map_name(pt_name)
        if mlx_path is None:
            unmapped_keys.append(base_name)
            continue

        # Convert and load
        mlx_value = torch_to_mlx(weight)

        # Transpose Conv1d weights: PyTorch (out, in, kernel) -> MLX (out, kernel, in)
        if (
            mlx_value.ndim == 3
            and ".weight" in mlx_path
            and ("conv" in mlx_path.lower() or "cond_layer" in mlx_path or "res_skip" in mlx_path)
        ):
            mlx_value = mx.transpose(mlx_value, (0, 2, 1))

        try:
            set_nested_attr(mlx_model, mlx_path, mlx_value)
            loaded_count += 1
            if verbose and loaded_count % 20 == 0:
                print(f"  Loaded {loaded_count} parameters...")
        except Exception as e:
            if verbose:
                print(f"  ⚠️  Failed to load {mlx_path}: {e}")

        # Also load bias if present
        bias_name = base_name + ".bias"
        if bias_name in pytorch_checkpoint:
            mlx_bias_path = mapper.map_name(bias_name)
            if mlx_bias_path:
                mlx_bias = torch_to_mlx(pytorch_checkpoint[bias_name])
                try:
                    set_nested_attr(mlx_model, mlx_bias_path, mlx_bias)
                    loaded_count += 1
                except Exception as e:
                    if verbose:
                        print(f"  ⚠️  Failed to load bias {mlx_bias_path}: {e}")

    if verbose:
        print(f"\n✓ Loaded {loaded_count} parameters")
        if unmapped_keys:
            print(f"\n⚠️  {len(unmapped_keys)} parameters not mapped:")
            for key in unmapped_keys[:10]:
                print(f"    {key}")
            if len(unmapped_keys) > 10:
                print(f"    ... and {len(unmapped_keys) - 10} more")

    return loaded_count, len(pytorch_checkpoint), unmapped_keys


def set_nested_attr(obj, path: str, value):
    """Set nested attribute using dot-separated path.

    Args:
        obj: Object to set attribute on
        path: Dot-separated path (e.g., "transformer.layers.0.attention.wqkv.weight")
        value: Value to set
    """
    parts = path.split(".")
    for part in parts[:-1]:
        # Handle list indexing
        if part.isdigit():
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part)

    # Set final attribute
    final_part = parts[-1]
    if hasattr(obj, final_part):
        setattr(obj, final_part, value)
    else:
        raise AttributeError(f"No attribute '{final_part}' in {type(obj)}")
