"""
MLX S2Mel Pipeline

Complete end-to-end semantic-to-mel generation pipeline using MLX.
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Optional
import numpy as np
from pathlib import Path

from .s2mel_dit_v2 import create_dit_v2_from_config
from .s2mel_cfm import create_cfm_from_dit
from .s2mel_regulator_v2 import InterpolateRegulator

# Hardcoded config matching the IndexTTS-2 s2mel checkpoint (from config.yaml)
_S2MEL_CONFIG = {
    "s2mel": {
        "style_encoder": {"dim": 192},
        "DiT": {
            "hidden_dim": 512,
            "num_heads": 8,
            "depth": 13,
            "in_channels": 80,
            "content_dim": 512,
            "long_skip_connection": True,
            "uvit_skip_connection": True,
            "is_causal": False,
            "final_layer_type": "wavenet",
            "block_size": 8192,
        },
        "wavenet": {
            "hidden_dim": 512,
            "kernel_size": 5,
            "dilation_rate": 1,
            "n_layers": 8,
            "p_dropout": 0.0,
        },
        "length_regulator": {
            "in_channels": 1024,
            "channels": 512,
            "sampling_ratios": [1, 1, 1, 1],
        },
    }
}


def _load_npz_into_model(model, npz_data: dict, prefix: str):
    """Load weights from flat npz dict with a given prefix into an MLX model."""
    updates = {}
    for key, val in npz_data.items():
        if key.startswith(prefix + "."):
            subkey = key[len(prefix) + 1 :]
            updates[subkey] = mx.array(val)
    if updates:
        model.load_weights(list(updates.items()))


class MLXS2MelPipeline:
    """MLX implementation of s2mel pipeline."""

    def __init__(self, config_path: Optional[str] = None, checkpoint_path: Optional[str] = None):
        """Initialize MLX s2mel pipeline.

        Args:
            config_path: Unused (kept for API compatibility). Config is hardcoded.
            checkpoint_path: Path to s2mel weights (.npz preferred, .pth also supported).
                             Defaults to the s2mel_pytorch.npz in the standard weights dir.
        """
        if checkpoint_path is None:
            checkpoint_path = str(
                Path.home()
                / "code/index-tts-m3-port/prototypes/s2mel_mlx/mlx_weights/s2mel_pytorch.npz"
            )

        self.config = _S2MEL_CONFIG

        if str(checkpoint_path).endswith(".npz"):
            self._load_from_npz(checkpoint_path)
        else:
            self._load_from_pth(checkpoint_path)

        self.in_channels = self.config["s2mel"]["DiT"]["in_channels"]

    def _load_from_npz(self, checkpoint_path: str):
        """Load weights from a pre-converted .npz checkpoint."""
        npz = np.load(checkpoint_path)
        npz_dict = {k: npz[k] for k in npz.files}

        # Create regulator
        self.regulator = InterpolateRegulator(
            in_channels=1024,
            channels=512,
            sampling_ratios=[1, 1, 1, 1],
            groups=1,
        )
        reg_updates = {}
        for key, val in npz_dict.items():
            if key.startswith("length_regulator."):
                subkey = key[len("length_regulator.") :]
                arr = mx.array(val)
                # Transpose Conv1d weights: PyTorch (out, in, kernel) → MLX (out, kernel, in)
                if arr.ndim == 3 and "weight" in subkey:
                    arr = mx.transpose(arr, (0, 2, 1))
                reg_updates[subkey] = arr
        self.regulator.load_weights(list(reg_updates.items()), strict=False)

        # Create DiT + CFM — use weight mapper to resolve estimator.* and conv.conv.* keys
        from .s2mel_weight_loader import create_dit_weight_mapper, set_nested_attr

        self.dit, self.dit_config = create_dit_v2_from_config(self.config["s2mel"])
        mapper = create_dit_weight_mapper()
        for key, val in npz_dict.items():
            if not key.startswith("cfm."):
                continue
            pt_name = key[len("cfm.") :]  # strip 'cfm.' → 'estimator.*'
            mlx_path = mapper.map_name(pt_name)
            if mlx_path is None:
                continue
            mlx_val = mx.array(val)
            try:
                set_nested_attr(self.dit, mlx_path, mlx_val)
            except Exception:
                pass
        self.cfm = create_cfm_from_dit(self.dit, n_timesteps=25)

        # gpt_layer
        self.gpt_layer = nn.Sequential(
            nn.Linear(1280, 256),
            nn.Linear(256, 128),
            nn.Linear(128, 1024),
        )
        self.gpt_layer.layers[0].weight = mx.array(npz_dict["gpt_layer.0.weight"])
        self.gpt_layer.layers[0].bias = mx.array(npz_dict["gpt_layer.0.bias"])
        self.gpt_layer.layers[1].weight = mx.array(npz_dict["gpt_layer.1.weight"])
        self.gpt_layer.layers[1].bias = mx.array(npz_dict["gpt_layer.1.bias"])
        self.gpt_layer.layers[2].weight = mx.array(npz_dict["gpt_layer.2.weight"])
        self.gpt_layer.layers[2].bias = mx.array(npz_dict["gpt_layer.2.bias"])

    def _load_from_pth(self, checkpoint_path: str):
        """Load weights from a PyTorch .pth checkpoint (requires torch + yaml)."""
        import torch
        import yaml
        from .s2mel_weight_loader import load_dit_weights, load_regulator_weights

        hf_config = str(
            Path.home()
            / ".cache/huggingface/hub/models--IndexTeam--IndexTTS-2/snapshots/740dcaff396282ffb241903d150ac011cd4b1ede/config.yaml"
        )
        with open(hf_config) as f:
            full_config = yaml.safe_load(f)
        full_config["s2mel"]["wavenet"]["p_dropout"] = 0.0
        self.config = full_config

        checkpoint_torch = torch.load(checkpoint_path, map_location="cpu")

        reg_config = full_config["s2mel"]["length_regulator"]
        self.regulator = InterpolateRegulator(
            in_channels=reg_config["in_channels"],
            channels=reg_config["channels"],
            sampling_ratios=reg_config["sampling_ratios"],
            groups=1,
        )
        load_regulator_weights(
            self.regulator, checkpoint_torch["net"]["length_regulator"], verbose=False
        )

        self.dit, self.dit_config = create_dit_v2_from_config(full_config["s2mel"])
        load_dit_weights(self.dit, checkpoint_torch["net"]["cfm"], verbose=False)
        self.cfm = create_cfm_from_dit(self.dit, n_timesteps=25)

        gpt_layer_weights = checkpoint_torch["net"]["gpt_layer"]
        self.gpt_layer = nn.Sequential(
            nn.Linear(1280, 256),
            nn.Linear(256, 128),
            nn.Linear(128, 1024),
        )
        self.gpt_layer.layers[0].weight = mx.array(gpt_layer_weights["0.weight"].numpy())
        self.gpt_layer.layers[0].bias = mx.array(gpt_layer_weights["0.bias"].numpy())
        self.gpt_layer.layers[1].weight = mx.array(gpt_layer_weights["1.weight"].numpy())
        self.gpt_layer.layers[1].bias = mx.array(gpt_layer_weights["1.bias"].numpy())
        self.gpt_layer.layers[2].weight = mx.array(gpt_layer_weights["2.weight"].numpy())
        self.gpt_layer.layers[2].bias = mx.array(gpt_layer_weights["2.bias"].numpy())

    def generate_mel(
        self,
        semantic_tokens: mx.array,
        style: mx.array,
        target_len: int,
        prompt: Optional[mx.array] = None,
        n_timesteps: int = 25,
        temperature: float = 1.0,
    ) -> mx.array:
        """Generate mel spectrogram using MLX pipeline.

        Args:
            semantic_tokens: Semantic tokens (batch, T_in, content_dim)
            style: Style embedding (batch, style_dim)
            target_len: Target mel length
            prompt: Optional prompt mel (batch, mel_channels, prompt_len)
            n_timesteps: Number of diffusion steps
            temperature: Noise temperature

        Returns:
            Mel spectrogram (batch, mel_channels, T)
        """
        batch_size = semantic_tokens.shape[0]

        # Create target lengths
        x_lens = mx.array([target_len] * batch_size, dtype=mx.int32)

        # Run regulator (upsampling)
        mu, olens = self.regulator(semantic_tokens, ylens=x_lens, f0=None)
        mx.eval(mu)

        # Create prompt if not provided
        if prompt is None:
            prompt = mx.zeros((batch_size, self.in_channels, target_len))

        # Run CFM (diffusion)
        mel = self.cfm.inference(
            mu=mu,
            x_lens=x_lens,
            prompt=prompt,
            style=style,
            f0=None,
            n_timesteps=n_timesteps,
            temperature=temperature,
            inference_cfg_rate=0.0,
        )
        mx.eval(mel)

        return mel

    def generate_mel_regulator_only(
        self,
        semantic_tokens: mx.array,
        target_len: int,
    ) -> mx.array:
        """Run only the regulator (for component-level testing).

        Args:
            semantic_tokens: Semantic tokens (batch, T_in, content_dim)
            target_len: Target mel length

        Returns:
            Upsampled conditioning (batch, T_out, hidden_dim)
        """
        batch_size = semantic_tokens.shape[0]
        ylens = mx.array([target_len] * batch_size, dtype=mx.int32)
        mu, olens = self.regulator(semantic_tokens, ylens=ylens, f0=None)
        mx.eval(mu)
        return mu


def create_mlx_s2mel_pipeline(
    config_path: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
) -> MLXS2MelPipeline:
    """Create MLX s2mel pipeline.

    Args:
        config_path: Path to config.yaml
        checkpoint_path: Path to s2mel checkpoint (.npz or .pth). If None, uses default HuggingFace cache path.

    Returns:
        MLXS2MelPipeline instance
    """
    return MLXS2MelPipeline(config_path=config_path, checkpoint_path=checkpoint_path)
