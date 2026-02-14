"""Download and convert all model weights required by indextts-mlx.

Downloads source checkpoints from HuggingFace and converts them from
PyTorch / SafeTensors format to the compressed NumPy (.npz) format expected
by the MLX inference engine.

Models downloaded / converted
------------------------------
1.  IndexTTS-2 checkpoints   (IndexTeam/IndexTTS-2 on HuggingFace)
    - gpt.pth            → gpt.npz
    - s2mel.pth          → s2mel_pytorch.npz
    - wav2vec2bert_stats.pt → semantic_stats.npz
    - feat1.pt           → speaker_matrix.npz
    - feat2.pt           → emotion_matrix.npz
    - bpe.model          → bpe.model  (copied as-is)

2.  CAMPPlus speaker encoder (funasr/campplus on HuggingFace)
    - campplus_cn_common.bin → campplus.npz

3.  W2V-BERT feature extractor (facebook/w2v-bert-2.0 on HuggingFace)
    - model.safetensors  → w2vbert.npz

4.  Semantic codec       (amphion/MaskGCT on HuggingFace)
    - semantic_codec/model.safetensors → semantic_codec.npz

5.  BigVGAN vocoder      (nvidia/bigvgan_v2_22khz_80band_256x on HuggingFace)
    - bigvgan_generator.pt → bigvgan.npz

Usage
-----
    indextts-download-weights --out-dir ~/my_weights

    # or via Python
    python -m cli.download_weights --out-dir ~/my_weights
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import click
import numpy as np

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _denormalize_weight_norm(weight_g: "np.ndarray", weight_v: "np.ndarray") -> "np.ndarray":
    """Reconstruct weight from PyTorch weight_norm components.

    PyTorch weight_norm stores:  weight = weight_g * (weight_v / ||weight_v||)
    """
    original_shape = weight_v.shape
    flattened = weight_v.reshape(original_shape[0], -1)
    norm = np.linalg.norm(flattened, axis=-1, keepdims=True)
    norm = norm.reshape(original_shape[0], *([1] * (len(original_shape) - 1)))
    return weight_g * (weight_v / (norm + 1e-8))


def _process_weight_norm(state: dict) -> dict:
    """Merge weight_g / weight_v pairs into a single reconstructed weight."""
    processed = {}
    pairs: dict[str, dict] = {}

    for name in state:
        if name.endswith(".weight_v"):
            base = name[:-9]
            pairs.setdefault(base, {})["v"] = name
        elif name.endswith(".weight_g"):
            base = name[:-9]
            pairs.setdefault(base, {})["g"] = name

    for base, p in pairs.items():
        if "v" in p and "g" in p:
            processed[f"{base}.weight"] = _denormalize_weight_norm(state[p["g"]], state[p["v"]])

    for name, arr in state.items():
        if not (name.endswith(".weight_v") or name.endswith(".weight_g")):
            processed[name] = arr

    return processed


def _transpose_conv1d(weight: "np.ndarray") -> "np.ndarray":
    """PyTorch Conv1d (O, I, K) → MLX (O, K, I)."""
    if weight.ndim == 3:
        return np.transpose(weight, (0, 2, 1))
    return weight


def _transpose_conv2d(weight: "np.ndarray") -> "np.ndarray":
    """PyTorch Conv2d (O, I, H, W) → MLX (O, H, W, I)."""
    if weight.ndim == 4:
        return np.transpose(weight, (0, 2, 3, 1))
    return weight


def _strip_module_prefix(state: dict) -> dict:
    """Remove DDP 'module.' prefix from state dict keys."""
    return {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}


def _torch_load(path: Path) -> dict:
    """Load a PyTorch checkpoint, returning a flat numpy state dict."""
    try:
        import torch
    except ImportError:
        raise SystemExit(
            "torch is required for weight conversion.  "
            "Install it in the current environment:\n"
            "  pip install torch --index-url https://download.pytorch.org/whl/cpu"
        )
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt:
        ckpt = ckpt["model"]
    return {k: v.detach().cpu().numpy() for k, v in ckpt.items()}


# ---------------------------------------------------------------------------
# Per-model converters
# ---------------------------------------------------------------------------


def _convert_gpt(src_dir: Path, out_dir: Path) -> None:
    """gpt.pth → gpt.npz"""
    out = out_dir / "gpt.npz"
    if out.exists():
        click.echo(f"  skip  gpt.npz (already exists)")
        return

    click.echo("  converting gpt.pth …")
    state = _torch_load(src_dir / "gpt.pth")
    state = _strip_module_prefix(state)
    state = _process_weight_norm(state)

    npz = {}
    for name, arr in state.items():
        if "conv" in name.lower() and name.endswith(".weight"):
            if arr.ndim == 3:
                arr = _transpose_conv1d(arr)
            elif arr.ndim == 4:
                arr = _transpose_conv2d(arr)
        npz[name] = arr

    np.savez_compressed(str(out), **npz)
    click.echo(f"  saved  gpt.npz  ({out.stat().st_size / 1024**2:.1f} MB, {len(npz)} params)")


def _convert_s2mel(src_dir: Path, out_dir: Path) -> None:
    """s2mel.pth → s2mel_pytorch.npz"""
    out = out_dir / "s2mel_pytorch.npz"
    if out.exists():
        click.echo(f"  skip  s2mel_pytorch.npz (already exists)")
        return

    click.echo("  converting s2mel.pth …")
    try:
        import torch
    except ImportError:
        raise SystemExit("torch required — see gpt conversion note above")

    ckpt = torch.load(str(src_dir / "s2mel.pth"), map_location="cpu", weights_only=False)
    state_groups = ckpt.get("net", ckpt)

    npz = {}
    for model_name, params in state_groups.items():
        params_np = {k: v.detach().cpu().numpy() for k, v in params.items()}
        params_np = _strip_module_prefix(params_np)
        params_np = _process_weight_norm(params_np)
        for param_name, arr in params_np.items():
            full = f"{model_name}.{param_name}"
            if "conv" in param_name.lower() and param_name.endswith(".weight"):
                if arr.ndim == 3:
                    arr = _transpose_conv1d(arr)
                elif arr.ndim == 4:
                    arr = _transpose_conv2d(arr)
            npz[full] = arr

    np.savez_compressed(str(out), **npz)
    click.echo(
        f"  saved  s2mel_pytorch.npz  ({out.stat().st_size / 1024**2:.1f} MB, {len(npz)} params)"
    )


def _convert_semantic_stats(src_dir: Path, out_dir: Path) -> None:
    """wav2vec2bert_stats.pt → semantic_stats.npz"""
    out = out_dir / "semantic_stats.npz"
    if out.exists():
        click.echo("  skip  semantic_stats.npz (already exists)")
        return

    click.echo("  converting wav2vec2bert_stats.pt …")
    try:
        import torch
    except ImportError:
        raise SystemExit("torch required")

    stats = torch.load(
        str(src_dir / "wav2vec2bert_stats.pt"), map_location="cpu", weights_only=False
    )
    npz = {
        "mean": stats["mean"].numpy(),
        "std": torch.sqrt(stats["var"]).numpy(),
    }
    np.savez_compressed(str(out), **npz)
    click.echo(f"  saved  semantic_stats.npz  (mean {npz['mean'].shape}, std {npz['std'].shape})")


def _convert_speaker_emotion(src_dir: Path, out_dir: Path) -> None:
    """feat1.pt → speaker_matrix.npz, feat2.pt → emotion_matrix.npz"""
    for fname, outname in [("feat1.pt", "speaker_matrix.npz"), ("feat2.pt", "emotion_matrix.npz")]:
        out = out_dir / outname
        if out.exists():
            click.echo(f"  skip  {outname} (already exists)")
            continue

        click.echo(f"  converting {fname} …")
        try:
            import torch
        except ImportError:
            raise SystemExit("torch required")

        feat = torch.load(str(src_dir / fname), map_location="cpu", weights_only=False)
        if isinstance(feat, dict):
            npz = {k: v.detach().cpu().numpy() for k, v in feat.items()}
        else:
            npz = {"matrix": feat.detach().cpu().numpy()}

        np.savez_compressed(str(out), **npz)
        first_shape = next(iter(npz.values())).shape
        click.echo(f"  saved  {outname}  ({first_shape})")


def _copy_bpe(src_dir: Path, out_dir: Path) -> None:
    """bpe.model → bpe.model  (no conversion needed)"""
    out = out_dir / "bpe.model"
    if out.exists():
        click.echo("  skip  bpe.model (already exists)")
        return

    src = src_dir / "bpe.model"
    shutil.copy2(str(src), str(out))
    click.echo(f"  copied bpe.model  ({out.stat().st_size / 1024:.0f} KB)")


def _convert_campplus(out_dir: Path) -> None:
    """funasr/campplus  campplus_cn_common.bin → campplus.npz"""
    out = out_dir / "campplus.npz"
    if out.exists():
        click.echo("  skip  campplus.npz (already exists)")
        return

    click.echo("  downloading campplus_cn_common.bin from funasr/campplus …")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise SystemExit("huggingface_hub is required.  pip install huggingface_hub")

    try:
        import torch
    except ImportError:
        raise SystemExit("torch required")

    path = hf_hub_download("funasr/campplus", filename="campplus_cn_common.bin")
    click.echo(f"  downloaded → {path}")

    state = torch.load(path, map_location="cpu", weights_only=False)
    npz = {k: v.detach().cpu().numpy() for k, v in state.items()}

    np.savez_compressed(str(out), **npz)
    click.echo(f"  saved  campplus.npz  ({out.stat().st_size / 1024**2:.1f} MB, {len(npz)} params)")


def _convert_w2vbert(out_dir: Path) -> None:
    """facebook/w2v-bert-2.0  model.safetensors → w2vbert.npz"""
    out = out_dir / "w2vbert.npz"
    if out.exists():
        click.echo("  skip  w2vbert.npz (already exists)")
        return

    click.echo("  downloading w2v-bert-2.0 from facebook/w2v-bert-2.0 …")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise SystemExit("huggingface_hub is required.  pip install huggingface_hub")

    try:
        from safetensors import safe_open
    except ImportError:
        raise SystemExit("safetensors is required.  pip install safetensors")

    path = hf_hub_download("facebook/w2v-bert-2.0", filename="model.safetensors")
    click.echo(f"  downloaded → {path}")

    npz = {}
    with safe_open(path, framework="numpy") as f:
        for key in f.keys():
            npz[key] = f.get_tensor(key)

    np.savez_compressed(str(out), **npz)
    click.echo(f"  saved  w2vbert.npz  ({out.stat().st_size / 1024**2:.1f} MB, {len(npz)} params)")


def _convert_semantic_codec(out_dir: Path) -> None:
    """amphion/MaskGCT  semantic_codec/model.safetensors → semantic_codec.npz"""
    out = out_dir / "semantic_codec.npz"
    if out.exists():
        click.echo("  skip  semantic_codec.npz (already exists)")
        return

    click.echo("  downloading semantic_codec from amphion/MaskGCT …")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise SystemExit("huggingface_hub is required.  pip install huggingface_hub")

    try:
        from safetensors.torch import load_file
    except ImportError:
        raise SystemExit("safetensors is required.  pip install safetensors")

    path = hf_hub_download(
        "amphion/MaskGCT",
        filename="semantic_codec/model.safetensors",
    )
    click.echo(f"  downloaded → {path}")

    weights_pt = load_file(path)
    npz = {}
    for name, param in weights_pt.items():
        arr = param.cpu().numpy()
        # Transpose Conv1d weights — detected by 3-D shape and typical kernel sizes
        if ".weight" in name and arr.ndim == 3 and arr.shape[-1] in (1, 3, 7, 31):
            arr = np.transpose(arr, (0, 2, 1))
        npz[name] = arr

    np.savez_compressed(str(out), **npz)
    click.echo(
        f"  saved  semantic_codec.npz  ({out.stat().st_size / 1024**2:.1f} MB, {len(npz)} params)"
    )


def _convert_bigvgan(out_dir: Path) -> None:
    """nvidia/bigvgan_v2_22khz_80band_256x  bigvgan_generator.pt → bigvgan.npz"""
    out = out_dir / "bigvgan.npz"
    if out.exists():
        click.echo("  skip  bigvgan.npz (already exists)")
        return

    click.echo("  downloading bigvgan_generator.pt from nvidia/bigvgan_v2_22khz_80band_256x …")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise SystemExit("huggingface_hub is required.  pip install huggingface_hub")

    try:
        import torch
    except ImportError:
        raise SystemExit("torch required")

    path = hf_hub_download(
        "nvidia/bigvgan_v2_22khz_80band_256x",
        filename="bigvgan_generator.pt",
    )
    click.echo(f"  downloaded → {path}")

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt.get("generator", ckpt) if isinstance(ckpt, dict) else ckpt

    npz = {}
    pending_g: dict[str, np.ndarray] = {}

    for name, param in state.items():
        arr = param.cpu().numpy()

        if name.endswith("_g"):
            pending_g[name] = arr
            continue
        elif name.endswith("_v"):
            base = name[:-2]
            g_name = base + "_g"
            if g_name in state:
                weight_g = state[g_name].cpu().numpy()
                arr = _denormalize_weight_norm(weight_g, arr)
            name = base

        # Transpose conv weights
        if ".weight" in name and arr.ndim == 3:
            if "ups." in name:
                # ConvTranspose1d: PyTorch (I, O, K) → MLX (O, K, I)
                arr = np.transpose(arr, (1, 2, 0))
            else:
                # Conv1d: (O, I, K) → (O, K, I)
                arr = np.transpose(arr, (0, 2, 1))

        npz[name] = arr

    np.savez_compressed(str(out), **npz)
    click.echo(f"  saved  bigvgan.npz  ({out.stat().st_size / 1024**2:.1f} MB, {len(npz)} params)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_INDEXTTS2_HF_REPO = "IndexTeam/IndexTTS-2"
_INDEXTTS2_FILES = [
    "gpt.pth",
    "s2mel.pth",
    "wav2vec2bert_stats.pt",
    "feat1.pt",
    "feat2.pt",
    "bpe.model",
]


def _download_indextts2_checkpoints(tmp_dir: Path) -> Path:
    """Download raw IndexTTS-2 checkpoints from HuggingFace into tmp_dir."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise SystemExit("huggingface_hub is required.  pip install huggingface_hub")

    click.echo(f"  downloading IndexTTS-2 checkpoints from {_INDEXTTS2_HF_REPO} …")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    for fname in _INDEXTTS2_FILES:
        dest = tmp_dir / fname
        if dest.exists():
            click.echo(f"    skip  {fname} (cached)")
            continue
        click.echo(f"    fetching {fname} …")
        src = hf_hub_download(_INDEXTTS2_HF_REPO, filename=fname, local_dir=str(tmp_dir))
        click.echo(f"    → {src}")

    return tmp_dir


@click.command("download-weights")
@click.option(
    "--out-dir",
    required=True,
    type=click.Path(file_okay=False, writable=True),
    help="Directory to write converted .npz weights into.",
)
@click.option(
    "--cache-dir",
    default=None,
    type=click.Path(file_okay=False, writable=True),
    help=(
        "Temporary directory for raw downloaded checkpoints.  "
        "Defaults to <out-dir>/.download_cache.  "
        "Can be reused across runs to avoid re-downloading."
    ),
)
@click.option(
    "--skip-existing/--no-skip-existing",
    default=True,
    show_default=True,
    help="Skip conversion steps whose output file already exists.",
)
def download_weights(out_dir: str, cache_dir: str | None, skip_existing: bool) -> None:
    """Download and convert all IndexTTS-2 model weights to MLX format.

    Downloads source checkpoints from HuggingFace and converts them from
    PyTorch / SafeTensors format to the compressed NumPy (.npz) format
    expected by indextts-mlx.

    After this command completes, point indextts-tts at the output directory:

        indextts synthesize --weights-dir <out-dir> --text "Hello world." --voice speaker.wav

    or set the environment variable permanently:

        export INDEXTTS_MLX_WEIGHTS_DIR=<out-dir>

    \b
    Requirements: torch, huggingface_hub, safetensors must be installed.
    Install them with:

        pip install torch --index-url https://download.pytorch.org/whl/cpu
        pip install huggingface_hub safetensors
    """
    out = Path(out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    cache = Path(cache_dir).expanduser().resolve() if cache_dir else out / ".download_cache"

    click.echo(f"Output directory : {out}")
    click.echo(f"Download cache   : {cache}")
    click.echo()

    # ── Step 1: IndexTTS-2 (gpt, s2mel, stats, matrices, bpe) ────────────────
    click.echo("── IndexTTS-2 checkpoints ──────────────────────────────────────")
    raw_dir = _download_indextts2_checkpoints(cache / "IndexTTS-2")
    _convert_gpt(raw_dir, out)
    _convert_s2mel(raw_dir, out)
    _convert_semantic_stats(raw_dir, out)
    _convert_speaker_emotion(raw_dir, out)
    _copy_bpe(raw_dir, out)

    # ── Step 2: CAMPPlus ─────────────────────────────────────────────────────
    click.echo()
    click.echo("── CAMPPlus speaker encoder ────────────────────────────────────")
    _convert_campplus(out)

    # ── Step 3: W2V-BERT ─────────────────────────────────────────────────────
    click.echo()
    click.echo("── W2V-BERT feature extractor ──────────────────────────────────")
    _convert_w2vbert(out)

    # ── Step 4: Semantic codec ────────────────────────────────────────────────
    click.echo()
    click.echo("── Semantic codec (MaskGCT) ────────────────────────────────────")
    _convert_semantic_codec(out)

    # ── Step 5: BigVGAN ──────────────────────────────────────────────────────
    click.echo()
    click.echo("── BigVGAN vocoder ─────────────────────────────────────────────")
    _convert_bigvgan(out)

    # ── Done ──────────────────────────────────────────────────────────────────
    click.echo()
    click.echo("All weights ready.")
    click.echo()

    files = sorted(out.glob("*.npz")) + sorted(out.glob("*.model"))
    total_mb = sum(f.stat().st_size for f in files) / 1024**2
    click.echo(f"Files in {out}:")
    for f in files:
        click.echo(f"  {f.name:35s}  {f.stat().st_size / 1024**2:6.1f} MB")
    click.echo(f"  {'TOTAL':35s}  {total_mb:6.1f} MB")
    click.echo()
    click.echo("Next step — run synthesis:")
    click.echo(
        f'  indextts synthesize --weights-dir "{out}" '
        '--text "Hello world." --voice speaker.wav --out out.wav'
    )


if __name__ == "__main__":
    download_weights()
