# CLAUDE.md — Development Notes for indextts-mlx

## Quick orientation

This is a pure-MLX inference package for IndexTTS-2 TTS on Apple Silicon. It wraps six models in a single `IndexTTS2` class with a `synthesize()` method and a `indextts-tts` CLI entry point.

**Two virtual environments:**

| Venv | Purpose | Activate |
|------|---------|---------|
| `venv/` | Normal inference + tests (MLX only) | `source venv/bin/activate` |
| `venv_parity/` | PyTorch parity comparison (has torch, transformers, torchaudio) | `source venv_parity/bin/activate` |

Create them once:

```bash
# Inference venv (Python 3.14, MLX stack)
python3 -m venv venv && venv/bin/pip install -e ".[dev]"

# Parity venv (Python 3.11, PyTorch stack — used for compare_pytorch_mlx.py)
python3.11 -m venv venv_parity && venv_parity/bin/pip install -e ".[parity]"
```

---

## Running things

### Tests

```bash
pytest tests/ -v                         # all tests
pytest tests/test_pipeline.py -v        # end-to-end only
pytest tests/test_voices.py -v          # voice/emo logic only (no weights needed)
pytest tests/ -v -k "not parity"        # skip torchaudio/transformers tests
pytest tests/ -x                        # stop on first failure
```

`tests/test_voices.py` is fast (no model loading). All other tests require weights + reference audio.

Session-scoped fixtures in `tests/conftest.py` load models once per pytest session. The first run is slow (~30 s model loading); subsequent tests in the same session are fast.

### CLI

```bash
# Direct audio file (new: --spk-audio-prompt)
indextts-tts "Hello world." --spk-audio-prompt ~/audiobooks/voices/prunella_scales.wav --play

# Voices directory
indextts-tts "Hello world." --voices-dir ~/audiobooks/voices --voice prunella_scales --out /tmp/out.wav

# List available voices
indextts-tts --list-voices --voices-dir ~/audiobooks/voices

# Emotion vector
indextts-tts "What a day!" --spk-audio-prompt speaker.wav \
    --emo-vector "0.8,0,0,0,0,0,0.2,0" --emo-alpha 0.5

# Deterministic audiobook render
indextts-tts "Hello world." --spk-audio-prompt speaker.wav --seed 42 --no-use-random

# JSONL chapter render
indextts-tts --segments-jsonl chapter01.jsonl --voices-dir ~/voices --out chapter01.wav

# Legacy --voice still works (treated as direct path if no --voices-dir)
indextts-tts "Hello." --voice ~/audiobooks/voices/prunella_scales.wav
```

### Python

```python
from indextts_mlx import IndexTTS2

tts = IndexTTS2()

# New API: spk_audio_prompt
audio = tts.synthesize("Hello.", spk_audio_prompt="speaker.wav")

# Voices directory
audio = tts.synthesize("Hello.", voices_dir="~/voices", voice="Emma")

# Backward-compat: reference_audio still works
audio = tts.synthesize("Hello.", reference_audio="speaker.wav")

# Deterministic
audio = tts.synthesize("Hello.", spk_audio_prompt="speaker.wav", seed=0, use_random=False)
```

### JSONL chapter rendering

```python
from indextts_mlx import render_segments_jsonl, IndexTTS2

tts = IndexTTS2()
render_segments_jsonl(
    "chapter01.jsonl",
    "chapter01.wav",
    tts=tts,
    voices_dir="~/audiobooks/voices",
    seed=0,
    cfm_steps=25,
    cache_dir="/tmp/tts_cache",
)
```

### PyTorch parity comparison

`tests/compare_pytorch_mlx.py` runs both the PyTorch reference implementation and the MLX port on the same input and prints a stage-by-stage tensor comparison table. Use this to identify where outputs diverge.

**Requires** `venv_parity` (has `torch`, `transformers`, `torchaudio`, `mlx` all together).

```bash
venv_parity/bin/python tests/compare_pytorch_mlx.py \
    --voice ~/audiobooks/voices/prunella_scales.wav \
    --text "Despite a deadlock over funding for the agency, lawmakers left town."
```

Options:

| Flag | Description |
|------|-------------|
| `--voice PATH` | Reference audio file (required) |
| `--text TEXT` | Text to synthesize (required) |
| `--save-dir PATH` | Save intermediate tensors as `.npy` files for offline inspection |

Output columns: `name | shape | max|Δ| | cos_sim | pt_mean | mlx_mean`

**Interpreting results:**
- Stages up through `inference regulator` should have `cos≈1.000` and `max|Δ| < 0.1` — large divergence here indicates a weight loading or architecture bug
- `CFM mel output` and `BigVGAN waveform` will differ between runs due to independent random noise in the diffusion sampler — this is expected; `cos≈0.99` on the mel is healthy

**Known baseline (after all fixes):**

| Stage | Expected max\|Δ\| | Expected cos |
|-------|-----------------|-------------|
| seamless fbank | 0.000 | 1.000 |
| W2V-BERT hidden[17] | <0.003 | 1.000 |
| normalized semantic features | <0.002 | 1.000 |
| CAMPPlus speaker style | <0.015 | 1.000 |
| semantic_codec S_ref | 0.000 | 1.000 |
| prompt regulator | 0.000 | 1.000 |
| GPT latent | <0.02 | 1.000 |
| GPT latent proj | <0.003 | 1.000 |
| S_infer | <0.003 | 1.000 |
| inference regulator | <0.001 | 1.000 |
| CFM mel | varies (noise) | ~0.995 |
| BigVGAN waveform | varies (noise) | ~0.02 |

---

## Weights

Default directory: `~/code/index-tts-m3-port/prototypes/s2mel_mlx/mlx_weights/`

```
bigvgan.npz
campplus.npz
emotion_matrix.npz   ← optional; enables emo_vector
gpt.npz
s2mel_pytorch.npz
semantic_codec.npz
semantic_stats.npz
speaker_matrix.npz   ← optional; required alongside emotion_matrix.npz
w2vbert.npz
```

BPE model: `~/code/tts/index-tts/checkpoints/bpe.model`

Override via env vars `INDEXTTS_MLX_WEIGHTS_DIR` / `INDEXTTS_MLX_BPE_MODEL` or constructor args.

---

## Package layout

```
indextts_mlx/
├── __init__.py           exports: IndexTTS2, WeightsConfig, synthesize
├── config.py             WeightsConfig dataclass, env var defaults, validate()
├── pipeline.py           IndexTTS2.__init__ + synthesize()
├── voices.py             resolve_voice(), list_voices(), parse_emo_vector()
├── renderer.py           render_segments_jsonl() — JSONL chapter renderer
│
├── models/               MLX model definitions (no weight loading here)
│   ├── gpt.py            UnifiedVoice: conformer + perceiver + GPT2 + mel head
│   ├── gpt2.py           GPT2Model used inside gpt.py
│   ├── w2vbert.py        Wav2Vec2Bert feature extractor
│   ├── campplus.py       CAMPPlus speaker encoder
│   ├── semantic_codec.py RepCodec quantizer
│   ├── s2mel.py          MLXS2MelPipeline: loads regulator + DiT + CFM from npz
│   ├── s2mel_cfm.py      Continuous Flow Matching solver
│   ├── s2mel_dit_v2.py   DiT v2 + create_dit_v2_from_config()
│   ├── s2mel_regulator_v2.py  InterpolateRegulator
│   ├── s2mel_layers.py   Shared primitives (reflect_pad1d, etc.)
│   ├── s2mel_transformer.py   Transformer blocks for DiT
│   ├── bigvgan.py        BigVGAN vocoder
│   ├── bigvgan_wavenet.py     WaveNet blocks (shared by DiT and BigVGAN)
│   ├── bigvgan_alias_free.py  Anti-alias activation wrappers
│   ├── bigvgan_activations.py Snake/SnakeBeta activations
│   ├── conformer.py      ConformerEncoder used in GPT conditioning
│   ├── perceiver.py      PerceiverResampler used in GPT conditioning
│   └── s2mel_weight_loader.py  WeightMapper + load_dit_weights (used by _load_from_pth)
│
├── loaders/              NPZ → MLX weight loaders (one per model)
│   ├── gpt_loader.py
│   ├── w2vbert_loader.py
│   ├── campplus_loader.py
│   ├── semantic_codec_loader.py
│   ├── bigvgan_loader.py
│   └── s2mel_loader.py   (thin wrapper — s2mel loads itself via MLXS2MelPipeline)
│
└── audio/                Numpy/MLX feature extraction (no model weights)
    ├── kaldi_fbank.py    compute_kaldi_fbank_mlx() → (T, 80) for CAMPPlus
    ├── seamless_fbank.py compute_seamless_fbank() → (T, 160) for W2V-BERT
    └── mel.py            compute_mel_s2mel() → (1, 80, T) reference mel for CFM
```

---

## Key design decisions

### Speaker + emotion resolution rules

Speaker priority (highest → lowest): `spk_audio_prompt` > `voice + voices_dir` > error.

`reference_audio` is a backward-compat alias for `spk_audio_prompt` (positional arg → keyword). If both are supplied, `spk_audio_prompt` wins with a warning.

`voice` without `voices_dir` is treated as a direct file path (backward compat for the old `--voice PATH` CLI flag).

Emotion priority: `emo_vector` > `emo_text` (when both given, `emo_vector` wins and `emo_text` is cleared with a warning). `use_emo_text` is tri-state (None = auto: enabled when `emo_text` is provided).

### Determinism

`use_random=False` (default) seeds both `numpy` and `mlx.random` with `seed` (default 0). This makes top-k sampling and CFM diffusion noise deterministic — important for audiobook production where re-renders should be identical.

Set `use_random=True` to restore stochastic behavior (appropriate for interactive one-shot use).

### Voices directory

`voices.py` provides `resolve_voice(dir, name)` and `list_voices(dir)`. Voice names are `.wav` file stems. Case-sensitive match first; case-insensitive fallback with a warning (handles `Emma` vs `emma`).

### JSONL segment schema

`schemas/segment.schema.json` defines the per-segment record format. Every field except `text` is optional and overrides the global render defaults. Fields: `text`, `voice`, `voices_dir`, `spk_audio_prompt`, `emo_alpha`, `emo_vector`, `emo_text`, `use_emo_text`, `emo_audio_prompt`, `seed`, `use_random`, `pause_before_ms`, `pause_after_ms`, `sample_rate`, `audio_format`.

`renderer.py` caches per-segment audio in `cache_dir/` keyed on a SHA-256 of all synthesis params. Re-running with the same JSONL and params is free after the first render.

### S2Mel config is hardcoded

`models/s2mel.py` contains `_S2MEL_CONFIG` — a hardcoded dict with all DiT, WaveNet, and regulator parameters extracted from IndexTTS-2's `config.yaml`. This avoids a yaml dependency at runtime. If loading from `.pth` (the `_load_from_pth` fallback path), the live config.yaml is still read. The NPZ path is the primary/fast path.

### Weight transposition

MLX Conv1d expects `(out_channels, kernel_size, in_channels)`. PyTorch stores `(out_channels, in_channels, kernel_size)`. All loaders transpose 3-D conv weights on load. The `s2mel_pytorch.npz` was already converted by the prototype's conversion script, so conv weights inside it are already in MLX layout.

### DiT weight mapping

`s2mel_weight_loader.py` contains `WeightMapper` / `create_dit_weight_mapper()`. This maps `cfm.estimator.wavenet.in_layers.0.conv.conv.weight` → `wavenet.in_layers.0.conv.weight` (strips the PyTorch weight-norm `conv.conv` double nesting and the `estimator.` wrapper). Used in `MLXS2MelPipeline._load_from_npz`.

### Regulator strict=False

The NPZ has `length_regulator.embedding.weight` and `length_regulator.mask_token` which aren't present in `InterpolateRegulator`. Load is done with `strict=False` to ignore them silently.

### Emotion parameter

`emotion` (float 0–2) is passed as `emotion_scale` into `UnifiedVoice.get_full_conditioning_34()`. Internally: `cond_with_emo = cond_32 + emotion_scale * emo_vec[:, None, :]`. 0.0 = the base conditioning only; 1.0 = default blend; values above 1.0 amplify the emotion vector.

### emo_vector and emo_matrix

`emo_vector` (8 floats: happy, angry, sad, afraid, disgusted, melancholic, surprised, calm) maps to the `emotion_matrix.npz` / `speaker_matrix.npz` files in the weights directory. Both files have 73 rows split by `emo_num = [3,17,2,8,4,5,10,24]` into 8 per-category groups.

Pipeline:
1. For each category, find the row in `speaker_matrix` (192-dim) most cosine-similar to the speaker's CAMPPlus style embedding.
2. Pick the corresponding row from `emotion_matrix` (1280-dim).
3. Weighted sum: `emovec_mat = sum(emo_vector[i] * emo_matrix_row[i])`
4. Blend with audio-derived emovec: `final = emovec_mat + (1 - sum(emo_vector)) * audio_emovec`

`emo_alpha` is used with `emo_audio_prompt`: `final = base_emovec + alpha * (emo_emovec - base_emovec)` (interpolation between speaker's own emovec and reference emotion audio emovec).

`emo_text` / `use_emo_text` are accepted as API parameters but not yet wired into computation (future work: requires text-to-emovec encoder).

### GPT generation loop

The autoregressive loop in `pipeline.py` runs up to `max_codes` steps (default 1500). The stop token is `gpt_model.stop_mel_token` (8193). A 200-character sentence typically produces ~545 codes before the stop token.

The loop uses **top-k sampling** with `gpt_temperature=0.8, top_k=200` (matching original IndexTTS-2 defaults), not greedy argmax. Pure greedy decoding gets stuck in repetitive loops on many sentences, producing noise in the tail of the audio.

---

## Adding a new test

Tests live in `tests/`. Session-scoped fixtures in `conftest.py` provide loaded models. Example skeleton:

```python
# tests/test_mymodel.py
import numpy as np
import mlx.core as mx

def test_mymodel_shape(some_fixture):
    x = mx.zeros((1, 100, 512))
    out = some_fixture(x)
    mx.eval(out)
    assert np.array(out).shape == (1, 100, 80)
    assert np.isfinite(np.array(out)).all()
```

Add the fixture in `conftest.py` if it needs to be session-scoped and shared.

---

## Common failure modes

| Symptom | Likely cause |
|---------|-------------|
| `KeyError: 'hidden_dim'` in s2mel | `_S2MEL_CONFIG['s2mel']['DiT']` has wrong key names — must match `create_dit_v2_from_config` |
| `ValueError: [broadcast_shapes]` in WaveNet | WaveNet `kernel_size` in config doesn't match loaded weights (must be 5, not 3) |
| `ValueError: Received N parameters not in model` | Using `strict=True` on a model with extra keys in the NPZ — add `strict=False` |
| `ValueError: indices must be integral` in semantic_codec | `quantize()` returns `(codes, quantized_out)` — use `codes, _ = quantize(...)`, not `_, codes = ...` |
| GPT loop never hits stop token | `max_codes` too small for long input. Default is 1500; each character ≈ 2–4 codes |
| GPT generates repetitive codes → noise | Greedy argmax sampling gets stuck in loops; use `gpt_temperature=0.8, top_k=30` (matching original IndexTTS-2) |
| GPT latent diverges from PyTorch (cos < 1.000) | `forward_for_latent` must apply `self.final_norm` before extracting mel hidden states — matches PyTorch `get_logits` which does `enc = self.final_norm(enc)` |
| S2Mel sounds wrong despite correct architecture | NPZ may be missing biases from weight-normed layers (`x_embedder.bias`, wavenet biases, etc.). Patch NPZ by running the extraction script or loading from `.pth` |
| CFM output is wrong length | Check `n_timesteps` — DiT WaveNet residual requires output T = input T (padding must be same-length) |

---

## Dependency notes

`torch` and `yaml` are **not** imported at module level anywhere in this package. They are lazy-imported inside `MLXS2MelPipeline._load_from_pth()` only, which is the fallback path for loading raw `.pth` checkpoints. Normal NPZ-based loading requires only `mlx`, `numpy`, `soundfile`, `librosa`, `sentencepiece`, and `click`.

Parity tests (`tests/test_kaldi_fbank.py`, `tests/test_seamless_fbank.py`) import `torchaudio` and `transformers`. These are not in `pyproject.toml` and must be installed separately if needed.
