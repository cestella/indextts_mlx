# CLAUDE.md — Development Notes for indextts-mlx

## Quick orientation

This is a pure-MLX inference package for IndexTTS-2 TTS on Apple Silicon. It wraps six models in a single `IndexTTS2` class with a `synthesize()` method and a `indextts-tts` CLI entry point.

**Always activate the venv before running anything:**

```bash
source /Users/cstella/code/indextts_mlx/venv/bin/activate
```

---

## Running things

### Tests

```bash
pytest tests/ -v                         # all 22 tests
pytest tests/test_pipeline.py -v        # end-to-end only
pytest tests/ -v -k "not parity"        # skip torchaudio/transformers tests
pytest tests/ -x                        # stop on first failure
```

All tests are non-skipping: they fail if weights or reference audio are missing.

Session-scoped fixtures in `tests/conftest.py` load models once per pytest session. The first run is slow (~30 s model loading); subsequent tests in the same session are fast.

### CLI

```bash
indextts-tts "Hello world." --voice ~/audiobooks/voices/prunella_scales.wav --play
indextts-tts "Hello world." --voice ~/audiobooks/voices/prunella_scales.wav --out /tmp/out.wav --steps 25
```

### Python

```python
from indextts_mlx import IndexTTS2
tts = IndexTTS2()
audio = tts.synthesize("Hello.", reference_audio="~/audiobooks/voices/prunella_scales.wav")
```

---

## Weights

Default directory: `~/code/index-tts-m3-port/prototypes/s2mel_mlx/mlx_weights/`

```
bigvgan.npz
campplus.npz
gpt.npz
s2mel_pytorch.npz
semantic_codec.npz
semantic_stats.npz
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
| GPT generates repetitive codes → noise | Greedy argmax sampling gets stuck in loops; use `gpt_temperature=0.8, top_k=200` (the defaults, matching original IndexTTS-2) |
| S2Mel sounds wrong despite correct architecture | NPZ may be missing biases from weight-normed layers (`x_embedder.bias`, wavenet biases, etc.). Patch NPZ by running the extraction script or loading from `.pth` |
| CFM output is wrong length | Check `n_timesteps` — DiT WaveNet residual requires output T = input T (padding must be same-length) |

---

## Dependency notes

`torch` and `yaml` are **not** imported at module level anywhere in this package. They are lazy-imported inside `MLXS2MelPipeline._load_from_pth()` only, which is the fallback path for loading raw `.pth` checkpoints. Normal NPZ-based loading requires only `mlx`, `numpy`, `soundfile`, `librosa`, `sentencepiece`, and `click`.

Parity tests (`tests/test_kaldi_fbank.py`, `tests/test_seamless_fbank.py`) import `torchaudio` and `transformers`. These are not in `pyproject.toml` and must be installed separately if needed.
