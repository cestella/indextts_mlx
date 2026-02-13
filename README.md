# indextts-mlx

IndexTTS-2 text-to-speech inference on Apple Silicon via [MLX](https://github.com/ml-explore/mlx). No PyTorch dependency.

## Requirements

- Apple Silicon Mac (M1/M2/M3/M4)
- Python 3.10+
- IndexTTS-2 model weights converted to `.npz` format

## Installation

```bash
git clone https://github.com/you/indextts-mlx
cd indextts_mlx
python3 -m venv venv
source venv/bin/activate
pip install -e .
```

For development (includes pytest):

```bash
pip install -e ".[dev]"
```

## Weights

All model weights must be in `.npz` format in a single directory. The default location is:

```
~/code/index-tts-m3-port/prototypes/s2mel_mlx/mlx_weights/
```

Required files:

| File | Description |
|------|-------------|
| `gpt.npz` | UnifiedVoice GPT (text → semantic codes) |
| `w2vbert.npz` | W2V-BERT semantic feature extractor |
| `campplus.npz` | CAMPPlus speaker style encoder |
| `semantic_codec.npz` | RepCodec semantic quantizer |
| `semantic_stats.npz` | W2V-BERT normalization statistics |
| `bigvgan.npz` | BigVGAN vocoder (mel → waveform) |
| `s2mel_pytorch.npz` | S2Mel DiT + CFM + regulator |

You also need a SentencePiece BPE tokenizer model (default: `~/code/tts/index-tts/checkpoints/bpe.model`).

Override defaults with environment variables:

```bash
export INDEXTTS_MLX_WEIGHTS_DIR=/path/to/weights
export INDEXTTS_MLX_BPE_MODEL=/path/to/bpe.model
```

## CLI

```
indextts-tts TEXT --voice PATH [OPTIONS]
```

**Required:**
- `TEXT` — text to synthesize
- `--voice PATH` — reference audio file (sets speaker timbre)

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--out PATH` | `output.wav` | Output WAV file |
| `--emotion FLOAT` | `1.0` | Emotion intensity: 0.0 = neutral, 1.0 = default, 2.0 = expressive |
| `--steps INT` | `10` | CFM diffusion steps (10 = fast, 25 = higher quality) |
| `--temperature FLOAT` | `1.0` | Sampling temperature |
| `--cfg-rate FLOAT` | `0.7` | Classifier-free guidance rate |
| `--max-codes INT` | `1500` | Maximum GPT tokens (caps output length) |
| `--gpt-temperature FLOAT` | `0.8` | GPT sampling temperature (matches original IndexTTS-2) |
| `--top-k INT` | `30` | Top-k for GPT sampling |
| `--weights-dir PATH` | — | Override weights directory |
| `--bpe-model PATH` | — | Override BPE model path |
| `--play` | — | Play output via `afplay` after synthesis (macOS) |

**Examples:**

```bash
# Basic synthesis
indextts-tts "Hello, world." --voice speaker.wav

# High-quality with more expressive delivery
indextts-tts "What a remarkable discovery!" \
  --voice speaker.wav --out out.wav --emotion 1.8 --steps 25

# Fast draft with custom weights
indextts-tts "Quick test." \
  --voice speaker.wav --weights-dir /my/weights --steps 5 --play
```

## Python API

### Quick start

```python
from indextts_mlx import IndexTTS2
import soundfile as sf

tts = IndexTTS2()  # loads all models once

audio = tts.synthesize(
    text="Hello, world.",
    reference_audio="speaker.wav",
)
sf.write("output.wav", audio, 22050)
```

### `IndexTTS2`

```python
IndexTTS2(config: WeightsConfig = None)
```

Loads all six models on construction. Create once and reuse for repeated synthesis.

#### `synthesize()`

```python
audio = tts.synthesize(
    text="...",
    reference_audio="speaker.wav",   # str, Path, or np.ndarray
    *,
    sample_rate=None,     # required if reference_audio is np.ndarray
    emotion=1.0,          # 0.0–2.0
    cfm_steps=10,         # 10=fast, 25=quality
    temperature=1.0,
    max_codes=1500,
    cfg_rate=0.7,
) -> np.ndarray           # float32, mono, 22050 Hz
```

| Parameter | Default | Notes |
|-----------|---------|-------|
| `text` | — | Input text. Uppercase is recommended. |
| `reference_audio` | — | Path string, `pathlib.Path`, or `np.ndarray` float32 |
| `sample_rate` | `None` | Required when `reference_audio` is a numpy array |
| `emotion` | `1.0` | Scales emotional expressiveness of the output |
| `cfm_steps` | `10` | Diffusion steps — more steps = slower but smoother mel |
| `temperature` | `1.0` | CFM noise temperature |
| `max_codes` | `1500` | Hard cap on autoregressive GPT tokens |
| `cfg_rate` | `0.7` | Classifier-free guidance; 0.0 disables it |
| `gpt_temperature` | `0.8` | GPT sampling temperature (0.8 matches original IndexTTS-2) |
| `top_k` | `200` | Top-k for GPT token sampling |

Returns a `np.ndarray` of shape `(N,)`, dtype `float32`, at 22050 Hz.

### `WeightsConfig`

```python
from indextts_mlx import WeightsConfig
from pathlib import Path

config = WeightsConfig(
    weights_dir=Path("/my/weights"),
    bpe_model=Path("/my/bpe.model"),
)
tts = IndexTTS2(config=config)
```

Fields can also be set via environment variables (`INDEXTTS_MLX_WEIGHTS_DIR`, `INDEXTTS_MLX_BPE_MODEL`); constructor arguments take precedence.

Call `config.validate()` to verify all required files exist before loading models.

### One-shot helper

```python
from indextts_mlx import synthesize

audio = synthesize("Hello.", reference_audio="speaker.wav")
```

This loads all models on every call. Use `IndexTTS2` directly for anything beyond a single synthesis.

### Numpy input

```python
import numpy as np, soundfile as sf

audio_ref, sr = sf.read("speaker.wav")
audio_ref = audio_ref.mean(axis=1).astype(np.float32)  # mono

audio = tts.synthesize(
    text="Test.",
    reference_audio=audio_ref,
    sample_rate=sr,
)
```

## Pipeline overview

```
Reference audio (any SR)
  ├─ resample → 16 kHz → [W2V-BERT]  → semantic features (1, T, 1024)
  └─ resample → 22 kHz → [CAMPPlus]  → speaker style    (1, 192)

Text → [SentencePiece BPE] → text tokens

semantic features + emotion → [GPT conditioning]      → cond (1, 34, 1280)
text tokens + conditioning  → [GPT autoregressive]    → semantic codes

semantic codes + gpt latent → [S2Mel regulator]       → upsampled condition
upsampled condition + speaker style + reference mel
                            → [CFM diffusion]          → mel spectrogram (1, 80, T)

mel spectrogram → [BigVGAN] → waveform @ 22050 Hz
```

## Audio specs

| | Value |
|-|-------|
| Output sample rate | 22050 Hz |
| Output format | float32, mono |
| Max reference length | 3 seconds (internal) |
| Mel bins | 80 |
| Hop length | 256 samples |

## Tests

```bash
source venv/bin/activate
pytest tests/ -v
```

Tests require the default weights and a reference audio file at `~/audiobooks/voices/prunella_scales.wav`. Parity tests (marked `parity`) compare output against torchaudio / HuggingFace Transformers and require those to be installed separately.

## Project structure

```
indextts_mlx/
├── indextts_mlx/
│   ├── __init__.py          # Public API: IndexTTS2, WeightsConfig, synthesize
│   ├── config.py            # WeightsConfig dataclass
│   ├── pipeline.py          # IndexTTS2 class + synthesize()
│   ├── models/              # MLX model architectures
│   ├── loaders/             # NPZ → MLX weight loaders
│   └── audio/               # Feature extraction (FBANK, mel)
├── cli/
│   └── tts.py               # indextts-tts entry point
└── tests/                   # pytest suite
```
