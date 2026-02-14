# indextts-mlx

IndexTTS-2 text-to-speech inference on Apple Silicon via [MLX](https://github.com/ml-explore/mlx). No PyTorch dependency.

## Requirements

- Apple Silicon Mac (M1/M2/M3/M4)
- Python 3.10+
- IndexTTS-2 model weights converted to `.npz` format

## Installation

```bash
git clone https://github.com/cestella/indextts_mlx
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

Optional files (enable `--emo-vector` and `--emo-text`):

| File | Description |
|------|-------------|
| `emotion_matrix.npz` | 73×1280 per-category emotion direction vectors |
| `speaker_matrix.npz` | 73×192 speaker-similarity lookup matrix |

You also need a SentencePiece BPE tokenizer model (default: `~/code/tts/index-tts/checkpoints/bpe.model`).

For `--emo-text`, the fine-tuned Qwen3-0.6B emotion classifier must be present (default: `~/code/tts/index-tts/checkpoints/qwen0.6bemo4-merge`).

Override defaults with environment variables:

```bash
export INDEXTTS_MLX_WEIGHTS_DIR=/path/to/weights
export INDEXTTS_MLX_BPE_MODEL=/path/to/bpe.model
export INDEXTTS_MLX_QWEN_EMO=/path/to/qwen0.6bemo4-merge
```

## CLI

```
indextts-tts TEXT [OPTIONS]
```

**Speaker source (pick one; `--spk-audio-prompt` takes priority):**

| Flag | Description |
|------|-------------|
| `--spk-audio-prompt PATH` | Reference audio file (sets speaker timbre) |
| `--voice NAME --voices-dir PATH` | Voice name resolved to `voices_dir/NAME.wav` |

**Emotion controls:**

| Flag | Default | Description |
|------|---------|-------------|
| `--emotion FLOAT` | `1.0` | Internal emo-vec scale: 0.0 = neutral, 1.0 = default, 2.0 = expressive |
| `--emo-vector "f,f,f,f,f,f,f,f"` | — | 8 comma-separated floats: `happy,angry,sad,afraid,disgusted,melancholic,surprised,calm` |
| `--emo-alpha FLOAT` | `0.0` | Blend strength for `--emo-audio-prompt` (0 = no blend, 1 = full) |
| `--emo-text TEXT` | — | Natural-language emotion description; classified by Qwen3-0.6B into an `--emo-vector` automatically |
| `--emo-audio-prompt PATH` | — | Reference audio whose emotional character is blended in |

**Determinism:**

| Flag | Default | Description |
|------|---------|-------------|
| `--seed INT` | — | Random seed (default 0 when `--no-use-random`) |
| `--no-use-random` | on | Deterministic output (default; good for audiobooks) |
| `--use-random` | — | Non-deterministic sampling |

**Quality / output:**

| Flag | Default | Description |
|------|---------|-------------|
| `--out PATH` | `output.wav` | Output WAV file |
| `--steps INT` | `10` | CFM diffusion steps (10 = fast, 25 = higher quality) |
| `--temperature FLOAT` | `1.0` | CFM sampling temperature |
| `--cfg-rate FLOAT` | `0.7` | Classifier-free guidance rate |
| `--max-codes INT` | `1500` | Maximum GPT tokens (caps output length) |
| `--gpt-temperature FLOAT` | `0.8` | GPT sampling temperature |
| `--top-k INT` | `30` | Top-k for GPT sampling |
| `--sample-rate INT` | `22050` | Output sample rate (resamples if != 22050) |
| `--audio-format wav\|pcm` | `wav` | Output format |
| `--play` | — | Play output via `afplay` after synthesis (macOS) |
| `--weights-dir PATH` | — | Override weights directory |
| `--bpe-model PATH` | — | Override BPE model path |

**Audiobook / JSONL chapter mode:**

| Flag | Description |
|------|-------------|
| `--segments-jsonl PATH` | JSONL file; each line is a segment with `text`, optional per-segment overrides |
| `--cache-dir PATH` | Cache directory for segment audio (SHA-256 keyed; re-renders are free) |
| `--list-voices` | List available voice names in `--voices-dir` and exit |

**Examples:**

```bash
# Basic synthesis
indextts-tts "Hello, world." --spk-audio-prompt speaker.wav

# Voices directory
indextts-tts "Hello." --voices-dir ~/voices --voice Emma --out out.wav

# Emotion via text description (uses Qwen3-0.6B classifier)
indextts-tts "What a day!" --spk-audio-prompt speaker.wav --emo-text "joyful and excited"

# Emotion via explicit vector (happy=0.8, calm=0.2)
indextts-tts "What a day!" --spk-audio-prompt speaker.wav \
    --emo-vector "0.8,0,0,0,0,0,0,0.2"

# High-quality deterministic render
indextts-tts "Chapter one." --spk-audio-prompt speaker.wav \
    --steps 25 --seed 42 --no-use-random --out chapter01.wav

# JSONL chapter render
indextts-tts --segments-jsonl chapter01.jsonl --voices-dir ~/voices --out chapter01.wav
```

## Python API

### Quick start

```python
from indextts_mlx import IndexTTS2
import soundfile as sf

tts = IndexTTS2()  # loads all models once

audio = tts.synthesize(
    text="Hello, world.",
    spk_audio_prompt="speaker.wav",
)
sf.write("output.wav", audio, 22050)
```

### `IndexTTS2`

```python
IndexTTS2(config: WeightsConfig = None)
```

Loads all models on construction. Create once and reuse for repeated synthesis.

#### `synthesize()`

```python
audio = tts.synthesize(
    text="...",

    # Speaker source — pick one (spk_audio_prompt wins if both given)
    spk_audio_prompt="speaker.wav",   # str, Path, or np.ndarray
    voices_dir="~/voices",            # directory of .wav files
    voice="Emma",                     # resolved to voices_dir/Emma.wav

    # Backward-compat alias for spk_audio_prompt
    reference_audio=None,

    # Emotion controls
    emotion=1.0,            # internal emo-vec scale (0=neutral, 2=expressive)
    emo_vector=None,        # list of 8 floats: [happy,angry,sad,afraid,
                            #   disgusted,melancholic,surprised,calm]
    emo_text=None,          # natural-language description → Qwen3 → emo_vector
    use_emo_text=None,      # None=auto (True when emo_text is set)
    emo_alpha=0.0,          # blend strength for emo_audio_prompt
    emo_audio_prompt=None,  # path to reference emotion audio

    # Determinism
    seed=None,              # integer seed; default 0 when use_random=False
    use_random=False,       # False = deterministic (default)

    # Quality
    cfm_steps=10,           # 10=fast, 25=quality
    temperature=1.0,
    max_codes=1500,
    cfg_rate=0.7,
    gpt_temperature=0.8,
    top_k=30,
) -> np.ndarray             # float32, mono, 22050 Hz
```

**Speaker resolution priority:** `spk_audio_prompt` > `voice + voices_dir` > error.
`reference_audio` is a backward-compat alias; `spk_audio_prompt` wins if both are supplied.

**Emotion resolution priority:** `emo_vector` > `emo_text` (when both given, `emo_vector` wins and a warning is issued).
`emo_text` auto-enables `use_emo_text` when provided.

**Emotion controls explained:**

| Parameter | Behaviour |
|-----------|-----------|
| `emotion` | Scales the audio-derived emotion vector. Works standalone — no extra files needed. |
| `emo_vector` | 8-float blend of per-category emotion directions from `emotion_matrix.npz`. Requires `emotion_matrix.npz` + `speaker_matrix.npz`. |
| `emo_text` | Natural-language description classified by Qwen3-0.6B into an `emo_vector`. Requires the Qwen checkpoint. |
| `emo_audio_prompt` + `emo_alpha` | Blends the speaker's own emotion vector with that of a reference audio clip: `base + alpha * (ref - base)`. |

### `WeightsConfig`

```python
from indextts_mlx import WeightsConfig
from pathlib import Path

config = WeightsConfig(
    weights_dir=Path("/my/weights"),
    bpe_model=Path("/my/bpe.model"),
    qwen_emo=Path("/my/qwen0.6bemo4-merge"),  # optional; needed for emo_text
)
tts = IndexTTS2(config=config)
```

Fields can also be set via environment variables (`INDEXTTS_MLX_WEIGHTS_DIR`, `INDEXTTS_MLX_BPE_MODEL`, `INDEXTTS_MLX_QWEN_EMO`); constructor arguments take precedence.

Call `config.validate()` to verify all required files exist before loading models.

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

Each JSONL line is a segment record. Per-segment fields override the global defaults:

```jsonl
{"segment_id": 1, "text": "Chapter One.", "segment_type": "heading", "pause_after_ms": 800}
{"segment_id": 2, "text": "She stepped outside.", "segment_type": "narration"}
{"segment_id": 3, "text": "Good morning!", "voice": "Eleanor", "emo_vector": [0.6,0,0,0,0,0,0.3,0.1], "emo_alpha": 0.4}
```

### One-shot helper

```python
from indextts_mlx import synthesize

audio = synthesize("Hello.", spk_audio_prompt="speaker.wav")
```

Loads all models on every call. Use `IndexTTS2` directly for anything beyond a single synthesis.

## Pipeline overview

```
Reference audio (any SR)
  ├─ resample → 16 kHz → [W2V-BERT]  → semantic features (1, T, 1024)
  └─ resample → 22 kHz → [CAMPPlus]  → speaker style    (1, 192)

Text → [SentencePiece BPE] → text tokens

emo_text → [Qwen3-0.6B] → emo_vector (8 floats)   (optional)
emo_vector + speaker_matrix → [emotion_matrix lookup] → emo_vec (1, 1280)  (optional)

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
| Max reference length | 15 seconds (internal) |
| Mel bins | 80 |
| Hop length | 256 samples |

## Tests

```bash
source venv/bin/activate
pytest tests/ -v
```

Tests require the default weights and a reference audio file at `~/audiobooks/voices/prunella_scales.wav`. Parity tests (marked `parity`) compare output against torchaudio / HuggingFace Transformers and require those to be installed separately.
