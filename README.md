# indextts-mlx

IndexTTS-2 text-to-speech inference on Apple Silicon via [MLX](https://github.com/ml-explore/mlx). No PyTorch dependency at runtime.

## Requirements

- Apple Silicon Mac (M1/M2/M3/M4)
- Python 3.11 or 3.12 recommended (required for NeMo/spaCy compatibility; 3.10+ for core synthesis only)
- IndexTTS-2 model weights converted to `.npz` format (see [Weights](#weights))

## Installation

```bash
git clone https://github.com/cestella/indextts_mlx
cd indextts_mlx
python3.11 -m venv venv          # use 3.11 or 3.12 for full feature support
source venv/bin/activate
pip install -e .
```

For development (includes pytest + black):

```bash
pip install -e ".[dev]"
```

## Weights

### Automatic download and conversion (recommended)

`indextts download-weights` downloads all source checkpoints from HuggingFace and converts them to the `.npz` format required by the MLX engine in one step.

**Install conversion dependencies first:**

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install huggingface_hub safetensors
```

**Run the downloader:**

```bash
indextts download-weights --out-dir ~/indextts_weights
```

Or with a separate download cache (useful for re-running without re-downloading):

```bash
indextts download-weights --out-dir ~/indextts_weights --cache-dir ~/indextts_cache
```

Models downloaded and converted:

| Source (HuggingFace) | File(s) | Output |
|---|---|---|
| `IndexTeam/IndexTTS-2` | `gpt.pth` | `gpt.npz` |
| `IndexTeam/IndexTTS-2` | `s2mel.pth` | `s2mel_pytorch.npz` |
| `IndexTeam/IndexTTS-2` | `wav2vec2bert_stats.pt` | `semantic_stats.npz` |
| `IndexTeam/IndexTTS-2` | `feat1.pt` / `feat2.pt` | `speaker_matrix.npz` / `emotion_matrix.npz` |
| `IndexTeam/IndexTTS-2` | `bpe.model` | `bpe.model` |
| `funasr/campplus` | `campplus_cn_common.bin` | `campplus.npz` |
| `facebook/w2v-bert-2.0` | `model.safetensors` | `w2vbert.npz` |
| `amphion/MaskGCT` | `semantic_codec/model.safetensors` | `semantic_codec.npz` |
| `nvidia/bigvgan_v2_22khz_80band_256x` | `bigvgan_generator.pt` | `bigvgan.npz` |

After the download completes, point the CLI at the output directory:

```bash
export INDEXTTS_MLX_WEIGHTS_DIR=~/indextts_weights
# or pass --weights-dir ~/indextts_weights on each invocation
```

### Manual weights

All model weights must be in `.npz` format in a single directory. The default location is read from `INDEXTTS_MLX_WEIGHTS_DIR` (see [Configuration](#configuration)).

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

You also need a SentencePiece BPE tokenizer model. Default: `~/code/tts/index-tts/checkpoints/bpe.model` (override with `INDEXTTS_MLX_BPE_MODEL`).

For `--emo-text`, the fine-tuned Qwen3-0.6B emotion classifier must be present. Default: `~/code/tts/index-tts/checkpoints/qwen0.6bemo4-merge` (override with `INDEXTTS_MLX_QWEN_EMO`).

## Optional dependencies

### Sentence segmentation (long-text synthesis + EPUB extract)

`synthesize_long()` and `indextts extract` use spaCy for sentence boundary detection. Install spaCy and the appropriate language model:

```bash
pip install -e ".[long]"                          # installs spacy + pysbd
python -m spacy download en_core_web_sm           # English
python -m spacy download fr_core_news_sm          # French
python -m spacy download es_core_news_sm          # Spanish
python -m spacy download it_core_news_sm          # Italian
python -m spacy download de_core_news_sm          # German
python -m spacy download pt_core_news_sm          # Portuguese
```

### Text normalization

`synthesize_long()` can normalize text before synthesis (numbers → words, currency, dates, etc.) using [NeMo Text Processing](https://github.com/NVIDIA/NeMo-text-processing). Normalization is optional — if `nemo_text_processing` is not installed it is silently skipped.

**Linux:** prebuilt wheels are available:

```bash
pip install pynini nemo_text_processing
```

**macOS:** `pynini` must be compiled against OpenFst. Requires Python 3.11 or 3.12.

1. **Install OpenFst:**
   ```bash
   brew install openfst
   ```

2. **Build pynini:**
   ```bash
   export CFLAGS="-I/opt/homebrew/opt/openfst/include"
   export CXXFLAGS="-I/opt/homebrew/opt/openfst/include"
   export LDFLAGS="-L/opt/homebrew/opt/openfst/lib"
   pip install --no-cache-dir pynini
   ```

   > **Note:** `nemo_text_processing` pins `pynini==2.1.6.post1`, which does not build against OpenFst 1.8+. Install without a version pin and ignore the warning — it works at runtime.

3. **Install nemo_text_processing:**
   ```bash
   pip install --no-deps nemo_text_processing
   pip install sacremoses cdifflib editdistance inflect joblib pandas regex transformers wget
   ```

4. **Verify:**
   ```bash
   python -c "from nemo_text_processing.text_normalization.normalize import Normalizer; print('ok')"
   ```

### M4B audiobook packaging

`indextts m4b` requires `m4b-tool` (wraps `ffmpeg`):

```bash
brew install m4b-tool
```

`isbnlib` (for ISBN metadata lookup) is included in the package dependencies.

### EPUB extraction

`indextts extract` requires `ebooklib`, `beautifulsoup4`, and `lxml`, all of which are included in the package dependencies. spaCy (see above) is required for `--sentence-per-line` (the default).

## Configuration

All paths can be set via environment variables; constructor arguments take precedence.

| Variable | Default | Description |
|----------|---------|-------------|
| `INDEXTTS_MLX_WEIGHTS_DIR` | `~/code/index-tts-m3-port/prototypes/s2mel_mlx/mlx_weights/` | Directory of `.npz` weight files |
| `INDEXTTS_MLX_BPE_MODEL` | `~/code/tts/index-tts/checkpoints/bpe.model` | SentencePiece BPE tokenizer |
| `INDEXTTS_MLX_QWEN_EMO` | `~/code/tts/index-tts/checkpoints/qwen0.6bemo4-merge` | Qwen3-0.6B emotion classifier (optional) |

## CLI

All commands are subcommands of the unified `indextts` entry point:

```
indextts synthesize         Text-to-speech synthesis
indextts classify-emotions  Classify emotions for JSONL pipeline preparation
indextts extract            Extract plain-text chapters from an EPUB
indextts m4b                Package audio chapters into a .m4b audiobook
indextts web                Web UI: queue, download, extract, synthesize, and package audiobooks
indextts download-weights   Download and convert all model weights from HuggingFace
```

Run `indextts <command> --help` for full option listings.

---

### `indextts synthesize`

Synthesizes speech from text or a file. All input is automatically run through:
**normalize → segment at sentence boundaries → synthesize → stitch**.

```bash
indextts synthesize --text "Hello, world." --spk-audio-prompt speaker.wav
indextts synthesize --file chapter01.txt --voices-dir ~/voices --voice Emma --out ch01.wav
indextts synthesize --file chapter01.jsonl --voices-dir ~/voices --out chapter01.wav
indextts synthesize --file ~/chapters --out ~/audio --out-ext mp3 --voices-dir ~/voices --voice Emma
```

**Input (pick one):**

| Flag | Description |
|------|-------------|
| `--text TEXT` | Inline text to synthesize |
| `--file PATH` | UTF-8 `.txt` file, `.jsonl` segments file, or directory of input files |

**Directory batch mode** (`--file DIR --out DIR`): processes every `.txt` and `.jsonl` file in the input directory, skipping files whose output already exists in the output directory.

| Flag | Default | Description |
|------|---------|-------------|
| `--out-ext wav\|mp3\|pcm` | `mp3` | Output format for directory batch mode |

**Text pipeline controls:**

| Flag | Default | Description |
|------|---------|-------------|
| `--normalize / --no-normalize` | on | Run NeMo text normalization (numbers, dates, currency → spoken form) |
| `--language TEXT` | `english` | Language for normalization and segmentation |
| `--token-target INT` | `250` | BPE tokens per synthesis chunk (~2–3 sentences); chunks always break on sentence boundaries |
| `--silence-ms INT` | `300` | Silence in ms between chunks |
| `--crossfade-ms INT` | `10` | Linear crossfade overlap in ms between chunks; replaces silence when non-zero |

**Speaker source (pick one; `--spk-audio-prompt` wins):**

| Flag | Description |
|------|-------------|
| `--spk-audio-prompt PATH` | Reference audio file (sets speaker timbre) |
| `--voice NAME --voices-dir PATH` | Voice name resolved to `voices_dir/NAME.wav` |

**Emotion controls:**

| Flag | Default | Description |
|------|---------|-------------|
| `--emotion FLOAT` | `1.0` | Internal emo-vec scale: 0.0 = neutral, 1.0 = default, 2.0 = expressive |
| `--emo-vector "f,…,f"` | — | 8 comma-separated floats: `happy,angry,sad,afraid,disgusted,melancholic,surprised,calm` |
| `--emo-alpha FLOAT` | `0.0` | Blend strength for `--emo-audio-prompt` (0 = off, 1 = full blend) |
| `--emo-text TEXT` | — | Natural-language emotion description; auto-classified into an `--emo-vector` by Qwen3-0.6B |
| `--emo-audio-prompt PATH` | — | Reference audio whose emotional character is blended in |

**Determinism:**

| Flag | Default | Description |
|------|---------|-------------|
| `--seed INT` | — | Random seed (default 0 when `--no-use-random`) |
| `--no-use-random` | on | Deterministic output (recommended for audiobooks) |
| `--use-random` | — | Non-deterministic sampling |

**Quality / output:**

| Flag | Default | Description |
|------|---------|-------------|
| `--out PATH` | `output.wav` | Output file; extension sets format (`.wav`, `.mp3`, `.pcm`) |
| `--audio-format wav\|mp3\|pcm` | *(from ext)* | Override format detection |
| `--steps INT` | `10` | CFM diffusion steps (10 = fast, 25 = higher quality) |
| `--temperature FLOAT` | `1.0` | CFM sampling temperature |
| `--cfg-rate FLOAT` | `0.7` | Classifier-free guidance rate |
| `--max-codes INT` | `1500` | Maximum GPT tokens (caps output length) |
| `--gpt-temperature FLOAT` | `0.8` | GPT sampling temperature |
| `--top-k INT` | `30` | Top-k for GPT sampling |
| `--sample-rate INT` | `22050` | Output sample rate (resamples if ≠ 22050) |
| `--play` | — | Play output via `afplay` after synthesis (macOS) |
| `--weights-dir PATH` | — | Override weights directory |
| `--bpe-model PATH` | — | Override BPE model path |

**Batch-mode progress and status:**

| Flag | Description |
|------|-------------|
| `--status DIR` | After every chunk and every completed file, write `DIR/synth_status.json` with chunk ETA, job ETA, files remaining, and avg wall-time per chunk. Used by `indextts web` to display live progress. |
| `-v / --verbose` | Print per-chunk preview, ETA, and realtime factor to stdout. Suppresses tqdm bars. |

**JSONL chapter mode extras:**

| Flag | Description |
|------|-------------|
| `--cache-dir PATH` | Cache directory for segment audio (SHA-256 keyed; re-renders are free) |
| `--emotion-config PATH` | Path to `emotions.json` preset config (auto-detected from `--voices-dir` if not set) |
| `--enable-drift` | Apply bounded per-segment drift to emotion vectors |
| `--end-chime PATH` | Audio file appended to the end of every output file (all modes; resampled if needed) |
| `--list-voices` | List available voice names in `--voices-dir` and exit |

---

### `indextts classify-emotions`

Classifies emotions for each sentence in a plain-text chapter file and writes a JSONL segments file suitable for use with `indextts synthesize --file`.

```bash
indextts classify-emotions chapter01.txt chapter01.jsonl
indextts classify-emotions chapter01.txt chapter01.jsonl --model mlx-community/Qwen2.5-32B-Instruct-4bit
```

| Flag | Default | Description |
|------|---------|-------------|
| `--model TEXT` | *(ClassifierConfig default)* | MLX-LM model repo ID or local path |
| `--chapter-id TEXT` | — | Optional chapter identifier written to every JSONL record |
| `--language TEXT` | `english` | Language for sentence segmentation |
| `--context-window INT` | `5` | Surrounding sentences included as paragraph context |
| `--hysteresis-min-run INT` | `2` | Minimum consecutive non-neutral predictions to keep an emotion label |
| `--max-retries INT` | `3` | Retries per sentence if the model returns an invalid token |
| `--quiet` | — | Suppress per-sentence progress output |

Each output line is a JSONL segment record conforming to `schemas/segment.schema.json`.

---

### `indextts extract`

Extracts chapters from an EPUB file as plain-text `.txt` files. Strips footnotes, tables, code blocks, images, and other non-narration elements. Optionally runs spaCy sentence segmentation so the output has one sentence per line — the format expected by `indextts synthesize --file`.

```bash
# Extract to plain text, one sentence per line (default)
indextts extract book.epub ~/chapters/

# Verbose — show per-chapter word counts
indextts extract book.epub ~/chapters/ -v

# Force spine order instead of TOC
indextts extract book.epub ~/chapters/ --no-toc

# Also write a combined all_chapters.txt
indextts extract book.epub ~/chapters/ --combined

# Skip sentence segmentation (raw paragraph text)
indextts extract book.epub ~/chapters/ --no-sentence-per-line
```

| Flag | Default | Description |
|------|---------|-------------|
| `--toc / --no-toc` | `--toc` | Use the book's Table of Contents; falls back to spine order if TOC yields < 3 chapters |
| `--min-words INT` | `100` | Minimum word count for a spine item to be kept as a chapter |
| `--sentence-per-line / --no-sentence-per-line` | on | Run spaCy sentence segmentation; one sentence per line in output |
| `--language TEXT` | `english` | Language for spaCy segmentation |
| `--combined / --no-combined` | off | Also write `all_chapters.txt` concatenating every chapter |
| `-v / --verbose` | — | Print per-chapter title and word count |

Output files are named `chapter_NN_<title>.txt`. Front and back matter (contents, index, notes, acknowledgments, copyright, etc.) are automatically excluded.

**Typical audiobook workflow:**

```bash
# 1. Extract EPUB to per-chapter text files
indextts extract book.epub ~/chapters/ -v

# 2. Classify emotions for each chapter
for f in ~/chapters/chapter_*.txt; do
    indextts classify-emotions "$f" "${f%.txt}.jsonl"
done

# 3. Synthesize each chapter
indextts synthesize --file ~/chapters --out ~/audio --out-ext mp3 \
    --voices-dir ~/voices --voice Emma

# 4. Package into M4B
indextts m4b --isbn 9780743273565 --chapters-dir ~/audio --out ~/audiobooks
```

---

### `indextts m4b`

Packages a directory of audio chapter files into a `.m4b` audiobook. Fetches book metadata (title, author, year, cover art, description) from an ISBN lookup and invokes `m4b-tool`.

```bash
indextts m4b --isbn 9780743273565 --chapters-dir ~/audio/chapters --out ~/audiobooks
```

| Flag | Default | Description |
|------|---------|-------------|
| `--isbn TEXT` | *(required)* | ISBN-10 or ISBN-13 for metadata and cover art lookup |
| `--chapters-dir PATH` | *(required)* | Directory of audio chapter files (mp3, m4a, wav, …) |
| `--out PATH` | *(required)* | Output directory where the `.m4b` file will be written |
| `--bitrate TEXT` | `64k` | Audio bitrate passed to `m4b-tool` |
| `--jobs TEXT` | `4` | Parallel encoding jobs |
| `--use-filenames-as-chapters / --no-use-filenames-as-chapters` | on | Pass `--use-filenames-as-chapters` to `m4b-tool` |
| `--m4b-arg KEY=VALUE` | — | Extra arguments forwarded to `m4b-tool` (repeatable) |
| `-v / --verbose` | — | Print `m4b-tool` output |

Requires `m4b-tool`: `brew install m4b-tool`.

---

### `indextts web`

A local web application that automates the full audiobook pipeline (download → extract → synthesize → package) via a browser UI. Designed for use from Safari on iOS and Chrome on desktop.

```bash
indextts web --audiobooks-dir ~/audiobooks \
             --voices-dir ./voices \
             --voice british_female \
             --port 5000
```

Then open `http://localhost:5000/` in a browser (or `http://<mac-ip>:5000/` from another device on the same network).

| Flag | Default | Description |
|------|---------|-------------|
| `--audiobooks-dir PATH` | *(required)* | Root directory for all audiobooks. Stores EPUBs, chapter text, chapter MP3s, and final M4B files. The job queue is persisted here as `.queue.json`. |
| `--voices-dir PATH` | — | Directory of voice reference audio files; populates the voice dropdown in the UI |
| `--voice NAME` | — | Default voice name pre-selected in the UI |
| `--port INT` | `5000` | Port to listen on |
| `--host TEXT` | `0.0.0.0` | Bind address; use `127.0.0.1` to restrict to localhost |

**Submitting a job:** provide an EPUB URL and ISBN. The server creates a directory named after the stripped ISBN (e.g. `978-0-06-112008-4` → `9780061120084/`), downloads the EPUB, and runs the full pipeline automatically.

**Directory layout** for each book:

```
{audiobooks-dir}/
└── 9780061120084/
    ├── 9780061120084.epub       ← downloaded EPUB
    ├── chapters_txt/            ← output of indextts extract
    │   ├── chapter_01_….txt
    │   └── …
    ├── chapters_mp3/            ← output of indextts synthesize
    │   ├── chapter_01_….mp3
    │   └── …
    ├── .status/
    │   └── synth_status.json   ← live progress (polled by UI)
    └── <Title> - <Author>.m4b  ← final output
```

**API endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Single-page UI |
| `GET` | `/api/queue` | All jobs, active job, current synth status, voice list |
| `POST` | `/api/submit` | Submit a new job (JSON body: `isbn`, `epub_url`, `voice`, `steps`, `emotion`, `token_target`) |
| `POST` | `/api/cancel/<id>` | Cancel a queued or running job |
| `GET` | `/api/status/<id>` | Job metadata + current `synth_status.json` content |
| `GET` | `/files/<path>` | Serve or browse files under `--audiobooks-dir` |

**Job lifecycle:**

```
queued → running → done
                 → failed
                 → cancelled
          (crash) → interrupted   ← re-submit to retry
```

**Restart behaviour:** The job queue is persisted to `{audiobooks-dir}/.queue.json` after every mutation (atomic tmp-then-rename). On restart:

- **Queued** jobs are fully restored and the worker picks them up automatically — no action required.
- **Running** jobs are marked `interrupted` (not silently re-queued, since partial output may exist in the job directory). Re-submit via the UI if you want to retry.
- **Done / failed / cancelled** jobs remain in the history panel.

#### Web architecture

```
cli/web.py              click entry point; wires together the three components and starts Flask

indextts_mlx/web/
├── queue_manager.py    QueueManager — thread-safe, JSON-backed persistent job store
│                         • submit() / cancel() / update() / get_next_queued()
│                         • atomically writes {audiobooks_dir}/.queue.json after every mutation
│                         • on reload: resets "running" → "interrupted"
│
├── worker.py           Worker(threading.Thread) — single background thread
│                         • polls QueueManager every 2 s for the next queued job
│                         • runs pipeline stages sequentially via subprocess.Popen:
│                             1. wget          (download EPUB)
│                             2. indextts extract  (chapters_txt/)
│                             3. indextts synthesize --status .status/ (chapters_mp3/)
│                             4. indextts m4b  (final .m4b)
│                         • stores Popen handle so cancel() can SIGTERM/SIGKILL it
│                         • reads epub metadata (title/author) via ebooklib after download
│
├── app.py              Flask application factory (create_app)
│                         • all routes are pure JSON except GET / (HTML) and GET /files/
│                         • /files/ path-traversal protected (resolve + relative_to check)
│                         • reads .status/synth_status.json on demand (no background thread)
│
└── templates/
    └── index.html      Single-page UI — vanilla JS, no framework dependencies
                          • polls /api/queue every 2 s (active) or 5 s (idle)
                          • dual progress bars: file-level + chunk-level during synthesis
                          • stat boxes: chunk ETA, job ETA, files remaining, avg s/chunk
                          • mobile-responsive dark theme (CSS custom properties + grid)
```

The `--status DIR` option added to `indextts synthesize` writes `DIR/synth_status.json` atomically after every chunk and every completed file. The Flask `/api/queue` and `/api/status/<id>` endpoints read this file on demand and forward it to the UI, keeping the server stateless with respect to synthesis progress.

---

### `indextts download-weights`

Downloads all required model weights from HuggingFace and converts them to `.npz` format.

```bash
indextts download-weights --out-dir ~/indextts_weights
indextts download-weights --out-dir ~/indextts_weights --cache-dir ~/indextts_cache
```

Requires `torch`, `huggingface_hub`, and `safetensors`:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install huggingface_hub safetensors
```

---

## Python API

### Quick start

```python
from indextts_mlx import IndexTTS2
import soundfile as sf

tts = IndexTTS2()   # loads all models once

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
    voices_dir="~/voices",
    voice="Emma",                     # resolved to voices_dir/Emma.wav

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

**Emotion resolution priority:** `emo_vector` > `emo_text` (when both given, `emo_vector` wins and a warning is issued). `emo_text` auto-enables `use_emo_text`.

**Emotion controls explained:**

| Parameter | Behaviour |
|-----------|-----------|
| `emotion` | Scales the audio-derived emotion vector. Works standalone — no extra files needed. |
| `emo_vector` | 8-float blend of per-category emotion directions from `emotion_matrix.npz`. Requires `emotion_matrix.npz` + `speaker_matrix.npz`. |
| `emo_text` | Natural-language description classified by Qwen3-0.6B into an `emo_vector`. Requires the Qwen checkpoint. |
| `emo_audio_prompt` + `emo_alpha` | Blends the speaker's emotion vector with that of a reference audio clip: `base + alpha * (ref - base)`. |

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

Each JSONL line is a segment record. Per-segment fields override global defaults:

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

### Long-text synthesis

```python
from indextts_mlx.synthesize_long import synthesize_long, LongSynthesisConfig
from indextts_mlx.segmenter import SegmenterConfig

seg_config = SegmenterConfig(language="english", strategy="token_count", token_target=250)
long_config = LongSynthesisConfig(normalize=True, segmenter_config=seg_config)

audio = synthesize_long(
    text,
    tts=tts,
    spk_audio_prompt="speaker.wav",
    config=long_config,
    on_chunk=lambda i, total, text: print(f"[{i+1}/{total}] {text[:60]}"),
    on_chunk_done=lambda i, total, stats: print(f"  {stats['realtime_factor']:.1f}x realtime"),
)
```

### EPUB extraction

```python
from indextts_mlx.epub_extractor import EPUBParser
from pathlib import Path

parser = EPUBParser(Path("book.epub"))
chapters = parser.extract_chapters(use_toc=True, min_words=100)

for ch in chapters:
    print(ch.number, ch.title, ch.word_count)
    Path(f"chapter_{ch.number:02d}.txt").write_text(ch.content)
```

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

## Development

### Code formatting

The project uses [Black](https://black.readthedocs.io/) (line length 100, target Python 3.11):

```bash
source venv/bin/activate
black indextts_mlx/ cli/ tests/          # format in-place
black --check indextts_mlx/ cli/ tests/  # CI check (no writes)
```

### Tests

```bash
source venv/bin/activate
pytest tests/ -v
```

Tests require the default weights and a reference audio file at `~/audiobooks/voices/prunella_scales.wav`. Parity tests (marked `parity`) compare output against torchaudio / HuggingFace Transformers and require those to be installed separately.
