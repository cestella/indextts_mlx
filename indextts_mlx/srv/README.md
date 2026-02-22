# indextts-srv — GPU Resource Management Service

A FastAPI service that serializes GPU-bound ML workloads through a single-model-at-a-time queue. Runs over a Unix socket, manages model loading/unloading, and supports priority-based job scheduling with application affinity.

## Architecture

```
Client (httpx/CLI)
    │
    ▼  Unix socket (/tmp/indextts_srv.sock)
┌─────────────────────────────────────────────┐
│  FastAPI App                                │
│  ├── POST /jobs          submit job         │
│  ├── GET  /jobs/{id}     poll status        │
│  ├── DELETE /jobs/{id}   cancel             │
│  ├── GET  /health        health check       │
│  ├── GET  /queue         queue snapshot     │
│  ├── POST /cpu/normalize NeMo normalize     │
│  └── POST /cpu/segment   spaCy segment      │
│                                             │
│  Background Worker Thread                   │
│  └── picks jobs → ModelManager              │
│       └── ensures one model loaded at once  │
│       └── calls backend.execute(payload)    │
└─────────────────────────────────────────────┘
```

Only one GPU model is loaded at a time. When a job arrives for a different model type, the current model is unloaded (with gc + MLX/MPS cache clearing) before the new one loads.

CPU endpoints (`/cpu/normalize`, `/cpu/segment`) run concurrently with GPU jobs — they don't contend for GPU memory.

## Starting the service

```bash
# Start with defaults (socket at /tmp/indextts_srv.sock)
indextts srv start

# Custom socket and config
indextts srv start --socket /tmp/my.sock --config ~/models.yaml

# CLI health check
indextts srv health
indextts srv queue
```

## API Reference

### POST /jobs

Submit a job to the queue.

```json
{
  "model_type": "tts_indextts",
  "model": "indextts2",
  "application_id": "audiobook-worker",
  "priority": 10,
  "payload": { ... },
  "result_path": "/tmp/output.wav"
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model_type` | string | required | Backend type (see backends below) |
| `model` | string | null | Named model from config; null uses the backend's default |
| `application_id` | string | `"default"` | Groups jobs for affinity scheduling |
| `priority` | int | `10` | 0 = urgent (bypasses grouping), higher = lower priority |
| `payload` | dict | `{}` | Backend-specific parameters (see below) |
| `result_path` | string | null | Path for output file (required for TTS/whisper) |

**Response:** `{"job_id": "uuid"}`

### GET /jobs/{job_id}

Poll job status. Also refreshes the heartbeat (prevents expiry).

**Response:**
```json
{
  "id": "uuid",
  "model_type": "tts_indextts",
  "model": "indextts2",
  "application_id": "audiobook-worker",
  "priority": 10,
  "status": "done",
  "created_at": 1708000000.0,
  "last_heartbeat": 1708000030.0,
  "result_path": "/tmp/output.wav",
  "result": {"status": "ok", "sample_rate": 22050},
  "error": null
}
```

Job statuses: `queued` → `running` → `done` | `failed` | `cancelled` | `expired`

### DELETE /jobs/{job_id}

Cancel a queued job. Returns 404 if the job is already running or completed.

### GET /health

```json
{
  "status": "ok",
  "loaded_model": {"type": "tts_indextts", "name": "indextts2"},
  "queue_depth": 3,
  "uptime_s": 1234.5
}
```

### GET /queue

```json
{
  "active": { ... },
  "queued": [ ... ],
  "recent": [ ... ]
}
```

### POST /cpu/normalize

NeMo text normalization (CPU-only, no GPU contention). Converts written text to spoken form: numbers, dates, currency, times.

```json
{"text": "The $45.50 item ships Jan 3, 2025.", "language": "en"}
```

**Response:** `{"text": "The forty five dollars fifty cents item ships january third twenty twenty five."}`

Requires `nemo_text_processing` + `pynini` (returns 503 if not installed).

### POST /cpu/segment

spaCy-based text segmentation into TTS-sized chunks.

```json
{"text": "Long text...", "language": "english", "strategy": "char_count", "max_chars": 300}
```

**Response:** `{"segments": ["chunk1...", "chunk2..."]}`

Requires `spacy` (returns 503 if not installed).

## Backends

### `tts_indextts` — IndexTTS-2 TTS

Synthesizes speech from text using a reference voice.

**Model config:**
```yaml
tts_indextts:
  default: indextts2
  models:
    indextts2:
      weights_dir: ~/path/to/weights  # optional, uses env default
      bpe_model: ~/path/to/bpe.model  # optional, uses env default
```

**Payload:**
```json
{
  "text": "Hello world.",
  "result_path": "/tmp/out.wav",
  "spk_audio_prompt": "/path/to/speaker.wav",
  "voice": "emma",
  "voices_dir": "~/audiobooks/voices",
  "emotion": 1.0,
  "emo_alpha": 0.5,
  "emo_vector": "0.8,0,0,0,0,0,0.2,0",
  "seed": 42,
  "use_random": false,
  "cfm_steps": 10,
  "temperature": 1.0,
  "gpt_temperature": 0.8,
  "top_k": 200,
  "cfg_rate": 0.7,
  "max_codes": 1500
}
```

Required: `text`, `result_path`, and one of `spk_audio_prompt` or `voice`+`voices_dir`.

**Result:** `{"status": "ok", "result_path": "/tmp/out.wav", "sample_rate": 22050}`

### `llm` — MLX-LM Text Generation

Generates text using a quantized LLM via mlx-lm.

**Model config:**
```yaml
llm:
  default: qwen2.5-7b
  models:
    qwen2.5-7b:
      repo: mlx-community/Qwen2.5-7B-Instruct-4bit
    qwen2.5-14b:
      repo: mlx-community/Qwen2.5-14B-Instruct-4bit
```

**Payload:**
```json
{
  "prompt": "Summarize this article: ...",
  "system_prompt": "You are a news editor.",
  "max_tokens": 1024
}
```

Required: `prompt`. Optional: `system_prompt`, `max_tokens` (default 1024).

**Result:** `{"text": "The generated response..."}`

### `whisperx` — MLX-Whisper Transcription

Transcribes audio files to text.

**Model config:**
```yaml
whisperx:
  default: whisper-large-v3-turbo
  models:
    whisper-large-v3-turbo:
      repo: mlx-community/whisper-large-v3-turbo
    whisper-medium:
      repo: mlx-community/whisper-medium-mlx
```

**Payload:**
```json
{
  "audio_path": "/path/to/audio.mp3",
  "language": "en",
  "result_path": "/tmp/transcript.txt"
}
```

Required: `audio_path`. Optional: `language` (default `"en"`), `result_path` (writes transcript to file).

**Result:** `{"text": "The transcribed text..."}`

### `translation` — SeamlessM4T Text-to-Text

Translates text between languages using SeamlessM4Tv2 (PyTorch/MPS).

**Model config:**
```yaml
translation:
  default: seamless-m4t-v2
  models:
    seamless-m4t-v2:
      repo: facebook/seamless-m4t-v2-large
```

**Payload:**
```json
{
  "text": "Buongiorno, come stai?",
  "src_lang": "ita",
  "tgt_lang": "eng",
  "max_tokens": 1024
}
```

Required: `text`. Optional: `src_lang` (default `"ita"`), `tgt_lang` (default `"eng"`), `max_tokens` (default 1024).

**Result:** `{"text": "Good morning, how are you?"}`

### `tts_mlx_audio` — mlx-audio TTS (Stub)

Placeholder for future F5-TTS and Kokoro model support. Currently raises `NotImplementedError` on execute.

**Model config:**
```yaml
tts_mlx_audio:
  default: f5-tts
  models:
    f5-tts:
      repo: lucasnewman/f5-tts-mlx
```

## Queue Scheduling

The worker picks the next job using priority + affinity:

1. **Priority 0** (urgent): Pure FIFO, no grouping — always picked first
2. **Same model + same app**: If a model is already loaded and more jobs from the same application are queued, prefer those (avoids model thrashing and batches related work)
3. **Same app**: Prefer jobs from the current application even if model differs
4. **FIFO within priority band**: Tiebreak by submission order

Jobs expire if not polled within `heartbeat_timeout_s` (default 300s). Each `GET /jobs/{id}` refreshes the heartbeat.

## Models Config

Located at `~/.config/indextts_srv/models.yaml` (auto-created on first run). Defines available backends and their named model configurations:

```yaml
backends:
  tts_indextts:
    default: indextts2
    models:
      indextts2: {}
  llm:
    default: qwen2.5-7b
    models:
      qwen2.5-7b:
        repo: mlx-community/Qwen2.5-7B-Instruct-4bit
      qwen2.5-14b:
        repo: mlx-community/Qwen2.5-14B-Instruct-4bit
  whisperx:
    default: whisper-large-v3-turbo
    models:
      whisper-large-v3-turbo:
        repo: mlx-community/whisper-large-v3-turbo
  translation:
    default: seamless-m4t-v2
    models:
      seamless-m4t-v2:
        repo: facebook/seamless-m4t-v2-large
  tts_mlx_audio:
    default: f5-tts
    models:
      f5-tts:
        repo: lucasnewman/f5-tts-mlx
```

## CLI Usage

```bash
# Submit a TTS job
indextts srv submit tts_indextts \
  --payload '{"text": "Hello.", "spk_audio_prompt": "speaker.wav", "result_path": "/tmp/out.wav"}' \
  --app-id audiobook-worker

# Submit an LLM job
indextts srv submit llm \
  --payload '{"prompt": "What is 2+2?", "max_tokens": 32}'

# Submit a transcription job
indextts srv submit whisperx \
  --payload '{"audio_path": "/tmp/episode.mp3", "language": "en"}'

# Submit a translation job
indextts srv submit translation \
  --payload '{"text": "Buongiorno", "src_lang": "ita", "tgt_lang": "eng"}'

# Submit with priority 0 (urgent)
indextts srv submit tts_indextts --priority 0 \
  --payload '{"text": "Breaking news.", "spk_audio_prompt": "speaker.wav", "result_path": "/tmp/urgent.wav"}'
```

## Testing

```bash
# Unit tests (always, no models loaded)
uv run pytest tests/test_srv_backends.py tests/test_srv_queue.py tests/test_srv_api.py -v

# Integration tests (loads real models)
uv run pytest tests/test_srv_backends_integration.py -v --srv-integration

# All srv tests
uv run pytest tests/test_srv_*.py -v --srv-integration
```

## Gap Analysis: Web App and news_tracker Coverage

### Can the web app use srv?

The web app (`indextts_mlx/web/`) currently spawns CLI subprocesses for each pipeline stage. Migrating to srv would replace subprocess calls with HTTP job submissions. Coverage:

| Web App Operation | srv Backend | Status |
|---|---|---|
| TTS synthesis (`indextts synthesize`) | `tts_indextts` | Covered |
| Emotion classification (`indextts classify-emotions`) | `llm` | Covered (LLM backend can run the classifier prompt) |
| Text normalization | `/cpu/normalize` | Covered |
| Text segmentation | `/cpu/segment` | Covered |
| EPUB extraction | N/A (CPU, no ML) | Not needed — stays as direct Python call |
| M4B packaging | N/A (CPU, no ML) | Not needed — stays as direct Python call |

**Verdict:** The srv API covers all GPU/ML operations the web app needs. EPUB extraction and M4B packaging are CPU-only and don't need the GPU queue.

### Can news_tracker use srv?

The news_tracker (`~/code/news_tracker/`) pipeline uses five ML models. Coverage:

| news_tracker Operation | srv Backend | Status |
|---|---|---|
| Article translation (SeamlessM4T) | `translation` | Covered |
| Article summarization (Qwen LLM) | `llm` | Covered |
| Story synthesis (Qwen LLM) | `llm` | Covered |
| Op-ed generation (Qwen LLM) | `llm` | Covered |
| Podcast transcription (mlx-whisper) | `whisperx` | Covered |
| Text normalization (NeMo) | `/cpu/normalize` | Covered |
| Sentence splitting (spaCy) | `/cpu/segment` | Covered |
| Embedding similarity (sentence-transformers) | — | **Gap** |

**Gaps:**

1. **Embedding/similarity** — news_tracker uses `sentence-transformers/all-MiniLM-L6-v2` for article grouping and topic matching. This is a lightweight CPU model (not GPU-bound on Apple Silicon), so it arguably doesn't belong in the GPU queue. But if consolidation is desired, a new `embedding` backend would be needed.

2. **Streaming/progressive LLM** — The op-ed builder runs a multi-turn loop (chunk → notes → outline → prose) with the same model loaded. This works naturally with srv since the model stays loaded across sequential jobs from the same `application_id`. However, there's no way to pass state between jobs — each job is independent. The caller would need to manage the progressive context itself and submit each LLM call as a separate job with the accumulated context in the prompt.

3. **Model naming** — news_tracker defaults to `Qwen2.5-14B-Instruct-4bit` while the srv default config has `Qwen2.5-7B-Instruct-4bit`. This is just a config issue — add the 14B model as a named model in `models.yaml`.

**Verdict:** The srv API covers all GPU-bound operations. The embedding gap is minor (CPU model). The main integration effort is replacing direct `mlx_lm.generate()` / `mlx_whisper.transcribe()` / `SeamlessM4T` calls with HTTP job submissions and polling.
