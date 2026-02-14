"""Emotion classifier for audiobook segments using an MLX LLM.

Classifies each sentence in a chapter text into one of 7 emotion labels,
then applies a hysteresis smoothing pass to prevent emotional flicker.

Emotion labels:
    0 = neutral
    1 = mild_emphasis
    2 = indignant
    3 = joyful
    4 = suspense
    5 = melancholic
    6 = calm_authority
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

EMOTION_LABELS = [
    "neutral",
    "mild_emphasis",
    "indignant",
    "joyful",
    "suspense",
    "melancholic",
    "calm_authority",
]

_PROMPT_TEMPLATE = """\
You are an audiobook delivery classifier.

Your task is NOT to classify the emotional content of a sentence.

Your task is to decide whether a professional audiobook narrator
would audibly change vocal delivery from neutral narration
for this sentence, given its paragraph context.

This is professional long-form narration.
Narrators maintain a steady, neutral delivery almost all the time.
Emotion shifts are rare and deliberate.

PRIMARY DECISION:

First ask yourself:
Would a skilled narrator actually change their vocal tone here?

If the answer is no, output 0 (neutral).

Only if the answer is clearly yes should you select a non-neutral label.

CRITICAL PRINCIPLE:

A narrator almost never shifts emotion in expository, analytical,
historical, or essayistic prose.

Even if a sentence contains words like "threatening", "dramatic",
"extraordinary", or evaluative language, that does NOT mean
the narrator changes tone.

The presence of emotional vocabulary does NOT imply a change in delivery.

Neutral (0) should be chosen for the overwhelming majority of sentences.

If unsure, choose 0.

NON-NEUTRAL LABELS (USE RARELY):

1 = mild_emphasis
Only when the sentence structurally demands a different delivery.
This must be visible in the sentence itself:
- Direct address: "Admit it."
- A short, isolated punch sentence after buildup.
- Clear structural stress markers (colon introducing emphasis, deliberate inversion).
- A true exclamation that is not joyful or indignant.

Never use mild_emphasis for:
- Analytical prose
- Historical commentary
- Wit or dry humour
- Rhetorical questions in essays
- Mere interesting or surprising statements

3 = joyful
Only when the narrator would audibly sound happy.
Requires sustained, significant happiness.
Not small wins.
Not mild satisfaction.
Not polite enthusiasm.

4 = suspense
Only when the narration describes an unfolding moment of tension,
uncertainty, or imminent danger.
Not conceptual threats.
Not analytical discussion of risk.
Not historical danger described at a distance.

5 = melancholic
Only when the narrator would audibly sound heavy or sorrowful.
Requires genuine emotional weight.
Not inconvenience.
Not abstract sadness.
Not reflective tone alone.

Rule: If you cannot clearly imagine a narrator changing vocal delivery,
output 0.

OUTPUT RULES:

- Output exactly ONE digit.
- Must be one of: 0, 1, 2, 3, 4, 5
- No words.
- No explanation.
- No whitespace.

Context paragraph:
<<<
{PARAGRAPH_CONTEXT}
>>>

Target sentence:
<<<
{TARGET_SENTENCE}
>>>
"""


@dataclass
class ClassifierConfig:
    """Configuration for the emotion classifier.

    Args:
        model: MLX-LM model path or HuggingFace repo ID.
        max_retries: How many times to retry if the model returns an invalid token.
        context_window: Number of surrounding sentences to include as paragraph context.
        language: Language passed to the segmenter.
        hysteresis_min_run: Minimum consecutive non-neutral predictions required before
                            switching away from neutral. Isolated spikes shorter than
                            this are collapsed back to neutral.
    """

    model: str = "mlx-community/Qwen2.5-32B-Instruct-4bit"
    max_retries: int = 3
    context_window: int = 5
    language: str = "english"
    hysteresis_min_run: int = 2


@dataclass
class SentenceRecord:
    """Internal record for a single classified sentence."""

    segment_id: int
    text: str
    raw_emotion_idx: int  # pre-smoothing
    emotion_idx: int  # post-smoothing
    paragraph_context: str
    chapter_id: Optional[int | str] = None

    @property
    def emotion(self) -> str:
        return EMOTION_LABELS[self.emotion_idx]

    def to_jsonl_dict(self) -> dict:
        d: dict = {
            "segment_id": self.segment_id,
            "text": self.text,
            "emotion": self.emotion,
        }
        if self.chapter_id is not None:
            d["chapter_id"] = self.chapter_id
        return d


def _build_prompt(paragraph_context: str, target_sentence: str) -> str:
    return _PROMPT_TEMPLATE.format(
        PARAGRAPH_CONTEXT=paragraph_context,
        TARGET_SENTENCE=target_sentence,
    )


def _apply_hysteresis(labels: List[int], min_run: int = 2) -> List[int]:
    """Smooth a sequence of emotion labels using a run-length hysteresis rule.

    Any run of non-neutral labels shorter than ``min_run`` is collapsed back to
    neutral (0).  Runs of ``min_run`` or longer are preserved unchanged.

    Args:
        labels: Raw per-sentence emotion indices.
        min_run: Minimum run length to keep a non-neutral emotion.

    Returns:
        Smoothed label list of the same length.
    """
    if not labels:
        return []

    result = list(labels)
    n = len(result)
    i = 0
    while i < n:
        if result[i] != 0:
            # Find the end of this non-neutral run
            j = i
            while j < n and result[j] == result[i]:
                j += 1
            run_len = j - i
            if run_len < min_run:
                for k in range(i, j):
                    result[k] = 0
            i = j
        else:
            i += 1

    return result


class EmotionClassifier:
    """Classifies sentences in a chapter text into emotion labels.

    Lazy-loads the MLX-LM model on first use.

    Example::

        clf = EmotionClassifier(ClassifierConfig())
        records = clf.classify_chapter("path/to/chapter.txt")
        clf.write_jsonl(records, "path/to/output.jsonl")
    """

    def __init__(self, config: ClassifierConfig = None):
        if config is None:
            config = ClassifierConfig()
        self.config = config
        self._model = None
        self._tokenizer = None
        self._generate_kwargs: Optional[dict] = None

    # ── lazy model loading ────────────────────────────────────────────────────

    def _load_model(self):
        if self._model is not None:
            return
        try:
            from mlx_lm import load
        except ImportError as e:
            raise ImportError(
                "mlx_lm is required for emotion classification. " "Install with: pip install mlx-lm"
            ) from e
        print(f"Loading LLM: {self.config.model} …")
        self._model, self._tokenizer = load(self.config.model)
        # Resolve generate kwargs once: newer mlx_lm uses make_sampler,
        # older versions accepted temp/top_p directly.
        try:
            from mlx_lm.sample_utils import make_sampler

            self._generate_kwargs = {"sampler": make_sampler(temp=0.0, top_p=1.0)}
        except ImportError:
            self._generate_kwargs = {"temp": 0.0, "top_p": 1.0, "repetition_penalty": None}
        print("  model loaded.")

    # ── sentence extraction ───────────────────────────────────────────────────

    def _get_sentences(self, text: str) -> List[str]:
        """Return sentences using the project segmenter (spaCy + pySBD)."""
        from indextts_mlx.segmenter import Segmenter, SegmenterConfig

        # Use sentence_count=1 so every sentence is its own chunk, giving us
        # the finest granularity for classification.
        cfg = SegmenterConfig(
            language=self.config.language,
            strategy="sentence_count",
            sentences_per_chunk=1,
        )
        seg = Segmenter(cfg)
        return seg.segment(text)

    # ── single-sentence classification ───────────────────────────────────────

    def _classify_one(self, paragraph_context: str, sentence: str) -> int:
        """Return an emotion index (0-6) for a single sentence.

        Retries up to ``config.max_retries`` times if the model returns an
        invalid token.
        """
        from mlx_lm import generate

        self._load_model()
        prompt = _build_prompt(paragraph_context, sentence)

        # Apply chat template if available
        if hasattr(self._tokenizer, "apply_chat_template") and getattr(
            self._tokenizer, "chat_template", None
        ):
            messages = [{"role": "user", "content": prompt}]
            formatted = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            formatted = prompt

        for attempt in range(self.config.max_retries):
            response = generate(
                self._model,
                self._tokenizer,
                prompt=formatted,
                max_tokens=1,
                verbose=False,
                **self._generate_kwargs,
            )
            token = response.strip()
            if token in ("0", "1", "2", "3", "4", "5", "6"):
                return int(token)
            print(
                f"  [retry {attempt + 1}/{self.config.max_retries}] "
                f"invalid response {token!r} for: {sentence[:60]!r}"
            )

        # Fall back to neutral after exhausting retries
        print(f"  [fallback] defaulting to neutral for: {sentence[:60]!r}")
        return 0

    # ── public API ────────────────────────────────────────────────────────────

    def classify_chapter(
        self,
        text_path: str | Path,
        chapter_id: Optional[str | int] = None,
        verbose: bool = True,
    ) -> List[SentenceRecord]:
        """Classify every sentence in a chapter text file.

        Args:
            text_path: Path to plain-text chapter file.
            chapter_id: Optional chapter identifier added to each record.
            verbose: Print progress while classifying.

        Returns:
            List of SentenceRecord, one per sentence, with smoothed emotions.
        """
        text = Path(text_path).read_text(encoding="utf-8")
        return self.classify_text(text, chapter_id=chapter_id, verbose=verbose)

    def classify_text(
        self,
        text: str,
        chapter_id: Optional[str | int] = None,
        verbose: bool = True,
    ) -> List[SentenceRecord]:
        """Classify every sentence in a text string.

        Args:
            text: Raw chapter text.
            chapter_id: Optional chapter identifier.
            verbose: Print progress while classifying.

        Returns:
            List of SentenceRecord with smoothed emotions.
        """
        sentences = self._get_sentences(text)
        n = len(sentences)
        if verbose:
            print(f"Classifying {n} sentences …")

        raw_labels: List[int] = []

        for i, sentence in enumerate(sentences):
            # Build paragraph context from surrounding sentences
            lo = max(0, i - self.config.context_window)
            hi = min(n, i + self.config.context_window + 1)
            context_sents = sentences[lo:hi]
            paragraph_context = " ".join(context_sents)

            label = self._classify_one(paragraph_context, sentence)
            raw_labels.append(label)

            if verbose:
                print(f"  [{i + 1:4d}/{n}] {EMOTION_LABELS[label]:16s}  {sentence[:70]!r}")

        smoothed_labels = _apply_hysteresis(raw_labels, self.config.hysteresis_min_run)

        records: List[SentenceRecord] = []
        for i, (sentence, raw, smoothed) in enumerate(zip(sentences, raw_labels, smoothed_labels)):
            lo = max(0, i - self.config.context_window)
            hi = min(n, i + self.config.context_window + 1)
            ctx = " ".join(sentences[lo:hi])
            rec = SentenceRecord(
                segment_id=i,
                text=sentence,
                raw_emotion_idx=raw,
                emotion_idx=smoothed,
                paragraph_context=ctx,
                chapter_id=chapter_id,
            )
            records.append(rec)

        if verbose:
            changed = sum(1 for r, s in zip(raw_labels, smoothed_labels) if r != s)
            print(f"Hysteresis smoothed {changed}/{n} labels.")

        return records

    @staticmethod
    def write_jsonl(records: List[SentenceRecord], output_path: str | Path) -> None:
        """Write classified records to a JSONL file, grouping consecutive
        sentences with the same emotion into a single segment.

        Consecutive SentenceRecords sharing the same smoothed emotion are
        merged into one line, with their texts joined by a space. This means
        each output segment represents a run of same-emotion prose and may
        contain multiple sentences — the renderer will chunk it by token
        budget at synthesis time.

        Args:
            records: List of SentenceRecord from classify_chapter/classify_text.
            output_path: Destination .jsonl file path (created or overwritten).
        """
        if not records:
            return

        # Group consecutive records with identical emotion
        groups: List[List[SentenceRecord]] = []
        current: List[SentenceRecord] = [records[0]]
        for rec in records[1:]:
            if rec.emotion == current[0].emotion:
                current.append(rec)
            else:
                groups.append(current)
                current = [rec]
        groups.append(current)

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            for seg_id, group in enumerate(groups):
                d: dict = {
                    "segment_id": seg_id,
                    "text": " ".join(r.text for r in group),
                    "emotion": group[0].emotion,
                }
                if group[0].chapter_id is not None:
                    d["chapter_id"] = group[0].chapter_id
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        print(f"Wrote {len(groups)} segments ({len(records)} sentences) → {out}")
