"""Emotion + pause classifier for audiobook segments using an MLX LLM.

Classifies each sentence in a chapter text into one of 7 emotion labels and
one of 5 pause labels, then applies a hysteresis smoothing pass to emotion to
prevent emotional flicker.

Emotion labels:
    0 = neutral
    1 = mild_emphasis
    2 = indignant
    3 = joyful
    4 = suspense
    5 = melancholic
    6 = calm_authority

Pause labels (pause *after* this sentence):
    0 = none       (~0 ms)
    1 = short      (~250 ms)
    2 = neutral    (~550 ms)
    3 = long       (~1100 ms)
    4 = dramatic   (~2500 ms)
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

PAUSE_LABELS = [
    "none",
    "short",
    "neutral",
    "long",
    "dramatic",
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
Only when the NARRATOR would audibly lower their voice and slow their
pace to create dread or unease — not when the content merely involves
tension or danger.

Suspense requires ALL of the following to be true:
- The sentence is pure third-person narration (not dialogue, not action-beat
  sound effects like "Zap." or "Bang.").
- The narrator would genuinely sound hushed, slow, or ominous.
- The moment is actively unfolding RIGHT NOW in the story — not described
  in retrospect, not summarised, not mentioned in passing.

Never use suspense for:
- Dialogue, even if the speaker describes something frightening.
- Sentences spoken by a character (calm, robotic, or otherwise).
- Short punchy action beats (sound effects, single-word impacts).
- Scientific, analytical, or explanatory sentences.
- Sentences where a calm narrator would still read flatly despite the topic.
- Historical or backstory descriptions of past danger.
- Statements of fact or status reports, even alarming ones
  (e.g. "Priscilla is missing." is information delivery, not suspense).
- Physical descriptions of a character or object moving or approaching,
  unless the sentence itself contains explicit fear/horror language.

If in doubt, use mild_emphasis (1) or neutral (0).

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

_PAUSE_PROMPT_TEMPLATE = """\
You are an audiobook pacing director.

Your task is to decide how long a pause the narrator should insert
AFTER the following sentence, considering both the punctuation at the
end of the sentence and the narrative context.

PAUSE TYPES AND THEIR MEANINGS:

0 = none
  No silence. Sentence runs directly into the next.
  Use for mid-clause fragments, em-dash continuations, or list items
  that flow into each other without any structural break.

1 = short (~0.25 s)
  Brief micro-pause.
  Use after a comma-like beat at sentence end, after dialogue tags
  where narration continues immediately, or after an em dash (—) that
  ends a sentence mid-thought.

2 = neutral (~0.55 s)
  Standard sentence-end pause.
  Default for ordinary prose sentences ending with a period, question
  mark, or exclamation point with no special context.

3 = long (~1.1 s)
  Extended pause for structural transitions.
  Use when moving between clearly distinct topics or scenes, after a
  section-ending summary sentence, or after ellipsis (...) trailing off.
  Ellipsis (...) strongly suggests a long or dramatic pause.

4 = dramatic (~2.5 s)
  Maximum pause for high-impact moments.
  Use only when the sentence ends an important scene, a major revelation,
  or a deliberate cliff-hanger. Ellipsis (...) at a climactic moment
  can warrant dramatic. Reserve for genuine page-turning beats.

PUNCTUATION HINTS (apply these first, then adjust for context):

  .   → neutral (default)
  ?   → neutral
  !   → neutral or short (exclamations rarely need a long beat)
  —   → short (interrupted thought; listener expects immediate continuation)
  …   → long or dramatic (trailing off; let the silence land)
  ,   → none or short (mid-clause; unusual for a sentence to end here)

OUTPUT RULES:

- Output exactly ONE digit.
- Must be one of: 0, 1, 2, 3, 4
- No words.
- No explanation.
- No whitespace.

Context paragraph:
<<<
{PARAGRAPH_CONTEXT}
>>>

Target sentence (classify the pause AFTER this sentence):
<<<
{TARGET_SENTENCE}
>>>
"""


@dataclass
class ClassifierConfig:
    """Configuration for the emotion + pause classifier.

    Args:
        model: MLX-LM model path or HuggingFace repo ID.
        max_retries: How many times to retry if the model returns an invalid token.
        context_window: Number of surrounding sentences to include as paragraph context.
        language: Language passed to the segmenter.
        hysteresis_min_run: Minimum consecutive non-neutral predictions required before
                            switching away from neutral. Isolated spikes shorter than
                            this are collapsed back to neutral.
        use_boundary_detection: If True, use semantic boundary detection to upgrade
                                pause labels at detected boundaries.
        boundary_model: Sentence-transformers model for boundary detection.
        boundary_window: Window size (sentences per side) for cosine similarity.
        boundary_threshold: Cosine similarity threshold for boundary detection.
        boundary_min_pause: Minimum pause label (int) assigned at a detected boundary.
    """

    model: str = "mlx-community/Qwen2.5-32B-Instruct-4bit"
    max_retries: int = 3
    context_window: int = 5
    language: str = "english"
    hysteresis_min_run: int = 2
    use_boundary_detection: bool = True
    boundary_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    boundary_window: int = 3
    boundary_threshold: float = 0.85
    boundary_min_pause: int = 3  # "long"


@dataclass
class SentenceRecord:
    """Internal record for a single classified sentence."""

    segment_id: int
    text: str
    raw_emotion_idx: int  # pre-smoothing
    emotion_idx: int  # post-smoothing
    pause_idx: int  # pause after this sentence
    paragraph_context: str
    chapter_id: Optional[int | str] = None

    @property
    def emotion(self) -> str:
        return EMOTION_LABELS[self.emotion_idx]

    @property
    def pause(self) -> str:
        return PAUSE_LABELS[self.pause_idx]

    def to_jsonl_dict(self) -> dict:
        d: dict = {
            "segment_id": self.segment_id,
            "text": self.text,
            "emotion": self.emotion,
            "pause_after": self.pause,
        }
        if self.chapter_id is not None:
            d["chapter_id"] = self.chapter_id
        return d


def _build_emotion_prompt(paragraph_context: str, target_sentence: str) -> str:
    return _PROMPT_TEMPLATE.format(
        PARAGRAPH_CONTEXT=paragraph_context,
        TARGET_SENTENCE=target_sentence,
    )


def _build_pause_prompt(paragraph_context: str, target_sentence: str) -> str:
    return _PAUSE_PROMPT_TEMPLATE.format(
        PARAGRAPH_CONTEXT=paragraph_context,
        TARGET_SENTENCE=target_sentence,
    )


def _update_dialogue_state(in_dialogue: bool, sentence: str) -> bool:
    """Return updated dialogue-tracking state after processing *sentence*.

    Rules:
    - A sentence that starts with a ``"`` character opens (or continues)
      dialogue, so the result is ``True``.
    - If we are already inside dialogue, we stay inside until the sentence
      contains a closing ``"`` somewhere after its first character, at which
      point dialogue ends (result is ``False``).
    - Otherwise the state is unchanged.

    This lets the classifier know that ``Priscilla is missing.`` is still
    inside the robot's speech even though its opening quote appeared on the
    previous sentence.
    """
    stripped = sentence.strip()
    if stripped.startswith('"'):
        # Sentence opens (or continues) dialogue.
        # Check whether it also closes (i.e. has a closing " after position 0).
        closes = '"' in stripped[1:]
        return not closes  # still in dialogue iff no closing quote
    if in_dialogue:
        # Mid-dialogue sentence: closes when it contains a " anywhere.
        closes = '"' in stripped
        return not closes
    return False


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
    """Classifies sentences in a chapter text into emotion + pause labels.

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
        """Return sentences using the project segmenter (spaCy + pySBD).

        Orphan punctuation fragments (e.g. ``".`` left over after spaCy splits
        a dialogue sentence from its closing quote) are dropped — they contain
        no speakable content and would confuse the classifier.
        """
        import re as _re
        from indextts_mlx.segmenter import Segmenter, SegmenterConfig

        # Use sentence_count=1 so every sentence is its own chunk, giving us
        # the finest granularity for classification.
        cfg = SegmenterConfig(
            language=self.config.language,
            strategy="sentence_count",
            sentences_per_chunk=1,
        )
        seg = Segmenter(cfg)
        raw = seg.segment(text)
        # Drop fragments that contain fewer than 3 alphabetic characters —
        # these are orphan punctuation/quote artefacts, not real sentences.
        return [s for s in raw if len(_re.findall(r"[A-Za-z]", s)) >= 3]

    # ── single-sentence classification ───────────────────────────────────────

    def _classify_one_emotion(self, paragraph_context: str, sentence: str) -> int:
        """Return an emotion index (0-6) for a single sentence."""
        from mlx_lm import generate

        self._load_model()
        prompt = _build_emotion_prompt(paragraph_context, sentence)
        formatted = self._format_prompt(prompt)

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
                f"invalid emotion response {token!r} for: {sentence[:60]!r}"
            )

        print(f"  [fallback] defaulting to neutral emotion for: {sentence[:60]!r}")
        return 0

    def _classify_one_pause(self, paragraph_context: str, sentence: str) -> int:
        """Return a pause index (0-4) for the pause after a single sentence."""
        from mlx_lm import generate

        self._load_model()
        prompt = _build_pause_prompt(paragraph_context, sentence)
        formatted = self._format_prompt(prompt)

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
            if token in ("0", "1", "2", "3", "4"):
                return int(token)
            print(
                f"  [retry {attempt + 1}/{self.config.max_retries}] "
                f"invalid pause response {token!r} for: {sentence[:60]!r}"
            )

        print(f"  [fallback] defaulting to neutral pause for: {sentence[:60]!r}")
        return 2  # "neutral"

    def _format_prompt(self, prompt: str) -> str:
        """Apply chat template if available, else return prompt as-is."""
        if hasattr(self._tokenizer, "apply_chat_template") and getattr(
            self._tokenizer, "chat_template", None
        ):
            messages = [{"role": "user", "content": prompt}]
            return self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return prompt

    # ── public API ────────────────────────────────────────────────────────────

    def classify_chapter(
        self,
        text_path: str | Path,
        chapter_id: Optional[str | int] = None,
        verbose: bool = True,
        on_sentence: Optional[callable] = None,
        head: Optional[int] = None,
    ) -> List[SentenceRecord]:
        """Classify every sentence in a chapter text file.

        Args:
            text_path: Path to plain-text chapter file.
            chapter_id: Optional chapter identifier added to each record.
            verbose: Print progress while classifying.
            on_sentence: Optional callback(sentence_index, total) called after
                         each sentence is classified (for status reporting).
            head: If set, only classify the first N sentences.

        Returns:
            List of SentenceRecord, one per sentence, with smoothed emotions.
        """
        text = Path(text_path).read_text(encoding="utf-8")
        return self.classify_text(text, chapter_id=chapter_id, verbose=verbose, on_sentence=on_sentence, head=head)

    def classify_text(
        self,
        text: str,
        chapter_id: Optional[str | int] = None,
        verbose: bool = True,
        on_sentence: Optional[callable] = None,
        head: Optional[int] = None,
    ) -> List[SentenceRecord]:
        """Classify every sentence in a text string.

        Two LLM calls are made per sentence: one for emotion, one for the
        pause that follows it.

        Args:
            text: Raw chapter text.
            chapter_id: Optional chapter identifier.
            verbose: Print progress while classifying.
            on_sentence: Optional callback(sentence_index, total) called after
                         each sentence is classified.
            head: If set, only classify the first N sentences.

        Returns:
            List of SentenceRecord with smoothed emotions.
        """
        sentences = self._get_sentences(text)
        if head is not None:
            sentences = sentences[:head]
        n = len(sentences)
        if verbose:
            print(f"Classifying {n} sentences (emotion + pause) …")

        raw_emotion_labels: List[int] = []
        raw_pause_labels: List[int] = []

        # Track whether we are inside an open dialogue quote across sentences.
        # A sentence that starts with " opens dialogue; it closes when a sentence
        # contains a closing " (after the first character).
        _in_dialogue = False

        for i, sentence in enumerate(sentences):
            # Build paragraph context from surrounding sentences
            lo = max(0, i - self.config.context_window)
            hi = min(n, i + self.config.context_window + 1)
            context_sents = sentences[lo:hi]
            paragraph_context = " ".join(context_sents)

            # Update dialogue-tracking state before classifying.
            _in_dialogue = _update_dialogue_state(_in_dialogue, sentence)

            emo_label = self._classify_one_emotion(paragraph_context, sentence)
            # Hard-cap: dialogue sentences must never be classified as suspense,
            # melancholic, or joyful — only neutral (0) or mild_emphasis (1).
            _DIALOGUE_MAX_EMO = 1  # mild_emphasis
            if _in_dialogue and emo_label > _DIALOGUE_MAX_EMO:
                emo_label = 0

            pause_label = self._classify_one_pause(paragraph_context, sentence)
            raw_emotion_labels.append(emo_label)
            raw_pause_labels.append(pause_label)

            if verbose:
                print(
                    f"  [{i + 1:4d}/{n}] emo={EMOTION_LABELS[emo_label]:16s} "
                    f"pause={PAUSE_LABELS[pause_label]:8s}  {sentence[:60]!r}"
                )

            if on_sentence is not None:
                on_sentence(i, n)

        # Apply hysteresis smoothing to emotions only
        smoothed_labels = _apply_hysteresis(raw_emotion_labels, self.config.hysteresis_min_run)

        # Optionally apply boundary detection to boost pause labels
        if self.config.use_boundary_detection:
            try:
                from indextts_mlx.boundary_detector import detect_boundaries

                boundaries = detect_boundaries(
                    sentences,
                    text,
                    model_name=self.config.boundary_model,
                    window=self.config.boundary_window,
                    threshold=self.config.boundary_threshold,
                )
                for idx in boundaries:
                    if idx < len(raw_pause_labels):
                        raw_pause_labels[idx] = max(
                            raw_pause_labels[idx], self.config.boundary_min_pause
                        )
                if verbose and boundaries:
                    print(f"Boundary detection inserted {len(boundaries)} boundary pause upgrades.")
            except Exception as exc:
                if verbose:
                    print(f"  [boundary detection skipped: {exc}]")

        records: List[SentenceRecord] = []
        for i, (sentence, raw_emo, smoothed_emo, pause) in enumerate(
            zip(sentences, raw_emotion_labels, smoothed_labels, raw_pause_labels)
        ):
            lo = max(0, i - self.config.context_window)
            hi = min(n, i + self.config.context_window + 1)
            ctx = " ".join(sentences[lo:hi])
            rec = SentenceRecord(
                segment_id=i,
                text=sentence,
                raw_emotion_idx=raw_emo,
                emotion_idx=smoothed_emo,
                pause_idx=pause,
                paragraph_context=ctx,
                chapter_id=chapter_id,
            )
            records.append(rec)

        if verbose:
            changed = sum(1 for r, s in zip(raw_emotion_labels, smoothed_labels) if r != s)
            print(f"Hysteresis smoothed {changed}/{n} emotion labels.")

        return records

    @staticmethod
    def write_jsonl(records: List[SentenceRecord], output_path: str | Path) -> None:
        """Write classified records to a JSONL file, grouping consecutive
        sentences with the same emotion+pause into a single segment.

        Consecutive SentenceRecords sharing the same smoothed emotion AND pause
        are merged into one line, with their texts joined by a space. This means
        each output segment represents a run of same-emotion/same-pause prose.

        The last sentence in each group carries the group's pause_after value
        (since the pause belongs at the end of the segment).

        Args:
            records: List of SentenceRecord from classify_chapter/classify_text.
            output_path: Destination .jsonl file path (created or overwritten).
        """
        if not records:
            return

        # Group consecutive records with identical emotion AND pause
        groups: List[List[SentenceRecord]] = []
        current: List[SentenceRecord] = [records[0]]
        for rec in records[1:]:
            if rec.emotion == current[0].emotion and rec.pause_idx == current[0].pause_idx:
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
                    "pause_after": group[0].pause,
                }
                if group[0].chapter_id is not None:
                    d["chapter_id"] = group[0].chapter_id
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        print(f"Wrote {len(groups)} segments ({len(records)} sentences) → {out}")
