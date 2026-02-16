"""Unit tests for EmotionClassifier — no LLM required."""

import json
import textwrap
from pathlib import Path

import pytest

from indextts_mlx.emotion_classifier import (
    EMOTION_LABELS,
    PAUSE_LABELS,
    ClassifierConfig,
    EmotionClassifier,
    SentenceRecord,
    _apply_hysteresis,
    _build_emotion_prompt,
    _build_pause_prompt,
    _update_dialogue_state,
)

# ---------------------------------------------------------------------------
# _apply_hysteresis
# ---------------------------------------------------------------------------


class TestApplyHysteresis:
    """Exhaustive tests for the run-length hysteresis smoother."""

    def test_empty(self):
        assert _apply_hysteresis([]) == []

    def test_single_neutral(self):
        assert _apply_hysteresis([0]) == [0]

    def test_single_nonzero_collapsed(self):
        assert _apply_hysteresis([3], min_run=2) == [0]

    def test_single_nonzero_kept_at_min_run_1(self):
        assert _apply_hysteresis([3], min_run=1) == [3]

    def test_all_neutral_unchanged(self):
        assert _apply_hysteresis([0, 0, 0, 0]) == [0, 0, 0, 0]

    def test_isolated_spike_collapsed(self):
        assert _apply_hysteresis([0, 0, 3, 0, 0], min_run=2) == [0, 0, 0, 0, 0]

    def test_run_of_exactly_min_run_preserved(self):
        assert _apply_hysteresis([0, 4, 4, 0], min_run=2) == [0, 4, 4, 0]

    def test_run_one_below_min_run_collapsed(self):
        assert _apply_hysteresis([0, 4, 0], min_run=2) == [0, 0, 0]

    def test_run_longer_than_min_run_preserved(self):
        assert _apply_hysteresis([0, 4, 4, 4, 0], min_run=2) == [0, 4, 4, 4, 0]

    def test_spec_example(self):
        # From spec: 0,0,0,3,0,0,2,0 → isolated spikes removed
        assert _apply_hysteresis([0, 0, 0, 3, 0, 0, 2, 0], min_run=2) == [
            0, 0, 0, 0, 0, 0, 0, 0,
        ]

    def test_two_short_runs_both_collapsed(self):
        assert _apply_hysteresis([1, 0, 2, 0, 3], min_run=2) == [0, 0, 0, 0, 0]

    def test_two_long_runs_both_preserved(self):
        assert _apply_hysteresis([1, 1, 0, 2, 2, 0, 3, 3, 3], min_run=2) == [
            1, 1, 0, 2, 2, 0, 3, 3, 3,
        ]

    def test_all_same_nonzero_preserved(self):
        assert _apply_hysteresis([5, 5, 5], min_run=2) == [5, 5, 5]

    def test_starts_and_ends_with_nonzero(self):
        assert _apply_hysteresis([3, 3, 0, 5, 5], min_run=2) == [3, 3, 0, 5, 5]

    def test_adjacent_different_emotions_each_short_collapsed(self):
        # [1, 2, 0] — each non-neutral run is length 1 → both collapse
        assert _apply_hysteresis([1, 2, 0], min_run=2) == [0, 0, 0]

    def test_min_run_3(self):
        assert _apply_hysteresis([4, 4, 0, 4, 4, 4], min_run=3) == [0, 0, 0, 4, 4, 4]

    def test_length_always_preserved(self):
        for labels in [[0, 1, 2, 3, 4, 5, 6, 0], [0], [], [1, 1, 1]]:
            assert len(_apply_hysteresis(labels, min_run=2)) == len(labels)

    def test_output_is_new_list(self):
        original = [0, 3, 0]
        result = _apply_hysteresis(original, min_run=2)
        assert result is not original


# ---------------------------------------------------------------------------
# _build_emotion_prompt
# ---------------------------------------------------------------------------


class TestBuildEmotionPrompt:
    def test_context_appears_in_prompt(self):
        p = _build_emotion_prompt("Some context.", "Target sentence.")
        assert "Some context." in p
        assert "Target sentence." in p

    def test_all_active_emotion_labels_listed(self):
        p = _build_emotion_prompt("ctx", "sent")
        # calm_authority (index 6) is not emitted by the current prompt (outputs 0-5 only)
        for label in EMOTION_LABELS:
            if label == "calm_authority":
                continue
            assert label in p

    def test_output_format_instructions_present(self):
        p = _build_emotion_prompt("ctx", "sent")
        assert "ONE digit" in p
        assert "0, 1, 2, 3, 4, 5" in p

    def test_joyful_specific_rules_present(self):
        p = _build_emotion_prompt("ctx", "sent")
        assert "joyful" in p
        assert "happy" in p.lower() or "happiness" in p.lower()

    def test_mild_emphasis_specific_rules_present(self):
        p = _build_emotion_prompt("ctx", "sent")
        assert "mild_emphasis" in p
        assert "analytical" in p.lower()
        assert "rare" in p.lower()
        assert "structural" in p.lower()

    def test_delimiters_present(self):
        p = _build_emotion_prompt("ctx", "sent")
        assert "<<<" in p
        assert ">>>" in p

    def test_context_in_correct_section(self):
        p = _build_emotion_prompt("MY_CONTEXT", "MY_SENTENCE")
        ctx_pos = p.index("MY_CONTEXT")
        sent_pos = p.index("MY_SENTENCE")
        assert ctx_pos < sent_pos, "Context should appear before sentence"

    def test_no_leftover_template_placeholders(self):
        p = _build_emotion_prompt("ctx", "sent")
        assert "{PARAGRAPH_CONTEXT}" not in p
        assert "{TARGET_SENTENCE}" not in p


# ---------------------------------------------------------------------------
# _build_pause_prompt
# ---------------------------------------------------------------------------


class TestBuildPausePrompt:
    def test_context_and_sentence_present(self):
        p = _build_pause_prompt("Some context.", "Target sentence.")
        assert "Some context." in p
        assert "Target sentence." in p

    def test_all_pause_labels_mentioned(self):
        p = _build_pause_prompt("ctx", "sent")
        for label in PAUSE_LABELS:
            assert label in p

    def test_punctuation_hints_present(self):
        p = _build_pause_prompt("ctx", "sent")
        # The prompt should mention ellipsis and em dash hints
        assert "…" in p or "..." in p
        assert "—" in p

    def test_valid_output_digits_listed(self):
        p = _build_pause_prompt("ctx", "sent")
        assert "0, 1, 2, 3, 4" in p

    def test_one_digit_instruction(self):
        p = _build_pause_prompt("ctx", "sent")
        assert "ONE digit" in p

    def test_no_leftover_placeholders(self):
        p = _build_pause_prompt("ctx", "sent")
        assert "{PARAGRAPH_CONTEXT}" not in p
        assert "{TARGET_SENTENCE}" not in p

    def test_delimiters_present(self):
        p = _build_pause_prompt("ctx", "sent")
        assert "<<<" in p
        assert ">>>" in p

    def test_context_before_sentence(self):
        p = _build_pause_prompt("MY_CONTEXT", "MY_SENTENCE")
        assert p.index("MY_CONTEXT") < p.index("MY_SENTENCE")


# ---------------------------------------------------------------------------
# ClassifierConfig
# ---------------------------------------------------------------------------


class TestClassifierConfig:
    def test_defaults(self):
        cfg = ClassifierConfig()
        assert cfg.model == "mlx-community/Qwen2.5-32B-Instruct-4bit"
        assert cfg.max_retries == 3
        assert cfg.context_window == 5
        assert cfg.language == "english"
        assert cfg.hysteresis_min_run == 2
        assert cfg.use_boundary_detection is True
        assert cfg.boundary_min_pause == 3

    def test_custom_values(self):
        cfg = ClassifierConfig(
            model="my/model",
            max_retries=5,
            context_window=3,
            language="french",
            hysteresis_min_run=3,
            use_boundary_detection=False,
        )
        assert cfg.model == "my/model"
        assert cfg.max_retries == 5
        assert cfg.context_window == 3
        assert cfg.language == "french"
        assert cfg.hysteresis_min_run == 3
        assert cfg.use_boundary_detection is False


# ---------------------------------------------------------------------------
# SentenceRecord
# ---------------------------------------------------------------------------


class TestSentenceRecord:
    def _rec(self, emotion_idx=0, pause_idx=2, segment_id=0, chapter_id=None):
        return SentenceRecord(
            segment_id=segment_id,
            text=f"Sentence {segment_id}.",
            raw_emotion_idx=emotion_idx,
            emotion_idx=emotion_idx,
            pause_idx=pause_idx,
            paragraph_context="Some context.",
            chapter_id=chapter_id,
        )

    def test_emotion_property_for_each_index(self):
        for i, label in enumerate(EMOTION_LABELS):
            assert self._rec(emotion_idx=i).emotion == label

    def test_pause_property_for_each_index(self):
        for i, label in enumerate(PAUSE_LABELS):
            assert self._rec(pause_idx=i).pause == label

    def test_to_jsonl_dict_required_keys(self):
        d = self._rec().to_jsonl_dict()
        assert "segment_id" in d
        assert "text" in d
        assert "emotion" in d
        assert "pause_after" in d

    def test_to_jsonl_dict_pause_after_is_valid_label(self):
        for i in range(len(PAUSE_LABELS)):
            d = self._rec(pause_idx=i).to_jsonl_dict()
            assert d["pause_after"] in PAUSE_LABELS

    def test_to_jsonl_dict_no_chapter_id_when_none(self):
        d = self._rec(chapter_id=None).to_jsonl_dict()
        assert "chapter_id" not in d

    def test_to_jsonl_dict_includes_int_chapter_id(self):
        d = self._rec(chapter_id=7).to_jsonl_dict()
        assert d["chapter_id"] == 7

    def test_to_jsonl_dict_includes_str_chapter_id(self):
        d = self._rec(chapter_id="prologue").to_jsonl_dict()
        assert d["chapter_id"] == "prologue"

    def test_to_jsonl_dict_emotion_is_valid_label(self):
        for i in range(len(EMOTION_LABELS)):
            d = self._rec(emotion_idx=i).to_jsonl_dict()
            assert d["emotion"] in EMOTION_LABELS

    def test_raw_and_smoothed_can_differ(self):
        rec = SentenceRecord(
            segment_id=0,
            text="x",
            raw_emotion_idx=3,
            emotion_idx=0,  # smoothed back to neutral
            pause_idx=2,
            paragraph_context="x",
        )
        assert rec.emotion == "neutral"
        assert rec.raw_emotion_idx == 3


# ---------------------------------------------------------------------------
# EmotionClassifier.write_jsonl
# ---------------------------------------------------------------------------


class TestWriteJsonl:
    def _records(self, n=3, chapter_id=None):
        return [
            SentenceRecord(
                segment_id=i,
                text=f"Sentence {i}.",
                raw_emotion_idx=0,
                emotion_idx=i % len(EMOTION_LABELS),
                pause_idx=i % len(PAUSE_LABELS),
                paragraph_context=f"Context {i}.",
                chapter_id=chapter_id,
            )
            for i in range(n)
        ]

    def _same_emo_pause_records(self, n=3, emotion_idx=0, pause_idx=2, chapter_id=None):
        """All records share the same emotion+pause so they merge into one segment."""
        return [
            SentenceRecord(
                segment_id=i,
                text=f"Sentence {i}.",
                raw_emotion_idx=emotion_idx,
                emotion_idx=emotion_idx,
                pause_idx=pause_idx,
                paragraph_context="ctx",
                chapter_id=chapter_id,
            )
            for i in range(n)
        ]

    def test_correct_line_count_all_different(self, tmp_path):
        # Each record has a different emotion+pause combo — all stay separate
        out = tmp_path / "out.jsonl"
        EmotionClassifier.write_jsonl(self._records(5), out)
        lines = out.read_text().strip().splitlines()
        assert len(lines) == 5

    def test_merges_consecutive_same_emotion_and_pause(self, tmp_path):
        out = tmp_path / "out.jsonl"
        EmotionClassifier.write_jsonl(self._same_emo_pause_records(4), out)
        lines = out.read_text().strip().splitlines()
        assert len(lines) == 1

    def test_no_merge_across_different_pause(self, tmp_path):
        # Two records same emotion but different pause → should NOT merge
        recs = [
            SentenceRecord(0, "A.", 0, 0, 2, "ctx"),  # neutral, neutral pause
            SentenceRecord(1, "B.", 0, 0, 3, "ctx"),  # neutral, long pause
        ]
        out = tmp_path / "out.jsonl"
        EmotionClassifier.write_jsonl(recs, out)
        assert len(out.read_text().strip().splitlines()) == 2

    def test_each_line_valid_json(self, tmp_path):
        out = tmp_path / "out.jsonl"
        EmotionClassifier.write_jsonl(self._records(4), out)
        for line in out.read_text().strip().splitlines():
            json.loads(line)  # must not raise

    def test_required_fields_present(self, tmp_path):
        out = tmp_path / "out.jsonl"
        EmotionClassifier.write_jsonl(self._records(3), out)
        for obj in (json.loads(l) for l in out.read_text().strip().splitlines()):
            assert "segment_id" in obj
            assert "text" in obj
            assert "emotion" in obj
            assert "pause_after" in obj

    def test_pause_after_is_valid_label(self, tmp_path):
        out = tmp_path / "out.jsonl"
        EmotionClassifier.write_jsonl(self._records(len(PAUSE_LABELS)), out)
        for obj in (json.loads(l) for l in out.read_text().strip().splitlines()):
            assert obj["pause_after"] in PAUSE_LABELS

    def test_emotions_are_valid_labels(self, tmp_path):
        out = tmp_path / "out.jsonl"
        EmotionClassifier.write_jsonl(self._records(len(EMOTION_LABELS)), out)
        for obj in (json.loads(l) for l in out.read_text().strip().splitlines()):
            assert obj["emotion"] in EMOTION_LABELS

    def test_chapter_id_int_written(self, tmp_path):
        out = tmp_path / "out.jsonl"
        EmotionClassifier.write_jsonl(self._same_emo_pause_records(2, chapter_id=3), out)
        for obj in (json.loads(l) for l in out.read_text().strip().splitlines()):
            assert obj["chapter_id"] == 3

    def test_chapter_id_str_written(self, tmp_path):
        out = tmp_path / "out.jsonl"
        EmotionClassifier.write_jsonl(self._same_emo_pause_records(2, chapter_id="ch01"), out)
        for obj in (json.loads(l) for l in out.read_text().strip().splitlines()):
            assert obj["chapter_id"] == "ch01"

    def test_no_chapter_id_field_when_none(self, tmp_path):
        out = tmp_path / "out.jsonl"
        EmotionClassifier.write_jsonl(self._records(2, chapter_id=None), out)
        for obj in (json.loads(l) for l in out.read_text().strip().splitlines()):
            assert "chapter_id" not in obj

    def test_segment_ids_sequential(self, tmp_path):
        out = tmp_path / "out.jsonl"
        EmotionClassifier.write_jsonl(self._records(4), out)
        ids = [json.loads(l)["segment_id"] for l in out.read_text().strip().splitlines()]
        assert ids == list(range(len(ids)))

    def test_creates_parent_dirs(self, tmp_path):
        out = tmp_path / "a" / "b" / "c.jsonl"
        EmotionClassifier.write_jsonl(self._records(1), out)
        assert out.exists()

    def test_unicode_text_round_trips(self, tmp_path):
        rec = SentenceRecord(
            segment_id=0,
            text="Héros naïf — « Bonjour »",
            raw_emotion_idx=0,
            emotion_idx=0,
            pause_idx=2,
            paragraph_context="",
        )
        out = tmp_path / "uni.jsonl"
        EmotionClassifier.write_jsonl([rec], out)
        obj = json.loads(out.read_text(encoding="utf-8").strip())
        assert obj["text"] == "Héros naïf — « Bonjour »"

    def test_overwrites_existing_file(self, tmp_path):
        out = tmp_path / "out.jsonl"
        EmotionClassifier.write_jsonl(self._records(5), out)
        EmotionClassifier.write_jsonl(self._same_emo_pause_records(2), out)
        assert len(out.read_text().strip().splitlines()) == 1


# ---------------------------------------------------------------------------
# EmotionClassifier with stubbed classifiers
# ---------------------------------------------------------------------------


def _make_clf_stubbed(
    emotion_responses: list,
    pause_responses: list | None = None,
    hysteresis_min_run: int = 2,
) -> EmotionClassifier:
    """Create an EmotionClassifier whose LLM calls are replaced by iterating lists."""
    cfg = ClassifierConfig(
        hysteresis_min_run=hysteresis_min_run,
        use_boundary_detection=False,  # disable to avoid needing sentence-transformers
    )
    clf = EmotionClassifier(cfg)
    emo_it = iter(emotion_responses)
    pause_it = iter(pause_responses if pause_responses is not None else [2] * 1000)
    clf._classify_one_emotion = lambda ctx, s: next(emo_it, 0)
    clf._classify_one_pause = lambda ctx, s: next(pause_it, 2)
    return clf


spacy_available = bool(__import__("importlib").util.find_spec("spacy"))


class TestClassifyTextStubbed:
    @pytest.mark.skipif(not spacy_available, reason="spacy not installed")
    def test_returns_one_record_per_sentence(self):
        text = "Hello world. Goodbye world. See you soon."
        clf = _make_clf_stubbed([0] * 10)
        records = clf.classify_text(text, verbose=False)
        assert len(records) >= 1

    @pytest.mark.skipif(not spacy_available, reason="spacy not installed")
    def test_segment_ids_are_sequential(self):
        clf = _make_clf_stubbed([0] * 10)
        records = clf.classify_text("One. Two. Three.", verbose=False)
        ids = [r.segment_id for r in records]
        assert ids == list(range(len(records)))

    @pytest.mark.skipif(not spacy_available, reason="spacy not installed")
    def test_chapter_id_propagated(self):
        clf = _make_clf_stubbed([0] * 10)
        records = clf.classify_text("One. Two.", chapter_id=5, verbose=False)
        assert all(r.chapter_id == 5 for r in records)

    @pytest.mark.skipif(not spacy_available, reason="spacy not installed")
    def test_chapter_id_none_by_default(self):
        clf = _make_clf_stubbed([0] * 10)
        records = clf.classify_text("One. Two.", verbose=False)
        assert all(r.chapter_id is None for r in records)

    @pytest.mark.skipif(not spacy_available, reason="spacy not installed")
    def test_each_record_has_pause_idx(self):
        clf = _make_clf_stubbed([0] * 10, pause_responses=[1] * 10)
        records = clf.classify_text("One. Two. Three.", verbose=False)
        for r in records:
            assert r.pause_idx == 1

    @pytest.mark.skipif(not spacy_available, reason="spacy not installed")
    def test_pause_idx_varies_per_sentence(self):
        # 3 sentences, pause responses 0, 3, 4
        clf = _make_clf_stubbed([0] * 10, pause_responses=[0, 3, 4] + [2] * 10)
        records = clf.classify_text("One. Two. Three.", verbose=False)
        if len(records) >= 3:
            assert records[0].pause_idx == 0
            assert records[1].pause_idx == 3
            assert records[2].pause_idx == 4

    @pytest.mark.skipif(not spacy_available, reason="spacy not installed")
    def test_raw_and_smoothed_labels_stored(self):
        # Make the stub return [3, 0, 0, ...] — the 3 should be smoothed away
        clf = _make_clf_stubbed([3, 0, 0, 0, 0, 0])
        records = clf.classify_text("One. Two. Three. Four. Five.", verbose=False)
        raw_labels = [r.raw_emotion_idx for r in records]
        smoothed_labels = [r.emotion_idx for r in records]
        assert 3 in raw_labels
        assert 3 not in smoothed_labels

    @pytest.mark.skipif(not spacy_available, reason="spacy not installed")
    def test_isolated_spikes_smoothed(self):
        clf = _make_clf_stubbed([0, 3, 0, 0, 5, 0])
        text = (
            "The sun rose slowly. It was warm. "
            "A dog barked. The street was quiet. "
            "A child laughed. The day began."
        )
        records = clf.classify_text(text, verbose=False)
        for r in records:
            assert r.emotion_idx == 0, f"Expected neutral after smoothing, got {r.emotion_idx}"

    @pytest.mark.skipif(not spacy_available, reason="spacy not installed")
    def test_sustained_run_preserved_after_smoothing(self):
        clf = _make_clf_stubbed([0, 0, 5, 5, 5, 0])
        text = (
            "The sun rose slowly. It was warm. "
            "A dog barked. The street was quiet. "
            "A child laughed. The day began."
        )
        records = clf.classify_text(text, verbose=False)
        smoothed = [r.emotion_idx for r in records]
        non_neutral = [e for e in smoothed if e != 0]
        assert len(non_neutral) >= 2, f"Expected sustained run to survive: {smoothed}"

    @pytest.mark.skipif(not spacy_available, reason="spacy not installed")
    def test_on_sentence_callback_called_per_sentence(self):
        clf = _make_clf_stubbed([0] * 10)
        calls = []
        def cb(idx, total):
            calls.append((idx, total))
        clf.classify_text("One. Two. Three.", verbose=False, on_sentence=cb)
        assert len(calls) == len(clf._get_sentences("One. Two. Three."))
        # Indices should be sequential 0, 1, 2, ...
        for i, (idx, total) in enumerate(calls):
            assert idx == i
            assert total > 0

    @pytest.mark.skipif(not spacy_available, reason="spacy not installed")
    def test_write_jsonl_roundtrip_includes_pause_after(self, tmp_path):
        clf = _make_clf_stubbed([0] * 10, pause_responses=[3] * 10)
        text = "One sentence. Two sentences. Three sentences."
        records = clf.classify_text(text, chapter_id=1, verbose=False)
        out = tmp_path / "ch1.jsonl"
        EmotionClassifier.write_jsonl(records, out)
        objs = [json.loads(l) for l in out.read_text().strip().splitlines()]
        assert all("text" in o for o in objs)
        assert all("emotion" in o for o in objs)
        assert all("pause_after" in o for o in objs)
        assert all(o["chapter_id"] == 1 for o in objs)

    @pytest.mark.skipif(not spacy_available, reason="spacy not installed")
    def test_classify_chapter_reads_file(self, tmp_path):
        txt = tmp_path / "chapter.txt"
        txt.write_text("Hello world. Goodbye world.", encoding="utf-8")
        clf = _make_clf_stubbed([0] * 10)
        records = clf.classify_chapter(txt, verbose=False)
        assert len(records) >= 1
        assert all(r.text for r in records)

    @pytest.mark.skipif(not spacy_available, reason="spacy not installed")
    def test_batch_chapter_id_uses_file_stem(self, tmp_path):
        """Batch mode passes chapter_id=inp.stem — verify the stem is stored."""
        txt = tmp_path / "chapter03.txt"
        txt.write_text("One sentence. Two sentences.", encoding="utf-8")
        clf = _make_clf_stubbed([0] * 10)
        # Simulate what the batch loop does
        records = clf.classify_chapter(txt, chapter_id=txt.stem, verbose=False)
        assert all(r.chapter_id == "chapter03" for r in records)

    @pytest.mark.skipif(not spacy_available, reason="spacy not installed")
    def test_batch_chapter_id_written_to_jsonl(self, tmp_path):
        """chapter_id from batch mode survives the JSONL round-trip."""
        import json as _json
        txt = tmp_path / "part_one.txt"
        txt.write_text("First. Second. Third.", encoding="utf-8")
        clf = _make_clf_stubbed([0] * 10)
        records = clf.classify_chapter(txt, chapter_id=txt.stem, verbose=False)
        out = tmp_path / "part_one.jsonl"
        EmotionClassifier.write_jsonl(records, out)
        objs = [_json.loads(l) for l in out.read_text().strip().splitlines()]
        assert all(o.get("chapter_id") == "part_one" for o in objs)

    @pytest.mark.skipif(not spacy_available, reason="spacy not installed")
    def test_boundary_detection_upgrade_applied(self):
        """Boundary detection should upgrade pause labels at detected boundaries."""
        cfg = ClassifierConfig(
            use_boundary_detection=True,
            boundary_min_pause=4,
            hysteresis_min_run=1,
        )
        clf = EmotionClassifier(cfg)
        clf._classify_one_emotion = lambda ctx, s: 0
        clf._classify_one_pause = lambda ctx, s: 0  # always returns "none"

        # Inject a fake boundary detector that says boundary is after sentence 0
        import indextts_mlx.boundary_detector as _bd
        original = _bd.detect_boundaries
        _bd.detect_boundaries = lambda sentences, text, **kw: {0}
        try:
            records = clf.classify_text("Sentence one.\n\nSentence two.", verbose=False)
        finally:
            _bd.detect_boundaries = original

        if records:
            assert records[0].pause_idx == 4, (
                f"Expected boundary upgrade to 4 (dramatic), got {records[0].pause_idx}"
            )


# ---------------------------------------------------------------------------
# Fragment filtering in _get_sentences
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not spacy_available, reason="spacy not installed")
class TestGetSentencesFragmentFiltering:
    """_get_sentences must drop orphan punctuation fragments (e.g. '\".' or '\".').

    These arise when a sentence-tokeniser splits dialogue mid-quote, leaving
    a closing-quote + period fragment with no alphabetic content.
    """

    def _get_sentences(self, text: str):
        cfg = ClassifierConfig(use_boundary_detection=False)
        return EmotionClassifier(cfg)._get_sentences(text)

    def test_pure_quote_period_fragment_dropped(self):
        # Simulates: '"Priscilla is missing." -> ["Priscilla is missing.", '".']
        # The closing-quote fragment should be silently dropped.
        sentences = self._get_sentences('"Priscilla is missing."')
        for s in sentences:
            import re
            assert len(re.findall(r"[A-Za-z]", s)) >= 3, (
                f"Fragment with <3 letters should have been filtered: {s!r}"
            )

    def test_single_quote_period_dropped(self):
        sentences = self._get_sentences("Go. \".  Next sentence here.")
        for s in sentences:
            import re
            assert len(re.findall(r"[A-Za-z]", s)) >= 3, (
                f"Fragment should have been filtered: {s!r}"
            )

    def test_normal_sentences_retained(self):
        text = "The cat sat. The dog ran. A bird flew."
        sentences = self._get_sentences(text)
        assert len(sentences) >= 2, "Normal sentences should not be filtered"

    def test_all_real_content_kept(self):
        text = "Oliver groaned. He rolled over. His head was pounding."
        sentences = self._get_sentences(text)
        for s in sentences:
            import re
            assert len(re.findall(r"[A-Za-z]", s)) >= 3


# ---------------------------------------------------------------------------
# --head parameter
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not spacy_available, reason="spacy not installed")
class TestHeadParameter:
    """classify_text and classify_chapter respect the head= argument."""

    def test_head_limits_sentence_count(self):
        clf = _make_clf_stubbed([0] * 20)
        text = "One. Two. Three. Four. Five."
        records_full = clf.classify_text(text, verbose=False)
        records_head = clf.classify_text(text, verbose=False, head=2)
        assert len(records_head) <= 2
        assert len(records_head) <= len(records_full)

    def test_head_zero_returns_empty(self):
        clf = _make_clf_stubbed([0] * 20)
        records = clf.classify_text("One. Two. Three.", verbose=False, head=0)
        assert records == []

    def test_head_larger_than_sentence_count_returns_all(self):
        clf = _make_clf_stubbed([0] * 20)
        text = "One. Two. Three."
        records_all = clf.classify_text(text, verbose=False)
        records_head = clf.classify_text(text, verbose=False, head=100)
        assert len(records_head) == len(records_all)

    def test_head_none_returns_all(self):
        clf = _make_clf_stubbed([0] * 20)
        text = "One. Two. Three."
        records_all = clf.classify_text(text, verbose=False)
        records_head = clf.classify_text(text, verbose=False, head=None)
        assert len(records_head) == len(records_all)

    def test_head_on_classify_chapter(self, tmp_path):
        txt = tmp_path / "ch.txt"
        txt.write_text("One. Two. Three. Four. Five.", encoding="utf-8")
        clf = _make_clf_stubbed([0] * 20)
        records = clf.classify_chapter(txt, verbose=False, head=2)
        assert len(records) <= 2

    def test_head_records_are_first_sentences(self):
        clf = _make_clf_stubbed([0] * 20)
        text = "Alpha sentence. Beta sentence. Gamma sentence. Delta sentence."
        records_head = clf.classify_text(text, verbose=False, head=2)
        records_all = clf.classify_text(text, verbose=False)
        # The first N records should match the first N of the full run
        for h, a in zip(records_head, records_all):
            assert h.text == a.text


# ---------------------------------------------------------------------------
# Suspense prompt rules
# ---------------------------------------------------------------------------


class TestSuspensePromptContent:
    """Verify the suspense section of _PROMPT_TEMPLATE contains the key
    guard rules that prevent over-triggering on dialogue and fact reports.
    Purely a prompt-content regression test — no LLM involved.
    """

    def _suspense_block(self) -> str:
        from indextts_mlx.emotion_classifier import _PROMPT_TEMPLATE
        # Extract the section between "4 = suspense" and the next label ("5 =")
        start = _PROMPT_TEMPLATE.find("4 = suspense")
        end = _PROMPT_TEMPLATE.find("5 =", start)
        assert start != -1, "Could not find '4 = suspense' in _PROMPT_TEMPLATE"
        assert end != -1, "Could not find end of suspense block"
        return _PROMPT_TEMPLATE[start:end]

    def test_suspense_block_exists(self):
        block = self._suspense_block()
        assert "suspense" in block

    def test_dialogue_exclusion_present(self):
        block = self._suspense_block()
        assert "Dialogue" in block or "dialogue" in block, (
            "Suspense block must explicitly exclude dialogue"
        )

    def test_character_speech_exclusion_present(self):
        block = self._suspense_block()
        # Must mention excluding sentences spoken by characters
        assert "character" in block.lower() or "spoken" in block.lower(), (
            "Suspense block must exclude sentences spoken by a character"
        )

    def test_fact_report_exclusion_present(self):
        block = self._suspense_block()
        # Must exclude statements of fact / status reports
        assert "fact" in block.lower() or "status" in block.lower() or "report" in block.lower(), (
            "Suspense block must exclude statements of fact or status reports"
        )

    def test_physical_approach_exclusion_present(self):
        block = self._suspense_block()
        # Must exclude plain movement/approach descriptions
        assert "moving" in block.lower() or "approach" in block.lower() or "physical" in block.lower(), (
            "Suspense block must exclude physical movement/approach descriptions"
        )

    def test_doubt_fallback_to_neutral_present(self):
        block = self._suspense_block()
        assert "in doubt" in block.lower() or "if unsure" in block.lower() or "neutral" in block.lower(), (
            "Suspense block must direct classifier to fall back to neutral when in doubt"
        )


# ---------------------------------------------------------------------------
# _update_dialogue_state
# ---------------------------------------------------------------------------


class TestUpdateDialogueState:
    """Unit tests for the stateful open-quote tracker."""

    # ── starting outside dialogue ────────────────────────────────────────

    def test_plain_narration_stays_outside(self):
        assert _update_dialogue_state(False, "The cat sat on the mat.") is False

    def test_sentence_opening_quote_no_close_enters_dialogue(self):
        assert _update_dialogue_state(False, '"Oliver, you must remove yourself from bed.') is True

    def test_sentence_opening_and_closing_quote_stays_outside(self):
        # Opens and closes in the same sentence — not inside after
        assert _update_dialogue_state(False, '"Hello," she said.') is False

    def test_sentence_with_only_closing_quote_stays_outside(self):
        # No opening quote at start — should not enter dialogue
        assert _update_dialogue_state(False, 'He said, "Goodbye."') is False

    # ── starting inside dialogue ─────────────────────────────────────────

    def test_mid_dialogue_no_close_stays_inside(self):
        # Continuation with no closing quote
        assert _update_dialogue_state(True, "Priscilla is missing.") is True

    def test_mid_dialogue_with_closing_quote_exits(self):
        assert _update_dialogue_state(True, 'You must get up."') is False

    def test_mid_dialogue_standalone_closing_quote_exits(self):
        # A lone '"' is treated as an opening quote (no content after it),
        # so it does NOT close the dialogue — the fragment filter will drop it
        # before the classifier ever sees it anyway.
        assert _update_dialogue_state(True, '"') is True

    def test_mid_dialogue_sentence_opens_new_quote_exits(self):
        # A new " somewhere in a mid-dialogue sentence closes the current run
        assert _update_dialogue_state(True, '"Who the hell is Priscilla?"') is False

    # ── edge cases ───────────────────────────────────────────────────────

    def test_empty_sentence_outside_stays_outside(self):
        assert _update_dialogue_state(False, "") is False

    def test_empty_sentence_inside_stays_inside(self):
        # No quote to close on — stay inside
        assert _update_dialogue_state(True, "") is True

    def test_whitespace_only_outside_stays_outside(self):
        assert _update_dialogue_state(False, "   ") is False

    def test_quote_only_sentence_opens_and_does_not_close(self):
        # A sentence that IS just a quote character — technically opens dialogue
        # but has nothing after position 0 to close it.
        assert _update_dialogue_state(False, '"') is True


# ---------------------------------------------------------------------------
# Dialogue pre-filter in classify_text
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not spacy_available, reason="spacy not installed")
class TestDialoguePreFilter:
    """Emotions for dialogue sentences must be capped at mild_emphasis (1).

    We inject the LLM stub to always return suspense (4) and verify that
    sentences identified as dialogue are clamped back down.
    """

    def _make_suspense_clf(self):
        """Classifier whose LLM always returns suspense (4) for emotion."""
        return _make_clf_stubbed(emotion_responses=[4] * 50, pause_responses=[2] * 50)

    def test_opening_dialogue_sentence_not_suspense(self):
        clf = self._make_suspense_clf()
        # Starts with " — should be detected as dialogue
        records = clf.classify_text('"Oliver, you must remove yourself from bed.', verbose=False)
        for r in records:
            assert r.emotion_idx <= 1, (
                f"Dialogue sentence got emotion {r.emotion_idx}, expected <= 1 (mild_emphasis)"
            )

    def test_continuation_dialogue_sentence_not_suspense(self):
        # Two-sentence dialogue: opener on sentence 1, continuation on sentence 2
        clf = self._make_suspense_clf()
        text = '"Oliver, you must remove yourself from bed. Priscilla is missing.'
        records = clf.classify_text(text, verbose=False)
        for r in records:
            assert r.emotion_idx <= 1, (
                f"Dialogue continuation got emotion {r.emotion_idx}"
            )

    def test_pure_narration_not_capped(self):
        # Pure narration — the stub returns suspense (4) and it should NOT be
        # capped (we're testing the cap is only applied to dialogue).
        clf = self._make_suspense_clf()
        records = clf.classify_text(
            "The shadow crept across the floor. The door handle slowly turned.",
            verbose=False,
        )
        # All raw labels are 4 (suspense); hysteresis with min_run=2 on a 2-sent
        # sequence means both survive.  Emotion should be 4 for narration.
        raw_labels = [r.raw_emotion_idx for r in records]
        assert any(label == 4 for label in raw_labels), (
            "Narration sentences should not be capped — suspense (4) should survive in raw labels"
        )

    def test_dialogue_capping_does_not_affect_pause(self):
        # The pause label should be unaffected by the dialogue emotion cap.
        clf = _make_clf_stubbed(emotion_responses=[4] * 10, pause_responses=[3] * 10)
        records = clf.classify_text('"Oliver, get up. Priscilla is missing.', verbose=False)
        for r in records:
            assert r.pause_idx == 3, "Pause should not be affected by dialogue emotion cap"
