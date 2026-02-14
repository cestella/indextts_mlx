"""Unit tests for EmotionClassifier — no LLM required."""

import json
import textwrap
from pathlib import Path

import pytest

from indextts_mlx.emotion_classifier import (
    EMOTION_LABELS,
    ClassifierConfig,
    EmotionClassifier,
    SentenceRecord,
    _apply_hysteresis,
    _build_prompt,
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
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ]

    def test_two_short_runs_both_collapsed(self):
        assert _apply_hysteresis([1, 0, 2, 0, 3], min_run=2) == [0, 0, 0, 0, 0]

    def test_two_long_runs_both_preserved(self):
        assert _apply_hysteresis([1, 1, 0, 2, 2, 0, 3, 3, 3], min_run=2) == [
            1,
            1,
            0,
            2,
            2,
            0,
            3,
            3,
            3,
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
# _build_prompt
# ---------------------------------------------------------------------------


class TestBuildPrompt:
    def test_context_appears_in_prompt(self):
        p = _build_prompt("Some context.", "Target sentence.")
        assert "Some context." in p
        assert "Target sentence." in p

    def test_all_emotion_labels_listed(self):
        p = _build_prompt("ctx", "sent")
        for label in EMOTION_LABELS:
            assert label in p

    def test_output_format_instructions_present(self):
        p = _build_prompt("ctx", "sent")
        assert "ONE digit" in p
        assert "0, 1, 2, 3, 4, 5, 6" in p

    def test_joyful_specific_rules_present(self):
        p = _build_prompt("ctx", "sent")
        assert "JOYFUL" in p
        assert "small practical wins" in p.lower() or "small practical" in p.lower()
        assert "mundane" in p.lower()

    def test_mild_emphasis_specific_rules_present(self):
        p = _build_prompt("ctx", "sent")
        assert "MILD_EMPHASIS" in p
        assert "discursive" in p.lower() or "analytical" in p.lower()
        assert "wit" in p.lower() or "irony" in p.lower()
        assert "rare" in p.lower()
        assert "structural" in p.lower()

    def test_delimiters_present(self):
        p = _build_prompt("ctx", "sent")
        assert "<<<" in p
        assert ">>>" in p

    def test_context_in_correct_section(self):
        p = _build_prompt("MY_CONTEXT", "MY_SENTENCE")
        ctx_pos = p.index("MY_CONTEXT")
        sent_pos = p.index("MY_SENTENCE")
        assert ctx_pos < sent_pos, "Context should appear before sentence"

    def test_no_leftover_template_placeholders(self):
        p = _build_prompt("ctx", "sent")
        assert "{PARAGRAPH_CONTEXT}" not in p
        assert "{TARGET_SENTENCE}" not in p


# ---------------------------------------------------------------------------
# ClassifierConfig
# ---------------------------------------------------------------------------


class TestClassifierConfig:
    def test_defaults(self):
        cfg = ClassifierConfig()
        assert cfg.model == "mlx-community/Hermes-3-Llama-3.1-8B-MLX"
        assert cfg.max_retries == 3
        assert cfg.context_window == 5
        assert cfg.language == "english"
        assert cfg.hysteresis_min_run == 2

    def test_custom_values(self):
        cfg = ClassifierConfig(
            model="my/model",
            max_retries=5,
            context_window=3,
            language="french",
            hysteresis_min_run=3,
        )
        assert cfg.model == "my/model"
        assert cfg.max_retries == 5
        assert cfg.context_window == 3
        assert cfg.language == "french"
        assert cfg.hysteresis_min_run == 3


# ---------------------------------------------------------------------------
# SentenceRecord
# ---------------------------------------------------------------------------


class TestSentenceRecord:
    def _rec(self, emotion_idx=0, segment_id=0, chapter_id=None):
        return SentenceRecord(
            segment_id=segment_id,
            text=f"Sentence {segment_id}.",
            raw_emotion_idx=emotion_idx,
            emotion_idx=emotion_idx,
            paragraph_context="Some context.",
            chapter_id=chapter_id,
        )

    def test_emotion_property_for_each_index(self):
        for i, label in enumerate(EMOTION_LABELS):
            assert self._rec(emotion_idx=i).emotion == label

    def test_to_jsonl_dict_required_keys(self):
        d = self._rec().to_jsonl_dict()
        assert "segment_id" in d
        assert "text" in d
        assert "emotion" in d

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
                paragraph_context=f"Context {i}.",
                chapter_id=chapter_id,
            )
            for i in range(n)
        ]

    def test_correct_line_count(self, tmp_path):
        out = tmp_path / "out.jsonl"
        EmotionClassifier.write_jsonl(self._records(5), out)
        assert len(out.read_text().strip().splitlines()) == 5

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

    def test_emotions_are_valid_labels(self, tmp_path):
        out = tmp_path / "out.jsonl"
        EmotionClassifier.write_jsonl(self._records(len(EMOTION_LABELS)), out)
        for obj in (json.loads(l) for l in out.read_text().strip().splitlines()):
            assert obj["emotion"] in EMOTION_LABELS

    def test_chapter_id_int_written(self, tmp_path):
        out = tmp_path / "out.jsonl"
        EmotionClassifier.write_jsonl(self._records(2, chapter_id=3), out)
        for obj in (json.loads(l) for l in out.read_text().strip().splitlines()):
            assert obj["chapter_id"] == 3

    def test_chapter_id_str_written(self, tmp_path):
        out = tmp_path / "out.jsonl"
        EmotionClassifier.write_jsonl(self._records(2, chapter_id="ch01"), out)
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
        assert ids == [0, 1, 2, 3]

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
            paragraph_context="",
        )
        out = tmp_path / "uni.jsonl"
        EmotionClassifier.write_jsonl([rec], out)
        obj = json.loads(out.read_text(encoding="utf-8").strip())
        assert obj["text"] == "Héros naïf — « Bonjour »"

    def test_overwrites_existing_file(self, tmp_path):
        out = tmp_path / "out.jsonl"
        EmotionClassifier.write_jsonl(self._records(5), out)
        EmotionClassifier.write_jsonl(self._records(2), out)
        assert len(out.read_text().strip().splitlines()) == 2


# ---------------------------------------------------------------------------
# EmotionClassifier with stubbed _classify_one
# ---------------------------------------------------------------------------


def _make_clf_with_responses(responses: list, hysteresis_min_run: int = 2) -> EmotionClassifier:
    clf = EmotionClassifier(ClassifierConfig(hysteresis_min_run=hysteresis_min_run))
    it = iter(responses)
    clf._classify_one = lambda ctx, s: next(it, 0)
    return clf


spacy_available = bool(__import__("importlib").util.find_spec("spacy"))


class TestClassifyTextStubbed:
    @pytest.mark.skipif(not spacy_available, reason="spacy not installed")
    def test_returns_one_record_per_sentence(self):
        text = "Hello world. Goodbye world. See you soon."
        clf = _make_clf_with_responses([0] * 10)
        records = clf.classify_text(text, verbose=False)
        assert len(records) >= 1

    @pytest.mark.skipif(not spacy_available, reason="spacy not installed")
    def test_segment_ids_are_sequential(self):
        clf = _make_clf_with_responses([0] * 10)
        records = clf.classify_text("One. Two. Three.", verbose=False)
        ids = [r.segment_id for r in records]
        assert ids == list(range(len(records)))

    @pytest.mark.skipif(not spacy_available, reason="spacy not installed")
    def test_chapter_id_propagated(self):
        clf = _make_clf_with_responses([0] * 10)
        records = clf.classify_text("One. Two.", chapter_id=5, verbose=False)
        assert all(r.chapter_id == 5 for r in records)

    @pytest.mark.skipif(not spacy_available, reason="spacy not installed")
    def test_chapter_id_none_by_default(self):
        clf = _make_clf_with_responses([0] * 10)
        records = clf.classify_text("One. Two.", verbose=False)
        assert all(r.chapter_id is None for r in records)

    @pytest.mark.skipif(not spacy_available, reason="spacy not installed")
    def test_raw_and_smoothed_labels_stored(self):
        # Make the stub return [3, 0, 0, ...] — the 3 should be smoothed away
        clf = _make_clf_with_responses([3, 0, 0, 0, 0, 0])
        records = clf.classify_text("One. Two. Three. Four. Five.", verbose=False)
        # At least the first record had a raw spike
        raw_labels = [r.raw_emotion_idx for r in records]
        smoothed_labels = [r.emotion_idx for r in records]
        assert 3 in raw_labels
        assert 3 not in smoothed_labels

    @pytest.mark.skipif(not spacy_available, reason="spacy not installed")
    def test_isolated_spikes_smoothed(self):
        # 6 sentences: stub returns isolated spikes at positions 1 and 4
        clf = _make_clf_with_responses([0, 3, 0, 0, 5, 0])
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
        # 6 sentences: positions 2-4 are melancholic (run=3, preserved)
        clf = _make_clf_with_responses([0, 0, 5, 5, 5, 0])
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
    def test_write_jsonl_roundtrip(self, tmp_path):
        clf = _make_clf_with_responses([0] * 10)
        text = "One sentence. Two sentences. Three sentences."
        records = clf.classify_text(text, chapter_id=1, verbose=False)
        out = tmp_path / "ch1.jsonl"
        EmotionClassifier.write_jsonl(records, out)
        objs = [json.loads(l) for l in out.read_text().strip().splitlines()]
        assert all("text" in o for o in objs)
        assert all("emotion" in o for o in objs)
        assert all(o["chapter_id"] == 1 for o in objs)

    @pytest.mark.skipif(not spacy_available, reason="spacy not installed")
    def test_classify_chapter_reads_file(self, tmp_path):
        txt = tmp_path / "chapter.txt"
        txt.write_text("Hello world. Goodbye world.", encoding="utf-8")
        clf = _make_clf_with_responses([0] * 10)
        records = clf.classify_chapter(txt, verbose=False)
        assert len(records) >= 1
        assert all(r.text for r in records)
