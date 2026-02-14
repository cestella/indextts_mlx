"""Qwen3-based text emotion classifier.

Wraps the fine-tuned qwen0.6bemo4-merge checkpoint via mlx-lm to convert
a text description (English or Chinese) into an 8-float emotion vector:
  [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]

Matches the PyTorch QwenEmotion class from IndexTTS-2's infer_v2.py.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Optional


# Emotion categories in the required order (matches emo_num split order)
_DESIRED_ORDER = ["高兴", "愤怒", "悲伤", "恐惧", "反感", "低落", "惊讶", "自然"]

_CN_TO_EN = {
    "高兴": "happy",
    "愤怒": "angry",
    "悲伤": "sad",
    "恐惧": "afraid",
    "反感": "disgusted",
    "低落": "melancholic",
    "惊讶": "surprised",
    "自然": "calm",
}

# Words that force "sad" detection to be remapped to "melancholic"
# (the model can't distinguish them; this is the same workaround as PyTorch)
_MELANCHOLIC_WORDS = {
    "低落", "melancholy", "melancholic", "depression", "depressed", "gloomy",
}


class QwenEmotion:
    """Text-to-emotion-vector classifier using the fine-tuned Qwen3-0.6B model.

    Lazy-loads the model on first use (adds ~3-4 s startup, ~500 MB RAM).

    Usage:
        qe = QwenEmotion("/path/to/qwen0.6bemo4-merge")
        vec = qe.inference("What a wonderful day!")
        # [0.95, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02, 0.03]
    """

    def __init__(self, model_dir: str | Path):
        self.model_dir = str(model_dir)
        self._model = None
        self._tokenizer = None

    def _load(self):
        if self._model is not None:
            return
        from mlx_lm import load
        self._model, self._tokenizer = load(self.model_dir)

    @staticmethod
    def _clamp(value: float, lo: float = 0.0, hi: float = 1.2) -> float:
        return max(lo, min(hi, value))

    def _parse_response(self, content: str) -> dict:
        """Parse the model's JSON output into a Chinese-keyed dict."""
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # Fallback: regex scan for key:value pairs
            return {
                m.group(1): float(m.group(2))
                for m in re.finditer(r'([^\s":.,]+?)"?\s*:\s*([\d.]+)', content)
            }

    def _convert(self, raw: dict, text_input: str) -> List[float]:
        """Convert raw Chinese-keyed dict to ordered 8-float list.

        Applies:
        - melancholic/sad swap workaround (model can't distinguish them)
        - clamping to [0.0, 1.2]
        - fallback to calm=1.0 if all zeros
        """
        # Melancholic workaround: if input text contains melancholic keywords,
        # swap the sad and melancholic values so the melancholic slot gets the score
        text_lower = text_input.lower()
        if any(word in text_lower for word in _MELANCHOLIC_WORDS):
            raw["悲伤"], raw["低落"] = raw.get("低落", 0.0), raw.get("悲伤", 0.0)

        vec = [self._clamp(raw.get(cn_key, 0.0)) for cn_key in _DESIRED_ORDER]

        # Default to calm/neutral if all zeros
        if all(v <= 0.0 for v in vec):
            vec[7] = 1.0  # calm is index 7

        return vec

    def inference(self, text_input: str) -> List[float]:
        """Classify emotion in text_input.

        Args:
            text_input: Natural language description of desired emotion, e.g.
                        "joyful and excited" or "sad and melancholic"

        Returns:
            8-float list: [happy, angry, sad, afraid, disgusted, melancholic,
                           surprised, calm], each in [0.0, 1.2].
        """
        self._load()
        from mlx_lm import generate

        messages = [
            {"role": "system", "content": "文本情感分类"},
            {"role": "user", "content": text_input},
        ]
        prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        response = generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=200,
            verbose=False,
        )

        raw = self._parse_response(response.strip())
        return self._convert(raw, text_input)

    def inference_to_dict(self, text_input: str) -> dict:
        """Like inference() but returns {emotion_name: float} dict."""
        vec = self.inference(text_input)
        return {_CN_TO_EN[cn]: v for cn, v in zip(_DESIRED_ORDER, vec)}
