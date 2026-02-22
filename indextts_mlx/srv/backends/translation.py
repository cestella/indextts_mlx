"""SeamlessM4T text-to-text translation backend (PyTorch/MPS)."""

from __future__ import annotations

from indextts_mlx.srv.backends.base import ModelBackend


class TranslationBackend(ModelBackend):
    model_type = "translation"

    def __init__(self) -> None:
        self._processor = None
        self._model = None

    def load(self, model_params: dict) -> None:
        from transformers import AutoProcessor, SeamlessM4Tv2ForTextToText

        repo = model_params.get("repo", "facebook/seamless-m4t-v2-large")
        self._processor = AutoProcessor.from_pretrained(repo)
        self._model = SeamlessM4Tv2ForTextToText.from_pretrained(repo)

    def unload(self) -> None:
        self._processor = None
        self._model = None

    def execute(self, request: dict) -> dict:
        if self._model is None:
            raise RuntimeError("TranslationBackend not loaded")

        text = request["text"]
        src_lang = request.get("src_lang", "ita")
        tgt_lang = request.get("tgt_lang", "eng")
        max_tokens = request.get("max_tokens", 1024)

        inputs = self._processor(text=text, src_lang=src_lang, return_tensors="pt")
        output_tokens = self._model.generate(
            **inputs, tgt_lang=tgt_lang, max_new_tokens=max_tokens
        )
        translated = self._processor.decode(output_tokens[0].tolist(), skip_special_tokens=True)

        return {"text": translated}
