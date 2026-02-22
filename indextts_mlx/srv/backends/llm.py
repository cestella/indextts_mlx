"""MLX-LM text generation backend."""

from __future__ import annotations

from indextts_mlx.srv.backends.base import ModelBackend


class LLMBackend(ModelBackend):
    model_type = "llm"

    def __init__(self) -> None:
        self._model = None
        self._tokenizer = None

    def load(self, model_params: dict) -> None:
        import mlx_lm

        repo = model_params.get("repo", "mlx-community/Qwen2.5-7B-Instruct-4bit")
        self._model, self._tokenizer = mlx_lm.load(repo)

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None

    def execute(self, request: dict) -> dict:
        if self._model is None:
            raise RuntimeError("LLMBackend not loaded")

        import mlx_lm

        prompt = request["prompt"]
        system_prompt = request.get("system_prompt")
        max_tokens = request.get("max_tokens", 1024)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        formatted = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        text = mlx_lm.generate(
            self._model, self._tokenizer, prompt=formatted, max_tokens=max_tokens
        )

        return {"text": text}
