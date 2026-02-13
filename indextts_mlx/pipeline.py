"""IndexTTS-2 MLX inference pipeline."""
from __future__ import annotations

import numpy as np
import mlx.core as mx
import soundfile as sf
import librosa
from pathlib import Path
from typing import Union

from .config import WeightsConfig
from .models.gpt import UnifiedVoice, create_unifiedvoice
from .models.w2vbert import create_w2vbert_model
from .models.campplus import CAMPPlus
from .models.semantic_codec import RepCodec
from .models.s2mel import MLXS2MelPipeline, create_mlx_s2mel_pipeline
from .models.bigvgan import BigVGAN
from .loaders.gpt_loader import load_gpt_model
from .loaders.w2vbert_loader import load_w2vbert_model
from .loaders.campplus_loader import load_campplus_model
from .loaders.semantic_codec_loader import load_semantic_codec_model
from .loaders.bigvgan_loader import load_bigvgan_model
from .audio.kaldi_fbank import compute_kaldi_fbank_mlx
from .audio.seamless_fbank import compute_seamless_fbank
from .audio.mel import compute_mel_s2mel
from sentencepiece import SentencePieceProcessor


def _sample_top_k(logits: mx.array, temperature: float = 0.8, top_k: int = 200) -> int:
    """Sample from logits with temperature and top-k filtering."""
    if temperature <= 0.0 or top_k <= 1:
        return int(mx.argmax(logits).item())
    logits = logits / temperature
    # Top-k: zero out everything except the top-k logits
    if top_k < logits.shape[-1]:
        # Sort descending, keep top_k
        sorted_indices = mx.argsort(-logits)  # descending
        cutoff = float(logits[sorted_indices[top_k - 1]].item())
        logits = mx.where(logits >= cutoff, logits, mx.full(logits.shape, float('-inf')))
    probs = mx.softmax(logits, axis=-1)
    # Multinomial sampling via inverse CDF
    probs_np = np.array(probs)
    return int(np.random.choice(len(probs_np), p=probs_np / probs_np.sum()))


class IndexTTS2:
    """IndexTTS-2 text-to-speech pipeline using pure MLX inference.
    
    Usage:
        tts = IndexTTS2()
        audio = tts.synthesize("Hello world", "reference.wav")
        import soundfile as sf
        sf.write("output.wav", audio, 22050)
    """

    def __init__(self, config: WeightsConfig = None):
        """Load all models. This is expensive -- create once and reuse."""
        if config is None:
            config = WeightsConfig()
        self.config = config

        # Load BPE tokenizer
        self.sp = SentencePieceProcessor(model_file=str(config.bpe_model))

        # Load W2V-BERT
        self.w2vbert = create_w2vbert_model()
        self.w2vbert = load_w2vbert_model(self.w2vbert, str(config.w2vbert))
        stats = np.load(str(config.semantic_stats))
        self.semantic_mean = mx.array(stats['mean'])
        self.semantic_std = mx.array(stats['std'])

        # Load CAMPPlus speaker encoder
        self.campplus = CAMPPlus(feat_dim=80, embedding_size=192)
        self.campplus = load_campplus_model(self.campplus, str(config.campplus))
        self.campplus.eval()

        # Load GPT (UnifiedVoice)
        self.gpt = create_unifiedvoice()
        self.gpt = load_gpt_model(self.gpt, str(config.gpt))

        # Load semantic codec
        self.semantic_codec = RepCodec()
        self.semantic_codec = load_semantic_codec_model(self.semantic_codec, str(config.semantic_codec))

        # Load S2Mel pipeline
        self.s2mel = create_mlx_s2mel_pipeline(checkpoint_path=str(config.s2mel))

        # Load BigVGAN vocoder
        self.bigvgan = BigVGAN()
        self.bigvgan = load_bigvgan_model(self.bigvgan, str(config.bigvgan))
        self.bigvgan.eval()

    def _load_audio(self, reference_audio, sample_rate=None):
        """Load and resample reference audio to 16kHz and 22kHz."""
        if isinstance(reference_audio, (str, Path)):
            audio, sr = sf.read(str(reference_audio))
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
        else:
            audio = np.asarray(reference_audio, dtype=np.float32)
            sr = sample_rate
            if sr is None:
                raise ValueError("sample_rate must be provided when reference_audio is a numpy array")

        audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=16000).astype(np.float32)[:15*16000]
        audio_22k = librosa.resample(audio, orig_sr=sr, target_sr=22050).astype(np.float32)[:15*22050]
        return audio_16k, audio_22k

    def synthesize(
        self,
        text: str,
        reference_audio: Union[str, Path, np.ndarray],
        *,
        sample_rate: int = None,
        emotion: float = 1.0,
        cfm_steps: int = 10,
        temperature: float = 1.0,
        max_codes: int = 1500,
        cfg_rate: float = 0.7,
        gpt_temperature: float = 0.8,
        top_k: int = 30,
    ) -> np.ndarray:
        """Synthesize speech from text using a reference voice.
        
        Args:
            text: Text to synthesize.
            reference_audio: Path to reference audio file, or numpy array of audio samples.
            sample_rate: Sample rate of reference_audio (required if numpy array).
            emotion: Emotion intensity scale. 0.0=neutral, 1.0=default, 2.0=expressive.
            cfm_steps: Number of CFM diffusion steps. 10=fast, 25=quality.
            temperature: CFM sampling temperature.
            max_codes: Maximum number of GPT tokens to generate.
            cfg_rate: Classifier-free guidance rate.
            gpt_temperature: GPT sampling temperature. 0.8 matches original IndexTTS-2.
            top_k: Top-k for GPT token sampling. 200 matches original IndexTTS-2.
        
        Returns:
            Audio as float32 numpy array at 22050 Hz.
        """
        audio_16k, audio_22k = self._load_audio(reference_audio, sample_rate)

        # Tokenize text
        text_tokens = mx.array([self.sp.encode(text.upper())])

        # Compute reference mel
        ref_mel_80 = compute_mel_s2mel(audio_22k)
        ref_target_lengths = mx.array([ref_mel_80.shape[2]], dtype=mx.int32)
        mx.eval(ref_mel_80)

        # 1. Feature extraction (W2V-BERT)
        feats_np = compute_seamless_fbank(audio_16k)
        mlx_feats = mx.array(feats_np[None])
        T_feat = feats_np.shape[0]
        mask = mx.ones((1, T_feat), dtype=mx.int32)
        out = self.w2vbert(input_features=mlx_feats, attention_mask=mask, output_hidden_states=True)
        semantic_features = (out.hidden_states[17] - self.semantic_mean) / self.semantic_std
        mx.eval(semantic_features)

        # 2. CAMPPlus speaker style (16 kHz, matching PyTorch IndexTTS-2)
        feat = compute_kaldi_fbank_mlx(mx.array(audio_16k), num_mel_bins=80, sample_frequency=16000)
        feat = feat - feat.mean(axis=0, keepdims=True)
        speaker_style = self.campplus(feat[None])
        mx.eval(speaker_style)

        # 3. Semantic codec ref
        _, S_ref = self.semantic_codec.quantize(semantic_features)
        prompt_condition, _ = self.s2mel.regulator(S_ref, ylens=ref_target_lengths, f0=None)
        mx.eval(prompt_condition)

        # 4a. GPT conditioning (with emotion scaling)
        cond_latents_34 = self.gpt.get_full_conditioning_34(semantic_features, emotion_scale=emotion)
        inputs_embeds, _ = self.gpt.prepare_inputs(cond_latents_34, text_tokens)
        mx.eval(cond_latents_34, inputs_embeds)

        # 4b. GPT autoregressive generation
        generated = [[self.gpt.start_mel_token]]
        cache = None
        for step_idx in range(max_codes):
            if step_idx == 0:
                sme = self.gpt.mel_embedding(mx.array([[self.gpt.start_mel_token]]))
                smp = self.gpt.mel_pos_embedding.emb(mx.array([0]))
                cur_input = mx.concatenate([inputs_embeds, sme + smp[None]], axis=1)
                cur_mask = mx.ones((1, cur_input.shape[1]), dtype=mx.int32)
            else:
                last_tok = mx.array([[generated[0][-1]]])
                cur_input = self.gpt.mel_embedding(last_tok)
                pos_idx = len(generated[0])
                if pos_idx < self.gpt.mel_pos_embedding.emb.weight.shape[0]:
                    cur_input = cur_input + self.gpt.mel_pos_embedding.emb(mx.array([pos_idx]))[None]
                cur_mask = None
            hidden, cache = self.gpt.gpt(cur_input, cur_mask, cache)
            logits = self.gpt.mel_head(self.gpt.final_norm(hidden[:, -1, :]))
            next_tok = _sample_top_k(logits[0], temperature=gpt_temperature, top_k=top_k)
            mx.eval(next_tok)
            generated[0].append(next_tok)
            if next_tok == self.gpt.stop_mel_token:
                break

        codes_np = np.array(generated[0])
        start, stop = self.gpt.start_mel_token, self.gpt.stop_mel_token
        codes_np = codes_np[(codes_np != start) & (codes_np != stop)]
        semantic_codes = mx.array([codes_np.tolist()])

        # 5. GPT forward for latent
        gpt_latent = self.gpt.forward_for_latent(cond_latents_34, text_tokens, semantic_codes)
        gpt_latent_proj = self.s2mel.gpt_layer(gpt_latent)
        codes_for_codec = semantic_codes[np.newaxis, :, :]
        vq_emb = self.semantic_codec.vq2emb(codes_for_codec)
        S_infer = vq_emb + gpt_latent_proj
        mx.eval(S_infer)

        # 6. S2Mel
        target_lengths = mx.array([int(codes_np.shape[0] * 1.72)], dtype=mx.int32)
        cond, _ = self.s2mel.regulator(S_infer, ylens=target_lengths, f0=None)
        cat_condition = mx.concatenate([prompt_condition, cond], axis=1)
        cat_len = mx.array([cat_condition.shape[1]], dtype=mx.int32)
        mel = self.s2mel.cfm.inference(
            mu=cat_condition, x_lens=cat_len, prompt=ref_mel_80,
            style=speaker_style, f0=None, n_timesteps=cfm_steps,
            temperature=temperature, inference_cfg_rate=cfg_rate)
        mel = mel[:, :, ref_mel_80.shape[2]:]
        mx.eval(mel)

        # 7. BigVGAN vocoder
        audio_out = self.bigvgan(mel)
        mx.eval(audio_out)

        return np.array(audio_out).squeeze().astype(np.float32)


def synthesize(
    text: str,
    reference_audio: Union[str, Path, np.ndarray],
    **kwargs,
) -> np.ndarray:
    """One-shot synthesis. Loads all models on each call.
    
    For repeated synthesis, use IndexTTS2 class directly to avoid reloading weights.
    """
    tts = IndexTTS2(config=kwargs.pop('config', None))
    return tts.synthesize(text, reference_audio, **kwargs)
