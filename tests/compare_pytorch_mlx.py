#!/usr/bin/env python3
"""
Systematic pipeline comparison: MLX port vs PyTorch IndexTTS-2.

Runs both pipelines on the same (text, voice) input and compares intermediate
tensors at each stage. Helps pinpoint where the MLX port diverges from
the reference PyTorch implementation.

Usage:
    venv_parity/bin/python tests/compare_pytorch_mlx.py \
        --voice ~/audiobooks/voices/prunella_scales.wav \
        --text "Despite a deadlock over funding..."

Set up the parity venv once with:
    python3.11 -m venv venv_parity
    venv_parity/bin/pip install -e ".[parity]"

The parity venv requires: torch, torchaudio, transformers, safetensors,
    huggingface_hub, omegaconf, sentencepiece, mlx, soundfile, librosa.
"""

import sys
import os
import argparse
import numpy as np
from pathlib import Path

# Add paths for both repos
INDEXTTS_MLX_DIR = Path(__file__).parent.parent
PT_REPO_DIR = Path.home() / "code/temp_repos/index-tts"
sys.path.insert(0, str(INDEXTTS_MLX_DIR))
sys.path.insert(0, str(PT_REPO_DIR))

# ── helpers ──────────────────────────────────────────────────────────────────

def compare(name, pt_arr, mlx_arr, verbose=False):
    """Compare two numpy arrays and print a summary row."""
    if pt_arr is None or mlx_arr is None:
        print(f"  {name:40s} SKIPPED (None)")
        return
    pt = np.array(pt_arr, dtype=np.float32)
    mlx = np.array(mlx_arr, dtype=np.float32)
    if pt.shape != mlx.shape:
        print(f"  {name:40s} SHAPE MISMATCH  pt={pt.shape}  mlx={mlx.shape}")
        return
    diff = np.abs(pt - mlx)
    cos = float(np.dot(pt.ravel(), mlx.ravel()) /
                (np.linalg.norm(pt.ravel()) * np.linalg.norm(mlx.ravel()) + 1e-12))
    print(f"  {name:40s} shape={str(pt.shape):18s}  "
          f"max|Δ|={diff.max():.5f}  cos={cos:.5f}  "
          f"pt_μ={pt.mean():.4f}  mlx_μ={mlx.mean():.4f}")
    if verbose and diff.max() > 0.1:
        idx = np.unravel_index(diff.argmax(), diff.shape)
        print(f"    largest diff at {idx}: pt={pt[idx]:.4f}  mlx={mlx[idx]:.4f}")


def header(title):
    print(f"\n{'─'*70}")
    print(f"  {title}")
    print(f"{'─'*70}")


# ── load shared audio ─────────────────────────────────────────────────────────

def load_audio(voice_path):
    import soundfile as sf
    import librosa
    audio, sr = sf.read(voice_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=16000).astype(np.float32)[:15*16000]
    audio_22k = librosa.resample(audio, orig_sr=sr, target_sr=22050).astype(np.float32)[:15*22050]
    return audio_16k, audio_22k


# ── PyTorch pipeline ─────────────────────────────────────────────────────────

def run_pytorch(text, audio_16k, audio_22k, gpt_codes_np=None):
    """Run each stage of the PyTorch pipeline and return intermediate tensors."""
    import torch
    import torchaudio
    from transformers import Wav2Vec2BertModel, SeamlessM4TFeatureExtractor
    from huggingface_hub import hf_hub_download
    import safetensors

    tensors = {}

    # ── Stage 0-2: seamless fbank → W2V-BERT → normalized features ──────────
    header("PyTorch: W2V-BERT")
    from transformers import SeamlessM4TFeatureExtractor
    extractor = SeamlessM4TFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
    audio_t = torch.from_numpy(audio_16k).unsqueeze(0)
    inputs = extractor(audio_t, sampling_rate=16000, return_tensors="pt")
    input_features = inputs["input_features"]
    attention_mask = inputs["attention_mask"]
    tensors["seamless_input_features"] = input_features.numpy()

    w2vbert = Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0")
    w2vbert.eval()
    with torch.no_grad():
        out = w2vbert(input_features=input_features, attention_mask=attention_mask,
                      output_hidden_states=True)
    hidden17 = out.hidden_states[17]  # (1, T, 1024)
    tensors["w2vbert_hidden17"] = hidden17.numpy()

    stats = np.load(str(Path.home() / "code/index-tts-m3-port/prototypes/s2mel_mlx/mlx_weights/semantic_stats.npz"))
    mean_t = torch.from_numpy(stats["mean"])
    std_t = torch.from_numpy(stats["std"])
    sem_feat = (hidden17 - mean_t) / std_t
    tensors["semantic_features"] = sem_feat.numpy()
    print("  semantic_features:", sem_feat.shape)

    # ── Stage 3: CAMPPlus speaker style (16 kHz) ─────────────────────────────
    header("PyTorch: CAMPPlus")
    sys.path.insert(0, str(PT_REPO_DIR / "indextts/s2mel/modules/campplus"))
    from DTDNN import CAMPPlus as PTCAMPPlus
    campplus_ckpt = hf_hub_download("funasr/campplus", "campplus_cn_common.bin")
    campplus = PTCAMPPlus(feat_dim=80, embedding_size=192)
    campplus.load_state_dict(torch.load(campplus_ckpt, map_location="cpu"))
    campplus.eval()
    feat = torchaudio.compliance.kaldi.fbank(
        torch.from_numpy(audio_16k).unsqueeze(0),
        num_mel_bins=80, dither=0, sample_frequency=16000)
    feat = feat - feat.mean(dim=0, keepdim=True)
    with torch.no_grad():
        style = campplus(feat.unsqueeze(0))
    tensors["speaker_style"] = style.numpy()
    print("  speaker_style:", style.shape)

    # ── Stage 4: Semantic codec S_ref ────────────────────────────────────────
    header("PyTorch: Semantic Codec (ref)")
    sys.path.insert(0, str(PT_REPO_DIR))
    from indextts.utils.maskgct_utils import build_semantic_codec
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(str(PT_REPO_DIR / "checkpoints/config.yaml"))
    semantic_codec = build_semantic_codec(cfg.semantic_codec)
    codec_ckpt = hf_hub_download("amphion/MaskGCT", filename="semantic_codec/model.safetensors")
    safetensors.torch.load_model(semantic_codec, codec_ckpt)
    semantic_codec.eval()
    with torch.no_grad():
        _, S_ref = semantic_codec.quantize(sem_feat)
    tensors["S_ref"] = S_ref.numpy()
    print("  S_ref:", S_ref.shape)

    # ── Stage 5: Prompt regulator ─────────────────────────────────────────────
    header("PyTorch: S2Mel (load)")
    from indextts.s2mel.modules.commons import load_checkpoint2, MyModel
    s2mel_path = str(PT_REPO_DIR / "checkpoints/s2mel.pth")
    if not os.path.exists(s2mel_path):
        s2mel_path = str(Path.home() / ".cache/huggingface/hub/models--IndexTeam--IndexTTS-2/snapshots/740dcaff396282ffb241903d150ac011cd4b1ede/s2mel.pth")
    s2mel = MyModel(cfg.s2mel, use_gpt_latent=True)
    s2mel, _, _, _ = load_checkpoint2(s2mel, None, s2mel_path, load_only_params=True,
                                       ignore_modules=[], is_distributed=False)
    s2mel.eval()
    s2mel.models['cfm'].estimator.setup_caches(max_batch_size=2, max_seq_length=8192)

    import librosa
    ref_mel_np = _compute_mel_pytorch(audio_22k)  # (1, 80, T)
    ref_mel_t = torch.from_numpy(ref_mel_np)
    ref_target_lengths = torch.LongTensor([ref_mel_t.size(2)])
    with torch.no_grad():
        prompt_condition = s2mel.models['length_regulator'](
            S_ref, ylens=ref_target_lengths, n_quantizers=3, f0=None)[0]
    tensors["prompt_condition"] = prompt_condition.numpy()
    print("  prompt_condition:", prompt_condition.shape)

    # ── Stage 6: GPT conditioning ─────────────────────────────────────────────
    header("PyTorch: GPT conditioning")
    from indextts.gpt.model_v2 import UnifiedVoice
    from indextts.utils.checkpoint import load_checkpoint
    gpt = UnifiedVoice(**cfg.gpt, use_accel=False)
    gpt_ckpt = str(Path.home() / "code/tts/index-tts/checkpoints/gpt.pth")
    load_checkpoint(gpt, gpt_ckpt)
    gpt.eval()
    gpt.post_init_gpt2_config(use_deepspeed=False, kv_cache=True, half=False)

    # For conditioning, use merge_emovec equivalent
    # emo_cond_emb = spk_cond_emb (same audio for emo and spk)
    with torch.no_grad():
        emovec = gpt.merge_emovec(
            sem_feat, sem_feat,
            torch.tensor([sem_feat.shape[-1]]),
            torch.tensor([sem_feat.shape[-1]]),
            alpha=1.0)
    tensors["emovec"] = emovec.numpy()
    print("  emovec:", emovec.shape)

    # ── GPT codes (either provided or generate) ──────────────────────────────
    from sentencepiece import SentencePieceProcessor
    sp = SentencePieceProcessor(model_file=str(Path.home() / "code/tts/index-tts/checkpoints/bpe.model"))
    text_tokens = torch.tensor(sp.encode(text.upper()), dtype=torch.int32).unsqueeze(0)

    if gpt_codes_np is None:
        header("PyTorch: GPT generation")
        with torch.no_grad():
            codes, speech_latent = gpt.inference_speech(
                sem_feat, text_tokens,
                sem_feat,
                cond_lengths=torch.tensor([sem_feat.shape[-1]]),
                emo_cond_lengths=torch.tensor([sem_feat.shape[-1]]),
                emo_vec=emovec,
                do_sample=True, top_p=0.8, top_k=30, temperature=0.8,
                num_return_sequences=1, length_penalty=0.0,
                num_beams=3, repetition_penalty=10.0,
                max_generate_length=1500,
            )
        # strip stop token
        stop = cfg.gpt.stop_mel_token
        if (codes[0] == stop).any():
            stop_idx = (codes[0] == stop).nonzero(as_tuple=True)[0][0].item()
            codes = codes[:, :stop_idx]
        tensors["gpt_codes"] = codes.numpy()
        print(f"  gpt_codes: {codes.shape} (stop found: {stop_idx if 'stop_idx' in dir() else 'max'})")
        gpt_codes_np = codes.numpy()
    else:
        print(f"  (using provided codes: {gpt_codes_np.shape})")
        codes = torch.from_numpy(gpt_codes_np)

    # ── Stage 7-9: GPT latent, vq2emb, S_infer ───────────────────────────────
    header("PyTorch: GPT latent + vq2emb + S_infer")
    code_lens_t = torch.tensor([codes.shape[-1]])
    use_speed = torch.zeros(sem_feat.size(0)).long()
    with torch.no_grad():
        if 'speech_latent' not in dir():
            # When codes were provided externally, re-run inference_speech to get speech_latent
            _, speech_latent = gpt.inference_speech(
                sem_feat, text_tokens,
                sem_feat,
                cond_lengths=torch.tensor([sem_feat.shape[-1]]),
                emo_cond_lengths=torch.tensor([sem_feat.shape[-1]]),
                emo_vec=emovec,
                do_sample=True, top_p=0.8, top_k=30, temperature=0.8,
                num_return_sequences=1, length_penalty=0.0,
                num_beams=3, repetition_penalty=10.0,
                max_generate_length=codes.shape[-1] + 1,
            )
        latent = gpt(
            speech_latent, text_tokens,
            torch.tensor([text_tokens.shape[-1]]),
            codes, torch.tensor([codes.shape[-1]]),
            sem_feat,
            cond_mel_lengths=torch.tensor([sem_feat.shape[-1]]),
            emo_cond_mel_lengths=torch.tensor([sem_feat.shape[-1]]),
            emo_vec=emovec,
            use_speed=use_speed,
        )
        tensors["gpt_latent"] = latent.numpy()
        print("  gpt_latent:", latent.shape)

        latent_proj = s2mel.models['gpt_layer'](latent)
        tensors["gpt_latent_proj"] = latent_proj.numpy()
        print("  gpt_latent_proj:", latent_proj.shape)

        vq_emb = semantic_codec.quantizer.vq2emb(codes.unsqueeze(1))  # (1, dim, T)
        vq_emb = vq_emb.transpose(1, 2)                               # (1, T, dim)
        tensors["vq_emb"] = vq_emb.numpy()
        print("  vq_emb:", vq_emb.shape)

        S_infer = vq_emb + latent_proj
        tensors["S_infer"] = S_infer.numpy()
        print("  S_infer:", S_infer.shape)

    # ── Stage 10-11: regulator + CFM ─────────────────────────────────────────
    header("PyTorch: regulator + CFM")
    target_lengths = (code_lens_t * 1.72).long()
    with torch.no_grad():
        cond = s2mel.models['length_regulator'](
            S_infer, ylens=target_lengths, n_quantizers=3, f0=None)[0]
        tensors["cond"] = cond.numpy()
        print("  cond:", cond.shape)

        cat_condition = torch.cat([prompt_condition, cond], dim=1)
        mel = s2mel.models['cfm'].inference(
            cat_condition,
            torch.LongTensor([cat_condition.size(1)]),
            ref_mel_t, style, None, 25,
            inference_cfg_rate=0.7)
        mel = mel[:, :, ref_mel_t.size(-1):]
        tensors["mel"] = mel.numpy()
        print("  mel:", mel.shape)

    # ── Stage 12: BigVGAN ────────────────────────────────────────────────────
    header("PyTorch: BigVGAN")
    from indextts.s2mel.modules.bigvgan import bigvgan
    bvgan_name = cfg.vocoder.name
    bvgan = bigvgan.BigVGAN.from_pretrained(bvgan_name, use_cuda_kernel=False)
    bvgan.remove_weight_norm()
    bvgan.eval()
    with torch.no_grad():
        wav = bvgan(mel.float()).squeeze()
    tensors["waveform"] = wav.numpy()
    print("  waveform:", wav.shape)

    return tensors, gpt_codes_np


# ── MLX pipeline ─────────────────────────────────────────────────────────────

def run_mlx(text, audio_16k, audio_22k, gpt_codes_np=None):
    """Run each stage of the MLX pipeline and return intermediate tensors."""
    import mlx.core as mx
    sys.path.insert(0, str(INDEXTTS_MLX_DIR))
    from indextts_mlx.config import WeightsConfig
    from indextts_mlx.models.gpt import create_unifiedvoice
    from indextts_mlx.models.w2vbert import create_w2vbert_model
    from indextts_mlx.models.campplus import CAMPPlus
    from indextts_mlx.models.semantic_codec import RepCodec
    from indextts_mlx.models.s2mel import create_mlx_s2mel_pipeline
    from indextts_mlx.models.bigvgan import BigVGAN
    from indextts_mlx.loaders.gpt_loader import load_gpt_model
    from indextts_mlx.loaders.w2vbert_loader import load_w2vbert_model
    from indextts_mlx.loaders.campplus_loader import load_campplus_model
    from indextts_mlx.loaders.semantic_codec_loader import load_semantic_codec_model
    from indextts_mlx.loaders.bigvgan_loader import load_bigvgan_model
    from indextts_mlx.audio.kaldi_fbank import compute_kaldi_fbank_mlx
    from indextts_mlx.audio.seamless_fbank import compute_seamless_fbank
    from indextts_mlx.audio.mel import compute_mel_s2mel
    from indextts_mlx.pipeline import _sample_top_k
    from sentencepiece import SentencePieceProcessor

    cfg = WeightsConfig()
    tensors = {}

    header("MLX: loading models (quiet)")
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        w2v = create_w2vbert_model()
        w2v = load_w2vbert_model(w2v, str(cfg.w2vbert))
        stats = np.load(str(cfg.semantic_stats))
        sem_mean = mx.array(stats["mean"])
        sem_std = mx.array(stats["std"])
        campplus = CAMPPlus(feat_dim=80, embedding_size=192)
        campplus = load_campplus_model(campplus, str(cfg.campplus))
        campplus.eval()
        gpt = create_unifiedvoice()
        gpt = load_gpt_model(gpt, str(cfg.gpt))
        semantic_codec = RepCodec()
        semantic_codec = load_semantic_codec_model(semantic_codec, str(cfg.semantic_codec))
        s2mel = create_mlx_s2mel_pipeline(checkpoint_path=str(cfg.s2mel))
        bigvgan = BigVGAN()
        bigvgan = load_bigvgan_model(bigvgan, str(cfg.bigvgan))
        bigvgan.eval()
    print("  models loaded")

    # ── W2V-BERT ──────────────────────────────────────────────────────────────
    header("MLX: W2V-BERT")
    feats_np = compute_seamless_fbank(audio_16k)
    tensors["seamless_input_features"] = feats_np[None]  # (1, T, 160)
    mlx_feats = mx.array(feats_np[None])
    T = feats_np.shape[0]
    mask = mx.ones((1, T), dtype=mx.int32)
    out = w2v(input_features=mlx_feats, attention_mask=mask, output_hidden_states=True)
    hidden17 = out.hidden_states[17]
    mx.eval(hidden17)
    tensors["w2vbert_hidden17"] = np.array(hidden17)
    sem_feat = (hidden17 - sem_mean) / sem_std
    mx.eval(sem_feat)
    tensors["semantic_features"] = np.array(sem_feat)
    print("  semantic_features:", sem_feat.shape)

    # ── CAMPPlus (16 kHz) ─────────────────────────────────────────────────────
    header("MLX: CAMPPlus")
    feat = compute_kaldi_fbank_mlx(mx.array(audio_16k), num_mel_bins=80, sample_frequency=16000)
    feat = feat - feat.mean(axis=0, keepdims=True)
    style = campplus(feat[None])
    mx.eval(style)
    tensors["speaker_style"] = np.array(style)
    print("  speaker_style:", style.shape)

    # ── Semantic codec ref ────────────────────────────────────────────────────
    header("MLX: Semantic Codec (ref)")
    _, S_ref = semantic_codec.quantize(sem_feat)
    mx.eval(S_ref)
    tensors["S_ref"] = np.array(S_ref)
    print("  S_ref:", S_ref.shape)

    # ── Prompt regulator ──────────────────────────────────────────────────────
    header("MLX: prompt regulator")
    ref_mel_80 = compute_mel_s2mel(audio_22k)
    ref_target_lengths = mx.array([ref_mel_80.shape[2]], dtype=mx.int32)
    mx.eval(ref_mel_80)
    prompt_condition, _ = s2mel.regulator(S_ref, ylens=ref_target_lengths, f0=None)
    mx.eval(prompt_condition)
    tensors["prompt_condition"] = np.array(prompt_condition)
    tensors["ref_mel"] = np.array(ref_mel_80)
    print("  prompt_condition:", prompt_condition.shape)

    # ── GPT conditioning ──────────────────────────────────────────────────────
    header("MLX: GPT conditioning")
    cond_latents = gpt.get_full_conditioning_34(sem_feat, emotion_scale=1.0)
    mx.eval(cond_latents)
    tensors["emovec"] = np.array(cond_latents)  # (1, 34, 1280) — use as proxy
    print("  cond_latents_34:", cond_latents.shape)

    sp = SentencePieceProcessor(model_file=str(cfg.bpe_model))
    text_tokens = mx.array([sp.encode(text.upper())])
    inputs_embeds, _ = gpt.prepare_inputs(cond_latents, text_tokens)
    mx.eval(inputs_embeds)

    # ── GPT generation ────────────────────────────────────────────────────────
    if gpt_codes_np is None:
        header("MLX: GPT generation")
        generated = [[gpt.start_mel_token]]
        cache = None
        np.random.seed(42)  # reproducible for comparison
        for step_idx in range(1500):
            if step_idx == 0:
                sme = gpt.mel_embedding(mx.array([[gpt.start_mel_token]]))
                smp = gpt.mel_pos_embedding.emb(mx.array([0]))
                cur_input = mx.concatenate([inputs_embeds, sme + smp[None]], axis=1)
                cur_mask = mx.ones((1, cur_input.shape[1]), dtype=mx.int32)
            else:
                last_tok = mx.array([[generated[0][-1]]])
                cur_input = gpt.mel_embedding(last_tok)
                pos_idx = len(generated[0])
                if pos_idx < gpt.mel_pos_embedding.emb.weight.shape[0]:
                    cur_input = cur_input + gpt.mel_pos_embedding.emb(mx.array([pos_idx]))[None]
                cur_mask = None
            hidden, cache = gpt.gpt(cur_input, cur_mask, cache)
            logits = gpt.mel_head(gpt.final_norm(hidden[:, -1, :]))
            next_tok = _sample_top_k(logits[0], temperature=0.8, top_k=30)
            mx.eval(next_tok)
            generated[0].append(next_tok)
            if next_tok == gpt.stop_mel_token:
                break
        codes_np = np.array(generated[0])
        start, stop = gpt.start_mel_token, gpt.stop_mel_token
        codes_np = codes_np[(codes_np != start) & (codes_np != stop)]
        tensors["gpt_codes"] = codes_np[None]
        print(f"  gpt_codes: {codes_np.shape}")
        gpt_codes_np = codes_np[None]
    else:
        print(f"  (using provided codes: {gpt_codes_np.shape})")
        codes_np = gpt_codes_np[0]

    semantic_codes = mx.array([codes_np.tolist()])

    # ── GPT forward for latent ────────────────────────────────────────────────
    header("MLX: GPT latent + vq2emb + S_infer")
    gpt_latent = gpt.forward_for_latent(cond_latents, text_tokens, semantic_codes)
    gpt_latent_proj = s2mel.gpt_layer(gpt_latent)
    mx.eval(gpt_latent, gpt_latent_proj)
    tensors["gpt_latent"] = np.array(gpt_latent)
    tensors["gpt_latent_proj"] = np.array(gpt_latent_proj)
    print("  gpt_latent:", gpt_latent.shape)

    codes_for_codec = semantic_codes[np.newaxis, :, :]
    vq_emb = semantic_codec.vq2emb(codes_for_codec)
    mx.eval(vq_emb)
    tensors["vq_emb"] = np.array(vq_emb)
    print("  vq_emb:", vq_emb.shape)

    S_infer = vq_emb + gpt_latent_proj
    mx.eval(S_infer)
    tensors["S_infer"] = np.array(S_infer)
    print("  S_infer:", S_infer.shape)

    # ── regulator + CFM ───────────────────────────────────────────────────────
    header("MLX: regulator + CFM")
    target_lengths = mx.array([int(codes_np.shape[0] * 1.72)], dtype=mx.int32)
    cond, _ = s2mel.regulator(S_infer, ylens=target_lengths, f0=None)
    mx.eval(cond)
    tensors["cond"] = np.array(cond)
    print("  cond:", cond.shape)

    cat_condition = mx.concatenate([prompt_condition, cond], axis=1)
    cat_len = mx.array([cat_condition.shape[1]], dtype=mx.int32)
    mel = s2mel.cfm.inference(
        mu=cat_condition, x_lens=cat_len, prompt=ref_mel_80,
        style=style, f0=None, n_timesteps=25,
        temperature=1.0, inference_cfg_rate=0.7)
    mel = mel[:, :, ref_mel_80.shape[2]:]
    mx.eval(mel)
    tensors["mel"] = np.array(mel)
    print("  mel:", mel.shape)

    # ── BigVGAN ───────────────────────────────────────────────────────────────
    header("MLX: BigVGAN")
    audio_out = bigvgan(mel)
    mx.eval(audio_out)
    tensors["waveform"] = np.array(audio_out).squeeze()
    print("  waveform:", tensors["waveform"].shape)

    return tensors, gpt_codes_np


# ── helper ────────────────────────────────────────────────────────────────────

def _compute_mel_pytorch(audio_22k):
    import librosa
    n_fft, hop_length, win_length, n_mels = 1024, 256, 1024, 80
    pad = (n_fft - hop_length) // 2
    audio_padded = np.pad(audio_22k, pad, mode='reflect')
    stft = librosa.stft(audio_padded, n_fft=n_fft, hop_length=hop_length,
                        win_length=win_length, window='hann', center=False)
    spec = np.sqrt(stft.real**2 + stft.imag**2 + 1e-9)
    mel_basis = librosa.filters.mel(sr=22050, n_fft=n_fft, n_mels=n_mels,
                                    fmin=0, fmax=None, norm='slaney', htk=False)
    mel_log = np.log(np.clip(mel_basis @ spec, a_min=1e-5, a_max=None))
    return mel_log[np.newaxis, :, :].astype(np.float32)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compare MLX vs PyTorch IndexTTS-2 pipeline")
    parser.add_argument("--text", default=(
        "Despite a deadlock over funding for the agency, lawmakers left town "
        "and left Democratic and White House negotiators to try to work out a "
        "deal in their absence."))
    parser.add_argument("--voice", default=str(Path.home() / "audiobooks/voices/prunella_scales.wav"))
    parser.add_argument("--save-dir", default=None, help="Directory to save tensors as .npz")
    parser.add_argument("--mlx-only", action="store_true", help="Run only the MLX pipeline")
    parser.add_argument("--pt-only", action="store_true", help="Run only the PyTorch pipeline")
    args = parser.parse_args()

    print(f"Text: {args.text[:80]}")
    print(f"Voice: {args.voice}")

    audio_16k, audio_22k = load_audio(args.voice)
    print(f"Audio: 16k={len(audio_16k)/16000:.2f}s  22k={len(audio_22k)/22050:.2f}s")

    pt_tensors = None
    mlx_tensors = None
    shared_codes = None

    if not args.mlx_only:
        print("\n" + "="*70)
        print("  PYTORCH PIPELINE")
        print("="*70)
        pt_tensors, shared_codes = run_pytorch(args.text, audio_16k, audio_22k)

    if not args.pt_only:
        print("\n" + "="*70)
        print("  MLX PIPELINE")
        print("="*70)
        # Use same GPT codes from PyTorch if available (for fair comparison of downstream stages)
        mlx_tensors, _ = run_mlx(args.text, audio_16k, audio_22k,
                                  gpt_codes_np=shared_codes)

    if pt_tensors is not None and mlx_tensors is not None:
        print("\n" + "="*70)
        print("  COMPARISON")
        print("="*70)
        stages = [
            ("seamless_input_features", "seamless fbank features"),
            ("w2vbert_hidden17",        "W2V-BERT hidden[17]"),
            ("semantic_features",       "normalized semantic features"),
            ("speaker_style",           "CAMPPlus speaker style"),
            ("S_ref",                   "semantic_codec S_ref"),
            ("prompt_condition",        "prompt regulator output"),
            ("emovec",                  "GPT conditioning"),
            ("gpt_latent",              "GPT latent"),
            ("gpt_latent_proj",         "GPT latent proj"),
            ("vq_emb",                  "vq2emb"),
            ("S_infer",                 "S_infer (vq+latent)"),
            ("cond",                    "inference regulator"),
            ("mel",                     "CFM mel output"),
            ("waveform",                "BigVGAN waveform"),
        ]
        for key, name in stages:
            pt = pt_tensors.get(key)
            mlx = mlx_tensors.get(key)
            compare(name, pt, mlx)

    if args.save_dir:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        if pt_tensors:
            np.savez(save_dir / "pt_tensors.npz", **{k: v for k, v in pt_tensors.items() if v is not None})
        if mlx_tensors:
            np.savez(save_dir / "mlx_tensors.npz", **{k: v for k, v in mlx_tensors.items() if v is not None})
        print(f"\nSaved tensors to {save_dir}")


if __name__ == "__main__":
    main()
