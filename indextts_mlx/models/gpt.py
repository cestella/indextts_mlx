"""
UnifiedVoice GPT model for text-to-semantic-code generation (MLX).

Architecture:
1. Conformer encoder: mel → conditioning latents
2. Perceiver resampler: variable length → fixed 32 latents
3. Text/mel embeddings + positional embeddings
4. GPT2 transformer: [cond || text] → semantic codes
5. Output heads: mel_head (8194 codes)
"""

from typing import Optional, Tuple
import numpy as np
import mlx.core as mx
import mlx.nn as nn

from .conformer import ConformerEncoder
from .perceiver import PerceiverResampler
from .gpt2 import GPT2Model


class LearnedPositionEmbeddings(nn.Module):
    """Learned positional embeddings (matches PyTorch implementation)."""

    def __init__(self, max_len: int, dim: int):
        super().__init__()
        # Use Embedding layer like PyTorch, not a raw matrix
        self.emb = nn.Embedding(max_len, dim)
        # Initialize with same std as PyTorch (0.02)
        self.emb.weight = mx.random.normal((max_len, dim)) * 0.02

    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: (B, T, dim) input sequence (only T is used)
        Returns:
            pos_emb: (T, dim) positional embeddings
        """
        T = x.shape[1]
        # Create position indices [0, 1, 2, ..., T-1] and lookup embeddings
        positions = mx.arange(T)
        return self.emb(positions)  # (T, dim)


class UnifiedVoice(nn.Module):
    """UnifiedVoice: GPT model for text-to-semantic-token generation.

    Generation flow:
    1. Encode reference mel → conditioning latents (32 tokens)
    2. Embed text tokens → text embeddings
    3. Concatenate: [cond_latents || text_emb || START_MEL]
    4. Autoregressive generation: predict semantic codes
    5. Stop when STOP_MEL_TOKEN is generated
    """

    def __init__(
        self,
        # GPT config
        num_layers: int = 24,
        model_dim: int = 1280,
        num_heads: int = 20,  # GPT model has 20 attention heads
        # Vocabulary config
        number_text_tokens: int = 12001,
        number_mel_codes: int = 8194,
        start_mel_token: int = 8192,
        stop_mel_token: int = 8193,
        # Position embedding config
        max_text_tokens: int = 602,
        max_mel_tokens: int = 1818,
        # Conditioning config
        conditioning_dim: int = 512,
        num_latents: int = 32,
        # Conformer config (for conditioning encoder)
        conformer_layers: int = 6,
        conformer_heads: int = 8,
        conformer_dim: int = 512,
        conformer_intermediate_dim: int = 2048,
        mel_input_channels: int = 1024,  # GPT expects 1024-bin mel spectrogram!
    ):
        super().__init__()
        self.model_dim = model_dim
        self.num_layers = num_layers
        self.num_heads = num_heads

        self.number_text_tokens = number_text_tokens
        self.number_mel_codes = number_mel_codes
        self.start_mel_token = start_mel_token
        self.stop_mel_token = stop_mel_token

        self.num_latents = num_latents

        # Conditioning encoder: mel → (B, T, conditioning_dim)
        self.conditioning_encoder = ConformerEncoder(
            input_dim=mel_input_channels,
            output_dim=conformer_dim,
            num_layers=conformer_layers,
            num_heads=conformer_heads,
            intermediate_dim=conformer_intermediate_dim,
        )

        # Perceiver resampler: (B, T, conditioning_dim) → (B, num_latents, model_dim)
        # Note: Perceiver has 8 heads (not num_heads=20 which is for GPT)
        self.perceiver_encoder = PerceiverResampler(
            dim=model_dim,
            depth=2,
            dim_context=conditioning_dim,
            num_latents=num_latents,
            dim_head=64,
            heads=8,  # Perceiver uses 8 heads, not GPT's 20
            ff_mult=4.0,
        )

        # Emotion conditioning encoder: 4 layers, 4 heads × 128 dim_head
        # (different from conditioning_encoder: 6 layers, 8 heads × 64 dim_head)
        self.emo_conditioning_encoder = ConformerEncoder(
            input_dim=mel_input_channels,
            output_dim=conformer_dim,
            num_layers=4,
            num_heads=4,
            intermediate_dim=conformer_intermediate_dim,
        )

        # Emotion perceiver resampler: compresses to 1 latent at dim=1024
        # Weights show: latents=(1,1024), proj_context=(1024,512), heads=4, dim_head=64, ff_mult=2.0
        self.emo_perceiver_encoder = PerceiverResampler(
            dim=1024,
            depth=2,
            dim_context=conformer_dim,
            num_latents=1,
            dim_head=64,
            heads=4,
            ff_mult=2.0,
        )

        # Emotion vector layers
        self.emovec_layer = nn.Linear(1024, model_dim)   # (1024 → 1280)
        self.emo_layer = nn.Linear(model_dim, model_dim)  # (1280 → 1280)

        # Duration/speed embeddings (2 entries: speed_emb(0)=normal, speed_emb(1)=half)
        self.speed_emb = nn.Embedding(2, model_dim)

        # Text and mel embeddings (use nn.Embedding like PyTorch)
        self.text_embedding = nn.Embedding(number_text_tokens, model_dim)
        self.text_embedding.weight = mx.random.normal((number_text_tokens, model_dim)) * 0.02

        self.mel_embedding = nn.Embedding(number_mel_codes, model_dim)
        self.mel_embedding.weight = mx.random.normal((number_mel_codes, model_dim)) * 0.02

        # Position embeddings
        self.text_pos_embedding = LearnedPositionEmbeddings(max_text_tokens, model_dim)
        self.mel_pos_embedding = LearnedPositionEmbeddings(max_mel_tokens, model_dim)

        # GPT2 transformer
        self.gpt = GPT2Model(
            num_layers=num_layers,
            model_dim=model_dim,
            num_heads=num_heads,
            intermediate_dim=4 * model_dim,  # 5120 for model_dim=1280
        )

        # Output head
        self.final_norm = nn.LayerNorm(model_dim)
        self.mel_head = nn.Linear(model_dim, number_mel_codes)

    def get_conditioning(self, mel: mx.array, mel_lengths: Optional[mx.array] = None) -> mx.array:
        """Encode mel spectrogram to conditioning latents.

        Args:
            mel: (B, T_mel, mel_channels) mel spectrogram
            mel_lengths: (B,) lengths of mel spectrograms
        Returns:
            cond_latents: (B, num_latents, model_dim) conditioning
        """
        # Conformer encoder
        encoded, mask = self.conditioning_encoder(mel, mel_lengths)

        # Perceiver resampler
        # mask: (B, 1, T') -> (B, T') for perceiver
        if mask is not None:
            mask = mask.squeeze(1)

        cond_latents = self.perceiver_encoder(encoded, mask)

        return cond_latents

    def get_emo_conditioning(self, mel: mx.array, mel_lengths: Optional[mx.array] = None) -> mx.array:
        """Encode mel to emotion vector.

        Args:
            mel: (B, T_mel, mel_channels) mel spectrogram
        Returns:
            emo_vec: (B, 1280) emotion vector after emovec_layer + emo_layer
        """
        encoded, mask = self.emo_conditioning_encoder(mel, mel_lengths)
        if mask is not None:
            mask = mask.squeeze(1)
        # emo_perceiver produces (B, 1, 1024) → squeeze to (B, 1024)
        emo_raw = self.emo_perceiver_encoder(encoded, mask)  # (B, 1, 1024)
        emo_raw = emo_raw[:, 0, :]  # (B, 1024)
        emo_vec = self.emovec_layer(emo_raw)   # (B, 1280)
        emo_vec = self.emo_layer(emo_vec)      # (B, 1280)
        return emo_vec

    def get_full_conditioning_34(
        self,
        mel: mx.array,
        mel_lengths: Optional[mx.array] = None,
        speed: int = 0,
        emotion_scale: float = 1.0,
        emo_vec_override: Optional[mx.array] = None,
    ) -> mx.array:
        """Compute the full 34-latent conditioning used by inference.

        Replicates PyTorch's inference_speech conditioning:
          conds = [speech_conditioning_latent + emo_vec, duration_emb_half, duration_emb]

        Args:
            mel: (B, T_mel, mel_channels) mel spectrogram (1024-bin)
            mel_lengths: (B,) lengths
            speed: 0=normal, 1=half speed
            emotion_scale: Scale factor for emotion vector. 0.0=neutral, 1.0=default, 2.0=expressive.
        Returns:
            cond_34: (B, 34, 1280) full conditioning latents
        """
        B = mel.shape[0]

        # 32 base latents
        cond_32 = self.get_conditioning(mel, mel_lengths)  # (B, 32, 1280)

        # Emotion vector (use override if provided, otherwise derive from mel)
        if emo_vec_override is not None:
            emo_vec = emo_vec_override  # (B, 1280)
        else:
            emo_vec = self.get_emo_conditioning(mel, mel_lengths)  # (B, 1280)

        # Combine: speech_conditioning_latent + emo_vec
        cond_with_emo = cond_32 + emotion_scale * emo_vec[:, None, :]  # (B, 32, 1280)

        # Duration embeddings
        speed_idx = mx.array([speed] * B)
        half_idx = mx.array([1] * B)  # speed_emb(1) = duration_emb_half
        norm_idx = mx.array([0] * B)  # speed_emb(0) = duration_emb

        duration_emb_half = self.speed_emb(half_idx)  # (B, 1280)
        duration_emb = self.speed_emb(norm_idx)        # (B, 1280)

        # Concatenate: [cond_with_emo (32) || duration_half (1) || duration (1)]
        cond_34 = mx.concatenate([
            cond_with_emo,
            duration_emb_half[:, None, :],  # (B, 1, 1280)
            duration_emb[:, None, :],       # (B, 1, 1280)
        ], axis=1)  # (B, 34, 1280)

        return cond_34

    def prepare_inputs(
        self,
        cond_latents: mx.array,
        text_tokens: mx.array,
    ) -> Tuple[mx.array, mx.array]:
        """Prepare inputs for GPT2 generation.

        Args:
            cond_latents: (B, num_latents, model_dim) conditioning
            text_tokens: (B, T_text) text token ids
        Returns:
            inputs_embeds: (B, num_latents + T_text + 2, model_dim) combined embeddings
            attention_mask: (B, num_latents + T_text + 2) attention mask
        """
        B, T_text = text_tokens.shape

        # Filter out any existing START/STOP tokens (0, 1) and wrap with new ones
        # This matches PyTorch's prepare_gpt_inputs behavior
        START_TEXT_TOKEN = 0
        STOP_TEXT_TOKEN = 1

        # For each batch element, filter and add START/STOP
        text_tokens_list = []
        for b in range(B):
            tokens_np = np.array(text_tokens[b])

            # Remove any existing START (0) or STOP (1) tokens
            valid_indices = np.where((tokens_np != START_TEXT_TOKEN) & (tokens_np != STOP_TEXT_TOKEN))[0]
            valid_tokens = tokens_np[valid_indices]

            # Add START at beginning and STOP at end: [0, ...tokens..., 1]
            wrapped_tokens = np.concatenate([
                np.array([START_TEXT_TOKEN]),
                valid_tokens,
                np.array([STOP_TEXT_TOKEN])
            ])
            text_tokens_list.append(mx.array(wrapped_tokens))

        # Stack wrapped tokens
        text_tokens_wrapped = mx.stack(text_tokens_list, axis=0)  # (B, T_text + 2)

        # Embed text tokens
        text_emb = self.text_embedding(text_tokens_wrapped)  # (B, T_text + 2, model_dim)

        # Add positional embeddings (returns (T, dim), broadcast to (B, T, dim))
        text_pos_emb = self.text_pos_embedding(text_emb)  # (T_text + 2, model_dim)
        text_emb = text_emb + text_pos_emb[None, :, :]  # Broadcast to (B, T_text + 2, model_dim)

        # Concatenate conditioning and text embeddings
        inputs_embeds = mx.concatenate([cond_latents, text_emb], axis=1)  # (B, num_latents + T_text + 2, model_dim)

        # Attention mask (attend to all tokens)
        attention_mask = mx.ones((B, inputs_embeds.shape[1]), dtype=mx.int32)

        return inputs_embeds, attention_mask

    def forward(
        self,
        mel: mx.array,
        text_tokens: mx.array,
        mel_lengths: Optional[mx.array] = None,
    ) -> mx.array:
        """Forward pass (for testing/validation).

        Args:
            mel: (B, T_mel, mel_channels) conditioning mel
            text_tokens: (B, T_text) text tokens
            mel_lengths: (B,) mel lengths
        Returns:
            logits: (B, num_latents + T_text, number_mel_codes)
        """
        # Get conditioning
        cond_latents = self.get_conditioning(mel, mel_lengths)

        # Prepare inputs
        inputs_embeds, attention_mask = self.prepare_inputs(cond_latents, text_tokens)

        # GPT2 forward
        hidden_states, _ = self.gpt(inputs_embeds, attention_mask)

        # Final norm and output head
        hidden_states = self.final_norm(hidden_states)
        logits = self.mel_head(hidden_states)

        return logits

    def forward_for_latent(
        self,
        cond_latents: mx.array,
        text_tokens: mx.array,
        mel_codes: mx.array,
    ) -> mx.array:
        """GPT forward pass to extract mel hidden states for s2mel conditioning.

        Mirrors PyTorch's gpt.forward(speech_conditioning_latent, ..., mel_codes, ...).
        Returns mel hidden states: (B, T_mel, model_dim) - same shape as mel_codes length.

        Args:
            cond_latents: (B, 34, model_dim) - precomputed 34-latent conditioning
            text_tokens: (B, T_text) - text token ids
            mel_codes: (B, T_mel) - generated mel codes WITHOUT START/STOP tokens
        Returns:
            latent: (B, T_mel, model_dim) mel hidden states
        """
        B = cond_latents.shape[0]
        T_mel = mel_codes.shape[1]

        # Prepare text embeddings (matching prepare_inputs logic)
        START_TEXT_TOKEN = 0
        STOP_TEXT_TOKEN = 1
        text_tokens_list = []
        for b in range(B):
            tokens_np = np.array(text_tokens[b])
            valid_indices = np.where((tokens_np != START_TEXT_TOKEN) & (tokens_np != STOP_TEXT_TOKEN))[0]
            valid_tokens = tokens_np[valid_indices]
            wrapped_tokens = np.concatenate([
                np.array([START_TEXT_TOKEN]),
                valid_tokens,
                np.array([STOP_TEXT_TOKEN])
            ])
            text_tokens_list.append(mx.array(wrapped_tokens))
        text_tokens_wrapped = mx.stack(text_tokens_list, axis=0)
        text_emb = self.text_embedding(text_tokens_wrapped)
        text_pos_emb = self.text_pos_embedding(text_emb)
        text_emb = text_emb + text_pos_emb[None, :, :]
        T_text_wrapped = text_tokens_wrapped.shape[1]

        # Prepare mel embeddings: [START, codes..., STOP]
        # PyTorch: set_mel_padding + pad(STOP) + build_aligned_inputs_and_targets
        start_tokens = mx.full((B, 1), self.start_mel_token, dtype=mx.int32)
        stop_tokens = mx.full((B, 1), self.stop_mel_token, dtype=mx.int32)
        mel_input = mx.concatenate([start_tokens, mel_codes, stop_tokens], axis=1)  # (B, T_mel+2)
        mel_emb = self.mel_embedding(mel_input)  # (B, T_mel+2, model_dim)

        # Add mel positional embeddings
        mel_seq_len = mel_input.shape[1]
        mel_positions = mx.arange(mel_seq_len)
        mel_pos_emb = self.mel_pos_embedding.emb(mel_positions)  # (T_mel+2, model_dim)
        mel_emb = mel_emb + mel_pos_emb[None, :, :]

        # Full input: [cond (34) || text || mel]
        full_input = mx.concatenate([cond_latents, text_emb, mel_emb], axis=1)
        attention_mask = mx.ones((B, full_input.shape[1]), dtype=mx.int32)

        # GPT2 forward
        hidden_states, _ = self.gpt(full_input, attention_mask, None)

        # Apply final_norm — matches PyTorch get_logits: enc = self.final_norm(enc)
        hidden_states = self.final_norm(hidden_states)

        # Extract mel hidden states (positions after cond + text, then strip last 2)
        # PyTorch: get_logits extracts mel portion, then returns mel_logits[:, :-2]
        n_cond = cond_latents.shape[1]
        mel_hidden = hidden_states[:, n_cond + T_text_wrapped:, :]  # (B, T_mel+2, model_dim)
        mel_hidden = mel_hidden[:, :-2, :]  # (B, T_mel, model_dim) - strip last 2

        return mel_hidden

    def generate(
        self,
        mel: mx.array,
        text_tokens: mx.array,
        mel_lengths: Optional[mx.array] = None,
        max_length: int = 250,
        temperature: float = 0.8,
        top_k: Optional[int] = None,
    ) -> mx.array:
        """Autoregressive generation of semantic codes.

        Args:
            mel: (B, T_mel, mel_channels) conditioning mel
            text_tokens: (B, T_text) text tokens
            mel_lengths: (B,) mel lengths
            max_length: maximum number of tokens to generate
            temperature: sampling temperature
            top_k: top-k sampling (if None, use full distribution)
        Returns:
            generated_codes: (B, T_generated) semantic code sequence
        """
        B = mel.shape[0]

        # Get conditioning
        cond_latents = self.get_conditioning(mel, mel_lengths)

        # Prepare inputs (conditioning + text)
        inputs_embeds, attention_mask = self.prepare_inputs(cond_latents, text_tokens)

        # Initialize generation with START_MEL_TOKEN
        generated = [[self.start_mel_token] for _ in range(B)]

        # KV cache for efficient generation
        cache = None
        use_cache = True  # Enable KV caching for efficiency

        for step_idx in range(max_length):
            # For the first step, use full inputs_embeds (conditioning + text) + START_MEL
            # For subsequent steps, only use the last generated token embedding
            if step_idx == 0 or not use_cache:
                # First step or no caching: use full sequence
                if step_idx == 0:
                    # CRITICAL: On first step, concatenate START_MEL_TOKEN embedding
                    # PyTorch does: emb = cat([cached_mel_emb (43), START_MEL_emb (1)]) = 44 tokens
                    start_mel_tokens = mx.array([[self.start_mel_token] for _ in range(B)])
                    start_mel_emb = self.mel_embedding(start_mel_tokens)  # (B, 1, model_dim)

                    # Add MEL positional embedding
                    # PyTorch's text_pos_embedding.forward() returns position 0 for sequence length 1
                    mel_pos_idx = 0
                    start_mel_pos = self.mel_pos_embedding.emb(mx.array([mel_pos_idx]))  # (1, model_dim)
                    start_mel_emb = start_mel_emb + start_mel_pos[None, :, :]

                    # Concatenate: [conditioning + text] (43) + START_MEL (1) = 44 tokens
                    current_input = mx.concatenate([inputs_embeds, start_mel_emb], axis=1)
                    current_mask = mx.ones((B, current_input.shape[1]), dtype=mx.int32)
                else:
                    # Without cache: reconstruct full sequence each time
                    mel_tokens = mx.array([generated[b] for b in range(B)])  # (B, len(generated))
                    mel_embs = self.mel_embedding(mel_tokens)  # (B, len(generated), model_dim)

                    # Add mel positional embeddings
                    for i in range(mel_embs.shape[1]):
                        if i < self.mel_pos_embedding.emb.weight.shape[0]:
                            pos_emb = self.mel_pos_embedding.emb(mx.array([i]))  # (1, model_dim)
                            mel_embs = mel_embs.at[:, i:i+1, :].add(pos_emb[None, :, :])

                    # Concatenate conditioning + text + mel
                    current_input = mx.concatenate([inputs_embeds, mel_embs], axis=1)
                    current_mask = mx.ones((B, current_input.shape[1]), dtype=mx.int32)
            else:
                # Subsequent steps with caching: embed last generated token
                last_tokens = mx.array([[generated[b][-1]] for b in range(B)])  # (B, 1)
                current_input = self.mel_embedding(last_tokens)  # (B, 1, model_dim)

                # Add mel positional embedding
                # PyTorch: position = attention_mask.shape[1] - mel_len
                # attention_mask grows by 1 each step from initial size (input_len+1),
                # so pos = (input_len+1+step) - input_len = 1+step, but START_MEL
                # was at position 0, and HF adds 1 col before the 2nd forward, giving:
                # tok1→pos2, tok2→pos3, tok3→pos4, ...
                # = len(generated[0]) matches this: [START, tok1]=2, [START,tok1,tok2]=3, ...
                pos_idx = len(generated[0])
                if pos_idx < self.mel_pos_embedding.emb.weight.shape[0]:
                    pos_emb = self.mel_pos_embedding.emb(mx.array([pos_idx]))  # (1, model_dim)
                    current_input = current_input + pos_emb[None, :, :]

                # For cached generation, don't pass attention mask
                # The model handles causal masking automatically
                current_mask = None

            # GPT2 forward with caching (or without if use_cache=False)
            if use_cache:
                hidden_states, cache = self.gpt(current_input, current_mask, cache)
            else:
                hidden_states, _ = self.gpt(current_input, current_mask, None)

            # Take last token's hidden state
            last_hidden = hidden_states[:, -1, :]  # (B, model_dim)

            # Final norm and output head
            last_hidden = self.final_norm(last_hidden)
            logits = self.mel_head(last_hidden)  # (B, number_mel_codes)

            # Apply temperature
            logits = logits / temperature

            # Top-k sampling (TODO: implement properly)
            if top_k is not None:
                # For now, just sample from full distribution
                # TODO: implement top-k filtering
                pass

            # Sample next token
            probs = mx.softmax(logits, axis=-1)

            # Debug first few iterations
            if len(generated[0]) <= 3:
                sorted_indices = mx.argsort(probs[0])[::-1][:5]
                sorted_probs = probs[0][sorted_indices]
                print(f"  Step {len(generated[0])}: top tokens = {sorted_indices.tolist()}, probs = {sorted_probs.tolist()}")
                print(f"    logits range: [{logits[0].min():.2f}, {logits[0].max():.2f}]")
                print(f"    hidden_states shape: {hidden_states.shape}, last_hidden range: [{last_hidden.min():.2f}, {last_hidden.max():.2f}]")
                if len(generated[0]) == 2:
                    print(f"    pos_idx for this token: {len(generated[0]) - 1}")
                    print(f"    current_input shape: {current_input.shape}")

            # Use multinomial sampling (not greedy!)
            # MLX's categorical samples from probs
            next_tokens = mx.random.categorical(mx.log(probs), axis=-1)  # (B,)

            # Add to generated sequences
            for b in range(B):
                next_token = int(next_tokens[b])
                generated[b].append(next_token)

                # Check for stop token
                if next_token == self.stop_mel_token:
                    break

            # If all sequences have stop token, break
            if all(gen[-1] == self.stop_mel_token for gen in generated):
                break

        # Convert to array
        max_gen_len = max(len(g) for g in generated)
        generated_array = mx.zeros((B, max_gen_len), dtype=mx.int32)
        for b in range(B):
            generated_array[b, :len(generated[b])] = mx.array(generated[b])

        return generated_array


def create_unifiedvoice(
    num_layers: int = 24,
    model_dim: int = 1280,
    num_heads: int = 20,  # GPT model has 20 heads, not 8!
    number_text_tokens: int = 12001,
    number_mel_codes: int = 8194,
) -> UnifiedVoice:
    """Create UnifiedVoice model with default parameters."""
    return UnifiedVoice(
        num_layers=num_layers,
        model_dim=model_dim,
        num_heads=num_heads,
        number_text_tokens=number_text_tokens,
        number_mel_codes=number_mel_codes,
    )
