# Voxtral TTS Rust -- Implementation Notes

Development notes, architecture decisions, and lessons learned during the port of Voxtral-4B-TTS from Python to Rust.

## Build Commands

```bash
# macOS (MLX)
git submodule update --init --recursive
cargo build --release --no-default-features --features mlx

# Linux (libtorch)
export LIBTORCH=$(pwd)/libtorch
cargo build --release

# Run tests
cargo test

# Run with debug logging
RUST_LOG=debug ./target/release/voxtral-tts models/voxtral-4b-tts --text "Hello." --voice neutral_female --output output.wav
```

## Weight Key Naming

The safetensors checkpoint uses Mistral-style weight naming, **not** HuggingFace convention. This is the most common source of load errors.

### Backbone (237 keys)

| Weight | Key pattern |
|--------|-------------|
| Token embeddings | `mm_audio_embeddings.tok_embeddings.weight` [131072, 3072] |
| Audio codebook embeddings | `mm_audio_embeddings.audio_codebook_embeddings.embeddings.weight` [9088, 3072] |
| Layer attention Q/K/V/O | `layers.{i}.attention.wq.weight`, `.wk.weight`, `.wv.weight`, `.wo.weight` |
| Layer FFN gate/down/up | `layers.{i}.feed_forward.w1.weight`, `.w2.weight`, `.w3.weight` |
| Layer norms | `layers.{i}.attention_norm.weight`, `layers.{i}.ffn_norm.weight` |
| Final norm | `norm.weight` |
| LM head | Absent -- tied to `tok_embeddings.weight` |

### Flow-Matching Transformer (33 keys)

Prefix: `acoustic_transformer.` (NOT `multimodal.acoustic_transformer.`)

| Weight | Key |
|--------|-----|
| Input projection | `acoustic_transformer.input_projection.weight` [3072, 36] |
| LLM projection | `acoustic_transformer.llm_projection.weight` [3072, 3072] |
| Time projection | `acoustic_transformer.time_projection.weight` [3072, 3072] |
| Semantic output | `acoustic_transformer.semantic_codebook_output.weight` [8320, 3072] |
| Acoustic output | `acoustic_transformer.acoustic_codebook_output.weight` [36, 3072] |
| Layers | `acoustic_transformer.layers.{i}.attention.{wq,wk,wv,wo}.weight` |

### Codec Decoder (116 keys)

Prefix: `audio_tokenizer.` (NOT `multimodal.audio_tokenizer.`)

Convolutions use weight normalization with `parametrizations.weight.original0` (g, magnitude [out, 1, 1]) and `parametrizations.weight.original1` (v, direction [out, in, kernel]).

| Weight | Key pattern |
|--------|-------------|
| Decoder conv blocks | `audio_tokenizer.decoder_blocks.{i}.conv.parametrizations.weight.original{0,1}` |
| Decoder transformer layers | `audio_tokenizer.decoder_blocks.{i}.layers.{j}.attention.{wq,wk,wv,wo}.weight` |
| Layer scale | `audio_tokenizer.decoder_blocks.{i}.layers.{j}.attention_scale`, `.ffn_scale` |
| QK norm | `audio_tokenizer.decoder_blocks.{i}.layers.{j}.attention.q_norm.weight`, `.k_norm.weight` |
| Semantic codebook (EMA) | `audio_tokenizer.quantizer.semantic_codebook.embedding_sum` [8192, 256] |
| Cluster usage (EMA) | `audio_tokenizer.quantizer.semantic_codebook.cluster_usage` [8192] |
| Output projection | `audio_tokenizer.output_proj.conv.parametrizations.weight.original{0,1}` |

Decoder block layout (8 blocks total):
- Even indices (0, 2, 4, 6): Conv blocks
- Odd indices (1, 3, 5, 7): Transformer blocks (2 layers each)

## Config Parsing Pitfall

`params.json` stores decoder config as **comma-separated strings**, not JSON arrays:

```json
{
  "decoder_convs_strides_str": "1,2,2,2",
  "decoder_convs_kernels_str": "3,4,4,4",
  "decoder_transformer_lengths_str": "2,2,2,2"
}
```

The code must parse these with `resolve_str_fields()` after serde deserialization. Without this, the codec decoder loads with 0 blocks and the dequantized latent passes straight through to the output projection, causing a channel mismatch.

## MLX Backend -- Critical Lessons

### 1. Lazy Evaluation — eval() at Outer Loop Boundaries Only

MLX builds a computation graph lazily. The graph must be evaluated periodically to prevent unbounded growth, but **over-evaluating kills performance**. Per the MLX documentation and mlx-lm reference implementation, eval() should be called at outer loop boundaries, not per-layer.

**Where eval() is required (current optimized placement):**

- After the full 26-layer backbone forward pass in `forward_one_embedding()` — 1 eval per frame
- After the full 26-layer backbone forward pass in `forward_prefill_embeddings()` — 1 eval total
- After each Euler step in `decode_acoustic()` (flow matching ODE) — 7 evals per frame
- After the full codec decoder (all 4 blocks) in `run_decoder()` — 1 eval per decode

**Where eval() is NOT needed:**
- Per transformer layer (the graph for 26 layers is fine — "thousands of ops" per eval is OK)
- Per flow matching `predict_velocity()` call (only 3 layers on 3 tokens — tiny graph)
- Per codec conv/transformer block

**Symptom of too few eval():** Graph grows across iterations, causing exponential slowdowns.
**Symptom of too many eval():** Each eval() has fixed overhead for graph traversal, scheduling, and GPU synchronization. Reducing from ~130 to ~8 eval() calls per frame improved flow matching from 0.53s to 0.28s per frame.

### 2. KV Cache Must Replace, Not Concatenate

The attention layer already concatenates old cache + new K/V internally before returning. The KV cache `update()` method must **replace** the stored tensors, not concatenate again:

```rust
// CORRECT: simple replacement (attention already did the cat)
pub fn update(&mut self, layer_idx: usize, new_k: Tensor, new_v: Tensor) {
    self.k_cache[layer_idx] = Some(new_k);
    self.v_cache[layer_idx] = Some(new_v);
}

// WRONG: double-concatenation causes exponential cache growth
// 221 -> 443 -> 886 -> 1773 -> ...
```

**Symptom:** Matmul shape errors with large unexpected dimensions (e.g., 113664 instead of 222).

### 3. PyTorch vs MLX Tensor Format

| Operation | PyTorch | MLX |
|-----------|---------|-----|
| Conv1d input | `[N, C, L]` (NCL) | `[N, L, C]` (NLC) |
| Conv1d weight | `[C_out, C_in, K]` | `[C_out, K, C_in]` |
| ConvTranspose1d weight | `[C_in, C_out, K]` | `[C_out, K, C_in]` |

The `tensor.rs` conv methods handle these transposes automatically. The weights from safetensors are in PyTorch format and get transposed before calling MLX ops.

### 4. MLX Initialization

`Device::best_available()` must call `init_mlx(true)` before any MLX operations. Without this, all MLX calls panic with "MLX not initialized".

## Inference Pipeline Details

### Prefill Sequence Construction

```
[text_token_0, text_token_1, ..., begin_audio_token, voice_emb_0, voice_emb_1, ..., voice_emb_N]
```

- Text tokens: looked up in `tok_embeddings` [131072, 3072]
- Begin-audio token (ID 25): looked up in `tok_embeddings`
- Voice embeddings: pre-computed backbone hidden states [N, 3072], injected directly

### Per-Frame Audio Code Generation

1. **Semantic code**: `semantic_codebook_output.forward(hidden_state)` -> argmax over [8320] logits
   - Codes 0-1 are special (EOS, padding), valid range is [2, 8194)
2. **36 acoustic codes**: Euler ODE flow matching
   - Initialize `x_0 ~ N(0, 1)` with shape [1, 36]
   - 7 Euler steps from t=0 to t=1
   - Each step: build batched 3-token sequence `[acoustic_proj, time_emb, llm_proj]` with batch=2 (cond + uncond), run 3 bidirectional transformer layers
   - Classifier-free guidance: `v = 1.2 * v_cond - 0.2 * v_uncond` (batched CFG: both passes in single forward)
   - Quantize output to FSQ levels: map [-1,1] to [0,20], add +2 offset for special tokens

### Audio Codebook Embeddings

The backbone has a single [9088, 3072] codebook embedding table for 37 codebooks:
- Codebook 0 (semantic): 8192 + 2 special = 8194 entries, offset 0
- Codebooks 1-36 (acoustic): 21 + 2 special = 23 entries each

Each frame's 37 embeddings are summed together with the text embedding for `audio_token_id` (24).

### Codec Decoder

- Input: dequantized codes [1, 292, T] where 292 = 256 (semantic) + 36 (acoustic)
- Block 0: Conv1d [292 -> 1024], stride 1, kernel 3
- Blocks 1-3: ConvTranspose1d [1024 -> 1024], strides [2, 2, 2], kernels [4, 4, 4]
- Each conv is followed by 2 transformer layers
- Output projection: Conv1d [1024 -> 240], kernel 7
- Final reshape: [1, 240, T'] -> [1, 1, T' * 240] (patch_size=240)
- Total upsampling: T * 1 * 2 * 2 * 2 = 8T frames -> 8T * 240 = 1920T samples at 24kHz

Semantic dequantization: look up code in EMA codebook (`embedding_sum / cluster_usage`) -> [256] vector.
Acoustic dequantization: FSQ decode level [0,20] -> value in [-1, 1].

### Weight Normalization (Codec Convolutions)

Effective weight = `g * v / ||v||` where:
- `g` = `parametrizations.weight.original0` (magnitude, shape [out_ch, 1, 1])
- `v` = `parametrizations.weight.original1` (direction, shape [out_ch, in_ch, kernel])
- `||v||` = L2 norm over (in_ch, kernel) dimensions

## Voice Embeddings

Voice embeddings are pre-computed backbone hidden states. Each voice is a tensor of shape [N, 3072] where N varies by voice (typically 100-300 frames = 8-24 seconds of reference).

The original checkpoint stores these as PyTorch `.pt` files. For the MLX backend, they must be converted to `.safetensors` format (key: `embedding`).

## Codec QK Norm and Layer Scale

The codec transformer layers have two features not present in the backbone:

1. **QK Norm**: RMSNorm applied separately to Q and K projections before attention (weights: `q_norm.weight`, `k_norm.weight`). Currently loaded but not applied (the standard attention path is used).

2. **Layer Scale**: Learnable per-channel scales applied after attention and FFN outputs (`attention_scale`, `ffn_scale`). Currently loaded but approximated (scales are close to 1.0 after training).

Both are areas for improvement if audio quality needs to be refined.

## Performance Notes

On Apple M4 Max (MLX backend):
- Model loading: ~0.1s (memory-mapped safetensors)
- Prefill (221 tokens, 26 layers): ~3.3s
- Per-frame generation: ~0.34s (backbone ~70ms + flow matching ~270ms)
- Codec decoding: ~0.2s
- Total for "Hello." (110 frames): ~41s (~2.9 frames/s, RTF ~4.7)

### Optimizations Applied (2.5x total speedup)

1. **Fused MLX ops**: SDPA (`fast_scaled_dot_product_attention`), RMSNorm (`fast_rms_norm`), and RoPE (`fast_rope`) use fused Metal kernels. SDPA handles GQA natively (no `repeat_kv` expansion needed). RMSNorm replaces 6 discrete ops with 1 kernel.

2. **Batched CFG**: The flow matching CFG conditional and unconditional passes are batched together (batch=2) into a single forward pass through the 3 transformer layers, halving Metal kernel dispatch overhead.

3. **Reduced eval() frequency**: From ~130 eval() calls per frame to ~2 (backbone + flow matching). The backbone does a single eval() after all 26 layers, and the flow matching does one eval() after all 7 Euler steps.

4. **BF16 flow matching**: Random noise, zeros, and time embeddings are cast to BF16 to match weight dtype, avoiding implicit F32 promotion.

5. **Pre-computed time projections**: The 7 sinusoidal time step embeddings (constant across all frames) are pre-computed at model init.

6. **SDPA mask NaN fix**: Causal attention mask uses `-1e9` instead of `-inf` to avoid `0 * -inf = NaN` in IEEE 754.

### Remaining Bottleneck

Flow matching is 80% of per-frame time (270ms vs 70ms backbone). Each frame requires 7 Euler steps × 3 transformer layers (batch=2, seq=3). The small matrix sizes [6, 3072] cannot saturate the GPU efficiently. The backbone is relatively efficient at ~2.6ms/layer for single-token decode with KV cache.
