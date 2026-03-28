// Copyright 2026 Claude Code on behalf of Michael Yuan.
// SPDX-License-Identifier: Apache-2.0

//! Voxtral Codec decoder (300M parameters).
//!
//! Decodes discrete audio codes (1 semantic + 36 acoustic per frame) back to
//! 24kHz mono waveform.  The decoder alternates CausalConv1d and Transformer
//! blocks, with a final 240-channel output reshaped to raw audio samples.
//!
//! There is no encoder in the inference checkpoint — the backbone generates
//! audio codes directly.

use std::collections::HashMap;

use crate::config::AudioTokenizerConfig;
use crate::error::{Result, VoxtralError};
use crate::tensor::{DType, Device, Tensor};

use super::layers::{RMSNorm, RotaryEmbedding, TransformerLayer};

// ---------------------------------------------------------------------------
// Weight-normalized Conv1d
// ---------------------------------------------------------------------------

/// 1D convolution with optional weight normalization.
///
/// When weight-normalized, the effective weight is `g * v / ||v||` where:
/// - `g` = `parametrizations.weight.original0` (per-output-channel magnitude)
/// - `v` = `parametrizations.weight.original1` (direction)
/// Causal 1D convolution with weight normalization.
///
/// Left-pads the input with replicate padding, then applies Conv1d with no
/// built-in padding. This ensures the output at time t only depends on
/// inputs at times <= t.
struct CausalConv1d {
    weight: Tensor,
    stride: i64,
    /// Total left padding = (kernel_size - 1) * dilation + 1 - stride.
    left_pad: i64,
}

impl CausalConv1d {
    fn from_weight_norm(g: &Tensor, v: &Tensor, stride: i64, kernel_size: i64) -> Self {
        let v_f32 = v.to_dtype(DType::Float32);
        let g_f32 = g.to_dtype(DType::Float32);

        let v_norm = v_f32
            .pow_scalar(2.0)
            .sum_dim(&[-1, -2], true)
            .sqrt()
            .clamp_min(1e-12);
        let weight = &g_f32 * &(&v_f32 / &v_norm);

        // Causal: all padding on left side
        let left_pad = kernel_size - stride;

        Self {
            weight,
            stride,
            left_pad,
        }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        // x: [B, C, T]
        // Pad left with replicate (edge) padding: repeat first sample left_pad times
        let x = if self.left_pad > 0 {
            let first = x.narrow(2, 0, 1); // [B, C, 1]
            let pad = first.expand(&[-1, -1, self.left_pad], false); // [B, C, left_pad]
            Tensor::cat(&[pad, x.clone()], 2) // [B, C, left_pad + T]
        } else {
            x.clone()
        };
        x.conv1d(
            &self.weight,
            None,
            &[self.stride],
            &[0], // no built-in padding
            &[1], // dilation
            1,    // groups
        )
    }
}

/// Causal transposed 1D convolution with weight normalization (for upsampling).
///
/// Runs ConvTranspose1d with no padding, then trims excess samples from
/// the RIGHT side only (causal: output at time t depends on input at times <= t).
struct CausalConvTranspose1d {
    weight: Tensor,
    stride: i64,
    /// Number of samples to trim from the right of the raw output.
    right_trim: i64,
}

impl CausalConvTranspose1d {
    fn from_weight_norm(g: &Tensor, v: &Tensor, stride: i64, kernel_size: i64) -> Self {
        let v_f32 = v.to_dtype(DType::Float32);
        let g_f32 = g.to_dtype(DType::Float32);

        let v_norm = v_f32
            .pow_scalar(2.0)
            .sum_dim(&[-1, -2], true)
            .sqrt()
            .clamp_min(1e-12);
        let weight = &g_f32 * &(&v_f32 / &v_norm);

        // total_padding = kernel_size - stride; trim all from right (causal)
        let right_trim = kernel_size - stride;

        Self {
            weight,
            stride,
            right_trim,
        }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        // x: [B, C, T]
        // Run conv_transpose with NO padding
        let out = x.conv_transpose1d(
            &self.weight,
            None,
            &[self.stride],
            &[0], // no built-in padding
            &[0], // output_padding
            1,    // groups
            &[1], // dilation
        );
        // Trim right_trim samples from the right
        if self.right_trim > 0 {
            let out_len = out.size()[2];
            out.narrow(2, 0, out_len - self.right_trim)
        } else {
            out
        }
    }
}

// ---------------------------------------------------------------------------
// Codec transformer layer with LayerScale and QK norm
// ---------------------------------------------------------------------------

/// A codec transformer layer: pre-norm attention + LayerScale + pre-norm FFN + LayerScale.
///
/// Differences from backbone TransformerLayer:
/// - QK norm (RMSNorm on Q and K projections)
/// - Layer scale (per-channel learnable scale after attention and FFN)
struct CodecTransformerLayer {
    /// The base transformer layer (attention + FFN).
    base: TransformerLayer,
    /// Per-channel scale after attention output.
    attention_scale: Option<Tensor>,
    /// Per-channel scale after FFN output.
    ffn_scale: Option<Tensor>,
    /// Whether to use causal attention.
    causal: bool,
}

impl CodecTransformerLayer {
    fn from_weights(
        weights: &HashMap<String, Tensor>,
        prefix: &str,
        config: &AudioTokenizerConfig,
        sliding_window: usize,
    ) -> Self {
        // Use config.norm_eps (0.01) for attention_norm/ffn_norm, NOT qk_norm_eps (1e-6)
        let mut base = TransformerLayer::from_weights(
            weights,
            prefix,
            config.n_heads,
            config.n_kv_heads,
            config.head_dim,
            config.norm_eps,
        );

        // Load QK norms (codec transformer applies RMSNorm to Q and K projections)
        let q_norm_key = format!("{}.attention.q_norm.weight", prefix);
        let k_norm_key = format!("{}.attention.k_norm.weight", prefix);
        if let (Some(q_w), Some(k_w)) = (weights.get(&q_norm_key), weights.get(&k_norm_key)) {
            base.attention.set_qk_norms(
                RMSNorm::from_weights(q_w.clone(), config.qk_norm_eps),
                RMSNorm::from_weights(k_w.clone(), config.qk_norm_eps),
            );
            tracing::debug!("Loaded QK norms for {}", prefix);
        }

        // Set sliding window attention
        if sliding_window > 0 {
            base.attention.set_sliding_window(sliding_window);
            tracing::debug!("Set sliding window={} for {}", sliding_window, prefix);
        }

        let attention_scale = weights.get(&format!("{}.attention_scale", prefix)).cloned();
        let ffn_scale = weights.get(&format!("{}.ffn_scale", prefix)).cloned();

        Self {
            base,
            attention_scale,
            ffn_scale,
            causal: config.causal,
        }
    }

    fn forward(&self, x: &Tensor, rotary_emb: &RotaryEmbedding) -> Tensor {
        // Pre-norm attention with residual + layer scale
        let normed = self.base.attention_norm.forward(x);
        let seq_len = x.size()[1] as usize;
        let (attn_out, _k, _v) =
            self.base
                .attention
                .forward(&normed, rotary_emb, 0, None, self.causal && seq_len > 1);
        let attn_out = match &self.attention_scale {
            Some(scale) => &attn_out * scale,
            None => attn_out,
        };
        let h = x + &attn_out;

        // Pre-norm FFN with residual + layer scale
        let normed = self.base.ffn_norm.forward(&h);
        let ffn_out = self.base.feed_forward.forward(&normed);
        let ffn_out = match &self.ffn_scale {
            Some(scale) => &ffn_out * scale,
            None => ffn_out,
        };
        &h + &ffn_out
    }
}

// ---------------------------------------------------------------------------
// Codec
// ---------------------------------------------------------------------------

/// Decoder-only codec for reconstructing waveforms from audio codes.
pub struct Codec {
    /// Semantic codebook: `[8192, 256]`.
    semantic_codebook: Tensor,
    /// Decoder conv blocks (even indices) and transformer blocks (odd indices).
    /// Block 0: CausalConv1d (input projection, stride 1)
    /// Block 1: 2 transformer layers
    /// Block 2: CausalConvTranspose1d (upsampling)
    /// Block 3: 2 transformer layers
    /// ...
    decoder_convs: Vec<DecoderConv>,
    decoder_transformers: Vec<DecoderTransformerBlock>,
    /// Output projection: dim → 240 channels (causal conv, kernel=7).
    output_proj: CausalConv1d,
    /// Configuration.
    config: AudioTokenizerConfig,
    /// RoPE for codec transformer layers.
    #[allow(dead_code)]
    rotary_emb: RotaryEmbedding,
}

enum DecoderConv {
    Conv1d(CausalConv1d),
    ConvTranspose1d(CausalConvTranspose1d),
}

struct DecoderTransformerBlock {
    layers: Vec<CodecTransformerLayer>,
    rotary_emb: RotaryEmbedding,
}

impl Codec {
    /// Load the codec from a weight map.
    ///
    /// All weights are prefixed with `audio_tokenizer.`.
    pub fn from_weights(
        weights: &HashMap<String, Tensor>,
        config: AudioTokenizerConfig,
        device: Device,
    ) -> Self {
        let prefix = "audio_tokenizer";

        // Compute semantic codebook from EMA buffers:
        // codebook = embedding_sum / cluster_usage
        let semantic_codebook = {
            let embedding_sum = weights
                .get(&format!(
                    "{}.quantizer.semantic_codebook.embedding_sum",
                    prefix
                ))
                .cloned()
                .unwrap_or_else(|| {
                    tracing::warn!("Semantic codebook embedding_sum not found");
                    Tensor::zeros(
                        &[
                            config.semantic_codebook_size as i64,
                            config.semantic_dim as i64,
                        ],
                        DType::Float32,
                        device,
                    )
                })
                .to_dtype(DType::Float32)
                .to_device(device);

            let cluster_usage = weights
                .get(&format!(
                    "{}.quantizer.semantic_codebook.cluster_usage",
                    prefix
                ))
                .cloned()
                .unwrap_or_else(|| {
                    tracing::warn!("Semantic codebook cluster_usage not found");
                    Tensor::ones(
                        &[config.semantic_codebook_size as i64],
                        DType::Float32,
                        device,
                    )
                })
                .to_dtype(DType::Float32)
                .to_device(device);

            // codebook[i] = embedding_sum[i] / max(cluster_usage[i], 1e-5)
            let usage_clamped = cluster_usage.clamp_min(1e-5).unsqueeze(-1); // [8192, 1]
            &embedding_sum / &usage_clamped
        };

        tracing::debug!("Semantic codebook computed: {:?}", semantic_codebook.size());

        // Build decoder blocks
        // The decoder has alternating conv and transformer blocks:
        // Block 0: Conv1d (input projection)
        // Block 1: Transformer layers
        // Block 2: ConvTranspose1d (upsample)
        // Block 3: Transformer layers
        // ... etc.
        // Based on config: decoder_conv_strides = [1, 2, 2, 2]
        // decoder_conv_kernels = [3, 4, 4, 4]
        // decoder_transformer_lengths = [2, 2, 2, 2]

        let n_conv_blocks = config.decoder_conv_strides.len();
        let mut decoder_convs = Vec::new();
        let mut decoder_transformers = Vec::new();

        // Sliding window sizes: base_window / 2^(n_blocks-1-i)
        // e.g., with base=16 and 4 blocks: [2, 4, 8, 16]
        let base_window = config.attn_sliding_window_size;

        for i in 0..n_conv_blocks {
            let conv_block_idx = i * 2; // even indices are conv blocks
            let transformer_block_idx = i * 2 + 1; // odd indices are transformer blocks

            let stride = config.decoder_conv_strides[i] as i64;
            let kernel = config.decoder_conv_kernels[i] as i64;

            // Window doubles per decoder stage: 2, 4, 8, 16
            let window = base_window >> (n_conv_blocks - 1 - i);

            // Load conv block (weight-normalized, causal)
            let g_key = format!(
                "{}.decoder_blocks.{}.conv.parametrizations.weight.original0",
                prefix, conv_block_idx
            );
            let v_key = format!(
                "{}.decoder_blocks.{}.conv.parametrizations.weight.original1",
                prefix, conv_block_idx
            );

            if let (Some(g), Some(v)) = (weights.get(&g_key), weights.get(&v_key)) {
                let g = g.to_dtype(DType::Float32).to_device(device);
                let v = v.to_dtype(DType::Float32).to_device(device);

                if stride == 1 {
                    decoder_convs.push(DecoderConv::Conv1d(
                        CausalConv1d::from_weight_norm(&g, &v, stride, kernel),
                    ));
                } else {
                    decoder_convs.push(DecoderConv::ConvTranspose1d(
                        CausalConvTranspose1d::from_weight_norm(&g, &v, stride, kernel),
                    ));
                }
            } else {
                tracing::warn!(
                    "Decoder conv block {} not found, using identity",
                    conv_block_idx
                );
                let w = Tensor::zeros(
                    &[config.dim as i64, config.dim as i64, 1],
                    DType::Float32,
                    device,
                );
                decoder_convs.push(DecoderConv::Conv1d(CausalConv1d {
                    weight: w,
                    stride: 1,
                    left_pad: 0,
                }));
            }

            // Load transformer block
            let n_layers = config.decoder_transformer_lengths[i];
            let rotary_emb = RotaryEmbedding::new(config.head_dim, 8192, 10000.0, device);

            let mut layers = Vec::with_capacity(n_layers);
            for j in 0..n_layers {
                let layer_prefix = format!(
                    "{}.decoder_blocks.{}.layers.{}",
                    prefix, transformer_block_idx, j
                );
                let layer =
                    CodecTransformerLayer::from_weights(weights, &layer_prefix, &config, window);
                layers.push(layer);
            }

            decoder_transformers.push(DecoderTransformerBlock { layers, rotary_emb });
        }

        // Output projection conv
        let output_g_key = format!(
            "{}.output_proj.conv.parametrizations.weight.original0",
            prefix
        );
        let output_v_key = format!(
            "{}.output_proj.conv.parametrizations.weight.original1",
            prefix
        );

        let output_proj =
            if let (Some(g), Some(v)) = (weights.get(&output_g_key), weights.get(&output_v_key)) {
                let g = g.to_dtype(DType::Float32).to_device(device);
                let v = v.to_dtype(DType::Float32).to_device(device);
                CausalConv1d::from_weight_norm(&g, &v, 1, 7) // kernel=7, causal left-pad=6
            } else {
                tracing::warn!("Output projection not found, using zeros");
                CausalConv1d {
                    weight: Tensor::zeros(
                        &[crate::PRETRANSFORM_PATCH_SIZE as i64, config.dim as i64, 7],
                        DType::Float32,
                        device,
                    ),
                    stride: 1,
                    left_pad: 6,
                }
            };

        let rotary_emb = RotaryEmbedding::new(config.head_dim, 8192, 10000.0, device);

        Self {
            semantic_codebook,
            decoder_convs,
            decoder_transformers,
            output_proj,
            config,
            rotary_emb,
        }
    }

    /// Decode audio codes to waveform samples.
    ///
    /// * `codes` – `Vec<Vec<i64>>` where each inner vec is 37 codes for one frame:
    ///   [semantic_code, acoustic_code_0, ..., acoustic_code_35].
    /// * `device` – target device.
    ///
    /// Returns mono 24kHz audio samples in `[-1, 1]`.
    pub fn decode(&self, codes: &[Vec<i64>], device: Device) -> Result<Vec<f32>> {
        let n_frames = codes.len();
        if n_frames == 0 {
            return Err(VoxtralError::Inference("No codes to decode".to_string()));
        }

        tracing::debug!("Decoding {} frames", n_frames);

        // Dequantize codes to continuous latents: [B, 292, T]
        let latent = self.dequantize_codes(codes, device);

        // Run decoder
        let waveform = self.run_decoder(&latent);

        // Output: [1, 240, T'] → reshape to [1, 1, T'*240] → flatten
        let shape = waveform.size();
        let n_channels = shape[1]; // 240
        let time_steps = shape[2];
        let total_samples = n_channels * time_steps;

        // Reshape: [1, 240, T'] → [1, 1, T' * 240]
        // Using rearrange pattern: b (c h) t -> b c (t h) where h=patch_size
        let patch_size = crate::PRETRANSFORM_PATCH_SIZE as i64;
        let out = waveform
            .reshape(&[1, 1, patch_size, time_steps])
            .transpose(2, 3)
            .reshape(&[1, 1, total_samples]);

        let samples = out.reshape(&[-1]).to_vec_f32();
        let samples: Vec<f32> = samples.iter().map(|&s| s.clamp(-1.0, 1.0)).collect();

        Ok(samples)
    }

    /// Dequantize 37 codes per frame into continuous latents [1, 292, T].
    fn dequantize_codes(&self, codes: &[Vec<i64>], device: Device) -> Tensor {
        let n_frames = codes.len();
        let semantic_dim = self.config.semantic_dim;
        let acoustic_dim = self.config.acoustic_dim;
        let total_dim = semantic_dim + acoustic_dim; // 256 + 36 = 292

        let mut latent_data = vec![0.0f32; n_frames * total_dim];

        for (frame_idx, frame_codes) in codes.iter().enumerate() {
            // Semantic: look up codebook embedding
            // The semantic code includes AudioSpecialTokens offset, but the actual
            // codebook index needs the offset removed (codes in [2, 8193] → index [0, 8191])
            let sem_code = frame_codes[0];
            // Clamp to valid range and remove special token offset
            let sem_idx = (sem_code - crate::NUM_AUDIO_SPECIAL_TOKENS as i64)
                .clamp(0, self.config.semantic_codebook_size as i64 - 1);
            let sem_emb = self.semantic_codebook.select(0, sem_idx);
            let sem_vals = sem_emb.to_vec_f32();

            for (d, &val) in sem_vals.iter().enumerate().take(semantic_dim) {
                latent_data[frame_idx * total_dim + d] = val;
            }

            // Acoustic: FSQ dequantize
            // Each acoustic code is in [2, 22] (with +2 offset), actual level = code - 2
            for (d, &code) in frame_codes[1..].iter().enumerate().take(acoustic_dim) {
                let level = (code - crate::NUM_AUDIO_SPECIAL_TOKENS as i64)
                    .clamp(0, crate::FSQ_LEVELS as i64 - 1);
                // Map [0, 20] → [-1, 1]
                let value = level as f32 * 2.0 / (crate::FSQ_LEVELS as f32 - 1.0) - 1.0;
                latent_data[frame_idx * total_dim + semantic_dim + d] = value;
            }
        }

        // Shape: [1, total_dim, n_frames] (conv format: batch, channels, time)
        Tensor::from_slice_f32(&latent_data)
            .reshape(&[n_frames as i64, total_dim as i64])
            .transpose(0, 1) // [total_dim, n_frames]
            .unsqueeze(0) // [1, total_dim, n_frames]
            .to_device(device)
    }

    /// Run the decoder network.
    fn run_decoder(&self, x: &Tensor) -> Tensor {
        // x: [B, 292, T]
        tracing::debug!(
            "Decoder input: {:?}, {} conv blocks, {} transformer blocks",
            x.size(),
            self.decoder_convs.len(),
            self.decoder_transformers.len()
        );

        let mut h = x.transpose(1, 2); // [B, T, D=292]

        for (i, (conv, transformer_block)) in self
            .decoder_convs
            .iter()
            .zip(self.decoder_transformers.iter())
            .enumerate()
        {
            // Conv: [B, T, D] → [B, D, T] → conv → [B, D', T'] → [B, T', D']
            let h_conv = h.transpose(1, 2); // [B, D, T]
            let h_conv = match conv {
                DecoderConv::Conv1d(c) => c.forward(&h_conv),
                DecoderConv::ConvTranspose1d(c) => c.forward(&h_conv),
            };
            h = h_conv.transpose(1, 2); // [B, T', D']

            // Transformer layers
            for layer in &transformer_block.layers {
                h = layer.forward(&h, &transformer_block.rotary_emb);
            }
            tracing::debug!("Decoder block {} done: shape={:?}", i, h.size());
        }

        // Output projection: [B, T, D] → [B, D, T] → conv → [B, 240, T']
        let h = h.transpose(1, 2); // [B, D, T]
        let out = self.output_proj.forward(&h);
        // Single eval materializes the entire decoder (4 conv+transformer blocks + output proj).
        out.eval();
        out
    }

    /// Encode a voice embedding directly (voice embeddings bypass the codec encoder).
    /// This is a pass-through since voice embeddings are already in backbone space.
    pub fn encode_voice_embedding(&self, _embedding: &Tensor) -> Result<Vec<Tensor>> {
        // Voice embeddings are used directly by the backbone; they don't go through
        // the codec. Return empty to indicate the backbone should use them as-is.
        Ok(Vec::new())
    }

    /// Access the configuration.
    pub fn config(&self) -> &AudioTokenizerConfig {
        &self.config
    }
}
