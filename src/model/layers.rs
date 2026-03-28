// Copyright 2026 Claude Code on behalf of Michael Yuan.
// SPDX-License-Identifier: Apache-2.0

//! Neural network layer primitives for the Mistral-family transformer.
//!
//! All layers follow Mistral weight naming conventions and use the unified
//! `Tensor` abstraction so they work on both tch (libtorch) and MLX backends.

use std::collections::HashMap;

use crate::tensor::{DType, Device, Tensor};

// ---------------------------------------------------------------------------
// RMSNorm
// ---------------------------------------------------------------------------

/// Root Mean Square Layer Normalization (no bias, no centering).
///
/// Computes:  `x * weight / sqrt(mean(x^2) + eps)`
pub struct RMSNorm {
    weight: Tensor,
    eps: f64,
}

impl RMSNorm {
    /// Build from a pre-loaded weight tensor and epsilon.
    pub fn from_weights(weight: Tensor, eps: f64) -> Self {
        Self { weight, eps }
    }

    /// Apply RMS normalization.
    pub fn forward(&self, x: &Tensor) -> Tensor {
        #[cfg(feature = "mlx")]
        {
            // Use MLX fused RMSNorm kernel — single GPU dispatch instead of 6 ops
            Tensor::from_mlx(crate::backend::mlx::ops::fast_rms_norm(
                x.as_mlx(),
                self.weight.as_mlx(),
                self.eps as f32,
            ))
        }
        #[cfg(not(feature = "mlx"))]
        {
            // x: [..., dim]
            let x_f32 = x.to_dtype(DType::Float32);
            let variance = x_f32.pow_scalar(2.0).mean_dim(&[-1], true);
            let normed = &x_f32 / &((&variance + self.eps).sqrt());
            let out = &normed * &self.weight.to_dtype(DType::Float32);
            out.to_dtype(x.kind())
        }
    }
}

// ---------------------------------------------------------------------------
// Linear
// ---------------------------------------------------------------------------

/// Standard dense (fully connected) layer: `y = x @ W^T + bias`.
pub struct Linear {
    weight: Tensor,
    bias: Option<Tensor>,
}

impl Linear {
    /// Build from a weight matrix (no bias).
    pub fn from_weights(weight: Tensor) -> Self {
        Self { weight, bias: None }
    }

    /// Build from a weight matrix and bias vector.
    pub fn from_weights_with_bias(weight: Tensor, bias: Tensor) -> Self {
        Self {
            weight,
            bias: Some(bias),
        }
    }

    /// Forward pass: `x @ W^T [+ bias]`.
    ///
    /// `x` can be any shape `[..., in_features]`; the result is `[..., out_features]`.
    pub fn forward(&self, x: &Tensor) -> Tensor {
        // x @ weight^T
        let out = x.matmul(&self.weight.transpose(-2, -1));
        match &self.bias {
            Some(b) => &out + b,
            None => out,
        }
    }
}

// ---------------------------------------------------------------------------
// RotaryEmbedding (RoPE)
// ---------------------------------------------------------------------------

/// Rotary Positional Embedding.
///
/// Pre-computes sin/cos tables up to `max_seq_len` and applies rotary
/// transformations to query and key tensors.
pub struct RotaryEmbedding {
    /// Cosine component: `[max_seq_len, dim/2]` (used by tch backend)
    #[allow(dead_code)]
    cos_cached: Tensor,
    /// Sine component: `[max_seq_len, dim/2]` (used by tch backend)
    #[allow(dead_code)]
    sin_cached: Tensor,
    /// Head dimension (for fused RoPE).
    dim: usize,
    /// Base frequency (for fused RoPE).
    theta: f64,
}

impl RotaryEmbedding {
    /// Create a new RoPE instance.
    ///
    /// * `dim` – head dimension (per-head, not total).
    /// * `max_seq_len` – maximum sequence length to pre-compute.
    /// * `theta` – RoPE base frequency (1_000_000 for Mistral).
    /// * `device` – target device.
    pub fn new(dim: usize, max_seq_len: usize, theta: f64, device: Device) -> Self {
        // freq_i = 1 / theta^(2i / dim)  for i in 0..dim/2
        let half_dim = dim / 2;
        let inv_freq: Vec<f32> = (0..half_dim)
            .map(|i| 1.0 / (theta as f32).powf(2.0 * i as f32 / dim as f32))
            .collect();
        let inv_freq_t = Tensor::from_slice_f32(&inv_freq)
            .to_device(device)
            .reshape(&[1, half_dim as i64]); // [1, dim/2]

        // positions = [0, 1, ..., max_seq_len-1]
        let positions = Tensor::arange(0, max_seq_len as i64, device)
            .to_dtype(DType::Float32)
            .reshape(&[max_seq_len as i64, 1]); // [seq, 1]

        // outer product: [seq, dim/2]
        let freqs = positions.matmul(&inv_freq_t);

        let cos_cached = freqs.cos();
        let sin_cached = freqs.sin();

        Self {
            cos_cached,
            sin_cached,
            dim,
            theta,
        }
    }

    /// Apply rotary embedding to query and key tensors.
    ///
    /// * `q` – `[batch, n_heads, seq_len, head_dim]`
    /// * `k` – `[batch, n_kv_heads, seq_len, head_dim]`
    /// * `seq_len` – actual sequence length (for slicing cached tables).
    ///
    /// Returns `(q_rotated, k_rotated)` with the same shapes.
    pub fn forward(&self, q: &Tensor, k: &Tensor, _seq_len: usize) -> (Tensor, Tensor) {
        #[cfg(feature = "mlx")]
        {
            // Use MLX fused RoPE kernel — single GPU dispatch
            // traditional=true: split-half format (d, d+dim/2) matching Llama/Mistral convention
            let q_rot = Tensor::from_mlx(crate::backend::mlx::ops::fast_rope(
                q.as_mlx(), self.dim as i32, true, Some(self.theta as f32), 1.0, 0,
            ));
            let k_rot = Tensor::from_mlx(crate::backend::mlx::ops::fast_rope(
                k.as_mlx(), self.dim as i32, true, Some(self.theta as f32), 1.0, 0,
            ));
            (q_rot, k_rot)
        }
        #[cfg(not(feature = "mlx"))]
        {
            let cos = self.cos_cached.narrow(0, 0, _seq_len as i64);
            let sin = self.sin_cached.narrow(0, 0, _seq_len as i64);
            let q_rot = apply_rotary_emb(q, &cos, &sin);
            let k_rot = apply_rotary_emb(k, &cos, &sin);
            (q_rot, k_rot)
        }
    }

    /// Apply rotary embedding starting from a specific position offset.
    ///
    /// Used during autoregressive decoding where only one new token is
    /// processed at a time and the position is the current cache length.
    pub fn forward_at_pos(
        &self,
        q: &Tensor,
        k: &Tensor,
        pos: usize,
        _seq_len: usize,
    ) -> (Tensor, Tensor) {
        #[cfg(feature = "mlx")]
        {
            // Use MLX fused RoPE kernel with offset — single GPU dispatch
            // traditional=true: split-half format (d, d+dim/2) matching Llama/Mistral convention
            let q_rot = Tensor::from_mlx(crate::backend::mlx::ops::fast_rope(
                q.as_mlx(), self.dim as i32, true, Some(self.theta as f32), 1.0, pos as i32,
            ));
            let k_rot = Tensor::from_mlx(crate::backend::mlx::ops::fast_rope(
                k.as_mlx(), self.dim as i32, true, Some(self.theta as f32), 1.0, pos as i32,
            ));
            (q_rot, k_rot)
        }
        #[cfg(not(feature = "mlx"))]
        {
            let cos = self.cos_cached.narrow(0, pos as i64, _seq_len as i64);
            let sin = self.sin_cached.narrow(0, pos as i64, _seq_len as i64);
            let q_rot = apply_rotary_emb(q, &cos, &sin);
            let k_rot = apply_rotary_emb(k, &cos, &sin);
            (q_rot, k_rot)
        }
    }
}

/// Apply the rotary embedding to a single tensor (tch backend only).
///
/// `x`: `[batch, n_heads, seq_len, head_dim]`
/// `cos`, `sin`: `[seq_len, head_dim/2]`
#[cfg(not(feature = "mlx"))]
fn apply_rotary_emb(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Tensor {
    let shape = x.size(); // [B, H, S, D]
    let head_dim = *shape.last().unwrap();
    let half = head_dim / 2;

    // Split into first half and second half
    let x1 = x.narrow(-1, 0, half); // [..., :half]
    let x2 = x.narrow(-1, half, half); // [..., half:]

    // cos/sin: [seq_len, half] -> [1, 1, seq_len, half] for broadcasting
    let cos = cos.unsqueeze(0).unsqueeze(0);
    let sin = sin.unsqueeze(0).unsqueeze(0);

    // rotary: [x1*cos - x2*sin, x1*sin + x2*cos]
    let out1 = &(&x1 * &cos) - &(&x2 * &sin);
    let out2 = &(&x1 * &sin) + &(&x2 * &cos);

    Tensor::cat(&[out1, out2], -1)
}

// ---------------------------------------------------------------------------
// Attention (Grouped Query Attention)
// ---------------------------------------------------------------------------

/// Grouped Query Attention (GQA) as used in Mistral models.
///
/// Supports different numbers of query heads (`n_heads`) and key-value heads
/// (`n_kv_heads`).  When `n_kv_heads < n_heads`, each KV head is shared
/// across `n_heads / n_kv_heads` query heads.
pub struct Attention {
    wq: Linear,
    wk: Linear,
    wv: Linear,
    wo: Linear,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    q_norm: Option<RMSNorm>,
    k_norm: Option<RMSNorm>,
    sliding_window: Option<usize>,
}

impl Attention {
    /// Load attention weights from a weight map using Mistral naming:
    ///
    /// * `{prefix}.wq.weight`
    /// * `{prefix}.wk.weight`
    /// * `{prefix}.wv.weight`
    /// * `{prefix}.wo.weight`
    pub fn from_weights(
        weights: &HashMap<String, Tensor>,
        prefix: &str,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> Self {
        let wq = Linear::from_weights(weights[&format!("{}.wq.weight", prefix)].clone());
        let wk = Linear::from_weights(weights[&format!("{}.wk.weight", prefix)].clone());
        let wv = Linear::from_weights(weights[&format!("{}.wv.weight", prefix)].clone());
        let wo = Linear::from_weights(weights[&format!("{}.wo.weight", prefix)].clone());
        Self {
            wq,
            wk,
            wv,
            wo,
            n_heads,
            n_kv_heads,
            head_dim,
            q_norm: None,
            k_norm: None,
            sliding_window: None,
        }
    }

    /// Set optional QK norms (used by codec transformer layers).
    pub fn set_qk_norms(&mut self, q_norm: RMSNorm, k_norm: RMSNorm) {
        self.q_norm = Some(q_norm);
        self.k_norm = Some(k_norm);
    }

    /// Set sliding window attention size (used by codec transformer layers).
    pub fn set_sliding_window(&mut self, size: usize) {
        self.sliding_window = Some(size);
    }

    /// Forward pass with optional KV cache for autoregressive decoding.
    ///
    /// * `x` – `[batch, seq_len, dim]`
    /// * `rotary_emb` – rotary embedding to apply to Q and K.
    /// * `pos` – starting position in the sequence (used with KV cache).
    /// * `kv_cache` – optional `(cached_k, cached_v)` from prior steps.
    /// * `causal` – whether to apply a causal attention mask.
    ///
    /// Returns `(output, new_k, new_v)` where `output` is `[batch, seq_len, dim]`.
    pub fn forward(
        &self,
        x: &Tensor,
        rotary_emb: &RotaryEmbedding,
        pos: usize,
        kv_cache: Option<(&Tensor, &Tensor)>,
        causal: bool,
    ) -> (Tensor, Tensor, Tensor) {
        let shape = x.size(); // [B, S, D]
        let batch = shape[0];
        let seq_len = shape[1] as usize;

        tracing::trace!(
            "Attention input: {:?}, heads={}, kv_heads={}, head_dim={}",
            shape,
            self.n_heads,
            self.n_kv_heads,
            self.head_dim
        );

        // Project Q, K, V
        let q = self.wq.forward(x); // [B, S, n_heads * head_dim]
        let k = self.wk.forward(x); // [B, S, n_kv_heads * head_dim]
        let v = self.wv.forward(x); // [B, S, n_kv_heads * head_dim]

        // Apply QK norm if present (used by codec transformer)
        // Applied before reshape, on [B, S, dim] where dim = n_heads * head_dim
        let q = match &self.q_norm {
            Some(norm) => norm.forward(&q),
            None => q,
        };
        let k = match &self.k_norm {
            Some(norm) => norm.forward(&k),
            None => k,
        };

        // Reshape to multi-head format: [B, S, nH, D] -> [B, nH, S, D]
        let q = q
            .reshape(&[
                batch,
                seq_len as i64,
                self.n_heads as i64,
                self.head_dim as i64,
            ])
            .transpose(1, 2);
        let k = k
            .reshape(&[
                batch,
                seq_len as i64,
                self.n_kv_heads as i64,
                self.head_dim as i64,
            ])
            .transpose(1, 2);
        let v = v
            .reshape(&[
                batch,
                seq_len as i64,
                self.n_kv_heads as i64,
                self.head_dim as i64,
            ])
            .transpose(1, 2);

        // Apply rotary positional embedding
        let (q, k) = rotary_emb.forward_at_pos(&q, &k, pos, seq_len);

        // Concatenate with cached KV if present
        let (k, v) = if let Some((ck, cv)) = kv_cache {
            let k = Tensor::cat(&[ck.clone(), k], 2); // cat along seq dim
            let v = Tensor::cat(&[cv.clone(), v], 2);
            (k, v)
        } else {
            (k, v)
        };

        // Save new KV for cache before expansion
        let new_k = k.clone();
        let new_v = v.clone();

        // Scaled dot-product attention
        let context = {
            #[cfg(feature = "mlx")]
            {
                // Use MLX fused SDPA kernel — handles GQA natively, no repeat_kv needed
                let scale = 1.0 / (self.head_dim as f32).sqrt();
                let mask = if causal && seq_len > 1 {
                    // Build additive causal mask: 0 for attend, -inf for mask.
                    // Cannot use ones().triu() * -inf because 0 * -inf = NaN in IEEE 754.
                    // Instead, use a large finite negative value.
                    let kv_seq_len = k.size()[2];
                    let mut mask = Tensor::ones(&[seq_len as i64, kv_seq_len], DType::Float32, x.device())
                        .triu(kv_seq_len - seq_len as i64 + 1)
                        * (-1e9);
                    // Add sliding window mask: also mask positions > window_size in the past
                    if let Some(window) = self.sliding_window {
                        let window_mask = Tensor::ones(&[seq_len as i64, kv_seq_len], DType::Float32, x.device())
                            .tril(-(window as i64))
                            * (-1e9);
                        mask = &mask + &window_mask;
                    }
                    Some(mask.to_dtype(q.kind()))
                } else if let Some(window) = self.sliding_window {
                    // Sliding window without causal (bidirectional with limited range)
                    let kv_seq_len = k.size()[2];
                    let above = Tensor::ones(&[seq_len as i64, kv_seq_len], DType::Float32, x.device())
                        .triu(window as i64)
                        * (-1e9);
                    let below = Tensor::ones(&[seq_len as i64, kv_seq_len], DType::Float32, x.device())
                        .tril(-(window as i64))
                        * (-1e9);
                    Some((&above + &below).to_dtype(q.kind()))
                } else {
                    None
                };
                Tensor::from_mlx(crate::backend::mlx::ops::fast_scaled_dot_product_attention(
                    q.as_mlx(),
                    k.as_mlx(),
                    v.as_mlx(),
                    scale,
                    mask.as_ref().map(|m| m.as_mlx()),
                ))
            }
            #[cfg(not(feature = "mlx"))]
            {
                // Manual attention for tch backend
                let (k, v) = if self.n_kv_heads < self.n_heads {
                    let n_rep = self.n_heads / self.n_kv_heads;
                    (repeat_kv(&k, n_rep), repeat_kv(&v, n_rep))
                } else {
                    (k, v)
                };

                // Make all tensors contiguous for matmul:
                // - Q: non-contiguous after reshape+transpose(1,2)
                // - K, V: non-contiguous after GQA repeat_kv (expand+reshape)
                let q = q.contiguous();
                let k = k.contiguous();
                let v = v.contiguous();

                let kv_seq_len = k.size()[2];
                let scale = 1.0 / (self.head_dim as f64).sqrt();
                let scores = &q.matmul(&k.transpose(-2, -1).contiguous()) * scale;

                let scores = if causal && seq_len > 1 {
                    let mut mask = Tensor::ones(&[seq_len as i64, kv_seq_len], DType::Bool, x.device())
                        .triu(kv_seq_len - seq_len as i64 + 1);
                    if let Some(window) = self.sliding_window {
                        let window_mask = Tensor::ones(&[seq_len as i64, kv_seq_len], DType::Bool, x.device())
                            .tril(-(window as i64));
                        mask = mask.logical_or(&window_mask);
                    }
                    scores.masked_fill(&mask, f64::NEG_INFINITY)
                } else {
                    scores
                };

                let attn = scores.softmax(-1);
                attn.matmul(&v)
            }
        }; // [B, nH, S, D]

        // Reshape back: [B, nH, S, D] -> [B, S, nH*D]
        let context = context.transpose(1, 2).contiguous().reshape(&[
            batch,
            seq_len as i64,
            (self.n_heads * self.head_dim) as i64,
        ]);

        // Output projection
        let output = self.wo.forward(&context);

        (output, new_k, new_v)
    }

    /// Forward pass without rotary positional embedding.
    ///
    /// Used by the flow-matching acoustic transformer which uses bidirectional
    /// attention without any positional encoding.
    pub fn forward_no_rope(
        &self,
        x: &Tensor,
        causal: bool,
    ) -> Tensor {
        let shape = x.size(); // [B, S, D]
        let batch = shape[0];
        let seq_len = shape[1] as usize;

        // Project Q, K, V
        let q = self.wq.forward(x);
        let k = self.wk.forward(x);
        let v = self.wv.forward(x);

        // Reshape to multi-head format: [B, S, nH, D] -> [B, nH, S, D]
        let q = q
            .reshape(&[batch, seq_len as i64, self.n_heads as i64, self.head_dim as i64])
            .transpose(1, 2);
        let k = k
            .reshape(&[batch, seq_len as i64, self.n_kv_heads as i64, self.head_dim as i64])
            .transpose(1, 2);
        let v = v
            .reshape(&[batch, seq_len as i64, self.n_kv_heads as i64, self.head_dim as i64])
            .transpose(1, 2);

        // NO RoPE applied — bidirectional attention without positional encoding

        // Scaled dot-product attention
        let context = {
            #[cfg(feature = "mlx")]
            {
                let scale = 1.0 / (self.head_dim as f32).sqrt();
                let mask = if causal && seq_len > 1 {
                    let kv_seq_len = k.size()[2];
                    let mask = Tensor::ones(&[seq_len as i64, kv_seq_len], DType::Float32, x.device())
                        .triu(kv_seq_len - seq_len as i64 + 1)
                        * (-1e9);
                    Some(mask.to_dtype(q.kind()))
                } else {
                    None
                };
                Tensor::from_mlx(crate::backend::mlx::ops::fast_scaled_dot_product_attention(
                    q.as_mlx(),
                    k.as_mlx(),
                    v.as_mlx(),
                    scale,
                    mask.as_ref().map(|m| m.as_mlx()),
                ))
            }
            #[cfg(not(feature = "mlx"))]
            {
                let (k, v) = if self.n_kv_heads < self.n_heads {
                    let n_rep = self.n_heads / self.n_kv_heads;
                    (repeat_kv(&k, n_rep), repeat_kv(&v, n_rep))
                } else {
                    (k, v)
                };

                let q = q.contiguous();
                let k = k.contiguous();
                let v = v.contiguous();

                let scale = 1.0 / (self.head_dim as f64).sqrt();
                let scores = &q.matmul(&k.transpose(-2, -1).contiguous()) * scale;

                let scores = if causal && seq_len > 1 {
                    let kv_seq_len = k.size()[2];
                    let mask = Tensor::ones(&[seq_len as i64, kv_seq_len], DType::Bool, x.device())
                        .triu(kv_seq_len - seq_len as i64 + 1);
                    scores.masked_fill(&mask, f64::NEG_INFINITY)
                } else {
                    scores
                };

                let attn = scores.softmax(-1);
                attn.matmul(&v)
            }
        };

        // Reshape back: [B, nH, S, D] -> [B, S, nH*D]
        let context = context.transpose(1, 2).contiguous().reshape(&[
            batch,
            seq_len as i64,
            (self.n_heads * self.head_dim) as i64,
        ]);

        self.wo.forward(&context)
    }
}

/// Repeat KV heads to match the number of query heads for GQA.
///
/// `x`: `[B, n_kv_heads, S, D]` -> `[B, n_heads, S, D]`
#[cfg(not(feature = "mlx"))]
fn repeat_kv(x: &Tensor, n_rep: usize) -> Tensor {
    if n_rep == 1 {
        return x.clone();
    }
    let shape = x.size(); // [B, n_kv_heads, S, D]
    let batch = shape[0];
    let n_kv = shape[1];
    let seq = shape[2];
    let dim = shape[3];

    // [B, n_kv, S, D] -> [B, n_kv, 1, S, D] -> expand [B, n_kv, n_rep, S, D] -> reshape
    x.unsqueeze(2)
        .expand(&[batch, n_kv, n_rep as i64, seq, dim], false)
        .reshape(&[batch, n_kv * n_rep as i64, seq, dim])
}

// ---------------------------------------------------------------------------
// MLP (SwiGLU)
// ---------------------------------------------------------------------------

/// SwiGLU Feed-Forward Network as used in Mistral/Llama models.
///
/// Computes: `down( silu(gate(x)) * up(x) )`
pub struct MLP {
    /// Gate projection (w1).
    gate: Linear,
    /// Down projection (w2).
    down: Linear,
    /// Up projection (w3).
    up: Linear,
}

impl MLP {
    /// Load MLP weights using Mistral naming:
    ///
    /// * `{prefix}.w1.weight` – gate projection
    /// * `{prefix}.w2.weight` – down projection
    /// * `{prefix}.w3.weight` – up projection
    pub fn from_weights(weights: &HashMap<String, Tensor>, prefix: &str) -> Self {
        let gate = Linear::from_weights(weights[&format!("{}.w1.weight", prefix)].clone());
        let down = Linear::from_weights(weights[&format!("{}.w2.weight", prefix)].clone());
        let up = Linear::from_weights(weights[&format!("{}.w3.weight", prefix)].clone());
        Self { gate, down, up }
    }

    /// Forward pass: `down( silu(gate(x)) * up(x) )`.
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let gate_out = self.gate.forward(x).silu();
        let up_out = self.up.forward(x);
        self.down.forward(&(&gate_out * &up_out))
    }
}

// ---------------------------------------------------------------------------
// TransformerLayer
// ---------------------------------------------------------------------------

/// A single transformer decoder layer with pre-norm, GQA attention, and SwiGLU MLP.
///
/// Architecture:
/// ```text
/// x -> RMSNorm -> Attention -> + (residual) -> RMSNorm -> MLP -> + (residual)
/// ```
pub struct TransformerLayer {
    pub(crate) attention_norm: RMSNorm,
    pub(crate) attention: Attention,
    pub(crate) ffn_norm: RMSNorm,
    pub(crate) feed_forward: MLP,
}

impl TransformerLayer {
    /// Load a transformer layer using Mistral naming:
    ///
    /// * `{prefix}.attention_norm.weight`
    /// * `{prefix}.ffn_norm.weight`
    /// * `{prefix}.attention.{wq,wk,wv,wo}.weight`
    /// * `{prefix}.feed_forward.{w1,w2,w3}.weight`
    pub fn from_weights(
        weights: &HashMap<String, Tensor>,
        prefix: &str,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        norm_eps: f64,
    ) -> Self {
        let attention_norm = RMSNorm::from_weights(
            weights[&format!("{}.attention_norm.weight", prefix)].clone(),
            norm_eps,
        );
        let ffn_norm = RMSNorm::from_weights(
            weights[&format!("{}.ffn_norm.weight", prefix)].clone(),
            norm_eps,
        );
        let attention = Attention::from_weights(
            weights,
            &format!("{}.attention", prefix),
            n_heads,
            n_kv_heads,
            head_dim,
        );
        let feed_forward = MLP::from_weights(weights, &format!("{}.feed_forward", prefix));

        Self {
            attention_norm,
            attention,
            ffn_norm,
            feed_forward,
        }
    }

    /// Forward pass through the transformer layer.
    ///
    /// * `x` – `[batch, seq_len, dim]`
    /// * `rotary_emb` – shared rotary embedding instance.
    /// * `pos` – starting position for RoPE (used during autoregressive decoding).
    /// * `kv_cache` – optional `(cached_k, cached_v)` from a previous step.
    /// * `causal` – whether to apply a causal attention mask.
    ///
    /// Returns `(output, new_k, new_v)`.
    pub fn forward(
        &self,
        x: &Tensor,
        rotary_emb: &RotaryEmbedding,
        pos: usize,
        kv_cache: Option<(&Tensor, &Tensor)>,
        causal: bool,
    ) -> (Tensor, Tensor, Tensor) {
        // Pre-norm attention with residual
        let normed = self.attention_norm.forward(x);
        let (attn_out, new_k, new_v) = self
            .attention
            .forward(&normed, rotary_emb, pos, kv_cache, causal);
        let h = x + &attn_out;

        // Pre-norm FFN with residual
        let normed = self.ffn_norm.forward(&h);
        let ffn_out = self.feed_forward.forward(&normed);
        let out = &h + &ffn_out;

        (out, new_k, new_v)
    }

    /// Forward pass without rotary positional embedding.
    ///
    /// Used by the flow-matching acoustic transformer which uses bidirectional
    /// attention without any positional encoding (no RoPE, no KV cache).
    pub fn forward_no_rope(&self, x: &Tensor, causal: bool) -> Tensor {
        // Pre-norm attention with residual (no RoPE)
        let normed = self.attention_norm.forward(x);
        let attn_out = self.attention.forward_no_rope(&normed, causal);
        let h = x + &attn_out;

        // Pre-norm FFN with residual
        let normed = self.ffn_norm.forward(&h);
        let ffn_out = self.feed_forward.forward(&normed);
        &h + &ffn_out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() {
        #[cfg(feature = "mlx")]
        crate::backend::mlx::stream::init_mlx(false);
    }

    #[test]
    fn test_rms_norm_shape() {
        setup();
        let dim = 64;
        let weight = Tensor::ones(&[dim], DType::Float32, Device::Cpu);
        let norm = RMSNorm::from_weights(weight, 1e-5);
        let x = Tensor::ones(&[1, 4, dim], DType::Float32, Device::Cpu);
        let out = norm.forward(&x);
        assert_eq!(out.size(), vec![1, 4, dim]);
    }

    #[test]
    fn test_linear_shape() {
        setup();
        let weight = Tensor::ones(&[32, 64], DType::Float32, Device::Cpu);
        let linear = Linear::from_weights(weight);
        let x = Tensor::ones(&[1, 4, 64], DType::Float32, Device::Cpu);
        let out = linear.forward(&x);
        assert_eq!(out.size(), vec![1, 4, 32]);
    }
}
