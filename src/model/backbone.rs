// Copyright 2026 Claude Code on behalf of Michael Yuan.
// SPDX-License-Identifier: Apache-2.0

//! The 3.4B parameter Mistral decoder backbone.
//!
//! Autoregressive transformer that produces hidden states conditioned on
//! text tokens and voice reference embeddings.  At each audio frame position
//! the hidden state is passed to the flow-matching acoustic transformer.

use std::collections::HashMap;

use crate::config::VoxtralConfig;
use crate::tensor::{Device, Tensor};

use super::kv_cache::KVCache;
use super::layers::{RMSNorm, RotaryEmbedding, TransformerLayer};

// ---------------------------------------------------------------------------
// BackboneConfig
// ---------------------------------------------------------------------------

/// Configuration for the Mistral decoder backbone.
#[derive(Debug, Clone)]
pub struct BackboneConfig {
    pub dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub hidden_dim: usize,
    pub vocab_size: usize,
    pub rope_theta: f64,
    pub norm_eps: f64,
    pub max_seq_len: usize,
}

impl From<&VoxtralConfig> for BackboneConfig {
    fn from(cfg: &VoxtralConfig) -> Self {
        Self {
            dim: cfg.dim,
            n_layers: cfg.n_layers,
            n_heads: cfg.n_heads,
            n_kv_heads: cfg.n_kv_heads,
            head_dim: cfg.head_dim,
            hidden_dim: cfg.hidden_dim,
            vocab_size: cfg.vocab_size,
            rope_theta: cfg.rope_theta,
            norm_eps: cfg.norm_eps,
            max_seq_len: cfg.max_seq_len,
        }
    }
}

// ---------------------------------------------------------------------------
// Backbone
// ---------------------------------------------------------------------------

/// The Mistral decoder backbone (26 layers, 3.4B parameters).
pub struct Backbone {
    /// Token embedding table: `[vocab_size, dim]`.
    tok_embeddings: Tensor,
    /// Audio codebook embedding table: `[9088, dim]`.
    /// Contains embeddings for 37 codebooks (1 semantic + 36 acoustic) with offsets.
    audio_codebook_embeddings: Tensor,
    /// Cumulative offsets into the audio codebook embedding table.
    /// [0, 8194, 8217, 8240, ...] for 37 codebooks.
    codebook_offsets: Vec<i64>,
    /// Transformer decoder layers.
    layers: Vec<TransformerLayer>,
    /// Final RMSNorm before the output projection.
    norm: RMSNorm,
    /// Output projection (lm_head): `[vocab_size, dim]`.
    /// Tied with `tok_embeddings` when `tied_embeddings=true`.
    _output: Tensor,
    /// Shared rotary positional embedding.
    rotary_emb: RotaryEmbedding,
    /// Model configuration.
    config: BackboneConfig,
}

impl Backbone {
    /// Load the backbone from a weight map.
    ///
    /// Expected weight keys:
    /// * `mm_audio_embeddings.tok_embeddings.weight` – `[vocab_size, dim]`
    /// * `mm_audio_embeddings.audio_codebook_embeddings.embeddings.weight` – `[9088, dim]`
    /// * `layers.{i}.attention_norm.weight`
    /// * `layers.{i}.attention.{wq,wk,wv,wo}.weight`
    /// * `layers.{i}.ffn_norm.weight`
    /// * `layers.{i}.feed_forward.{w1,w2,w3}.weight`
    /// * `norm.weight` – final norm
    /// * `output.weight` – lm_head (optional; if absent, tied with tok_embeddings)
    pub fn from_weights(
        weights: &HashMap<String, Tensor>,
        config: BackboneConfig,
        device: Device,
    ) -> Self {
        let tok_embeddings = weights["mm_audio_embeddings.tok_embeddings.weight"]
            .clone()
            .to_device(device);

        let audio_codebook_embeddings = weights
            ["mm_audio_embeddings.audio_codebook_embeddings.embeddings.weight"]
            .clone()
            .to_device(device);

        // Build codebook offsets: semantic has 8192+2=8194 entries, each acoustic has 21+2=23
        let semantic_cb_size =
            (crate::SEMANTIC_CODEBOOK_SIZE + crate::NUM_AUDIO_SPECIAL_TOKENS) as i64;
        let acoustic_cb_size = (crate::FSQ_LEVELS + crate::NUM_AUDIO_SPECIAL_TOKENS) as i64;
        let mut codebook_offsets = Vec::with_capacity(crate::TOKENS_PER_FRAME);
        codebook_offsets.push(0); // semantic codebook at offset 0
        let mut offset = semantic_cb_size;
        for _ in 0..crate::ACOUSTIC_DIM {
            codebook_offsets.push(offset);
            offset += acoustic_cb_size;
        }

        let mut layers = Vec::with_capacity(config.n_layers);
        for i in 0..config.n_layers {
            let prefix = format!("layers.{}", i);
            let layer = TransformerLayer::from_weights(
                weights,
                &prefix,
                config.n_heads,
                config.n_kv_heads,
                config.head_dim,
                config.norm_eps,
            );
            layers.push(layer);
        }

        let norm = RMSNorm::from_weights(
            weights["norm.weight"].clone().to_device(device),
            config.norm_eps,
        );

        // Output projection; fall back to tied embeddings if key is absent.
        let output = weights
            .get("output.weight")
            .cloned()
            .unwrap_or_else(|| tok_embeddings.clone())
            .to_device(device);

        let rotary_emb = RotaryEmbedding::new(
            config.head_dim,
            config.max_seq_len,
            config.rope_theta,
            device,
        );

        Self {
            tok_embeddings,
            audio_codebook_embeddings,
            codebook_offsets,
            layers,
            norm,
            _output: output,
            rotary_emb,
            config,
        }
    }

    /// Prefill with pre-computed embeddings (e.g. text + voice).
    ///
    /// * `embeddings` – `[1, seq_len, dim]` pre-computed input embeddings.
    /// * `kv_cache` – KV cache to populate.
    ///
    /// Returns the hidden state at the last position: `[dim]`.
    pub fn forward_prefill_embeddings(
        &self,
        embeddings: &Tensor,
        kv_cache: &mut KVCache,
    ) -> Tensor {
        tracing::debug!("Prefill embeddings shape: {:?}", embeddings.size());
        let seq_len = embeddings.size()[1] as usize;
        let mut h = embeddings.clone();

        for (i, layer) in self.layers.iter().enumerate() {
            let (out, new_k, new_v) = layer.forward(&h, &self.rotary_emb, 0, kv_cache.get(i), true);
            kv_cache.update(i, new_k, new_v);
            h = out;
        }

        let h = self.norm.forward(&h);
        // Return last position hidden state: [1, seq_len, dim] → [dim]
        let last_pos = seq_len as i64 - 1;
        let out = h.select(1, last_pos).squeeze_dim(0);
        // Single eval materializes the entire 26-layer prefill graph at once,
        // allowing MLX to optimize the full computation on the GPU.
        out.eval();
        tracing::debug!("Backbone prefill done, output shape: {:?}", out.size());
        out
    }

    /// Forward one step with a pre-computed embedding vector.
    ///
    /// * `embedding` – `[dim]` input embedding for this step.
    /// * `kv_cache` – KV cache (updated in place).
    ///
    /// Returns the hidden state: `[dim]`.
    pub fn forward_one_embedding(&self, embedding: &Tensor, kv_cache: &mut KVCache) -> Tensor {
        let h = embedding.reshape(&[1, 1, self.config.dim as i64]);
        let pos = kv_cache.seq_len();
        let mut h = h;

        for (i, layer) in self.layers.iter().enumerate() {
            let (out, new_k, new_v) =
                layer.forward(&h, &self.rotary_emb, pos, kv_cache.get(i), true);
            kv_cache.update(i, new_k, new_v);
            h = out;
        }

        let out = self.norm.forward(&h).squeeze_dim(0).squeeze_dim(0); // [dim]
        // Single eval materializes the full 26-layer forward pass at once.
        // MLX lazy evaluation lets the GPU optimize the entire graph together
        // rather than synchronizing after each layer.
        out.eval();
        out
    }

    /// Embed a text token via the token embedding table.
    pub fn embed_text_token(&self, token_id: i64) -> Tensor {
        self.tok_embeddings.select(0, token_id)
    }

    /// Embed 37 audio codes into a single [dim] vector.
    ///
    /// Each code is looked up in the audio codebook embedding table at its
    /// codebook-specific offset, and all 37 embeddings are summed together.
    /// The text embedding for `audio_token_id` is also added (sum embedding).
    ///
    /// * `codes` – slice of 37 codes: [semantic_code, acoustic_code_0, ..., acoustic_code_35].
    pub fn embed_audio_codes(&self, codes: &[i64]) -> Tensor {
        // Start with the text embedding for the audio marker token
        let mut sum = self.tok_embeddings.select(0, crate::AUDIO_TOKEN_ID);

        // Add each codebook embedding
        for (i, &code) in codes.iter().enumerate() {
            let idx = self.codebook_offsets[i] + code;
            let emb = self.audio_codebook_embeddings.select(0, idx);
            sum = &sum + &emb;
        }

        sum
    }

    /// Create a new empty KV cache for this backbone.
    pub fn new_kv_cache(&self) -> KVCache {
        KVCache::new(self.config.n_layers)
    }

    /// Access the model configuration.
    pub fn config(&self) -> &BackboneConfig {
        &self.config
    }

    /// Get the model dimension.
    pub fn dim(&self) -> usize {
        self.config.dim
    }
}
