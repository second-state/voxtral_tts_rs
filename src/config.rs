//! Model configuration parsing from params.json.

use serde::Deserialize;
use std::path::Path;

use crate::error::{Result, VoxtralError};

/// Top-level Voxtral TTS configuration (params.json).
#[derive(Debug, Clone, Deserialize)]
pub struct VoxtralConfig {
    /// Model dimension.
    pub dim: usize,
    /// Number of transformer layers.
    pub n_layers: usize,
    /// Number of attention heads.
    pub n_heads: usize,
    /// Number of key-value heads (for GQA).
    pub n_kv_heads: usize,
    /// Head dimension.
    pub head_dim: usize,
    /// FFN hidden dimension.
    pub hidden_dim: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// RMSNorm epsilon.
    pub norm_eps: f64,
    /// RoPE theta.
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    /// Maximum sequence length.
    #[serde(default = "default_max_seq_len")]
    pub max_seq_len: usize,
    /// Multimodal configuration (codec + acoustic transformer).
    #[serde(default)]
    pub multimodal: Option<MultimodalConfig>,
}

fn default_rope_theta() -> f64 {
    1_000_000.0
}

fn default_max_seq_len() -> usize {
    65536
}

/// Multimodal configuration containing audio model args.
#[derive(Debug, Clone, Deserialize)]
pub struct MultimodalConfig {
    /// Audio tokenizer (codec) configuration.
    #[serde(default)]
    pub audio_tokenizer_args: Option<AudioTokenizerConfig>,
    /// Audio model configuration.
    #[serde(default)]
    pub audio_model_args: Option<AudioModelConfig>,
}

/// Audio tokenizer (Voxtral Codec) configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct AudioTokenizerConfig {
    /// Number of audio channels (1 = mono).
    #[serde(default = "default_channels")]
    pub channels: usize,
    /// Audio sampling rate.
    #[serde(default = "default_sampling_rate")]
    pub sampling_rate: usize,
    /// Patch size for pretransform.
    #[serde(default)]
    pub pretransform_patch_size: usize,
    /// Patch projection kernel size.
    #[serde(default)]
    pub patch_proj_kernel_size: usize,
    /// Semantic codebook size (VQ vocabulary).
    #[serde(default = "default_semantic_codebook_size")]
    pub semantic_codebook_size: usize,
    /// Semantic embedding dimension.
    #[serde(default)]
    pub semantic_dim: usize,
    /// Acoustic codebook size (FSQ levels per dimension).
    #[serde(default = "default_acoustic_codebook_size")]
    pub acoustic_codebook_size: usize,
    /// Acoustic dimensions (number of FSQ dimensions).
    #[serde(default = "default_acoustic_dim")]
    pub acoustic_dim: usize,
    /// Codec transformer dimension.
    #[serde(default)]
    pub dim: usize,
    /// Codec FFN hidden dimension.
    #[serde(default)]
    pub hidden_dim: usize,
    /// Codec head dimension.
    #[serde(default)]
    pub head_dim: usize,
    /// Codec number of heads.
    #[serde(default)]
    pub n_heads: usize,
    /// Codec number of KV heads.
    #[serde(default)]
    pub n_kv_heads: usize,
    /// Whether to use QK normalization.
    #[serde(default)]
    pub qk_norm: bool,
    /// QK norm epsilon.
    #[serde(default = "default_qk_norm_eps")]
    pub qk_norm_eps: f64,
    /// Whether to use layer scale.
    #[serde(default)]
    pub layer_scale: bool,
    /// Layer scale initial value.
    #[serde(default)]
    pub layer_scale_init: f64,
    /// Whether to use weight normalization on convolutions.
    #[serde(default)]
    pub conv_weight_norm: bool,
    /// Norm epsilon for attention_norm and ffn_norm (separate from qk_norm_eps).
    #[serde(default = "default_codec_norm_eps")]
    pub norm_eps: f64,
    /// Whether attention is causal.
    #[serde(default = "default_codec_causal")]
    pub causal: bool,
    /// Attention sliding window size.
    #[serde(default)]
    pub attn_sliding_window_size: usize,
    /// Decoder transformer lengths per block.
    #[serde(default)]
    pub decoder_transformer_lengths: Vec<usize>,
    /// Decoder convolution kernel sizes.
    #[serde(default)]
    pub decoder_conv_kernels: Vec<usize>,
    /// Decoder convolution strides (for upsampling).
    #[serde(default)]
    pub decoder_conv_strides: Vec<usize>,
    /// Encoder transformer lengths per block.
    #[serde(default)]
    pub encoder_transformer_lengths: Vec<usize>,
    /// Encoder convolution kernel sizes.
    #[serde(default)]
    pub encoder_conv_kernels: Vec<usize>,
    /// Encoder convolution strides (for downsampling).
    #[serde(default)]
    pub encoder_conv_strides: Vec<usize>,

    // Comma-separated string variants (params.json uses these)
    #[serde(default)]
    decoder_transformer_lengths_str: Option<String>,
    #[serde(default)]
    decoder_convs_kernels_str: Option<String>,
    #[serde(default)]
    decoder_convs_strides_str: Option<String>,
}

impl AudioTokenizerConfig {
    /// Resolve `_str` fields into the corresponding Vec fields.
    /// Call this after deserialization.
    pub fn resolve_str_fields(&mut self) {
        if self.decoder_transformer_lengths.is_empty() {
            if let Some(s) = &self.decoder_transformer_lengths_str {
                self.decoder_transformer_lengths = parse_csv_usize(s);
            }
        }
        if self.decoder_conv_kernels.is_empty() {
            if let Some(s) = &self.decoder_convs_kernels_str {
                self.decoder_conv_kernels = parse_csv_usize(s);
            }
        }
        if self.decoder_conv_strides.is_empty() {
            if let Some(s) = &self.decoder_convs_strides_str {
                self.decoder_conv_strides = parse_csv_usize(s);
            }
        }
    }
}

fn parse_csv_usize(s: &str) -> Vec<usize> {
    s.split(',')
        .filter_map(|v| v.trim().parse::<usize>().ok())
        .collect()
}

fn default_channels() -> usize {
    1
}

fn default_sampling_rate() -> usize {
    24000
}

fn default_semantic_codebook_size() -> usize {
    8192
}

fn default_acoustic_codebook_size() -> usize {
    21
}

fn default_acoustic_dim() -> usize {
    36
}

fn default_qk_norm_eps() -> f64 {
    1e-6
}

fn default_codec_norm_eps() -> f64 {
    0.01
}

fn default_codec_causal() -> bool {
    true
}

/// Audio model configuration (backbone extensions for audio).
#[derive(Debug, Clone, Deserialize)]
pub struct AudioModelConfig {
    /// Semantic codebook size for audio tokens.
    #[serde(default = "default_semantic_codebook_size")]
    pub semantic_codebook_size: usize,
    /// Acoustic transformer configuration.
    #[serde(default)]
    pub acoustic_transformer_args: Option<AcousticTransformerConfig>,
}

/// Acoustic (flow-matching) transformer configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct AcousticTransformerConfig {
    /// Input dimension (from backbone hidden states).
    #[serde(default)]
    pub input_dim: usize,
    /// Transformer dimension.
    #[serde(default)]
    pub dim: usize,
    /// Number of layers.
    #[serde(default = "default_acoustic_n_layers")]
    pub n_layers: usize,
    /// Head dimension.
    #[serde(default)]
    pub head_dim: usize,
    /// Number of attention heads.
    #[serde(default)]
    pub n_heads: usize,
    /// Number of KV heads.
    #[serde(default)]
    pub n_kv_heads: usize,
    /// FFN hidden dimension.
    #[serde(default)]
    pub hidden_dim: usize,
    /// Noise sigma for flow matching.
    #[serde(default = "default_sigma")]
    pub sigma: f64,
    /// Maximum sigma for flow matching.
    #[serde(default = "default_sigma_max")]
    pub sigma_max: f64,
    /// RoPE theta for acoustic transformer.
    #[serde(default = "default_acoustic_rope_theta")]
    pub rope_theta: f64,
}

fn default_acoustic_n_layers() -> usize {
    3
}

fn default_sigma() -> f64 {
    1e-5
}

fn default_sigma_max() -> f64 {
    1.0
}

fn default_acoustic_rope_theta() -> f64 {
    10000.0
}

impl VoxtralConfig {
    /// Load configuration from a params.json file.
    pub fn from_file(path: &Path) -> Result<Self> {
        let contents = std::fs::read_to_string(path).map_err(|e| {
            VoxtralError::Config(format!("Failed to read {}: {}", path.display(), e))
        })?;
        let config: Self = serde_json::from_str(&contents)
            .map_err(|e| VoxtralError::Config(format!("Failed to parse params.json: {}", e)))?;
        Ok(config)
    }

    /// Load configuration from a model directory containing params.json.
    pub fn from_dir(model_dir: &Path) -> Result<Self> {
        Self::from_file(&model_dir.join("params.json"))
    }

    /// Get the acoustic transformer config, or defaults.
    pub fn acoustic_transformer_config(&self) -> AcousticTransformerConfig {
        self.multimodal
            .as_ref()
            .and_then(|m| m.audio_model_args.as_ref())
            .and_then(|a| a.acoustic_transformer_args.clone())
            .unwrap_or(AcousticTransformerConfig {
                input_dim: self.dim,
                dim: self.dim,
                n_layers: 3,
                head_dim: self.head_dim,
                n_heads: self.n_heads,
                n_kv_heads: self.n_kv_heads,
                hidden_dim: self.hidden_dim,
                sigma: 1e-5,
                sigma_max: 1.0,
                rope_theta: 10000.0,
            })
    }

    /// Get the audio tokenizer config, or defaults.
    pub fn audio_tokenizer_config(&self) -> AudioTokenizerConfig {
        let mut config = self
            .multimodal
            .as_ref()
            .and_then(|m| m.audio_tokenizer_args.clone())
            .unwrap_or(AudioTokenizerConfig {
                channels: 1,
                sampling_rate: 24000,
                pretransform_patch_size: 240,
                patch_proj_kernel_size: 7,
                semantic_codebook_size: 8192,
                semantic_dim: 256,
                acoustic_codebook_size: 21,
                acoustic_dim: 36,
                dim: 1024,
                hidden_dim: 4096,
                head_dim: 128,
                n_heads: 8,
                n_kv_heads: 8,
                qk_norm: true,
                qk_norm_eps: 1e-6,
                layer_scale: true,
                layer_scale_init: 0.01,
                conv_weight_norm: true,
                norm_eps: 0.01,
                causal: true,
                attn_sliding_window_size: 16,
                decoder_transformer_lengths: vec![2, 2, 2, 2],
                decoder_conv_kernels: vec![3, 4, 4, 4],
                decoder_conv_strides: vec![1, 2, 2, 2],
                encoder_transformer_lengths: vec![2, 2, 2, 2],
                encoder_conv_kernels: vec![3, 4, 4, 4],
                encoder_conv_strides: vec![1, 2, 2, 2],
                decoder_transformer_lengths_str: None,
                decoder_convs_kernels_str: None,
                decoder_convs_strides_str: None,
            });
        config.resolve_str_fields();
        config
    }
}
