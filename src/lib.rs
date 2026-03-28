//! # Voxtral TTS - Rust Port
//!
//! A Rust implementation of the Voxtral 4B Text-to-Speech model from Mistral AI.
//!
//! This crate provides a high-level API for generating speech from text using
//! the Voxtral TTS model, with support for:
//!
//! - **Preset voices**: 20 built-in voices across 9 languages
//! - **Voice cloning**: Clone any voice from 3-30 seconds of reference audio
//! - **Streaming**: SSE-based streaming audio generation
//!
//! ## Backends
//!
//! - `tch-backend` (default): Uses libtorch for Linux/CUDA inference
//! - `mlx`: Uses Apple MLX for macOS Metal inference

// Ensure exactly one backend is selected
#[cfg(all(feature = "tch-backend", feature = "mlx"))]
compile_error!("Features 'tch-backend' and 'mlx' are mutually exclusive");

#[cfg(not(any(feature = "tch-backend", feature = "mlx")))]
compile_error!("Either 'tch-backend' or 'mlx' feature must be enabled");

#[cfg(feature = "mlx")]
pub mod backend;
pub mod tensor;

pub mod audio;
pub mod config;
pub mod error;
pub mod inference;
pub mod model;
pub mod text;
pub mod tokenizer;
pub mod voice;

// Re-export main types
pub use config::VoxtralConfig;
pub use error::{Result, VoxtralError};
pub use tensor::{DType, Device, Tensor};

/// Default output sample rate in Hz.
pub const DEFAULT_SAMPLE_RATE: u32 = 24000;

/// Audio frame rate in Hz (12.5 Hz = 80ms per frame).
pub const FRAME_RATE: f32 = 12.5;

/// Number of tokens per audio frame (1 semantic + 36 acoustic).
pub const TOKENS_PER_FRAME: usize = 37;

/// Semantic codebook size.
pub const SEMANTIC_CODEBOOK_SIZE: usize = 8192;

/// Number of acoustic dimensions (FSQ).
pub const ACOUSTIC_DIM: usize = 36;

/// Number of FSQ quantization levels per acoustic dimension.
pub const FSQ_LEVELS: usize = 21;

/// Number of special audio tokens (padding, end-of-audio).
pub const NUM_AUDIO_SPECIAL_TOKENS: usize = 2;

/// Token ID in the text vocabulary that marks an audio frame position.
pub const AUDIO_TOKEN_ID: i64 = 24;

/// Token ID for begin-of-audio.
pub const BEGIN_AUDIO_TOKEN_ID: i64 = 25;

/// Empty-audio token ID in the semantic codebook (index 0, never predicted).
pub const EMPTY_AUDIO_TOKEN_ID: i64 = 0;

/// End-of-audio token ID in the semantic codebook (index 1).
pub const END_AUDIO_TOKEN_ID: i64 = 1;

/// BOS token ID in the text vocabulary.
pub const BOS_TOKEN_ID: i64 = 1;

/// [REPEAT_AUDIO_TEXT] token ID — signals the model to repeat (generate) audio for the text.
/// Placed between text tokens and the final [BEGIN_AUDIO].
pub const REPEAT_AUDIO_TEXT_TOKEN_ID: i64 = 35;

/// [NEXT_AUDIO_TEXT] token ID — separates voice reference from the text to synthesize.
/// Placed between the voice embeddings and the text tokens.
pub const NEXT_AUDIO_TEXT_TOKEN_ID: i64 = 36;

/// Pretransform patch size (audio samples per frame output).
pub const PRETRANSFORM_PATCH_SIZE: usize = 240;
