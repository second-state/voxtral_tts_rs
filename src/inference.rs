//! High-level TTS inference pipeline.
//!
//! Orchestrates the full text-to-speech pipeline:
//! 1. Tokenize input text via Tekken tokenizer
//! 2. Inject voice reference embeddings into the input sequence
//! 3. Run backbone + flow-matching per frame to produce audio codes
//! 4. Decode audio codes to waveform via codec decoder

use std::path::Path;

use crate::config::VoxtralConfig;
use crate::error::{Result, VoxtralError};
use crate::model::backbone::Backbone;
use crate::model::codec::Codec;
use crate::model::flow_matching::FlowMatchingTransformer;
use crate::model::weights::load_model_weights;
use crate::tensor::{Device, Tensor};
use crate::tokenizer::TekkenTokenizer;
use crate::voice::VoiceStore;

/// The complete Voxtral TTS inference engine.
pub struct VoxtralTTS {
    config: VoxtralConfig,
    backbone: Backbone,
    flow_matching: FlowMatchingTransformer,
    codec: Codec,
    tokenizer: TekkenTokenizer,
    voices: VoiceStore,
    device: Device,
}

impl VoxtralTTS {
    /// Load the full model from a directory.
    pub fn from_dir(model_dir: &Path, device: Device) -> Result<Self> {
        tracing::info!("Loading Voxtral TTS from {}", model_dir.display());

        // Load configuration
        let config = VoxtralConfig::from_dir(model_dir)?;
        tracing::info!(
            "Config: dim={}, layers={}, heads={}, vocab={}",
            config.dim,
            config.n_layers,
            config.n_heads,
            config.vocab_size,
        );

        // Load tokenizer (cap to model vocab_size to avoid OOB on tok_embeddings)
        let tokenizer = TekkenTokenizer::from_dir(model_dir, Some(config.vocab_size))?;
        tracing::info!("Tokenizer loaded ({} tokens)", tokenizer.vocab_size());

        // Load voice embeddings
        let voices = VoiceStore::from_dir(model_dir, device)?;

        // Load model weights
        let (backbone, flow_matching, codec) = load_model_weights(model_dir, &config, device)?;
        tracing::info!("Model weights loaded");

        Ok(Self {
            config,
            backbone,
            flow_matching,
            codec,
            tokenizer,
            voices,
            device,
        })
    }

    /// Generate speech from text using a preset voice.
    ///
    /// Returns (waveform_samples, sample_rate).
    pub fn generate(
        &self,
        text: &str,
        voice: &str,
        _temperature: f32,
        max_tokens: usize,
    ) -> Result<(Vec<f32>, u32)> {
        if text.is_empty() {
            return Err(VoxtralError::Inference("Input text is empty".to_string()));
        }

        // Get voice embedding: [N, dim]
        let voice_embedding = self.voices.get(voice)?;
        let voice_shape = voice_embedding.size();
        let n_voice_frames = voice_shape[0] as usize;
        tracing::debug!(
            "Voice embedding: {} frames x {} dim",
            n_voice_frames,
            voice_shape[1]
        );

        // Tokenize text
        let text_tokens: Vec<u32> = self.tokenizer.encode(text);
        tracing::debug!("Text tokens: {} tokens", text_tokens.len());

        // Build input embedding sequence:
        // [text_token_embeddings..., begin_audio_token, voice_embeddings...]
        let total_prefix_len = text_tokens.len() + 1 + n_voice_frames; // text + begin_audio + voice
        let mut embeddings_data = Vec::with_capacity(total_prefix_len);

        // Text token embeddings
        for &token_id in &text_tokens {
            let emb = self.backbone.embed_text_token(token_id as i64);
            embeddings_data.push(emb);
        }

        // Begin-audio token embedding
        let begin_audio_emb = self.backbone.embed_text_token(crate::BEGIN_AUDIO_TOKEN_ID);
        embeddings_data.push(begin_audio_emb);

        // Voice embeddings (pre-computed, directly injected)
        for i in 0..n_voice_frames {
            let frame_emb = voice_embedding.select(0, i as i64); // [dim]
            embeddings_data.push(frame_emb);
        }

        // Stack into [1, total_prefix_len, dim]
        let prefix_embeddings = Tensor::stack(&embeddings_data, 0) // [total_prefix_len, dim]
            .unsqueeze(0); // [1, total_prefix_len, dim]

        tracing::debug!(
            "Prefix: {} tokens (text={}, voice={})",
            total_prefix_len,
            text_tokens.len() + 1,
            n_voice_frames
        );

        // Prefill backbone
        let mut kv_cache = self.backbone.new_kv_cache();
        let prefill_start = std::time::Instant::now();
        let mut hidden_state = self
            .backbone
            .forward_prefill_embeddings(&prefix_embeddings, &mut kv_cache);
        tracing::info!("Prefill: {:.2}s (seq_len={})", prefill_start.elapsed().as_secs_f64(), kv_cache.seq_len());

        // Autoregressive generation: produce audio frames
        let mut all_codes: Vec<Vec<i64>> = Vec::new();
        let max_frames = max_tokens / crate::TOKENS_PER_FRAME;
        let gen_start = std::time::Instant::now();

        for frame_idx in 0..max_frames {
            // Generate one frame of 37 codes from the current hidden state
            let codes = match self
                .flow_matching
                .generate_frame(&hidden_state, self.device)
            {
                Some(codes) => codes,
                None => {
                    tracing::debug!("End-of-audio at frame {}", frame_idx);
                    break;
                }
            };

            // Embed the generated codes for the next backbone step
            let next_embedding = self.backbone.embed_audio_codes(&codes);

            // Store the codes
            all_codes.push(codes);

            // Forward one step through the backbone
            hidden_state = self
                .backbone
                .forward_one_embedding(&next_embedding, &mut kv_cache);

            if frame_idx % 50 == 0 && frame_idx > 0 {
                let elapsed = gen_start.elapsed().as_secs_f64();
                tracing::info!(
                    "Generated {} frames ({:.1}s audio) in {:.1}s ({:.2} frames/s)",
                    frame_idx,
                    frame_idx as f64 / crate::FRAME_RATE as f64,
                    elapsed,
                    frame_idx as f64 / elapsed,
                );
            }
        }

        if all_codes.is_empty() {
            return Err(VoxtralError::Inference(
                "Generated zero audio frames".to_string(),
            ));
        }

        tracing::info!(
            "Generated {} audio frames ({:.2}s)",
            all_codes.len(),
            all_codes.len() as f64 / crate::FRAME_RATE as f64
        );

        // Decode audio codes to waveform via codec
        let waveform = self.codec.decode(&all_codes, self.device)?;

        tracing::info!(
            "Generated {:.2}s of audio ({} samples at {}Hz)",
            waveform.len() as f64 / crate::DEFAULT_SAMPLE_RATE as f64,
            waveform.len(),
            crate::DEFAULT_SAMPLE_RATE,
        );

        Ok((waveform, crate::DEFAULT_SAMPLE_RATE))
    }

    /// Generate speech from text using a reference audio file for voice cloning.
    pub fn generate_with_reference(
        &self,
        _text: &str,
        _reference_audio_path: &str,
        _temperature: f32,
        _max_tokens: usize,
    ) -> Result<(Vec<f32>, u32)> {
        Err(VoxtralError::Inference(
            "Voice cloning from reference audio requires the codec encoder, \
             which is not included in the inference checkpoint. \
             Use a preset voice instead."
                .to_string(),
        ))
    }

    /// List available preset voices.
    pub fn list_voices(&self) -> Vec<&str> {
        self.voices.list_voices()
    }

    /// Get the model configuration.
    pub fn config(&self) -> &VoxtralConfig {
        &self.config
    }
}
