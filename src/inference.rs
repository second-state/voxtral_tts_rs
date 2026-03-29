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
        tracing::debug!("Text tokens: {} tokens: {:?}", text_tokens.len(), text_tokens);

        // Build input embedding sequence (Voxtral TTS format):
        // [BOS] [BEGIN_AUDIO] voice_embeddings... [NEXT_AUDIO_TEXT] text_tokens [REPEAT_AUDIO_TEXT] [BEGIN_AUDIO]
        let total_prefix_len = 2 + n_voice_frames + 1 + text_tokens.len() + 2;
        let mut embeddings_data = Vec::with_capacity(total_prefix_len);

        // [BOS]
        embeddings_data.push(self.backbone.embed_text_token(crate::BOS_TOKEN_ID));
        // [BEGIN_AUDIO]
        embeddings_data.push(self.backbone.embed_text_token(crate::BEGIN_AUDIO_TOKEN_ID));
        // Voice embeddings (pre-computed backbone hidden states, directly injected)
        for i in 0..n_voice_frames {
            embeddings_data.push(voice_embedding.select(0, i as i64));
        }
        // [NEXT_AUDIO_TEXT]
        embeddings_data.push(self.backbone.embed_text_token(crate::NEXT_AUDIO_TEXT_TOKEN_ID));
        // Text token embeddings
        for &token_id in &text_tokens {
            embeddings_data.push(self.backbone.embed_text_token(token_id as i64));
        }
        // [REPEAT_AUDIO_TEXT]
        embeddings_data.push(self.backbone.embed_text_token(crate::REPEAT_AUDIO_TEXT_TOKEN_ID));
        // [BEGIN_AUDIO] (signals start of audio generation)
        embeddings_data.push(self.backbone.embed_text_token(crate::BEGIN_AUDIO_TOKEN_ID));

        // Stack into [1, total_prefix_len, dim]
        let prefix_embeddings = Tensor::stack(&embeddings_data, 0)
            .unsqueeze(0);

        tracing::debug!(
            "Prefix: {} tokens (voice={}, text={})",
            total_prefix_len,
            n_voice_frames,
            text_tokens.len()
        );

        // Prefill backbone
        let mut kv_cache = self.backbone.new_kv_cache();
        let prefill_start = std::time::Instant::now();
        self.backbone
            .forward_prefill_embeddings(&prefix_embeddings, &mut kv_cache);
        tracing::info!("Prefill: {:.2}s (seq_len={})", prefill_start.elapsed().as_secs_f64(), kv_cache.seq_len());

        // First decode step: feed AUDIO token to get first hidden state
        // (matches reference: prefill ends with BEGIN_AUDIO, then AUDIO token starts generation)
        let audio_embed = self.backbone.embed_text_token(crate::AUDIO_TOKEN_ID);
        let mut hidden_state = self
            .backbone
            .forward_one_embedding(&audio_embed, &mut kv_cache);

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
                    tracing::info!("End-of-audio at frame {}", frame_idx);
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

    /// Generate speech with frame-level streaming.
    ///
    /// Instead of waiting for all frames, decodes and sends audio every
    /// `chunk_frames` frames. The first chunk uses `first_chunk_frames` for
    /// faster time-to-first-audio.
    ///
    /// `on_audio` is called with each PCM chunk. Return `true` to continue,
    /// `false` to stop (e.g., client disconnected).
    pub fn generate_streaming<F>(
        &self,
        text: &str,
        voice: &str,
        max_tokens: usize,
        first_chunk_frames: usize,
        chunk_frames: usize,
        mut on_audio: F,
    ) -> Result<()>
    where
        F: FnMut(&[f32]) -> bool,
    {
        if text.is_empty() {
            return Err(VoxtralError::Inference("Input text is empty".to_string()));
        }

        let voice_embedding = self.voices.get(voice)?;
        let voice_shape = voice_embedding.size();
        let n_voice_frames = voice_shape[0] as usize;

        let text_tokens: Vec<u32> = self.tokenizer.encode(text);
        // [BOS] [BEGIN_AUDIO] voice_embeddings... [NEXT_AUDIO_TEXT] text_tokens [REPEAT_AUDIO_TEXT] [BEGIN_AUDIO]
        let total_prefix_len = 2 + n_voice_frames + 1 + text_tokens.len() + 2;
        let mut embeddings_data = Vec::with_capacity(total_prefix_len);

        embeddings_data.push(self.backbone.embed_text_token(crate::BOS_TOKEN_ID));
        embeddings_data.push(self.backbone.embed_text_token(crate::BEGIN_AUDIO_TOKEN_ID));
        for i in 0..n_voice_frames {
            embeddings_data.push(voice_embedding.select(0, i as i64));
        }
        embeddings_data.push(self.backbone.embed_text_token(crate::NEXT_AUDIO_TEXT_TOKEN_ID));
        for &token_id in &text_tokens {
            embeddings_data.push(self.backbone.embed_text_token(token_id as i64));
        }
        embeddings_data.push(self.backbone.embed_text_token(crate::REPEAT_AUDIO_TEXT_TOKEN_ID));
        embeddings_data.push(self.backbone.embed_text_token(crate::BEGIN_AUDIO_TOKEN_ID));

        let prefix_embeddings = Tensor::stack(&embeddings_data, 0).unsqueeze(0);

        let mut kv_cache = self.backbone.new_kv_cache();
        let prefill_start = std::time::Instant::now();
        self.backbone
            .forward_prefill_embeddings(&prefix_embeddings, &mut kv_cache);
        tracing::info!(
            "Prefill: {:.2}s (seq_len={})",
            prefill_start.elapsed().as_secs_f64(),
            kv_cache.seq_len()
        );

        // First decode step: feed AUDIO token to get first hidden state
        let audio_embed = self.backbone.embed_text_token(crate::AUDIO_TOKEN_ID);
        let mut hidden_state = self
            .backbone
            .forward_one_embedding(&audio_embed, &mut kv_cache);

        let max_frames = max_tokens / crate::TOKENS_PER_FRAME;
        let gen_start = std::time::Instant::now();
        let mut pending_codes: Vec<Vec<i64>> = Vec::new();
        let mut total_frames = 0usize;
        let mut total_audio_samples = 0usize;

        for frame_idx in 0..max_frames {
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

            let next_embedding = self.backbone.embed_audio_codes(&codes);
            pending_codes.push(codes);
            total_frames += 1;

            hidden_state = self
                .backbone
                .forward_one_embedding(&next_embedding, &mut kv_cache);

            // Determine if we should flush this batch
            let target = if total_audio_samples == 0 {
                first_chunk_frames
            } else {
                chunk_frames
            };

            if pending_codes.len() >= target {
                let waveform = self.codec.decode(&pending_codes, self.device)?;
                let audio_dur = waveform.len() as f64 / crate::DEFAULT_SAMPLE_RATE as f64;
                total_audio_samples += waveform.len();
                tracing::info!(
                    "Streaming chunk: {} frames, {:.1}s audio (total {:.1}s, {:.1}s elapsed)",
                    pending_codes.len(),
                    audio_dur,
                    total_audio_samples as f64 / crate::DEFAULT_SAMPLE_RATE as f64,
                    gen_start.elapsed().as_secs_f64(),
                );
                pending_codes.clear();

                if !on_audio(&waveform) {
                    tracing::info!("Client disconnected at frame {}", frame_idx);
                    return Ok(());
                }
            }
        }

        // Flush remaining frames
        if !pending_codes.is_empty() {
            let waveform = self.codec.decode(&pending_codes, self.device)?;
            total_audio_samples += waveform.len();
            tracing::info!(
                "Streaming final chunk: {} frames (total {:.1}s in {:.1}s)",
                pending_codes.len(),
                total_audio_samples as f64 / crate::DEFAULT_SAMPLE_RATE as f64,
                gen_start.elapsed().as_secs_f64(),
            );
            on_audio(&waveform);
        }

        if total_frames == 0 {
            return Err(VoxtralError::Inference(
                "Generated zero audio frames".to_string(),
            ));
        }

        Ok(())
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
