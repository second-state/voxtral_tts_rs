//! Voice embedding loading and management.

use std::collections::HashMap;
use std::path::Path;

use crate::error::{Result, VoxtralError};
use crate::tensor::{DType, Device, Tensor};

/// Preset voice names with OpenAI-compatible aliases.
pub const PRESET_VOICES: &[&str] = &[
    "casual_female",
    "casual_male",
    "cheerful_female",
    "neutral_female",
    "neutral_male",
    "pt_male",
    "pt_female",
    "nl_male",
    "nl_female",
    "it_male",
    "it_female",
    "fr_male",
    "fr_female",
    "es_male",
    "es_female",
    "de_male",
    "de_female",
    "ar_male",
    "hi_male",
    "hi_female",
];

/// Map OpenAI TTS API voice names to Voxtral preset names.
pub fn resolve_voice_alias(name: &str) -> String {
    match name.to_lowercase().as_str() {
        "alloy" => "neutral_female".to_string(),
        "echo" => "casual_male".to_string(),
        "fable" => "cheerful_female".to_string(),
        "onyx" => "neutral_male".to_string(),
        "nova" => "casual_female".to_string(),
        "shimmer" => "fr_female".to_string(),
        _ => name.to_string(),
    }
}

/// Voice embedding store.
pub struct VoiceStore {
    embeddings: HashMap<String, Tensor>,
}

impl VoiceStore {
    /// Load all voice embeddings from a model directory.
    /// Expects `<model_dir>/voice_embedding/<name>.pt` files.
    pub fn from_dir(model_dir: &Path, device: Device) -> Result<Self> {
        let voice_dir = model_dir.join("voice_embedding");
        let mut embeddings = HashMap::new();

        if !voice_dir.exists() {
            tracing::warn!(
                "Voice embedding directory not found: {}",
                voice_dir.display()
            );
            return Ok(Self { embeddings });
        }

        // On CPU, libtorch cannot do BF16 matmul — cast voice embeddings to F32
        // (same logic as model weight loading in weights.rs)
        let need_f32 = matches!(device, Device::Cpu);

        for name in PRESET_VOICES {
            let pt_path = voice_dir.join(format!("{}.pt", name));
            let safetensors_path = voice_dir.join(format!("{}.safetensors", name));

            if safetensors_path.exists() {
                // Prefer safetensors format (works on both backends)
                let tensors = Tensor::load_safetensors(&safetensors_path)?;
                if let Some((_, tensor)) = tensors.into_iter().next() {
                    let tensor = if need_f32 {
                        tensor.to_dtype(DType::Float32).to_device(device)
                    } else {
                        tensor.to_device(device)
                    };
                    embeddings.insert(name.to_string(), tensor);
                    tracing::debug!("Loaded voice embedding: {} (safetensors)", name);
                }
            } else if pt_path.exists() {
                // Fall back to .pt format (tch backend only)
                match Tensor::load_pt(&pt_path) {
                    Ok(tensor) => {
                        let tensor = if need_f32 {
                            tensor.to_dtype(DType::Float32).to_device(device)
                        } else {
                            tensor.to_device(device)
                        };
                        embeddings.insert(name.to_string(), tensor);
                        tracing::debug!("Loaded voice embedding: {} (.pt)", name);
                    }
                    Err(e) => {
                        tracing::warn!("Failed to load {}: {}", pt_path.display(), e);
                    }
                }
            } else {
                tracing::debug!("Voice embedding not found: {}", name);
            }
        }

        tracing::info!("Loaded {} voice embeddings", embeddings.len());
        Ok(Self { embeddings })
    }

    /// Get a voice embedding by name (resolves aliases).
    pub fn get(&self, name: &str) -> Result<&Tensor> {
        let resolved = resolve_voice_alias(name);
        self.embeddings.get(&resolved).ok_or_else(|| {
            VoxtralError::VoiceNotFound(format!(
                "Voice '{}' (resolved to '{}') not found. Available: {:?}",
                name,
                resolved,
                self.list_voices(),
            ))
        })
    }

    /// List all available voice names.
    pub fn list_voices(&self) -> Vec<&str> {
        self.embeddings.keys().map(|s| s.as_str()).collect()
    }

    /// Check if a voice name is available.
    pub fn has_voice(&self, name: &str) -> bool {
        let resolved = resolve_voice_alias(name);
        self.embeddings.contains_key(&resolved)
    }
}
