// Copyright 2026 Claude Code on behalf of Michael Yuan.
// SPDX-License-Identifier: Apache-2.0

//! Weight loading from `consolidated.safetensors`.
//!
//! Routes tensors from a single safetensors checkpoint file to the three
//! model components (Backbone, FlowMatchingTransformer, Codec) based on
//! their key prefixes.

use std::collections::HashMap;
use std::path::Path;

use crate::config::VoxtralConfig;
use crate::error::{Result, VoxtralError};
use crate::tensor::{Device, Tensor};

use super::backbone::{Backbone, BackboneConfig};
use super::codec::Codec;
use super::flow_matching::FlowMatchingTransformer;

/// Load all model weights and construct the three main components.
///
/// Reads `consolidated.safetensors` (or multiple shards) from `model_dir`,
/// partitions the tensors by prefix, and instantiates:
///
/// 1. **Backbone** – weights with prefixes `tok_embeddings`, `layers.*`, `norm`, `output`
/// 2. **FlowMatchingTransformer** – weights with prefix `multimodal.acoustic_transformer`
/// 3. **Codec** – weights with prefix `multimodal.audio_tokenizer`
///
/// # Arguments
///
/// * `model_dir` – directory containing `consolidated.safetensors` and `params.json`.
/// * `config` – parsed model configuration.
/// * `device` – target compute device.
///
/// # Returns
///
/// A tuple `(Backbone, FlowMatchingTransformer, Codec)`.
pub fn load_model_weights(
    model_dir: &Path,
    config: &VoxtralConfig,
    device: Device,
) -> Result<(Backbone, FlowMatchingTransformer, Codec)> {
    // Discover safetensors files
    let safetensors_files = find_safetensors_files(model_dir)?;
    if safetensors_files.is_empty() {
        return Err(VoxtralError::ModelLoad(format!(
            "No safetensors files found in {}",
            model_dir.display()
        )));
    }

    tracing::info!(
        "Loading weights from {} safetensors file(s) in {}",
        safetensors_files.len(),
        model_dir.display()
    );

    // Keep weights in BF16 on all backends — libtorch 2.7+ supports BF16 matmul
    // on CPU (Apple Silicon). This matches MLX's native BF16 computation exactly,
    // avoiding precision divergence that causes degenerate semantic codes.

    // Load all tensors into a single HashMap
    let mut all_weights: HashMap<String, Tensor> = HashMap::new();
    for path in &safetensors_files {
        tracing::debug!("Loading {}", path.display());
        let tensors = Tensor::load_safetensors(path)?;
        for (name, tensor) in tensors {
            let tensor = tensor.to_device(device);
            all_weights.insert(name, tensor);
        }
    }

    tracing::info!("Loaded {} weight tensors total", all_weights.len());

    // Partition weights by component
    let mut backbone_weights: HashMap<String, Tensor> = HashMap::new();
    let mut flow_weights: HashMap<String, Tensor> = HashMap::new();
    let mut codec_weights: HashMap<String, Tensor> = HashMap::new();

    for (name, tensor) in all_weights {
        if name.starts_with("acoustic_transformer.") {
            flow_weights.insert(name, tensor);
        } else if name.starts_with("audio_tokenizer.") {
            codec_weights.insert(name, tensor);
        } else {
            // Everything else goes to the backbone:
            // mm_audio_embeddings.*, layers.*, norm.*, output.*
            backbone_weights.insert(name, tensor);
        }
    }

    tracing::info!(
        "Weight partitions: backbone={}, flow_matching={}, codec={}",
        backbone_weights.len(),
        flow_weights.len(),
        codec_weights.len(),
    );

    // Build backbone
    let backbone_config = BackboneConfig::from(config);
    let backbone = Backbone::from_weights(&backbone_weights, backbone_config, device);
    tracing::info!(
        "Backbone loaded: {} layers, dim={}, heads={}",
        config.n_layers,
        config.dim,
        config.n_heads,
    );

    // Build flow-matching transformer
    let acoustic_config = config.acoustic_transformer_config();
    let flow_matching =
        FlowMatchingTransformer::from_weights(&flow_weights, acoustic_config, device);
    tracing::info!("Flow-matching transformer loaded");

    // Build codec
    let codec_config = config.audio_tokenizer_config();
    let codec = Codec::from_weights(&codec_weights, codec_config, device);
    tracing::info!("Codec loaded");

    Ok((backbone, flow_matching, codec))
}

/// Find all safetensors files in the model directory.
///
/// Looks for:
/// 1. `consolidated.safetensors` (single-file checkpoint)
/// 2. `model-00001-of-*.safetensors` (sharded checkpoint)
/// 3. Any `.safetensors` file (fallback)
fn find_safetensors_files(model_dir: &Path) -> Result<Vec<std::path::PathBuf>> {
    // Try consolidated first
    let consolidated = model_dir.join("consolidated.safetensors");
    if consolidated.exists() {
        return Ok(vec![consolidated]);
    }

    // Try sharded format
    let mut shards: Vec<std::path::PathBuf> = Vec::new();
    if let Ok(entries) = std::fs::read_dir(model_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                if name.starts_with("model-") && name.ends_with(".safetensors") {
                    shards.push(path);
                }
            }
        }
    }

    if !shards.is_empty() {
        // Sort shards to ensure consistent ordering (model-00001, model-00002, ...)
        shards.sort();
        return Ok(shards);
    }

    // Fallback: any safetensors file
    let mut files: Vec<std::path::PathBuf> = Vec::new();
    if let Ok(entries) = std::fs::read_dir(model_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("safetensors") {
                files.push(path);
            }
        }
    }

    if files.is_empty() {
        return Err(VoxtralError::ModelLoad(format!(
            "No safetensors files found in {}. Expected consolidated.safetensors or sharded model-*.safetensors",
            model_dir.display()
        )));
    }

    files.sort();
    Ok(files)
}
