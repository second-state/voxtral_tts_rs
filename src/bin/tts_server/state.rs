//! Shared application state for the TTS server.

use std::sync::{Arc, Mutex};

use voxtral_tts_rs::inference::VoxtralTTS;

/// Shared state holding the loaded TTS model behind a mutex.
///
/// Inference is CPU/GPU-bound and not `Send`-safe across threads without
/// synchronisation, so we wrap it in `Arc<Mutex<_>>` and use
/// `tokio::task::spawn_blocking` when performing inference.
pub type AppState = Arc<Mutex<VoxtralTTS>>;
