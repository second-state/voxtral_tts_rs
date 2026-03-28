//! Speech synthesis endpoint (OpenAI-compatible).
//!
//! `POST /v1/audio/speech` - Generate speech from text.

use axum::extract::State;
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response, Sse};
use axum::Json;
use serde::{Deserialize, Serialize};
use tokio_stream::wrappers::ReceiverStream;

use crate::state::AppState;

/// Maximum allowed input text length (in characters).
const MAX_INPUT_LENGTH: usize = 4096;

/// Minimum allowed speed multiplier.
const MIN_SPEED: f32 = 0.25;

/// Maximum allowed speed multiplier.
const MAX_SPEED: f32 = 4.0;

// ---------------------------------------------------------------------------
// Request / Response types
// ---------------------------------------------------------------------------

/// OpenAI-compatible speech request body.
#[derive(Debug, Deserialize)]
pub struct SpeechRequest {
    /// Model name (currently only "voxtral-4b-tts" is accepted).
    #[serde(default = "default_model")]
    #[allow(dead_code)]
    pub model: String,

    /// Text to synthesize.
    pub input: String,

    /// Voice name (preset name or OpenAI alias).
    #[serde(default = "default_voice")]
    pub voice: String,

    /// Response audio format. Currently only "wav" is supported.
    #[serde(default = "default_format")]
    pub response_format: String,

    /// Speed multiplier (0.25 - 4.0). Reserved for future use.
    #[serde(default = "default_speed")]
    pub speed: f32,

    /// Whether to stream the response via SSE.
    #[serde(default)]
    pub stream: bool,
}

fn default_model() -> String {
    "voxtral-4b-tts".to_string()
}

fn default_voice() -> String {
    "alloy".to_string()
}

fn default_format() -> String {
    "wav".to_string()
}

fn default_speed() -> f32 {
    1.0
}

/// Error response body.
#[derive(Serialize)]
struct ErrorBody {
    error: ErrorDetail,
}

#[derive(Serialize)]
struct ErrorDetail {
    message: String,
    r#type: &'static str,
    code: &'static str,
}

/// Number of audio frames to accumulate before sending the first streaming chunk.
/// Lower = faster time-to-first-audio. 10 frames ≈ 0.8s audio, arrives after ~6.7s.
const STREAMING_FIRST_CHUNK_FRAMES: usize = 10;

/// Number of audio frames per subsequent streaming chunk.
/// 25 frames ≈ 2s audio.
const STREAMING_CHUNK_FRAMES: usize = 25;

// ---------------------------------------------------------------------------
// Handler
// ---------------------------------------------------------------------------

/// `POST /v1/audio/speech` - Generate speech from text.
pub async fn create_speech(
    State(state): State<AppState>,
    Json(req): Json<SpeechRequest>,
) -> Response {
    // ---- Input validation ----

    if req.input.is_empty() {
        return error_response(
            StatusCode::BAD_REQUEST,
            "Input text must not be empty.",
            "invalid_request_error",
            "invalid_input",
        );
    }

    if req.input.len() > MAX_INPUT_LENGTH {
        return error_response(
            StatusCode::BAD_REQUEST,
            &format!(
                "Input text too long ({} chars). Maximum is {} characters.",
                req.input.len(),
                MAX_INPUT_LENGTH,
            ),
            "invalid_request_error",
            "input_too_long",
        );
    }

    if req.speed < MIN_SPEED || req.speed > MAX_SPEED {
        return error_response(
            StatusCode::BAD_REQUEST,
            &format!(
                "Speed must be between {} and {}. Got {}.",
                MIN_SPEED, MAX_SPEED, req.speed
            ),
            "invalid_request_error",
            "invalid_speed",
        );
    }

    let supported_formats = ["wav", "pcm", "mp3", "flac", "ogg", "opus"];
    if !supported_formats.contains(&req.response_format.as_str()) {
        return error_response(
            StatusCode::BAD_REQUEST,
            &format!(
                "Unsupported response_format '{}'. Supported: wav, pcm, mp3, flac, ogg.",
                req.response_format,
            ),
            "invalid_request_error",
            "invalid_format",
        );
    }

    // ---- Dispatch ----

    if req.stream {
        handle_streaming(state, req).await
    } else {
        handle_non_streaming(state, req).await
    }
}

// ---------------------------------------------------------------------------
// Non-streaming response
// ---------------------------------------------------------------------------

async fn handle_non_streaming(state: AppState, req: SpeechRequest) -> Response {
    let voice = req.voice.clone();
    let input = req.input.clone();
    let format = req.response_format.clone();

    // Run inference on a blocking thread to avoid starving the Tokio runtime.
    let result = tokio::task::spawn_blocking(move || {
        let tts = state
            .lock()
            .map_err(|e| VoxtralError::Inference(format!("Failed to acquire model lock: {}", e)))?;
        tts.generate(&input, &voice, 0.7, 4096)
    })
    .await;

    match result {
        Ok(Ok((samples, sample_rate))) => {
            match voxtral_tts::audio::encode_audio(&samples, sample_rate, &format) {
                Ok((body, content_type)) => {
                    (StatusCode::OK, [(header::CONTENT_TYPE, content_type)], body).into_response()
                }
                Err(e) => {
                    return error_response(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        &format!("Failed to encode audio: {}", e),
                        "server_error",
                        "encoding_error",
                    );
                }
            }
        }
        Ok(Err(e)) => error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            &format!("Inference failed: {}", e),
            "server_error",
            "inference_error",
        ),
        Err(e) => error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            &format!("Task join error: {}", e),
            "server_error",
            "internal_error",
        ),
    }
}

// ---------------------------------------------------------------------------
// Streaming response (SSE with base64-encoded PCM chunks)
// ---------------------------------------------------------------------------

async fn handle_streaming(state: AppState, req: SpeechRequest) -> Response {
    let voice = req.voice.clone();
    let input = req.input.clone();

    let (tx, rx) = tokio::sync::mpsc::channel::<
        Result<axum::response::sse::Event, std::convert::Infallible>,
    >(8);

    // Spawn inference on a blocking thread.
    tokio::task::spawn_blocking(move || {
        let tts = match state.lock() {
            Ok(guard) => guard,
            Err(e) => {
                let payload = serde_json::json!({
                    "type": "error",
                    "error": { "message": format!("Failed to acquire model lock: {}", e) }
                });
                let _ = tx.blocking_send(Ok(
                    axum::response::sse::Event::default().data(payload.to_string())
                ));
                return;
            }
        };

        let b64 = base64::engine::general_purpose::STANDARD;

        // Frame-level streaming: generate frames and send audio every N frames.
        // First chunk is smaller for faster time-to-first-audio.
        let result = tts.generate_streaming(
            &input,
            &voice,
            4096,
            STREAMING_FIRST_CHUNK_FRAMES,
            STREAMING_CHUNK_FRAMES,
            |samples| {
                let pcm_bytes = voxtral_tts::audio::encode_pcm_i16(samples);
                let delta = b64.encode(&pcm_bytes);

                let payload = serde_json::json!({
                    "type": "speech.audio.delta",
                    "delta": delta,
                });
                let event =
                    axum::response::sse::Event::default().data(payload.to_string());
                // Returns false if client disconnected
                tx.blocking_send(Ok(event)).is_ok()
            },
        );

        if let Err(e) = result {
            tracing::error!("Streaming inference failed: {}", e);
            let payload = serde_json::json!({
                "type": "error",
                "error": { "message": e.to_string() }
            });
            let _ = tx.blocking_send(Ok(
                axum::response::sse::Event::default().data(payload.to_string())
            ));
            return;
        }

        // Send done event
        let payload = serde_json::json!({ "type": "speech.audio.done" });
        let _ = tx.blocking_send(Ok(
            axum::response::sse::Event::default().data(payload.to_string())
        ));
    });

    let stream = ReceiverStream::new(rx);
    Sse::new(stream)
        .keep_alive(axum::response::sse::KeepAlive::default())
        .into_response()
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

use base64::Engine;
use voxtral_tts::error::VoxtralError;

fn error_response(
    status: StatusCode,
    message: &str,
    error_type: &'static str,
    code: &'static str,
) -> Response {
    let body = ErrorBody {
        error: ErrorDetail {
            message: message.to_string(),
            r#type: error_type,
            code,
        },
    };
    (status, Json(body)).into_response()
}
