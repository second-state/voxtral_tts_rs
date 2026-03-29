//! Voxtral TTS Server - OpenAI-compatible TTS API.
//!
//! Provides an HTTP API compatible with the OpenAI audio/speech endpoint.
//!
//! Usage:
//!   voxtral-tts-server <MODEL_DIR> [--host 127.0.0.1] [--port 8080]

mod routes;
mod state;

use clap::Parser;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use crate::state::AppState;

#[derive(Parser)]
#[command(
    name = "voxtral-tts-server",
    about = "Voxtral TTS - OpenAI-compatible TTS API server",
    version
)]
struct Cli {
    /// Path to model directory
    model_dir: PathBuf,

    /// Bind host address
    #[arg(long, default_value = "127.0.0.1")]
    host: String,

    /// Bind port
    #[arg(long, default_value = "8080")]
    port: u16,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialise tracing.
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();

    // Validate model directory.
    if !cli.model_dir.is_dir() {
        anyhow::bail!(
            "Model directory does not exist: {}",
            cli.model_dir.display()
        );
    }

    // Determine compute device.
    let device = voxtral_tts_rs::Device::best_available();
    tracing::info!("Using device: {:?}", device);

    // Load the full model.
    tracing::info!("Loading model from {} ...", cli.model_dir.display());
    let tts = voxtral_tts_rs::inference::VoxtralTTS::from_dir(&cli.model_dir, device)?;
    tracing::info!("Model loaded successfully");

    // Wrap in shared state.
    let state: AppState = Arc::new(Mutex::new(tts));

    // Build Axum router.
    let app = axum::Router::new()
        .route("/health", axum::routing::get(routes::health::health_check))
        .route(
            "/v1/models",
            axum::routing::get(routes::models::list_models),
        )
        .route(
            "/v1/audio/speech",
            axum::routing::post(routes::speech::create_speech),
        )
        .with_state(state);

    // Bind and serve.
    let addr = format!("{}:{}", cli.host, cli.port);
    tracing::info!("Listening on http://{}", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
