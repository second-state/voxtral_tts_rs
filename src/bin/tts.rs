//! Voxtral TTS - Text-to-Speech CLI binary.
//!
//! Usage:
//!   voxtral-tts <MODEL_DIR> --text "Hello world" [--voice neutral_female] [--output output.wav]

use clap::Parser;
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    name = "voxtral-tts",
    about = "Voxtral TTS - Text-to-Speech CLI",
    version
)]
struct Cli {
    /// Path to model directory (containing params.json, tekken.json, weights, etc.)
    model_dir: PathBuf,

    /// Text to synthesize
    #[arg(short, long)]
    text: String,

    /// Voice name (preset or OpenAI alias: alloy, echo, fable, onyx, nova, shimmer)
    #[arg(short, long, default_value = "neutral_female")]
    voice: String,

    /// Output WAV file path
    #[arg(short, long, default_value = "output.wav")]
    output: String,

    /// Sampling temperature (higher = more variation)
    #[arg(long, default_value = "0.7")]
    temperature: f32,

    /// Maximum generation tokens
    #[arg(long, default_value = "16384")]
    max_tokens: usize,

    /// Voice reference audio file (for voice cloning instead of preset voices)
    #[arg(long)]
    reference_audio: Option<PathBuf>,

    /// List available voices and exit
    #[arg(long)]
    list_voices: bool,
}

fn main() -> anyhow::Result<()> {
    // Initialise tracing (respects RUST_LOG env var).
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();

    // Validate model directory exists.
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
    let tts = voxtral_tts_rs::inference::VoxtralTTS::from_dir(&cli.model_dir, device)?;

    // Handle --list-voices.
    if cli.list_voices {
        println!("Available voices:");
        let mut voices = tts.list_voices();
        voices.sort();
        for v in &voices {
            println!("  {}", v);
        }
        println!("\nOpenAI-compatible aliases:");
        println!("  alloy   -> neutral_female");
        println!("  echo    -> casual_male");
        println!("  fable   -> cheerful_female");
        println!("  onyx    -> neutral_male");
        println!("  nova    -> casual_female");
        println!("  shimmer -> fr_female");
        return Ok(());
    }

    // Validate text.
    if cli.text.is_empty() {
        anyhow::bail!("--text must not be empty");
    }

    // Generate speech.
    let (samples, sample_rate) = if let Some(ref ref_path) = cli.reference_audio {
        tracing::info!(
            "Using reference audio for voice cloning: {}",
            ref_path.display()
        );
        tts.generate_with_reference(
            &cli.text,
            ref_path.to_str().unwrap_or_default(),
            cli.temperature,
            cli.max_tokens,
        )?
    } else {
        tracing::info!("Using preset voice: {}", cli.voice);
        tts.generate(&cli.text, &cli.voice, cli.temperature, cli.max_tokens)?
    };

    // Write output WAV.
    voxtral_tts_rs::audio::write_wav_file(&cli.output, &samples, sample_rate)?;
    tracing::info!(
        "Wrote {:.2}s of audio to {}",
        samples.len() as f64 / sample_rate as f64,
        cli.output,
    );

    Ok(())
}
