# Rust CLI and API server for Voxtral TTS from Mistral

Rust implemenation of [Voxtral-4B-TTS](https://huggingface.co/mistralai/Voxtral-4B-TTS-2603) — Mistral AI's 4B-parameter text-to-speech model. Runs on macOS (Apple Silicon via MLX) and Linux (CPU or CUDA via libtorch). No Python required. Includes both CLI and API server. Ready for agent harness.

## Quick Start

### 1. Download the release

Download the platform-specific zip from [GitHub Releases](https://github.com/second-state/voxtral_tts_rs/releases):

| Platform | Asset |
|----------|-------|
| macOS (Apple Silicon) | `voxtral-tts-macos-aarch64.zip` |
| Linux x86_64 (CPU) | `voxtral-tts-linux-x86_64.zip` |
| Linux x86_64 (CUDA) | `voxtral-tts-linux-x86_64-cuda.zip` |
| Linux ARM64 (CPU) | `voxtral-tts-linux-aarch64.zip` |
| Linux ARM64 (CUDA) | `voxtral-tts-linux-aarch64-cuda.zip` |

```bash
# Example: macOS Apple Silicon
curl -LO https://github.com/second-state/voxtral_tts_rs/releases/latest/download/voxtral-tts-macos-aarch64.zip
unzip voxtral-tts-macos-aarch64.zip
cd voxtral-tts-macos-aarch64
```

### 2. Download the model

```bash
bash <(curl -sSf https://raw.githubusercontent.com/second-state/voxtral_tts_rs/main/scripts/download_model.sh)
```

This downloads `consolidated.safetensors` (8 GB), `params.json`, `tekken.json`, and 20 voice embeddings into `models/voxtral-4b-tts/`.

### 3. Copy voice embeddings to the model folder

The release zip includes pre-converted voice embeddings (`.safetensors`). Copy them into the model directory:

```bash
cp voice_embedding/*.safetensors models/voxtral-4b-tts/voice_embedding/
```

### 4. Generate speech (CLI)

```bash
./voxtral-tts models/voxtral-4b-tts \
    --text "Hello, this is Voxtral TTS!" \
    --voice neutral_female \
    --output output.wav
```

### 5. Start the API server

```bash
./voxtral-tts-server models/voxtral-4b-tts --port 8080
```

```bash
curl -X POST http://localhost:8080/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{"input":"Hello world","voice":"alloy"}' \
    -o output.wav
```

## CLI Reference

```
voxtral-tts <MODEL_DIR> --text "..." [OPTIONS]
```

| Flag | Default | Description |
|------|---------|-------------|
| `<MODEL_DIR>` | (required) | Path to model directory |
| `--text`, `-t` | (required) | Text to synthesize |
| `--voice`, `-v` | `neutral_female` | Voice name or OpenAI alias |
| `--output`, `-o` | `output.wav` | Output WAV file path |
| `--temperature` | `0.7` | Sampling temperature (higher = more variation) |
| `--max-tokens` | `4096` | Maximum generation tokens |
| `--reference-audio` | | Voice reference audio file (for voice cloning) |
| `--list-voices` | | Print available voices and exit |

Examples:

```bash
# English with a casual voice
./voxtral-tts models/voxtral-4b-tts --text "Hey, what's up?" --voice casual_male -o casual.wav

# French
./voxtral-tts models/voxtral-4b-tts --text "Bonjour le monde!" --voice fr_female -o bonjour.wav

# List all voices
./voxtral-tts models/voxtral-4b-tts --list-voices --text ""
```

## API Server Reference

```
voxtral-tts-server <MODEL_DIR> [OPTIONS]
```

| Flag | Default | Description |
|------|---------|-------------|
| `<MODEL_DIR>` | (required) | Path to model directory |
| `--host` | `127.0.0.1` | Bind host address |
| `--port` | `8080` | Bind port |

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check — returns `{"status":"ok"}` |
| `/v1/models` | GET | List available models |
| `/v1/audio/speech` | POST | Generate speech (OpenAI-compatible) |

### POST /v1/audio/speech

Request body:

```json
{
    "model": "voxtral-4b-tts",
    "input": "Text to synthesize",
    "voice": "neutral_female",
    "response_format": "wav",
    "speed": 1.0,
    "stream": false
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `input` | string | (required) | Text to synthesize (max 4096 chars) |
| `model` | string | `voxtral-4b-tts` | Model name |
| `voice` | string | `alloy` | Voice name or OpenAI alias |
| `response_format` | string | `wav` | Output format: `wav`, `pcm`, `mp3`, `flac`, `ogg`, `opus` |
| `speed` | float | `1.0` | Speed multiplier (0.25–4.0, reserved) |
| `stream` | bool | `false` | Enable SSE streaming |

**Non-streaming** returns binary audio with the appropriate content type (`audio/wav`, `audio/pcm`, `audio/mpeg`, `audio/flac`, or `audio/ogg`).

**Streaming** (`"stream": true`) returns Server-Sent Events with base64 PCM chunks:

```
data: {"type":"speech.audio.delta","delta":"<base64 16-bit LE PCM>"}
data: {"type":"speech.audio.delta","delta":"<base64 16-bit LE PCM>"}
data: {"type":"speech.audio.done"}
```

Examples:

```bash
# Non-streaming WAV
curl -X POST http://localhost:8080/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{"input":"Hello world","voice":"alloy"}' \
    -o output.wav

# Streaming
curl -N -X POST http://localhost:8080/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{"input":"Hello world","voice":"alloy","stream":true}'
```

## Voices

### 20 Preset Voices

| Voice | Language | Gender |
|-------|----------|--------|
| `casual_female`, `casual_male` | English | F, M |
| `cheerful_female` | English | F |
| `neutral_female`, `neutral_male` | English | F, M |
| `fr_male`, `fr_female` | French | M, F |
| `es_male`, `es_female` | Spanish | M, F |
| `de_male`, `de_female` | German | M, F |
| `pt_male`, `pt_female` | Portuguese | M, F |
| `it_male`, `it_female` | Italian | M, F |
| `nl_male`, `nl_female` | Dutch | M, F |
| `ar_male` | Arabic | M |
| `hi_male`, `hi_female` | Hindi | M, F |

### OpenAI Voice Aliases

| Alias | Maps to |
|-------|---------|
| `alloy` | `neutral_female` |
| `echo` | `casual_male` |
| `fable` | `cheerful_female` |
| `onyx` | `neutral_male` |
| `nova` | `casual_female` |
| `shimmer` | `fr_female` |

## Build from Source

### Prerequisites

| Platform | Requirements |
|----------|-------------|
| macOS (Apple Silicon) | Xcode Command Line Tools, CMake, Rust 1.75+ |
| Linux (CPU) | GCC/Clang, Rust 1.75+ |
| Linux (CUDA) | NVIDIA driver 535+, CUDA toolkit, Rust 1.75+ |

### macOS (MLX backend)

```bash
git clone https://github.com/second-state/voxtral_tts_rs.git
cd voxtral_tts_rs
git submodule update --init --recursive
cargo build --release --no-default-features --features mlx
```

### Linux (libtorch backend)

```bash
git clone https://github.com/second-state/voxtral_tts_rs.git
cd voxtral_tts_rs

# Download libtorch (auto-detects x86_64 or aarch64)
bash scripts/download_libtorch.sh cpu      # CPU only
bash scripts/download_libtorch.sh cuda     # CUDA 12.6

# Build
export LIBTORCH=$(pwd)/libtorch
export LIBTORCH_BYPASS_VERSION_CHECK=1
cargo build --release
```

### Convert voice embeddings

The model ships voice embeddings as PyTorch `.pt` files. Convert them to `.safetensors` (required for MLX, optional for libtorch):

```bash
pip install torch safetensors
python3 -c "
import torch, os
from safetensors.torch import save_file
d = 'models/voxtral-4b-tts/voice_embedding'
for f in os.listdir(d):
    if f.endswith('.pt'):
        t = torch.load(os.path.join(d, f), map_location='cpu', weights_only=True)
        save_file({'embedding': t}, os.path.join(d, f.replace('.pt', '.safetensors')))
        print(f'Converted {f}')
"
```

### Run tests

```bash
cargo test
```

## Architecture

The model has three components totalling 4B parameters:

```
Text ──> Tekken Tokenizer ──> Token IDs
                                  │
                                  v
Voice Embedding ──> Backbone Decoder (3.4B, 26 layers) ──> Hidden States
                                                               │
                                                               v
                    Flow-Matching Transformer (390M) ──> 37 Audio Codes/Frame
                                                               │
                                                               v
                    Voxtral Codec Decoder (300M) ──> 24kHz Mono Waveform
```

| Component | Parameters | Architecture |
|-----------|-----------|--------------|
| Backbone Decoder | 3.4B | 26-layer Mistral transformer, dim=3072, 32 heads (8 KV), SwiGLU, RoPE |
| Flow-Matching Transformer | 390M | 3-layer bidirectional transformer, Euler ODE (7 steps), CFG |
| Voxtral Codec Decoder | 300M | 4 conv+transformer blocks, strides [1,2,2,2], 240-channel output |

## Environment Variables

| Variable | Description |
|----------|-------------|
| `RUST_LOG` | Log verbosity: `error`, `warn`, `info` (default), `debug`, `trace` |
| `LIBTORCH` | Path to libtorch directory (Linux/tch backend only) |
| `LIBTORCH_BYPASS_VERSION_CHECK` | Set to `1` to skip libtorch version check |

## License

Apache-2.0

The Voxtral model weights are licensed under [Mistral AI Non-Production License](https://mistral.ai/licenses/MNPL-0.1.md).
