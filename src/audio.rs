//! Audio I/O utilities: WAV read/write, PCM encoding, resampling.

use crate::error::{Result, VoxtralError};
use crate::DEFAULT_SAMPLE_RATE;

/// Load a WAV file and return (samples, sample_rate).
/// Samples are normalized to [-1.0, 1.0] as mono f32.
pub fn load_wav_file(path: &str) -> Result<(Vec<f32>, u32)> {
    let reader = hound::WavReader::open(path)
        .map_err(|e| VoxtralError::Audio(format!("Failed to open {}: {}", path, e)))?;

    let spec = reader.spec();
    let sample_rate = spec.sample_rate;

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .into_samples::<f32>()
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| VoxtralError::Audio(format!("Failed to read samples: {}", e)))?,
        hound::SampleFormat::Int => {
            let max_val = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .into_samples::<i32>()
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(|e| VoxtralError::Audio(format!("Failed to read samples: {}", e)))?
                .into_iter()
                .map(|s| s as f32 / max_val)
                .collect()
        }
    };

    // Convert stereo to mono if needed
    let samples = if spec.channels == 2 {
        samples
            .chunks(2)
            .map(|pair| (pair[0] + pair.get(1).copied().unwrap_or(0.0)) / 2.0)
            .collect()
    } else {
        samples
    };

    Ok((samples, sample_rate))
}

/// Write samples to a WAV file (24kHz, 16-bit mono).
pub fn write_wav_file(path: &str, samples: &[f32], sample_rate: u32) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = hound::WavWriter::create(path, spec)
        .map_err(|e| VoxtralError::Audio(format!("Failed to create {}: {}", path, e)))?;

    for &sample in samples {
        let s = (sample * 32767.0).clamp(-32768.0, 32767.0) as i16;
        writer
            .write_sample(s)
            .map_err(|e| VoxtralError::Audio(format!("Failed to write sample: {}", e)))?;
    }

    writer
        .finalize()
        .map_err(|e| VoxtralError::Audio(format!("Failed to finalize WAV: {}", e)))?;

    Ok(())
}

/// Encode samples to WAV bytes in memory (24kHz, 16-bit mono).
pub fn write_wav_bytes(samples: &[f32], sample_rate: u32) -> Result<Vec<u8>> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut buffer = std::io::Cursor::new(Vec::new());
    {
        let mut writer = hound::WavWriter::new(&mut buffer, spec)
            .map_err(|e| VoxtralError::Audio(format!("Failed to create WAV writer: {}", e)))?;

        for &sample in samples {
            let s = (sample * 32767.0).clamp(-32768.0, 32767.0) as i16;
            writer
                .write_sample(s)
                .map_err(|e| VoxtralError::Audio(format!("Failed to write sample: {}", e)))?;
        }

        writer
            .finalize()
            .map_err(|e| VoxtralError::Audio(format!("Failed to finalize WAV: {}", e)))?;
    }

    Ok(buffer.into_inner())
}

/// Encode f32 samples to 16-bit signed little-endian PCM bytes.
pub fn encode_pcm_i16(samples: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(samples.len() * 2);
    for &sample in samples {
        let s = (sample * 32767.0).clamp(-32768.0, 32767.0) as i16;
        bytes.extend_from_slice(&s.to_le_bytes());
    }
    bytes
}

/// Encode f32 samples to MP3 bytes (mono, 128kbps CBR).
pub fn encode_mp3(samples: &[f32], sample_rate: u32) -> Result<Vec<u8>> {
    use mp3lame_encoder::{Builder, FlushNoGap, InterleavedPcm};

    let mut encoder = Builder::new().ok_or_else(|| {
        VoxtralError::Audio("Failed to create MP3 encoder".to_string())
    })?;
    encoder.set_num_channels(1).map_err(|e| {
        VoxtralError::Audio(format!("MP3 encoder set_num_channels failed: {:?}", e))
    })?;
    encoder.set_sample_rate(sample_rate).map_err(|e| {
        VoxtralError::Audio(format!("MP3 encoder set_sample_rate failed: {:?}", e))
    })?;
    encoder
        .set_brate(mp3lame_encoder::Bitrate::Kbps128)
        .map_err(|e| {
            VoxtralError::Audio(format!("MP3 encoder set_brate failed: {:?}", e))
        })?;
    encoder
        .set_quality(mp3lame_encoder::Quality::Best)
        .map_err(|e| {
            VoxtralError::Audio(format!("MP3 encoder set_quality failed: {:?}", e))
        })?;
    let mut encoder = encoder.build().map_err(|e| {
        VoxtralError::Audio(format!("MP3 encoder build failed: {:?}", e))
    })?;

    // Convert f32 to i16
    let pcm: Vec<i16> = samples
        .iter()
        .map(|&s| (s * 32767.0).clamp(-32768.0, 32767.0) as i16)
        .collect();

    let input = InterleavedPcm(&pcm);
    let mut mp3_out = Vec::with_capacity(mp3lame_encoder::max_required_buffer_size(pcm.len()));

    encoder.encode_to_vec(input, &mut mp3_out).map_err(|e| {
        VoxtralError::Audio(format!("MP3 encoding failed: {:?}", e))
    })?;

    // Flush remaining data
    encoder.flush_to_vec::<FlushNoGap>(&mut mp3_out).map_err(|e| {
        VoxtralError::Audio(format!("MP3 flush failed: {:?}", e))
    })?;

    Ok(mp3_out)
}

/// Encode f32 samples to FLAC bytes (16-bit mono, lossless).
pub fn encode_flac(samples: &[f32], sample_rate: u32) -> Result<Vec<u8>> {
    use flacenc::bitsink::ByteSink;
    use flacenc::component::BitRepr;
    use flacenc::error::Verify;

    let pcm_i32: Vec<i32> = samples
        .iter()
        .map(|&s| (s * 32767.0).clamp(-32768.0, 32767.0) as i32)
        .collect();

    let config = flacenc::config::Encoder::default()
        .into_verified()
        .map_err(|(_enc, e)| VoxtralError::Audio(format!("FLAC config error: {:?}", e)))?;
    let source = flacenc::source::MemSource::from_samples(&pcm_i32, 1, 16, sample_rate as usize);
    let stream = flacenc::encode_with_fixed_block_size(&config, source, config.block_size)
        .map_err(|e| VoxtralError::Audio(format!("FLAC encoding failed: {}", e)))?;

    let mut sink = ByteSink::new();
    stream
        .write(&mut sink)
        .map_err(|e| VoxtralError::Audio(format!("FLAC write failed: {:?}", e)))?;
    Ok(sink.into_inner())
}

/// Encode f32 samples to OGG Opus bytes (48kHz mono).
pub fn encode_ogg_opus(samples: &[f32], sample_rate: u32) -> Result<Vec<u8>> {
    use audiopus::coder::Encoder as OpusEncoder;
    use audiopus::{Application, Channels, SampleRate as OpusSampleRate};

    // Opus requires 48kHz
    let samples_48k = if sample_rate != 48000 {
        resample(samples, sample_rate, 48000)?
    } else {
        samples.to_vec()
    };

    let opus_sr = OpusSampleRate::Hz48000;
    let channels = Channels::Mono;
    let encoder = OpusEncoder::new(opus_sr, channels, Application::Voip).map_err(|e| {
        VoxtralError::Audio(format!("Opus encoder init failed: {}", e))
    })?;

    // OGG page writer
    let mut ogg_buf = Vec::new();
    let serial = 1u32;
    {
        let mut writer = ogg::PacketWriter::new(&mut ogg_buf);

        // OpusHead header (RFC 7845)
        let mut head = Vec::with_capacity(19);
        head.extend_from_slice(b"OpusHead"); // magic
        head.push(1); // version
        head.push(1); // channel count
        head.extend_from_slice(&0u16.to_le_bytes()); // pre-skip
        head.extend_from_slice(&48000u32.to_le_bytes()); // input sample rate
        head.extend_from_slice(&0i16.to_le_bytes()); // output gain
        head.push(0); // channel mapping family
        writer
            .write_packet(head, serial, ogg::PacketWriteEndInfo::EndPage, 0)
            .map_err(|e| VoxtralError::Audio(format!("OGG write head failed: {}", e)))?;

        // OpusTags header
        let vendor = b"voxtral-tts";
        let mut tags = Vec::new();
        tags.extend_from_slice(b"OpusTags");
        tags.extend_from_slice(&(vendor.len() as u32).to_le_bytes());
        tags.extend_from_slice(vendor);
        tags.extend_from_slice(&0u32.to_le_bytes()); // no user comments
        writer
            .write_packet(tags, serial, ogg::PacketWriteEndInfo::EndPage, 0)
            .map_err(|e| VoxtralError::Audio(format!("OGG write tags failed: {}", e)))?;

        // Encode audio in 20ms frames (960 samples at 48kHz)
        let frame_size = 960;
        let mut opus_out = vec![0u8; 4000]; // max opus packet
        let mut granule: u64 = 0;
        let total_frames = (samples_48k.len() + frame_size - 1) / frame_size;

        for (i, chunk) in samples_48k.chunks(frame_size).enumerate() {
            // Pad last frame if needed
            let frame: Vec<f32> = if chunk.len() < frame_size {
                let mut padded = chunk.to_vec();
                padded.resize(frame_size, 0.0);
                padded
            } else {
                chunk.to_vec()
            };

            let encoded_len = encoder.encode_float(&frame, &mut opus_out).map_err(|e| {
                VoxtralError::Audio(format!("Opus encode failed: {}", e))
            })?;

            granule += frame_size as u64;
            let end_info = if i == total_frames - 1 {
                ogg::PacketWriteEndInfo::EndStream
            } else {
                ogg::PacketWriteEndInfo::NormalPacket
            };

            writer
                .write_packet(
                    opus_out[..encoded_len].to_vec(),
                    serial,
                    end_info,
                    granule,
                )
                .map_err(|e| VoxtralError::Audio(format!("OGG write failed: {}", e)))?;
        }
    }

    Ok(ogg_buf)
}

/// Encode audio samples to the specified format.
/// Returns (encoded_bytes, content_type).
pub fn encode_audio(samples: &[f32], sample_rate: u32, format: &str) -> Result<(Vec<u8>, &'static str)> {
    match format {
        "wav" => {
            let bytes = write_wav_bytes(samples, sample_rate)?;
            Ok((bytes, "audio/wav"))
        }
        "pcm" => {
            let bytes = encode_pcm_i16(samples);
            Ok((bytes, "audio/pcm"))
        }
        "mp3" => {
            let bytes = encode_mp3(samples, sample_rate)?;
            Ok((bytes, "audio/mpeg"))
        }
        "flac" => {
            let bytes = encode_flac(samples, sample_rate)?;
            Ok((bytes, "audio/flac"))
        }
        "ogg" | "opus" => {
            let bytes = encode_ogg_opus(samples, sample_rate)?;
            Ok((bytes, "audio/ogg"))
        }
        _ => Err(VoxtralError::Audio(format!(
            "Unsupported format: {}. Supported: wav, pcm, mp3, flac, ogg",
            format
        ))),
    }
}

/// Resample audio from source_sr to target_sr using high-quality sinc interpolation.
pub fn resample(samples: &[f32], source_sr: u32, target_sr: u32) -> Result<Vec<f32>> {
    if source_sr == target_sr {
        return Ok(samples.to_vec());
    }

    use rubato::{
        Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
    };

    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };

    let mut resampler = SincFixedIn::<f32>::new(
        target_sr as f64 / source_sr as f64,
        2.0,
        params,
        samples.len(),
        1,
    )
    .map_err(|e| VoxtralError::Audio(format!("Resampler init failed: {}", e)))?;

    let input = vec![samples.to_vec()];
    let output = resampler
        .process(&input, None)
        .map_err(|e| VoxtralError::Audio(format!("Resampling failed: {}", e)))?;

    Ok(output.into_iter().next().unwrap_or_default())
}

/// Load and preprocess audio for the codec: load, resample to 24kHz, normalize.
pub fn load_and_preprocess(path: &str) -> Result<Vec<f32>> {
    let (samples, sr) = load_wav_file(path)?;

    // Resample to 24kHz if needed
    let samples = resample(&samples, sr, DEFAULT_SAMPLE_RATE)?;

    // Peak normalization to 0.95
    let max_abs = samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    let samples = if max_abs > 0.0 {
        let scale = 0.95 / max_abs;
        samples.iter().map(|&s| s * scale).collect()
    } else {
        samples
    };

    Ok(samples)
}
