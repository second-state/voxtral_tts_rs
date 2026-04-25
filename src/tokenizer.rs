//! Pure Rust implementation of the Tekken BPE tokenizer.
//!
//! Loads from `tekken.json` which contains BPE vocabulary entries, special tokens,
//! and configuration. BPE token IDs are offset by `num_special_tokens` so that
//! special/control tokens occupy IDs 0..(num_special_tokens-1) and BPE tokens
//! start at `num_special_tokens`.

use std::collections::HashMap;
use std::path::Path;

use base64::Engine;
use serde::Deserialize;

use crate::error::{Result, VoxtralError};

/// A single vocabulary entry as stored in tekken.json.
#[derive(Debug, Deserialize)]
struct VocabEntry {
    /// Merge rank (lower = higher priority). Also used as the sort key.
    rank: u32,
    /// Base64-encoded byte sequence for this token.
    token_bytes: String,
    // There may be other fields like `token_str`; we ignore them.
}

/// Config section of tekken.json (v7).
#[derive(Debug, Deserialize)]
struct TekkenConfig {
    #[serde(default = "default_num_special")]
    default_num_special_tokens: usize,
    #[serde(default)]
    #[allow(dead_code)]
    default_vocab_size: usize,
}

fn default_num_special() -> usize {
    0
}

/// Full tekken.json with config, vocab, and special_tokens.
#[derive(Debug, Deserialize)]
struct TekkenJsonFull {
    #[serde(default)]
    config: Option<TekkenConfig>,
    vocab: Vec<VocabEntry>,
}

/// Wrapper to handle both possible top-level JSON formats.
///
/// Format A: `{ "config": ..., "vocab": [...], "special_tokens": [...] }`
/// Format B: bare array `[ ... ]`
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum TekkenJson {
    /// Object with config and vocab keys.
    Full(TekkenJsonFull),
    /// Bare array of entries (legacy format).
    BareArray(Vec<VocabEntry>),
}

/// Pure-Rust Tekken BPE tokenizer.
pub struct TekkenTokenizer {
    /// Encode: byte sequence -> BPE rank (internal, before offset).
    token_to_rank: HashMap<Vec<u8>, u32>,
    /// Decode: final token ID -> byte sequence.
    id_to_bytes: Vec<Option<Vec<u8>>>,
    /// BPE merge pairs ordered by rank (index 0 = highest priority).
    /// Each entry is `(left_bytes, right_bytes)`.
    #[allow(dead_code)]
    merges: Vec<(Vec<u8>, Vec<u8>)>,
    /// Reverse lookup: merged pair -> rank (lower = higher priority).
    merge_rank: HashMap<(Vec<u8>, Vec<u8>), usize>,
    /// Number of special tokens that precede BPE tokens in the model vocab.
    num_special_tokens: u32,
    /// Total vocabulary size (special + BPE).
    vocab_size: usize,
}

impl TekkenTokenizer {
    // --------------------------------------------------------------------
    // Construction
    // --------------------------------------------------------------------

    /// Load the tokenizer from a `tekken.json` file, capping to `max_vocab` entries.
    /// If `max_vocab` is `None`, all entries are loaded.
    pub fn from_file(path: &Path, max_vocab: Option<usize>) -> Result<Self> {
        let data = std::fs::read_to_string(path).map_err(|e| {
            VoxtralError::Tokenizer(format!("Failed to read {}: {}", path.display(), e))
        })?;

        let parsed: TekkenJson = serde_json::from_str(&data)
            .map_err(|e| VoxtralError::Tokenizer(format!("Failed to parse tekken.json: {}", e)))?;

        let (mut entries, num_special) = match parsed {
            TekkenJson::Full(full) => {
                let num_special = full
                    .config
                    .as_ref()
                    .map(|c| c.default_num_special_tokens)
                    .unwrap_or(0);
                (full.vocab, num_special)
            }
            TekkenJson::BareArray(arr) => (arr, 0),
        };

        // Cap BPE entries so total (special + BPE) fits within max_vocab.
        if let Some(max) = max_vocab {
            entries.sort_by_key(|e| e.rank);
            let max_bpe = max.saturating_sub(num_special);
            entries.truncate(max_bpe);
        }

        tracing::info!(
            "Tekken tokenizer: {} BPE tokens, {} special tokens (offset)",
            entries.len(),
            num_special,
        );

        Self::from_entries(entries, num_special as u32)
    }

    /// Load the tokenizer from `<model_dir>/tekken.json`, capping to `max_vocab` entries.
    pub fn from_dir(model_dir: &Path, max_vocab: Option<usize>) -> Result<Self> {
        Self::from_file(&model_dir.join("tekken.json"), max_vocab)
    }

    /// Build internal data structures from parsed vocabulary entries.
    fn from_entries(mut entries: Vec<VocabEntry>, num_special_tokens: u32) -> Result<Self> {
        let engine = base64::engine::general_purpose::STANDARD;

        // Sort entries by rank (ascending) so that index == priority order.
        entries.sort_by_key(|e| e.rank);

        let bpe_count = entries.len();
        let total_vocab = num_special_tokens as usize + bpe_count;

        // Pre-allocate decode table (indexed by final token ID).
        let mut id_to_bytes: Vec<Option<Vec<u8>>> = vec![None; total_vocab];
        // BPE internal mapping: byte sequence -> BPE rank (0-based, before offset)
        let mut token_to_rank: HashMap<Vec<u8>, u32> = HashMap::with_capacity(bpe_count);

        let mut merges: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
        let mut merge_rank: HashMap<(Vec<u8>, Vec<u8>), usize> = HashMap::new();

        // Collect all token byte sequences keyed by BPE rank.
        let mut rank_to_bytes: Vec<Vec<u8>> = Vec::with_capacity(bpe_count);

        for entry in &entries {
            let bytes = engine.decode(&entry.token_bytes).map_err(|e| {
                VoxtralError::Tokenizer(format!(
                    "Failed to decode base64 for rank {}: {}",
                    entry.rank, e
                ))
            })?;
            rank_to_bytes.push(bytes);
        }

        // Populate forward / reverse maps.
        // BPE rank `r` maps to final token ID `r + num_special_tokens`.
        for (rank, bytes) in rank_to_bytes.iter().enumerate() {
            let final_id = rank as u32 + num_special_tokens;
            if (final_id as usize) < id_to_bytes.len() {
                id_to_bytes[final_id as usize] = Some(bytes.clone());
            }
            token_to_rank.insert(bytes.clone(), rank as u32);
        }

        // Build merge table. For every token whose byte sequence is >1 byte
        // we find the best split into two existing tokens (both of which must
        // have a *lower* rank) and record that as a merge pair.
        for (idx, bytes) in rank_to_bytes.iter().enumerate() {
            if bytes.len() <= 1 {
                continue;
            }

            let mut best_split: Option<(usize, u32)> = None;

            for split in 1..bytes.len() {
                let left = &bytes[..split];
                let right = &bytes[split..];

                if let (Some(&lid), Some(&rid)) =
                    (token_to_rank.get(left), token_to_rank.get(right))
                {
                    if (lid as usize) < idx && (rid as usize) < idx {
                        let worst = lid.max(rid);
                        if best_split.is_none() || worst < best_split.unwrap().1 {
                            best_split = Some((split, worst));
                        }
                    }
                }
            }

            if let Some((split, _)) = best_split {
                let left = bytes[..split].to_vec();
                let right = bytes[split..].to_vec();
                let rank = merges.len();
                merge_rank.insert((left.clone(), right.clone()), rank);
                merges.push((left, right));
            }
        }

        tracing::debug!(
            "Tokenizer loaded: {} BPE tokens + {} special = {} total, {} merges",
            bpe_count,
            num_special_tokens,
            total_vocab,
            merges.len()
        );

        Ok(Self {
            token_to_rank,
            id_to_bytes,
            merges,
            merge_rank,
            num_special_tokens,
            vocab_size: total_vocab,
        })
    }

    // --------------------------------------------------------------------
    // Encoding
    // --------------------------------------------------------------------

    /// Encode `text` to a sequence of token IDs using BPE.
    ///
    /// Returns final token IDs (BPE ranks + num_special_tokens offset).
    pub fn encode(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return Vec::new();
        }

        let bytes = text.as_bytes();

        // Initialise: each byte is its own segment.
        let mut segments: Vec<Vec<u8>> = bytes.iter().map(|&b| vec![b]).collect();

        // Iterative merge loop.
        loop {
            if segments.len() < 2 {
                break;
            }

            // Find the adjacent pair with the lowest (best) merge rank.
            let mut best_rank: Option<usize> = None;
            let mut best_idx: usize = 0;

            for i in 0..segments.len() - 1 {
                let pair = (segments[i].clone(), segments[i + 1].clone());
                if let Some(&rank) = self.merge_rank.get(&pair) {
                    if best_rank.is_none() || rank < best_rank.unwrap() {
                        best_rank = Some(rank);
                        best_idx = i;
                    }
                }
            }

            // No mergeable pair found; we are done.
            let Some(_) = best_rank else {
                break;
            };

            // Merge all occurrences of the best pair in a single pass
            // (left-to-right, non-overlapping).
            let best_left = segments[best_idx].clone();
            let best_right = segments[best_idx + 1].clone();

            let mut new_segments: Vec<Vec<u8>> = Vec::with_capacity(segments.len());
            let mut i = 0;
            while i < segments.len() {
                if i + 1 < segments.len()
                    && segments[i] == best_left
                    && segments[i + 1] == best_right
                {
                    let mut merged = segments[i].clone();
                    merged.extend_from_slice(&segments[i + 1]);
                    new_segments.push(merged);
                    i += 2;
                } else {
                    new_segments.push(segments[i].clone());
                    i += 1;
                }
            }
            segments = new_segments;
        }

        // Map segments to final token IDs (BPE rank + offset).
        let offset = self.num_special_tokens;
        segments
            .iter()
            .map(|seg| {
                self.token_to_rank.get(seg).copied().map(|r| r + offset).unwrap_or_else(|| {
                    tracing::warn!("Token not found for byte sequence of length {}", seg.len());
                    0
                })
            })
            .collect()
    }

    // --------------------------------------------------------------------
    // Decoding
    // --------------------------------------------------------------------

    /// Decode a sequence of token IDs back to a string.
    ///
    /// Unknown token IDs are silently skipped. The resulting bytes are
    /// interpreted as UTF-8 with lossy replacement.
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes: Vec<u8> = Vec::new();

        for &id in ids {
            let idx = id as usize;
            if idx < self.id_to_bytes.len() {
                if let Some(ref b) = self.id_to_bytes[idx] {
                    bytes.extend_from_slice(b);
                }
            }
        }

        String::from_utf8_lossy(&bytes).into_owned()
    }

    // --------------------------------------------------------------------
    // Accessors
    // --------------------------------------------------------------------

    /// Return the vocabulary size (special + BPE).
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Smoke test: round-trip encode -> decode should recover the original text
    /// (for ASCII text that maps cleanly to byte tokens).
    #[test]
    fn test_byte_level_roundtrip() {
        // Build a minimal tokenizer with only byte tokens (no merges), no offset.
        let mut entries = Vec::new();
        for i in 0u32..256 {
            let token_bytes = base64::engine::general_purpose::STANDARD.encode([i as u8]);
            entries.push(VocabEntry {
                rank: i,
                token_bytes,
            });
        }
        let tok = TekkenTokenizer::from_entries(entries, 0).unwrap();

        let text = "hello world";
        let ids = tok.encode(text);
        assert_eq!(ids.len(), text.len());
        let decoded = tok.decode(&ids);
        assert_eq!(decoded, text);
    }

    /// Test that BPE token IDs include the special token offset.
    #[test]
    fn test_special_token_offset() {
        let mut entries = Vec::new();
        for i in 0u32..256 {
            let token_bytes = base64::engine::general_purpose::STANDARD.encode([i as u8]);
            entries.push(VocabEntry {
                rank: i,
                token_bytes,
            });
        }
        let tok = TekkenTokenizer::from_entries(entries, 1000).unwrap();

        // 'A' = 0x41 = 65, with offset 1000 should be 1065
        let ids = tok.encode("A");
        assert_eq!(ids.len(), 1);
        assert_eq!(ids[0], 1065);
    }
}
