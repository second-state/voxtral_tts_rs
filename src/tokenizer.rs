//! Pure Rust implementation of the Tekken BPE tokenizer.
//!
//! Loads from `tekken.json` which contains a vocabulary of 131,072 tokens
//! with base64-encoded byte sequences and merge ranks.

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

/// Wrapper to handle both possible top-level JSON formats.
///
/// Format A: `{ "vocab": [ ... ] }`
/// Format B: bare array `[ ... ]`
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum TekkenJson {
    /// Object with a `vocab` key containing the entry array.
    WithVocab { vocab: Vec<VocabEntry> },
    /// Bare array of entries.
    BareArray(Vec<VocabEntry>),
}

/// Pure-Rust Tekken BPE tokenizer.
pub struct TekkenTokenizer {
    /// Encode: byte sequence -> token ID.
    token_to_id: HashMap<Vec<u8>, u32>,
    /// Decode: token ID -> byte sequence.
    id_to_bytes: Vec<Option<Vec<u8>>>,
    /// BPE merge pairs ordered by rank (index 0 = highest priority).
    /// Each entry is `(left_bytes, right_bytes)`.
    #[allow(dead_code)]
    merges: Vec<(Vec<u8>, Vec<u8>)>,
    /// Reverse lookup: merged pair -> rank (lower = higher priority).
    merge_rank: HashMap<(Vec<u8>, Vec<u8>), usize>,
    /// Total vocabulary size.
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

        let mut entries = match parsed {
            TekkenJson::WithVocab { vocab } => vocab,
            TekkenJson::BareArray(arr) => arr,
        };

        // Cap to model vocab size — tekken.json may have more entries than the
        // model's embedding table (e.g. 150K entries vs 131072 embeddings).
        if let Some(max) = max_vocab {
            entries.sort_by_key(|e| e.rank);
            entries.truncate(max);
        }

        Self::from_entries(entries)
    }

    /// Load the tokenizer from `<model_dir>/tekken.json`, capping to `max_vocab` entries.
    pub fn from_dir(model_dir: &Path, max_vocab: Option<usize>) -> Result<Self> {
        Self::from_file(&model_dir.join("tekken.json"), max_vocab)
    }

    /// Build internal data structures from parsed vocabulary entries.
    fn from_entries(mut entries: Vec<VocabEntry>) -> Result<Self> {
        let engine = base64::engine::general_purpose::STANDARD;

        // Sort entries by rank (ascending) so that index == priority order.
        entries.sort_by_key(|e| e.rank);

        let vocab_size = entries.len();

        // Pre-allocate decode table.
        let mut id_to_bytes: Vec<Option<Vec<u8>>> = vec![None; vocab_size];
        let mut token_to_id: HashMap<Vec<u8>, u32> = HashMap::with_capacity(vocab_size);

        // First 256 entries (ranks 0..255) are typically the raw byte tokens.
        // Entries beyond that are merge results; we derive merge pairs from them.
        let mut merges: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
        let mut merge_rank: HashMap<(Vec<u8>, Vec<u8>), usize> = HashMap::new();

        // Collect all token byte sequences keyed by rank (== token ID).
        let mut rank_to_bytes: Vec<Vec<u8>> = Vec::with_capacity(vocab_size);

        for entry in &entries {
            let bytes = engine.decode(&entry.token_bytes).map_err(|e| {
                VoxtralError::Tokenizer(format!(
                    "Failed to decode base64 for rank {}: {}",
                    entry.rank, e
                ))
            })?;
            rank_to_bytes.push(bytes);
        }

        // Populate forward / reverse maps using rank as token ID.
        for (idx, bytes) in rank_to_bytes.iter().enumerate() {
            let id = idx as u32;
            id_to_bytes[idx] = Some(bytes.clone());
            token_to_id.insert(bytes.clone(), id);
        }

        // Build merge table. For every token whose byte sequence is >1 byte
        // we find the best split into two existing tokens (both of which must
        // have a *lower* rank) and record that as a merge pair.
        for (idx, bytes) in rank_to_bytes.iter().enumerate() {
            if bytes.len() <= 1 {
                continue;
            }

            // Try all possible split points and pick the one where both halves
            // are known tokens with the maximum combined rank being minimised.
            let mut best_split: Option<(usize, u32)> = None; // (split_pos, max_rank_of_halves)

            for split in 1..bytes.len() {
                let left = &bytes[..split];
                let right = &bytes[split..];

                if let (Some(&lid), Some(&rid)) = (token_to_id.get(left), token_to_id.get(right)) {
                    // Both halves must have strictly lower rank (== earlier in vocab).
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
            "Tokenizer loaded: {} tokens, {} merges",
            vocab_size,
            merges.len()
        );

        Ok(Self {
            token_to_id,
            id_to_bytes,
            merges,
            merge_rank,
            vocab_size,
        })
    }

    // --------------------------------------------------------------------
    // Encoding
    // --------------------------------------------------------------------

    /// Encode `text` to a sequence of token IDs using BPE.
    ///
    /// Algorithm:
    /// 1. Convert the text to its UTF-8 byte representation.
    /// 2. Start with each byte as its own token (byte tokens are assumed to
    ///    be the first 256 entries in the vocabulary).
    /// 3. Iteratively merge the adjacent pair with the lowest merge rank
    ///    (highest priority) until no more merges are possible.
    /// 4. Map the resulting byte sequences to their token IDs.
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

        // Map segments to token IDs.
        segments
            .iter()
            .map(|seg| {
                self.token_to_id.get(seg).copied().unwrap_or_else(|| {
                    // Fallback: encode as individual byte tokens.
                    // This should not happen if the vocab is complete, but
                    // it prevents panics.
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

    /// Return the vocabulary size.
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
        // Build a minimal tokenizer with only byte tokens (no merges).
        let mut entries = Vec::new();
        for i in 0u32..256 {
            let token_bytes = base64::engine::general_purpose::STANDARD.encode([i as u8]);
            entries.push(VocabEntry {
                rank: i,
                token_bytes,
            });
        }
        let tok = TekkenTokenizer::from_entries(entries).unwrap();

        let text = "hello world";
        let ids = tok.encode(text);
        assert_eq!(ids.len(), text.len());
        let decoded = tok.decode(&ids);
        assert_eq!(decoded, text);
    }
}
