//! Vocabulary management for BPE tokenizers.
//!
//! A `Vocab` maps between token byte sequences and their integer ids.
//! The byte-level representation allows the tokenizer to handle arbitrary
//! UTF-8 text including non-ASCII characters without unknown token
//! penalties.

use std::collections::HashMap;

use crate::error::{LmError, LmResult};

// ─── Vocab ───────────────────────────────────────────────────────────────────

/// Token vocabulary: bidirectional mapping between byte sequences and ids.
///
/// Token ids are dense unsigned 32-bit integers starting at 0.
/// The first 256 tokens (ids 0–255) are always single-byte tokens
/// (bytes 0x00–0xFF) for byte-level BPE vocabularies.
#[derive(Debug, Clone)]
pub struct Vocab {
    /// `id_to_bytes[i]` is the byte sequence for token i.
    id_to_bytes: Vec<Vec<u8>>,
    /// Reverse map.
    bytes_to_id: HashMap<Vec<u8>, u32>,
    /// Named special tokens (e.g., `"<|endoftext|>"` for GPT-2,
    /// `"<s>"` and `"</s>"` for LLaMA).
    special_tokens: HashMap<String, u32>,
}

impl Vocab {
    // ── Constructors ─────────────────────────────────────────────────────

    /// Build a vocabulary from an ordered list of byte sequences.
    ///
    /// The special token map may reference any existing index in `tokens`.
    /// Duplicate byte sequences are detected and cause an error.
    pub fn from_tokens(
        tokens: Vec<Vec<u8>>,
        special_tokens: HashMap<String, u32>,
    ) -> LmResult<Self> {
        let mut bytes_to_id = HashMap::with_capacity(tokens.len());
        for (id, bytes) in tokens.iter().enumerate() {
            if bytes_to_id.insert(bytes.clone(), id as u32).is_some() {
                return Err(LmError::InvalidConfig {
                    msg: format!("duplicate token bytes at id {id}"),
                });
            }
        }
        // Validate special token ids are in range.
        for (name, &id) in &special_tokens {
            if id as usize >= tokens.len() {
                return Err(LmError::OutOfVocab { token: id });
            }
            let _ = name; // consumed for validation only
        }
        Ok(Self {
            id_to_bytes: tokens,
            bytes_to_id,
            special_tokens,
        })
    }

    /// Standard GPT-2 / byte-level BPE base vocabulary (256 single-byte tokens).
    ///
    /// No merge tokens or special tokens are included; callers should add
    /// merged and special tokens on top via [`Self::with_extra_tokens`].
    pub fn gpt2_byte_vocab() -> Self {
        let tokens: Vec<Vec<u8>> = (0u8..=255).map(|b| vec![b]).collect();
        let mut bytes_to_id = HashMap::with_capacity(256);
        for (id, bytes) in tokens.iter().enumerate() {
            bytes_to_id.insert(bytes.clone(), id as u32);
        }
        Self {
            id_to_bytes: tokens,
            bytes_to_id,
            special_tokens: HashMap::new(),
        }
    }

    /// Return a clone with additional tokens appended.
    ///
    /// `extra` is a list of byte sequences to add; they receive ids
    /// starting at `self.size()`.  `extra_special` maps names to their
    /// absolute ids (which must be within the final vocabulary range).
    pub fn with_extra_tokens(
        &self,
        extra: Vec<Vec<u8>>,
        extra_special: HashMap<String, u32>,
    ) -> LmResult<Self> {
        let mut tokens = self.id_to_bytes.clone();
        for bytes in extra {
            tokens.push(bytes);
        }
        let mut special = self.special_tokens.clone();
        special.extend(extra_special);
        Self::from_tokens(tokens, special)
    }

    // ── Accessors ─────────────────────────────────────────────────────────

    /// Total vocabulary size.
    pub fn size(&self) -> usize {
        self.id_to_bytes.len()
    }

    /// Look up the id for a byte sequence; returns `None` if absent.
    pub fn bytes_to_id(&self, bytes: &[u8]) -> Option<u32> {
        self.bytes_to_id.get(bytes).copied()
    }

    /// Byte sequence for a token id.
    pub fn id_to_bytes(&self, id: u32) -> LmResult<&[u8]> {
        self.id_to_bytes
            .get(id as usize)
            .map(|v| v.as_slice())
            .ok_or(LmError::OutOfVocab { token: id })
    }

    /// Decode a single token id to a UTF-8 `String`.
    ///
    /// Returns an error if the byte sequence is not valid UTF-8.
    pub fn decode_token(&self, id: u32) -> LmResult<String> {
        let bytes = self.id_to_bytes(id)?;
        String::from_utf8(bytes.to_vec()).map_err(|_| LmError::Utf8Decode { token: id })
    }

    /// Look up the id of a special token by name.
    pub fn special_id(&self, name: &str) -> Option<u32> {
        self.special_tokens.get(name).copied()
    }

    /// Iterate over all `(name, id)` special token pairs.
    pub fn special_tokens(&self) -> impl Iterator<Item = (&str, u32)> {
        self.special_tokens.iter().map(|(k, &v)| (k.as_str(), v))
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn small_vocab() -> Vocab {
        // tokens: 'a'=0, 'b'=1, 'c'=2, "ab"=3, "abc"=4
        let tokens = vec![
            vec![b'a'],
            vec![b'b'],
            vec![b'c'],
            vec![b'a', b'b'],
            vec![b'a', b'b', b'c'],
        ];
        let special: HashMap<String, u32> = [("<eos>".into(), 2u32)].into_iter().collect();
        Vocab::from_tokens(tokens, special).unwrap()
    }

    #[test]
    fn vocab_size() {
        assert_eq!(small_vocab().size(), 5);
    }

    #[test]
    fn bytes_to_id_found() {
        let v = small_vocab();
        assert_eq!(v.bytes_to_id(b"a"), Some(0));
        assert_eq!(v.bytes_to_id(b"ab"), Some(3));
    }

    #[test]
    fn bytes_to_id_not_found() {
        let v = small_vocab();
        assert_eq!(v.bytes_to_id(b"d"), None);
    }

    #[test]
    fn id_to_bytes_ok() {
        let v = small_vocab();
        assert_eq!(v.id_to_bytes(0).unwrap(), b"a");
        assert_eq!(v.id_to_bytes(4).unwrap(), b"abc");
    }

    #[test]
    fn id_to_bytes_out_of_range() {
        let v = small_vocab();
        assert!(matches!(
            v.id_to_bytes(99),
            Err(LmError::OutOfVocab { token: 99 })
        ));
    }

    #[test]
    fn decode_token_ascii() {
        let v = small_vocab();
        assert_eq!(v.decode_token(0).unwrap(), "a");
        assert_eq!(v.decode_token(3).unwrap(), "ab");
    }

    #[test]
    fn decode_token_utf8_error() {
        // Build a vocab with a token whose bytes are not valid UTF-8.
        let tokens = vec![vec![0xFF_u8]];
        let v = Vocab::from_tokens(tokens, HashMap::new()).unwrap();
        assert!(matches!(
            v.decode_token(0),
            Err(LmError::Utf8Decode { token: 0 })
        ));
    }

    #[test]
    fn special_id_lookup() {
        let v = small_vocab();
        assert_eq!(v.special_id("<eos>"), Some(2));
        assert_eq!(v.special_id("<unk>"), None);
    }

    #[test]
    fn gpt2_byte_vocab_size() {
        let v = Vocab::gpt2_byte_vocab();
        assert_eq!(v.size(), 256);
        // Byte 65 ('A') maps to id 65
        assert_eq!(v.bytes_to_id(&[65_u8]), Some(65));
    }

    #[test]
    fn with_extra_tokens_appends() {
        let base = Vocab::gpt2_byte_vocab();
        let extra = vec![vec![b'a', b'b']]; // "ab" → id 256
        let extra_special: HashMap<String, u32> = [("<eos>".into(), 256u32)].into_iter().collect();
        let v = base.with_extra_tokens(extra, extra_special).unwrap();
        assert_eq!(v.size(), 257);
        assert_eq!(v.bytes_to_id(b"ab"), Some(256));
        assert_eq!(v.special_id("<eos>"), Some(256));
    }

    #[test]
    fn duplicate_token_errors() {
        let tokens = vec![vec![b'a'], vec![b'a']]; // duplicate
        assert!(matches!(
            Vocab::from_tokens(tokens, HashMap::new()),
            Err(LmError::InvalidConfig { .. })
        ));
    }

    #[test]
    fn special_out_of_range_errors() {
        let tokens = vec![vec![b'a']];
        let special: HashMap<String, u32> = [("<eos>".into(), 99u32)].into_iter().collect();
        assert!(matches!(
            Vocab::from_tokens(tokens, special),
            Err(LmError::OutOfVocab { token: 99 })
        ));
    }
}
