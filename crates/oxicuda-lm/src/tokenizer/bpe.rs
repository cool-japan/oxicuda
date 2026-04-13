//! Byte-Pair Encoding (BPE) tokenizer.
//!
//! Implements the GPT-2 style byte-level BPE algorithm:
//!
//! 1. **Pre-tokenisation**: the input text is converted to a sequence of byte
//!    values (0–255) treated as individual tokens.
//! 2. **Merge iteration**: pairs are merged in priority order (lowest rank
//!    first) until no mergeable pair remains.
//! 3. **Decode**: token ids are concatenated back to bytes and decoded as
//!    UTF-8.
//!
//! # Vocabulary layout
//!
//! Token ids 0–255 represent single bytes.  Merged tokens start at id 256.
//! Special tokens (e.g., `<|endoftext|>`, `<s>`, `</s>`) are appended at the
//! end of the vocabulary.
//!
//! # Limitations
//!
//! This implementation does **not** include the GPT-2 regex pre-tokenisation
//! split (which splits on whitespace / punctuation boundaries before applying
//! BPE within each word).  That split is language-specific and requires an
//! external regex crate.  The pure byte-level algorithm here is correct for
//! all BPE vocabularies and matches the SentencePiece / LLaMA tokeniser
//! approach.

use std::collections::HashMap;

use crate::error::{LmError, LmResult};
use crate::tokenizer::vocab::Vocab;

// ─── BpeTokenizer ────────────────────────────────────────────────────────────

/// Byte-Pair Encoding tokenizer.
///
/// Created from a [`Vocab`] and an ordered list of merge rules.
/// The merge rules are pairs of byte sequences; the `k`-th merge pair
/// has rank `k` (lower rank = higher priority).
#[derive(Debug, Clone)]
pub struct BpeTokenizer {
    vocab: Vocab,
    /// `merge_ranks[(a, b)]` = rank of merging tokens a and b.
    merge_ranks: HashMap<(u32, u32), u32>,
    /// `pair_to_merged[(a, b)]` = token id of the merged token.
    pair_to_merged: HashMap<(u32, u32), u32>,
}

impl BpeTokenizer {
    // ── Constructor ──────────────────────────────────────────────────────

    /// Build a tokenizer from a vocabulary and an ordered merge list.
    ///
    /// `merges` is a list of byte-sequence pairs in priority order
    /// (index 0 = highest priority = rank 0).  Each pair
    /// `(left_bytes, right_bytes)` defines a merge rule where the
    /// two tokens whose byte sequences are `left_bytes` and `right_bytes`
    /// merge into the token whose byte sequence is `left_bytes ++ right_bytes`.
    ///
    /// # Errors
    ///
    /// Returns [`LmError::InvalidMergePair`] if any merge pair references
    /// bytes that are absent from the vocabulary.
    pub fn new(vocab: Vocab, merges: Vec<(Vec<u8>, Vec<u8>)>) -> LmResult<Self> {
        let n_merges = merges.len();
        let mut merge_ranks = HashMap::with_capacity(n_merges);
        let mut pair_to_merged = HashMap::with_capacity(n_merges);

        for (rank, (left_bytes, right_bytes)) in merges.into_iter().enumerate() {
            let a = vocab
                .bytes_to_id(&left_bytes)
                .ok_or(LmError::InvalidMergePair {
                    a: u32::MAX,
                    b: u32::MAX,
                })?;
            let b = vocab
                .bytes_to_id(&right_bytes)
                .ok_or(LmError::InvalidMergePair { a, b: u32::MAX })?;

            // The merged token's bytes are the concatenation.
            let mut merged_bytes = left_bytes.clone();
            merged_bytes.extend_from_slice(&right_bytes);
            let merged = vocab
                .bytes_to_id(&merged_bytes)
                .ok_or(LmError::InvalidMergePair { a, b })?;

            merge_ranks.insert((a, b), rank as u32);
            pair_to_merged.insert((a, b), merged);
        }

        Ok(Self {
            vocab,
            merge_ranks,
            pair_to_merged,
        })
    }

    // ── Public API ───────────────────────────────────────────────────────

    /// Vocabulary size (total token count).
    pub fn vocab_size(&self) -> usize {
        self.vocab.size()
    }

    /// Id of a special token by name; `None` if absent.
    pub fn special_id(&self, name: &str) -> Option<u32> {
        self.vocab.special_id(name)
    }

    /// Encode `text` into a sequence of token ids.
    ///
    /// The encoding is byte-level: the text is converted to raw bytes,
    /// each byte becomes a single-byte token (id = byte value), and
    /// BPE merges are applied in priority order until no more merges apply.
    pub fn encode(&self, text: &str) -> LmResult<Vec<u32>> {
        if text.is_empty() {
            return Ok(vec![]);
        }

        // Initialize: look up each byte in the vocabulary so that the token
        // id matches the vocab layout (which may not be identity for non-GPT-2
        // byte vocabularies).
        let mut tokens: Vec<u32> = text
            .as_bytes()
            .iter()
            .map(|&b| {
                self.vocab
                    .bytes_to_id(&[b])
                    .ok_or(LmError::OutOfVocab { token: b as u32 })
            })
            .collect::<LmResult<Vec<u32>>>()?;

        // Iterative BPE: find the lowest-rank mergeable pair and apply it.
        loop {
            // Find best (lowest-rank) pair.
            let mut best_rank = u32::MAX;
            let mut best_pos: Option<usize> = None;

            for i in 0..tokens.len().saturating_sub(1) {
                let pair = (tokens[i], tokens[i + 1]);
                if let Some(&rank) = self.merge_ranks.get(&pair) {
                    if rank < best_rank {
                        best_rank = rank;
                        best_pos = Some(i);
                    }
                }
            }

            match best_pos {
                None => break, // No more applicable merges.
                Some(pos) => {
                    let pair = (tokens[pos], tokens[pos + 1]);
                    let merged =
                        *self
                            .pair_to_merged
                            .get(&pair)
                            .ok_or(LmError::InvalidMergePair {
                                a: pair.0,
                                b: pair.1,
                            })?;
                    tokens[pos] = merged;
                    tokens.remove(pos + 1);
                }
            }
        }

        Ok(tokens)
    }

    /// Encode `text`, optionally prepending a BOS and appending an EOS token.
    ///
    /// `bos_name` and `eos_name` are special token names (e.g., `"<s>"`,
    /// `"</s>"`, `"<|endoftext|>"`).  The relevant token must exist in the
    /// vocabulary's special token map; if not, encoding silently skips it.
    pub fn encode_with_special(
        &self,
        text: &str,
        bos_name: Option<&str>,
        eos_name: Option<&str>,
    ) -> LmResult<Vec<u32>> {
        let mut ids = Vec::new();
        if let Some(name) = bos_name {
            if let Some(id) = self.vocab.special_id(name) {
                ids.push(id);
            }
        }
        ids.extend(self.encode(text)?);
        if let Some(name) = eos_name {
            if let Some(id) = self.vocab.special_id(name) {
                ids.push(id);
            }
        }
        Ok(ids)
    }

    /// Decode a sequence of token ids back to a `String`.
    ///
    /// Concatenates the byte sequences for each id and interprets the result
    /// as UTF-8.  Returns [`LmError::Utf8Decode`] if the byte stream is not
    /// valid UTF-8.
    pub fn decode(&self, ids: &[u32]) -> LmResult<String> {
        let mut bytes = Vec::new();
        for &id in ids {
            let tok_bytes = self.vocab.id_to_bytes(id)?;
            bytes.extend_from_slice(tok_bytes);
        }
        String::from_utf8(bytes).map_err(|_| {
            // Return the first non-decodable token id as context.
            LmError::Utf8Decode {
                token: ids.first().copied().unwrap_or(0),
            }
        })
    }

    /// Decode a single token id.
    pub fn decode_one(&self, id: u32) -> LmResult<String> {
        self.decode(&[id])
    }

    /// Expose the underlying vocabulary for introspection.
    pub fn vocab(&self) -> &Vocab {
        &self.vocab
    }
}

// ─── Builder helper ──────────────────────────────────────────────────────────

/// Convenience builder for test tokenizers.
///
/// Starts with a byte-level vocabulary (256 single-byte tokens),
/// then adds merged tokens and merge rules in order.
#[derive(Debug, Default)]
pub struct BpeBuilder {
    /// `(left_bytes, right_bytes)` pairs in priority order (index 0 = rank 0).
    merges: Vec<(Vec<u8>, Vec<u8>)>,
    /// Extra tokens appended after the 256 byte tokens.
    extra_tokens: Vec<Vec<u8>>,
    /// Special token name → id.
    special: HashMap<String, u32>,
}

impl BpeBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a merged token and its merge rule.
    ///
    /// `left` and `right` are the constituent byte sequences.
    /// The merged token's bytes are `left ++ right` and it receives
    /// the next available id (256 + number of previously added extras).
    pub fn add_merge(mut self, left: &[u8], right: &[u8]) -> Self {
        let mut merged = left.to_vec();
        merged.extend_from_slice(right);
        self.extra_tokens.push(merged);
        self.merges.push((left.to_vec(), right.to_vec()));
        self
    }

    /// Register a special token; `id` must be a valid vocabulary id
    /// at construction time.
    pub fn add_special(mut self, name: &str, id: u32) -> Self {
        self.special.insert(name.into(), id);
        self
    }

    /// Build the tokenizer.
    pub fn build(self) -> LmResult<BpeTokenizer> {
        let base = Vocab::gpt2_byte_vocab();
        let vocab = base.with_extra_tokens(self.extra_tokens, self.special)?;
        BpeTokenizer::new(vocab, self.merges)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal vocabulary: 'a'=0, 'b'=1, 'c'=2, 'd'=3 (raw byte tokens),
    /// plus merges:
    ///   rank 0: (a, b) → "ab" (id 4)
    ///   rank 1: (c, d) → "cd" (id 5)
    ///   rank 2: (ab, cd) → "abcd" (id 6)
    fn minimal_tokenizer() -> BpeTokenizer {
        let tokens: Vec<Vec<u8>> = vec![
            vec![b'a'],                   // 0
            vec![b'b'],                   // 1
            vec![b'c'],                   // 2
            vec![b'd'],                   // 3
            vec![b'a', b'b'],             // 4  = merge(0,1)
            vec![b'c', b'd'],             // 5  = merge(2,3)
            vec![b'a', b'b', b'c', b'd'], // 6 = merge(4,5)
        ];
        let merges = vec![
            (vec![b'a'], vec![b'b']),             // rank 0
            (vec![b'c'], vec![b'd']),             // rank 1
            (vec![b'a', b'b'], vec![b'c', b'd']), // rank 2
        ];
        let vocab = Vocab::from_tokens(tokens, HashMap::new()).unwrap();
        BpeTokenizer::new(vocab, merges).unwrap()
    }

    #[test]
    fn encode_single_char() {
        let t = minimal_tokenizer();
        assert_eq!(t.encode("a").unwrap(), vec![0u32]);
        assert_eq!(t.encode("b").unwrap(), vec![1u32]);
    }

    #[test]
    fn encode_empty_string() {
        let t = minimal_tokenizer();
        assert_eq!(t.encode("").unwrap(), vec![]);
    }

    #[test]
    fn encode_ab_merges_to_one_token() {
        let t = minimal_tokenizer();
        assert_eq!(t.encode("ab").unwrap(), vec![4u32]);
    }

    #[test]
    fn encode_cd_merges_to_one_token() {
        let t = minimal_tokenizer();
        assert_eq!(t.encode("cd").unwrap(), vec![5u32]);
    }

    #[test]
    fn encode_abcd_fully_merged() {
        let t = minimal_tokenizer();
        assert_eq!(t.encode("abcd").unwrap(), vec![6u32]);
    }

    #[test]
    fn encode_abc_partial_merge() {
        // "abc" → [ab=4, c=2] (cd merge does not apply because no 'd')
        let t = minimal_tokenizer();
        assert_eq!(t.encode("abc").unwrap(), vec![4u32, 2]);
    }

    #[test]
    fn encode_abcdabcd_two_full_merges() {
        let t = minimal_tokenizer();
        // "abcdabcd" → [6, 6]
        assert_eq!(t.encode("abcdabcd").unwrap(), vec![6u32, 6]);
    }

    #[test]
    fn decode_single_token() {
        let t = minimal_tokenizer();
        assert_eq!(t.decode(&[0]).unwrap(), "a");
        assert_eq!(t.decode(&[4]).unwrap(), "ab");
        assert_eq!(t.decode(&[6]).unwrap(), "abcd");
    }

    #[test]
    fn decode_multiple_tokens() {
        let t = minimal_tokenizer();
        assert_eq!(t.decode(&[4, 2]).unwrap(), "abc");
        assert_eq!(t.decode(&[6, 6]).unwrap(), "abcdabcd");
    }

    #[test]
    fn encode_then_decode_roundtrip() {
        let t = minimal_tokenizer();
        for text in &["a", "ab", "abc", "abcd", "abcdabcd", "ba", "dcba"] {
            let ids = t.encode(text).unwrap();
            let decoded = t.decode(&ids).unwrap();
            assert_eq!(&decoded, text, "roundtrip failed for '{text}'");
        }
    }

    #[test]
    fn out_of_vocab_token_errors() {
        let t = minimal_tokenizer();
        assert!(matches!(
            t.decode(&[99]),
            Err(LmError::OutOfVocab { token: 99 })
        ));
    }

    #[test]
    fn encode_with_special_bos_eos() {
        let tokens: Vec<Vec<u8>> = vec![
            vec![b'a'],       // 0
            vec![b'b'],       // 1
            vec![1_u8, 0_u8], // 2: BOS special (arbitrary bytes not in 'a','b' domain)
            vec![2_u8, 0_u8], // 3: EOS special
        ];
        let special: HashMap<String, u32> = [("<bos>".into(), 2u32), ("<eos>".into(), 3u32)]
            .into_iter()
            .collect();
        let vocab = Vocab::from_tokens(tokens, special).unwrap();
        let t = BpeTokenizer::new(vocab, vec![]).unwrap();
        let ids = t
            .encode_with_special("ab", Some("<bos>"), Some("<eos>"))
            .unwrap();
        assert_eq!(ids, vec![2, 0, 1, 3]);
    }

    #[test]
    fn encode_with_special_no_bos_eos_absent() {
        // When special token name is not in vocab, skip silently.
        let t = minimal_tokenizer();
        let ids = t.encode_with_special("ab", Some("<bos>"), None).unwrap();
        // <bos> not registered → skip; "ab" → [4]
        assert_eq!(ids, vec![4u32]);
    }

    #[test]
    fn bpe_builder_basic() {
        // Build a tokenizer using BpeBuilder on top of 256 byte tokens.
        // Merge ('a', 'b') → "ab" at rank 0.
        let t = BpeBuilder::new()
            .add_merge(b"a", b"b") // "ab" gets id 256
            .build()
            .unwrap();
        assert_eq!(t.vocab_size(), 257);
        assert_eq!(t.encode("ab").unwrap(), vec![256u32]);
        assert_eq!(t.decode(&[256]).unwrap(), "ab");
    }

    #[test]
    fn bpe_builder_chained_merges() {
        // Merge a+b → ab (rank 0), then ab+c → abc (rank 1)
        let t = BpeBuilder::new()
            .add_merge(b"a", b"b") // id 256: "ab"
            .add_merge(b"ab", b"c") // id 257: "abc"
            .build()
            .unwrap();
        assert_eq!(t.encode("abc").unwrap(), vec![257u32]);
        assert_eq!(t.decode(&[257]).unwrap(), "abc");
    }

    #[test]
    fn bpe_builder_with_special_token() {
        let t = BpeBuilder::new()
            .add_special("<eos>", 10) // byte 10 = 0x0A (newline) repurposed
            .build()
            .unwrap();
        assert_eq!(t.special_id("<eos>"), Some(10));
    }

    #[test]
    fn vocab_size_matches_builder() {
        let t = BpeBuilder::new()
            .add_merge(b"x", b"y")
            .add_merge(b"xy", b"z")
            .build()
            .unwrap();
        assert_eq!(t.vocab_size(), 258); // 256 + 2 merged tokens
    }

    #[test]
    fn priority_order_matters() {
        // Two possible merges on "abcd":
        //   rank 0: (b, c) → "bc" (id 256)
        //   rank 1: (a, b) → "ab" (id 257)
        // With rank 0 = (b,c): "abcd" → [a, bc, d] → no further merges
        let t = BpeBuilder::new()
            .add_merge(b"b", b"c") // rank 0
            .add_merge(b"a", b"b") // rank 1
            .build()
            .unwrap();
        let ids = t.encode("abcd").unwrap();
        // 'a'=97, 'bc'=256, 'd'=100  (GPT-2 byte tokens use raw byte values)
        assert_eq!(ids, vec![b'a' as u32, 256u32, b'd' as u32]);
    }
}
