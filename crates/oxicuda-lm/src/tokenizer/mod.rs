//! BPE tokenizer and vocabulary management.
//!
//! # Usage
//!
//! ```ignore
//! use oxicuda_lm::tokenizer::{BpeBuilder, BpeTokenizer};
//!
//! let tokenizer = BpeBuilder::new()
//!     .add_merge(b"a", b"b")   // "ab" → id 256
//!     .add_special("<eos>", 0)
//!     .build()?;
//!
//! let ids = tokenizer.encode("ab")?;
//! assert_eq!(ids, vec![256]);
//! let text = tokenizer.decode(&ids)?;
//! assert_eq!(text, "ab");
//! ```

pub mod bpe;
pub mod vocab;

pub use bpe::{BpeBuilder, BpeTokenizer};
pub use vocab::Vocab;
