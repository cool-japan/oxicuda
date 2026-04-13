//! Sequence parallelism for long-context LLM inference.
//!
//! For an input sequence of `T` tokens with hidden dimension `H`, each rank
//! processes a contiguous chunk of `T / sp` tokens.  Communication is needed
//! only at the boundaries of attention layers where full sequence information
//! is required (all-gather before attention, reduce-scatter after).
//!
//! # Modules
//!
//! * `splitter` — `SeqSplitter`: partition / reconstruct a token sequence.
//! * `boundary` — `BoundaryExchange`: simulate the all-gather and reduce-scatter
//!   communication at attention layer boundaries.

pub mod boundary;
pub mod splitter;

pub use boundary::BoundaryExchange;
pub use splitter::SeqSplitter;
