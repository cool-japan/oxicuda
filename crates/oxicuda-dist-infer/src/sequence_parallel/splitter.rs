//! Sequence splitter — partition and reconstruct token sequences.

use crate::error::{DistInferError, DistInferResult};
use crate::handle::DistInferHandle;

// ─── ChunkInfo ───────────────────────────────────────────────────────────────

/// Describes one rank's chunk of the full token sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkInfo {
    /// Global token index of the first token in this chunk.
    pub start: usize,
    /// Number of tokens in this chunk.
    pub len: usize,
    /// Total tokens in the full sequence.
    pub total_tokens: usize,
    /// Hidden dimension per token.
    pub hidden_dim: usize,
}

impl ChunkInfo {
    /// Byte length (f32 elements) of this chunk's embedding buffer.
    pub fn n_elements(&self) -> usize {
        self.len * self.hidden_dim
    }

    /// Byte length (f32 elements) of the full sequence buffer.
    pub fn full_n_elements(&self) -> usize {
        self.total_tokens * self.hidden_dim
    }
}

// ─── SeqSplitter ─────────────────────────────────────────────────────────────

/// Splits a token sequence across `sp` ranks and reassembles it.
///
/// The sequence is divided into `sp` contiguous chunks of equal length.
/// `total_tokens` must be divisible by `sp`.
#[derive(Debug, Clone)]
pub struct SeqSplitter {
    handle: DistInferHandle,
    total_tokens: usize,
    hidden_dim: usize,
    chunk_len: usize,
}

impl SeqSplitter {
    /// Construct a splitter for a sequence of `total_tokens` tokens.
    pub fn new(
        handle: DistInferHandle,
        total_tokens: usize,
        hidden_dim: usize,
    ) -> DistInferResult<Self> {
        let sp = handle.config.sp;
        if total_tokens % sp != 0 {
            return Err(DistInferError::SpSeqLenMisaligned {
                seq_len: total_tokens,
                degree: sp,
            });
        }
        let chunk_len = total_tokens / sp;
        if chunk_len == 0 {
            return Err(DistInferError::EmptyChunk {
                seq_len: total_tokens,
                degree: sp,
            });
        }
        Ok(Self {
            handle,
            total_tokens,
            hidden_dim,
            chunk_len,
        })
    }

    /// Metadata describing this rank's chunk.
    pub fn chunk_info(&self) -> ChunkInfo {
        ChunkInfo {
            start: self.handle.sp_rank() * self.chunk_len,
            len: self.chunk_len,
            total_tokens: self.total_tokens,
            hidden_dim: self.hidden_dim,
        }
    }

    /// Extract this rank's chunk from the full sequence buffer.
    ///
    /// `full_seq` shape: `[total_tokens × hidden_dim]`.
    /// Returns shape: `[chunk_len × hidden_dim]`.
    pub fn extract_chunk(&self, full_seq: &[f32]) -> DistInferResult<Vec<f32>> {
        let expected = self.total_tokens * self.hidden_dim;
        if full_seq.len() != expected {
            return Err(DistInferError::DimensionMismatch {
                expected,
                got: full_seq.len(),
            });
        }
        let info = self.chunk_info();
        let start_elem = info.start * self.hidden_dim;
        let end_elem = start_elem + info.n_elements();
        Ok(full_seq[start_elem..end_elem].to_vec())
    }

    /// Insert this rank's chunk back into the full sequence buffer.
    ///
    /// `chunk` shape: `[chunk_len × hidden_dim]`.
    /// `full_seq` shape: `[total_tokens × hidden_dim]` (mutable, may be partial).
    pub fn insert_chunk(&self, chunk: &[f32], full_seq: &mut Vec<f32>) -> DistInferResult<()> {
        let info = self.chunk_info();
        if chunk.len() != info.n_elements() {
            return Err(DistInferError::DimensionMismatch {
                expected: info.n_elements(),
                got: chunk.len(),
            });
        }
        let full_elems = self.total_tokens * self.hidden_dim;
        if full_seq.len() < full_elems {
            full_seq.resize(full_elems, 0.0);
        }
        let start = info.start * self.hidden_dim;
        full_seq[start..start + info.n_elements()].copy_from_slice(chunk);
        Ok(())
    }

    /// Simulate an all-gather: collect chunks from all `sp` ranks and
    /// reconstruct the full sequence.
    ///
    /// `chunks[r]` must be the output of `extract_chunk` for rank `r`.
    pub fn all_gather(
        total_tokens: usize,
        hidden_dim: usize,
        sp: usize,
        chunks: &[Vec<f32>],
    ) -> DistInferResult<Vec<f32>> {
        if chunks.len() != sp {
            return Err(DistInferError::DimensionMismatch {
                expected: sp,
                got: chunks.len(),
            });
        }
        if total_tokens % sp != 0 {
            return Err(DistInferError::SpSeqLenMisaligned {
                seq_len: total_tokens,
                degree: sp,
            });
        }
        let chunk_len = total_tokens / sp;
        let mut full = vec![0.0_f32; total_tokens * hidden_dim];
        for (rank, chunk) in chunks.iter().enumerate() {
            let start = rank * chunk_len * hidden_dim;
            let end = start + chunk_len * hidden_dim;
            if chunk.len() != chunk_len * hidden_dim {
                return Err(DistInferError::DimensionMismatch {
                    expected: chunk_len * hidden_dim,
                    got: chunk.len(),
                });
            }
            full[start..end].copy_from_slice(chunk);
        }
        Ok(full)
    }

    /// Simulate a reduce-scatter: sum corresponding token embeddings across
    /// ranks (as happens after a sequence-parallel matrix multiply) and
    /// scatter back the local chunk.
    ///
    /// `partials[r][t, d]` = rank `r`'s partial sum for token `t`, feature `d`.
    /// All partials have shape `[total_tokens × hidden_dim]`.
    /// Returns this rank's chunk after reduction, shape `[chunk_len × hidden_dim]`.
    pub fn reduce_scatter(&self, partials: &[Vec<f32>]) -> DistInferResult<Vec<f32>> {
        let full_elems = self.total_tokens * self.hidden_dim;
        // Sum all partials element-wise
        let mut acc = vec![0.0_f32; full_elems];
        for part in partials {
            if part.len() != full_elems {
                return Err(DistInferError::DimensionMismatch {
                    expected: full_elems,
                    got: part.len(),
                });
            }
            for (a, &p) in acc.iter_mut().zip(part.iter()) {
                *a += p;
            }
        }
        // Scatter: return only this rank's chunk
        let info = self.chunk_info();
        let start = info.start * self.hidden_dim;
        Ok(acc[start..start + info.n_elements()].to_vec())
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::{DistInferHandle, ParallelismConfig, SmVersion};

    fn handle_sp(sp: usize, rank: usize) -> DistInferHandle {
        DistInferHandle::new(
            rank as i32,
            SmVersion(80),
            rank,
            ParallelismConfig { tp: 1, sp, ep: 1 },
        )
        .unwrap()
    }

    #[test]
    fn extract_and_insert_roundtrip_sp2() {
        let sp = 2;
        let total = 4;
        let hd = 3;
        // full sequence: tokens 0..3, each with features [t*10, t*10+1, t*10+2]
        let full: Vec<f32> = (0..total * hd).map(|i| i as f32).collect();

        let chunks: Vec<Vec<f32>> = (0..sp)
            .map(|r| {
                let h = handle_sp(sp, r);
                let s = SeqSplitter::new(h, total, hd).unwrap();
                s.extract_chunk(&full).unwrap()
            })
            .collect();

        // All-gather reconstructs
        let reconstructed = SeqSplitter::all_gather(total, hd, sp, &chunks).unwrap();
        assert_eq!(reconstructed, full);
    }

    #[test]
    fn reduce_scatter_sums_across_ranks() {
        let sp = 4;
        let total = 4;
        let hd = 2;
        // Each partial = all-ones (total×hd)
        let partial = vec![1.0_f32; total * hd];
        let partials = vec![partial; sp]; // sp copies

        // rank 0: reduce_scatter should return chunk[0..hd] = sum of sp ones = 4.0
        let h0 = handle_sp(sp, 0);
        let s0 = SeqSplitter::new(h0, total, hd).unwrap();
        let chunk0 = s0.reduce_scatter(&partials).unwrap();
        // Each element in the chunk: sp * 1.0 = 4.0
        assert_eq!(chunk0, vec![4.0_f32; hd]);
    }

    #[test]
    fn seq_not_divisible_errors() {
        let h = handle_sp(3, 0);
        let err = SeqSplitter::new(h, 5, 4).unwrap_err();
        assert!(matches!(
            err,
            DistInferError::SpSeqLenMisaligned {
                seq_len: 5,
                degree: 3
            }
        ));
    }

    #[test]
    fn chunk_info_is_correct() {
        let sp = 4;
        for rank in 0..sp {
            let h = handle_sp(sp, rank);
            let s = SeqSplitter::new(h, 8, 2).unwrap();
            let ci = s.chunk_info();
            assert_eq!(ci.start, rank * 2, "wrong start for rank {rank}");
            assert_eq!(ci.len, 2);
        }
    }

    #[test]
    fn insert_chunk_fills_correct_region() {
        let sp = 2;
        let total = 4;
        let hd = 2;
        let h1 = handle_sp(sp, 1);
        let s1 = SeqSplitter::new(h1, total, hd).unwrap();
        let chunk = vec![99.0_f32; 2 * hd]; // rank 1 chunk (tokens 2..3)
        let mut full = vec![0.0_f32; total * hd];
        s1.insert_chunk(&chunk, &mut full).unwrap();
        // First chunk = zeros, second chunk = 99s
        assert_eq!(&full[0..hd * 2], &[0.0, 0.0, 0.0, 0.0]);
        assert_eq!(&full[hd * 2..], &[99.0, 99.0, 99.0, 99.0]);
    }
}
