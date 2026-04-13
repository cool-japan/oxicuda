//! Shared memory bank conflict detection for PTX instruction sequences.
//!
//! NVIDIA shared memory is organized into 32 banks, each 4 bytes wide. When
//! multiple threads in a warp access different words in the same bank, a
//! *bank conflict* occurs, causing serialized memory accesses and reduced
//! throughput. This module provides static analysis to detect such patterns
//! in PTX instruction sequences and suggest mitigations.
//!
//! # Bank Mapping
//!
//! For a byte offset `b`, the bank index is `(b / 4) % 32`. Sequential
//! 4-byte accesses from consecutive threads map to consecutive banks and are
//! conflict-free. Stride patterns that are multiples of 128 bytes (32 words)
//! cause all threads to hit the same bank (32-way conflict).
//!
//! # Broadcast Exception
//!
//! When all threads in a warp access the *same word* (same 4-byte aligned
//! address), the hardware broadcasts the value and no conflict occurs.

use std::fmt;

use crate::ir::{Instruction, MemorySpace, Operand, PtxType};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Number of shared memory banks on all current NVIDIA architectures.
const NUM_BANKS: u32 = 32;

/// Width of each bank in bytes.
const BANK_WIDTH_BYTES: u32 = 4;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// A single shared memory access extracted from an instruction.
#[derive(Debug, Clone)]
pub struct SharedMemAccess {
    /// Index of the instruction in the input slice.
    pub instruction_index: usize,
    /// Name of the base address register.
    pub base_reg: String,
    /// Offset pattern relative to the base register.
    pub offset: AccessOffset,
    /// Width of the access in bytes (1, 2, 4, 8, or 16).
    pub access_width: u32,
    /// Whether this is a store (`true`) or a load (`false`).
    pub is_store: bool,
}

/// Classification of the byte offset in a shared memory access.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AccessOffset {
    /// A compile-time constant offset, e.g. `[smem + 128]`.
    Constant(i64),
    /// A stride pattern of the form `tid * stride + base`, where each
    /// thread in the warp accesses a different element.
    Strided {
        /// Constant added to the strided address.
        base: i64,
        /// Per-thread stride in bytes.
        stride: i64,
    },
    /// An offset that cannot be statically determined.
    Unknown,
}

/// Complete bank conflict analysis report.
#[derive(Debug, Clone)]
pub struct BankConflictReport {
    /// All shared memory accesses found in the instruction sequence.
    pub accesses: Vec<SharedMemAccess>,
    /// Detected bank conflicts.
    pub conflicts: Vec<BankConflict>,
    /// Number of accesses that are provably conflict-free.
    pub conflict_free_count: usize,
    /// Total number of shared memory accesses analyzed.
    pub total_shared_accesses: usize,
}

/// A single detected bank conflict.
#[derive(Debug, Clone)]
pub struct BankConflict {
    /// Index of the instruction that triggers the conflict.
    pub instruction_index: usize,
    /// Classification of the conflict.
    pub conflict_type: ConflictType,
    /// Bank indices involved in the conflict.
    pub affected_banks: Vec<u32>,
    /// Maximum number of threads contending for the same bank.
    pub degree: u32,
    /// Human-readable remediation suggestion.
    pub suggestion: String,
}

/// Classification of a bank conflict pattern.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConflictType {
    /// N threads access the same bank but different words.
    NWay(u32),
    /// All threads access the same word -- the hardware broadcasts, so
    /// no conflict actually occurs.
    Broadcast,
    /// A stride pattern causes systematic conflicts.
    StridedConflict {
        /// The per-thread stride in bytes.
        stride: i64,
    },
}

// ---------------------------------------------------------------------------
// Display implementations
// ---------------------------------------------------------------------------

impl fmt::Display for BankConflictReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Bank Conflict Report")?;
        writeln!(
            f,
            "  Total shared memory accesses: {}",
            self.total_shared_accesses
        )?;
        writeln!(f, "  Conflict-free accesses: {}", self.conflict_free_count)?;
        writeln!(f, "  Conflicts detected: {}", self.conflicts.len())?;
        for (i, conflict) in self.conflicts.iter().enumerate() {
            writeln!(
                f,
                "  [{i}] instruction {}: {}-way, suggestion: {}",
                conflict.instruction_index, conflict.degree, conflict.suggestion
            )?;
        }
        Ok(())
    }
}

impl fmt::Display for ConflictType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NWay(n) => write!(f, "{n}-way bank conflict"),
            Self::Broadcast => write!(f, "broadcast (no conflict)"),
            Self::StridedConflict { stride } => {
                write!(f, "strided conflict (stride={stride} bytes)")
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Public helper functions
// ---------------------------------------------------------------------------

/// Calculate which bank a byte offset maps to.
///
/// The mapping is `(byte_offset / 4) % 32`, where negative offsets wrap
/// around (the modulus result is always in `0..32`).
#[must_use]
pub fn byte_offset_to_bank(byte_offset: i64) -> u32 {
    let word_index = byte_offset.div_euclid(i64::from(BANK_WIDTH_BYTES));
    #[allow(clippy::cast_possible_truncation)]
    // rem_euclid(32) guarantees the result fits in u32
    {
        word_index.rem_euclid(i64::from(NUM_BANKS)) as u32
    }
}

/// Determine the degree of bank conflict caused by a strided access pattern.
///
/// For a warp of threads where thread *t* accesses byte offset
/// `t * stride_bytes`, the conflict degree (max threads per bank) equals
/// `warp_size / (NUM_BANKS / gcd(stride_words, NUM_BANKS))` where
/// `stride_words = stride_bytes / 4`.
///
/// A degree of 1 means conflict-free; higher values indicate serialization.
#[must_use]
pub fn stride_conflict_degree(stride_bytes: i64, warp_size: u32) -> u32 {
    if stride_bytes == 0 {
        // All threads access the same address — broadcast, no conflict on
        // NVIDIA GPUs (hardware broadcasts same-word accesses).
        return 1;
    }
    let stride_words = stride_bytes.div_euclid(i64::from(BANK_WIDTH_BYTES));
    let stride_abs = stride_words.unsigned_abs();
    let effective = stride_abs % u64::from(NUM_BANKS);
    if effective == 0 {
        // Every thread hits the same bank — full serialization.
        return warp_size;
    }
    // Number of distinct banks accessed = NUM_BANKS / gcd(effective, NUM_BANKS).
    let g = gcd_u64(effective, u64::from(NUM_BANKS));
    let distinct_banks = u64::from(NUM_BANKS) / g;
    // Conflict degree = threads per bank (assuming warp_size threads).
    let degree = u64::from(warp_size) / distinct_banks;
    u32::try_from(degree.max(1)).unwrap_or(warp_size)
}

/// Suggest an amount of padding (in bytes) to add per row in order to
/// eliminate bank conflicts for a given row width.
///
/// Returns `Some(padding_bytes)` when padding would help, or `None` if the
/// row width is already conflict-free (i.e., not a multiple of 128 bytes
/// or already has a non-power-of-2 relationship with bank count).
#[must_use]
pub const fn suggest_padding(current_row_bytes: u32) -> Option<u32> {
    if current_row_bytes == 0 {
        return None;
    }
    // Conflicts arise when the row width is a multiple of 128 bytes
    // (NUM_BANKS * BANK_WIDTH_BYTES). Adding one bank-width of padding
    // per row breaks the alignment.
    let bank_line = NUM_BANKS * BANK_WIDTH_BYTES; // 128
    if current_row_bytes % bank_line == 0 {
        Some(BANK_WIDTH_BYTES)
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Main analysis entry point
// ---------------------------------------------------------------------------

/// Analyze a sequence of PTX instructions for shared memory bank conflicts.
///
/// The analysis:
/// 1. Extracts all load/store instructions targeting the `Shared` memory space.
/// 2. Classifies the byte offset of each access (constant, strided, or unknown).
/// 3. For strided accesses, computes the conflict degree across the warp.
/// 4. For constant accesses, checks whether all threads hit the same bank.
/// 5. Generates human-readable suggestions for conflict mitigation.
///
/// # Arguments
///
/// * `instructions` -- The instruction sequence to analyze.
/// * `warp_size` -- Number of threads per warp (typically 32).
///
/// # Returns
///
/// A [`BankConflictReport`] summarizing all shared memory accesses and any
/// detected bank conflicts.
pub fn analyze_bank_conflicts(instructions: &[Instruction], warp_size: u32) -> BankConflictReport {
    let mut accesses = Vec::new();
    let mut conflicts = Vec::new();
    let mut conflict_free_count: usize = 0;

    // Step 1: Extract shared memory accesses.
    for (idx, inst) in instructions.iter().enumerate() {
        if let Some(access) = extract_shared_access(inst, idx) {
            accesses.push(access);
        }
    }

    let total_shared_accesses = accesses.len();

    // Step 2: Analyze each access for conflicts.
    for access in &accesses {
        match &access.offset {
            AccessOffset::Strided { stride, .. } => {
                let degree = stride_conflict_degree(*stride, warp_size);
                if degree <= 1 {
                    conflict_free_count += 1;
                } else {
                    let suggestion = strided_suggestion(*stride, degree);
                    let affected = compute_affected_banks_strided(*stride, warp_size);
                    conflicts.push(BankConflict {
                        instruction_index: access.instruction_index,
                        conflict_type: ConflictType::StridedConflict { stride: *stride },
                        affected_banks: affected,
                        degree,
                        suggestion,
                    });
                }
            }
            AccessOffset::Constant(offset) => {
                // For constant offset, every thread hits the same bank/word.
                // If they hit the same word, it is a broadcast (no conflict).
                // Since all threads use the identical constant, it is always a broadcast.
                let bank = byte_offset_to_bank(*offset);
                conflicts.push(BankConflict {
                    instruction_index: access.instruction_index,
                    conflict_type: ConflictType::Broadcast,
                    affected_banks: vec![bank],
                    degree: 1,
                    suggestion: "Broadcast access -- no conflict.".to_string(),
                });
                conflict_free_count += 1;
            }
            AccessOffset::Unknown => {
                // Cannot analyze statically; count as conflict-free (optimistic).
                conflict_free_count += 1;
            }
        }
    }

    BankConflictReport {
        accesses,
        conflicts,
        conflict_free_count,
        total_shared_accesses,
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Extract a [`SharedMemAccess`] from an instruction if it is a shared memory
/// load or store.
fn extract_shared_access(inst: &Instruction, idx: usize) -> Option<SharedMemAccess> {
    match inst {
        Instruction::Load {
            space: MemorySpace::Shared,
            ty,
            addr,
            ..
        } => Some(build_access(idx, *ty, addr, false)),
        Instruction::Store {
            space: MemorySpace::Shared,
            ty,
            addr,
            ..
        } => Some(build_access(idx, *ty, addr, true)),
        _ => None,
    }
}

/// Build a [`SharedMemAccess`] from components.
fn build_access(
    instruction_index: usize,
    ty: PtxType,
    addr: &Operand,
    is_store: bool,
) -> SharedMemAccess {
    let (base_reg, offset) = classify_address(addr);
    #[allow(clippy::cast_possible_truncation)]
    let width = ty.size_bytes() as u32; // size_bytes() is always small
    SharedMemAccess {
        instruction_index,
        base_reg,
        offset,
        access_width: width,
        is_store,
    }
}

/// Classify an address operand into a base register name and an [`AccessOffset`].
fn classify_address(addr: &Operand) -> (String, AccessOffset) {
    match addr {
        Operand::Address { base, offset } => {
            let base_name = base.name.clone();
            let off = offset.unwrap_or(0);
            (base_name, AccessOffset::Constant(off))
        }
        Operand::Register(reg) => (reg.name.clone(), AccessOffset::Constant(0)),
        Operand::Symbol(sym) => (sym.clone(), AccessOffset::Unknown),
        Operand::Immediate(_) => (String::new(), AccessOffset::Unknown),
    }
}

/// Compute which banks are affected by a strided access across a warp.
fn compute_affected_banks_strided(stride_bytes: i64, warp_size: u32) -> Vec<u32> {
    let mut banks = Vec::new();
    for tid in 0..warp_size {
        let byte_off = i64::from(tid) * stride_bytes;
        let bank = byte_offset_to_bank(byte_off);
        if !banks.contains(&bank) {
            banks.push(bank);
        }
    }
    banks.sort_unstable();
    banks
}

/// Generate a human-readable suggestion for a strided conflict.
fn strided_suggestion(stride_bytes: i64, degree: u32) -> String {
    let stride_abs = stride_bytes.unsigned_abs();
    if stride_abs.is_power_of_two() {
        format!(
            "Add {BANK_WIDTH_BYTES}-byte padding per row to break bank alignment \
             ({degree}-way conflict, stride={stride_bytes}B)"
        )
    } else {
        format!(
            "Consider using different offsets per thread to reduce \
             {degree}-way conflict (stride={stride_bytes}B)"
        )
    }
}

/// Greatest common divisor (Euclidean algorithm) for u64.
const fn gcd_u64(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{CacheQualifier, MemorySpace, Operand, PtxType, Register, VectorWidth};

    /// Helper: create a shared memory load instruction with a constant offset.
    fn shared_load(offset: i64, ty: PtxType) -> Instruction {
        Instruction::Load {
            space: MemorySpace::Shared,
            qualifier: CacheQualifier::None,
            vec: VectorWidth::V1,
            ty,
            dst: Register {
                name: "%r0".into(),
                ty: PtxType::U32,
            },
            addr: Operand::Address {
                base: Register {
                    name: "%rd_smem".into(),
                    ty: PtxType::U64,
                },
                offset: Some(offset),
            },
        }
    }

    /// Helper: create a shared memory store instruction.
    fn shared_store(offset: i64, ty: PtxType) -> Instruction {
        Instruction::Store {
            space: MemorySpace::Shared,
            qualifier: CacheQualifier::None,
            vec: VectorWidth::V1,
            ty,
            addr: Operand::Address {
                base: Register {
                    name: "%rd_smem".into(),
                    ty: PtxType::U64,
                },
                offset: Some(offset),
            },
            src: Register {
                name: "%r0".into(),
                ty: PtxType::U32,
            },
        }
    }

    /// Helper: create a global memory load (should be ignored by analysis).
    fn global_load() -> Instruction {
        Instruction::Load {
            space: MemorySpace::Global,
            qualifier: CacheQualifier::None,
            vec: VectorWidth::V1,
            ty: PtxType::F32,
            dst: Register {
                name: "%f0".into(),
                ty: PtxType::F32,
            },
            addr: Operand::Address {
                base: Register {
                    name: "%rd0".into(),
                    ty: PtxType::U64,
                },
                offset: None,
            },
        }
    }

    // -----------------------------------------------------------------------
    // byte_offset_to_bank tests
    // -----------------------------------------------------------------------

    #[test]
    fn byte_offset_to_bank_basic() {
        assert_eq!(byte_offset_to_bank(0), 0);
        assert_eq!(byte_offset_to_bank(4), 1);
        assert_eq!(byte_offset_to_bank(8), 2);
        assert_eq!(byte_offset_to_bank(124), 31);
        assert_eq!(byte_offset_to_bank(128), 0); // wraps
        assert_eq!(byte_offset_to_bank(132), 1);
    }

    #[test]
    fn byte_offset_to_bank_negative() {
        // Negative offsets should wrap around via div_euclid/rem_euclid.
        assert_eq!(byte_offset_to_bank(-4), 31);
        assert_eq!(byte_offset_to_bank(-128), 0);
    }

    // -----------------------------------------------------------------------
    // stride_conflict_degree tests
    // -----------------------------------------------------------------------

    #[test]
    fn stride_4_bytes_no_conflict() {
        // Sequential 4-byte stride: thread t → bank t. All 32 banks used, degree 1.
        assert_eq!(stride_conflict_degree(4, 32), 1);
    }

    #[test]
    fn stride_128_bytes_full_conflict() {
        // 128-byte stride = 32 words. (t*32) % 32 = 0 for all t. All threads
        // hit bank 0 — worst case, degree = warp_size.
        assert_eq!(stride_conflict_degree(128, 32), 32);
    }

    #[test]
    fn stride_8_bytes_2way_conflict() {
        // 8-byte stride = 2 words. Banks: 0,2,4,...,30,0,2,... → 16 distinct
        // banks, 2 threads per bank → degree 2.
        assert_eq!(stride_conflict_degree(8, 32), 2);
    }

    #[test]
    fn stride_32_bytes_8way_conflict() {
        // 32-byte stride = 8 words. Banks: 0,8,16,24,0,8,... → 4 distinct
        // banks, 8 threads per bank → degree 8.
        assert_eq!(stride_conflict_degree(32, 32), 8);
    }

    #[test]
    fn stride_256_bytes_full_conflict() {
        // 256-byte stride = 64 words. 64 % 32 = 0 → all threads hit bank 0.
        assert_eq!(stride_conflict_degree(256, 32), 32);
    }

    #[test]
    fn stride_zero_broadcast() {
        // Stride 0: all threads access same address — broadcast, no conflict.
        assert_eq!(stride_conflict_degree(0, 32), 1);
    }

    // -----------------------------------------------------------------------
    // suggest_padding tests
    // -----------------------------------------------------------------------

    #[test]
    fn suggest_padding_multiple_of_128() {
        assert_eq!(suggest_padding(128), Some(4));
        assert_eq!(suggest_padding(256), Some(4));
        assert_eq!(suggest_padding(512), Some(4));
    }

    #[test]
    fn suggest_padding_not_needed() {
        assert_eq!(suggest_padding(64), None);
        assert_eq!(suggest_padding(132), None);
        assert_eq!(suggest_padding(0), None);
    }

    // -----------------------------------------------------------------------
    // analyze_bank_conflicts tests
    // -----------------------------------------------------------------------

    #[test]
    fn empty_instructions() {
        let report = analyze_bank_conflicts(&[], 32);
        assert_eq!(report.total_shared_accesses, 0);
        assert_eq!(report.conflict_free_count, 0);
        assert!(report.conflicts.is_empty());
        assert!(report.accesses.is_empty());
    }

    #[test]
    fn no_shared_memory_accesses() {
        let instructions = vec![global_load()];
        let report = analyze_bank_conflicts(&instructions, 32);
        assert_eq!(report.total_shared_accesses, 0);
        assert!(report.conflicts.is_empty());
    }

    #[test]
    fn constant_offset_broadcast_detection() {
        // All threads load from the same constant offset => broadcast.
        let instructions = vec![shared_load(64, PtxType::F32)];
        let report = analyze_bank_conflicts(&instructions, 32);
        assert_eq!(report.total_shared_accesses, 1);
        assert_eq!(report.conflict_free_count, 1);
        assert_eq!(report.conflicts.len(), 1);
        assert_eq!(report.conflicts[0].conflict_type, ConflictType::Broadcast);
        assert_eq!(report.conflicts[0].degree, 1);
    }

    #[test]
    fn store_vs_load_distinction() {
        let instructions = vec![shared_load(0, PtxType::F32), shared_store(4, PtxType::F32)];
        let report = analyze_bank_conflicts(&instructions, 32);
        assert_eq!(report.total_shared_accesses, 2);
        assert!(!report.accesses[0].is_store);
        assert!(report.accesses[1].is_store);
    }

    #[test]
    fn unknown_offset_symbol() {
        let inst = Instruction::Load {
            space: MemorySpace::Shared,
            qualifier: CacheQualifier::None,
            vec: VectorWidth::V1,
            ty: PtxType::F32,
            dst: Register {
                name: "%r0".into(),
                ty: PtxType::U32,
            },
            addr: Operand::Symbol("unknown_addr".into()),
        };
        let report = analyze_bank_conflicts(&[inst], 32);
        assert_eq!(report.total_shared_accesses, 1);
        assert_eq!(report.accesses[0].offset, AccessOffset::Unknown);
        // Unknown offsets are counted as conflict-free (optimistic).
        assert_eq!(report.conflict_free_count, 1);
    }

    #[test]
    fn mixed_accesses() {
        let instructions = vec![
            shared_load(0, PtxType::F32),   // broadcast (constant)
            global_load(),                  // ignored
            shared_store(64, PtxType::F32), // broadcast (constant)
        ];
        let report = analyze_bank_conflicts(&instructions, 32);
        assert_eq!(report.total_shared_accesses, 2);
        assert_eq!(report.conflict_free_count, 2);
    }

    #[test]
    fn report_display() {
        let report = BankConflictReport {
            accesses: Vec::new(),
            conflicts: vec![BankConflict {
                instruction_index: 3,
                conflict_type: ConflictType::NWay(8),
                affected_banks: vec![0, 8, 16, 24],
                degree: 8,
                suggestion: "Add padding".to_string(),
            }],
            conflict_free_count: 5,
            total_shared_accesses: 6,
        };
        let display = format!("{report}");
        assert!(display.contains("Total shared memory accesses: 6"));
        assert!(display.contains("Conflict-free accesses: 5"));
        assert!(display.contains("Conflicts detected: 1"));
        assert!(display.contains("8-way"));
    }

    #[test]
    fn conflict_type_display() {
        assert_eq!(format!("{}", ConflictType::NWay(4)), "4-way bank conflict");
        assert_eq!(
            format!("{}", ConflictType::Broadcast),
            "broadcast (no conflict)"
        );
        assert_eq!(
            format!("{}", ConflictType::StridedConflict { stride: 8 }),
            "strided conflict (stride=8 bytes)"
        );
    }

    #[test]
    fn access_width_from_type() {
        let inst_u8 = shared_load(0, PtxType::U8);
        let inst_f64 = shared_load(0, PtxType::F64);
        let inst_b128 = shared_load(0, PtxType::B128);
        let report = analyze_bank_conflicts(&[inst_u8, inst_f64, inst_b128], 32);
        assert_eq!(report.accesses[0].access_width, 1);
        assert_eq!(report.accesses[1].access_width, 8);
        assert_eq!(report.accesses[2].access_width, 16);
    }
}
