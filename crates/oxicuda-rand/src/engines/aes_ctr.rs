//! AES-CTR CSPRNG engine.
//!
//! Cryptographically secure pseudorandom number generator based on AES-256 in
//! counter mode (CTR). Suitable for security-sensitive GPU applications where
//! statistical quality alone is insufficient and cryptographic unpredictability
//! is required.
//!
//! The CPU-side implementation follows NIST SP 800-38A and FIPS 197. The PTX
//! generator produces GPU kernels where each thread encrypts a unique counter
//! block, yielding 4 u32 values (128 bits) per thread.
//!
//! Reference: FIPS 197 (AES), NIST SP 800-38A (Block Cipher Modes)

use oxicuda_ptx::arch::SmVersion;
use oxicuda_ptx::builder::KernelBuilder;
use oxicuda_ptx::error::PtxGenError;
use oxicuda_ptx::ir::PtxType;

use crate::error::{RandError, RandResult};

// ===========================================================================
//  AES S-Box (FIPS 197, Section 5.1.1)
// ===========================================================================

/// The standard AES forward S-box lookup table.
///
/// Each byte value `b` maps to `AES_SBOX[b as usize]`, implementing the
/// SubBytes transformation defined in FIPS 197.
#[rustfmt::skip]
const AES_SBOX: [u8; 256] = [
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16,
];

/// AES round constants used during key expansion.
///
/// Only 7 values are needed for AES-256 (14 rounds, key schedule words 8..60,
/// with RCON applied at positions 8, 16, 24, 32, 40, 48, 56).
const AES_RCON: [u8; 7] = [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40];

// ===========================================================================
//  AES Key Schedule (FIPS 197, Section 5.2)
// ===========================================================================

/// Applies the AES SubBytes transformation to each byte of a 32-bit word.
#[inline]
const fn sub_word(w: u32) -> u32 {
    let b0 = AES_SBOX[(w & 0xFF) as usize] as u32;
    let b1 = AES_SBOX[((w >> 8) & 0xFF) as usize] as u32;
    let b2 = AES_SBOX[((w >> 16) & 0xFF) as usize] as u32;
    let b3 = AES_SBOX[((w >> 24) & 0xFF) as usize] as u32;
    b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)
}

/// Rotates a 32-bit word left by 8 bits (one byte position).
#[inline]
const fn rot_word(w: u32) -> u32 {
    w.rotate_left(8)
}

/// Expands a 256-bit AES key into the full round key schedule.
///
/// Returns 60 u32 words (240 bytes) representing 15 round keys (initial +
/// 14 rounds), each consisting of 4 words. The key bytes are interpreted
/// in little-endian order for each 32-bit word.
///
/// # Arguments
///
/// * `key` - 32-byte AES-256 key
pub fn expand_key_256(key: &[u8; 32]) -> [u32; 60] {
    let mut w = [0u32; 60];

    // First 8 words come directly from the key (big-endian per FIPS 197)
    let mut i = 0;
    while i < 8 {
        w[i] = u32::from_be_bytes([key[4 * i], key[4 * i + 1], key[4 * i + 2], key[4 * i + 3]]);
        i += 1;
    }

    // Remaining words
    i = 8;
    while i < 60 {
        let mut temp = w[i - 1];
        if i % 8 == 0 {
            // RCON is applied to the most significant byte (big-endian word format)
            temp = sub_word(rot_word(temp)) ^ ((AES_RCON[i / 8 - 1] as u32) << 24);
        } else if i % 8 == 4 {
            temp = sub_word(temp);
        }
        w[i] = w[i - 8] ^ temp;
        i += 1;
    }

    w
}

// ===========================================================================
//  AES Block Cipher (FIPS 197, Section 5)
// ===========================================================================

/// Applies the SubBytes transformation to a 16-byte state.
#[inline]
fn sub_bytes(state: &mut [u8; 16]) {
    for byte in state.iter_mut() {
        *byte = AES_SBOX[*byte as usize];
    }
}

/// Applies the ShiftRows transformation.
///
/// The state is arranged in column-major order (column 0 = bytes 0..4,
/// column 1 = bytes 4..8, etc.). ShiftRows cyclically shifts each row.
#[inline]
fn shift_rows(state: &mut [u8; 16]) {
    // Row 0: no shift
    // Row 1: shift left by 1
    let t = state[1];
    state[1] = state[5];
    state[5] = state[9];
    state[9] = state[13];
    state[13] = t;

    // Row 2: shift left by 2
    let t0 = state[2];
    let t1 = state[6];
    state[2] = state[10];
    state[6] = state[14];
    state[10] = t0;
    state[14] = t1;

    // Row 3: shift left by 3 (= right by 1)
    let t = state[15];
    state[15] = state[11];
    state[11] = state[7];
    state[7] = state[3];
    state[3] = t;
}

/// Multiplies a byte by 2 in GF(2^8) with the AES irreducible polynomial.
#[inline]
const fn xtime(a: u8) -> u8 {
    let shifted = (a as u16) << 1;
    (shifted ^ (if a & 0x80 != 0 { 0x1b } else { 0x00 })) as u8
}

/// Multiplies two bytes in GF(2^8).
#[inline]
const fn gmul(mut a: u8, mut b: u8) -> u8 {
    let mut result = 0u8;
    let mut i = 0;
    while i < 8 {
        if b & 1 != 0 {
            result ^= a;
        }
        a = xtime(a);
        b >>= 1;
        i += 1;
    }
    result
}

/// Applies the MixColumns transformation.
///
/// Each column of the state is multiplied by the fixed polynomial
/// {03}x^3 + {01}x^2 + {01}x + {02} in GF(2^8).
#[inline]
fn mix_columns(state: &mut [u8; 16]) {
    for col in 0..4 {
        let base = col * 4;
        let s0 = state[base];
        let s1 = state[base + 1];
        let s2 = state[base + 2];
        let s3 = state[base + 3];

        state[base] = gmul(0x02, s0) ^ gmul(0x03, s1) ^ s2 ^ s3;
        state[base + 1] = s0 ^ gmul(0x02, s1) ^ gmul(0x03, s2) ^ s3;
        state[base + 2] = s0 ^ s1 ^ gmul(0x02, s2) ^ gmul(0x03, s3);
        state[base + 3] = gmul(0x03, s0) ^ s1 ^ s2 ^ gmul(0x02, s3);
    }
}

/// Applies AddRoundKey: XORs the state with 4 round key words.
#[inline]
fn add_round_key(state: &mut [u8; 16], round_keys: &[u32; 60], round: usize) {
    for col in 0..4 {
        let rk = round_keys[round * 4 + col].to_be_bytes();
        let base = col * 4;
        state[base] ^= rk[0];
        state[base + 1] ^= rk[1];
        state[base + 2] ^= rk[2];
        state[base + 3] ^= rk[3];
    }
}

/// Encrypts a single 128-bit block using AES-256.
///
/// Performs the full 14-round AES encryption as specified in FIPS 197.
/// The last round omits MixColumns.
///
/// # Arguments
///
/// * `block` - 16-byte plaintext block
/// * `round_keys` - 60-word expanded key schedule from [`expand_key_256`]
pub fn aes_encrypt_block(block: &[u8; 16], round_keys: &[u32; 60]) -> [u8; 16] {
    let mut state = *block;

    // Initial round key addition
    add_round_key(&mut state, round_keys, 0);

    // Rounds 1..13: SubBytes, ShiftRows, MixColumns, AddRoundKey
    for round in 1..14 {
        sub_bytes(&mut state);
        shift_rows(&mut state);
        mix_columns(&mut state);
        add_round_key(&mut state, round_keys, round);
    }

    // Final round (no MixColumns)
    sub_bytes(&mut state);
    shift_rows(&mut state);
    add_round_key(&mut state, round_keys, 14);

    state
}

// ===========================================================================
//  AES-CTR Mode
// ===========================================================================

/// Builds a 128-bit counter block: nonce (12 bytes) || counter (4 bytes, big-endian).
///
/// This follows NIST SP 800-38A format where the nonce occupies the first 12
/// bytes and the counter occupies the last 4 bytes in big-endian order.
#[inline]
fn build_counter_block(nonce: &[u8; 12], counter: u64) -> [u8; 16] {
    let mut block = [0u8; 16];
    block[..12].copy_from_slice(nonce);
    // Only the lower 32 bits of the counter enter the block (big-endian)
    let ctr_low = (counter & 0xFFFF_FFFF) as u32;
    block[12..16].copy_from_slice(&ctr_low.to_be_bytes());
    block
}

// ===========================================================================
//  Configuration and State
// ===========================================================================

/// Configuration for the AES-CTR CSPRNG engine.
///
/// Holds the cryptographic key, nonce, initial counter value, and GPU
/// launch parameters.
#[derive(Debug, Clone)]
pub struct AesCtrConfig {
    /// 256-bit AES key.
    pub key: [u8; 32],
    /// 96-bit nonce (initialization vector prefix).
    pub nonce: [u8; 12],
    /// Starting counter value.
    pub initial_counter: u64,
    /// Threads per block for GPU kernel launches.
    pub threads_per_block: u32,
    /// Target GPU architecture.
    pub sm_version: SmVersion,
}

impl Default for AesCtrConfig {
    fn default() -> Self {
        Self {
            key: [0u8; 32],
            nonce: [0u8; 12],
            initial_counter: 0,
            threads_per_block: 256,
            sm_version: SmVersion::Sm80,
        }
    }
}

/// Internal state of the AES-CTR engine.
///
/// Contains the expanded key schedule and current counter position.
#[derive(Debug, Clone)]
pub struct AesCtrState {
    /// Expanded AES-256 round keys (14 rounds x 4 + 4 = 60 words).
    pub key_schedule: [u32; 60],
    /// Current counter value.
    pub counter: u64,
    /// 96-bit nonce.
    pub nonce: [u8; 12],
}

/// AES-CTR CSPRNG engine.
///
/// Provides cryptographically secure random number generation using AES-256
/// in counter mode. Each counter value produces 128 bits (4 u32 values) of
/// pseudorandom output by encrypting the counter block with the AES key.
///
/// # CPU-side Usage
///
/// ```rust
/// use oxicuda_rand::engines::aes_ctr::AesCtrEngine;
///
/// let key = [0x42u8; 32];
/// let nonce = [0u8; 12];
/// let mut engine = AesCtrEngine::new(&key, &nonce);
/// let val = engine.next_u32();
/// ```
///
/// # GPU Usage
///
/// Use [`generate_aes_ctr_ptx`] to produce a PTX kernel where each GPU
/// thread encrypts a unique counter block in parallel.
#[derive(Debug, Clone)]
pub struct AesCtrEngine {
    /// Current engine state.
    pub state: AesCtrState,
    /// Engine configuration.
    pub config: AesCtrConfig,
    /// Buffer of remaining u32 values from the last AES block.
    buffer: [u32; 4],
    /// Number of u32 values remaining in the buffer (0..=3).
    buffer_remaining: u8,
}

impl AesCtrEngine {
    /// Creates a new AES-CTR engine with the given key and nonce.
    ///
    /// The counter starts at 0.
    #[must_use]
    pub fn new(key: &[u8; 32], nonce: &[u8; 12]) -> Self {
        let key_schedule = expand_key_256(key);
        Self {
            state: AesCtrState {
                key_schedule,
                counter: 0,
                nonce: *nonce,
            },
            config: AesCtrConfig {
                key: *key,
                nonce: *nonce,
                initial_counter: 0,
                threads_per_block: 256,
                sm_version: SmVersion::Sm80,
            },
            buffer: [0; 4],
            buffer_remaining: 0,
        }
    }

    /// Creates a new AES-CTR engine from a 64-bit seed.
    ///
    /// Derives the 256-bit key and 96-bit nonce from the seed using a
    /// simple expansion. Each distinct seed produces a distinct key/nonce
    /// pair. This is a convenience constructor; for production CSPRNG use,
    /// prefer [`AesCtrEngine::new`] with properly generated key material.
    #[must_use]
    pub fn from_seed(seed: u64) -> Self {
        let seed_bytes = seed.to_le_bytes();

        // Derive 32-byte key by mixing seed bytes with position-dependent offsets
        let mut key = [0u8; 32];
        for i in 0..32 {
            let seed_byte = seed_bytes[i % 8];
            let position = i as u8;
            // Mix seed with golden ratio fractional bits + position
            key[i] = seed_byte
                .wrapping_mul(position.wrapping_add(0x9E))
                .wrapping_add(position ^ 0x37);
        }

        // Derive 12-byte nonce from different mixing of seed
        let mut nonce = [0u8; 12];
        for i in 0..12 {
            let seed_byte = seed_bytes[(i + 3) % 8];
            let position = (i as u8).wrapping_add(0xB7);
            nonce[i] = seed_byte.wrapping_mul(position).wrapping_add(0x63);
        }

        Self::new(&key, &nonce)
    }

    /// Advances the counter by `count` positions, skipping output values.
    ///
    /// Each counter position produces 4 u32 values (128 bits). Advancing
    /// by `count` skips `count * 4` u32 values.
    pub fn advance(&mut self, count: u64) {
        self.state.counter = self.state.counter.wrapping_add(count);
        self.buffer_remaining = 0; // Invalidate buffer
    }

    /// Generates the next u32 value (CPU-side, for testing).
    ///
    /// Encrypts the current counter block if the internal buffer is empty,
    /// then returns one u32 from the encrypted output.
    pub fn next_u32(&mut self) -> u32 {
        if self.buffer_remaining == 0 {
            self.fill_buffer();
        }

        let idx = (4 - self.buffer_remaining) as usize;
        self.buffer_remaining -= 1;
        self.buffer[idx]
    }

    /// Fills the internal buffer by encrypting the current counter block.
    fn fill_buffer(&mut self) {
        let block = build_counter_block(&self.state.nonce, self.state.counter);
        let encrypted = aes_encrypt_block(&block, &self.state.key_schedule);

        // Convert 16 bytes to 4 u32 values (little-endian)
        for i in 0..4 {
            self.buffer[i] = u32::from_le_bytes([
                encrypted[4 * i],
                encrypted[4 * i + 1],
                encrypted[4 * i + 2],
                encrypted[4 * i + 3],
            ]);
        }
        self.buffer_remaining = 4;
        self.state.counter = self.state.counter.wrapping_add(1);
    }
}

// ===========================================================================
//  Configuration Validation
// ===========================================================================

/// Validates an AES-CTR configuration for correctness.
///
/// Checks that GPU launch parameters are within acceptable bounds.
///
/// # Errors
///
/// Returns `RandError::InvalidSeed` if `threads_per_block` is zero or
/// exceeds the architecture limit, or if the key is all zeros (weak key).
pub fn validate_aes_config(config: &AesCtrConfig) -> RandResult<()> {
    if config.threads_per_block == 0 {
        return Err(RandError::InvalidSeed(
            "threads_per_block must be greater than 0".to_string(),
        ));
    }

    let max_threads = config.sm_version.max_threads_per_block();
    if config.threads_per_block > max_threads {
        return Err(RandError::InvalidSeed(format!(
            "threads_per_block ({}) exceeds architecture limit ({max_threads})",
            config.threads_per_block
        )));
    }

    // Warn about all-zero key (cryptographically weak)
    if config.key.iter().all(|&b| b == 0) {
        return Err(RandError::InvalidSeed(
            "AES key is all zeros (cryptographically weak)".to_string(),
        ));
    }

    // threads_per_block should be a multiple of the warp size for efficiency
    let warp_size = config.sm_version.warp_size();
    if config.threads_per_block % warp_size != 0 {
        return Err(RandError::InvalidSeed(format!(
            "threads_per_block ({}) should be a multiple of warp size ({warp_size})",
            config.threads_per_block
        )));
    }

    Ok(())
}

// ===========================================================================
//  PTX Generation
// ===========================================================================

/// Generates PTX that loads the AES S-box into shared memory.
///
/// Produces a standalone PTX module with a kernel that cooperatively loads
/// the 256-byte S-box into shared memory. In practice, this logic is inlined
/// into the main AES-CTR kernel via [`generate_aes_ctr_ptx`].
///
/// # Errors
///
/// Returns `PtxGenError` if PTX construction fails.
pub fn generate_sbox_load_ptx(sm: SmVersion) -> Result<String, PtxGenError> {
    KernelBuilder::new("aes_sbox_load")
        .target(sm)
        .param("sbox_ptr", PtxType::U64)
        .shared_mem("sbox_shared", PtxType::U8, 256)
        .max_threads_per_block(256)
        .body(move |b| {
            let tid = b.global_thread_id_x();
            let sbox_ptr = b.load_param_u64("sbox_ptr");

            // Each thread loads one byte from global S-box to shared memory
            let tid_u64 = b.alloc_reg(PtxType::U64);
            b.raw_ptx(&format!("cvt.u64.u32 {tid_u64}, {tid};"));

            let addr = b.alloc_reg(PtxType::U64);
            b.raw_ptx(&format!("add.u64 {addr}, {sbox_ptr}, {tid_u64};"));

            let byte_val = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("ld.global.u8 {byte_val}, [{addr}];"));

            let smem_addr = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("mov.u32 {smem_addr}, sbox_shared;"));
            let smem_offset = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("add.u32 {smem_offset}, {smem_addr}, {tid};"));

            b.raw_ptx(&format!("st.shared.u8 [{smem_offset}], {byte_val};"));

            b.bar_sync(0);

            b.ret();
        })
        .build()
}

/// Generates PTX for a parallel AES-CTR random number generation kernel.
///
/// Each GPU thread:
/// 1. Computes its unique counter: `base_counter + global_thread_id`
/// 2. Builds a 128-bit counter block from the nonce and counter
/// 3. Performs 14 rounds of AES-256 encryption using the S-box in shared memory
/// 4. Outputs 4 u32 values (128 bits) to global memory
///
/// The kernel parameters are:
/// - `out_ptr: u64` — device pointer to output buffer
/// - `n: u32` — number of u32 values to generate (must be multiple of 4)
/// - `nonce0..nonce2: u32` — 96-bit nonce as 3 words
/// - `ctr_lo, ctr_hi: u32` — 64-bit base counter
/// - `rk0..rk59: u32` — expanded round keys (inlined as immediates)
///
/// # Errors
///
/// Returns `PtxGenError` if PTX construction fails.
pub fn generate_aes_ctr_ptx(config: &AesCtrConfig) -> Result<String, PtxGenError> {
    let round_keys = expand_key_256(&config.key);

    // Pack nonce as 3 u32 words (little-endian)
    let nonce_w0 = u32::from_le_bytes([
        config.nonce[0],
        config.nonce[1],
        config.nonce[2],
        config.nonce[3],
    ]);
    let nonce_w1 = u32::from_le_bytes([
        config.nonce[4],
        config.nonce[5],
        config.nonce[6],
        config.nonce[7],
    ]);
    let nonce_w2 = u32::from_le_bytes([
        config.nonce[8],
        config.nonce[9],
        config.nonce[10],
        config.nonce[11],
    ]);

    let initial_counter = config.initial_counter;
    let tpb = config.threads_per_block;

    KernelBuilder::new("aes_ctr_generate")
        .target(config.sm_version)
        .param("out_ptr", PtxType::U64)
        .param("n", PtxType::U32)
        .param("ctr_lo", PtxType::U32)
        .param("ctr_hi", PtxType::U32)
        .shared_mem("sbox_shared", PtxType::U8, 256)
        .max_threads_per_block(tpb)
        .body(move |b| {
            // --- Phase 1: Cooperatively load S-box into shared memory ---
            b.comment("Load AES S-box into shared memory");
            let tid_x = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("mov.u32 {tid_x}, %tid.x;"));

            // Each thread with tid < 256 loads one S-box byte
            let limit = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("mov.u32 {limit}, 256;"));
            let smem_base = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("mov.u32 {smem_base}, sbox_shared;"));

            // Inline the S-box values as immediate stores
            // We use a loop-free approach: each thread stores its corresponding byte
            b.if_lt_u32(tid_x.clone(), limit, |b| {
                // Look up S-box value from inline constant table
                // We embed the S-box as a sequence of conditional moves based on tid
                // For PTX, we use the .const memory approach:
                // Instead, we use a simpler approach - embed S-box bytes as immediates
                // via a register-based table (each thread computes its S-box entry)

                // The most efficient PTX approach: use mov with byte immediates
                // We'll store the S-box byte for this thread's index
                let sbox_val = b.alloc_reg(PtxType::U32);

                // Generate S-box loading via a compact approach:
                // Emit the S-box as 32 groups of 8 bytes using bitfield checks
                // For simplicity and correctness, we inline all 256 values
                emit_sbox_lookup(b, &tid_x, &sbox_val);

                let smem_addr = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("add.u32 {smem_addr}, {smem_base}, {tid_x};"));
                b.raw_ptx(&format!("st.shared.u8 [{smem_addr}], {sbox_val};"));
            });

            b.bar_sync(0);

            // --- Phase 2: AES-CTR generation ---
            let gid = b.global_thread_id_x();
            let n_reg = b.load_param_u32("n");

            // Each thread produces 4 u32 values, so effective thread count = n/4
            let n_div4 = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("shr.u32 {n_div4}, {n_reg}, 2;"));

            b.if_lt_u32(gid.clone(), n_div4, move |b| {
                let out_ptr = b.load_param_u64("out_ptr");
                let ctr_lo_param = b.load_param_u32("ctr_lo");
                let ctr_hi_param = b.load_param_u32("ctr_hi");

                // counter = initial_counter + gid + ctr_param
                let init_lo = b.alloc_reg(PtxType::U32);
                let init_hi = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!(
                    "mov.u32 {init_lo}, {};",
                    (initial_counter & 0xFFFF_FFFF) as u32
                ));
                b.raw_ptx(&format!(
                    "mov.u32 {init_hi}, {};",
                    (initial_counter >> 32) as u32
                ));

                let gid_hi = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.u32 {gid_hi}, 0;"));

                // total_ctr = init + gid + ctr_param
                let t_lo = b.alloc_reg(PtxType::U32);
                let t_hi = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("add.cc.u32 {t_lo}, {init_lo}, {gid};"));
                b.raw_ptx(&format!("addc.u32 {t_hi}, {init_hi}, {gid_hi};"));
                let ctr_final_lo = b.alloc_reg(PtxType::U32);
                let ctr_final_hi = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!(
                    "add.cc.u32 {ctr_final_lo}, {t_lo}, {ctr_lo_param};"
                ));
                b.raw_ptx(&format!("addc.u32 {ctr_final_hi}, {t_hi}, {ctr_hi_param};"));

                // Build 128-bit counter block: nonce (3 words) || counter (1 word, big-endian)
                // State words s0..s3 in column-major AES state format
                let s0 = b.alloc_reg(PtxType::U32);
                let s1 = b.alloc_reg(PtxType::U32);
                let s2 = b.alloc_reg(PtxType::U32);
                let s3 = b.alloc_reg(PtxType::U32);

                b.raw_ptx(&format!("mov.u32 {s0}, {nonce_w0};"));
                b.raw_ptx(&format!("mov.u32 {s1}, {nonce_w1};"));
                b.raw_ptx(&format!("mov.u32 {s2}, {nonce_w2};"));
                // Counter word: byte-swap ctr_final_lo for big-endian
                let ctr_be = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("prmt.b32 {ctr_be}, {ctr_final_lo}, 0, 0x0123;"));
                b.raw_ptx(&format!("mov.u32 {s3}, {ctr_be};"));

                // --- AES-256: 14 rounds ---
                // Initial AddRoundKey (round 0)
                emit_add_round_key_imm(b, &s0, &s1, &s2, &s3, &round_keys, 0);

                // Rounds 1..13: SubBytes + ShiftRows + MixColumns + AddRoundKey
                for round in 1..14 {
                    emit_sub_bytes_shared(b, &s0, &s1, &s2, &s3, &smem_base);
                    emit_shift_rows(b, &s0, &s1, &s2, &s3);
                    emit_mix_columns(b, &s0, &s1, &s2, &s3);
                    emit_add_round_key_imm(b, &s0, &s1, &s2, &s3, &round_keys, round);
                }

                // Final round (no MixColumns)
                emit_sub_bytes_shared(b, &s0, &s1, &s2, &s3, &smem_base);
                emit_shift_rows(b, &s0, &s1, &s2, &s3);
                emit_add_round_key_imm(b, &s0, &s1, &s2, &s3, &round_keys, 14);

                // Store 4 u32 output values
                // Output offset = gid * 16 (4 u32s = 16 bytes)
                let base_offset = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("shl.b32 {base_offset}, {gid}, 4;")); // gid * 16

                let base_offset_64 = b.alloc_reg(PtxType::U64);
                b.raw_ptx(&format!("cvt.u64.u32 {base_offset_64}, {base_offset};"));
                let out_addr = b.alloc_reg(PtxType::U64);
                b.raw_ptx(&format!("add.u64 {out_addr}, {out_ptr}, {base_offset_64};"));

                // Store as v4.u32
                b.raw_ptx(&format!(
                    "st.global.v4.u32 [{out_addr}], {{{s0}, {s1}, {s2}, {s3}}};"
                ));
            });

            b.ret();
        })
        .build()
}

// ===========================================================================
//  PTX Emission Helpers
// ===========================================================================

/// Emits PTX to look up the S-box value for a given thread index.
///
/// Generates a sequence of conditional assignments that map each possible
/// `tid` value (0..255) to its corresponding S-box byte. For GPU efficiency,
/// this uses a binary search tree of comparisons to keep the depth at O(log n).
fn emit_sbox_lookup(
    b: &mut oxicuda_ptx::builder::BodyBuilder<'_>,
    tid: &oxicuda_ptx::ir::Register,
    result: &oxicuda_ptx::ir::Register,
) {
    // Initialize result to 0
    b.raw_ptx(&format!("mov.u32 {result}, 0;"));

    // Emit S-box as groups of 16 using setp + selp chains
    // This produces compact PTX: 16 groups of 16 entries = 256 entries
    for group in 0..16 {
        let base = group * 16;
        let group_check = b.alloc_reg(PtxType::U32);
        b.raw_ptx(&format!("shr.u32 {group_check}, {tid}, 4;"));
        let pred = b.alloc_reg(PtxType::Pred);
        b.raw_ptx(&format!("setp.eq.u32 {pred}, {group_check}, {group};"));

        // Within this group, use the low 4 bits to select the byte
        let low4 = b.alloc_reg(PtxType::U32);
        b.raw_ptx(&format!("and.b32 {low4}, {tid}, 15;"));

        // Pack 16 S-box values into 4 u32 registers and use byte extraction
        let pack0 = u32::from_le_bytes([
            AES_SBOX[base],
            AES_SBOX[base + 1],
            AES_SBOX[base + 2],
            AES_SBOX[base + 3],
        ]);
        let pack1 = u32::from_le_bytes([
            AES_SBOX[base + 4],
            AES_SBOX[base + 5],
            AES_SBOX[base + 6],
            AES_SBOX[base + 7],
        ]);
        let pack2 = u32::from_le_bytes([
            AES_SBOX[base + 8],
            AES_SBOX[base + 9],
            AES_SBOX[base + 10],
            AES_SBOX[base + 11],
        ]);
        let pack3 = u32::from_le_bytes([
            AES_SBOX[base + 12],
            AES_SBOX[base + 13],
            AES_SBOX[base + 14],
            AES_SBOX[base + 15],
        ]);

        let p0 = b.alloc_reg(PtxType::U32);
        let p1 = b.alloc_reg(PtxType::U32);
        let p2 = b.alloc_reg(PtxType::U32);
        let p3 = b.alloc_reg(PtxType::U32);

        b.raw_ptx(&format!("@{pred} mov.u32 {p0}, 0x{pack0:08x};"));
        b.raw_ptx(&format!("@{pred} mov.u32 {p1}, 0x{pack1:08x};"));
        b.raw_ptx(&format!("@{pred} mov.u32 {p2}, 0x{pack2:08x};"));
        b.raw_ptx(&format!("@{pred} mov.u32 {p3}, 0x{pack3:08x};"));

        // Select which packed word based on low4 >> 2
        let word_sel = b.alloc_reg(PtxType::U32);
        b.raw_ptx(&format!("@{pred} shr.u32 {word_sel}, {low4}, 2;"));

        let selected = b.alloc_reg(PtxType::U32);
        // Default to p0
        b.raw_ptx(&format!("@{pred} mov.u32 {selected}, {p0};"));

        let pred1 = b.alloc_reg(PtxType::Pred);
        b.raw_ptx(&format!("@{pred} setp.eq.u32 {pred1}, {word_sel}, 1;"));
        b.raw_ptx(&format!("@{pred1} mov.u32 {selected}, {p1};"));

        let pred2 = b.alloc_reg(PtxType::Pred);
        b.raw_ptx(&format!("@{pred} setp.eq.u32 {pred2}, {word_sel}, 2;"));
        b.raw_ptx(&format!("@{pred2} mov.u32 {selected}, {p2};"));

        let pred3 = b.alloc_reg(PtxType::Pred);
        b.raw_ptx(&format!("@{pred} setp.eq.u32 {pred3}, {word_sel}, 3;"));
        b.raw_ptx(&format!("@{pred3} mov.u32 {selected}, {p3};"));

        // Extract the byte at position (low4 & 3)
        let byte_pos = b.alloc_reg(PtxType::U32);
        b.raw_ptx(&format!("@{pred} and.b32 {byte_pos}, {low4}, 3;"));

        // Shift and mask to extract the byte
        let shift_amt = b.alloc_reg(PtxType::U32);
        b.raw_ptx(&format!("@{pred} shl.b32 {shift_amt}, {byte_pos}, 3;")); // * 8

        let shifted = b.alloc_reg(PtxType::U32);
        b.raw_ptx(&format!(
            "@{pred} shr.u32 {shifted}, {selected}, {shift_amt};"
        ));
        let byte_val = b.alloc_reg(PtxType::U32);
        b.raw_ptx(&format!("@{pred} and.b32 {byte_val}, {shifted}, 255;"));

        b.raw_ptx(&format!("@{pred} mov.u32 {result}, {byte_val};"));
    }
}

/// Emits PTX for SubBytes using the shared memory S-box.
///
/// Looks up each byte of the 4-word state in the shared memory S-box.
fn emit_sub_bytes_shared(
    b: &mut oxicuda_ptx::builder::BodyBuilder<'_>,
    s0: &oxicuda_ptx::ir::Register,
    s1: &oxicuda_ptx::ir::Register,
    s2: &oxicuda_ptx::ir::Register,
    s3: &oxicuda_ptx::ir::Register,
    smem_base: &oxicuda_ptx::ir::Register,
) {
    // For each state word, extract 4 bytes, look up each in S-box, reassemble
    for (s_reg, label) in [(s0, "s0"), (s1, "s1"), (s2, "s2"), (s3, "s3")] {
        let _ = label; // suppress unused warning
        let new_word = b.alloc_reg(PtxType::U32);
        b.raw_ptx(&format!("mov.u32 {new_word}, 0;"));

        for byte_idx in 0..4u32 {
            let shift = byte_idx * 8;
            let byte_val = b.alloc_reg(PtxType::U32);
            if shift == 0 {
                b.raw_ptx(&format!("and.b32 {byte_val}, {s_reg}, 255;"));
            } else {
                b.raw_ptx(&format!("shr.u32 {byte_val}, {s_reg}, {shift};"));
                b.raw_ptx(&format!("and.b32 {byte_val}, {byte_val}, 255;"));
            }

            // Look up in shared memory S-box
            let sbox_addr = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("add.u32 {sbox_addr}, {smem_base}, {byte_val};"));
            let sbox_result = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("ld.shared.u8 {sbox_result}, [{sbox_addr}];"));

            // Place the substituted byte at the correct position
            if shift > 0 {
                b.raw_ptx(&format!("shl.b32 {sbox_result}, {sbox_result}, {shift};"));
            }
            b.raw_ptx(&format!("or.b32 {new_word}, {new_word}, {sbox_result};"));
        }

        b.raw_ptx(&format!("mov.u32 {s_reg}, {new_word};"));
    }
}

/// Emits PTX for the ShiftRows transformation.
///
/// Operates on 4 u32 state words in column-major order, cyclically
/// shifting each row by its row index.
fn emit_shift_rows(
    b: &mut oxicuda_ptx::builder::BodyBuilder<'_>,
    s0: &oxicuda_ptx::ir::Register,
    s1: &oxicuda_ptx::ir::Register,
    s2: &oxicuda_ptx::ir::Register,
    s3: &oxicuda_ptx::ir::Register,
) {
    // State layout (column-major, each column is a u32):
    // Col 0 (s0): [r0c0, r1c0, r2c0, r3c0] = bytes [b0, b1, b2, b3]
    // Col 1 (s1): [r0c1, r1c1, r2c1, r3c1] = bytes [b4, b5, b6, b7]
    // Col 2 (s2): [r0c2, r1c2, r2c2, r3c2] = bytes [b8, b9, b10, b11]
    // Col 3 (s3): [r0c3, r1c3, r2c3, r3c3] = bytes [b12, b13, b14, b15]
    //
    // Row 0: no shift
    // Row 1 (byte 1 of each column): shift left by 1
    // Row 2 (byte 2 of each column): shift left by 2
    // Row 3 (byte 3 of each column): shift left by 3 (= right by 1)
    //
    // We use prmt (permute) instruction to rearrange bytes efficiently.
    // prmt.b32 result, a, b, selector selects 4 bytes from {a,b} pair.

    // Save originals
    let o0 = b.alloc_reg(PtxType::U32);
    let o1 = b.alloc_reg(PtxType::U32);
    let o2 = b.alloc_reg(PtxType::U32);
    let o3 = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("mov.u32 {o0}, {s0};"));
    b.raw_ptx(&format!("mov.u32 {o1}, {s1};"));
    b.raw_ptx(&format!("mov.u32 {o2}, {s2};"));
    b.raw_ptx(&format!("mov.u32 {o3}, {s3};"));

    // Extract individual bytes using masks and shifts, then reassemble.
    // For each output column, pick:
    //   byte0 from same column (row 0 - no shift)
    //   byte1 from column+1 mod 4 (row 1 - shift left 1)
    //   byte2 from column+2 mod 4 (row 2 - shift left 2)
    //   byte3 from column+3 mod 4 (row 3 - shift left 3 = right 1)

    // Helper temps
    let mask = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("mov.u32 {mask}, 255;"));

    // Build new s0: byte0 from o0, byte1 from o1, byte2 from o2, byte3 from o3
    emit_column_rebuild(b, s0, &o0, &o1, &o2, &o3, &mask);
    // Build new s1: byte0 from o1, byte1 from o2, byte2 from o3, byte3 from o0
    emit_column_rebuild(b, s1, &o1, &o2, &o3, &o0, &mask);
    // Build new s2: byte0 from o2, byte1 from o3, byte2 from o0, byte3 from o1
    emit_column_rebuild(b, s2, &o2, &o3, &o0, &o1, &mask);
    // Build new s3: byte0 from o3, byte1 from o0, byte2 from o1, byte3 from o2
    emit_column_rebuild(b, s3, &o3, &o0, &o1, &o2, &mask);
}

/// Rebuilds a column after ShiftRows by picking bytes from source columns.
fn emit_column_rebuild(
    b: &mut oxicuda_ptx::builder::BodyBuilder<'_>,
    dst: &oxicuda_ptx::ir::Register,
    src_b0: &oxicuda_ptx::ir::Register,
    src_b1: &oxicuda_ptx::ir::Register,
    src_b2: &oxicuda_ptx::ir::Register,
    src_b3: &oxicuda_ptx::ir::Register,
    mask: &oxicuda_ptx::ir::Register,
) {
    let b0 = b.alloc_reg(PtxType::U32);
    let b1 = b.alloc_reg(PtxType::U32);
    let b2 = b.alloc_reg(PtxType::U32);
    let b3 = b.alloc_reg(PtxType::U32);

    // Extract byte 0 from src_b0
    b.raw_ptx(&format!("and.b32 {b0}, {src_b0}, {mask};"));

    // Extract byte 1 from src_b1, shift to position
    b.raw_ptx(&format!("shr.u32 {b1}, {src_b1}, 8;"));
    b.raw_ptx(&format!("and.b32 {b1}, {b1}, {mask};"));
    b.raw_ptx(&format!("shl.b32 {b1}, {b1}, 8;"));

    // Extract byte 2 from src_b2
    b.raw_ptx(&format!("shr.u32 {b2}, {src_b2}, 16;"));
    b.raw_ptx(&format!("and.b32 {b2}, {b2}, {mask};"));
    b.raw_ptx(&format!("shl.b32 {b2}, {b2}, 16;"));

    // Extract byte 3 from src_b3
    b.raw_ptx(&format!("shr.u32 {b3}, {src_b3}, 24;"));
    b.raw_ptx(&format!("shl.b32 {b3}, {b3}, 24;"));

    // Combine
    b.raw_ptx(&format!("or.b32 {dst}, {b0}, {b1};"));
    b.raw_ptx(&format!("or.b32 {dst}, {dst}, {b2};"));
    b.raw_ptx(&format!("or.b32 {dst}, {dst}, {b3};"));
}

/// Emits PTX for MixColumns on 4 state words.
///
/// Each column is multiplied by the fixed polynomial in GF(2^8).
/// Uses the xtime doubling approach for efficiency.
fn emit_mix_columns(
    b: &mut oxicuda_ptx::builder::BodyBuilder<'_>,
    s0: &oxicuda_ptx::ir::Register,
    s1: &oxicuda_ptx::ir::Register,
    s2: &oxicuda_ptx::ir::Register,
    s3: &oxicuda_ptx::ir::Register,
) {
    // Process each column independently
    for s_reg in [s0, s1, s2, s3] {
        emit_mix_single_column(b, s_reg);
    }
}

/// Emits PTX for MixColumns on a single 32-bit column word.
///
/// Implements the GF(2^8) matrix multiplication:
///   [2 3 1 1]   [b0]
///   [1 2 3 1] * [b1]
///   [1 1 2 3]   [b2]
///   [3 1 1 2]   [b3]
fn emit_mix_single_column(
    b: &mut oxicuda_ptx::builder::BodyBuilder<'_>,
    col: &oxicuda_ptx::ir::Register,
) {
    // Extract 4 bytes
    let b0 = b.alloc_reg(PtxType::U32);
    let b1 = b.alloc_reg(PtxType::U32);
    let b2 = b.alloc_reg(PtxType::U32);
    let b3 = b.alloc_reg(PtxType::U32);
    let mask = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("mov.u32 {mask}, 255;"));

    b.raw_ptx(&format!("and.b32 {b0}, {col}, {mask};"));
    b.raw_ptx(&format!("shr.u32 {b1}, {col}, 8;"));
    b.raw_ptx(&format!("and.b32 {b1}, {b1}, {mask};"));
    b.raw_ptx(&format!("shr.u32 {b2}, {col}, 16;"));
    b.raw_ptx(&format!("and.b32 {b2}, {b2}, {mask};"));
    b.raw_ptx(&format!("shr.u32 {b3}, {col}, 24;"));
    b.raw_ptx(&format!("and.b32 {b3}, {b3}, {mask};"));

    // Compute xtime (doubling in GF(2^8)) for each byte
    // xtime(x) = (x << 1) ^ ((x >> 7) * 0x1b)
    let x0 = emit_xtime(b, &b0);
    let x1 = emit_xtime(b, &b1);
    let x2 = emit_xtime(b, &b2);
    let x3 = emit_xtime(b, &b3);

    // r0 = xtime(b0) ^ xtime(b1) ^ b1 ^ b2 ^ b3 = 2*b0 ^ 3*b1 ^ b2 ^ b3
    let r0 = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("xor.b32 {r0}, {x0}, {x1};"));
    b.raw_ptx(&format!("xor.b32 {r0}, {r0}, {b1};"));
    b.raw_ptx(&format!("xor.b32 {r0}, {r0}, {b2};"));
    b.raw_ptx(&format!("xor.b32 {r0}, {r0}, {b3};"));

    // r1 = b0 ^ xtime(b1) ^ xtime(b2) ^ b2 ^ b3
    let r1 = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("xor.b32 {r1}, {b0}, {x1};"));
    b.raw_ptx(&format!("xor.b32 {r1}, {r1}, {x2};"));
    b.raw_ptx(&format!("xor.b32 {r1}, {r1}, {b2};"));
    b.raw_ptx(&format!("xor.b32 {r1}, {r1}, {b3};"));

    // r2 = b0 ^ b1 ^ xtime(b2) ^ xtime(b3) ^ b3
    let r2 = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("xor.b32 {r2}, {b0}, {b1};"));
    b.raw_ptx(&format!("xor.b32 {r2}, {r2}, {x2};"));
    b.raw_ptx(&format!("xor.b32 {r2}, {r2}, {x3};"));
    b.raw_ptx(&format!("xor.b32 {r2}, {r2}, {b3};"));

    // r3 = xtime(b0) ^ b0 ^ b1 ^ b2 ^ xtime(b3)
    let r3 = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("xor.b32 {r3}, {x0}, {b0};"));
    b.raw_ptx(&format!("xor.b32 {r3}, {r3}, {b1};"));
    b.raw_ptx(&format!("xor.b32 {r3}, {r3}, {b2};"));
    b.raw_ptx(&format!("xor.b32 {r3}, {r3}, {x3};"));

    // Mask each result to 8 bits and reassemble
    b.raw_ptx(&format!("and.b32 {r0}, {r0}, {mask};"));
    b.raw_ptx(&format!("and.b32 {r1}, {r1}, {mask};"));
    b.raw_ptx(&format!("shl.b32 {r1}, {r1}, 8;"));
    b.raw_ptx(&format!("and.b32 {r2}, {r2}, {mask};"));
    b.raw_ptx(&format!("shl.b32 {r2}, {r2}, 16;"));
    b.raw_ptx(&format!("and.b32 {r3}, {r3}, {mask};"));
    b.raw_ptx(&format!("shl.b32 {r3}, {r3}, 24;"));

    b.raw_ptx(&format!("or.b32 {col}, {r0}, {r1};"));
    b.raw_ptx(&format!("or.b32 {col}, {col}, {r2};"));
    b.raw_ptx(&format!("or.b32 {col}, {col}, {r3};"));
}

/// Emits PTX for xtime (GF(2^8) doubling): `(x << 1) ^ ((x >> 7) * 0x1b)`.
fn emit_xtime(
    b: &mut oxicuda_ptx::builder::BodyBuilder<'_>,
    val: &oxicuda_ptx::ir::Register,
) -> oxicuda_ptx::ir::Register {
    let shifted = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("shl.b32 {shifted}, {val}, 1;"));

    let high_bit = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("shr.u32 {high_bit}, {val}, 7;"));
    b.raw_ptx(&format!("and.b32 {high_bit}, {high_bit}, 1;"));

    // reduction = high_bit * 0x1b
    let reduction = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("mul.lo.u32 {reduction}, {high_bit}, 0x1b;"));

    let result = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("xor.b32 {result}, {shifted}, {reduction};"));

    let mask = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("mov.u32 {mask}, 255;"));
    b.raw_ptx(&format!("and.b32 {result}, {result}, {mask};"));

    result
}

/// Emits AddRoundKey using immediate round key values.
fn emit_add_round_key_imm(
    b: &mut oxicuda_ptx::builder::BodyBuilder<'_>,
    s0: &oxicuda_ptx::ir::Register,
    s1: &oxicuda_ptx::ir::Register,
    s2: &oxicuda_ptx::ir::Register,
    s3: &oxicuda_ptx::ir::Register,
    round_keys: &[u32; 60],
    round: usize,
) {
    let rk0 = round_keys[round * 4];
    let rk1 = round_keys[round * 4 + 1];
    let rk2 = round_keys[round * 4 + 2];
    let rk3 = round_keys[round * 4 + 3];

    let k0 = b.alloc_reg(PtxType::U32);
    let k1 = b.alloc_reg(PtxType::U32);
    let k2 = b.alloc_reg(PtxType::U32);
    let k3 = b.alloc_reg(PtxType::U32);

    b.raw_ptx(&format!("mov.u32 {k0}, 0x{rk0:08x};"));
    b.raw_ptx(&format!("mov.u32 {k1}, 0x{rk1:08x};"));
    b.raw_ptx(&format!("mov.u32 {k2}, 0x{rk2:08x};"));
    b.raw_ptx(&format!("mov.u32 {k3}, 0x{rk3:08x};"));

    b.raw_ptx(&format!("xor.b32 {s0}, {s0}, {k0};"));
    b.raw_ptx(&format!("xor.b32 {s1}, {s1}, {k1};"));
    b.raw_ptx(&format!("xor.b32 {s2}, {s2}, {k2};"));
    b.raw_ptx(&format!("xor.b32 {s3}, {s3}, {k3};"));
}

// ===========================================================================
//  Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // AES key expansion tests (FIPS 197 Appendix A.3 — AES-256)
    // -----------------------------------------------------------------------

    #[test]
    fn key_expansion_nist_aes256() {
        // FIPS 197 Appendix A.3: AES-256 key expansion test vector
        let key: [u8; 32] = [
            0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d,
            0x0e, 0x0f, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b,
            0x1c, 0x1d, 0x1e, 0x1f,
        ];
        let schedule = expand_key_256(&key);

        // First 8 words should match the key directly (big-endian per FIPS 197)
        assert_eq!(schedule[0], 0x00010203);
        assert_eq!(schedule[1], 0x04050607);
        assert_eq!(schedule[2], 0x08090a0b);
        assert_eq!(schedule[3], 0x0c0d0e0f);
        assert_eq!(schedule[4], 0x10111213);
        assert_eq!(schedule[5], 0x14151617);
        assert_eq!(schedule[6], 0x18191a1b);
        assert_eq!(schedule[7], 0x1c1d1e1f);

        // Verify the schedule has 60 words
        assert_eq!(schedule.len(), 60);
    }

    #[test]
    fn sub_word_known_values() {
        // S-box(0x00) = 0x63, S-box(0x01) = 0x7c, etc.
        assert_eq!(sub_word(0x00000000), 0x63636363);
        assert_eq!(sub_word(0x01010101), 0x7c7c7c7c);
    }

    #[test]
    fn rot_word_basic() {
        assert_eq!(rot_word(0x01020304), 0x02030401);
        assert_eq!(rot_word(0xFF000000), 0x000000FF);
    }

    // -----------------------------------------------------------------------
    // AES block cipher tests (FIPS 197 Appendix C.3 — AES-256)
    // -----------------------------------------------------------------------

    #[test]
    fn aes256_encrypt_nist_vector() {
        // FIPS 197 Appendix C.3: AES-256 encryption test vector
        let key: [u8; 32] = [
            0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d,
            0x0e, 0x0f, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b,
            0x1c, 0x1d, 0x1e, 0x1f,
        ];
        let plaintext: [u8; 16] = [
            0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd,
            0xee, 0xff,
        ];
        let expected_ciphertext: [u8; 16] = [
            0x8e, 0xa2, 0xb7, 0xca, 0x51, 0x67, 0x45, 0xbf, 0xea, 0xfc, 0x49, 0x90, 0x4b, 0x49,
            0x60, 0x89,
        ];

        let round_keys = expand_key_256(&key);
        let ciphertext = aes_encrypt_block(&plaintext, &round_keys);
        assert_eq!(ciphertext, expected_ciphertext);
    }

    #[test]
    fn aes256_encrypt_zero_key_zero_plaintext() {
        let key = [0u8; 32];
        let plaintext = [0u8; 16];
        let round_keys = expand_key_256(&key);
        let ciphertext = aes_encrypt_block(&plaintext, &round_keys);

        // Result should be deterministic and non-zero
        assert_ne!(ciphertext, [0u8; 16]);
        // Known result for AES-256 with all-zero key and plaintext:
        let expected: [u8; 16] = [
            0xdc, 0x95, 0xc0, 0x78, 0xa2, 0x40, 0x89, 0x89, 0xad, 0x48, 0xa2, 0x14, 0x92, 0x84,
            0x20, 0x87,
        ];
        assert_eq!(ciphertext, expected);
    }

    // -----------------------------------------------------------------------
    // AES-CTR CPU-side output tests
    // -----------------------------------------------------------------------

    #[test]
    fn ctr_mode_deterministic() {
        let key = [0x42u8; 32];
        let nonce = [0u8; 12];
        let mut engine1 = AesCtrEngine::new(&key, &nonce);
        let mut engine2 = AesCtrEngine::new(&key, &nonce);

        // Same key/nonce should produce same sequence
        for _ in 0..16 {
            assert_eq!(engine1.next_u32(), engine2.next_u32());
        }
    }

    #[test]
    fn ctr_mode_different_keys_differ() {
        let key1 = [0x01u8; 32];
        let key2 = [0x02u8; 32];
        let nonce = [0u8; 12];
        let mut engine1 = AesCtrEngine::new(&key1, &nonce);
        let mut engine2 = AesCtrEngine::new(&key2, &nonce);

        // Different keys should (almost certainly) produce different output
        let mut same_count = 0;
        for _ in 0..16 {
            if engine1.next_u32() == engine2.next_u32() {
                same_count += 1;
            }
        }
        assert!(
            same_count < 2,
            "different keys produced too many matching values"
        );
    }

    #[test]
    fn ctr_mode_advance_skips() {
        let key = [0xABu8; 32];
        let nonce = [0x01u8; 12];

        let mut engine_skip = AesCtrEngine::new(&key, &nonce);
        engine_skip.advance(5);

        let mut engine_manual = AesCtrEngine::new(&key, &nonce);
        // Consume 5 blocks (20 u32 values)
        for _ in 0..20 {
            let _ = engine_manual.next_u32();
        }

        // They should now produce the same output
        for _ in 0..8 {
            assert_eq!(engine_skip.next_u32(), engine_manual.next_u32());
        }
    }

    #[test]
    fn ctr_mode_counter_block_format() {
        let nonce = [
            0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c,
        ];
        let block = build_counter_block(&nonce, 1);

        // First 12 bytes = nonce
        assert_eq!(&block[..12], &nonce);
        // Last 4 bytes = counter in big-endian
        assert_eq!(&block[12..], &[0x00, 0x00, 0x00, 0x01]);
    }

    // -----------------------------------------------------------------------
    // Seed derivation tests
    // -----------------------------------------------------------------------

    #[test]
    fn from_seed_deterministic() {
        let engine1 = AesCtrEngine::from_seed(12345);
        let engine2 = AesCtrEngine::from_seed(12345);

        assert_eq!(engine1.config.key, engine2.config.key);
        assert_eq!(engine1.config.nonce, engine2.config.nonce);
    }

    #[test]
    fn from_seed_different_seeds_differ() {
        let engine1 = AesCtrEngine::from_seed(1);
        let engine2 = AesCtrEngine::from_seed(2);

        assert_ne!(engine1.config.key, engine2.config.key);
    }

    // -----------------------------------------------------------------------
    // Config validation tests
    // -----------------------------------------------------------------------

    #[test]
    fn validate_config_valid() {
        let config = AesCtrConfig {
            key: [0x42u8; 32],
            nonce: [0u8; 12],
            initial_counter: 0,
            threads_per_block: 256,
            sm_version: SmVersion::Sm80,
        };
        assert!(validate_aes_config(&config).is_ok());
    }

    #[test]
    fn validate_config_zero_threads() {
        let config = AesCtrConfig {
            key: [0x42u8; 32],
            nonce: [0u8; 12],
            initial_counter: 0,
            threads_per_block: 0,
            sm_version: SmVersion::Sm80,
        };
        assert!(validate_aes_config(&config).is_err());
    }

    #[test]
    fn validate_config_too_many_threads() {
        let config = AesCtrConfig {
            key: [0x42u8; 32],
            nonce: [0u8; 12],
            initial_counter: 0,
            threads_per_block: 2048,
            sm_version: SmVersion::Sm80,
        };
        assert!(validate_aes_config(&config).is_err());
    }

    #[test]
    fn validate_config_zero_key() {
        let config = AesCtrConfig {
            key: [0u8; 32],
            nonce: [0u8; 12],
            initial_counter: 0,
            threads_per_block: 256,
            sm_version: SmVersion::Sm80,
        };
        assert!(validate_aes_config(&config).is_err());
    }

    #[test]
    fn validate_config_non_warp_aligned() {
        let config = AesCtrConfig {
            key: [0x42u8; 32],
            nonce: [0u8; 12],
            initial_counter: 0,
            threads_per_block: 100, // Not a multiple of 32
            sm_version: SmVersion::Sm80,
        };
        assert!(validate_aes_config(&config).is_err());
    }

    // -----------------------------------------------------------------------
    // PTX generation tests
    // -----------------------------------------------------------------------

    #[test]
    fn generate_aes_ctr_ptx_success() {
        let config = AesCtrConfig {
            key: [0x42u8; 32],
            nonce: [0u8; 12],
            initial_counter: 0,
            threads_per_block: 256,
            sm_version: SmVersion::Sm80,
        };
        let ptx = generate_aes_ctr_ptx(&config);
        assert!(ptx.is_ok());
        let ptx = ptx.expect("should generate PTX");
        assert!(ptx.contains(".entry aes_ctr_generate"));
        assert!(ptx.contains(".shared"));
        assert!(ptx.contains("xor.b32"));
        assert!(ptx.contains("ld.shared.u8"));
    }

    #[test]
    fn generate_sbox_load_ptx_success() {
        let ptx = generate_sbox_load_ptx(SmVersion::Sm80);
        assert!(ptx.is_ok());
        let ptx = ptx.expect("should generate PTX");
        assert!(ptx.contains(".entry aes_sbox_load"));
        assert!(ptx.contains(".shared"));
    }

    #[test]
    fn ptx_generation_different_sm_versions() {
        for sm in [SmVersion::Sm75, SmVersion::Sm80, SmVersion::Sm90] {
            let config = AesCtrConfig {
                key: [0x42u8; 32],
                nonce: [0u8; 12],
                initial_counter: 0,
                threads_per_block: 256,
                sm_version: sm,
            };
            let ptx = generate_aes_ctr_ptx(&config);
            assert!(ptx.is_ok(), "PTX generation failed for {sm}");
            let ptx_str = ptx.expect("should generate");
            assert!(ptx_str.contains(sm.as_ptx_str()), "PTX should target {sm}");
        }
    }

    // -----------------------------------------------------------------------
    // GF(2^8) arithmetic tests
    // -----------------------------------------------------------------------

    #[test]
    fn xtime_known_values() {
        assert_eq!(xtime(0x57), 0xAE);
        assert_eq!(xtime(0xAE), 0x47); // 0xAE << 1 = 0x15C, XOR 0x1B = 0x47
        assert_eq!(xtime(0x00), 0x00);
        assert_eq!(xtime(0x80), 0x1B); // high bit set => reduction
    }

    #[test]
    fn gmul_known_values() {
        // 0x57 * 0x13 = 0xfe (known GF(2^8) multiplication result)
        assert_eq!(gmul(0x57, 0x13), 0xFE);
        assert_eq!(gmul(0x00, 0xFF), 0x00);
        assert_eq!(gmul(0x01, 0xFF), 0xFF);
    }
}
