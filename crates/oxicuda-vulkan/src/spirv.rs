//! SPIR-V utilities for the Vulkan compute backend.
//!
//! This module provides a minimal SPIR-V module builder and a pre-encoded
//! placeholder compute shader.  The placeholder is a valid "do nothing"
//! `void main()` with `LocalSize(1,1,1)` that satisfies the Vulkan spec
//! requirements for shader module creation, even though all compute
//! operations in this backend currently return `Unsupported`.
//!
//! # SPIR-V binary format
//!
//! SPIR-V is a sequence of 32-bit words (little-endian on all current
//! platforms).  Each instruction is encoded as:
//!
//! ```text
//! word[0] = (word_count << 16) | opcode
//! word[1..n] = operands
//! ```
//!
//! The module header is always five words:
//!
//! ```text
//! 0x07230203  magic
//! version     e.g. 0x00010500 = 1.5
//! generator   arbitrary; we use 0x000D_000D
//! bound       highest ID used + 1
//! schema      reserved (0)
//! ```

/// SPIR-V magic number (identifies a binary as SPIR-V).
pub const SPIRV_MAGIC: u32 = 0x07230203;
/// SPIR-V version 1.2 (widely supported, no ray-tracing extras needed).
pub const SPIRV_VERSION_1_2: u32 = 0x0001_0200;
/// Generator magic — OxiCUDA Vulkan backend.
pub const SPIRV_GENERATOR: u32 = 0x000D_0001;

// ─── Minimal SPIR-V builder ──────────────────────────────────

/// Lightweight SPIR-V word-stream builder.
///
/// Emits valid SPIR-V instructions for simple compute shaders without
/// pulling in a full compiler.
pub struct SpvModule {
    words: Vec<u32>,
    /// Next available result ID.
    id_bound: u32,
}

impl SpvModule {
    /// Create a new module with a placeholder header (bound filled at finalise).
    pub fn new() -> Self {
        // Five header words; bound (word[3]) is filled by `finalize`.
        let words = vec![
            SPIRV_MAGIC,
            SPIRV_VERSION_1_2,
            SPIRV_GENERATOR,
            0, // bound — filled in finalize()
            0, // schema
        ];
        Self { words, id_bound: 1 }
    }

    /// Allocate a fresh result ID.
    pub fn alloc_id(&mut self) -> u32 {
        let id = self.id_bound;
        self.id_bound += 1;
        id
    }

    /// Emit a SPIR-V instruction.
    ///
    /// `opcode` is the raw opcode value; `operands` are the additional words.
    pub fn emit(&mut self, opcode: u32, operands: &[u32]) {
        let word_count = (1 + operands.len()) as u32;
        self.words.push((word_count << 16) | opcode);
        self.words.extend_from_slice(operands);
    }

    /// Emit a string as null-terminated UTF-8 packed into 32-bit words.
    pub fn string_words(s: &str) -> Vec<u32> {
        let bytes = s.as_bytes();
        // Pad to a multiple of 4, with at least one null terminator.
        let padded_len = (bytes.len() + 4) & !3;
        let mut out = vec![0u32; padded_len / 4];
        for (i, &b) in bytes.iter().enumerate() {
            let word_idx = i / 4;
            let byte_idx = i % 4;
            out[word_idx] |= (b as u32) << (byte_idx * 8);
        }
        out
    }

    /// Finalise the module: patch the ID bound and return the word vector.
    pub fn finalize(mut self) -> Vec<u32> {
        self.words[3] = self.id_bound;
        self.words
    }
}

impl Default for SpvModule {
    fn default() -> Self {
        Self::new()
    }
}

// ─── SPIR-V opcode constants ─────────────────────────────────

const OP_CAPABILITY: u32 = 17;
const OP_MEMORY_MODEL: u32 = 14;
const OP_ENTRY_POINT: u32 = 15;
const OP_EXECUTION_MODE: u32 = 16;
const OP_TYPE_VOID: u32 = 19;
const OP_TYPE_FUNCTION: u32 = 33;
const OP_FUNCTION: u32 = 54;
const OP_LABEL: u32 = 248;
const OP_RETURN: u32 = 253;
const OP_FUNCTION_END: u32 = 56;

// Capability
const CAPABILITY_SHADER: u32 = 1;
// Addressing / memory model
const ADDRESSING_MODEL_LOGICAL: u32 = 0;
const MEMORY_MODEL_GLSL450: u32 = 1;
// Execution model
const EXECUTION_MODEL_GLCOMPUTE: u32 = 5;
// Execution mode
const EXECUTION_MODE_LOCAL_SIZE: u32 = 17;
// Function control
const FUNCTION_CONTROL_NONE: u32 = 0;

/// Build a minimal valid compute shader: `void main() {}` with `LocalSize(1,1,1)`.
///
/// The resulting SPIR-V module is suitable for `vkCreateShaderModule` and
/// serves as a placeholder while real kernel SPIR-V is not yet available.
pub fn trivial_compute_shader() -> Vec<u32> {
    let mut m = SpvModule::new();

    // IDs
    let id_main_fn = m.alloc_id(); // 1 — the entry-point function
    let id_void = m.alloc_id(); // 2 — OpTypeVoid
    let id_void_fn = m.alloc_id(); // 3 — OpTypeFunction %void
    let id_label = m.alloc_id(); // 4 — OpLabel inside main

    // ── Global section ──────────────────────────────────────

    // OpCapability Shader
    m.emit(OP_CAPABILITY, &[CAPABILITY_SHADER]);

    // OpMemoryModel Logical GLSL450
    m.emit(
        OP_MEMORY_MODEL,
        &[ADDRESSING_MODEL_LOGICAL, MEMORY_MODEL_GLSL450],
    );

    // OpEntryPoint GLCompute %main "main"
    // Format: execution_model result_id name_words...
    let mut entry_words = vec![EXECUTION_MODEL_GLCOMPUTE, id_main_fn];
    entry_words.extend(SpvModule::string_words("main"));
    m.emit(OP_ENTRY_POINT, &entry_words);

    // OpExecutionMode %main LocalSize 1 1 1
    m.emit(
        OP_EXECUTION_MODE,
        &[id_main_fn, EXECUTION_MODE_LOCAL_SIZE, 1, 1, 1],
    );

    // ── Type declarations ────────────────────────────────────

    // OpTypeVoid %void
    m.emit(OP_TYPE_VOID, &[id_void]);

    // OpTypeFunction %void_fn %void
    m.emit(OP_TYPE_FUNCTION, &[id_void_fn, id_void]);

    // ── Function body ────────────────────────────────────────

    // OpFunction %void %main None %void_fn
    m.emit(
        OP_FUNCTION,
        &[id_void, id_main_fn, FUNCTION_CONTROL_NONE, id_void_fn],
    );

    // OpLabel %entry
    m.emit(OP_LABEL, &[id_label]);

    // OpReturn
    m.emit(OP_RETURN, &[]);

    // OpFunctionEnd
    m.emit(OP_FUNCTION_END, &[]);

    m.finalize()
}

/// Return the trivial compute shader as a byte slice suitable for
/// passing to `vkCreateShaderModule`.
///
/// The bytes are the native-endian representation of the SPIR-V words,
/// which is correct for the current platform.  Vulkan requires the module
/// to be a valid SPIR-V binary.
pub fn trivial_compute_shader_bytes() -> Vec<u8> {
    trivial_compute_shader()
        .iter()
        .flat_map(|w| w.to_ne_bytes())
        .collect()
}

// ─── Tests ──────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn placeholder_spv_valid_magic() {
        let words = trivial_compute_shader();
        assert!(!words.is_empty(), "SPIR-V module must not be empty");
        assert_eq!(words[0], SPIRV_MAGIC, "first word must be SPIR-V magic");
    }

    #[test]
    fn placeholder_spv_word_aligned() {
        let bytes = trivial_compute_shader_bytes();
        assert_eq!(bytes.len() % 4, 0, "SPIR-V must be 4-byte aligned");
    }

    #[test]
    fn placeholder_spv_version_and_schema() {
        let words = trivial_compute_shader();
        assert!(words.len() >= 5, "header must have 5 words");
        // version >= 1.0 (0x00010000)
        assert!(words[1] >= 0x0001_0000, "SPIR-V version must be >= 1.0");
        assert_eq!(words[4], 0, "schema word must be 0");
    }

    #[test]
    fn placeholder_spv_nonzero_bound() {
        let words = trivial_compute_shader();
        assert!(words[3] > 0, "ID bound must be > 0 when IDs are allocated");
    }

    #[test]
    fn spv_module_id_allocation_is_monotonic() {
        let mut m = SpvModule::new();
        let id1 = m.alloc_id();
        let id2 = m.alloc_id();
        assert!(id2 > id1);
    }

    #[test]
    fn string_words_null_terminated() {
        let words = SpvModule::string_words("abc");
        // "abc\0" packed into one 32-bit word
        assert!(!words.is_empty());
        // Reconstruct bytes
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        // Must contain 'a', 'b', 'c' followed by a null byte
        assert_eq!(bytes[0], b'a');
        assert_eq!(bytes[1], b'b');
        assert_eq!(bytes[2], b'c');
        assert_eq!(bytes[3], 0);
    }

    #[test]
    fn string_words_empty_string() {
        let words = SpvModule::string_words("");
        // Even the empty string must produce at least one word (null terminator).
        assert!(!words.is_empty());
        // First byte must be null.
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        assert_eq!(bytes[0], 0);
    }
}
