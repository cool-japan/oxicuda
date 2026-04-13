//! Texture and surface operation types for PTX instructions.
//!
//! This module defines enumerations for texture dimensionality and surface
//! operation kinds, used by the `tex` and `suld`/`sust` instruction families.

// ---------------------------------------------------------------------------
// Texture dimension
// ---------------------------------------------------------------------------

/// Texture dimension for `tex` instructions.
///
/// PTX texture fetch instructions are suffixed with the dimensionality
/// (`.1d`, `.2d`, `.3d`) to specify the coordinate space.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TextureDim {
    /// 1-dimensional texture.
    Tex1d,
    /// 2-dimensional texture.
    Tex2d,
    /// 3-dimensional texture.
    Tex3d,
}

impl TextureDim {
    /// Returns the PTX dimension suffix (e.g., `".1d"`, `".2d"`, `".3d"`).
    #[must_use]
    pub const fn as_ptx_str(self) -> &'static str {
        match self {
            Self::Tex1d => ".1d",
            Self::Tex2d => ".2d",
            Self::Tex3d => ".3d",
        }
    }
}

// ---------------------------------------------------------------------------
// Surface operation
// ---------------------------------------------------------------------------

/// Surface operation kind for `suld` (load) and `sust` (store) instructions.
///
/// Surfaces provide typed access to CUDA surface memory, supporting both
/// read and write operations through the hardware texture unit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SurfaceOp {
    /// Surface load (`suld`).
    Load,
    /// Surface store (`sust`).
    Store,
}

impl SurfaceOp {
    /// Returns the PTX instruction prefix (e.g., `"suld"`, `"sust"`).
    #[must_use]
    pub const fn as_ptx_str(self) -> &'static str {
        match self {
            Self::Load => "suld",
            Self::Store => "sust",
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn texture_dim_ptx_str() {
        assert_eq!(TextureDim::Tex1d.as_ptx_str(), ".1d");
        assert_eq!(TextureDim::Tex2d.as_ptx_str(), ".2d");
        assert_eq!(TextureDim::Tex3d.as_ptx_str(), ".3d");
    }

    #[test]
    fn surface_op_ptx_str() {
        assert_eq!(SurfaceOp::Load.as_ptx_str(), "suld");
        assert_eq!(SurfaceOp::Store.as_ptx_str(), "sust");
    }
}
