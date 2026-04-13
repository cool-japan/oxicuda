//! Disk-based PTX kernel cache.
//!
//! [`PtxCache`] provides persistent caching of generated PTX text on disk,
//! keyed by kernel name, parameter hash, and target architecture. This avoids
//! redundant PTX generation for kernels that have already been compiled.
//!
//! The cache stores files at `~/.cache/oxicuda/ptx/` (or a fallback location
//! under `std::env::temp_dir()` if the home directory is unavailable).
//!
//! # Example
//!
//! ```
//! use oxicuda_ptx::cache::{PtxCache, PtxCacheKey};
//! use oxicuda_ptx::arch::SmVersion;
//!
//! let cache = PtxCache::new().expect("cache init failed");
//! let key = PtxCacheKey {
//!     kernel_name: "vector_add".to_string(),
//!     params_hash: 0x12345678,
//!     sm_version: SmVersion::Sm80,
//! };
//!
//! let ptx = cache.get_or_generate(&key, || {
//!     Ok("// generated PTX".to_string())
//! }).expect("generation failed");
//! assert!(ptx.contains("generated PTX"));
//! # cache.clear().ok();
//! ```

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;

use crate::arch::SmVersion;
use crate::error::PtxGenError;

/// Disk-based PTX kernel cache.
///
/// Caches generated PTX text files on disk to avoid redundant code generation.
/// Files are stored as `{kernel_name}_{sm}_{hash:016x}.ptx` in the cache
/// directory.
pub struct PtxCache {
    /// The root cache directory.
    cache_dir: PathBuf,
}

/// Cache lookup key for PTX kernels.
///
/// The key combines the kernel name, a hash of the generation parameters,
/// and the target architecture to produce a unique filename.
#[derive(Debug, Clone, Hash)]
pub struct PtxCacheKey {
    /// The kernel function name.
    pub kernel_name: String,
    /// A hash of the kernel generation parameters (tile sizes, precisions, etc.).
    pub params_hash: u64,
    /// The target GPU architecture.
    pub sm_version: SmVersion,
}

impl PtxCacheKey {
    /// Converts this key to a filename suitable for disk storage.
    ///
    /// Format: `{kernel_name}_{sm}_{combined_hash:016x}.ptx`
    ///
    /// The combined hash includes both the `params_hash` and the full key hash
    /// to minimize collision risk.
    #[must_use]
    pub fn to_filename(&self) -> String {
        let mut hasher = DefaultHasher::new();
        self.hash(&mut hasher);
        let full_hash = hasher.finish();
        format!(
            "{}_{}_{:016x}.ptx",
            sanitize_filename(&self.kernel_name),
            self.sm_version.as_ptx_str(),
            full_hash
        )
    }
}

impl PtxCache {
    /// Creates a new PTX cache, initializing the cache directory.
    ///
    /// The cache directory is `~/.cache/oxicuda/ptx/`. If the home directory
    /// cannot be determined, falls back to `{temp_dir}/oxicuda_ptx_cache/`.
    ///
    /// # Errors
    ///
    /// Returns `std::io::Error` if the cache directory cannot be created.
    pub fn new() -> Result<Self, std::io::Error> {
        let cache_dir = resolve_cache_dir();
        std::fs::create_dir_all(&cache_dir)?;
        Ok(Self { cache_dir })
    }

    /// Creates a new PTX cache at a specific directory.
    ///
    /// Useful for testing or when a custom cache location is desired.
    ///
    /// # Errors
    ///
    /// Returns `std::io::Error` if the directory cannot be created.
    pub fn with_dir(dir: PathBuf) -> Result<Self, std::io::Error> {
        std::fs::create_dir_all(&dir)?;
        Ok(Self { cache_dir: dir })
    }

    /// Returns the cache directory path.
    #[must_use]
    pub const fn cache_dir(&self) -> &PathBuf {
        &self.cache_dir
    }

    /// Looks up a cached PTX string, or generates and caches it if not found.
    ///
    /// If the cache contains a file matching the key, its contents are returned
    /// directly. Otherwise, the `generate` closure is called to produce the PTX
    /// text, which is then written to the cache before being returned.
    ///
    /// # Errors
    ///
    /// Returns [`PtxGenError`] if:
    /// - The generate closure fails
    /// - Disk I/O fails during read or write
    pub fn get_or_generate<F>(&self, key: &PtxCacheKey, generate: F) -> Result<String, PtxGenError>
    where
        F: FnOnce() -> Result<String, PtxGenError>,
    {
        let path = self.cache_dir.join(key.to_filename());

        // Try to read from cache
        match std::fs::read_to_string(&path) {
            Ok(contents) if !contents.is_empty() => return Ok(contents),
            _ => {}
        }

        // Generate fresh PTX
        let ptx = generate()?;

        // Write to cache (best-effort; cache write failure is non-fatal)
        if let Err(e) = std::fs::write(&path, &ptx) {
            // Log the error but don't fail the generation
            eprintln!(
                "oxicuda-ptx: cache write failed for {}: {e}",
                path.display()
            );
        }

        Ok(ptx)
    }

    /// Retrieves cached PTX for the given key, if it exists.
    ///
    /// Returns `None` if no cached entry is found or the file is empty.
    #[must_use]
    pub fn get(&self, key: &PtxCacheKey) -> Option<String> {
        let path = self.cache_dir.join(key.to_filename());
        match std::fs::read_to_string(&path) {
            Ok(contents) if !contents.is_empty() => Some(contents),
            _ => None,
        }
    }

    /// Stores PTX text in the cache under the given key.
    ///
    /// # Errors
    ///
    /// Returns `std::io::Error` if the write fails.
    pub fn put(&self, key: &PtxCacheKey, ptx: &str) -> Result<(), std::io::Error> {
        let path = self.cache_dir.join(key.to_filename());
        std::fs::write(&path, ptx)
    }

    /// Removes all cached PTX files from the cache directory.
    ///
    /// Only removes `.ptx` files; other files and subdirectories are left intact.
    ///
    /// # Errors
    ///
    /// Returns `std::io::Error` if directory listing or file removal fails.
    pub fn clear(&self) -> Result<(), std::io::Error> {
        let entries = std::fs::read_dir(&self.cache_dir)?;
        for entry in entries {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("ptx") {
                std::fs::remove_file(&path)?;
            }
        }
        Ok(())
    }

    /// Returns the number of cached PTX files.
    ///
    /// # Errors
    ///
    /// Returns `std::io::Error` if the directory cannot be read.
    pub fn len(&self) -> Result<usize, std::io::Error> {
        let entries = std::fs::read_dir(&self.cache_dir)?;
        let count = entries
            .filter_map(Result::ok)
            .filter(|e| e.path().extension().and_then(|ext| ext.to_str()) == Some("ptx"))
            .count();
        Ok(count)
    }

    /// Returns `true` if the cache contains no PTX files.
    ///
    /// # Errors
    ///
    /// Returns `std::io::Error` if the directory cannot be read.
    pub fn is_empty(&self) -> Result<bool, std::io::Error> {
        self.len().map(|n| n == 0)
    }
}

/// Resolves the cache directory path, with fallback.
fn resolve_cache_dir() -> PathBuf {
    // Try ~/.cache/oxicuda/ptx/
    if let Some(home) = home_dir() {
        let cache = home.join(".cache").join("oxicuda").join("ptx");
        return cache;
    }

    // Fallback to temp dir
    std::env::temp_dir().join("oxicuda_ptx_cache")
}

/// Attempts to determine the user's home directory.
///
/// Checks `HOME` (Unix) and `USERPROFILE` (Windows) environment variables.
fn home_dir() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
}

/// Sanitizes a string for use as part of a filename.
///
/// Replaces any character that is not alphanumeric, underscore, or hyphen
/// with an underscore.
fn sanitize_filename(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Returns a unique temp directory for a test, using a counter to avoid collisions.
    fn test_cache_dir_named(name: &str) -> PathBuf {
        std::env::temp_dir()
            .join("oxicuda_ptx_cache_test")
            .join(format!("{}_{}", name, std::process::id()))
    }

    fn cleanup(dir: &PathBuf) {
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn cache_key_to_filename() {
        let key = PtxCacheKey {
            kernel_name: "vector_add".to_string(),
            params_hash: 0xDEAD_BEEF,
            sm_version: SmVersion::Sm80,
        };
        let filename = key.to_filename();
        assert!(filename.starts_with("vector_add_sm_80_"));
        assert!(
            std::path::Path::new(&filename)
                .extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("ptx"))
        );
    }

    #[test]
    fn cache_key_sanitization() {
        let key = PtxCacheKey {
            kernel_name: "my.kernel/v2".to_string(),
            params_hash: 42,
            sm_version: SmVersion::Sm90,
        };
        let filename = key.to_filename();
        assert!(
            !filename.contains('.')
                || std::path::Path::new(&filename)
                    .extension()
                    .is_some_and(|ext| ext.eq_ignore_ascii_case("ptx"))
        );
        // The kernel name part should not contain dots or slashes
        let prefix = filename.split("_sm_90_").next().unwrap_or("");
        assert!(!prefix.contains('/'));
    }

    #[test]
    fn cache_new_and_clear() {
        let dir = test_cache_dir_named("new_and_clear");
        cleanup(&dir);

        let cache = PtxCache::with_dir(dir.clone()).expect("cache creation should succeed");
        assert!(cache.is_empty().expect("should check empty"));

        let key = PtxCacheKey {
            kernel_name: "test".to_string(),
            params_hash: 1,
            sm_version: SmVersion::Sm80,
        };
        cache.put(&key, "// test ptx").expect("put should succeed");
        assert!(!cache.is_empty().expect("should check non-empty"));
        assert_eq!(cache.len().expect("len"), 1);

        cache.clear().expect("clear should succeed");
        assert!(cache.is_empty().expect("should be empty after clear"));

        cleanup(&dir);
    }

    #[test]
    fn get_or_generate_caches_result() {
        let dir = test_cache_dir_named("get_or_generate");
        cleanup(&dir);

        let cache = PtxCache::with_dir(dir.clone()).expect("cache creation should succeed");

        let key = PtxCacheKey {
            kernel_name: "cached_kernel".to_string(),
            params_hash: 42,
            sm_version: SmVersion::Sm80,
        };

        let mut call_count = 0u32;

        // First call should generate
        let ptx1 = cache
            .get_or_generate(&key, || {
                call_count += 1;
                Ok("// generated ptx v1".to_string())
            })
            .expect("should generate");
        assert_eq!(ptx1, "// generated ptx v1");
        assert_eq!(call_count, 1);

        // Second call should hit cache
        let ptx2 = cache
            .get_or_generate(&key, || {
                call_count += 1;
                Ok("// should not be called".to_string())
            })
            .expect("should cache hit");
        assert_eq!(ptx2, "// generated ptx v1");
        assert_eq!(call_count, 1);

        cleanup(&dir);
    }

    #[test]
    fn get_nonexistent_returns_none() {
        let dir = test_cache_dir_named("get_nonexistent");
        cleanup(&dir);

        let cache = PtxCache::with_dir(dir.clone()).expect("cache creation should succeed");
        let key = PtxCacheKey {
            kernel_name: "nonexistent".to_string(),
            params_hash: 0,
            sm_version: SmVersion::Sm80,
        };
        assert!(cache.get(&key).is_none());

        cleanup(&dir);
    }

    #[test]
    fn sanitize_filename_fn() {
        assert_eq!(sanitize_filename("hello_world"), "hello_world");
        assert_eq!(sanitize_filename("foo.bar/baz"), "foo_bar_baz");
        assert_eq!(sanitize_filename("a b c"), "a_b_c");
    }

    // -------------------------------------------------------------------------
    // P7: PTX disk cache round-trip tests
    // -------------------------------------------------------------------------

    /// Store a PTX string, retrieve it, and verify it is byte-for-byte identical.
    #[test]
    fn test_cache_round_trip() {
        let dir = test_cache_dir_named("round_trip");
        cleanup(&dir);

        let cache = PtxCache::with_dir(dir.clone()).expect("cache creation should succeed");
        let key = PtxCacheKey {
            kernel_name: "round_trip_kernel".to_string(),
            params_hash: 0xABCD_1234,
            sm_version: SmVersion::Sm80,
        };
        let original = "// round-trip PTX content\n.version 8.0\n.target sm_80\n";

        cache.put(&key, original).expect("put should succeed");
        let retrieved = cache.get(&key).expect("get should return cached value");
        assert_eq!(
            original, retrieved,
            "retrieved PTX must be identical to stored"
        );

        cleanup(&dir);
    }

    /// The same key always retrieves the same content.
    #[test]
    fn test_cache_same_key_same_content() {
        let dir = test_cache_dir_named("same_key");
        cleanup(&dir);

        let cache = PtxCache::with_dir(dir.clone()).expect("cache creation should succeed");
        let key = PtxCacheKey {
            kernel_name: "stable_kernel".to_string(),
            params_hash: 0x1111_2222,
            sm_version: SmVersion::Sm90,
        };
        let ptx = "// stable content";

        cache.put(&key, ptx).expect("first put should succeed");
        let first = cache.get(&key).expect("first get should succeed");
        let second = cache.get(&key).expect("second get should succeed");
        assert_eq!(
            first, second,
            "same key must return identical content on repeated lookups"
        );

        cleanup(&dir);
    }

    /// Different keys store and retrieve independent content.
    #[test]
    fn test_cache_different_keys() {
        let dir = test_cache_dir_named("diff_keys");
        cleanup(&dir);

        let cache = PtxCache::with_dir(dir.clone()).expect("cache creation should succeed");
        let key_a = PtxCacheKey {
            kernel_name: "kernel_a".to_string(),
            params_hash: 0x0000_0001,
            sm_version: SmVersion::Sm80,
        };
        let key_b = PtxCacheKey {
            kernel_name: "kernel_b".to_string(),
            params_hash: 0x0000_0002,
            sm_version: SmVersion::Sm80,
        };

        cache
            .put(&key_a, "// PTX for kernel A")
            .expect("put A should succeed");
        cache
            .put(&key_b, "// PTX for kernel B")
            .expect("put B should succeed");

        let content_a = cache.get(&key_a).expect("get A should succeed");
        let content_b = cache.get(&key_b).expect("get B should succeed");

        assert_eq!(content_a, "// PTX for kernel A");
        assert_eq!(content_b, "// PTX for kernel B");
        assert_ne!(
            content_a, content_b,
            "different keys must retrieve different content"
        );

        cleanup(&dir);
    }

    /// A cache hit must avoid calling the generation closure a second time.
    #[test]
    fn test_cache_hit_avoids_regeneration() {
        let dir = test_cache_dir_named("hit_avoids_regen");
        cleanup(&dir);

        let cache = PtxCache::with_dir(dir.clone()).expect("cache creation should succeed");
        let key = PtxCacheKey {
            kernel_name: "hit_kernel".to_string(),
            params_hash: 0xCAFE_BABE,
            sm_version: SmVersion::Sm80,
        };

        let mut call_count: u32 = 0;

        // First call — cache miss, generation runs
        let ptx_first = cache
            .get_or_generate(&key, || {
                call_count += 1;
                Ok("// generated".to_string())
            })
            .expect("first generation should succeed");
        assert_eq!(
            call_count, 1,
            "generation closure must be called on cache miss"
        );

        // Second call — cache hit, generation must NOT run
        let ptx_second = cache
            .get_or_generate(&key, || {
                call_count += 1;
                Ok("// should not be called".to_string())
            })
            .expect("second call should hit cache");
        assert_eq!(
            call_count, 1,
            "generation closure must not be called on cache hit"
        );
        assert_eq!(
            ptx_first, ptx_second,
            "cache hit must return original content"
        );

        cleanup(&dir);
    }

    /// A cache miss for an unknown key always triggers the generation closure.
    #[test]
    fn test_cache_miss_for_new_key() {
        let dir = test_cache_dir_named("miss_new_key");
        cleanup(&dir);

        let cache = PtxCache::with_dir(dir.clone()).expect("cache creation should succeed");

        let mut call_count: u32 = 0;

        // Each distinct key must produce a cache miss
        for i in 0u64..3 {
            let key = PtxCacheKey {
                kernel_name: format!("miss_kernel_{i}"),
                params_hash: i,
                sm_version: SmVersion::Sm80,
            };
            cache
                .get_or_generate(&key, || {
                    call_count += 1;
                    Ok(format!("// ptx for key {i}"))
                })
                .expect("generation should succeed");
        }

        assert_eq!(
            call_count, 3,
            "each new key must trigger one generation call"
        );

        cleanup(&dir);
    }
}
