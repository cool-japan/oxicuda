//! Request type and routing decision.

/// An inference request awaiting dispatch to a rank.
#[derive(Debug, Clone)]
pub struct Request {
    /// Unique request identifier.
    pub request_id: u64,
    /// Prompt token ids.
    pub token_ids: Vec<u32>,
    /// Maximum new tokens to generate.
    pub max_new_tokens: usize,
    /// Optional client-supplied priority (higher = more urgent).
    pub priority: i32,
}

impl Request {
    /// FNV-1a hash of the first `prefix_len` tokens.
    ///
    /// Used for prefix-affinity routing: requests sharing the same prefix hash
    /// are sent to the rank that likely has those KV blocks cached.
    pub fn prefix_hash(&self, prefix_len: usize) -> u64 {
        let len = prefix_len.min(self.token_ids.len());
        let mut h: u64 = 14_695_981_039_346_656_037;
        for &tok in &self.token_ids[..len] {
            h ^= tok as u64;
            h = h.wrapping_mul(1_099_511_628_211);
        }
        h
    }
}

/// Result of dispatching a request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RoutingDecision {
    /// Which rank should handle the request.
    pub rank: usize,
    /// The policy that selected this rank.
    pub policy_used: DispatchPolicy,
    /// Whether a prefix cache hit was detected for this rank.
    pub prefix_hit: bool,
}

/// Which policy produced the routing decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DispatchPolicy {
    /// Cyclic round-robin.
    RoundRobin,
    /// Rank with fewest allocated KV blocks.
    LeastLoaded,
    /// Rank with a matching prefix cache entry.
    PrefixAffinity,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefix_hash_deterministic() {
        let req = Request {
            request_id: 1,
            token_ids: vec![10, 20, 30, 40, 50],
            max_new_tokens: 64,
            priority: 0,
        };
        let h1 = req.prefix_hash(3);
        let h2 = req.prefix_hash(3);
        assert_eq!(h1, h2, "prefix hash must be deterministic");
    }

    #[test]
    fn prefix_hash_differs_for_different_prefixes() {
        let req = Request {
            request_id: 2,
            token_ids: vec![1, 2, 3, 4, 5],
            max_new_tokens: 32,
            priority: 0,
        };
        let ha = req.prefix_hash(2);
        let hb = req.prefix_hash(3);
        assert_ne!(
            ha, hb,
            "different prefix lengths should give different hashes"
        );
    }

    #[test]
    fn prefix_hash_empty_tokens_is_fnv_offset() {
        let req = Request {
            request_id: 3,
            token_ids: vec![],
            max_new_tokens: 1,
            priority: 0,
        };
        // Empty prefix → FNV offset basis
        let h = req.prefix_hash(0);
        assert_eq!(h, 14_695_981_039_346_656_037_u64);
    }

    #[test]
    fn prefix_hash_clamped_to_token_len() {
        let req = Request {
            request_id: 4,
            token_ids: vec![1, 2],
            max_new_tokens: 1,
            priority: 0,
        };
        // prefix_len=10 but only 2 tokens → same as prefix_len=2
        assert_eq!(req.prefix_hash(10), req.prefix_hash(2));
    }
}
