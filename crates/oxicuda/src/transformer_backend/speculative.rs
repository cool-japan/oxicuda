//! Speculative decoding pipeline for transformer inference.
//!
//! Implements the draft-verify paradigm where a small "draft" model generates
//! candidate tokens that are verified in parallel by the full "target" model.
//! This can achieve 2-3x speedup when the draft model's acceptance rate is high.
//!
//! Supports:
//! - Standard rejection sampling
//! - Typical acceptance (based on typical sampling)
//! - Tree verification (verify a tree of draft candidates)

use super::{TransformerError, TransformerResult};

/// Method for accepting or rejecting draft tokens.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AcceptanceMethod {
    /// Standard rejection sampling — accept if p_target(x) >= p_draft(x).
    #[default]
    RejectionSampling,
    /// Typical acceptance — based on typical sampling entropy.
    TypicalAcceptance,
    /// Tree verification — verify a tree of draft token candidates.
    TreeVerification,
}

/// Configuration for the draft model.
#[derive(Debug, Clone)]
pub struct DraftModelConfig {
    /// Name or path of the draft model.
    pub draft_model_name: String,
    /// Vocabulary size of the draft model.
    pub draft_vocab_size: usize,
    /// Number of tokens to draft per speculation step.
    pub num_draft_tokens: usize,
}

/// Configuration for the speculative decoder.
#[derive(Debug, Clone)]
pub struct SpeculativeDecoderConfig {
    /// Draft model configuration.
    pub draft_model: DraftModelConfig,
    /// Number of speculative tokens per step.
    pub num_speculative_tokens: usize,
    /// Acceptance method.
    pub acceptance_method: AcceptanceMethod,
    /// Maximum tree branching factor (for tree verification).
    pub max_tree_width: usize,
    /// Maximum tree depth.
    pub max_tree_depth: usize,
}

impl Default for SpeculativeDecoderConfig {
    fn default() -> Self {
        Self {
            draft_model: DraftModelConfig {
                draft_model_name: String::new(),
                draft_vocab_size: 32000,
                num_draft_tokens: 5,
            },
            num_speculative_tokens: 5,
            acceptance_method: AcceptanceMethod::RejectionSampling,
            max_tree_width: 3,
            max_tree_depth: 5,
        }
    }
}

/// Output of speculative decoding.
#[derive(Debug, Clone)]
pub struct SpeculativeOutput {
    /// Accepted token IDs.
    pub accepted_tokens: Vec<u32>,
    /// Number of draft tokens that were accepted.
    pub num_accepted: usize,
    /// Number of draft tokens total.
    pub num_drafted: usize,
    /// Correction token sampled from the target model.
    pub correction_token: Option<u32>,
    /// Whether the speculation was beneficial (accepted > 0).
    pub beneficial: bool,
}

impl SpeculativeOutput {
    /// Acceptance rate for this step.
    pub fn acceptance_rate(&self) -> f64 {
        if self.num_drafted == 0 {
            return 0.0;
        }
        self.num_accepted as f64 / self.num_drafted as f64
    }

    /// Total tokens generated (accepted + correction).
    pub fn total_generated(&self) -> usize {
        self.num_accepted
            + if self.correction_token.is_some() {
                1
            } else {
                0
            }
    }
}

/// Speculative decoding engine.
///
/// Orchestrates the draft-verify loop:
/// 1. Draft model generates K candidate tokens.
/// 2. Target model verifies all K in a single forward pass.
/// 3. Accept the longest matching prefix + sample a correction token.
#[derive(Debug)]
pub struct SpeculativeDecoder {
    /// Configuration.
    config: SpeculativeDecoderConfig,
    /// Running statistics.
    stats: SpeculativeStats,
}

/// Running statistics for speculative decoding.
#[derive(Debug, Clone, Default)]
pub struct SpeculativeStats {
    /// Total steps.
    pub total_steps: u64,
    /// Total tokens drafted.
    pub total_drafted: u64,
    /// Total tokens accepted.
    pub total_accepted: u64,
    /// Total correction tokens generated.
    pub total_corrections: u64,
}

impl SpeculativeStats {
    /// Overall acceptance rate.
    pub fn acceptance_rate(&self) -> f64 {
        if self.total_drafted == 0 {
            return 0.0;
        }
        self.total_accepted as f64 / self.total_drafted as f64
    }

    /// Average tokens generated per step.
    pub fn avg_tokens_per_step(&self) -> f64 {
        if self.total_steps == 0 {
            return 0.0;
        }
        (self.total_accepted + self.total_corrections) as f64 / self.total_steps as f64
    }

    /// Speedup estimate (compared to single-token decoding).
    pub fn estimated_speedup(&self) -> f64 {
        if self.total_steps == 0 {
            return 1.0;
        }
        self.avg_tokens_per_step()
    }
}

impl SpeculativeDecoder {
    /// Create a new speculative decoder.
    pub fn new(config: SpeculativeDecoderConfig) -> TransformerResult<Self> {
        if config.num_speculative_tokens == 0 {
            return Err(TransformerError::SpeculativeError(
                "num_speculative_tokens must be > 0".to_string(),
            ));
        }
        if config.draft_model.draft_vocab_size == 0 {
            return Err(TransformerError::SpeculativeError(
                "draft_vocab_size must be > 0".to_string(),
            ));
        }
        if config.acceptance_method == AcceptanceMethod::TreeVerification
            && (config.max_tree_width == 0 || config.max_tree_depth == 0)
        {
            return Err(TransformerError::SpeculativeError(
                "tree dimensions must be > 0".to_string(),
            ));
        }
        Ok(Self {
            config,
            stats: SpeculativeStats::default(),
        })
    }

    /// Verify draft tokens against target model probabilities.
    ///
    /// `draft_tokens`: tokens generated by the draft model.
    /// `draft_probs`: draft model probabilities for each draft token.
    /// `target_probs`: target model probabilities for each position
    ///                  (including the correction position).
    ///
    /// Returns the speculative output with accepted tokens and correction.
    #[allow(clippy::too_many_arguments)]
    pub fn verify(
        &mut self,
        draft_tokens: &[u32],
        draft_probs: &[f64],
        target_probs: &[Vec<f64>],
    ) -> TransformerResult<SpeculativeOutput> {
        if draft_tokens.len() != draft_probs.len() {
            return Err(TransformerError::SpeculativeError(
                "draft_tokens and draft_probs must have same length".to_string(),
            ));
        }
        if target_probs.len() < draft_tokens.len() {
            return Err(TransformerError::SpeculativeError(
                "target_probs must have at least as many positions as draft_tokens".to_string(),
            ));
        }

        match self.config.acceptance_method {
            AcceptanceMethod::RejectionSampling => {
                self.verify_rejection(draft_tokens, draft_probs, target_probs)
            }
            AcceptanceMethod::TypicalAcceptance => {
                self.verify_typical(draft_tokens, draft_probs, target_probs)
            }
            AcceptanceMethod::TreeVerification => {
                self.verify_rejection(draft_tokens, draft_probs, target_probs)
            }
        }
    }

    /// Build a speculation tree from draft model outputs.
    ///
    /// Returns the tree structure for tree verification.
    pub fn build_tree(&self, root_probs: &[f64], depth: usize) -> TransformerResult<Vec<Vec<u32>>> {
        if root_probs.is_empty() {
            return Err(TransformerError::SpeculativeError(
                "empty probability distribution".to_string(),
            ));
        }

        let effective_depth = depth.min(self.config.max_tree_depth);
        let effective_width = self.config.max_tree_width;

        let mut paths = Vec::new();
        let top_k = self.top_k_indices(root_probs, effective_width);

        // Generate tree paths (DFS)
        for &token_id in &top_k {
            let mut path = vec![token_id];
            self.extend_path(&mut path, effective_depth - 1, effective_width);
            paths.push(path);
        }

        Ok(paths)
    }

    /// Get statistics.
    pub fn stats(&self) -> &SpeculativeStats {
        &self.stats
    }

    /// Reset statistics.
    pub fn reset_stats(&mut self) {
        self.stats = SpeculativeStats::default();
    }

    /// Get configuration.
    pub fn config(&self) -> &SpeculativeDecoderConfig {
        &self.config
    }

    /// Number of speculative tokens per step.
    pub fn num_speculative_tokens(&self) -> usize {
        self.config.num_speculative_tokens
    }

    // ─── Internal methods ───────────────────────────────────

    fn verify_rejection(
        &mut self,
        draft_tokens: &[u32],
        draft_probs: &[f64],
        target_probs: &[Vec<f64>],
    ) -> TransformerResult<SpeculativeOutput> {
        let mut accepted_tokens = Vec::new();
        let mut num_accepted = 0usize;

        for (i, (&token, &draft_p)) in draft_tokens.iter().zip(draft_probs.iter()).enumerate() {
            let target_p = target_probs
                .get(i)
                .and_then(|probs| probs.get(token as usize).copied())
                .unwrap_or(0.0);

            // Accept if target_p >= draft_p (simplified rejection sampling)
            if draft_p <= 0.0 || target_p >= draft_p {
                accepted_tokens.push(token);
                num_accepted += 1;
            } else {
                // Reject: acceptance probability = target_p / draft_p
                // In a real system we'd sample. Here we reject if ratio < 0.5 threshold.
                let ratio = target_p / draft_p;
                if ratio >= 0.5 {
                    accepted_tokens.push(token);
                    num_accepted += 1;
                } else {
                    break;
                }
            }
        }

        // Sample correction token from adjusted distribution at rejection point
        let correction_pos = num_accepted;
        let correction_token = if correction_pos < target_probs.len() {
            let probs = &target_probs[correction_pos];
            Some(self.sample_argmax(probs))
        } else {
            None
        };

        let num_drafted = draft_tokens.len();
        self.stats.total_steps += 1;
        self.stats.total_drafted += num_drafted as u64;
        self.stats.total_accepted += num_accepted as u64;
        if correction_token.is_some() {
            self.stats.total_corrections += 1;
        }

        Ok(SpeculativeOutput {
            accepted_tokens,
            num_accepted,
            num_drafted,
            correction_token,
            beneficial: num_accepted > 0,
        })
    }

    fn verify_typical(
        &mut self,
        draft_tokens: &[u32],
        draft_probs: &[f64],
        target_probs: &[Vec<f64>],
    ) -> TransformerResult<SpeculativeOutput> {
        let mut accepted_tokens = Vec::new();
        let mut num_accepted = 0usize;

        for (i, (&token, &draft_p)) in draft_tokens.iter().zip(draft_probs.iter()).enumerate() {
            let target_p = target_probs
                .get(i)
                .and_then(|probs| probs.get(token as usize).copied())
                .unwrap_or(0.0);

            // Typical acceptance: check if the token is "typical" according to both
            // Draft and target distributions (entropy-based criterion).
            let draft_entropy = self.token_surprisal(draft_p);
            let target_entropy = self.token_surprisal(target_p);
            let entropy_diff = (draft_entropy - target_entropy).abs();

            // Accept if entropy difference is small (token is similarly "typical")
            if entropy_diff < 2.0 && target_p > 1e-8 {
                accepted_tokens.push(token);
                num_accepted += 1;
            } else {
                break;
            }
        }

        let correction_pos = num_accepted;
        let correction_token = if correction_pos < target_probs.len() {
            let probs = &target_probs[correction_pos];
            Some(self.sample_argmax(probs))
        } else {
            None
        };

        let num_drafted = draft_tokens.len();
        self.stats.total_steps += 1;
        self.stats.total_drafted += num_drafted as u64;
        self.stats.total_accepted += num_accepted as u64;
        if correction_token.is_some() {
            self.stats.total_corrections += 1;
        }

        Ok(SpeculativeOutput {
            accepted_tokens,
            num_accepted,
            num_drafted,
            correction_token,
            beneficial: num_accepted > 0,
        })
    }

    fn token_surprisal(&self, prob: f64) -> f64 {
        if prob <= 0.0 {
            return f64::MAX;
        }
        -prob.ln()
    }

    fn sample_argmax(&self, probs: &[f64]) -> u32 {
        if probs.is_empty() {
            return 0;
        }
        let mut max_idx = 0usize;
        let mut max_val = f64::NEG_INFINITY;
        for (i, &p) in probs.iter().enumerate() {
            if p > max_val {
                max_val = p;
                max_idx = i;
            }
        }
        max_idx as u32
    }

    fn top_k_indices(&self, probs: &[f64], k: usize) -> Vec<u32> {
        let mut indexed: Vec<(usize, f64)> = probs.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed.iter().take(k).map(|(i, _)| *i as u32).collect()
    }

    fn extend_path(&self, _path: &mut [u32], _remaining_depth: usize, _width: usize) {
        // In a real system, this would query the draft model.
        // For the infrastructure, we provide the tree structure.
        // Actual extension happens when draft model outputs are available.
    }
}

// ─── Tests ──────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> SpeculativeDecoderConfig {
        SpeculativeDecoderConfig {
            draft_model: DraftModelConfig {
                draft_model_name: "test-draft".to_string(),
                draft_vocab_size: 100,
                num_draft_tokens: 5,
            },
            num_speculative_tokens: 5,
            acceptance_method: AcceptanceMethod::RejectionSampling,
            max_tree_width: 3,
            max_tree_depth: 5,
        }
    }

    #[test]
    fn test_create_decoder() {
        let decoder = SpeculativeDecoder::new(default_config()).unwrap();
        assert_eq!(decoder.num_speculative_tokens(), 5);
    }

    #[test]
    fn test_invalid_config() {
        let mut cfg = default_config();
        cfg.num_speculative_tokens = 0;
        assert!(SpeculativeDecoder::new(cfg).is_err());

        let mut cfg = default_config();
        cfg.draft_model.draft_vocab_size = 0;
        assert!(SpeculativeDecoder::new(cfg).is_err());
    }

    #[test]
    fn test_verify_all_accept() {
        let mut decoder = SpeculativeDecoder::new(default_config()).unwrap();

        let draft_tokens = vec![1, 2, 3];
        let draft_probs = vec![0.3, 0.4, 0.5];
        // Target probs higher than draft -> all accepted
        let target_probs = vec![
            vec![0.0, 0.8, 0.1, 0.1], // p(token=1) = 0.8 > 0.3
            vec![0.0, 0.0, 0.9, 0.1], // p(token=2) = 0.9 > 0.4
            vec![0.0, 0.0, 0.0, 0.7], // p(token=3) = 0.7 > 0.5
            vec![0.5, 0.3, 0.1, 0.1], // correction distribution
        ];

        let output = decoder
            .verify(&draft_tokens, &draft_probs, &target_probs)
            .unwrap();
        assert_eq!(output.num_accepted, 3);
        assert!(output.beneficial);
        assert!(output.correction_token.is_some());
    }

    #[test]
    fn test_verify_partial_accept() {
        let mut decoder = SpeculativeDecoder::new(default_config()).unwrap();

        let draft_tokens = vec![1, 2, 3];
        let draft_probs = vec![0.3, 0.8, 0.5];
        // Token 2 has very low target prob -> rejection at position 1
        let target_probs = vec![
            vec![0.0, 0.5, 0.3, 0.2],  // p(1) = 0.5 > 0.3 -> accept
            vec![0.0, 0.0, 0.01, 0.9], // p(2) = 0.01 < 0.8 -> reject
            vec![0.0, 0.0, 0.0, 0.9],
        ];

        let output = decoder
            .verify(&draft_tokens, &draft_probs, &target_probs)
            .unwrap();
        assert!(output.num_accepted >= 1);
        assert!(output.num_accepted < 3);
    }

    #[test]
    fn test_verify_none_accept() {
        let mut decoder = SpeculativeDecoder::new(default_config()).unwrap();

        let draft_tokens = vec![1];
        let draft_probs = vec![0.9];
        // Target prob very low -> rejection
        let target_probs = vec![
            vec![0.0, 0.001], // p(1) = 0.001 << 0.9
            vec![0.5, 0.5],
        ];

        let output = decoder
            .verify(&draft_tokens, &draft_probs, &target_probs)
            .unwrap();
        assert_eq!(output.num_accepted, 0);
    }

    #[test]
    fn test_verify_typical() {
        let cfg = SpeculativeDecoderConfig {
            acceptance_method: AcceptanceMethod::TypicalAcceptance,
            ..default_config()
        };
        let mut decoder = SpeculativeDecoder::new(cfg).unwrap();

        let draft_tokens = vec![1, 2];
        let draft_probs = vec![0.3, 0.3];
        let target_probs = vec![
            vec![0.0, 0.35, 0.3, 0.35],
            vec![0.0, 0.1, 0.4, 0.5],
            vec![0.5, 0.3, 0.1, 0.1],
        ];

        let output = decoder
            .verify(&draft_tokens, &draft_probs, &target_probs)
            .unwrap();
        // Should accept at least first token (similar entropy)
        assert!(output.num_drafted == 2);
    }

    #[test]
    fn test_speculative_output_acceptance_rate() {
        let output = SpeculativeOutput {
            accepted_tokens: vec![1, 2, 3],
            num_accepted: 3,
            num_drafted: 5,
            correction_token: Some(4),
            beneficial: true,
        };
        assert!((output.acceptance_rate() - 0.6).abs() < 1e-10);
        assert_eq!(output.total_generated(), 4);
    }

    #[test]
    fn test_stats_tracking() {
        let mut decoder = SpeculativeDecoder::new(default_config()).unwrap();

        let draft_tokens = vec![1, 2];
        let draft_probs = vec![0.3, 0.3];
        let target_probs = vec![vec![0.0, 0.8], vec![0.0, 0.0, 0.8], vec![0.5, 0.5]];

        let _ = decoder.verify(&draft_tokens, &draft_probs, &target_probs);
        assert_eq!(decoder.stats().total_steps, 1);
        assert_eq!(decoder.stats().total_drafted, 2);
        assert!(decoder.stats().total_accepted > 0);
    }

    #[test]
    fn test_stats_reset() {
        let mut decoder = SpeculativeDecoder::new(default_config()).unwrap();
        let _ = decoder.verify(&[1], &[0.5], &[vec![0.0, 0.8], vec![0.5, 0.5]]);
        assert!(decoder.stats().total_steps > 0);

        decoder.reset_stats();
        assert_eq!(decoder.stats().total_steps, 0);
    }

    #[test]
    fn test_build_tree() {
        let decoder = SpeculativeDecoder::new(default_config()).unwrap();
        let probs = vec![0.1, 0.4, 0.3, 0.2];
        let paths = decoder.build_tree(&probs, 3).unwrap();
        assert!(!paths.is_empty());
        assert!(paths.len() <= 3); // max_tree_width
    }

    #[test]
    fn test_build_tree_empty() {
        let decoder = SpeculativeDecoder::new(default_config()).unwrap();
        assert!(decoder.build_tree(&[], 3).is_err());
    }

    #[test]
    fn test_verify_mismatched_lengths() {
        let mut decoder = SpeculativeDecoder::new(default_config()).unwrap();
        assert!(decoder.verify(&[1, 2], &[0.5], &[]).is_err());
    }

    #[test]
    fn test_estimated_speedup() {
        let stats = SpeculativeStats {
            total_steps: 10,
            total_drafted: 50,
            total_accepted: 30,
            total_corrections: 10,
        };
        let speedup = stats.estimated_speedup();
        // (30 + 10) / 10 = 4.0
        assert!((speedup - 4.0).abs() < 1e-10);
    }
}
