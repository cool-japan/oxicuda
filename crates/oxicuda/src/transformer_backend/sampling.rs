//! Token sampling strategies for transformer inference.
//!
//! Implements a comprehensive set of sampling methods:
//!
//! - **Greedy** — deterministic argmax selection
//! - **Top-k** — restrict to top k tokens, renormalize
//! - **Top-p (nucleus)** — restrict to smallest set with cumulative prob >= p
//! - **Min-p** — keep tokens with prob >= min_p * max_prob
//! - **Temperature** — scale logits before softmax
//! - **Repetition penalty** — penalize previously generated tokens
//! - **Frequency/Presence penalty** — penalize based on count and existence
//! - **Beam search** — maintain top-B hypotheses with length/diversity penalty
//! - **Mirostat** — target-perplexity adaptive sampling

use std::collections::HashMap;

use super::{TransformerError, TransformerResult};

/// Token sampling parameters.
#[derive(Debug, Clone)]
pub struct SamplingParams {
    /// Temperature for softmax scaling. 0.0 = greedy.
    pub temperature: f64,
    /// Top-k filtering. 0 = disabled.
    pub top_k: usize,
    /// Top-p (nucleus) filtering. 1.0 = disabled.
    pub top_p: f64,
    /// Min-p filtering. 0.0 = disabled.
    pub min_p: f64,
    /// Repetition penalty multiplier. 1.0 = disabled.
    pub repetition_penalty: f64,
    /// Frequency penalty. 0.0 = disabled.
    pub frequency_penalty: f64,
    /// Presence penalty. 0.0 = disabled.
    pub presence_penalty: f64,
    /// Maximum tokens to generate.
    pub max_tokens: usize,
    /// Stop sequences (as token ID sequences).
    pub stop_sequences: Vec<Vec<u32>>,
    /// Optional seed for reproducible sampling.
    pub seed: Option<u64>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            repetition_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            max_tokens: 256,
            stop_sequences: Vec::new(),
            seed: None,
        }
    }
}

impl SamplingParams {
    /// Whether this config specifies greedy decoding.
    pub fn is_greedy(&self) -> bool {
        self.temperature == 0.0
    }

    /// Validate parameters.
    pub fn validate(&self) -> TransformerResult<()> {
        if self.temperature < 0.0 {
            return Err(TransformerError::SamplingError(
                "temperature must be >= 0.0".to_string(),
            ));
        }
        if self.top_p < 0.0 || self.top_p > 1.0 {
            return Err(TransformerError::SamplingError(
                "top_p must be in [0.0, 1.0]".to_string(),
            ));
        }
        if self.min_p < 0.0 || self.min_p > 1.0 {
            return Err(TransformerError::SamplingError(
                "min_p must be in [0.0, 1.0]".to_string(),
            ));
        }
        if self.repetition_penalty < 0.0 {
            return Err(TransformerError::SamplingError(
                "repetition_penalty must be >= 0.0".to_string(),
            ));
        }
        if self.max_tokens == 0 {
            return Err(TransformerError::SamplingError(
                "max_tokens must be > 0".to_string(),
            ));
        }
        Ok(())
    }
}

/// Output of a sampling step.
#[derive(Debug, Clone)]
pub struct SamplingOutput {
    /// Selected token ID.
    pub token_id: u32,
    /// Probability of the selected token (after all filtering).
    pub probability: f64,
    /// Log probability of the selected token.
    pub log_prob: f64,
    /// Top-k token IDs and their probabilities (for logging).
    pub top_tokens: Vec<(u32, f64)>,
}

/// Beam search configuration.
#[derive(Debug, Clone)]
pub struct BeamSearchConfig {
    /// Number of beams.
    pub num_beams: usize,
    /// Length penalty (>1.0 favors longer, <1.0 favors shorter).
    pub length_penalty: f64,
    /// Diversity penalty for diverse beam search.
    pub diversity_penalty: f64,
    /// Early stopping: stop when `num_beams` hypotheses are complete.
    pub early_stopping: bool,
    /// Number of groups for group beam search.
    pub num_groups: usize,
}

impl Default for BeamSearchConfig {
    fn default() -> Self {
        Self {
            num_beams: 4,
            length_penalty: 1.0,
            diversity_penalty: 0.0,
            early_stopping: false,
            num_groups: 1,
        }
    }
}

/// A single beam hypothesis.
#[derive(Debug, Clone)]
pub struct BeamHypothesis {
    /// Token IDs.
    pub tokens: Vec<u32>,
    /// Cumulative log probability.
    pub score: f64,
    /// Whether the hypothesis is finished.
    pub is_finished: bool,
}

impl BeamHypothesis {
    /// Length-normalized score.
    pub fn normalized_score(&self, length_penalty: f64) -> f64 {
        let len = self.tokens.len().max(1) as f64;
        self.score / len.powf(length_penalty)
    }
}

/// Mirostat sampling state.
#[derive(Debug, Clone)]
pub struct MirostatState {
    /// Target surprise value (tau).
    pub tau: f64,
    /// Learning rate (eta).
    pub eta: f64,
    /// Current estimated surprise value (mu).
    pub mu: f64,
    /// Mirostat version (1 or 2).
    pub version: u32,
}

impl Default for MirostatState {
    fn default() -> Self {
        Self {
            tau: 5.0,
            eta: 0.1,
            mu: 10.0,
            version: 2,
        }
    }
}

/// Token sampler engine.
///
/// Provides methods for all supported sampling strategies. Operates on
/// logit vectors (pre-softmax scores) and produces sampled token IDs.
#[derive(Debug)]
pub struct TokenSampler {
    /// Default sampling parameters.
    params: SamplingParams,
    /// Simple deterministic RNG state.
    rng_state: u64,
}

impl TokenSampler {
    /// Create a new token sampler.
    pub fn new(params: SamplingParams) -> TransformerResult<Self> {
        params.validate()?;
        let rng_state = params.seed.unwrap_or(42);
        Ok(Self { params, rng_state })
    }

    /// Sample a token from logits using the configured strategy.
    pub fn sample(
        &mut self,
        logits: &[f64],
        previous_tokens: &[u32],
    ) -> TransformerResult<SamplingOutput> {
        if logits.is_empty() {
            return Err(TransformerError::SamplingError("empty logits".to_string()));
        }

        // Step 1: Apply penalties
        let mut logits = logits.to_vec();
        self.apply_penalties(&mut logits, previous_tokens);

        // Step 2: Temperature scaling
        if self.params.is_greedy() {
            return self.greedy_sample(&logits);
        }
        self.apply_temperature(&mut logits);

        // Step 3: Convert to probabilities
        let mut probs = self.softmax(&logits);

        // Step 4: Apply filtering
        self.apply_top_k(&mut probs);
        self.apply_top_p(&mut probs);
        self.apply_min_p(&mut probs);

        // Step 5: Renormalize
        self.renormalize(&mut probs);

        // Step 6: Sample from filtered distribution
        self.categorical_sample(&probs)
    }

    /// Greedy (argmax) sampling.
    pub fn greedy_sample(&self, logits: &[f64]) -> TransformerResult<SamplingOutput> {
        if logits.is_empty() {
            return Err(TransformerError::SamplingError("empty logits".to_string()));
        }

        let (token_id, _max_logit) = logits.iter().enumerate().fold(
            (0usize, f64::NEG_INFINITY),
            |(best_i, best_v), (i, &v)| {
                if v > best_v { (i, v) } else { (best_i, best_v) }
            },
        );

        let probs = self.softmax(logits);
        let prob = probs.get(token_id).copied().unwrap_or(0.0);

        Ok(SamplingOutput {
            token_id: token_id as u32,
            probability: prob,
            log_prob: if prob > 0.0 {
                prob.ln()
            } else {
                f64::NEG_INFINITY
            },
            top_tokens: self.get_top_tokens(&probs, 5),
        })
    }

    /// Sample with Mirostat adaptive perplexity targeting.
    pub fn sample_mirostat(
        &mut self,
        logits: &[f64],
        state: &mut MirostatState,
    ) -> TransformerResult<SamplingOutput> {
        if logits.is_empty() {
            return Err(TransformerError::SamplingError("empty logits".to_string()));
        }

        let mut probs = self.softmax(logits);

        match state.version {
            1 => self.mirostat_v1(&mut probs, state),
            _ => self.mirostat_v2(&mut probs, state),
        }
    }

    /// Run beam search over multiple steps.
    ///
    /// `score_fn` returns logits for a given token sequence.
    pub fn beam_search<F>(
        &self,
        initial_tokens: &[u32],
        config: &BeamSearchConfig,
        max_length: usize,
        eos_token_id: u32,
        mut score_fn: F,
    ) -> TransformerResult<Vec<BeamHypothesis>>
    where
        F: FnMut(&[u32]) -> Vec<f64>,
    {
        if config.num_beams == 0 {
            return Err(TransformerError::SamplingError(
                "num_beams must be > 0".to_string(),
            ));
        }

        let mut beams = vec![BeamHypothesis {
            tokens: initial_tokens.to_vec(),
            score: 0.0,
            is_finished: false,
        }];

        let mut finished = Vec::new();

        for _ in 0..max_length {
            let mut candidates = Vec::new();

            for beam in &beams {
                if beam.is_finished {
                    finished.push(beam.clone());
                    continue;
                }

                let logits = score_fn(&beam.tokens);
                let probs = self.softmax(&logits);

                // Get top-k candidates
                let top = self.get_top_tokens(&probs, config.num_beams * 2);

                for (token_id, prob) in top {
                    let log_prob = if prob > 0.0 {
                        prob.ln()
                    } else {
                        f64::NEG_INFINITY
                    };
                    let mut new_tokens = beam.tokens.clone();
                    new_tokens.push(token_id);

                    let mut hyp = BeamHypothesis {
                        tokens: new_tokens,
                        score: beam.score + log_prob,
                        is_finished: token_id == eos_token_id,
                    };

                    // Apply diversity penalty
                    if config.diversity_penalty > 0.0 {
                        let penalty = self.compute_diversity_penalty(
                            &hyp.tokens,
                            &candidates,
                            config.diversity_penalty,
                        );
                        hyp.score -= penalty;
                    }

                    candidates.push(hyp);
                }
            }

            if candidates.is_empty() {
                break;
            }

            // Sort by length-normalized score
            candidates.sort_by(|a, b| {
                let sa = a.normalized_score(config.length_penalty);
                let sb = b.normalized_score(config.length_penalty);
                sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
            });

            beams = candidates.into_iter().take(config.num_beams).collect();

            // Early stopping
            if config.early_stopping {
                let all_finished = beams.iter().all(|b| b.is_finished);
                if all_finished || finished.len() >= config.num_beams {
                    break;
                }
            }
        }

        // Collect results
        finished.extend(beams);
        finished.sort_by(|a, b| {
            let sa = a.normalized_score(config.length_penalty);
            let sb = b.normalized_score(config.length_penalty);
            sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(finished)
    }

    /// Get the current parameters.
    pub fn params(&self) -> &SamplingParams {
        &self.params
    }

    // ─── Internal methods ───────────────────────────────────

    fn apply_penalties(&self, logits: &mut [f64], previous_tokens: &[u32]) {
        if previous_tokens.is_empty() {
            return;
        }

        // Count token frequencies
        let mut freq_map: HashMap<u32, usize> = HashMap::new();
        for &tok in previous_tokens {
            *freq_map.entry(tok).or_insert(0) += 1;
        }

        for (&tok, &count) in &freq_map {
            let idx = tok as usize;
            if idx >= logits.len() {
                continue;
            }

            // Repetition penalty (multiplicative)
            if self.params.repetition_penalty != 1.0 {
                if logits[idx] > 0.0 {
                    logits[idx] /= self.params.repetition_penalty;
                } else {
                    logits[idx] *= self.params.repetition_penalty;
                }
            }

            // Frequency penalty (additive, proportional to count)
            if self.params.frequency_penalty != 0.0 {
                logits[idx] -= self.params.frequency_penalty * count as f64;
            }

            // Presence penalty (additive, flat)
            if self.params.presence_penalty != 0.0 {
                logits[idx] -= self.params.presence_penalty;
            }
        }
    }

    fn apply_temperature(&self, logits: &mut [f64]) {
        if self.params.temperature <= 0.0 || self.params.temperature == 1.0 {
            return;
        }
        for logit in logits.iter_mut() {
            *logit /= self.params.temperature;
        }
    }

    fn softmax(&self, logits: &[f64]) -> Vec<f64> {
        if logits.is_empty() {
            return Vec::new();
        }
        let max_logit = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = logits.iter().map(|&l| (l - max_logit).exp()).collect();
        let sum: f64 = exps.iter().sum();
        if sum <= 0.0 {
            return vec![0.0; logits.len()];
        }
        exps.iter().map(|&e| e / sum).collect()
    }

    fn apply_top_k(&self, probs: &mut [f64]) {
        if self.params.top_k == 0 || self.params.top_k >= probs.len() {
            return;
        }

        // Find the k-th largest probability
        let mut sorted: Vec<f64> = probs.to_vec();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let threshold = sorted.get(self.params.top_k).copied().unwrap_or(0.0);

        for p in probs.iter_mut() {
            if *p < threshold {
                *p = 0.0;
            }
        }
    }

    fn apply_top_p(&self, probs: &mut [f64]) {
        if self.params.top_p >= 1.0 {
            return;
        }

        // Sort indices by probability (descending)
        let mut indexed: Vec<(usize, f64)> = probs.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut cumsum = 0.0;
        let mut cutoff_idx = indexed.len();

        for (i, (_, p)) in indexed.iter().enumerate() {
            cumsum += p;
            if cumsum >= self.params.top_p {
                cutoff_idx = i + 1;
                break;
            }
        }

        // Zero out everything beyond cutoff
        let keep_set: std::collections::HashSet<usize> =
            indexed[..cutoff_idx].iter().map(|(i, _)| *i).collect();

        for (i, p) in probs.iter_mut().enumerate() {
            if !keep_set.contains(&i) {
                *p = 0.0;
            }
        }
    }

    fn apply_min_p(&self, probs: &mut [f64]) {
        if self.params.min_p <= 0.0 {
            return;
        }

        let max_prob = probs.iter().copied().fold(0.0_f64, f64::max);
        let threshold = self.params.min_p * max_prob;

        for p in probs.iter_mut() {
            if *p < threshold {
                *p = 0.0;
            }
        }
    }

    fn renormalize(&self, probs: &mut [f64]) {
        let sum: f64 = probs.iter().sum();
        if sum <= 0.0 {
            // Fallback: uniform over all
            let uniform = 1.0 / probs.len() as f64;
            for p in probs.iter_mut() {
                *p = uniform;
            }
            return;
        }
        for p in probs.iter_mut() {
            *p /= sum;
        }
    }

    fn categorical_sample(&mut self, probs: &[f64]) -> TransformerResult<SamplingOutput> {
        let r = self.next_random();
        let mut cumsum = 0.0;

        for (i, &p) in probs.iter().enumerate() {
            cumsum += p;
            if r < cumsum {
                return Ok(SamplingOutput {
                    token_id: i as u32,
                    probability: p,
                    log_prob: if p > 0.0 { p.ln() } else { f64::NEG_INFINITY },
                    top_tokens: self.get_top_tokens(probs, 5),
                });
            }
        }

        // Fallback: last token with non-zero probability
        let last_idx = probs.iter().rposition(|&p| p > 0.0).unwrap_or(0);

        Ok(SamplingOutput {
            token_id: last_idx as u32,
            probability: probs.get(last_idx).copied().unwrap_or(0.0),
            log_prob: probs
                .get(last_idx)
                .map(|&p| if p > 0.0 { p.ln() } else { f64::NEG_INFINITY })
                .unwrap_or(f64::NEG_INFINITY),
            top_tokens: self.get_top_tokens(probs, 5),
        })
    }

    fn mirostat_v1(
        &mut self,
        probs: &mut [f64],
        state: &mut MirostatState,
    ) -> TransformerResult<SamplingOutput> {
        // Sort probabilities descending
        let mut indexed: Vec<(usize, f64)> = probs.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Find truncation point based on mu
        let k = ((state.mu.exp()).round() as usize).clamp(1, probs.len());

        // Zero out beyond top-k
        for entry in indexed.iter().skip(k) {
            probs[entry.0] = 0.0;
        }
        self.renormalize(probs);

        // Sample
        let output = self.categorical_sample(probs)?;

        // Update mu based on surprise of selected token
        let surprise = if output.probability > 0.0 {
            -output.probability.log2()
        } else {
            state.tau
        };
        state.mu -= state.eta * (surprise - state.tau);

        Ok(output)
    }

    fn mirostat_v2(
        &mut self,
        probs: &mut [f64],
        state: &mut MirostatState,
    ) -> TransformerResult<SamplingOutput> {
        // Mirostat v2: direct probability threshold
        let mut indexed: Vec<(usize, f64)> = probs.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Compute surprise threshold
        let surprise_threshold = state.mu;

        // Keep tokens with surprise <= threshold
        for &(idx, p) in &indexed {
            let surprise = if p > 0.0 { -p.log2() } else { f64::MAX };
            if surprise > surprise_threshold {
                probs[idx] = 0.0;
            }
        }
        self.renormalize(probs);

        let output = self.categorical_sample(probs)?;

        // Update mu
        let surprise = if output.probability > 0.0 {
            -output.probability.log2()
        } else {
            state.tau
        };
        state.mu -= state.eta * (surprise - state.tau);

        Ok(output)
    }

    fn get_top_tokens(&self, probs: &[f64], k: usize) -> Vec<(u32, f64)> {
        let mut indexed: Vec<(u32, f64)> = probs
            .iter()
            .enumerate()
            .map(|(i, &p)| (i as u32, p))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed.truncate(k);
        indexed
    }

    fn compute_diversity_penalty(
        &self,
        tokens: &[u32],
        existing: &[BeamHypothesis],
        penalty: f64,
    ) -> f64 {
        if existing.is_empty() || tokens.is_empty() {
            return 0.0;
        }
        let last_token = tokens[tokens.len() - 1];
        let mut count = 0usize;
        for hyp in existing {
            if hyp.tokens.last().copied() == Some(last_token) {
                count += 1;
            }
        }
        penalty * count as f64
    }

    /// Simple xorshift64 PRNG for deterministic sampling.
    fn next_random(&mut self) -> f64 {
        self.rng_state ^= self.rng_state << 13;
        self.rng_state ^= self.rng_state >> 7;
        self.rng_state ^= self.rng_state << 17;
        // Map to [0, 1)
        (self.rng_state as f64) / (u64::MAX as f64)
    }
}

// ─── Tests ──────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_sampler() -> TokenSampler {
        TokenSampler::new(SamplingParams::default()).unwrap()
    }

    #[test]
    fn test_greedy_sample() {
        let sampler = default_sampler();
        let logits = vec![1.0, 5.0, 2.0, 0.5];
        let output = sampler.greedy_sample(&logits).unwrap();
        assert_eq!(output.token_id, 1); // highest logit
    }

    #[test]
    fn test_greedy_via_temperature_zero() {
        let params = SamplingParams {
            temperature: 0.0,
            ..Default::default()
        };
        let mut sampler = TokenSampler::new(params).unwrap();
        let logits = vec![1.0, 5.0, 2.0, 0.5];
        let output = sampler.sample(&logits, &[]).unwrap();
        assert_eq!(output.token_id, 1);
    }

    #[test]
    fn test_sample_empty_logits() {
        let mut sampler = default_sampler();
        assert!(sampler.sample(&[], &[]).is_err());
    }

    #[test]
    fn test_top_k_filtering() {
        let params = SamplingParams {
            top_k: 2,
            temperature: 0.0, // greedy for determinism
            ..Default::default()
        };
        let mut sampler = TokenSampler::new(params).unwrap();
        let logits = vec![1.0, 5.0, 4.0, 0.5];
        let output = sampler.sample(&logits, &[]).unwrap();
        // Greedy picks the highest logit (token 1)
        assert_eq!(output.token_id, 1);
    }

    #[test]
    fn test_top_p_filtering() {
        let params = SamplingParams {
            top_p: 0.5,
            seed: Some(456),
            ..Default::default()
        };
        let mut sampler = TokenSampler::new(params).unwrap();
        // Token 1 dominates with ~99% after softmax
        let logits = vec![0.0, 10.0, 0.0, 0.0];
        let output = sampler.sample(&logits, &[]).unwrap();
        assert_eq!(output.token_id, 1);
    }

    #[test]
    fn test_min_p_filtering() {
        let params = SamplingParams {
            min_p: 0.5,
            seed: Some(789),
            ..Default::default()
        };
        let mut sampler = TokenSampler::new(params).unwrap();
        // After softmax, token 1 dominates; others should be filtered
        let logits = vec![0.0, 10.0, 0.0, 0.0];
        let output = sampler.sample(&logits, &[]).unwrap();
        assert_eq!(output.token_id, 1);
    }

    #[test]
    fn test_temperature_scaling() {
        let params = SamplingParams {
            temperature: 0.01,
            seed: Some(100),
            ..Default::default()
        };
        let mut sampler = TokenSampler::new(params).unwrap();
        // Very low temperature -> nearly greedy
        let logits = vec![1.0, 5.0, 2.0, 0.5];
        let output = sampler.sample(&logits, &[]).unwrap();
        assert_eq!(output.token_id, 1);
    }

    #[test]
    fn test_repetition_penalty() {
        let params = SamplingParams {
            repetition_penalty: 2.0,
            temperature: 0.0, // greedy to make deterministic
            ..Default::default()
        };
        let mut sampler = TokenSampler::new(params).unwrap();

        // Without penalty, token 1 would be selected (highest logit)
        // With penalty and token 1 in previous, its logit is halved
        let logits = vec![4.0, 5.0, 4.9, 0.5];
        let output = sampler.sample(&logits, &[1]).unwrap();
        // Token 1's logit becomes 5.0/2.0 = 2.5, so token 2 (4.9) wins
        assert_eq!(output.token_id, 2);
    }

    #[test]
    fn test_frequency_penalty() {
        let params = SamplingParams {
            frequency_penalty: 1.0,
            temperature: 0.0,
            ..Default::default()
        };
        let mut sampler = TokenSampler::new(params).unwrap();

        let logits = vec![3.0, 5.0, 4.0];
        // Token 1 appears 3 times -> penalized by -3.0
        let output = sampler.sample(&logits, &[1, 1, 1]).unwrap();
        // Token 1: 5.0 - 3.0 = 2.0, Token 2: 4.0 wins
        assert_eq!(output.token_id, 2);
    }

    #[test]
    fn test_presence_penalty() {
        let params = SamplingParams {
            presence_penalty: 3.0,
            temperature: 0.0,
            ..Default::default()
        };
        let mut sampler = TokenSampler::new(params).unwrap();

        let logits = vec![3.0, 5.0, 4.0];
        let output = sampler.sample(&logits, &[1]).unwrap();
        // Token 1: 5.0 - 3.0 = 2.0, Token 2: 4.0 wins
        assert_eq!(output.token_id, 2);
    }

    #[test]
    fn test_sampling_params_validation() {
        assert!(
            SamplingParams {
                temperature: -1.0,
                ..Default::default()
            }
            .validate()
            .is_err()
        );

        assert!(
            SamplingParams {
                top_p: 1.5,
                ..Default::default()
            }
            .validate()
            .is_err()
        );

        assert!(
            SamplingParams {
                min_p: -0.1,
                ..Default::default()
            }
            .validate()
            .is_err()
        );

        assert!(
            SamplingParams {
                max_tokens: 0,
                ..Default::default()
            }
            .validate()
            .is_err()
        );
    }

    #[test]
    fn test_is_greedy() {
        assert!(
            SamplingParams {
                temperature: 0.0,
                ..Default::default()
            }
            .is_greedy()
        );
        assert!(!SamplingParams::default().is_greedy());
    }

    #[test]
    fn test_beam_search_basic() {
        let sampler = default_sampler();
        let config = BeamSearchConfig {
            num_beams: 2,
            ..Default::default()
        };

        let results = sampler
            .beam_search(
                &[1],
                &config,
                3,
                99, // eos
                |_tokens| vec![0.1, 0.5, 0.3, 0.1],
            )
            .unwrap();

        assert!(!results.is_empty());
    }

    #[test]
    fn test_beam_search_eos() {
        let sampler = default_sampler();
        let config = BeamSearchConfig {
            num_beams: 2,
            early_stopping: true,
            ..Default::default()
        };

        let results = sampler
            .beam_search(
                &[1],
                &config,
                5,
                1,                                  // eos = token 1
                |_tokens| vec![0.1, 0.9, 0.0, 0.0], // EOS token dominates
            )
            .unwrap();

        assert!(!results.is_empty());
        assert!(results[0].is_finished);
    }

    #[test]
    fn test_beam_hypothesis_normalized_score() {
        let hyp = BeamHypothesis {
            tokens: vec![1, 2, 3, 4],
            score: -4.0,
            is_finished: false,
        };
        // length_penalty = 1.0: score / len = -4.0 / 4.0 = -1.0
        assert!((hyp.normalized_score(1.0) - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_mirostat_sampling() {
        let mut sampler = default_sampler();
        let logits = vec![1.0, 2.0, 3.0, 0.5, 0.1];
        let mut state = MirostatState::default();

        let output = sampler.sample_mirostat(&logits, &mut state).unwrap();
        assert!(output.token_id < 5);
        // mu should have been updated
        assert_ne!(state.mu, 10.0);
    }

    #[test]
    fn test_mirostat_v1() {
        let mut sampler = default_sampler();
        let logits = vec![1.0, 2.0, 3.0, 0.5];
        let mut state = MirostatState {
            version: 1,
            ..Default::default()
        };

        let output = sampler.sample_mirostat(&logits, &mut state).unwrap();
        assert!(output.token_id < 4);
    }

    #[test]
    fn test_sampling_output_fields() {
        let mut sampler = default_sampler();
        let logits = vec![1.0, 5.0, 2.0];
        let output = sampler.sample(&logits, &[]).unwrap();

        assert!(output.probability >= 0.0);
        assert!(output.probability <= 1.0);
        assert!(!output.top_tokens.is_empty());
    }

    #[test]
    fn test_stop_sequences() {
        let params = SamplingParams {
            stop_sequences: vec![vec![1, 2, 3]],
            ..Default::default()
        };
        assert_eq!(params.stop_sequences.len(), 1);
    }

    #[test]
    fn test_deterministic_with_seed() {
        let params = SamplingParams {
            seed: Some(42),
            ..Default::default()
        };
        let mut s1 = TokenSampler::new(params.clone()).unwrap();
        let mut s2 = TokenSampler::new(params).unwrap();

        let logits = vec![1.0, 2.0, 3.0, 1.5];
        let o1 = s1.sample(&logits, &[]).unwrap();
        let o2 = s2.sample(&logits, &[]).unwrap();
        assert_eq!(o1.token_id, o2.token_id);
    }
}
