//! Recurrent Neural Network cells for DNN.
//!
//! This module provides GPU-accelerated RNN cells:
//!
//! - [`lstm`] -- Long Short-Term Memory (LSTM) cell with 4-gate architecture
//! - [`gru`] -- Gated Recurrent Unit (GRU) cell with reset/update gates

pub mod gru;
pub mod lstm;

pub use gru::{GruWeights, gru_cell_forward, gru_sequence_forward};
pub use lstm::{LstmWeights, lstm_cell_forward, lstm_sequence_forward};

// ---------------------------------------------------------------------------
// Common types
// ---------------------------------------------------------------------------

/// Direction of RNN processing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RnnDirection {
    /// Process the sequence from the first timestep to the last.
    Forward,
    /// Process the sequence from the last timestep to the first.
    Backward,
    /// Process in both directions; output is concatenated.
    Bidirectional,
}

/// Common configuration for RNN layers.
#[derive(Debug, Clone)]
pub struct RnnConfig {
    /// Dimensionality of the input features at each timestep.
    pub input_size: usize,
    /// Dimensionality of the hidden state.
    pub hidden_size: usize,
    /// Number of stacked RNN layers.
    pub num_layers: usize,
    /// Processing direction.
    pub direction: RnnDirection,
    /// Dropout probability applied between layers (0.0 = no dropout).
    pub dropout: f32,
}

impl RnnConfig {
    /// Creates a new RNN configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if any size parameter is zero or dropout is out of
    /// the valid range `[0.0, 1.0)`.
    pub fn new(
        input_size: usize,
        hidden_size: usize,
        num_layers: usize,
        direction: RnnDirection,
        dropout: f32,
    ) -> Result<Self, crate::error::DnnError> {
        if input_size == 0 {
            return Err(crate::error::DnnError::InvalidArgument(
                "RNN input_size must be non-zero".into(),
            ));
        }
        if hidden_size == 0 {
            return Err(crate::error::DnnError::InvalidArgument(
                "RNN hidden_size must be non-zero".into(),
            ));
        }
        if num_layers == 0 {
            return Err(crate::error::DnnError::InvalidArgument(
                "RNN num_layers must be non-zero".into(),
            ));
        }
        if !(0.0..1.0).contains(&dropout) {
            return Err(crate::error::DnnError::InvalidArgument(format!(
                "RNN dropout must be in [0.0, 1.0), got {dropout}"
            )));
        }
        Ok(Self {
            input_size,
            hidden_size,
            num_layers,
            direction,
            dropout,
        })
    }

    /// Returns the multiplier for the output hidden size based on direction.
    ///
    /// Bidirectional RNNs produce outputs with `2 * hidden_size` features.
    #[must_use]
    pub fn direction_multiplier(&self) -> usize {
        match self.direction {
            RnnDirection::Forward | RnnDirection::Backward => 1,
            RnnDirection::Bidirectional => 2,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rnn_config_valid() {
        let cfg = RnnConfig::new(128, 256, 2, RnnDirection::Forward, 0.1);
        assert!(cfg.is_ok());
    }

    #[test]
    fn rnn_config_zero_input_size() {
        let cfg = RnnConfig::new(0, 256, 1, RnnDirection::Forward, 0.0);
        assert!(cfg.is_err());
    }

    #[test]
    fn rnn_config_zero_hidden_size() {
        let cfg = RnnConfig::new(128, 0, 1, RnnDirection::Forward, 0.0);
        assert!(cfg.is_err());
    }

    #[test]
    fn rnn_config_zero_layers() {
        let cfg = RnnConfig::new(128, 256, 0, RnnDirection::Forward, 0.0);
        assert!(cfg.is_err());
    }

    #[test]
    fn rnn_config_invalid_dropout() {
        let cfg = RnnConfig::new(128, 256, 1, RnnDirection::Forward, 1.0);
        assert!(cfg.is_err());
        let cfg2 = RnnConfig::new(128, 256, 1, RnnDirection::Forward, -0.1);
        assert!(cfg2.is_err());
    }

    #[test]
    fn direction_multiplier_values() {
        assert_eq!(
            RnnConfig::new(1, 1, 1, RnnDirection::Forward, 0.0)
                .map(|c| c.direction_multiplier())
                .ok(),
            Some(1)
        );
        assert_eq!(
            RnnConfig::new(1, 1, 1, RnnDirection::Backward, 0.0)
                .map(|c| c.direction_multiplier())
                .ok(),
            Some(1)
        );
        assert_eq!(
            RnnConfig::new(1, 1, 1, RnnDirection::Bidirectional, 0.0)
                .map(|c| c.direction_multiplier())
                .ok(),
            Some(2)
        );
    }

    #[test]
    fn rnn_direction_debug() {
        let _ = format!("{:?}", RnnDirection::Bidirectional);
    }
}
