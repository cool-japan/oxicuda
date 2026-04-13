//! # Quantization-Aware Training (QAT)
//!
//! This module provides the building blocks for QAT:
//!
//! | Module        | Contents                                         |
//! |---------------|--------------------------------------------------|
//! | `observer`    | MinMax, MovingAvg, Histogram calibration        |
//! | `fake_quant`  | FakeQuantize with Straight-Through Estimator     |

pub mod fake_quant;
pub mod observer;

pub use fake_quant::FakeQuantize;
pub use observer::{HistogramObserver, MinMaxObserver, MovingAvgObserver, Observer};
