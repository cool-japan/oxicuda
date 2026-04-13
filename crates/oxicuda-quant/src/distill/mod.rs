//! # Knowledge Distillation
//!
//! Compress models by transferring knowledge from a large teacher to a small student.
//!
//! | Module     | Contents                                             |
//! |------------|------------------------------------------------------|
//! | `loss`     | [`DistilLoss`] — KL, MSE, cosine, combined losses   |
//! | `response` | [`ResponseDistiller`] — soft + hard label training  |
//! | `feature`  | [`FeatureDistiller`] — intermediate activation matching |

pub mod feature;
pub mod loss;
pub mod response;

pub use feature::FeatureDistiller;
pub use loss::{DistilLoss, DistilLossType};
pub use response::ResponseDistiller;
