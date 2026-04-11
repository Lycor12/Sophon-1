//! sophon-loss — Loss functions for Sophon-1 training.
//!
//! Provides free-energy-aligned loss for active inference integration.
//! The cross-entropy loss has been removed in favor of a pure free energy
//! objective, which naturally combines prediction error (likelihood) with
//! model complexity regularization (KL divergence to prior).
//!
//! # Free Energy Loss
//!
//! Variational free energy: F = E_q[log q(s) - log p(o, s)]
//! = KL(q(s) || p(s)) - E_q[log p(o | s)]
//! = (KL divergence to prior) + (negative log-likelihood of observations)
//!
//! For Gaussian variational approximation q(s) = N(mu, diag(sigma^2)):
//! KL(q || p) = 0.5 * sum_i [sigma_i^2 / sigma_prior^2 + (mu_i - mu_prior)^2 / sigma_prior^2
//! - 1 - log(sigma_i^2 / sigma_prior^2)]
//!
//! With standard normal prior p(s) = N(0, I):
//! KL = 0.5 * sum_i [sigma_i^2 + mu_i^2 - 1 - log(sigma_i^2)]
//!
//! The prediction_error term represents the negative log-likelihood of the
//! observations under the model, computed from the model's outputs.
//!
//! # Migration from Cross-Entropy
//!
//! Previously, training used a hybrid loss: L = alpha * CE + beta * FE.
//! Now we use pure free energy: the prediction_error in FE replaces CE.
//! This is theoretically motivated by the active inference framework and
//! provides a unified objective that balances accuracy with model simplicity.

#![forbid(unsafe_code)]

pub mod free_energy;

pub use free_energy::{
    accuracy, free_energy_components, free_energy_loss, kl_divergence_grad,
    kl_divergence_standard_normal, prediction_error_batch, prediction_error_grad,
    prediction_error_loss, FreeEnergyComponents,
};
