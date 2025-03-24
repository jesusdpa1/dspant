// rust/src/processors/spatial/mod.rs
pub mod common_reference;
pub mod whitening;

// Re-export common reference functions
pub use common_reference::{
    compute_channel_median,
    compute_channel_median_parallel,
    compute_channel_mean,
    compute_channel_mean_parallel,
    apply_global_reference,
    apply_global_reference_parallel,
    apply_channel_reference,
    apply_group_reference,
};

// Re-export whitening functions (already done)
pub use whitening::{
    compute_whitening_matrix,
    apply_whitening_parallel,
    compute_covariance_parallel,
    compute_mean_parallel,
};