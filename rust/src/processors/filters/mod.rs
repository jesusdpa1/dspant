// rust/src/processors/filters/mod.rs
pub mod base_filters;
pub mod iir_filters;
pub mod fir_filters;  // Add the new module

// Re-export functions
pub use base_filters::parallel_filter_channels;
pub use iir_filters::{
    apply_butter_filter,
    apply_cheby_filter,
    apply_elliptic_filter,
    apply_bessel_filter,
    apply_cascaded_filters,
};
// Add exports for the new FIR filter functions
pub use fir_filters::{
    apply_moving_average,
    apply_weighted_moving_average,
    apply_moving_rms,
    generate_sinc_window,
    apply_sinc_filter,
    apply_fir_filter,
    apply_exponential_moving_average,
    apply_median_filter,
    apply_savgol_filter,
};