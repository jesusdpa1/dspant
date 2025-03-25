// rust/src/filters/mod.rs
pub mod base_filters;
pub mod iir_filters;

// Re-export functions
pub use base_filters::parallel_filter_channels;
pub use iir_filters::{
    apply_butter_filter,
    apply_cheby_filter,
    apply_elliptic_filter,
    apply_bessel_filter,
    apply_cascaded_filters,
};