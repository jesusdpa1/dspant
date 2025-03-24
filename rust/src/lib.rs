// rust/src/lib.rs

use pyo3::prelude::*;

mod processors;

// Re-export the TKEO functions directly from the processors module
use processors::transforms::tkeo::{
    compute_tkeo,
    compute_tkeo_parallel,
    compute_tkeo_classic,
    compute_tkeo_modified,
};

// Re-export the normalization functions
use processors::basics::normalization::{
    apply_zscore,
    apply_zscore_parallel,
    apply_minmax,
    apply_minmax_parallel,
    apply_robust,
    apply_robust_parallel,
    apply_mad,
    apply_mad_parallel,
};

// Re-export the whitening functions
use processors::spatial::whitening::{
    compute_whitening_matrix,
    apply_whitening,
    apply_whitening_parallel,
    compute_covariance,
    compute_covariance_parallel,
    compute_mean,
    compute_mean_parallel,
};


// Add to the imports at the top of lib.rs
use processors::spatial::common_reference::{
    compute_channel_median,
    compute_channel_median_parallel,
    compute_channel_mean,
    compute_channel_mean_parallel,
    apply_global_reference,
    apply_global_reference_parallel,
    apply_channel_reference,
    apply_group_reference,
};


// Original functions
#[pyfunction]
fn print_hello(name: &str) -> PyResult<String> {
    let message = format!("Hello, {}!", name);
    println!("{}", message);
    Ok(message)
}

#[pyfunction]
fn guess_the_number() -> PyResult<()> {
    // You'll implement the guessing game logic here
    Ok(())
}

/// Python module entry point
#[pymodule]
fn _rs(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(print_hello, py)?)?;
    m.add_function(wrap_pyfunction!(guess_the_number, py)?)?;
    
    // Add the TKEO functions directly to the _rs module
    m.add_function(wrap_pyfunction!(compute_tkeo, py)?)?;
    m.add_function(wrap_pyfunction!(compute_tkeo_parallel, py)?)?;
    m.add_function(wrap_pyfunction!(compute_tkeo_classic, py)?)?;
    m.add_function(wrap_pyfunction!(compute_tkeo_modified, py)?)?;
    
    // Add the normalization functions
    m.add_function(wrap_pyfunction!(apply_zscore, py)?)?;
    m.add_function(wrap_pyfunction!(apply_zscore_parallel, py)?)?;
    m.add_function(wrap_pyfunction!(apply_minmax, py)?)?;
    m.add_function(wrap_pyfunction!(apply_minmax_parallel, py)?)?;
    m.add_function(wrap_pyfunction!(apply_robust, py)?)?;
    m.add_function(wrap_pyfunction!(apply_robust_parallel, py)?)?;
    m.add_function(wrap_pyfunction!(apply_mad, py)?)?;
    m.add_function(wrap_pyfunction!(apply_mad_parallel, py)?)?;
    
    // Add the whitening functions
    m.add_function(wrap_pyfunction!(compute_whitening_matrix, py)?)?;
    m.add_function(wrap_pyfunction!(apply_whitening, py)?)?;
    m.add_function(wrap_pyfunction!(apply_whitening_parallel, py)?)?;
    m.add_function(wrap_pyfunction!(compute_covariance, py)?)?;
    m.add_function(wrap_pyfunction!(compute_covariance_parallel, py)?)?;
    m.add_function(wrap_pyfunction!(compute_mean, py)?)?;
    m.add_function(wrap_pyfunction!(compute_mean_parallel, py)?)?;


    // Add the common reference functions
    m.add_function(wrap_pyfunction!(compute_channel_median, py)?)?;
    m.add_function(wrap_pyfunction!(compute_channel_median_parallel, py)?)?;
    m.add_function(wrap_pyfunction!(compute_channel_mean, py)?)?;
    m.add_function(wrap_pyfunction!(compute_channel_mean_parallel, py)?)?;
    m.add_function(wrap_pyfunction!(apply_global_reference, py)?)?;
    m.add_function(wrap_pyfunction!(apply_global_reference_parallel, py)?)?;
    m.add_function(wrap_pyfunction!(apply_channel_reference, py)?)?;
    m.add_function(wrap_pyfunction!(apply_group_reference, py)?)?;


    Ok(())
}