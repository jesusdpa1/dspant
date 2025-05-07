// dspant_neuroproc/rust/src/lib.rs

use pyo3::prelude::*;
mod processors;

// Import the correlogram functions
use processors::spike_analytics::correlogram::{
    compute_correlogram,
    compute_autocorrelogram,
    compute_all_cross_correlograms,
    compute_jitter_corrected_correlogram,
    compute_spike_time_tiling_coefficient,
};

// Original functions (kept from your code)
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
    
    // Add the correlogram functions
    m.add_function(wrap_pyfunction!(compute_correlogram, py)?)?;
    m.add_function(wrap_pyfunction!(compute_autocorrelogram, py)?)?;
    m.add_function(wrap_pyfunction!(compute_all_cross_correlograms, py)?)?;
    m.add_function(wrap_pyfunction!(compute_jitter_corrected_correlogram, py)?)?;
    m.add_function(wrap_pyfunction!(compute_spike_time_tiling_coefficient, py)?)?;

    Ok(())
}