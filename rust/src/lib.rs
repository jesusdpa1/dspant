use pyo3::prelude::*;

mod preprocessing;

// Re-export the TKEO functions directly from the preprocessing module
use preprocessing::transforms::tkeo::{
    compute_tkeo,
    compute_tkeo_parallel,
    compute_tkeo_classic,
    compute_tkeo_modified,
};

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
    
    Ok(())
}