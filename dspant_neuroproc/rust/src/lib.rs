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

// Import psth functions
use processors::spike_analytics::psth::{
    compute_psth,
    compute_psth_all,
    compute_psth_parallel,
    compute_raster_data, 
    bin_spikes_by_events
};

// Python module entry point
#[pymodule]
fn _rs(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // correlogram functions
    m.add_function(wrap_pyfunction!(compute_correlogram, py)?)?;
    m.add_function(wrap_pyfunction!(compute_autocorrelogram, py)?)?;
    m.add_function(wrap_pyfunction!(compute_all_cross_correlograms, py)?)?;
    m.add_function(wrap_pyfunction!(compute_jitter_corrected_correlogram, py)?)?;
    m.add_function(wrap_pyfunction!(compute_spike_time_tiling_coefficient, py)?)?;

    // psth functions
    m.add_function(wrap_pyfunction!(compute_psth, py)?)?;
    m.add_function(wrap_pyfunction!(compute_psth_all, py)?)?;
    m.add_function(wrap_pyfunction!(compute_psth_parallel, py)?)?;
    m.add_function(wrap_pyfunction!(compute_raster_data, py)?)?;
    m.add_function(wrap_pyfunction!(bin_spikes_by_events, py)?)?;
    Ok(())
}