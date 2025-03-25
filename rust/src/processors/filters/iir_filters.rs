use ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

use super::base_filters::{parallel_filter_channels, sosfilt, sosfiltfilt};

/// Apply Butterworth filter to multiple channels in parallel
#[pyfunction]
pub fn apply_butter_filter(
    py: Python<'_>,
    data: PyReadonlyArray2<f32>,
    sos: PyReadonlyArray2<f32>,
    filtfilt: Option<bool>
) -> PyResult<Py<PyArray2<f32>>> {
    parallel_filter_channels(py, data, sos, filtfilt)
}

/// Apply Chebyshev filter to multiple channels in parallel
#[pyfunction]
pub fn apply_cheby_filter(
    py: Python<'_>,
    data: PyReadonlyArray2<f32>,
    sos: PyReadonlyArray2<f32>,
    filtfilt: Option<bool>
) -> PyResult<Py<PyArray2<f32>>> {
    parallel_filter_channels(py, data, sos, filtfilt)
}

/// Apply Elliptic filter to multiple channels in parallel
#[pyfunction]
pub fn apply_elliptic_filter(
    py: Python<'_>,
    data: PyReadonlyArray2<f32>,
    sos: PyReadonlyArray2<f32>,
    filtfilt: Option<bool>
) -> PyResult<Py<PyArray2<f32>>> {
    parallel_filter_channels(py, data, sos, filtfilt)
}

/// Apply Bessel filter to multiple channels in parallel
#[pyfunction]
pub fn apply_bessel_filter(
    py: Python<'_>,
    data: PyReadonlyArray2<f32>,
    sos: PyReadonlyArray2<f32>,
    filtfilt: Option<bool>
) -> PyResult<Py<PyArray2<f32>>> {
    parallel_filter_channels(py, data, sos, filtfilt)
}

/// Apply cascaded filters (multiple filter stages) in parallel
#[pyfunction]
pub fn apply_cascaded_filters(
    py: Python<'_>,
    data: PyReadonlyArray2<f32>,
    sos_list: Vec<PyReadonlyArray2<f32>>,
    filtfilt: Option<bool>
) -> PyResult<Py<PyArray2<f32>>> {
    let use_filtfilt = filtfilt.unwrap_or(true);
    
    // Extract data to Rust arrays
    let mut current_data = data.as_array().to_owned();
    
    // Convert all SOS matrices to owned arrays
    let sos_arrays: Vec<Array2<f32>> = sos_list.iter()
        .map(|sos| sos.as_array().to_owned())
        .collect();
    
    // Get dimensions
    let (n_samples, n_channels) = (current_data.shape()[0], current_data.shape()[1]);
    
    // Allow Python threads to run during computation
    let result = Python::allow_threads(py, || {
        // For each filter in the cascade
        for sos_array in &sos_arrays {
            // Process each channel in parallel
            let channel_results: Vec<_> = (0..n_channels)
                .into_par_iter()
                .map(|c| {
                    // Extract channel data
                    let mut channel_data = Vec::with_capacity(n_samples);
                    for i in 0..n_samples {
                        channel_data.push(current_data[[i, c]]);
                    }
                    
                    // Apply filter to channel
                    let channel_array = Array1::from(channel_data);
                    let filtered = if use_filtfilt {
                        sosfiltfilt(&channel_array, sos_array)
                    } else {
                        sosfilt(&channel_array, sos_array)
                    };
                    
                    (c, filtered)
                })
                .collect();
            
            // Update current_data with filtered results
            for (c, filtered) in channel_results {
                for i in 0..n_samples {
                    current_data[[i, c]] = filtered[i];
                }
            }
        }
        
        current_data
    });
    
    // Convert back to Python array
    Ok(result.into_pyarray(py).to_owned().into())
}