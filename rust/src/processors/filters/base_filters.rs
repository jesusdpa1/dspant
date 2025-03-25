use ndarray::{Array1, Array2, Axis};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

/// Apply filter to multiple channels in parallel
#[pyfunction]
pub fn parallel_filter_channels(
    py: Python<'_>,
    data: PyReadonlyArray2<f32>,
    sos: PyReadonlyArray2<f32>,
    filtfilt: Option<bool>
) -> PyResult<Py<PyArray2<f32>>> {
    // Default to true for filtfilt
    let use_filtfilt = filtfilt.unwrap_or(true);
    
    // Extract data to owned Rust arrays
    let data_array = data.as_array().to_owned();
    let sos_array = sos.as_array().to_owned();
    
    // Get dimensions
    let (n_samples, n_channels) = (data_array.shape()[0], data_array.shape()[1]);
    
    // Need at least 3 points for filtering
    if n_samples < 3 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Signal must have at least 3 points for filtering"
        ));
    }
    
    // Allow Python threads to run during computation
    let output = Python::allow_threads(py, || {
        // Create output array
        let mut output = Array2::<f32>::zeros((n_samples, n_channels));
        
        // Process each channel in parallel
        let channel_results: Vec<_> = (0..n_channels)
            .into_par_iter()
            .map(|c| {
                // Extract channel data
                let mut channel_data = Vec::with_capacity(n_samples);
                for i in 0..n_samples {
                    channel_data.push(data_array[[i, c]]);
                }
                
                // Convert to Array1
                let channel_array = Array1::from(channel_data);
                
                // Apply the filter
                let filtered = if use_filtfilt {
                    sosfiltfilt(&channel_array, &sos_array)
                } else {
                    sosfilt(&channel_array, &sos_array)
                };
                
                (c, filtered)
            })
            .collect();
        
        // Copy results back to output array
        for (c, filtered) in channel_results {
            for i in 0..filtered.len() {
                if i < n_samples {
                    output[[i, c]] = filtered[i];
                }
            }
        }
        
        output
    });
    
    // Return as Python array
    Ok(output.into_pyarray(py).to_owned().into())
}

// SOS filter implementation (single pass)
pub fn sosfilt(data: &Array1<f32>, sos: &Array2<f32>) -> Array1<f32> {
    let n_sections = sos.shape()[0];
    let n_samples = data.len();
    let mut result = data.clone();
    
    // Apply each SOS section sequentially
    for section in 0..n_sections {
        // Get coefficients for this section
        let b0 = sos[[section, 0]];
        let b1 = sos[[section, 1]];
        let b2 = sos[[section, 2]];
        let a0 = sos[[section, 3]]; // Usually 1.0
        let a1 = sos[[section, 4]];
        let a2 = sos[[section, 5]];
        
        // Normalize by a0
        let b0_norm = b0 / a0;
        let b1_norm = b1 / a0;
        let b2_norm = b2 / a0;
        let a1_norm = a1 / a0;
        let a2_norm = a2 / a0;
        
        // Initialize state
        let mut z1 = 0.0;
        let mut z2 = 0.0;
        
        // Apply the filter
        for i in 0..n_samples {
            let x = result[i];
            let y = b0_norm * x + z1;
            z1 = b1_norm * x - a1_norm * y + z2;
            z2 = b2_norm * x - a2_norm * y;
            result[i] = y;
        }
    }
    
    result
}

// SOS filter implementation (forward-backward for zero phase)
pub fn sosfiltfilt(data: &Array1<f32>, sos: &Array2<f32>) -> Array1<f32> {
    // Forward pass
    let forward = sosfilt(data, sos);
    
    // Reverse the data
    let mut reversed = forward.clone();
    reversed.invert_axis(Axis(0));
    
    // Backward pass
    let backward = sosfilt(&reversed, sos);
    
    // Reverse again to get the final result
    let mut result = backward;
    result.invert_axis(Axis(0));
    
    result
}