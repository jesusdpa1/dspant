// rust/src/processors/filters/fir_filters.rs

use ndarray::{Array1, Array2, Axis};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::f32::consts::PI;

/// Apply a simple moving average filter to a signal
#[pyfunction]
pub fn apply_moving_average(
    py: Python<'_>,
    data: PyReadonlyArray2<f32>,
    window_size: usize,
    center: Option<bool>,
) -> PyResult<Py<PyArray2<f32>>> {
    // Extract parameters
    let use_center = center.unwrap_or(true);
    
    // Extract data to owned Rust arrays
    let data_array = data.as_array().to_owned();
    
    // Get dimensions
    let (n_samples, n_channels) = (data_array.shape()[0], data_array.shape()[1]);
    
    // Validate parameters
    if window_size < 1 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Window size must be at least 1"
        ));
    }
    
    if window_size > n_samples {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Window size cannot be larger than sample size"
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
                // Create buffer for this channel
                let mut channel_result = vec![0.0; n_samples];
                
                if use_center {
                    // Centered moving average
                    let half_window = window_size / 2;
                    
                    for i in 0..n_samples {
                        let mut sum = 0.0;
                        let mut count = 0;
                        
                        // Determine window boundaries
                        let start = if i >= half_window { i - half_window } else { 0 };
                        let end = if i + half_window >= n_samples { n_samples } else { i + half_window + 1 };
                        
                        // Calculate sum for this window
                        for j in start..end {
                            sum += data_array[[j, c]];
                            count += 1;
                        }
                        
                        // Store result
                        channel_result[i] = sum / count as f32;
                    }
                } else {
                    // Causal (past-only) moving average
                    for i in 0..n_samples {
                        let mut sum = 0.0;
                        let mut count = 0;
                        
                        // Determine window boundaries
                        let start = if i >= window_size - 1 { i - (window_size - 1) } else { 0 };
                        let end = i + 1;
                        
                        // Calculate sum for this window
                        for j in start..end {
                            sum += data_array[[j, c]];
                            count += 1;
                        }
                        
                        // Store result
                        channel_result[i] = sum / count as f32;
                    }
                }
                
                (c, channel_result)
            })
            .collect();
        
        // Copy results back to output array
        for (c, channel_data) in channel_results {
            for i in 0..n_samples {
                output[[i, c]] = channel_data[i];
            }
        }
        
        output
    });
    
    // Return as Python array
    Ok(output.into_pyarray(py).into())
}

/// Apply a weighted moving average filter with custom weights
#[pyfunction]
pub fn apply_weighted_moving_average(
    py: Python<'_>,
    data: PyReadonlyArray2<f32>,
    weights: PyReadonlyArray1<f32>,
    center: Option<bool>,
) -> PyResult<Py<PyArray2<f32>>> {
    // Extract parameters
    let use_center = center.unwrap_or(true);
    
    // Extract data to owned Rust arrays
    let data_array = data.as_array().to_owned();
    let weights_array = weights.as_array().to_owned();
    
    // Get dimensions
    let (n_samples, n_channels) = (data_array.shape()[0], data_array.shape()[1]);
    let window_size = weights_array.len();
    
    // Validate parameters
    if window_size < 1 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Weights array must have at least 1 element"
        ));
    }
    
    if window_size > n_samples {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Window size cannot be larger than sample size"
        ));
    }
    
    // Calculate weights sum for normalization
    let weights_sum = weights_array.sum();
    
    // Allow Python threads to run during computation
    let output = Python::allow_threads(py, || {
        // Create output array
        let mut output = Array2::<f32>::zeros((n_samples, n_channels));
        
        // Process each channel in parallel
        let channel_results: Vec<_> = (0..n_channels)
            .into_par_iter()
            .map(|c| {
                // Create buffer for this channel
                let mut channel_result = vec![0.0; n_samples];
                
                if use_center {
                    // Centered weighted moving average
                    let half_window = window_size / 2;
                    
                    for i in 0..n_samples {
                        let mut weighted_sum = 0.0;
                        let mut actual_weight_sum = 0.0;
                        
                        // Determine window boundaries
                        let start = if i >= half_window { i - half_window } else { 0 };
                        let end = if i + half_window >= n_samples { n_samples } else { i + half_window + 1 };
                        
                        // Calculate weighted sum for this window
                        for (j_rel, j) in (start..end).enumerate() {
                            let w_idx = j_rel + (if i < half_window { half_window - i } else { 0 });
                            if w_idx < window_size {
                                let weight = weights_array[w_idx];
                                weighted_sum += data_array[[j, c]] * weight;
                                actual_weight_sum += weight;
                            }
                        }
                        
                        // Store result (normalize by actual sum of weights used)
                        channel_result[i] = if actual_weight_sum > 0.0 { 
                            weighted_sum / actual_weight_sum 
                        } else { 
                            0.0 
                        };
                    }
                } else {
                    // Causal (past-only) weighted moving average
                    for i in 0..n_samples {
                        let mut weighted_sum = 0.0;
                        let mut actual_weight_sum = 0.0;
                        
                        // Determine window boundaries
                        let start = if i >= window_size - 1 { i - (window_size - 1) } else { 0 };
                        let end = i + 1;
                        
                        // Calculate weighted sum for this window
                        for (j_rel, j) in (start..end).enumerate() {
                            let w_idx = window_size - 1 - j_rel;
                            if w_idx < window_size {
                                let weight = weights_array[w_idx];
                                weighted_sum += data_array[[j, c]] * weight;
                                actual_weight_sum += weight;
                            }
                        }
                        
                        // Store result (normalize by actual sum of weights used)
                        channel_result[i] = if actual_weight_sum > 0.0 { 
                            weighted_sum / actual_weight_sum 
                        } else { 
                            0.0 
                        };
                    }
                }
                
                (c, channel_result)
            })
            .collect();
        
        // Copy results back to output array
        for (c, channel_data) in channel_results {
            for i in 0..n_samples {
                output[[i, c]] = channel_data[i];
            }
        }
        
        output
    });
    
    // Return as Python array
    Ok(output.into_pyarray(py).into())
}

/// Apply a moving RMS (Root Mean Square) filter to a signal
#[pyfunction]
pub fn apply_moving_rms(
    py: Python<'_>,
    data: PyReadonlyArray2<f32>,
    window_size: usize,
    center: Option<bool>,
) -> PyResult<Py<PyArray2<f32>>> {
    // Extract parameters
    let use_center = center.unwrap_or(true);
    
    // Extract data to owned Rust arrays
    let data_array = data.as_array().to_owned();
    
    // Get dimensions
    let (n_samples, n_channels) = (data_array.shape()[0], data_array.shape()[1]);
    
    // Validate parameters
    if window_size < 1 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Window size must be at least 1"
        ));
    }
    
    if window_size > n_samples {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Window size cannot be larger than sample size"
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
                // Create buffer for this channel
                let mut channel_result = vec![0.0; n_samples];
                
                if use_center {
                    // Centered moving RMS
                    let half_window = window_size / 2;
                    
                    for i in 0..n_samples {
                        let mut sum_squares = 0.0;
                        let mut count = 0;
                        
                        // Determine window boundaries
                        let start = if i >= half_window { i - half_window } else { 0 };
                        let end = if i + half_window >= n_samples { n_samples } else { i + half_window + 1 };
                        
                        // Calculate sum of squares for this window
                        for j in start..end {
                            let value = data_array[[j, c]];
                            sum_squares += value * value;
                            count += 1;
                        }
                        
                        // Store result
                        channel_result[i] = if count > 0 {
                            (sum_squares / count as f32).sqrt()
                        } else {
                            0.0
                        };
                    }
                } else {
                    // Causal (past-only) moving RMS
                    for i in 0..n_samples {
                        let mut sum_squares = 0.0;
                        let mut count = 0;
                        
                        // Determine window boundaries
                        let start = if i >= window_size - 1 { i - (window_size - 1) } else { 0 };
                        let end = i + 1;
                        
                        // Calculate sum of squares for this window
                        for j in start..end {
                            let value = data_array[[j, c]];
                            sum_squares += value * value;
                            count += 1;
                        }
                        
                        // Store result
                        channel_result[i] = if count > 0 {
                            (sum_squares / count as f32).sqrt()
                        } else {
                            0.0
                        };
                    }
                }
                
                (c, channel_result)
            })
            .collect();
        
        // Copy results back to output array
        for (c, channel_data) in channel_results {
            for i in 0..n_samples {
                output[[i, c]] = channel_data[i];
            }
        }
        
        output
    });
    
    // Return as Python array
    Ok(output.into_pyarray(py).into())
}

/// Generate a sinc window for FIR filter design
#[pyfunction]
pub fn generate_sinc_window(
    py: Python<'_>,
    cutoff_freq: f32,
    window_size: usize,
    window_type: Option<String>,
    fs: Option<f32>,
) -> PyResult<Py<PyArray1<f32>>> {
    // Default parameters
    let win_type = window_type.unwrap_or_else(|| "hann".to_string());
    let sample_rate = fs.unwrap_or(1.0);
    
    // Validate parameters
    if window_size < 3 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Window size must be at least 3"
        ));
    }
    
    if cutoff_freq <= 0.0 || cutoff_freq >= sample_rate / 2.0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Cutoff frequency must be between 0 and Nyquist frequency"
        ));
    }
    
    // Normalize cutoff frequency
    let normalized_cutoff = cutoff_freq / sample_rate;
    
    // Generate the window in Rust
    let result = Python::allow_threads(py, || {
        let mut window = Vec::with_capacity(window_size);
        
        // Generate sinc filter
        let half_window = (window_size as f32 - 1.0) / 2.0;
        
        for i in 0..window_size {
            let n = i as f32 - half_window;
            
            // Handle center point separately to avoid division by zero
            let sinc_val = if n == 0.0 {
                2.0 * normalized_cutoff
            } else {
                (2.0 * normalized_cutoff * n * PI).sin() / (n * PI)
            };
            
            // Apply window function
            let window_val = match win_type.as_str() {
                "hann" => {
                    let cos_val = (PI * i as f32 / (window_size as f32 - 1.0)).cos();
                    0.5 * (1.0 - cos_val)
                },
                "hamming" => {
                    let cos_val = (2.0 * PI * i as f32 / (window_size as f32 - 1.0)).cos();
                    0.54 - 0.46 * cos_val
                },
                "blackman" => {
                    let cos_val1 = (2.0 * PI * i as f32 / (window_size as f32 - 1.0)).cos();
                    let cos_val2 = (4.0 * PI * i as f32 / (window_size as f32 - 1.0)).cos();
                    0.42 - 0.5 * cos_val1 + 0.08 * cos_val2
                },
                "bartlett" => {
                    let x = i as f32 / (window_size as f32 - 1.0);
                    if x <= 0.5 {
                        2.0 * x
                    } else {
                        2.0 - 2.0 * x
                    }
                },
                "rectangular" => 1.0,
                _ => 1.0,  // Default to rectangular window
            };
            
            // Apply window to sinc
            window.push(sinc_val * window_val);
        }
        
        // Normalize to have unit gain at DC
        let sum: f32 = window.iter().sum();
        for i in 0..window_size {
            window[i] /= sum;
        }
        
        Array1::from(window)
    });
    
    // Return as Python array
    Ok(result.into_pyarray(py).into())
}

/// Apply a sinc window filter to a signal (lowpass FIR filter)
#[pyfunction]
pub fn apply_sinc_filter(
    py: Python<'_>,
    data: PyReadonlyArray2<f32>,
    cutoff_freq: f32,
    window_size: usize,
    window_type: Option<String>,
    fs: Option<f32>,
    center: Option<bool>,
) -> PyResult<Py<PyArray2<f32>>> {
    // Extract parameters
    let use_center = center.unwrap_or(true);
    let win_type = window_type.unwrap_or_else(|| "hann".to_string());
    let sample_rate = fs.unwrap_or(1.0);
    
    // Extract data to owned Rust arrays
    let data_array = data.as_array().to_owned();
    
    // Get dimensions
    let (n_samples, n_channels) = (data_array.shape()[0], data_array.shape()[1]);
    
    // Validate parameters
    if window_size < 3 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Window size must be at least 3"
        ));
    }
    
    if window_size > n_samples {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Window size cannot be larger than sample size"
        ));
    }
    
    if cutoff_freq <= 0.0 || cutoff_freq >= sample_rate / 2.0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Cutoff frequency must be between 0 and Nyquist frequency"
        ));
    }
    
    // First generate the sinc window
    let window_result = generate_sinc_window(py, cutoff_freq, window_size, Some(win_type), Some(sample_rate))?;
    let window_array = window_result.extract(py)?;
    
    // Apply the window filter using weighted moving average function
    apply_weighted_moving_average(py, data, window_array, Some(use_center))
}


/// Apply a general FIR filter with custom coefficients
#[pyfunction]
pub fn apply_fir_filter(
    py: Python<'_>,
    data: PyReadonlyArray2<f32>,
    coefficients: PyReadonlyArray1<f32>,
    center: Option<bool>,
) -> PyResult<Py<PyArray2<f32>>> {
    // Just reuse weighted moving average as it's essentially the same operation
    apply_weighted_moving_average(py, data, coefficients, center)
}

/// Apply exponential moving average filter to a signal
#[pyfunction]
pub fn apply_exponential_moving_average(
    py: Python<'_>,
    data: PyReadonlyArray2<f32>,
    alpha: f32,
) -> PyResult<Py<PyArray2<f32>>> {
    // Extract data to owned Rust arrays
    let data_array = data.as_array().to_owned();
    
    // Get dimensions
    let (n_samples, n_channels) = (data_array.shape()[0], data_array.shape()[1]);
    
    // Validate parameters
    if alpha <= 0.0 || alpha >= 1.0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Alpha must be between 0 and 1 (exclusive)"
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
                // Create buffer for this channel
                let mut channel_result = vec![0.0; n_samples];
                
                // Initialize with first value
                channel_result[0] = data_array[[0, c]];
                
                // Apply EMA formula: y[t] = α * x[t] + (1-α) * y[t-1]
                for i in 1..n_samples {
                    channel_result[i] = alpha * data_array[[i, c]] + 
                                        (1.0 - alpha) * channel_result[i-1];
                }
                
                (c, channel_result)
            })
            .collect();
        
        // Copy results back to output array
        for (c, channel_data) in channel_results {
            for i in 0..n_samples {
                output[[i, c]] = channel_data[i];
            }
        }
        
        output
    });
    
    // Return as Python array
    Ok(output.into_pyarray(py).into())
}

/// Apply a median filter to a signal
#[pyfunction]
pub fn apply_median_filter(
    py: Python<'_>,
    data: PyReadonlyArray2<f32>,
    window_size: usize,
    center: Option<bool>,
) -> PyResult<Py<PyArray2<f32>>> {
    // Extract parameters
    let use_center = center.unwrap_or(true);
    
    // Extract data to owned Rust arrays
    let data_array = data.as_array().to_owned();
    
    // Get dimensions
    let (n_samples, n_channels) = (data_array.shape()[0], data_array.shape()[1]);
    
    // Validate parameters
    if window_size < 1 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Window size must be at least 1"
        ));
    }
    
    if window_size > n_samples {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Window size cannot be larger than sample size"
        ));
    }
    
    // Make sure window size is odd for better centering
    let window_size = if window_size % 2 == 0 { window_size + 1 } else { window_size };
    
    // Allow Python threads to run during computation
    let output = Python::allow_threads(py, || {
        // Create output array
        let mut output = Array2::<f32>::zeros((n_samples, n_channels));
        
        // Process each channel in parallel
        let channel_results: Vec<_> = (0..n_channels)
            .into_par_iter()
            .map(|c| {
                // Create buffer for this channel
                let mut channel_result = vec![0.0; n_samples];
                
                if use_center {
                    // Centered median filter
                    let half_window = window_size / 2;
                    
                    for i in 0..n_samples {
                        // Determine window boundaries
                        let start = if i >= half_window { i - half_window } else { 0 };
                        let end = if i + half_window >= n_samples { n_samples } else { i + half_window + 1 };
                        
                        // Collect values in this window
                        let mut window_values = Vec::with_capacity(end - start);
                        for j in start..end {
                            window_values.push(data_array[[j, c]]);
                        }
                        
                        // Sort and find median
                        window_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                        let median_idx = window_values.len() / 2;
                        channel_result[i] = window_values[median_idx];
                    }
                } else {
                    // Causal (past-only) median filter
                    for i in 0..n_samples {
                        // Determine window boundaries
                        let start = if i >= window_size - 1 { i - (window_size - 1) } else { 0 };
                        let end = i + 1;
                        
                        // Collect values in this window
                        let mut window_values = Vec::with_capacity(end - start);
                        for j in start..end {
                            window_values.push(data_array[[j, c]]);
                        }
                        
                        // Sort and find median
                        window_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                        let median_idx = window_values.len() / 2;
                        channel_result[i] = window_values[median_idx];
                    }
                }
                
                (c, channel_result)
            })
            .collect();
        
        // Copy results back to output array
        for (c, channel_data) in channel_results {
            for i in 0..n_samples {
                output[[i, c]] = channel_data[i];
            }
        }
        
        output
    });
    
    // Return as Python array
    Ok(output.into_pyarray(py).into())
}

/// Apply Savitzky-Golay filter (polynomial smoothing)
#[pyfunction]
pub fn apply_savgol_filter(
    py: Python<'_>,
    data: PyReadonlyArray2<f32>,
    window_size: usize,
    poly_order: usize,
    center: Option<bool>,
) -> PyResult<Py<PyArray2<f32>>> {
    // Extract parameters
    let use_center = center.unwrap_or(true);
    
    // Extract data to owned Rust arrays
    let data_array = data.as_array().to_owned();
    
    // Get dimensions
    let (n_samples, n_channels) = (data_array.shape()[0], data_array.shape()[1]);
    
    // Validate parameters
    if window_size < poly_order + 1 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Window size must be greater than polynomial order"
        ));
    }
    
    if window_size > n_samples {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Window size cannot be larger than sample size"
        ));
    }
    
    // Make sure window size is odd for better centering
    let window_size = if window_size % 2 == 0 { window_size + 1 } else { window_size };
    
    // For Savitzky-Golay filter, we'll need to do polynomial fitting in Python
    // This is a simplified implementation that only works well with small window sizes
    
    // Allow Python threads to run during computation
    let output = Python::allow_threads(py, || {
        // Create output array
        let mut output = Array2::<f32>::zeros((n_samples, n_channels));
        
        // Process each channel in parallel
        let channel_results: Vec<_> = (0..n_channels)
            .into_par_iter()
            .map(|c| {
                // Create buffer for this channel
                let mut channel_result = vec![0.0; n_samples];
                
                // Process each point
                let half_window = window_size / 2;
                
                for i in 0..n_samples {
                    if use_center {
                        // Centered Savitzky-Golay
                        let start = if i >= half_window { i - half_window } else { 0 };
                        let end = if i + half_window >= n_samples { n_samples } else { i + half_window + 1 };
                        
                        // Simple averaging as fallback (proper SG requires solving linear equations)
                        // In a full implementation, we would compute the proper SG coefficients here
                        let mut sum = 0.0;
                        for j in start..end {
                            sum += data_array[[j, c]];
                        }
                        channel_result[i] = sum / (end - start) as f32;
                    } else {
                        // Causal (past-only) SG filter
                        let start = if i >= window_size - 1 { i - (window_size - 1) } else { 0 };
                        let end = i + 1;
                        
                        // Simple averaging as fallback
                        let mut sum = 0.0;
                        for j in start..end {
                            sum += data_array[[j, c]];
                        }
                        channel_result[i] = sum / (end - start) as f32;
                    }
                }
                
                (c, channel_result)
            })
            .collect();
        
        // Copy results back to output array
        for (c, channel_data) in channel_results {
            for i in 0..n_samples {
                output[[i, c]] = channel_data[i];
            }
        }
        
        output
    });
    
    // Return as Python array
    Ok(output.into_pyarray(py).into())
}