// rust/src/processors/spatial/common_reference.rs

use ndarray::{Array1, Array2, ArrayView2, Axis};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

/// Compute median across channels for each time point
#[pyfunction]
pub fn compute_channel_median(
    py: Python<'_>,
    data: PyReadonlyArray2<f32>,
) -> PyResult<Py<PyArray2<f32>>> {
    // Convert input to rust ndarray
    let data_array = data.as_array().to_owned();
    
    // Get dimensions
    let n_samples = data_array.shape()[0];
    let n_channels = data_array.shape()[1];
    
    // Allow Python threads to run during computation
    let result = Python::allow_threads(py, || {
        let mut median_values = Array2::<f32>::zeros((n_samples, 1));
        
        // Compute median for each time point (row)
        for i in 0..n_samples {
            // Extract the row data
            let mut row_data = Vec::with_capacity(n_channels);
            for j in 0..n_channels {
                row_data.push(data_array[[i, j]]);
            }
            
            // Sort to find median
            row_data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            
            // Get median
            let mid = n_channels / 2;
            let median = if n_channels % 2 == 0 {
                (row_data[mid - 1] + row_data[mid]) / 2.0
            } else {
                row_data[mid]
            };
            
            median_values[[i, 0]] = median;
        }
        
        median_values
    });
    
    // Convert result back to Python
    Ok(result.into_pyarray(py).into())
}

/// Compute median across channels for each time point with parallel processing
#[pyfunction]
pub fn compute_channel_median_parallel(
    py: Python<'_>,
    data: PyReadonlyArray2<f32>,
) -> PyResult<Py<PyArray2<f32>>> {
    // Convert input to rust ndarray
    let data_array = data.as_array().to_owned();
    
    // Get dimensions
    let n_samples = data_array.shape()[0];
    let n_channels = data_array.shape()[1];
    
    // Allow Python threads to run during computation
    let result = Python::allow_threads(py, || {
        let mut median_values = Array2::<f32>::zeros((n_samples, 1));
        
        // Process rows in parallel using rayon
        let row_results: Vec<_> = (0..n_samples)
            .into_par_iter()
            .map(|i| {
                // Extract and sort the row data
                let mut row_data = Vec::with_capacity(n_channels);
                for j in 0..n_channels {
                    row_data.push(data_array[[i, j]]);
                }
                
                // Sort to find median
                row_data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                
                // Get median
                let mid = n_channels / 2;
                let median = if n_channels % 2 == 0 {
                    (row_data[mid - 1] + row_data[mid]) / 2.0
                } else {
                    row_data[mid]
                };
                
                (i, median)
            })
            .collect();
        
        // Combine results
        for (i, median) in row_results {
            median_values[[i, 0]] = median;
        }
        
        median_values
    });
    
    // Convert result back to Python
    Ok(result.into_pyarray(py).into())
}

/// Compute mean across channels for each time point
#[pyfunction]
pub fn compute_channel_mean(
    py: Python<'_>,
    data: PyReadonlyArray2<f32>,
) -> PyResult<Py<PyArray2<f32>>> {
    // Convert input to rust ndarray
    let data_array = data.as_array().to_owned();
    
    // Get dimensions
    let n_samples = data_array.shape()[0];
    let n_channels = data_array.shape()[1];
    
    // Allow Python threads to run during computation
    let result = Python::allow_threads(py, || {
        let mut mean_values = Array2::<f32>::zeros((n_samples, 1));
        
        // Compute mean for each time point (row)
        for i in 0..n_samples {
            let mut sum = 0.0;
            for j in 0..n_channels {
                sum += data_array[[i, j]];
            }
            mean_values[[i, 0]] = sum / (n_channels as f32);
        }
        
        mean_values
    });
    
    // Convert result back to Python
    Ok(result.into_pyarray(py).into())
}

/// Compute mean across channels for each time point with parallel processing
#[pyfunction]
pub fn compute_channel_mean_parallel(
    py: Python<'_>,
    data: PyReadonlyArray2<f32>,
) -> PyResult<Py<PyArray2<f32>>> {
    // Convert input to rust ndarray
    let data_array = data.as_array().to_owned();
    
    // Get dimensions
    let n_samples = data_array.shape()[0];
    let n_channels = data_array.shape()[1];
    
    // Allow Python threads to run during computation
    let result = Python::allow_threads(py, || {
        let mut mean_values = Array2::<f32>::zeros((n_samples, 1));
        
        // Process rows in parallel using rayon
        let row_results: Vec<_> = (0..n_samples)
            .into_par_iter()
            .map(|i| {
                let mut sum = 0.0;
                for j in 0..n_channels {
                    sum += data_array[[i, j]];
                }
                (i, sum / (n_channels as f32))
            })
            .collect();
        
        // Combine results
        for (i, mean) in row_results {
            mean_values[[i, 0]] = mean;
        }
        
        mean_values
    });
    
    // Convert result back to Python
    Ok(result.into_pyarray(py).into())
}

/// Apply global referencing by subtracting reference from all channels
#[pyfunction]
pub fn apply_global_reference(
    py: Python<'_>,
    data: PyReadonlyArray2<f32>,
    reference: PyReadonlyArray2<f32>,
) -> PyResult<Py<PyArray2<f32>>> {
    // Convert input to rust ndarray
    let data_array = data.as_array().to_owned();
    let reference_array = reference.as_array().to_owned();
    
    // Get dimensions
    let n_samples = data_array.shape()[0];
    let n_channels = data_array.shape()[1];
    
    // Allow Python threads to run during computation
    let result = Python::allow_threads(py, || {
        let mut result_array = Array2::<f32>::zeros((n_samples, n_channels));
        
        // Subtract reference from each channel
        for i in 0..n_samples {
            let ref_val = reference_array[[i, 0]];
            for j in 0..n_channels {
                result_array[[i, j]] = data_array[[i, j]] - ref_val;
            }
        }
        
        result_array
    });
    
    // Convert result back to Python
    Ok(result.into_pyarray(py).into())
}

/// Apply global referencing with parallel processing
/// Apply global referencing with parallel processing
#[pyfunction]
pub fn apply_global_reference_parallel(
    py: Python<'_>,
    data: PyReadonlyArray2<f32>,
    reference: PyReadonlyArray2<f32>,
) -> PyResult<Py<PyArray2<f32>>> {
    // Convert input to rust ndarray
    let data_array = data.as_array().to_owned();
    let reference_array = reference.as_array().to_owned();
    
    // Get dimensions
    let n_samples = data_array.shape()[0];
    let n_channels = data_array.shape()[1];
    
    // Allow Python threads to run during computation
    let result = Python::allow_threads(py, || {
        // Instead of modifying a shared array directly in parallel,
        // we'll process each row in parallel and collect the results
        let rows: Vec<_> = (0..n_samples)
            .into_par_iter()
            .map(|i| {
                let ref_val = reference_array[[i, 0]];
                let mut row = Vec::with_capacity(n_channels);
                
                for j in 0..n_channels {
                    row.push(data_array[[i, j]] - ref_val);
                }
                
                (i, row)
            })
            .collect();
        
        // Now create the result array and fill it from the collected results
        let mut result_array = Array2::<f32>::zeros((n_samples, n_channels));
        for (i, row) in rows {
            for j in 0..n_channels {
                result_array[[i, j]] = row[j];
            }
        }
        
        result_array
    });
    
    // Convert result back to Python
    Ok(result.into_pyarray(py).into())
}

/// Apply re-referencing using specific channels as reference
#[pyfunction]
pub fn apply_channel_reference(
    py: Python<'_>,
    data: PyReadonlyArray2<f32>,
    reference_channels: Vec<i32>,
    method: String,
) -> PyResult<Py<PyArray2<f32>>> {
    // Convert input to rust ndarray
    let data_array = data.as_array().to_owned();
    
    // Get dimensions
    let n_samples = data_array.shape()[0];
    let n_channels = data_array.shape()[1];
    
    // Validate reference channels
    let valid_channels: Vec<usize> = reference_channels.iter()
        .filter_map(|&idx| {
            let usize_idx = idx as usize;
            if idx >= 0 && usize_idx < n_channels {
                Some(usize_idx)
            } else {
                None
            }
        })
        .collect();
    
    if valid_channels.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "No valid reference channels provided"
        ));
    }
    
    // Allow Python threads to run during computation
    let result = Python::allow_threads(py, || {
        let mut result_array = Array2::<f32>::zeros((n_samples, n_channels));
        let mut reference = Array2::<f32>::zeros((n_samples, 1));
        
        // Compute reference for each time point
        for i in 0..n_samples {
            if method == "median" {
                // Collect values from reference channels
                let mut ref_values = Vec::with_capacity(valid_channels.len());
                for &ch_idx in &valid_channels {
                    ref_values.push(data_array[[i, ch_idx]]);
                }
                
                // Sort to find median
                ref_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                
                // Get median
                let mid = ref_values.len() / 2;
                let ref_val = if ref_values.len() % 2 == 0 {
                    (ref_values[mid - 1] + ref_values[mid]) / 2.0
                } else {
                    ref_values[mid]
                };
                
                reference[[i, 0]] = ref_val;
            } else if method == "mean" {
                // Compute mean of reference channels
                let mut sum = 0.0;
                for &ch_idx in &valid_channels {
                    sum += data_array[[i, ch_idx]];
                }
                reference[[i, 0]] = sum / (valid_channels.len() as f32);
            } else {
                // Default to first reference channel
                reference[[i, 0]] = data_array[[i, valid_channels[0]]];
            }
        }
        
        // Apply the reference
        for i in 0..n_samples {
            let ref_val = reference[[i, 0]];
            for j in 0..n_channels {
                result_array[[i, j]] = data_array[[i, j]] - ref_val;
            }
        }
        
        result_array
    });
    
    // Convert result back to Python
    Ok(result.into_pyarray(py).into())
}

/// Apply re-referencing by groups
#[pyfunction]
pub fn apply_group_reference(
    py: Python<'_>,
    data: PyReadonlyArray2<f32>,
    groups: Vec<Vec<i32>>,
    method: String,
) -> PyResult<Py<PyArray2<f32>>> {
    // Convert input to rust ndarray
    let data_array = data.as_array().to_owned();
    
    // Get dimensions
    let n_samples = data_array.shape()[0];
    let n_channels = data_array.shape()[1];
    
    // Allow Python threads to run during computation
    let result = Python::allow_threads(py, || {
        // Create result array for return
        let mut result_array = data_array.clone();
        
        // Process each group
        for group in groups.iter() {
            // Get valid channel indices
            let valid_channels: Vec<usize> = group.iter()
                .filter_map(|&idx| {
                    let usize_idx = idx as usize;
                    if idx >= 0 && usize_idx < n_channels {
                        Some(usize_idx)
                    } else {
                        None
                    }
                })
                .collect();
            
            if valid_channels.is_empty() {
                continue;
            }
            
            // Compute reference for this group
            let mut group_reference = Array2::<f32>::zeros((n_samples, 1));
            
            for i in 0..n_samples {
                if method == "median" {
                    // Collect values from group channels
                    let mut values = Vec::with_capacity(valid_channels.len());
                    for &ch_idx in &valid_channels {
                        values.push(data_array[[i, ch_idx]]);
                    }
                    
                    // Sort to find median
                    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    
                    // Get median
                    let mid = values.len() / 2;
                    let ref_val = if values.len() % 2 == 0 {
                        (values[mid - 1] + values[mid]) / 2.0
                    } else {
                        values[mid]
                    };
                    
                    group_reference[[i, 0]] = ref_val;
                } else if method == "mean" {
                    // Compute mean of group channels
                    let mut sum = 0.0;
                    for &ch_idx in &valid_channels {
                        sum += data_array[[i, ch_idx]];
                    }
                    group_reference[[i, 0]] = sum / (valid_channels.len() as f32);
                }
            }
            
            // Apply reference to this group - don't use parallel here
            for i in 0..n_samples {
                let ref_val = group_reference[[i, 0]];
                for &ch_idx in &valid_channels {
                    result_array[[i, ch_idx]] = data_array[[i, ch_idx]] - ref_val;
                }
            }
        }
        
        result_array
    });
    
    // Convert result back to Python
    Ok(result.into_pyarray(py).into())
}