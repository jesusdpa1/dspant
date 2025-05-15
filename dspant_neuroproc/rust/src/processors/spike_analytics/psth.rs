// rust/src/processors/spike_analytics/psth.rs

use ndarray::{Array1, Array2, Axis};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rayon::prelude::*;



/// Bin spikes around event times with parallel processing
/// 
/// Args:
///     spike_times: Spike times in samples
///     event_times: Event times in samples
///     pre_samples: Number of samples before events to include
///     post_samples: Number of samples after events to include
///     bin_edges: Bin edges relative to event times in samples
///
/// Returns:
///     Binned spike counts (shape: n_bins × n_events)
#[pyfunction]
pub fn bin_spikes_by_events(
    py: Python<'_>,
    spike_times: PyReadonlyArray1<i32>,
    event_times: PyReadonlyArray1<i32>,
    pre_samples: i32,
    post_samples: i32,
    bin_edges: PyReadonlyArray1<f32>,
) -> PyResult<Py<PyArray2<i32>>> {
    // Convert inputs to Rust arrays
    let spikes = spike_times.as_array().to_owned();
    let events_array = event_times.as_array().to_owned();
    let edges = bin_edges.as_array().to_owned();
    
    // Convert to Vec for parallelization
    let events: Vec<i32> = events_array.iter().cloned().collect();
    let spike_vec: Vec<i32> = spikes.iter().cloned().collect();
    let bin_edges_vec: Vec<f32> = edges.iter().cloned().collect();
    
    // Calculate dimensions
    let n_events = events.len();
    let n_bins = bin_edges_vec.len() - 1;
    
    // Allow Python threads to run during computation
    let binned_spikes = Python::allow_threads(py, || {
        // Create output array: bins × events
        let mut binned_spikes = Array2::<i32>::zeros((n_bins, n_events));
        
        // Process each event in parallel
        let event_counts: Vec<_> = events
            .par_iter()
            .enumerate()
            .map(|(e_idx, &event_time)| {
                // Create histogram for this event
                let mut event_counts = vec![0i32; n_bins];
                
                // Calculate window boundaries
                let window_start = event_time - pre_samples;
                let window_end = event_time + post_samples;
                
                // Find spikes within this window
                for &spike_time in spike_vec.iter() {
                    // Skip if spike is outside window
                    if spike_time < window_start || spike_time >= window_end {
                        continue;
                    }
                    
                    // Calculate relative time
                    let rel_time = (spike_time - event_time) as f32;
                    
                    // Find bin index using binary search
                    let bin_idx = match bin_edges_vec.binary_search_by(|&edge| {
                        edge.partial_cmp(&rel_time).unwrap_or(std::cmp::Ordering::Equal)
                    }) {
                        Ok(idx) => idx - 1,  // Exact match - use previous bin
                        Err(idx) => idx - 1, // Not found - insertion point minus 1
                    };
                    
                    // Ensure bin index is valid
                    if bin_idx < n_bins {
                        event_counts[bin_idx] += 1;
                    }
                }
                
                (e_idx, event_counts)
            })
            .collect();
        
        // Combine results
        for (e_idx, counts) in event_counts {
            for bin_idx in 0..n_bins {
                binned_spikes[[bin_idx, e_idx]] = counts[bin_idx];
            }
        }
        
        binned_spikes
    });
    
    // Convert back to Python array
    Ok(binned_spikes.into_pyarray(py).into())
}


/// Compute PSTH (Peristimulus Time Histogram) for a single unit
/// 
/// Args:
///     spike_times: Spike times in seconds
///     event_times: Event/stimulus times in seconds
///     pre_time: Time before events to include (seconds)
///     post_time: Time after events to include (seconds)
///     bin_size: Size of bins in seconds
///
/// Returns:
///     Tuple of (binned_spikes, time_bins)
/// 
#[pyfunction]
pub fn compute_psth(
    py: Python<'_>,
    spike_times: PyReadonlyArray1<f32>,
    event_times: PyReadonlyArray1<f32>,
    pre_time: f32,
    post_time: f32,
    bin_size: f32,
) -> PyResult<(Py<PyArray2<f32>>, Py<PyArray1<f32>>)> {
    // Convert inputs to Rust arrays
    let spikes = spike_times.as_array().to_owned();
    let events = event_times.as_array().to_owned();
    
    // Calculate number of bins
    let total_time = pre_time + post_time;
    let n_bins = (total_time / bin_size) as usize;
    if n_bins == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Window size must be larger than bin size"
        ));
    }
    
    // Create time bins centered on event onset
    let time_bins = Array1::linspace(-pre_time, post_time, n_bins);
    
    // Create bin edges for digitizing
    let mut bin_edges = Vec::with_capacity(n_bins + 1);
    for i in 0..=n_bins {
        bin_edges.push(-pre_time + (i as f32 * bin_size));
    }
    
    // Allow Python threads to run during computation
    let result = Python::allow_threads(py, || {
        // Create output array: bins × events
        let mut binned_spikes = Array2::<f32>::zeros((n_bins, events.len()));
        
        // Process each event
        for (e_idx, &event_time) in events.iter().enumerate() {
            // Calculate window boundaries
            let window_start = event_time - pre_time;
            let window_end = event_time + post_time;
            
            // Process spikes within window
            for &spike_time in spikes.iter() {
                // Skip if spike is outside window
                if spike_time < window_start || spike_time >= window_end {
                    continue;
                }
                
                // Find relative time and bin
                let rel_time = spike_time - event_time;
                let bin_idx = ((rel_time + pre_time) / bin_size) as usize;
                
                // Safety check for bin index
                if bin_idx < n_bins {
                    binned_spikes[[bin_idx, e_idx]] += 1.0;
                }
            }
        }
        
        (binned_spikes, time_bins)
    });
    
    // Convert back to Python arrays
    Ok((
        result.0.into_pyarray(py).into(),
        result.1.into_pyarray(py).into(),
    ))
}

/// Compute PSTH (Peristimulus Time Histogram) for a single unit with parallel processing
/// 
/// Args:
///     spike_times: Spike times in seconds
///     event_times: Event/stimulus times in seconds
///     pre_time: Time before events to include (seconds)
///     post_time: Time after events to include (seconds)
///     bin_size: Size of bins in seconds
///     smoothing_sigma: Standard deviation for Gaussian smoothing (in bins, None for no smoothing)
///
/// Returns:
///     Tuple of (binned_spikes, time_bins, smoothed_data if requested)
#[pyfunction]
pub fn compute_psth_parallel(
    py: Python<'_>,
    spike_times: PyReadonlyArray1<f32>,
    event_times: PyReadonlyArray1<f32>,
    pre_time: f32,
    post_time: f32,
    bin_size: f32,
    smoothing_sigma: Option<f32>,
) -> PyResult<(Py<PyArray2<f32>>, Py<PyArray1<f32>>, Option<Py<PyArray2<f32>>>)> {
    // Convert inputs to Rust arrays
    let spikes = spike_times.as_array().to_owned();
    let events_array = event_times.as_array().to_owned();
    
    // Convert ndarray to Vec for using par_iter
    let events: Vec<f32> = events_array.iter().cloned().collect();
    
    // Calculate number of bins
    let total_time = pre_time + post_time;
    let n_bins = (total_time / bin_size) as usize;
    if n_bins == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Window size must be larger than bin size"
        ));
    }
    
    // Create time bins centered on event onset
    let time_bins = Array1::linspace(-pre_time, post_time, n_bins);
    
    // Allow Python threads to run during computation
    let result = Python::allow_threads(py, || {
        // Create output array: bins × events
        let mut binned_spikes = Array2::<f32>::zeros((n_bins, events.len()));
        
        // Process each event in parallel with Rayon
        let event_counts: Vec<_> = events
            .par_iter()
            .enumerate()
            .map(|(e_idx, &event_time)| {
                // Create histogram for this event
                let mut event_counts = vec![0.0f32; n_bins];
                
                // Calculate window boundaries
                let window_start = event_time - pre_time;
                let window_end = event_time + post_time;
                
                // Process spikes within window
                for &spike_time in spikes.iter() {
                    // Skip if spike is outside window
                    if spike_time < window_start || spike_time >= window_end {
                        continue;
                    }
                    
                    // Find relative time and bin
                    let rel_time = spike_time - event_time;
                    let bin_idx = ((rel_time + pre_time) / bin_size) as usize;
                    
                    // Safety check for bin index
                    if bin_idx < n_bins {
                        event_counts[bin_idx] += 1.0;
                    }
                }
                
                (e_idx, event_counts)
            })
            .collect();
        
        // Combine results
        for (e_idx, counts) in event_counts {
            for bin_idx in 0..n_bins {
                binned_spikes[[bin_idx, e_idx]] = counts[bin_idx];
            }
        }
        
        // Apply smoothing if requested
        let smoothed = if let Some(sigma) = smoothing_sigma {
            Some(apply_gaussian_smoothing_parallel(&binned_spikes, sigma))
        } else {
            None
        };
        
        (binned_spikes, time_bins, smoothed)
    });
    
    // Convert back to Python arrays
    match result.2 {
        Some(smoothed) => Ok((
            result.0.into_pyarray(py).into(),
            result.1.into_pyarray(py).into(),
            Some(smoothed.into_pyarray(py).into()),
        )),
        None => Ok((
            result.0.into_pyarray(py).into(),
            result.1.into_pyarray(py).into(),
            None,
        )),
    }
}

/// Apply Gaussian smoothing with parallel processing
fn apply_gaussian_smoothing_parallel(data: &Array2<f32>, sigma: f32) -> Array2<f32> {
    if sigma <= 0.0 {
        return data.clone();
    }
    
    // Create optimized Gaussian kernel (odd size for symmetric window)
    let kernel_radius = (sigma * 3.0).ceil() as usize;
    let kernel_size = kernel_radius * 2 + 1;
    let mut kernel = Vec::with_capacity(kernel_size);
    
    // Pre-compute kernel values
    let scale = -0.5 / (sigma * sigma);
    for i in 0..kernel_size {
        let x = (i as f32) - (kernel_size as f32 - 1.0) / 2.0;
        kernel.push((x * x * scale).exp());
    }
    
    // Normalize kernel
    let kernel_sum: f32 = kernel.iter().sum();
    if kernel_sum > 0.0 {
        for val in &mut kernel {
            *val /= kernel_sum;
        }
    }
    
    // Get dimensions
    let (n_bins, n_events) = data.dim();
    let mut smoothed = Array2::<f32>::zeros((n_bins, n_events));
    
    // Convert to Vec of columns for parallelization
    let columns: Vec<_> = (0..n_events)
        .map(|e_idx| {
            let data_col: Vec<f32> = data.column(e_idx).iter().cloned().collect();
            (e_idx, data_col)
        })
        .collect();
    
    // Apply convolution in parallel across events
    let smoothed_cols: Vec<_> = columns
        .par_iter()
        .map(|(e_idx, data_col)| {
            let mut smoothed_col = vec![0.0f32; n_bins];
            let kernel_half = kernel_size / 2;
            
            // Apply convolution
            for i in 0..n_bins {
                let mut sum = 0.0;
                
                for k in 0..kernel_size {
                    // Calculate source index with edge padding (mirror)
                    let src_idx = i as isize + (k as isize) - (kernel_half as isize);
                    let padded_idx = if src_idx < 0 {
                        -src_idx
                    } else if src_idx >= n_bins as isize {
                        2 * (n_bins as isize) - src_idx - 2
                    } else {
                        src_idx
                    } as usize;
                    
                    // Ensure index is valid
                    if padded_idx < n_bins {
                        sum += data_col[padded_idx] * kernel[k];
                    }
                }
                
                smoothed_col[i] = sum;
            }
            
            (*e_idx, smoothed_col)
        })
        .collect();
    
    // Copy smoothed columns back to the result array
    for (e_idx, col) in smoothed_cols {
        for i in 0..n_bins {
            smoothed[[i, e_idx]] = col[i];
        }
    }
    
    smoothed
}

/// Compute PSTH for all units in parallel
/// 
/// Args:
///     spike_times_list: List of spike times (one per unit) in seconds
///     event_times: Event/stimulus times in seconds
///     unit_ids: List of unit IDs
///     pre_time: Time before events to include (seconds)
///     post_time: Time after events to include (seconds)
///     bin_size: Size of bins in seconds
///     smoothing_sigma: Standard deviation for Gaussian smoothing (in bins, None for no smoothing)
///
/// Returns:
///     Dictionary with PSTH results
#[pyfunction]
pub fn compute_psth_all(
    py: Python<'_>,
    spike_times_list: Vec<PyReadonlyArray1<f32>>,
    event_times: PyReadonlyArray1<f32>,
    unit_ids: Vec<i32>,
    pre_time: f32,
    post_time: f32,
    bin_size: f32,
    smoothing_sigma: Option<f32>,
) -> PyResult<PyObject> {
    // Convert events to Vec for parallelization
    let events_array = event_times.as_array().to_owned();
    let events: Vec<f32> = events_array.iter().cloned().collect();
    
    // Calculate number of bins
    let total_time = pre_time + post_time;
    let n_bins = (total_time / bin_size) as usize;
    if n_bins == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Window size must be larger than bin size"
        ));
    }
    
    // Create time bins centered on event onset
    let time_bins = Array1::linspace(-pre_time, post_time, n_bins);
    
    // Prepare output dictionary
    let result_dict = PyDict::new(py);
    let psth_list = PyList::empty(py);
    
    // Convert spike_times_list to a form suitable for parallel processing
    let unit_data: Vec<_> = spike_times_list.iter()
        .enumerate()
        .filter_map(|(i, spike_times_py)| {
            if i >= unit_ids.len() {
                return None;
            }
            
            let spikes = spike_times_py.as_array().to_owned();
            let unit_id = unit_ids[i];
            
            // Convert to Vec for easier parallelization
            let spike_vec: Vec<f32> = spikes.iter().cloned().collect();
            
            Some((i, unit_id, spike_vec))
        })
        .collect();
    
    // Process each unit in parallel
    let unit_results: Vec<_> = unit_data
        .par_iter()
        .map(|(i, unit_id, spikes)| {
            // Create output array for this unit: bins × events
            let mut binned_spikes = Array2::<f32>::zeros((n_bins, events.len()));
            
            // Process each event for this unit
            for (e_idx, &event_time) in events.iter().enumerate() {
                // Calculate window boundaries
                let window_start = event_time - pre_time;
                let window_end = event_time + post_time;
                
                // Process spikes within window
                for &spike_time in spikes.iter() {
                    // Skip if spike is outside window
                    if spike_time < window_start || spike_time >= window_end {
                        continue;
                    }
                    
                    // Find relative time and bin
                    let rel_time = spike_time - event_time;
                    let bin_idx = ((rel_time + pre_time) / bin_size) as usize;
                    
                    // Safety check for bin index
                    if bin_idx < n_bins {
                        binned_spikes[[bin_idx, e_idx]] += 1.0;
                    }
                }
            }
            
            // Apply smoothing if requested
            let smoothed = if let Some(sigma) = smoothing_sigma {
                Some(apply_gaussian_smoothing_parallel(&binned_spikes, sigma))
            } else {
                None
            };
            
            (*i, unit_id, binned_spikes, smoothed)
        })
        .collect();
    
    // Convert results to Python objects
    for (_i, unit_id, binned_spikes, smoothed) in unit_results {
        // Create dictionary for this unit's PSTH
        let psth_dict = PyDict::new(py);
        psth_dict.set_item("unit_id", unit_id)?;
        psth_dict.set_item("psth_counts", binned_spikes.into_pyarray(py))?;
        
        if let Some(smoothed_data) = smoothed {
            psth_dict.set_item("psth_smoothed", smoothed_data.into_pyarray(py))?;
        }
        
        // Add to list
        psth_list.append(psth_dict)?;
    }
    
    // Add results to output dictionary
    result_dict.set_item("psth_data", psth_list)?;
    result_dict.set_item("time_bins", time_bins.into_pyarray(py))?;
    result_dict.set_item("event_count", events.len())?;
    
    Ok(result_dict.into())
}

/// Compute spike raster data for visualization with parallel processing
/// 
/// Args:
///     spike_times: Spike times in seconds
///     event_times: Event/stimulus times in seconds
///     pre_time: Time before events to include (seconds)
///     post_time: Time after events to include (seconds)
///
/// Returns:
///     Dictionary with trial indices and relative spike times
#[pyfunction]
pub fn compute_raster_data(
    py: Python<'_>,
    spike_times: PyReadonlyArray1<f32>,
    event_times: PyReadonlyArray1<f32>,
    pre_time: f32,
    post_time: f32,
) -> PyResult<PyObject> {
    // Convert inputs to Rust arrays
    let spikes = spike_times.as_array().to_owned();
    let events_array = event_times.as_array().to_owned();
    
    // Convert to Vec for parallelization
    let events: Vec<f32> = events_array.iter().cloned().collect();
    let spike_vec: Vec<f32> = spikes.iter().cloned().collect();
    
    // Allow Python threads to run during computation
    let result = Python::allow_threads(py, || {
        // Process each event in parallel
        let event_results: Vec<_> = events
            .par_iter()
            .enumerate()
            .map(|(e_idx, &event_time)| {
                // Calculate window boundaries
                let window_start = event_time - pre_time;
                let window_end = event_time + post_time;
                
                // Find spikes within this window and compute relative times
                let mut event_spikes = Vec::new();
                
                for &spike_time in spike_vec.iter() {
                    if spike_time >= window_start && spike_time < window_end {
                        // Record relative spike time and trial index
                        event_spikes.push((spike_time - event_time, e_idx as i32));
                    }
                }
                
                event_spikes
            })
            .collect();
        
        // Flatten results
        let mut all_spikes = Vec::new();
        for event_spikes in event_results {
            all_spikes.extend(event_spikes);
        }
        
        // Split into separate vectors
        let (rel_times, trial_indices): (Vec<f32>, Vec<i32>) = all_spikes.into_iter().unzip();
        
        (rel_times, trial_indices)
    });
    
    // Create result dictionary
    let dict = PyDict::new(py);
    
    // Convert vectors to numpy arrays
    let rel_times_array = Array1::from(result.0);
    let trial_indices_array = Array1::from(result.1);
    
    dict.set_item("spike_times", rel_times_array.into_pyarray(py))?;
    dict.set_item("trials", trial_indices_array.into_pyarray(py))?;
    
    Ok(dict.into())
}