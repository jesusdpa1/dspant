// rust/src/processors/spike_analytics/correlogram.rs

use ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rayon::prelude::*;
use rand::Rng;
// Import Distribution trait and Uniform distribution
use rand::distr::{Distribution, Uniform};

/// Compute a correlogram between two spike trains
/// 
/// Args:
///     spike_times1: First spike train times in seconds
///     spike_times2: Second spike train times in seconds
///     bin_size: Size of bins in seconds
///     window_size: Total window size in seconds
///     normalize: Normalization method (none, rate, prob)
///
/// Returns:
///     Tuple of (correlogram counts, time bins)
#[pyfunction]
pub fn compute_correlogram(
    py: Python<'_>,
    spike_times1: PyReadonlyArray1<f32>,
    spike_times2: PyReadonlyArray1<f32>,
    bin_size: f32,
    window_size: f32,
    normalize: Option<String>,
) -> PyResult<(Py<PyArray1<f32>>, Py<PyArray1<f32>>)> {
    
    // Convert inputs to Rust arrays
    let spikes1 = spike_times1.as_array().to_owned();
    let spikes2 = spike_times2.as_array().to_owned();
    
    // Calculate the number of bins needed
    let n_bins = (window_size / bin_size) as usize;
    if n_bins == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Window size must be larger than bin size"
        ));
    }
    
    let half_window = window_size / 2.0;
    
    // Create output arrays
    let mut correlogram = Array1::<f32>::zeros(n_bins);
    let time_bins = Array1::linspace(-half_window, half_window, n_bins);
    
    // Allow Python threads to run during computation
    let result = Python::allow_threads(py, || {
        // Compute correlogram with parallelization
        compute_correlogram_parallel(&spikes1, &spikes2, bin_size, window_size, &mut correlogram);
        
        // Apply normalization if requested
        if let Some(norm_method) = normalize {
            match norm_method.as_str() {
                "rate" => {
                    // Convert to firing rate (Hz)
                    let n_spikes1 = spikes1.len() as f32;
                    // Fix float comparison
                    if n_spikes1 > 0.0 {
                        // Calculate the max value to get recording duration
                        if let Some(_max_time) = spikes1.iter().cloned().reduce(f32::max) {
                            // Scale by number of reference spikes and bin size
                            let scale_factor = 1.0 / (n_spikes1 * bin_size);
                            for i in 0..correlogram.len() {
                                correlogram[i] *= scale_factor;
                            }
                        }
                    }
                },
                "probability" => {
                    // Convert to probability
                    let total_count: f32 = correlogram.sum();
                    if total_count > 0.0 {
                        for i in 0..correlogram.len() {
                            correlogram[i] /= total_count;
                        }
                    }
                },
                _ => {} // No normalization (raw counts)
            }
        }
        
        (correlogram, time_bins)
    });
    
    // Convert back to Python arrays with proper .into() conversion
    Ok((
        result.0.into_pyarray(py).into(),
        result.1.into_pyarray(py).into(),
    ))
}

/// Parallel implementation of correlogram computation
fn compute_correlogram_parallel(
    spikes1: &Array1<f32>,
    spikes2: &Array1<f32>,
    bin_size: f32,
    window_size: f32,
    output: &mut Array1<f32>,
) {
    let n_bins = output.len();
    let half_window = window_size / 2.0;
    
    // Process each spike in the first train in parallel
    let spike_counts: Vec<Array1<f32>> = spikes1
        .par_iter()
        .map(|&t1| {
            let mut counts = Array1::<f32>::zeros(n_bins);
            
            // Find all spikes in train 2 that could be within the window
            for &t2 in spikes2.iter() {
                let time_diff = t2 - t1;
                
                // Skip if time difference is outside window
                if time_diff.abs() > half_window {
                    continue;
                }
                
                // Find bin index
                let bin_idx = ((time_diff + half_window) / bin_size) as usize;
                
                // Ensure the bin index is valid
                if bin_idx < n_bins {
                    counts[bin_idx] += 1.0;
                }
            }
            
            counts
        })
        .collect();
    
    // Combine results
    if !spike_counts.is_empty() {
        // Sum all individual correlograms
        for counts in spike_counts {
            for i in 0..n_bins {
                output[i] += counts[i];
            }
        }
    }
}

/// Compute autocorrelogram for a single spike train
#[pyfunction]
pub fn compute_autocorrelogram(
    py: Python<'_>,
    spike_times: PyReadonlyArray1<f32>,
    bin_size: f32,
    window_size: f32,
    normalize: Option<String>,
) -> PyResult<(Py<PyArray1<f32>>, Py<PyArray1<f32>>)> {
    
    // Use compute_correlogram with both inputs being the same spike train
    compute_correlogram(
        py,
        spike_times.clone(),
        spike_times,
        bin_size,
        window_size,
        normalize,
    )
}

/// Compute cross-correlograms for multiple spike trains efficiently
#[pyfunction]
pub fn compute_all_cross_correlograms(
    py: Python<'_>,
    spike_times_list: Vec<PyReadonlyArray1<f32>>,
    unit_ids: Option<Vec<i32>>,
    bin_size: f32,
    window_size: f32,
    normalize: Option<String>,
) -> PyResult<PyObject> {
    // Create Python dictionaries and lists
    let result_dict = PyDict::new(py);
    // Create empty lists correctly
    let autocorrelograms = PyList::empty(py);
    let crosscorrelograms = PyList::empty(py);
    
    // Convert unit IDs or create default ones
    let ids: Vec<i32> = match unit_ids {
        Some(ids) => ids,
        None => (0..spike_times_list.len() as i32).collect(),
    };
    
    // Check if we have enough spike trains
    if spike_times_list.is_empty() {
        result_dict.set_item("autocorrelograms", autocorrelograms)?;
        result_dict.set_item("crosscorrelograms", crosscorrelograms)?;
        return Ok(result_dict.into());
    }
    
    // Compute autocorrelograms
    for (i, spike_train) in spike_times_list.iter().enumerate() {
        let (counts, bins) = compute_autocorrelogram(
            py,
            spike_train.clone(),
            bin_size,
            window_size,
            normalize.clone(),
        )?;
        
        // Create dictionary for this autocorrelogram
        let auto_dict = PyDict::new(py);
        auto_dict.set_item("unit_id", ids[i])?;
        auto_dict.set_item("autocorrelogram", counts)?;
        auto_dict.set_item("time_bins", bins)?;
        
        autocorrelograms.append(auto_dict)?;
    }
    
    // Compute cross-correlograms
    for i in 0..spike_times_list.len() {
        for j in (i+1)..spike_times_list.len() {
            let (counts, bins) = compute_correlogram(
                py,
                spike_times_list[i].clone(),
                spike_times_list[j].clone(),
                bin_size,
                window_size,
                normalize.clone(),
            )?;
            
            // Create dictionary for this cross-correlogram
            let cross_dict = PyDict::new(py);
            cross_dict.set_item("unit1", ids[i])?;
            cross_dict.set_item("unit2", ids[j])?;
            cross_dict.set_item("crosscorrelogram", counts)?;
            cross_dict.set_item("time_bins", bins)?;
            
            crosscorrelograms.append(cross_dict)?;
        }
    }
    
    // Add results to output dictionary
    result_dict.set_item("autocorrelograms", autocorrelograms)?;
    result_dict.set_item("crosscorrelograms", crosscorrelograms)?;
    
    Ok(result_dict.into())
}

/// More efficient computation of jitter-corrected correlogram
#[pyfunction]
pub fn compute_jitter_corrected_correlogram(
    py: Python<'_>,
    spike_times1: PyReadonlyArray1<f32>,
    spike_times2: PyReadonlyArray1<f32>,
    bin_size: f32,
    window_size: f32,
    jitter_window: f32,
    jitter_iterations: usize,
) -> PyResult<(Py<PyArray1<f32>>, Py<PyArray1<f32>>)> {
    // Convert inputs to Rust arrays
    let spikes1 = spike_times1.as_array().to_owned();
    let spikes2 = spike_times2.as_array().to_owned();
    
    // Calculate the number of bins needed
    let n_bins = (window_size / bin_size) as usize;
    if n_bins == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Window size must be larger than bin size"
        ));
    }
    
    let half_window = window_size / 2.0;
    
    // Create output arrays
    let mut correlogram = Array1::<f32>::zeros(n_bins);
    let mut jitter_correlogram = Array1::<f32>::zeros(n_bins);
    let time_bins = Array1::linspace(-half_window, half_window, n_bins);
    
    // Allow Python threads to run during computation
    let result = Python::allow_threads(py, || {
        // Compute original correlogram
        compute_correlogram_parallel(&spikes1, &spikes2, bin_size, window_size, &mut correlogram);
        
        // Compute jittered correlogram by averaging over multiple iterations
        let mut rng = rand::rng();
        
        // Create Uniform distribution in rand 0.9 - proper way to handle Result
        let uniform_result = Uniform::new(-jitter_window/2.0, jitter_window/2.0);
        
        // Safely unwrap the Result with a fallback
        let jitter_dist = match uniform_result {
            Ok(dist) => dist,
            Err(_) => {
                // Fallback to a smaller range if the original fails
                Uniform::new(-0.1, 0.1).unwrap_or_else(|_| {
                    // If even that fails, use zero jitter
                    Uniform::new(0.0, 0.0).unwrap()
                })
            }
        };
        
        // Create jittered spike trains for each iteration
        for _ in 0..jitter_iterations {
            // Create jittered spike train 2 - note that we sample FROM the distribution
            let jittered_spikes2: Array1<f32> = spikes2
                .iter()
                .map(|&t| t + jitter_dist.sample(&mut rng))
                .collect();
            
            // Compute correlogram for this jittered version
            let mut iter_correlogram = Array1::<f32>::zeros(n_bins);
            compute_correlogram_parallel(&spikes1, &jittered_spikes2, bin_size, window_size, &mut iter_correlogram);
            
            // Add to average
            for i in 0..n_bins {
                jitter_correlogram[i] += iter_correlogram[i];
            }
        }
        
        // Average the jittered correlograms
        if jitter_iterations > 0 {
            for i in 0..n_bins {
                jitter_correlogram[i] /= jitter_iterations as f32;
            }
        }
        
        // Subtract jitter correlogram from original
        for i in 0..n_bins {
            correlogram[i] -= jitter_correlogram[i];
            
            // Set negative values to zero (optional)
            if correlogram[i] < 0.0 {
                correlogram[i] = 0.0;
            }
        }
        
        (correlogram, time_bins)
    });
    
    // Convert back to Python arrays with proper conversion
    Ok((
        result.0.into_pyarray(py).into(),
        result.1.into_pyarray(py).into(),
    ))
}

/// Compute a spike time tiling coefficient, which is less biased by firing rate differences
#[pyfunction]
pub fn compute_spike_time_tiling_coefficient(
    py: Python<'_>,
    spike_times1: PyReadonlyArray1<f32>,
    spike_times2: PyReadonlyArray1<f32>,
    delta_t: f32,
) -> PyResult<f32> {
    // Convert inputs to Rust arrays
    let spikes1 = spike_times1.as_array().to_owned();
    let spikes2 = spike_times2.as_array().to_owned();
    
    // Handle edge cases
    if spikes1.len() == 0 || spikes2.len() == 0 {
        return Ok(0.0);
    }
    
    // Allow Python threads to run during computation
    let result = Python::allow_threads(py, || {
        // Count spikes from train 1 that have at least one spike from train 2 within ±delta_t
        let mut count_1 = 0;
        for &t1 in spikes1.iter() {
            let has_nearby = spikes2.iter().any(|&t2| (t2 - t1).abs() <= delta_t);
            if has_nearby {
                count_1 += 1;
            }
        }
        
        // Count spikes from train 2 that have at least one spike from train 1 within ±delta_t
        let mut count_2 = 0;
        for &t2 in spikes2.iter() {
            let has_nearby = spikes1.iter().any(|&t1| (t1 - t2).abs() <= delta_t);
            if has_nearby {
                count_2 += 1;
            }
        }
        
        // Calculate the tiling coefficient
        let ta = count_1 as f32 / spikes1.len() as f32;
        let tb = count_2 as f32 / spikes2.len() as f32;
        
        // Avoid division by zero
        let denominator = (1.0 - ta * tb).max(1e-10);
        (ta + tb - ta * tb) / denominator
    });
    
    Ok(result)
}