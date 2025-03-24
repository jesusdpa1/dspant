// rust/src/processors/basics/normalization.rs

use ndarray::{Array2, ArrayView2};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

/// Apply z-score normalization to a numpy array
///
/// z-score = (x - mean) / std
#[pyfunction]
pub fn apply_zscore(
    py: Python<'_>,
    data: PyReadonlyArray2<f32>,
    mean: f32,
    std: f32,
) -> PyResult<Py<PyArray2<f32>>> {
    // Convert input to rust ndarray
    let data_array = data.as_array();
    
    // Handle zero std case
    let std_safe = if std == 0.0 { 1.0 } else { std };
    
    // Apply normalization
    let result = data_array.mapv(|x| (x - mean) / std_safe);
    
    // Convert back to Python and explicitly call into() to get owned Py<PyArray>
    Ok(result.into_pyarray(py).into())
}

/// Apply z-score normalization with parallel processing for large arrays
#[pyfunction]
pub fn apply_zscore_parallel(
    py: Python<'_>,
    data: PyReadonlyArray2<f32>,
    mean: f32,
    std: f32,
) -> PyResult<Py<PyArray2<f32>>> {
    // Extract data to a Rust array before releasing the GIL
    let data_array = data.as_array().to_owned();
    
    // Handle zero std case
    let std_safe = if std == 0.0 { 1.0 } else { std };
    
    // Allow Python threads to run during computation
    let output = Python::allow_threads(py, || {
        // Apply normalization in parallel
        let mut result = data_array.clone();
        
        result.par_mapv_inplace(|x| (x - mean) / std_safe);
        
        result
    });
    
    // Convert the result back to Python with into()
    Ok(output.into_pyarray(py).into())
}

/// Apply min-max normalization to a numpy array
///
/// min-max = (x - min) / (max - min)
#[pyfunction]
pub fn apply_minmax(
    py: Python<'_>,
    data: PyReadonlyArray2<f32>,
    min_val: f32,
    max_val: f32,
) -> PyResult<Py<PyArray2<f32>>> {
    // Convert input to rust ndarray
    let data_array = data.as_array();
    
    // Handle case where min equals max
    if min_val == max_val {
        let zeros = Array2::zeros(data_array.dim());
        return Ok(zeros.into_pyarray(py).into());
    }
    
    // Apply normalization
    let range = max_val - min_val;
    let result = data_array.mapv(|x| (x - min_val) / range);
    
    // Convert back to Python with into()
    Ok(result.into_pyarray(py).into())
}

/// Apply min-max normalization with parallel processing for large arrays
#[pyfunction]
pub fn apply_minmax_parallel(
    py: Python<'_>,
    data: PyReadonlyArray2<f32>,
    min_val: f32,
    max_val: f32,
) -> PyResult<Py<PyArray2<f32>>> {
    // Extract data to a Rust array before releasing the GIL
    let data_array = data.as_array().to_owned();
    
    // Handle case where min equals max
    if min_val == max_val {
        let zeros = Array2::zeros(data_array.dim());
        return Ok(zeros.into_pyarray(py).into());
    }
    
    // Range for normalization
    let range = max_val - min_val;
    
    // Allow Python threads to run during computation
    let output = Python::allow_threads(py, || {
        // Apply normalization in parallel
        let mut result = data_array.clone();
        
        result.par_mapv_inplace(|x| (x - min_val) / range);
        
        result
    });
    
    // Convert the result back to Python with into()
    Ok(output.into_pyarray(py).into())
}

/// Apply robust normalization to a numpy array
///
/// robust = (x - median) / iqr
#[pyfunction]
pub fn apply_robust(
    py: Python<'_>,
    data: PyReadonlyArray2<f32>,
    median: f32,
    iqr: f32,
) -> PyResult<Py<PyArray2<f32>>> {
    // Convert input to rust ndarray
    let data_array = data.as_array();
    
    // Handle zero iqr case
    let iqr_safe = if iqr == 0.0 { 1.0 } else { iqr };
    
    // Apply normalization
    let result = data_array.mapv(|x| (x - median) / iqr_safe);
    
    // Convert back to Python with into()
    Ok(result.into_pyarray(py).into())
}

/// Apply robust normalization with parallel processing for large arrays
#[pyfunction]
pub fn apply_robust_parallel(
    py: Python<'_>,
    data: PyReadonlyArray2<f32>,
    median: f32,
    iqr: f32,
) -> PyResult<Py<PyArray2<f32>>> {
    // Extract data to a Rust array before releasing the GIL
    let data_array = data.as_array().to_owned();
    
    // Handle zero iqr case
    let iqr_safe = if iqr == 0.0 { 1.0 } else { iqr };
    
    // Allow Python threads to run during computation
    let output = Python::allow_threads(py, || {
        // Apply normalization in parallel
        let mut result = data_array.clone();
        
        result.par_mapv_inplace(|x| (x - median) / iqr_safe);
        
        result
    });
    
    // Convert the result back to Python with into()
    Ok(output.into_pyarray(py).into())
}

/// Apply MAD (Median Absolute Deviation) normalization to a numpy array
///
/// mad = (x - median) / (k * mad)
/// where k is a constant (typically 1.4826 for normally distributed data)
#[pyfunction]
pub fn apply_mad(
    py: Python<'_>,
    data: PyReadonlyArray2<f32>,
    median: f32,
    mad: f32,
    k: f32,
) -> PyResult<Py<PyArray2<f32>>> {
    // Convert input to rust ndarray
    let data_array = data.as_array();
    
    // Handle zero mad case
    let mad_safe = if mad == 0.0 { 1.0 } else { mad };
    
    // Apply normalization
    let k_mad = k * mad_safe;
    let result = data_array.mapv(|x| (x - median) / k_mad);
    
    // Convert back to Python with into()
    Ok(result.into_pyarray(py).into())
}

/// Apply MAD normalization with parallel processing for large arrays
#[pyfunction]
pub fn apply_mad_parallel(
    py: Python<'_>,
    data: PyReadonlyArray2<f32>,
    median: f32,
    mad: f32,
    k: f32,
) -> PyResult<Py<PyArray2<f32>>> {
    // Extract data to a Rust array before releasing the GIL
    let data_array = data.as_array().to_owned();
    
    // Handle zero mad case
    let mad_safe = if mad == 0.0 { 1.0 } else { mad };
    
    // Allow Python threads to run during computation
    let output = Python::allow_threads(py, || {
        // Apply normalization in parallel
        let mut result = data_array.clone();
        
        let k_mad = k * mad_safe;
        result.par_mapv_inplace(|x| (x - median) / k_mad);
        
        result
    });
    
    // Convert the result back to Python with into()
    Ok(output.into_pyarray(py).into())
}