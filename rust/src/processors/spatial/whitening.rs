// whitening.rs
use ndarray::{Array1, Array2, ArrayBase, ArrayView2, Axis, Dim, Ix2};
use ndarray_linalg::{SVD, UPLO}; // Using SVD instead of Eigh
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;

/// Compute whitening matrix from a covariance matrix using ZCA whitening with SVD.
/// 
/// This implements the ZCA (Zero-phase Component Analysis) whitening:
/// W = U * diag(1/sqrt(S + eps)) * U^T
/// where U and S come from SVD decomposition of the covariance matrix
/// 
/// Args:
///     cov: Covariance matrix
///     eps: Regularization parameter for singular values
/// 
/// Returns:
///     Whitening matrix W
#[pyfunction]
pub fn compute_whitening_matrix(
    py: Python<'_>,
    cov: PyReadonlyArray2<f32>,
    eps: f32,
) -> PyResult<Py<PyArray2<f32>>> {
    // Convert input to rust ndarray
    let cov_array = cov.as_array().to_owned();
    
    // Allow Python threads to run during computation
    let whitening = Python::allow_threads(py, || {
        // Get dimensions
        let n = cov_array.shape()[0];
        
        // Check if covariance matrix is valid
        if n == 0 || cov_array.shape()[1] != n {
            return Array2::<f32>::eye(n);
        }
        
        // Perform SVD decomposition using ndarray-linalg
        match cov_array.svd(true, true) {
            Ok((Some(u), s, Some(vt))) => {
                // Create diagonal matrix with 1/sqrt(S + eps)
                let s_inv: Vec<f32> = s.iter()
                    .map(|&x| 1.0 / (x + eps).sqrt())
                    .collect();
                
                // For ZCA whitening, we use U*S^(-1/2)*U^T
                // (not V^T since covariance matrix is symmetric and U ≈ V)
                let diag_inv = Array2::from_diag(&Array1::from(s_inv));
                let result = u.dot(&diag_inv).dot(&u.t());
                
                result
            },
            _ => {
                // Fallback to a simple identity matrix if decomposition fails
                Array2::<f32>::eye(n)
            }
        }
    });
    
    // Convert back to Python
    Ok(whitening.into_pyarray(py).to_owned().into())
}

/// Compute covariance matrix with parallel processing
#[pyfunction]
pub fn compute_covariance_parallel(
    py: Python<'_>,
    data: PyReadonlyArray2<f32>,
) -> PyResult<Py<PyArray2<f32>>> {
    // Convert input to rust ndarray
    let data_array = data.as_array().to_owned();
    
    // Get dimensions
    let n_samples = data_array.shape()[0];
    let n_features = data_array.shape()[1];
    
    // Handle edge case
    if n_samples == 0 || n_features == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Empty data array provided"
        ));
    }
    
    // Compute covariance matrix
    let cov = Python::allow_threads(py, || {
        // For better performance, transpose once and use dot product
        let data_t = data_array.t();
        data_t.dot(&data_array) / (n_samples as f32)
    });
    
    // Convert back to Python
    Ok(cov.into_pyarray(py).to_owned().into())
}

/// Compute the mean of the data with parallel processing
#[pyfunction]
pub fn compute_mean_parallel(
    py: Python<'_>,
    data: PyReadonlyArray2<f32>,
) -> PyResult<Py<PyArray2<f32>>> {
    // Convert input to rust ndarray
    let data_array = data.as_array().to_owned();
    
    // Get dimensions
    let n_samples = data_array.shape()[0];
    let n_features = data_array.shape()[1];
    
    // Handle edge case
    if n_samples == 0 || n_features == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Empty data array provided"
        ));
    }
    
    // Compute mean in parallel
    let mean = Python::allow_threads(py, || {
        // Mean across samples (axis 0)
        match data_array.mean_axis(Axis(0)) {
            Some(mean_vec) => {
                // Convert to 2D for consistency with Python's reshape(-1, 1)
                Array2::from_shape_fn((1, n_features), |(_, j)| mean_vec[j])
            },
            None => {
                // Return zeros if mean calculation fails
                Array2::<f32>::zeros((1, n_features))
            }
        }
    });
    
    // Convert back to Python
    Ok(mean.into_pyarray(py).to_owned().into())
}

/// Apply whitening transformation to data with parallel processing
#[pyfunction]
pub fn apply_whitening_parallel(
    py: Python<'_>,
    data: PyReadonlyArray2<f32>,
    whitening_matrix: PyReadonlyArray2<f32>,
    mean: Option<PyReadonlyArray2<f32>>,
    int_scale: Option<f32>,
) -> PyResult<Py<PyArray2<f32>>> {
    // Convert inputs to rust ndarrays
    let data_array = data.as_array().to_owned();
    let whitening_array = whitening_matrix.as_array().to_owned();
    
    // Convert optional mean to ndarray if provided
    let mean_array = mean.map(|m| m.as_array().to_owned());
    
    // Validate dimensions
    let n_samples = data_array.shape()[0];
    let n_features = data_array.shape()[1];
    
    if whitening_array.shape()[0] != n_features || whitening_array.shape()[1] != n_features {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("Whitening matrix shape {:?} incompatible with data shape {:?}", 
                    whitening_array.shape(), data_array.shape())
        ));
    }
    
    if let Some(ref mean_arr) = mean_array {
        if mean_arr.shape()[1] != n_features {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Mean shape {:?} incompatible with data shape {:?}", 
                        mean_arr.shape(), data_array.shape())
            ));
        }
    }
    
    // Allow Python threads to run during computation
    let whitened = Python::allow_threads(py, || {
        // Prepare centered data
        let centered_data = match &mean_array {
            Some(mean) => {
                // Subtract mean from each row
                Array2::from_shape_fn((n_samples, n_features), |(i, j)| {
                    data_array[[i, j]] - mean[[0, j]]
                })
            },
            None => data_array.clone(),
        };
        
        // Apply whitening matrix
        let mut result = centered_data.dot(&whitening_array);
        
        // Apply optional scaling
        if let Some(scale) = int_scale {
            result.mapv_inplace(|x| x * scale);
        }
        
        result
    });
    
    // Convert result back to Python
    Ok(whitened.into_pyarray(py).to_owned().into())
}

/// Optimize whitening by combining center, covariance, and whitening in one function
/// This reduces intermediate allocations and may be more efficient for large datasets
#[pyfunction]
pub fn whiten_data(
    py: Python<'_>,
    data: PyReadonlyArray2<f32>,
    eps: f32,
    apply_mean: bool,
    int_scale: Option<f32>,
) -> PyResult<Py<PyArray2<f32>>> {
    // Convert input to rust ndarray
    let data_array = data.as_array().to_owned();
    
    // Validate dimensions
    let n_samples = data_array.shape()[0];
    let n_features = data_array.shape()[1];
    
    if n_samples == 0 || n_features == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Empty data array provided"
        ));
    }
    
    let whitened = Python::allow_threads(py, || {
        // Compute mean if required
        let (centered_data, mean) = if apply_mean {
            match data_array.mean_axis(Axis(0)) {
                Some(mean_vec) => {
                    // Center the data by subtracting the mean
                    let centered = Array2::from_shape_fn((n_samples, n_features), |(i, j)| {
                        data_array[[i, j]] - mean_vec[j]
                    });
                    
                    (centered, Some(mean_vec))
                },
                None => (data_array.clone(), None)
            }
        } else {
            (data_array.clone(), None)
        };
        
        // Compute covariance matrix
        let cov_matrix = centered_data.t().dot(&centered_data) / (n_samples as f32);
        
        // Compute whitening matrix via SVD
        let whitening_matrix = match cov_matrix.svd(true, true) {
            Ok((Some(u), s, Some(_))) => {
                // Create diagonal matrix with 1/sqrt(S + eps)
                let s_inv: Vec<f32> = s.iter()
                    .map(|&x| 1.0 / (x + eps).sqrt())
                    .collect();
                
                // For ZCA whitening, we use U*S^(-1/2)*U^T
                let diag_inv = Array2::from_diag(&Array1::from(s_inv));
                u.dot(&diag_inv).dot(&u.t())
            },
            _ => {
                // Fallback to identity if decomposition fails
                Array2::<f32>::eye(n_features)
            }
        };
        
        // Apply whitening
        let mut whitened_data = centered_data.dot(&whitening_matrix);
        
        // Apply optional scaling
        if let Some(scale) = int_scale {
            whitened_data.mapv_inplace(|x| x * scale);
        }
        
        whitened_data
    });
    
    // Convert back to Python
    Ok(whitened.into_pyarray(py).to_owned().into())
}