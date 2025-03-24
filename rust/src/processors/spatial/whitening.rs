// rust/src/processors/spatial/whitening.rs

use ndarray::{Array1, Array2, ArrayView2};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;
use ndarray_linalg::{Eigh, UPLO};  // For eigendecomposition

/// Compute the whitening matrix from a covariance matrix using ZCA whitening
///
/// ZCA whitening: W = U * diag(1/sqrt(S + eps)) * U^T where U, S come from eigendecomposition of covariance
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
        // Since covariance matrix is symmetric positive semi-definite,
        // we can use eigendecomposition instead of SVD (more efficient)
        let n = cov_array.shape()[0];
        
        // Perform eigendecomposition
        match cov_array.eigh(UPLO::Upper) {
            Ok((eigvals, eigvecs)) => {
                // Create diagonal matrix with 1/sqrt(S + eps)
                let s_inv: Vec<f32> = eigvals.iter().map(|&x| 1.0 / (x + eps).sqrt()).collect();
                
                // Compute whitening matrix: W = U * diag(1/sqrt(S + eps)) * U^T
                let mut result = Array2::<f32>::zeros((n, n));
                
                // Manually compute the matrix multiplication
                for i in 0..n {
                    for j in 0..n {
                        let mut sum = 0.0;
                        for k in 0..n {
                            sum += eigvecs[[i, k]] * s_inv[k] * eigvecs[[j, k]];
                        }
                        result[[i, j]] = sum;
                    }
                }
                
                result
            },
            Err(_) => {
                // Fallback to a simple identity matrix if eigendecomposition fails
                Array2::<f32>::eye(n)
            }
        }
    });
    
    // Convert back to Python
    Ok(whitening.into_pyarray(py).into())
}

/// Apply whitening transformation to a data matrix
///
/// whitened = (data - mean) @ whitening_matrix
#[pyfunction]
pub fn apply_whitening(
    py: Python<'_>,
    data: PyReadonlyArray2<f32>,
    whitening_matrix: PyReadonlyArray2<f32>,
    mean: Option<PyReadonlyArray2<f32>>,
    int_scale: Option<f32>,
) -> PyResult<Py<PyArray2<f32>>> {
    // Convert inputs to rust ndarrays - use .to_owned() to get owned data
    let data_array = data.as_array().to_owned();
    let whitening_array = whitening_matrix.as_array().to_owned();
    
    // Convert optional mean to owned ndarray if provided
    let mean_array = if let Some(m) = mean {
        Some(m.as_array().to_owned())
    } else {
        None
    };
    
    // Allow Python threads to run during computation
    let whitened = Python::allow_threads(py, || {
        let n_samples = data_array.shape()[0];
        let n_features = data_array.shape()[1];
        let mut result = Array2::<f32>::zeros((n_samples, n_features));
        
        // Apply whitening transformation
        if let Some(mean_arr) = &mean_array {
            // Apply mean centering and whitening
            for i in 0..n_samples {
                for j in 0..n_features {
                    let mut sum = 0.0;
                    for k in 0..n_features {
                        sum += (data_array[[i, k]] - mean_arr[[0, k]]) * whitening_array[[k, j]];
                    }
                    result[[i, j]] = sum;
                }
            }
        } else {
            // Apply whitening without mean centering
            for i in 0..n_samples {
                for j in 0..n_features {
                    let mut sum = 0.0;
                    for k in 0..n_features {
                        sum += data_array[[i, k]] * whitening_array[[k, j]];
                    }
                    result[[i, j]] = sum;
                }
            }
        }
        
        // Apply scaling if needed
        if let Some(scale) = int_scale {
            for i in 0..n_samples {
                for j in 0..n_features {
                    result[[i, j]] *= scale;
                }
            }
        }
        
        result
    });
    
    // Convert result back to Python
    Ok(whitened.into_pyarray(py).into())
}

/// Apply whitening transformation to a data matrix with parallel processing
///
/// whitened = (data - mean) @ whitening_matrix
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
    let mean_array = if let Some(m) = mean {
        Some(m.as_array().to_owned())
    } else {
        None
    };
    
    // Allow Python threads to run during computation
    let whitened = Python::allow_threads(py, || {
        let n_samples = data_array.shape()[0];
        let n_features = data_array.shape()[1];
        
        // Create the result array
        let mut result = Array2::<f32>::zeros((n_samples, n_features));
        
        // Process each row in parallel using approach similar to tkeo.rs
        let row_results: Vec<_> = (0..n_samples)
            .into_par_iter()
            .map(|i| {
                let mut row_output = vec![0.0f32; n_features];
                
                if let Some(ref mean_arr) = mean_array {
                    // Apply mean centering and whitening
                    for j in 0..n_features {
                        let mut sum = 0.0;
                        for k in 0..n_features {
                            sum += (data_array[[i, k]] - mean_arr[[0, k]]) * whitening_array[[k, j]];
                        }
                        row_output[j] = sum;
                    }
                } else {
                    // Apply whitening without mean centering
                    for j in 0..n_features {
                        let mut sum = 0.0;
                        for k in 0..n_features {
                            sum += data_array[[i, k]] * whitening_array[[k, j]];
                        }
                        row_output[j] = sum;
                    }
                }
                
                // Apply scaling if needed
                if let Some(scale) = int_scale {
                    for j in 0..n_features {
                        row_output[j] *= scale;
                    }
                }
                
                (i, row_output)
            })
            .collect();
        
        // Combine results from all rows
        for (i, row_data) in row_results {
            for j in 0..n_features {
                result[[i, j]] = row_data[j];
            }
        }
        
        result
    });
    
    // Convert result back to Python
    Ok(whitened.into_pyarray(py).into())
}

/// Compute covariance matrix from data
///
/// cov = data.T @ data / n_samples
#[pyfunction]
pub fn compute_covariance(
    py: Python<'_>,
    data: PyReadonlyArray2<f32>,
) -> PyResult<Py<PyArray2<f32>>> {
    // Convert input to rust ndarray
    let data_array = data.as_array();
    
    // Get dimensions
    let n_samples = data_array.shape()[0];
    let n_features = data_array.shape()[1];
    
    // Compute covariance matrix
    let cov = Python::allow_threads(py, || {
        let mut cov_matrix = Array2::<f32>::zeros((n_features, n_features));
        
        // Manually compute covariance for better control
        for i in 0..n_features {
            for j in 0..=i {  // Exploit symmetry
                let mut sum = 0.0;
                for k in 0..n_samples {
                    sum += data_array[[k, i]] * data_array[[k, j]];
                }
                let value = sum / (n_samples as f32);
                cov_matrix[[i, j]] = value;
                cov_matrix[[j, i]] = value;  // Fill in symmetric part
            }
        }
        
        cov_matrix
    });
    
    // Convert back to Python
    Ok(cov.into_pyarray(py).into())
}

/// Compute covariance matrix from data with parallel processing
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
    
    // Compute covariance matrix
    let cov = Python::allow_threads(py, || {
        // Create the result matrix
        let mut cov_matrix = Array2::<f32>::zeros((n_features, n_features));
        
        // Process features in parallel
        let feature_results: Vec<_> = (0..n_features)
            .into_par_iter()
            .map(|i| {
                // For each feature, calculate its covariance with all features up to i
                let mut row_output = vec![0.0f32; i+1];
                
                for j in 0..=i {
                    let mut sum = 0.0;
                    for k in 0..n_samples {
                        sum += data_array[[k, i]] * data_array[[k, j]];
                    }
                    row_output[j] = sum / (n_samples as f32);
                }
                
                (i, row_output)
            })
            .collect();
        
        // Combine results
        for (i, row_data) in feature_results {
            for j in 0..=i {
                cov_matrix[[i, j]] = row_data[j];
                cov_matrix[[j, i]] = row_data[j];  // Fill in symmetric part
            }
        }
        
        cov_matrix
    });
    
    // Convert back to Python
    Ok(cov.into_pyarray(py).into())
}

/// Compute the mean of the data along the first axis
#[pyfunction]
pub fn compute_mean(
    py: Python<'_>,
    data: PyReadonlyArray2<f32>,
) -> PyResult<Py<PyArray2<f32>>> {
    // Convert input to rust ndarray
    let data_array = data.as_array();
    
    // Get dimensions
    let n_samples = data_array.shape()[0];
    let n_features = data_array.shape()[1];
    
    // Compute mean
    let mean = Python::allow_threads(py, || {
        let mut mean_vec = Array2::<f32>::zeros((1, n_features));
        
        for j in 0..n_features {
            let mut sum = 0.0;
            for i in 0..n_samples {
                sum += data_array[[i, j]];
            }
            mean_vec[[0, j]] = sum / (n_samples as f32);
        }
        
        mean_vec
    });
    
    // Convert back to Python
    Ok(mean.into_pyarray(py).into())
}

/// Compute the mean of the data along the first axis with parallel processing
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
    
    // Compute mean
    let mean = Python::allow_threads(py, || {
        // Create the result vector
        let mut mean_vec = Array2::<f32>::zeros((1, n_features));
        
        // Process features in parallel
        let feature_results: Vec<_> = (0..n_features)
            .into_par_iter()
            .map(|j| {
                let mut sum = 0.0;
                for i in 0..n_samples {
                    sum += data_array[[i, j]];
                }
                (j, sum / (n_samples as f32))
            })
            .collect();
        
        // Combine results
        for (j, mean_val) in feature_results {
            mean_vec[[0, j]] = mean_val;
        }
        
        mean_vec
    });
    
    // Convert back to Python
    Ok(mean.into_pyarray(py).into())
}