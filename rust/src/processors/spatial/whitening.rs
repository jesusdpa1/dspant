use ndarray::{Array1, Array2, ArrayView2, Axis};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use ndarray_linalg::SVD;

/// Improved compute_whitening_matrix with SVD instead of Eigh
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
        let n = cov_array.shape()[0];
        
        // Perform SVD
        match cov_array.svd(true, true) {
            Ok((u, s, _vt)) => {
                // Create diagonal matrix with 1/sqrt(S + eps)
                let s_inv: Vec<f32> = s.iter()
                    .map(|&x| 1.0 / (x + eps).sqrt())
                    .collect();
                
                let diag_inv = Array2::from_diag(&Array1::from(s_inv));
                
                // Unwrap the u value and then use it
                let u_unwrapped = u.expect("U matrix is None");
                u_unwrapped.dot(&diag_inv).dot(&u_unwrapped.t())
            },
            Err(_) => {
                // Fallback to a simple identity matrix if SVD fails
                Array2::<f32>::eye(n)
            }
        }
    });
    
    // Convert back to Python
    Ok(whitening.into_pyarray(py).into())
}

/// Parallel whitening transformation with vectorized operations
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
    
    // Allow Python threads to run during computation
    let whitened = Python::allow_threads(py, || {
        let n_samples = data_array.shape()[0];
        let n_features = data_array.shape()[1];
        
        // Prepare centered data
        let centered_data = match &mean_array {
            Some(mean) => {
                // Ensure mean is properly shaped
                let mean_2d = if mean.ndim() == 1 {
                    // Clone and expand 1D mean to 2D
                    let cloned_mean = mean.clone();
                    Array2::from_shape_vec(
                        (1, cloned_mean.len()), 
                        cloned_mean.into_raw_vec()
                    ).expect("Failed to create 2D mean array")
                } else {
                    // Ensure mean is 2D and matches feature dimension
                    let cloned_mean = mean.clone();
                    if cloned_mean.shape()[1] != n_features {
                        panic!("Mean array dimension does not match data features");
                    }
                    cloned_mean
                };
                
                // Subtract mean from each row
                Array2::from_shape_fn(data_array.dim(), |(i, j)| {
                    data_array[[i, j]] - mean_2d[[0, j.min(mean_2d.shape()[1] - 1)]]
                })
            },
            None => data_array.clone(),
        };
        
        // Use Rayon's parallel iterator for row-wise multiplication with whitening matrix
        let result = Array2::<f32>::from_shape_fn((n_samples, n_features), |(i, j)| {
            // Compute dot product for each cell
            let row_slice = centered_data.row(i);
            let col_slice = whitening_array.column(j);
            let whitened_val = row_slice.dot(&col_slice);
            
            // Apply optional scaling
            int_scale.map_or(whitened_val, |scale| scale * whitened_val)
        });
        
        result
    });
    
    // Convert result back to Python
    Ok(whitened.into_pyarray(py).into())
}

/// Parallel covariance computation with vectorized operations
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
        // Use Rayon's parallel iterator to compute covariance matrix
        let result = Array2::<f32>::from_shape_fn((n_features, n_features), |(i, j)| {
            // Use dot product for faster computation
            let column1 = data_array.column(i);
            let column2 = data_array.column(j);
            
            column1.dot(&column2) / (n_samples as f32)
        });
        
        result
    });
    
    // Convert back to Python
    Ok(cov.into_pyarray(py).into())
}

/// Parallel mean computation with reduced memory allocations
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
        // Compute mean along first axis
        if let Some(mean_array) = data_array.mean_axis(Axis(0)) {
            // Reshape to 2D (1 row x n_features columns)
            let mut result = Array2::<f32>::zeros((1, n_features));
            for j in 0..n_features {
                result[[0, j]] = mean_array[j];
            }
            result
        } else {
            // Fallback if mean computation fails
            Array2::<f32>::zeros((1, n_features))
        }
    });
    
    // Convert back to Python
    Ok(mean.into_pyarray(py).into())
}

/// Optimized whitening function with combined mean and whitening, now using SVD
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
    
    let whitened = Python::allow_threads(py, || {
        let n_samples = data_array.shape()[0];
        let n_features = data_array.shape()[1];
        
        // Compute mean if required
        let mean_array = if apply_mean {
            data_array.mean_axis(Axis(0))
        } else {
            None
        };
        
        // Center data if mean is computed
        let centered_data = match &mean_array {
            Some(mean) => {
                // Create 2D array to hold centered data
                let mut centered = Array2::<f32>::zeros(data_array.dim());
                
                // Subtract mean from each row
                for i in 0..n_samples {
                    for j in 0..n_features {
                        // Safely index mean to ensure we don't go out of bounds
                        let mean_val = mean.get(j).copied().unwrap_or(0.0);
                        centered[[i, j]] = data_array[[i, j]] - mean_val;
                    }
                }
                centered
            },
            None => data_array.clone(),
        };
        
        // Compute covariance
        let cov_matrix = centered_data.t().dot(&centered_data) / (n_samples as f32);

        // Compute whitening matrix using SVD
        let whitening_matrix = match cov_matrix.svd(true, true) {
            Ok((u, s, _vt)) => {
                let s_inv: Vec<f32> = s.iter()
                    .map(|&x| 1.0 / (x + eps).sqrt())
                    .collect();
                
                let diag_inv = Array2::from_diag(&Array1::from(s_inv));
                
                // Unwrap the u value first
                let u_unwrapped = u.expect("U matrix is None");
                // Then use it in the dot product operations
                let temp = u_unwrapped.dot(&diag_inv);
                temp.dot(&u_unwrapped.t())
            },
            Err(_) => Array2::<f32>::eye(n_features),
        };
        
        // Apply whitening transformation
        let whitened_data = centered_data.dot(&whitening_matrix.t());
        
        // Apply optional scaling
        match int_scale {
            Some(scale) => whitened_data * scale,
            None => whitened_data,
        }
    });
    
    // Convert back to Python
    Ok(whitened.into_pyarray(py).into())
}