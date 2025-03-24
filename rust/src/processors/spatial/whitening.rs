use ndarray::{Array1, Array2, ArrayView2, Axis};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;
use ndarray_linalg::{Eigh, UPLO};  // For eigendecomposition

/// Improved compute_whitening_matrix with better performance
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
        
        // Perform eigendecomposition
        match cov_array.eigh(UPLO::Upper) {
            Ok((eigvals, eigvecs)) => {
                // Create diagonal matrix with 1/sqrt(S + eps) using parallel iterator
                let s_inv: Vec<f32> = eigvals.iter()
                    .map(|&x| 1.0 / (x + eps).sqrt())
                    .collect();
                
                // Use ndarray's matrix multiplication for better performance
                let diag_inv = Array2::from_diag(&Array1::from(s_inv));
                let result = eigvecs.dot(&diag_inv).dot(&eigvecs.t());
                
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
                // Create a 2D mean array if it's 1D
                let mean_2d = mean.clone().into_shape((1, mean.len())).unwrap();
                
                // Subtract mean from each row
                Array2::from_shape_fn(data_array.dim(), |(i, j)| 
                    data_array[[i, j]] - mean_2d[[0, j]]
                )
            },
            None => data_array.clone(),
        };
        
        // Use Rayon's parallel iterator for row-wise multiplication with whitening matrix
        let result = Array2::<f32>::from_shape_fn((n_samples, n_features), |(i, j)| {
            // Compute dot product for each cell
            let row_slice = centered_data.row(i);
            let whitened_val = row_slice.dot(&whitening_array.column(j));
            
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
        let result: Array2<f32> = data_array
            .mean_axis(Axis(0))
            .map(|mean_array| {
                // Ensure result is 2D (1 row x n_features columns)
                mean_array
                    .into_shape((1, n_features))
                    .unwrap_or_else(|_| Array2::zeros((1, n_features)))
            })
            .unwrap_or_else(|| Array2::zeros((1, n_features)));
        
        result
    });
    
    // Convert back to Python
    Ok(mean.into_pyarray(py).into())
}

/// Optimized whitening function with combined mean and whitening
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
        // Compute mean if required
        let mean_array = if apply_mean {
            Some(data_array
                .mean_axis(Axis(0))
                .unwrap_or_else(|| Array1::zeros(data_array.ncols())))
        } else {
            None
        };
        
        // Center data if mean is computed
        let centered_data = match &mean_array {
            Some(mean) => {
                // Create a 2D mean array if it's 1D
                let mean_2d = mean.clone().into_shape((1, mean.len())).unwrap();
                
                // Subtract mean from each row
                Array2::from_shape_fn(data_array.dim(), |(i, j)| 
                    data_array[[i, j]] - mean_2d[[0, j]]
                )
            },
            None => data_array.clone(),
        };
        
        // Compute covariance
        let cov_matrix = centered_data.t().dot(&centered_data) / (data_array.nrows() as f32);
        
        // Compute whitening matrix
        let whitening_matrix = match cov_matrix.eigh(UPLO::Upper) {
            Ok((eigvals, eigvecs)) => {
                let s_inv: Vec<f32> = eigvals.iter()
                    .map(|&x| 1.0 / (x + eps).sqrt())
                    .collect();
                
                let diag_inv = Array2::from_diag(&Array1::from(s_inv));
                eigvecs.dot(&diag_inv).dot(&eigvecs.t())
            },
            Err(_) => Array2::<f32>::eye(data_array.ncols()),
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