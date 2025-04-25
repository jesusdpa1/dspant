# src/dspant_viz/utils/normalization.py
import numba as nb
import numpy as np

from dspant_viz.core.internals import public_api


@public_api(module_override="dspant_viz.utils")
@nb.jit(nopython=True, cache=True)
def zscore_normalization(data: np.ndarray) -> np.ndarray:
    """
    Perform z-score normalization on input data.

    Parameters
    ----------
    data : np.ndarray
        Input data array to normalize

    Returns
    -------
    np.ndarray
        Z-score normalized data
    """
    # Compute mean and standard deviation
    mean = np.mean(data)
    std = np.std(data)

    # Avoid division by zero
    if std == 0:
        return np.zeros_like(data)

    return (data - mean) / std


@public_api(module_override="dspant_viz.utils")
@nb.jit(nopython=True, cache=True)
def minmax_normalization(data: np.ndarray) -> np.ndarray:
    """
    Perform min-max normalization on input data.

    Parameters
    ----------
    data : np.ndarray
        Input data array to normalize

    Returns
    -------
    np.ndarray
        Min-max normalized data (scaled to [0, 1])
    """
    # Compute min and max
    data_min = np.min(data)
    data_max = np.max(data)

    # Avoid division by zero
    if data_min == data_max:
        return np.zeros_like(data)

    return (data - data_min) / (data_max - data_min)


@public_api(module_override="dspant_viz.utils")
def normalize_data(data: np.ndarray, method: str = None) -> np.ndarray:
    """
    Normalize input data using specified method.

    Parameters
    ----------
    data : np.ndarray
        Input data array to normalize
    method : str, optional
        Normalization method
        - None: No normalization
        - 'zscore': Z-score normalization
        - 'minmax': Min-max normalization

    Returns
    -------
    np.ndarray
        Normalized data
    """
    if method is None:
        return data

    if method == "zscore":
        return zscore_normalization(data)

    if method == "minmax":
        return minmax_normalization(data)

    raise ValueError(f"Unknown normalization method: {method}")
