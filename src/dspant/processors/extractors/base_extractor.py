from typing import Optional, Union

import dask.array as da
import numpy as np
import polars as pl

from dspant.core.internals import public_api


@public_api
class BaseExtractor:
    """
    Base class for data extraction processors in dspant.

    Provides common utility methods for data extraction and validation.
    """

    @staticmethod
    def _convert_time_to_samples(
        times: Union[np.ndarray, list], fs: float, time_unit: str = "seconds"
    ) -> np.ndarray:
        """
        Convert time values to sample indices.

        Parameters:
        -----------
        times : array-like
            Time values to convert
        fs : float
            Sampling frequency in Hz
        time_unit : str, optional
            Unit of input times ('seconds' or 'samples')

        Returns:
        --------
        np.ndarray
            Sample indices
        """
        if time_unit == "seconds":
            return np.round(np.asarray(times) * fs).astype(int)
        elif time_unit == "samples":
            return np.asarray(times).astype(int)
        else:
            raise ValueError("time_unit must be 'seconds' or 'samples'")

    @staticmethod
    def _validate_time_input(
        input_times: Union[np.ndarray, list, pl.Series, pl.DataFrame],
        column: Optional[str] = None,
    ) -> np.ndarray:
        """
        Validate and convert time input to numpy array.

        Parameters:
        -----------
        input_times : array-like or DataFrame
            Input time values
        column : str, optional
            Column name if input is a DataFrame

        Returns:
        --------
        np.ndarray
            Validated time values
        """
        if isinstance(input_times, pl.DataFrame):
            if column is None:
                raise ValueError("Column must be specified for DataFrame input")
            return input_times[column].to_numpy()
        elif isinstance(input_times, pl.Series):
            return input_times.to_numpy()
        else:
            return np.asarray(input_times)

    @staticmethod
    def _check_data_shape(
        data: Union[np.ndarray, da.Array], expected_dims: int = 2
    ) -> Union[np.ndarray, da.Array]:
        """
        Ensure data has the expected number of dimensions.

        Parameters:
        -----------
        data : array
            Input data array
        expected_dims : int, optional
            Expected number of dimensions

        Returns:
        --------
        array
            Validated data array
        """
        if data.ndim < expected_dims:
            return data.reshape(-1, 1)
        elif data.ndim > expected_dims:
            raise ValueError(f"Data should have {expected_dims} dimensions")
        return data
