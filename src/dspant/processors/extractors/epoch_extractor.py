import logging
from typing import List, Literal, Optional, Union

import dask.array as da
import numpy as np
import polars as pl

from dspant.core.internals import public_api

from .base_extractor import BaseExtractor


@public_api
class EpochExtractor(BaseExtractor):
    """
    Advanced epoch extraction processor with comprehensive features.

    Supports flexible epoch extraction from time series data with
    advanced preprocessing, filtering, and alignment capabilities.
    """

    def __init__(
        self,
        time_array: Union[np.ndarray, da.Array],
        data_array: Union[np.ndarray, da.Array],
        fs: float,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the EpochExtractor.

        Parameters:
        -----------
        time_array : array
            Time points corresponding to data
        data_array : array
            Time series data (samples × channels)
        fs : float
            Sampling frequency in Hz
        logger : logging.Logger, optional
            Logger for tracking extraction process
        """
        # Set up logging
        self.logger = logger or logging.getLogger(__name__)

        # Convert to dask arrays for consistency
        self.time_array = da.asarray(time_array)
        self.data_array = da.asarray(data_array)
        self.fs = fs

        # Validate input shapes
        self._validate_input_shapes()

    def _validate_input_shapes(self):
        """
        Validate the shapes of input arrays.

        Raises:
        -------
        ValueError
            If input arrays have incompatible shapes
        """
        if self.time_array.ndim != 1:
            raise ValueError("Time array must be 1-dimensional")

        if self.data_array.ndim < 2:
            self.data_array = self.data_array.reshape(-1, 1)
        elif self.data_array.ndim > 2:
            raise ValueError("Data array must be 2-dimensional (samples × channels)")

        if len(self.time_array) != self.data_array.shape[0]:
            raise ValueError(
                "Time array length must match first dimension of data array"
            )

    def extract_epochs(
        self,
        onsets: Union[np.ndarray, list, pl.Series, pl.DataFrame],
        offsets: Optional[Union[np.ndarray, list, pl.Series, pl.DataFrame]] = None,
        window_size: Optional[float] = None,
        time_unit: Literal["seconds", "samples"] = "seconds",
        padding_strategy: Literal["zero", "repeat", "nan", "reflect"] = "zero",
        channel_selection: Optional[Union[int, List[int]]] = None,
        min_epoch_length: Optional[int] = None,
        max_epoch_length: Optional[int] = None,
        reject_outliers: bool = False,
        outlier_threshold: float = 3.0,
    ) -> da.Array:
        """
        Extract and align data epochs with advanced filtering and preprocessing.

        Parameters:
        -----------
        onsets : array-like
            Start times/indices of epochs
        offsets : array-like, optional
            End times/indices of epochs
        window_size : float, optional
            Fixed window size if offsets not provided
        time_unit : {'seconds', 'samples'}, default 'seconds'
            Unit of onset/offset/window_size specification
        padding_strategy : {'zero', 'repeat', 'nan', 'reflect'}, default 'zero'
            Strategy for handling unequal epoch lengths
        channel_selection : int or list, optional
            Specific channel(s) to extract epochs from
        min_epoch_length : int, optional
            Minimum acceptable epoch length
        max_epoch_length : int, optional
            Maximum acceptable epoch length
        reject_outliers : bool, default False
            Whether to reject epochs with extreme values
        outlier_threshold : float, default 3.0
            Standard deviation threshold for outlier rejection

        Returns:
        --------
        da.Array
            Extracted and aligned epochs
        """
        # Validate and convert onset times to samples
        try:
            onset_samples = self._convert_time_to_samples(
                self._validate_time_input(onsets), self.fs, time_unit
            )
        except Exception as e:
            self.logger.error(f"Error converting onset times: {str(e)}")
            raise

        # Handle offsets
        try:
            if offsets is not None:
                offset_samples = self._convert_time_to_samples(
                    self._validate_time_input(offsets), self.fs, time_unit
                )
            elif window_size is not None:
                # Convert window size to samples
                window_samples = (
                    int(window_size * self.fs)
                    if time_unit == "seconds"
                    else int(window_size)
                )
                offset_samples = onset_samples + window_samples
            else:
                raise ValueError("Must provide either offsets or window_size")
        except Exception as e:
            self.logger.error(f"Error handling offsets: {str(e)}")
            raise

        # Ensure offsets don't exceed data length
        offset_samples = np.minimum(offset_samples, len(self.time_array))

        # Channel selection
        if channel_selection is not None:
            if isinstance(channel_selection, int):
                channel_selection = [channel_selection]
            data = self.data_array[:, channel_selection]
        else:
            data = self.data_array

        # Compute epoch lengths
        epoch_lengths = offset_samples - onset_samples

        # Filter epochs based on length constraints
        valid_mask = np.ones(len(onset_samples), dtype=bool)
        if min_epoch_length is not None:
            valid_mask &= epoch_lengths >= min_epoch_length
        if max_epoch_length is not None:
            valid_mask &= epoch_lengths <= max_epoch_length

        # Update onset and offset samples
        onset_samples = onset_samples[valid_mask]
        offset_samples = offset_samples[valid_mask]
        epoch_lengths = epoch_lengths[valid_mask]

        # Compute max epoch length for padding
        max_epoch_length = int(np.max(epoch_lengths))

        # Initialize output array based on padding strategy
        try:
            epochs = self._initialize_epochs(
                len(onset_samples),
                max_epoch_length,
                data.shape[1],
                padding_strategy,
                data.dtype,
            )
        except Exception as e:
            self.logger.error(f"Error initializing epochs: {str(e)}")
            raise

        # Fill epochs with data
        for i, (start, end) in enumerate(zip(onset_samples, offset_samples)):
            chunk_length = end - start

            try:
                epoch_data = data[start:end, :]

                # Reject outliers if requested
                if reject_outliers:
                    if self._is_outlier_epoch(epoch_data, outlier_threshold):
                        continue

                # Fill epochs based on padding strategy
                epochs = self._fill_epoch(
                    epochs,
                    epoch_data,
                    i,
                    chunk_length,
                    max_epoch_length,
                    padding_strategy,
                )
            except Exception as e:
                self.logger.warning(f"Error processing epoch {i}: {str(e)}")
                continue

        return epochs

    def _initialize_epochs(
        self,
        n_epochs: int,
        max_epoch_length: int,
        n_channels: int,
        padding_strategy: str,
        dtype: np.dtype,
    ) -> da.Array:
        """
        Initialize epochs array based on padding strategy.

        Parameters:
        -----------
        n_epochs : int
            Number of epochs
        max_epoch_length : int
            Maximum epoch length
        n_channels : int
            Number of channels
        padding_strategy : str
            Padding strategy
        dtype : np.dtype
            Data type of the array

        Returns:
        --------
        da.Array
            Initialized epochs array
        """
        if padding_strategy == "zero":
            return da.zeros((n_epochs, max_epoch_length, n_channels), dtype=dtype)
        elif padding_strategy == "repeat":
            return da.zeros((n_epochs, max_epoch_length, n_channels), dtype=dtype)
        elif padding_strategy == "nan":
            return da.full(
                (n_epochs, max_epoch_length, n_channels), np.nan, dtype=dtype
            )
        elif padding_strategy == "reflect":
            # For reflect, we'll use zeros initially and handle in filling
            return da.zeros((n_epochs, max_epoch_length, n_channels), dtype=dtype)
        else:
            raise ValueError(f"Unsupported padding strategy: {padding_strategy}")

    def _fill_epoch(
        self,
        epochs: da.Array,
        epoch_data: da.Array,
        epoch_index: int,
        chunk_length: int,
        max_epoch_length: int,
        padding_strategy: str,
    ) -> da.Array:
        """
        Fill an epoch in the epochs array based on padding strategy.

        Parameters:
        -----------
        epochs : da.Array
            Epochs array to fill
        epoch_data : da.Array
            Data for the current epoch
        epoch_index : int
            Index of the current epoch
        chunk_length : int
            Length of the current epoch
        max_epoch_length : int
            Maximum epoch length
        padding_strategy : str
            Padding strategy to use

        Returns:
        --------
        da.Array
            Updated epochs array
        """
        if padding_strategy == "zero":
            epochs[epoch_index, :chunk_length, :] = epoch_data
        elif padding_strategy == "repeat":
            epochs[epoch_index, :chunk_length, :] = epoch_data
            # Repeat last value for remaining
            if chunk_length < max_epoch_length:
                epochs[epoch_index, chunk_length:, :] = epoch_data[-1, :]
        elif padding_strategy == "nan":
            epochs[epoch_index, :chunk_length, :] = epoch_data
        elif padding_strategy == "reflect":
            # Pad with reflection
            if chunk_length < max_epoch_length:
                # Reflect the data to fill the remaining space
                reflection_length = max_epoch_length - chunk_length
                reflected_data = da.flip(epoch_data, axis=0)
                epochs[epoch_index, :chunk_length, :] = epoch_data
                epochs[
                    epoch_index, chunk_length : chunk_length + reflection_length // 2, :
                ] = reflected_data[: reflection_length // 2, :]
                if reflection_length % 2 != 0:
                    epochs[epoch_index, -1, :] = epoch_data[0, :]

        return epochs

    def _is_outlier_epoch(self, epoch_data: da.Array, threshold: float) -> bool:
        """
        Detect if an epoch contains outlier values.

        Parameters:
        -----------
        epoch_data : da.Array
            Epoch data to check
        threshold : float
            Standard deviation threshold for outlier detection

        Returns:
        --------
        bool
            True if the epoch is an outlier, False otherwise
        """
        # Compute mean and standard deviation
        mean = da.mean(epoch_data, axis=0)
        std = da.std(epoch_data, axis=0)

        # Check if any value exceeds the threshold
        outlier_mask = da.abs(epoch_data - mean) > (threshold * std)

        return da.any(outlier_mask).compute()
