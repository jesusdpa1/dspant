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
        data_array: Union[np.ndarray, da.Array],
        fs: float,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the EpochExtractor.

        Parameters:
        -----------
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
        if self.data_array.ndim < 2:
            self.data_array = self.data_array.reshape(-1, 1)
        elif self.data_array.ndim > 2:
            raise ValueError("Data array must be 2-dimensional (samples × channels)")

    def extract_epochs(
        self,
        onsets: Union[np.ndarray, list, pl.Series, pl.DataFrame],
        pre_samples: int = 10,
        post_samples: int = 40,
        time_unit: Literal["seconds", "samples"] = "seconds",
        channel_selection: Optional[Union[int, List[int]]] = None,
        reject_outliers: bool = False,
        outlier_threshold: float = 3.0,
    ) -> da.Array:
        """
        Extract data epochs with equal-length windows.

        Parameters:
        -----------
        onsets : array-like
            Start times/indices of epochs
        pre_samples : int, optional
            Number of samples before the onset (default: 10)
        post_samples : int, optional
            Number of samples after the onset (default: 40)
        time_unit : {'seconds', 'samples'}, default 'seconds'
            Unit of onset specification
        channel_selection : int or list, optional
            Specific channel(s) to extract epochs from
        reject_outliers : bool, default False
            Whether to reject epochs with extreme values
        outlier_threshold : float, default 3.0
            Standard deviation threshold for outlier rejection

        Returns:
        --------
        da.Array
            Extracted epochs with shape (n_epochs, window_length, n_channels)
        """
        # Convert onsets to samples
        onset_samples = self._convert_time_to_samples(
            self._validate_time_input(onsets), self.fs, time_unit
        )

        # Channel selection
        if channel_selection is not None:
            if isinstance(channel_selection, int):
                channel_selection = [channel_selection]
            data = self.data_array[:, channel_selection]
        else:
            data = self.data_array

        # Total window length
        window_length = pre_samples + post_samples + 1

        # Initialize epoch list
        epochs = []

        # Extract epochs using direct slicing
        for onset in onset_samples:
            # Check bounds before extraction
            if onset - pre_samples < 0 or onset + post_samples + 1 > data.shape[0]:
                continue

            # Extract epoch using direct slicing
            epoch_data = data[onset - pre_samples : onset + post_samples + 1, :]

            # Reject outliers if requested
            if reject_outliers and self._is_outlier_epoch(
                epoch_data, outlier_threshold
            ):
                continue

            epochs.append(epoch_data)

        # Stack all valid epochs
        if epochs:
            return da.stack(epochs)
        else:
            # Return empty array if no valid epochs
            n_channels = data.shape[1]
            return da.empty((0, window_length, n_channels), dtype=data.dtype)

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

    def extract_variable_epochs(
        self,
        onsets: Union[np.ndarray, list, pl.Series, pl.DataFrame],
        offsets: Union[np.ndarray, list, pl.Series, pl.DataFrame],
        time_unit: Literal["seconds", "samples"] = "seconds",
        channel_selection: Optional[Union[int, List[int]]] = None,
        min_epoch_length: Optional[int] = None,
        max_epoch_length: Optional[int] = None,
        reject_outliers: bool = False,
        outlier_threshold: float = 3.0,
    ) -> List[da.Array]:
        """
        Extract epochs with variable lengths (for non-fixed window extraction).

        This method should be used when epochs have different durations.

        Returns:
        --------
        List[da.Array]
            List of extracted epochs with potentially different lengths
        """
        # Convert to samples
        onset_samples = self._convert_time_to_samples(
            self._validate_time_input(onsets), self.fs, time_unit
        )
        offset_samples = self._convert_time_to_samples(
            self._validate_time_input(offsets), self.fs, time_unit
        )

        # Ensure offsets don't exceed data length
        offset_samples = np.minimum(offset_samples, self.data_array.shape[0])

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

        # Channel selection
        if channel_selection is not None:
            if isinstance(channel_selection, int):
                channel_selection = [channel_selection]
            data = self.data_array[:, channel_selection]
        else:
            data = self.data_array

        # Extract epochs with variable lengths
        epochs = []
        for start, end in zip(onset_samples, offset_samples):
            epoch_data = data[start:end, :]

            # Reject outliers if requested
            if reject_outliers and self._is_outlier_epoch(
                epoch_data, outlier_threshold
            ):
                continue

            epochs.append(epoch_data)

        return epochs
