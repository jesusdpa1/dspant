from typing import Dict, Literal, Optional, Tuple, Union

import dask.array as da
import numpy as np

from dspant.core.internals import public_api
from dspant.processors.basic import NormalizationProcessor


@public_api
class TemplateExtractor:
    """
    Specialized template extraction from multichannel waveform data.

    This class provides methods to extract templates (average waveforms) from
    electrophysiological data across multiple channels, with support for various
    normalization strategies and statistical analyses.

    The input data is expected to be of shape (n_waveforms, n_samples, n_channels).

    Attributes:
        None
    """

    @staticmethod
    def extract_template(
        waveforms: da.Array,
        normalization: Optional[
            Literal["zscore", "minmax", "robust", "mad", None]
        ] = None,
        axis: int = 0,
    ) -> np.ndarray:
        """
        Extract template from multichannel waveforms with optional normalization.

        Parameters:
        -----------
        waveforms : da.Array
            Extracted waveforms with shape (n_waveforms, n_samples, n_channels)
        normalization : str, optional
            Normalization method:
            - 'zscore': Zero mean, unit variance
            - 'minmax': Scale to [0, 1]
            - 'robust': Median and interquartile range
            - 'mad': Median absolute deviation
            - None: No normalization
        axis : int, default=0
            The axis along which to compute the template (usually the waveforms axis)

        Returns:
        --------
        template : np.ndarray
            Extracted and optionally normalized template with shape (n_samples, n_channels)
        """
        # Ensure input array is suitable for processing
        if waveforms.ndim < 3:
            raise ValueError(f"Expected 3D array, got shape {waveforms.shape}")

        # Compute mean template
        template = waveforms.mean(axis=axis).compute()

        # Apply normalization if specified
        if normalization:
            # For multichannel data, we need to normalize each channel separately
            normalized_template = np.zeros_like(template)

            # Iterate through channels
            for channel in range(template.shape[-1]):
                channel_data = template[..., channel]
                normalizer = NormalizationProcessor(method=normalization)
                # Reshape to 2D for normalization processor (expects samples x features)
                channel_data_reshaped = channel_data.reshape(-1, 1)
                normalized_channel = (
                    normalizer.process(channel_data_reshaped)
                    .compute()
                    .reshape(channel_data.shape)
                )
                normalized_template[..., channel] = normalized_channel

            return normalized_template

        return template

    @staticmethod
    def extract_template_distributions(
        waveforms: da.Array,
        normalization: Optional[
            Literal["zscore", "minmax", "robust", "mad", None]
        ] = None,
        axis: int = 0,
    ) -> Dict[str, Union[np.ndarray, da.Array]]:
        """
        Extract template with comprehensive statistical distributions across channels.

        Parameters:
        -----------
        waveforms : da.Array
            Extracted waveforms with shape (n_waveforms, n_samples, n_channels)
        normalization : str, optional
            Normalization method for the mean template
        axis : int, default=0
            The axis along which to compute statistics (usually the waveforms axis)

        Returns:
        --------
        Dict containing template statistics:
            - template_mean: Mean waveform (n_samples, n_channels)
            - template_std: Standard deviation (n_samples, n_channels)
            - template_median: Median waveform (n_samples, n_channels)
            - template_var: Variance (n_samples, n_channels)
            - template_min: Minimum values (n_samples, n_channels)
            - template_max: Maximum values (n_samples, n_channels)
            - template_q25: 25th percentile (n_samples, n_channels)
            - template_q75: 75th percentile (n_samples, n_channels)
            - waveforms: Original waveforms data
            - normalized_method: Normalization method used
            - n_waveforms: Number of waveforms
            - n_channels: Number of channels
        """
        # Ensure input array is suitable for processing
        if waveforms.ndim < 3:
            raise ValueError(f"Expected 3D array, got shape {waveforms.shape}")

        # Compute template statistics along the specified axis
        template_mean = waveforms.mean(axis=axis).compute()
        template_std = waveforms.std(axis=axis).compute()
        template_var = waveforms.var(axis=axis).compute()

        # Compute these statistics using numpy for efficiency
        waveforms_np = waveforms.compute()
        template_median = np.median(waveforms_np, axis=axis)
        template_min = np.min(waveforms_np, axis=axis)
        template_max = np.max(waveforms_np, axis=axis)
        template_q25 = np.percentile(waveforms_np, 25, axis=axis)
        template_q75 = np.percentile(waveforms_np, 75, axis=axis)

        # Apply normalization to mean template if specified
        normalized_mean = template_mean
        if normalization:
            normalized_mean = np.zeros_like(template_mean)

            # Normalize each channel separately
            for channel in range(template_mean.shape[-1]):
                channel_data = template_mean[..., channel]
                normalizer = NormalizationProcessor(method=normalization)
                # Reshape to 2D for normalization processor
                channel_data_reshaped = channel_data.reshape(-1, 1)
                normalized_channel = (
                    normalizer.process(channel_data_reshaped)
                    .compute()
                    .reshape(channel_data.shape)
                )
                normalized_mean[..., channel] = normalized_channel

        n_waveforms, n_samples, n_channels = waveforms.shape

        return {
            "template_mean": normalized_mean,
            "template_std": template_std,
            "template_median": template_median,
            "template_var": template_var,
            "template_min": template_min,
            "template_max": template_max,
            "template_q25": template_q25,
            "template_q75": template_q75,
            "waveforms": waveforms,
            "normalized_method": normalization,
            "n_waveforms": n_waveforms,
            "n_channels": n_channels,
        }

    @staticmethod
    def extract_channel_template(
        waveforms: da.Array,
        channel: int,
        normalization: Optional[
            Literal["zscore", "minmax", "robust", "mad", None]
        ] = None,
        axis: int = 0,
    ) -> np.ndarray:
        """
        Extract template for a specific channel from multichannel waveforms.

        Parameters:
        -----------
        waveforms : da.Array
            Extracted waveforms with shape (n_waveforms, n_samples, n_channels)
        channel : int
            Channel index to extract template for
        normalization : str, optional
            Normalization method
        axis : int, default=0
            The axis along which to compute the template

        Returns:
        --------
        template : np.ndarray
            Extracted and optionally normalized template for the specified channel
        """
        # Check if channel is valid
        n_channels = waveforms.shape[-1]
        if channel < 0 or channel >= n_channels:
            raise ValueError(
                f"Channel index {channel} out of range (0-{n_channels - 1})"
            )

        # Extract data for specified channel
        channel_data = waveforms[..., channel]

        # Compute mean template
        template = channel_data.mean(axis=axis).compute()

        # Apply normalization if specified
        if normalization:
            normalizer = NormalizationProcessor(method=normalization)
            # Reshape to 2D for normalization processor
            template_reshaped = template.reshape(-1, 1)
            template = (
                normalizer.process(template_reshaped).compute().reshape(template.shape)
            )

        return template


# Convenience functions
@public_api
def extract_template(
    waveforms: da.Array,
    normalization: Optional[Literal["zscore", "minmax", "robust", "mad", None]] = None,
    axis: int = 0,
) -> np.ndarray:
    """
    Convenience function for template extraction from multichannel data.

    Parameters:
    -----------
    waveforms : da.Array
        Extracted waveforms with shape (n_waveforms, n_samples, n_channels)
    normalization : str, optional
        Normalization method
    axis : int, default=0
        The axis along which to compute the template

    Returns:
    --------
    template : np.ndarray
        Extracted and optionally normalized template with shape (n_samples, n_channels)
    """
    return TemplateExtractor.extract_template(waveforms, normalization, axis)


@public_api
def extract_template_distributions(
    waveforms: da.Array,
    normalization: Optional[Literal["zscore", "minmax", "robust", "mad", None]] = None,
    axis: int = 0,
) -> Dict:
    """
    Convenience function for template distribution extraction from multichannel data.

    Parameters:
    -----------
    waveforms : da.Array
        Extracted waveforms with shape (n_waveforms, n_samples, n_channels)
    normalization : str, optional
        Normalization method
    axis : int, default=0
        The axis along which to compute statistics

    Returns:
    --------
    Dict containing template statistics
    """
    return TemplateExtractor.extract_template_distributions(
        waveforms, normalization, axis
    )


@public_api
def extract_channel_template(
    waveforms: da.Array,
    channel: int,
    normalization: Optional[Literal["zscore", "minmax", "robust", "mad", None]] = None,
    axis: int = 0,
) -> np.ndarray:
    """
    Convenience function for template extraction from a specific channel.

    Parameters:
    -----------
    waveforms : da.Array
        Extracted waveforms with shape (n_waveforms, n_samples, n_channels)
    channel : int
        Channel index to extract template for
    normalization : str, optional
        Normalization method
    axis : int, default=0
        The axis along which to compute the template

    Returns:
    --------
    template : np.ndarray
        Extracted and optionally normalized template for the specified channel
    """
    return TemplateExtractor.extract_channel_template(
        waveforms, channel, normalization, axis
    )
