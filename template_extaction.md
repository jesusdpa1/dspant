```py
from typing import Dict, Literal, Optional, Union

import dask.array as da
import numpy as np

from dspant.core.internals import public_api
from dspant.processors.basic import NormalizationProcessor


@public_api
class TemplateExtractor:
    """
    Specialized template extraction from waveform data

    Supports multiple normalization strategies and statistical analyses
    """

    @staticmethod
    def extract_template(
        waveforms: da.Array,
        normalization: Optional[
            Literal["zscore", "minmax", "robust", "mad", None]
        ] = None,
    ) -> np.ndarray:
        """
        Extract template from waveforms with optional normalization

        Parameters:
        -----------
        waveforms : da.Array
            Extracted waveforms
        normalization : str, optional
            Normalization method:
            - 'zscore': Zero mean, unit variance
            - 'minmax': Scale to [0, 1]
            - 'robust': Median and interquartile range
            - 'mad': Median absolute deviation
            - None: No normalization

        Returns:
        --------
        template : np.ndarray
            Extracted and optionally normalized template
        """
        # Compute mean template
        template = np.mean(waveforms, axis=0)

        # Apply normalization if specified
        if normalization:
            normalizer = NormalizationProcessor(method=normalization)
            template = normalizer.process(template.T).T.compute()

        return template

    @staticmethod
    def extract_template_distributions(
        waveforms: da.Array,
        normalization: Optional[
            Literal["zscore", "minmax", "robust", "mad", None]
        ] = None,
    ) -> Dict[str, Union[np.ndarray, da.Array]]:
        """
        Extract template with comprehensive statistical distributions

        Parameters:
        -----------
        waveforms : da.Array
            Extracted waveforms
        normalization : str, optional
            Normalization method

        Returns:
        --------
        Dict containing template statistics
        """
        # Compute template statistics
        template_mean = np.mean(waveforms, axis=0)
        template_std = np.std(waveforms, axis=0)
        template_median = np.median(waveforms, axis=0)
        template_var = np.var(waveforms, axis=0)

        # Optional normalization
        if normalization:
            normalizer = NormalizationProcessor(method=normalization)
            normalized_mean = normalizer.process(template_mean.T).T.compute()
        else:
            normalized_mean = template_mean

        return {
            "template_mean": normalized_mean,
            "template_std": template_std,
            "template_median": template_median,
            "template_var": template_var,
            "waveforms": waveforms,
            "normalized_method": normalization,
        }


# Convenience functions
def extract_template(
    waveforms: da.Array,
    normalization: Optional[Literal["zscore", "minmax", "robust", "mad", None]] = None,
) -> np.ndarray:
    """
    Convenience function for template extraction
    """
    return TemplateExtractor.extract_template(waveforms, normalization)


def extract_template_distributions(
    waveforms: da.Array,
    normalization: Optional[Literal["zscore", "minmax", "robust", "mad", None]] = None,
) -> Dict:
    """
    Convenience function for template distribution extraction
    """
    return TemplateExtractor.extract_template_distributions(waveforms, normalization)



```