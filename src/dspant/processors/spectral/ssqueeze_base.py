"""
Synchrosqueezing time-frequency analysis processor.

This module provides a processor for synchrosqueezing transforms using ssqueezepy,
which offers sharper time-frequency representations than standard spectrograms
by reassigning energy in the time-frequency plane.
"""

from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import dask.array as da
import numpy as np

try:
    import ssqueezepy as ssq
    from ssqueezepy import cwt, ssq_cwt, ssq_stft

    HAVE_SSQUEEZEPY = True
except ImportError:
    HAVE_SSQUEEZEPY = False

from ...engine.base import BaseProcessor


class SynchrosqueezingProcessor(BaseProcessor):
    """
    Processor for computing synchrosqueezed time-frequency representations.

    This processor implements synchrosqueezing transforms for sharper
    time-frequency analysis using the ssqueezepy package.
    """

    def __init__(
        self,
        transform_type: Literal["cwt", "stft"] = "cwt",
        wavelet: Union[str, Tuple[str, Dict[str, Any]]] = "gmw",
        scales: Union[str, np.ndarray] = "log-piecewise",
        nv: Optional[int] = None,
        window: Optional[Union[str, np.ndarray]] = None,
        n_fft: Optional[int] = None,
        hop_length: Optional[int] = None,
        squeezing: Union[bool, str] = True,
        preserve_transform: bool = False,
        gamma: Optional[float] = None,
        difftype: str = "trig",
        difforder: Optional[int] = None,
        padtype: str = "reflect",
        maprange: str = "peak",
        vectorized: bool = True,
        cache_wavelet: Optional[bool] = None,
        workers: int = 1,
    ):
        """
        Initialize the synchrosqueezing processor.

        Parameters
        ----------
        transform_type : str, default: "cwt"
            Type of transform to use: "cwt" (continuous wavelet) or "stft" (short-time Fourier)
        wavelet : str or tuple, default: "gmw"
            Wavelet to use for CWT. If tuple, should be (name, params).
            Choices: "gmw" (Generalized Morse Wavelet), "morlet", "bump", "cmhat", etc.
        scales : str or ndarray, default: "log-piecewise"
            Scale distribution. Options: "log", "log-piecewise", "linear", or custom array.
        nv : int, optional
            Number of voices. If None, defaults to 32 for the wavelet.
        window : str or ndarray, optional
            Window function for STFT. If string, uses predefined windows.
            Choices: "hann", "hamming", "blackman", etc.
        n_fft : int, optional
            FFT size for STFT
        hop_length : int, optional
            Hop length for STFT. If None, defaults to n_fft//4
        squeezing : bool or str, default: True
            Whether to perform the synchrosqueezing step. If string, specifies
            squeezing method: "sum" (default), "lebesgue".
        preserve_transform : bool, default: False
            Whether to preserve the original transform (CWT or STFT) in addition to Tx
        gamma : float, optional
            Threshold for synchrosqueezing. Areas where |dWx| < gamma will have Wx zeroed
        difftype : str, default: "trig"
            Method for computing frequency transform: "trig" (default), "diff", or "phase"
        difforder : int, optional
            Order of the finite difference scheme used if difftype="diff"
        padtype : str, default: "reflect"
            Signal padding method for boundary handling
        maprange : str, default: "peak"
            Method for mapping scales to frequencies
        vectorized : bool, default: True
            Whether to use vectorized computation
        cache_wavelet : bool, optional
            Whether to cache the wavelet for reuse
        workers : int, default: 1
            Number of workers for parallel processing

        Raises
        ------
        ImportError
            If ssqueezepy is not installed
        """
        if not HAVE_SSQUEEZEPY:
            raise ImportError(
                "ssqueezepy is required for this processor. Install it with: "
                "pip install ssqueezepy"
            )

        self.transform_type = transform_type
        self.wavelet = wavelet
        self.scales = scales
        self.nv = nv
        self.window = window
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.squeezing = (
            "sum"
            if squeezing is True
            else (squeezing if isinstance(squeezing, str) else False)
        )
        self.preserve_transform = preserve_transform
        self.gamma = gamma
        self.difftype = difftype
        self.difforder = difforder
        self.padtype = padtype
        self.maprange = maprange
        self.vectorized = vectorized
        self.cache_wavelet = cache_wavelet
        self.workers = workers

        # Set overlap samples based on transform type
        if transform_type == "stft":
            self._overlap_samples = n_fft if n_fft is not None else 1024
        else:  # CWT
            # For CWT, use a reasonable default overlap
            self._overlap_samples = 128

    def process(
        self, data: da.Array, fs: Optional[float] = None, **kwargs
    ) -> Union[da.Array, Dict[str, da.Array]]:
        """
        Process input data to compute synchrosqueezed time-frequency representation.

        Parameters
        ----------
        data : da.Array
            Input dask array (samples × channels)
        fs : float, optional
            Sampling frequency in Hz, required for correctly scaled frequencies
        **kwargs : dict
            Additional keyword arguments

        Returns
        -------
        outputs : Union[da.Array, Dict[str, da.Array]]
            Dictionary of computed outputs if preserve_transform is True,
            otherwise just the synchrosqueezed transform
        """
        if fs is None:
            raise ValueError("Sampling frequency (fs) must be provided")

        # Ensure input is 2D
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        elif data.ndim > 2:
            raise ValueError(f"Expected 1D or 2D input, got shape {data.shape}")

        # Process based on transform type
        if self.transform_type == "cwt":
            return self._process_cwt(data, fs, **kwargs)
        else:  # STFT
            return self._process_stft(data, fs, **kwargs)

    def _process_cwt(
        self, data: da.Array, fs: float, **kwargs
    ) -> Union[da.Array, Dict[str, da.Array]]:
        """
        Process using CWT-based synchrosqueezing.

        Parameters
        ----------
        data : da.Array
            Input data (samples × channels)
        fs : float
            Sampling frequency
        **kwargs : dict
            Additional keyword arguments

        Returns
        -------
        outputs : Union[da.Array, Dict[str, da.Array]]
            Processed transform(s)
        """

        # Function to apply to chunks
        def process_chunk(
            chunk: np.ndarray,
        ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
            # Ensure the input is a contiguous array and has correct memory layout
            chunk = np.ascontiguousarray(chunk)

            # Handle single-channel case by adding extra dimension if needed
            if chunk.ndim == 1:
                chunk = chunk.reshape(-1, 1)

            n_channels = chunk.shape[1]
            results = []

            for ch in range(n_channels):
                signal = chunk[:, ch]

                if self.squeezing:
                    # Apply synchrosqueezed CWT with appropriate parameters
                    # Create a clean kwargs dict for ssq_cwt
                    cwt_params = {
                        "wavelet": self.wavelet,
                        "fs": fs,
                        "scales": self.scales,
                        "padtype": self.padtype,
                        "difftype": self.difftype,
                        "squeezing": self.squeezing,
                        "maprange": self.maprange,
                        "gamma": self.gamma,
                        "preserve_transform": self.preserve_transform,
                        "vectorized": self.vectorized,
                    }

                    # Only add optional parameters if they are not None
                    if self.nv is not None:
                        cwt_params["nv"] = self.nv
                    if self.difforder is not None:
                        cwt_params["difforder"] = self.difforder
                    if self.cache_wavelet is not None:
                        cwt_params["cache_wavelet"] = self.cache_wavelet

                    # Apply synchrosqueezed CWT
                    Tx, _, ssq_freqs, *rest = ssq_cwt(signal, **cwt_params)

                    if self.preserve_transform:
                        Wx, scales = rest
                        result = {
                            "Tx": Tx,
                            "Wx": Wx,
                            "ssq_freqs": ssq_freqs,
                            "scales": scales,
                        }
                    else:
                        result = np.abs(Tx)
                else:
                    # Apply only CWT (no synchrosqueezing)
                    # Create a clean kwargs dict for cwt
                    cwt_params = {
                        "wavelet": self.wavelet,
                        "fs": fs,
                        "scales": self.scales,
                        "padtype": self.padtype,
                    }

                    # Only add optional parameters if they are not None
                    if self.nv is not None:
                        cwt_params["nv"] = self.nv
                    if self.cache_wavelet is not None:
                        cwt_params["cache_wavelet"] = self.cache_wavelet

                    # Apply CWT
                    Wx, scales = cwt(signal, **cwt_params)
                    result = np.abs(Wx)

                results.append(result)

            if self.preserve_transform:
                # Combine all channels
                combined_results = {
                    key: np.stack([r[key] for r in results], axis=-1)
                    for key in results[0].keys()
                    if key not in ["ssq_freqs", "scales"]
                }
                # Add frequency information (same for all channels)
                combined_results["ssq_freqs"] = results[0]["ssq_freqs"]
                combined_results["scales"] = results[0]["scales"]
                return combined_results
            else:
                # Stack results from all channels
                return np.stack(results, axis=-1)

        # Process the data using map_overlap instead of map_blocks
        # Get a sample output to determine the output shape
        sample_chunk = data[
            : min(1024 + self._overlap_samples, data.shape[0]), :
        ].compute()
        sample_output = process_chunk(sample_chunk)

        if isinstance(sample_output, dict):
            # Create a dictionary of results for each key
            results = {}
            for key, value in sample_output.items():
                if key not in ["ssq_freqs", "scales"]:
                    # Process data arrays
                    result = data.map_overlap(
                        lambda x: process_chunk(x)[key],
                        depth={-2: self._overlap_samples},
                        boundary="reflect",
                        dtype=value.dtype,
                        new_axis=list(range(data.ndim, value.ndim)),
                    )
                    results[key] = result
                else:
                    # Store frequency or scale information
                    results[key] = value
            return results
        else:
            # Process with a single output array
            result = data.map_overlap(
                process_chunk,
                depth={-2: self._overlap_samples},
                boundary="reflect",
                dtype=sample_output.dtype,
                new_axis=list(range(data.ndim, sample_output.ndim)),
            )
            return result

    def _process_stft(
        self, data: da.Array, fs: float, **kwargs
    ) -> Union[da.Array, Dict[str, da.Array]]:
        """
        Process using STFT-based synchrosqueezing.

        Parameters
        ----------
        data : da.Array
            Input data (samples × channels)
        fs : float
            Sampling frequency
        **kwargs : dict
            Additional keyword arguments

        Returns
        -------
        outputs : Union[da.Array, Dict[str, da.Array]]
            Processed transform(s)
        """
        # Extract STFT parameters
        n_fft = self.n_fft or 1024
        hop_length = self.hop_length or n_fft // 4
        window = self.window or "hann"

        # Function to apply to chunks
        def process_chunk(
            chunk: np.ndarray,
        ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
            # Ensure the input is a contiguous array and has correct memory layout
            chunk = np.ascontiguousarray(chunk)

            # Handle single-channel case by adding extra dimension if needed
            if chunk.ndim == 1:
                chunk = chunk.reshape(-1, 1)

            n_channels = chunk.shape[1]
            results = []

            for ch in range(n_channels):
                signal = chunk[:, ch]

                if self.squeezing:
                    # Apply synchrosqueezed STFT with appropriate parameters
                    # Create a clean kwargs dict for ssq_stft
                    stft_params = {
                        "fs": fs,
                        "n_fft": n_fft,
                        "hop_length": hop_length,
                        "window": window,
                        "padtype": self.padtype,
                        "difftype": self.difftype,
                        "squeezing": self.squeezing,
                        "gamma": self.gamma,
                        "preserve_transform": self.preserve_transform,
                    }

                    # Only add optional parameters if they are not None
                    if self.difforder is not None:
                        stft_params["difforder"] = self.difforder

                    # Apply synchrosqueezed STFT
                    Tx, _, ssq_freqs, *rest = ssq_stft(signal, **stft_params)

                    if self.preserve_transform:
                        Sx, stft_freqs = rest
                        result = {
                            "Tx": Tx,
                            "Sx": Sx,
                            "ssq_freqs": ssq_freqs,
                            "stft_freqs": stft_freqs,
                        }
                    else:
                        result = np.abs(Tx)
                else:
                    # Apply only STFT (no synchrosqueezing)
                    Sx, stft_freqs = ssq.stft(
                        signal,
                        fs=fs,
                        n_fft=n_fft,
                        hop_length=hop_length,
                        window=window,
                        padtype=self.padtype,
                    )
                    result = np.abs(Sx)

                results.append(result)

            if self.preserve_transform:
                # Combine all channels
                combined_results = {
                    key: np.stack([r[key] for r in results], axis=-1)
                    for key in results[0].keys()
                    if key not in ["ssq_freqs", "stft_freqs"]
                }
                # Add frequency information (same for all channels)
                combined_results["ssq_freqs"] = results[0]["ssq_freqs"]
                combined_results["stft_freqs"] = results[0]["stft_freqs"]
                return combined_results
            else:
                # Stack results from all channels
                return np.stack(results, axis=-1)

        # Use map_overlap for STFT to handle boundary effects
        # Determine output shape based on settings
        sample_chunk = data[
            : min(1024 + self._overlap_samples, data.shape[0]), :
        ].compute()
        sample_output = process_chunk(sample_chunk)

        if isinstance(sample_output, dict):
            # Create a dictionary of results for each key
            results = {}
            for key, value in sample_output.items():
                if key not in ["ssq_freqs", "stft_freqs"]:
                    # Process data arrays
                    result = data.map_overlap(
                        lambda x: process_chunk(x)[key],
                        depth={-2: self._overlap_samples},
                        boundary="reflect",
                        dtype=value.dtype,
                        new_axis=list(range(data.ndim, value.ndim)),
                    )
                    results[key] = result
                else:
                    # Store frequency information
                    results[key] = value
            return results
        else:
            # Process with a single output array
            result = data.map_overlap(
                process_chunk,
                depth={-2: self._overlap_samples},
                boundary="reflect",
                dtype=sample_output.dtype,
                new_axis=list(range(data.ndim, sample_output.ndim)),
            )
            return result

    @property
    def overlap_samples(self) -> int:
        """Return number of samples needed for overlap"""
        return self._overlap_samples

    @property
    def summary(self) -> Dict[str, Any]:
        """Get a summary of processor configuration"""
        base_summary = super().summary
        base_summary.update(
            {
                "transform_type": self.transform_type,
                "wavelet": str(self.wavelet),
                "nv": self.nv,
                "scales": str(self.scales)
                if isinstance(self.scales, str)
                else "custom",
                "squeezing": str(self.squeezing),
                "preserve_transform": self.preserve_transform,
                "difftype": self.difftype,
                "padtype": self.padtype,
                "maprange": self.maprange,
            }
        )

        if self.transform_type == "stft":
            base_summary.update(
                {
                    "n_fft": self.n_fft,
                    "hop_length": self.hop_length,
                    "window": str(self.window),
                }
            )

        return base_summary


# Factory functions for easy processor creation


def create_cwt_processor(
    wavelet: str = "gmw",
    nv: Optional[int] = None,
    scales: Union[str, np.ndarray] = "log-piecewise",
    squeezing: bool = False,
    workers: int = 1,
) -> SynchrosqueezingProcessor:
    """
    Create a CWT processor (without synchrosqueezing).

    Parameters
    ----------
    wavelet : str, default: "gmw"
        Wavelet to use. Options include: "gmw", "morlet", "bump", "cmhat"
    nv : int, optional
        Number of voices. If None, defaults to 32 for the wavelet.
    scales : str or ndarray, default: "log-piecewise"
        Scale distribution method or custom scales array
    squeezing : bool, default: False
        Whether to perform synchrosqueezing (False for standard CWT)
    workers : int, default: 1
        Number of workers for parallel processing

    Returns
    -------
    processor : SynchrosqueezingProcessor
        Configured CWT processor
    """
    return SynchrosqueezingProcessor(
        transform_type="cwt",
        wavelet=wavelet,
        nv=nv,
        scales=scales,
        squeezing=squeezing,
        preserve_transform=False,
        workers=workers,
    )


def create_ssq_cwt_processor(
    wavelet: str = "gmw",
    nv: Optional[int] = None,
    scales: Union[str, np.ndarray] = "log-piecewise",
    squeezing: Union[bool, str] = "sum",
    preserve_transform: bool = False,
    gamma: Optional[float] = None,
    difftype: str = "trig",
    workers: int = 1,
) -> SynchrosqueezingProcessor:
    """
    Create a synchrosqueezed CWT processor.

    Parameters
    ----------
    wavelet : str, default: "gmw"
        Wavelet to use. Options include: "gmw", "morlet", "bump", "cmhat"
    nv : int, optional
        Number of voices. If None, defaults to 32 for the wavelet.
    scales : str or ndarray, default: "log-piecewise"
        Scale distribution method or custom scales array
    squeezing : bool or str, default: "sum"
        Squeezing method: "sum" (default) or "lebesgue"
    preserve_transform : bool, default: False
        Whether to preserve the original CWT in addition to synchrosqueezed transform
    gamma : float, optional
        Threshold for synchrosqueezing. Areas where |dWx| < gamma will have Wx zeroed
    difftype : str, default: "trig"
        Method for computing frequency transform: "trig", "diff", or "phase"
    workers : int, default: 1
        Number of workers for parallel processing

    Returns
    -------
    processor : SynchrosqueezingProcessor
        Configured synchrosqueezed CWT processor
    """
    return SynchrosqueezingProcessor(
        transform_type="cwt",
        wavelet=wavelet,
        nv=nv,
        scales=scales,
        squeezing=squeezing,
        preserve_transform=preserve_transform,
        gamma=gamma,
        difftype=difftype,
        workers=workers,
    )


def create_ssq_stft_processor(
    n_fft: int = 1024,
    hop_length: Optional[int] = None,
    window: str = "hann",
    preserve_transform: bool = False,
    gamma: Optional[float] = None,
    workers: int = 1,
) -> SynchrosqueezingProcessor:
    """
    Create a synchrosqueezed STFT processor.

    Parameters
    ----------
    n_fft : int, default: 1024
        FFT size
    hop_length : int, optional
        Hop length. If None, defaults to n_fft//4
    window : str, default: "hann"
        Window function. Options include: "hann", "hamming", "blackman"
    preserve_transform : bool, default: False
        Whether to preserve the original STFT in addition to synchrosqueezed transform
    gamma : float, optional
        Threshold for synchrosqueezing. Areas where |dWx| < gamma will have Wx zeroed
    workers : int, default: 1
        Number of workers for parallel processing

    Returns
    -------
    processor : SynchrosqueezingProcessor
        Configured synchrosqueezed STFT processor
    """
    return SynchrosqueezingProcessor(
        transform_type="stft",
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        squeezing=True,
        preserve_transform=preserve_transform,
        gamma=gamma,
        workers=workers,
    )
