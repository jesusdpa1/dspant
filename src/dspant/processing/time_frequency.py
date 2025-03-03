from typing import Any, Callable, Dict, Optional

import dask.array as da
import numpy as np
import torch
from torchaudio import functional, transforms

from ..core.nodes.stream_processing import BaseProcessor


class SpectrogramProcessor(BaseProcessor):
    """
    Spectrogram processor that handles various input array configurations
    and ensures compatibility with PyTorch tensor conversion
    """

    def __init__(
        self,
        n_fft: int = 400,
        win_length: Optional[int] = None,
        hop_length: Optional[int] = None,
        pad: int = 0,
        window_fn: Callable[[int], torch.Tensor] = torch.hann_window,
        power: Optional[float] = 2.0,
        normalized: bool = False,
        wkwargs: Optional[dict] = None,
        center: bool = True,
        pad_mode: str = "reflect",
        onesided: bool = True,
    ):
        self.spectrogram = transforms.Spectrogram(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            pad=pad,
            window_fn=window_fn,
            power=power,
            normalized=normalized,
            wkwargs=wkwargs or {},
            center=center,
            pad_mode=pad_mode,
            onesided=onesided,
        )

        self.n_fft = n_fft
        self.hop_length = hop_length if hop_length is not None else n_fft // 2

        if center:
            self._overlap_samples = n_fft
        else:
            self._overlap_samples = n_fft - self.hop_length

    def process(self, data: da.Array, fs: Optional[float] = None, **kwargs) -> da.Array:
        if fs is None:
            raise ValueError("Sampling frequency (fs) must be provided")

        # Ensure input is 2D
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        elif data.ndim > 2:
            raise ValueError(f"Expected 1D or 2D input, got shape {data.shape}")

        def process_chunk(x: np.ndarray) -> np.ndarray:
            # Ensure the input is a contiguous array and has correct memory layout
            x = np.ascontiguousarray(x)

            # Handle single-channel case by adding extra dimension if needed
            if x.ndim == 1:
                x = x.reshape(-1, 1)

            # Transpose to match PyTorch's expected input (channels, samples)
            x_torch = torch.from_numpy(x).float().T

            # Compute spectrogram
            spec = self.spectrogram(x_torch)

            # Move axes to match original expected output (freq_bins, time, channels)
            return np.moveaxis(spec.numpy(), 0, -1)

        # Use map_overlap with adjusted parameters
        result = data.map_overlap(
            process_chunk,
            depth={-2: self._overlap_samples},  # Using dict form like STFT
            boundary="reflect",
            dtype=np.float32,
            new_axis=-3,  # Add frequency bin dimension
        )

        return result

    @property
    def overlap_samples(self) -> int:
        return self._overlap_samples

    @property
    def summary(self) -> Dict[str, Any]:
        base_summary = super().summary
        base_summary.update(
            {
                "n_fft": self.n_fft,
                "hop_length": self.hop_length,
                "freq_bins": self.n_fft // 2 + 1,
            }
        )
        return base_summary


class LFCCProcessor(BaseProcessor):
    def __init__(
        self,
        n_filter: int = 128,
        n_lfcc: int = 40,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
        dct_type: int = 2,
        norm: str = "ortho",
        log_lf: bool = False,
        speckwargs: Optional[dict] = None,
    ):
        # Store parameters for spectrogram computation
        self.speckwargs = speckwargs or {
            "n_fft": 400,
            "hop_length": 200,
            "power": 2.0,
        }

        # Initialize overlap samples based on spectrogram parameters
        self.n_fft = self.speckwargs.get("n_fft", 400)
        self.hop_length = self.speckwargs.get("hop_length", self.n_fft // 2)

        if self.speckwargs.get("center", True):
            self._overlap_samples = self.n_fft
        else:
            self._overlap_samples = self.n_fft - self.hop_length

        # LFCC specific parameters
        self.n_filter = n_filter
        self.n_lfcc = n_lfcc
        self.f_min = f_min
        self.f_max = f_max
        self.dct_type = dct_type
        self.norm = norm
        self.log_lf = log_lf

        # LFCC transform will be initialized in process since it needs sampling rate
        self.lfcc = None
        self.freqs = None

    def get_frequencies(self) -> Optional[np.ndarray]:
        """Return the center frequencies of the linear filterbank"""
        if self.lfcc is None:
            return None
        # The filterbank is linear, so we can compute the frequencies directly
        f_max = self.f_max if self.f_max is not None else self.lfcc.sample_rate / 2
        return np.linspace(self.f_min, f_max, self.n_filter)

    def process(self, data: da.Array, fs: Optional[float] = None, **kwargs) -> da.Array:
        if fs is None:
            raise ValueError("Sampling frequency (fs) must be provided")

        # Ensure input is 2D
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        elif data.ndim > 2:
            raise ValueError(f"Expected 1D or 2D input, got shape {data.shape}")

        # Initialize LFCC transform with the sampling rate
        self.lfcc = transforms.LFCC(
            sample_rate=int(fs),  # LFCC requires integer sample rate
            n_filter=self.n_filter,
            n_lfcc=self.n_lfcc,
            f_min=self.f_min,
            f_max=self.f_max if self.f_max is not None else fs / 2,
            dct_type=self.dct_type,
            norm=self.norm,
            log_lf=self.log_lf,
            speckwargs=self.speckwargs,
        )

        # Store the frequencies once LFCC is initialized
        self.freqs = self.get_frequencies()

        def process_chunk(x: np.ndarray) -> np.ndarray:
            # Ensure the input is a contiguous array and has correct memory layout
            x = np.ascontiguousarray(x)

            # Handle single-channel case by adding extra dimension if needed
            if x.ndim == 1:
                x = x.reshape(-1, 1)

            # Transpose to match PyTorch's expected input (channels, samples)
            x_torch = torch.from_numpy(x).float().T

            # Compute LFCC features
            lfcc_features = self.lfcc(x_torch)

            # Move axes to match original expected output (lfcc_bins, time, channels)
            lfcc_np = np.moveaxis(lfcc_features.numpy(), 0, -1)

            # Ensure 3D output even for single channel
            if lfcc_np.ndim == 2:
                lfcc_np = lfcc_np[..., np.newaxis]

            return lfcc_np

        result = data.map_overlap(
            process_chunk,
            depth={-2: self._overlap_samples},
            boundary="reflect",
            dtype=np.float32,
            new_axis=-3,
        )

        if self.freqs is not None:
            result.attrs = {"frequencies": self.freqs}

        return result

    @property
    def overlap_samples(self) -> int:
        return self._overlap_samples

    @property
    def summary(self) -> Dict[str, Any]:
        base_summary = super().summary
        base_summary.update(
            {
                "n_filter": self.n_filter,
                "n_lfcc": self.n_lfcc,
                "f_min": self.f_min,
                "f_max": self.f_max,
                "dct_type": self.dct_type,
                "log_lf": self.log_lf,
            }
        )
        return base_summary


class MFCCProcessor(BaseProcessor):
    def __init__(
        self,
        n_mfcc: int = 40,
        dct_type: int = 2,
        norm: str = "ortho",
        log_mels: bool = False,
        melkwargs: Optional[dict] = None,
    ):
        # Store mel spectrogram parameters
        self.melkwargs = melkwargs or {
            "n_fft": 400,
            "hop_length": 200,
            "n_mels": 128,
            "power": 2.0,
        }

        # Initialize overlap samples based on mel spectrogram parameters
        self.n_fft = self.melkwargs.get("n_fft", 400)
        self.hop_length = self.melkwargs.get("hop_length", self.n_fft // 2)

        if self.melkwargs.get("center", True):
            self._overlap_samples = self.n_fft
        else:
            self._overlap_samples = self.n_fft - self.hop_length

        # MFCC specific parameters
        self.n_mfcc = n_mfcc
        self.dct_type = dct_type
        self.norm = norm
        self.log_mels = log_mels

        # MFCC transform will be initialized in process since it needs sampling rate
        self.mfcc = None
        self.mel_fbanks = None

    def get_mel_filterbanks(self, fs) -> Optional[torch.Tensor]:
        """Get mel filterbank matrix"""
        if not fs:
            return None

        n_mels = self.melkwargs.get("n_mels", 128)
        f_min = self.melkwargs.get("f_min", 0.0)

        # Ensure f_max is not None by using fs/2 as default
        f_max = self.melkwargs.get("f_max") or (fs / 2)

        # Number of FFT bins
        n_freqs = self.n_fft // 2 + 1

        return functional.melscale_fbanks(
            n_freqs=n_freqs,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            sample_rate=fs,
            norm=self.melkwargs.get("norm", None),
            mel_scale=self.melkwargs.get("mel_scale", "htk"),
        )

    def process(self, data: da.Array, fs: Optional[float] = None, **kwargs) -> da.Array:
        if fs is None:
            raise ValueError("Sampling frequency (fs) must be provided")

        # Ensure input is 2D
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        elif data.ndim > 2:
            raise ValueError(f"Expected 1D or 2D input, got shape {data.shape}")

        # Initialize MFCC transform with the sampling rate
        self.mfcc = transforms.MFCC(
            sample_rate=int(fs),  # MFCC requires integer sample rate
            n_mfcc=self.n_mfcc,
            dct_type=self.dct_type,
            norm=self.norm,
            log_mels=self.log_mels,
            melkwargs=self.melkwargs,
        )

        # Get mel filterbanks
        self.mel_fbanks = self.get_mel_filterbanks(fs)

        def process_chunk(x: np.ndarray) -> np.ndarray:
            # Ensure the input is a contiguous array and has correct memory layout
            x = np.ascontiguousarray(x)

            # Handle single-channel case by adding extra dimension if needed
            if x.ndim == 1:
                x = x.reshape(-1, 1)

            # Transpose to match PyTorch's expected input (channels, samples)
            x_torch = torch.from_numpy(x).float().T

            # Compute MFCC features
            mfcc_features = self.mfcc(x_torch)

            # Move axes to match original expected output (mfcc_bins, time, channels)
            mfcc_np = np.moveaxis(mfcc_features.numpy(), 0, -1)

            # Ensure 3D output even for single channel
            if mfcc_np.ndim == 2:
                mfcc_np = mfcc_np[..., np.newaxis]

            return mfcc_np

        result = data.map_overlap(
            process_chunk,
            depth={-2: self._overlap_samples},
            boundary="reflect",
            dtype=np.float32,
            new_axis=-3,
        )

        # Store filterbank information as attributes
        if self.mel_fbanks is not None:
            mel_fbanks_np = self.mel_fbanks.numpy()
            result.attrs = {
                "mel_filterbanks": mel_fbanks_np,
                "freq_bins": np.linspace(0, fs / 2, mel_fbanks_np.shape[1]),
                "n_mels": mel_fbanks_np.shape[0],
            }

        return result

    @property
    def overlap_samples(self) -> int:
        return self._overlap_samples

    @property
    def summary(self) -> Dict[str, Any]:
        base_summary = super().summary
        base_summary.update(
            {
                "n_mfcc": self.n_mfcc,
                "dct_type": self.dct_type,
                "log_mels": self.log_mels,
                "n_mels": self.melkwargs.get("n_mels", 128),
            }
        )
        return base_summary
