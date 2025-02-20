class StftProcessor(BaseProcessor):
    """STFT processor implementation with flexible multi-channel support."""

    def __init__(
        self,
        n_fft: int = 256,
        hop_length: Optional[int] = None,
        window: str = "hann",
        center: bool = True,
        normalized: bool = True,
        power: Optional[float] = 1.0,
        channel_axis: int = -1,
    ):
        """
        Initialize STFT processor.

        Args:
            n_fft: Length of FFT window
            hop_length: Number of samples between successive frames
            window: Window type ('hann', 'hamming', 'blackman', 'bartlett', or 'none')
            center: Whether to pad signal on both sides
            normalized: Whether to normalize the STFT
            power: If None, returns complex STFT; if 1.0, returns magnitude; if 2.0, returns power
            channel_axis: Axis for channels in input array (-1 for last axis)
        """
        self.n_fft = n_fft
        self.hop_length = hop_length if hop_length is not None else n_fft // 4
        self.window = window.lower()
        self.center = center
        self.normalized = normalized
        self.power = power
        self.channel_axis = channel_axis

        # Pre-create window tensor
        self._window_tensor = self._create_window()

        # Overlap size depends on whether centering is used
        self._overlap_samples = n_fft if center else n_fft - self.hop_length

    def _create_window(self) -> torch.Tensor:
        """Create the window tensor based on the specified window type."""
        if self.window == "none":
            return torch.ones(self.n_fft)
        elif self.window == "hann":
            return torch.hann_window(self.n_fft)
        elif self.window == "hamming":
            return torch.hamming_window(self.n_fft)
        elif self.window == "blackman":
            return torch.blackman_window(self.n_fft)
        elif self.window == "bartlett":
            return torch.bartlett_window(self.n_fft)
        else:
            raise ValueError(f"Unsupported window type: {self.window}")

    def _compute_output_shape(self, n_samples: int) -> Tuple[int, int]:
        """
        Compute the output shape for STFT transform.

        Args:
            n_samples: Number of samples in input

        Returns:
            Tuple of (freq_bins, time_frames)
        """
        freq_bins = self.n_fft // 2 + 1

        if self.center:
            n_frames = (
                1 + (n_samples + 2 * (self.n_fft // 2) - self.n_fft) // self.hop_length
            )
        else:
            n_frames = 1 + (n_samples - self.n_fft) // self.hop_length

        return freq_bins, n_frames

    def _stft_block(self, data: torch.Tensor) -> torch.Tensor:
        """
        Compute STFT for a block of data.

        Args:
            data: Input tensor of shape (..., time_samples, channels)

        Returns:
            STFT tensor of shape (..., freq_bins, time_frames, channels)
        """
        # Process each channel
        stft_results = []
        for ch in range(data.shape[-1]):
            channel_data = data[..., ch]

            # Compute STFT
            stft = torch.stft(
                channel_data,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window=self._window_tensor,
                center=self.center,
                normalized=self.normalized,
                return_complex=True,
                onesided=True,
            )

            # Apply power transform if needed
            if self.power is not None:
                stft = torch.abs(stft)
                if self.power != 1.0:
                    stft = stft.pow(self.power)

            stft_results.append(stft)

        # Stack results along new channel dimension
        result = torch.stack(stft_results, dim=-1)

        return result

    def process(self, data: da.Array, fs: Optional[float] = None, **kwargs) -> da.Array:
        """
        Compute STFT across all channels.

        Args:
            data: Input data array of shape (..., time_samples, channels)
            fs: Sampling frequency (optional)
            **kwargs: Additional keyword arguments

        Returns:
            STFT array of shape (..., freq_bins, time_frames, channels)
        """
        # Validate input
        if data.ndim < 2:
            raise ValueError(f"Expected at least 2D input, got shape {data.shape}")

        # Move channel axis to end if needed
        if self.channel_axis != -1:
            data = da.moveaxis(data, self.channel_axis, -1)

        # Get important dimensions
        n_samples = data.shape[-2]
        n_channels = data.shape[-1]
        freq_bins, n_frames = self._compute_output_shape(n_samples)

        # Calculate optimal chunk size
        optimal_chunk = 4 * self.n_fft
        chunks = {-2: optimal_chunk, -1: -1}  # Chunk time axis, preserve channels
        data = data.rechunk(chunks)

        def process_chunk(x: np.ndarray) -> np.ndarray:
            # Convert to torch tensor
            x_torch = torch.from_numpy(x)

            # Process
            result = self._stft_block(x_torch)

            # Convert back to numpy
            return result.numpy()

        # Process with overlap
        result = data.map_overlap(
            process_chunk,
            depth={-2: self._overlap_samples},
            boundary="reflect",
            dtype=np.complex64 if self.power is None else np.float32,
            new_axis=-3,  # Add new axis for frequency bins
        )

        # Restore original channel axis position if needed
        if self.channel_axis != -1:
            result = da.moveaxis(result, -1, self.channel_axis)

        return result

    @property
    def overlap_samples(self) -> int:
        """Return the number of samples needed for overlap."""
        return self._overlap_samples

    def get_frequency_axis(self, fs: Optional[float] = None) -> np.ndarray:
        """Get the frequency axis for the STFT output."""
        if fs is None:
            return np.arange(self.n_fft // 2 + 1)
        return np.linspace(0, fs / 2, self.n_fft // 2 + 1)

    def get_time_axis(self, n_samples: int, fs: Optional[float] = None) -> np.ndarray:
        """Get the time axis for the STFT output."""
        _, n_frames = self._compute_output_shape(n_samples)
        if fs is None:
            return np.arange(n_frames)
        return np.arange(n_frames) * self.hop_length / fs


class FilterProcessor(BaseProcessor):
    """Filter processor implementation"""

    def __init__(self, filter_func: ProcessingFunction, overlap_samples: int):
        self.filter_func = filter_func
        self._overlap_samples = overlap_samples

    def process(self, data: da.Array, fs: Optional[float] = None, **kwargs) -> da.Array:
        return data.map_overlap(
            self.filter_func,
            depth=(self.overlap_samples, 0),
            boundary="reflect",
            fs=fs,
            dtype=data.dtype,
            **kwargs,
        )

    @property
    def overlap_samples(self) -> int:
        return self._overlap_samples


# this is working


class SpectrogramProcessor(BaseProcessor):
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

    def process(
        self, data: da.Array, fs: Optional[float] = None, **kwargs
    ) -> np.ndarray:
        print("Input shape before:", data.shape)
        print("Input chunks before:", data.chunks)  # Add this line

        if fs is None:
            raise ValueError("Sampling frequency (fs) must be provided")

        if data.ndim != 2:
            raise ValueError(f"Expected 2D input, got shape {data.shape}")

        def process_chunk(x: np.ndarray) -> np.ndarray:
            x_torch = torch.from_numpy(x).float().T
            spec = self.spectrogram(x_torch)
            return np.moveaxis(spec.numpy(), 0, -1)

        result = data.map_overlap(
            process_chunk,
            depth={-2: self._overlap_samples},  # Using dict form like STFT
            boundary="reflect",
            dtype=np.float32,
            new_axis=-3,
        )

        print("Result chunks:", result.chunks)  # Add this line
        return result

    @property
    def overlap_samples(self) -> int:
        return self._overlap_samples
