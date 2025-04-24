from typing import Optional, Union, List
import numpy as np
import dask.array as da
from pydantic import BaseModel, ConfigDict, field_validator

class BasePlotModel(BaseModel):
    """
    Base class for visualization data models.
    Enforces 2D array shape of [samples, channels]
    """
    # Use model_config instead of Config in Pydantic v2
    model_config = ConfigDict(arbitrary_types_allowed=True)

    data: Union[np.ndarray, da.Array]
    channel: Optional[int] = None
    fs: Optional[float] = None

    @field_validator('data')
    @classmethod
    def validate_data_shape(cls, data):
        """
        Ensure data is 2D with [samples, channels] shape
        """
        if data.ndim == 1:
            # Reshape 1D to 2D
            data = data.reshape(-1, 1)

        if data.ndim != 2:
            raise ValueError(f"Data must be 2D. Current shape: {data.shape}")

        return data

    def get_channel_data(self, channel: Optional[int] = None) -> Union[np.ndarray, da.Array]:
        """
        Extract data for a specific channel

        Args:
            channel: Channel index. If None, uses self.channel

        Returns:
            Array of channel data
        """
        if channel is None:
            channel = self.channel

        if channel is None:
            raise ValueError("No channel specified")

        if channel < 0 or channel >= self.data.shape[1]:
            raise ValueError(f"Invalid channel index. Must be between 0 and {self.data.shape[1] - 1}")

        return self.data[:, channel]

    def get_time_array(self) -> Union[np.ndarray, da.Array]:
        """
        Generate time array based on sampling frequency

        Returns:
            Time array in seconds
        """
        if self.fs is None:
            return np.arange(self.data.shape[0])

        return np.arange(self.data.shape[0]) / self.fs
