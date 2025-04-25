# dspant_viz API Reference

## Core Module

### Base Classes

#### `VisualizationComponent`

Abstract base class for all visualization components.

```python
class VisualizationComponent(ABC):
    def __init__(self, data, **kwargs):
        """
        Initialize the visualization component.
        
        Parameters
        ----------
        data : Any
            Input data for visualization
        **kwargs : dict
            Additional configuration parameters
        """
        
    @abstractmethod
    def plot(self, backend: str = "mpl", **kwargs):
        """
        Generate a plot using the specified backend.
        
        Parameters
        ----------
        backend : str, optional
            Backend to use for plotting ('mpl', 'plotly')
        **kwargs : dict
            Additional plot-specific parameters
            
        Returns
        -------
        Any
            Plot figure from the specified backend
        """
        
    @abstractmethod
    def get_data(self) -> Dict:
        """
        Prepare data for rendering.
        
        Returns
        -------
        dict
            Data and parameters for rendering
        """
        
    @abstractmethod
    def update(self, **kwargs) -> None:
        """
        Update component parameters.
        
        Parameters
        ----------
        **kwargs : dict
            Parameters to update
        """
```

#### `CompositeVisualization`

Abstract base class for composite visualizations that combine multiple components.

```python
class CompositeVisualization(ABC):
    def __init__(self, components: List[VisualizationComponent], **kwargs):
        """
        Initialize the composite visualization.
        
        Parameters
        ----------
        components : List[VisualizationComponent]
            List of visualization components to combine
        **kwargs : dict
            Additional configuration parameters
        """
        
    @abstractmethod
    def plot(self, backend: str = "mpl", **kwargs):
        """
        Generate a composite plot using the specified backend.
        
        Parameters
        ----------
        backend : str, optional
            Backend to use for plotting ('mpl', 'plotly')
        **kwargs : dict
            Additional plot-specific parameters
            
        Returns
        -------
        Any
            Composite figure from the specified backend
        """
        
    @abstractmethod
    def get_data(self) -> Dict:
        """
        Prepare data from all components for rendering.
        
        Returns
        -------
        dict
            Combined data and parameters for rendering
        """
        
    @abstractmethod
    def update(self, **kwargs) -> None:
        """
        Update component parameters.
        
        Parameters
        ----------
        **kwargs : dict
            Parameters to update
        """
```

### Data Models

#### `SpikeData`

Data model for spike times organized by units.

```python
class SpikeData(BaseModel):
    """Data model for spike times organized by units"""
    
    spikes: Dict[int, np.ndarray] = Field(
        ..., description="Dictionary mapping unit IDs to spike times"
    )
    unit_labels: Optional[Dict[int, str]] = Field(
        None, description="Optional custom labels for units"
    )
    
    def get_unit_ids(self) -> List[int]:
        """Get available unit IDs"""
        
    def get_unit_spikes(self, unit_id: int) -> np.ndarray:
        """
        Get spike times for a specific unit
        
        Parameters
        ----------
        unit_id : int
            ID of the unit to retrieve
            
        Returns
        -------
        np.ndarray
            Array of spike times for the unit
        """
```

#### `PSTHData`

Data model for PSTH results.

```python
class PSTHData(BaseModel):
    """Data model for PSTH results"""
    
    time_bins: List[float] = Field(..., description="Time bin centers in seconds")
    firing_rates: List[float] = Field(..., description="Firing rates in Hz")
    sem: Optional[List[float]] = None
    unit_id: Optional[int] = None
    baseline_window: Optional[Tuple[float, float]] = None
```

#### `TimeSeriesData`

Data model for time series.

```python
class TimeSeriesData(BaseModel):
    """Data model for time series"""
    
    times: List[float] = Field(..., description="Time points in seconds")
    values: List[float] = Field(..., description="Signal values")
    sampling_rate: Optional[float] = None
    channel_id: Optional[Union[int, str]] = None
    channel_name: Optional[str] = None
```

#### `MultiChannelData`

Data model for multi-channel time series.

```python
class MultiChannelData(BaseModel):
    """Data model for multi-channel time series"""
    
    times: List[float] = Field(..., description="Time points in seconds")
    channels: Dict[Union[int, str], List[float]] = Field(
        ..., description="Dictionary mapping channel IDs to signal values"
    )
    sampling_rate: Optional[float] = None
```

### Theme Manager

#### `ThemeManager`

Manager for visualization themes across backends.

```python
class ThemeManager:
    """Comprehensive theme management for visualization across backends"""
    
    def __init__(
        self,
        mpl_theme: str = "seaborn-darkgrid",
        plotly_theme: str = "seaborn",
        custom_themes_dir: Optional[Path] = None,
    ):
        """
        Initialize ThemeManager with theme preferences.
        
        Args:
            mpl_theme: Matplotlib theme (default: 'seaborn-darkgrid')
            plotly_theme: Plotly theme (default: 'seaborn')
            custom_themes_dir: Optional directory for custom TOML themes
        """
        
    def apply_matplotlib_theme(self, theme: Optional[str] = None):
        """
        Apply theme to Matplotlib.
        
        Args:
            theme: Theme name. Uses default if not specified.
        """
        
    def apply_plotly_theme(self, theme: Optional[str] = None):
        """
        Apply theme to Plotly.
        
        Args:
            theme: Theme name. Uses default if not specified.
        """
        
    def load_custom_theme(self, theme_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load a custom theme from a TOML file.
        
        Args:
            theme_path: Path to the theme TOML file
            
        Returns:
            Parsed theme configuration
        """
        
    def create_custom_theme(self, theme_name: str, theme_config: Dict[str, Any]):
        """
        Create and save a custom theme to a TOML file.
        
        Args:
            theme_name: Name of the custom theme
            theme_config: Dictionary containing theme configuration
        """
        
    def list_available_themes(self) -> Dict[str, List[str]]:
        """
        List available themes for different backends.
        
        Returns:
            Dictionary of available themes
        """
        
    def example_themes(self):
        """
        Generate example plots to demonstrate available themes.
        
        Useful for theme comparison and exploration.
        """
```

## Visualization Module

### Stream Visualizations

#### `BaseStreamVisualization`

Base class for time series visualization components.

```python
class BaseStreamVisualization(VisualizationComponent, ABC):
    """
    Base class for time series visualization components with common functionality.
    """
    
    def __init__(
        self,
        data: Union[da.Array, Dict[int, np.ndarray]],
        sampling_rate: float,
        elements: Optional[List[int]] = None,  # Generic name for selectable elements
        time_window: Optional[Tuple[float, float]] = None,
        **kwargs,
    ):
        """
        Initialize the base stream visualization.
        
        Parameters
        ----------
        data : Union[da.Array, Dict[int, np.ndarray]]
            Either:
            - Dask array with data in the format (samples × elements)
            - Dictionary mapping element IDs to time series data
        sampling_rate : float
            Sampling frequency in Hz
        elements : list of int, optional
            Specific elements to display. If None, uses all available.
        time_window : tuple of (float, float), optional
            Time window to display (start_time, end_time) in seconds
        **kwargs : dict
            Additional configuration parameters
        """
        
    def _get_element_data(self, element_id: int) -> np.ndarray:
        """
        Get data for a specific element.
        
        Parameters
        ----------
        element_id : int
            Element ID to retrieve
            
        Returns
        -------
        np.ndarray
            Time series data for the specified element
        """
        
    def _get_time_array(self, data_length: int = None) -> np.ndarray:
        """
        Get time array based on data length and sampling rate.
        
        Parameters
        ----------
        data_length : int, optional
            Length of data array. If None, determines from data.
            
        Returns
        -------
        np.ndarray
            Time values in seconds
        """
        
    def _apply_time_window(
        self, data: np.ndarray, time: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply time window to data and time arrays.
        
        Parameters
        ----------
        data : np.ndarray
            Data array to filter
        time : np.ndarray
            Time array to filter
            
        Returns
        -------
        Tuple[np.n