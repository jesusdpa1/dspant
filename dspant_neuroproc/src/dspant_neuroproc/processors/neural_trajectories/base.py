# src/dspant/processors/neural_trajectories/base.py

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import BaseEstimator

from dspant.core.internals import public_api


@public_api
class BaseTrajectoryAnalyzer(ABC):
    """
    Abstract base class for neural trajectory analysis.

    This class defines the common interface for all trajectory analysis methods,
    regardless of the specific dimensionality reduction technique used.
    """

    def __init__(
        self,
        n_components: int = 3,
        random_state: Optional[int] = None,
        compute_immediately: bool = False,
    ):
        """
        Initialize the trajectory analyzer.

        Parameters
        ----------
        n_components : int
            Number of components/dimensions to extract
        random_state : int, optional
            Random seed for reproducibility
        compute_immediately : bool
            Whether to compute results immediately when fit is called
        """
        self.n_components = n_components
        self.random_state = random_state
        self.compute_immediately = compute_immediately
        self._is_fitted = False
        self._model = None
        self._components = None
        self._trajectories = None
        self._explained_variance_ratio = None

    @abstractmethod
    def _create_model(self) -> BaseEstimator:
        """
        Create the dimensionality reduction model.

        This method must be implemented by subclasses to create
        the specific dimensionality reduction model to be used.

        Returns
        -------
        BaseEstimator
            The sklearn-compatible dimensionality reduction model
        """
        pass

    def fit(self, data: Union[np.ndarray, da.Array]) -> "BaseTrajectoryAnalyzer":
        """
        Fit the dimensionality reduction model to the data.

        Parameters
        ----------
        data : array, shape (n_trials, n_timepoints, n_neurons)
            Neural data array

        Returns
        -------
        self
            The fitted analyzer
        """
        # Reshape data if needed: (trials, timepoints, neurons) -> (trials*timepoints, neurons)
        data_shape = data.shape
        if len(data_shape) == 3:
            n_trials, n_timepoints, n_neurons = data_shape
            reshaped_data = data.reshape(n_trials * n_timepoints, n_neurons)
        else:
            reshaped_data = data

        # Create and fit the model
        self._model = self._create_model()

        # Handle both numpy and dask arrays
        if isinstance(reshaped_data, da.Array):
            if self.compute_immediately:
                # Compute immediately
                self._model.fit(reshaped_data.compute())
            else:
                # Schedule computation but don't execute
                self._model = da.map_blocks(
                    lambda x: self._model.fit(x), reshaped_data, dtype=object
                )
        else:
            # Regular numpy array
            self._model.fit(reshaped_data)

        # Store components and explained variance if available
        if hasattr(self._model, "components_"):
            self._components = self._model.components_
        elif hasattr(self._model, "components"):
            self._components = self._model.components

        if hasattr(self._model, "explained_variance_ratio_"):
            self._explained_variance_ratio = self._model.explained_variance_ratio_

        self._is_fitted = True
        return self

    def transform(self, data: Union[np.ndarray, da.Array]) -> "BaseTrajectoryAnalyzer":
        """
        Transform data to the low-dimensional space and store it internally.

        Parameters
        ----------
        data : array, shape (n_trials, n_timepoints, n_neurons) or (n_samples, n_neurons)
            Neural data array

        Returns
        -------
        self
            The analyzer with stored trajectories
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call 'fit' before 'transform'.")

        # Preserve original shape for reshaping after transform
        original_shape = data.shape

        # Reshape data if needed
        if len(original_shape) == 3:
            n_trials, n_timepoints, n_neurons = original_shape
            reshaped_data = data.reshape(n_trials * n_timepoints, n_neurons)
        else:
            reshaped_data = data

        # Apply transform
        if isinstance(reshaped_data, da.Array):
            low_dim_data = self._model.transform(reshaped_data.compute())
        else:
            low_dim_data = self._model.transform(reshaped_data)

        # Reshape back to original format but with fewer dimensions
        if len(original_shape) == 3:
            low_dim_data = low_dim_data.reshape(
                n_trials, n_timepoints, self.n_components
            )

        # Store trajectories
        self._trajectories = low_dim_data
        return self

    def fit_transform(
        self, data: Union[np.ndarray, da.Array]
    ) -> "BaseTrajectoryAnalyzer":
        """
        Fit the model and transform the data in one step, storing results internally.

        Parameters
        ----------
        data : array, shape (n_trials, n_timepoints, n_neurons) or (n_samples, n_neurons)
            Neural data array

        Returns
        -------
        self
            The analyzer with stored trajectories
        """
        return self.fit(data).transform(data)

    @property
    def components_(self) -> np.ndarray:
        """Get the principal components/factors/directions."""
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call 'fit' first.")

        return self._components

    @property
    def explained_variance_ratio_(self) -> np.ndarray:
        """Get the explained variance ratio (if available)."""
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call 'fit' first.")

        return self._explained_variance_ratio

    @property
    def trajectories(self) -> Union[np.ndarray, da.Array]:
        """
        Get the trajectories in the low-dimensional space.

        Returns
        -------
        array, shape (n_trials, n_timepoints, n_components)
            Neural trajectories in the low-dimensional space
        """
        if self._trajectories is None:
            raise ValueError("No trajectories computed yet. Call 'transform' first.")
        return self._trajectories

    def plot_trajectories(
        self,
        dimensions: Tuple[int, int, int] = (0, 1, 2),
        ax: Optional[plt.Axes] = None,
        colormap: str = "viridis",
        show_markers: bool = True,
        **kwargs,
    ) -> plt.Figure:
        """
        Plot trajectories in 3D space.

        Parameters
        ----------
        dimensions : tuple of int
            Which dimensions to plot (default: first three)
        ax : matplotlib.Axes, optional
            Axes to plot on
        colormap : str
            Colormap for trajectories
        show_markers : bool
            Whether to show markers at each timepoint
        **kwargs
            Additional arguments passed to plot

        Returns
        -------
        matplotlib.Figure
            The figure containing the plot
        """
        if self._trajectories is None:
            raise ValueError("No trajectories computed yet. Call 'transform' first.")

        # Create figure if needed
        if ax is None:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection="3d")
        else:
            fig = ax.figure

        # Get dimensions to plot
        dim1, dim2, dim3 = dimensions

        # Plot each trajectory
        trajectories = self._trajectories

        if isinstance(trajectories, da.Array):
            trajectories = trajectories.compute()

        n_trials, n_timepoints, _ = trajectories.shape

        for trial in range(n_trials):
            traj = trajectories[trial]

            # Get coordinates
            x = traj[:, dim1]
            y = traj[:, dim2]
            z = traj[:, dim3]

            # Create colormap based on timepoints
            colors = plt.cm.get_cmap(colormap)(np.linspace(0, 1, n_timepoints))

            # Plot line
            ax.plot(x, y, z, **kwargs)

            # Plot markers
            if show_markers:
                ax.scatter(x, y, z, c=colors, s=20, alpha=0.8)

            # Mark start and end
            ax.scatter(x[0], y[0], z[0], c="green", s=100, marker="^")
            ax.scatter(x[-1], y[-1], z[-1], c="red", s=100, marker="s")

        # Set labels
        ax.set_xlabel(f"Dimension {dim1 + 1}")
        ax.set_ylabel(f"Dimension {dim2 + 1}")
        ax.set_zlabel(f"Dimension {dim3 + 1}")

        return fig

    def calculate_trajectory_metrics(self) -> Dict:
        """
        Calculate metrics for the trajectories.

        Returns
        -------
        dict
            Dictionary of trajectory metrics
        """
        if self._trajectories is None:
            raise ValueError("No trajectories computed yet. Call 'transform' first.")

        trajectories = self._trajectories
        if isinstance(trajectories, da.Array):
            trajectories = trajectories.compute()

        # Calculate metrics
        metrics = {}

        # Path length
        path_lengths = []
        for trial in range(trajectories.shape[0]):
            traj = trajectories[trial]
            # Calculate Euclidean distance between consecutive points
            diffs = np.diff(traj, axis=0)
            path_length = np.sum(np.sqrt(np.sum(diffs**2, axis=1)))
            path_lengths.append(path_length)

        metrics["path_length"] = np.array(path_lengths)
        metrics["average_path_length"] = np.mean(path_lengths)

        # Trajectory speed (average distance between consecutive points)
        speeds = []
        for trial in range(trajectories.shape[0]):
            traj = trajectories[trial]
            diffs = np.diff(traj, axis=0)
            speeds.append(np.sqrt(np.sum(diffs**2, axis=1)))

        metrics["speeds"] = speeds
        metrics["average_speed"] = np.mean([np.mean(s) for s in speeds])

        # Total distance covered in each dimension
        dim_distances = []
        for dim in range(trajectories.shape[2]):
            distances = []
            for trial in range(trajectories.shape[0]):
                traj = trajectories[trial]
                distances.append(np.abs(traj[-1, dim] - traj[0, dim]))
            dim_distances.append(np.mean(distances))

        metrics["dimension_distances"] = np.array(dim_distances)

        return metrics
