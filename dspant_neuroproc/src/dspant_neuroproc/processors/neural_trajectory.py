from typing import Optional, Tuple, Union

import dask.array as da
import numpy as np
from sklearn.decomposition import PCA


class NeuralTrajectoryAnalyzer:
    def __init__(
        self,
        epochs_data: Union[np.ndarray, da.Array],
        used_unit_ids: Optional[list] = None,
    ):
        """
        Initialize Neural Trajectory Analyzer

        Parameters:
        -----------
        epochs_data : array
            Extracted neural epochs (epochs × time × units)
        used_unit_ids : list, optional
            List of unit identifiers
        """
        # Ensure data is numpy array
        self.epochs_data = (
            epochs_data.compute() if hasattr(epochs_data, "compute") else epochs_data
        )
        self.used_unit_ids = used_unit_ids

    def compute_pca(self, n_components: int = 3, normalize: bool = True) -> np.ndarray:
        """
        Compute PCA for neural trajectories

        Parameters:
        -----------
        n_components : int, optional
            Number of principal components to retain
        normalize : bool, optional
            Whether to normalize data before PCA

        Returns:
        --------
        np.ndarray
            Reduced dimensionality trajectories
        """
        # Reshape data for PCA
        # Shape: (epochs * time, units)
        reshaped_data = self.epochs_data.reshape(-1, self.epochs_data.shape[-1])

        # Optional normalization
        if normalize:
            reshaped_data = (
                reshaped_data - reshaped_data.mean(axis=0)
            ) / reshaped_data.std(axis=0)

        # Perform PCA
        pca = PCA(n_components=n_components)
        reduced_trajectories = pca.fit_transform(reshaped_data)

        # Reshape back to original epoch structure
        return reduced_trajectories.reshape(
            self.epochs_data.shape[0], self.epochs_data.shape[1], n_components
        )

    def compute_trajectory_metrics(self, trajectories: np.ndarray) -> dict:
        """
        Compute trajectory analysis metrics

        Parameters:
        -----------
        trajectories : np.ndarray
            Reduced dimensionality trajectories

        Returns:
        --------
        dict
            Trajectory analysis metrics
        """
        metrics = {
            "variance_explained": None,  # Placeholder for variance explained by components
            "trajectory_length": None,  # Total trajectory length
            "curvature": None,  # Trajectory curvature
            "stability": None,  # Trajectory stability
        }

        return metrics

    def visualize_trajectories(self, trajectories: np.ndarray, plot_type: str = "3d"):
        """
        Visualize neural trajectories

        Parameters:
        -----------
        trajectories : np.ndarray
            Reduced dimensionality trajectories
        plot_type : str, optional
            Type of plot ('3d', '2d')
        """
        import matplotlib.pyplot as plt

        if plot_type == "3d":
            # 3D trajectory plot
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection="3d")

            # Plot each epoch's trajectory
            for epoch_traj in trajectories:
                ax.plot(epoch_traj[:, 0], epoch_traj[:, 1], epoch_traj[:, 2])

            ax.set_title("Neural Trajectories")
            plt.show()

        elif plot_type == "2d":
            # 2D trajectory plot
            plt.figure(figsize=(10, 8))

            # Plot each epoch's trajectory
            for epoch_traj in trajectories:
                plt.plot(epoch_traj[:, 0], epoch_traj[:, 1])

            plt.title("Neural Trajectories")
            plt.show()
