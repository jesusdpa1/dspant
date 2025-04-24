import plotly.graph_objs as go
from plotly_resampler import FigureResampler
import numpy as np
import dask.array as da
from dspant_viz.core.base import BasePlotModel
import streamlit as st
import streamlit.components.v1 as components
import plotly.io as pio
from multiprocessing import Process


# Ensure colored traces in Streamlit
pio.templates.default = "plotly"

class SingleChannelPlotWidget:
    def __init__(
        self,
        plot_model: BasePlotModel,
        title: str = "Single Channel Plot",
        max_points: int = 5000
    ):
        self.plot_model = plot_model
        self.title = title
        self.max_points = max_points

        self._create_plot()

    def _create_plot(self):
        """
        Create the Plotly Resampler plot
        """
        time = self.plot_model.get_time_array()
        channel_data = self.plot_model.get_channel_data()

        # Convert Dask arrays to NumPy
        if isinstance(time, da.Array):
            time = time.compute()
        if isinstance(channel_data, da.Array):
            channel_data = channel_data.compute()

        # Create an empty figure first
        base_fig = go.Figure()

        # Create FigureResampler with more explicit parameters
        self.fig = FigureResampler(
            figure=base_fig,
            default_n_shown_samples=self.max_points,
            convert_existing_traces=True
        )

        # Add trace after figure creation
        self.fig.add_trace(
            go.Scatter(name=f'Channel {self.plot_model.channel}'),
            hf_x=time,
            hf_y=channel_data
        )

        self.fig.update_layout(
            title=self.title,
            xaxis_title='Time (s)',
            yaxis_title='Amplitude'
        )

    def show(self):
        """
        Display the plot
        """
        return self.fig

class StreamlitSingleChannelPlot:
    def __init__(
        self,
        plot_model: BasePlotModel,
        title: str = "Single Channel Plot",
        max_points: int = 5000
    ):
        self.plot_model = plot_model
        self.title = title
        self.max_points = max_points
        self.num_channels = self.plot_model.data.shape[1]

        self.port = 9022  # Configurable port

    def create_plotly_resampler_figure(self, selected_channel):
        """
        Create Plotly Resampler Figure
        """
        time = self.plot_model.get_time_array()
        channel_data = self.plot_model.get_channel_data(selected_channel)

        # Convert Dask arrays to NumPy
        if isinstance(time, da.Array):
            time = time.compute()
        if isinstance(channel_data, da.Array):
            channel_data = channel_data.compute()

        # Create an empty figure first with dark background
        base_fig = go.Figure()

        fig = FigureResampler(
            figure=base_fig,
            default_n_shown_samples=self.max_points,
            convert_existing_traces=True
        )

        # Add trace after figure creation
        fig.add_trace(
            go.Scatter(
                name=f'Channel {selected_channel}',
                line=dict(color='#1E90FF')  # Dodger Blue to stand out in dark theme
            ),
            hf_x=time,
            hf_y=channel_data
        )

        # Update layout with dark theme and additional features
        fig.update_layout(
            title=f"{self.title} - Channel {selected_channel}",
            xaxis_title='Time (s)',
            yaxis_title='Amplitude',
            height=700,
            # Seaborn dark-like background
            plot_bgcolor='rgb(30,30,30)',
            paper_bgcolor='rgb(20,20,20)',
            font=dict(color='white'),
        )

        # Add range slider
        fig.update_layout(
            xaxis=dict(
                rangeslider=dict(visible=True),
                type="linear"
            )
        )

        return fig

    def show(self):
        """
        Show plot in Streamlit
        """
        st.title(self.title)

        # Channel selection slider
        selected_channel = st.select_slider(
            'Select Channel',
            options=list(range(self.num_channels)),
            value=0
        )

        fig = self.create_plotly_resampler_figure(selected_channel)

        # Start Dash app in a separate process
        proc = Process(
            target=fig.show_dash,
            kwargs=dict(mode="external", port=self.port)
        ).start()

        # Embed as iframe in Streamlit

        components.iframe(
            f"http://localhost:{self.port}",
            height=700
        )
