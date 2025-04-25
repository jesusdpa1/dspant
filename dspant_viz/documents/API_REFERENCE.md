# 1. Define a class that inherits from VisualizationComponent
class MyComponent(VisualizationComponent):
    def __init__(self, data, **kwargs):
        super().__init__(data, **kwargs)
        # Initialize custom attributes
    
    # 2. Implement get_data() to prepare data for rendering
    def get_data(self) -> Dict:
        return {
            "data": {...},  # Data for rendering
            "params": {...}  # Parameters for rendering
        }
    
    # 3. Implement update() to handle parameter updates
    def update(self, **kwargs) -> None:
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.config[key] = value
    
    # 4. Implement plot() to dispatch to the appropriate backend
    def plot(self, backend="mpl", **kwargs):
        if backend == "mpl":
            from dspant_viz.backends.mpl.my_component import render_my_component
        elif backend == "plotly":
            from dspant_viz.backends.plotly.my_component import render_my_component
        else:
            raise ValueError(f"Unsupported backend: {backend}")
        
        return render_my_component(self.get_data(), **kwargs)