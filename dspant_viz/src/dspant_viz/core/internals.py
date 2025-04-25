# src/dspant_viz/core/internals.py
import sys
from functools import wraps
from typing import Any, Callable, TypeVar, Union

T = TypeVar("T")


def public_api(
    module_override: Union[str, None] = None, export: bool = True
) -> Callable:
    """
    Enhanced decorator to mark functions or classes as part of the public API.

    Parameters
    ----------
    module_override : str, optional
        Explicitly specify the module to add the item to __all__
    export : bool, default True
        Whether to automatically export the item to __all__

    Usage:
    ------
    @public_api()  # Simple usage
    def my_function():
        pass

    @public_api(module_override='dspant_viz.visualization')  # Custom module
    def another_function():
        pass

    @public_api(export=False)  # Don't export to __all__
    def internal_function():
        pass
    """

    def decorator(obj: T) -> T:
        # Determine the module
        module = module_override or obj.__module__
        module_obj = sys.modules[module]

        # Initialize __all__ if not exists
        if not hasattr(module_obj, "__all__"):
            module_obj.__all__ = []

        # Add to __all__ if export is True and not already present
        if export and obj.__name__ not in module_obj.__all__:
            module_obj.__all__.append(obj.__name__)

        return obj

    return decorator


def require_backend(backend_type: str) -> Callable:
    """
    Decorator to specify which backend(s) a method or function requires.

    Usage:
        @require_backend('mpl')
        def plot_with_matplotlib(self):
            pass
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # In the future, we could add validation here
            return func(*args, **kwargs)

        # Add metadata to the function
        if not hasattr(wrapper, "_backend_requirements"):
            wrapper._backend_requirements = []
        wrapper._backend_requirements.append(backend_type)

        return wrapper

    return decorator


def register_module_components(module_name: str = None):
    """Register all classes in a module that should be public"""

    def decorator(module):
        for name, obj in module.__dict__.items():
            if isinstance(obj, type) and not name.startswith("_"):
                public_api(module_override=module_name)(obj)
        return module

    return decorator
