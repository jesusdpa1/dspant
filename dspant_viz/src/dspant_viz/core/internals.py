# core/internals.py
import sys
from functools import wraps
from typing import Any, Callable, TypeVar

T = TypeVar('T')


def public_api(obj: T) -> T:
    """
    Decorator to mark a function or class as part of the public API,
    automatically adding it to the module's __all__ list.

    Usage:
        @public_api
        def my_function():
            pass
    """
    module = obj.__module__
    module_obj = sys.modules[module]
    if not hasattr(module_obj, "__all__"):
        module_obj.__all__ = []
    if obj.__name__ not in module_obj.__all__:
        module_obj.__all__.append(obj.__name__)
    return obj


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
        if not hasattr(wrapper, '_backend_requirements'):
            wrapper._backend_requirements = []
        wrapper._backend_requirements.append(backend_type)

        return wrapper
    return decorator
