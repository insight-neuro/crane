import inspect
from functools import wraps

from crane.featurizer import CraneFeature


def allow_inplace(func):
    """Allow a preprocessing function to modify the input data in-place if `inplace=True` is passed, or return a modified copy if `inplace=False` (default)."""
    sig = inspect.signature(func)

    @wraps(func)
    def wrapper(*args, inplace: bool = False, **kwargs) -> CraneFeature:
        bound = sig.bind_partial(*args, **kwargs)
        bound.apply_defaults()

        if "data" not in bound.arguments:
            raise ValueError("The wrapped function must have a 'data' argument")

        if not inplace:
            bound.arguments["data"] = bound.arguments["data"].copy()

        return func(*bound.args, **bound.kwargs)

    return wrapper
