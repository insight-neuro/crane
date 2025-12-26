from collections.abc import Callable, Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class Sweep:
    """A parameter sweep definition."""

    name: str
    """Unique name of the sweep parameter."""

    values_or_fn: Iterable | Callable[..., Iterable]
    """An iterable of values or a callable that returns an iterable of values to sweep over."""

    def resolve(self):
        """Resolve the sweep values. If `values_or_fn` is a
        callable, it will be called to obtain the values.

        Raises:
            ValueError: If the sweep produces no values.
            ValueError: If the sweep has duplicate values.

        Returns:
            tuple: A tuple of resolved sweep values.
        """
        if callable(self.values_or_fn):
            values = tuple(self.values_or_fn())
        else:
            values = tuple(self.values_or_fn)

        if not values:
            raise ValueError(f"Sweep `{self.name}` produced no values!")

        if len(set(values)) != len(values):
            raise ValueError(f"Sweep `{self.name}` has duplicate values!")

        return values


def sweep(name: str, values_or_fn: Iterable | Callable[..., Iterable]) -> Sweep:
    """Create a Sweep instance."""
    return Sweep(name=name, values_or_fn=values_or_fn)
