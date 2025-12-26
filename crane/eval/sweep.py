from collections.abc import Callable, Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class Sweep:
    name: str
    values_or_fn: Iterable | Callable[..., Iterable]

    def resolve(self):
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
    return Sweep(name=name, values_or_fn=values_or_fn)
