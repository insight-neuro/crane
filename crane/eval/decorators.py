from collections.abc import Callable
from dataclasses import dataclass
from itertools import product
from typing import Any, Literal

from ..core import BrainFeatureExtractor, BrainModel
from .sweep import Sweep

type Role = Literal["eval", "finetune"]

@dataclass(frozen=True)
class TaskSpec:
    role: Role
    name: str
    base: str
    uses: str | None
    tags: set[str]
    fn: Callable


@dataclass(frozen=True)
class TaskDescriptor:
    role: Role
    name: str
    uses: str | None
    tags: set[str]
    sweeps: dict[str, Sweep]
    static: dict[str, Any]
    fn: Callable

    def expand(self, instance: object) -> list[TaskSpec]:
        def make_bound(**kwargs):
            def bound(model: BrainModel, featurizer: BrainFeatureExtractor):
                return self.fn(instance, model, featurizer, **kwargs)

            return bound

        resolved = {name: sweep.resolve() for name, sweep in self.sweeps.items()}

        sweep_names = sorted(resolved.keys())
        sweep_values = [resolved[k] for k in sweep_names]

        # Special case: no sweeps -> exactly one task
        combinations = product(*sweep_values if sweep_values else [()])

        specs = []
        for combo in combinations:
            axis_values = dict(zip(sweep_names, combo, strict=True))

            kwargs = {**axis_values, **self.static}

            suffix = ",".join(f"{name}={value}" for name, value in axis_values.items())
            name = f"{self.name}[{suffix}]" if suffix else self.name

            specs.append(TaskSpec(role=self.role, name=name, base=self.name, uses=self.uses, tags=self.tags, fn=make_bound(**kwargs)))

        return specs


def _build_descriptor(role: Role, name: str, uses: str | None, tags: list[str] | None, kwargs: dict[str, Any], fn: Callable) -> TaskDescriptor:
    sweeps = {}
    static = {}

    for k, v in kwargs:
        if isinstance(v, Sweep):
            if v.name != k:
                raise ValueError(f"Sweep name '{v.name}' must match kwarg '{k}'")
            sweeps[name] = v
        else:
            static[name] = v

    return TaskDescriptor(role=role, name=name, uses=uses, tags=set(tags or []), sweeps=sweeps, static=static, fn=fn)


def finetune(name: str, **kwargs):
    def decorator(fn: Callable[[BrainModel, BrainFeatureExtractor], BrainModel]):
        desc = _build_descriptor(role="finetune", name=name, uses=None, tags=None, kwargs=kwargs, fn=fn)
        fn.__bench_tasks__ = getattr(fn, "__bench_tasks__", []) + [desc]  # type: ignore
        return fn

    return decorator


def eval(name: str, uses: str | None, tags: list[str] | None = None, **kwargs):
    def decorator(fn: Callable[[BrainModel, BrainFeatureExtractor], dict[str, Any]]):
        desc = _build_descriptor(role="eval", name=name, uses=uses, tags=tags, kwargs=kwargs, fn=fn)
        fn.__bench_tasks__ = getattr(fn, "__bench_tasks__", []) + [desc]  # type: ignore
        return fn

    return decorator
