from collections.abc import Callable
from dataclasses import dataclass
from itertools import product
from typing import Any, Concatenate, Literal, ParamSpec

from ..core import BrainFeatureExtractor, BrainModel
from .sweep import Sweep

type Role = Literal["eval", "finetune"]
P = ParamSpec("P")


@dataclass(frozen=True)
class TaskSpec:
    """A specific task instance with all sweeps resolved."""

    base: str
    """Base name of the task before sweep expansion."""
    sweep_axes: dict[str, Any]
    """Mapping of sweep parameter names to their selected values."""
    role: Role
    """Specifies whether this is an eval or finetune task."""
    uses: str | None
    """The resource this task uses, if any."""
    tags: set[str]
    """A set of tags associated with this task."""
    fn: Callable
    """The callable that executes the task."""

    @property
    def name(self) -> str:
        """Unique name of the task instance."""
        suffix = ",".join(f"{name}={value}" for name, value in self.sweep_axes.items())
        name = f"{self.name}[{suffix}]" if suffix else self.name
        return name


@dataclass(frozen=True)
class TaskDescriptor:
    """A task definition that may include parameter sweeps."""

    name: str
    """Unique name of the task."""
    role: Role
    """Specifies whether this is an eval or finetune task."""
    uses: str | None
    """The resource this task uses, if any."""
    tags: set[str]
    """A set of tags associated with this task."""
    sweeps: dict[str, Sweep]
    """A mapping of sweep parameter names to Sweep instances."""
    static: dict[str, Any]
    """A mapping of static parameter names to their fixed values."""
    fn: Callable
    """The callable that executes the task."""

    def expand(self, instance: object) -> list[TaskSpec]:
        """Expand the task descriptor into a list of task
        specifications by resolving all parameter sweeps.

        Args:
            instance (object): The instance to bind to the task function.

        Returns:
            list[TaskSpec]: A list of expanded task specifications.
        """
        resolved = {name: sweep.resolve() for name, sweep in self.sweeps.items()}

        sweep_names = sorted(resolved.keys())
        sweep_values = [resolved[k] for k in sweep_names]

        # Special case: no sweeps -> exactly one task
        combinations = product(*sweep_values if sweep_values else [()])

        def make_bound(**kwargs):
            def bound(model: BrainModel, featurizer: BrainFeatureExtractor):
                return self.fn(instance, model, featurizer, **kwargs)

            return bound

        specs = []
        for combo in combinations:
            axis_values = dict(zip(sweep_names, combo, strict=True))

            kwargs = {**axis_values, **self.static}
            specs.append(
                TaskSpec(
                    base=self.name,
                    role=self.role,
                    sweep_axes=axis_values,
                    uses=self.uses,
                    tags=self.tags,
                    fn=make_bound(**kwargs),
                )
            )

        return specs


def _build_descriptor(
    name: str, role: Role, uses: str | None, tags: list[str] | None, kwargs: dict[str, Any], fn: Callable
) -> TaskDescriptor:
    """Build a TaskDescriptor from the given parameters.

    Args:
        name (str): Name of the task.
        role (Role): Role of the task, either "eval" or "finetune".
        uses (str | None): Resource used by the task.
        tags (list[str] | None): Tags associated with the task.
        kwargs (dict[str, Any]): Keyword arguments, some of which may be Sweeps.
        fn (Callable): The function implementing the task.

    Raises:
        ValueError: If a sweep name does not match the corresponding kwarg name.

    Returns:
        TaskDescriptor: The constructed task descriptor.
    """
    sweeps = {}
    static = {}

    for k, v in kwargs.items():
        if isinstance(v, Sweep):
            if v.name != k:
                raise ValueError(f"Sweep name `{v.name}` must match kwarg `{k}`")
            sweeps[k] = v
        else:
            static[k] = v

    return TaskDescriptor(role=role, name=name, uses=uses, tags=set(tags or []), sweeps=sweeps, static=static, fn=fn)


def eval(name: str, uses: str | None, tags: list[str] | None = None, **kwargs):
    """
    Define an evaluation task.

    Args:
        name (str): Name of the evaluation task.
        uses (str | None): Resource used by the evaluation task.
        tags (list[str] | None): Tags associated with the evaluation task.
        **kwargs: Keyword arguments for the evaluation task, which may include Sweeps.
    """

    def decorator(fn: Callable[Concatenate[object, BrainModel, BrainFeatureExtractor, P], dict[str, Any]]):
        desc = _build_descriptor(name=name, role="eval", uses=uses, tags=tags, kwargs=kwargs, fn=fn)
        fn.__bench_tasks__ = getattr(fn, "__bench_tasks__", []) + [desc]  # type: ignore
        return fn

    return decorator


def finetune(name: str, **kwargs):
    """
    Define a finetuning task.

    Args:
        name (str): Name of the finetuning task.
        **kwargs: Keyword arguments for the finetuning task, which may include Sweeps.
    """

    def decorator(fn: Callable[Concatenate[object, BrainModel, BrainFeatureExtractor, P], BrainModel]):
        desc = _build_descriptor(name=name, role="finetune", uses=None, tags=None, kwargs=kwargs, fn=fn)
        fn.__bench_tasks__ = getattr(fn, "__bench_tasks__", []) + [desc]  # type: ignore
        return fn

    return decorator
