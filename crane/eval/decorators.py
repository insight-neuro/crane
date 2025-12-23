from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from ..core import BrainFeatureExtractor, BrainModel


@dataclass(frozen=True)
class TaskSpec:
    role: Literal["eval", "finetune"]
    name: str
    uses: str | None
    tags: set[str]
    kwargs: dict
    fn: Callable


class TaskDescriptor:
    def __init__(self, role: Literal["eval", "finetune"], name: str, uses: str | None, tags: list[str] | None, kwargs: dict[str, Any], fn: Callable):
        self.role: Literal["eval", "finetune"] = role
        self.name = name
        self.uses = uses
        self.tags = set(tags or [])
        self.kwargs = kwargs
        self._fn = fn

    def bind(self, instance: object):
        def bound(model: BrainModel, featurizer: BrainFeatureExtractor):
            return self._fn(instance, model, featurizer, **self.kwargs)

        return TaskSpec(role=self.role, name=self.name, uses=self.uses, tags=self.tags, kwargs=self.kwargs, fn=bound)


def finetune(name: str, **kwargs):
    def decorator(fn: Callable[[BrainModel, BrainFeatureExtractor], BrainModel]):
        desc = TaskDescriptor(role="finetune", name=name, uses=None, tags=None, kwargs=kwargs, fn=fn)
        fn.__bench_tasks__ = getattr(fn, "__bench_tasks__", []) + [desc]  # type: ignore
        return fn

    return decorator


def eval(name: str, uses: str | None, tags: list[str] | None = None, **kwargs):
    def decorator(fn: Callable[[BrainModel, BrainFeatureExtractor], dict[str, Any]]):
        desc = TaskDescriptor(role="finetune", name=name, uses=uses, tags=tags, kwargs=kwargs, fn=fn)
        fn.__bench_tasks__ = getattr(fn, "__bench_tasks__", []) + [desc]  # type: ignore
        return fn

    return decorator
