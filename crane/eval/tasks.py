from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from typing import Any, overload

from crane.eval.artifacts import TaskSpec
from crane.eval.data import NeuralData


@dataclass(frozen=True, slots=True)
class Task:
    """Specification of a single evaluation task."""

    group: str = field(init=False, default="")
    """Name of the task group."""
    train: NeuralData
    """Training data for the task."""
    test: NeuralData
    """Testing data for the task."""
    tags: frozenset[str]
    """Tags associated with the task."""

    def __init__(
        self,
        train: NeuralData,
        test: NeuralData,
        *,
        tags: Iterable[str] | None = None,
    ):
        super().__setattr__("train", train)
        super().__setattr__("test", test)
        super().__setattr__("tags", frozenset(tags or []))


class TaskGroup:
    """Group of tasks for evaluation. Iterable over contained tasks.

    Args:
        name (str): Name of the task group.
        tasks (list[Task]): List of tasks in the group.
        tags (Iterable[str] | None, optional): Tags associated with the evaluation tasks in this
            group. Will be added to each task in the group. Defaults to None.
    """

    def __init__(self, name: str, tasks: list[Task], *, tags: Iterable[str] | None = None):
        extra_tags = frozenset(tags or [])
        self.name = name
        self.tasks = [replace(task, tags=task.tags | extra_tags) for task in tasks]

    def __iter__(self):
        return iter(self.tasks)

    def __len__(self):
        return len(self.tasks)


@overload
def task_group(fn: Callable[..., TaskGroup], /) -> Callable[..., TaskGroup]:
    """
    Define a task group from function that returns a TaskGroup. Uses the TaskGroup's name as the function name.

    Args:
        tags (list[str] | None, optional): Tags associated with the evaluation task. Will be added to each task in the group.
        **kwargs: Additional keyword arguments passed to the function for this task group.

    Returns:
        Callable[..., TaskGroup]: Decorated function that returns a TaskGroup.
    """
    ...


@overload
def task_group(
    *, tags: list[str] | None = None, **kwargs: Any
) -> Callable[[Callable[..., TaskGroup]], Callable[..., TaskGroup]]:
    """
    Define a task group from function that returns a TaskGroup. Uses the TaskGroup's name as the function name.

    Args:
        tags (list[str] | None, optional): Tags associated with the evaluation task. Will be added to each task in the group.
        **kwargs: Additional keyword arguments passed to the function for this task group.

    Returns:
        Callable[[Callable[..., TaskGroup]], Callable[..., TaskGroup]]: Decorated function that returns a TaskGroup.
    """
    ...


@overload
def task_group(
    name: str, /, *, tags: list[str] | None = None, **kwargs: Any
) -> Callable[[Callable[..., list[Task]]], Callable[..., list[Task]]]:
    """Define a task group from function that returns a list of tasks.

    Args:
        name (str): Name of the task group.
        tags (list[str] | None, optional): Tags associated with the evaluation task. Will be added to each task in the group.
        **kwargs: Additional keyword arguments passed to the function for this task group.

    Returns:
        Callable[[Callable[..., list[Task]]], Callable[..., list[Task]]]: Decorated function that returns a list of tasks.
    """
    ...


def task_group(arg: Callable | str | None = None, /, *, tags: list[str] | None = None, **kwargs: Any):
    frozen_tags = frozenset(tags or [])

    def decorate(fn: Callable) -> Callable:
        name = None if callable(arg) else arg
        spec = TaskSpec(group=name, kwargs=kwargs, tags=frozen_tags)
        fn.__bench_tasks__ = getattr(fn, "__bench_tasks__", []) + [spec]  # type: ignore[attr-defined]
        return fn

    return decorate(arg) if callable(arg) else decorate
