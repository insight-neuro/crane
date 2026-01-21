from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, Literal, overload

from .data import NeuralData


@dataclass(frozen=True, slots=True)
class Task:
    train: NeuralData
    """Training data for the task."""
    test: NeuralData
    """Testing data for the task."""
    train_fn: Callable | Literal["default"] = "default"
    """Override default training function for this task."""
    test_fn: Callable | Literal["default"] = "default"
    """Override default testing function for this task."""
    tags: frozenset[str] | None = None
    """Tags associated with the task."""

    def bind(
        self, *, group: str, default_train_fn: Callable, default_test_fn: Callable, additional_tags: list[str]
    ) -> BoundTask:
        return BoundTask(
            group=group,
            train=self.train,
            test=self.test,
            train_fn=self.train_fn if self.train_fn != "default" else default_train_fn,
            test_fn=self.test_fn if self.test_fn != "default" else default_test_fn,
            tags=(self.tags or frozenset()) | frozenset(additional_tags),
        )


@dataclass(frozen=True, slots=True)
class BoundTask:
    group: str
    """Name of the task group this task belongs to."""
    train: NeuralData
    """Training data for the task."""
    test: NeuralData
    """Testing data for the task."""
    train_fn: Callable
    """Training function for this task."""
    test_fn: Callable
    """Testing function for this task."""
    tags: frozenset[str]
    """Tags associated with the task."""


class TaskGroup:
    def __init__(
        self,
        name: str,
        tasks: list[Task],
        *,
        train_fn: Callable | Literal["default"] = "default",
        test_fn: Callable | Literal["default"] = "default",
        tags: list[str] | None = None,
    ):
        self.name = name
        self.tasks = [
            replace(
                task,
                train_fn=task.train_fn if task.train_fn != "default" else train_fn,
                test_fn=task.test_fn if task.test_fn != "default" else test_fn,
                tags=frozenset(task.tags or tags or []),
            )
            for task in tasks
        ]

    def __iter__(self):
        return iter(self.tasks)


@dataclass
class TaskSpec:
    name: str | None
    kwargs: dict[str, Any]
    tags: list[str]


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
    def decorate(fn: Callable) -> Callable:
        name = None if callable(arg) else arg
        spec = TaskSpec(name=name, kwargs=kwargs, tags=tags or [])

        fn.__bench_tasks__ = getattr(fn, "__bench_tasks__", []) + [spec]  # type: ignore[attr-defined]
        return fn

    return decorate(arg) if callable(arg) else decorate
