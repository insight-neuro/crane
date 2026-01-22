from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from crane.eval.data.base import NeuralData
from crane.eval.types import TestFn, TrainFn


@dataclass(frozen=True, slots=True)
class TaskSpec:
    group: str | None
    """Name of the task group specification."""
    kwargs: dict[str, Any]
    """Keyword arguments passed to the task instantiation function."""
    tags: frozenset[str]
    """Tags associated with the task."""


@dataclass(frozen=True, slots=True)
class BoundTask:
    group: str
    """Name of the task group this task belongs to."""
    train: NeuralData
    """Training data for the task."""
    test: NeuralData
    """Testing data for the task."""
    train_fn: TrainFn
    """Training function for this task."""
    test_fn: TestFn
    """Testing function for this task."""
    tags: frozenset[str]
    """Tags associated with the task."""


@dataclass(frozen=True, slots=True)
class ExecutionUnit:
    """Unit of execution in an evaluation plan. Iterable over its tasks."""

    train_key: str
    """Unique key identifying the training configuration."""
    train_fn: TrainFn
    """Training function to be used."""
    train_data: NeuralData
    """Training data to be used."""
    tasks: tuple[BoundTask, ...]
    """Tuple of tasks to be executed with this training configuration."""

    @staticmethod
    def from_tasks(
        train_fn: TrainFn,
        train_data: NeuralData,
        tasks: list[BoundTask],
    ) -> ExecutionUnit:
        """Create an ExecutionUnit from the given training configuration and tasks."""
        return ExecutionUnit(
            train_key=f"{train_fn.__name__}_{id(train_data)}",
            train_fn=train_fn,
            train_data=train_data,
            tasks=tuple(tasks),
        )

    def __len__(self) -> int:
        return len(self.tasks)

    def __iter__(self):
        return iter(self.tasks)


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    plan: tuple[ExecutionUnit, ...]
    """Tuple of execution units in the plan."""

    def __len__(self) -> int:
        return len(self.plan)

    def __iter__(self):
        return iter(self.plan)


@dataclass(frozen=True, slots=True)
class TaskResult:
    task_fn: str
    """Name of the task function used for evaluation."""
    group: str
    """Group name."""
    task_id: str
    """Task identifier."""
    metrics: dict[str, Any]
    """Mapping of metric names to their values."""
    artifacts: dict[str, Any] = field(default_factory=dict)
    """Additional artifacts produced during evaluation."""


@dataclass(frozen=True, slots=True)
class RunResult:
    benchmark: str
    """Benchmark name."""
    version: str | None
    """Benchmark version."""
    model: str
    """Evaluated model name."""
    task_results: tuple[TaskResult, ...]
    """Results for individual tasks."""

    def __len__(self) -> int:
        return len(self.task_results)

    def by_group(self) -> dict[str, list[TaskResult]]:
        """Group task results by their group name."""
        out: dict[str, list[TaskResult]] = defaultdict(list)
        for result in self.task_results:
            out[result.group].append(result)
        return out
