import inspect
from abc import ABC
from collections import defaultdict
from dataclasses import replace
from functools import cached_property
from types import MappingProxyType
from typing import Literal, ClassVar

import torch.nn.functional as F

from crane.core import BrainFeatureExtractor, BrainModel
from crane.eval.artifacts import RunResult
from crane.eval.exec import Executor, SequentialExecutor
from crane.eval.filter import MatchGroups, MatchTags, TaskFilter
from crane.eval.fns import LinearTrain, ZeroShotEval
from crane.eval.plan import GroupedPlanner, Planner
from crane.eval.protocols import TestFn, TrainFn
from crane.eval.tasks import Task


class BrainBench(ABC):
    """Abstract base class for executable brain benchmarks.

    A BrainBench defines a fixed set of evaluation tasks, organized into
    named task groups and annotated with tags for selection and filtering.
    Task definitions are declared by subclasses via decorated methods.

    Tasks are defined with the `@task_group` decorator and
    collected eagerly at subclass initialization time.

    Args:
        train_fn (TrainFn): Default training function for finetuning.
        test_fn (TestFn): Default testing function for evaluation.

    Attributes:
        name (str): Human-readable name of the benchmark.
        version (str | None): Version of the benchmark.
        reference (str | None): Reference or citation for the benchmark.
        default_tags (list[str] | None): Default tags to use if none are specified.

    Methods:
        run(): Evaluate the model on the given dataset.
        description(): Structured, machine-readable description of the benchmark.

    To override (if needed):
        train_fn: Default training function for finetuning. By default, LinearTrain(), requires NeuralLabeledData.
        test_fn: Default testing function for evaluation. By default, ZeroShotEval with MSE metric, requires NeuralLabeledData.
        planner: Default planner for building evaluation plans.
        executor: Default executor for running evaluation plans.
    """

    name: ClassVar[str]
    """Human-readable name of the benchmark."""
    version: ClassVar[str | None] = None
    """Version of the benchmark."""
    reference: ClassVar[str | None] = None
    """Reference or citation for the benchmark."""
    default_tags: ClassVar[list[str] | None] = None
    """Default tags to use if none are specified."""

    train_fn: TrainFn = LinearTrain()
    """Training function for finetuning."""
    test_fn: TestFn = ZeroShotEval({"mse": F.mse_loss})
    """Testing function for evaluation."""

    planner: Planner = GroupedPlanner()
    """Planner to use for building evaluation plans."""
    executor: Executor = SequentialExecutor()
    """Executor to use for running evaluation plans."""

    def __init__(
        self,
        train_fn: TrainFn | None = None,
        test_fn: TestFn | None = None,
    ) -> None:
        self.train_fn = train_fn or self.train_fn
        self.test_fn = test_fn or self.test_fn

        task_groups = self._collect_tasks()
        self.task_groups = MappingProxyType({k: frozenset(v) for k, v in task_groups.items()})

    @property
    def tasks(self) -> set[Task]:
        """All collected tasks in the benchmark."""
        tasks: set[Task] = set()
        for task_group in self.task_groups.values():
            tasks.update(task_group)
        return tasks

    def select_tasks(
        self,
        *,
        task_groups: list[str] | None = None,
        tags: list[str] | Literal["default"] | None = None,
        filters: list[TaskFilter] | None = None,
    ):
        """Select tasks based on task groups and tags.

        Args:
            task_groups (list[str] | None, default=None): Specific task group names to run. If None, run all.
            tags (list[str] | None, default=None): Filter away tasks that don't have any of these. If None, use BenchMark's default_tags if set.
            filters (list[TaskFilter] | None, default=None): Additional custom filters to apply.
        Returns:
            list[Task]: Filtered list of tasks.
        """
        selected_tasks: set[Task] = set(self.tasks)

        # Filter by task groups
        if task_groups is not None:
            selected_tasks = MatchGroups(*task_groups).filter(selected_tasks)

        # Filter by tags
        if tags == "default":
            tags = self.default_tags

        if tags is not None:
            selected_tasks = MatchTags(*tags).filter(selected_tasks)

        for filter in filters or []:
            selected_tasks = filter.filter(selected_tasks)

        return sorted(selected_tasks, key=str)

    def run(
        self,
        model: BrainModel,
        featurizer: BrainFeatureExtractor,
        *,
        task_groups: list[str] | None = None,
        tags: list[str] | Literal["default"] | None = None,
        filters: list[TaskFilter] | None = None,
        planner: Planner | None = None,
        executor: Executor | None = None,
    ) -> RunResult:
        """Evaluate the model on the given dataset.

        Args:
            model (BrainModel): The model to be evaluated.
            featurizer (BrainFeatureExtractor): The feature extractor to process data.
            task_groups (list[str] | None, default=None): Specific task group names to run. If None, run all. Equivalent to MatchGroups() selector.
            tags (list[str] | None, default=None): Filter away tasks that don't have any of these. Equivalent to MatchTags(match_all=False) selector. If None, use BenchMark's default_tags if set.
            filters (list[TaskFilter] | None, default=None): Additional custom filters to apply.
            planner (Planner | None, default=None): Custom planner to use. If None, use the benchmark's default planner.
            executor (Executor | None, default=None): Custom executor to use. If None, use the benchmark's default executor.

        Returns:
            RunResult: The results of the benchmark run.
        """

        # Filter tasks
        tasks = self.select_tasks(
            task_groups=task_groups,
            tags=tags,
            filters=filters,
        )

        # Build plan
        planner = planner or self.planner
        plan = planner.build_plan(tasks)

        # Execute plan
        executor = executor or self.executor
        task_results = executor.run(self, model, featurizer, plan)

        # Finalize results
        return RunResult(
            benchmark=self.name,
            version=self.version,
            model=str(model),
            task_results=tuple(task_results),
        )

    @cached_property
    def description(self) -> dict:
        """Structured, machine-readable description of the benchmark."""
        return {
            "name": self.name,
            "version": self.version,
            "reference": self.reference,
            "default_tags": self.default_tags,
            "task_groups": {
                group: {
                    "name": group,
                    "count": len(tasks),
                    "instances": [
                        {
                            "train": task.train,
                            "test": task.test,
                            "tags": task.tags,
                        }
                        for task in tasks
                    ],
                }
                for group, tasks in self.task_groups.items()
            },
        }

    def _collect_tasks(self):
        """Collect tasks from decorated methods."""
        task_groups: dict[str, set[Task]] = defaultdict(set)

        # Inspect methods for decorated task group definitions
        for _, method in inspect.getmembers(self, inspect.ismethod):
            for spec in getattr(method.__func__, "__bench_tasks__", []):
                task_group = method(**spec.kwargs)  # Call method to create Task instances
                group = spec.group or task_group.name  # If set: list[Task], else TaskGroup

                if not group:
                    raise ValueError(
                        f"Task group name must be specified for tasks defined in {method.__name__}. Cannot be empty."
                    )

                if group in task_groups:
                    raise ValueError(f"Duplicate task group '{group}' defined in {method.__name__}")

                for task in task_group:
                    tags = frozenset(task.tags).union(spec.tags)
                    bound = replace(task, group=group, tags=tags)
                    task_groups[group].add(bound)

        return task_groups

    def __str__(self) -> str:
        d = self.description
        lines = [f"Benchmark: {d['name']}"]

        if d["version"]:
            lines.append(f"Version: {d['version']}")
        if d["reference"]:
            lines.append(f"Reference: {d['reference']}")

        lines.append("Task groups:")
        for group, info in d["task_groups"].items():
            lines.append(f"  - {group}: {info['count']} tasks")
        if not d["task_groups"]:
            lines.append("  (none)")

        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, tasks={len(self.tasks)} tasks)"
