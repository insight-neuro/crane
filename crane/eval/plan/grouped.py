from collections import defaultdict
from collections.abc import Callable

from ..tasks import BoundTask
from .base import ExecutionPlan, ExecutionUnit, Planner


class GroupedPlanner(Planner):
    """Planner that groups tasks by their training configuration."""

    def build_plan(self, tasks: list[BoundTask]) -> ExecutionPlan:
        plan_dict: dict[tuple[Callable, object], list[BoundTask]] = defaultdict(list)

        for task in tasks:
            key = (task.train_fn, task.train)
            plan_dict[key].append(task)

        execution_units = [
            ExecutionUnit(
                train_key=f"{train_fn.__name__}_{id(train)}",
                train_fn=train_fn,
                train_data=train,
                tasks=tuple(task_batch),
            )
            for (train_fn, train), task_batch in plan_dict.items()
        ]
        for (train_fn, train), task_batch in plan_dict.items():
            # Sort tasks in each group by test data to maximize caching
            sorted_tasks = sorted(task_batch, key=lambda t: id(t.test))
            execution_units.append(
                ExecutionUnit(
                    train_key=f"{train_fn.__name__}_{id(train)}",
                    train_fn=train_fn,
                    train_data=train,
                    tasks=tuple(sorted_tasks),
                )
            )

        # Sort execution units by train key for consistency
        execution_units.sort(key=lambda eu: eu.train_key)

        return ExecutionPlan(
            benchmark="GroupedBenchmark",
            plan=tuple(execution_units),
        )
