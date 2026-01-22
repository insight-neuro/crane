from collections import defaultdict

from crane.eval.artifacts import BoundTask, ExecutionPlan, ExecutionUnit, TrainFn
from crane.eval.data import NeuralData
from crane.eval.plan.base import Planner


class GroupedPlanner(Planner):
    """Planner that groups tasks by their training configuration."""

    def build_plan(self, tasks: list[BoundTask]) -> ExecutionPlan:
        plan_dict: dict[tuple[TrainFn, NeuralData], list[BoundTask]] = defaultdict(list)

        for task in tasks:
            key = (task.train_fn, task.train)
            plan_dict[key].append(task)

        execution_units: list[ExecutionUnit] = []

        for (train_fn, train), task_batch in plan_dict.items():
            # Sort tasks in each group by test data to maximize caching
            sorted_tasks = sorted(task_batch, key=lambda t: id(t.test))
            execution_units.append(
                ExecutionUnit.from_tasks(
                    train_fn=train_fn,
                    train_data=train,
                    tasks=sorted_tasks,
                )
            )

        # Sort execution units by train key for consistency
        execution_units.sort(key=lambda eu: eu.train_key)

        return ExecutionPlan(plan=tuple(execution_units))
