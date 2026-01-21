from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

from ..tasks import BoundTask


@dataclass(frozen=True, slots=True)
class ExecutionUnit:
    train_key: str
    train_fn: Callable
    train_data: object
    tasks: tuple[BoundTask, ...]

    def __len__(self) -> int:
        return len(self.tasks)

    def __iter__(self):
        return iter(self.tasks)


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    benchmark: str
    plan: tuple[ExecutionUnit, ...]

    def __len__(self) -> int:
        return len(self.plan)

    def __iter__(self):
        return iter(self.plan)


class Planner(ABC):
    """Abstract base class for planners that create evaluation plans."""

    @abstractmethod
    def build_plan(self, tasks: list[BoundTask]) -> ExecutionPlan:
        """Create an evaluation plan from the provided tasks.

        Args:
            tasks: A list of BoundTasks to include in the plan.

        Returns:
            An ExecutionPlan detailing how to execute the tasks.
        """
        pass
