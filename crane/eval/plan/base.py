from abc import ABC, abstractmethod

from crane.eval.artifacts import ExecutionPlan
from crane.eval.tasks import Task


class Planner(ABC):
    """Abstract base class for planners that create evaluation plans."""

    @abstractmethod
    def build_plan(self, tasks: list[Task]) -> ExecutionPlan:
        """Create an evaluation plan from the provided tasks.

        Args:
            tasks: A list of Tasks to include in the plan.

        Returns:
            An ExecutionPlan detailing how to execute the tasks.
        """
        ...
