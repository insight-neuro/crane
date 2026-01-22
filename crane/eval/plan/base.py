from abc import ABC, abstractmethod

from crane.eval.artifacts import BoundTask, ExecutionPlan


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
        ...
