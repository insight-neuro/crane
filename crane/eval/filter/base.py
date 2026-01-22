from abc import ABC, abstractmethod

from crane.eval.artifacts import BoundTask


class TaskFilter(ABC):
    """Abstract base class for filtering evaluation tasks."""

    @abstractmethod
    def filter(self, tasks: set[BoundTask]) -> set[BoundTask]:
        """Filter the provided tasks based on selection criteria.

        Args:
            tasks: A set of BoundTasks to select from.

        Returns:
            A subset of the provided tasks that meet the selection criteria.
        """
        ...
