from abc import ABC, abstractmethod

from crane.eval.tasks import Task


class TaskFilter(ABC):
    """Abstract base class for filtering evaluation tasks."""

    @abstractmethod
    def filter(self, tasks: set[Task]) -> set[Task]:
        """Filter the provided tasks based on selection criteria.

        Args:
            tasks: A set of Tasks to select from.
        Returns:
            A subset of the provided tasks that meet the selection criteria.
        """
        ...
