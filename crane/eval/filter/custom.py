from collections.abc import Callable

from crane.eval.filter.base import TaskFilter
from crane.eval.tasks import Task


class LambdaFilter(TaskFilter):
    """
    Filter that selects tasks based on a user-defined function.

    Args:
        func: A function that takes a Task and returns a boolean.
    """

    def __init__(self, func: Callable[[Task], bool]) -> None:
        self.func = func

    def filter(self, tasks: set[Task]) -> set[Task]:
        return {task for task in tasks if self.func(task)}
