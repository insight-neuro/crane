from collections.abc import Callable

from crane.eval.artifacts import BoundTask
from crane.eval.filter.base import TaskFilter


class LambdaFilter(TaskFilter):
    """
    Filter that selects tasks based on a user-defined function.

    Args:
        func: A function that takes a BoundTask and returns a boolean.
    """

    def __init__(self, func: Callable[[BoundTask], bool]) -> None:
        self.func = func

    def filter(self, tasks: set[BoundTask]) -> set[BoundTask]:
        return {task for task in tasks if self.func(task)}
