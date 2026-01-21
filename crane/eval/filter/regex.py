import re

from ..tasks import BoundTask
from .base import TaskFilter


class MatchGroupsRe(TaskFilter):
    """
    Filter that selects tasks belonging to groups matching any of the specified regular expressions.

    Args:
        groups: List of group names to match.
        regex: If True, treat group names as regular expressions.
    """

    def __init__(self, *groups: str) -> None:
        self.patterns = [re.compile(g) for g in groups]

    def filter(self, tasks: set[BoundTask]) -> set[BoundTask]:
        return {task for task in tasks if any(pat.match(task.group) for pat in self.patterns)}


class MatchTagsRe(TaskFilter):
    """
    Filter that selects tasks that have at least one tag matching any of the specified regular expressions.

    Args:
        regex: If True, treat tags as regular expressions.
    """

    def __init__(self, *tags: str) -> None:
        self.patterns = [re.compile(t) for t in tags]

    def filter(self, tasks: set[BoundTask]) -> set[BoundTask]:
        selected_tasks = set()
        for task in tasks:
            if all(any(pat.match(tag) for tag in task.tags) for pat in self.patterns):
                selected_tasks.add(task)

        return selected_tasks
