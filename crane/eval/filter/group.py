from crane.eval.filter.base import TaskFilter
from crane.eval.tasks import Task


class MatchGroups(TaskFilter):
    """
    Filter that selects tasks belonging to a list of groups.

    Args:
        groups: List of group names to match.
    """

    def __init__(self, *groups: str) -> None:
        self.groups = set(groups)

    def filter(self, tasks: set[Task]) -> set[Task]:
        return {task for task in tasks if task.group in self.groups}


class MatchTags(TaskFilter):
    """
    Filter that selects tasks containing any of the specified tags.

    Args:s
        tags: List of tags to match.
        match_all: If True, task must contain all tags to be selected.
    """

    def __init__(self, *tags: str, match_all: bool = False) -> None:
        self.tags = set(tags)

        if match_all:
            self.check = lambda task_tags: self.tags.issubset(task_tags)
        else:
            self.check = lambda task_tags: self.tags.intersection(task_tags)

    def filter(self, tasks: set[Task]) -> set[Task]:
        selected_tasks = set()
        for task in tasks:
            if self.check(task.tags):
                selected_tasks.add(task)
        return selected_tasks
