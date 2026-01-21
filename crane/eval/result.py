from collections import defaultdict
from dataclasses import dataclass

from crane.eval.exec.base import TaskResult


@dataclass(frozen=True, slots=True)
class RunResult:
    benchmark: str
    """Benchmark name."""
    version: str | None
    """Benchmark version."""
    model: str
    """Evaluated model name."""
    task_results: tuple[TaskResult, ...]
    """Results for individual tasks."""

    def __len__(self) -> int:
        return len(self.task_results)

    def by_group(self) -> dict[str, list[TaskResult]]:
        """Group task results by their group name."""
        out: dict[str, list[TaskResult]] = defaultdict(list)
        for result in self.task_results:
            out[result.group].append(result)
        return out
