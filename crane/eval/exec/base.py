from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from crane import BrainFeatureExtractor, BrainModel
from crane.eval.plan import ExecutionPlan


@dataclass(frozen=True, slots=True)
class TaskResult:
    group: str
    """Group name."""
    task_id: str
    """Task identifier."""
    metrics: Mapping[str, Any]
    """Mapping of metric names to their values."""
    artifacts: Mapping[str, Any] = field(default_factory=dict)
    """Additional artifacts produced during evaluation."""


class Executor(ABC):
    """Abstract base class for executors that run benchmarks."""

    @abstractmethod
    def run(self, model: BrainModel, featurizer: BrainFeatureExtractor, plan: ExecutionPlan) -> list[TaskResult]:
        """Run the benchmark according to the provided plan.

        Args:
            model: The brain model to evaluate.
            featurizer: The brain feature extractor to use.
            plan: A mapping from (train_fn, train_data) pairs to lists of BoundTasks.

        Returns:
            List of TaskResults containing the evaluation results.
        """
        pass
