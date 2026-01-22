from abc import ABC, abstractmethod

from crane import BrainFeatureExtractor, BrainModel
from crane.eval.artifacts import ExecutionPlan, TaskResult


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
        ...
