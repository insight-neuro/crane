from abc import ABC, abstractmethod

from crane import BrainFeatureExtractor, BrainModel
from crane.eval.artifacts import ExecutionPlan, TaskResult
from crane.eval.bench import BrainBench


class Executor(ABC):
    """Abstract base class for executors that run benchmarks."""

    @abstractmethod
    def run(
        self,
        bench: BrainBench,
        model: BrainModel,
        featurizer: BrainFeatureExtractor,
        plan: ExecutionPlan,
    ) -> list[TaskResult]:
        """Run the benchmark according to the provided plan.

        Args:
            bench: The BrainBench being executed.
            model: The brain model to evaluate.
            featurizer: The brain feature extractor to use.
            plan: The execution plan defining how to run the benchmark.

        Returns:
            List of TaskResults containing the evaluation results.
        """
        ...
