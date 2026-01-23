from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import replace

from crane import BrainFeatureExtractor, BrainModel
from crane.eval.artifacts import ExecutionPlan, TaskResult
from crane.eval.bench import BrainBench
from crane.eval.tasks import Task


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


class TestRunner:
    """Helper runner class to execute benchmarks with a given test function."""

    def __init__(self, bench: BrainBench, featurizer: BrainFeatureExtractor) -> None:
        self.bench = bench
        self.featurizer = featurizer

    def run(self, model: BrainModel, tasks: Sequence[Task]) -> list[TaskResult]:
        """Run the given tasks and return their results.

        Args:
            model (BrainModel): The brain model to evaluate.
            tasks (Sequence[Task]): The tasks to run.

        Returns:
            list[TaskResult]: The results of running the tasks.
        """
        results: list[TaskResult] = []
        test_fn = self.bench.test_fn

        for task in tasks:
            result = test_fn(self.bench, model, self.featurizer, task.test)
            bound = replace(result, group=task.group, task_id=id(task))
            results.append(bound)

        return results
