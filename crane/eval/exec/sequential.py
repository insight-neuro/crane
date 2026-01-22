import torch

from crane import BrainFeatureExtractor, BrainModel
from crane.eval.artifacts import ExecutionPlan, TaskResult
from crane.eval.bench import BrainBench
from crane.eval.exec.base import Executor
from crane.eval.protocols import TestFn, TrainFn


class SequentialExecutor(Executor):
    """Executor that runs tasks sequentially."""

    def run(
        self,
        bench: BrainBench,
        model: BrainModel,
        featurizer: BrainFeatureExtractor,
        plan: ExecutionPlan,
    ) -> list[TaskResult]:
        results: list[TaskResult] = []
        
        train_fn: TrainFn = bench.train_fn
        test_fn: TestFn = bench.test_fn

        for task_unit in plan:
            trained_model = train_fn(bench, model, featurizer, task_unit.train_data)

            with torch.no_grad():
                for task in task_unit:
                    results.append(test_fn(bench, trained_model, featurizer, task.test))
        return results
