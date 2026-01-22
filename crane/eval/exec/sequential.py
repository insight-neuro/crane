import torch

from crane import BrainFeatureExtractor, BrainModel
from crane.eval.artifacts import ExecutionPlan, TaskResult
from crane.eval.exec.base import Executor


class SequentialExecutor(Executor):
    """Executor that runs tasks sequentially."""

    def run(
        self,
        model: BrainModel,
        featurizer: BrainFeatureExtractor,
        plan: ExecutionPlan,
    ) -> list[TaskResult]:
        results: list[TaskResult] = []

        for task_unit in plan:
            trained_model = task_unit.train_fn(model, featurizer, task_unit.train_data)

            with torch.no_grad():
                for task in task_unit:
                    results.append(task.test_fn(trained_model, featurizer, task.test))

        return results
