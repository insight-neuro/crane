from collections.abc import Callable
from typing import Concatenate, ParamSpec

import torch

from crane import BrainFeatureExtractor, BrainModel
from crane.eval.data import NeuralLabeledData

from ..artifacts import TaskResult
from ..bench import BrainBench

P = ParamSpec("P")

Metric = Callable[Concatenate[torch.Tensor, torch.Tensor, P], torch.Tensor]


class ZeroShotEval:
    """Zero-shot evaluation test function.

    Args:
        metrics (dict[str, Metric]): A dictionary mapping metric names to metric functions.
            Each metric function should take predictions and labels as input and return a float score.
    """

    def __init__(self, metrics: dict[str, Metric]):
        self.metrics = metrics

    def __call__(
        self, bench: BrainBench, model: BrainModel, featurizer: BrainFeatureExtractor, test_data: NeuralLabeledData
    ) -> TaskResult:
        data = test_data.data
        labels = test_data.labels

        features = featurizer(data)
        predictions = model(**features)

        results: dict[str, float] = {}
        for metric, fn in self.metrics.items():
            results[metric] = fn(predictions, labels).item()

        return TaskResult(
            task_fn="zero_shot",
            metrics=results,
        )
