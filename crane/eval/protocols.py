from typing import Protocol, runtime_checkable

from crane.core import BrainFeatureExtractor, BrainModel
from crane.eval.artifacts import TaskResult
from crane.eval.bench import BrainBench
from crane.eval.data import NeuralData


@runtime_checkable
class TrainFn(Protocol):
    def __call__(
        self,
        bench: BrainBench,
        model: BrainModel,
        featurizer: BrainFeatureExtractor,
        train_data: NeuralData,
    ) -> BrainModel: ...


@runtime_checkable
class TestFn(Protocol):
    def __call__(
        self,
        bench: BrainBench,
        model: BrainModel,
        featurizer: BrainFeatureExtractor,
        test_data: NeuralData,
    ) -> TaskResult: ...
