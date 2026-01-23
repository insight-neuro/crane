from typing import Protocol, TypeVar, runtime_checkable

from crane.core import BrainFeatureExtractor, BrainModel
from crane.eval.artifacts import TaskResult
from crane.eval.bench import BrainBench
from crane.eval.data import NeuralData

ND = TypeVar("ND", bound=NeuralData, contravariant=True)


@runtime_checkable
class TrainFn(Protocol[ND]):
    def __call__(
        self,
        bench: BrainBench,
        model: BrainModel,
        featurizer: BrainFeatureExtractor,
        train_data: ND,
    ) -> BrainModel: ...


@runtime_checkable
class TestFn(Protocol[ND]):
    def __call__(
        self,
        bench: BrainBench,
        model: BrainModel,
        featurizer: BrainFeatureExtractor,
        test_data: ND,
    ) -> TaskResult: ...
