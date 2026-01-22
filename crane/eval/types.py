from collections.abc import Callable

from crane.core import BrainFeatureExtractor, BrainModel
from crane.eval.artifacts import TaskResult
from crane.eval.bench import BrainBench
from crane.eval.data import NeuralData

type TrainFn = Callable[[BrainBench, BrainModel, BrainFeatureExtractor, NeuralData], BrainModel]
type TestFn = Callable[[BrainBench, BrainModel, BrainFeatureExtractor, NeuralData], TaskResult]
