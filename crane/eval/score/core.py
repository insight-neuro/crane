from crane import BrainFeatureExtractor, BrainModel

from ..artifacts import TaskResult
from ..bench import BrainBench
from ..data import NeuralData


def zero_shot(
    bench: BrainBench, model: BrainModel, featurizer: BrainFeatureExtractor, test_data: NeuralData
) -> TaskResult:
    data = test_data.data
    labels = test_data.labels

    features = featurizer(data)
    predictions = model(**features)

    results: dict[str, float] = {}
    for metric, fn in bench.default_metrics.items():
        results[metric] = fn(predictions, labels)

    return TaskResult(
        fn="zero_shot",
        metrics=results,
    )
