import pytest
from transformers.utils.generic import TensorType

from crane.core.featurizer import BrainFeature, BrainFeatureExtractor
from temporaldata import Data


# ---------------------------
# BrainFeature tests
# ---------------------------

def test_brain_feature_empty_init():
    bf = BrainFeature()
    assert len(bf) == 0


def test_brain_feature_with_data():
    data = {"a": 1, "b": 2}
    bf = BrainFeature(data)
    assert bf["a"] == 1
    assert bf["b"] == 2


def test_brain_feature_with_kwargs_updates_data():
    bf = BrainFeature({"a": 1}, b=2, c=3)
    assert bf["a"] == 1
    assert bf["b"] == 2
    assert bf["c"] == 3


def test_brain_feature_behaves_like_dict():
    bf = BrainFeature(a=1)
    bf["b"] = 2
    assert set(bf.keys()) == {"a", "b"}
    assert bf.get("a") == 1


# ---------------------------
# BrainFeatureExtractor tests
# ---------------------------

class DummyExtractor(BrainFeatureExtractor):
    def __init__(self):
        super().__init__(feature_size=1, sampling_rate=100, padding_value=0)
        
    def forward(self, batch):
        return BrainFeature({"processed": batch})


def test_brain_feature_extractor_call_delegates_to_forward():
    extractor = DummyExtractor()
    batch = Data(some="data")
    result = extractor(batch)

    assert isinstance(result, BrainFeature)
    assert result["processed"] == batch


def test_brain_feature_extractor_is_abstract():
    with pytest.raises(TypeError):
        BrainFeatureExtractor()  # type: ignore - should raise because forward is not implemented
