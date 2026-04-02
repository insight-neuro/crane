import pytest
import torch
from temporaldata import Data
from torch_brain.data.collate import collate

from crane.featurizer import BrainFeature, BrainFeatureExtractor, CraneFeature

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

    datadata = Data(a=1, b=2)  # type: ignore
    bf2 = BrainFeature(datadata)
    assert bf2["a"] == 1
    assert bf2["b"] == 2
    assert hasattr(bf2, "_domain")


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
# CraneFeatures tests
# ---------------------------


def crane_inputs():
    return {
        "brainset": "test_brainset",
        "subject": "test_subject",
        "session": "test_session",
        "signals": torch.randn(64, 1000),  # (batch, channels, samples)
        "channel_labels": [f"ch{i}" for i in range(64)],
        "channel_coordinates": torch.randn(64, 3),
        "sampling_rate": 1000,
    }


def test_crane_feature_one_signal():
    inputs = crane_inputs()
    feature = CraneFeature(**inputs)

    for key in inputs:
        if key == "signals":
            assert key in feature
            assert torch.allclose(feature[key], inputs[key])
        else:
            assert key not in feature

    assert feature.batched is False
    assert feature.channel_dim == 0


def test_crane_feature_batched_signal():
    inputs = crane_inputs()
    inputs["signals"] = torch.randn(10, 64, 1000)  # (batch, channels, samples)

    feature = CraneFeature(**inputs)

    for key in inputs:
        if key == "signals":
            assert key in feature
            assert torch.allclose(feature[key], inputs[key])
        else:
            assert key not in feature

    assert feature.device == inputs["signals"].device
    assert feature.batched is True
    assert feature.channel_dim == 1


def test_crane_feature_channel_mismatch():
    inputs = crane_inputs()
    inputs["channel_labels"] = [f"ch{i}" for i in range(63)]  # mismatch
    with pytest.raises(ValueError):
        CraneFeature(**inputs)


def test_crane_feature_collate():
    batch = [CraneFeature(**crane_inputs()) for _ in range(4)]
    collated = collate(batch)

    inputs = [crane_inputs() for _ in range(4)]

    batch = [CraneFeature(**inp) for inp in inputs]
    collated = collate(batch)
    assert isinstance(collated, CraneFeature)

    expect_inps = inputs[0].copy()
    expect_inps["signals"] = collate([inp["signals"] for inp in inputs])
    expected = CraneFeature(**expect_inps)

    for x, y in zip(collated.values(), expected.values(), strict=True):
        if isinstance(x, torch.Tensor):
            assert torch.allclose(x, y)
        else:
            assert x == y


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
    batch = BrainFeature(some="data")
    result = extractor(batch)

    assert isinstance(result, BrainFeature)
    assert result["processed"] == batch


def test_brain_feature_extractor_is_abstract():
    with pytest.raises(TypeError):  # should raise because forward is not implemented.
        BrainFeatureExtractor()  # type: ignore
