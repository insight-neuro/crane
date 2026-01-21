from typing import Any

import pytest
import torch

from crane.core import BrainFeatureExtractor, BrainModel, BrainOutput
from crane.eval.decorators import (
    TaskDescriptor,
    _build_descriptor,
    eval,
    finetune,
)
from crane.eval.sweep import Sweep

sweep1 = Sweep(name="sweep1", values_or_fn=[1, 2, 3])
sweep2 = Sweep(name="sweep2", values_or_fn=["a", "b"])


class DummyModel(BrainModel):
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def forward(self, batch: dict, *args, **kwargs):
        return BrainOutput(last_hidden_state=torch.tensor([]))


@pytest.fixture
def task_descriptor():
    return TaskDescriptor(
        role="eval",
        name="dummy_task",
        uses=None,
        tags={"test", "dummy"},
        sweeps={
            "sweep1": sweep1,
            "sweep2": sweep2,
        },
        static={"param1": 42, "param2": "value"},
        fn=lambda instance, model, extractor, **kwargs: {"score": 0.95},
    )


def test_default_task_descriptor(task_descriptor):
    assert task_descriptor.role == "eval"
    assert task_descriptor.name == "dummy_task"
    assert task_descriptor.static["param1"] == 42
    assert task_descriptor.fn(None, None, None)["score"] == 0.95


def test_expand_task_descriptor(task_descriptor):
    dummy_obj = object()
    specs = task_descriptor.expand(dummy_obj)
    assert len(specs) == 6  # 3 values in sweep1 * 2 values in sweep2
    spec = specs[0]

    assert spec.name == "dummy_task[sweep1=1,sweep2=a]"
    assert spec.role == "eval"
    assert spec.fn(None, None)["score"] == 0.95

    for spec in specs:
        assert spec.role == "eval"
        assert spec.base == "dummy_task"


def test_build_descriptor(task_descriptor):
    built_descriptor = _build_descriptor(
        role="eval",
        name="dummy_task",
        uses=None,
        tags=["test", "dummy"],
        kwargs={
            "sweep1": sweep1,
            "sweep2": sweep2,
            "param1": 42,
            "param2": "value",
        },
        fn=task_descriptor.fn,
    )
    assert built_descriptor == task_descriptor
    assert built_descriptor.fn(None, None, None)["score"] == 0.95


def test_task_spec_name(task_descriptor):
    spec = task_descriptor.expand(object())[0]
    assert "sweep1=1" in spec.name
    assert "sweep2=a" in spec.name


def test_build_descriptor_tags():
    built_descriptor = _build_descriptor(
        role="eval",
        name="dummy_task",
        uses=None,
        tags=None,
        kwargs={},
        fn=lambda instance, model, extractor, **kwargs: {"score": 0.95},
    )
    assert built_descriptor.tags == set()


def test_build_descriptor_raise():
    with pytest.raises(ValueError):
        _build_descriptor(
            role="eval",
            name="dummy_task",
            uses=None,
            tags=["test"],
            kwargs={"sweep": Sweep(name="not_sweep", values_or_fn=[1, 2])},
            fn=lambda instance, model, extractor, **kwargs: {"score": 0.95},
        )


def test_finetune_decorator():
    @finetune("finetune_task", param=10)
    @finetune("finetune_task_2", param=20)
    def finetune_fn(instance: object, model: BrainModel, extractor: BrainFeatureExtractor, param: int) -> BrainModel:
        return DummyModel(param=param)

    assert hasattr(finetune_fn, "__bench_tasks__")
    assert len(finetune_fn.__bench_tasks__) == 2  # type: ignore

    names = {desc.name for desc in finetune_fn.__bench_tasks__}  # type: ignore
    assert names == {"finetune_task", "finetune_task_2"}

    assert finetune_fn.__bench_tasks__[0].fn(None, None, None, 10).param in {10, 20}  # type: ignore
    assert finetune_fn.__bench_tasks__[1].fn(None, None, None, 20).param in {10, 20}  # type: ignore


def test_eval_decorator():
    @eval("eval_task", uses="resource", tags=["tag1", "tag2"], sweep=Sweep(name="sweep", values_or_fn=[True, False]))
    def eval_fn(instance: object, model: BrainModel, extractor: BrainFeatureExtractor, sweep: bool) -> dict[str, Any]:
        return {"result": sweep}

    assert hasattr(eval_fn, "__bench_tasks__")
    assert len(eval_fn.__bench_tasks__) == 1  # type: ignore
    descriptor = eval_fn.__bench_tasks__[0]  # type: ignore
    assert descriptor.name == "eval_task"
    assert descriptor.role == "eval"
    assert descriptor.uses == "resource"
    assert descriptor.tags == {"tag1", "tag2"}
    assert descriptor.sweeps["sweep"].name == "sweep"
    assert descriptor.fn(None, None, None, True)["result"] is True
    assert descriptor.fn(None, None, None, False)["result"] is False
