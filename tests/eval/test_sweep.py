import pytest

from crane.eval.sweep import Sweep, sweep


def test_sweep_resolve():
    s1 = sweep("lr", [0.001, 0.01, 0.1])
    values1 = s1.resolve()
    assert values1 == (0.001, 0.01, 0.1)

    s2 = sweep("batch_size", lambda: [16, 32, 64])
    values2 = s2.resolve()
    assert values2 == (16, 32, 64)

    s3 = sweep("empty", [])  # No error yet

    with pytest.raises(ValueError, match="no values"):
        s3.resolve()

    s4 = sweep("duplicates", [1, 2, 2])
    with pytest.raises(ValueError, match="duplicate values"):
        s4.resolve()


def test_sweep_constructors_equiv():
    s1 = sweep("param", [1, 2, 3])
    s2 = Sweep(name="param", values_or_fn=[1, 2, 3])
    assert s1 == s2
