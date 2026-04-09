import pytest

from crane.preprocess.subset import subset_electrodes


def test_subset_max_n_electrodes(make_batch):
    batch = make_batch(n_elec=10)
    max_n_electrodes = 5
    out = subset_electrodes(batch, max_n_electrodes=max_n_electrodes)
    assert out.signals.shape[1] == max_n_electrodes
    assert len(out.channel_labels) == max_n_electrodes
    assert out.channel_coordinates.shape[0] == max_n_electrodes

    max_n_electrodes = 15
    out = subset_electrodes(batch, max_n_electrodes=max_n_electrodes)
    assert out.signals.shape[1] == 10  # no change since max_n_electrodes > n
    assert len(out.channel_labels) == 10
    assert out.channel_coordinates.shape[0] == 10


def test_subset_electrodes_indices(make_batch):
    batch = make_batch(n_elec=10)
    subset = [0, 2, 4]
    out = subset_electrodes(batch, subset=subset)
    assert out.signals.shape[1] == len(subset)
    assert len(out.channel_labels) == len(subset)
    assert out.channel_coordinates.shape[0] == len(subset)
    assert out.channel_labels == [batch.channel_labels[i] for i in subset]


def test_subset_electrodes_ids(make_batch):
    batch = make_batch(n_elec=10)
    subset = [batch.channel_labels[i] for i in [1, 3, 5]]
    out = subset_electrodes(batch, subset=subset)
    assert out.signals.shape[1] == len(subset)
    assert len(out.channel_labels) == len(subset)
    assert out.channel_coordinates.shape[0] == len(subset)
    assert out.channel_labels == subset


def test_subset_electrodes_unbatched(make_batch):
    batch = make_batch(batch_size=1, n_elec=10)
    subset = [0, 2, 4]
    out = subset_electrodes(batch, subset=subset)
    assert out.signals.shape[0] == len(subset)
    assert len(out.channel_labels) == len(subset)
    assert out.channel_coordinates.shape[0] == len(subset)
    assert out.channel_labels == [batch.channel_labels[i] for i in subset]


def test_subset_electrodes_invalid_input(make_batch):
    batch = make_batch(n_elec=10)
    with pytest.raises(ValueError):
        subset_electrodes(batch)  # neither provided
    with pytest.raises(ValueError):
        subset_electrodes(batch, max_n_electrodes=5, subset=[0, 1])  # both provided
