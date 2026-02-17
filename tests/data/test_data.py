from pathlib import Path

import pytest

from crane.data.data import CraneDataset
from crane.data.selectors import Subjects, SubjectSessions


def _touch(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


@pytest.fixture
def dataset_dir(tmp_path):
    # Create fake files
    _touch(tmp_path / "sub-001_ses-01.h5")
    _touch(tmp_path / "sub-001_ses-02.h5")
    _touch(tmp_path / "sub-002_ses-01.h5")
    return tmp_path


@pytest.fixture(autouse=True)
def patch_parent(monkeypatch):
    def fake_init(self, **kwargs):
        # store attributes so tests can inspect them
        self.dataset_dir = kwargs["dataset_dir"]
        self._recording_ids = kwargs["recording_ids"]
        self.transform = kwargs["transform"]
        self.keep_files_open = kwargs["keep_files_open"]

    monkeypatch.setattr(
        "torch_brain.dataset.Dataset.__init__",
        fake_init,
    )


def test_load_all(dataset_dir):
    ds = CraneDataset(dataset_dir)
    assert len(ds.recording_ids) == 3


def test_select_subjects(dataset_dir):
    ds = CraneDataset(dataset_dir, select=[1])
    assert all("sub-001_" in rid for rid in ds.recording_ids)
    assert len(ds.recording_ids) == 2

    ds2 = CraneDataset(dataset_dir, select=Subjects(1))
    assert ds.recording_ids == ds2.recording_ids


def test_select_subject_sessions(dataset_dir):
    ds = CraneDataset(dataset_dir, select=[(1, 2)])
    assert ds.recording_ids == ["sub-001_ses-02"]

    ds2 = CraneDataset(dataset_dir, select=SubjectSessions((1, 2)))
    assert ds.recording_ids == ds2.recording_ids


def test_empty_selection_raises(dataset_dir):
    with pytest.raises(ValueError):
        CraneDataset(dataset_dir, select=[999])


def test_mixed_selection_raises(dataset_dir):
    with pytest.raises(ValueError):
        CraneDataset(dataset_dir, select=[999, (1, 1)])


def test_transform_passed(dataset_dir):
    def fn(x):
        return x

    ds = CraneDataset(dataset_dir, transform=fn)
    assert ds.transform is fn
