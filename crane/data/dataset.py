from collections.abc import Callable, Iterable
from pathlib import Path
from typing import override

import torch
from temporaldata import Data
from torch_brain.dataset import Dataset, DatasetIndex

from crane.data.selectors import Selector, Subjects, SubjectSessions
from crane.featurizer import BrainFeature, BrainFeatureExtractor, CraneFeature


class CraneDataset(Dataset):
    """Dataset for loading and transforming Crane-formatted neural recordings.

    Args:
        dataset_dir: Path to the directory containing the neural recordings.
        select: Optional selection criteria for filtering recordings. Either:
            - Selector instance (e.g., Subjects, SubjectSessions, or combinations thereof)
            - List of subject IDs (e.g., [1, 2, 3])
            - List of (subject ID, session ID) pairs (e.g., [(1, 1), (2, 1), (3, 2)])
            - None (default) to include all recordings.
        transform: Optional function to apply to each loaded Data object.
        featurizer: Optional BrainFeatureExtractor to apply to each loaded Data object.
        keep_files_open: Whether to keep HDF5 files open for faster access (default: True).
    """

    def __init__(
        self,
        dataset_dir: str | Path,
        select: Selector | Iterable[int | tuple[int, int]] | None = None,
        transform: Callable[[Data], Data] | None = None,
        featurizer: BrainFeatureExtractor | None = None,
        keep_files_open: bool = True,
    ):
        dataset_dir = Path(dataset_dir)
        subject_sessions = sorted(x.stem for x in dataset_dir.glob("sub-*_ses-*.h5"))
        self.featurizer = featurizer

        #  Apply selection criteria
        if select is not None:
            if not isinstance(select, Selector):
                select = list(select)
                if all(isinstance(x, (str, int)) for x in select):
                    select = Subjects(*select)  # type: ignore
                elif all(isinstance(x, tuple) and len(x) == 2 for x in select):
                    select = SubjectSessions(*select)  # type: ignore
                else:
                    raise ValueError("select must be Selector, list[int], or list[(sub, ses)]")

            subject_sessions = [s for s in subject_sessions if select.match(s)]

        if not subject_sessions:
            raise ValueError(
                "No recordings found matching the selection criteria. Must be of the form 'sub-<id>_ses-<id>*.h5' with valid selection criteria."
            )

        super().__init__(
            dataset_dir=dataset_dir,
            recording_ids=subject_sessions,
            transform=transform,
            keep_files_open=keep_files_open,
            namespace_attributes=["session.id", "subject.id", "channels.id"],
        )

    @override
    def __getitem__(self, index: DatasetIndex) -> BrainFeature:  # type: ignore[override]
        data = super().__getitem__(index)

        signals = torch.from_numpy(data.signals.data.T).float()  # type: ignore[attr-defined]
        coords = torch.from_numpy(data.channel_coordinates).float()  # type: ignore[attr-defined]
        feat = CraneFeature(
            brainset=data.brainset.id,  # type: ignore[attr-defined]
            subject=data.subject.id,  # type: ignore[attr-defined]
            session=data.session.id,  # type: ignore[attr-defined]
            signals=signals,
            channel_labels=data.channel_labels,  # type: ignore[attr-defined]
            channel_coordinates=coords,
            sampling_rate=data.signals.sampling_rate,  # type: ignore[attr-defined]
        )

        if self.featurizer is not None:
            feat = self.featurizer(feat)
        return feat
