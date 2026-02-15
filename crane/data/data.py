from collections.abc import Callable, Iterable
from pathlib import Path

from temporaldata import Data
from torch_brain.dataset import Dataset

from crane.data.selectors import SelectNone, Selector, Subjects, SubjectSessions


class CraneDataset(Dataset):
    """Dataset for loading and transforming data from CRANE recordings.

    Args:
        dataset_dir: Path to the directory containing the CRANE recordings.
        select: Optional selection criteria for filtering recordings. Either:
            - Selector instance (e.g., Subjects, SubjectSessions, or combinations thereof)
            - List of subject IDs (e.g., [1, 2, 3])
            - List of (subject ID, session ID) pairs (e.g., [(1, 1), (2, 1), (3, 2)])
            - None (default) to include all recordings.
        transform: Optional function to apply to each loaded Data object.
        keep_files_open: Whether to keep HDF5 files open for faster access (default: True).
    """

    def __init__(
        self,
        dataset_dir: str | Path,
        select: Selector | Iterable[int | tuple[int, int]] | None = None,
        transform: Callable[[Data], Data] | None = None,
        keep_files_open: bool = True,
    ):
        dataset_dir = Path(dataset_dir)
        subject_sessions = sorted(x.stem for x in dataset_dir.glob("sub-*_ses-*.h5"))

        #  Apply selection criteria
        if select is not None:
            if not isinstance(select, Selector):
                select = list(select)
                if not select:
                    select = SelectNone()
                elif all(isinstance(x, str | int) for x in select):
                    select = Subjects(*select)  # type: ignore
                elif all(isinstance(x, tuple) and len(x) == 2 for x in select):
                    select = SubjectSessions(*select)  # type: ignore
                else:
                    raise ValueError("select must be Selector, list[int], or list[(sub, ses)]")

            subject_sessions = [s for s in subject_sessions if select.match(s)]

        super().__init__(
            dataset_dir=dataset_dir,
            recording_ids=subject_sessions,
            transform=transform,
            keep_files_open=keep_files_open,
            namespace_attributes=["session.id", "subject.id", "channels.id"],
        )
