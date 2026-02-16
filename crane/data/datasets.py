import re
from fnmatch import fnmatch
from functools import partial
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset

from crane.data.structures import CraneBatch, CraneData


class SingleSessionDataset(Dataset):
    """
    Dataset for a single session of iEEG data. Reads data converted to `ieeg-data` format.

    .. deprecated::
       `SingleSessionDataset` is deprecated, use `CraneDataset` instead. It will be removed in a future release.


    Args:
        session_id (str): A string in the format "brainset/subject/session[from:to]". If [from:to] is not provided, the entire session will be used.
        context_length (float): The length of the context window in seconds.
        data_root_dir (Path | str): Root directory of the data.
    """

    def __init__(self, session_id: str, context_length: float, data_root_dir: Path | str):
        # Parse session_id (brainset/subject/session[from:to])
        m = re.match(r"^([^/]+)/([^/]+)/([^\[\]]+)(?:\[(\d+(?:\.\d+)?):(\d+(?:\.\d+)?)\])?$", session_id)
        if not m:
            raise ValueError(
                f"Invalid session_id format: {session_id}. Expected format: 'brainset/subject/session[from:to]'"
            )
        brainset, subject, session, time_from, time_to = m.groups()

        self.context_length = context_length
        self.data_file = Path(data_root_dir) / brainset / subject / session / "data.h5"

        self.start_times = np.array([])  # list of start times of the context windows

        with h5py.File(self.data_file, "r") as f:
            data = CraneData.from_hdf5(f)

            sampling_rate = int(data.ieeg.sampling_rate)
            data.channels.materialize()

            self.batch_builder = partial(
                CraneBatch,
                brainset=data.brainset,
                subject=data.subject,
                session=data.session,
                citation=data.citation,
                sampling_rate=sampling_rate,
                channels=data.channels,
            )

            time_from = float(time_from) if time_from is not None else data.domain.start[0]  # type: ignore[attr-defined]
            time_to = float(time_to) if time_to is not None else data.domain.end[0]  # type: ignore[attr-defined]

            # TODO: Handle artifacts here. Goal: go over every start_time,
            # if it contains some artifact channels, remove those channels from that time.
            # If as result, too few channels remain, remove the start_time.
            self.start_times = np.arange(time_from, time_to - self.context_length, self.context_length)

    def __len__(self):
        return len(self.start_times)

    def __getitem__(self, idx: int) -> CraneBatch:
        """
        Get a context window of iEEG data as a CraneBatch.

        Args:
            idx (int): The index of the item to return.

        Returns:
            torch.Tensor: Tensor of shape (n_samples, n_channels) for the context window.
        """
        start_time = self.start_times[idx]
        end_time = start_time + self.context_length
        with h5py.File(self.data_file, "r") as f:
            data = CraneData.from_hdf5(f)
            data = data.slice(start_time, end_time)
            data.ieeg.materialize()
            tensor = torch.from_numpy(data.ieeg.data.T)  # shape: (n_samples, n_channels)

        return self.batch_builder(data=tensor)


class MultiSessionDataset(ConcatDataset[SingleSessionDataset]):
    """
    Dataset that concatenates multiple SingleSessionDatasets, with support for glob pattern expansion.
    Assumes all sessions have the same context length, sampling rate, and channel layout.

    .. deprecated::
       `MultiSessionDataset` is deprecated, use `torch_brain.datasets.NestedDataset` instead. It will be removed in a future release.

    Args:
        session_strings (list): List of session strings, potentially containing glob patterns (*, ?, [seq], [!seq])
        context_length (float): Context length in seconds
        data_root_dir (Path | str): Root directory containing the datasets
    """

    datasets: list[SingleSessionDataset]  # type: ignore[override]

    def __init__(self, session_strings: list, context_length: float, data_root_dir: Path | str):
        data_root_dir = Path(data_root_dir)
        super().__init__(
            [
                SingleSessionDataset(session_string, context_length, data_root_dir)
                for session_string in self._expand_session_wildcards(session_strings, data_root_dir)
            ]
        )

    def _discover_dirs(self, path: Path, require_data_h5: bool = False) -> list[str]:
        """Helper to discover directories at a path, optionally requiring data.h5."""
        if not path.exists():
            return []

        dirs = [
            d.name
            for d in path.iterdir()
            if d.is_dir() and not d.name.startswith(".") and (not require_data_h5 or (d / "data.h5").exists())
        ]
        return sorted(dirs)

    def _expand_session_pattern(
        self, brainset_pattern: str, subject_pattern: str, session_pattern: str, root_dir: Path
    ) -> list[str]:
        """Recursively expand glob patterns in brainset/subject/session pattern."""

        # Get brainsets - filter using glob pattern
        all_brainsets = self._discover_dirs(root_dir)
        brainsets = [b for b in all_brainsets if fnmatch(b, brainset_pattern)]

        # Recursively expand subjects and sessions
        expanded: list[str] = []
        for brainset in brainsets:
            all_subjects = self._discover_dirs(root_dir / brainset)
            subjects = [s for s in all_subjects if fnmatch(s, subject_pattern)]
            for subject in subjects:
                all_sessions = self._discover_dirs(root_dir / brainset / subject, require_data_h5=True)
                sessions = [s for s in all_sessions if fnmatch(s, session_pattern)]
                expanded.extend(f"{brainset}/{subject}/{session}" for session in sessions)

        return expanded

    def _expand_session_wildcards(self, session_strings: list[str], root_dir: Path) -> list[str]:
        """
        Expand glob patterns in session strings (format: "brainset/subject/session[time_from:time_to]").
        Glob patterns (e.g., *, ?, [seq], [!seq]) can be used in any position. Time ranges are preserved.

        Examples:
            - "*/sub-01/*" - all brainsets, subject sub-01, all sessions
            - "dataset1/sub-*/ses-1" - dataset1, all subjects starting with 'sub-', session ses-1
            - "*/*/ses-*task-SPESclin*" - all brainsets/subjects, sessions matching the pattern
        """
        expanded: list[str] = []
        for session_string in session_strings:
            time_range = ("[" + session_string.split("[")[1]) if "[" in session_string else ""

            # Parse and expand pattern
            brainset_pattern, subject_pattern, session_pattern = session_string.split("[")[0].split("/")
            matches = self._expand_session_pattern(brainset_pattern, subject_pattern, session_pattern, root_dir)
            expanded.extend(f"{m}{time_range}" for m in matches)

        return sorted(expanded)
