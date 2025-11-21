import os
from fnmatch import fnmatch
from pathlib import Path

from torch.utils.data import ConcatDataset

from .single_session import SingleSessionDataset


class MultiSessionDataset(ConcatDataset):
    """
    Dataset that concatenates multiple SingleSessionDatasets, with support for glob pattern expansion.


    Args:
        session_strings (list): List of session strings, potentially containing glob patterns (*, ?, [seq], [!seq])
        context_length (float): Context length in seconds
    """

    def __init__(self, session_strings: list, context_length: float):
        super().__init__([SingleSessionDataset(session_string, context_length) for session_string in self._expand_session_wildcards(session_strings)])

    # Code that can be used to discover directories in the dataset
    def _discover_dirs(self, path, require_data_h5=False):
        """Helper to discover directories at a path, optionally requiring data.h5."""
        path = Path(path)
        if not path.exists():
            return []
        dirs = [d.name for d in path.iterdir() if d.is_dir() and not d.name.startswith(".") and (not require_data_h5 or (d / "data.h5").exists())]
        return sorted(dirs)

    def _expand_session_pattern(self, data_root, brainset_pattern, subject_pattern, session_pattern):
        """Recursively expand glob patterns in brainset/subject/session pattern."""
        data_root = Path(data_root)

        # Get brainsets - filter using glob pattern
        all_brainsets = self._discover_dirs(data_root)
        brainsets = [b for b in all_brainsets if fnmatch(b, brainset_pattern)]

        # Recursively expand subjects and sessions
        expanded = []
        for brainset in brainsets:
            all_subjects = self._discover_dirs(data_root / brainset)
            subjects = [s for s in all_subjects if fnmatch(s, subject_pattern)]
            for subject in subjects:
                all_sessions = self._discover_dirs(data_root / brainset / subject, require_data_h5=True)
                sessions = [s for s in all_sessions if fnmatch(s, session_pattern)]
                expanded.extend(f"{brainset}/{subject}/{session}" for session in sessions)

        return expanded

    def _expand_session_wildcards(self, session_strings: list) -> list:
        """
        Expand glob patterns in session strings (format: "brainset/subject/session[time_from:time_to]").
        Glob patterns (e.g., *, ?, [seq], [!seq]) can be used in any position. Time ranges are preserved.

        Examples:
            - "*/sub-01/*" - all brainsets, subject sub-01, all sessions
            - "dataset1/sub-*/ses-1" - dataset1, all subjects starting with 'sub-', session ses-1
            - "*/*/ses-*task-SPESclin*" - all brainsets/subjects, sessions matching the pattern
        """
        data_root = Path(os.environ.get("DATA_ROOT_DIR", ""))
        if not data_root.exists():
            raise ValueError("DATA_ROOT_DIR environment variable must be set")

        expanded: list[str] = []
        for session_string in session_strings:
            time_range = ("[" + session_string.split("[")[1]) if "[" in session_string else ""

            # Parse and expand pattern
            brainset_pattern, subject_pattern, session_pattern = session_string.split("[")[0].split("/")
            matches = self._expand_session_pattern(data_root, brainset_pattern, subject_pattern, session_pattern)
            expanded.extend(f"{m}{time_range}" for m in matches)

        return sorted(expanded)
