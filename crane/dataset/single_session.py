import logging
import os

import h5py
import numpy as np
import torch
from temporaldata import Data
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class SingleSessionDataset(Dataset):
    """
    Dataset for a single session of iEEG data.

    Args:
        session_string (str): A string in the format "brainset/subject/session[from:to]". If [from:to] is not provided, the entire session will be used.
        context_length (float): The length of the context window in seconds.
    """

    def __init__(self, session_string: str, context_length: float):
        self.context_length = context_length
        self.session_string = session_string.split("[")[0]  # "brainset/subject/session"

        self.brainset, self.subject, self.session = self.session_string.split("/")
        self.data_file = os.path.join(
            os.environ["DATA_ROOT_DIR"],
            self.brainset,
            self.subject,
            self.session,
            "data.h5",
        )
        self.start_times = []  # list of start times of the context windows

        with h5py.File(self.data_file, "r") as f:
            data = Data.from_hdf5(f)

            if "[" in session_string:
                self.time_from, self.time_to = session_string.split("[")[1][:-1].split(":")
                self.time_from, self.time_to = (
                    float(self.time_from),
                    float(self.time_to),
                )
            else:
                self.time_from, self.time_to = data.domain.start[0], data.domain.end[0]

            # TODO: Handle artifacts here. Goal: go over every start_time, if it contians some artifact channels, remove those channels from that time.
            # If as result, too few channels remain, remove the start_time.
            self.start_times = np.arange(self.time_from, self.time_to - self.context_length, self.context_length)

    def __len__(self):
        return len(self.start_times)

    def __getitem__(self, idx: int) -> dict:
        """
        Return a dictionary containing the iEEG data, channels, and metadata for the given index.

        Args:
            idx (int): The index of the item to return.

        Returns:
            dict: A dictionary containing the iEEG data, channels, and metadata for the given index. Keys: "ieeg": {data: torch.Tensor[n_channels, n_samples], sampling_rate: int}, "channels": {id: np.array}, "metadata": {brainset: str, subject: str, session: str}.
        """
        start_time = self.start_times[idx]
        end_time = start_time + self.context_length
        with h5py.File(self.data_file, "r") as f:
            data = Data.from_hdf5(f)
            data = data.slice(start_time, end_time).materialize()

        return {
            "ieeg": {
                "data": torch.from_numpy(data.ieeg.data.T),  # shape: (n_channels, n_samples)
                "sampling_rate": int(data.ieeg.sampling_rate),
            },
            "channels": {"id": data.channels.id},
            "metadata": {
                "brainset": data.brainset,
                "subject": data.subject,
                "session": data.session,
            },
        }
