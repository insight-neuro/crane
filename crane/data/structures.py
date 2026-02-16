from dataclasses import dataclass

import numpy as np
import torch
from temporaldata import ArrayDict, Data, RegularTimeSeries


class RawData(RegularTimeSeries):
    data: np.ndarray
    """Data array of shape (n_channels, n_samples)"""


class ChannelDict(ArrayDict):  # shape: (n_channels,)
    id: np.ndarray
    """Channel ID"""
    x: np.ndarray
    """x coordinate"""
    y: np.ndarray
    """y coordinate"""
    z: np.ndarray
    """z coordinate"""
    type: np.ndarray
    """Channel type (e.g., 'ecog', 'seeg')"""
    brain_area: np.ndarray
    """Brain area"""

    def __getitem__(self, key):  # Splicing support
        return ChannelDict(**{field: getattr(self, field)[key] for field in self.__annotations__})


class CraneData(Data):
    brainset: str
    """ID of the brainset"""
    subject: str
    """ID of the subject"""
    session: str
    """ID of the session"""
    citation: str
    """Citation for the dataset (bibtex format)"""
    ieeg: RawData
    """iEEG data"""
    channels: ChannelDict
    """Channel metadata"""

    def to_crane_batch(self, device: torch.device | str | None = None) -> "CraneBatch":
        """
        Convert CraneData to CraneBatch for use in PyTorch.

        Args:
            device (torch.device | str | None): Device to move the tensor to. If None, no movement is done.

        Returns:
            CraneBatch: Converted batch.
        """
        data_tensor = torch.from_numpy(self.ieeg.data.T)  # shape: (n_samples, n_channels)
        if device is not None:
            data_tensor = data_tensor.to(device)

        return CraneBatch(
            brainset=self.brainset,
            subject=self.subject,
            session=self.session,
            citation=self.citation,
            sampling_rate=int(self.ieeg.sampling_rate),
            data=data_tensor,
            channels=self.channels,
        )


@dataclass
class CraneBatch:
    """
    Batch of iEEG data for PyTorch models. Data tensor shape:
    (n_samples, batch_size, n_channels), can be accessed as `batch.data`.

    .. deprecated::
    """

    brainset: str
    """ID of the brainset"""
    subject: str
    """ID of the subject"""
    session: str
    """ID of the session"""
    citation: str
    """Citation for the dataset (bibtex format)"""
    sampling_rate: int
    """Sampling rate of the iEEG data"""
    data: torch.Tensor
    """Data tensor of shape (n_samples, batch_size, n_channels)"""
    channels: ChannelDict
    """Channel metadata"""

    @property
    def n_samples(self) -> int:
        """Number of samples in the data tensor."""
        return self.data.shape[0]

    @property
    def n_channels(self) -> int:
        """Number of channels in the data tensor."""
        return self.channels.id.shape[0]
