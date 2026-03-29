from abc import ABC, abstractmethod
from typing import Any

import torch
from jaxtyping import Float
from temporaldata import Data
from torch import Tensor
from transformers import BatchFeature, SequenceFeatureExtractor
from transformers.utils.generic import TensorType


class BrainFeature(BatchFeature):
    """
    BatchFeature subclass for brain data. Can be used to store features extracted from neural recordings.

    This class is derived from a python dictionary and can be used as a dictionary.
    """

    def __init__(
        self,
        data: dict[str, Any] | Data | None = None,
        /,
        tensor_type: None | str | TensorType = "pt",
        **kwargs,
    ):
        if isinstance(data, Data):
            data = data.to_dict()
        data = {} if data is None else data
        data.update(kwargs)
        super().__init__(data=data, tensor_type=tensor_type)


class CraneFeature(BrainFeature):
    """
    Standarized features for neural signal data.
    """

    brainset: str
    """ID of the brainset"""
    subject: str
    """ID of the subject"""
    session: str
    """ID of the session"""
    signals: Float[Tensor, "n_channels n_samples"]
    """Neural time series data (e.g., iEEG, ECoG, SEEG)"""
    channel_labels: list[str]
    """List of channel labels corresponding to the channels in the signals"""
    channel_coordinates: Float[Tensor, "n_channels 3"]
    """Tensor of shape (n_channels, 3) containing the x, y, z coordinates of each channel"""
    sampling_rate: int
    """Sampling rate of the signals"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        channel_dim = 1 if self.batched else 0

        if self.signals.shape[channel_dim] != self.channel_coordinates.shape[channel_dim] != len(self.channel_labels):
            raise ValueError(
                f"Number of channels in signals ({self.signals.shape[channel_dim]}), channel_coordinates ({self.channel_coordinates.shape[channel_dim]}), and channel_labels ({len(self.channel_labels)}) must all match."
            )

        if self.channel_coordinates.device != self.signals.device:
            raise ValueError("channel_coordinates must be on the same device as signals")

    def __len__(self) -> int:
        return self.signals.shape[0]

    @property
    def device(self) -> torch.device:
        return self.signals.device

    @property
    def batched(self) -> bool:
        return self.signals.ndim == 3


class BrainFeatureExtractor(SequenceFeatureExtractor, ABC):
    """
    Base feature extractor for brain models.
    Inherits from SequenceFeatureExtractor for HuggingFace Hub compatibility
    and padding utilization.

    Override the `forward` method to implement custom feature extraction logic.
    """

    def __call__(self, batch: BrainFeature) -> BrainFeature:
        return self.forward(batch)

    @abstractmethod
    def forward(self, batch: BrainFeature) -> BrainFeature:
        """Forward method to be implemented by subclasses."""
        ...
