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
        tensor_type: None | str | TensorType = None,
        skip_tensor_conversion: list[str] | set[str] | None = None,
        **kwargs,
    ):
        if isinstance(data, Data):
            data = data.to_dict()
        data = {} if data is None else data
        data.update(kwargs)
        super().__init__(data=data, tensor_type=tensor_type, skip_tensor_conversion=skip_tensor_conversion)


class CraneFeature(BrainFeature):
    """
    Standarized features for neural signal data. When batched,
    all signals in the batch must share the same channels and metadata
    (e.g., brainset, subject, session, sampling_rate).

    Args:
        brainset (str): ID of the brainset
        subject (str): ID of the subject
        session (str): ID of the session
        signals (Float[Tensor, "[batch] n_channels n_samples"]): Neural time series data (e.g., iEEG, ECoG, SEEG)
        channel_labels (list[str]): List of channel labels corresponding to the channels in the signals
        channel_coordinates (Float[Tensor, "n_channels 3"]): Tensor of shape (n_channels, 3) containing the x, y, z coordinates of each channel
        sampling_rate (int): Sampling rate of the signals
    """

    signals: Float[Tensor, "n_channels n_samples"]

    def __init__(
        self,
        *args,
        brainset: str,
        subject: str,
        session: str,
        signals: Float[Tensor, "[batch] n_channels n_samples"],
        channel_labels: list[str],
        channel_coordinates: Float[Tensor, "[batch] n_channels 3"],
        sampling_rate: int,
        **kwargs,
    ):
        super().__init__(*args, **kwargs, signals=signals)

        self.brainset = brainset
        self.subject = subject
        self.session = session
        self.signals = signals
        self.channel_labels = channel_labels
        self.channel_coordinates = channel_coordinates
        self.sampling_rate = sampling_rate

        n_signal_channels = self.signals.shape[self.channel_dim]
        n_channel_coords = self.channel_coordinates.shape[0]
        n_channel_labels = len(self.channel_labels)

        if len({n_signal_channels, n_channel_coords, n_channel_labels}) > 1:
            raise ValueError(
                f"Number of channels in signals ({n_signal_channels}), channel_coordinates ({n_channel_coords}), and channel_labels ({n_channel_labels}) must all match."
            )

    @property
    def device(self) -> torch.device:
        """Returns the device of the signals tensor."""
        return self.signals.device

    @property
    def batched(self) -> bool:
        """Returns True if the signals tensor has a batch dimension (i.e., is 3D)."""
        return self.signals.ndim == 3

    @property
    def channel_dim(self) -> int:
        """Returns the dimension index of the channels in the signals tensor."""
        return 1 if self.batched else 0


class BrainFeatureExtractor[InT: BrainFeature, OutT: BrainFeature](SequenceFeatureExtractor, ABC):
    """
    Base feature extractor for brain models.
    Inherits from SequenceFeatureExtractor for HuggingFace Hub compatibility
    and padding utilization.

    Override the `forward` method to implement custom feature extraction logic.
    """

    def __call__(self, batch: InT) -> OutT:
        return self.forward(batch)

    @abstractmethod
    def forward(self, batch: InT) -> OutT:
        """Forward method to be implemented by subclasses."""
        ...
