from abc import ABC, abstractmethod
from typing import Any

from temporaldata import Data
from transformers import BatchFeature, SequenceFeatureExtractor
from transformers.utils.generic import TensorType


class BrainFeature(BatchFeature):
    """
    BatchFeature subclass for brain data. Can be used to store features extracted from neural recordings.

    This class is derived from a python dictionary and can be used as a dictionary.
    """

    def __init__(
        self,
        data: dict[str, Any],
        *,
        tensor_type: None | str | TensorType = None,
        **kwargs,
    ):
        data.update(kwargs)
        super().__init__(data=data, tensor_type=tensor_type)


class BrainFeatureExtractor(SequenceFeatureExtractor, ABC):
    """
    Base feature extractor for brain models.
    Inherits from SequenceFeatureExtractor for HuggingFace Hub compatibility
    and padding utilization.

    Override the `forward` method to implement custom feature extraction logic.
    """

    def __call__(self, data: Data, **kwargs) -> BrainFeature:
        return self.forward(data, **kwargs)

    @abstractmethod
    def forward(self, data: Data, **kwargs) -> BrainFeature:
        """Forward method to be implemented by subclasses."""
        ...
