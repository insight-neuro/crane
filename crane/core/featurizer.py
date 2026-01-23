from abc import ABC, abstractmethod
from typing import Any

from transformers import SequenceFeatureExtractor


class BrainFeatureExtractor(SequenceFeatureExtractor, ABC):
    """
    Base feature extractor for brain models.
    Inherits from SequenceFeatureExtractor for HuggingFace Hub compatibility
    and padding utilization.

    Override the `forward` method to implement custom feature extraction logic.
    """

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args, **kwds)

    @abstractmethod
    def forward(self, *args: Any, **kwds: Any) -> Any:
        """Forward method to be implemented by subclasses."""
        ...
