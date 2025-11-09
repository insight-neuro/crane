from abc import ABC

from transformers import SequenceFeatureExtractor


class BrainFeatureExtractor(SequenceFeatureExtractor, ABC):
    """
    Base feature extractor for brain models.
    Inherits from SequenceFeatureExtractor for HuggingFace Hub compatibility.
    """

    pass
