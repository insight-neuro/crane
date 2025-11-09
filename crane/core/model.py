from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from transformers import PreTrainedModel
from transformers.utils.generic import ModelOutput

from .config import BrainConfig


@dataclass
class BrainOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor


class BrainModel(PreTrainedModel, ABC):
    """
    Base interface for all brain models in neurocrane.
    Inherits from PreTrainedModel for HuggingFace Hub compatibility.
    """

    config_class = BrainConfig
    base_model_prefix = "brain_model"
    supports_gradient_checkpointing = True

    @abstractmethod
    def forward(self, batch: dict, *args, **kwargs) -> BrainOutput:
        """
        Forward pass through the brain model.

        Args:
            batch: Dictionary containing input data.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Dictionary containing model outputs.
        """
        ...
