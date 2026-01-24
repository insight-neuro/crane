from abc import ABC, abstractmethod

import torch
from transformers import PreTrainedModel

from .config import BrainConfig
from .output import BrainHeadOutput, BrainOutput


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

    @staticmethod
    def with_head(model: "BrainModel", head: torch.nn.Module) -> "BrainModel":
        """
        Wraps the base brain model with a task-specific head.

        Args:
            model: The base brain model.
            head: The task-specific head module.

        Returns:
            A new module combining the brain model and the head.
        """

        class BrainModelWithHead(BrainModel):
            def __init__(self, backbone: BrainModel, head_module: torch.nn.Module):
                super().__init__(backbone.config)
                self.backbone = backbone
                self.head = head_module

            def forward(self, batch: dict, *args, **kwargs) -> BrainHeadOutput:
                features = self.backbone(batch, *args, **kwargs)
                outputs = self.head(features.last_hidden_state)
                return BrainHeadOutput.from_features(features, outputs)

        return BrainModelWithHead(model, head)
