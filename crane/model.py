from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from transformers import PretrainedConfig, PreTrainedModel
from transformers.utils.generic import ModelOutput


class BrainConfig(PretrainedConfig):
    model_type = "brain_model"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


@dataclass
class BrainOutput(ModelOutput):
    """Model output for neural data models.

    Attributes:
        last_hidden_state (torch.Tensor): The last hidden state tensor (leaned features).
    """

    last_hidden_state: torch.Tensor

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class BrainHeadOutput(BrainOutput):
    """Model output for brain models with task-specific heads.

    Attributes:
        last_hidden_state (torch.Tensor): The last hidden state tensor (leaned features).
        outputs (torch.Tensor): The output tensor from the task-specific head.
    """

    outputs: torch.Tensor

    @staticmethod
    def from_features(features: BrainOutput, outputs: torch.Tensor) -> "BrainHeadOutput":
        return BrainHeadOutput(**features.to_dict(), outputs=outputs)

    def to_dict(self) -> dict:
        return super().to_dict()


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
