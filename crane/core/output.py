from dataclasses import dataclass

import torch
from transformers.utils.generic import ModelOutput


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
