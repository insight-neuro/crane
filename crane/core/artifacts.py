from dataclasses import dataclass

import torch
from transformers.utils.generic import ModelOutput


@dataclass
class BrainOutput(ModelOutput):
    last_hidden_state: torch.Tensor

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class BrainHeadOutput(BrainOutput):
    output: torch.Tensor

    @staticmethod
    def from_features(features: BrainOutput, output: torch.Tensor) -> "BrainHeadOutput":
        return BrainHeadOutput(**features.to_dict(), output=output)

    def to_dict(self) -> dict:
        return super().to_dict()
