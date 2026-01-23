from functools import cached_property

import torch


class NeuralData:
    """Base class for neural data representations.

    Args:
        data_path (str): Path to the data file.

    Properties:
        data (torch.Tensor): Input data tensor. Lazily loaded.

    """

    def __init__(self, data_path: str):
        self.data_path = data_path

    @cached_property
    def data(self) -> torch.Tensor:
        """Input data tensor. Lazily loaded."""
        return torch.load(self.data_path)


class NeuralLabeledData(NeuralData):
    """Neural data with associated labels.

    Args:
        data_path (str): Path to the data file.
        labels_path (str): Path to the labels file.

    Properties:
        data (torch.Tensor): Input data tensor. Lazily loaded.
        labels (torch.Tensor): Labels tensor. Lazily loaded.
    """

    def __init__(self, data_path: str, labels_path: str):
        super().__init__(data_path)
        self.labels_path = labels_path

    @cached_property
    def labels(self) -> torch.Tensor:
        """Labels tensor. Lazily loaded."""
        return torch.load(self.labels_path)
