from functools import cached_property

import torch


class NeuralData:
    """Base class for neural data representations.

    Args:
        data_path (str): Path to the data file.
        data (torch.Tensor | None): Input data tensor. If provided, data_path is ignored.

    Properties:
        data (torch.Tensor): Input data tensor. Lazily loaded if provided by data_path.

    """

    def __init__(self, data_path: str | None, *, data=None):
        self.data_path = data_path
        self._data = data

    @cached_property
    def data(self) -> torch.Tensor:
        """Input data tensor. Lazily loaded."""
        if self._data is not None:
            return self._data
        if self.data_path is None:
            raise ValueError("data_path is None, cannot load data.")
        return torch.load(self.data_path)

    def subset(self, indices: torch.Tensor) -> "NeuralData":
        """Return a subset of the neural data.

        Args:
            indices (torch.Tensor): Indices to select.

        Returns:
            NeuralData: Subset of the neural data.
        """
        return NeuralData(None, data=self.data[indices])

    def __len__(self) -> int:
        return len(self.data)


class NeuralLabeledData(NeuralData):
    """Neural data with associated labels.

    Args:
        data_path (str): Path to the data file.
        labels_path (str): Path to the labels file.
        data (torch.Tensor | None): Input data tensor. If provided, data_path is ignored.
        labels (torch.Tensor | None): Labels tensor. If provided, labels_path is ignored.

    Properties:
        data (torch.Tensor): Input data tensor. Lazily loaded if provided by data_path.
        labels (torch.Tensor): Labels tensor. Lazily loaded if provided by labels_path.
    """

    def __init__(
        self,
        data_path: str | None = None,
        labels_path: str | None = None,
        *,
        data: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ):
        super().__init__(data_path, data=data)
        self.labels_path = labels_path
        self._labels = labels

    @cached_property
    def labels(self) -> torch.Tensor:
        """Labels tensor. Lazily loaded."""
        if self._labels is not None:
            return self._labels
        if self.labels_path is None:
            raise ValueError("labels_path is None, cannot load labels.")
        return torch.load(self.labels_path)

    def subset(self, indices: torch.Tensor) -> "NeuralLabeledData":
        """Return a subset of the neural labeled data.

        Args:
            indices (torch.Tensor): Indices to select.

        Returns:
            NeuralLabeledData: Subset of the neural labeled data.
        """
        return NeuralLabeledData(
            data=self.data[indices],
            labels=self.labels[indices],
        )

    def __len__(self) -> int:
        return len(self.data)
