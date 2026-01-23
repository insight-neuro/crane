from collections.abc import Callable

import torch
import torch.nn as nn

from crane.core.featurizer import BrainFeatureExtractor
from crane.core.model import BrainModel
from crane.eval.bench import BrainBench
from crane.eval.data import NeuralLabeledData


class LinearTrain:
    def __init__(
        self,
        criterion: Callable = nn.MSELoss,
        optimizer_cls: Callable = torch.optim.SGD,
        lr: float = 0.01,
        epochs: int = 100,
    ) -> None:
        self.criterion = criterion()
        self.optimizer_cls = optimizer_cls
        self.lr = lr
        self.epochs = epochs

    def __call__(
        self,
        bench: BrainBench,
        model: BrainModel,
        featurizer: BrainFeatureExtractor,
        train_data: NeuralLabeledData,
    ) -> BrainModel:
        features = featurizer(train_data.data)
        inputs = features["inputs"]
        targets = train_data.labels

        # Simple linear layer training
        linear_layer = torch.nn.Linear(inputs.shape[1], targets.shape[1])
        optimizer = self.optimizer_cls(linear_layer.parameters(), lr=self.lr)
        loss_fn = self.criterion

        # Training loop
        for _ in range(self.epochs):
            optimizer.zero_grad()
            outputs = linear_layer(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

        # Wrap the trained linear layer into the model
        trained_model = BrainModel.with_head(model, linear_layer)
        return trained_model
