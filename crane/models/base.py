from abc import ABC, abstractmethod

from transformers import PreTrainedModel


class Module(PreTrainedModel, ABC):
    """
    Base interface for all neurocrane models.

    All models must implement:
    - preprocess(): Model-specific data transformations
    - forward(): Standard forward pass with loss computation
    - extract_features(): Feature extraction for analysis

    Inherits from PreTrainedModel for HuggingFace Hub compatibility.
    """

    @abstractmethod
    def preprocess(self, batch: dict) -> dict:
        """
        Apply model-specific preprocessing transformations to the batch.

        This is where models define their specific requirements:
        - Re-referencing scheme (Laplacian, CAR, bipolar)
        - Electrode selection/subset
        - Normalization strategy (z-score, min-max, robust)
        - Filtering or spectral transforms

        Note: DataModule handles data quality (NaN removal, dtype).
              Model handles transformations (re-referencing, normalization).

        Args:
            batch: Dictionary containing the following keys:
                - 'neural_data': torch.Tensor - main neural data
                - Optional: 'channels', 'electrode_positions', 'sampling_rate', etc.
                - Optional: 'eyetracking', 'behavior_labels', etc.

        Returns:
            Preprocessed batch dictionary with same structure.
            At minimum, 'neural_data' should be transformed.

        Example:
            def preprocess(self, batch):
                # Get neural data
                neural_data = batch['neural_data']

                # Re-reference using electrode positions if available
                if 'electrode_positions' in batch:
                    neural_data = laplacian_rereference(
                        neural_data,
                        electrode_positions=batch['electrode_positions']
                    )

                # Normalize
                neural_data = zscore_normalize(neural_data)

                # Update batch
                batch['neural_data'] = neural_data
                return batch
        """
        pass

    @abstractmethod
    def forward(self, batch: dict, **kwargs) -> dict:
        """
        Forward pass through the model.

        Args:
            batch: Dictionary containing the following keys:
                - 'ieeg': {data: torch.Tensor[n_channels, n_samples], sampling_rate: int}
                - 'channels': {id: np.array}
                - 'metadata': {brainset: str, subject: str, session: str}
            **kwargs: Additional model-specific arguments

        Returns:
            Dictionary with keys:
                - 'logits': torch.Tensor - model predictions
                - 'hidden_states': torch.Tensor - final layer representations
        """
        pass

    @abstractmethod
    def extract_features(self, batch: dict, **kwargs) -> dict:
        """
        Extract intermediate or final representations for analysis.

        Critical for neuroscience: enables extracting features from
        any layer for downstream analysis or visualization.

        Args:
            batch: Dictionary containing the following keys:
                - 'ieeg': {data: torch.Tensor[n_channels, n_samples], sampling_rate: int}
                - 'channels': {id: np.array}
                - 'metadata': {brainset: str, subject: str, session: str}
            **kwargs: Additional model-specific arguments

        Returns:
            Dictionary with keys:
                - 'features': torch.Tensor - extracted features
        """
        pass

    def __call__(self, batch: dict, **kwargs) -> dict:
        """
        Automatically applies preprocessing before forward pass.

        This ensures preprocessing is always applied consistently.
        This is what LightningModule calls with **batch unpacking.

        Args:
            batch: Dictionary containing the following keys:
                - 'ieeg': {data: torch.Tensor[n_channels, n_samples], sampling_rate: int}
                - 'channels': {id: np.array}
                - 'metadata': {brainset: str, subject: str, session: str}
            **kwargs: Additional model-specific arguments

        Returns:
            Dictionary from forward() with loss, logits, hidden_states
        """
        # Apply model-specific preprocessing
        batch = self.preprocess(batch, **kwargs)

        # Call forward
        return self.forward(batch, **kwargs)
