from abc import ABC, abstractmethod

from ..core import BrainFeatureExtractor, BrainModel


class BrainBench(ABC):
    """Abstract base class for brain benchmarks."""

    benchmark_name: str

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.requires_finetuning = cls.finetune is not BrainBench.finetune

    def finetune(
        self,
        model: BrainModel,
        featurizer: BrainFeatureExtractor,
    ) -> BrainModel:
        """Finetune the model on the given dataset (optional).

        Args:
            model: The model to be finetuned.
            featurizer: The feature extractor to process data.

        Returns:
            The finetuned model.
        """
        raise NotImplementedError("Finetune method not implemented.")

    def run(
        self,
        model: BrainModel,
        featurizer: BrainFeatureExtractor,
        force_zero_shot: bool = False,
        save_finetuned_directory: str | None = None,
    ) -> dict[str, float]:
        """Evaluate the model on the given dataset.

        Args:
            model (BrainModel): The model to be evaluated.
            featurizer (BrainFeatureExtractor): The feature extractor to process data.
            force_zero_shot (bool, default=False): If True, skip finetuning even if required.
            save_finetuned_directory (str | None, default=None): Directory to save the finetuned model. If None, the model is not saved.

        Returns:
            A dictionary of evaluation metrics.
        """

        if self.requires_finetuning and not force_zero_shot:
            model = self.finetune(model, featurizer)

            if save_finetuned_directory is not None:
                model.save_pretrained(save_finetuned_directory)

        return self.evaluate(model, featurizer)

    @abstractmethod
    def evaluate(
        self,
        model: BrainModel,
        featurizer: BrainFeatureExtractor,
    ) -> dict[str, float]:
        """Evaluate the model on the given dataset.

        Args:
            model (BrainModel): The model to be evaluated.
            featurizer (BrainFeatureExtractor): The feature extractor to process data.

        Returns:
            A dictionary of evaluation metrics.
        """
        ...
