import inspect
from abc import ABC
from typing import Any

from ..core import BrainFeatureExtractor, BrainModel
from .decorators import TaskSpec


class BrainBench(ABC):
    """Abstract base class for brain benchmarks.

    Subclasses only define decorated eval and finetune methods, as well as the `name`, `version`, `reference`.

    """

    name: str
    version: str | None = None
    reference: str | None = None

    def __init__(self) -> None:
        self._finetunes: dict[str, TaskSpec] = {}
        self._evals: dict[str, TaskSpec] = {}
        self._collect_tasks()

    def _collect_tasks(self):
        for _, method in inspect.getmembers(self, inspect.ismethod):
            for desc in getattr(method.__func__, "__bench_specs__", []):
                spec = desc.bind(self)

                if spec.name == "__base__":
                    raise ValueError(f"[{self.name}] Task name `__base__` reserved.")

                if spec.name in self._finetunes or self.name in self._evals:
                    raise ValueError(f"[{self.name}] Duplicate task name: `{spec.name}`")

                if spec.role == "finetune":
                    self._finetunes[spec.name] = spec
                else:
                    self._evals[spec.name] = spec

        # Ensure all dependencies are available
        for spec in self._finetunes.values():
            if spec.uses not in self._evals:
                raise ValueError(f"[{self.name}] Eval `{spec.name} depends on unknown finetune `{spec.name}`")

    def _select_tasks(self, evals: list[str] | None, tags: list[str] | None) -> dict[str, TaskSpec]:
        # Base case: return all
        if not tags and not evals:
            return self._evals

        # Select all with correct tags
        tag_set = set(tags or [])
        tasks = {name: desc for name, desc in self._evals.items() if desc.tags & tag_set} if tags else {}

        # Add all single evals chosen
        if evals:
            for eval in evals:
                tasks[eval] = self._evals[eval]

        return tasks

    def _build_plan(self, tasks: dict[str, TaskSpec]) -> dict[str, list[TaskSpec]]:
        groups: dict[str, list[TaskSpec]] = {}

        for spec in tasks.values():
            key = spec.uses or "__base__"
            groups.setdefault(key, []).append(spec)

        return groups

    def _load_or_finetune(self, spec: TaskSpec, model: BrainModel, featurizer: BrainFeatureExtractor) -> BrainModel:
        # TODO: Load from cache
        model = spec.fn(model, featurizer)
        return model

    def run(
        self,
        model: BrainModel,
        featurizer: BrainFeatureExtractor,
        force_zero_shot: bool = False,
        evals: list[str] | None = None,
        tags: list[str] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Evaluate the model on the given dataset.

        Args:
            model (BrainModel): The model to be evaluated.
            featurizer (BrainFeatureExtractor): The feature extractor to process data.
            force_zero_shot (bool, default=False): If True, skip finetuning even if required.

        Returns:
            A dictionary of evaluation metrics.
        """

        tasks = self._select_tasks(evals, tags)

        if force_zero_shot:
            plan = {"__base__": list(tasks.values())}
        else:
            plan = self._build_plan(tasks)

        results: dict[str, dict[str, Any]] = {}

        for eval in plan.get("__base__", []):
            results[eval.name] = eval.fn(model, featurizer)

        for ft_name, eval_specs in plan.items():
            if ft_name == "__base__":
                continue

            ft_spec = self._finetunes[ft_name]
            ft_model = self._load_or_finetune(ft_spec, model, featurizer)

            for eval in eval_specs:
                results[eval.name] = eval.fn(ft_model, featurizer)

            # Free memory
            del ft_model

        return results
