import inspect
from abc import ABC
from collections import defaultdict
from typing import Any

from ..core import BrainFeatureExtractor, BrainModel
from .decorators import TaskSpec


class BrainBench(ABC):
    """Abstract base class for brain benchmarks.

    Subclasses only define decorated eval and finetune methods,
    as well as the `name`, `version`, `reference`.

    """

    name: str
    """Human-readable name of the benchmark."""
    version: str | None = None
    """Version of the benchmark."""
    reference: str | None = None
    """Reference or citation for the benchmark."""

    def __init__(self) -> None:
        # Concrete tasks
        self._finetunes: dict[str, TaskSpec] = {}
        self._evals: dict[str, TaskSpec] = {}

        # Grouping by base task name
        self._finetunes_base: dict[str, list[TaskSpec]] = defaultdict(list)
        self._evals_base: dict[str, list[TaskSpec]] = defaultdict(list)

        # Indices for efficient lookup (TODO)

        self._collect_tasks()
        self._validate_tasks()

    def _collect_tasks(self):
        """Collect all decorated methods as tasks."""
        for _, method in inspect.getmembers(self, inspect.ismethod):
            for desc in getattr(method.__func__, "__bench_tasks__", []):
                base = desc.name
                if desc.name == "__base__":
                    raise ValueError(f"[{self.name}] Task name `__base__` reserved.")
                if base in self._finetunes or base in self._evals:
                    raise ValueError(f"[{self.name}] Duplicate task name: `{base}`")

                # Expand sweeps and register tasks
                for spec in desc.expand(self):
                    match spec.role:
                        case "finetune":
                            self._finetunes[spec.name] = spec
                            self._finetunes_base[base].append(spec)
                        case "eval":
                            self._evals[spec.name] = spec
                            self._evals_base[base].append(spec)
                        case _:
                            raise ValueError(f"[{self.name}] Unknown task role: `{spec.role}`")

    def _validate_tasks(self):
        """Validate task dependencies."""
        for spec in self._evals.values():
            if spec.uses is None:
                continue

            # Ensure dependencies are available
            if spec.uses not in self._finetunes:
                raise ValueError(f"[{self.name}] Eval `{spec.name}` depends on unknown finetune `{spec.uses}`")

    def _select_tasks(self, evals: list[str] | None, tags: list[str] | None) -> dict[str, TaskSpec]:
        """Select tasks based on eval names and/or tags."""
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
        """Build a DAG to determine which finetunes are needed for which evals."""
        groups: dict[str, list[TaskSpec]] = {}

        for spec in tasks.values():
            key = spec.uses or "__base__"
            groups.setdefault(key, []).append(spec)

        return groups

    def _load_or_finetune(self, spec: TaskSpec, model: BrainModel, featurizer: BrainFeatureExtractor) -> BrainModel:
        """Load a finetuned model from cache, or finetune it if not available."""
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
            evals (list[str] | None, default=None): Specific eval names to run. If None, run all.
            tags (list[str] | None, default=None): Tags to filter evals. If None, no tag filtering.

        Returns:
            A dictionary of evaluation metrics.
        """

        tasks = self._select_tasks(evals, tags)

        if force_zero_shot:  # TODO: Does force zero shot even make sense in bfms? I think not.
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

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(name={self.name!r}, finetunes={len(self._finetunes)}, evals={len(self._evals)})"
        )

    def __str__(self) -> str:
        d = self.describe()
        lines = [f"Benchmark: {d['name']}"]

        if d["version"]:
            lines.append(f"Version: {d['version']}")
        if d["reference"]:
            lines.append(f"Reference: {d['reference']}")

        lines.append("Finetune tasks:")
        for base, info in d["finetunes"].items():
            axes = ", ".join(info["sweeps"]) or "none"
            lines.append(f"  - {base}: {info['count']} runs (axes: {axes})")
        if not d["finetunes"]:
            lines.append("  (none)")

        lines.append("Eval tasks:")
        for base, info in d["evals"].items():
            uses = info["uses"] or "base"
            axes = ", ".join(info["sweeps"]) or "none"
            tags = ", ".join(info["tags"]) or "none"
            lines.append(f"  - {base}: {info['count']} runs (uses: {uses}, axes: {axes}, tags: {tags})")
        if not d["evals"]:
            lines.append("  (none)")

        return "\n".join(lines)

    def describe(self) -> dict:
        """Structured, machine-readable description of the benchmark."""
        return {
            "name": self.name,
            "version": self.version,
            "reference": self.reference,
            "finetunes": {
                base: {
                    "count": len(specs),
                    "sweeps": sorted({k for s in specs for k in s.sweep_axes.keys()}),
                    "instances": [
                        {
                            "name": s.name,
                            "axes": s.sweep_axes,
                        }
                        for s in specs
                    ],
                }
                for base, specs in self._finetunes_base.items()
            },
            "evals": {
                base: {
                    "uses": specs[0].uses,
                    "count": len(specs),
                    "tags": list(specs[0].tags),
                    "sweeps": sorted({k for s in specs for k in s.sweep_axes.keys()}),
                    "instances": [
                        {
                            "name": s.name,
                            "axes": s.sweep_axes,
                        }
                        for s in specs
                    ],
                }
                for base, specs in self._evals_base.items()
            },
        }
