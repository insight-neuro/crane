# Crane: Machine Learning for Neural Interfaces

**Package Name:** `neurocrane`  
**Import Convention:** `import neurocrane as crane`  
**Purpose:** Standardized framework for iEEG/SEEG foundation model training, evaluation, and sharing

Crane is a comprehensive library designed to facilitate the development and training of machine learning models for neural data. It provides tools for data handling and common neural model components, making it easier to build and experiment with models in this domain.

## 1. Core Architecture

### 1.1 Design Philosophy

- **Framework, not implementation**: Provides interfaces and infrastructure, not models
- **HuggingFace compatible**: Inherits from `PreTrainedModel` for Hub integration
- **PyTorch Lightning based**: Training infrastructure built on Lightning
- **Hydra configured**: All settings managed via composable configs
- **Lab-to-lab interoperability**: Standard interface enables model sharing

### 1.2 Package Structure

```
neurocrane/
├── pyproject.toml
├── README.md
│
├── crane/
│   ├── __init__.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   └── base.py              # BaseModel interface
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── lightning_module.py  # LightningModule
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── benchmark.py         # Benchmark base class
│   │   └── evaluator.py         # evaluate() function
│   │
│   ├── dataset/
│   │   ├── __init__.py
│   │   ├── single_session.py
│   │   ├── multi_session.py
│   │   ├── collator.py
│   │   └── sampler.py
│   │
│   └── preprocess/              # Common preprocessing operations
│       ├── __init__.py
│       ├── electrode_subset.py
│       ├── rereferencing.py     # Laplacian, CAR, bipolar
│       └── spectrogram.py
│
├── docs/
└── tests/
```

---


**Usage in User Models:**

```python
import neurocrane as crane
from crane.preprocessing import (
    laplacian_rereference,
    zscore_normalize,
    compute_spectrogram,
    select_electrodes,
)

class SEEGTransformer(crane.BaseModel):
    
    def preprocess(self, neural_data, **kwargs):
        # Use crane's preprocessing utilities
        
        # 1. Re-reference
        if self.config.rereferencing == 'laplacian':
            neural_data = laplacian_rereference(neural_data)
        elif self.config.rereferencing == 'car':
            neural_data = common_average_rereference(neural_data)
        
        # 2. Select electrodes
        neural_data = select_electrodes(
            neural_data,
            max_electrodes=self.config.max_electrodes,
            strategy='variance'
        )
        
        # 3. Normalize
        neural_data = zscore_normalize(neural_data, dim=(1, 2))
        
        return neural_data
```

```python
class SpectralModel(crane.BaseModel):
    """Model that works in frequency domain."""
    
    def preprocess(self, neural_data, **kwargs):
        # Convert to spectrogram
        spectrogram = compute_spectrogram(
            neural_data,
            sampling_rate=self.config.sampling_rate,
            window_size=256,
            hop_length=128,
            freq_range=(1, 100),  # 1-100 Hz
        )
        
        # Log-transform and normalize
        spectrogram = torch.log(spectrogram + 1e-8)
        spectrogram = zscore_normalize(spectrogram)
        
        return spectrogram
```

**Key Design Decisions:**
- All functions are pure, stateless transformations
- Work with batched data (batch, channels, time)
- Consistent API across all preprocessing functions
- Type hints and comprehensive docstrings
- Examples for each function
- Users can import and compose as needed

---

### 3.6 LightningModule

**File:** `crane/training/lightning_module.py`

```python
import pytorch_lightning as pl
from crane.models.base import BaseModel
from typing import Any, Dict
import torch


class LightningModule(pl.LightningModule):
    """
    PyTorch Lightning wrapper for crane.BaseModel.
    
    Handles:
    - Training loop
    - Validation loop
    - Optimizer configuration
    - Logging
    
    Works with ANY model implementing BaseModel interface.
    """
    
    def __init__(
        self,
        model: BaseModel,
        optimizer_config: Dict[str, Any],
        scheduler_config: Dict[str, Any] = None,
    ):
        super().__init__()
        self.model = model
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config
        self.save_hyperparameters(ignore=['model'])
    
    def forward(self, **batch):
        return self.model(**batch)
    
    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        outputs = self.model(**batch)
        loss = outputs['loss']
        
        # Log metrics
        self.log('train_loss', loss, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch: Dict, batch_idx: int):
        outputs = self.model(**batch)
        loss = outputs['loss']
        
        self.log('val_loss', loss, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        # Create optimizer from config
        optimizer_cls = self.optimizer_config.pop('_target_')
        optimizer = optimizer_cls(self.parameters(), **self.optimizer_config)
        
        if self.scheduler_config:
            scheduler_cls = self.scheduler_config.pop('_target_')
            scheduler = scheduler_cls(optimizer, **self.scheduler_config)
            return [optimizer], [scheduler]
        
        return optimizer
```

**Key Design Decisions:**
- Model-agnostic: works with any `BaseModel`
- Standard Lightning patterns
- Config-driven optimizer/scheduler setup
- Minimal boilerplate for users

---

## 3. Cross-Lab Workflow

### Lab A: Pretrain and Share

```python
# Train model with specific preprocessing
python train.py

# Model automatically pushed to HuggingFace Hub
# → "lab-a/seeg-foundation-v1"
# Preprocessing config saved automatically in model config
```

### Lab B: Load and Analyze

```python
import neurocrane as crane

# Load pretrained model (preprocessing config included!)
model = crane.BaseModel.from_pretrained("lab-a/seeg-foundation-v1")

# Extract features from their own raw data
# Preprocessing is applied automatically via __call__
features = model.extract_features(raw_patient_data)

# Do novel analysis
analyze_neural_representations(features)
```

### Lab C: Fine-tune

```python
import neurocrane as crane

# Load and fine-tune
model = crane.BaseModel.from_pretrained("lab-a/seeg-foundation-v1")
# Model's preprocessing is preserved

# Fine-tune on new task
lightning_module = crane.LightningModule(model, optimizer_config)
trainer.fit(lightning_module, seizure_datamodule)

# Share fine-tuned model (with same preprocessing)
model.push_to_hub("lab-c/seeg-seizure-detector")
```

### Lab D: Benchmark

```python
import neurocrane as crane

# Load model
model = crane.BaseModel.from_pretrained("lab-a/seeg-foundation-v1")

# Evaluate on benchmark (auto fine-tunes if needed)
benchmark = SeizureDetectionBenchmark(data_path="...")
results = benchmark.evaluate(model)
# Preprocessing is applied correctly throughout

print(results)
# {'accuracy': 0.89, 'f1': 0.87, 'auc': 0.92}
```

**Key Benefit:** Preprocessing travels with the model automatically. Lab B doesn't need to know Lab A's preprocessing details - it just works.

---

### 3.7 Benchmark (NOT SURE HOW TO DO THIS YET)

**File:** `crane/evaluation/benchmark.py`

```python
from abc import ABC, abstractmethod
from crane.models.base import BaseModel
from crane.training.lightning_module import LightningModule
import pytorch_lightning as pl
from typing import Dict, Any, Optional


class Benchmark(ABC):
    """
    Base class for neurocrane benchmarks.
    
    Benchmarks can:
    - Require fine-tuning on train set before evaluation
    - Support zero-shot evaluation
    - Provide train/val/test splits
    - Compute standardized metrics
    """
    
    def __init__(
        self,
        name: str,
        dataset: Any,  # Should provide train/val/test splits
        requires_finetuning: bool = True,
        finetune_config: Optional[Dict] = None,
    ):
        self.name = name
        self.dataset = dataset
        self.requires_finetuning = requires_finetuning
        self.finetune_config = finetune_config or self._default_finetune_config()
    
    @abstractmethod
    def _default_finetune_config(self) -> Dict:
        """Return default fine-tuning configuration."""
        pass
    
    def evaluate(
        self,
        model: BaseModel,
        force_zero_shot: bool = False,
    ) -> Dict[str, float]:
        """
        Evaluate model on benchmark.
        
        Args:
            model: Any model implementing BaseModel
            force_zero_shot: Skip fine-tuning even if benchmark requires it
            
        Returns:
            Dictionary of metric_name -> value
        """
        if self.requires_finetuning and not force_zero_shot:
            model = self._finetune(model)
        
        # Evaluate on test set
        metrics = self._evaluate_split(model, self.dataset.test)
        
        return metrics
    
    def _finetune(self, model: BaseModel) -> BaseModel:
        """Fine-tune model on benchmark's train/val sets."""
        # Wrap in Lightning
        lightning_module = LightningModule(
            model=model,
            optimizer_config=self.finetune_config['optimizer'],
        )
        
        # Create trainer
        trainer = pl.Trainer(**self.finetune_config['trainer'])
        
        # Fine-tune
        datamodule = self.dataset.get_datamodule()
        trainer.fit(lightning_module, datamodule)
        
        return lightning_module.model
    
    @abstractmethod
    def _evaluate_split(
        self,
        model: BaseModel,
        split_data: Any,
    ) -> Dict[str, float]:
        """
        Compute metrics on a data split.
        
        Must be implemented by specific benchmarks.
        """
        pass
```