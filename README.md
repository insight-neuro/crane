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
├── setup.py
├── pyproject.toml
├── README.md
│
├── crane/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py              # BaseModel interface
│   │   └── config.py            # Base config classes
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── lightning_module.py  # LightningModule
│   │   ├── trainer.py           # train() function
│   │   └── callbacks.py         # Standard callbacks
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── benchmark.py         # Benchmark base class
│   │   ├── metrics.py           # Common metrics
│   │   └── evaluator.py         # evaluate() function
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── base.py              # Optional base patterns
│   │   ├── samplers.py          # SessionBatchSampler, SubjectBatchSampler
│   │   └── preprocessing.py     # BasicPreprocessor (data quality only)
│   │
│   ├── preprocessing/           # Common preprocessing operations
│   │   ├── __init__.py
│   │   ├── rereferencing.py     # Laplacian, CAR, bipolar
│   │   ├── spectral.py          # Spectrogram, wavelet, bandpass
│   │   ├── normalization.py     # Z-score, min-max, robust scaling
│   │   └── spatial.py           # Electrode selection, spatial filtering
│   │
│   └── utils/
│       ├── __init__.py
│       ├── hub.py               # HuggingFace Hub helpers
│       ├── checkpointing.py
│       └── logging.py
│
├── docs/
│
└── tests/
```

---

## 2. Design Principles: Preprocessing vs Batch Composition

### 2.1 Separation of Concerns

NeuroCrane distinguishes between two critical concepts:

**Preprocessing (Model's Responsibility):**
- How to transform individual samples
- Re-referencing schemes (Laplacian, CAR, bipolar)
- Normalization strategies (z-score, min-max)
- Electrode selection/subset
- Filtering

**Batch Composition (DataLoader's Responsibility):**
- Which samples should be grouped together
- Same-subject batching (for within-subject contrastive learning)
- Same-session batching (for consistent channel counts)
- Balanced class batching (for classification)
- Random batching (for standard supervised learning)

### 2.2 Architecture Overview

```
Raw Data → [DataModule: Basic Cleaning + Batch Composition] 
         → [Model: Model-Specific Preprocessing] 
         → Forward Pass
```

**Level 1 - DataModule:**
- Data quality checks (NaN removal, dtype validation)
- Batch composition strategy (which samples go together)
- Basic cleaning that ALL models need

**Level 2 - Model:**
- Model-specific transformations
- Preprocessing that defines this model's requirements
- Saved and loaded automatically with model from Hub

---

## 3. Core Interfaces

### 3.1 BaseModel (Abstract Base Class)

**File:** `crane/models/base.py`

```python
from abc import ABC, abstractmethod
from transformers import PreTrainedModel
from typing import Optional, Dict, List
import torch


class BaseModel(PreTrainedModel, ABC):
    """
    Base interface for all neurocrane models.
    
    All models must implement:
    - preprocess(): Model-specific data transformations
    - forward(): Standard forward pass with loss computation
    - compute_loss(): Custom loss function
    - extract_features(): Feature extraction for analysis
    
    Inherits from PreTrainedModel for HuggingFace Hub compatibility.
    """
    
    @abstractmethod
    def preprocess(
        self,
        neural_data: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Apply model-specific preprocessing transformations.
        
        This is where models define their specific requirements:
        - Re-referencing scheme (Laplacian, CAR, bipolar)
        - Electrode selection/subset
        - Normalization strategy (z-score, min-max, robust)
        - Filtering or spectral transforms
        
        Note: DataModule handles data quality (NaN removal, dtype).
              Model handles transformations (re-referencing, normalization).
        
        Args:
            neural_data: Clean neural data from DataModule
            **kwargs: Additional preprocessing parameters
            
        Returns:
            Preprocessed neural data tensor (typically same shape as input)
            
        Example:
            def preprocess(self, neural_data, **kwargs):
                # Re-reference
                neural_data = laplacian_rereference(neural_data)
                # Normalize
                neural_data = (neural_data - neural_data.mean()) / neural_data.std()
                return neural_data
        """
        pass
    
    @abstractmethod
    def forward(
        self,
        neural_data: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            neural_data: Shape (batch, channels, time) or (batch, time, channels)
            attention_mask: Optional mask for padding
            labels: Optional labels for supervised training
            **kwargs: Additional model-specific arguments
            
        Returns:
            Dictionary with keys:
                - 'loss': Optional[torch.Tensor] - computed if labels provided
                - 'logits': torch.Tensor - model predictions
                - 'hidden_states': torch.Tensor - final layer representations
        """
        pass
    
    @abstractmethod
    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute task-specific loss.
        
        Allows different models to use different loss functions
        (contrastive, MSE, cross-entropy, etc.)
        
        Args:
            outputs: Dictionary from forward() containing 'logits', etc.
            labels: Ground truth labels
            
        Returns:
            loss: Scalar loss tensor
        """
        pass
    
    @abstractmethod
    def extract_features(
        self,
        neural_data: torch.Tensor,
        layer_ids: Optional[List[int]] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Extract intermediate or final representations for analysis.
        
        Critical for neuroscience: enables extracting features from
        any layer for downstream analysis or visualization.
        
        Args:
            neural_data: Input data
            layer_ids: Optional list of layer indices to extract.
                      If None, return all layers or final features.
            
        Returns:
            Dictionary mapping layer identifier -> features tensor
            Example: {'layer_0': tensor, 'layer_3': tensor, 'final': tensor}
        """
        pass
    
    def __call__(self, neural_data, **kwargs):
        """
        Automatically applies preprocessing before forward pass.
        
        This ensures preprocessing is always applied consistently.
        Users should call this method (not forward directly).
        
        Args:
            neural_data: Raw (but cleaned) neural data
            **kwargs: Passed to both preprocess and forward
            
        Returns:
            Dictionary from forward() with loss, logits, hidden_states
        """
        # Apply model-specific preprocessing
        neural_data = self.preprocess(neural_data, **kwargs)
        
        # Then do forward pass
        return self.forward(neural_data, **kwargs)
```

**Key Design Decisions:**
- Inherits from `PreTrainedModel` → automatic Hub integration
- Abstract methods enforce interface compliance
- `preprocess()` is mandatory → ensures each model declares transformations
- `__call__()` automatically applies preprocessing → prevents user errors
- Returns dictionaries for flexibility
- `extract_features` is mandatory → critical for neuroscience analysis
- `batch_requirements` is optional → for documentation only

---

### 3.2 DataModule and Batch Composition

**File:** `crane/data/base.py` (optional base patterns)

NeuroCrane provides batch sampling strategies to handle different training scenarios:

```python
# crane/data/samplers.py

class SessionBatchSampler:
    # ... implementation details
    pass
```

**Example DataModule using batch strategies:**

```python
# User's datamodule implementation
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from crane.data.samplers import SessionBatchSampler, SubjectBatchSampler

class iEEGDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_subject_trials_file: str,
        context_length: float,
        batch_size: int = 32,
        num_workers: int = 4,
        val_split: float = 0.1,
        data_root_dir: Optional[str] = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Only basic data quality preprocessing
        self.basic_preprocessor = BasicPreprocessor()
    
    def _get_batch_sampler(self, dataset, shuffle=True):
        """Create appropriate batch sampler based on strategy."""
        
        return SessionBatchSampler(
            dataset_sizes=[len(d) for d in dataset.datasets],
            batch_size=self.hparams.batch_size,
            shuffle=shuffle,
            drop_last=True,
        )
    
    def train_dataloader(self):
        batch_sampler = self._get_batch_sampler(self.train_dataset, shuffle=True)
        
        return DataLoader(
            self.train_dataset,
            batch_sampler=batch_sampler,
            num_workers=self.hparams.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True,
        )
```

**Basic Preprocessor (data quality only):**

```python
# crane/data/preprocessing.py

class BasicPreprocessor:
    """
    Handles data quality issues that ALL models need.
    Does NOT do model-specific transformations.
    
    Responsibilities:
    - Remove NaN/Inf values
    - Ensure correct dtypes
    - Handle missing channels (if metadata available)
    """
    
    def __call__(self, batch: Dict) -> Dict:
        # Remove NaN/Inf
        batch['neural_data'] = torch.nan_to_num(batch['neural_data'], 
                                                 nan=0.0, 
                                                 posinf=0.0, 
                                                 neginf=0.0)
        
        # Ensure float32
        batch['neural_data'] = batch['neural_data'].float()
        
        # Apply channel mask if provided
        if 'channel_mask' in batch:
            batch['neural_data'] = batch['neural_data'] * batch['channel_mask']
        
        return batch
```

### 3.5 Preprocessing Utilities

**File:** `crane/preprocessing/`

NeuroCrane provides common preprocessing operations as reusable functions. Models can use these in their `preprocess()` method:

```python
# crane/preprocessing/rereferencing.py
import torch
from typing import Optional

def laplacian_rereference(
    neural_data: torch.Tensor,
    neighbor_distance: float = 1.0,
    electrode_positions: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Apply Laplacian re-referencing to neural data.
    
    Laplacian referencing computes the difference between each electrode
    and the average of its spatial neighbors, emphasizing local activity.
    
    Args:
        neural_data: Shape (batch, channels, time)
        neighbor_distance: Maximum distance for neighbor selection (mm)
        electrode_positions: Optional (channels, 3) array of xyz positions
                           If None, uses simple adjacency (channel i ± 1)
        
    Returns:
        Re-referenced data, same shape as input
        
    Example:
        >>> data = torch.randn(32, 128, 2000)  # batch, channels, time
        >>> rereferenced = laplacian_rereference(data)
    """
    pass


def common_average_rereference(
    neural_data: torch.Tensor,
    keepdim: bool = True,
) -> torch.Tensor:
    """
    Apply common average reference (CAR).
    
    Subtracts the mean across all electrodes at each time point.
    
    Args:
        neural_data: Shape (batch, channels, time)
        keepdim: If True, maintain original shape
        
    Returns:
        Re-referenced data, same shape as input
        
    Example:
        >>> data = torch.randn(32, 128, 2000)
        >>> rereferenced = common_average_rereference(data)
    """
    pass


def bipolar_rereference(
    neural_data: torch.Tensor,
    pairs: Optional[List[Tuple[int, int]]] = None,
) -> torch.Tensor:
    """
    Apply bipolar referencing.
    
    Computes differences between specified electrode pairs.
    
    Args:
        neural_data: Shape (batch, channels, time)
        pairs: List of (channel_i, channel_j) tuples
              If None, creates sequential pairs: (0,1), (1,2), ...
        
    Returns:
        Bipolar-referenced data, shape (batch, n_pairs, time)
        
    Example:
        >>> data = torch.randn(32, 128, 2000)
        >>> bipolar = bipolar_rereference(data)  # Sequential pairs
        >>> # Or specify custom pairs
        >>> pairs = [(0, 5), (10, 15), (20, 25)]
        >>> bipolar = bipolar_rereference(data, pairs=pairs)
    """
    pass
```

```python
# crane/preprocessing/spectral.py
import torch
from typing import Optional, Tuple

def compute_spectrogram(
    neural_data: torch.Tensor,
    sampling_rate: float,
    window_size: int = 256,
    hop_length: int = 128,
    n_fft: Optional[int] = None,
    freq_range: Optional[Tuple[float, float]] = None,
) -> torch.Tensor:
    """
    Compute spectrogram using Short-Time Fourier Transform.
    
    Args:
        neural_data: Shape (batch, channels, time)
        sampling_rate: Sampling rate in Hz
        window_size: Window size in samples
        hop_length: Hop length in samples
        n_fft: FFT size (default: window_size)
        freq_range: Optional (low_hz, high_hz) to restrict frequencies
        
    Returns:
        Spectrogram, shape (batch, channels, freq_bins, time_bins)
        
    Example:
        >>> data = torch.randn(32, 128, 2000)
        >>> spec = compute_spectrogram(data, sampling_rate=1000)
        >>> spec.shape  # (32, 128, freq_bins, time_bins)
    """
   pass

```python
# crane/preprocessing/spatial.py
import torch
from typing import Optional, List

def select_electrodes(
    neural_data: torch.Tensor,
    max_electrodes: int,
    strategy: str = "first",
    variance: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Select subset of electrodes.
    
    Args:
        neural_data: Shape (batch, channels, time)
        max_electrodes: Maximum number of electrodes to keep
        strategy: Selection strategy:
                 'first' - first N electrodes
                 'random' - random N electrodes
                 'variance' - N highest variance electrodes
        variance: Pre-computed variance per electrode (optional)
        
    Returns:
        Data with selected electrodes, shape (batch, max_electrodes, time)
        
    Example:
        >>> data = torch.randn(32, 256, 2000)
        >>> # Select 128 electrodes with highest variance
        >>> selected = select_electrodes(data, max_electrodes=128, 
        ...                              strategy='variance')
    """
    pass

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

### 3.7 Benchmark

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