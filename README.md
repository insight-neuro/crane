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
    
    # Optional: Models can declare batch requirements for documentation
    batch_requirements = {
        'same_subject': False,
        'same_session': False,
        'balanced_classes': False,
    }
    
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
    """
    Ensures each batch contains samples from a single session only.
    
    Critical when different sessions have:
    - Different numbers of channels
    - Different sampling rates
    - Different electrode configurations
    
    Args:
        dataset_sizes: List of sizes for each session
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle within sessions and batch order
        drop_last: Whether to drop incomplete batches
    """
    
    def __init__(
        self,
        dataset_sizes: List[int],
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = True,
    ):
        self.dataset_sizes = dataset_sizes
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
    
    def __iter__(self):
        all_batches = []
        start_idx = 0
        
        for size in self.dataset_sizes:
            # Create indices for this session
            session_indices = list(range(start_idx, start_idx + size))
            
            if self.shuffle:
                random.shuffle(session_indices)
            
            # Create batches for this session
            for i in range(0, len(session_indices), self.batch_size):
                batch = session_indices[i:i + self.batch_size]
                if not self.drop_last or len(batch) == self.batch_size:
                    all_batches.append(batch)
            
            start_idx += size
        
        # Shuffle batch order across sessions
        if self.shuffle:
            random.shuffle(all_batches)
        
        return iter(all_batches)
    
    def __len__(self):
        if self.drop_last:
            return sum(size // self.batch_size for size in self.dataset_sizes)
        return sum((size + self.batch_size - 1) // self.batch_size 
                   for size in self.dataset_sizes)


class SubjectBatchSampler:
    """
    Ensures each batch contains samples from a single subject only.
    
    Use when:
    - Training with contrastive learning (within-subject contrasts)
    - Subject-specific normalization needed
    - Avoiding subject-level data leakage
    
    Args:
        dataset: Dataset with metadata['subject'] field
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle within subjects and batch order
        drop_last: Whether to drop incomplete batches
    """
    
    def __init__(
        self,
        dataset,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = True,
    ):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        
        # Group indices by subject
        from collections import defaultdict
        self.subject_to_indices = defaultdict(list)
        
        for idx in range(len(dataset)):
            sample = dataset[idx]
            subject = sample['metadata']['subject']
            self.subject_to_indices[subject].append(idx)
    
    def __iter__(self):
        all_batches = []
        
        for subject, indices in self.subject_to_indices.items():
            if self.shuffle:
                random.shuffle(indices)
            
            # Create batches for this subject
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                if not self.drop_last or len(batch) == self.batch_size:
                    all_batches.append(batch)
        
        # Shuffle batch order across subjects
        if self.shuffle:
            random.shuffle(all_batches)
        
        return iter(all_batches)
    
    def __len__(self):
        if self.drop_last:
            return sum(len(indices) // self.batch_size 
                      for indices in self.subject_to_indices.values())
        return sum((len(indices) + self.batch_size - 1) // self.batch_size 
                  for indices in self.subject_to_indices.values())


class BalancedBatchSampler:
    """
    Ensures each batch has balanced representation of classes.
    
    Use when:
    - Classification with severe class imbalance
    - Want each batch to see all classes
    - Training stability requires balanced batches
    """
    # Implementation details...
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
        batch_strategy: str = "session",  # 'session', 'subject', 'random'
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
        
        if self.hparams.batch_strategy == "session":
            # For models needing consistent channels per batch
            return SessionBatchSampler(
                dataset_sizes=[len(d) for d in dataset.datasets],
                batch_size=self.hparams.batch_size,
                shuffle=shuffle,
                drop_last=True,
            )
        
        elif self.hparams.batch_strategy == "subject":
            # For contrastive learning within subjects
            return SubjectBatchSampler(
                dataset=dataset,
                batch_size=self.hparams.batch_size,
                shuffle=shuffle,
                drop_last=True,
            )
        
        elif self.hparams.batch_strategy == "random":
            # Standard random batching
            import torch
            return torch.utils.data.BatchSampler(
                torch.utils.data.RandomSampler(dataset) if shuffle 
                else torch.utils.data.SequentialSampler(dataset),
                batch_size=self.hparams.batch_size,
                drop_last=True,
            )
        
        else:
            raise ValueError(f"Unknown batch strategy: {self.hparams.batch_strategy}")
    
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

**Key Design Decisions:**
- DataModule handles batch composition (which samples go together)
- DataModule does NOT do model-specific preprocessing
- Different batch strategies for different loss functions
- Batch strategy is a config parameter, not hardcoded

---

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
    batch_size, n_channels, n_times = neural_data.shape
    
    if electrode_positions is None:
        # Simple adjacency: each channel referenced to neighbors
        rereferenced = torch.zeros_like(neural_data)
        
        for i in range(n_channels):
            neighbors = []
            if i > 0:
                neighbors.append(neural_data[:, i-1, :])
            if i < n_channels - 1:
                neighbors.append(neural_data[:, i+1, :])
            
            if neighbors:
                neighbor_avg = torch.stack(neighbors).mean(dim=0)
                rereferenced[:, i, :] = neural_data[:, i, :] - neighbor_avg
            else:
                rereferenced[:, i, :] = neural_data[:, i, :]
    else:
        # Spatial distance-based neighbors
        rereferenced = torch.zeros_like(neural_data)
        
        # Compute pairwise distances
        distances = torch.cdist(electrode_positions, electrode_positions)
        
        for i in range(n_channels):
            # Find spatial neighbors
            neighbor_mask = (distances[i] < neighbor_distance) & (distances[i] > 0)
            
            if neighbor_mask.any():
                neighbor_avg = neural_data[:, neighbor_mask, :].mean(dim=1)
                rereferenced[:, i, :] = neural_data[:, i, :] - neighbor_avg
            else:
                rereferenced[:, i, :] = neural_data[:, i, :]
    
    return rereferenced


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
    # Compute mean across channels
    mean = neural_data.mean(dim=1, keepdim=True)
    return neural_data - mean


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
    batch_size, n_channels, n_times = neural_data.shape
    
    if pairs is None:
        # Create sequential pairs
        pairs = [(i, i+1) for i in range(n_channels - 1)]
    
    # Compute differences
    bipolar_data = []
    for ch_i, ch_j in pairs:
        diff = neural_data[:, ch_i, :] - neural_data[:, ch_j, :]
        bipolar_data.append(diff)
    
    return torch.stack(bipolar_data, dim=1)
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
    if n_fft is None:
        n_fft = window_size
    
    batch_size, n_channels, n_times = neural_data.shape
    
    # Apply STFT per channel
    spectrograms = []
    for ch in range(n_channels):
        # Use torch.stft
        stft = torch.stft(
            neural_data[:, ch, :],
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=window_size,
            window=torch.hann_window(window_size, device=neural_data.device),
            return_complex=True,
        )
        # Compute magnitude
        magnitude = torch.abs(stft)
        spectrograms.append(magnitude)
    
    # Stack channels
    spectrogram = torch.stack(spectrograms, dim=1)
    
    # Optionally filter frequency range
    if freq_range is not None:
        low_hz, high_hz = freq_range
        freqs = torch.fft.rfftfreq(n_fft, 1/sampling_rate)
        freq_mask = (freqs >= low_hz) & (freqs <= high_hz)
        spectrogram = spectrogram[:, :, freq_mask, :]
    
    return spectrogram


def bandpass_filter(
    neural_data: torch.Tensor,
    sampling_rate: float,
    low_freq: float,
    high_freq: float,
    order: int = 4,
) -> torch.Tensor:
    """
    Apply bandpass filter to neural data.
    
    Args:
        neural_data: Shape (batch, channels, time)
        sampling_rate: Sampling rate in Hz
        low_freq: Low cutoff frequency in Hz
        high_freq: High cutoff frequency in Hz
        order: Filter order
        
    Returns:
        Filtered data, same shape as input
        
    Example:
        >>> data = torch.randn(32, 128, 2000)
        >>> # Extract theta band (4-8 Hz)
        >>> theta = bandpass_filter(data, sampling_rate=1000, 
        ...                         low_freq=4, high_freq=8)
    """
    # Note: This is a simplified implementation
    # For production, use scipy.signal or torchaudio
    from scipy import signal
    
    # Design Butterworth filter
    nyquist = sampling_rate / 2
    low = low_freq / nyquist
    high = high_freq / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    
    # Apply filter
    filtered = torch.zeros_like(neural_data)
    for i in range(neural_data.shape[0]):  # batch
        for j in range(neural_data.shape[1]):  # channels
            data_np = neural_data[i, j].cpu().numpy()
            filtered_np = signal.filtfilt(b, a, data_np)
            filtered[i, j] = torch.from_numpy(filtered_np)
    
    return filtered.to(neural_data.device)


def compute_wavelet_transform(
    neural_data: torch.Tensor,
    sampling_rate: float,
    frequencies: torch.Tensor,
    n_cycles: int = 7,
) -> torch.Tensor:
    """
    Compute continuous wavelet transform (Morlet wavelets).
    
    Args:
        neural_data: Shape (batch, channels, time)
        sampling_rate: Sampling rate in Hz
        frequencies: Frequencies to compute (Hz)
        n_cycles: Number of cycles in Morlet wavelet
        
    Returns:
        Complex wavelet coefficients, shape (batch, channels, n_freqs, time)
        
    Example:
        >>> data = torch.randn(32, 128, 2000)
        >>> freqs = torch.logspace(0, 2, 20)  # 1-100 Hz, log-spaced
        >>> wavelet = compute_wavelet_transform(data, 1000, freqs)
    """
    # Implementation using torch.fft
    # This is a simplified version
    batch_size, n_channels, n_times = neural_data.shape
    n_freqs = len(frequencies)
    
    # Create Morlet wavelets
    wavelets = []
    for freq in frequencies:
        # Time vector for wavelet
        time_wavelet = torch.arange(-3, 3, 1/sampling_rate)
        # Morlet wavelet
        wavelet = torch.exp(2j * torch.pi * freq * time_wavelet) * \
                  torch.exp(-time_wavelet**2 / (2 * (n_cycles / (2*torch.pi*freq))**2))
        wavelets.append(wavelet)
    
    # Convolve with data (frequency domain multiplication)
    # ... implementation details ...
    
    return wavelet_coefficients
```

```python
# crane/preprocessing/normalization.py
import torch
from typing import Optional, Tuple

def zscore_normalize(
    neural_data: torch.Tensor,
    dim: Optional[Tuple[int, ...]] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Z-score normalization (zero mean, unit variance).
    
    Args:
        neural_data: Shape (batch, channels, time)
        dim: Dimensions to compute statistics over
             Default: (1, 2) - across channels and time
        eps: Small constant for numerical stability
        
    Returns:
        Normalized data, same shape as input
        
    Example:
        >>> data = torch.randn(32, 128, 2000)
        >>> # Normalize across channels and time per sample
        >>> normalized = zscore_normalize(data, dim=(1, 2))
        >>> # Normalize per channel (across time only)
        >>> normalized = zscore_normalize(data, dim=2)
    """
    if dim is None:
        dim = (1, 2)  # Default: normalize across channels and time
    
    mean = neural_data.mean(dim=dim, keepdim=True)
    std = neural_data.std(dim=dim, keepdim=True)
    
    return (neural_data - mean) / (std + eps)


def robust_normalize(
    neural_data: torch.Tensor,
    dim: Optional[Tuple[int, ...]] = None,
    quantile_range: Tuple[float, float] = (0.25, 0.75),
) -> torch.Tensor:
    """
    Robust normalization using median and IQR.
    
    Less sensitive to outliers than z-score.
    
    Args:
        neural_data: Shape (batch, channels, time)
        dim: Dimensions to compute statistics over
        quantile_range: Tuple of (low, high) quantiles for IQR
        
    Returns:
        Normalized data, same shape as input
        
    Example:
        >>> data = torch.randn(32, 128, 2000)
        >>> # Robust normalization
        >>> normalized = robust_normalize(data)
    """
    if dim is None:
        dim = (1, 2)
    
    # Compute median
    median = neural_data.median(dim=dim, keepdim=True)[0]
    
    # Compute IQR
    q_low = torch.quantile(neural_data, quantile_range[0], dim=dim, keepdim=True)
    q_high = torch.quantile(neural_data, quantile_range[1], dim=dim, keepdim=True)
    iqr = q_high - q_low
    
    return (neural_data - median) / (iqr + 1e-8)


def minmax_normalize(
    neural_data: torch.Tensor,
    dim: Optional[Tuple[int, ...]] = None,
    feature_range: Tuple[float, float] = (0.0, 1.0),
) -> torch.Tensor:
    """
    Min-max normalization to specified range.
    
    Args:
        neural_data: Shape (batch, channels, time)
        dim: Dimensions to compute min/max over
        feature_range: Target (min, max) range
        
    Returns:
        Normalized data, same shape as input
        
    Example:
        >>> data = torch.randn(32, 128, 2000)
        >>> # Normalize to [0, 1]
        >>> normalized = minmax_normalize(data)
        >>> # Normalize to [-1, 1]
        >>> normalized = minmax_normalize(data, feature_range=(-1, 1))
    """
    if dim is None:
        dim = (1, 2)
    
    data_min = neural_data.amin(dim=dim, keepdim=True)
    data_max = neural_data.amax(dim=dim, keepdim=True)
    
    # Scale to [0, 1]
    normalized = (neural_data - data_min) / (data_max - data_min + 1e-8)
    
    # Scale to feature_range
    range_min, range_max = feature_range
    normalized = normalized * (range_max - range_min) + range_min
    
    return normalized
```

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
    batch_size, n_channels, n_times = neural_data.shape
    
    if n_channels <= max_electrodes:
        return neural_data
    
    if strategy == "first":
        return neural_data[:, :max_electrodes, :]
    
    elif strategy == "random":
        indices = torch.randperm(n_channels)[:max_electrodes]
        return neural_data[:, indices, :]
    
    elif strategy == "variance":
        if variance is None:
            # Compute variance across time
            variance = neural_data.var(dim=2).mean(dim=0)  # Average across batch
        
        # Select top variance electrodes
        top_indices = torch.argsort(variance, descending=True)[:max_electrodes]
        return neural_data[:, top_indices, :]
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def pad_electrodes(
    neural_data: torch.Tensor,
    target_channels: int,
    pad_value: float = 0.0,
) -> torch.Tensor:
    """
    Pad channels dimension to target size.
    
    Args:
        neural_data: Shape (batch, channels, time)
        target_channels: Target number of channels
        pad_value: Value to use for padding
        
    Returns:
        Padded data, shape (batch, target_channels, time)
        
    Example:
        >>> data = torch.randn(32, 100, 2000)
        >>> padded = pad_electrodes(data, target_channels=128)
        >>> padded.shape  # (32, 128, 2000)
    """
    batch_size, n_channels, n_times = neural_data.shape
    
    if n_channels >= target_channels:
        return neural_data[:, :target_channels, :]
    
    # Pad
    pad_size = target_channels - n_channels
    padding = torch.full((batch_size, pad_size, n_times), 
                        pad_value, 
                        device=neural_data.device,
                        dtype=neural_data.dtype)
    
    return torch.cat([neural_data, padding], dim=1)
```

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

**Key Design Decisions:**
- Supports both fine-tuning and zero-shot evaluation
- Handles fine-tuning internally
- Benchmark-specific metrics via abstract method
- Standard interface for all benchmarks

---

## 3. User Implementation Pattern

### 3.1 User's Repository Structure

```
my-seeg-models/                   # User's repo
├── models/
│   ├── __init__.py
│   ├── seeg_transformer.py       # Implements crane.BaseModel
│   └── config.py
│
├── data/
│   ├── __init__.py
│   └── ieeg_datamodule.py        # LightningDataModule
│
├── configs/
│   ├── train.yaml
│   ├── model/
│   │   └── seeg_transformer.yaml
│   └── data/
│       └── ieeg_data.yaml
│
├── train.py                       # Training script
├── finetune.py                    # Fine-tuning script
└── requirements.txt               # Includes: neurocrane
```

### 3.2 User's Model Implementation

```python
# my-seeg-models/models/seeg_transformer.py
import torch
import torch.nn as nn
import neurocrane as crane
from transformers import PretrainedConfig


class SEEGTransformerConfig(PretrainedConfig):
    model_type = "seeg_transformer"
    
    def __init__(
        self,
        hidden_size: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads


class SEEGTransformer(crane.BaseModel):
    config_class = SEEGTransformerConfig
    
    # Declare batch requirements (optional, for documentation)
    batch_requirements = {
        'same_subject': False,  # This model works with mixed subjects
        'same_session': True,   # But needs consistent channels (same session)
    }
    
    def __init__(self, config: SEEGTransformerConfig):
        super().__init__(config)
        
        # User defines their architecture
        self.embeddings = nn.Linear(config.n_channels, config.hidden_size)
        self.encoder = nn.TransformerEncoder(...)
        self.head = nn.Linear(config.hidden_size, config.output_dim)
        
        # Store preprocessing config (saved to Hub automatically)
        self.preprocessing_config = {
            'rereferencing': config.rereferencing,  # 'laplacian', 'car', 'bipolar', 'none'
            'max_electrodes': config.max_electrodes,
            'normalization': config.normalization,  # 'zscore', 'minmax', 'robust'
        }
    
    def preprocess(self, neural_data, **kwargs):
        """Model-specific preprocessing."""
        
        # 1. Re-referencing (model's choice of scheme)
        if self.preprocessing_config['rereferencing'] == 'laplacian':
            neural_data = self._laplacian_rereference(neural_data)
        elif self.preprocessing_config['rereferencing'] == 'car':
            neural_data = self._common_average_rereference(neural_data)
        
        # 2. Electrode selection (model's architectural constraint)
        max_elec = self.preprocessing_config['max_electrodes']
        if neural_data.shape[1] > max_elec:
            # Take first N electrodes (or could sample, select by variance, etc.)
            neural_data = neural_data[:, :max_elec, :]
        
        # 3. Normalization (model's training requirement)
        if self.preprocessing_config['normalization'] == 'zscore':
            mean = neural_data.mean(dim=(1, 2), keepdim=True)
            std = neural_data.std(dim=(1, 2), keepdim=True)
            neural_data = (neural_data - mean) / (std + 1e-8)
        
        return neural_data
    
    def _laplacian_rereference(self, data):
        # Implementation...
        return data
    
    def _common_average_rereference(self, data):
        # Implementation...
        return data
    
    def forward(self, neural_data, attention_mask=None, labels=None, **kwargs):
        # Receives already-preprocessed data from __call__
        embeddings = self.embeddings(neural_data)
        hidden = self.encoder(embeddings, mask=attention_mask)
        logits = self.head(hidden)
        
        loss = None
        if labels is not None:
            loss = self.compute_loss({'logits': logits}, labels)
        
        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': hidden,
        }
    
    def compute_loss(self, outputs, labels):
        # User's loss function
        return nn.functional.mse_loss(outputs['logits'], labels)
    
    def extract_features(self, neural_data, layer_ids=None):
        # User's feature extraction logic
        embeddings = self.embeddings(neural_data)
        features = {'embeddings': embeddings}
        
        for i, layer in enumerate(self.encoder.layers):
            embeddings = layer(embeddings)
            if layer_ids is None or i in layer_ids:
                features[f'layer_{i}'] = embeddings
        
        return features
```

**Example: Contrastive Learning Model (needs same-subject batches)**

```python
class ContrastiveModel(crane.BaseModel):
    config_class = ContrastiveConfig
    
    # Declare batch requirements
    batch_requirements = {
        'same_subject': True,   # Required for within-subject contrastive loss
        'same_session': False,  # Can handle different sessions
    }
    
    def preprocess(self, neural_data, **kwargs):
        # Aggressive augmentation for contrastive learning
        neural_data = self._augment(neural_data)
        neural_data = (neural_data - neural_data.mean()) / neural_data.std()
        return neural_data
    
    def compute_loss(self, outputs, labels=None):
        # Contrastive loss within batch
        # Assumes all samples from same subject (DataModule's job to ensure this)
        embeddings = outputs['embeddings']
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(embeddings, embeddings.T)
        
        # Contrastive loss (positive pairs from same subject)
        loss = contrastive_loss(sim_matrix)
        
        return loss
```

### 3.3 User's Training Script

```python
# my-seeg-models/train.py
import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
import neurocrane as crane

from models.seeg_transformer import SEEGTransformer, SEEGTransformerConfig
from data.ieeg_datamodule import iEEGDataModule


@hydra.main(version_base=None, config_path="configs", config_name="train")
def train(cfg: DictConfig):
    # Instantiate data
    datamodule = hydra.utils.instantiate(cfg.data)
    
    # Instantiate model
    model_config = SEEGTransformerConfig(**cfg.model.config)
    model = SEEGTransformer(model_config)
    
    # Wrap in neurocrane Lightning module
    lightning_module = crane.LightningModule(
        model=model,
        optimizer_config=cfg.optimizer,
    )
    
    # Create trainer
    trainer = pl.Trainer(**cfg.trainer)
    
    # Train
    trainer.fit(lightning_module, datamodule)
    
    # Push to Hub using neurocrane utilities
    if cfg.push_to_hub:
        from neurocrane.utils.hub import push_to_hub
        push_to_hub(model, cfg.hub_model_id)


if __name__ == "__main__":
    train()
```

### 3.4 User's Config

```yaml
# my-seeg-models/configs/train.yaml
defaults:
  - data: ieeg_data
  - model: seeg_transformer
  - optimizer: adamw
  - _self_

trainer:
  max_epochs: 100
  devices: 1
  accelerator: gpu
  precision: 16

push_to_hub: true
hub_model_id: "my-lab/seeg-foundation-v1"
```

```yaml
# my-seeg-models/configs/data/ieeg_data.yaml
_target_: data.ieeg_datamodule.iEEGDataModule

train_subject_trials_file: /path/to/sessions.yaml
context_length: 2.0
batch_size: 32
batch_strategy: "session"  # 'session', 'subject', 'random'
num_workers: 4
val_split: 0.1
```

```yaml
# my-seeg-models/configs/data/contrastive_data.yaml
# Same datamodule, different batch strategy for contrastive learning
_target_: data.ieeg_datamodule.iEEGDataModule

train_subject_trials_file: /path/to/sessions.yaml
context_length: 2.0
batch_size: 32
batch_strategy: "subject"  # ← Ensures same-subject batches for contrastive loss
num_workers: 4
val_split: 0.1
```

```yaml
# my-seeg-models/configs/model/seeg_transformer.yaml
_target_: models.seeg_transformer.SEEGTransformer

config:
  hidden_size: 512
  num_layers: 6
  num_heads: 8
  n_channels: 128
  output_dim: 512
  # Preprocessing config (saved to Hub with model)
  rereferencing: "laplacian"  # or "car", "bipolar", "none"
  max_electrodes: 128
  normalization: "zscore"  # or "minmax", "robust"
```

---

## 4. Cross-Lab Workflow

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

## 5. Implementation Roadmap

### Phase 1: Core Interfaces (Week 1-2)
**Priority: CRITICAL**

1. **Implement `BaseModel`** (`crane/models/base.py`)
   - Inherit from `PreTrainedModel`
   - Define abstract methods: `preprocess`, `forward`, `compute_loss`, `extract_features`
   - Implement `__call__` to automatically apply preprocessing
   - Add `batch_requirements` class attribute (optional, for documentation)
   - Add docstrings with clear specifications
   - Test that interface constraints work

2. **Implement `LightningModule`** (`crane/training/lightning_module.py`)
   - Wrap any `BaseModel`
   - Handle training/validation loops
   - Support config-driven optimizer setup
   - Add logging callbacks

3. **Implement `Benchmark`** (`crane/evaluation/benchmark.py`)
   - Abstract base class
   - Support fine-tuning workflow
   - Support zero-shot evaluation
   - Define metric computation interface

4. **Implement Batch Samplers** (`crane/data/samplers.py`)
   - `SessionBatchSampler` - same session per batch
   - `SubjectBatchSampler` - same subject per batch
   - `BalancedBatchSampler` - balanced classes per batch
   - Document when to use each strategy

5. **Implement Basic Preprocessor** (`crane/data/preprocessing.py`)
   - `BasicPreprocessor` - data quality only (NaN removal, dtype)
   - NOT model-specific transformations

**Deliverable:** Core package that can be pip installed

```bash
pip install neurocrane
```

### Phase 2: Utilities (Week 3)
**Priority: HIGH**

1. **Preprocessing Utilities** (`crane/preprocessing/`)
   ```python
   # crane/preprocessing/rereferencing.py
   def laplacian_rereference(neural_data, ...)
   def common_average_rereference(neural_data, ...)
   def bipolar_rereference(neural_data, ...)
   
   # crane/preprocessing/spectral.py
   def compute_spectrogram(neural_data, sampling_rate, ...)
   def bandpass_filter(neural_data, low_freq, high_freq, ...)
   def compute_wavelet_transform(neural_data, frequencies, ...)
   
   # crane/preprocessing/normalization.py
   def zscore_normalize(neural_data, ...)
   def robust_normalize(neural_data, ...)
   def minmax_normalize(neural_data, ...)
   
   # crane/preprocessing/spatial.py
   def select_electrodes(neural_data, max_electrodes, strategy, ...)
   def pad_electrodes(neural_data, target_channels, ...)
   ```

2. **Hub Integration** (`crane/utils/hub.py`)
   ```python
   def push_to_hub(model: BaseModel, repo_id: str, **kwargs)
   def load_from_hub(repo_id: str) -> BaseModel
   ```

3. **Common Metrics** (`crane/evaluation/metrics.py`)
   ```python
   def compute_accuracy(preds, labels)
   def compute_f1(preds, labels)
   def compute_auc(preds, labels)
   ```

4. **Checkpointing Helpers** (`crane/utils/checkpointing.py`)
   ```python
   def save_checkpoint(model, optimizer, epoch, path)
   def load_checkpoint(path) -> Dict
   ```

### Phase 3: Documentation (Week 4)
**Priority: HIGH**

1. **API Documentation**
   - Sphinx or MkDocs
   - Full API reference
   - Type hints throughout

2. **User Guides**
   - `docs/creating_models.md` - How to implement a model with preprocessing
   - `docs/creating_benchmarks.md` - How to create benchmarks
   - `docs/data_loading.md` - Data module patterns and batch strategies
   - `docs/preprocessing_guide.md` - When to preprocess in DataModule vs Model
   - `docs/batch_composition.md` - Choosing the right batch sampler

3. **Architecture Diagrams**
   - Use Excalidraw initially
   - Convert to Mermaid for docs
   - Interface contract diagram (most critical)
   - Preprocessing flow diagram (DataModule → Model)
   - Batch composition strategies diagram

### Phase 4: Examples (Week 5)
**Priority: MEDIUM**

1. **Example Model** (`examples/models/simple_transformer.py`)
   - Minimal working transformer
   - Shows all interface methods including `preprocess()`
   - Demonstrates different preprocessing strategies
   - Well-commented

2. **Example Contrastive Model** (`examples/models/contrastive_model.py`)
   - Shows same-subject batch requirement
   - Implements contrastive loss
   - Demonstrates batch_requirements usage

3. **Example Data Module** (`examples/data/toy_datamodule.py`)
   - Synthetic iEEG data
   - Shows session-aware batching pattern
   - Demonstrates different batch strategies (session, subject, random)
   - Shows BasicPreprocessor usage

4. **Example Benchmark** (`examples/benchmarks/toy_benchmark.py`)
   - Simple classification task
   - Shows fine-tuning workflow

5. **Notebooks**
   - `examples/notebooks/01_quickstart.ipynb` - Basic usage
   - `examples/notebooks/02_custom_model.ipynb` - Implement your own model
   - `examples/notebooks/03_preprocessing.ipynb` - Preprocessing best practices
   - `examples/notebooks/04_batch_strategies.ipynb` - When to use which batch sampler
   - `examples/notebooks/05_feature_extraction.ipynb` - Extract and analyze features

### Phase 5: Testing (Ongoing)
**Priority: CRITICAL**

1. **Unit Tests** (`tests/`)
   - Test interface enforcement
   - Test Lightning module with mock models
   - Test benchmark workflow
   - Test Hub integration

2. **Integration Tests**
   - End-to-end training
   - End-to-end evaluation
   - Hub push/pull cycle

3. **CI/CD**
   - GitHub Actions
   - Automated testing on PR
   - Version bumping and PyPI release

---

## 6. Technical Requirements

### Dependencies

```toml
# pyproject.toml
[project]
name = "neurocrane"
version = "0.1.0"
dependencies = [
    "torch>=2.0.0",
    "pytorch-lightning>=2.0.0",
    "transformers>=4.30.0",
    "hydra-core>=1.3.0",
    "numpy>=1.24.0",
    "h5py>=3.8.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "black>=23.0.0",
    "ruff>=0.0.285",
    "mypy>=1.0.0",
]
docs = [
    "sphinx>=6.0.0",
    "sphinx-rtd-theme>=1.2.0",
]
examples = [
    "jupyter>=1.0.0",
    "matplotlib>=3.7.0",
    "seaborn>=0.12.0",
]
```

### Python Version

- Minimum: Python 3.9
- Recommended: Python 3.10+
- Use `from __future__ import annotations` for type hints

### Code Quality

- **Formatting:** Black
- **Linting:** Ruff
- **Type Checking:** MyPy
- **Docstrings:** Google style
- **Testing:** pytest with >80% coverage

---

## 7. Critical Design Principles

### 7.1 Interface Over Implementation

- neurocrane provides **contracts**, not **code**
- Users implement models, neurocrane validates interface compliance
- Think: "What must ALL iEEG models support?"

### 7.2 Minimal Core, Rich Ecosystem

- Core package is small and stable
- Examples live in `examples/` (not imported by default)
- Community contributions go in separate repos

### 7.3 HuggingFace First

- Inherit from `PreTrainedModel`
- Use their config system
- Use their Hub infrastructure
- Don't reinvent serialization

### 7.4 Lightning Best Practices

- Don't fight Lightning's patterns
- Use `LightningModule` for training
- Use `LightningDataModule` pattern (but don't enforce)
- Leverage callbacks, not custom loops

### 7.5 Configuration as Code

- Hydra for all configuration
- No hardcoded hyperparameters
- Composable configs
- Command-line overrides

### 7.6 Preprocessing and Batch Composition Separation

- **Preprocessing = Model's job**: Re-referencing, normalization, filtering
- **Batch composition = DataLoader's job**: Which samples go together
- Preprocessing travels with model to Hub
- Batch strategy is a config parameter
- Clear separation enables flexibility and reusability

---

## 8. Success Metrics

### Technical Metrics

- [ ] Can install via `pip install neurocrane`
- [ ] Can implement a model in <100 lines
- [ ] Can train a model with <20 lines
- [ ] Can push/pull models from Hub
- [ ] Test coverage >80%

### Adoption Metrics

- [ ] 3+ labs using neurocrane
- [ ] 5+ models on HuggingFace Hub with `neurocrane` tag
- [ ] 2+ benchmarks implemented
- [ ] GitHub stars >100

### Quality Metrics

- [ ] Documentation covers all use cases
- [ ] Examples run without errors
- [ ] Clear error messages for interface violations
- [ ] Type hints throughout

---

## 9. Example: Interface Violation Error

```python
# User forgets to implement preprocess
class BadModel(crane.BaseModel):
    def forward(self, neural_data, **kwargs):
        return {'logits': ...}
    
    def compute_loss(self, outputs, labels):
        return ...
    
    def extract_features(self, neural_data, **kwargs):
        return {'final': ...}
    
    # Missing: preprocess!

# When instantiated:
model = BadModel(config)
# TypeError: Can't instantiate abstract class BadModel with abstract method preprocess
```

```python
# User forgets to implement extract_features
class AnotherBadModel(crane.BaseModel):
    def preprocess(self, neural_data, **kwargs):
        return neural_data
    
    def forward(self, neural_data, **kwargs):
        return {'logits': ...}
    
    def compute_loss(self, outputs, labels):
        return ...
    
    # Missing: extract_features!

# When instantiated:
model = AnotherBadModel(config)
# TypeError: Can't instantiate abstract class AnotherBadModel with abstract method extract_features
```

**Good error messages are critical for adoption.**

---

## 10. Launch Checklist

### Pre-Release (v0.1.0)

- [ ] Core interfaces implemented and tested
- [ ] At least 1 working example model
- [ ] Basic documentation published
- [ ] PyPI package published
- [ ] GitHub repo public

### Post-Release

- [ ] Present at lab meetings / conferences
- [ ] Blog post explaining design
- [ ] Reach out to 3 neuroscience labs for feedback
- [ ] Iterate based on real usage

---

## Appendix: Key Files Content

### A. `setup.py`

```python
from setuptools import setup, find_packages

setup(
    name="neurocrane",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "pytorch-lightning>=2.0.0",
        "transformers>=4.30.0",
        "hydra-core>=1.3.0",
    ],
    python_requires=">=3.9",
    author="Your Lab",
    description="Standardized framework for iEEG foundation models",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourlab/neurocrane",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
```

### B. `crane/__init__.py`

```python
"""NeuroCrane: Framework for iEEG Foundation Models"""

__version__ = "0.1.0"

from crane.models.base import BaseModel
from crane.training.lightning_module import LightningModule
from crane.evaluation.benchmark import Benchmark

# Preprocessing utilities
from crane.preprocessing import (
    # Re-referencing
    laplacian_rereference,
    common_average_rereference,
    bipolar_rereference,
    # Spectral
    compute_spectrogram,
    bandpass_filter,
    compute_wavelet_transform,
    # Normalization
    zscore_normalize,
    robust_normalize,
    minmax_normalize,
    # Spatial
    select_electrodes,
    pad_electrodes,
)

# Batch samplers
from crane.data.samplers import (
    SessionBatchSampler,
    SubjectBatchSampler,
    BalancedBatchSampler,
)

__all__ = [
    # Core interfaces
    "BaseModel",
    "LightningModule",
    "Benchmark",
    # Preprocessing
    "laplacian_rereference",
    "common_average_rereference",
    "bipolar_rereference",
    "compute_spectrogram",
    "bandpass_filter",
    "compute_wavelet_transform",
    "zscore_normalize",
    "robust_normalize",
    "minmax_normalize",
    "select_electrodes",
    "pad_electrodes",
    # Samplers
    "SessionBatchSampler",
    "SubjectBatchSampler",
    "BalancedBatchSampler",
]
```

---

**END OF TECHNICAL SPECIFICATION**