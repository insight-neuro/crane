# Crane: Machine Learning for Neural Interfaces

Crane is a comprehensive library designed to facilitate the development and training of machine learning models for neural data. It provides tools for data handling and common neural model components, making it easier to build and experiment with models in this domain. Crane emphasizes interoperability, reproducibility, and ease of use, allowing researchers to share models and preprocessing pipelines seamlessly.

## Building your Model

To build a model in Crane, you typically need to implement two main components:
- A `BrainModel` subclass that defines your model architecture and forward pass
- A `BrainFeatureExtractor` subclass that handles data preprocessing, feature extraction and tokenization.

Once these components are defined, you can easily train, evaluate, and share your model using Crane's built-in tools.

Example:

```python
from crane import BrainFeaturizer, BrainModel, BrainOutput
from crane.preprocessing import (
    laplacian_rereference,
    zscore_normalize,
    select_electrodes,
)

class SEEGFeaturizer(BrainFeatureExtractor):

    def preprocess(self, neural_data, **kwargs):
        # Use crane's preprocessing utilities
        
        # 1. Re-reference
        neural_data = laplacian_rereference(neural_data)
        
        # 2. Select electrodes
        neural_data = select_electrodes(
            neural_data,
            max_electrodes=self.config.max_electrodes,
            strategy='variance'
        )
        
        # 3. Normalize
        neural_data = zscore_normalize(neural_data, dim=(1, 2))
        
        return neural_data


class SEEGTransformer(BrainModel):
    def __init__(self, config):
        super().__init__(config)
        # Define model layers here
        self.transformer = torch.nn.Transformer(
            d_model=config.hidden_size,
            nhead=config.num_heads,
            num_encoder_layers=config.num_layers,
            num_decoder_layers=config.num_layers,
        )

    def forward(self, features, **kwargs) > BrainOutput:
        # Forward pass through transformer
        last_hidden_state = self.transformer(features)
        return BrainOutput(last_hidden_state=last_hidden_state)
```

## Data Format
While not required for `crane`, we recommend storing the data in the following `temporaldata` format, which strikes a balance between flexibility and speed of I/O.
Please see the [ieeg-data](https://github.com/insight-neuro/ieeg-data) repository for the detailed documentation and implementation.
```python
session = Data(
    # metadata
    brainset = "STRING",
    subject = "STRING",
    session = "STRING",
    citation = "STRING", # in bib format
    
    # In case the data includes iEEG. NTOE: EEG is also included here, as it is a generalization (the same data format).
    # NOTE: can be extended to fMRI, FUS, or any high-dimensional time series.
    ieeg = RegularTimeSeries(
        data = seeg_data,  # Shape: (n_timepoints, n_electrodes). Voltage in uV
        sampling_rate = 2048,  # Hz

        domain_start = 0.0,  # Start at 0 seconds
        domain = "auto"  # Let it infer from data length and sampling rate
    ),
    channels = ArrayDict(
        id = np.array(["LAMY1"]), # Shape: (n_electrodes, )
        
        # Coordinates of the corresponding electrodes. Usually, these will be the MNI coordinates. 
        # Note: in some datasets, there will be an exception (if MNI are unavailable or type of probe is different)
        x = np.array([0.0]), # Shape: (n_electrodes, ). if unknown, can be np.nan
        y = np.array([0.0]),
        z = np.array([0.0]),
        brain_area = np.array(["UNKNOWN"]),

        type = np.array(["SEEG"]) # options: SEEG, ECOG, EEG, etc
    ),
    artifacts = Interval() # Specify the affected intervals and channels in a standardized format

    # In case the data includes any type of triggers. Note: These could be redundant with the other tags below.
    triggers = IrregularTimeSeries(
        timestamps = trigger_times,
        type = np.array(["MOUSE_CLICK"]),
        note = np.array([""]), # Optional note together with the trigger. Can be empty.
        timekeys = ['timestamps'],  # Only timestamps should be adjusted during operations
    ),
    
    # In case the data includes stimulation. Note: frequency is not a parameter here! Use many electrical_stimulation events (as separate pulses) to denote the stimulation at a particular frequency.
    stimulation = IrregularTimeSeries(
        timestamps = stimulus_times,  # Shape (n_stim,). If multiple electrodes/electrode pairs at the same time, there will be multiple entries in the timestamp
        waveform_type = np.array(["BIPHASIC"]),  # (n_stim,).
        stimulation_site = np.array(["CHANNEL1-CHANNEL2"]),  # (n_stim,). Can be two channel labels separated by a dash
        amplitude = np.array([1.0]), # mA
        pulsewidth = np.array([0.001]), # seconds
        duration = np.array([0.014]), # seconds
        timekeys = ['timestamps'],  # Only timestamps should be adjusted during operations
    ),
    
    # In case the data includes images shown
    images = IrregularTimeSeries()
    
    # In case the data includes sound played
    sounds = None, # same structure as images
    continuous_sound = None, # unused for now, but may be RegularTimeSeries - the actual raw waveform of the sound
    
    domain=ieeg.domain
)
```

## Design Reusable Benchmarks

Crane is designed to make it simple to share benchmarks across labs. A benchmark typically inherits from `crane.eval.BrainBenchmark` and implements `task_group` functions.

```python
from crane.eval import BrainBenchmark
from trl import SFTTrainer

class SeizureDetectionBenchmark(BrainBenchmark):

    def __init__(self, data_path):
        self.data_path = data_path

    def fine_tune(self, model, featurizer, train_data): # Optional
        # Fine-tune model on seizure detection task 
        trainer = SFTTrainer(
            model=model,
            processing_class=featurizer,
            train_dataset=train_data,
            args=fine_tuning_args,
        )
        trainer.train()
        return model

    def evaluate(self, model, featurizer):
        # Load benchmark dataset
        test_data = load_seizure_detection_data(self.data_path)
        
        all_predictions = []
        all_labels = []
        
        for raw_data, labels in test_data:
            # Preprocess data
            features = featurizer(raw_data)
            
            with torch.inference_mode():
                outputs = model(features)
            
            predictions = postprocess_outputs(outputs)
            all_predictions.extend(predictions)
            all_labels.extend(labels)
        
        # Compute metrics
        accuracy = compute_accuracy(all_predictions, all_labels)
        f1 = compute_f1_score(all_predictions, all_labels)
        auc = compute_auc(all_predictions, all_labels)
        
        return {'accuracy': accuracy, 'f1': f1, 'auc': auc}
```

## Cross-Lab Workflow

### Lab A: Pretrain and Share

```python
from transformers import Trainer

# Initialize model and featurizer
model = SEEGTransformer(config)
featurizer = SEEGFeaturizer()

# Train the model
trainer = Trainer(
    model=model,
    processing_class=featurizer,
    train_dataset=pretraining_data,
    eval_dataset=validation_data,
    args=training_args,
)

# Push to Hub
trainer.push_to_hub("lab-a/seeg-foundation-v1")
# Preprocessing config saved automatically in model config
```

### Lab B: Load and Analyze

```python
from crane import BrainModel, BrainFeatureExtractor
import torch

# Load pretrained model and featurizer
featurizer = BrainFeatureExtractor.from_pretrained("lab-a/seeg-foundation-v1")
model = BrainModel.from_pretrained("lab-a/seeg-foundation-v1")

# Extract learned representation from own raw data
features = featurizer(raw_patient_data)

with torch.inference_mode():
    representations = model(features)

# Do novel analysis
analyze_neural_representations(representations)
```

### Lab C: Fine-tune

```python
from crane import BrainModel, BrainFeatureExtractor
from trl import SFTTrainer

# Load and fine-tune
featurizer = BrainFeatureExtractor.from_pretrained("lab-a/seeg-foundation-v1")
model = BrainModel.from_pretrained("lab-a/seeg-foundation-v1")

# Fine-tune on new task
trainer = SFTTrainer(
    model=model,
    processing_class=featurizer,
    train_dataset=seizure_train_data,
    eval_dataset=seizure_val_data,
    args=fine_tuning_args,
)
trainer.train()

# Share fine-tuned model (with same preprocessing)
trainer.push_to_hub("lab-c/seeg-seizure-detector")
```

### Lab D: Benchmark

```python
from crane import BrainModel, BrainFeatureExtractor

# Load model
featurizer = BrainFeatureExtractor.from_pretrained("lab-a/seeg-foundation-v1")
model = BrainModel.from_pretrained("lab-a/seeg-foundation-v1")

# Evaluate on benchmark (auto fine-tunes if needed)
benchmark = SeizureDetectionBenchmark(data_path="...") 
results = benchmark.run(model, featurizer) # Preprocessing is applied correctly throughout

print(results)
# {'accuracy': 0.89, 'f1': 0.87, 'auc': 0.92}
```

**Key Benefit:** Preprocessing travels with the model automatically. Lab B doesn't need to know Lab A's preprocessing details - it just works.






