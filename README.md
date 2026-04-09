# Crane: Machine Learning for Neural Interfaces

Crane provides useful components to facilitate the development and training of machine learning models for neural data. It includes tools for data handling and common neural model components, making it easier to build and experiment with models in this domain. It is designed to be compatible with [Hugging Face Transformers](https://github.com/huggingface/transformers), [PyTorch](https://pytorch.org/), and the [Insight-Neuro Ecosystem](https://github.com/insight-neuro), allowing researchers to share models and preprocessing pipelines seamlessly.

## Installation

The recommended way to install Crane is via pip:

```bash
pip install "crane @ git+https://github.com/insight-neuro/crane"
```

## Quick Start

To get started with Crane, you can build your own model by subclassing `BrainModel` and `BrainFeatureExtractor`. These classes provide a structured way to define your model architecture and preprocessing steps.