# Indian Statistical Institute:-Summer-School-Project
Scientific Machine Learning For Computational Physics:-- Neural Operator Benchmarking on Darcy Flow Resolution Generalization

## code1.py

This file contains the complete implementation of the proposed **Dual-Embedding Multi-Particle Neural Operator Transformer (Dual-MPNOT)** for the Darcy Flow benchmark. It is still in development. 

### Features

* Automatic download and preprocessing of the Darcy Flow dataset.
* Bilinear downsampling and normalization pipeline.
* Construction of custom PyTorch datasets and dataloaders.
* Dual-embedding architecture that independently encodes physical field values and spatial coordinates.
* Multi-Particle Attention mechanism with learnable diffusion-based spatial decay.
* Hyperparameter sweep over multiple learning rates and training epochs.
* Automated model selection using the lowest Relative L2 error.
* Performance evaluation using Mean Squared Error (MSE), Relative L2 Error, training time, and inference time.
* Visualization of learning curves, prediction fields, error maps, scatter plots, and error histograms.
* Zero-shot evaluation on higher-resolution spatial grids to investigate resolution generalization.

The implementation is designed as an end-to-end experimental pipeline, covering data preparation, model training, evaluation, visualization, and performance analysis within a single executable script.

# Code 2: Neural Operator Benchmark Suite

This code provides a unified PyTorch implementation of four transformer-based neural operator architectures for solving the Darcy Flow benchmark:

- **CATO (Charted Axial Transformer Operator)**
- **LinearNO (Kernelized Linear Attention Neural Operator)**
- **Transolver (Physics Slice-Attention Neural Operator)**
- **FactFormer (Axial Factorized Transformer Operator)**

The entire benchmarking pipeline is contained within a single script, enabling direct comparison of the four architectures under identical preprocessing, training, and evaluation settings.

## Features

- Automatic download and extraction of the Darcy Flow benchmark dataset.
- Data preprocessing including bilinear downsampling and z-score normalization.
- Unified training pipeline for all four neural operator architectures.
- Hyperparameter sweep over multiple learning rates and training epochs.
- Automatic selection of the best-performing model based on Relative L2 Error.
- Performance evaluation using:
  - Mean Squared Error (MSE)
  - Relative L2 Error
  - Training Time
  - Inference Time
- Comprehensive visualization of:
  - Training and validation loss curves
  - Relative L2 convergence
  - Training and inference time
  - Ground truth fields
  - Predicted fields
  - Absolute error maps
  - Prediction vs. Ground Truth scatter plots
  - Error histograms
- Cross-resolution (zero-shot) evaluation for investigating spatial generalization.

## Implemented Architectures

### CATO
Charted Axial Transformer Operator utilizing learned chart coordinates together with axial attention to efficiently model spatial dependencies on structured grids.

### LinearNO
Kernelized linear attention neural operator that replaces quadratic self-attention with linear attention for improved computational efficiency on large spatial domains.

### Transolver
Physics Slice-Attention Neural Operator employing learnable slice representations to reduce computational complexity while preserving long-range interactions.

### FactFormer
Axial Factorized Transformer Operator performing attention independently along the height and width dimensions to efficiently model two-dimensional spatial dependencies.

## Dataset

- Darcy Flow Benchmark Dataset
- Automatic download from Zenodo
- Configurable spatial resolution
- Configurable train/test sample sizes

## Technologies

- Python
- PyTorch
- NumPy
- Pandas
- Matplotlib

## Purpose

This implementation serves as a unified benchmarking framework for comparing multiple state-of-the-art transformer-based neural operator architectures on the Darcy Flow PDE benchmark under a consistent experimental setup.
# Code 3 – Geometric Neural Operator (GNOT) for Darcy Flow

This repository contains a from-scratch implementation of the **Geometric Neural Operator (GNOT)** in **PyTorch** for solving the Darcy Flow benchmark. The model combines **multi-head self-attention** with a **coordinate-aware Mixture-of-Experts (MoE) feed-forward network**, enabling spatially adaptive operator learning. The project includes automated data preprocessing, hyperparameter sweeps, extensive runtime sanity checks, performance evaluation, and zero-shot resolution generalization.

## Features

- From-scratch implementation of the Geometric Neural Operator (GNOT)
- Automatic download and preprocessing of the Darcy Flow dataset
- Bilinear interpolation for configurable working resolutions
- Z-score normalization using training-set statistics
- Coordinate-aware geometric gating with a Mixture-of-Experts (MoE) module
- Multi-head self-attention with residual connections and Layer Normalization
- Hyperparameter sweep across multiple learning rates and epoch configurations
- Automatic best-model selection based on Relative L2 Error
- Evaluation using MSE, Relative L2 Error, training time, and inference time
- Zero-shot resolution transfer from **8×8** training to **16×16** inference
- Automatic generation of performance plots and prediction visualizations

## Model Architecture

The GNOT architecture consists of:

- **Lifting Layer** – Projects scalar input values into a latent feature space.
- **Heterogeneous Normalized Attention** – Captures long-range interactions using multi-head self-attention.
- **Geometric Gated Feed-Forward Network** – Uses spatial coordinates to generate gating weights for a Mixture-of-Experts module.
- **Residual Connections & Layer Normalization** – Improve optimization stability and feature propagation.
- **Projection Layer** – Maps latent representations back to the predicted solution field.

## Dataset

- **Darcy Flow Benchmark**
- Original Resolution: **64 × 64**
- Training Resolution: **8 × 8**
- Zero-Shot Evaluation Resolution: **16 × 16**

## Training Pipeline

1. Download the Darcy Flow dataset automatically.
2. Verify dataset integrity using built-in sanity checks.
3. Downsample fields using bilinear interpolation.
4. Normalize inputs and targets with z-score normalization.
5. Construct spatial coordinate grids.
6. Train GNOT across multiple hyperparameter configurations.
7. Select the best-performing model based on Relative L2 Error.
8. Evaluate performance on the test dataset.
9. Assess zero-shot generalization on a higher spatial resolution.

## Evaluation Metrics

- Mean Squared Error (MSE)
- Relative L2 Error
- Training Loss
- Test Loss
- Training Time
- Inference Time

## Generated Outputs

The implementation automatically saves:

- `GNOT_All_Plots.png`
- `GNOT_Resolution_16_Plots.png`

These figures include:

- Training loss curves
- Test loss curves
- Relative L2 error curves
- Training time analysis
- Inference time analysis
- Ground truth field visualization
- Predicted field visualization
- Absolute error map
- Prediction vs. Ground Truth scatter plot
- Error histogram

## Requirements

- Python 3
- PyTorch
- NumPy
- Pandas
- Matplotlib
- tqdm

This implementation serves as a research-oriented baseline for studying **Geometric Neural Operators**, attention-based operator learning, coordinate-aware Mixture-of-Experts architectures, and zero-shot resolution generalization on PDE surrogate modeling tasks.
# Code 4: Multi-Particle Neural Operator Transformer (MPNOT) for Darcy Flow

## Overview

**Code 4** implements a Multi-Particle Neural Operator Transformer (MPNOT) for learning solution operators of the two-dimensional Darcy Flow equation. The implementation includes automatic dataset acquisition, preprocessing, hyperparameter optimization, training, evaluation, visualization, and zero-shot resolution generalization experiments.

The workflow includes:

- Automatic download of the Darcy Flow benchmark dataset
- Data preprocessing and normalization
- Bilinear downsampling to an 8×8 working resolution
- MPNOT architecture implementation
- Hyperparameter sweep over learning rates and epochs
- Training and evaluation
- Performance visualization
- Zero-shot evaluation at 16×16 and 64×64 resolutions

---
# Code 5: Geometry-Aware Attentional Operator Transformer (MAGNO)

## Overview

This notebook implements a complete **Geometry-Aware Attentional Operator Transformer (MAGNO)** for solving the Darcy Flow operator learning benchmark.

Unlike conventional neural operators, this implementation combines:

- Geometry-aware feature embedding
- Multi-scale graph neural operator attention
- Transformer-based latent processing
- Resolution transfer (8×8 → 16×16)
- Hyperparameter sweeping
- Extensive runtime sanity verification

The implementation is written entirely in **PyTorch**.

---

# Workflow

The execution pipeline follows the sequence below:

1. Download Darcy Flow dataset
2. Downsample data to working resolution
3. Normalize inputs and outputs
4. Generate spatial coordinate grid
5. Construct operator datasets
6. Train MAGNO across multiple hyperparameter settings
7. Select the best-performing model
8. Evaluate on the training resolution
9. Perform zero-shot resolution transfer to 16×16
10. Generate performance plots

---

# Dataset

Dataset:
- Darcy Flow Benchmark
- Zenodo Release

Original resolution:

- 64 × 64

Working resolution:

- 8 × 8

Zero-shot evaluation resolution:

- 16 × 16

---

# Configuration

```python
CONFIG = {
    "dataset_resolution": 64,
    "working_resolution": 8,
    "train_samples": 200,
    "test_samples": 50,
    "batch_size": 8,
    "epochs_list": [20, 50, 100],
    "learning_rates": [1e-2, 1e-3, 1e-4],
}
```

---

# Data Preprocessing

The preprocessing pipeline performs:

- Dataset download
- Dataset extraction
- Bilinear downsampling
- Z-score normalization
- Dataset slicing
- Coordinate grid generation
- Branch/target vector construction
- DataLoader creation

The branch input is flattened into

```
8 × 8 = 64
```

features.

The target field is flattened into the same dimensionality.

---

# Geometry Embedding

The `GeometryEmbedding` module extracts continuous geometric descriptors from the spatial grid.

For every node it computes

- coordinate location
- pairwise spatial distances
- local density statistics

These descriptors are projected into a learnable embedding space before attention is applied.

---

# MAGNO Layer

The core operator layer is implemented by `MultiscaleAttentionalGNO`.

Each layer performs

- query projection
- key projection
- value projection
- edge feature construction
- radius-based neighborhood masking
- multi-head attention
- multi-scale attention fusion

Three interaction radii are used:

```
0.15
0.35
0.65
```

The outputs from every scale are combined using learnable fusion weights.

---

# Transformer Processor

After graph encoding, latent representations are processed using Transformer blocks.

Each block contains

- Multi-head self-attention
- Feed-forward network
- Layer normalization
- Residual connections

These layers model long-range interactions within the latent operator representation.

---

# MAGNONet Architecture

The complete architecture is

```
Input Field
      │
      ▼
Lifting Layer
      │
      ▼
Geometry Embedding
      │
      ▼
MAGNO Encoder
      │
      ▼
Transformer Block ×2
      │
      ▼
MAGNO Decoder
      │
      ▼
Projection Layer
      │
      ▼
Predicted Solution Field
```

---

# Training

Training is performed using

- Mean Squared Error loss
- Adam optimizer

Hyperparameter sweep:

Learning rates

```
1e-2
1e-3
1e-4
```

Epochs

```
20
50
100
```

This produces nine independent training experiments.

---

# Evaluation

For every experiment the following metrics are recorded:

- Training loss
- Test loss
- Relative L2 error
- Training time
- Inference time
- Final MSE

The best model is selected using the minimum Relative L2 error.

---

# Resolution Generalization

After training on

```
8 × 8
```

fields,

the best model is directly evaluated on

```
16 × 16
```

fields without retraining.

The notebook reports

- MSE
- Relative L2
- Inference time

for this zero-shot resolution transfer experiment.

---

# Visualization

The notebook automatically generates:

1. Training loss
2. Test loss
3. Relative L2 error
4. Training time
5. Inference time
6. Train error
7. Test error
8. Ground truth field
9. Predicted field
10. Absolute error map
11. Prediction vs ground truth scatter plot
12. Error histogram

Figures are saved as

```
MAGNO_All_Plots.png
```

and

```
MAGNO_Resolution_16_Plots.png
```

---

# Runtime Verification

The implementation contains extensive assertion checks that verify:

- tensor dimensions
- grid dimensions
- normalization validity
- DataLoader outputs
- encoder and decoder tensor shapes
- Transformer dimensions
- prediction dimensions
- visualization compatibility

These checks are intended to detect shape mismatches during execution.

---

# Dependencies

- Python 3
- PyTorch
- NumPy
- Pandas
- Matplotlib
- tqdm

---

# Output

The notebook produces

- trained MAGNO models
- experiment summary table
- performance metrics
- visualization figures
- zero-shot resolution transfer evaluation
- saved performance plots
