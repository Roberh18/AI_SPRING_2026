# State Space Models for Automatic Speech Transcription

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the implementation for my ICT project at the University of Agder (IKT464).

## Overview

This work investigates **State Space Models (SSM)** for Automatic Speech Recognition (ASR). Unlike standard SSM implementations that initialize all layers uniformly, this architecture initializes layers with **progressively slower temporal decay rates**, explicitly encoding an inductive bias that mirrors human auditory processing:

- **Early layers**: Fast-decaying states for short-term acoustic cues (10-50 ms)
- **Deep layers**: Slow-decaying states for longer linguistic timescales (100s of ms to seconds)

## Architecture
```
Input Audio → Log-Mel Spectrogram → Conv Subsampler (4×) → S-SSSM Encoder (L layers) → CTC Classifier
```

Each S-SSSM layer contains:
- Layer Normalization
- Input Projection (with gating split)
- Depthwise Convolution (local context)
- Selective SSM Recurrence (with hierarchical dynamics)
- GLU Gating (optional)
- Output Projection + Residual

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- ~10GB disk space for LibriSpeech 100h dataset

### Quick Install
```bash
# Clone the repository
git clone https://github.com/Roberh18/IKT464.git
cd IKT464

# Install dependencies (one-liner)
pip3 install --user torch torchaudio torchvision jiwer "lightning>=2.0" matplotlib "datasets[audio]==2.18.0" soundfile librosa && pip3 install --user "numpy<2.0" --force-reinstall

# Verify installation
python3 -c "import torch, torchaudio, jiwer, lightning, datasets, numpy, matplotlib; print(' All dependencies installed successfully!')"
```

### Step-by-Step Installation

If the one-liner fails:
```bash
# Step 1: Core PyTorch
pip3 install --user torch torchaudio torchvision

# Step 2: ML/ASR libraries
pip3 install --user jiwer "lightning>=2.0"

# Step 3: Audio processing (CRITICAL: specific version)
pip3 install --user "datasets[audio]==2.18.0" soundfile librosa

# Step 4: Plotting
pip3 install --user matplotlib

# Step 5: Fix NumPy compatibility (MUST be last)
pip3 install --user "numpy<2.0" --force-reinstall
```

## Dataset

This project uses the [LibriSpeech](https://www.openslr.org/12) corpus from Hugging Face.

### Download Dataset

Use the provided notebook or download directly:
```python
from datasets import load_dataset

# Download LibriSpeech (this will cache locally)
dataset = load_dataset("openslr/librispeech_asr", "clean")

# Save to disk for faster loading
dataset.save_to_disk("./hub_data/librispeech")
```

Or run the `download_dataset.ipynb` notebook.

## Usage

### Training
```bash
# Basic training with hierarchical initialization and gating (recommended)
python IKT464_AST_SSSSM.py --data-path ./hub_data/librispeech --exp-name my_experiment

# Baseline without hierarchical features
python IKT464_AST_SSSSM.py --data-path ./hub_data/librispeech --no-hierarchical --no-gating --exp-name baseline

# Custom configuration
python IKT464_AST_SSSSM.py \
    --data-path ./hub_data/librispeech \
    --d-model 336 \
    --n-layers 12 \
    --batch-size 32 \
    --epochs 30 \
    --lr 1e-3 \
    --exp-name custom_experiment
```

### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data-path` | `../hub_data/librispeech` | Path to LibriSpeech dataset |
| `--dataset-config` | `100h` | Training set: `100h` or `460h` |
| `--d-model` | `384` | Model dimension |
| `--n-layers` | `8` | Number of encoder layers |
| `--batch-size` | `32` | Training batch size |
| `--epochs` | `30` | Number of training epochs |
| `--lr` | `1e-3` | Learning rate |
| `--no-hierarchical` | False | Disable hierarchical state dynamics |
| `--no-gating` | False | Disable gating mechanism |
| `--seed` | `456` | Random seed |

### Quick Test
```bash
# Small-scale test to verify setup
python IKT464_AST_SSSSM.py \
    --data-path ./hub_data/librispeech \
    --subset-train 4000 \
    --subset-val 400 \
    --d-model 64 \
    --n-layers 6 \
    --epochs 5 \
    --exp-name test_run
```

## Pre-trained Models

Pre-trained model checkpoints are available on Hugging Face:

🤗 **[Roberh18/IKT464](https://huggingface.co/Roberh18/IKT464)**

### Available Checkpoints

| Experiment | Configuration | Link |
|------------|---------------|------|
| exp-01 | Baseline W386×D9 | [best_model.ckpt](https://huggingface.co/Roberh18/IKT464/blob/main/experiments/exp-01_baseline_W386_D9/best_model.ckpt) |
| exp-02 | Hier+Gate W358×D9 | [best_model.ckpt](https://huggingface.co/Roberh18/IKT464/blob/main/experiments/exp-02_hier_gating_W358_D9/best_model.ckpt) |
| ... | More experiments | [Browse all](https://huggingface.co/Roberh18/IKT464/tree/main/experiments) |

### Loading a Checkpoint
```python
import torch
from IKT464_AST_SSSSM import SSMASRModel

# Load checkpoint
checkpoint = torch.load("path/to/best_model.ckpt")
model = SSMASRModel.load_from_checkpoint("path/to/best_model.ckpt")
model.eval()
```

## Experiments

### Architecture Ablation 
```bash
# Exp01: Baseline
python IKT464_AST_SSSSM.py --no-hierarchical --no-gating --d-model 336 --n-layers 12 --exp-name baseline_W336_D12

# Exp02: + Hierarchical
python IKT464_AST_SSSSM.py --no-gating --d-model 336 --n-layers 12 --exp-name hier_W336_D12

# Exp03: + Gating
python IKT464_AST_SSSSM.py --no-hierarchical --d-model 312 --n-layers 12 --exp-name gating_W312_D12

# Exp04: + Both (recommended)
python IKT464_AST_SSSSM.py --d-model 312 --n-layers 12 --exp-name hier_gating_W312_D12
```

### Depth vs Width Analysis 

All configurations maintain ~8.6M parameters:

| Depth | Width (Baseline) | Width (Hier+Gate) |
|-------|------------------|-------------------|
| 9 | 386 | 358 |
| 12 | 336 | 312 |
| 18 | 276 | 256 |
| 24 | 240 | 222 |
| 30 | 216 | 200 |

## Troubleshooting

### TorchCodec Errors
```bash
pip3 uninstall -y torchcodec
export HF_DATASETS_DISABLE_TORCHCODEC=1
```

### NumPy Compatibility
```bash
pip3 install --user "numpy<2.0" --force-reinstall
```

### Dataset Issues
```bash
pip3 uninstall -y datasets
pip3 install --user "datasets[audio]==2.18.0"
```

### CUDA Out of Memory

- Reduce `--batch-size` (try 16 or 8)
- Reduce `--d-model` or `--n-layers`
- Use gradient accumulation (modify code)

## Project Structure
```
IKT464/
├── experiments/                       # Experiment outputs (checkpoints, logs)
│   ├── exp-01_baseline_W386_D9/       # (best_model.ckpt ~100MB only on HuggingFace)
│   ├── exp-02_hier_gating_W358_D9/    # (best_model.ckpt ~100MB only on HuggingFace)
│   ├── exp-03_baseline_W336_D12/      # (best_model.ckpt ~100MB only on HuggingFace)
│   │   ...
│   └── exp-10_hier_gating_W200_D30/   # (best_model.ckpt ~100MB only on HuggingFace)
├── src/                               # Source code
│   ├── download_dataset.ipynb         # Dataset download notebook
│   └── IKT464_AST_SSSSM.py            # Main training script
├── README.md                          # This file
└── requirements.txt                   # Python dependencies
```
