# Setup Guide

This guide separates the standard environment from the Mamba-specific setup.

## 1. Standard environment

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install PyTorch for your CUDA version. Example for CUDA 12.4:

```bash
pip install --upgrade pip
pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
```

Install repository dependencies:

```bash
pip install -r requirements.txt
```

Verify script entry points:

```bash
python scripts/IKT590_evaluate_ASR_pipeline.py --help
python scripts/IKT590_train_ASR_encoder_SSSM.py --help
python scripts/IKT590_train_LM_char_ngram.py --help
```

## 2. Dataset setup

Download LibriSpeech ASR from HuggingFace and save it to disk:

```bash
python scripts/download_dataset.py
```

Expected layout:

```text
hub_data/librispeech/
├── clean/
│   ├── train.100
│   ├── train.360
│   ├── validation
│   └── test
└── other/
    ├── train.500
    ├── validation
    └── test
```

All ASR scripts should use:

```bash
--data-path hub_data/librispeech
```

## 3. Mamba-1 and CharMamba-1 setup

The Mamba-1 encoder and CharMamba-1 LM use `mamba-ssm` and `causal-conv1d`.

Recommended versions from the script docstrings:

```bash
pip install causal-conv1d==1.5.0.post8
pip install mamba-ssm==2.2.4
```

Verify import:

```bash
python - <<'PY'
from mamba_ssm import Mamba
print("Mamba-1 import OK")
PY
```

If this fails, check:

```bash
python --version
python -c "import torch; print(torch.__version__, torch.version.cuda)"
nvidia-smi
```

## 4. Mamba-3 / Triton setup

Mamba-3 support is more restrictive than the standard environment. The CharMamba-3 script is intended for A100/H100-style execution and may fail on older GPUs.

The script expects `Mamba3` to be importable from `mamba_ssm`. PyPI versions may not be sufficient. A GitHub install may be required:

```bash
TORCH_CUDA_ARCH_LIST="8.0" pip install --force-reinstall --no-deps \
  git+https://github.com/state-spaces/mamba.git \
  --no-build-isolation
```

Some setups may also require:

```bash
pip install tilelang==0.1.8 quack-kernels==0.3.1
```

Verify Mamba-3 availability:

```bash
python scripts/IKT590_train_LM_CharMamba-3_Triton.py --mode verify
```

If the verification fails on an older GPU, this is expected. Use S-SSSM, Mamba-1, or provided checkpoints for evaluation.

## 5. Checkpoint placement

Place encoder checkpoints under:

```text
encoder_checkpoints/
```

Place LM checkpoints under:

```text
lm_checkpoints/
```

See `CHECKPOINTS.md` for the exact recommended layout.

## 6. Minimal local verification checklist

Run:

```bash
python scripts/IKT590_evaluate_ASR_pipeline.py --help
python scripts/download_dataset.py
python scripts/IKT590_train_ASR_encoder_SSSM.py \
  --data-path hub_data/librispeech \
  --subset-train 4000 \
  --subset-val 400 \
  --batch-size 16 \
  --d-model 64 \
  --n-layers 6 \
  --epochs 2 \
  --exp-name smoke_ssssm
```

Then run one evaluation command with real checkpoints:

```bash
python scripts/IKT590_evaluate_ASR_pipeline.py \
  --mode full_eval \
  --data-path hub_data/librispeech \
  --asr-checkpoint encoder_checkpoints/S-SSSM/960h/<checkpoint>.ckpt \
  --asr-type sssm \
  --elm-path lm_checkpoints/elm_mamba_MaxChars-1000000000_d320_L18 \
  --alpha <ALPHA> \
  --beta <BETA> \
  --beam-width 10 \
  --batch-size 64 \
  --exp-name local_final_eval_check
```
