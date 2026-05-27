# Reproducibility Notes

This document explains what can be reproduced from the repository, what requires supplied checkpoints, and what requires substantial compute.

## Reproducibility levels

### 1. Smoke-test reproducibility

Smoke tests verify that the scripts import correctly, the dataset layout is valid, and small training/evaluation paths run without crashing.

These tests do not reproduce the experiments and thesis WER values.

Recommended smoke tests:

```bash
python scripts/IKT590_evaluate_ASR_pipeline.py --help

python scripts/IKT590_train_ASR_encoder_SSSM.py \
  --data-path hub_data/librispeech \
  --subset-train 4000 \
  --subset-val 400 \
  --batch-size 16 \
  --d-model 64 \
  --n-layers 6 \
  --epochs 2 \
  --exp-name smoke_ssssm

python scripts/IKT590_train_LM_char_ngram.py \
  --train-text lm_checkpoints/kenlm/char_level_text.txt \
  --orders 7 \
  --max-lines 10000 \
  --output-dir lm_checkpoints/ngram_smoke
```

### 2. Evaluation reproducibility with provided checkpoints


Given the saved LibriSpeech datasets and the required encoder/LM checkpoints, the final evaluation can be reproduced by running the evaluation script with the same checkpoint paths, beam width, and shallow-fusion weights used in the thesis.

```bash
python scripts/IKT590_evaluate_ASR_pipeline.py \
  --mode full_eval \
  --data-path hub_data/librispeech \
  --asr-checkpoint encoder_checkpoints/S-SSSM/960h/<checkpoint>.ckpt \
  --asr-type sssm \
  --elm-path lm_checkpoints/elm_mamba_MaxChars-1000000000_d320_L18 \
  --alpha <ALPHA> \
  --beta 0
  --beam-width 10 \
  --batch-size 64 \
  --exp-name final_ssssm_charmamba1_eval
```

If the tuned `alpha` values are not known, run:

```bash
python scripts/IKT590_evaluate_ASR_pipeline.py \
  --mode tune \
  --data-path hub_data/librispeech \
  --asr-checkpoint encoder_checkpoints/S-SSSM/960h/<checkpoint>.ckpt \
  --asr-type sssm \
  --elm-path lm_checkpoints/elm_mamba_MaxChars-1000000000_d320_L18 \
  --beam-width 10 \
  --batch-size 64 \
  --subset-val 1000 \
  --exp-name tune_ssssm_charmamba1
```

### 3. Full training reproducibility

Full training can be rerun from the scripts, but it is compute-heavy and may not exactly reproduce checkpoint weights or WER values due to nondeterminism in GPU training and library kernels.

The full 960 h S-SSSM training command is:

```bash
python scripts/IKT590_train_ASR_encoder_SSSM.py \
  --data-path hub_data/librispeech \
  --dataset-config 960h \
  --d-model 320 \
  --n-layers 48 \
  --batch-size 64 \
  --seed 456 \
  --epochs 50 \
  --exp-name ssssm_960h_W-320_D-46_B-128_E-50
```

## Software assumptions

Recommended base environment:

- Python 3.10 or 3.11.
- PyTorch and torchaudio built for the installed CUDA version.
- Lightning.
- HuggingFace `datasets`.
- `jiwer`.
- `numpy`, `pandas`, and `matplotlib`.

The scripts were written around recent PyTorch/Lightning usage. The Mamba-related scripts are more sensitive to exact CUDA, PyTorch, and `mamba-ssm` versions than the S-SSSM and n-gram scripts.

Known working setup indicated by script docstrings for Mamba-1/CharMamba-1:

```bash
pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install causal-conv1d==1.5.0.post8
pip install mamba-ssm==2.2.4
pip install lightning jiwer datasets numpy pandas matplotlib
```

Mamba-3 may require installing `mamba-ssm` from GitHub rather than PyPI:

```bash
TORCH_CUDA_ARCH_LIST="8.0" pip install --force-reinstall --no-deps \
  git+https://github.com/state-spaces/mamba.git \
  --no-build-isolation
```

Depending on the system, additional packages such as `tilelang` or `quack-kernels` may be needed for Mamba-3/Triton kernels.

## Hardware assumptions

- CPU:
  - Suitable for syntax checks, `--help`, dataset download, and very small n-gram tests.
  - Not suitable for full encoder training.
- Consumer NVIDIA GPU:
  - Suitable for S-SSSM smoke tests and smaller experiments.
  - May be suitable for Mamba-1 if `mamba-ssm` and `causal-conv1d` install correctly.
- A100/H100 or compatible modern NVIDIA GPU:
  - Recommended for Mamba-3/Triton scripts.
  - The CharMamba-3 script explicitly targets A100/H100-style execution and may fail on older GPUs such as V100.

## Dataset source and layout

The repository uses LibriSpeech ASR from HuggingFace:

```bash
python scripts/download_dataset.py
```

Expected output:

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

All ASR scripts should be given:

```bash
--data-path hub_data/librispeech
```

## Seed handling

The training and evaluation scripts expose a `--seed` argument. The default seed used in the thesis commands is `456`.

Example:

```bash
python scripts/IKT590_train_ASR_encoder_SSSM.py \
  --data-path hub_data/librispeech \
  --dataset-config 100h \
  --seed 456 \
  --epochs 30
```

Setting the seed improves repeatability, but it does not guarantee bit-identical results across hardware, CUDA versions, PyTorch versions, or Mamba kernels.

## Known nondeterminism sources

Expected nondeterminism sources include:

- CUDA kernel scheduling and floating-point reduction order.
- cuDNN and PyTorch backend choices.
- PyTorch Lightning training loops and checkpoint timing.
- Multi-worker data loading.
- Mixed precision and TF32 execution.
- CTC loss implementation details.
- Mamba CUDA kernels.
- Triton kernels used by Mamba-3.
- Hardware differences across GPU architectures.

For stricter comparisons, document:

```bash
python --version
pip freeze
nvidia-smi
```

and keep the same checkpoint files.

## Checkpoint dependencies

Final thesis evaluation requires:

```text
encoder_checkpoints/S-SSSM/960h/<checkpoint>.ckpt
lm_checkpoints/elm_mamba_MaxChars-1000000000_d320_L18/elm_best.pt
```

The evaluation script loads the LM by directory for CharMamba models:

```bash
--elm-path lm_checkpoints/elm_mamba_MaxChars-1000000000_d320_L18
```

For n-gram control evaluation, use the `.pkl` file directly:

```bash
--elm-path lm_checkpoints/ngram/char_10gram.pkl
```

See `CHECKPOINTS.md` for details.

## Required commands

### Dataset download

```bash
python scripts/download_dataset.py
```

### Quick smoke test

```bash
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

### Final full evaluation

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
  --exp-name final_ssssm_charmamba1_eval
```

### n-gram control evaluation

```bash
python scripts/IKT590_evaluate_ASR_pipeline.py \
  --mode full_eval \
  --data-path hub_data/librispeech \
  --asr-checkpoint encoder_checkpoints/S-SSSM/960h/<checkpoint>.ckpt \
  --asr-type sssm \
  --elm-path lm_checkpoints/ngram/char_10gram.pkl \
  --alpha <ALPHA> \
  --beta 0.0 \
  --beam-width 10 \
  --batch-size 64 \
  --exp-name final_ssssm_ngram10_eval
```

### RTFX measurement

```bash
python scripts/IKT590_evaluate_ASR_pipeline.py \
  --mode rtfx \
  --data-path hub_data/librispeech \
  --asr-checkpoint encoder_checkpoints/S-SSSM/960h/<checkpoint>.ckpt \
  --asr-type sssm \
  --elm-path lm_checkpoints/elm_mamba_MaxChars-1000000000_d320_L18 \
  --alpha <ALPHA> \
  --beta <BETA> \
  --beam-width 10 \
  --batch-size 64 \
  --exp-name rtfx_ssssm_charmamba1
```

### Streaming verification

```bash
python scripts/IKT590_evaluate_ASR_pipeline.py \
  --mode verify_streaming \
  --data-path hub_data/librispeech \
  --asr-checkpoint encoder_checkpoints/S-SSSM/960h/<checkpoint>.ckpt \
  --asr-type sssm \
  --elm-path lm_checkpoints/elm_mamba_MaxChars-1000000000_d320_L18 \
  --beam-width 10 \
  --batch-size 16 \
  --exp-name verify_streaming_charmamba1
```

## Troubleshooting

### `datasets` cannot load LibriSpeech from disk

Check that the dataset was saved by `download_dataset.py` and that the expected folders exist:

```bash
ls hub_data/librispeech/clean
ls hub_data/librispeech/other
```

### `mamba_ssm` import fails

For S-SSSM-only experiments, this is not fatal. For Mamba-1 and CharMamba-1, install:

```bash
pip install causal-conv1d==1.5.0.post8
pip install mamba-ssm==2.2.4
```

For Mamba-3, install from GitHub as described in `setup_guide.md`.

### Mamba-3 fails on GPU

Verify GPU compatibility:

```bash
nvidia-smi
```

Mamba-3/Triton scripts are intended for A100/H100-class hardware. Older GPUs may fail even when PyTorch itself works.

### Evaluation cannot find checkpoints

Confirm the exact paths:

```bash
find encoder_checkpoints -name "*.ckpt"
find lm_checkpoints -name "elm_best.pt"
find lm_checkpoints -name "*.pkl"
```

Then pass those exact paths to `--asr-checkpoint` and `--elm-path`.

### Results differ slightly from thesis values

Small differences can occur due to CUDA, PyTorch, Lightning, Mamba kernels, batch size, or tuned shallow-fusion weights. For thesis-level comparison, use the provided checkpoints and the same `alpha`, `beta`, `beam-width`, and evaluation mode.
