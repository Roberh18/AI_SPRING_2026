# State Space Models for Automatic Speech Transcription

This repository contains the cleaned code for a master's thesis project on attention-free automatic speech transcription with State Space Models (SSMs). It supports training and evaluating S-SSSM, Mamba-1, and Mamba-3 CTC acoustic encoders on LibriSpeech, plus character-level language-model rescoring using CharMamba-1, CharMamba-3, or a pure Python character n-gram baseline with Stupid Backoff.

## What this repository contains

- S-SSSM CTC acoustic encoder training.
- Mamba-1 CTC acoustic encoder training.
- Mamba-3 CTC acoustic encoder training.
- CharMamba-1 character-level language model training.
- CharMamba-3 character-level language model training.
- Pure Python character n-gram LM training with Stupid Backoff.
- ASR evaluation with greedy CTC, prefix beam search, shallow fusion, WER, RTFX, hallucination analysis, score analysis, and streaming verification.
- LibriSpeech dataset download/setup helper.
- Optional LM diagnostic utilities, if included in the repository.

Recommended structure:

```text
.
├── README.md
├── REPRODUCIBILITY.md
├── CHECKPOINTS.md
├── setup_guide.md
├── SCRIPT_OVERVIEW.md
├── requirements.txt
├── environment.yml
├── .gitignore
├── scripts/
│   ├── download_dataset.py
│   ├── IKT590_train_ASR_encoder_SSSM.py
│   ├── IKT590_train_ASR_encoder_mamba-1.py
│   ├── IKT590_train_ASR_encoder_mamba-3.py
│   ├── IKT590_train_LM_char_ngram.py
│   ├── IKT590_train_LM_CharMamba-1.py
│   ├── IKT590_train_LM_CharMamba-3_Triton.py
│   └── IKT590_evaluate_ASR_pipeline.py
├── hub_data/
│   └── librispeech/
│       ├── clean/
│       └── other/
├── encoder_checkpoints/
├── lm_checkpoints/
└── results/
```

If your local script names differ, use the names actually present in `scripts/`.

## Fast path: evaluate the final pipeline

The intended examiner path is evaluation with existing checkpoints. Full 960 h training is compute-heavy and is not expected to run quickly on a normal workstation.

Expected inputs:

```text
hub_data/librispeech/clean/
hub_data/librispeech/other/
encoder_checkpoints/S-SSSM/960h/<encoder-checkpoint>.ckpt
lm_checkpoints/elm_mamba_MaxChars-1000000000_d320_L18/elm_best.pt
```

Example final evaluation command:

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

Use the thesis-tuned `alpha` and `beta` values if documented. Otherwise, run `--mode tune` first on the development split and record the selected values.

## Installation

A standard environment is sufficient for S-SSSM training, n-gram LM training, dataset setup, and basic script checks.

```bash
python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

For CUDA PyTorch, install the PyTorch wheel matching your CUDA version before or while installing the remaining dependencies. Example for CUDA 12.4:

```bash
pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Mamba-1 acceleration requires `mamba-ssm` and `causal-conv1d`. Mamba-3/Triton scripts may require installation from GitHub and an A100/H100-class GPU. See `setup_guide.md`.

## Dataset preparation

Download and save LibriSpeech ASR splits in HuggingFace `save_to_disk` format:

```bash
python scripts/download_dataset.py
```

Expected dataset layout:

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

## Checkpoint placement

See `CHECKPOINTS.md` for the full checkpoint layout. In brief:

```text
encoder_checkpoints/
└── S-SSSM/
    └── 960h/
        └── <checkpoint>.ckpt

lm_checkpoints/
├── elm_mamba_MaxChars-1000000000_d320_L18/
│   └── elm_best.pt
└── ngram/
    └── char_10gram.pkl
```

The evaluation script accepts alternative checkpoint paths through:

```bash
--asr-checkpoint <path-to-encoder.ckpt>
--elm-path <path-to-lm-dir-or-pkl>
```

## Quick smoke tests

These commands verify installation, dataset layout, and script entry points. They do not reproduce thesis results.

### Dataset availability check

```bash
python scripts/download_dataset.py
```

If the data is already downloaded, the script should still verify that the saved HuggingFace datasets can be reloaded.

### Evaluation help check

```bash
python scripts/IKT590_evaluate_ASR_pipeline.py --help
```

### S-SSSM tiny training run

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

### n-gram LM tiny training run

This requires a character-level training text file. If unavailable, create it using the same preprocessing used for the thesis experiments, or provide a small compatible file with one character-tokenized sentence per line.

```bash
python scripts/IKT590_train_LM_char_ngram.py \
  --train-text lm_checkpoints/kenlm/char_level_text.txt \
  --orders 7 \
  --max-lines 10000 \
  --output-dir lm_checkpoints/ngram_smoke
```

## Evaluation commands

### Tune shallow-fusion weights

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

### Full evaluation

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
  --exp-name eval_ssssm_charmamba1
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
  --exp-name eval_ssssm_ngram10
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

### Hallucination and score analysis

```bash
python scripts/IKT590_evaluate_ASR_pipeline.py \
  --mode hallucination \
  --data-path hub_data/librispeech \
  --asr-checkpoint encoder_checkpoints/S-SSSM/960h/<checkpoint>.ckpt \
  --asr-type sssm \
  --elm-path lm_checkpoints/elm_mamba_MaxChars-1000000000_d320_L18 \
  --alpha <ALPHA> \
  --beta <BETA> \
  --beam-width 10 \
  --batch-size 64 \
  --exp-name hallucination_ssssm_charmamba1

python scripts/IKT590_evaluate_ASR_pipeline.py \
  --mode score_analysis \
  --data-path hub_data/librispeech \
  --asr-checkpoint encoder_checkpoints/S-SSSM/960h/<checkpoint>.ckpt \
  --asr-type sssm \
  --elm-path lm_checkpoints/elm_mamba_MaxChars-1000000000_d320_L18 \
  --alpha <ALPHA> \
  --beta <BETA> \
  --beam-width 10 \
  --batch-size 64 \
  --exp-name score_analysis_ssssm_charmamba1
```

## Training commands

Full training commands are expensive. Use them only with adequate GPU resources and disk space.

### S-SSSM encoder

Smoke test:

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

Full 960 h run:

```bash
python scripts/IKT590_train_ASR_encoder_SSSM.py \
  --data-path hub_data/librispeech \
  --dataset-config 960h \
  --d-model 384 \
  --n-layers 60 \
  --batch-size 64 \
  --seed 456 \
  --epochs 50 \
  --exp-name ssssm_960h
```

Uniform-initialization ablation:

```bash
python scripts/IKT590_train_ASR_encoder_SSSM.py \
  --data-path hub_data/librispeech \
  --dataset-config 100h \
  --no-hierarchical \
  --no-gating \
  --d-model 200 \
  --n-layers 24 \
  --batch-size 64 \
  --seed 456 \
  --epochs 30 \
  --exp-name sssm_uniform_100h
```

### Mamba-1 encoder

```bash
python scripts/IKT590_train_ASR_encoder_mamba-1.py \
  --data-path hub_data/librispeech \
  --dataset-config 100h \
  --encoder-type mamba \
  --d-model 256 \
  --n-layers 12 \
  --d-state 16 \
  --batch-size 32 \
  --seed 456 \
  --epochs 30 \
  --exp-name mamba1_100h
```

Pure PyTorch fallback, slower:

```bash
python scripts/IKT590_train_ASR_encoder_mamba-1.py \
  --data-path hub_data/librispeech \
  --dataset-config 100h \
  --encoder-type mamba \
  --no-cuda-kernels \
  --d-model 128 \
  --n-layers 6 \
  --subset-train 2000 \
  --subset-val 200 \
  --epochs 3 \
  --exp-name smoke_mamba1_pytorch
```

### Mamba-3 encoder

Mamba-3 requires a compatible CUDA/Triton setup. See `setup_guide.md`.

```bash
python scripts/IKT590_train_ASR_encoder_mamba-3.py \
  --data-path hub_data/librispeech \
  --dataset-config 100h \
  --encoder-type mamba3 \
  --d-model 256 \
  --n-layers 12 \
  --d-state 64 \
  --headdim 64 \
  --batch-size 32 \
  --seed 456 \
  --epochs 30 \
  --exp-name mamba3_100h
```

### CharMamba-1 LM

```bash
python scripts/IKT590_train_LM_CharMamba-1.py \
  --lm-hf-dataset openslr/librispeech_lm \
  --lm-max-chars 1000000000 \
  --elm-d-model 320 \
  --elm-n-layers 18 \
  --elm-d-state 16 \
  --lm-epochs 20 \
  --lm-batch-size 64 \
  --lm-lr 1e-3
```

Expected output:

```text
lm_checkpoints/elm_mamba_MaxChars-1000000000_d320_L18/
├── elm_best.pt
├── elm_hparams.json
├── elm_history.json
└── elm_training.log
```

### CharMamba-3 LM

Verify Mamba-3 availability:

```bash
python scripts/IKT590_train_LM_CharMamba-3_Triton.py --mode verify
```

Train:

```bash
python scripts/IKT590_train_LM_CharMamba-3_Triton.py \
  --mode train \
  --lm-hf-dataset openslr/librispeech_lm \
  --max-chars 1000000000 \
  --d-model 320 \
  --n-layers 18 \
  --d-state 64 \
  --headdim 64 \
  --epochs 20 \
  --batch-size 64 \
  --accum-steps 2 \
  --lr 1e-3
```

### Character n-gram LM

```bash
python scripts/IKT590_train_LM_char_ngram.py \
  --train-text lm_checkpoints/kenlm/char_level_text.txt \
  --orders 7,10 \
  --output-dir lm_checkpoints/ngram
```

Quick version:

```bash
python scripts/IKT590_train_LM_char_ngram.py \
  --train-text lm_checkpoints/kenlm/char_level_text.txt \
  --orders 7 \
  --max-lines 10000 \
  --output-dir lm_checkpoints/ngram_smoke
```

## Optional analysis tools

The evaluation script includes these analysis modes:

- `rtfx`: decoding throughput and latency.
- `hallucination`: confidence/probability quadrant analysis.
- `score_analysis`: CTC and LM score breakdown with figures.
- `verify_streaming`: verifies consistency between batch and streaming LM scoring.


## Legacy naming note

Some code, checkpoint configs, and CLI flags retain legacy names such as:

- `--no-hierarchical`
- `use_hierarchical`
- `HIER_CONFIG`
- `--elm-hierarchical`

These names are retained for compatibility with existing checkpoints and historical experiment commands. In final thesis terminology, this mechanism should be described as non-uniform layer-wise initialization, non-uniform initialization of decay/timescale parameters, or initialization diversity. It should not be presented as evidence for a proven acoustic-to-linguistic hierarchy.

## Compute requirements

CPU execution is suitable for `--help`, dataset layout checks, and very small n-gram tests. Encoder training requires a GPU for practical runtimes. Full 960 h LibriSpeech training and Mamba-3/Triton experiments require substantial compute and may require A100/H100 or otherwise compatible NVIDIA GPUs.
