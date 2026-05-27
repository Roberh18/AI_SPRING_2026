# Script Overview

This document summarizes the main repository scripts and their role in the thesis pipeline.

Commands assume the scripts are placed in `scripts/`. If your local filenames differ, replace the script path with the actual filename.

## Essential scripts

### `scripts/download_dataset.py`

Purpose: Downloads the LibriSpeech ASR `clean` and `other` configurations from HuggingFace and saves them to disk.

Pipeline role: Dataset preparation.

Inputs:

- Internet access.
- HuggingFace `datasets`.

Outputs:

```text
hub_data/librispeech/clean/
hub_data/librispeech/other/
```

Main dependencies:

- `datasets`
- `os`

Minimal command:

```bash
python scripts/download_dataset.py
```

Essential: Yes.

GPU/checkpoint requirements: None.

Notes: The script downloads `openslr/librispeech_asr` with both `clean` and `other` configurations, saves them with `save_to_disk`, reloads them, and prints the resulting folder tree.

---

### `scripts/IKT590_train_ASR_encoder_SSSM.py`

Purpose: Trains the S-SSSM CTC acoustic encoder on LibriSpeech.

Pipeline role: Acoustic encoder training.

Inputs:

- LibriSpeech saved at `hub_data/librispeech`.
- Optional subset limits through `--subset-train` and `--subset-val`.

Outputs:

- PyTorch Lightning checkpoints.
- Config JSON.
- Training logs.
- Diagnostics JSON.
- Figures.

The script writes to a directory named from `--exp-name`, for example:

```text
checkpoints_smoke_ssssm/
```

Main dependencies:

- `torch`
- `torchaudio`
- `lightning`
- `datasets`
- `jiwer`
- `numpy`
- `pandas`
- `matplotlib`

Minimal help command:

```bash
python scripts/IKT590_train_ASR_encoder_SSSM.py --help
```

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

Full run:

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

Essential: Yes, if retraining the S-SSSM encoder. For examiner evaluation with provided checkpoints, training is not required.

GPU/checkpoint requirements: GPU strongly recommended. No checkpoint required for training from scratch.

Important CLI arguments:

- `--data-path`
- `--dataset-config`
- `--subset-train`
- `--subset-val`
- `--batch-size`
- `--epochs`
- `--d-model`
- `--n-layers`
- `--lr`
- `--no-hierarchical`
- `--no-gating`
- `--no-specaugment`
- `--speed-perturb`
- `--speed-factors`
- `--fp32`
- `--seed`
- `--freeze-epochs`
- `--hier-params`

Terminology note: `--no-hierarchical` and `--hier-params` are legacy names. The thesis terminology is non-uniform layer-wise initialization or initialization diversity.

---

### `scripts/IKT590_train_ASR_encoder_mamba-1.py`

Purpose: Trains a Mamba-1 CTC acoustic encoder. The script can also select an S-SSSM backend through `--encoder-type sssm`.

Pipeline role: Acoustic encoder training and Mamba-1 comparison.

Inputs:

- LibriSpeech saved at `hub_data/librispeech`.

Outputs:

- PyTorch Lightning checkpoints.
- Logs.
- Training diagnostics.
- Figures.

Main dependencies:

- `torch`
- `torchaudio`
- `lightning`
- `datasets`
- `jiwer`
- `numpy`
- `pandas`
- `matplotlib`
- Optional: `mamba-ssm`, `causal-conv1d`

Minimal help command:

```bash
python scripts/IKT590_train_ASR_encoder_mamba-1.py --help
```

Smoke test:

```bash
python scripts/IKT590_train_ASR_encoder_mamba-1.py \
  --data-path hub_data/librispeech \
  --subset-train 2000 \
  --subset-val 200 \
  --epochs 3 \
  --d-model 128 \
  --n-layers 6 \
  --encoder-type mamba \
  --exp-name smoke_mamba1
```

Training command:

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

Essential: Essential for reproducing Mamba-1 encoder experiments. Not required for the default S-SSSM final evaluation if a checkpoint is provided.

GPU/checkpoint requirements: GPU recommended. CUDA Mamba kernels are much faster than the pure PyTorch fallback. Use `--no-cuda-kernels` only for small checks or debugging.

Important CLI arguments:

- `--encoder-type mamba`
- `--d-state`
- `--d-conv`
- `--expand`
- `--no-cuda-kernels`
- common ASR training arguments.

Terminology note: `--no-hierarchical` disables non-uniform layer-wise initialization.

---

### `scripts/IKT590_train_ASR_encoder_mamba-3.py`

Purpose: Trains a CTC acoustic encoder with selectable S-SSSM, Mamba-1, or Mamba-3 backend.

Pipeline role: Acoustic encoder training and Mamba-3 comparison.

Inputs:

- LibriSpeech saved at `hub_data/librispeech`.

Outputs:

- PyTorch Lightning checkpoints.
- Logs.
- Training diagnostics.
- Figures.

Main dependencies:

- Standard ASR dependencies.
- `mamba-ssm` for Mamba-1.
- GitHub/Triton Mamba-3 installation for `Mamba3`.

Minimal help command:

```bash
python scripts/IKT590_train_ASR_encoder_mamba-3.py --help
```

Mamba-3 training command:

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

Essential: Essential only for reproducing Mamba-3 encoder experiments. Not required for default final evaluation.

GPU/checkpoint requirements: Mamba-3 may require A100/H100 or compatible modern NVIDIA GPU and a special `mamba-ssm` GitHub/Triton setup.

Important CLI arguments:

- `--encoder-type mamba3`
- `--d-state`
- `--headdim`
- `--is-mimo`
- `--mimo-rank`
- `--no-cuda-kernels`
- common ASR training arguments.

Terminology note: legacy `hierarchical` names refer to non-uniform initialization, not a proven acoustic-to-linguistic hierarchy.

---

### `scripts/IKT590_train_LM_CharMamba-1.py`

Purpose: Trains the CharMamba-1 character-level language model used for shallow-fusion rescoring.

Pipeline role: External language-model training.

Inputs:

- HuggingFace LM dataset, usually `openslr/librispeech_lm`, or a local text file via `--lm-text-path`.

Outputs:

```text
lm_checkpoints/elm_mamba_MaxChars-1000000000_d320_L18/
├── elm_best.pt
├── elm_hparams.json
├── elm_history.json
└── elm_training.log
```

Main dependencies:

- `torch`
- `datasets`
- `mamba-ssm`
- `causal-conv1d`
- `numpy`

Minimal help command:

```bash
python scripts/IKT590_train_LM_CharMamba-1.py --help
```

Training command:

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

Essential: Required only if retraining CharMamba-1. For final evaluation, a provided LM checkpoint directory is sufficient.

GPU/checkpoint requirements: GPU strongly recommended. Requires `mamba-ssm`.

Important CLI arguments:

- `--lm-hf-dataset`
- `--lm-text-path`
- `--lm-save-path`
- `--elm-d-model`
- `--elm-n-layers`
- `--elm-d-state`
- `--elm-d-conv`
- `--elm-expand`
- `--elm-hierarchical`
- `--lm-epochs`
- `--lm-batch-size`
- `--lm-lr`
- `--lm-max-chars`
- `--lm-dropout`
- `--lm-label-smoothing`
- `--lm-max-seq-len`
- `--lm-weight-decay`
- `--lm-patience`
- `--seed`

Terminology note: `--elm-hierarchical` applies non-uniform `dt_bias` initialization across layers.

---

### `scripts/IKT590_train_LM_CharMamba-3_Triton.py`

Purpose: Trains a character-level Mamba-3 language model for ASR rescoring.

Pipeline role: External language-model training and Mamba-3 LM comparison.

Inputs:

- HuggingFace LM dataset, usually `openslr/librispeech_lm`, or a local text file.

Outputs:

- CharMamba-3 LM checkpoint directory under `lm_checkpoints/`, unless `--save-dir` is specified.

Main dependencies:

- `torch`
- `datasets`
- GitHub/Triton-enabled `mamba-ssm` with `Mamba3`

Minimal verification command:

```bash
python scripts/IKT590_train_LM_CharMamba-3_Triton.py --mode verify
```

Training command:

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

Essential: Optional for final evaluation. Required only for Mamba-3 LM comparison.

GPU/checkpoint requirements: Requires modern NVIDIA GPU and special Mamba-3/Triton setup. The script is optimized for A100/H100 and may fail on V100 or older GPUs.

Important CLI arguments:

- `--mode verify|train`
- `--lm-hf-dataset`
- `--lm-text-path`
- `--max-chars`
- `--d-model`
- `--n-layers`
- `--d-state`
- `--headdim`
- `--mimo`
- `--mimo-rank`
- `--epochs`
- `--batch-size`
- `--accum-steps`
- `--lr`
- `--dropout`
- `--label-smoothing`
- `--save-dir`

---

### `scripts/IKT590_train_LM_char_ngram.py`

Purpose: Trains pure Python character-level n-gram language models with Stupid Backoff smoothing.

Pipeline role: Fixed-context LM baseline/control.

Inputs:

- Character-level text file, usually:

```text
lm_checkpoints/kenlm/char_level_text.txt
```

Outputs:

```text
lm_checkpoints/ngram/char_7gram.pkl
lm_checkpoints/ngram/char_10gram.pkl
```

Main dependencies:

- Python standard library only.

Minimal help command:

```bash
python scripts/IKT590_train_LM_char_ngram.py --help
```

Training command:

```bash
python scripts/IKT590_train_LM_char_ngram.py \
  --train-text lm_checkpoints/kenlm/char_level_text.txt \
  --orders 7,10 \
  --output-dir lm_checkpoints/ngram
```

Smoke test:

```bash
python scripts/IKT590_train_LM_char_ngram.py \
  --train-text lm_checkpoints/kenlm/char_level_text.txt \
  --orders 7 \
  --max-lines 10000 \
  --output-dir lm_checkpoints/ngram_smoke
```

Essential: Essential for n-gram control comparisons. Not required for CharMamba-only final evaluation.

GPU/checkpoint requirements: None.

Important CLI arguments:

- `--train-text`
- `--orders`
- `--output-dir`
- `--max-lines`

---

### `scripts/IKT590_evaluate_ASR_pipeline.py`

Purpose: Runs final ASR evaluation. It does not train models. It loads pretrained encoder and LM checkpoints and runs decoding, WER evaluation, tuning, RTFX measurement, hallucination analysis, score analysis, and streaming verification.

Pipeline role: Final evaluation and analysis.

Inputs:

- LibriSpeech saved at `hub_data/librispeech`.
- Encoder `.ckpt` file.
- CharMamba LM directory, CharMamba-3 LM directory, n-gram `.pkl`, or other supported LM path.

Outputs:

```text
ilme_mamba_results_<exp-name>/
├── full_eval_test-clean.json
├── full_eval_test-other.json
├── best_params.json
├── grid_results.json
├── rtfx_greedy.json
├── rtfx_beam.json
├── rtfx_nbest_ilme.json
├── hallucination.json
├── score_analysis_alpha<A>_beam<W>.txt
├── score_analysis_summary.json
└── figures/
```

Main dependencies:

- `torch`
- `torchaudio`
- `datasets`
- `jiwer`
- `numpy`
- `matplotlib`
- Optional: `mamba-ssm`
- Optional: `kenlm`

Minimal help command:

```bash
python scripts/IKT590_evaluate_ASR_pipeline.py --help
```

Final evaluation command:

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

Essential: Yes. This is the main examiner evaluation script.

GPU/checkpoint requirements: Requires encoder and LM checkpoints for meaningful evaluation. GPU recommended for practical runtime.

Supported modes:

- `tune`
- `full_eval`
- `rtfx`
- `hallucination`
- `score_analysis`
- `verify_streaming`

Important CLI arguments:

- `--mode`
- `--data-path`
- `--dataset-config`
- `--asr-checkpoint`
- `--asr-type`
- `--elm-path`
- `--ilm-path`
- `--beam-width`
- `--alpha`
- `--beta`
- `--gamma-ilme`
- `--batch-size`
- `--subset-val`
- `--exp-name`
- `--seed`

## Optional diagnostic scripts

The uploaded file set did not include standalone diagnostic scripts. If the repository includes them, document them here.

### `scripts/IKT590_lm_diagnostic.py`

Purpose: Optional LM diagnostic for coherent vs scrambled sentences, real vs nonsense text, minimal pairs, long-range agreement, next-character prediction, and sentence completion.

Pipeline role: Optional qualitative/diagnostic LM analysis.

Inputs:

- LM checkpoint directory or n-gram `.pkl`.
- Diagnostic sentence sets.

Outputs:

- Diagnostic scores, logs, and optional figures.

Minimal command:

```bash
python scripts/IKT590_lm_diagnostic.py --help
```

Essential: No.

GPU/checkpoint requirements: Requires an LM checkpoint. GPU may be required for CharMamba models.

### `scripts/IKT590_lm_diagnostic_v2.py`

Purpose: Optional diagnostic intended to better differentiate Mamba LMs from n-gram LMs through long-range context, ASR-relevant minimal pairs, perplexity, and context-length tests.

Pipeline role: Optional LM comparison.

Inputs:

- LM checkpoint directory or n-gram `.pkl`.
- Diagnostic text examples.

Outputs:

- Diagnostic scores, logs, and optional figures.

Minimal command:

```bash
python scripts/IKT590_lm_diagnostic_v2.py --help
```

Essential: No.

GPU/checkpoint requirements: Requires an LM checkpoint. GPU may be required for CharMamba models.
