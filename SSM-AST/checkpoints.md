# Checkpoints and Model Artifacts

This repository does not require large model files to be committed directly to Git. Place checkpoints and language-model artifacts in the directories below before running final evaluation.

Large files should be distributed through GitHub release assets, institutional storage, or external storage if they exceed practical GitHub limits.

## Expected layout

```text
encoder_checkpoints/
├── S-SSSM/
│   ├── 100h/
│   │   └── <checkpoint>.ckpt
│   └── 960h/
│       └── <checkpoint>.ckpt
├── Mamba-1/
│   └── 100h/
│       └── <checkpoint>.ckpt
└── Mamba-3/
    └── 100h/
        └── <checkpoint>.ckpt

lm_checkpoints/
├── elm_mamba_MaxChars-1000000000_d320_L18/
│   ├── elm_best.pt
│   ├── elm_hparams.json
│   ├── elm_history.json
│   └── elm_training.log
├── elm_mamba3_MaxChars-1000000000_d320_L18/
│   └── <mamba3-lm-checkpoint-files>
├── ngram/
│   ├── char_7gram.pkl
│   └── char_10gram.pkl
└── kenlm/
    └── char_level_text.txt
```

The exact checkpoint filenames may differ depending on the saved Lightning checkpoint name. Use the actual filename with `--asr-checkpoint`.

## Final thesis evaluation dependencies

The final evaluation path requires:

```text
hub_data/librispeech/clean/
hub_data/librispeech/other/
encoder_checkpoints/S-SSSM/960h/<checkpoint>.ckpt
lm_checkpoints/elm_mamba_MaxChars-1000000000_d320_L18/elm_best.pt
```

Example evaluation command:

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

## Encoder checkpoints

Encoder checkpoints are PyTorch Lightning `.ckpt` files produced by the ASR encoder training scripts.

Recommended placement:

```text
encoder_checkpoints/S-SSSM/960h/<checkpoint>.ckpt
encoder_checkpoints/Mamba-1/100h/<checkpoint>.ckpt
encoder_checkpoints/Mamba-3/100h/<checkpoint>.ckpt
```

Example placeholder:

```text
encoder_checkpoints/S-SSSM/960h/best_epoch=98_val_wer=0.111.ckpt
```

Use the appropriate `--asr-type`:

```bash
--asr-type sssm
--asr-type mamba
--asr-type mamba3
```

## CharMamba-1 LM checkpoints

CharMamba-1 training writes an experiment directory. The evaluation script should receive the directory path, not necessarily the `.pt` file directly.

Expected example:

```text
lm_checkpoints/elm_mamba_MaxChars-1000000000_d320_L18/
├── elm_best.pt
├── elm_hparams.json
├── elm_history.json
└── elm_training.log
```

Evaluation usage:

```bash
--elm-path lm_checkpoints/elm_mamba_MaxChars-1000000000_d320_L18
```

## CharMamba-3 LM checkpoints

CharMamba-3 checkpoints are optional for final S-SSSM + CharMamba-1 evaluation, but may be used for additional comparison.

Recommended example:

```text
lm_checkpoints/elm_mamba3_MaxChars-1000000000_d320_L18/
└── <checkpoint-files>
```

Evaluation usage:

```bash
--elm-path lm_checkpoints/elm_mamba3_MaxChars-1000000000_d320_L18
```

Mamba-3 checkpoint loading may require the same Mamba-3/Triton environment used for training.

## Character n-gram LM checkpoints

The n-gram script writes `.pkl` files. These files are passed directly to the evaluation script.

Recommended placement:

```text
lm_checkpoints/ngram/char_7gram.pkl
lm_checkpoints/ngram/char_10gram.pkl
```

Example training command:

```bash
python scripts/IKT590_train_LM_char_ngram.py \
  --train-text lm_checkpoints/kenlm/char_level_text.txt \
  --orders 7,10 \
  --output-dir lm_checkpoints/ngram
```

Example evaluation command:

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

## Optional checkpoints

Optional artifacts include:

- Mamba-1 encoder checkpoints.
- Mamba-3 encoder checkpoints.
- CharMamba-3 LM checkpoints.
- Character n-gram `.pkl` files.
- Diagnostic outputs and figures.
- Intermediate training logs.

These are not required for the default final evaluation path unless the examiner wants to reproduce the corresponding ablation or comparison.

## Alternative checkpoint paths

All checkpoint paths can be overridden at runtime:

```bash
python scripts/IKT590_evaluate_ASR_pipeline.py \
  --mode full_eval \
  --data-path hub_data/librispeech \
  --asr-checkpoint /absolute/or/relative/path/to/model.ckpt \
  --asr-type sssm \
  --elm-path /absolute/or/relative/path/to/lm_directory_or_ngram.pkl \
  --alpha <ALPHA> \
  --beta <BETA> \
  --beam-width 10 \
  --batch-size 64 \
  --exp-name custom_checkpoint_eval
```

Optional internal LM path:

```bash
--ilm-path <path-to-internal-lm>
```

## Legacy naming in checkpoints

Some checkpoint configs may contain legacy keys such as:

```text
hierarchical
use_hierarchical
HIER_CONFIG
elm_hierarchical
```

These names are retained for compatibility with saved models and historical experiment commands. In thesis text and documentation, the mechanism should be described as non-uniform layer-wise initialization, non-uniform initialization of decay/timescale parameters, or initialization diversity.
