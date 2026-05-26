# IKT590 Thesis Experiment Outputs

This folder contains terminal logs, merged command outputs, and evaluation records for the experiments reported in the master's thesis:

**State Space Models for Automatic Speech Transcription**  
Robert Alexander Hanssen  
University of Agder, 2026

The purpose of this folder is to provide transparency and reproducibility support for the reported experiments. The files are primarily raw or lightly curated terminal outputs from encoder training, language-model training, language-model rescoring, alpha sweeps, latency measurements, seed variance experiments, and final pipeline evaluations.

## Important note

The thesis report should be treated as the authoritative summary of the experiments and final results. These output files provide the underlying command and log record.

Some files were produced during active development and may include older commands, superseded experiment notes, failed setup attempts, or labels such as `ToDo`, `Running`, or `missing`. Where possible, final completed outputs are separated from archival workflow logs.

## Recommended folder structure

```text
results/
  README.md

  final_results/
    phase6_sssm_960h_charmamba1_eval.txt
    phase7_sssm_960h_charmamba3_eval.txt
    phase8_sssm_960h_ngram_eval.txt
    phase9_mamba_enc_960h_eval.txt
    phase10_mamba_enc_960h_lm_alpha_sweeps.txt
    phase11_seed_variance_100h.txt
    phase51_460h_encoder_trials.txt
    phase52_960h_encoder_trials.txt
    sssm_960h_lm_alpha_additional_sweeps.txt

  training_logs/
    elm_training_log.txt
    460h_encoder_training_logs.txt
    960h_encoder_training_logs.txt

  archival_raw/
    ELM_Experiments_Merged_Outputs_RAW.txt
```

The exact filenames in this repository may differ slightly from the names above. The structure is intended to distinguish final thesis-supporting outputs from raw historical logs.

## File categories

### `final_results/`

This folder should contain the most relevant completed experiment outputs used to support the final thesis results.

| File | Description |
|---|---|
| `phase6_sssm_960h_charmamba1_eval.txt` | Final 960h S-SSSM encoder evaluated with CharMamba-1 shallow-fusion rescoring. |
| `phase7_sssm_960h_charmamba3_eval.txt` | Final 960h S-SSSM encoder evaluated with CharMamba-3 shallow-fusion rescoring. |
| `phase8_sssm_960h_ngram_eval.txt` | Final 960h S-SSSM encoder evaluated with the character-level n-gram control model. |
| `phase9_mamba_enc_960h_eval.txt` | 960h Mamba encoder evaluations with CharMamba rescoring. |
| `phase10_mamba_enc_960h_lm_alpha_sweeps.txt` | Additional alpha sweeps for 960h Mamba encoder + language-model rescoring experiments. |
| `phase11_seed_variance_100h.txt` | Seed variance experiments for representative 100h SSM encoder configurations. |
| `phase51_460h_encoder_trials.txt` | 460h encoder comparison trials. |
| `phase52_960h_encoder_trials.txt` | 960h encoder comparison trials. |
| `sssm_960h_lm_alpha_additional_sweeps.txt` | Additional S-SSSM 960h language-model alpha evaluations. |

### `training_logs/`

This folder should contain longer raw training logs for the encoder and external language models.

| File | Description |
|---|---|
| `elm_training_log.txt` | Training logs for CharMamba / external language models. |
| `460h_encoder_training_logs.txt` | Training logs for 460h S-SSSM, Mamba-1, and Mamba-3 encoder experiments. |
| `960h_encoder_training_logs.txt` | Training logs for 960h S-SSSM, Mamba-1, and Mamba-3 encoder experiments. |

### `archival_raw/`

This folder should contain raw merged workflow logs that are useful for transparency but should not be read as the clean final result record.

| File | Description |
|---|---|
| `ELM_Experiments_Merged_Outputs_RAW.txt` | Historical merged workflow notes and command outputs for ELM experiments. May contain old setup attempts, superseded commands, or in-progress markers. |

## Interpreting WER values

WER stands for **Word Error Rate**. It is the standard metric used in ASR evaluation.

```text
WER = (S + D + I) / N
```

where:

- `S` = substitutions
- `D` = deletions
- `I` = insertions
- `N` = number of words in the reference transcript

Lower WER is better. A WER of 5% roughly means 5 word-level errors per 100 reference words.

## Corpus-level WER vs. per-utterance WER

The final thesis results use **corpus-level WER**, which is the standard reporting convention for ASR benchmarks.

Corpus-level WER aggregates all substitutions, deletions, and insertions over the full test set, and divides by the total number of reference words:

```text
WER_corpus = sum_i(S_i + D_i + I_i) / sum_i(N_i)
```

Some development scripts and training logs may report **per-utterance mean WER**, where WER is first computed separately for each audio file and then averaged:

```text
WER_utt_mean = (1 / M) * sum_i((S_i + D_i + I_i) / N_i)
```

These values can differ because utterances vary in length. Per-utterance mean WER gives short and long utterances equal weight, while corpus-level WER weights utterances by the number of reference words.

For benchmark comparison and final thesis tables, use the corpus-level WER values reported in the thesis.

## Dataset configuration note

Some evaluation commands include arguments such as:

```bash
--dataset-config 100h
```

In several evaluation scripts, this argument is used for loading the LibriSpeech test/evaluation setup rather than indicating the training duration of the checkpoint being evaluated.

The actual training scale of the encoder is determined by the checkpoint path and experiment name, for example:

```text
encoder_checkpoints/S-SSSM/960h/...
checkpoints_v77_960h_hier_gating_320_48
```

Therefore, an evaluation command may contain `--dataset-config 100h` while still evaluating a 460h or 960h-trained checkpoint.

## First-epoch `nan` loss entries

Some training logs may contain `nan` values for train or validation loss in the first logged epoch. In the recorded experiments, these entries did not necessarily indicate failed training. The training continued, later epochs logged normally, and model selection/evaluation was based on recorded validation and test WER values.

These `nan` entries are best interpreted as logging or aggregation artifacts unless the surrounding log shows that the run stopped or failed.

## Raw logs and development artifacts

Some raw logs may contain:

- local machine names,
- virtual environment names,
- absolute paths,
- old command attempts,
- failed setup attempts,
- `ToDo` or `Running` markers,
- intermediate experiment names,
- superseded alpha sweeps,
- old result values from earlier thesis stages.

These are kept for transparency but should not be interpreted as the final cleaned thesis result table.

## Final result interpretation

The thesis investigates a compact, reproducible, attention-free ASR pipeline based on State Space Models.

The final system uses:

- an S-SSSM-based acoustic encoder,
- CTC prefix beam search,
- a character-level Mamba language model for shallow-fusion rescoring.

The final reported LibriSpeech results in the thesis are:

| System | test-clean WER | test-other WER |
|---|---:|---:|
| S-SSSM encoder, greedy CTC | 6.17% | 16.23% |
| S-SSSM encoder, beam search | 6.07% | 16.04% |
| S-SSSM + CharMamba-1 shallow fusion | 5.25% | 14.91% |

The final unweighted mean across LibriSpeech test-clean and test-other is reported as **10.08% WER**.

These final values should be treated as the authoritative result summary. The raw logs are included to support traceability and reproducibility.

## Main experimental themes

The uploaded logs support the following thesis components:

1. **Encoder comparison**
   - S-SSSM
   - Mamba-1
   - Mamba-3

2. **Training scale comparison**
   - 100h experiments
   - 460h experiments
   - 960h experiments

3. **S-SSSM design analysis**
   - gating
   - non-uniform initialization
   - sensitivity to initialization schedule
   - depth and width studies
   - seed variance

4. **Language-model integration**
   - CharMamba-1
   - CharMamba-3
   - character-level n-gram controls
   - shallow-fusion alpha sweeps
   - score analysis
   - hallucination diagnostics
   - latency / RTFX measurements

5. **Final system evaluation**
   - final S-SSSM encoder + CharMamba rescoring
   - comparison against n-gram controls
   - comparison across encoder families

## Reproducibility limitations

These logs are provided to support reproducibility, but they are not a complete standalone reproduction package by themselves.

To reproduce the experiments fully, the following are also required:

- source code,
- exact Python environment,
- required Python packages,
- LibriSpeech audio data,
- LibriSpeech LM text data,
- model checkpoints,
- hardware/GPU information,
- local preprocessing scripts where applicable,
- configuration files or command-line arguments.

Where possible, command-line invocations are included in the logs.

## Contact

Robert Alexander Hanssen  
Master's thesis candidate, University of Agder
