"""
CharMambaLM Training Script
===========================

Trains a character-level Mamba language model (CharMamba-1) on a large text
corpus (e.g. the LibriSpeech LM corpus). The resulting checkpoint is consumed
by the ASR pipeline evaluation script (IKT590_evaluate_ASR_pipeline.py) via
shallow fusion rescoring.

This script is the training-only companion to the evaluation pipeline.
Evaluation (tune / full_eval / rtfx / hallucination / score_analysis) lives
in IKT590_evaluate_ASR_pipeline.py.

The CharMambaLM uses mamba_ssm CUDA kernels (10-50x faster than a Python
recurrence). The same model file is loaded by the evaluation script through
load_lm(), which is why model state is saved with the architectural config
embedded in the checkpoint.

Author: Robert Alexander Hanssen
Course: IKT590 - Master's Thesis
Date:   March 2026

================================================================================
SETUP
================================================================================
    pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
    pip install causal-conv1d==1.5.0.post8
    pip install mamba-ssm==2.2.4
    pip install lightning datasets

================================================================================
TYPICAL USAGE
================================================================================

# Train CharMamba-1 ELM on the LibriSpeech LM corpus (~1B characters):
python IKT590_train_LM_CharMamba1_v1_4.py \\
    --lm-hf-dataset openslr/librispeech_lm \\
    --lm-max-chars 1000000000 \\
    --elm-d-model 320 \\
    --elm-n-layers 18 \\
    --elm-d-state 16 \\
    --lm-epochs 20 \\
    --lm-batch-size 64 \\
    --lm-lr 1e-3

# Output:
#   lm_checkpoints/elm_mamba_MaxChars-1000000000_d320_L18/
#       elm_best.pt          <- checkpoint loaded by the evaluation script
#       elm_hparams.json     <- training config
#       elm_history.json     <- per-epoch train/val loss
#       elm_training.log     <- training log
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# mamba_ssm is required for CharMambaLM training.
try:
    from mamba_ssm import Mamba as MambaCUDA
    _HAS_MAMBA_SSM = True
    print(f"[OK] mamba_ssm loaded — CUDA Mamba available")
except ImportError:
    _HAS_MAMBA_SSM = False
    MambaCUDA = None
    print(f"[ERROR] mamba_ssm not found — install with: pip install mamba-ssm>=2.2.0 causal-conv1d>=1.5.0")

# HuggingFace datasets — required when loading the LM training corpus from the Hub.
try:
    from datasets import load_dataset
    _HAS_HF = True
except ImportError:
    _HAS_HF = False

warnings.filterwarnings('ignore', category=UserWarning)


# ============================================================================
# VOCABULARY & CONSTANTS (must match ASR training scripts)
# ============================================================================

VOCAB_CHARS = list(" 'abcdefghijklmnopqrstuvwxyz")
BLANK_TOKEN = len(VOCAB_CHARS)  # 27
VOCAB_SIZE = len(VOCAB_CHARS) + 1  # 28 (27 chars + blank/BOS)
CHAR_TO_IDX = {c: i for i, c in enumerate(VOCAB_CHARS)}


def text_to_ids(text: str) -> List[int]:
    return [CHAR_TO_IDX[c] for c in text.lower() if c in CHAR_TO_IDX]


# ============================================================================
# HIERARCHICAL CONFIG FOR MAMBA LM
# ============================================================================

# Used when --elm-hierarchical is set. Biases the dt_proj.bias parameter
# across layers so that early layers start with faster discretization
# timescales and later layers with slower ones. See thesis Section 4.5.1.
MAMBA_LM_HIER_CONFIG = {
    'DT_MIN_EARLY': 0.001,
    'DT_MAX_EARLY': 0.01,
    'DT_MIN_LATE': 0.005,
    'DT_MAX_LATE': 0.05,
}


# ============================================================================
# CHARACTER MAMBA LM (model under training)
# ============================================================================

class CharMambaLM(nn.Module):
    """
    Character-level autoregressive LM using Mamba layers.

    During training only the forward() path is used (full CUDA selective
    scan). The streaming .step() interface used by the evaluation script
    lives in the evaluation file. The checkpoint saved here is fully
    compatible with the evaluation-time CharMambaLM class.
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = 256,
        n_layers: int = 8,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        use_hierarchical: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers_count = n_layers
        self.d_state = d_state

        # Character embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.embed_dropout = nn.Dropout(dropout)

        # Mamba layers
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        for _ in range(n_layers):
            if not _HAS_MAMBA_SSM:
                raise RuntimeError(
                    "mamba_ssm required for CharMambaLM. "
                    "Install: pip install mamba-ssm>=2.2.0 causal-conv1d>=1.5.0"
                )
            mamba = MambaCUDA(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            self.layers.append(mamba)
            self.norms.append(nn.LayerNorm(d_model))
            self.dropouts.append(nn.Dropout(dropout))

        # Apply hierarchical dt_bias initialization
        if use_hierarchical:
            self._apply_hierarchical_init()

        self.final_norm = nn.LayerNorm(d_model)
        self.output_head = nn.Linear(d_model, vocab_size)

        total_params = sum(p.numel() for p in self.parameters())
        print(f"\n[CharMambaLM] {total_params:,} params "
              f"(d={d_model}, L={n_layers}, N={d_state}, "
              f"expand={expand}, hier={use_hierarchical})")

    def _apply_hierarchical_init(self):
        """Apply hierarchical dt_bias initialization across layers."""
        cfg = MAMBA_LM_HIER_CONFIG
        n = self.n_layers_count

        print(f"\n  [CharMambaLM] Hierarchical dt_bias initialization:")
        for i, mamba in enumerate(self.layers):
            progress = i / max(1, n - 1)

            dt_min = cfg['DT_MIN_EARLY'] + (cfg['DT_MIN_LATE'] - cfg['DT_MIN_EARLY']) * progress
            dt_max = cfg['DT_MAX_EARLY'] + (cfg['DT_MAX_LATE'] - cfg['DT_MAX_EARLY']) * progress

            with torch.no_grad():
                d_inner = mamba.dt_proj.bias.shape[0]
                dt = torch.exp(
                    torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min))
                    + math.log(dt_min)
                ).clamp(min=1e-4)
                inv_dt = dt + torch.log(-torch.expm1(-dt))
                mamba.dt_proj.bias.copy_(inv_dt)

            layer_type = "fast" if progress < 0.33 else ("mid" if progress < 0.66 else "slow")
            print(f"    Layer {i:2d} ({layer_type:4s}): dt_mean={dt.mean():.4f}")

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Full sequence forward (training). Uses CUDA selective scan.
        Args:  input_ids: [B, T]
        Returns: logits: [B, T, V]
        """
        x = self.embed_dropout(self.embedding(input_ids))  # [B, T, D]

        for mamba, norm, drop in zip(self.layers, self.norms, self.dropouts):
            x_norm = norm(x)
            y = mamba(x_norm)
            x = x + drop(y)

        x = self.final_norm(x)
        return self.output_head(x)


# ============================================================================
# DATASETS
# ============================================================================

class ChunkedLMDataset(Dataset):
    """LM dataset from a pre-split character ID list. No overlap between train/val."""

    def __init__(self, char_ids: list, max_seq_len: int = 256, stride: int = None):
        self.max_seq_len = max_seq_len
        self.all_ids = char_ids
        self.stride = stride if stride is not None else max_seq_len
        self.n_examples = max(1, (len(self.all_ids) - max_seq_len) // self.stride)

    def __len__(self):
        return self.n_examples

    def __getitem__(self, idx):
        start = idx * self.stride
        chunk = self.all_ids[start:start + self.max_seq_len + 1]
        if len(chunk) < self.max_seq_len + 1:
            chunk += [BLANK_TOKEN] * (self.max_seq_len + 1 - len(chunk))
        inp = [BLANK_TOKEN] + chunk[:-1]
        tgt = chunk[:self.max_seq_len]
        return torch.LongTensor(inp[:self.max_seq_len]), torch.LongTensor(tgt)


def build_elm_datasets(hf_dataset, max_seq_len=512, max_chars=100_000_000, val_fraction=0.05):
    """
    Build train/val datasets for ELM from HuggingFace dataset.
    Splits raw text first, then chunks independently (no leakage).
    """
    all_ids = []
    print(f"  [ELM Data] Loading from HuggingFace...")
    for i, item in enumerate(hf_dataset):
        text = item['text'].strip().lower()
        if text:
            ids = text_to_ids(text)
            if ids:
                all_ids.extend(ids)
                all_ids.append(CHAR_TO_IDX[' '])
        if len(all_ids) >= max_chars:
            print(f"    Reached {max_chars:,} char limit at row {i:,}")
            break
        if (i + 1) % 2_000_000 == 0:
            print(f"    {i+1:,} rows, {len(all_ids):,} chars...")

    split_point = int(len(all_ids) * (1.0 - val_fraction))
    train_ids = all_ids[:split_point]
    val_ids = all_ids[split_point:]

    train_ds = ChunkedLMDataset(train_ids, max_seq_len, max_seq_len)  # non-overlapping for ELM
    val_ds = ChunkedLMDataset(val_ids, max_seq_len, max_seq_len)

    print(f"  [ELM Data] {len(all_ids):,} chars total")
    print(f"    Train: {len(train_ids):,} chars -> {len(train_ds):,} chunks (stride={max_seq_len})")
    print(f"    Val:   {len(val_ids):,} chars -> {len(val_ds):,} chunks (stride={max_seq_len})")
    return train_ds, val_ds


def build_elm_datasets_from_file(path, max_seq_len=512, max_chars=100_000_000, val_fraction=0.05):
    """Build train/val datasets for ELM from a text file."""
    all_ids = []
    print(f"  [ELM Data] Loading from {path}...")
    with open(path, 'r') as f:
        for line in f:
            t = line.strip().lower()
            if t:
                ids = text_to_ids(t)
                if ids:
                    all_ids.extend(ids)
                    all_ids.append(CHAR_TO_IDX[' '])
            if len(all_ids) >= max_chars:
                break

    split_point = int(len(all_ids) * (1.0 - val_fraction))
    train_ids = all_ids[:split_point]
    val_ids = all_ids[split_point:]

    train_ds = ChunkedLMDataset(train_ids, max_seq_len, max_seq_len)
    val_ds = ChunkedLMDataset(val_ids, max_seq_len, max_seq_len)

    print(f"  [ELM Data] {len(all_ids):,} chars total")
    print(f"    Train: {len(train_ids):,} chars -> {len(train_ds):,} chunks")
    print(f"    Val:   {len(val_ids):,} chars -> {len(val_ds):,} chunks")
    return train_ds, val_ds


# ============================================================================
# LM TRAINING
# ============================================================================

def train_lm(model, train_ds, val_ds, save_dir, epochs=20, batch_size=64, lr=1e-3,
             label="ELM", patience=5, weight_decay=0.01, label_smoothing=0.0):
    """
    LM training loop with:
    - Pre-split train/val (no leakage)
    - Hyperparameter logging
    - Log file output
    - Label smoothing
    - Early stopping on val loss
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{label.lower()}_best.pt")
    log_path = os.path.join(save_dir, f"{label.lower()}_training.log")

    # --- Log hyperparameters ---
    hparams = {
        'label': label,
        'model_type': 'mamba',
        'd_model': model.d_model,
        'n_layers': model.n_layers_count,
        'vocab_size': model.vocab_size,
        'd_state': model.d_state,
        'total_params': sum(p.numel() for p in model.parameters()),
        'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'train_chunks': len(train_ds),
        'val_chunks': len(val_ds),
        'epochs': epochs,
        'batch_size': batch_size,
        'lr': lr,
        'weight_decay': weight_decay,
        'label_smoothing': label_smoothing,
        'patience': patience,
        'device': str(device),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'save_dir': save_dir,
    }

    # Save hyperparameters to JSON
    with open(os.path.join(save_dir, f"{label.lower()}_hparams.json"), 'w') as f:
        json.dump(hparams, f, indent=2, default=str)

    # Open log file — log() prints to console AND writes to file
    logf = open(log_path, 'w')

    def log(msg):
        print(msg)
        logf.write(msg + '\n')
        logf.flush()

    # Print config once (via log, which handles both console + file)
    log(f"\n{'='*70}")
    log(f"  {label} Training Configuration")
    log(f"{'='*70}")
    for k, v in hparams.items():
        log(f"  {k:20s}: {v:,}" if k == 'total_params' else f"  {k:20s}: {v}")
    log(f"{'='*70}\n")

    # --- Dataloaders ---
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = len(train_loader) * epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    criterion = nn.CrossEntropyLoss(ignore_index=BLANK_TOKEN, label_smoothing=label_smoothing)

    if label_smoothing > 0:
        # Theoretical minimum loss with label smoothing (for reference)
        min_loss = -math.log(1.0 - label_smoothing + label_smoothing / model.vocab_size)
        log(f"  [{label}] Label smoothing={label_smoothing}, theoretical loss floor={min_loss:.4f} (PPL {math.exp(min_loss):.2f})")

    best_val_loss = float('inf')
    no_improve = 0
    history = []

    for epoch in range(epochs):
        # --- Train ---
        model.train()
        total_loss, n_batches = 0.0, 0
        t0 = time.time()

        for input_ids, target_ids in train_loader:
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            logits = model(input_ids)
            loss = criterion(logits.reshape(-1, model.vocab_size), target_ids.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            n_batches += 1

        train_avg = total_loss / max(n_batches, 1)
        train_ppl = math.exp(min(train_avg, 20))

        # --- Validate ---
        model.eval()
        val_loss, val_n = 0.0, 0
        with torch.no_grad():
            for input_ids, target_ids in val_loader:
                input_ids = input_ids.to(device)
                target_ids = target_ids.to(device)
                logits = model(input_ids)
                loss = criterion(logits.reshape(-1, model.vocab_size), target_ids.reshape(-1))
                val_loss += loss.item()
                val_n += 1
        val_avg = val_loss / max(val_n, 1)
        val_ppl = math.exp(min(val_avg, 20))

        elapsed = time.time() - t0
        gap = val_avg - train_avg
        star = " *" if val_avg < best_val_loss else ""
        msg = (f"  [{label}] Epoch {epoch+1:2d}/{epochs} — "
               f"Train: {train_avg:.4f} (PPL {train_ppl:.2f}), "
               f"Val: {val_avg:.4f} (PPL {val_ppl:.2f}), "
               f"Gap: {gap:+.4f}, "
               f"Time: {elapsed:.0f}s{star}")
        log(msg)

        history.append({'epoch': epoch+1, 'train_loss': train_avg, 'val_loss': val_avg,
                        'train_ppl': train_ppl, 'val_ppl': val_ppl})

        # Save on best val loss
        if val_avg < best_val_loss:
            best_val_loss = val_avg
            no_improve = 0
            config = {
                'model_type': 'mamba',
                'vocab_size': model.vocab_size,
                'd_model': model.d_model,
                'n_layers': model.n_layers_count,
                'd_state': model.d_state,
                'label': label,
            }
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': config,
                'best_loss': best_val_loss,
                'hparams': hparams,
            }, save_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                log(f"  [{label}] Early stopping at epoch {epoch+1} (no val improvement for {patience} epochs)")
                break

    # Save training history
    with open(os.path.join(save_dir, f"{label.lower()}_history.json"), 'w') as f:
        json.dump(history, f, indent=2)

    val_ppl = math.exp(min(best_val_loss, 20))
    log(f"\n  [{label}] Done. Best val loss={best_val_loss:.4f}, val PPL={val_ppl:.2f}")
    log(f"  [{label}] Checkpoint: {save_path}")
    log(f"  [{label}] Log: {log_path}")
    logf.close()
    return model


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train CharMamba-1 character-level LM for shallow-fusion rescoring"
    )

    # Data sources (exactly one required)
    parser.add_argument('--lm-hf-dataset', type=str, default=None,
                        help='HuggingFace dataset name (e.g. openslr/librispeech_lm)')
    parser.add_argument('--lm-text-path', type=str, default=None,
                        help='Path to plain-text file with one document per line')

    # Output
    parser.add_argument('--lm-save-path', type=str, default=None,
                        help='Experiment directory (auto-named if not set)')

    # Model architecture
    parser.add_argument('--elm-d-model', type=int, default=320,
                        help='Model dimension (default 320)')
    parser.add_argument('--elm-n-layers', type=int, default=18,
                        help='Number of Mamba layers (default 18)')
    parser.add_argument('--elm-d-state', type=int, default=16,
                        help='Mamba state dimension (default 16)')
    parser.add_argument('--elm-d-conv', type=int, default=4,
                        help='Mamba causal conv kernel size (default 4)')
    parser.add_argument('--elm-expand', type=int, default=2,
                        help='Mamba inner expansion factor (default 2)')
    parser.add_argument('--elm-hierarchical', action='store_true',
                        help='Apply non-uniform dt_bias initialization across layers')

    # Training
    parser.add_argument('--lm-epochs', type=int, default=20)
    parser.add_argument('--lm-batch-size', type=int, default=64)
    parser.add_argument('--lm-lr', type=float, default=1e-3)
    parser.add_argument('--lm-max-chars', type=int, default=1_000_000_000,
                        help='Max characters loaded from the corpus (default 1B)')
    parser.add_argument('--lm-dropout', type=float, default=0.3,
                        help='Dropout for LM training (default 0.3 to prevent memorization)')
    parser.add_argument('--lm-label-smoothing', type=float, default=0.1,
                        help='Label smoothing (default 0.1)')
    parser.add_argument('--lm-max-seq-len', type=int, default=512,
                        help='Training chunk length in characters (default 512)')
    parser.add_argument('--lm-weight-decay', type=float, default=0.01,
                        help='AdamW weight decay (default 0.01)')
    parser.add_argument('--lm-patience', type=int, default=5,
                        help='Early stopping patience in epochs (default 5)')

    # General
    parser.add_argument('--seed', type=int, default=456)

    args = parser.parse_args()

    if not _HAS_MAMBA_SSM:
        print("ERROR: mamba_ssm required. pip install mamba-ssm causal-conv1d")
        sys.exit(1)

    if not args.lm_hf_dataset and not args.lm_text_path:
        print("ERROR: --lm-hf-dataset or --lm-text-path required for ELM training")
        sys.exit(1)

    if args.lm_hf_dataset and not _HAS_HF:
        print("ERROR: HuggingFace datasets required for --lm-hf-dataset. pip install datasets")
        sys.exit(1)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Build output directory name
    hier_tag = "_hier" if args.elm_hierarchical else ""
    exp_tag = (f"elm_mamba_MaxChars-{args.lm_max_chars}_"
               f"d{args.elm_d_model}_L{args.elm_n_layers}{hier_tag}")
    save_dir = args.lm_save_path or os.path.join("lm_checkpoints", exp_tag)

    # Build datasets
    if args.lm_hf_dataset:
        print(f"\nLoading HF dataset: {args.lm_hf_dataset}")
        hf_ds = load_dataset(args.lm_hf_dataset, split='train', trust_remote_code=True)
        train_ds, val_ds = build_elm_datasets(
            hf_ds,
            max_seq_len=args.lm_max_seq_len,
            max_chars=args.lm_max_chars,
        )
    else:
        train_ds, val_ds = build_elm_datasets_from_file(
            args.lm_text_path,
            max_seq_len=args.lm_max_seq_len,
            max_chars=args.lm_max_chars,
        )

    # Build model
    model = CharMambaLM(
        vocab_size=VOCAB_SIZE,
        d_model=args.elm_d_model,
        n_layers=args.elm_n_layers,
        d_state=args.elm_d_state,
        d_conv=args.elm_d_conv,
        expand=args.elm_expand,
        dropout=args.lm_dropout,
        use_hierarchical=args.elm_hierarchical,
    )

    # Train
    train_lm(
        model, train_ds, val_ds, save_dir,
        epochs=args.lm_epochs,
        batch_size=args.lm_batch_size,
        lr=args.lm_lr,
        label="ELM",
        patience=args.lm_patience,
        weight_decay=args.lm_weight_decay,
        label_smoothing=args.lm_label_smoothing,
    )


if __name__ == '__main__':
    main()