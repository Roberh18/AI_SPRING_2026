"""
CharMamba3LM: Character-level Mamba-3 Language Model for ASR Rescoring (v1.2-A100)

Trains a character-level LM using the Mamba-3 SSM architecture for direct
comparison against the existing Mamba-1 CharMambaLM (d=320, L=18, 12.2M params).

v1.2-A100: Uses official mamba_ssm Triton kernels (requires sm_75+ GPU).
  Optimized for A100/H100. NOT compatible with V100 (sm_70).

Key Mamba-3 improvements over Mamba-1:
  - Complex-valued SSM with RoPE embeddings
  - Exponential-trapezoidal discretization (2nd order vs 1st order Euler)
  - Multi-input multi-output (MIMO) SSMs
  - No causal convolution (removed -- now implicit in recurrence)
  - BCNorm (QKNorm) for training stability

IMPORTANT:
  - Requires mamba-ssm >= 2.3.1 installed from GitHub:
      TORCH_CUDA_ARCH_LIST="8.0" pip install --force-reinstall --no-deps \
        git+https://github.com/state-spaces/mamba.git --no-build-isolation --break-system-packages
  - Requires A100 (sm_80) or newer GPU. V100 will fail with ptxas tanh errors.

Usage:
  # Step 0: Install mamba-ssm from GitHub (NOT PyPI)
  TORCH_CUDA_ARCH_LIST="8.0" pip install --force-reinstall --no-deps \
    git+https://github.com/state-spaces/mamba.git --no-build-isolation --break-system-packages

  # Step 1: Verify Mamba-3 availability and GPU compatibility
  python IKT590_train_LM_CharMamba-3_Triton.py --mode verify

  # Step 2: Train (matched to Mamba-1 comparison: same data, similar param count)
  python IKT590_train_LM_CharMamba-3_Triton.py --mode train \
    --lm-hf-dataset openslr/librispeech_lm \
    --d-model 320 --n-layers 18 --d-state 64 --headdim 64 \
    --epochs 20 --batch-size 64 --accum-steps 2 --max-chars 1000000000

  # Step 3: Evaluate with existing ASR pipeline (see the ASR evaluation script)
  #   --elm-path lm_checkpoints/elm_mamba3_MaxChars-1000000000_d320_L18/

Outputs (saved to lm_checkpoints/elm_mamba3_MaxChars-{N}_d{D}_L{L}/):
  elm_best.pt        - Best checkpoint (model_state_dict + config + hparams)
  elm_hparams.json   - Full hyperparameter record
  elm_history.json   - Per-epoch train/val loss and PPL
  elm_training.log   - Training log
  plot_loss.pdf      - Training curves (requires matplotlib)
  plot_ppl.pdf       - Perplexity curves
  plot_gap.pdf       - Generalization gap
"""

import argparse
import json
import math
import os
import sys
import time
import warnings
from datetime import datetime
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================================
# MAMBA-3 AVAILABILITY CHECK (using official mamba_ssm Triton kernels)
# ============================================================================

_HAS_MAMBA3 = False
_MAMBA3_ERROR = None
_mamba_version = 'unknown'

try:
    from mamba_ssm import Mamba3
    _HAS_MAMBA3 = True
    import mamba_ssm
    _mamba_version = getattr(mamba_ssm, '__version__', 'unknown')
    print(f"[OK] Mamba-3 loaded from mamba_ssm v{_mamba_version} (official Triton kernels)")
except ImportError as e:
    _MAMBA3_ERROR = str(e)
    print(f"[WARNING] Mamba-3 not available: {e}")
    print(f"  Install from GitHub:")
    print(f'  TORCH_CUDA_ARCH_LIST="8.0" pip install --force-reinstall --no-deps \\')
    print(f"    git+https://github.com/state-spaces/mamba.git --no-build-isolation --break-system-packages")

try:
    from datasets import load_dataset
    _HAS_HF = True
except ImportError:
    _HAS_HF = False

warnings.filterwarnings('ignore', category=UserWarning)

# ============================================================================
# VOCABULARY (must match main ASR script exactly)
# ============================================================================

VOCAB_CHARS = list(" 'abcdefghijklmnopqrstuvwxyz")
BLANK_TOKEN = len(VOCAB_CHARS)  # 27
VOCAB_SIZE = len(VOCAB_CHARS) + 1  # 28
CHAR_TO_IDX = {c: i for i, c in enumerate(VOCAB_CHARS)}
IDX_TO_CHAR = {i: c for i, c in enumerate(VOCAB_CHARS)}


def text_to_ids(text: str) -> List[int]:
    return [CHAR_TO_IDX[c] for c in text.lower() if c in CHAR_TO_IDX]

def ids_to_text(ids: List[int]) -> str:
    return "".join(IDX_TO_CHAR[i] for i in ids if i in IDX_TO_CHAR)


# ============================================================================
# HYPERPARAMETERS (all defaults in one place for easy tuning)
# ============================================================================
# Mamba-1 reference: d=320, L=18, N=16, batch=128, lr=1e-3, wd=0.01, ls=0.1
#                    12,184,349 params, val PPL=4.04

# -- Model architecture --
DEFAULT_D_MODEL    = 320     # model dimension (Mamba-1 used 320)
DEFAULT_N_LAYERS   = 18      # number of layers (Mamba-1 used 18)
DEFAULT_D_STATE    = 64      # SSM state size (Mamba-1 used 16; Mamba-3 paper recommends 64+)
DEFAULT_HEADDIM    = 64      # head dimension for multi-head SSM
DEFAULT_DROPOUT    = 0.3     # embedding dropout

# -- Training --
DEFAULT_EPOCHS     = 20      # max epochs (Mamba-1 used 20)
DEFAULT_BATCH_SIZE = 64      # micro-batch per forward pass (Triton kernels are memory-efficient)
DEFAULT_ACCUM_STEPS = 2      # gradient accumulation (effective batch = 64 * 2 = 128)
DEFAULT_LR         = 1e-3    # learning rate (Mamba-1 used 1e-3)
DEFAULT_WEIGHT_DECAY = 0.01  # AdamW weight decay (Mamba-1 used 0.01)
DEFAULT_LABEL_SMOOTHING = 0.1  # label smoothing (Mamba-1 used 0.1)
DEFAULT_GRAD_CLIP  = 5.0     # gradient clipping (Mamba-1 used 5.0)
DEFAULT_PATIENCE   = 5       # early stopping patience (Mamba-1 used 5)

# -- Data --
DEFAULT_MAX_CHARS  = 1_000_000_000  # max chars from dataset
DEFAULT_MAX_SEQ_LEN = 512    # sequence length per chunk (Mamba-1 used 512)
DEFAULT_VAL_FRAC   = 0.05    # validation fraction (Mamba-1 used 0.05)


# ============================================================================
# MAMBA-3 CHARACTER LM
# ============================================================================

class CharMamba3LM(nn.Module):
    """
    Character-level Language Model using Mamba-3 blocks.
    
    Designed as a drop-in comparison for CharMambaLM (Mamba-1).
    Same vocab, same embedding, same output head -- only the SSM block differs.
    
    Interface matches CharMambaLM exactly:
      - score_sequence(char_ids, device) --> float
      - score_sequence_detailed(char_ids, device) --> (float, list)
      - get_log_probs(input_ids) --> tensor
    """
    
    def __init__(self, vocab_size=VOCAB_SIZE, d_model=256, n_layers=18,
                 d_state=64, headdim=64, is_mimo=False, mimo_rank=4,
                 dropout=0.1):
        super().__init__()
        
        if not _HAS_MAMBA3:
            raise ImportError(
                f"Mamba-3 not available. Error: {_MAMBA3_ERROR}\n"
                f"Install from GitHub: pip install git+https://github.com/state-spaces/mamba.git"
            )
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers_count = n_layers
        self.d_state = d_state
        self.headdim = headdim
        self.is_mimo = is_mimo
        
        # Detect GPU capability for dtype selection
        if torch.cuda.is_available():
            capability = torch.cuda.get_device_capability()
            if capability[0] >= 8:  # Ampere+ (A100, H100)
                self.dtype = torch.bfloat16
                print(f"  Using bfloat16 (GPU compute capability {capability[0]}.{capability[1]})")
            else:
                self.dtype = torch.float16
                print(f"  Using float16 (compute capability {capability[0]}.{capability[1]})")
        else:
            self.dtype = torch.float32
            print(f"  Using float32 (CPU)")
        
        # Embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.embed_dropout = nn.Dropout(dropout)
        
        # Mamba-3 layers with RMSNorm (pre-norm architecture)
        # Uses official mamba_ssm.Mamba3 with Triton kernels
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for _ in range(n_layers):
            self.norms.append(nn.RMSNorm(d_model))
            self.layers.append(
                Mamba3(
                    d_model=d_model,
                    d_state=d_state,
                    headdim=headdim,
                    is_mimo=is_mimo,
                    mimo_rank=mimo_rank if is_mimo else 1,
                    is_outproj_norm=False,
                    dtype=self.dtype,
                )
            )
        
        self.final_norm = nn.RMSNorm(d_model)
        self.output_head = nn.Linear(d_model, vocab_size)
        
        # Parameter count
        total = sum(p.numel() for p in self.parameters())
        mimo_tag = f", MIMO R={mimo_rank}" if is_mimo else ", SISO"
        print(f"\n[CharMamba3LM] {total:,} params "
              f"(d={d_model}, L={n_layers}, N={d_state}, head={headdim}{mimo_tag})"
              f" [official mamba_ssm / Triton kernels]")
    
    def forward(self, input_ids):
        """Forward pass. input_ids: [B, T] --> logits: [B, T, V]"""
        x = self.embed_dropout(self.embedding(input_ids))
        
        # Cast to model dtype for Mamba-3 Triton kernels
        x = x.to(self.dtype)
        
        for norm, layer in zip(self.norms, self.layers):
            residual = x
            x = norm(x)
            x = layer(x)  # official Mamba3 returns tensor directly
            x = residual + x
        
        x = self.final_norm(x)
        x = x.to(self.output_head.weight.dtype)  # cast back for output head
        return self.output_head(x)
    
    def get_log_probs(self, input_ids):
        """Get log probabilities. input_ids: [B, T] --> log_probs: [B, T, V]"""
        return F.log_softmax(self.forward(input_ids), dim=-1)
    
    def score_sequence(self, char_ids: List[int], device=None) -> float:
        """Score a character sequence. Returns total log P (natural log)."""
        if len(char_ids) == 0:
            return 0.0
        if device is None:
            device = next(self.parameters()).device
        
        with torch.no_grad():
            input_seq = [BLANK_TOKEN] + char_ids[:-1]
            input_t = torch.LongTensor([input_seq]).to(device)
            log_probs = self.get_log_probs(input_t)
            
            total = 0.0
            for t, tgt in enumerate(char_ids):
                if 0 <= tgt < self.vocab_size:
                    total += log_probs[0, t, tgt].item()
            return total
    
    def score_sequence_detailed(self, char_ids: List[int], device=None):
        """Score with per-character breakdown. Matches CharMambaLM interface."""
        if len(char_ids) == 0:
            return 0.0, []
        if device is None:
            device = next(self.parameters()).device
        
        with torch.no_grad():
            input_seq = [BLANK_TOKEN] + char_ids[:-1]
            input_t = torch.LongTensor([input_seq]).to(device)
            log_probs = self.get_log_probs(input_t)
            probs = torch.exp(log_probs)
            
            total = 0.0
            details = []
            for t, tgt in enumerate(char_ids):
                if 0 <= tgt < self.vocab_size:
                    lp = log_probs[0, t, tgt].item()
                    p = probs[0, t, tgt].item()
                    total += lp
                    
                    top_vals, top_idx = probs[0, t, :self.vocab_size].topk(3)
                    top3 = [(VOCAB_CHARS[idx.item()] if idx.item() < len(VOCAB_CHARS) else '?',
                             top_vals[j].item()) for j, idx in enumerate(top_idx)]
                    
                    details.append({
                        'char': VOCAB_CHARS[tgt] if tgt < len(VOCAB_CHARS) else '?',
                        'log_prob': lp,
                        'prob': p,
                        'top3': top3,
                    })
            return total, details


# ============================================================================
# DATASET (same as main script)
# ============================================================================

from torch.utils.data import Dataset, DataLoader

class ChunkedLMDataset(Dataset):
    """LM dataset from a pre-split character ID list. No overlap between train/val.
    Matches IKT464 Mamba-1 dataset format exactly:
      input:  [BLANK_TOKEN] + chunk[:-1]  (teacher forcing with BOS)
      target: chunk[:max_seq_len]
    """
    def __init__(self, char_ids: list, max_seq_len: int = 256, stride: int = None):
        self.max_seq_len = max_seq_len
        self.all_ids = char_ids
        self.stride = stride if stride is not None else max_seq_len
        self.n_samples = max(1, (len(self.all_ids) - max_seq_len) // self.stride)
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        start = idx * self.stride
        chunk = self.all_ids[start:start + self.max_seq_len + 1]
        if len(chunk) < self.max_seq_len + 1:
            chunk += [BLANK_TOKEN] * (self.max_seq_len + 1 - len(chunk))
        inp = [BLANK_TOKEN] + chunk[:-1]
        tgt = chunk[:self.max_seq_len]
        return torch.LongTensor(inp[:self.max_seq_len]), torch.LongTensor(tgt)


def build_datasets(hf_dataset, max_seq_len=DEFAULT_MAX_SEQ_LEN, max_chars=DEFAULT_MAX_CHARS):
    """Build train/val datasets from HuggingFace text dataset.
    Matches IKT464 build_elm_datasets exactly:
      - 5% val split (not 10%)
      - stride = max_seq_len (non-overlapping, not half-overlap)
      - Splits raw text first, then chunks independently (no leakage)
    """
    print(f"  [ELM Data] Loading from HuggingFace...")
    all_ids = []
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
    
    # Val split -- matches Mamba-1 build_elm_datasets
    split_point = int(len(all_ids) * (1.0 - DEFAULT_VAL_FRAC))
    train_ids = all_ids[:split_point]
    val_ids = all_ids[split_point:]
    
    # Non-overlapping stride -- matches Mamba-1
    train_ds = ChunkedLMDataset(train_ids, max_seq_len, max_seq_len)
    val_ds = ChunkedLMDataset(val_ids, max_seq_len, max_seq_len)
    
    print(f"  [ELM Data] {len(all_ids):,} chars total")
    print(f"    Train: {len(train_ids):,} chars -> {len(train_ds):,} chunks (stride={max_seq_len})")
    print(f"    Val:   {len(val_ids):,} chars -> {len(val_ds):,} chunks (stride={max_seq_len})")
    return train_ds, val_ds


# ============================================================================
# TRAINING
# ============================================================================

def train_model(model, train_ds, val_ds, save_dir, epochs=DEFAULT_EPOCHS,
                batch_size=DEFAULT_BATCH_SIZE, accum_steps=DEFAULT_ACCUM_STEPS,
                lr=DEFAULT_LR, label_smoothing=DEFAULT_LABEL_SMOOTHING,
                weight_decay=DEFAULT_WEIGHT_DECAY):
    """Train the Mamba-3 character LM.
    
    Training procedure aligned to IKT464 train_lm() for fair comparison:
      - CrossEntropyLoss with ignore_index=BLANK_TOKEN
      - AdamW with CosineAnnealingLR (T_max=total_steps, stepped per batch)
      - Grad clipping at 5.0 (same as Mamba-1)
      - No mixed precision (fp32, same as Mamba-1)
      - Early stopping with patience
      - Gradient accumulation (effective_batch = batch_size * accum_steps)
    """
    label = "ELM"
    effective_batch = batch_size * accum_steps
    os.makedirs(save_dir, exist_ok=True)
    device = next(model.parameters()).device
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # total_steps counts optimizer steps, not micro-batches
    total_steps = (len(train_loader) // accum_steps) * epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    criterion = nn.CrossEntropyLoss(ignore_index=BLANK_TOKEN, label_smoothing=label_smoothing)
    
    # Hyperparameter logging (matches Mamba-1 format)
    hparams = {
        'label': label,
        'model_type': 'mamba3',
        'd_model': model.d_model,
        'n_layers': model.n_layers_count,
        'd_state': model.d_state,
        'headdim': model.headdim,
        'is_mimo': model.is_mimo,
        'vocab_size': model.vocab_size,
        'total_params': sum(p.numel() for p in model.parameters()),
        'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'train_chunks': len(train_ds),
        'val_chunks': len(val_ds),
        'epochs': epochs,
        'batch_size': batch_size,
        'accum_steps': accum_steps,
        'effective_batch': effective_batch,
        'lr': lr,
        'weight_decay': weight_decay,
        'label_smoothing': label_smoothing,
        'patience': DEFAULT_PATIENCE,
        'device': str(device),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'save_dir': save_dir,
        'backend': f'official mamba_ssm v{_mamba_version} (Triton kernels)',
    }
    
    with open(os.path.join(save_dir, 'elm_hparams.json'), 'w') as f:
        json.dump(hparams, f, indent=2, default=str)
    
    log_path = os.path.join(save_dir, 'elm_training.log')
    logf = open(log_path, 'w')
    
    def log(msg):
        print(msg)
        logf.write(msg + '\n')
        logf.flush()
    
    log(f"\n{'='*70}")
    log(f"  {label} Training Configuration")
    log(f"{'='*70}")
    for k, v in hparams.items():
        log(f"  {k:20s}: {v:,}" if k == 'total_params' else f"  {k:20s}: {v}")
    log(f"{'='*70}")
    
    if label_smoothing > 0:
        min_loss = -math.log(1.0 - label_smoothing + label_smoothing / model.vocab_size)
        log(f"  [{label}] Label smoothing={label_smoothing}, theoretical loss floor={min_loss:.4f} (PPL {math.exp(min_loss):.2f})")
    
    best_val_loss = float('inf')
    patience = DEFAULT_PATIENCE
    no_improve = 0
    history = []
    total_micro_batches = len(train_loader)
    opt_steps_per_epoch = total_micro_batches // accum_steps
    
    # Progress intervals: 5% for terminal (overwrite same line), 25% for log file
    pct1_interval = max(1, opt_steps_per_epoch // 20)
    pct10_interval = max(1, opt_steps_per_epoch // 4)
    
    log(f"  [{label}] Micro-batches/epoch: {total_micro_batches}, "
        f"Optimizer steps/epoch: {opt_steps_per_epoch}, "
        f"Effective batch: {effective_batch}")
    log("")
    
    training_start = time.time()
    
    for epoch in range(epochs):
        # --- Train ---
        model.train()
        total_loss, n_batches = 0.0, 0
        recent_losses = []
        opt_step_count = 0
        optimizer.zero_grad()
        epoch_start = time.time()
        
        for micro_step, (input_ids, target_ids) in enumerate(train_loader):
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            
            logits = model(input_ids)
            loss = criterion(logits.reshape(-1, model.vocab_size), target_ids.reshape(-1))
            (loss / accum_steps).backward()
            
            total_loss += loss.item()
            recent_losses.append(loss.item())
            n_batches += 1
            
            if (micro_step + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), DEFAULT_GRAD_CLIP)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                opt_step_count += 1
                
                # Within-epoch progress (1% terminal overwrite, 10% to log file)
                if opt_step_count % pct1_interval == 0:
                    elapsed = time.time() - epoch_start
                    pct = 100.0 * (micro_step + 1) / total_micro_batches
                    window = min(len(recent_losses), pct1_interval * accum_steps)
                    avg_recent = sum(recent_losses[-window:]) / window
                    ppl_recent = math.exp(min(avg_recent, 20))
                    avg_all = total_loss / n_batches
                    ppl_all = math.exp(min(avg_all, 20))
                    samples_per_sec = (micro_step + 1) * batch_size / elapsed
                    eta_epoch = elapsed / pct * (100 - pct) if pct > 0 else 0
                    current_lr = scheduler.get_last_lr()[0]
                    
                    mem_str = ""
                    if torch.cuda.is_available():
                        peak_gb = torch.cuda.max_memory_allocated() / 1024**3
                        mem_str = f" | Peak: {peak_gb:.1f}GB"
                    
                    progress_msg = (
                        f"    [{label}] Ep {epoch+1} [{pct:5.1f}%] "
                        f"loss: {avg_recent:.4f} PPL: {ppl_recent:.2f} | "
                        f"avg: {avg_all:.4f} PPL: {ppl_all:.2f} | "
                        f"lr: {current_lr:.2e} | "
                        f"{samples_per_sec:.0f} samp/s | "
                        f"ETA: {eta_epoch/60:.0f}m{mem_str}")
                    
                    # Terminal: overwrite same line
                    print(f"\r{progress_msg}", end="", flush=True)
                    
                    # Log file: write every 10% (permanent record)
                    if opt_step_count % pct10_interval == 0:
                        logf.write(progress_msg + "\n")
                        logf.flush()
        
        # Clear the progress line before printing epoch summary
        print("", flush=True)
        
        train_avg = total_loss / max(n_batches, 1)
        train_ppl = math.exp(min(train_avg, 20))
        
        # --- Validate ---
        model.eval()
        val_loss, val_n = 0.0, 0
        val_start = time.time()
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
        val_time = time.time() - val_start
        
        epoch_time = time.time() - epoch_start
        total_elapsed = time.time() - training_start
        gap = val_avg - train_avg
        star = " *" if val_avg < best_val_loss else ""
        
        # GPU peak memory for this epoch
        peak_str = ""
        if torch.cuda.is_available():
            peak_gb = torch.cuda.max_memory_allocated() / 1024**3
            peak_str = f" | GPU Peak: {peak_gb:.1f}GB"
            torch.cuda.reset_peak_memory_stats()
        
        msg = (f"  [{label}] Epoch {epoch+1:2d}/{epochs} -- "
               f"Train: {train_avg:.4f} (PPL {train_ppl:.2f}), "
               f"Val: {val_avg:.4f} (PPL {val_ppl:.2f}), "
               f"Gap: {gap:+.4f}, "
               f"Time: {epoch_time:.0f}s (val: {val_time:.0f}s), "
               f"Total: {total_elapsed/3600:.1f}h{peak_str}{star}")
        log(msg)
        
        history.append({'epoch': epoch+1, 'train_loss': train_avg, 'val_loss': val_avg,
                        'train_ppl': train_ppl, 'val_ppl': val_ppl,
                        'epoch_time_s': epoch_time, 'total_time_h': total_elapsed/3600})
        
        # Save on best val loss
        if val_avg < best_val_loss:
            best_val_loss = val_avg
            no_improve = 0
            
            config = {
                'model_type': 'mamba3',
                'vocab_size': model.vocab_size,
                'd_model': model.d_model,
                'n_layers': model.n_layers_count,
                'd_state': model.d_state,
                'headdim': model.headdim,
                'is_mimo': model.is_mimo,
                'label': f"ELM-Mamba3 (d={model.d_model}, L={model.n_layers_count})",
            }
            
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': config,
                'best_loss': best_val_loss,
                'hparams': hparams,
            }, os.path.join(save_dir, 'elm_best.pt'))
            log(f"    [{label}] Best model saved (val_loss={val_avg:.4f}, PPL={val_ppl:.2f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                log(f"  [{label}] Early stopping at epoch {epoch+1} (no val improvement for {patience} epochs)")
                break
    
    # Save training history
    with open(os.path.join(save_dir, 'elm_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    val_ppl = math.exp(min(best_val_loss, 20))
    log(f"\n  [{label}] Done. Best val loss={best_val_loss:.4f}, val PPL={val_ppl:.2f}")
    log(f"  [{label}] Checkpoint: {os.path.join(save_dir, 'elm_best.pt')}")
    log(f"  [{label}] Log: {log_path}")
    
    # Generate training plots
    plot_paths = generate_training_plots(history, save_dir, hparams)
    for p in plot_paths:
        log(f"  [{label}] Plot: {p}")
    
    logf.close()


def generate_training_plots(history, save_dir, hparams):
    """Generate PDF training plots for thesis report."""
    try:
        import matplotlib
        matplotlib.use('Agg')  # non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [WARNING] matplotlib not available, skipping plots")
        print("  Install with: pip install matplotlib --break-system-packages")
        return []
    
    epochs = [h['epoch'] for h in history]
    train_loss = [h['train_loss'] for h in history]
    val_loss = [h['val_loss'] for h in history]
    train_ppl = [h['train_ppl'] for h in history]
    val_ppl = [h['val_ppl'] for h in history]
    
    model_label = (f"Mamba-3 (d={hparams['d_model']}, L={hparams['n_layers']}, "
                   f"N={hparams['d_state']}, {hparams['total_params']:,} params)")
    
    plot_paths = []
    
    # --- Plot 1: Training & Validation Loss ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_loss, 'b-o', markersize=4, label='Train Loss')
    ax.plot(epochs, val_loss, 'r-o', markersize=4, label='Val Loss')
    best_epoch = min(range(len(val_loss)), key=lambda i: val_loss[i])
    ax.axvline(x=epochs[best_epoch], color='gray', linestyle='--', alpha=0.5,
               label=f'Best epoch {epochs[best_epoch]}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Cross-Entropy Loss')
    ax.set_title(f'Training Loss -- {model_label}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = os.path.join(save_dir, 'plot_loss.pdf')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    plot_paths.append(path)
    
    # --- Plot 2: Validation Perplexity ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, val_ppl, 'r-o', markersize=4, label='Val PPL')
    ax.plot(epochs, train_ppl, 'b-o', markersize=4, label='Train PPL', alpha=0.6)
    ax.axhline(y=val_ppl[best_epoch], color='gray', linestyle='--', alpha=0.5,
               label=f'Best val PPL = {val_ppl[best_epoch]:.2f}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Perplexity')
    ax.set_title(f'Perplexity -- {model_label}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = os.path.join(save_dir, 'plot_ppl.pdf')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    plot_paths.append(path)
    
    # --- Plot 3: Train-Val Gap (overfitting monitor) ---
    gaps = [v - t for t, v in zip(train_loss, val_loss)]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, gaps, 'g-o', markersize=4, label='Val - Train Loss')
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss Gap (Val - Train)')
    ax.set_title(f'Generalization Gap -- {model_label}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = os.path.join(save_dir, 'plot_gap.pdf')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    plot_paths.append(path)
    
    print(f"  [ELM] Generated {len(plot_paths)} PDF plots in {save_dir}/")
    return plot_paths


# ============================================================================
# LOADING (for integration with main script)
# ============================================================================

def load_mamba3_lm(path, device='cuda'):
    """Load a trained CharMamba3LM checkpoint.
    
    To integrate with the ASR evaluation script's load_lm(), add:
        if cfg['model_type'] == 'mamba3':
            from <this_module> import CharMamba3LM
            model = CharMamba3LM(...)
    """
    if os.path.isdir(path):
        pt_files = [f for f in os.listdir(path) if f.endswith('_best.pt')]
        if not pt_files:
            raise FileNotFoundError(f"No *_best.pt found in {path}")
        pt_path = os.path.join(path, pt_files[0])
    else:
        pt_path = path
    
    ckpt = torch.load(pt_path, map_location=device, weights_only=False)
    cfg = ckpt['config']
    
    model = CharMamba3LM(
        vocab_size=cfg['vocab_size'],
        d_model=cfg['d_model'],
        n_layers=cfg['n_layers'],
        d_state=cfg.get('d_state', 64),
        headdim=cfg.get('headdim', 64),
        is_mimo=cfg.get('is_mimo', False),
    )
    
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device).eval()
    print(f"[LoadLM] {cfg['label']} from {pt_path} (loss={ckpt['best_loss']:.4f})")
    return model


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='CharMamba3LM Training')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['verify', 'train'],
                        help='verify = check Mamba-3 availability, train = train model')
    
    # Data
    parser.add_argument('--lm-hf-dataset', type=str, default='openslr/librispeech_lm')
    parser.add_argument('--lm-text-path', type=str, default=None)
    parser.add_argument('--max-chars', type=int, default=DEFAULT_MAX_CHARS)
    
    # Model architecture (see HYPERPARAMETERS section above)
    parser.add_argument('--d-model', type=int, default=DEFAULT_D_MODEL)
    parser.add_argument('--n-layers', type=int, default=DEFAULT_N_LAYERS)
    parser.add_argument('--d-state', type=int, default=DEFAULT_D_STATE)
    parser.add_argument('--headdim', type=int, default=DEFAULT_HEADDIM)
    parser.add_argument('--mimo', action='store_true', default=False)
    parser.add_argument('--mimo-rank', type=int, default=4)
    
    # Training (see HYPERPARAMETERS section above)
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS)
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE,
                        help='Micro-batch per forward pass (limited by GPU memory)')
    parser.add_argument('--accum-steps', type=int, default=DEFAULT_ACCUM_STEPS,
                        help='Gradient accumulation steps (effective batch = batch-size * accum-steps)')
    parser.add_argument('--lr', type=float, default=DEFAULT_LR)
    parser.add_argument('--dropout', type=float, default=DEFAULT_DROPOUT)
    parser.add_argument('--label-smoothing', type=float, default=DEFAULT_LABEL_SMOOTHING)
    parser.add_argument('--save-dir', type=str, default=None)
    
    args = parser.parse_args()
    
    # ================================================================
    # VERIFY MODE
    # ================================================================
    if args.mode == 'verify':
        print(f"\n{'='*70}")
        print(f"  Mamba-3 Availability Check (official mamba_ssm)")
        print(f"{'='*70}\n")
        
        # Check mamba_ssm
        print(f"  Mamba-3 available: {_HAS_MAMBA3}")
        if not _HAS_MAMBA3:
            print(f"  Error: {_MAMBA3_ERROR}")
            print(f"  Install from GitHub (NOT PyPI):")
            print(f'  TORCH_CUDA_ARCH_LIST="8.0" pip install --force-reinstall --no-deps \\')
            print(f"    git+https://github.com/state-spaces/mamba.git --no-build-isolation --break-system-packages")
            return
        print(f"  Backend: official mamba_ssm v{_mamba_version} (Triton kernels)")
        
        # Check GPU
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            cap = torch.cuda.get_device_capability()
            bf16 = cap[0] >= 8
            print(f"  GPU: {name} ({mem:.1f} GB)")
            print(f"  Compute capability: {cap[0]}.{cap[1]}")
            print(f"  bfloat16 support: {'Yes' if bf16 else 'No -- will use float16'}")
            if cap[0] < 8:
                print(f"  [WARNING] sm_{cap[0]}{cap[1]} may not support Mamba-3 Triton kernels (need sm_75+)")
        else:
            print(f"  GPU: None (CPU only)")
        
        # Quick test: create a small Mamba-3 block
        print(f"\n  Creating test Mamba-3 block...")
        try:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
            
            test = Mamba3(d_model=64, d_state=64, headdim=32, dtype=dtype).to(device)
            x = torch.randn(1, 32, 64, dtype=dtype, device=device)
            y = test(x)
            print(f"  [OK] Mamba-3 block works! Input: {x.shape} --> Output: {y.shape}")
            del test, x, y
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"  [FAIL] Mamba-3 block FAILED: {e}")
            import traceback
            traceback.print_exc()
            return
        
        # Test full model
        print(f"\n  Creating test CharMamba3LM...")
        try:
            model = CharMamba3LM(d_model=64, n_layers=2, d_state=64, headdim=32).to(device)
            ids = text_to_ids("the cat sat on the mat")
            score = model.score_sequence(ids, device=device)
            print(f"  [OK] CharMamba3LM works! Score: {score:.4f}")
            
            total, details = model.score_sequence_detailed(ids, device=device)
            print(f"  [OK] score_sequence_detailed works! {len(details)} chars scored")
            
            # Quick memory check
            if torch.cuda.is_available():
                peak_mb = torch.cuda.max_memory_allocated() / 1024**2
                print(f"  [OK] Peak GPU memory (tiny model): {peak_mb:.0f} MB")
                torch.cuda.empty_cache()
            
            del model
        except Exception as e:
            print(f"  [FAIL] CharMamba3LM FAILED: {e}")
            import traceback
            traceback.print_exc()
            return
        
        print(f"\n  [OK] All checks passed. Ready to train with --mode train")
        print(f"{'='*70}\n")
        return
    
    # ================================================================
    # TRAIN MODE
    # ================================================================
    if not _HAS_MAMBA3:
        print(f"ERROR: Mamba-3 not available. Run with --mode verify first.")
        sys.exit(1)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Build save dir name (matches Mamba-1 convention)
    mimo_tag = f"_MIMO-R{args.mimo_rank}" if args.mimo else ""
    save_dir = args.save_dir or os.path.join(
        "lm_checkpoints",
        f"elm_mamba3_MaxChars-{args.max_chars}_d{args.d_model}_L{args.n_layers}{mimo_tag}"
    )
    
    # Load data
    if args.lm_hf_dataset:
        print(f"\nLoading HF dataset: {args.lm_hf_dataset}")
        hf_ds = load_dataset(args.lm_hf_dataset, split='train', trust_remote_code=True)
        train_ds, val_ds = build_datasets(hf_ds, max_seq_len=DEFAULT_MAX_SEQ_LEN, max_chars=args.max_chars)
    elif args.lm_text_path:
        # Read text file
        print(f"\nLoading text from: {args.lm_text_path}")
        with open(args.lm_text_path, 'r') as f:
            lines = f.readlines()
        class FakeHF:
            def __iter__(self):
                for line in lines:
                    yield {'text': line}
        train_ds, val_ds = build_datasets(FakeHF(), max_seq_len=DEFAULT_MAX_SEQ_LEN, max_chars=args.max_chars)
    else:
        print("ERROR: --lm-hf-dataset or --lm-text-path required")
        sys.exit(1)
    
    # Create model
    model = CharMamba3LM(
        vocab_size=VOCAB_SIZE,
        d_model=args.d_model,
        n_layers=args.n_layers,
        d_state=args.d_state,
        headdim=args.headdim,
        is_mimo=args.mimo,
        mimo_rank=args.mimo_rank,
        dropout=args.dropout,
    ).to(device)
    
    # Compare param count to Mamba-1
    mamba3_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Mamba-3 params:  {mamba3_params:,}")
    print(f"  Mamba-1 params:  12,184,349 (d=320, L=18 -- your existing ELM)")
    print(f"  Ratio:           {mamba3_params / 12_184_349:.2f}x")
    
    # Train
    train_model(model, train_ds, val_ds, save_dir,
                epochs=args.epochs, batch_size=args.batch_size,
                accum_steps=args.accum_steps, lr=args.lr,
                label_smoothing=args.label_smoothing)
    
    # Integration hint for the ASR evaluation script
    print(f"\n  To use with the ASR evaluation script, update load_lm() to handle model_type='mamba3':")
    print(f"    elif cfg['model_type'] == 'mamba3':")
    print(f"        from {os.path.splitext(os.path.basename(__file__))[0]} import CharMamba3LM")
    print(f"        model = CharMamba3LM(...)")
    print(f"\n  Then run: --elm-path {save_dir}")


if __name__ == '__main__':
    main()
