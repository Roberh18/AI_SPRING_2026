"""
SSM-ASR: CTC Encoder Training Mamba-3
===========================================================

Trains a CTC acoustic encoder for LibriSpeech using one of three SSM backends:
  - Mamba-3  (trapezoidal discretization, complex SSM with RoPE, Triton kernels)
  - Mamba-1  (ZOH discretization, causal conv, CUDA kernels via mamba_ssm)
  - S-SSSM   (original selective SSM from the thesis, pure PyTorch)

Optionally applies non-uniform layer-wise initialization of dt_bias / decay
parameters across encoder depth (--no-hierarchical to disable).

NOTE The flag ``--no-hierarchical`` and related identifiers
(``use_hierarchical``, ``HIER_CONFIG``, ``_apply_hierarchical_init``, …) are
kept for compatibility with existing experiment commands, checkpoints, and logs.
In the final thesis ("State Space Models for Automatic Speech Transcription",
R. A. Hanssen, University of Agder, 2025), this is described as *non-uniform
layer-wise initialization* or *initialization diversity* of decay / timescale
parameters — not as evidence for a specific acoustic-to-linguistic hierarchy.

References:
  - Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State
    Spaces", 2023 (arXiv:2312.00752)
  - Lahoti et al., "Mamba-3: Improved Sequence Modeling using State Space
    Principles", 2025 (arXiv:2603.15569)

SETUP:
    # Mamba-1 CUDA kernels:
    pip install mamba-ssm>=2.0.0 causal-conv1d>=1.4.0

    # Mamba-3 Triton kernels (requires A100/H100, sm_75+ GPU):
    TORCH_CUDA_ARCH_LIST="8.0" pip install --force-reinstall --no-deps \\
      git+https://github.com/state-spaces/mamba.git --no-build-isolation
    pip install tilelang==0.1.8 quack-kernels==0.3.1

USAGE EXAMPLES:

    # Mamba-3 encoder
    python IKT590_train_ASR_encoder_mamba-3.py \\
      --data-path hub_data/librispeech --encoder-type mamba3 \\
      --d-model 256 --n-layers 12 --d-state 64 --headdim 64

"""



# ============================================================================
# IMPORTS
# ============================================================================

from __future__ import annotations

# Standard library imports
import argparse
import json
import math
import os
import random
import sys
import time
import traceback
import warnings
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Third-party library imports
import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from jiwer import wer as jiwer_wer
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader, Dataset

# Optional dependency (check dataset availability)
try:
    from datasets import load_from_disk, concatenate_datasets
    _HAS_HF = True
except ImportError:
    _HAS_HF = False

# Optional dependency: mamba_ssm for CUDA-optimized kernels (10-100x faster)
# Install with: pip install mamba-ssm>=2.0.0 causal-conv1d>=1.4.0
try:
    from mamba_ssm import Mamba as MambaCUDA
    _HAS_MAMBA_SSM = True
except ImportError:
    _HAS_MAMBA_SSM = False
    MambaCUDA = None

# Optional: Mamba-3 (requires mamba-ssm >= 2.3.1 installed from GitHub, sm_75+ GPU)
try:
    from mamba_ssm import Mamba3 as Mamba3CUDA
    _HAS_MAMBA3 = True
except ImportError:
    _HAS_MAMBA3 = False
    Mamba3CUDA = None

# Suppress PyTorch Lightning's TF32 deprecation warning
warnings.filterwarnings('ignore', category=UserWarning, message='.*Please use the new API settings to control TF32.*')

# Suppress the "Checkpoint directory exists" warning 
warnings.filterwarnings('ignore', category=UserWarning, message='Checkpoint directory .* exists and is not empty.')


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(log_file: str, level=logging.INFO):
    """
    Set up logging to both console and file.
    
    This is multiprocessing-safe and won't cause deadlocks with DataLoader workers.
    """
    logger = logging.getLogger('ASR_Training')
    logger.setLevel(level)
    logger.handlers.clear()
    
    formatter = logging.Formatter('%(message)s')
    
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


# Store original print for fallback
_original_print = print


def logged_print(*args, **kwargs):
    """
    Module-level print replacement that logs to both file and console.
    Must be at module level for numba compatibility.
    """
    logger = logging.getLogger('ASR_Training')
    
    # If logger has no handlers yet, fall back to original print
    if not logger.handlers:
        _original_print(*args, **kwargs)
        return
    
    message = ' '.join(str(arg) for arg in args)
    end = kwargs.get('end', '\n')
    
    # Handle progress bar updates (carriage return) - terminal only
    if '\r' in message or (end != '\n' and end != ''):
        _original_print(*args, **kwargs)
        return
    
    # Log normal messages
    if message.strip():
        logger.info(message)
    elif end == '\n':
        logger.info('')


def override_print_with_logger():
    """
    Override the built-in print() to use logged_print.
    Call this AFTER setup_logging().
    """
    import builtins
    builtins.print = logged_print


# ============================================================================
# SYSTEM SETUP
# ============================================================================

print("\n")
print("SSM-ASR: CTC ENCODER TRAINING")
print("\n")

if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')  # The new unified API for TF32
    device_name = torch.cuda.get_device_name(0)
    memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    print(f"GPU: {device_name}")
    print(f"Memory: {memory_gb:.1f} GB")
    print("Compute: TF32 enabled (high precision)")
else:
    print("WARNING: Running on CPU - training will be very slow")

# Report mamba_ssm availability
if _HAS_MAMBA_SSM:
    print("Mamba Backend: CUDA kernels (mamba_ssm) - FAST")
else:
    print("Mamba Backend: Pure PyTorch (reference) - install mamba-ssm for 10-100x speedup")


# ============================================================================
# VOCABULARY & CONSTANTS
# ============================================================================

VOCAB_CHARS = list(" 'abcdefghijklmnopqrstuvwxyz")
BLANK_TOKEN = len(VOCAB_CHARS)
CHAR_TO_IDX = {c: i for i, c in enumerate(VOCAB_CHARS)}
IDX_TO_CHAR = {i: c for i, c in enumerate(VOCAB_CHARS)}

SAMPLE_RATE = 16000
N_MELS = 80

print(f"Vocabulary: {len(VOCAB_CHARS)} characters + blank token")
print(f"Audio: {SAMPLE_RATE}Hz, {N_MELS}-dim log mel-spectrogram")
print(f"Subsampling: 4x\n")

# ============================================================================
# NON-UNIFORM INIT CONFIGURATION (Module Level)
# ============================================================================
# Legacy name "HIER_CONFIG" kept for checkpoint / config compatibility.
# In the final thesis this is described as non-uniform layer-wise
# initialization (initialization diversity), not as evidence for a
# specific acoustic-to-linguistic hierarchy.

HIER_CONFIG = {
    # State Decay (a): High = Long Memory, Low = Short Memory
    # Note: tanh(0.85) ≈ 0.69 retention, tanh(0.15) ≈ 0.15 retention
    'A0_EARLY': 0.15,   # Lower initial decay (forward schedule)
    'A0_LATE':  0.85,   # Higher initial decay (forward schedule)
    
    # Input Sensitivity (b): High = Sensitive to new input
    'B0_EARLY': 0.35,   # Higher sensitivity in early layers
    'B0_LATE':  0.12,   # Lower sensitivity in late layers
    
    # Output Weight (c): High = Stronger contribution to output
    'C0_EARLY': -0.35,  # Lower state output → sigmoid ≈ 0.41
    'C0_LATE':  0.20    # Higher state output → sigmoid ≈ 0.55
}

MAMBA_HIER_CONFIG = {
    # dt (timescale) initialization range — forward non-uniform schedule.
    # Early layers: smaller dt → faster timescale
    # Late layers: larger dt → slower timescale
    'DT_MIN_EARLY': 0.001,
    'DT_MAX_EARLY': 0.01,
    'DT_MIN_LATE': 0.005,
    'DT_MAX_LATE': 0.05,
}





def text_to_ids(text: str) -> List[int]:
    """Convert text to character IDs."""
    return [CHAR_TO_IDX[c] for c in text.lower() if c in CHAR_TO_IDX]


def ids_to_text(ids: List[int]) -> str:
    """Convert character IDs to text."""
    return "".join(IDX_TO_CHAR[i] for i in ids if i in IDX_TO_CHAR)


# ============================================================================
# AUDIO FEATURE EXTRACTION (FROM v4.0)
# ============================================================================

_mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=400,
    win_length=400,
    hop_length=160,
    n_mels=N_MELS,
    f_min=20.0,
    f_max=7600.0,
    power=1.0
)

_amp_to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80.0)


def extract_features(
    waveform: torch.Tensor,
    sample_rate: int,
    apply_speed_perturb: bool = False,
    speed_perturb_factors: Tuple[float, ...] = (0.9, 1.0, 1.1)
) -> torch.Tensor:
    """
    Extract log mel-spectrogram features with optional speed perturbation.
    """
    # Resample if needed
    if sample_rate != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sample_rate, SAMPLE_RATE)
    
    # Convert to mono
    if waveform.ndim > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    elif waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    
    # Apply speed perturbation (training only)
    if apply_speed_perturb:
        speed_factor = random.choice(speed_perturb_factors)
        waveform = apply_speed_perturbation(waveform, SAMPLE_RATE, speed_factor)
    
    # Extract features (v4.0 stable approach)
    with torch.no_grad():
        mel_spec = _mel_transform(waveform)
        log_mel = _amp_to_db(mel_spec)
        mean = log_mel.mean()
        std = log_mel.std()
        log_mel = (log_mel - mean) / (std + 1e-5)
    
    return log_mel.squeeze(0).transpose(0, 1)


def apply_speed_perturbation(
    waveform: torch.Tensor,
    sample_rate: int,
    speed_factor: float = 1.0
) -> torch.Tensor:
    """
    Apply speed perturbation to waveform.
    
    Args:
        waveform: [channels, samples] or [samples]
        sample_rate: Original sample rate
        speed_factor: Speed multiplication factor (0.9, 1.0, 1.1)
    
    Returns:
        waveform: Speed-perturbed waveform
    """
    if speed_factor == 1.0:
        return waveform
    
    new_sample_rate = int(sample_rate * speed_factor)
    waveform = torchaudio.functional.resample(
        waveform,
        orig_freq=new_sample_rate,
        new_freq=sample_rate
    )
    
    return waveform


# ============================================================================
# DATASET & DATALOADER (FROM v4.0)
# ============================================================================

class LibriSpeechDataset(Dataset):
    """Lazy-loading LibriSpeech dataset."""

    def __init__(
            self,
            hf_dataset,
            subset: Optional[int] = None,
            split_name: str = "",
            use_specaugment: bool = False,
            use_speed_perturb: bool = False,
            speed_perturb_factors: Tuple[float, ...] = (0.9, 1.0, 1.1),
        ):
        self.split_name = split_name
        self.use_speed_perturb = use_speed_perturb 
        self.speed_perturb_factors = speed_perturb_factors 

        if subset and subset < len(hf_dataset):
            indices = list(range(subset))
            hf_dataset = hf_dataset.select(indices)
        
        self.dataset = hf_dataset
        
        print(f"\nDataset: {split_name}")
        print(f"  Size: {len(self.dataset)} examples")
        print(f"  Mode: LAZY LOADING (memory efficient)")

        # Initialize SpecAugment
        self.use_specaugment = use_specaugment
        if self.use_specaugment:
            print(f"  SpecAugment: ENABLED (Applied before padding)")
            self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=15)
            self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=50)
        
        if self.use_speed_perturb:
            print(f"  Speed perturbation: ENABLED")
            print(f"  Speed perturbation: {speed_perturb_factors}")
            print(f"  Factors: {speed_perturb_factors}")
        else:
            print(f"  Speed perturbation: DISABLED")  

        sample_size = min(1000, len(self.dataset))
        sample_indices = np.linspace(0, len(self.dataset)-1, sample_size, dtype=int)
        
        audio_lengths = []
        text_lengths = []
        
        for idx in sample_indices:
            item = self.dataset[int(idx)]
            audio_lengths.append(len(item['audio']['array']))
            text_lengths.append(len(item['text']))
        
        print(f"\nDataset statistics (sampled from {sample_size} examples):")
        print(f"  Audio length: min={min(audio_lengths)}, max={max(audio_lengths)}, "
              f"mean={int(np.mean(audio_lengths))}")
        print(f"  Audio duration: min={min(audio_lengths)/SAMPLE_RATE:.1f}s, "
              f"max={max(audio_lengths)/SAMPLE_RATE:.1f}s, "
              f"mean={np.mean(audio_lengths)/SAMPLE_RATE:.1f}s")
        print(f"  Text length: min={min(text_lengths)}, max={max(text_lengths)}, "
              f"mean={np.mean(text_lengths):.1f}")

    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        audio = item['audio']
        waveform = torch.FloatTensor(audio['array'])
        text = item['text']
        features = extract_features(
            waveform,
            SAMPLE_RATE,
            apply_speed_perturb=self.use_speed_perturb,
            speed_perturb_factors=self.speed_perturb_factors
        )
        
        # Apply SpecAugment (before padding)
        # Features shape is [Time, Freq]. Transforms expect [..., Freq, Time]
        if self.use_specaugment:
            # Transpose to [Freq, Time] for torchaudio
            features_t = features.transpose(0, 1)
            # Apply masks (2 freq masks, 2 time masks is standard)
            for _ in range(2):
                features_t = self.freq_mask(features_t)
            for _ in range(2):
                features_t = self.time_mask(features_t)
            # Transpose back to [Time, Freq]
            features = features_t.transpose(0, 1)

        return features, text


def collate_fn(batch):
    """Collate function with dynamic padding."""
    features, texts = zip(*batch)
    
    feature_lengths = torch.LongTensor([f.shape[0] for f in features])
    
    max_len = max(f.shape[0] for f in features)
    n_mels = features[0].shape[1]
    features_padded = torch.zeros(len(batch), max_len, n_mels)
    
    for i, f in enumerate(features):
        features_padded[i, :f.shape[0], :] = f
    
    text_ids = [torch.LongTensor(text_to_ids(t)) for t in texts]
    text_lengths = torch.LongTensor([len(t) for t in text_ids])
    
    max_text_len = max(len(t) for t in text_ids)
    text_ids_padded = torch.full(
        (len(batch), max_text_len),
        fill_value=BLANK_TOKEN,
        dtype=torch.long
    )
    
    for i, t in enumerate(text_ids):
        if len(t) > 0:
            text_ids_padded[i, :len(t)] = t
    
    return {
        'features': features_padded,
        'feature_lengths': feature_lengths,
        'text_ids': text_ids_padded,
        'text_lengths': text_lengths,
        'texts': texts,
    }




# ============================================================================
# MAMBA (with mamba_ssm CUDA kernel support)
# ============================================================================

class MambaLayer(nn.Module):
    """
    Mamba-1 layer with optional non-uniform layer-wise initialization.
    
    HYBRID implementation:
    - Uses mamba_ssm CUDA kernels when available (10-100x faster)
    - Falls back to pure PyTorch when not available
    
    Architecture:
        Input -> Norm -> Mamba Block -> Residual + Dropout -> Output
    
    Non-uniform init (when use_hierarchical=True) modifies dt_bias based
    on layer depth to create timescale diversity across the encoder stack.
    Flag name kept for compatibility; see module docstring for terminology.
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        layer_idx: int = 0,
        n_layers: int = 1,
        use_hierarchical: bool = True,
        use_cuda_kernels: bool = True,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)
        self.layer_idx = layer_idx
        self.n_layers = n_layers
        self.use_hierarchical = use_hierarchical
        
        # Determine if we should use CUDA kernels
        self.use_cuda = use_cuda_kernels and _HAS_MAMBA_SSM
        
        # Compute layer position [0.0 = first, 1.0 = last]
        self.layer_progress = layer_idx / max(1, n_layers - 1)
        
        # dt_rank: bottleneck dimension for delta projection
        self.dt_rank = math.ceil(d_model / 16)
        
        # Layer Norm (Pre-norm)
        self.norm = nn.LayerNorm(d_model)
        
        # Mamba Core - choose implementation
        if self.use_cuda:
            self._init_cuda_mamba()
        else:
            self._init_pytorch_mamba()
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Layer type label for logging (labels are historical; see module docstring)
        if self.layer_progress < 0.33:
            self.layer_type = "acoustic"
        elif self.layer_progress < 0.66:
            self.layer_type = "phonetic"
        else:
            self.layer_type = "linguistic"
    
    def _init_cuda_mamba(self):
        """Initialize using official mamba_ssm library with CUDA kernels."""
        self.mamba = MambaCUDA(
            d_model=self.d_model,
            d_state=self.d_state,
            d_conv=self.d_conv,
            expand=self.expand,
        )
        
        if self.use_hierarchical:
            self._apply_hierarchical_init_cuda()
        else:
            self.a_scale = 1.0
            self.dt_init_mean = 0.01
    
    def _apply_hierarchical_init_cuda(self):
        """
        Apply non-uniform layer-wise initialization (dt_bias only).
        
        Preserves Mamba's full frequency spectrum in A_log while biasing
        which timescales each layer prefers via dt_bias.  Early layers get
        smaller dt (faster timescale), late layers get larger dt (slower).
        Method name kept for checkpoint compatibility.
        """
        cfg = MAMBA_HIER_CONFIG
        progress = self.layer_progress  # 0.0 = first layer, 1.0 = last layer
        
        with torch.no_grad():
            # ============================================
            # DO NOT MODIFY A_log - preserve full spectrum
            # ============================================
            self.a_scale = 1.0  # No scaling applied
            
            # ============================================
            # ONLY modify dt_bias for hierarchical effect
            # ============================================
            # Interpolate dt range based on layer depth
            dt_min = cfg['DT_MIN_EARLY'] + (cfg['DT_MIN_LATE'] - cfg['DT_MIN_EARLY']) * progress
            dt_max = cfg['DT_MAX_EARLY'] + (cfg['DT_MAX_LATE'] - cfg['DT_MAX_EARLY']) * progress
            
            # Sample dt values from log-uniform distribution
            dt = torch.exp(
                torch.rand(self.d_inner, device=self.mamba.dt_proj.bias.device)
                * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
            ).clamp(min=1e-4)
            
            # Convert to inverse softplus for bias initialization
            # dt_proj applies softplus, so we need inverse: inv_softplus(x) = x + log(1 - exp(-x))
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            
            # Apply to dt_proj bias
            self.mamba.dt_proj.bias.copy_(inv_dt)
            
            # Store for logging
            self.dt_init_mean = dt.mean().item()
            
            # Log what we did
            if self.layer_idx == 0:
                print(f"\n  [non-uniform init] Preserving A_log spectrum, modifying dt_bias only")
    
    def _init_pytorch_mamba(self):
        """Initialize pure PyTorch implementation (fallback)."""
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False)
        
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=self.d_conv,
            padding=self.d_conv - 1,
            groups=self.d_inner,
            bias=True,
        )
        
        self.x_proj = nn.Linear(
            self.d_inner,
            self.dt_rank + self.d_state * 2,
            bias=False
        )
        
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self._init_dt_proj_pytorch()
        
        self._init_A_matrix_pytorch()
        
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True
        
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)
    
    def _init_dt_proj_pytorch(self):
        """Initialize dt_proj for PyTorch implementation."""
        dt_init_std = self.dt_rank ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        
        cfg = MAMBA_HIER_CONFIG
        
        if self.use_hierarchical:
            progress = self.layer_progress
            dt_min = cfg['DT_MIN_EARLY'] + (cfg['DT_MIN_LATE'] - cfg['DT_MIN_EARLY']) * progress
            dt_max = cfg['DT_MAX_EARLY'] + (cfg['DT_MAX_LATE'] - cfg['DT_MAX_EARLY']) * progress
        else:
            dt_min = 0.001
            dt_max = 0.1
        
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=1e-4)
        
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        
        self.dt_init_mean = dt.mean().item()
    
    def _init_A_matrix_pytorch(self):
        """Initialize A matrix for PyTorch implementation."""
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32)
        A = A.repeat(self.d_inner, 1)
        A_log = torch.log(A)
        
        if self.use_hierarchical:
            cfg = MAMBA_HIER_CONFIG
            scale_early = cfg['A_SCALE_EARLY']
            scale_late = cfg['A_SCALE_LATE']
            self.a_scale = scale_early * ((scale_late / scale_early) ** self.layer_progress)
            A_log = A_log * self.a_scale
        else:
            self.a_scale = 1.0
        
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through Mamba layer."""
        x_norm = self.norm(x)

        if self.use_cuda and x_norm.dtype in (torch.float16, torch.bfloat16):
            with torch.autocast(device_type="cuda", enabled=False):
                y = self.mamba(x_norm.float())
            y = y.to(x_norm.dtype)
        else:
            y = self.mamba(x_norm)  # or pytorch fallback
        
        return x + self.dropout(y)
    
    def _forward_pytorch(self, x_norm: torch.Tensor) -> torch.Tensor:
        """Pure PyTorch forward pass.
        
        TODO: This method is not called by forward() in the current code path.
        The CUDA path (self.mamba) is used when use_cuda=True; when use_cuda=False,
        forward() still calls self.mamba which is not defined. This is a known
        limitation of the PyTorch fallback — thesis experiments use CUDA kernels.
        """
        batch, length, _ = x_norm.shape
        
        xz = self.in_proj(x_norm)
        x_ssm, z = xz.chunk(2, dim=-1)
        
        x_conv = x_ssm.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :length]
        x_conv = x_conv.transpose(1, 2)
        x_conv = F.silu(x_conv)
        
        x_dbl = self.x_proj(x_conv)
        delta_pre, B, C = torch.split(
            x_dbl,
            [self.dt_rank, self.d_state, self.d_state],
            dim=-1
        )
        
        delta = self.dt_proj(delta_pre)
        delta = F.softplus(delta)
        
        A = -torch.exp(self.A_log)
        
        y = selective_scan_ref(
            u=x_conv, delta=delta, A=A, B=B, C=C, D=self.D,
        )
        
        y = y * F.silu(z)
        return self.out_proj(y)


class MambaEncoder(nn.Module):
    """Stack of Mamba-1 layers with optional non-uniform layer-wise init."""
    
    def __init__(
        self,
        d_model: int = 256,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        n_layers: int = 12,
        dropout: float = 0.1,
        use_hierarchical: bool = True,
        use_cuda_kernels: bool = True,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.d_state = d_state
        self.n_layers = n_layers
        
        actual_cuda = use_cuda_kernels and _HAS_MAMBA_SSM
        backend_name = "CUDA (mamba_ssm)" if actual_cuda else "PyTorch (reference)"
        
        print(f"\nMamba Encoder Configuration:")
        print(f"  d_model={d_model}, d_state={d_state}, d_conv={d_conv}, expand={expand}")
        print(f"  n_layers={n_layers}, hierarchical={use_hierarchical}")
        print(f"  d_inner={int(expand * d_model)}")
        print(f"  Backend: {backend_name}")
        
        if use_hierarchical:
            print(f"\nNon-uniform Init (dt_bias + A_log scaling):")
            print(f"  {'Layer':<6} {'Type':<11} {'A_scale':<10} {'dt_mean':<12}")
            print(f"  {'-'*6} {'-'*11} {'-'*10} {'-'*12}")
        
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            layer = MambaLayer(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout,
                layer_idx=i,
                n_layers=n_layers,
                use_hierarchical=use_hierarchical,
                use_cuda_kernels=use_cuda_kernels,
            )
            self.layers.append(layer)
            
            if use_hierarchical:
                print(f"  {i:<6} {layer.layer_type:<11} {layer.a_scale:<10.3f} {layer.dt_init_mean:<12.4f}")
        
        self.final_norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.final_norm(x)


# ============================================================================
# MAMBA-3 ENCODER (official mamba_ssm Triton kernels, requires sm_75+ GPU)
# ============================================================================

class Mamba3Layer(nn.Module):
    """
    Mamba-3 layer for ASR encoder.
    
    Architecture:
        Input -> RMSNorm -> Mamba3 Block -> Residual + Dropout -> Output
    
    Key differences from MambaLayer (Mamba-1):
        - Uses RMSNorm instead of LayerNorm (Mamba-3 design)
        - No causal convolution (implicit in trapezoidal recurrence)
        - Complex-valued SSM with RoPE for state tracking
        - BCNorm (QKNorm) for training stability
        - dt_bias is a direct parameter (shape: nheads), not dt_proj.bias (shape: d_inner)
    
    Non-uniform init (optional, use_hierarchical=True):
        Modifies dt_bias based on layer depth to create timescale diversity.
        Same principle as Mamba-1, adapted for Mamba-3's per-head parameterization.
        Flag name kept for compatibility; see module docstring for terminology.
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        headdim: int = 64,
        is_mimo: bool = False,
        mimo_rank: int = 4,
        dropout: float = 0.1,
        layer_idx: int = 0,
        n_layers: int = 1,
        use_hierarchical: bool = False,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.d_state = d_state
        self.headdim = headdim
        self.layer_idx = layer_idx
        self.n_layers = n_layers
        self.use_hierarchical = use_hierarchical
        
        # Layer position [0.0 = first, 1.0 = last]
        self.layer_progress = layer_idx / max(1, n_layers - 1)
        
        # Layer type label for logging (labels are historical; see module docstring)
        if self.layer_progress < 0.33:
            self.layer_type = "acoustic"
        elif self.layer_progress < 0.66:
            self.layer_type = "phonetic"
        else:
            self.layer_type = "linguistic"
        
        # Pre-norm (RMSNorm for Mamba-3)
        self.norm = nn.RMSNorm(d_model)
        
        # Mamba-3 block (official Triton kernels)
        self.mamba3 = Mamba3CUDA(
            d_model=d_model,
            d_state=d_state,
            headdim=headdim,
            is_mimo=is_mimo,
            mimo_rank=mimo_rank if is_mimo else 1,
            is_outproj_norm=False,
            dtype=dtype,
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Hierarchical dt_bias init
        self.dt_init_mean = 0.0
        if use_hierarchical:
            self._apply_hierarchical_init()
    
    def _apply_hierarchical_init(self):
        """
        Apply non-uniform dt_bias initialization for Mamba-3.
        
        Mamba-3 dt_bias is shape (nheads,) — one value per head.
        Initialized based on layer depth: early layers get smaller dt
        (faster timescale), late layers get larger dt (slower timescale).
        Method name kept for checkpoint compatibility.
        """
        cfg = MAMBA_HIER_CONFIG
        progress = self.layer_progress
        
        with torch.no_grad():
            dt_min = cfg['DT_MIN_EARLY'] + (cfg['DT_MIN_LATE'] - cfg['DT_MIN_EARLY']) * progress
            dt_max = cfg['DT_MAX_EARLY'] + (cfg['DT_MAX_LATE'] - cfg['DT_MAX_EARLY']) * progress
            
            nheads = self.mamba3.dt_bias.shape[0]
            
            # Sample dt values from log-uniform distribution
            dt = torch.exp(
                torch.rand(nheads, device=self.mamba3.dt_bias.device)
                * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
            ).clamp(min=1e-4)
            
            # dt_bias is added to dt before softplus in Mamba-3 forward,
            # so we store inverse softplus: inv_softplus(x) = x + log(1 - exp(-x))
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            self.mamba3.dt_bias.copy_(inv_dt)
            
            self.dt_init_mean = dt.mean().item()
            
            if self.layer_idx == 0:
                print(f"\n  [Mamba-3 non-uniform init] Modifying dt_bias per layer depth")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through Mamba-3 layer."""
        x_norm = self.norm(x)
        
        # Cast to model dtype for Triton kernels, then cast back
        input_dtype = x_norm.dtype
        if hasattr(self.mamba3, 'dtype') and self.mamba3.dtype is not None:
            x_norm = x_norm.to(self.mamba3.dtype)
        
        y = self.mamba3(x_norm)
        y = y.to(input_dtype)
        
        return x + self.dropout(y)


class Mamba3Encoder(nn.Module):
    """Stack of Mamba-3 layers for ASR encoding."""
    
    def __init__(
        self,
        d_model: int = 256,
        d_state: int = 64,
        headdim: int = 64,
        is_mimo: bool = False,
        mimo_rank: int = 4,
        n_layers: int = 12,
        dropout: float = 0.1,
        use_hierarchical: bool = False,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.d_state = d_state
        self.n_layers = n_layers
        
        # Detect dtype
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability()
            dtype = torch.bfloat16 if cap[0] >= 8 else torch.float16
        else:
            dtype = torch.float32
        
        mimo_tag = f", MIMO R={mimo_rank}" if is_mimo else ", SISO"
        print(f"\nMamba-3 Encoder Configuration:")
        print(f"  d_model={d_model}, d_state={d_state}, headdim={headdim}{mimo_tag}")
        print(f"  n_layers={n_layers}, hierarchical={use_hierarchical}")
        print(f"  dtype={dtype}, backend=official mamba_ssm (Triton)")
        
        if use_hierarchical:
            print(f"\n  {'Layer':<6} {'Type':<11} {'dt_mean':<12}")
            print(f"  {'-'*6} {'-'*11} {'-'*12}")
        
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            layer = Mamba3Layer(
                d_model=d_model,
                d_state=d_state,
                headdim=headdim,
                is_mimo=is_mimo,
                mimo_rank=mimo_rank,
                dropout=dropout,
                layer_idx=i,
                n_layers=n_layers,
                use_hierarchical=use_hierarchical,
                dtype=dtype,
            )
            self.layers.append(layer)
            
            if use_hierarchical:
                print(f"  {i:<6} {layer.layer_type:<11} {layer.dt_init_mean:<12.6f}")
        
        self.final_norm = nn.RMSNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.final_norm(x)


def selective_scan_ref(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor = None,
) -> torch.Tensor:
    """
    Reference implementation of Mamba's selective scan (pure PyTorch fallback).
    
    Used when mamba_ssm CUDA kernels are not available.
    """
    batch, length, d_inner = u.shape
    d_state = A.shape[1]
    
    delta_expanded = delta.unsqueeze(-1)
    A_expanded = A.unsqueeze(0).unsqueeze(0)
    
    deltaA = delta_expanded * A_expanded
    A_bar = torch.exp(deltaA)
    
    B_expanded = B.unsqueeze(2)
    deltaB = delta_expanded * B_expanded
    
    u_expanded = u.unsqueeze(-1)
    
    h = torch.zeros(batch, d_inner, d_state, device=u.device, dtype=u.dtype)
    outputs = []
    
    for t in range(length):
        h = A_bar[:, t, :, :] * h + deltaB[:, t, :, :] * u_expanded[:, t, :, :]
        y_t = torch.einsum('bn,bdn->bd', C[:, t, :], h)
        outputs.append(y_t)
    
    y = torch.stack(outputs, dim=1)
    
    if D is not None:
        y = y + u * D.unsqueeze(0).unsqueeze(0)
    
    return y



# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class ConvSubsample(nn.Module):
    """4x subsampling via two strided convolutions (from v4.0)."""
    
    def __init__(self, in_channels: int = 80, out_channels: int = 256):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels // 2, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv1d(out_channels // 2, out_channels, kernel_size=5, stride=2, padding=2)
    
    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, time, features]
            lengths: [batch]
        
        Returns:
            x: [batch, time//4, out_channels]
            lengths: [batch] reduced by factor of 4
        """
        x = x.transpose(1, 2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.transpose(1, 2)
        lengths = torch.div(lengths, 4, rounding_mode="floor")
        return x, lengths


class HierarchicalSelectiveSSMLayer(nn.Module):
    """
    Selective SSM Layer with optional non-uniform layer-wise init and gating.
    
    Class name kept for checkpoint compatibility.  "Hierarchical" here refers
    to non-uniform initialization of decay/timescale parameters across layers
    (see module-level docstring for full terminology note).
    """
    
    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        layer_idx: int = 0,
        n_layers: int = 1,
        use_hierarchical: bool = True,
        use_gating: bool = True,
        enable_diagnostics: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.layer_idx = layer_idx
        self.n_layers = n_layers
        self.use_hierarchical = use_hierarchical
        self.use_gating = use_gating
        self.enable_diagnostics = enable_diagnostics
        
        # Normalization
        self.norm = nn.LayerNorm(d_model)
        
        # Input projection
        if use_gating:
            self.in_proj = nn.Linear(d_model, d_model * 2)
        else:
            self.in_proj = nn.Linear(d_model, d_model)
        
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)

        self.dwconv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=5,
            padding=2,
            groups=d_model
        )

        # Projections
        self.Wa = nn.Linear(d_model, d_model, bias=True)  
        self.Wb = nn.Linear(d_model, d_model, bias=True)  
        self.Wc = nn.Linear(d_model, d_model, bias=True)  
        self.Wd = nn.Linear(d_model, d_model, bias=True) 
        
        # Parameters
        self.a0 = nn.Parameter(torch.zeros(d_model))
        self.b0 = nn.Parameter(torch.zeros(d_model))
        self.c0 = nn.Parameter(torch.zeros(d_model))
        self.d0 = nn.Parameter(torch.zeros(d_model))

        # Hierarchical initialization
        if use_hierarchical:
            with torch.no_grad():
                # Compute layer position (0.0 = early, 1.0 = late)
                # Safe division even if n_layers=1
                denom = max(1, n_layers - 1)
                layer_progress = layer_idx / denom
                
                # Load constants
                hc = HIER_CONFIG
                
                # a0: Exponential decay schedule
                a_mean = hc['A0_EARLY'] * ((hc['A0_LATE'] / hc['A0_EARLY']) ** layer_progress)
                
                # b0: Exponential decay schedule
                b_mean = hc['B0_EARLY'] * ((hc['B0_LATE'] / hc['B0_EARLY']) ** layer_progress)
                
                # c0: Linear schedule
                c_mean = hc['C0_EARLY'] + (hc['C0_LATE'] - hc['C0_EARLY']) * layer_progress
                
                # Initialization with random variance
                a_init = torch.normal(mean=a_mean, std=0.08, size=(d_model,))
                self.a0.copy_(torch.clamp(a_init, -0.95, 0.95))
                
                b_init = torch.normal(mean=b_mean, std=0.05, size=(d_model,))
                self.b0.copy_(torch.clamp(b_init, 0.01, 0.95))
                
                c_init = torch.normal(mean=c_mean, std=0.05, size=(d_model,))
                self.c0.copy_(torch.clamp(c_init, -0.95, 0.95))
                
                # d0: Skip connection
                self.d0.copy_(torch.normal(0.0, std=0.02, size=(d_model,)))
                
                # Determine Layer Type dynamically based on progress
                if layer_progress < 0.33:
                    layer_type = "acoustic"
                elif layer_progress < 0.66:
                    layer_type = "phonetic"
                else:
                    layer_type = "linguistic"
                
                self.layer_type = layer_type
                
                # Logging actual values
                a_str = f"{self.a0.mean():.3f}±{self.a0.std():.3f}"
                b_str = f"{self.b0.mean():.3f}±{self.b0.std():.3f}"
                c_str = f"{self.c0.mean():.3f}±{self.c0.std():.3f}"
                print(f"  {layer_idx:<6} {layer_type:<11} {a_str:<20} {b_str:<20} {c_str:<20}")
                
                # Store for diagnostics
                self.init_params = {
                    'a_mean': a_mean, 'b_mean': b_mean, 'c_mean': c_mean,
                    'layer_type': layer_type,
                }
                               
    def save_diagnostics(self, save_dir: str):
        """Save diagnostic statistics to file."""
        if hasattr(self, '_diag_stats'):
            os.makedirs(save_dir, exist_ok=True)
            filepath = os.path.join(save_dir, f'layer_{self.layer_idx}_diagnostics.json')
            
            # Safe computation of stats
            def safe_stat(data, func):
                return func(data) if len(data) > 0 else 0.0

            summary = {
                'layer_idx': self.layer_idx,
                'layer_type': getattr(self, 'layer_type', 'unknown'),
                'stats_count': len(self._diag_stats['at_mean'])
            }
            
            # Add stats if available
            if summary['stats_count'] > 0:
                summary.update({
                    'at_mean': np.mean(self._diag_stats['at_mean']),
                    'bt_mean': np.mean(self._diag_stats['bt_mean']),
                    'ct_mean': np.mean(self._diag_stats['ct_mean']),
                    'state_norm': np.mean(self._diag_stats['state_norm']),
                })

            with open(filepath, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"Saved diagnostics for layer {self.layer_idx} to {filepath}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        
        x_norm = self.norm(x)
        
        if self.use_gating:
            xz = self.in_proj(x_norm)
            x_proj, z = xz.chunk(2, dim=-1)
        else:
            x_proj = self.in_proj(x_norm)
        
        x_conv = self.dwconv(x_proj.transpose(1, 2)).transpose(1, 2)
        
        # Bounded parameters
        at = torch.tanh(self.a0 + self.Wa(x_conv))
        bt = torch.sigmoid(self.b0 + self.Wb(x_conv))
        ct = torch.sigmoid(self.c0 + self.Wc(x_conv))
        dt = self.Wd(x_conv) + self.d0
        
        # SSM Recurrence
        s = torch.zeros(B, self.d_model, device=x.device, dtype=x.dtype)
        ys = []
        
        for t in range(T):
            xt = x_proj[:, t, :]
            s = at[:, t, :] * s + bt[:, t, :] * xt
            yt = ct[:, t, :] * s + dt[:, t, :] * xt
            ys.append(yt)
        
        y = torch.stack(ys, dim=1)

        # Diagnostics
        if self.enable_diagnostics and self.training:
            if not hasattr(self, '_diag_step'): self._diag_step = 0
            self._diag_step += 1
            
            if self._diag_step % 100 == 0:
                with torch.no_grad():
                    if not hasattr(self, '_diag_stats'):
                        self._diag_stats = {'at_mean': [], 'bt_mean': [], 'ct_mean': [], 'state_norm': []}
                    self._diag_stats['at_mean'].append(at.mean().item())
                    self._diag_stats['bt_mean'].append(bt.mean().item())
                    self._diag_stats['ct_mean'].append(ct.mean().item())
                    self._diag_stats['state_norm'].append(s.norm(dim=-1).mean().item())
        
        if self.use_gating:
            gate = torch.sigmoid(z)
            y = y * gate
        
        y = self.out_proj(y)
        return x + self.dropout(y)


class SSMEncoder(nn.Module):
    """Stack of SSM layers."""
    
    def __init__(
        self, 
        d_model: int = 256, 
        n_layers: int = 6, 
        dropout: float = 0.1,
        use_hierarchical: bool = True,
        use_gating: bool = True,  
        enable_diagnostics: bool = False,
    ):
        super().__init__()

        print(f"\nSSM Encoder Configuration: d_model={d_model}, layers={n_layers}, " 
              f"hierarchical={use_hierarchical}, gating={use_gating}")
        
        if use_hierarchical:
            # Calculate dynamic preview values based on config
            hc = HIER_CONFIG
            
            # Calculate theoretical mean 'a' (tanh of parameter) for display
            # Note: tanh(0.85) = 0.69, tanh(0.15) = 0.15
            # We show the raw parameter value to match the table below
            a_early = hc['A0_EARLY']
            a_mid = hc['A0_EARLY'] * ((hc['A0_LATE'] / hc['A0_EARLY']) ** 0.5)
            a_late = hc['A0_LATE']

            print(f"\nNon-uniform Init (Target Means):")
            print(f"   Early layers:  a0≈{a_early:.2f} (lower initial decay in forward schedule)")
            print(f"   Mid layers:    a0≈{a_mid:.2f}")
            print(f"   Late layers:   a0≈{a_late:.2f} (higher initial decay in forward schedule)\n")
            
            print(f"  {'Layer':<6} {'Type':<11} {'a0 (mean±std)':<20} {'b0 (mean±std)':<20} {'c0 (mean±std)':<20}")
            print(f"  {'-'*6} {'-'*11} {'-'*20} {'-'*20} {'-'*20}")
        else:
            print(f"\nUniform Init:")
            print(f"  All layers: Standard initialization")
        
        self.layers = nn.ModuleList([
            HierarchicalSelectiveSSMLayer(
                d_model=d_model,
                dropout=dropout,
                layer_idx=i,
                n_layers=n_layers,
                use_hierarchical=use_hierarchical,
                use_gating=use_gating,
                enable_diagnostics=enable_diagnostics,
            )
            for i in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.final_norm(x)



class ASRModel(nn.Module):
    """
    Complete ASR model with encoder selection.
    
    Supports:
    - 'mamba': Mamba-1 architecture with optional non-uniform init
    - 'mamba3': Mamba-3 architecture (Triton kernels, requires sm_75+ GPU)
    - 'sssm': Original S-SSSM from thesis (for comparison)
    """
    
    def __init__(
        self,
        n_classes: int,
        d_model: int = 256,
        n_layers: int = 6,
        dropout: float = 0.1,
        use_hierarchical: bool = True,
        use_gating: bool = True,
        encoder_type: str = "mamba",
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        enable_diagnostics: bool = False,
        use_cuda_kernels: bool = True,
        **kwargs,  # Mamba-3 params: headdim, is_mimo, mimo_rank
    ):
        super().__init__()
        
        self.encoder_type = encoder_type
        
        # Subsampling (shared between encoders)
        self.subsample = ConvSubsample(N_MELS, d_model)
        
        # Encoder selection
        if encoder_type.lower() == "mamba":
            self.encoder = MambaEncoder(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                n_layers=n_layers,
                dropout=dropout,
                use_hierarchical=use_hierarchical,
                use_cuda_kernels=use_cuda_kernels,
            )
        elif encoder_type.lower() == "mamba3":
            if not _HAS_MAMBA3:
                raise RuntimeError(
                    "mamba_ssm.Mamba3 not available. Install from GitHub:\n"
                    '  TORCH_CUDA_ARCH_LIST="8.0" pip install --force-reinstall --no-deps \\\n'
                    "    git+https://github.com/state-spaces/mamba.git "
                    "--no-build-isolation --break-system-packages"
                )
            self.encoder = Mamba3Encoder(
                d_model=d_model,
                d_state=d_state,
                headdim=kwargs.get('headdim', 64),
                is_mimo=kwargs.get('is_mimo', False),
                mimo_rank=kwargs.get('mimo_rank', 4),
                n_layers=n_layers,
                dropout=dropout,
                use_hierarchical=use_hierarchical,
            )
        elif encoder_type.lower() == "sssm":
            self.encoder = SSMEncoder(
                d_model=d_model,
                n_layers=n_layers,
                dropout=dropout,
                use_hierarchical=use_hierarchical,
                use_gating=use_gating,
                enable_diagnostics=enable_diagnostics,
            )
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")
        
        # Output head
        self.output_head = nn.Linear(d_model, n_classes)
        
        # Parameter count
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"\nModel Parameters:")
        print(f"  Encoder type: {encoder_type.upper()}")
        print(f"  Total: {total_params:,} ({total_params/1e6:.2f}M)")
        print(f"  Trainable: {trainable_params:,}")
    
    def forward(
        self,
        features: torch.Tensor,
        feature_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Subsample
        x, lengths = self.subsample(features, feature_lengths)
        
        # Encode
        x = self.encoder(x)
        
        # Classify
        logits = self.output_head(x)
        
        return logits, lengths


# ============================================================================
# DECODING
# ============================================================================

def ctc_greedy_decode(
    logits: torch.Tensor,
    lengths: torch.Tensor
) -> List[List[int]]:
    """Greedy CTC decoding."""
    pred = logits.argmax(dim=-1)
    hyps = []
    
    for i, row in enumerate(pred):
        L = int(lengths[i].item())
        row = row[:L]
        
        # CTC collapse
        last = -1
        seq = []
        for p in row.tolist():
            if p != last and p != BLANK_TOKEN and p < len(VOCAB_CHARS):
                seq.append(p)
            last = p
        
        hyps.append(seq)
    
    return hyps


# ============================================================================
# LIGHTNING MODULE
# ============================================================================

class LitASR(L.LightningModule):
    """Lightning module for training."""
    
    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 6,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        warmup_steps: int = 500,
        weight_decay: float = 0.05,
        use_speed_perturb: bool = False,
        use_hierarchical: bool = True,
        use_gating: bool = True,
        enable_diagnostics: bool = False,
        # Mamba-1 parameters
        encoder_type: str = "mamba",
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        use_cuda_kernels: bool = True,
        # Mamba-3 parameters
        headdim: int = 64,
        is_mimo: bool = False,
        mimo_rank: int = 4,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        print("\nTraining Config:")
        print(f"   Learning rate: {learning_rate}")
        print(f"   Warmup steps: {warmup_steps}")
        print(f"   Weight decay: {weight_decay}")
        print(f"   SpeedPerturbation: {use_speed_perturb}")
        
        # Model with encoder selection
        self.model = ASRModel(
            n_classes=len(VOCAB_CHARS) + 1,
            d_model=d_model,
            n_layers=n_layers,
            dropout=dropout,
            use_hierarchical=use_hierarchical,
            use_gating=use_gating,
            encoder_type=encoder_type,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            enable_diagnostics=enable_diagnostics,
            use_cuda_kernels=use_cuda_kernels,
            # Mamba-3 kwargs (only used when encoder_type='mamba3')
            headdim=headdim,
            is_mimo=is_mimo,
            mimo_rank=mimo_rank,
        )
        
        # Loss
        self.ctc_loss = nn.CTCLoss(blank=BLANK_TOKEN, zero_infinity=True)
        
        # Tracking
        self.train_step_outputs = []
        self.val_step_outputs = {0: [], 1: []} 
        self.val_loss_outputs = {0: [], 1: []}
        self.val_wer_outputs = {0: [], 1: []} 
    
    def forward(self, features, feature_lengths):
        return self.model(features, feature_lengths)
    
    def training_step(self, batch, batch_idx):
        features = batch['features']
        feature_lengths = batch['feature_lengths']
        text_ids = batch['text_ids']
        text_lengths = batch['text_lengths']
        
        # Forward pass
        logits, lengths = self.model(features, feature_lengths)
        
        # CTC loss expects [time, batch, classes]
        log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
        
        # Flatten targets for CTC
        targets = torch.cat([
            text_ids[i, :text_lengths[i]]
            for i in range(text_ids.size(0))
        ]).long()
        
    
        # Compute loss (cast to float32 because ctc_loss_cuda doesn't support bfloat16)
        loss = self.ctc_loss(log_probs.float(), targets, lengths, text_lengths)
        
        # Check for numerical issues
        if not torch.isfinite(loss):
            print(f"\nWARNING: Non-finite loss detected at step {batch_idx}")
            print(f"  Loss: {loss.item()}")
            print(f"  Logits: min={logits.min():.3f}, max={logits.max():.3f}")
            return None

        batch_size = features.shape[0]
        self.log(
            'train_loss', 
            loss, 
            prog_bar=True, 
            on_step=True, 
            on_epoch=True, 
            batch_size=batch_size
        )
        self.train_step_outputs.append(loss.detach())
        
        return loss
    
    def on_train_epoch_end(self):
        if torch.cuda.is_available():
            # Get peak memory since last reset
            peak_memory = torch.cuda.max_memory_allocated(0) / (1024 ** 3)
            self.log('gpu_memory_peak_gb', peak_memory, on_epoch=True, 
                    prog_bar=False, logger=True, sync_dist=True)
            # Reset peak memory tracker for next epoch
            torch.cuda.reset_peak_memory_stats(0)
        self.train_step_outputs.clear()
    
    def _shared_eval(self, batch, batch_idx, dataloader_idx, prefix):
        # Determine suffix based on loader index
        suffix = "clean" if dataloader_idx == 0 else "other"
        
        features = batch['features']
        feature_lengths = batch['feature_lengths']
        text_ids = batch['text_ids']
        text_lengths = batch['text_lengths']
        
        batch_size = features.shape[0]

        # Forward pass
        logits, lengths = self.model(features, feature_lengths)

        # Calculate loss
        log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
        targets = torch.cat([
            text_ids[i, :text_lengths[i]] for i in range(text_ids.size(0))
        ]).long()
        loss = self.ctc_loss(log_probs.float(), targets, lengths, text_lengths)
        
        # Decode
        hyps = ctc_greedy_decode(logits, lengths)
        
        # WER Calculation
        wers = []
        for i, hyp in enumerate(hyps):
            ref = ids_to_text(text_ids[i, :text_lengths[i]].tolist())
            hyp_text = ids_to_text(hyp)
            wers.append(jiwer_wer(ref, hyp_text))
            
            # Store detailed results (Only during validation, Only clean set)
            if prefix == 'val' and dataloader_idx == 0:
                duration = feature_lengths[i].item() / SAMPLE_RATE 
                result = {
                    'reference': ref, 'hypothesis': hyp_text,
                    'wer': wers[-1] * 100, 'duration': duration,
                    'audio_length': feature_lengths[i].item()
                }
                if not hasattr(self, 'detailed_results'): self.detailed_results = []
                self.detailed_results.append(result)
            
        mean_wer = float(np.mean(wers))
        
        # --- DYNAMIC LOGGING (Uses 'val' or 'test' prefix) ---
        self.log(
            f'{prefix}_wer_{suffix}', 
            mean_wer, 
            prog_bar=True, on_step=False, on_epoch=True, 
            add_dataloader_idx=False, batch_size=batch_size
        )
        
        # Only track outputs for Epoch End aggregation during VALIDATION
        if prefix == 'val':
            if dataloader_idx not in self.val_wer_outputs: self.val_wer_outputs[dataloader_idx] = []
            if dataloader_idx not in self.val_loss_outputs: self.val_loss_outputs[dataloader_idx] = []
            self.val_wer_outputs[dataloader_idx].append(mean_wer)
            self.val_loss_outputs[dataloader_idx].append(loss.detach())
        
        return loss

    # --- WRAPPER METHODS ---
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        return self._shared_eval(batch, batch_idx, dataloader_idx, prefix="val")

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self._shared_eval(batch, batch_idx, dataloader_idx, prefix="test")
    
    def on_validation_epoch_end(self):
        # Calculate Main Metric for Early Stopping (Average of both or just clean)
        if hasattr(self, 'val_wer_outputs'):
            all_wers = []
            if 0 in self.val_wer_outputs: all_wers.extend(self.val_wer_outputs[0])
            if 1 in self.val_wer_outputs: all_wers.extend(self.val_wer_outputs[1])
            
            if all_wers:
                avg_wer = np.mean(all_wers)
                self.log('val_wer', avg_wer, prog_bar=True)
                self.log('val_wer_epoch', avg_wer)
        
        if hasattr(self, 'val_loss_outputs'):
            all_losses = []
            if 0 in self.val_loss_outputs: all_losses.extend(self.val_loss_outputs[0])
            if 1 in self.val_loss_outputs: all_losses.extend(self.val_loss_outputs[1])
            if all_losses:
                avg_loss = torch.stack(all_losses).mean()
                self.log('val_loss_epoch', avg_loss)

        self.val_wer_outputs = {0: [], 1: []}
        self.val_loss_outputs = {0: [], 1: []}

        # Save detailed results
        if hasattr(self, 'detailed_results') and self.detailed_results:
            if not hasattr(self, 'final_detailed_results'):
                self.final_detailed_results = self.detailed_results.copy()
            self.detailed_results.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            betas=(0.9, 0.98),
            weight_decay=self.hparams.weight_decay
        )
        
        # Learning rate schedule with warmup and cosine decay
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = self.hparams.warmup_steps
        
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
                return 0.5 * (1.0 + math.cos(math.pi * progress))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
            }
        }



# ============================================================================
# CALLBACKS
# ============================================================================

class HierarchyFreezeCallback(Callback):
    """
    Freezes the non-uniform init parameters (a0, b0) for the first N epochs
    to force the model to utilise the imposed timescale diversity, then
    unfreezes them to allow fine-tuning.  Class name kept for compatibility.
    """
    def __init__(self, unfreeze_epoch: int = 5):
        super().__init__()
        self.unfreeze_epoch = unfreeze_epoch
        self.frozen = False

    def on_train_start(self, trainer, pl_module):
        # Freeze a0 and b0 in all layers
        print(f"\n[HierarchyFreeze] Freezing State Dynamics (a0, b0) until epoch {self.unfreeze_epoch}...")
        self._set_grads(pl_module, requires_grad=False)
        self.frozen = True

    def on_train_epoch_start(self, trainer, pl_module):
        current_epoch = trainer.current_epoch
        
        if self.frozen and current_epoch >= self.unfreeze_epoch:
            print(f"\n[HierarchyFreeze] Unfreezing State Dynamics at Epoch {current_epoch}!")
            self._set_grads(pl_module, requires_grad=True)
            self.frozen = False

    def _set_grads(self, pl_module, requires_grad: bool):
        # Navigate to the encoders. 
        # Note: Handle both standard and BiDirectional cases
        encoders = []
        if hasattr(pl_module.model.encoder, 'layers'): # Standard
            encoders.append(pl_module.model.encoder)
        elif hasattr(pl_module.model.encoder, 'fwd_encoder'): # BiDirectional
            encoders.append(pl_module.model.encoder.fwd_encoder)
            encoders.append(pl_module.model.encoder.bwd_encoder)
            
        count = 0
        for enc in encoders:
            for layer in enc.layers:
                # Freeze a0 (decay) and b0 (input scale)
                # We allow c0 (output) to learn, to adapt the mix
                if hasattr(layer, 'a0'): layer.a0.requires_grad = requires_grad
                if hasattr(layer, 'b0'): layer.b0.requires_grad = requires_grad
                count += 2
        
        status = "Unfrozen" if requires_grad else "Frozen"
        print(f"[HierarchyFreeze] {status} {count} parameter tensors.")


class CleanProgressCallback(Callback):
    """Clean progress reporting without special characters."""
    
    def __init__(self, print_every_n_batches: int = 50):
        super().__init__()
        self.print_every_n_batches = print_every_n_batches
        self.epoch_start_time = None
        self.fit_start_time = None
        self.history = {'gpu_memory_allocated': [], 'gpu_memory_reserved': []}
        
    def on_fit_start(self, trainer, pl_module):
        self.fit_start_time = time.time()
        print("\n\nTRAINING STARTED\n")
    
    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_start_time = time.time()
        epoch = trainer.current_epoch + 1
        max_epochs = trainer.max_epochs
        print("")
        print(f"{'-'*80}")
        print(f"Epoch {epoch}/{max_epochs}")
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if (batch_idx + 1) % self.print_every_n_batches == 0:
            total_batches = trainer.num_training_batches
            progress_pct = 100.0 * (batch_idx + 1) / total_batches
            
            # Get current loss
            current_loss = trainer.callback_metrics.get("train_loss", float("nan"))
            
            # Calculate ETA
            elapsed = time.time() - self.epoch_start_time
            batches_done = batch_idx + 1
            batches_left = total_batches - batches_done
            
            if batches_done > 0:
                time_per_batch = elapsed / batches_done
                eta_seconds = int(time_per_batch * batches_left)
                eta_min = eta_seconds // 60
                eta_sec = eta_seconds % 60
                
                line = (f"  Batch {batches_done:4d}/{total_batches} "
                        f"({progress_pct:5.1f}%) | "
                        f"Loss: {current_loss:6.4f} | "
                        f"ETA: {eta_min:3d}m {eta_sec:2d}s")
                
                print(f"\r{line}{' ' * 10}", end="", flush=True)
    
    def on_validation_epoch_end(self, trainer, pl_module):
            if trainer.sanity_checking:
                return
            
            metrics = trainer.callback_metrics
            print()

            # 1. Get metrics from the correct epoch-level logs
            train_loss = float(metrics.get("train_loss_epoch", float("nan")))
            val_loss = float(metrics.get("val_loss_epoch", float("nan")))
            
            # 2. Calculate val_wer directly from pl_module's validation outputs
            val_wer = float("nan")

            if hasattr(pl_module, 'val_wer_outputs') and pl_module.val_wer_outputs:
                all_wers = []
                if 0 in pl_module.val_wer_outputs: 
                    all_wers.extend(pl_module.val_wer_outputs[0])
                if 1 in pl_module.val_wer_outputs: 
                    all_wers.extend(pl_module.val_wer_outputs[1])
                if all_wers:
                    val_wer = float(np.mean(all_wers))
            
            # 3. Get current_lr from the trainer's optimizer
            current_lr = trainer.optimizers[0].param_groups[0]['lr']
            
            epoch_time = time.time() - self.epoch_start_time
            total_time = time.time() - self.fit_start_time
            
            epoch_min = int(epoch_time // 60)
            epoch_sec = int(epoch_time % 60)
            total_min = int(total_time // 60)
            
            print("")
            print(f"  Train Loss:  {train_loss:.4f} | " 
            f"Val Loss:  {val_loss:.4f} | " 
            f"Val WER:  {val_wer:.4f} ({(val_wer * 100):.2f}%)| "
            f"Epoch Time:  {epoch_min}m {epoch_sec}s | "
            f"Total Time:  {total_min}m")
            print(f"  Current LR:        {current_lr:.6f}")
            if torch.cuda.is_available():
                # Log and reset peak memory for the plot
                peak_alloc = torch.cuda.max_memory_allocated() / 1024**3
                peak_res = torch.cuda.max_memory_reserved() / 1024**3
                self.history['gpu_memory_allocated'].append(peak_alloc)
                self.history['gpu_memory_reserved'].append(peak_res)
                torch.cuda.reset_peak_memory_stats(trainer.strategy.root_device) # Reset for next epoch
                print(f"  GPU Memory (Peak): {peak_alloc:.2f} GB")
            
            print(f"{'-'*80}")
    
    def on_fit_end(self, trainer, pl_module):
        total_time = time.time() - self.fit_start_time
        total_hours = int(total_time // 3600)
        total_min = int((total_time % 3600) // 60)
        

        print("Training Complete.")
        print(f"Total training time: {total_hours}h {total_min}m")

# ============================================================================
# VISUALIZATION HELPER
# ============================================================================

def get_param_counts(model: nn.Module) -> Dict[str, int]:
    """Get parameter counts for visualization."""
    try:
        counts = {
            'Convolutional Subsample': sum(p.numel() for p in model.model.subsample.parameters()),
            'SSM Encoder Layers': sum(p.numel() for p in model.model.encoder.parameters()),
            'Output Head': sum(p.numel() for p in model.model.output_head.parameters()),
        }
        return counts
    except Exception as e:
        print(f"Warning: Could not get param counts. {e}")
        return {}



# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train(config: Config):
    """Main training function."""

    # Setup checkpointing & logging to text file
    exp_name = config.get_exp_name()
    checkpoint_dir = f"checkpoints_{exp_name}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Set up safe logging (replaces DualLogger)
    log_file = os.path.join(checkpoint_dir, f"0utput_exp-{exp_name}.txt")
    setup_logging(log_file) 
    override_print_with_logger()
    print(f"Logging output to: {log_file}")

    # Lightning's CSV logger for metrics 
    csv_logger = CSVLogger(save_dir="logs", name=exp_name)

    # Set random seeds for reproducibility
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    
    if not _HAS_HF:
        raise RuntimeError("Install datasets library: pip install datasets")
    
    # Print configuration
    print("")
    print(f"Experiment Config: {config.get_exp_name()}")

    for key, value in asdict(config).items():
        print(f"  {key:.<30} {value}")

    print("")

    # Load datasets
    print("Loading datasets from disk...")
    
    clean_path = os.path.join(config.data_path, "clean")
    other_path = os.path.join(config.data_path, "other")
    
    if not os.path.isdir(clean_path):
        raise FileNotFoundError(f"Could not find clean dataset folder: {clean_path}")
    if not os.path.isdir(other_path):
        raise FileNotFoundError(f"Could not find other dataset folder: {other_path}")
    
    ds_dict_clean = load_from_disk(clean_path)   # hub_data/librispeech/clean
    ds_dict_other = load_from_disk(other_path)   # hub_data/librispeech/other
    
    # Select training set based on dataset_config
    cfg = config.dataset_config.lower()
    
    if cfg == "100h":
        train_dataset = ds_dict_clean["train.100"]
        train_split_name = "TRAIN (clean train.100)"
        use_dual_validation = False
    
    elif cfg == "460h":
        print("Creating merged 460h dataset: clean train.100 + clean train.360 (Lazy Concatenation)...")
        train_dataset = concatenate_datasets([ds_dict_clean["train.100"], ds_dict_clean["train.360"]])
        train_split_name = "TRAIN (clean train.100 + clean train.360)"
        use_dual_validation = True
    
    elif cfg == "500h":
        train_dataset = ds_dict_other["train.500"]
        train_split_name = "TRAIN (other train.500)"
        use_dual_validation = True
    
    elif cfg == "600h":
        print("Creating merged 600h dataset: clean train.100 + other train.500 (Lazy Concatenation)...")
        train_dataset = concatenate_datasets([ds_dict_clean["train.100"], ds_dict_other["train.500"]])
        train_split_name = "TRAIN (clean train.100 + other train.500)"
        use_dual_validation = True
    
    elif cfg == "960h":
        print("Creating merged 960h dataset: clean train.100 + clean train.360 + other train.500 (Lazy Concatenation)...")
        train_dataset = concatenate_datasets([
            ds_dict_clean["train.100"],
            ds_dict_clean["train.360"],
            ds_dict_other["train.500"],
        ])
        train_split_name = "TRAIN (clean train.100 + clean train.360 + other train.500)"
        use_dual_validation = True
    
    else:
        raise ValueError(f"Unknown dataset_config='{config.dataset_config}'. Valid: 100h, 460h, 500h, 600h, 960h")

    # Training dataset
    train_ds = LibriSpeechDataset(
        train_dataset,
        subset=config.subset_train,
        split_name=train_split_name,
        use_specaugment=config.use_specaugment,
        use_speed_perturb=config.use_speed_perturb,
        speed_perturb_factors=config.speed_perturb_factors
    )
    
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True if config.num_workers > 0 else False,
    )
    
    # Validation dataset(s)
    if use_dual_validation:
        # 460h mode: validate on both dev-clean and dev-other
        val_ds_clean = LibriSpeechDataset(
            ds_dict_clean["validation"], 
            subset=config.subset_val,
            split_name="DEV-CLEAN",
            use_specaugment=False
        )
        val_ds_other = LibriSpeechDataset(
            ds_dict_other["validation"], 
            subset=config.subset_val,
            split_name="DEV-OTHER",
            use_specaugment=False
        )
        
        val_loader_clean = DataLoader(
            val_ds_clean, batch_size=config.batch_size, shuffle=False,
            collate_fn=collate_fn, num_workers=config.num_workers,
            pin_memory=True, persistent_workers=True if config.num_workers > 0 else False,
        )
        val_loader_other = DataLoader(
            val_ds_other, batch_size=config.batch_size, shuffle=False,
            collate_fn=collate_fn, num_workers=config.num_workers,
            pin_memory=True, persistent_workers=True if config.num_workers > 0 else False,
        )
        
        val_loaders = [val_loader_clean, val_loader_other]
        monitor_metric = 'val_wer'  # Combined average
        
        print(f"\nDataLoader Configuration:")
        print(f"  Train batches per epoch: {len(train_loader)}")
        print(f"  Val batches per epoch (Clean): {len(val_loader_clean)}")
        print(f"  Val batches per epoch (Other): {len(val_loader_other)}")
        print(f"\n[Config] {cfg} mode: Validating on DEV-CLEAN + DEV-OTHER, monitoring '{monitor_metric}' (average)")        
    else:
        # 100h mode: validate on dev-clean only
        val_ds = LibriSpeechDataset(
            ds_dict_clean["validation"], 
            subset=config.subset_val,
            split_name="DEV-CLEAN",
            use_specaugment=False
        )
        
        val_loaders = DataLoader(
            val_ds, batch_size=config.batch_size, shuffle=False,
            collate_fn=collate_fn, num_workers=config.num_workers,
            pin_memory=True, persistent_workers=True if config.num_workers > 0 else False,
        )
        monitor_metric = 'val_wer'
        
        print(f"\nDataLoader Configuration:")
        print(f"  Train batches per epoch: {len(train_loader)}")
        print(f"  Val batches per epoch (DEV-CLEAN): {len(val_loaders)}")
        print(f"\n[Config] {cfg} mode: Validating on DEV-CLEAN only, monitoring '{monitor_metric}'")

    print(f"  Batch size: {config.batch_size}")
    print(f"  Accumulate: {config.accumulate_grad_batches} "
          f"(effective: {config.batch_size * config.accumulate_grad_batches})")
    
    # Calculate warmup steps
    total_steps = len(train_loader) * config.epochs // config.accumulate_grad_batches
    warmup_steps = config.warmup_steps
    
    # Create model
    model = LitASR(
        d_model=config.d_model,
        n_layers=config.n_layers,
        dropout=config.dropout,
        learning_rate=config.learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=config.weight_decay,
        use_speed_perturb=config.use_speed_perturb,
        use_hierarchical=config.use_hierarchical,
        use_gating=config.use_gating,
        enable_diagnostics=config.enable_diagnostics,
        # Mamba parameters
        encoder_type=config.encoder_type,
        d_state=config.d_state,
        d_conv=config.d_conv,
        expand=config.expand,
        use_cuda_kernels=config.use_cuda_kernels,
        # Mamba-3 parameters
        headdim=config.headdim,
        is_mimo=config.is_mimo,
        mimo_rank=config.mimo_rank,
    )
    
    # Save config
    config_path = os.path.join(checkpoint_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(asdict(config), f, indent=2)
    print(f"\nConfiguration saved to: {config_path}")
    print("")
    
    # Decide what to monitor for checkpointing / early stopping
    if use_dual_validation:
        monitor_metric = "val_wer"        # aggregate computed in on_validation_epoch_end
        print(f"\n[Config] {cfg} mode: Monitoring '{monitor_metric}' (Avg Clean+Other) for Early Stopping.")
    else:
        monitor_metric = "val_wer_clean"  # single loader is treated as clean (idx=0)
        print(f"\n[Config] {cfg} mode: Monitoring '{monitor_metric}' for Early Stopping.")

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor=monitor_metric,
        dirpath=checkpoint_dir,
        filename=f"best_{{epoch}}_{{{monitor_metric}:.3f}}", 
        mode='min',
        save_top_k=3,
        save_last=True,
    )
    
    early_stop_callback = EarlyStopping(
        monitor=monitor_metric,
        patience=8,
        mode='min',
        verbose=True,
    )
    
    progress_callback = CleanProgressCallback(print_every_n_batches=40)

    # 1. Create the base list of callbacks
    callbacks_list = [checkpoint_callback, early_stop_callback, progress_callback]

    # 2. Conditionally add the freeze callback (PASS PARAMETER HERE)
    if config.freeze_epochs > 0:
        freeze_cb = HierarchyFreezeCallback(unfreeze_epoch=config.freeze_epochs)
        callbacks_list.append(freeze_cb)
    
    # Logger (reuse csv_logger created above)
    
    # Trainer
    trainer = L.Trainer(
        max_epochs=config.epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        precision=config.precision,
        gradient_clip_val=config.gradient_clip_val,
        accumulate_grad_batches=config.accumulate_grad_batches,
        callbacks=callbacks_list,
        logger=csv_logger,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    
    # Train
    trainer.fit(model, train_loader, val_loaders)
    
    # Report results
    best_wer = checkpoint_callback.best_model_score.item()
    best_path = checkpoint_callback.best_model_path
    
    print(f"\n")
    print("FINAL RESULTS")
    print(f"   Best WER:        {best_wer*100:.2f}%")
    print(f"   Best checkpoint: {best_path}")
    print(f"   Results saved:   {checkpoint_dir}")


    # Final Evaluation
    print("\nEvaluating on test sets...")
    print(f"Loading best model: {best_path}")


    test_ds_clean = LibriSpeechDataset(
        ds_dict_clean["test"], 
        split_name="TEST-CLEAN",
        use_specaugment=False
    )
    test_loader_clean = DataLoader(test_ds_clean, batch_size=config.batch_size, collate_fn=collate_fn, num_workers=config.num_workers)

    test_ds_other = LibriSpeechDataset(
        ds_dict_other["test"], 
        split_name="TEST-OTHER",
        use_specaugment=False
    )
    test_loader_other = DataLoader(test_ds_other, batch_size=config.batch_size, collate_fn=collate_fn, num_workers=config.num_workers)
    
    print("\nWER results for 'test-clean' and 'test-other':")
    test_results = trainer.test(
        model,
        dataloaders=[test_loader_clean, test_loader_other],
        ckpt_path=best_path
    )
    
    # `test_results` is a list of dicts (one per dataloader)
    # Append a clean, deterministic summary to your experiment log file.
    with open(log_file, "a") as f:
        f.write("\n\n" + "=" * 80 + "\n")
        f.write("FINAL EVALUATION (trainer.test)\n")
        f.write(f"Checkpoint used: {best_path}\n")
        f.write(f"Num dataloaders: {len(test_results)}\n\n")
    
        for i, r in enumerate(test_results):
            f.write(f"[Dataloader {i}]\n")
            for k, v in sorted(r.items()):
                f.write(f"  {k}: {v}\n")
            f.write("\n")
    
    # Optional: also print the raw dicts so you see them in terminal without relying on Lightning's table
    print("\n=== FINAL TEST RESULTS (raw dicts) ===")
    for i, r in enumerate(test_results):
        print(f"Dataloader {i}: {r}")


    # Save layer diagnostics
    print("\nSaving layer diagnostics...")
    diag_dir = os.path.join(checkpoint_dir, 'diagnostics')
    for layer_idx, layer in enumerate(model.model.encoder.layers):
        if hasattr(layer, 'save_diagnostics'):
            layer.save_diagnostics(diag_dir)

    # Generate visualizations
    print("\nGenerating visualizations...")
    visualizer = Visualizer(
        log_dir=f"logs/{config.get_exp_name()}",
        output_dir=os.path.join(checkpoint_dir, 'figures'),
        exp_name=config.get_exp_name()
    )
    
    # 1. Run the main plots (loss, WER, ablations)
    visualizer.generate_all()
    
    # 2. Run the plots that need the trained model
    print(" Generating: parameter_breakdown.pdf")
    param_counts = get_param_counts(model)
    if param_counts:
        visualizer.plot_parameter_breakdown(param_counts)

    # 3. Run the fixed hierarchical dynamics plot
    visualizer.visualize_hierarchical_dynamics(model) # This is the fixed version

    # 4. State dynamics heatmap
    print(" Generating: state_dynamics_heatmap.pdf")
    visualizer.plot_state_dynamics_heatmap(model) 

    # 5. Run the GPU plot using data from the callback
    print(" Generating: gpu_memory.pdf")
    gpu_history = progress_callback.history
    if gpu_history['gpu_memory_allocated']:
        visualizer.plot_gpu_memory(gpu_history)
    else:
        print("  Skipped: No GPU memory data recorded.")

    return best_wer, best_path



# ============================================================================
# GENERATING PLOTS AND FIGURES (VISUALIZATION)
# ============================================================================


# Set publication-quality defaults
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'lines.linewidth': 2,
    'axes.grid': True,
    'grid.alpha': 0.3,
})

# Color palette (colorblind-friendly)
COLORS = {
    'primary': '#2E86AB',    # Blue
    'secondary': '#A23B72',  # Purple
    'accent': '#F18F01',     # Orange
    'success': '#06A77D',    # Green
    'baseline': '#808080',   # Gray
}


class Visualizer:
    """
    Complete visualization suite for ASR.
    Generates figures.
    """
    
    def __init__(
        self,
        log_dir: str,
        output_dir: str = 'model_figures',
        exp_name: str = 'experiment'
    ):
        """
        Initialize visualizer.
        
        Args:
            log_dir: Path to Lightning logs directory
            output_dir: Where to save figures
            exp_name: Experiment name for figure titles
        """
        self.log_dir = Path(log_dir)
        self.output_dir = Path(output_dir)
        self.exp_name = exp_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n")
        print("Visualizing results: ")
        print(f"   Log directory: {self.log_dir}")
        print(f"   Output directory: {self.output_dir}")
        print(f"   Experiment: {self.exp_name}")
    
    def generate_all(self):
        """Generate all figures."""
        print("Generating figures...")
        
        try:
            # Find latest metrics CSV
            csv_path = self._find_latest_csv()
            print(f"Using metrics: {csv_path}\n")
            
            # Generate all figures
            figures = [
                ('training_curves', self.plot_training_curves),
                ('wer_curves', self.plot_wer_curves),
                ('ablation_study', self.plot_ablation_study),
                ('hierarchical_init', self.plot_hierarchical_initialization),
                ('lr_schedule', self.plot_learning_rate_schedule),
            ]
            
            for name, func in figures:
                try:
                    if name in ['training_curves', 'wer_curves']:
                        func(csv_path)
                    else:
                        func()
                    print(f" Generated: {name}.pdf")
                except Exception as e:
                    print(f" Failed: {name}.pdf - {e}")
            

            print(f"Figures saved to: {self.output_dir}")
            
        except Exception as e:
            print(f"Error generating figures: {e}") 
            traceback.print_exc()
    
    def _find_latest_csv(self) -> Path:
        """Find latest metrics.csv file."""
        csv_files = list(self.log_dir.rglob('metrics.csv'))
        if not csv_files:
            raise FileNotFoundError(f"No metrics.csv found in {self.log_dir}")
        return max(csv_files, key=lambda p: p.stat().st_mtime)
    
    # ========================================================================
    # CATEGORY 1: Training Dynamics
    # ========================================================================
    
    def plot_training_curves(self, csv_path: Path):  
        """Plot training and validation loss curves."""
        try:
            # Read the CSV file
            metrics_df = pd.read_csv(csv_path)
            
            # Create single subplot
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Handle Lightning's column naming conventions
            train_loss_col = None
            for col in ['train_loss_epoch', 'train_loss', 'train_loss_step']:
                if col in metrics_df.columns:
                    train_loss_col = col
                    break
            
            val_loss_col = None
            for col in ['val_loss', 'val_loss_epoch']:
                if col in metrics_df.columns:
                    val_loss_col = col
                    break
            
            if train_loss_col is None or val_loss_col is None:
                print(f"Missing columns. Available: {metrics_df.columns.tolist()}")
                raise KeyError("Required loss columns not found")
            
            # Group by epoch and take mean (in case of multiple rows per epoch)
            epoch_data = metrics_df.groupby('epoch').agg({
                train_loss_col: 'mean',
                val_loss_col: 'mean'
            }).reset_index()
            
            # Plot losses
            ax.plot(epoch_data['epoch'], epoch_data[train_loss_col], 
                    'b-', label='Train Loss', linewidth=2, marker='o', markersize=4)
            ax.plot(epoch_data['epoch'], epoch_data[val_loss_col],
                    'r-', label='Val Loss', linewidth=2, marker='o', markersize=4)
            
            # Styling
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('CTC Loss', fontsize=12)
            ax.set_title('Training and Validation Loss Curves', fontsize=14, fontweight='bold')
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            
            # Save
            save_path = self.output_dir / 'training_curves.pdf'
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f" Failed: training_curves.pdf - {e}")
            
            traceback.print_exc()
    
    def plot_wer_curves(self, csv_path: Path):
        """Plot validation WER over epochs."""
        df = pd.read_csv(csv_path)
        val_wer = df[df['val_wer'].notna()][['epoch', 'val_wer']].dropna()
        
        # Convert to percentage
        val_wer['val_wer'] = val_wer['val_wer'] * 100
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Find best WER first
        best_wer = val_wer['val_wer'].min()
        best_epoch = val_wer.loc[val_wer['val_wer'].idxmin(), 'epoch']
        
        # Plot WER with best in title
        ax.plot(val_wer['epoch'], val_wer['val_wer'],
                color=COLORS['primary'], marker='o', markersize=4,
                linewidth=2, label=f'Val WER (Best: {best_wer:.2f}% @ Epoch {int(best_epoch)})')
        
        # Styling
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Word Error Rate (%)', fontsize=12)
        ax.set_title('Validation WER During Training', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.3)
        
        # Save
        save_path = self.output_dir / 'wer_curves.pdf'
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    # ========================================================================
    # CATEGORY 2: Ablation Study
    # ========================================================================
    
    def plot_ablation_study(
        self,
        configurations: Optional[List[str]] = None,
        wer_values: Optional[List[float]] = None
    ):
        """
        Plot ablation study results.
        
        Args:
            configurations: List of configuration names
            wer_values: List of WER values (in percentage)
        """
        # Default values (update with your actual results)
        if configurations is None:
            configurations = [
                'Baseline\nSSM',
                '+ Hierarchical',
                '+ Gating',
                '+ Both',
            ]
        
        if wer_values is None:
            wer_values = [
                32.88,  # Baseline
                32.01,  # + Hierarchical
                31.78,  # + Gating
                31.23,  # + Both
            ]
        
        # Calculate absolute improvements
        baseline = wer_values[0]
        improvements = [0] + [baseline - wer for wer in wer_values[1:]]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Colors: baseline gray, improvements blue
        colors = [COLORS['baseline']] + [COLORS['primary']] * (len(configurations) - 1)
        
        # Create bars
        bars = ax.bar(configurations, wer_values, color=colors,
                      alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for i, (bar, wer, imp) in enumerate(zip(bars, wer_values, improvements)):
            height = bar.get_height()
            
            # WER value on top
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{wer:.2f}%',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
            
            # Improvement inside bar (skip baseline)
            if i > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height - 2,
                        f'↓{imp:.2f}%',
                        ha='center', va='top', fontsize=10,
                        color='white', fontweight='bold')
        
        # Styling
        ax.set_ylabel('Word Error Rate (%)')
        ax.set_title('Ablation Study: Impact of Architectural Components')
        ax.set_ylim([0, max(wer_values) + 3])
        ax.grid(True, alpha=0.3, axis='y')
        
        # Save
        save_path = self.output_dir / 'ablation_study.pdf'
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    # ========================================================================
    # CATEGORY 3: SSM-Specific Visualizations
    # ========================================================================
    
    def plot_hierarchical_initialization(self, n_layers: int = 12):
            """Plot hierarchical initialization values across layers."""
            layers = np.arange(n_layers)
            # Avoid division by zero if n_layers=1
            denom = max(1, n_layers - 1)
            layer_progress = layers / denom
            
            # (Early=0.15, Late=0.85)
            # These formulas now match the logic in HierarchicalSelectiveSSMLayer
            a_early, a_late = 0.15, 0.85
            b_early, b_late = 0.35, 0.12
            c_early, c_late = -0.35, 0.20
            
            # Calculate curves
            a_means = a_early * ((a_late / a_early) ** layer_progress)
            b_means = b_early * ((b_late / b_early) ** layer_progress)
            c_means = c_early + (c_late - c_early) * layer_progress
            
            # Create figure with 3 subplots
            fig, axes = plt.subplots(3, 1, figsize=(10, 10))
            
            # Plot a (state decay)
            axes[0].plot(layers, a_means, marker='o', linewidth=2,
                         markersize=8, color=COLORS['primary'])
            axes[0].set_ylabel('State Decay (a)')
            axes[0].set_title('Hierarchical State Dynamics Initialization')
            axes[0].grid(True, alpha=0.3)
            
            # Plot b (input sensitivity)
            axes[1].plot(layers, b_means, marker='s', linewidth=2,
                         markersize=8, color=COLORS['secondary'])
            axes[1].set_ylabel('Input Sensitivity (b)')
            axes[1].grid(True, alpha=0.3)
            
            # Plot c (output weight)
            axes[2].plot(layers, c_means, marker='^', linewidth=2,
                         markersize=8, color=COLORS['accent'])
            axes[2].set_xlabel('Layer Index')
            axes[2].set_ylabel('Output Weight (c)')
            axes[2].grid(True, alpha=0.3)
            
            # Save
            save_path = self.output_dir / 'hierarchical_init.pdf'
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
    
    # ========================================================================
    # CATEGORY 4: Training Configuration
    # ========================================================================
    
    def plot_learning_rate_schedule(
        self,
        learning_rate: float = 1e-3,
        warmup_steps: int = 500,
        total_steps: int = 26700
    ):
        """Plot learning rate schedule."""
        steps = np.arange(total_steps)
        lrs = []
        
        for step in steps:
            if step < warmup_steps:
                # Linear warmup
                lr = learning_rate * (step / warmup_steps)
            else:
                # Cosine annealing
                progress = (step - warmup_steps) / (total_steps - warmup_steps)
                lr = learning_rate * 0.5 * (1 + np.cos(np.pi * progress))
            lrs.append(lr)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(steps, lrs, linewidth=2, color=COLORS['primary'])
        
        # Mark warmup end
        ax.axvline(x=warmup_steps, color='red', linestyle='--',
                   linewidth=1.5, alpha=0.7, label=f'End of Warmup (step {warmup_steps})')
        
        # Styling
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Cosine Learning Rate Schedule with Warmup')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Save
        save_path = self.output_dir / 'lr_schedule.pdf'
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    # ========================================================================
    # ADDITIONAL UTILITIES
    # ========================================================================
    
    def plot_parameter_breakdown(
        self,
        components: Optional[Dict[str, int]] = None
    ):
        """Plot parameter distribution across model components."""
        if components is None:
            # Default values (update with your model)
            components = {
                'Convolutional Subsample': 442368,
                'SSM Layers': 6750720,
                'Output Head': 11136,
            }
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Pie chart
        colors = [COLORS['primary'], COLORS['secondary'], COLORS['accent']]
        ax.pie(components.values(), labels=components.keys(),
               autopct='%1.1f%%', startangle=90, colors=colors,
               textprops={'fontsize': 11})
        
        ax.set_title('Parameter Distribution Across Model Components')
        
        # Save
        save_path = self.output_dir / 'parameter_breakdown.pdf'
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


    def visualize_hierarchical_dynamics(self, model):
        """Visualize learned temporal dynamics across layers (supports both encoders)"""
        print(" Generating: hierarchical_dynamics.pdf (Learned Parameters)")
        
        encoder_type = getattr(model.model, 'encoder_type', 'sssm')
        
        fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        
        layer_indices = []
        param1 = []  # a0 for SSSM, A_log for Mamba
        param2 = []  # b0 for SSSM, dt_bias for Mamba
        param3 = []  # c0 for SSSM, D for Mamba
        
        for idx, layer in enumerate(model.model.encoder.layers):
            layer_indices.append(idx)
            
            if encoder_type == 'mamba':
                param1.append(layer.A_log.detach().mean().cpu().item())
                param2.append(layer.dt_proj.bias.detach().mean().cpu().item())
                param3.append(layer.D.detach().mean().cpu().item())
            else:  # sssm
                param1.append(layer.a0.detach().mean().cpu().item())
                param2.append(layer.b0.detach().mean().cpu().item())
                param3.append(layer.c0.detach().mean().cpu().item())
        
        # Labels based on encoder type
        if encoder_type == 'mamba':
            labels = ['A_log (State Decay)', 'dt_bias (Timescale)', 'D (Skip)']
        else:
            labels = ['a0 (State Decay)', 'b0 (Input Gate)', 'c0 (Output Gate)']
        
        axes[0].plot(layer_indices, param1, marker='o', linewidth=2, color=COLORS['primary'])
        axes[0].set_ylabel(labels[0])
        axes[0].set_title(f'Learned Hierarchical Dynamics ({encoder_type.upper()})')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(layer_indices, param2, marker='s', linewidth=2, color=COLORS['secondary'])
        axes[1].set_ylabel(labels[1])
        axes[1].grid(True, alpha=0.3)
        
        axes[2].plot(layer_indices, param3, marker='^', linewidth=2, color=COLORS['accent'])
        axes[2].set_xlabel('Layer Index')
        axes[2].set_ylabel(labels[2])
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / 'hierarchical_dynamics.pdf'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_gpu_memory(self, history: Dict[str, list]):
        """Plot GPU memory usage over time"""
        if 'gpu_memory_allocated' not in history or not history['gpu_memory_allocated']:
            print("⚠ No GPU memory data to plot")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        epochs = range(1, len(history['gpu_memory_allocated']) + 1)
        
        ax.plot(epochs, history['gpu_memory_allocated'], 
                label='Allocated', color=COLORS['primary'], linewidth=2)
        ax.plot(epochs, history['gpu_memory_reserved'], 
                label='Reserved', color=COLORS['secondary'], linewidth=2)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('GPU Memory (GB)', fontsize=12)
        ax.set_title('GPU Memory Usage During Training', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        save_path = self.output_dir / 'gpu_memory.pdf'
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved GPU memory plot: {save_path}")

    def plot_wer_vs_duration(self, results: List[Dict]):
        """Plot WER as a function of audio duration."""
        durations = [r['duration'] for r in results]
        wers = [r['wer'] for r in results]
        
        # Bin by duration
        bins = [0, 5, 10, 15, 20, 30]
        bin_wers = []
        bin_labels = []
        
        for i in range(len(bins)-1):
            mask = [(d >= bins[i] and d < bins[i+1]) for d in durations]
            if any(mask):
                bin_wers.append(np.mean([w for w, m in zip(wers, mask) if m]))
                bin_labels.append(f'{bins[i]}-{bins[i+1]}s')
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(bin_labels, bin_wers, color=COLORS['primary'])
        ax.set_xlabel('Audio Duration (seconds)')
        ax.set_ylabel('Word Error Rate (%)')
        ax.set_title('WER vs Audio Duration')
        ax.grid(True, alpha=0.3, axis='y')
        
        save_path = self.output_dir / 'wer_vs_duration.pdf'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_state_dynamics_heatmap(self, model):
        """2D heatmap of learned SSM parameters across layers."""
        n_layers = len(model.model.encoder.layers)
        d_model = model.model.encoder.layers[0].d_model
        
        # Sample subset of dimensions for visualization
        sample_dims = min(64, d_model)
        dim_indices = np.linspace(0, d_model-1, sample_dims, dtype=int)
        
        a_matrix = np.zeros((n_layers, sample_dims))
        b_matrix = np.zeros((n_layers, sample_dims))
        
        for i, layer in enumerate(model.model.encoder.layers):
            a_matrix[i] = layer.a0.detach().cpu().numpy()[dim_indices]
            b_matrix[i] = layer.b0.detach().cpu().numpy()[dim_indices]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        im1 = ax1.imshow(a_matrix, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
        ax1.set_xlabel('Dimension Index')
        ax1.set_ylabel('Layer Index')
        ax1.set_title('State Decay (a0) Across Layers')
        plt.colorbar(im1, ax=ax1)
        
        im2 = ax2.imshow(b_matrix, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
        ax2.set_xlabel('Dimension Index')
        ax2.set_ylabel('Layer Index')
        ax2.set_title('Input Sensitivity (b0) Across Layers')
        plt.colorbar(im2, ax=ax2)
        
        save_path = self.output_dir / 'state_dynamics_heatmap.pdf'
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_convergence_comparison(self, experiment_csvs: Dict[str, Path]):
        """Compare training curves across experiments."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        for name, csv_path in experiment_csvs.items():
            df = pd.read_csv(csv_path)
            
            # Get loss column
            loss_col = 'val_loss_epoch' if 'val_loss_epoch' in df.columns else 'val_loss'
            wer_col = 'val_wer_epoch' if 'val_wer_epoch' in df.columns else 'val_wer'
            
            epoch_data = df.groupby('epoch').agg({
                loss_col: 'mean',
                wer_col: 'mean'
            }).reset_index()
            
            ax1.plot(epoch_data['epoch'], epoch_data[loss_col], 
                    label=name, linewidth=2, marker='o', markersize=3)
            ax2.plot(epoch_data['epoch'], epoch_data[wer_col] * 100,
                    label=name, linewidth=2, marker='o', markersize=3)
        
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Validation Loss')
        ax1.set_title('Convergence: Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Word Error Rate (%)')
        ax2.set_title('Convergence: Word Error Rate')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        save_path = self.output_dir / 'convergence_comparison.pdf'
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_example_predictions(self, examples: List[Dict]):
        """Display example predictions with reference text."""
        fig, ax = plt.subplots(figsize=(12, len(examples) * 1.5))
        ax.axis('off')
        
        text_content = []
        for i, ex in enumerate(examples):
            ref = ex['reference']
            hyp = ex['hypothesis']
            wer = ex['wer']
            
            text_content.append(f"Example {i+1} (WER: {wer:.1f}%)")
            text_content.append(f"REF: {ref}")
            text_content.append(f"HYP: {hyp}")
            text_content.append("")
        
        ax.text(0.05, 0.95, '\n'.join(text_content),
                transform=ax.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace')
        
        save_path = self.output_dir / 'example_predictions.pdf'
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    """Training configuration."""
    
    # Data
    data_path: str = "hub_data/librispeech"
    train_split: str = "train.100"
    val_split: str = "validation"
    dataset_config: str = "100h"
    subset_train: Optional[int] = None
    subset_val: Optional[int] = None
    
    # Training
    batch_size: int = 32
    num_workers: int = min(4, os.cpu_count() or 1)      # auto-detect
    epochs: int = 30
    learning_rate: float = 5e-4
    warmup_steps: int = 2000
    weight_decay: float = 0.05
    gradient_clip_val: float = 5.0
    accumulate_grad_batches: int = 1
    enable_diagnostics: bool = False       # Diagnostic Logging every 100 batches (at, bt, ct, s)
    freeze_epochs: int = 0  # 0 = disabled
    
    # Model - Core
    d_model: int = 256
    n_layers: int = 12
    dropout: float = 0.1
    use_hierarchical: bool = True
    use_gating: bool = True  # Only for S-SSSM

    # Model - Mamba specific
    encoder_type: str = "mamba"  # "mamba", "mamba3", or "sssm"
    d_state: int = 16            # SSM state dimension
    d_conv: int = 4              # Convolution kernel size (Mamba-1 only)
    expand: int = 2              # Expansion factor (Mamba-1 only)
    use_cuda_kernels: bool = True  # Use mamba_ssm CUDA kernels if available
    
    # Model - Mamba-3 specific
    headdim: int = 64            # Head dimension (Mamba-3 only)
    is_mimo: bool = False        # MIMO mode (Mamba-3 only)
    mimo_rank: int = 4           # MIMO rank (Mamba-3 only, when is_mimo=True)
    
    # Augmentation
    use_specaugment: bool = True
    use_speed_perturb: bool = False
    speed_perturb_factors: Tuple[float, ...] = (0.9, 1.0, 1.1)
    
    # System
    precision: str = "16-mixed"
    seed: int = 42
    
    # Experiment
    exp_name: str = "mamba_hierarchical"
    
    def get_exp_name(self) -> str:
        """Generate experiment name."""
        if self.exp_name:
            return self.exp_name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"{self.encoder_type}_hier{int(self.use_hierarchical)}_{timestamp}"
        return name


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point."""
    
    parser = argparse.ArgumentParser(
        description="SSM-ASR: CTC Encoder Training (Mamba-1 / Mamba-3 / S-SSSM)"
    )
    
    parser.add_argument(
        '--data-path',
        type=str,
        default='hub_data/librispeech',
        help='Path to LibriSpeech dataset'
    )
    
    parser.add_argument(
        '--dataset-config',
        type=str,
        default='100h',
        choices=['100h', '460h', '500h', '600h', '960h'],
        help='Select training set: 100h, 460h (100+360), 500h (other), 600h (100+500), 960h (100+360+500)'
    )
    
    parser.add_argument(
        '--exp-name',
        type=str,
        default='hierarchical_gating',
        help='Experiment name'
    )
    
    parser.add_argument(
        '--subset-train',
        type=int,
        default=None,
        help='Limit training examples (None for all)'
    )
    
    parser.add_argument(
        '--subset-val',
        type=int,
        default=None,
        help='Limit validation examples (None for all)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=30,
        help='Number of epochs'
    )
    
    parser.add_argument(
        '--d-model',
        type=int,
        default=384,
        help='Model dimension'
    )
    
    parser.add_argument(
        '--n-layers',
        type=int,
        default=8,
        help='Number of encoder layers'
    )
    
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-3,
        help='Learning rate'
    )
    
    parser.add_argument(
        '--no-hierarchical',
        action='store_true',
        help='Disable non-uniform layer-wise init (flag name kept for compatibility; '
             'see module docstring for terminology)'
    )
    
    parser.add_argument(
        '--no-gating',
        action='store_true',
        help='Disable gating mechanism'
    )
    
    parser.add_argument(
        '--no-specaugment',
        action='store_true',
        help='Disable SpecAugment'
    )
    
    parser.add_argument(
        '--speed-perturb',
        action='store_true',
        help='Enable speed perturbation (0.9x, 1.0x, 1.1x)'
    )
    
    parser.add_argument(
        '--speed-factors',
        type=float,
        nargs='+',
        default=[0.9, 1.0, 1.1],
        help='Speed perturbation factors (default: 0.9 1.0 1.1)'
    )
    
    parser.add_argument(
        '--fp32',
        action='store_true',
        help='Use FP32 instead of mixed precision'
    )
   
    parser.add_argument(
        '--seed',
        type=int,
        default=456,
        help='Random seed for reproducibility'
    )

    parser.add_argument(
        '--freeze-epochs',
        type=int,
        default=0,
        help='Number of epochs to freeze non-uniform init parameters (set to 0 to disable)'
    )
    
    # Mamba-specific arguments
    parser.add_argument(
        '--encoder-type',
        type=str,
        default='mamba',
        choices=['mamba', 'mamba3', 'sssm'],
        help='Encoder architecture: mamba (Mamba-1), mamba3 (Mamba-3), or sssm (original S-SSSM)'
    )
    
    parser.add_argument(
        '--d-state',
        type=int,
        default=16,
        help='SSM state dimension (typically 16 for Mamba-1, 64+ for Mamba-3)'
    )
    
    parser.add_argument(
        '--d-conv',
        type=int,
        default=4,
        help='Convolution kernel size for Mamba-1 (typically 4)'
    )
    
    parser.add_argument(
        '--expand',
        type=int,
        default=2,
        help='Expansion factor for Mamba-1 inner dimension (typically 2)'
    )

    parser.add_argument(
        '--no-cuda-kernels',
        action='store_true',
        help='Force PyTorch implementation even if mamba_ssm is available'
    )
    
    # Mamba-3 specific arguments
    parser.add_argument(
        '--headdim',
        type=int,
        default=64,
        help='Head dimension for Mamba-3 (typically 64)'
    )
    
    parser.add_argument(
        '--is-mimo',
        action='store_true',
        help='Enable MIMO mode for Mamba-3 (multi-input multi-output)'
    )
    
    parser.add_argument(
        '--mimo-rank',
        type=int,
        default=4,
        help='MIMO rank for Mamba-3 (default 4, only used with --is-mimo)'
    )

    
    args = parser.parse_args()
        
    # Create config
    config = Config(
        data_path=args.data_path,
        dataset_config=args.dataset_config,
        exp_name=args.exp_name,
        subset_train=args.subset_train,
        subset_val=args.subset_val,
        batch_size=args.batch_size,
        epochs=args.epochs,
        d_model=args.d_model,
        n_layers=args.n_layers,
        learning_rate=args.lr,
        use_hierarchical=not args.no_hierarchical,
        use_gating=not args.no_gating,
        freeze_epochs=args.freeze_epochs,
        use_specaugment=not args.no_specaugment,
        use_speed_perturb=args.speed_perturb,
        speed_perturb_factors=tuple(args.speed_factors),
        precision="32-true" if args.fp32 else ("bf16-true" if args.encoder_type == "mamba3" else "16-mixed"),
        seed=args.seed,
        # Mamba parameters
        encoder_type=args.encoder_type,
        d_state=args.d_state,
        d_conv=args.d_conv,
        expand=args.expand,
        use_cuda_kernels=not args.no_cuda_kernels,
        # Mamba-3 parameters
        headdim=args.headdim,
        is_mimo=args.is_mimo,
        mimo_rank=args.mimo_rank,
    )
    
    # Train
    try:
        best_wer, best_path = train(config)
        sys.exit(0)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nERROR: {e}")
        
        traceback.print_exc()
        sys.exit(1)




# ============================================================================
# MAIN FUNCTION CALL
# ============================================================================

if __name__ == '__main__':
    main()