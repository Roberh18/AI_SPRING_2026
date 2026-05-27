"""
ASR Pipeline Evaluation Script
==============================

End-to-end evaluation of the pure-SSM ASR pipeline on LibriSpeech. Supports
the SSSM, Mamba-1, and Mamba-3 encoder families, and CharMamba-1/CharMamba-3
or character n-gram language models for shallow-fusion rescoring.

Training of the CharMamba-1 LM lives in the companion script
IKT590_train_LM_CharMamba1_v1_4.py. This script only loads pre-trained
checkpoints and runs evaluation modes.

Author: Robert Alexander Hanssen
Thesis: State Space Models for Automatic Speech Transcription
Date:   March 2026

================================================================================
SETUP
================================================================================
    pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
    pip install causal-conv1d==1.5.0.post8
    pip install mamba-ssm==2.2.4
    pip install lightning torchaudio jiwer datasets

================================================================================
EVALUATION MODES
================================================================================

  tune              Grid-search alpha/beta on dev-clean and report best WER.
  full_eval         Run greedy CTC, beam search, and beam + LM rescoring on
                    test-clean and test-other; report WER, P, R, F1 per mode.
  rtfx              Measure decoding throughput (RTFX) and per-utterance
                    latency for greedy / beam / beam + LM decoding.
  hallucination     CTC confidence x LM probability quadrant analysis.
  score_analysis    Per-hypothesis CTC + LM score breakdown with figures.
  verify_streaming  Sanity-check that the LM's batch and streaming scoring
                    paths agree (useful after retraining the LM).

================================================================================
TYPICAL USAGE
================================================================================

# Tune alpha on dev-clean (S-SSSM 960h encoder + CharMamba-1 LM)
python IKT590_evaluate_ASR_pipeline.py \
    --mode tune \
    --data-path hub_data/librispeech \
    --asr-checkpoint encoder_checkpoints/S-SSSM/960h/best_epoch=98_val_wer=0.111.ckpt \
    --asr-type sssm \
    --elm-path lm_checkpoints/elm_mamba_MaxChars-1000000000_d320_L18 \
    --beam-width 10 --batch-size 64 \
    --exp-name CharMamba1_tune_ssssm_960h

# Full evaluation at alpha=100 on test-clean and test-other
python IKT590_evaluate_ASR_pipeline.py \
    --mode full_eval \
    --data-path hub_data/librispeech \
    --asr-checkpoint encoder_checkpoints/S-SSSM/960h/best_epoch=98_val_wer=0.111.ckpt \
    --asr-type sssm \
    --elm-path lm_checkpoints/elm_mamba_MaxChars-1000000000_d320_L18 \
    --alpha 100.0 --beta 0.0 --beam-width 10 --batch-size 64 \
    --exp-name CharMamba1_eval_ssssm_960h_a100

# RTFX measurement
python IKT590_evaluate_ASR_pipeline.py \
    --mode rtfx \
    --data-path hub_data/librispeech \
    --asr-checkpoint encoder_checkpoints/S-SSSM/960h/best_epoch=98_val_wer=0.111.ckpt \
    --asr-type sssm \
    --elm-path lm_checkpoints/elm_mamba_MaxChars-1000000000_d320_L18 \
    --alpha 32.0 --beam-width 10 --batch-size 64 \
    --exp-name CharMamba1_rtfx_ssssm_960h

# n-gram baseline (control LM)
python IKT590_evaluate_ASR_pipeline.py \
    --mode full_eval \
    --data-path hub_data/librispeech \
    --asr-checkpoint encoder_checkpoints/S-SSSM/960h/best_epoch=98_val_wer=0.111.ckpt \
    --asr-type sssm \
    --elm-path lm_checkpoints/ngram/char_10gram.pkl \
    --alpha 8.0 --beta 0.0 --beam-width 10 --batch-size 64 \
    --exp-name ngram10_eval_ssssm_960h_a8

================================================================================
OUTPUT STRUCTURE
================================================================================

ilme_mamba_results_<exp-name>/
    full_eval_test-clean.json
    full_eval_test-other.json
    best_params.json                (tune mode)
    grid_results.json               (tune mode)
    rtfx_greedy.json                (rtfx mode)
    rtfx_beam.json                  (rtfx mode)
    rtfx_nbest_ilme.json            (rtfx mode)
    hallucination.json              (hallucination mode)
    score_analysis_alpha<A>_beam<W>.txt  (score_analysis mode)
    score_analysis_summary.json     (score_analysis mode)
    figures/                        (score_analysis mode, PDF figures)
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
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Optional: mamba_ssm for CUDA-accelerated Mamba (required to load CharMamba LMs)
try:
    from mamba_ssm import Mamba as MambaCUDA
    _HAS_MAMBA_SSM = True
    print(f"[OK] mamba_ssm loaded — CUDA Mamba available")
except ImportError:
    _HAS_MAMBA_SSM = False
    MambaCUDA = None
    print(f"[WARNING] mamba_ssm not found — install with: pip install mamba-ssm>=2.2.0 causal-conv1d>=1.5.0")

# Optional: HuggingFace datasets — only load_from_disk is used (for LibriSpeech splits)
try:
    from datasets import load_from_disk
    _HAS_HF = True
except ImportError:
    _HAS_HF = False

# Optional: jiwer for WER
try:
    from jiwer import wer as jiwer_wer
    _HAS_JIWER = True
except ImportError:
    _HAS_JIWER = False

try:
    import torchaudio
    _HAS_TORCHAUDIO = True
except ImportError:
    _HAS_TORCHAUDIO = False

try:
    import kenlm as _kenlm
    _HAS_KENLM = True
except ImportError:
    _kenlm = None
    _HAS_KENLM = False

warnings.filterwarnings('ignore', category=UserWarning)


# ============================================================================
# VOCABULARY & CONSTANTS (must match ASR training scripts)
# ============================================================================

VOCAB_CHARS = list(" 'abcdefghijklmnopqrstuvwxyz")
BLANK_TOKEN = len(VOCAB_CHARS)  # 27
VOCAB_SIZE = len(VOCAB_CHARS) + 1  # 28 (27 chars + blank/BOS)
CHAR_TO_IDX = {c: i for i, c in enumerate(VOCAB_CHARS)}
IDX_TO_CHAR = {i: c for i, c in enumerate(VOCAB_CHARS)}
SAMPLE_RATE = 16000


def text_to_ids(text: str) -> List[int]:
    return [CHAR_TO_IDX[c] for c in text.lower() if c in CHAR_TO_IDX]

def ids_to_text(ids: List[int]) -> str:
    return "".join(IDX_TO_CHAR[i] for i in ids if i in IDX_TO_CHAR)


# ============================================================================
# NON-UNIFORM INIT CONFIG FOR MAMBA LM
# (Legacy name "hierarchical" kept for checkpoint/config compatibility;
#  in the final thesis this is described as non-uniform layer-wise
#  initialization of decay/timescale parameters.)
# ============================================================================

MAMBA_LM_HIER_CONFIG = {
    'DT_MIN_EARLY': 0.001,
    'DT_MAX_EARLY': 0.01,
    'DT_MIN_LATE': 0.005,
    'DT_MAX_LATE': 0.05,
}

# S-SSSM non-uniform init config (for ILM, matching ASR encoder)
HIER_CONFIG = {
    'A0_EARLY': 0.15, 'A0_LATE': 0.85,
    'B0_EARLY': 0.35, 'B0_LATE': 0.12,
    'C0_EARLY': -0.35, 'C0_LATE': 0.20,
}
HIER_MODE = 'normal'


# ============================================================================
# CHARACTER MAMBA LM (for ELM)
# ============================================================================

class CharMambaLM(nn.Module):
    """
    Character-level autoregressive LM using Mamba layers.
    
    Key advantages over S-SSSM LM:
    1. CUDA-accelerated selective scan (10-50x faster training)
    2. Native .step() method for correct streaming inference
    3. Input-dependent selectivity (B, C, delta are data-dependent)
    4. Proper ZOH discretization
    
    Modes:
    - Batch: forward() for training (full CUDA scan)
    - Step: forward_step() for streaming (native Mamba .step())
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
        
        for i in range(n_layers):
            if _HAS_MAMBA_SSM:
                mamba = MambaCUDA(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                )
            else:
                # Fallback: use a simple linear + activation (placeholder)
                raise RuntimeError(
                    "mamba_ssm required for CharMambaLM. "
                    "Install: pip install mamba-ssm>=2.2.0 causal-conv1d>=1.5.0"
                )
            
            self.layers.append(mamba)
            self.norms.append(nn.LayerNorm(d_model))
            self.dropouts.append(nn.Dropout(dropout))
        
        # Apply non-uniform layer-wise dt_bias initialization
        # (flag name kept for checkpoint compatibility)
        if use_hierarchical:
            self._apply_hierarchical_init()
        
        self.final_norm = nn.LayerNorm(d_model)
        self.output_head = nn.Linear(d_model, vocab_size)
        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"\n[CharMambaLM] {total_params:,} params "
              f"(d={d_model}, L={n_layers}, N={d_state}, "
              f"expand={expand}, hier={use_hierarchical})")
    
    def _apply_hierarchical_init(self):
        """Apply non-uniform layer-wise dt_bias initialization.
        
        Method name kept for checkpoint compatibility; in the final thesis
        this is described as non-uniform initialization of decay/timescale
        parameters (initialization diversity).
        """
        cfg = MAMBA_LM_HIER_CONFIG
        n = self.n_layers_count
        
        print(f"\n  [CharMambaLM] Non-uniform layer-wise dt_bias initialization:")
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
    
    # ----- Batch mode (training) -----
    
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
    
    def get_log_probs(self, input_ids: torch.Tensor) -> torch.Tensor:
        return F.log_softmax(self.forward(input_ids), dim=-1)
    
    # ----- Step mode (streaming inference via native Mamba .step()) -----
    
    def reset_state(self, batch_size: int = 1, device=None):
        """Reset all layer states for streaming. Call at start of each utterance."""
        if device is None:
            device = next(self.parameters()).device
        
        self._inference_params = []
        for mamba in self.layers:
            # Mamba.step() needs conv_state [B, d_inner, d_conv] and ssm_state [B, d_inner, d_state]
            # d_inner is stored as mamba.d_inner in the official mamba_ssm Mamba class
            d_inner = mamba.d_inner
            conv_state = torch.zeros(batch_size, d_inner, mamba.d_conv, device=device, dtype=torch.float32)
            ssm_state = torch.zeros(batch_size, d_inner, mamba.d_state, device=device, dtype=torch.float32)
            self._inference_params.append((conv_state, ssm_state))
    
    def forward_step(self, char_id: torch.Tensor) -> torch.Tensor:
        """
        Process ONE character using native Mamba .step().
        
        CRITICAL: mamba_ssm .step() expects hidden_states with shape [B, 1, D]
        (not [B, D]). It asserts hidden_states.shape[1] == 1.
        States (conv_state, ssm_state) are updated IN-PLACE by .step().
        
        This correctly handles:
        - Causal conv state (ring buffer, d_conv width)
        - SSM hidden state (d_state dimensions)
        - No approximation needed (unlike S-SSSM dwconv skip)
        
        O(1) time, O(1) memory per step.
        
        Args:  char_id: [B] or scalar
        Returns: log_probs: [B, V]
        """
        if not hasattr(self, '_inference_params') or self._inference_params is None:
            raise RuntimeError("Call reset_state() before forward_step()")
        
        if char_id.dim() == 0:
            char_id = char_id.unsqueeze(0)
        
        # Embed: [B] -> [B, D]
        x = self.embedding(char_id)  # no dropout at inference
        
        for i, (mamba, norm, drop) in enumerate(zip(self.layers, self.norms, self.dropouts)):
            x_norm = norm(x)
            
            # CRITICAL: .step() requires [B, 1, D] shape
            x_step = x_norm.unsqueeze(1)  # [B, D] -> [B, 1, D]
            
            conv_state, ssm_state = self._inference_params[i]
            
            # Native Mamba .step() — states updated in-place
            y, _, _ = mamba.step(x_step, conv_state, ssm_state)
            # y shape: [B, D] (mamba squeezes the L=1 dim internally)
            # conv_state and ssm_state are modified in-place (no need to reassign)
            
            x = x + y  # residual (no dropout at inference)
        
        x = self.final_norm(x)
        logits = self.output_head(x)
        return F.log_softmax(logits, dim=-1)
    
    # ----- Scoring utilities -----
    
    def score_sequence(self, char_ids: List[int], device=None) -> float:
        """Score complete sequence (batch mode). Returns total log P."""
        if len(char_ids) == 0:
            return 0.0
        if device is None:
            device = next(self.parameters()).device
        
        with torch.no_grad():
            input_seq = [BLANK_TOKEN] + char_ids[:-1]
            target_seq = char_ids
            input_t = torch.LongTensor([input_seq]).to(device)
            log_probs = self.get_log_probs(input_t)
            
            total = 0.0
            for t, tgt in enumerate(target_seq):
                if 0 <= tgt < self.vocab_size:
                    total += log_probs[0, t, tgt].item()
            return total
    
    def score_sequence_detailed(self, char_ids: List[int], device=None):
        """Score with per-character breakdown. Returns (total, per_char_details).
        
        Each entry in per_char_details is a dict with:
          - char: the character
          - log_prob: log P(char | context)
          - prob: P(char | context)
          - top3: list of (char, prob) for the 3 most likely next characters
        """
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
                    
                    # Top 3 predictions at this position
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
    
    def score_sequence_streaming(self, char_ids: List[int], device=None) -> Tuple[float, List[float]]:
        """Score using streaming step mode. Returns (total, per_char_scores).
        
        This exercises the exact same code path used in real-time decoding.
        Compare with score_sequence() to verify streaming correctness.
        """
        if len(char_ids) == 0:
            return 0.0, []
        if device is None:
            device = next(self.parameters()).device
        
        self.eval()
        self.reset_state(batch_size=1, device=device)
        total = 0.0
        per_char = []
        
        with torch.no_grad():
            prev = torch.tensor(BLANK_TOKEN, device=device)
            for tgt in char_ids:
                lp = self.forward_step(prev)  # [1, V]
                if 0 <= tgt < self.vocab_size:
                    s = lp[0, tgt].item()
                    total += s
                    per_char.append(s)
                prev = torch.tensor(tgt, device=device)
        
        return total, per_char


# ============================================================================
# CHARACTER S-SSSM LM (for ILM — matches ASR encoder architecture)
# (uses non-uniform layer-wise initialization; legacy "hierarchical"
#  identifiers kept for checkpoint compatibility)
# ============================================================================

class HierarchicalSelectiveSSMLayer(nn.Module):
    """S-SSSM layer (copied from v7.6 for ILM compatibility).
    
    Class name kept for checkpoint compatibility; 'hierarchical' refers
    to non-uniform layer-wise initialization of decay/timescale parameters.
    """
    
    def __init__(self, d_model, dropout=0.1, layer_idx=0, n_layers=1,
                 use_hierarchical=False, use_gating=True, hier_params='abc'):
        super().__init__()
        self.d_model = d_model
        self.use_gating = use_gating
        self.norm = nn.LayerNorm(d_model)
        
        if use_gating:
            self.in_proj = nn.Linear(d_model, d_model * 2)
        else:
            self.in_proj = nn.Linear(d_model, d_model)
        
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.dwconv = nn.Conv1d(d_model, d_model, kernel_size=5, padding=2, groups=d_model)
        
        self.Wa = nn.Linear(d_model, d_model, bias=True)
        self.Wb = nn.Linear(d_model, d_model, bias=True)
        self.Wc = nn.Linear(d_model, d_model, bias=True)
        self.Wd = nn.Linear(d_model, d_model, bias=True)
        
        self.a0 = nn.Parameter(torch.zeros(d_model))
        self.b0 = nn.Parameter(torch.zeros(d_model))
        self.c0 = nn.Parameter(torch.zeros(d_model))
        self.d0 = nn.Parameter(torch.zeros(d_model))
        
        if use_hierarchical:
            with torch.no_grad():
                denom = max(1, n_layers - 1)
                progress = layer_idx / denom
                hc = HIER_CONFIG
                
                if 'a' in hier_params:
                    a_mean = hc['A0_EARLY'] * ((hc['A0_LATE'] / hc['A0_EARLY']) ** progress)
                    self.a0.copy_(torch.clamp(torch.normal(mean=a_mean, std=0.08, size=(d_model,)), -0.95, 0.95))
                if 'b' in hier_params:
                    b_mean = hc['B0_EARLY'] * ((hc['B0_LATE'] / hc['B0_EARLY']) ** progress)
                    self.b0.copy_(torch.clamp(torch.normal(mean=b_mean, std=0.05, size=(d_model,)), 0.01, 0.95))
                if 'c' in hier_params:
                    c_mean = hc['C0_EARLY'] + (hc['C0_LATE'] - hc['C0_EARLY']) * progress
                    self.c0.copy_(torch.clamp(torch.normal(mean=c_mean, std=0.05, size=(d_model,)), -0.95, 0.95))
                self.d0.copy_(torch.normal(0.0, std=0.02, size=(d_model,)))
    
    def forward(self, x):
        B, T, D = x.shape
        x_norm = self.norm(x)
        
        if self.use_gating:
            xz = self.in_proj(x_norm)
            x_proj, z = xz.chunk(2, dim=-1)
        else:
            x_proj = self.in_proj(x_norm)
        
        x_conv = self.dwconv(x_proj.transpose(1, 2)).transpose(1, 2)
        
        at = torch.tanh(self.a0 + self.Wa(x_conv))
        bt = torch.sigmoid(self.b0 + self.Wb(x_conv))
        ct = torch.sigmoid(self.c0 + self.Wc(x_conv))
        dt = self.Wd(x_conv) + self.d0
        
        s = torch.zeros(B, self.d_model, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(T):
            xt = x_proj[:, t, :]
            s = at[:, t, :] * s + bt[:, t, :] * xt
            yt = ct[:, t, :] * s + dt[:, t, :] * xt
            ys.append(yt)
        y = torch.stack(ys, dim=1)
        
        if self.use_gating:
            y = y * torch.sigmoid(z)
        
        y = self.out_proj(y)
        return x + self.dropout(y)


class SSMEncoder(nn.Module):
    """Stack of S-SSSM layers (for ILM)."""
    def __init__(self, d_model=128, n_layers=4, dropout=0.1,
                 use_hierarchical=False, use_gating=True, hier_params='abc'):
        super().__init__()
        self.layers = nn.ModuleList([
            HierarchicalSelectiveSSMLayer(
                d_model, dropout, i, n_layers,
                use_hierarchical, use_gating, hier_params,
            ) for i in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.final_norm(x)


class CharSSMLM(nn.Module):
    """Character-level S-SSSM LM (for ILM — matches encoder arch)."""
    
    def __init__(self, vocab_size=VOCAB_SIZE, d_model=128, n_layers=4,
                 dropout=0.1, use_hierarchical=False, use_gating=True, hier_params='abc'):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers_count = n_layers
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.embed_dropout = nn.Dropout(dropout)
        self.encoder = SSMEncoder(d_model, n_layers, dropout,
                                   use_hierarchical, use_gating, hier_params)
        self.output_head = nn.Linear(d_model, vocab_size)
        
        total = sum(p.numel() for p in self.parameters())
        print(f"\n[CharSSMLM] {total:,} params (d={d_model}, L={n_layers})")
    
    def forward(self, input_ids):
        x = self.embed_dropout(self.embedding(input_ids))
        x = self.encoder(x)
        return self.output_head(x)
    
    def get_log_probs(self, input_ids):
        return F.log_softmax(self.forward(input_ids), dim=-1)
    
    def score_sequence(self, char_ids, device=None):
        if len(char_ids) == 0:
            return 0.0
        if device is None:
            device = next(self.parameters()).device
        with torch.no_grad():
            inp = torch.LongTensor([[BLANK_TOKEN] + char_ids[:-1]]).to(device)
            lp = self.get_log_probs(inp)
            return sum(lp[0, t, c].item() for t, c in enumerate(char_ids) if 0 <= c < self.vocab_size)


# ============================================================================
# CHARACTER-LEVEL KENLM N-GRAM LM
# ============================================================================

class CharKenLM:
    """
    KenLM n-gram wrapper with score_sequence() interface matching CharMambaLM.
    
    KenLM treats each character as a "word" in its n-gram model. Training text
    is space-separated characters with a special token for word boundaries:
        "the cat" → "t h e ▁ c a t"
    
    This enables direct comparison between n-gram and neural (Mamba) LMs on
    identical data, isolating the contribution of long-range context.
    """
    
    # Character used to represent whitespace in KenLM's token stream
    SPACE_TOKEN = '▁'
    
    def __init__(self, arpa_path: str):
        if not _HAS_KENLM:
            raise ImportError("kenlm not installed. pip install https://github.com/kpu/kenlm/archive/master.zip")
        
        self.model = _kenlm.Model(arpa_path)
        self.order = self.model.order
        self.arpa_path = arpa_path
        self.vocab_size = VOCAB_SIZE
        self.d_model = 0
        self.n_layers_count = self.order
        
        # File size as proxy for "parameters"
        file_size_mb = os.path.getsize(arpa_path) / (1024 * 1024)
        print(f"\n[CharKenLM] {self.order}-gram from {arpa_path} ({file_size_mb:.1f} MB)")
    
    def _ids_to_kenlm_str(self, char_ids: List[int]) -> str:
        """Convert char_ids to KenLM-compatible space-separated token string."""
        tokens = []
        for cid in char_ids:
            if 0 <= cid < len(VOCAB_CHARS):
                c = VOCAB_CHARS[cid]
                tokens.append(self.SPACE_TOKEN if c == ' ' else c)
        return ' '.join(tokens)
    
    def score_sequence(self, char_ids: List[int], device=None) -> float:
        """
        Score a character sequence. Returns total log prob (natural log).
        
        KenLM returns log10 scores; we convert to natural log (ln) to match
        the neural LM interface: log10(x) × ln(10) = ln(x)
        """
        if not char_ids:
            return 0.0
        char_str = self._ids_to_kenlm_str(char_ids)
        # KenLM .score() returns total log10 probability
        log10_score = self.model.score(char_str, bos=True, eos=True)
        return log10_score * math.log(10)  # Convert log10 → ln
    
    def score_sequence_detailed(self, char_ids: List[int], device=None):
        """
        Score with per-character breakdown (for score_analysis mode).
        Returns (total_ln, details_list).
        """
        if not char_ids:
            return 0.0, []
        
        char_str = self._ids_to_kenlm_str(char_ids)
        total_ln = 0.0
        details = []
        
        # full_scores() yields (log10_prob, ngram_length, oov) per token
        words = char_str.split()
        for i, (log10_p, ngram_len, is_oov) in enumerate(self.model.full_scores(char_str, bos=True, eos=False)):
            if i >= len(words):
                break  # Skip EOS score
            ln_p = log10_p * math.log(10)
            prob = math.pow(10, log10_p)
            total_ln += ln_p
            
            # Map token back to character
            token = words[i]
            char = ' ' if token == self.SPACE_TOKEN else token
            
            # KenLM doesn't give top-k alternatives, so fill with placeholder
            details.append({
                'char': char,
                'log_prob': ln_p,
                'prob': min(prob, 1.0),
                'top3': [(char, min(prob, 1.0)), ('?', 0.0), ('?', 0.0)],
            })
        
        return total_ln, details
    
    def eval(self):
        """No-op for interface compatibility."""
        return self
    
    def parameters(self):
        """Empty iterator for interface compatibility."""
        return iter([])


def load_lm(path, device='cuda'):
    """Load any LM (auto-detects type from file/checkpoint).
    
    Accepts:
    - Direct path to .pt file (neural LM)
    - Directory containing *_best.pt (experiment subfolder)
    - Path to .arpa or .binary file (KenLM n-gram)
    - Path to .pkl file (Python n-gram)
    """
    # KenLM detection
    if path.endswith('.arpa') or path.endswith('.binary'):
        return CharKenLM(path)
    
    # Python n-gram detection
    if path.endswith('.pkl'):
        from IKT590_train_char_ngram import CharNgramLM
        return CharNgramLM.load(path)
    
    # If path is a directory, check for various file types
    if os.path.isdir(path):
        arpa_files = [f for f in os.listdir(path) if f.endswith('.arpa')]
        pkl_files = [f for f in os.listdir(path) if f.endswith('.pkl')]
        pt_files = [f for f in os.listdir(path) if f.endswith('_best.pt')]
        if arpa_files and not pt_files:
            return CharKenLM(os.path.join(path, arpa_files[0]))
        if pkl_files and not pt_files:
            from IKT590_train_char_ngram import CharNgramLM
            return CharNgramLM.load(os.path.join(path, pkl_files[0]))
        if pt_files:
            pt_path = os.path.join(path, pt_files[0])
        else:
            raise FileNotFoundError(f"No *_best.pt, .arpa, or .pkl found in {path}")
    else:
        pt_path = path
    
    ckpt = torch.load(pt_path, map_location=device, weights_only=False)
    cfg = ckpt['config']
    
    if cfg['model_type'] == 'mamba3':
        from IKT590_train_CharMamba3_LM_Triton_v1_1 import CharMamba3LM
        model = CharMamba3LM(
            vocab_size=cfg['vocab_size'],
            d_model=cfg['d_model'],
            n_layers=cfg['n_layers'],
            d_state=cfg.get('d_state', 64),
            headdim=cfg.get('headdim', 64),
            is_mimo=cfg.get('is_mimo', False),
        )
    elif cfg['model_type'] == 'mamba':
        model = CharMambaLM(
            vocab_size=cfg['vocab_size'],
            d_model=cfg['d_model'],
            n_layers=cfg['n_layers'],
            d_state=cfg.get('d_state', 16),
        )
    else:
        model = CharSSMLM(
            vocab_size=cfg['vocab_size'],
            d_model=cfg['d_model'],
            n_layers=cfg['n_layers'],
        )
    
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device).eval()
    print(f"[LoadLM] {cfg['label']} ({cfg['model_type']}) from {pt_path} (loss={ckpt['best_loss']:.4f})")
    return model


# ============================================================================
# CTC BEAM SEARCH
# ============================================================================

def ctc_prefix_beam_search(log_probs, beam_width=10, blank_id=BLANK_TOKEN):
    """CTC prefix beam search. Returns N-best list for rescoring."""
    T, V = log_probs.shape
    NEG_INF = float('-inf')
    beams = {(): (0.0, NEG_INF)}
    
    for t in range(T):
        new_beams = {}
        for prefix, (p_b, p_nb) in beams.items():
            p_tot = np.logaddexp(p_b, p_nb)
            if p_tot < -300:
                continue
            
            new_p_b = np.logaddexp(p_b + log_probs[t, blank_id], p_nb + log_probs[t, blank_id])
            if prefix in new_beams:
                ob, onb = new_beams[prefix]
                new_beams[prefix] = (np.logaddexp(ob, new_p_b), onb)
            else:
                new_beams[prefix] = (new_p_b, NEG_INF)
            
            for c in range(V):
                if c == blank_id:
                    continue
                if len(prefix) > 0 and c == prefix[-1]:
                    emit = p_b + log_probs[t, c]
                    cont = p_nb + log_probs[t, c]
                    if prefix in new_beams:
                        ob, onb = new_beams[prefix]
                        new_beams[prefix] = (ob, np.logaddexp(onb, cont))
                    else:
                        new_beams[prefix] = (NEG_INF, cont)
                    np2 = prefix + (c,)
                    if np2 in new_beams:
                        ob, onb = new_beams[np2]
                        new_beams[np2] = (ob, np.logaddexp(onb, emit))
                    else:
                        new_beams[np2] = (NEG_INF, emit)
                else:
                    val = np.logaddexp(p_b + log_probs[t, c], p_nb + log_probs[t, c])
                    np2 = prefix + (c,)
                    if np2 in new_beams:
                        ob, onb = new_beams[np2]
                        new_beams[np2] = (ob, np.logaddexp(onb, val))
                    else:
                        new_beams[np2] = (NEG_INF, val)
        
        scored = [(k, np.logaddexp(pb, pnb), pb, pnb) for k, (pb, pnb) in new_beams.items()]
        scored.sort(key=lambda x: x[1], reverse=True)
        beams = {s[0]: (s[2], s[3]) for s in scored[:beam_width]}
    
    results = [(list(k), np.logaddexp(pb, pnb)) for k, (pb, pnb) in beams.items()]
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def rescore_nbest_ilme(nbest, elm=None, ilm=None, alpha=0.3, beta=0.1, gamma=0.0, device='cpu'):
    """ILME rescoring: CTC + alpha*ELM/|y| - beta*ILM/|y| + gamma*|y|"""
    rescored = []
    for char_ids, ctc_score in nbest:
        length = max(1, len(char_ids))
        elm_s = elm.score_sequence(char_ids, device=device) if elm and char_ids else 0.0
        ilm_s = ilm.score_sequence(char_ids, device=device) if ilm and char_ids else 0.0
        final = ctc_score + alpha * (elm_s / length) - beta * (ilm_s / length) + gamma * length
        rescored.append((char_ids, final, {'ctc': ctc_score, 'elm': elm_s, 'ilm': ilm_s}))
    rescored.sort(key=lambda x: x[1], reverse=True)
    return rescored


# ============================================================================
# METRICS
# ============================================================================

def compute_detailed_metrics(references, hypotheses):
    """WER with S/D/I breakdown and P/R/F1."""
    total_S, total_D, total_I, total_H = 0, 0, 0, 0
    per_utt = []
    
    for ref, hyp in zip(references, hypotheses):
        if not ref.strip(): continue
        if not hyp.strip(): hyp = " "
        try:
            from jiwer import process_words
            out = process_words(ref, hyp)
            S, D, I, H = out.substitutions, out.deletions, out.insertions, out.hits
        except (ImportError, AttributeError):
            w = jiwer_wer(ref, hyp)
            n = len(ref.split())
            err = int(round(w * n))
            S, D, I, H = err, 0, 0, max(0, n - err)
        total_S += S; total_D += D; total_I += I; total_H += H
        N = S + D + H
        per_utt.append((S + D + I) / max(N, 1) * 100)
    
    N = total_S + total_D + total_H
    C = total_H
    P = C / max(C + total_S + total_I, 1)
    R = C / max(C + total_S + total_D, 1)
    F1 = 2 * P * R / max(P + R, 1e-9)
    wer = (total_S + total_D + total_I) / max(N, 1)
    
    res = {'wer': wer*100, 'sub': total_S/max(N,1)*100, 'del': total_D/max(N,1)*100,
           'ins': total_I/max(N,1)*100, 'precision': P*100, 'recall': R*100, 'f1': F1*100,
           'wer_median': np.median(per_utt) if per_utt else 0,
           'wer_p90': np.percentile(per_utt, 90) if per_utt else 0}
    print(f"    WER={res['wer']:.2f}% (S={res['sub']:.1f}% D={res['del']:.1f}% I={res['ins']:.1f}%) "
          f"P={res['precision']:.1f}% R={res['recall']:.1f}% F1={res['f1']:.1f}%")
    return res


# ============================================================================
# RTFX MEASUREMENT
# ============================================================================

@torch.no_grad()
def measure_rtfx(asr_model, dataloader, decode_mode='greedy', beam_width=10,
                 elm=None, ilm=None, alpha=0.0, beta=0.0, gamma=0.0,
                 device='cuda', warmup=3):
    """Measure RTFX and latency."""
    asr_model.eval()
    total_audio, total_compute = 0.0, 0.0
    per_utt = []
    
    for bi, batch in enumerate(dataloader):
        feat = batch['features'].to(device)
        fl = batch['feature_lengths'].to(device)
        if bi < warmup:
            asr_model(feat, fl); continue
        
        for i in range(feat.shape[0]):
            uf, ul = feat[i:i+1], fl[i:i+1]
            dur = fl[i].item() * 160 / SAMPLE_RATE  # mel frames → seconds (hop=160 @ 16kHz)
            if device == 'cuda' or (isinstance(device, torch.device) and device.type == 'cuda'):
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            logits, lengths = asr_model(uf, ul)
            
            if decode_mode == 'greedy':
                pred = logits.argmax(dim=-1)[0]
                L = int(lengths[0].item())
                last, seq = -1, []
                for p in pred[:L].tolist():
                    if p != last and p != BLANK_TOKEN and p < len(VOCAB_CHARS):
                        seq.append(p)
                    last = p
            elif decode_mode == 'beam':
                lp = F.log_softmax(logits, dim=-1)
                L = int(lengths[0].item())
                nb = ctc_prefix_beam_search(lp[0,:L,:].cpu().numpy(), beam_width)
                seq = nb[0][0] if nb else []
            elif decode_mode == 'nbest_ilme':
                lp = F.log_softmax(logits, dim=-1)
                L = int(lengths[0].item())
                nb = ctc_prefix_beam_search(lp[0,:L,:].cpu().numpy(), beam_width)
                r = rescore_nbest_ilme(nb, elm, ilm, alpha, beta, gamma, device)
                seq = r[0][0] if r else []
            
            if device == 'cuda' or (isinstance(device, torch.device) and device.type == 'cuda'):
                torch.cuda.synchronize()
            comp = time.perf_counter() - t0
            total_audio += dur; total_compute += comp
            per_utt.append({'dur': dur, 'comp': comp, 'rtfx': dur/max(comp,1e-9)})
    
    rtfx = total_audio / max(total_compute, 1e-9)
    lats = [s['comp'] for s in per_utt]
    res = {'decode_mode': decode_mode, 'rtfx': rtfx, 'rtf': 1/rtfx,
           'mean_ms': np.mean(lats)*1000 if lats else 0,
           'p95_ms': np.percentile(lats, 95)*1000 if lats else 0,
           'n_utt': len(per_utt)}
    print(f"  [RTFX] {decode_mode}: RTFX={rtfx:.1f}x (RTF={1/rtfx:.4f}), lat_mean={res['mean_ms']:.1f}ms, "
          f"p95={res['p95_ms']:.1f}ms ({len(per_utt)} utts, {total_audio:.0f}s audio in {total_compute:.0f}s compute)")
    return res


# ============================================================================
# HALLUCINATION ANALYSIS
# ============================================================================

@torch.no_grad()
def analyze_hallucinations(asr_model, elm, dataloader, device='cuda', max_utt=500):
    """CTC confidence × LM probability quadrant analysis.
    
    Uses batch-mode LM scoring (not streaming .step()) for compatibility.
    """
    asr_model.eval(); elm.eval()
    all_chars = []
    n_utt = 0
    
    for batch in dataloader:
        if n_utt >= max_utt: break
        feat = batch['features'].to(device)
        fl = batch['feature_lengths'].to(device)
        logits, lengths = asr_model(feat, fl)
        probs = F.softmax(logits, dim=-1)
        
        for i in range(feat.shape[0]):
            if n_utt >= max_utt: break
            L = int(lengths[i].item())
            pred = logits[i,:L,:].argmax(dim=-1)
            conf = probs[i,:L,:].max(dim=-1).values
            
            # Collect emitted characters (CTC collapse)
            emitted = []
            last = -1
            for t in range(L):
                tok = pred[t].item()
                if tok != last and tok != BLANK_TOKEN and tok < len(VOCAB_CHARS):
                    emitted.append({'char_id': tok, 'ctc_conf': conf[t].item()})
                last = tok
            
            # Score with ELM using batch mode (no streaming needed)
            if emitted:
                char_ids = [e['char_id'] for e in emitted]
                with torch.no_grad():
                    inp = torch.LongTensor([[BLANK_TOKEN] + char_ids[:-1]]).to(device)
                    log_probs = elm.get_log_probs(inp)  # [1, T, V]
                    for t, ec in enumerate(emitted):
                        ec['lm_prob'] = torch.exp(log_probs[0, t, ec['char_id']]).item()
            
            for ec in emitted:
                hi_ctc, hi_lm = ec['ctc_conf'] > 0.5, ec['lm_prob'] > 0.1
                if hi_ctc and hi_lm: ec['q'] = 'confident_correct'
                elif hi_ctc: ec['q'] = 'acoustic_artifact'
                elif hi_lm: ec['q'] = 'lm_hallucination'
                else: ec['q'] = 'noise'
            all_chars.extend(emitted)
            n_utt += 1
    
    n = len(all_chars)
    counts = {}
    for c in all_chars:
        counts[c['q']] = counts.get(c['q'], 0) + 1
    
    print(f"\n  [Hallucination] {n} chars, {n_utt} utterances:")
    for q in ['confident_correct', 'acoustic_artifact', 'lm_hallucination', 'noise']:
        pct = counts.get(q, 0) / max(n, 1) * 100
        print(f"    {q:25s}: {pct:5.1f}% ({counts.get(q,0):,})")
    return {'n_chars': n, 'n_utt': n_utt, 'counts': counts,
            'pct': {k: v/max(n,1)*100 for k,v in counts.items()}}



# ============================================================================
# SCORE ANALYSIS (detailed scoring breakdown for thesis)
# ============================================================================

@torch.no_grad()
def run_score_analysis(asr_model, loader, elm, beam_width=10, alpha=32.0,
                       device='cuda', n_examples=10, save_path=None):
    """
    Detailed scoring breakdown showing how CTC + ELM rescoring works.
    
    Generates:
    1. Text log with full score tables for sample utterances
    2. PDF report with visualisations for thesis figures
    
    For each sample utterance, shows:
    - All beam hypotheses with CTC, ELM, and combined scores
    - Whether rescoring changed the winning hypothesis
    - Per-character ELM probability breakdown
    - LM top-3 predictions at each position
    """
    asr_model.eval(); elm.eval()
    
    out_dir = os.path.dirname(save_path) if save_path else "."
    log_path = save_path
    
    lines = []
    def log(msg=""):
        print(msg)
        lines.append(msg)
    
    log(f"\n{'='*90}")
    log(f"  SCORE ANALYSIS: Beam Search + Mamba ELM Rescoring")
    log(f"  ASR Encoder: d_model={asr_model.model.subsample.conv2.out_channels if hasattr(asr_model, 'model') else '?'}")
    log(f"  ELM: d_model={elm.d_model}, layers={elm.n_layers_count}, params={sum(p.numel() for p in elm.parameters()):,}")
    log(f"  Alpha: {alpha}, Beam width: {beam_width}")
    log(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"{'='*90}")
    
    # ---- Collect examples ----
    examples_changed = []
    examples_unchanged = []
    n_total, n_reranked = 0, 0
    
    all_ctc_scores, all_elm_per_len = [], []
    all_ctc_spreads, all_elm_spreads = [], []
    all_combined_scores = []
    rerank_details = []  # For PDF plots
    per_char_probs_collection = []  # For heatmap
    
    for batch in loader:
        if len(examples_changed) + len(examples_unchanged) >= n_examples * 3:
            break
        feat = batch['features'].to(device)
        fl = batch['feature_lengths'].to(device)
        logits, lengths = asr_model(feat, fl)
        lp = F.log_softmax(logits, dim=-1)
        
        for i in range(feat.shape[0]):
            L = int(lengths[i].item())
            ref = ids_to_text(batch['text_ids'][i,:batch['text_lengths'][i]].tolist())
            
            # Greedy decode
            pred = lp[i,:L,:].argmax(dim=-1)
            last, greedy_seq = -1, []
            for p in pred.tolist():
                if p != last and p != BLANK_TOKEN and p < len(VOCAB_CHARS):
                    greedy_seq.append(p)
                last = p
            greedy_text = ids_to_text(greedy_seq)
            
            # Beam search
            nb = ctc_prefix_beam_search(lp[i,:L,:].cpu().numpy(), beam_width)
            if not nb:
                continue
            
            # Score all hypotheses
            scored = []
            for char_ids, ctc_score in nb:
                length = max(1, len(char_ids))
                elm_total, elm_details = elm.score_sequence_detailed(char_ids, device=device)
                elm_per_len = elm_total / length
                combined = ctc_score + alpha * elm_per_len
                scored.append({
                    'char_ids': char_ids,
                    'text': ids_to_text(char_ids),
                    'length': length,
                    'ctc_score': ctc_score,
                    'elm_total': elm_total,
                    'elm_per_len': elm_per_len,
                    'elm_contribution': alpha * elm_per_len,
                    'combined': combined,
                    'elm_details': elm_details,
                })
            
            by_ctc = sorted(scored, key=lambda x: x['ctc_score'], reverse=True)
            by_combined = sorted(scored, key=lambda x: x['combined'], reverse=True)
            
            ctc_winner = by_ctc[0]['text']
            elm_winner = by_combined[0]['text']
            changed = ctc_winner != elm_winner
            
            n_total += 1
            if changed:
                n_reranked += 1
            
            ctc_vals = [s['ctc_score'] for s in scored]
            elm_vals = [s['elm_per_len'] for s in scored]
            combined_vals = [s['combined'] for s in scored]
            all_ctc_scores.extend(ctc_vals)
            all_elm_per_len.extend(elm_vals)
            all_combined_scores.extend(combined_vals)
            if len(scored) >= 2:
                all_ctc_spreads.append(max(ctc_vals) - min(ctc_vals))
                all_elm_spreads.append(max(elm_vals) - min(elm_vals))
            
            example = {
                'ref': ref, 'greedy': greedy_text, 'scored': scored,
                'by_ctc': by_ctc, 'by_combined': by_combined,
                'changed': changed, 'ctc_winner': ctc_winner, 'elm_winner': elm_winner,
            }
            
            # Collect rerank info for plots
            if changed:
                rerank_details.append({
                    'ref': ref,
                    'ctc_winner': ctc_winner, 'elm_winner': elm_winner,
                    'ctc_scores': ctc_vals, 'elm_scores': elm_vals,
                    'combined_scores': combined_vals,
                    'hypotheses': [s['text'] for s in by_combined],
                })
            
            # Collect per-char probs for heatmap (first few examples)
            if by_combined and by_combined[0]['elm_details'] and len(per_char_probs_collection) < 6:
                per_char_probs_collection.append({
                    'text': by_combined[0]['text'],
                    'probs': [d['prob'] for d in by_combined[0]['elm_details']],
                    'chars': [d['char'] for d in by_combined[0]['elm_details']],
                    'ref': ref,
                    'changed': changed,
                })
            
            if changed and len(examples_changed) < n_examples:
                examples_changed.append(example)
            elif not changed and len(examples_unchanged) < n_examples:
                examples_unchanged.append(example)
    
    # ================================================================
    # TEXT LOG
    # ================================================================
    
    log(f"\n\n{'='*90}")
    log(f"  EXAMPLES WHERE ELM RESCORING CHANGED THE TOP HYPOTHESIS")
    log(f"  ({n_reranked}/{n_total} utterances re-ranked = {100*n_reranked/max(n_total,1):.1f}%)")
    log(f"{'='*90}")
    
    for ei, ex in enumerate(examples_changed[:5]):
        _print_example(log, ei+1, ex, alpha, show_per_char=True)
    
    log(f"\n\n{'='*90}")
    log(f"  EXAMPLES WHERE ELM CONFIRMED THE CTC TOP HYPOTHESIS")
    log(f"{'='*90}")
    
    for ei, ex in enumerate(examples_unchanged[:3]):
        _print_example(log, ei+1, ex, alpha, show_per_char=False)
    
    # ---- Aggregate stats ----
    ctc_arr = np.array(all_ctc_scores)
    elm_arr = np.array(all_elm_per_len)
    combined_arr = np.array(all_combined_scores)
    
    log(f"\n\n{'='*90}")
    log(f"  AGGREGATE SCORE STATISTICS ({n_total} utterances, {len(ctc_arr)} hypotheses)")
    log(f"{'='*90}")
    
    log(f"\n  CTC log-prob scores (sum over all frames):")
    log(f"    Range:  [{ctc_arr.min():.2f}, {ctc_arr.max():.2f}]")
    log(f"    Mean:   {ctc_arr.mean():.2f}  (std: {ctc_arr.std():.2f})")
    log(f"    Median: {np.median(ctc_arr):.2f}")
    
    log(f"\n  ELM log-prob per character:")
    log(f"    Range:  [{elm_arr.min():.4f}, {elm_arr.max():.4f}]")
    log(f"    Mean:   {elm_arr.mean():.4f}  (std: {elm_arr.std():.4f})")
    
    log(f"\n  Score composition at alpha={alpha}:")
    log(f"    Typical CTC score:           {ctc_arr.mean():.2f}")
    log(f"    Typical ELM contribution:    {alpha * elm_arr.mean():.2f}  (alpha × {elm_arr.mean():.4f})")
    log(f"    Typical combined score:      {combined_arr.mean():.2f}")
    log(f"    ELM share of combined:       {abs(alpha*elm_arr.mean()) / (abs(ctc_arr.mean()) + abs(alpha*elm_arr.mean())) * 100:.1f}%")
    
    if all_ctc_spreads:
        log(f"\n  Score spread across beam hypotheses (median):")
        log(f"    CTC spread:                  {np.median(all_ctc_spreads):.4f}")
        log(f"    ELM/len spread:              {np.median(all_elm_spreads):.4f}")
        log(f"    ELM × alpha spread:          {alpha * np.median(all_elm_spreads):.4f}")
        ratio = alpha * np.median(all_elm_spreads) / np.median(all_ctc_spreads)
        log(f"    Ratio (ELM×alpha / CTC):     {ratio:.1%}")
    
    log(f"\n  Re-ranking summary:")
    log(f"    Utterances processed:        {n_total}")
    log(f"    Re-ranked by ELM:            {n_reranked} ({100*n_reranked/max(n_total,1):.1f}%)")
    log(f"    Confirmed CTC choice:        {n_total - n_reranked} ({100*(n_total-n_reranked)/max(n_total,1):.1f}%)")
    
    log(f"\n  How to read the combined score:")
    log(f"    combined = CTC_score + alpha * (ELM_total / |hypothesis|)")
    log(f"    Example: {ctc_arr[0]:.3f} + {alpha} × ({elm_arr[0]:.4f}) = {combined_arr[0]:.3f}")
    log(f"    The CTC score is a large negative number (sum of log-probs over ~100-500 frames).")
    log(f"    The ELM score is a small negative number (avg log-prob per character, ~20-100 chars).")
    log(f"    Alpha={alpha} amplifies the ELM so it can influence the ranking despite its small scale.")
    log(f"{'='*90}\n")
    
    # Save text log
    with open(log_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"\n  Text log saved: {log_path}")
    
    # ================================================================
    # PDF FIGURES (individual files for thesis)
    # ================================================================
    
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        saved_figs = []
        
        # --- Figure 1: Score Distributions (2×2) ---
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Score Distributions: S-SSSM Encoder + Mamba ELM (α={alpha})',
                    fontsize=14, fontweight='bold')
        
        ax = axes[0, 0]
        ax.hist(ctc_arr, bins=50, color='#2196F3', alpha=0.8, edgecolor='white')
        ax.set_xlabel('CTC Log-Probability Score')
        ax.set_ylabel('Count')
        ax.set_title('CTC Score Distribution')
        ax.axvline(ctc_arr.mean(), color='red', linestyle='--', label=f'Mean: {ctc_arr.mean():.1f}')
        ax.legend(fontsize=9)
        
        ax = axes[0, 1]
        ax.hist(elm_arr, bins=50, color='#4CAF50', alpha=0.8, edgecolor='white')
        ax.set_xlabel('ELM Log-Prob / Length')
        ax.set_ylabel('Count')
        ax.set_title('ELM Per-Character Score Distribution')
        ax.axvline(elm_arr.mean(), color='red', linestyle='--', label=f'Mean: {elm_arr.mean():.4f}')
        ax.legend(fontsize=9)
        
        ax = axes[1, 0]
        ax.hist(combined_arr, bins=50, color='#FF9800', alpha=0.8, edgecolor='white')
        ax.set_xlabel('Combined Score (CTC + α·ELM/len)')
        ax.set_ylabel('Count')
        ax.set_title(f'Combined Score Distribution (α={alpha})')
        ax.axvline(combined_arr.mean(), color='red', linestyle='--', label=f'Mean: {combined_arr.mean():.1f}')
        ax.legend(fontsize=9)
        
        ax = axes[1, 1]
        if all_ctc_spreads:
            spread_data = [all_ctc_spreads, [alpha * s for s in all_elm_spreads]]
            bp = ax.boxplot(spread_data, labels=['CTC Spread', f'ELM×α Spread'],
                           patch_artist=True, widths=0.5)
            bp['boxes'][0].set_facecolor('#2196F3')
            bp['boxes'][1].set_facecolor('#4CAF50')
            ax.set_ylabel('Score Spread (max − min)')
            ax.set_title('Score Spread Across Beam Hypotheses')
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        p = os.path.join(fig_dir, 'fig_score_distributions.pdf')
        fig.savefig(p, bbox_inches='tight')
        plt.close(fig)
        saved_figs.append(p)
        
        # --- Figure 2: Score Composition Pie ---
        fig, ax = plt.subplots(figsize=(7, 6))
        ctc_mag = abs(ctc_arr.mean())
        elm_mag = abs(alpha * elm_arr.mean())
        sizes = [ctc_mag, elm_mag]
        labels = [f'CTC\n({ctc_arr.mean():.1f})', f'ELM × α\n({alpha*elm_arr.mean():.1f})']
        colors = ['#2196F3', '#4CAF50']
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors,
                                           autopct='%1.1f%%', startangle=90,
                                           textprops={'fontsize': 11})
        ax.set_title(f'Score Composition at α={alpha}\n(Typical combined score: {combined_arr.mean():.1f})',
                    fontsize=13)
        plt.tight_layout()
        p = os.path.join(fig_dir, 'fig_score_composition_pie.pdf')
        fig.savefig(p, bbox_inches='tight')
        plt.close(fig)
        saved_figs.append(p)
        
        # --- Figure 3: CTC vs ELM Scatter ---
        fig, ax = plt.subplots(figsize=(8, 6))
        n_plot = min(len(ctc_arr), 3000)
        idx = np.random.choice(len(ctc_arr), n_plot, replace=False)
        ax.scatter(ctc_arr[idx], elm_arr[idx], alpha=0.3, s=8, c='#555')
        ax.set_xlabel('CTC Score (sum of frame log-probs)', fontsize=11)
        ax.set_ylabel('ELM Score / Length (avg char log-prob)', fontsize=11)
        ax.set_title('CTC vs ELM Score per Beam Hypothesis', fontsize=13)
        if len(ctc_arr) > 2:
            r = np.corrcoef(ctc_arr[:n_plot], elm_arr[:n_plot])[0, 1]
            ax.text(0.05, 0.95, f'Pearson r = {r:.3f}\n{n_plot:,} hypotheses',
                   transform=ax.transAxes, fontsize=11, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        plt.tight_layout()
        p = os.path.join(fig_dir, 'fig_ctc_vs_elm_scatter.pdf')
        fig.savefig(p, bbox_inches='tight')
        plt.close(fig)
        saved_figs.append(p)
        
        # --- Figure 4: Per-Character LM Probability (individual per example) ---
        for pi, pc in enumerate(per_char_probs_collection[:6]):
            chars = pc['chars']
            probs = pc['probs']
            n_chars = len(chars)
            if n_chars == 0:
                continue
            
            max_display = 80
            if n_chars > max_display:
                chars = chars[:max_display]
                probs = probs[:max_display]
                n_chars = max_display
            
            fig_width = max(10, min(20, n_chars * 0.25))
            fig, ax = plt.subplots(figsize=(fig_width, 3.5))
            
            colors_bar = ['#4CAF50' if p > 0.5 else '#FF9800' if p > 0.1 else '#F44336'
                          for p in probs]
            ax.bar(range(n_chars), probs, color=colors_bar, edgecolor='white', width=0.8)
            
            display_chars = [repr(c)[1:-1] if c == ' ' else c for c in chars]
            ax.set_xticks(range(n_chars))
            ax.set_xticklabels(display_chars, fontsize=6, fontfamily='monospace')
            ax.set_ylabel('P(char | context)', fontsize=10)
            ax.set_ylim(0, 1.05)
            ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='P=0.5')
            ax.axhline(y=0.1, color='red', linestyle=':', alpha=0.3, label='P=0.1')
            ax.legend(fontsize=8, loc='upper right')
            
            tag = "RE-RANKED" if pc['changed'] else "confirmed"
            text_short = pc['text'][:50] + ('...' if len(pc['text']) > 50 else '')
            ax.set_title(f'Per-Character ELM Probability: "{text_short}" [{tag}]',
                        fontsize=10, loc='left')
            
            plt.tight_layout()
            p = os.path.join(fig_dir, f'fig_per_char_probs_example{pi+1}.pdf')
            fig.savefig(p, bbox_inches='tight')
            plt.close(fig)
            saved_figs.append(p)
        
        # --- Figure 5: Rescoring Waterfall (individual per example) ---
        for wi, rd in enumerate(rerank_details[:4]):
            n_hyps = min(len(rd['ctc_scores']), beam_width)
            
            fig, ax = plt.subplots(figsize=(12, max(4, n_hyps * 0.5)))
            
            y_pos = np.arange(n_hyps)
            bar_height = 0.35
            
            ctc_s = np.array(rd['ctc_scores'][:n_hyps])
            comb_s = np.array(rd['combined_scores'][:n_hyps])
            ctc_norm = (ctc_s - ctc_s.min()) / max(ctc_s.max() - ctc_s.min(), 1e-8)
            comb_norm = (comb_s - comb_s.min()) / max(comb_s.max() - comb_s.min(), 1e-8)
            
            ax.barh(y_pos + bar_height/2, ctc_norm, bar_height,
                   color='#2196F3', alpha=0.7, label='CTC only')
            ax.barh(y_pos - bar_height/2, comb_norm, bar_height,
                   color='#4CAF50', alpha=0.7, label=f'CTC + ELM (α={alpha})')
            
            hyp_labels = [h[:40] + '...' if len(h) > 40 else h for h in rd['hypotheses'][:n_hyps]]
            ax.set_yticks(y_pos)
            ax.set_yticklabels(hyp_labels, fontsize=8, fontfamily='monospace')
            ax.invert_yaxis()
            ax.set_xlabel('Normalised Score (higher = better)')
            ax.legend(fontsize=9, loc='lower right')
            
            ref_short = rd['ref'][:70] + ('...' if len(rd['ref']) > 70 else '')
            ax.set_title(f'Rescoring Effect — Ref: "{ref_short}"', fontsize=10, loc='left')
            
            plt.tight_layout()
            p = os.path.join(fig_dir, f'fig_rescoring_waterfall_example{wi+1}.pdf')
            fig.savefig(p, bbox_inches='tight')
            plt.close(fig)
            saved_figs.append(p)
        
        # --- Figure 6: Alpha Sensitivity (if we have the data from the scored hypotheses) ---
        # Show how WER proxy (% of utterances where CTC top-1 changes) varies with alpha
        if all_ctc_spreads:
            test_alphas = [0, 1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 40, 50]
            rerank_pcts = []
            for test_a in test_alphas:
                n_changed = 0
                for utt_idx in range(len(all_ctc_spreads)):
                    # We don't have per-utterance data here, so use the spread ratio as proxy
                    pass
                rerank_pcts.append(None)
            
            # Instead: plot the score scale comparison across alpha
            fig, ax = plt.subplots(figsize=(8, 5))
            alphas_plot = np.linspace(0, 50, 100)
            ctc_spread_med = np.median(all_ctc_spreads)
            elm_spread_med = np.median(all_elm_spreads)
            
            ax.axhline(y=ctc_spread_med, color='#2196F3', linewidth=2, label=f'CTC spread (median: {ctc_spread_med:.2f})')
            ax.plot(alphas_plot, alphas_plot * elm_spread_med, color='#4CAF50', linewidth=2,
                   label=f'α × ELM spread')
            ax.axvline(x=alpha, color='red', linestyle='--', alpha=0.7,
                      label=f'α={alpha} (used)')
            
            # Mark where ELM×α = CTC
            crossover = ctc_spread_med / elm_spread_med if elm_spread_med > 0 else 0
            ax.axvline(x=crossover, color='gray', linestyle=':', alpha=0.5)
            ax.text(crossover + 0.5, ctc_spread_med * 1.05, f'α={crossover:.0f}\n(equal influence)',
                   fontsize=9, color='gray')
            
            ax.set_xlabel('Alpha (α)', fontsize=11)
            ax.set_ylabel('Score Spread', fontsize=11)
            ax.set_title('CTC vs ELM Score Influence as α Increases', fontsize=13)
            ax.legend(fontsize=10)
            ax.set_xlim(0, 50)
            ax.set_ylim(0, max(ctc_spread_med, 50 * elm_spread_med) * 1.2)
            
            plt.tight_layout()
            p = os.path.join(fig_dir, 'fig_alpha_scale_comparison.pdf')
            fig.savefig(p, bbox_inches='tight')
            plt.close(fig)
            saved_figs.append(p)
        
        # --- Figure 7: Summary Statistics Table ---
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis('off')
        
        table_data = [
            ['Metric', 'Value'],
            ['Utterances analysed', f'{n_total}'],
            ['Total hypotheses', f'{len(ctc_arr):,}'],
            ['Re-ranked by ELM', f'{n_reranked} ({100*n_reranked/max(n_total,1):.1f}%)'],
            ['', ''],
            ['CTC score (mean)', f'{ctc_arr.mean():.2f}'],
            ['ELM/len score (mean)', f'{elm_arr.mean():.4f}'],
            [f'ELM × α={alpha} (mean)', f'{alpha*elm_arr.mean():.2f}'],
            ['Combined score (mean)', f'{combined_arr.mean():.2f}'],
            ['ELM share of combined', f'{abs(alpha*elm_arr.mean())/(abs(ctc_arr.mean())+abs(alpha*elm_arr.mean()))*100:.1f}%'],
            ['', ''],
            ['CTC spread (median)', f'{np.median(all_ctc_spreads):.4f}' if all_ctc_spreads else 'N/A'],
            [f'ELM×α spread (median)', f'{alpha*np.median(all_elm_spreads):.4f}' if all_elm_spreads else 'N/A'],
            ['ELM×α / CTC ratio', f'{alpha*np.median(all_elm_spreads)/np.median(all_ctc_spreads):.1%}' if all_ctc_spreads else 'N/A'],
        ]
        
        table = ax.table(cellText=table_data, colWidths=[0.45, 0.35],
                       cellLoc='left', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 1.5)
        
        table[0, 0].set_facecolor('#333')
        table[0, 0].set_text_props(color='white', fontweight='bold')
        table[0, 1].set_facecolor('#333')
        table[0, 1].set_text_props(color='white', fontweight='bold')
        
        for i in range(1, len(table_data)):
            for j in range(2):
                if table_data[i][0] == '':
                    table[i, j].set_facecolor('#f0f0f0')
                    table[i, j].set_edgecolor('#f0f0f0')
                elif i % 2 == 0:
                    table[i, j].set_facecolor('#f8f8f8')
        
        ax.set_title(f'Score Analysis Summary\nS-SSSM + Mamba ELM Shallow Fusion (α={alpha})',
                    fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        p = os.path.join(fig_dir, 'fig_summary_table.pdf')
        fig.savefig(p, bbox_inches='tight')
        plt.close(fig)
        saved_figs.append(p)
        
        log(f"\n  Figures saved ({len(saved_figs)} files):")
        for fp in saved_figs:
            log(f"    {fp}")
    
    except Exception as e:
        print(f"  WARNING: Figure generation failed: {e}")
        import traceback
        traceback.print_exc()
        print(f"  Text log is still available at: {log_path}")
    
    return {'n_total': n_total, 'n_reranked': n_reranked,
            'ctc_mean': float(ctc_arr.mean()), 'elm_mean': float(elm_arr.mean()),
            'log_path': log_path, 'fig_dir': fig_dir}


def _print_example(log, idx, ex, alpha, show_per_char=True):
    """Print a single example with full score breakdown."""
    log(f"\n  {'─'*86}")
    log(f"  Example {idx}")
    log(f"  {'─'*86}")
    log(f"  Reference:     {ex['ref']}")
    log(f"  Greedy:        {ex['greedy']}")
    log(f"  CTC best:      {ex['ctc_winner']}")
    log(f"  ELM rescored:  {ex['elm_winner']}")
    if ex['changed']:
        log(f"  *** RESCORING CHANGED THE WINNER ***")
    
    # N-best table
    log(f"\n  {'Rank':>4}  {'CTC':>10}  {'ELM/len':>10}  {'α×ELM/len':>10}  {'Combined':>10}  Hypothesis")
    log(f"  {'─'*4}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*40}")
    
    by_combined = ex['by_combined']
    for ri, s in enumerate(by_combined[:10]):
        marker = " ←" if ri == 0 else ""
        ctc_rank = next(j for j, c in enumerate(ex['by_ctc']) if c['text'] == s['text'])
        rank_note = f" (CTC #{ctc_rank+1})" if ctc_rank != ri else ""
        log(f"  {ri+1:>4}  {s['ctc_score']:>10.3f}  {s['elm_per_len']:>10.4f}  "
            f"{s['elm_contribution']:>10.3f}  {s['combined']:>10.3f}  "
            f"\"{s['text']}\"{rank_note}{marker}")
    
    # Per-character breakdown for winner
    if show_per_char and by_combined and by_combined[0]['elm_details']:
        winner = by_combined[0]
        log(f"\n  Per-character ELM scoring for winner: \"{winner['text']}\"")
        log(f"  (Each row: what the LM predicted vs what appeared, P = probability the LM assigned)")
        log(f"  {'Pos':>4}  {'Char':>4}  {'logP':>8}  {'P':>7}  {'Top-1':>12}  {'Top-2':>12}  {'Top-3':>12}  Note")
        log(f"  {'─'*4}  {'─'*4}  {'─'*8}  {'─'*7}  {'─'*12}  {'─'*12}  {'─'*12}  {'─'*20}")
        
        for t, d in enumerate(winner['elm_details']):
            ch = repr(d['char']) if d['char'] == ' ' else d['char']
            top3_strs = [f"{repr(c) if c==' ' else c}={p:.3f}" for c, p in d['top3']]
            
            # Annotate interesting positions
            note = ""
            if d['prob'] < 0.05:
                note = "← LM surprised"
            elif d['prob'] > 0.8:
                note = "← LM confident"
            elif d['top3'][0][0] != d['char'] and d['top3'][0][1] > 0.5:
                note = f"← LM preferred '{d['top3'][0][0]}'"
            
            log(f"  {t:>4}  {ch:>4}  {d['log_prob']:>8.4f}  {d['prob']:>6.3f}  "
                f"{top3_strs[0]:>12}  {top3_strs[1]:>12}  {top3_strs[2]:>12}  {note}")
        
        log(f"\n  Score breakdown:")
        log(f"    ELM total log-prob:  {winner['elm_total']:.4f}  (sum of per-char log-probs)")
        log(f"    Hypothesis length:   {winner['length']} characters")
        log(f"    ELM / length:        {winner['elm_per_len']:.4f}")
        log(f"    α × ELM/length:      {winner['elm_contribution']:.3f}  (alpha={alpha})")
        log(f"    CTC score:           {winner['ctc_score']:.3f}")
        log(f"    ─────────────────────────────")
        log(f"    Combined:            {winner['combined']:.3f}  = {winner['ctc_score']:.3f} + {winner['elm_contribution']:.3f}")


# ============================================================================
# FULL EVALUATION PIPELINE
# ============================================================================

@torch.no_grad()
def run_full_evaluation(asr_model, loader, elm=None, ilm=None, beam_width=10,
                        alpha=0.3, beta=0.1, gamma=0.0, device='cuda', split='test'):
    """Run all decode modes and report metrics."""
    asr_model.eval()
    print(f"\n{'='*70}\n  Full Evaluation: {split}\n{'='*70}")
    
    # Pre-compute encoder outputs
    all_logits, all_lengths, all_refs = [], [], []
    for batch in loader:
        feat = batch['features'].to(device)
        fl = batch['feature_lengths'].to(device)
        logits, lengths = asr_model(feat, fl)
        for i in range(feat.shape[0]):
            L = int(lengths[i].item())
            all_logits.append(logits[i,:L,:].cpu())
            all_lengths.append(L)
            all_refs.append(ids_to_text(batch['text_ids'][i,:batch['text_lengths'][i]].tolist()))
    print(f"  {len(all_refs)} utterances.\n")
    
    modes = [('greedy', 0, 0, 0, 'Greedy (baseline)'),
             ('beam', 0, 0, 0, f'Beam W={beam_width}')]
    if elm: modes.append(('elm', alpha, 0, gamma, f'Beam+ELM (a={alpha})'))
    if elm and ilm: modes.append(('ilme', alpha, beta, gamma, f'Beam+ILME (a={alpha},b={beta})'))
    if ilm: modes.append(('ilm_sub', 0, beta, gamma, f'Beam-ILM (b={beta})'))
    
    results = {}
    for mode, a, b, g, label in modes:
        print(f"  --- {label} ---")
        hyps = []
        for i in range(len(all_refs)):
            lp = F.log_softmax(all_logits[i].to(device), dim=-1)
            if mode == 'greedy':
                pred = lp.argmax(dim=-1)
                last, seq = -1, []
                for p in pred.tolist():
                    if p != last and p != BLANK_TOKEN and p < len(VOCAB_CHARS):
                        seq.append(p)
                    last = p
                hyps.append(ids_to_text(seq))
            else:
                nb = ctc_prefix_beam_search(lp.cpu().numpy(), beam_width)
                if mode == 'beam':
                    hyps.append(ids_to_text(nb[0][0]) if nb else "")
                else:
                    use_e = elm if a > 0 else None
                    use_i = ilm if b > 0 else None
                    r = rescore_nbest_ilme(nb, use_e, use_i, a, b, g, device)
                    hyps.append(ids_to_text(r[0][0]) if r else "")
        metrics = compute_detailed_metrics(all_refs, hyps)
        metrics['label'] = label
        results[mode] = metrics
    
    print(f"\n{'='*90}")
    print(f"  {'Mode':<35} {'WER':>7} {'P':>6} {'R':>6} {'F1':>6}")
    print(f"  {'-'*35} {'-'*7} {'-'*6} {'-'*6} {'-'*6}")
    for _, m in results.items():
        print(f"  {m['label']:<35} {m['wer']:>6.2f}% {m['precision']:>5.1f}% {m['recall']:>5.1f}% {m['f1']:>5.1f}%")
    print(f"{'='*90}")
    return results


# ============================================================================
# TUNING
# ============================================================================

def tune_weights(asr_model, loader, elm, ilm, beam_width=10, device='cuda'):
    """Grid search alpha/beta/gamma on dev set.
    
    Pre-computes beam search AND LM scores once, then grid search is pure arithmetic.
    """
    asr_model.eval()
    all_nb, all_refs = [], []
    
    print("\n[Tune] Phase 1: Pre-computing N-best lists (beam search)...")
    t0 = time.time()
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            feat = batch['features'].to(device)
            fl = batch['feature_lengths'].to(device)
            logits, lengths = asr_model(feat, fl)
            lp = F.log_softmax(logits, dim=-1)
            for i in range(feat.shape[0]):
                L = int(lengths[i].item())
                nb = ctc_prefix_beam_search(lp[i,:L,:].cpu().numpy(), beam_width)
                all_nb.append(nb)
                all_refs.append(ids_to_text(batch['text_ids'][i,:batch['text_lengths'][i]].tolist()))
            if (bi + 1) % 50 == 0:
                print(f"    Batch {bi+1}, {sum(len(nb) for nb in all_nb)} hypotheses so far...")
    
    total_hyps = sum(len(nb) for nb in all_nb)
    print(f"  {len(all_refs)} utterances, {total_hyps} total hypotheses. ({time.time()-t0:.0f}s)")
    
    # Pre-compute ALL LM scores once (this is the expensive part)
    print(f"\n[Tune] Phase 2: Pre-computing LM scores for {total_hyps} hypotheses...")
    t0 = time.time()
    all_scores = []  # Parallel to all_nb: list of lists of (char_ids, ctc_score, elm_score, ilm_score, length)
    
    for ui, nb in enumerate(all_nb):
        utt_scores = []
        for char_ids, ctc_score in nb:
            length = max(1, len(char_ids))
            elm_s = elm.score_sequence(char_ids, device=device) if elm and char_ids else 0.0
            ilm_s = ilm.score_sequence(char_ids, device=device) if ilm and char_ids else 0.0
            utt_scores.append((char_ids, ctc_score, elm_s, ilm_s, length))
        all_scores.append(utt_scores)
        if (ui + 1) % 500 == 0:
            print(f"    Scored {ui+1}/{len(all_nb)} utterances...")
    
    print(f"  LM scoring done. ({time.time()-t0:.0f}s)")
    
    # --- Diagnostic: Score distribution analysis ---
    # This reveals WHY beta has no effect (if ILM scores have no variance across beams)
    print(f"\n[Tune] Score Distribution Analysis (across beam hypotheses per utterance):")
    ctc_spreads, elm_spreads, ilm_spreads = [], [], []
    elm_ilm_corrs = []
    n_changed_by_elm, n_changed_by_ilm = 0, 0
    
    for utt_scores in all_scores:
        if len(utt_scores) < 2:
            continue
        ctc_vals = [s[1] for s in utt_scores]  # ctc_score
        elm_vals = [s[2] / s[4] for s in utt_scores]  # elm/length
        ilm_vals = [s[3] / s[4] for s in utt_scores]  # ilm/length
        
        ctc_spreads.append(max(ctc_vals) - min(ctc_vals))
        elm_spreads.append(max(elm_vals) - min(elm_vals))
        ilm_spreads.append(max(ilm_vals) - min(ilm_vals))
        
        # Check: would ELM or ILM change the top-1 hypothesis? (at alpha=1.0 for signal detection)
        ctc_best = max(range(len(utt_scores)), key=lambda i: ctc_vals[i])
        elm_best = max(range(len(utt_scores)), key=lambda i: ctc_vals[i] + 1.0 * elm_vals[i])
        ilm_best = max(range(len(utt_scores)), key=lambda i: ctc_vals[i] - 1.0 * ilm_vals[i])
        if elm_best != ctc_best:
            n_changed_by_elm += 1
        if ilm_best != ctc_best:
            n_changed_by_ilm += 1
        
        # Correlation between ELM and ILM scores (high correlation = ILM subtraction cancels ELM)
        if np.std(elm_vals) > 1e-8 and np.std(ilm_vals) > 1e-8:
            corr = np.corrcoef(elm_vals, ilm_vals)[0, 1]
            elm_ilm_corrs.append(corr)
    
    n_utt = len(ctc_spreads)
    print(f"  Score spread (max-min) across beam hypotheses (median over {n_utt} utterances):")
    print(f"    CTC score spread:       {np.median(ctc_spreads):.4f}  (this is what beam search sees)")
    print(f"    ELM score/len spread:   {np.median(elm_spreads):.4f}")
    print(f"    ILM score/len spread:   {np.median(ilm_spreads):.4f}")
    print(f"  Effective signal-to-noise ratio (at alpha=1.0):")
    print(f"    ELM contribution / CTC spread: {np.median(elm_spreads)/max(np.median(ctc_spreads),1e-8):.2%}")
    print(f"    ILM contribution / CTC spread: {np.median(ilm_spreads)/max(np.median(ctc_spreads),1e-8):.2%}")
    print(f"  Re-ranking impact:")
    print(f"    Utterances where ELM changes top-1: {n_changed_by_elm}/{n_utt} ({100*n_changed_by_elm/max(n_utt,1):.1f}%)")
    print(f"    Utterances where ILM changes top-1: {n_changed_by_ilm}/{n_utt} ({100*n_changed_by_ilm/max(n_utt,1):.1f}%)")
    if elm_ilm_corrs:
        print(f"  ELM↔ILM score correlation: {np.median(elm_ilm_corrs):.3f} (median)")
        print(f"    (High correlation = ILM subtraction undoes ELM contribution)")
    print()
    
    # Grid search is now pure arithmetic — instant
    n_combos = 10 * 6 * 5  # alpha × beta × gamma
    print(f"\n[Tune] Phase 3: Grid search ({n_combos} combinations)...")
    t0 = time.time()
    best_wer, best_p = float('inf'), (0, 0, 0)
    all_results = []
    
    for a in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0]:
        for b in [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]:
            for g in [-0.3, -0.1, 0.0, 0.1, 0.3]:
                hyps = []
                for utt_scores in all_scores:
                    best_hyp, best_score = [], float('-inf')
                    for char_ids, ctc_score, elm_s, ilm_s, length in utt_scores:
                        final = ctc_score + a * (elm_s / length) - b * (ilm_s / length) + g * length
                        if final > best_score:
                            best_score = final
                            best_hyp = char_ids
                    hyps.append(ids_to_text(best_hyp) if best_hyp else "")
                w = jiwer_wer(all_refs, hyps)
                all_results.append((a, b, g, w))
                if w < best_wer:
                    best_wer = w; best_p = (a, b, g)
                    print(f"  NEW BEST: a={a}, b={b}, g={g:.1f}, WER={w*100:.2f}%")
    
    print(f"\n  Grid search done. ({time.time()-t0:.0f}s)")
    
    # Print top-10 results
    all_results.sort(key=lambda x: x[3])
    print(f"\n[Tune] Top 10 configurations:")
    print(f"  {'Rank':>4} {'alpha':>6} {'beta':>6} {'gamma':>6} {'WER':>8}")
    print(f"  {'-'*34}")
    for i, (a, b, g, w) in enumerate(all_results[:10]):
        print(f"  {i+1:>4} {a:>6.2f} {b:>6.2f} {g:>6.1f} {w*100:>7.2f}%")
    
    print(f"\n[Tune] BEST: alpha={best_p[0]}, beta={best_p[1]}, gamma={best_p[2]}, WER={best_wer*100:.2f}%")
    return best_p, all_results


# ============================================================================
# ASR MODEL LOADING
# Supports S-SSSM (v7.6), Mamba-1 (v2.1), and Mamba-3 (v1.0) checkpoints.
# ============================================================================

def load_asr_model(checkpoint_path, asr_type='sssm', device='cuda'):
    """
    Load ASR model from Lightning checkpoint.
    
    asr_type: 'sssm'  for S-SSSM encoder checkpoints
              'mamba' for Mamba-1 encoder checkpoints
              'mamba3' for Mamba-3 encoder checkpoints
    """
    print(f"\n[LoadASR] Loading {asr_type} checkpoint: {checkpoint_path}")
    
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    hparams = ckpt.get('hyper_parameters', {})
    state = ckpt['state_dict']
    
    # Extract model weights (strip 'model.' prefix from Lightning state dict)
    model_state = {}
    for k, v in state.items():
        if k.startswith('model.'):
            model_state[k[6:]] = v
    
    d_model = hparams.get('d_model', 256)
    n_layers = hparams.get('n_layers', 12)
    n_classes = VOCAB_SIZE
    
    if asr_type == 'sssm':
        # Need to import the ASR model architecture
        # We reconstruct it here to avoid importing the full training script
        from collections import OrderedDict
        
        class _ConvSub(nn.Module):
            def __init__(self, inc=80, outc=256):
                super().__init__()
                self.conv1 = nn.Conv1d(inc, outc//2, 5, 2, 2)
                self.conv2 = nn.Conv1d(outc//2, outc, 5, 2, 2)
            def forward(self, x, lengths):
                x = x.transpose(1,2)
                x = F.relu(self.conv1(x))
                x = F.relu(self.conv2(x))
                x = x.transpose(1,2)
                lengths = torch.div(lengths, 4, rounding_mode="floor")
                return x, lengths
        
        class _ASR(nn.Module):
            def __init__(self, d_model, n_layers, n_classes):
                super().__init__()
                hier_params = hparams.get('hier_params', 'abc')
                use_hier = hparams.get('use_hierarchical', True)
                use_gate = hparams.get('use_gating', True)
                self.subsample = _ConvSub(80, d_model)
                self.encoder = SSMEncoder(d_model, n_layers, 0.1, use_hier, use_gate, hier_params)
                self.output_head = nn.Linear(d_model, n_classes)
            def forward(self, features, feature_lengths):
                x, lengths = self.subsample(features, feature_lengths)
                x = self.encoder(x)
                return self.output_head(x), lengths
        
        model = _ASR(d_model, n_layers, n_classes)
        model.load_state_dict(model_state, strict=False)
    
    elif asr_type == 'mamba':
        import importlib
        mamba_mod = importlib.import_module('IKT590_Script_ASR_mamba_v2_1')
        
        encoder_type = hparams.get('encoder_type', 'mamba')
        d_state = hparams.get('d_state', 16)
        d_conv = hparams.get('d_conv', 4)
        expand = hparams.get('expand', 2)
        use_hier = hparams.get('use_hierarchical', True)
        use_gate = hparams.get('use_gating', True)
        use_cuda = hparams.get('use_cuda_kernels', True)
        
        model = mamba_mod.ASRModel(
            n_classes=n_classes,
            d_model=d_model,
            n_layers=n_layers,
            encoder_type=encoder_type,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            use_hierarchical=use_hier,
            use_gating=use_gate,
            use_cuda_kernels=use_cuda,
        )
        model.load_state_dict(model_state, strict=False)
    
    elif asr_type == 'mamba3':
        import importlib.util
        _spec = importlib.util.spec_from_file_location(
            "mamba3_asr",
            os.path.join(os.path.dirname(__file__), "IKT590_Script_ASR_mamba-3_v1_0.py")
        )

        mamba3_mod = importlib.util.module_from_spec(_spec)
        import sys as _sys
        _sys.modules["mamba3_asr"] = mamba3_mod
        _spec.loader.exec_module(mamba3_mod)

        encoder_type = hparams.get('encoder_type', 'mamba3')
        d_state = hparams.get('d_state', 16)
        d_conv = hparams.get('d_conv', 4)
        expand = hparams.get('expand', 2)
        use_hier = hparams.get('use_hierarchical', True)
        use_gate = hparams.get('use_gating', True)
        use_cuda = hparams.get('use_cuda_kernels', True)
        headdim = hparams.get('headdim', 64)
        is_mimo = hparams.get('is_mimo', False)
        mimo_rank = hparams.get('mimo_rank', 4)
        
        model = mamba3_mod.ASRModel(
            n_classes=n_classes,
            d_model=d_model,
            n_layers=n_layers,
            encoder_type=encoder_type,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            use_hierarchical=use_hier,
            use_gating=use_gate,
            use_cuda_kernels=use_cuda,
            headdim=headdim,
            is_mimo=is_mimo,
            mimo_rank=mimo_rank,
        )
        model.load_state_dict(model_state, strict=False)
        model.float()

    else:
        raise ValueError(f"asr_type '{asr_type}' not supported. Use 'sssm', 'mamba', or 'mamba3'.")
    
    model.to(device).eval()
    total = sum(p.numel() for p in model.parameters())
    print(f"[LoadASR] Loaded: d_model={d_model}, layers={n_layers}, params={total:,}")
    return model


# ============================================================================
# COLLATE FUNCTION (for evaluation data loading)
# ============================================================================

def make_eval_loader(hf_dataset, batch_size=16, subset=None):
    """Create evaluation DataLoader from HF dataset."""
    if not _HAS_TORCHAUDIO:
        raise RuntimeError("torchaudio required for audio processing")
    
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE, n_fft=400, win_length=400, hop_length=160,
        n_mels=80, f_min=20.0, f_max=7600.0, power=1.0)
    amp_to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80.0)
    
    class _DS(Dataset):
        def __init__(self, ds, subset=None):
            self.ds = ds.select(range(subset)) if subset and subset < len(ds) else ds
        def __len__(self): return len(self.ds)
        def __getitem__(self, idx):
            item = self.ds[idx]
            wav = torch.FloatTensor(item['audio']['array'])
            if wav.ndim == 1: wav = wav.unsqueeze(0)
            with torch.no_grad():
                mel = mel_transform(wav)
                feat = amp_to_db(mel)
                feat = (feat - feat.mean()) / (feat.std() + 1e-5)
            feat = feat.squeeze(0).transpose(0, 1)
            return feat, item['text']
    
    def _collate(batch):
        features, texts = zip(*batch)
        fl = torch.LongTensor([f.shape[0] for f in features])
        max_len = max(f.shape[0] for f in features)
        fp = torch.zeros(len(batch), max_len, 80)
        for i, f in enumerate(features):
            fp[i, :f.shape[0], :] = f
        tids = [torch.LongTensor(text_to_ids(t)) for t in texts]
        tl = torch.LongTensor([len(t) for t in tids])
        max_tl = max(len(t) for t in tids) if tids else 1
        tp = torch.full((len(batch), max_tl), BLANK_TOKEN, dtype=torch.long)
        for i, t in enumerate(tids):
            if len(t) > 0: tp[i, :len(t)] = t
        return {'features': fp, 'feature_lengths': fl, 'text_ids': tp,
                'text_lengths': tl, 'texts': texts}
    
    ds = _DS(hf_dataset, subset)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=_collate, num_workers=2)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the SSM ASR pipeline with shallow-fusion LM rescoring"
    )

    # Mode
    parser.add_argument('--mode', type=str, required=True,
                        choices=['tune', 'full_eval', 'rtfx', 'hallucination',
                                 'score_analysis', 'verify_streaming'],
                        help='Evaluation mode')

    # Data
    parser.add_argument('--data-path', type=str, default='hub_data/librispeech',
                        help='Root path containing LibriSpeech clean/ and other/ subsets')
    parser.add_argument('--dataset-config', type=str, default='100h',
                        choices=['100h', '360h', '460h'],
                        help='Unused for evaluation; kept for backwards-compatible CLI invocations')

    # ASR model
    parser.add_argument('--asr-checkpoint', type=str, default=None,
                        help='Path to Lightning ASR checkpoint (.ckpt)')
    parser.add_argument('--asr-type', type=str, default='sssm',
                        choices=['sssm', 'mamba', 'mamba3'],
                        help='ASR encoder family')

    # LM paths (point to checkpoint dirs / arpa / pkl files)
    parser.add_argument('--elm-path', type=str, default=None,
                        help='External LM path (CharMamba checkpoint dir, .arpa, or .pkl)')
    parser.add_argument('--ilm-path', type=str, default=None,
                        help='Internal LM path (optional, for ILME-style subtraction)')

    # ILME / shallow-fusion params
    parser.add_argument('--beam-width', type=int, default=10)
    parser.add_argument('--alpha', type=float, default=0.3,
                        help='ELM weight in shallow fusion')
    parser.add_argument('--beta', type=float, default=0.1,
                        help='ILM weight (subtractive, ILME-style)')
    parser.add_argument('--gamma-ilme', type=float, default=0.0,
                        help='Length bonus')

    # General
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--subset-val', type=int, default=None,
                        help='Optional cap on validation utterances used by tune mode')
    parser.add_argument('--exp-name', type=str, default='mamba_lm',
                        help='Used as a suffix for the output directory name')
    parser.add_argument('--seed', type=int, default=456)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ================================================================
    # MODE: VERIFY STREAMING (compare batch vs step scoring)
    # ================================================================
    if args.mode == 'verify_streaming':
        path = args.elm_path or args.ilm_path
        if not path:
            print("ERROR: --elm-path or --ilm-path required"); sys.exit(1)

        model = load_lm(path, device=str(device))

        test_texts = [
            "the quick brown fox",
            "hello world",
            "automatic speech recognition with state space models",
        ]

        print(f"\nVerifying streaming vs batch scoring:")
        for text in test_texts:
            ids = text_to_ids(text)
            batch_score = model.score_sequence(ids, device=device)

            if hasattr(model, 'forward_step'):
                stream_score, _ = model.score_sequence_streaming(ids, device=device)
                diff = abs(batch_score - stream_score)
                ok = "OK" if diff < 0.01 else "MISMATCH"
                print(f"  '{text[:30]:30s}' batch={batch_score:.2f} stream={stream_score:.2f} diff={diff:.4f} [{ok}]")
            else:
                print(f"  '{text[:30]:30s}' batch={batch_score:.2f} (no streaming)")
        return

    # ================================================================
    # EVALUATION MODES (need ASR + data)
    # ================================================================
    if args.mode not in ('full_eval', 'tune', 'rtfx', 'hallucination', 'score_analysis'):
        print(f"ERROR: unknown mode '{args.mode}'"); sys.exit(1)

    if not args.asr_checkpoint:
        print("ERROR: --asr-checkpoint required"); sys.exit(1)

    asr_model = load_asr_model(args.asr_checkpoint, args.asr_type, device)
    elm = load_lm(args.elm_path, str(device)) if args.elm_path else None
    ilm = load_lm(args.ilm_path, str(device)) if args.ilm_path else None

    ds_clean = load_from_disk(os.path.join(args.data_path, "clean"))
    ds_other = load_from_disk(os.path.join(args.data_path, "other"))

    test_clean = make_eval_loader(ds_clean["test"], args.batch_size)
    test_other = make_eval_loader(ds_other["test"], args.batch_size)
    val_clean = make_eval_loader(ds_clean["validation"], args.batch_size, args.subset_val)

    out_dir = f"ilme_mamba_results_{args.exp_name}"
    os.makedirs(out_dir, exist_ok=True)

    if args.mode == 'full_eval':
        for name, ldr in [("test-clean", test_clean), ("test-other", test_other)]:
            r = run_full_evaluation(asr_model, ldr, elm, ilm, args.beam_width,
                                     args.alpha, args.beta, args.gamma_ilme, device, name)
            with open(f"{out_dir}/full_eval_{name}.json", 'w') as f:
                json.dump(r, f, indent=2, default=str)

    elif args.mode == 'tune':
        bp, all_results = tune_weights(asr_model, val_clean, elm, ilm, args.beam_width, device)
        with open(f"{out_dir}/best_params.json", 'w') as f:
            json.dump({'alpha': bp[0], 'beta': bp[1], 'gamma': bp[2]}, f, indent=2)
        with open(f"{out_dir}/grid_results.json", 'w') as f:
            json.dump([{'alpha': a, 'beta': b, 'gamma': g, 'wer': w*100}
                       for a, b, g, w in all_results], f, indent=2)

    elif args.mode == 'rtfx':
        for dm in ['greedy', 'beam', 'nbest_ilme']:
            if dm == 'nbest_ilme' and not elm and not ilm: continue
            r = measure_rtfx(asr_model, test_clean, dm, args.beam_width,
                              elm, ilm, args.alpha, args.beta, args.gamma_ilme, device)
            with open(f"{out_dir}/rtfx_{dm}.json", 'w') as f:
                json.dump(r, f, indent=2, default=str)

    elif args.mode == 'hallucination':
        if not elm: print("ERROR: --elm-path required"); sys.exit(1)
        r = analyze_hallucinations(asr_model, elm, test_clean, device)
        with open(f"{out_dir}/hallucination.json", 'w') as f:
            json.dump(r, f, indent=2, default=str)

    elif args.mode == 'score_analysis':
        if not elm: print("ERROR: --elm-path required"); sys.exit(1)
        save_path = f"{out_dir}/score_analysis_alpha{args.alpha}_beam{args.beam_width}.txt"
        r = run_score_analysis(asr_model, test_clean, elm, args.beam_width,
                                args.alpha, device, n_examples=10, save_path=save_path)
        with open(f"{out_dir}/score_analysis_summary.json", 'w') as f:
            json.dump(r, f, indent=2, default=str)

    print(f"\nResults: {out_dir}/")


if __name__ == '__main__':
    main()