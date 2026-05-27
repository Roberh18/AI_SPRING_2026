#!/usr/bin/env python3
"""
Pure Python Character N-gram LM — Baseline for CharMamba comparison.

Trains character-level n-gram models with Stupid Backoff smoothing.
No external dependencies (no KenLM, no Boost, no C++).

Produces models with the same score_sequence() interface as CharMambaLM,
so the entire tune/eval/score_analysis pipeline works unchanged.

Prerequisites:
    A char_level_text.txt file produced by the train_kenlm mode of the
    main evaluation script (IKT590_eval_ASR.py). One sentence per line,
    characters space-separated, word boundaries marked with ▁.

Usage:
    # Train 7-gram and 10-gram baselines
    python IKT590_train_LM_char_ngram.py \\
        --train-text lm_checkpoints/kenlm/char_level_text.txt \\
        --orders 7,10 --output-dir lm_checkpoints/ngram

    # Train a single order with limited data (quick test)
    python IKT590_train_LM_char_ngram.py \\
        --train-text lm_checkpoints/kenlm/char_level_text.txt \\
        --orders 7 --max-lines 10000 --output-dir lm_checkpoints/ngram

    # Use the output .pkl files with load_lm() in the eval script:
    #   --elm-path lm_checkpoints/ngram/char_10gram.pkl
"""

import argparse
import math
import os
import pickle
import time
from collections import defaultdict, Counter
from typing import List

# Must match main script
VOCAB_CHARS = list(" 'abcdefghijklmnopqrstuvwxyz")
BLANK_TOKEN = len(VOCAB_CHARS)
VOCAB_SIZE = len(VOCAB_CHARS) + 1
CHAR_TO_IDX = {c: i for i, c in enumerate(VOCAB_CHARS)}
IDX_TO_CHAR = {i: c for i, c in enumerate(VOCAB_CHARS)}
SPACE_TOKEN = '▁'

BOS = '<s>'
EOS = '</s>'


def text_to_ids(text: str) -> List[int]:
    return [CHAR_TO_IDX[c] for c in text.lower() if c in CHAR_TO_IDX]


def ids_to_text(ids: List[int]) -> str:
    return "".join(IDX_TO_CHAR[i] for i in ids if i in IDX_TO_CHAR)


class CharNgramLM:
    """
    Character-level N-gram LM with Stupid Backoff smoothing.
    
    Stupid Backoff (Brants et al. 2007) is used instead of Kneser-Ney because:
    - Simpler to implement correctly
    - Works well with large training data
    - Standard baseline in large-scale LM literature
    
    The backoff factor alpha=0.4 is the standard value from the original paper.
    """
    
    def __init__(self, order: int, counts: dict = None, total_unigrams: int = 0,
                 backoff_alpha: float = 0.4):
        self.order = order
        self.counts = counts or {}  # {n: {context_tuple: {next_token: count}}}
        self.context_totals = {}     # {n: {context_tuple: total_count}}
        self.total_unigrams = total_unigrams
        self.backoff_alpha = backoff_alpha
        self.vocab_size = VOCAB_SIZE
        self.d_model = 0
        self.n_layers_count = order
    
    def _build_context_totals(self):
        """Pre-compute context totals for fast probability lookup."""
        self.context_totals = {}
        for n, ngram_counts in self.counts.items():
            self.context_totals[n] = {}
            for context, next_counts in ngram_counts.items():
                self.context_totals[n][context] = sum(next_counts.values())
    
    def score_token(self, context: tuple, token: str) -> float:
        """
        Score a single token given context using Stupid Backoff.
        Returns log probability (natural log).
        
        P_SB(w | context) = 
            count(context + w) / count(context)           if count(context + w) > 0
            alpha * P_SB(w | context[1:])                 otherwise (backoff)
            count(w) / total_unigrams                     at unigram level
            1 / vocab_size                                uniform fallback
        """
        # Try each order from highest to lowest
        for n in range(min(len(context) + 1, self.order), 0, -1):
            ctx = context[-(n-1):] if n > 1 else ()
            
            if n in self.counts and ctx in self.counts[n]:
                next_counts = self.counts[n][ctx]
                if token in next_counts:
                    total = self.context_totals[n][ctx]
                    prob = next_counts[token] / total
                    # Apply backoff penalty for each level we dropped
                    backoff_levels = min(len(context) + 1, self.order) - n
                    prob *= (self.backoff_alpha ** backoff_levels)
                    return math.log(max(prob, 1e-20))
        
        # Uniform fallback
        return math.log(1.0 / len(VOCAB_CHARS))
    
    def _ids_to_tokens(self, char_ids: List[int]) -> List[str]:
        """Convert char IDs to token strings (matching KenLM format)."""
        tokens = []
        for cid in char_ids:
            if 0 <= cid < len(VOCAB_CHARS):
                c = VOCAB_CHARS[cid]
                tokens.append(SPACE_TOKEN if c == ' ' else c)
        return tokens
    
    def score_sequence(self, char_ids: List[int], device=None) -> float:
        """
        Score a character sequence. Returns total log prob (natural log).
        Matches CharMambaLM.score_sequence() interface exactly.
        """
        if not char_ids:
            return 0.0
        
        tokens = [BOS] + self._ids_to_tokens(char_ids) + [EOS]
        total = 0.0
        
        for i in range(1, len(tokens)):  # Skip BOS, score everything including EOS
            context = tuple(tokens[max(0, i - self.order + 1):i])
            token = tokens[i]
            total += self.score_token(context, token)
        
        return total
    
    def score_sequence_detailed(self, char_ids: List[int], device=None):
        """
        Score with per-character breakdown.
        Matches CharMambaLM.score_sequence_detailed() interface.
        """
        if not char_ids:
            return 0.0, []
        
        tokens = [BOS] + self._ids_to_tokens(char_ids)
        total = 0.0
        details = []
        
        for i in range(1, len(tokens)):
            context = tuple(tokens[max(0, i - self.order + 1):i])
            token = tokens[i]
            lp = self.score_token(context, token)
            prob = math.exp(lp)
            total += lp
            
            # Map token back to character
            char = ' ' if token == SPACE_TOKEN else token
            
            details.append({
                'char': char,
                'log_prob': lp,
                'prob': min(prob, 1.0),
                'top3': [(char, min(prob, 1.0)), ('?', 0.0), ('?', 0.0)],
            })
        
        return total, details
    
    def get_log_probs(self, input_ids):
        """
        Compute log probabilities for all vocab items at each position.
        Matches CharMambaLM.get_log_probs() interface for analyze_hallucinations.
        
        Args:
            input_ids: torch.LongTensor [1, T] (token IDs, BOS-prepended)
        Returns:
            torch.Tensor [1, T, V] of log probabilities
        """
        import torch
        ids = input_ids[0].cpu().tolist()  # [T] — move to CPU if on GPU
        T = len(ids)
        V = self.vocab_size
        log_probs = torch.full((1, T, V), -20.0)  # Default very low
        
        # Build token sequence from IDs
        tokens = []
        for cid in ids:
            if cid == BLANK_TOKEN:
                tokens.append(BOS)
            elif 0 <= cid < len(VOCAB_CHARS):
                c = VOCAB_CHARS[cid]
                tokens.append(SPACE_TOKEN if c == ' ' else c)
            else:
                tokens.append(BOS)
        
        # For each position, score every vocab character
        for t in range(T):
            context = tuple(tokens[max(0, t - self.order + 2):t + 1])
            for cid in range(min(V - 1, len(VOCAB_CHARS))):
                c = VOCAB_CHARS[cid]
                token = SPACE_TOKEN if c == ' ' else c
                lp = self.score_token(context, token)
                log_probs[0, t, cid] = lp
        
        return log_probs
    
    def eval(self):
        """No-op for interface compatibility."""
        return self
    
    def parameters(self):
        """Empty iterator for interface compatibility."""
        return iter([])
    
    def save(self, path: str):
        """Save model to pickle file."""
        data = {
            'order': self.order,
            'counts': self.counts,
            'context_totals': self.context_totals,
            'total_unigrams': self.total_unigrams,
            'backoff_alpha': self.backoff_alpha,
            'model_type': 'ngram',
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f, protocol=4)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"  Saved: {path} ({size_mb:.1f} MB)")
    
    @classmethod
    def load(cls, path: str):
        """Load model from pickle file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        model = cls(
            order=data['order'],
            counts=data['counts'],
            total_unigrams=data['total_unigrams'],
            backoff_alpha=data.get('backoff_alpha', 0.4),
        )
        model.context_totals = data['context_totals']
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"\n[CharNgramLM] {data['order']}-gram from {path} ({size_mb:.1f} MB)")
        return model


def train_ngram(text_path: str, order: int, max_lines: int = None) -> CharNgramLM:
    """
    Train a character n-gram model from the char_level_text.txt file.
    
    The file format (produced by prepare_kenlm_text) has one sentence per line,
    with characters space-separated and word boundaries as ▁ tokens:
        t h e ▁ c a t ▁ s a t
    """
    print(f"\n  Training {order}-gram from {text_path}...")
    t0 = time.time()
    
    # Count n-grams
    counts = {n: defaultdict(Counter) for n in range(1, order + 1)}
    total_tokens = 0
    n_lines = 0
    
    with open(text_path, 'r', encoding='utf-8') as f:
        for line in f:
            tokens = [BOS] + line.strip().split() + [EOS]
            
            for i in range(1, len(tokens)):
                for n in range(1, min(i + 1, order) + 1):
                    context = tuple(tokens[i - n + 1:i]) if n > 1 else ()
                    counts[n][context][tokens[i]] += 1
                total_tokens += 1
            
            n_lines += 1
            if n_lines % 1_000_000 == 0:
                elapsed = time.time() - t0
                print(f"    {n_lines:,} lines, {total_tokens:,} tokens ({elapsed:.0f}s)")
            
            if max_lines and n_lines >= max_lines:
                break
    
    elapsed = time.time() - t0
    
    # Convert defaultdicts to regular dicts for pickling
    regular_counts = {}
    total_ngrams = 0
    for n in counts:
        regular_counts[n] = {}
        for ctx, counter in counts[n].items():
            regular_counts[n][ctx] = dict(counter)
            total_ngrams += len(counter)
    
    model = CharNgramLM(order=order, counts=regular_counts, total_unigrams=total_tokens)
    model._build_context_totals()
    
    print(f"  Done: {n_lines:,} lines, {total_tokens:,} tokens, "
          f"{total_ngrams:,} unique n-grams ({elapsed:.0f}s)")
    
    return model


def main():
    parser = argparse.ArgumentParser(description='Train character n-gram LMs')
    parser.add_argument('--train-text', type=str, required=True,
                        help='Path to char_level_text.txt (from train_kenlm mode)')
    parser.add_argument('--orders', type=str, default='7,10',
                        help='Comma-separated n-gram orders (default: 7,10)')
    parser.add_argument('--output-dir', type=str, default='lm_checkpoints/ngram',
                        help='Output directory for .pkl files')
    parser.add_argument('--max-lines', type=int, default=None,
                        help='Max training lines (default: all)')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    orders = [int(x) for x in args.orders.split(',')]
    
    print(f"\n{'='*70}")
    print(f"  Character N-gram LM Training")
    print(f"  Text: {args.train_text}")
    print(f"  Orders: {orders}")
    print(f"  Output: {args.output_dir}")
    print(f"{'='*70}")
    
    for order in orders:
        model = train_ngram(args.train_text, order, args.max_lines)
        
        save_path = os.path.join(args.output_dir, f"char_{order}gram.pkl")
        model.save(save_path)
        
        # Quick validation
        test_texts = [
            "the cat sat on the mat",
            "xzq plonk brrft",
            "she walked to the store",
        ]
        print(f"\n  Validation scores ({order}-gram):")
        for text in test_texts:
            ids = text_to_ids(text)
            score = model.score_sequence(ids)
            ppl = math.exp(-score / max(len(ids), 1))
            print(f"    \"{text}\" → score={score:.2f}, PPL={ppl:.1f}")
    
    print(f"\n{'='*70}")
    print(f"  Training complete. Models saved to {args.output_dir}/")
    print(f"{'='*70}")
    print(f"\n  Usage with main script:")
    for order in orders:
        print(f"    --elm-path {args.output_dir}/char_{order}gram.pkl")


if __name__ == '__main__':
    main()