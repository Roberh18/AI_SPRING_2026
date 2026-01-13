#!/usr/bin/env python3
"""
Thesis Figure Generator - Updated for v7.4 Experiment Structure
================================================================
Generates comparison plots from experiment output files for the thesis.

Expected directory structure (from IKT464_Script_ASR_SSSM_v7_4.py):
    working_directory/
    ├── checkpoints_v74_baseline_W386_D9/
    │   ├── 0utput_exp-v74_baseline_W386_D9.txt
    │   ├── config.json
    │   ├── best_epoch=X_val_wer_clean=Y.ckpt
    │   └── diagnostics/
    ├── checkpoints_v74_hier_gating_W358_D9/
    │   └── ...
    └── ...

Usage:
    python generate_thesis_figures.py --exp-dir /path/to/working/directory --output-dir ./thesis_figures

Author: Robert Hanssen
"""

import os
import re
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 9,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.figsize': (10, 6),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# Color scheme for consistency
COLORS = {
    'baseline': '#7f7f7f',      # Gray
    'hier': '#ff7f0e',          # Orange
    'gate': '#2ca02c',          # Green
    'hier_gate': '#1f77b4',     # Blue
}


def parse_output_file(filepath: str) -> Dict:
    """
    Parse a terminal output file to extract training metrics.
    
    Returns dict with:
        - epochs: list of epoch numbers
        - train_loss: list of training losses
        - val_wer: list of validation WER values (%)
        - best_wer: best WER achieved
        - best_epoch: epoch where best WER was achieved
    """
    results = {
        'epochs': [],
        'train_loss': [],
        'val_wer': [],
        'best_wer': None,
        'best_epoch': None,
    }
    
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # Extract epoch summaries
    # Pattern matches the output format from the training script
    # Example: "Epoch 1 Summary:" ... "Val WER:     0.8153 (81.53%)"
    epoch_pattern = re.compile(
        r'Epoch\s+(\d+)\s+Summary:.*?'
        r'Train Loss:\s+([\d.]+|nan).*?'
        r'Val WER.*?:\s+([\d.]+)\s+\(([\d.]+)%\)',
        re.DOTALL
    )
    
    for match in epoch_pattern.finditer(content):
        epoch = int(match.group(1))
        train_loss_str = match.group(2)
        train_loss = float(train_loss_str) if train_loss_str != 'nan' else np.nan
        val_wer = float(match.group(4))  # Use percentage value
        
        results['epochs'].append(epoch)
        results['train_loss'].append(train_loss)
        results['val_wer'].append(val_wer)
    
    # Try alternate pattern for 100h config with val_wer_clean
    if not results['epochs']:
        epoch_pattern_alt = re.compile(
            r'Epoch\s+(\d+)\s+Summary:.*?'
            r'Train Loss:\s+([\d.]+|nan).*?'
            r'Val WER \(clean\):\s+([\d.]+)%',
            re.DOTALL
        )
        for match in epoch_pattern_alt.finditer(content):
            epoch = int(match.group(1))
            train_loss_str = match.group(2)
            train_loss = float(train_loss_str) if train_loss_str != 'nan' else np.nan
            val_wer = float(match.group(3))
            
            results['epochs'].append(epoch)
            results['train_loss'].append(train_loss)
            results['val_wer'].append(val_wer)
    
    # Extract best WER from final results section
    best_wer_match = re.search(r'Best WER.*?:\s+([\d.]+)%', content)
    if best_wer_match:
        results['best_wer'] = float(best_wer_match.group(1))
    elif results['val_wer']:
        results['best_wer'] = min(results['val_wer'])
    
    # Extract best epoch from checkpoint filename in output
    best_epoch_match = re.search(r'best_epoch=(\d+)', content)
    if best_epoch_match:
        results['best_epoch'] = int(best_epoch_match.group(1))
    elif results['val_wer']:
        results['best_epoch'] = results['epochs'][np.argmin(results['val_wer'])]
    
    return results


def load_config(config_path: str) -> Dict:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def find_experiments(base_dir: str) -> List[Dict]:
    """
    Find all experiment directories matching the v7.4 structure.
    
    Looks for directories named 'checkpoints_*' containing config.json and output files.
    """
    experiments = []
    base_path = Path(base_dir)
    
    # Look for checkpoint directories
    for item in base_path.iterdir():
        if not item.is_dir():
            continue
        
        # Match directories starting with 'checkpoints_'
        if not item.name.startswith('checkpoints_'):
            continue
        
        exp_name = item.name.replace('checkpoints_', '')
        
        # Look for config.json
        config_file = item / 'config.json'
        if not config_file.exists():
            print(f"  Warning: No config.json in {item.name}, skipping")
            continue
        
        # Load config
        try:
            config = load_config(str(config_file))
        except Exception as e:
            print(f"  Warning: Could not load config from {config_file}: {e}")
            continue
        
        # Look for output file
        output_file = item / f'0utput_exp-{exp_name}.txt'
        if not output_file.exists():
            # Try alternate patterns
            output_files = list(item.glob('*utput*.txt'))
            if output_files:
                output_file = output_files[0]
            else:
                print(f"  Warning: No output file in {item.name}, skipping")
                continue
        
        # Parse output file
        try:
            results = parse_output_file(str(output_file))
        except Exception as e:
            print(f"  Warning: Could not parse {output_file}: {e}")
            continue
        
        # Combine config and results
        exp_data = {
            'exp_name': exp_name,
            'exp_dir': str(item),
            'n_layers': config.get('n_layers'),
            'd_model': config.get('d_model'),
            'use_hierarchical': config.get('use_hierarchical', False),
            'use_gating': config.get('use_gating', False),
            **results
        }
        
        experiments.append(exp_data)
    
    return experiments


def get_experiment_label(exp: Dict, include_width: bool = False) -> str:
    """Generate a human-readable label for an experiment."""
    parts = []
    
    if exp.get('n_layers'):
        parts.append(f"L={exp['n_layers']}")
    
    if include_width and exp.get('d_model'):
        parts.append(f"D={exp['d_model']}")
    
    hier = exp.get('use_hierarchical', False)
    gate = exp.get('use_gating', False)
    
    if hier and gate:
        parts.append('Hier+Gate')
    elif hier:
        parts.append('Hier')
    elif gate:
        parts.append('Gate')
    else:
        parts.append('Baseline')
    
    return ', '.join(parts)


def get_config_type(exp: Dict) -> str:
    """Return configuration type for coloring."""
    hier = exp.get('use_hierarchical', False)
    gate = exp.get('use_gating', False)
    
    if hier and gate:
        return 'hier_gate'
    elif hier:
        return 'hier'
    elif gate:
        return 'gate'
    else:
        return 'baseline'


def plot_wer_comparison(experiments: List[Dict], output_path: str):
    """
    Plot WER curves for all experiments on the same axes.
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Sort by depth, then by config type
    experiments = sorted(experiments, 
                        key=lambda x: (x.get('n_layers', 0), 
                                      get_config_type(x) == 'baseline'))
    
    # Use different line styles for different depths
    depths = sorted(set(e.get('n_layers') for e in experiments if e.get('n_layers')))
    linestyles = ['-', '--', '-.', ':']
    depth_styles = {d: linestyles[i % len(linestyles)] for i, d in enumerate(depths)}
    
    for exp in experiments:
        if not exp['epochs'] or not exp['val_wer']:
            continue
        
        config_type = get_config_type(exp)
        color = COLORS.get(config_type, '#333333')
        linestyle = depth_styles.get(exp.get('n_layers'), '-')
        
        label = get_experiment_label(exp)
        best_wer = exp.get('best_wer', min(exp['val_wer']))
        label_with_best = f"{label} ({best_wer:.2f}%)"
        
        ax.plot(exp['epochs'], exp['val_wer'], 
                marker='o', markersize=3, linewidth=1.5,
                color=color, linestyle=linestyle, label=label_with_best)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation WER (%)')
    ax.set_title('Validation WER During Training')
    ax.legend(loc='upper right', framealpha=0.9, fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_depth_comparison_bars(experiments: List[Dict], output_path: str):
    """
    Create bar chart comparing final WER across depths for baseline vs hier+gate.
    This matches Table 5.5 in the thesis.
    """
    # Group experiments by depth
    depths = sorted(set(e['n_layers'] for e in experiments if e.get('n_layers')))
    
    baseline_wers = []
    hiergating_wers = []
    baseline_present = []
    hiergating_present = []
    
    for depth in depths:
        depth_exps = [e for e in experiments if e.get('n_layers') == depth]
        
        # Find baseline (no hier, no gating)
        baseline = [e for e in depth_exps 
                   if not e.get('use_hierarchical') and not e.get('use_gating')]
        if baseline and baseline[0].get('best_wer'):
            baseline_wers.append(baseline[0]['best_wer'])
            baseline_present.append(True)
        else:
            baseline_wers.append(0)
            baseline_present.append(False)
        
        # Find hier+gating
        hiergating = [e for e in depth_exps 
                     if e.get('use_hierarchical') and e.get('use_gating')]
        if hiergating and hiergating[0].get('best_wer'):
            hiergating_wers.append(hiergating[0]['best_wer'])
            hiergating_present.append(True)
        else:
            hiergating_wers.append(0)
            hiergating_present.append(False)
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(depths))
    width = 0.35
    
    # Only plot bars where data exists
    baseline_heights = [w if p else 0 for w, p in zip(baseline_wers, baseline_present)]
    hiergating_heights = [w if p else 0 for w, p in zip(hiergating_wers, hiergating_present)]
    
    bars1 = ax.bar(x - width/2, baseline_heights, width, label='Baseline', 
                   color=COLORS['baseline'], edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, hiergating_heights, width, label='Hier + Gate',
                   color=COLORS['hier_gate'], edgecolor='black', linewidth=0.5)
    
    # Add value labels on bars
    def add_labels(bars, present_flags):
        for bar, present in zip(bars, present_flags):
            if present:
                height = bar.get_height()
                ax.annotate(f'{height:.2f}%',
                           xy=(bar.get_x() + bar.get_width()/2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)
    
    add_labels(bars1, baseline_present)
    add_labels(bars2, hiergating_present)
    
    ax.set_xlabel('Number of Layers')
    ax.set_ylabel('Test-Clean WER (%)')
    ax.set_title('Word Error Rate by Model Depth: Baseline vs Hierarchical + Gating')
    ax.set_xticks(x)
    ax.set_xticklabels([f'L={d}' for d in depths])
    ax.legend(loc='upper right')
    ax.grid(True, axis='y', alpha=0.3)
    
    # Set y-axis to show meaningful range
    all_wers = [w for w, p in zip(baseline_wers + hiergating_wers, 
                                   baseline_present + hiergating_present) if p and w > 0]
    if all_wers:
        min_wer = min(all_wers)
        max_wer = max(all_wers)
        ax.set_ylim(bottom=max(0, min_wer - 3), top=max_wer + 2)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_ablation_comparison(experiments: List[Dict], output_path: str,
                             target_depth: int = 12):
    """
    Create bar chart showing ablation study results for a specific depth.
    Shows: Baseline, +Hier, +Gate, +Hier+Gate
    
    Note: This requires having all 4 configurations at the target depth.
    """
    depth_exps = [e for e in experiments if e.get('n_layers') == target_depth]
    
    if not depth_exps:
        print(f"  No experiments found at depth {target_depth} for ablation chart")
        return
    
    # Categorize experiments
    categories = {
        'Baseline': None,
        '+Hierarchical': None,
        '+Gating': None,
        '+Hier & Gate': None,
    }
    
    for exp in depth_exps:
        hier = exp.get('use_hierarchical', False)
        gate = exp.get('use_gating', False)
        wer = exp.get('best_wer')
        
        if not hier and not gate:
            categories['Baseline'] = wer
        elif hier and not gate:
            categories['+Hierarchical'] = wer
        elif not hier and gate:
            categories['+Gating'] = wer
        elif hier and gate:
            categories['+Hier & Gate'] = wer
    
    # Check what we have
    available = {k: v for k, v in categories.items() if v is not None}
    if len(available) < 2:
        print(f"  Only {len(available)} configurations at depth {target_depth}, skipping ablation")
        return
    
    # Create bar chart with available data
    fig, ax = plt.subplots(figsize=(8, 5))
    
    labels = list(available.keys())
    values = list(available.values())
    colors = [COLORS['baseline'], COLORS['hier'], COLORS['gate'], COLORS['hier_gate']]
    bar_colors = []
    for label in labels:
        if label == 'Baseline':
            bar_colors.append(COLORS['baseline'])
        elif label == '+Hierarchical':
            bar_colors.append(COLORS['hier'])
        elif label == '+Gating':
            bar_colors.append(COLORS['gate'])
        else:
            bar_colors.append(COLORS['hier_gate'])
    
    bars = ax.bar(labels, values, color=bar_colors, edgecolor='black', linewidth=0.5)
    
    # Add value labels
    for bar, val in zip(bars, values):
        ax.annotate(f'{val:.2f}%',
                   xy=(bar.get_x() + bar.get_width()/2, val),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=10)
    
    ax.set_ylabel('Test-Clean WER (%)')
    ax.set_title(f'Ablation Study: Effect of Hierarchical Init and Gating (L={target_depth})')
    ax.grid(True, axis='y', alpha=0.3)
    
    # Set y-axis range
    if values:
        ax.set_ylim(bottom=min(values) - 1.5, top=max(values) + 1)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_relative_improvement(experiments: List[Dict], output_path: str):
    """
    Create bar chart showing relative error reduction at each depth.
    """
    depths = sorted(set(e['n_layers'] for e in experiments if e.get('n_layers')))
    
    improvements = []
    valid_depths = []
    
    for depth in depths:
        depth_exps = [e for e in experiments if e.get('n_layers') == depth]
        
        baseline = [e for e in depth_exps 
                   if not e.get('use_hierarchical') and not e.get('use_gating')]
        hiergating = [e for e in depth_exps 
                     if e.get('use_hierarchical') and e.get('use_gating')]
        
        if baseline and hiergating:
            b_wer = baseline[0].get('best_wer')
            h_wer = hiergating[0].get('best_wer')
            if b_wer and h_wer:
                # Relative Error Reduction = (baseline - improved) / baseline * 100
                rer = (b_wer - h_wer) / b_wer * 100
                improvements.append(rer)
                valid_depths.append(depth)
    
    if not improvements:
        print("  Not enough data for relative improvement chart")
        return
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    x = np.arange(len(valid_depths))
    bars = ax.bar(x, improvements, color=COLORS['hier_gate'], 
                  edgecolor='black', linewidth=0.5)
    
    # Add value labels
    for bar, val in zip(bars, improvements):
        ax.annotate(f'{val:.1f}%',
                   xy=(bar.get_x() + bar.get_width()/2, val),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=10)
    
    ax.set_xlabel('Number of Layers')
    ax.set_ylabel('Relative Error Reduction (%)')
    ax.set_title('Relative Improvement from Hierarchical Init + Gating')
    ax.set_xticks(x)
    ax.set_xticklabels([f'L={d}' for d in valid_depths])
    ax.grid(True, axis='y', alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def generate_results_table(experiments: List[Dict], output_path: str):
    """
    Generate a text file with tabulated results matching Table 5.5 format.
    """
    # Sort by depth, then by config type
    experiments = sorted(experiments, 
                        key=lambda x: (x.get('n_layers', 0), 
                                      get_config_type(x) != 'baseline'))
    
    with open(output_path, 'w') as f:
        f.write("EXPERIMENT RESULTS SUMMARY\n")
        f.write("=" * 90 + "\n")
        f.write("Table matching thesis Table 5.5 format\n")
        f.write("=" * 90 + "\n\n")
        
        # Header
        f.write(f"{'Depth':<8} {'Width':<8} {'Hier':<6} {'Gate':<6} {'Best WER':<12} {'Best Epoch':<12}\n")
        f.write("-" * 90 + "\n")
        
        current_depth = None
        for exp in experiments:
            depth = exp.get('n_layers', '-')
            
            # Add separator between depths
            if current_depth is not None and depth != current_depth:
                f.write("-" * 90 + "\n")
            current_depth = depth
            
            width = exp.get('d_model', '-')
            hier = 'Yes' if exp.get('use_hierarchical') else 'No'
            gate = 'Yes' if exp.get('use_gating') else 'No'
            wer = f"{exp.get('best_wer', 0):.2f}%" if exp.get('best_wer') else 'N/A'
            epoch = exp.get('best_epoch', 'N/A')
            
            f.write(f"{str(depth):<8} {str(width):<8} {hier:<6} {gate:<6} {wer:<12} {str(epoch):<12}\n")
        
        f.write("\n" + "=" * 90 + "\n")
        
        # Summary statistics
        f.write("\nSUMMARY BY DEPTH:\n")
        f.write("-" * 50 + "\n")
        
        depths = sorted(set(e.get('n_layers') for e in experiments if e.get('n_layers')))
        for depth in depths:
            depth_exps = [e for e in experiments if e.get('n_layers') == depth]
            baseline = [e for e in depth_exps 
                       if not e.get('use_hierarchical') and not e.get('use_gating')]
            hiergating = [e for e in depth_exps 
                        if e.get('use_hierarchical') and e.get('use_gating')]
            
            if baseline and hiergating:
                b_wer = baseline[0].get('best_wer')
                h_wer = hiergating[0].get('best_wer')
                if b_wer and h_wer:
                    abs_imp = b_wer - h_wer
                    rel_imp = abs_imp / b_wer * 100
                    f.write(f"L={depth}: Baseline {b_wer:.2f}% → Hier+Gate {h_wer:.2f}%  "
                           f"(Δ={abs_imp:.2f}pp, RER={rel_imp:.1f}%)\n")
    
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate thesis figures from v7.4 experiment outputs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Expected directory structure:
    base_dir/
    ├── checkpoints_v74_baseline_W386_D9/
    │   ├── 0utput_exp-v74_baseline_W386_D9.txt
    │   └── config.json
    ├── checkpoints_v74_hier_gating_W358_D9/
    │   └── ...
    └── ...

Example:
    python generate_thesis_figures.py --exp-dir ~/IKT464/Medium_Subset --output-dir ./figures
        """)
    parser.add_argument('--exp-dir', type=str, required=True,
                        help='Directory containing checkpoints_* subdirectories')
    parser.add_argument('--output-dir', type=str, default='./thesis_figures',
                        help='Directory to save generated figures')
    parser.add_argument('--ablation-depth', type=int, default=12,
                        help='Depth to use for ablation study bar chart')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find and parse all experiments
    print(f"Scanning for experiments in: {args.exp_dir}")
    print(f"Looking for directories matching 'checkpoints_*'\n")
    
    experiments = find_experiments(args.exp_dir)
    print(f"\nFound {len(experiments)} complete experiments\n")
    
    if not experiments:
        print("No experiments found. Check the directory path.")
        print("Expected structure: checkpoints_v74_*/config.json")
        return
    
    # Print summary
    print("Experiments found:")
    print("-" * 70)
    for exp in sorted(experiments, key=lambda x: (x.get('n_layers', 0), 
                                                   get_config_type(x) != 'baseline')):
        config_type = get_config_type(exp)
        print(f"  L={exp.get('n_layers'):2d}, D={exp.get('d_model'):3d}, "
              f"{'Hier' if exp.get('use_hierarchical') else '    '} "
              f"{'Gate' if exp.get('use_gating') else '    '} "
              f"→ Best WER: {exp.get('best_wer', 'N/A'):>6.2f}%" if exp.get('best_wer') else "")
    print("-" * 70)
    print()
    
    # Generate figures
    print("Generating figures...")
    
    # 1. WER comparison across all experiments
    print("\n1. WER comparison (all experiments)...")
    plot_wer_comparison(experiments, str(output_dir / 'wer_comparison_all.png'))
    
    # 2. Depth comparison bar chart (Baseline vs Hier+Gate)
    print("\n2. Depth comparison bars...")
    plot_depth_comparison_bars(experiments, str(output_dir / 'depth_comparison_bars.png'))
    
    # 3. Relative improvement chart
    print("\n3. Relative improvement chart...")
    plot_relative_improvement(experiments, str(output_dir / 'relative_improvement.png'))
    
    # 4. Ablation study bar chart (if data available)
    print(f"\n4. Ablation study (L={args.ablation_depth})...")
    plot_ablation_comparison(experiments, str(output_dir / 'ablation_study_bars.png'),
                            target_depth=args.ablation_depth)
    
    # 5. Results summary table
    print("\n5. Results summary table...")
    generate_results_table(experiments, str(output_dir / 'results_summary.txt'))
    
    print(f"\n{'='*70}")
    print(f"All outputs saved to: {output_dir}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()