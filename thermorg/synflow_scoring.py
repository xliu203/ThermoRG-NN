"""
NASWOT-style Zero-Cost Architecture Scoring
============================================
Implements SynFlow (Synaptic Flow) scoring for zero-cost architecture evaluation.
Based on: "SynFlow: Pruning Neural Networks without Data" (ID: 1906.04326)

The core idea: measure the importance of each weight by gradient × weight at initialization.
Architecture score = sum of |grad| for all weights after one forward-backward pass.

This runs on CPU and takes ~minutes for the entire search space.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import json
import time
from typing import List, Dict, Tuple

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# =============================================================================
# ThermoNet Architecture Definitions
# =============================================================================

class ConvBlock(nn.Module):
    """Basic conv block: Conv2d + Activation + (optional Norm)"""
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, activation='gelu', norm_type=None):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=kernel_size//2, bias=(norm_type is None))
        self.norm = None
        if norm_type == 'bn':
            self.norm = nn.BatchNorm2d(out_ch)
        elif norm_type == 'ln':
            self.norm = nn.LayerNorm(out_ch)
        self.activation = activation.lower()
    
    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation == 'gelu':
            x = F.gelu(x)
        elif self.activation == 'relu':
            x = F.relu(x)
        return x


class ThermoNet(nn.Module):
    """Plain ConvNet with configurable width, depth, normalization"""
    def __init__(self, width, depth, norm_type='bn', skip=False, in_ch=3, num_classes=10, kernel_size=3):
        super().__init__()
        self.skip = skip
        
        # First layer
        self.blocks = nn.ModuleList()
        current_ch = in_ch
        
        for layer_idx in range(depth):
            out_ch = width
            stride = 1
            self.blocks.append(ConvBlock(current_ch, out_ch, kernel_size, stride, 'gelu', norm_type))
            current_ch = out_ch
        
        # Global pooling + classifier
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(width, num_classes)
        
        # Track if skip connections are used
        self.skip = skip
        self.skip_idx = []  # layers with skip connections
        
    def forward(self, x):
        for i, block in enumerate(self.blocks):
            x = block(x)
        x = self.pool(x).flatten(1)
        x = self.fc(x)
        return x
    
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())


def create_thermonet(width, depth, norm_type='bn', skip=False):
    """Factory function to create ThermoNet"""
    return ThermoNet(width=width, depth=depth, norm_type=norm_type, skip=skip)


# =============================================================================
# SynFlow Score Computation
# =============================================================================

def compute_synflow_score(model, data_loader, device='cpu'):
    """
    Compute SynFlow score for a model architecture.
    
    SynFlow score = sum of |grad_i * w_i| for all parameters i
    where grad is computed with a constant loss (sum of logits) at initialization,
    making the score independent of training data.
    
    Args:
        model: PyTorch model
        data_loader: Dummy data loader (random tensors)
        device: 'cpu' or 'cuda'
    
    Returns:
        float: SynFlow score (higher = potentially better)
    """
    model.train()  # Enable gradient computation
    model.to(device)
    
    # Create dummy input
    x = torch.randn(16, 3, 32, 32, device=device)
    
    # Forward pass
    output = model(x)  # [batch, num_classes]
    
    # SynFlow trick: use sum of output logits as loss (data-independent)
    loss = output.sum()
    
    # Backward pass
    model.zero_grad()
    loss.backward()
    
    # Compute score: sum of |weight * gradient|
    total_score = 0.0
    for param in model.parameters():
        if param.grad is not None:
            total_score += (param.abs() * param.grad.abs()).sum().item()
    
    return total_score


def compute_naswot_score_batch(models: List[nn.Module], data_loader, device='cpu') -> List[float]:
    """
    Compute NASWOT (SynFlow) scores for a batch of architectures.
    Runs on CPU efficiently.
    """
    scores = []
    for model in models:
        score = compute_synflow_score(model, data_loader, device)
        scores.append(score)
    return scores


# =============================================================================
# Search Space Definition (matching Phase B2)
# =============================================================================

def build_search_space():
    """
    Build the search space matching Phase B2 experiments.
    Widths: 24, 32, 48, 64, 96
    Depths: 3, 5, 6
    Norm types: bn, none
    Skip: False (Phase B2 used no-skip architectures)
    
    Returns:
        List of architecture configs
    """
    widths = [24, 32, 48, 64, 96]
    depths = [3, 5, 6]
    norm_types = ['bn', 'none']
    skip_types = [False]  # Phase B2 used no-skip
    
    configs = []
    for w in widths:
        for d in depths:
            for norm in norm_types:
                for skip in skip_types:
                    config = {
                        'width': w,
                        'depth': d,
                        'norm': norm,
                        'skip': skip,
                        'name': f"TN-{d}-W{w}-{norm.upper()}"
                    }
                    configs.append(config)
    
    return configs


# =============================================================================
# Main: Score all architectures and compare with Phase B2 results
# =============================================================================

def main():
    print("=" * 70)
    print("NASWOT (SynFlow) Zero-Cost Architecture Scoring")
    print("=" * 70)
    
    device = 'cpu'
    print(f"\nDevice: {device}")
    
    # Build search space
    configs = build_search_space()
    print(f"\nSearch space: {len(configs)} architectures")
    print(f"  Widths: {[24, 32, 48, 64, 96]}")
    print(f"  Depths: {[3, 5, 6]}")
    print(f"  Norm types: ['bn', 'none']")
    print(f"  Skip: False")
    
    # Create dummy data for scoring
    dummy_x = torch.randn(16, 3, 32, 32)
    dummy_y = torch.zeros(16, dtype=torch.long)
    dummy_loader = DataLoader(TensorDataset(dummy_x, dummy_y), batch_size=16)
    
    # Compute scores for all architectures
    print("\n" + "-" * 70)
    print("Computing SynFlow scores...")
    print("-" * 70)
    
    start_time = time.time()
    results = []
    
    for i, cfg in enumerate(configs):
        model = create_thermonet(
            width=cfg['width'],
            depth=cfg['depth'],
            norm_type=cfg['norm'],
            skip=cfg['skip']
        )
        
        score = compute_synflow_score(model, dummy_loader, device)
        params = model.get_num_params()
        
        results.append({
            **cfg,
            'synflow_score': score,
            'params': params,
            'score_per_param': score / params if params > 0 else 0
        })
        
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{len(configs)}] {cfg['name']}: score={score:.2e}, params={params:,}")
    
    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    
    # Sort by SynFlow score (descending = better architecture)
    results_sorted = sorted(results, key=lambda x: x['synflow_score'], reverse=True)
    
    # Print rankings
    print("\n" + "=" * 70)
    print("NASWOT Architecture Ranking (by SynFlow Score)")
    print("=" * 70)
    print(f"{'Rank':<5} {'Config':<20} {'SynFlow Score':<15} {'Params':<12} {'Score/Param':<12}")
    print("-" * 70)
    for i, r in enumerate(results_sorted):
        print(f"{i+1:<5} {r['name']:<20} {r['synflow_score']:<15.2e} {r['params']:<12,} {r['score_per_param']:<12.2e}")
    
    # Load Phase B2 results for comparison
    print("\n" + "=" * 70)
    print("Comparison with Phase B2 (HBO vs Random results)")
    print("=" * 70)
    
    # Phase B2 best architectures found
    phase_b2_hbo = [
        ('96/6/BN/False', 0.703, 1),
        ('64/6/BN/False', 0.781, 2),
        ('48/6/BN/False', 0.82, 3),
        ('64/5/BN/False', 0.85, 4),
        ('48/5/BN/False', 0.90, 5),
    ]
    
    phase_b2_random = [
        ('64/6/BN/False', 0.781, 1),
        ('64/5/BN/False', 0.82, 2),
        ('48/6/BN/False', 0.85, 3),
        ('64/4/BN/False', 0.90, 4),
        ('48/5/BN/False', 0.90, 5),
    ]
    
    print("\nPhase B2 HBO top-5:")
    for name, loss, rank in phase_b2_hbo:
        print(f"  Rank {rank}: {name} (loss={loss})")
    
    print("\nPhase B2 Random top-5:")
    for name, loss, rank in phase_b2_random:
        print(f"  Rank {rank}: {name} (loss={loss})")
    
    # Compare with NASWOT ranking
    print("\n" + "-" * 70)
    print("Does NASWOT rank the same architectures at the top?")
    print("-" * 70)
    
    # Create lookup for NASWOT scores
    naswot_lookup = {r['name']: r['synflow_score'] for r in results}
    
    # Check overlap
    # Phase B2 naming: "96/6/BN/False" -> our naming: "TN-6-W96-BN"
    hbo_top5_configs = []
    for name, loss, rank in phase_b2_hbo:
        parts = name.split('/')  # "96/6/BN/False" -> ["96", "6", "BN", "False"]
        w, d, norm, skip = parts
        hbo_top5_configs.append({
            'width': int(w), 'depth': int(d), 'norm': norm.lower(), 'skip': skip.lower() == 'true'
        })
    
    print("\nNASWOT top-10:")
    for i, r in enumerate(results_sorted[:10]):
        # Check if in HBO top-5
        in_hbo = any(
            r['width'] == c['width'] and r['depth'] == c['depth'] and r['norm'] == c['norm']
            for c in hbo_top5_configs
        )
        match_str = "✓ HBO top-5" if in_hbo else ""
        print(f"  {i+1}. {r['name']}: score={r['synflow_score']:.2e} {match_str}")
    
    # Save results
    output = {
        'experiment': 'naswot_comparison',
        'date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'device': device,
        'runtime_seconds': elapsed,
        'search_space_size': len(configs),
        'results': results_sorted,
        'phase_b2_hbo_top5': phase_b2_hbo,
        'phase_b2_random_top5': phase_b2_random,
    }
    
    output_file = 'naswot_results.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    return results_sorted


if __name__ == '__main__':
    results = main()