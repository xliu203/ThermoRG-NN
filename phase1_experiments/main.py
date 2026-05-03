#!/usr/bin/env python3
"""
ThermoRG v4 Experiment - Main Execution Script (Kaggle-Adapted)

Kaggle-specific features:
- Auto-detects Kaggle environment (/kaggle/ path)
- Graceful SIGTERM handling for timeout
- Checkpoint-based resumability
- Incremental per-run results saving
- Phase 0.2 can be split into sub-runs

Usage:
    python main.py --phase 0.1    # EOS verification
    python main.py --phase 0.2    # BN vs LN core test
    python main.py --phase 0.2 --D-start 0 --D-end 1  # Split: first D value only
    python main.py --phase 0.2 --resume  # Resume partial run
    
    # Phase 0.2 split by lr group:
    python main.py --phase 0.2 --lr-group 0  # First half of learning rates
    python main.py --phase 0.2 --lr-group 1  # Second half of learning rates
"""

import argparse
import os
import sys
import json
import yaml
import torch
import signal
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional, List

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.convnet import create_model, ConvNetL5
from experiments.train import (
    train_and_measure, run_experiment_grid,
    get_run_id, CheckpointManager
)
from utils.measurements import (
    measure_gamma, measure_lambda_max_mean, is_stationary,
    fit_beta, save_results, save_run_result, load_run_results,
    PROTOCOL_CONSTANTS
)

from analysis.statistics import aggregate_results, fit_beta_vs_ln_gamma, f_test_equal_slopes


# =============================================================================
# KAGGLE ENVIRONMENT DETECTION
# =============================================================================

def is_kaggle_environment() -> bool:
    """Detect if running on Kaggle."""
    return os.path.exists('/kaggle/') and os.path.exists('/kaggle/input')


def get_kaggle_working_dir() -> str:
    """Get Kaggle working directory."""
    return '/kaggle/working/'


def get_kaggle_data_dir() -> str:
    """Get Kaggle data directory (where datasets are mounted)."""
    return '/kaggle/input'


def get_output_dir(base_output: str, phase: str, create: bool = True) -> str:
    """
    Get output directory with timestamp for uniqueness.
    
    On Kaggle: /kaggle/working/phase_{phase}_{timestamp}/
    Otherwise: {base_output}/phase_{phase}_{timestamp}/
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if is_kaggle_environment():
        out_dir = os.path.join(get_kaggle_working_dir(), f'phase_{phase}_{timestamp}')
    else:
        out_dir = os.path.join(base_output, f'phase_{phase}_{timestamp}')
    
    if create:
        os.makedirs(out_dir, exist_ok=True)
    
    return out_dir


# =============================================================================
# GRACEFUL EXIT HANDLING
# =============================================================================

class GracefulExit:
    """Context manager for graceful exit on SIGTERM."""
    
    def __init__(self):
        self.exit_requested = False
        self._original_sigterm = None
        
    def request_exit(self, signum, frame):
        print("\n[GRACEFUL] SIGTERM received - saving state and exiting...")
        self.exit_requested = True
        
    def __enter__(self):
        # Register SIGTERM handler
        self._original_sigterm = signal.signal(signal.SIGTERM, self.request_exit)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original handler
        signal.signal(signal.SIGTERM, self._original_sigterm)
        
        if exc_type is not None:
            # Exception occurred - will re-raise after cleanup
            return False
        return True


# =============================================================================
# DATA LOADING
# =============================================================================

def load_cifar10(batch_size: int = 128, data_dir: str = None):
    """
    Load CIFAR-10 dataset.
    
    Supports two modes:
    - Fast.ai image version (Kaggle): uses ImageFolder at /kaggle/working/cifar10_fastai/cifar10/
    - Standard version (local): uses torchvision.datasets.CIFAR10
    """
    import torchvision
    import torchvision.transforms as transforms
    from torchvision.datasets import ImageFolder
    
    # Normalization values (same as before)
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    # Determine which version to use
    fastai_path = '/kaggle/working/cifar10_fastai/cifar10'
    use_fastai = is_kaggle_environment() and os.path.exists(fastai_path)
    
    if use_fastai:
        # Fast.ai image version — use ImageFolder
        train_root = os.path.join(fastai_path, 'train')
        test_root = os.path.join(fastai_path, 'test')
        
        print(f"[Data] Loading CIFAR-10 (Fast.ai images) from: {fastai_path}")
        
        trainset = ImageFolder(root=train_root, transform=transform_train)
        testset = ImageFolder(root=test_root, transform=transform_test)
    else:
        # Standard torchvision CIFAR-10
        if data_dir is None:
            data_dir = './data' if is_kaggle_environment() else './data'
        
        print(f"[Data] Loading CIFAR-10 (torchvision) from: {data_dir}")
        
        trainset = torchvision.datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=transform_train
        )
        testset = torchvision.datasets.CIFAR10(
            root=data_dir, train=False, download=True, transform=transform_test
        )
    
    # num_workers=0 for Kaggle (same as previous experiments)
    num_workers = 0 if is_kaggle_environment() else 2
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    print(f"[Data] Loaded {len(trainset)} train, {len(testset)} test samples")
    
    return trainloader, testloader


# =============================================================================
# CONFIG LOADING
# =============================================================================

def load_phase_config(phase: str) -> dict:
    """Load configuration for a given phase."""
    config_path = Path(__file__).parent / 'configs' / f'phase_{phase}.yaml'
    if not config_path.exists():
        raise FileNotFoundError(f"Config for phase {phase} not found at {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def get_adjusted_batch_size(batch_size: int = 128) -> int:
    """Adjust batch size based on environment."""
    if is_kaggle_environment():
        # Kaggle typically has P100 or T4 with 16GB RAM
        # 128 should be fine, but reduce if OOM
        return min(batch_size, 128)
    return batch_size


# =============================================================================
# EXPERIMENT RUNNERS
# =============================================================================

def run_phase_0_1(config: dict, device: torch.device, output_dir: str, graceful_exit: GracefulExit):
    """
    Phase 0.1: EOS Boundary Verification
    - ConvNet L=5, D=256, BN vs LN
    - 8 learning rates, 3 seeds, 20 epochs
    - Measure: η_crit, λ_max_init
    """
    print("\n" + "="*70)
    print("PHASE 0.1: EOS BOUNDARY VERIFICATION")
    print("="*70)
    
    batch_size = get_adjusted_batch_size(config['training'].get('batch_size', 128))
    trainloader, testloader = load_cifar10(batch_size=batch_size)
    
    # Extract config
    norm_types = config['norm_types']
    D = config['D_values'][0]
    lr_values = config['lr_values']
    seeds = config['seeds']
    epochs = config['training']['epochs']
    
    results_all = {}
    eos_results = {}
    
    try:
        for norm_type in norm_types:
            print(f"\n>>> Normalization: {norm_type}")
            eos_results[norm_type] = {}
            
            for lr in lr_values:
                print(f"\n  LR = {lr}")
                eos_results[norm_type][lr] = {}
                
                for seed in seeds:
                    # Check for graceful exit
                    if graceful_exit.exit_requested:
                        print("[GRACEFUL] Exiting after current run completes...")
                        break
                    
                    print(f"    Seed {seed}...", end=" ")
                    
                    # Generate run ID for checkpointing
                    run_id = get_run_id(D, norm_type, lr, seed)
                    
                    # Check for existing checkpoint
                    checkpoint_manager = CheckpointManager(output_dir)
                    start_epoch = 0
                    if checkpoint_manager.checkpoint_exists(run_id):
                        print(f"[Resume] Found checkpoint for {run_id}, loading...")
                        checkpoint = checkpoint_manager.load(run_id)
                        start_epoch = checkpoint['epoch'] + 1
                        if start_epoch >= epochs:
                            print(f"  Run already complete (epoch {start_epoch}), skipping...")
                            continue
                    
                    # Train
                    result = train_and_measure(
                        D=D,
                        norm_type=norm_type,
                        lr=lr,
                        seed=seed,
                        dataloader_train=trainloader,
                        dataloader_eval=testloader,
                        device=device,
                        num_epochs=epochs,
                        verbose=False,
                        output_dir=output_dir,
                        start_epoch=start_epoch,
                    )
                    
                    # Store key metrics
                    eos_results[norm_type][lr][seed] = {
                        'lambda_max_init': result['lambda_max_init'],
                        'loss_final': result['loss_history'][-1],
                        'loss_init': result['loss_history'][0],
                        'is_converged': result['is_converged'],
                    }
                    
                    # Check EOS: λ_max * η should be ≈ 2
                    lambda_init = result['lambda_max_init']
                    eos_ratio = lambda_init * lr
                    eos_results[norm_type][lr][seed]['eos_ratio'] = eos_ratio
                    
                    print(f"λ_max={lambda_init:.3f}, λ_max*η={eos_ratio:.3f}, converged={result['is_converged']}")
                    
                    # Save result incrementally
                    save_run_result(result, output_dir)
                
                if graceful_exit.exit_requested:
                    break
            if graceful_exit.exit_requested:
                break
    
    finally:
        # Always save results on exit
        print("\n[Saving] Final results...")
        save_path = os.path.join(output_dir, 'phase_0.1_results.json')
        save_results({'eos_results': eos_results, 'config': config}, save_path)
        print(f"Results saved to {save_path}")
    
    return eos_results


def run_phase_0_2(
    config: dict,
    device: torch.device,
    output_dir: str,
    graceful_exit: GracefulExit,
    D_start: Optional[int] = None,
    D_end: Optional[int] = None,
    lr_group: Optional[int] = None,
):
    """
    Phase 0.2: BN vs LN Parallel Lines Core Test
    - ConvNet L=5, D ∈ {64, 256, 1024}, BN vs LN
    - 6 learning rates, 3 seeds, 200 epochs
    - Measure: γ, β (D-scaling), fit β vs ln(γ)
    
    Can be split via:
    - D_start/D_end: Train on D_values[D_start:D_end] only
    - lr_group: Train on roughly half of lr_values (split into 2 groups)
    """
    print("\n" + "="*70)
    print("PHASE 0.2: BN vs LN PARALLEL LINES CORE TEST")
    print("="*70)
    
    # Handle splits
    if D_start is not None or D_end is not None:
        all_D_values = config['D_values']
        start_idx = D_start if D_start is not None else 0
        end_idx = D_end if D_end is not None else len(all_D_values)
        D_values = all_D_values[start_idx:end_idx]
        print(f"[Split] Training on D_values: {D_values}")
    else:
        D_values = config['D_values']
    
    if lr_group is not None:
        all_lr_values = config['lr_values']
        n_lrs = len(all_lr_values)
        # Split into 2 groups of roughly equal size
        n_groups = 2
        group_size = (n_lrs + n_groups - 1) // n_groups  # Ceiling division
        lr_start = lr_group * group_size
        lr_end = min((lr_group + 1) * group_size, n_lrs)
        lr_values = all_lr_values[lr_start:lr_end]
        print(f"[Split] Training on lr_values[{lr_group}]: {lr_values}")
    else:
        lr_values = config['lr_values']
    
    batch_size = get_adjusted_batch_size(config['training'].get('batch_size', 128))
    trainloader, testloader = load_cifar10(batch_size=batch_size)
    
    # Extract config
    norm_types = config['norm_types']
    seeds = config['seeds']
    epochs = config['training']['epochs']
    
    # Build experiment grid
    experiment_grid = {
        'norm_types': norm_types,
        'D_values': D_values,
        'lr_values': lr_values,
        'seeds': seeds,
        'epochs': epochs,
    }
    
    # Check for resume - load existing run results
    results_all = load_run_results(output_dir)
    if results_all:
        print(f"[Resume] Loaded {len(results_all)} previous run results from {output_dir}")
    
    # Run experiment grid with graceful exit support
    try:
        new_results = run_experiment_grid(
            experiment_config=experiment_grid,
            dataloader_train=trainloader,
            dataloader_eval=testloader,
            device=device,
            output_dir=output_dir,
            verbose=True,
            graceful_exit=graceful_exit,
        )
        
        results_all.extend(new_results)
    
    finally:
        # Always save results on exit
        print("\n[Saving] Final results...")
        save_path = os.path.join(output_dir, 'phase_0.2_results.json')
        save_results({'runs': results_all, 'config': config}, save_path)
        print(f"Results saved to {save_path}")
    
    # Analyze results (only if we have all results or just do what we can)
    if not results_all:
        print("No results to analyze.")
        return [], {}, {}
    
    print("\n" + "="*70)
    print("ANALYSIS: β vs ln(γ) LINEAR REGRESSION")
    print("="*70)
    
    # Aggregate by (norm_type, lr) - average over seeds and D
    aggregated = aggregate_results(results_all)
    
    # Fit β vs ln(γ) per norm type
    regression_results = {}
    for norm_type in norm_types:
        beta_gamma_pairs = [(r['beta'], r['gamma']) for r in aggregated 
                           if r['norm_type'] == norm_type 
                           and r.get('beta') is not None 
                           and r.get('gamma') > 0]
        
        if len(beta_gamma_pairs) >= 3:
            reg_result = fit_beta_vs_ln_gamma(beta_gamma_pairs, norm_type)
            regression_results[norm_type] = reg_result
            
            print(f"\n{norm_type}:")
            print(f"  Slope m = {reg_result['m']:.4f}")
            print(f"  Intercept c = {reg_result['c']:.4f}")
            print(f"  R² = {reg_result['r_squared']:.4f}")
            print(f"  n_points = {reg_result['n_points']}")
        else:
            print(f"\n{norm_type}: Not enough valid points ({len(beta_gamma_pairs)})")
    
    # F-test for equal slopes
    f_test = None
    if len(regression_results) >= 2:
        f_test = f_test_equal_slopes(regression_results)
        print(f"\nF-test for equal slopes:")
        print(f"  F = {f_test['f_statistic']:.4f}")
        print(f"  p = {f_test['p_value']:.4f}")
        print(f"  Reject H0 (slopes differ)? {f_test['reject_null']}")
    
    # Save analysis
    analysis_path = os.path.join(output_dir, 'phase_0.2_analysis.json')
    save_results({
        'aggregated': aggregated,
        'regression': regression_results,
        'f_test': f_test if len(regression_results) >= 2 else None
    }, analysis_path)
    
    return results_all, aggregated, regression_results


def run_phase_1_1(config: dict, device: torch.device, output_dir: str, graceful_exit: GracefulExit):
    """
    Phase 1.1: Full TPU sweep
    - 4 norm types × 6 D × 8 η × 5 seeds
    - 40 epochs
    """
    print("\n" + "="*70)
    print("PHASE 1.1: FULL NORMALIZATION × D × η SWEEP")
    print("="*70)
    
    batch_size = get_adjusted_batch_size(config['training'].get('batch_size', 128))
    trainloader, testloader = load_cifar10(batch_size=batch_size)
    
    # Extract config
    norm_types = config['norm_types']
    D_values = config['D_values']
    lr_values = config['lr_values']
    seeds = config['seeds']
    epochs = config['training']['epochs']
    
    experiment_grid = {
        'norm_types': norm_types,
        'D_values': D_values,
        'lr_values': lr_values,
        'seeds': seeds,
        'epochs': epochs,
    }
    
    # Load existing results
    results_all = load_run_results(output_dir)
    if results_all:
        print(f"[Resume] Loaded {len(results_all)} previous run results")
    
    try:
        new_results = run_experiment_grid(
            experiment_config=experiment_grid,
            dataloader_train=trainloader,
            dataloader_eval=testloader,
            device=device,
            output_dir=output_dir,
            verbose=True,
            graceful_exit=graceful_exit,
        )
        
        results_all.extend(new_results)
    
    finally:
        # Save
        save_path = os.path.join(output_dir, 'phase_1.1_results.json')
        save_results({'runs': results_all, 'config': config}, save_path)
        print(f"Results saved to {save_path}")
    
    return results_all


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='ThermoRG v4 Experiments (Kaggle-Adapted)')
    parser.add_argument('--phase', type=str, default='0.1',
                       help='Experiment phase: 0.1, 0.2, 0.3, 1.1, etc.')
    parser.add_argument('--config', type=str, default=None,
                       help='Custom config file path (overrides --phase)')
    parser.add_argument('--output', type=str, default='./outputs',
                       help='Output directory (default: ./outputs)')
    parser.add_argument('--device', type=str, default=None,
                       help='Device (cuda/cpu), auto-detect if not specified')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from partial results')
    parser.add_argument('--analyze-only', type=str, default=None,
                       help='Path to results file for analysis only')
    
    # Phase 0.2 splitting options
    parser.add_argument('--D-start', type=int, default=None,
                       help='Phase 0.2: Start index for D_values slice')
    parser.add_argument('--D-end', type=int, default=None,
                       help='Phase 0.2: End index for D_values slice')
    parser.add_argument('--lr-group', type=int, default=None,
                       help='Phase 0.2: Learning rate group (0=first half, 1=second half)')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Batch size (default: 128)')
    
    args = parser.parse_args()
    
    # Print environment info
    print("\n" + "="*70)
    print("ThermoRG v4 Experiment Runner (Kaggle-Adapted)")
    print("="*70)
    print(f"Kaggle Environment: {is_kaggle_environment()}")
    
    # Auto-detect device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Device: {device}")
    print(f"Phase: {args.phase}")
    
    # Create output directory with timestamp
    output_dir = get_output_dir(args.output, args.phase)
    print(f"Output: {output_dir}")
    
    # Setup graceful exit
    graceful_exit = GracefulExit()
    
    # Load config
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = load_phase_config(args.phase)
    
    print(f"\nConfig loaded: {config['experiment']['name']}")
    
    # Handle splitting info for Phase 0.2
    split_info = ""
    if args.D_start is not None or args.D_end is not None:
        split_info = f"_D{args.D_start or 0}-{args.D_end or 'end'}"
    if args.lr_group is not None:
        split_info += f"_LRG{args.lr_group}"
    
    if split_info:
        print(f"Running split: {split_info}")
    
    try:
        with graceful_exit:
            # Run appropriate phase
            if args.phase == '0.1':
                run_phase_0_1(config, device, output_dir, graceful_exit)
                
            elif args.phase == '0.2':
                run_phase_0_2(
                    config, device, output_dir, graceful_exit,
                    D_start=args.D_start, D_end=args.D_end, lr_group=args.lr_group
                )
                
            elif args.phase == '0.3':
                # Phase 0.3: GN + None baseline
                config_03 = load_phase_config('0.3')
                run_phase_0_2(config_03, device, output_dir, graceful_exit)
                
            elif args.phase == '1.1':
                run_phase_1_1(config, device, output_dir, graceful_exit)
                
            else:
                print(f"Unknown phase: {args.phase}")
                sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n[Interrupted] Saving current state...")
        # State is saved via finally blocks
        sys.exit(0)
    
    if graceful_exit.exit_requested:
        print("\n[GRACEFUL] Exit completed. Results saved.")
    else:
        print(f"\n{'='*70}")
        print(f"PHASE {args.phase} COMPLETE")
        print(f"Results saved to: {output_dir}")
        print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
