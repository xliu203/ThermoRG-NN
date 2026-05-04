"""
Training loop for ThermoRG v4 experiments (Kaggle-Adapted).

Features for Kaggle:
- Checkpoint saving every N epochs
- Resume from checkpoint support
- Per-run results saving (JSON per run)
- Graceful SIGTERM handling
- Unique run IDs for resumability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
import json
import os
import re
from datetime import datetime
import hashlib

from models.convnet import ConvNetL5, create_model
from utils.measurements import (
    measure_gamma, measure_lambda_max_mean, is_stationary,
    find_stationary_epoch, power_iteration_single_layer,
    fit_beta, save_results, save_run_result, load_run_results
)


# =============================================================================
# CHECKPOINT UTILITIES
# =============================================================================

def get_run_id(D: int, norm_type: str, lr: float, seed: int) -> str:
    """
    Generate a unique run ID for a specific experiment configuration.

    Format: norm_{norm}_D{D}_lr{lr}_seed{seed}
    The lr is formatted to avoid decimal issues.
    """
    # Format lr to avoid floating point issues
    lr_str = f"{lr:.6f}".rstrip('0').rstrip('.')
    return f"norm_{norm_type}_D{D}_lr{lr_str}_seed{seed}"


def get_checkpoint_filename(run_id: str, epoch: int) -> str:
    """Get checkpoint filename for a run at a specific epoch."""
    return f"checkpoint_{run_id}_epoch_{epoch}.pt"


def get_results_filename(run_id: str) -> str:
    """Get results filename for a run."""
    return f"result_{run_id}.json"


class CheckpointManager:
    """
    Manages checkpoint saving and loading for Kaggle resumability.

    Checkpoint contains:
    - Model state dict
    - Optimizer state dict
    - Scheduler state dict
    - Epoch number
    - Loss history
    - Lambda max history
    - Training state flags
    """

    def __init__(self, output_dir: str, checkpoint_every: int = 20):
        """
        Args:
            output_dir: Directory to save checkpoints
            checkpoint_every: Save checkpoint every N epochs
        """
        self.output_dir = output_dir
        self.checkpoint_every = checkpoint_every

    def get_checkpoint_path(self, run_id: str, epoch: int) -> str:
        """Get full path to checkpoint file."""
        return os.path.join(self.output_dir, get_checkpoint_filename(run_id, epoch))

    def find_latest_checkpoint(self, run_id: str) -> Optional[Tuple[str, int]]:
        """Find the latest checkpoint epoch for a run."""
        prefix = f"checkpoint_{run_id}_epoch_"
        latest_epoch = -1
        latest_path = None

        for fname in os.listdir(self.output_dir):
            if fname.startswith(prefix) and fname.endswith('.pt'):
                match = re.search(r'epoch_(\d+)\.pt$', fname)
                if match:
                    epoch = int(match.group(1))
                    if epoch > latest_epoch:
                        latest_epoch = epoch
                        latest_path = os.path.join(self.output_dir, fname)

        return (latest_path, latest_epoch) if latest_path else None

    def save(
        self,
        run_id: str,
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler._LRScheduler,
        epoch: int,
        loss_history: List[float],
        lambda_max_history: List[float],
        training_state: Dict,
        additional_data: Optional[Dict] = None,
    ) -> str:
        """
        Save a checkpoint.

        Returns:
            Path to saved checkpoint
        """
        checkpoint_path = self.get_checkpoint_path(run_id, epoch)

        # Ensure output directory exists before writing
        os.makedirs(self.output_dir, exist_ok=True)

        checkpoint = {
            'run_id': run_id,
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'loss_history': loss_history,
            'lambda_max_history': lambda_max_history,
            'training_state': training_state,
            'timestamp': datetime.now().isoformat(),
        }

        if additional_data:
            checkpoint.update(additional_data)

        temp_path = checkpoint_path + '.tmp'
        torch.save(checkpoint, temp_path)
        os.rename(temp_path, checkpoint_path)  # Atomic on POSIX
        return checkpoint_path

    def load(self, run_id: str, epoch: Optional[int] = None, device: str = 'cpu') -> Dict:
        """
        Load a checkpoint.

        Args:
            run_id: Run identifier
            epoch: Specific epoch to load, or None for latest
            device: Device to map tensors to

        Returns:
            Checkpoint dict
        """
        if epoch is None:
            result = self.find_latest_checkpoint(run_id)
            if result is None:
                raise FileNotFoundError(f"No checkpoint found for run_id: {run_id}")
            checkpoint_path, epoch = result
        else:
            checkpoint_path = self.get_checkpoint_path(run_id, epoch)

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=device)
        return checkpoint


# =============================================================================
# TRAINING STATE
# =============================================================================

class TrainingState:
    """Tracks state during training."""

    def __init__(self):
        self.loss_history = []
        self.gamma_history = []
        self.lambda_max_history = []
        self.epoch = 0
        self.is_stationary = False
        self.stationary_epoch = -1
        self.sigma_init = None
        self.lambda_max_init = None
        self.stationary_lambda_max = None  # λ_max value at stationarity
        self.best_loss = float('inf')
        self.measurements_complete = False


# =============================================================================
# CORE TRAINING FUNCTIONS
# =============================================================================

def train_single_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device
) -> float:
    """Run one training epoch, return average loss."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for data, target in dataloader:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


@torch.no_grad()
def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device) -> float:
    """Evaluate model, return average loss."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for data, target in dataloader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = F.cross_entropy(output, target)
        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def capture_initial_activations(
    model: ConvNetL5,
    dataloader: DataLoader,
    device: torch.device
) -> List[torch.Tensor]:
    """Capture normalized activations at initialization."""
    model.eval()

    # Warm-up forward to populate BN running stats
    with torch.no_grad():
        for data, _ in dataloader:
            model(data.to(device))
            break

    activations = []

    for data, _ in dataloader:
        data = data.to(device)
        x = data

        for i in range(1, 6):
            conv = getattr(model, f'conv{i}')
            norm = getattr(model, f'norm{i}')

            x = conv(x)
            if norm is not None:
                if isinstance(norm, nn.BatchNorm2d):
                    x_norm = (x - norm.running_mean.view(1, -1, 1, 1)) / \
                             torch.sqrt(norm.running_var.view(1, -1, 1, 1) + norm.eps)
                elif isinstance(norm, nn.GroupNorm):
                    x_norm = norm(x)
                else:
                    x_norm = x
            else:
                x_norm = x

            activations.append(x_norm.cpu().detach())
            x = model.activation(x_norm if norm is not None else x)

        break  # Only first batch

    return activations


def capture_current_activations(
    model: ConvNetL5,
    dataloader: DataLoader,
    device: torch.device
) -> List[torch.Tensor]:
    """Capture normalized activations at current training state."""
    return capture_initial_activations(model, dataloader, device)


def compute_sigma_from_activations(activations: List[torch.Tensor]) -> np.ndarray:
    """Compute l2 norm per layer from activations."""
    sigmas = []
    for act in activations:
        act_flat = act.flatten(start_dim=1)
        l2_per_sample = act_flat.norm(p=2, dim=1)
        mean_l2 = l2_per_sample.mean().item()
        sigmas.append(mean_l2)
    return np.array(sigmas)


def train_and_measure(
    D: int,
    norm_type: str,
    lr: float,
    seed: int,
    dataloader_train: DataLoader,
    dataloader_eval: DataLoader,
    device: torch.device,
    num_epochs: int,
    measure_lambda_every: int = 10,
    verbose: bool = True,
    output_dir: Optional[str] = None,
    start_epoch: int = 0,
    checkpoint_every: int = 20,
    graceful_exit: Optional = None,
) -> Dict:
    """
    Complete training run with all measurements.

    Supports checkpoint-based resume: if output_dir is provided and a checkpoint
    exists for this run, training will resume from the checkpoint.

    Args:
        D: Channel width
        norm_type: Normalization type
        lr: Learning rate
        seed: Random seed
        dataloader_train: Training data loader
        dataloader_eval: Evaluation data loader
        device: torch device
        num_epochs: Number of training epochs
        measure_lambda_every: How often to measure λ_max
        verbose: Print progress
        output_dir: Directory for checkpoints and results (required for resume)
        start_epoch: Starting epoch (0 for new run, >0 for resume)
        checkpoint_every: Save checkpoint every N epochs

    Returns:
        Dict with all measurements
    """
    # Generate run ID
    run_id = get_run_id(D, norm_type, lr, seed)

    # Check for existing checkpoint if output_dir provided
    latest = None  # Initialize to prevent NameError when output_dir is None
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        checkpoint_manager = CheckpointManager(output_dir, checkpoint_every=checkpoint_every)
        result_path = os.path.join(output_dir, get_results_filename(run_id))

        # Check if run already completed
        if os.path.exists(result_path):
            if verbose:
                print(f"  [Skip] Result already exists for {run_id}")
            with open(result_path, 'r') as f:
                return json.load(f)

        # Check for checkpoint to resume
        latest = checkpoint_manager.find_latest_checkpoint(run_id)
        if latest and start_epoch == 0:
            checkpoint_path, checkpoint_epoch = latest
            if checkpoint_epoch >= num_epochs - 1:
                if verbose:
                    print(f"  [Skip] Checkpoint found at final epoch for {run_id}")
                # Run completed, but no result file - reconstruct from checkpoint
                ckpt = checkpoint_manager.load(run_id, device=str(device))
                return _reconstruct_result_from_checkpoint(ckpt)
            else:
                if verbose:
                    print(f"  [Resume] Loading checkpoint epoch {checkpoint_epoch} for {run_id}")
                start_epoch = checkpoint_epoch + 1
    else:
        checkpoint_manager = None

    # Set seed (use seed + start_epoch for reproducibility in resume)
    torch.manual_seed(seed + start_epoch)
    np.random.seed(seed + start_epoch)

    # Create model
    model = create_model(D=D, norm_type=norm_type).to(device)

    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Training state
    state = TrainingState()

    # If resuming, restore state from checkpoint
    # (Reuse latest from above instead of overwriting)

    if latest is not None and start_epoch > 0:
        # Restore from checkpoint
        checkpoint = checkpoint_manager.load(run_id, device=str(device))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint['scheduler_state_dict'] and scheduler:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        state.loss_history = checkpoint['loss_history']  # Only contains eval_loss (not train_loss)
        state.lambda_max_history = checkpoint['lambda_max_history']
        state.is_stationary = checkpoint['training_state'].get('is_stationary', False)
        state.stationary_epoch = checkpoint['training_state'].get('stationary_epoch', -1)
        state.sigma_init = checkpoint['training_state'].get('sigma_init')
        state.lambda_max_init = checkpoint['training_state'].get('lambda_max_init')
        state.stationary_lambda_max = checkpoint['training_state'].get('stationary_lambda_max')

        if verbose:
            print(f"  [Resume] Restored from epoch {checkpoint['epoch']}, continuing from epoch {start_epoch}")
    else:
        # ===== INITIAL MEASUREMENTS =====
        if verbose:
            print(f"  [Init] Capturing initial state...")

        # Capture initial activations
        activations_init = capture_initial_activations(model, dataloader_train, device)
        sigma_init = compute_sigma_from_activations(activations_init)
        state.sigma_init = sigma_init

        # Measure initial λ_max
        lambda_init = measure_lambda_max_mean(model, device, num_iterations=20)
        state.lambda_max_init = lambda_init

    # ===== TRAINING LOOP =====
    if verbose:
        print(f"  [Train] Starting training for {num_epochs} epochs (from epoch {start_epoch})...")

    for epoch in range(start_epoch, num_epochs):
        # Train one epoch
        train_loss = train_single_epoch(model, dataloader_train, optimizer, device)
        eval_loss = evaluate(model, dataloader_eval, device)

        state.loss_history.append(eval_loss)
        state.epoch = epoch

        # Check stationarity (skip if already stationary from resume)
        do_periodic_measurement = epoch % measure_lambda_every == 0
        if not state.is_stationary:
            loss_array = np.array(state.loss_history)
            is_stat, stat_epoch = is_stationary(loss_array)

            if is_stat:
                state.is_stationary = True
                state.stationary_epoch = stat_epoch
                if verbose:
                    print(f"  [Stationary] Reached at epoch {stat_epoch}")

                # Measure λ_max at stationarity
                lambda_stat = measure_lambda_max_mean(model, device, num_iterations=20)
                state.lambda_max_history.append(lambda_stat)
                state.stationary_lambda_max = lambda_stat
                # Skip periodic measurement this epoch since we just measured
                do_periodic_measurement = False

        scheduler.step()

        # Periodic λ_max measurement (skip if we just measured at stationarity)
        if do_periodic_measurement:
            lambda_curr = measure_lambda_max_mean(model, device, num_iterations=20)
            state.lambda_max_history.append(lambda_curr)

        if verbose and (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{num_epochs}: loss={eval_loss:.4f}")

        # Save checkpoint periodically
        if checkpoint_manager is not None and epoch % checkpoint_every == 0:
            checkpoint_manager.save(
                run_id=run_id,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                loss_history=state.loss_history,
                lambda_max_history=state.lambda_max_history,
                training_state={
                    'is_stationary': state.is_stationary,
                    'stationary_epoch': state.stationary_epoch,
                    'sigma_init': state.sigma_init.tolist() if state.sigma_init is not None else None,
                    'lambda_max_init': float(state.lambda_max_init) if state.lambda_max_init is not None else None,
                    'stationary_lambda_max': float(state.stationary_lambda_max) if state.stationary_lambda_max is not None else None,
                },
                additional_data={
                    'config': {
                        'D': D,
                        'norm_type': norm_type,
                        'lr': lr,
                        'seed': seed,
                        'num_epochs': num_epochs,
                    }
                },
            )
            if verbose:
                print(f"  [Checkpoint] Saved at epoch {epoch + 1}")

        # Check for graceful exit after each epoch (allow SIGTERM to interrupt mid-epoch)
        if graceful_exit is not None and graceful_exit.exit_requested:
            if verbose:
                print(f"  [Graceful] Exit requested at epoch {epoch + 1}, saving checkpoint and stopping")
            # Save final checkpoint before exiting
            if checkpoint_manager is not None:
                checkpoint_manager.save(
                    run_id=run_id,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    loss_history=state.loss_history,
                    lambda_max_history=state.lambda_max_history,
                    training_state={
                        'is_stationary': state.is_stationary,
                        'stationary_epoch': state.stationary_epoch,
                        'sigma_init': state.sigma_init.tolist() if state.sigma_init is not None else None,
                        'lambda_max_init': float(state.lambda_max_init) if state.lambda_max_init is not None else None,
                        'stationary_lambda_max': float(state.stationary_lambda_max) if state.stationary_lambda_max is not None else None,
                    },
                    additional_data={
                        'config': {
                            'D': D,
                            'norm_type': norm_type,
                            'lr': lr,
                            'seed': seed,
                            'num_epochs': num_epochs,
                        }
                    },
                )
                if verbose:
                    print(f"  [Checkpoint] Saved final checkpoint at epoch {epoch + 1}")
            break

    # ===== FINAL MEASUREMENTS =====
    if verbose:
        print(f"  [Final] Capturing final state...")

    # Capture final activations
    activations_final = capture_current_activations(model, dataloader_train, device)
    sigma_final = compute_sigma_from_activations(activations_final)

    # Measure final λ_max
    lambda_final = measure_lambda_max_mean(model, device, num_iterations=20)

    # Compute γ and γ_init
    if state.sigma_init is None:
        # sigma_init can be None when resuming from a legacy checkpoint
        # that didn't save sigma_init. Skip gamma computation.
        gamma = None
        gamma_init = None
        logger.warning("sigma_init is None — skipping gamma computation")
    else:
        sigma_init_arr = state.sigma_init if isinstance(state.sigma_init, np.ndarray) else np.array(state.sigma_init)
        sigma_ratio = sigma_final / (sigma_init_arr + 1e-8)
        gamma = np.mean(np.abs(np.log(sigma_ratio)))

        # γ_init: zero-point fluctuation indicator
        # σ_ref = 1 for normalized types (LN, BN, GN); None for 'none'
        if norm_type in ('batchnorm', 'layernorm', 'groupnorm'):
            sigma_ref = 1.0
            gamma_init = float(np.mean(np.abs(np.log(sigma_init_arr / sigma_ref))))
        else:
            gamma_init = None

    # ===== BUILD RESULTS DICT =====
    # β (D-scaling exponent) is not computed here because single-run training
    # doesn't span multiple D values. The beta field is set at the grid level
    # (run_experiment_grid) after collecting results across all D values for
    # a given norm_type and lr, where fit_beta can be applied properly.
    # Individual runs store gamma which is used in the Phase 0.2 β vs ln(γ) analysis.
    results = {
        'config': {
            'D': D,
            'norm_type': norm_type,
            'lr': lr,
            'seed': seed,
            'num_epochs': num_epochs,
        },
        'sigma_init': state.sigma_init.tolist() if state.sigma_init is not None else None,
        'sigma_final': sigma_final.tolist(),
        'gamma': float(gamma) if gamma is not None else None,
        'gamma_init': gamma_init,
        'lambda_max_init': float(state.lambda_max_init) if state.lambda_max_init is not None else None,
        'lambda_max_final': float(lambda_final),
        'lambda_max_stationary': float(state.stationary_lambda_max) if state.stationary_lambda_max is not None else None,
        'loss_history': [float(l) for l in state.loss_history],
        'lambda_max_history': [float(l) for l in state.lambda_max_history],
        'stationary': state.is_stationary,
        'stationary_epoch': state.stationary_epoch,
        'is_converged': len(state.loss_history) >= 2 and state.loss_history[0] > 0 and state.loss_history[-1] < state.loss_history[0] * 0.7,
        'completed': True,
    }

    # Save final result
    if output_dir:
        save_run_result(results, output_dir)

        # Save final checkpoint at end of training (only if checkpoint_manager was initialized)
        if checkpoint_manager is not None:
            checkpoint_manager.save(
                run_id=run_id,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=num_epochs - 1,
                loss_history=state.loss_history,
                lambda_max_history=state.lambda_max_history,
                training_state={
                    'is_stationary': state.is_stationary,
                    'stationary_epoch': state.stationary_epoch,
                    'sigma_init': state.sigma_init.tolist() if state.sigma_init is not None else None,
                    'lambda_max_init': float(state.lambda_max_init) if state.lambda_max_init is not None else None,
                    'stationary_lambda_max': float(state.stationary_lambda_max) if state.stationary_lambda_max is not None else None,
                },
                additional_data={
                    'config': {
                        'D': D,
                        'norm_type': norm_type,
                        'lr': lr,
                        'seed': seed,
                        'num_epochs': num_epochs,
                    }
                },
            )

    return results


def _reconstruct_result_from_checkpoint(checkpoint: Dict) -> Dict:
    """Reconstruct a result dict from a checkpoint."""
    training_state = checkpoint.get('training_state', {})
    stored_config = checkpoint.get('config', {})

    loss_history = checkpoint.get('loss_history', [])
    lambda_max_history = checkpoint.get('lambda_max_history', [])

    # NOTE: gamma cannot be faithfully reconstructed from a checkpoint
    # because sigma_final is not stored. Previously this block returned
    # loss_init/loss_final as a proxy, but that is a loss ratio, not gamma
    # (mean |ln(sigma_final / sigma_init)|). Using it as gamma is misleading.
    # We return gamma: None and flag gamma_reconstructed: True so downstream
    # code knows this is NOT a real measurement.
    gamma = None

    norm_type = stored_config.get('norm_type')
    sigma_init = training_state.get('sigma_init')

    # Reconstruct gamma_init from sigma_init for normalized types
    # gamma_init = mean(|ln(sigma_init / 1.0)|) for batchnorm/layernorm/groupnorm
    gamma_init = None
    if norm_type in ('batchnorm', 'layernorm', 'groupnorm') and sigma_init is not None:
        sigma_init_arr = np.array(sigma_init)
        gamma_init = float(np.mean(np.abs(np.log(sigma_init_arr / 1.0))))

    return {
        'config': {
            'D': stored_config.get('D'),
            'norm_type': norm_type,
            'lr': stored_config.get('lr'),
            'seed': stored_config.get('seed'),
            'num_epochs': stored_config.get('num_epochs', checkpoint['epoch'] + 1),
        },
        'sigma_init': sigma_init,
        'sigma_final': None,  # Not available
        'gamma': gamma,
        'gamma_reconstructed': True,  # Not a real measurement; gamma is None
        'gamma_init': gamma_init,
        'lambda_max_init': training_state.get('lambda_max_init'),
        'lambda_max_final': None,
        'lambda_max_stationary': training_state.get('stationary_lambda_max'),
        'loss_history': loss_history,
        'lambda_max_history': lambda_max_history,
        'stationary': training_state.get('is_stationary', False),
        'stationary_epoch': training_state.get('stationary_epoch', -1),
        'is_converged': len(loss_history) >= 2 and loss_history[0] > 0 and loss_history[-1] < loss_history[0] * 0.7,
        'completed': True,
        'reconstructed_from_checkpoint': True,
    }


def train_with_d_scaling(
    D_values: List[int],
    norm_type: str,
    lr: float,
    seed: int,
    dataloader_train: DataLoader,
    dataloader_eval: DataLoader,
    device: torch.device,
    num_epochs: int = 200,
    verbose: bool = True,
    output_dir: Optional[str] = None,
    graceful_exit: Optional = None,
) -> Tuple[Dict, Optional[float]]:
    """
    Train models across multiple D values to measure D-scaling for β extraction.
    """
    results_all = {}
    final_losses = []

    for D in D_values:
        if verbose:
            print(f"\n>>> Training D={D}, norm={norm_type}, lr={lr}, seed={seed}")

        result = train_and_measure(
            D=D,
            norm_type=norm_type,
            lr=lr,
            seed=seed,
            dataloader_train=dataloader_train,
            dataloader_eval=dataloader_eval,
            device=device,
            num_epochs=num_epochs,
            verbose=verbose,
            output_dir=output_dir,
            graceful_exit=graceful_exit,
        )

        results_all[D] = result
        final_losses.append(result['loss_history'][-1])

    # Fit D-scaling law to extract β
    losses_array = np.array(final_losses)
    D_array = np.array(D_values)

    beta, alpha, r_squared = fit_beta(losses_array, D_array)

    return results_all, beta


# =============================================================================
# BATCH RUNNING UTILITIES
# =============================================================================

def run_experiment_grid(
    experiment_config: Dict,
    dataloader_train: DataLoader,
    dataloader_eval: DataLoader,
    device: torch.device,
    output_dir: str,
    verbose: bool = True,
    graceful_exit: Optional = None,
) -> List[Dict]:
    """
    Run a complete experiment grid with Kaggle support.

    Features:
    - Per-run results saved immediately after each run
    - Checkpoints saved every N epochs
    - Graceful SIGTERM handling
    - Resume support: skips completed runs

    Args:
        experiment_config: Dict with grid specification
        dataloader_train: Training data loader
        dataloader_eval: Evaluation data loader
        device: torch device
        output_dir: Where to save results and checkpoints
        verbose: Print progress
        graceful_exit: GracefulExit object for SIGTERM handling

    Returns:
        List of result dicts for each completed run
    """
    norm_types = experiment_config['norm_types']
    D_values = experiment_config['D_values']
    lr_values = experiment_config['lr_values']
    seeds = experiment_config['seeds']
    epochs = experiment_config.get('epochs', 200)
    checkpoint_every = experiment_config.get('checkpoint_every', 20)

    all_results = []

    total_runs = len(norm_types) * len(D_values) * len(lr_values) * len(seeds)
    run_idx = 0

    # Load any existing results for resume
    existing_results = load_run_results(output_dir)
    existing_run_ids = set()
    for r in existing_results:
        if 'config' in r:
            cfg = r['config']
            rid = get_run_id(cfg.get('D'), cfg.get('norm_type'), cfg.get('lr'), cfg.get('seed'))
            existing_run_ids.add(rid)

    if existing_run_ids:
        print(f"[Resume] Found {len(existing_run_ids)} existing run results, will skip completed runs")

    for norm_type in norm_types:
        for D in D_values:
            for lr in lr_values:
                for seed in seeds:
                    run_idx += 1

                    # Check for graceful exit
                    if graceful_exit is not None and graceful_exit.exit_requested:
                        print(f"\n[Graceful] Exit requested, stopping after current batch")
                        break

                    # Generate run ID and check if completed
                    run_id = get_run_id(D, norm_type, lr, seed)

                    if run_id in existing_run_ids:
                        if verbose:
                            print(f"\n[Skip {run_idx}/{total_runs}] {run_id} - already completed")
                        continue

                    if verbose:
                        print(f"\n{'='*60}")
                        print(f"Run {run_idx}/{total_runs}")
                        print(f"norm={norm_type}, D={D}, lr={lr}, seed={seed}")
                        print('='*60)

                    # Train
                    result = train_and_measure(
                        D=D,
                        norm_type=norm_type,
                        lr=lr,
                        seed=seed,
                        dataloader_train=dataloader_train,
                        dataloader_eval=dataloader_eval,
                        device=device,
                        num_epochs=epochs,
                        verbose=verbose,
                        output_dir=output_dir,
                        checkpoint_every=checkpoint_every,
                    )

                    all_results.append(result)
                    existing_run_ids.add(run_id)  # Mark as complete

                    # Result already saved by train_and_measure
                    # Save incremental progress
                    save_path = os.path.join(output_dir, 'results_progress.json')
                    save_results({'runs': all_results}, save_path)

                if graceful_exit is not None and graceful_exit.exit_requested:
                    break
            if graceful_exit is not None and graceful_exit.exit_requested:
                break
        if graceful_exit is not None and graceful_exit.exit_requested:
            break


# =====================================================================
# FIT β (D-scaling exponent) FOR EACH (norm_type, lr) GROUP
# After all runs complete, group by (norm_type, lr) and fit D-scaling
# law: L = α·D^(-β) + E_floor to extract β.
# =====================================================================
    if all_results:
        _fit_beta_for_grid_results(all_results, output_dir)

    return all_results


def _fit_beta_for_grid_results(all_results: List[Dict], output_dir: str) -> None:
    """
    Fit D-scaling law to extract β for each (norm_type, lr) group.

    For each (norm_type, lr) group:
    1. Group runs by D value
    2. Average final loss across seeds for each D
    3. Fit L = α·D^(-β) + E_floor to get β
    4. Store beta in each run's result dict and re-save the result file
    """
    from collections import defaultdict

    # Group runs by (norm_type, lr)
    groups = defaultdict(list)
    for r in all_results:
        cfg = r.get('config', {})
        key = (cfg.get('norm_type'), cfg.get('lr'))
        groups[key].append(r)

    for (norm_type, lr), runs in groups.items():
        # Group by D and average loss across seeds
        d_to_losses = defaultdict(list)
        for r in runs:
            D = r.get('config', {}).get('D')
            loss_history = r.get('loss_history', [])
            loss_final = loss_history[-1] if loss_history else None
            if D is not None and loss_final is not None:
                d_to_losses[D].append(loss_final)

        if len(d_to_losses) < 2:
            # Need at least 2 D values to fit
            continue

        D_values = np.array(sorted(d_to_losses.keys()), dtype=float)
        mean_losses = np.array([np.mean(d_to_losses[D]) for D in D_values])

        # Fit D-scaling law
        beta, alpha, r_squared = fit_beta(mean_losses, D_values)

        if beta is not None:
            # Add beta to each run in this group
            for r in runs:
                r['beta'] = beta
                r['beta_r_squared'] = r_squared
                # Re-save the updated result
                if 'config' in r:
                    cfg = r['config']
                    run_id = get_run_id(cfg.get('D'), cfg.get('norm_type'),
                                       cfg.get('lr'), cfg.get('seed'))
                    save_run_result(r, output_dir)

            print(f"  [D-scaling fit] norm={norm_type}, lr={lr}: β={beta:.4f}, R2={r_squared:.4f}")


def load_partial_results(output_dir: str) -> List[Dict]:
    """Load partial results from a previous run (legacy compatibility)."""
    # Try new format first
    results = load_run_results(output_dir)
    if results:
        return results

    # Fallback to legacy format
    save_path = os.path.join(output_dir, 'results_partial.json')
    if os.path.exists(save_path):
        with open(save_path, 'r') as f:
            data = json.load(f)
        return data.get('runs', [])
    return []
