"""Training script for CIFAR-10 Lift Test.

This module implements two training phases:
- Phase A: Train all 15 architectures for 30 epochs
- Phase B: Train selected 5 architectures for 150 epochs (top 2, bottom 2, G4-1)

Key features:
- Tracks train/val/test loss and accuracy per epoch
- Detects grokking timing (when test loss drops >5% in one epoch)
- Logs results to CSV for later analysis
"""

import os
import json
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

from .architectures import get_model, list_models, count_parameters, estimate_model_flops
from .constants import (
    CIFAR10_MEAN, CIFAR10_STD, CIFAR10_NUM_CLASSES,
    PHASE_A_EPOCHS, PHASE_B_EPOCHS, PHASE_B_ARCHITECTURES,
    DEFAULT_BATCH_SIZE, DEFAULT_LR, DEFAULT_WEIGHT_DECAY
)


@dataclass
class EpochMetrics:
    """Metrics for a single epoch."""
    epoch: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float
    test_loss: float
    test_acc: float
    lr: float
    time_seconds: float


@dataclass
class TrainingResult:
    """Complete training result for an architecture."""
    arch_name: str
    phase: str  # 'A' or 'B'
    epochs: int
    final_train_loss: float
    final_train_acc: float
    final_val_loss: float
    final_val_acc: float
    final_test_loss: float
    final_test_acc: float
    best_test_acc: float
    grokking_epoch: Optional[int] = None
    grokking_improvement: float = 0.0
    total_params: int = 0
    total_flops: int = 0
    epoch_metrics: List[EpochMetrics] = field(default_factory=list)
    training_time: float = 0.0


def get_cifar10_loaders(batch_size: int = DEFAULT_BATCH_SIZE) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Get CIFAR-10 train/val/test dataloaders.
    
    Uses 50,000 training images with 90/10 train/val split.
    
    Args:
        batch_size: Batch size for dataloaders
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # CIFAR-10 preprocessing
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    
    # Load full training set
    full_trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=train_transform
    )
    
    # Split into train/val (90/10)
    total_size = len(full_trainset)
    val_size = int(0.1 * total_size)
    train_size = total_size - val_size
    
    train_subset, val_subset = torch.utils.data.random_split(
        full_trainset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create loaders
    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )
    
    # Test set
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=test_transform
    )
    test_loader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
    """Train for one epoch.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        
    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    return total_loss / total, 100.0 * correct / total


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """Evaluate model on a dataset.
    
    Args:
        model: Model to evaluate
        loader: Data loader
        criterion: Loss function
        device: Device to evaluate on
        
    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return total_loss / total, 100.0 * correct / total


def detect_grokking(epoch_metrics: List[EpochMetrics], threshold: float = 0.05) -> Tuple[Optional[int], float]:
    """Detect grokking epoch - when test loss drops >threshold in one epoch.
    
    Args:
        epoch_metrics: List of epoch metrics
        threshold: Minimum improvement threshold (default 5%)
        
    Returns:
        Tuple of (grokking_epoch, improvement) or (None, 0) if no grokking detected
    """
    if len(epoch_metrics) < 2:
        return None, 0.0
    
    for i in range(1, len(epoch_metrics)):
        prev_loss = epoch_metrics[i - 1].test_loss
        curr_loss = epoch_metrics[i].test_loss
        
        if prev_loss > 0:
            improvement = (prev_loss - curr_loss) / prev_loss
            if improvement > threshold:
                return epoch_metrics[i].epoch, improvement
    
    return None, 0.0


def train_architecture(
    arch_name: str,
    epochs: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    lr: float = DEFAULT_LR,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    save_path: Optional[str] = None
) -> TrainingResult:
    """Train a single architecture.
    
    Args:
        arch_name: Architecture name
        epochs: Number of epochs to train
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        device: Device to train on
        lr: Learning rate
        weight_decay: Weight decay
        save_path: Optional path to save best model
        
    Returns:
        TrainingResult with all metrics
    """
    # Create model
    model = get_model(arch_name, num_classes=CIFAR10_NUM_CLASSES)
    model = model.to(device)
    
    # Count params and FLOPs
    total_params = count_parameters(model)
    total_flops = estimate_model_flops(model)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Training
    best_test_acc = 0.0
    best_model_state = None
    epoch_metrics = []
    training_start = time.time()
    
    for epoch in tqdm(range(1, epochs + 1), desc=f"{arch_name}"):
        epoch_start = time.time()
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Evaluate
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        # Learning rate
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()
        
        # Metrics
        metrics = EpochMetrics(
            epoch=epoch,
            train_loss=train_loss,
            train_acc=train_acc,
            val_loss=val_loss,
            val_acc=val_acc,
            test_loss=test_loss,
            test_acc=test_acc,
            lr=current_lr,
            time_seconds=time.time() - epoch_start
        )
        epoch_metrics.append(metrics)
        
        # Track best
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_model_state = model.state_dict().copy()
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(best_model_state, save_path)
    
    # Final evaluation with best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    final_train_loss, final_train_acc = evaluate(model, train_loader, criterion, device)
    final_val_loss, final_val_acc = evaluate(model, val_loader, criterion, device)
    final_test_loss, final_test_acc = evaluate(model, test_loader, criterion, device)
    
    # Detect grokking
    grokking_epoch, grokking_improvement = detect_grokking(epoch_metrics)
    
    training_time = time.time() - training_start
    
    return TrainingResult(
        arch_name=arch_name,
        phase='A',  # Will be updated by caller
        epochs=epochs,
        final_train_loss=final_train_loss,
        final_train_acc=final_train_acc,
        final_val_loss=final_val_loss,
        final_val_acc=final_val_acc,
        final_test_loss=final_test_loss,
        final_test_acc=final_test_acc,
        best_test_acc=best_test_acc,
        grokking_epoch=grokking_epoch,
        grokking_improvement=grokking_improvement,
        total_params=total_params,
        total_flops=total_flops,
        epoch_metrics=epoch_metrics,
        training_time=training_time
    )


def save_results(results: List[TrainingResult], output_dir: str):
    """Save training results to disk.
    
    Args:
        results: List of TrainingResult objects
        output_dir: Directory to save results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for result in results:
        # Save per-architecture results
        arch_dir = output_dir / result.arch_name
        arch_dir.mkdir(parents=True, exist_ok=True)
        
        # Main result as JSON
        result_dict = {
            "arch_name": result.arch_name,
            "phase": result.phase,
            "epochs": result.epochs,
            "final_train_loss": result.final_train_loss,
            "final_train_acc": result.final_train_acc,
            "final_val_loss": result.final_val_loss,
            "final_val_acc": result.final_val_acc,
            "final_test_loss": result.final_test_loss,
            "final_test_acc": result.final_test_acc,
            "best_test_acc": result.best_test_acc,
            "grokking_epoch": result.grokking_epoch,
            "grokking_improvement": result.grokking_improvement,
            "total_params": result.total_params,
            "total_flops": result.total_flops,
            "training_time": result.training_time,
        }
        
        with open(arch_dir / "result.json", "w") as f:
            json.dump(result_dict, f, indent=2)
        
        # Epoch metrics as CSV
        import csv
        with open(arch_dir / "epoch_metrics.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "epoch", "train_loss", "train_acc", "val_loss", "val_acc",
                "test_loss", "test_acc", "lr", "time_seconds"
            ])
            writer.writeheader()
            for m in result.epoch_metrics:
                writer.writerow({
                    "epoch": m.epoch,
                    "train_loss": m.train_loss,
                    "train_acc": m.train_acc,
                    "val_loss": m.val_loss,
                    "val_acc": m.val_acc,
                    "test_loss": m.test_loss,
                    "test_acc": m.test_acc,
                    "lr": m.lr,
                    "time_seconds": m.time_seconds
                })


def train_phase_a(
    output_dir: str = "./experiments/lift_test/results",
    device: torch.device = None,
    architectures: List[str] = None
) -> List[TrainingResult]:
    """Phase A: Train all 15 architectures for 30 epochs.
    
    Args:
        output_dir: Directory to save results
        device: Device to train on (auto-detected if None)
        architectures: List of architectures to train (all if None)
        
    Returns:
        List of TrainingResult objects
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Phase A: Training all architectures for {PHASE_A_EPOCHS} epochs")
    print(f"Device: {device}")
    
    # Get architectures to train
    if architectures is None:
        architectures = list_models()
    
    # Get dataloaders
    train_loader, val_loader, test_loader = get_cifar10_loaders()
    
    results = []
    
    for arch_name in architectures:
        print(f"\n{'='*60}")
        print(f"Training {arch_name}")
        print(f"{'='*60}")
        
        arch_dir = Path(output_dir) / arch_name
        phase_a_csv = arch_dir / "phase_a_metrics.csv"
        
        # ──断点续传：跳过已完成的架构──
        if phase_a_csv.exists():
            print(f"✅ {arch_name} 数据已存在，跳过训练...")
            continue
        
        save_path = f"{output_dir}/{arch_name}/best_model.pt"
        
        result = train_architecture(
            arch_name=arch_name,
            epochs=PHASE_A_EPOCHS,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            save_path=save_path
        )
        result.phase = 'A'
        results.append(result)
        
        # ──实时落盘：每个架构训练完立刻保存 CSV──
        arch_dir.mkdir(parents=True, exist_ok=True)
        with open(phase_a_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "epoch", "train_loss", "train_acc", "val_loss", "val_acc",
                "test_loss", "test_acc", "lr", "time_seconds"
            ])
            writer.writeheader()
            for m in result.epoch_metrics:
                writer.writerow({
                    "epoch": m.epoch,
                    "train_loss": m.train_loss,
                    "train_acc": m.train_acc,
                    "val_loss": m.val_loss,
                    "val_acc": m.val_acc,
                    "test_loss": m.test_loss,
                    "test_acc": m.test_acc,
                    "lr": m.lr,
                    "time_seconds": m.time_seconds
                })
        print(f"💾 已落盘: {phase_a_csv}")
        
        print(f"\n{arch_name} Results:")
        print(f"  Final Test Accuracy: {result.final_test_acc:.2f}%")
        print(f"  Best Test Accuracy: {result.best_test_acc:.2f}%")
        print(f"  Grokking Epoch: {result.grokking_epoch}")
        print(f"  Parameters: {result.total_params:,}")
        print(f"  Training Time: {result.training_time:.1f}s")
    
    # Save all results
    save_results(results, output_dir)
    
    return results


def select_phase_b_architectures(phase_a_results: List[TrainingResult]) -> List[str]:
    """Select architectures for Phase B based on Phase A results.
    
    Selects:
    - Top 2 architectures by test accuracy
    - Bottom 2 architectures by test accuracy
    - G4-1 (ResNet-18-CIFAR)
    
    Args:
        phase_a_results: Results from Phase A
        
    Returns:
        List of architecture names for Phase B
    """
    # Sort by test accuracy
    sorted_results = sorted(phase_a_results, key=lambda r: r.final_test_acc, reverse=True)
    
    top_2 = [r.arch_name for r in sorted_results[:2]]
    bottom_2 = [r.arch_name for r in sorted_results[-2:]]
    
    # Ensure G4-1 is included
    g4_1 = "ResNet-18-CIFAR"
    if g4_1 not in top_2 and g4_1 not in bottom_2:
        phase_b_archs = top_2 + bottom_2 + [g4_1]
    else:
        phase_b_archs = top_2 + bottom_2
    
    return phase_b_archs


def train_phase_b(
    phase_a_results: List[TrainingResult] = None,
    output_dir: str = "./experiments/lift_test/results",
    device: torch.device = None,
    architectures: List[str] = None
) -> List[TrainingResult]:
    """Phase B: Train selected 5 architectures for 150 epochs.
    
    Args:
        phase_a_results: Results from Phase A (used to select architectures if not provided)
        output_dir: Directory to save results
        device: Device to train on (auto-detected if None)
        architectures: List of architectures to train (auto-selected if None)
        
    Returns:
        List of TrainingResult objects
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Select architectures
    if architectures is None:
        if phase_a_results is not None:
            architectures = select_phase_b_architectures(phase_a_results)
        else:
            architectures = PHASE_B_ARCHITECTURES
    
    print(f"Phase B: Training {len(architectures)} architectures for {PHASE_B_EPOCHS} epochs")
    print(f"Architectures: {architectures}")
    print(f"Device: {device}")
    
    # Get dataloaders
    train_loader, val_loader, test_loader = get_cifar10_loaders()
    
    results = []
    
    for arch_name in architectures:
        print(f"\n{'='*60}")
        print(f"Training {arch_name}")
        print(f"{'='*60}")
        
        arch_dir = Path(output_dir) / arch_name
        phase_b_csv = arch_dir / "phase_b_metrics.csv"
        
        # ──断点续传：跳过已完成的架构──
        if phase_b_csv.exists():
            print(f"✅ {arch_name} Phase B 数据已存在，跳过训练...")
            continue
        
        save_path = f"{output_dir}/{arch_name}/best_model_phaseB.pt"
        
        result = train_architecture(
            arch_name=arch_name,
            epochs=PHASE_B_EPOCHS,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            save_path=save_path
        )
        result.phase = 'B'
        results.append(result)
        
        # ──实时落盘：每个架构训练完立刻保存 CSV──
        arch_dir.mkdir(parents=True, exist_ok=True)
        with open(phase_b_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "epoch", "train_loss", "train_acc", "val_loss", "val_acc",
                "test_loss", "test_acc", "lr", "time_seconds"
            ])
            writer.writeheader()
            for m in result.epoch_metrics:
                writer.writerow({
                    "epoch": m.epoch,
                    "train_loss": m.train_loss,
                    "train_acc": m.train_acc,
                    "val_loss": m.val_loss,
                    "val_acc": m.val_acc,
                    "test_loss": m.test_loss,
                    "test_acc": m.test_acc,
                    "lr": m.lr,
                    "time_seconds": m.time_seconds
                })
        print(f"💾 已落盘: {phase_b_csv}")
        
        print(f"\n{arch_name} Phase B Results:")
        print(f"  Final Test Accuracy: {result.final_test_acc:.2f}%")
        print(f"  Best Test Accuracy: {result.best_test_acc:.2f}%")
        print(f"  Grokking Epoch: {result.grokking_epoch}")
        print(f"  Training Time: {result.training_time:.1f}s")
    
    # Save all results
    save_results(results, output_dir)
    
    return results


def main():
    """Main training entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="CIFAR-10 Lift Test Training")
    parser.add_argument("--phase", type=str, choices=["A", "B", "both"], default="both",
                        help="Training phase to run")
    parser.add_argument("--output_dir", type=str, default="./experiments/lift_test/results",
                        help="Output directory for results")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (cuda/cpu, auto-detected if None)")
    parser.add_argument("--architectures", type=str, nargs="+", default=None,
                        help="Specific architectures to train")
    
    args = parser.parse_args()
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.phase in ["A", "both"]:
        phase_a_results = train_phase_a(
            output_dir=args.output_dir,
            device=device,
            architectures=args.architectures
        )
        
        if args.phase == "both":
            phase_b_results = train_phase_b(
                phase_a_results=phase_a_results,
                output_dir=args.output_dir,
                device=device
            )
    else:
        phase_b_results = train_phase_b(
            output_dir=args.output_dir,
            device=device,
            architectures=args.architectures
        )
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
