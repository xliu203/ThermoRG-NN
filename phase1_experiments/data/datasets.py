"""
CIFAR-10 data loading utilities.
"""

import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split


def get_transforms(augment: bool = True):
    """Get train and test transforms for CIFAR-10."""
    normalize = transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2023, 0.1994, 0.2010)
    )
    
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    
    return train_transform, test_transform


def load_cifar10(
    batch_size: int = 128,
    data_dir: str = './data',
    augment: bool = True,
    num_workers: int = 2,
    download: bool = True
) -> tuple:
    """
    Load CIFAR-10 dataset.
    
    Returns:
        Tuple of (trainloader, testloader)
    """
    train_transform, test_transform = get_transforms(augment)
    
    os.makedirs(data_dir, exist_ok=True)

    try:
        trainset = torchvision.datasets.CIFAR10(
            root=data_dir,
            train=True,
            download=download,
            transform=train_transform
        )
    except Exception as e:
        raise RuntimeError(f"Failed to download CIFAR-10 dataset: {e}. Please check network connection or manually place data in {data_dir}")

    try:
        testset = torchvision.datasets.CIFAR10(
            root=data_dir,
            train=False,
            download=download,
            transform=test_transform
        )
    except Exception as e:
        raise RuntimeError(f"Failed to download CIFAR-10 dataset: {e}. Please check network connection or manually place data in {data_dir}")

    if len(trainset) == 0:
        raise ValueError(f"CIFAR-10 trainset is empty. Data may be corrupted in {data_dir}")
    if len(testset) == 0:
        raise ValueError(f"CIFAR-10 testset is empty. Data may be corrupted in {data_dir}")

    trainloader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    testloader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return trainloader, testloader


def create_split_loaders(
    dataset,
    train_ratio: float = 0.9,
    batch_size: int = 128,
    num_workers: int = 2
) -> tuple:
    """
    Split a dataset into train/val loaders.
    
    Args:
        dataset: PyTorch dataset
        train_ratio: Fraction for training
        batch_size: Batch size
        num_workers: Number of workers
        
    Returns:
        Tuple of (trainloader, valloader)
    """
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = total_size - train_size

    assert train_size > 0 and val_size > 0, f"Invalid split: train_size={train_size}, val_size={val_size}"

    train_subset, val_subset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    trainloader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    valloader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return trainloader, valloader
