"""CIFAR-10 Lift Test Experiment.

This module implements the CIFAR-10 Lift Test for evaluating 15 neural network
architectures across 4 groups:
- Group 1: Thermogeometric Optimal (4 architectures)
- Group 2: Topology Destroyer (4 architectures with bottlenecks)
- Group 3: Thermal Boiling Furnace (4 architectures - ReLU ablation)
- Group 4: Traditional Baselines (3 architectures)

Example usage:
    from experiments.lift_test import train, evaluate
    
    # Phase A: Train all 15 architectures for 30 epochs
    results = train.train_phase_a()
    
    # Phase B: Train selected architectures for 150 epochs
    phase_b_results = train.train_phase_b(results)
    
    # Evaluate and print summary
    metrics = evaluate.load_results("./experiments/lift_test/results")
    evaluate.print_summary(metrics)
"""

from .constants import (
    ALL_SPECS,
    ARCH_MAP,
    GROUP1_SPECS,
    GROUP2_SPECS,
    GROUP3_SPECS,
    GROUP4_SPECS,
    PHASE_A_EPOCHS,
    PHASE_B_EPOCHS,
    PHASE_B_ARCHITECTURES,
    CIFAR10_MEAN,
    CIFAR10_STD,
    CIFAR10_NUM_CLASSES,
    CIFAR10_INPUT_SIZE,
    DEFAULT_BATCH_SIZE,
    DEFAULT_LR,
    DEFAULT_WEIGHT_DECAY,
    get_group_name,
    get_architectures_by_group,
)

from .architectures import (
    get_model,
    list_models,
    get_model_info,
    count_parameters,
    estimate_model_flops,
    MODEL_REGISTRY,
    # All architecture classes
    ThermoNet3,
    ThermoNet5,
    ThermoNet7,
    ThermoNet9,
    ThermoBot3,
    ThermoBot5,
    ThermoBot7,
    ThermoBot9,
    ReLUFurnace3,
    ReLUFurnace5,
    ReLUFurnace7,
    ReLUFurnace9,
    ResNet18CIFAR,
    VGG11CIFAR,
    DenseNet40CIFAR,
)

from .train import (
    train_phase_a,
    train_phase_b,
    train_architecture,
    get_cifar10_loaders,
    TrainingResult,
    EpochMetrics,
)

from .evaluate import (
    load_results,
    compute_metrics,
    correlation_analysis,
    generate_comparison_table,
    rank_architectures,
    print_summary,
    ArchitectureMetrics,
)

__all__ = [
    # Constants
    "ALL_SPECS",
    "ARCH_MAP",
    "GROUP1_SPECS",
    "GROUP2_SPECS",
    "GROUP3_SPECS",
    "GROUP4_SPECS",
    "PHASE_A_EPOCHS",
    "PHASE_B_EPOCHS",
    "PHASE_B_ARCHITECTURES",
    "CIFAR10_MEAN",
    "CIFAR10_STD",
    "CIFAR10_NUM_CLASSES",
    "CIFAR10_INPUT_SIZE",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_LR",
    "DEFAULT_WEIGHT_DECAY",
    "get_group_name",
    "get_architectures_by_group",
    # Architectures
    "get_model",
    "list_models",
    "get_model_info",
    "count_parameters",
    "estimate_model_flops",
    "MODEL_REGISTRY",
    "ThermoNet3",
    "ThermoNet5",
    "ThermoNet7",
    "ThermoNet9",
    "ThermoBot3",
    "ThermoBot5",
    "ThermoBot7",
    "ThermoBot9",
    "ReLUFurnace3",
    "ReLUFurnace5",
    "ReLUFurnace7",
    "ReLUFurnace9",
    "ResNet18CIFAR",
    "VGG11CIFAR",
    "DenseNet40CIFAR",
    # Training
    "train_phase_a",
    "train_phase_b",
    "train_architecture",
    "get_cifar10_loaders",
    "TrainingResult",
    "EpochMetrics",
    # Evaluation
    "load_results",
    "compute_metrics",
    "correlation_analysis",
    "generate_comparison_table",
    "rank_architectures",
    "print_summary",
    "ArchitectureMetrics",
]
