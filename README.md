# ThermoRG-NN

**Thermodynamic Theory of Neural Architecture Scaling**

A framework for analyzing and optimizing neural network architectures using thermodynamic principles and manifold geometry.

## Project Structure

```
ThermoRG-NN/
├── thermorg/              # Core library (NEW in v0.2)
│   ├── __init__.py        # Clean API exports
│   ├── j_topo.py          # J_topo computation with stride correction
│   ├── scaling.py         # D-scaling law fitting
│   ├── cooling.py          # Cooling factor φ(γ) computation
│   └── utils.py            # Common utilities
│
├── theory/
│   └── THEORY.md          # Current theory framework (v9)
│
├── experiments/
│   ├── phase_s0/         # Phase S0: Simulation validation (RFF networks)
│   ├── phase_s1/          # Phase S1: Small-scale validation
│   ├── phase_a/           # Phase A: ThermoNet architecture scaling
│   ├── phase_b/           # Phase B: Joint selection (J_topo + capacity)
│   └── phase_4/           # Phase 4: Stride/pooling RG correction
│
├── src/                   # Legacy source packages
│   ├── thermorg/          # Original thermorg package (pre-v0.2)
│   ├── thermorg_hbo/      # HBO variant
│   └── thermorg_suhbo/    # SU-HBO variant
│
├── papers/                # Paper LaTeX sources
└── examples/             # Example scripts
```

## Quick Start

### Using the Core Library

```python
import torch.nn as nn
from thermorg import compute_J_topo, fit_scaling_law, phi_from_delta

# Build a model
model = nn.Sequential(
    nn.Conv2d(3, 64, 3, padding=1),
    nn.Conv2d(64, 128, 3, stride=2, padding=1),  # stride-2
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten(),
    nn.Linear(128, 10)
)

# Compute J_topo
J, etas = compute_J_topo(model)
print(f"J_topo = {J:.4f}")  # Information flow quality

# Compute cooling factor for stride-2 layers
phi = phi_from_delta(n_s=1)  # 1 stride-2 layer
print(f"φ = {phi:.2f}")  # ≈ 0.87

# Fit scaling law
from thermorg import scaling_law
D = [100, 500, 1000, 5000]
L = [0.8, 0.4, 0.25, 0.15]
alpha, beta, eps, rmse = fit_scaling_law(D, L)
```

### Available Modules

| Module | Functions |
|--------|-----------|
| `j_topo` | `compute_J_topo`, `compute_D_eff`, `compute_D_eff_total`, `count_parameters` |
| `scaling` | `scaling_law`, `fit_scaling_law`, `predict_loss`, `compute_optimal_temperature` |
| `cooling` | `phi_from_delta`, `get_cooling_factor`, `cooling_factor_*` schedules |
| `utils` | `estimate_d_manifold`, `compute_capacity_bound`, `count_stride2_layers` |

## Theory Overview

### D-Scaling Law

$$L(D) = \alpha \cdot D^{-\beta} + E$$

- $D$: Training set size
- $\alpha$: Pre-asymptotic coefficient (complexity penalty)
- $\beta$: Scaling exponent (learning efficiency)
- $E$: Asymptotic floor (irreducible error)

### J_topo Metric

$$J_{\mathrm{topo}} = \exp\!\Bigl(-\frac{1}{L}\sum_{l=1}^{L}|\log \eta_l|\Bigr)$$

- $\eta_l = D_{\mathrm{eff}}^{(l)} / D_{\mathrm{eff}}^{(l-1)}$
- $J \to 1$: Stable information flow
- $J \to 0$: Bottlenecks or expansion issues

### Stride Correction (v8+)

For stride-2 layers, apply correction factor:

$$\eta_{\text{corrected}} = \eta \cdot \frac{C_{\text{out}}}{C_{\text{in}} \cdot s^2}$$

where $s=2$ is the stride. This accounts for spatial-channel compression.

## Phase 4 Experiment

The Phase 4 experiment (`experiments/phase_4/phase_4a1_stride2_validation.ipynb`) validates the stride correction theory:

| Architecture | n_s | β_pred |
|--------------|-----|--------|
| ThermoNet-L3 | 0 | 0.86 |
| ThermoNet-S2-1 | 1 | 0.75 |
| ThermoNet-S2-2 | 2 | 0.65 |
| ThermoNet-MP-2 | 2 | 0.65 |
| ResNet-18 | 3 | 0.57 |

## Installation

```bash
pip install -e .
```

Or add to `sys.path`:

```python
import sys
sys.path.insert(0, '/path/to/ThermoRG-NN')
from thermorg import compute_J_topo
```

## References

- Theory: [`theory/THEORY.md`](theory/THEORY.md) (v9)
- Paper: [`papers/unified_framework_paper_final.tex`](papers/unified_framework_paper_final.tex)

## Changelog

### v0.2.0 (2026-04-06)
- **New**: `thermorg/` core library at root level
- **New**: J_topo with stride correction
- **New**: D-scaling law fitting
- **New**: Cooling factor computation
- **New**: `experiments/phase_4/` directory
- **Cleaned**: `theory/` directory (removed duplicate)
