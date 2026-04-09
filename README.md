# ThermoRG-NN

**Thermodynamic Theory of Neural Architecture Scaling**

A framework for analyzing and optimizing neural network architectures using thermodynamic principles and manifold geometry.

## Project Structure

```
ThermoRG-NN/
├── thermorg/              # Core library
│   ├── __init__.py        # Clean API exports
│   ├── j_topo.py          # J_topo computation with stride correction
│   ├── scaling.py         # D-scaling law fitting
│   ├── cooling.py         # Cooling factor φ(γ) computation
│   └── utils.py           # Common utilities
│
├── theory/
│   └── THEORY.md          # Current theory framework (v9+)
│
├── experiments/
│   ├── phase_s0/          # Phase S0: RFF simulation
│   ├── phase_s1/          # Phase S1: Cooling theory (BN, LN, None)
│   ├── phase_a/           # Phase A: ThermoNet training (87 runs)
│   ├── phase_b/            # Phase B: Architecture selection (HBO vs Random)
│   └── phase_4/           # Phase 4: Stride/pooling RG correction
│
├── src/                   # Legacy source packages
├── papers/                # Paper LaTeX sources
└── examples/              # Example scripts
```

## Quick Start

### Compute J_topo

```python
import torch.nn as nn
from thermorg import compute_J_topo

model = YourModel()
J, etas = compute_J_topo(model)
print(f"J_topo = {J:.4f}")  # 0 = bottleneck, 1 = uniform info flow
```

### D-Scaling Law

```python
from thermorg import scaling_law, fit_scaling_law

D = [100, 500, 1000, 5000]
L = [0.8, 0.4, 0.25, 0.15]
alpha, beta, E_floor, rmse = fit_scaling_law(D, L)
print(f"L(D) = {alpha:.3f} * D^(-{beta:.3f}) + {E_floor:.3f}")
```

### Cooling Theory

```python
from thermorg import beta_gamma

gamma = 2.29  # BatchNorm variance fluctuation
beta = beta_gamma(gamma)
print(f"β = {beta:.3f}")  # ≈ 0.950
```

---

## Theory Overview

### Core D-Scaling Law

$$L(D) = \alpha \cdot D^{-\beta} + E_{\mathrm{floor}}$$

- $D$: Network width (dominant scaling variable)
- $\alpha$: Pre-asymptotic coefficient
- $\beta$: Scaling exponent (learning efficiency per unit width)
- $E_{\mathrm{floor}}$: Asymptotic error floor

### J_topo Metric

$$J_{\mathrm{topo}} = \exp\!\Bigl(-\frac{1}{L}\sum_{l=1}^{L}|\log \eta_l|\Bigr)$$

- $\eta_l = D_{\mathrm{eff}}^{(l)} / D_{\mathrm{eff}}^{(l-1)}$ (per-layer expansion ratio)
- $D_{\mathrm{eff}} = \|W\|_F^2 / \lambda_{\max}^2$ (effective dimensionality)
- $J \to 1$: Uniform information flow (all $\eta_l \approx 1$)
- $J \to 0$: Bottlenecks or pathological expansions
- **Zero-cost**: Computed from initialization weights alone

### J_topo Width Dependence: α_GELU ≈ 0.45

J_topo decreases with width D due to the GELU nonlinearity:

$$J_{\mathrm{topo}}(D) \propto D^{-\alpha_{\mathrm{GELU}}/L}, \quad \alpha_{\mathrm{GELU}} \approx 0.45$$

- Verified: depth=3 slope=-0.150, depth=5 slope=-0.087 ✅
- Mechanism: GELU saturates for large |x|, making λ_max grow as D^0.52 (not √D)
- BN/LN do NOT affect this scaling (they are excluded from J_topo computation)

### H1: β ∝ J_topo

Higher J_topo → higher β → slower width scaling decay.
- Within families: r ≈ 0.97 ✅
- Cross-phase: r ≈ 0.68 (ResNet-18 outlier explained by stride-2 correction)

### Cooling Theory: β(γ)

$$\beta(\gamma) = 0.425 \cdot \ln(\gamma / \gamma_c) + 0.893, \quad \gamma_c \approx 2.0$$

Validated across γ ∈ [0.41, 3.39] (LN → BN → None):

| Configuration | γ | β (fitted) | β (theory) | Regime |
|-------------|---|------------|------------|--------|
| None | 3.39 | 1.117 | 1.117 ✅ | super-critical |
| BatchNorm | 2.29 | 0.950 | 0.950 ✅ | super-critical |
| LayerNorm | ~0.41 | 0.219 | 0.219 ✅ | **sub-critical** |

### E_floor Decomposition

$$E_{\mathrm{floor}} = E_{\mathrm{task}} + \frac{C}{D} + B \cdot J_{\mathrm{topo}}^{\nu}$$

Two independent channels:
1. **Capacity**: Width D → E_floor (dominant, r ≈ -0.83)
2. **Topology**: J_topo → optimization difficulty → E_floor (secondary, r ≈ -0.79 within width groups)

### Stride-2 as RG Blocking

For stride-2 downsampling, apply correction:

$$\eta_{\mathrm{corrected}} = \eta_l \cdot \frac{C_{\mathrm{out}}}{C_{\mathrm{in}} \cdot s^2}$$

- CIFAR-10: φ ≈ 1.0 (irrelevant)
- ImageNet: φ = 0.87 (suppressing)

---

## Phase Status

| Phase | Status | Key Result |
|-------|--------|------------|
| S0 (RFF Simulation) | ✅ Complete | α phase transition, universal J_topo |
| A (ThermoNet Training) | ✅ Complete | 87 runs, 9 architectures |
| 4A1 (Stride-2 RG) | ✅ Complete | φ task-dependent |
| S1 (Cooling β(γ)) | ✅ Complete | β(γ) validated for BN, None |
| B1 (LN Cooling) | ✅ Complete | β_LN=0.219, R²=0.9997, γ≈0.41 |
| B2 (HBO Round 1) | ✅ Complete | **Negative result**: Random best=0.386 vs HBO best=0.605 |
| B3 (HBO_revised) | 🔄 Pending | Width-first + J_topo HIGH (Leo: next week) |

### Phase B: Architecture Selection

**Round 1 (Negative Result):**
- HBO selected HIGH J_topo globally → picked narrow-deep nets (24/6) → capacity-limited
- Random selected wide nets (96/6) → won decisively

**Confounding Analysis (n=10):**
| Relationship | Simple r | Partial r (width fixed) |
|-------------|---------|------------------------|
| J_topo → loss | +0.588 | **-0.794** (p=0.006) |
| Width → loss | -0.829 | -0.891 |

**Simpson's Paradox Resolved:** Wide networks have LOW J_topo AND LOW loss (capacity). Within fixed width, HIGH J_topo → LOW loss (optimization efficiency).

**Round 2 (HBO_revised):**
- Notebook: `experiments/phase_b/notebooks/phase_b_hbo_revised.ipynb`
- Design: Width-first (W≥48) → top-30 by J_topo HIGH → L1 → L2
- Expected: Should beat Random

---

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

---

## References

- Theory: [`theory/THEORY.md`](theory/THEORY.md)
- Paper: [`papers/`](papers/)

## Changelog

### v0.3.0 (2026-04-09)
- **New**: LayerNorm cooling validation (β_LN=0.219, γ≈0.41, sub-critical regime)
- **New**: J_topo(D) scaling: α_GELU=0.45 from GELU nonlinearity
- **New**: Unified β(J_topo, γ) framework via condition number
- **New**: Phase B2 negative result + confounding analysis
- **New**: HBO_revised notebook (width-first + J_topo HIGH)
- **Fixed**: LayerNorm J_topo bug in compute_J_topo (prev_D_eff chain reset)
- **Updated**: Cooling theory β(γ) across [0.41, 3.39]

### v0.2.0 (2026-04-06)
- **New**: `thermorg/` core library at root level
- **New**: J_topo with stride correction
- **New**: D-scaling law fitting
- **New**: Cooling factor computation
