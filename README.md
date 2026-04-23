# ThermoRG-NN

[![DOI](https://img.shields.io/badge/DOI-10.5281/zenodo.19708932-blue)](https://doi.org/10.5281/zenodo.19708932)

**Thermodynamic Theory of Neural Architecture Scaling and Its Application to Calibration-Guided Architecture Search**

Xiaonan Liu — Arizona State University

> We present ThermoRG, a phenomenological thermodynamic framework for neural architecture scaling. We validate ThermoRG on plain convolutional networks with different normalization layers, demonstrating that D-scaling laws, J_topo quality metrics, and the β(γ) cooling theory all hold across controlled building blocks. Finally, we present ThermoRG-AL, a calibration-guided architecture search framework. After a one-time calibration phase (~1 GPU-hour), ThermoRG-AL uses a phenomenologically derived equation of state to pre-screen architectures semi-analytically, dramatically reducing search cost. ThermoRG-AL achieves comparable or better results than random search (best loss 0.377 vs 0.427) on CIFAR-10, independently cross-validated by SynFlow.

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

## Project Structure

```
ThermoRG-NN/
├── thermorg/              # Core library
│   ├── topology_calculator.py  # J_topo computation (zero-cost)
│   ├── analytical_predictor.py # Loss prediction (pure math)
│   ├── calibration/           # ThermoCalibrator
│   ├── scaling.py             # D-scaling law fitting
│   ├── cooling.py             # Cooling factor φ(γ)
│   └── j_topo.py              # J_topo with stride correction
│
├── legacy/
│   └── THEORY.md              # Full theory framework
│
├── experiments/
│   ├── notebooks/             # Experiment notebooks
│   └── results/
│       └── EXPERIMENTS_SUMMARY.md  # Phase results summary
│
├── papers/                    # LaTeX sources
└── examples/                  # Example scripts
```

---

## Installation

```bash
pip install -r requirements.txt
```

**Note:** This code was developed with PyTorch 2.0+:

```bash
pip install torch==2.0.0 numpy==1.24.0 scipy==1.10.0
```

---

## The 3 Modules

### 1. `thermorg/topology_calculator.py` — J_topo Computation

Computes J_topo from initialized network weights using Power Iteration (PI-20).

- Zero-cost (no training needed)
- ~23× faster than full SVD, ~2.5% error
- Fixed random seed (42) for reproducibility

```python
from thermorg.topology_calculator import compute_J_topo

model = ThermoNet(width=64, depth=5, norm_type='bn')
J_topo, eta_list = compute_J_topo(model)
print(f"J_topo = {J_topo:.4f}")
```

### 2. `thermorg/calibration/thermo_calibrator.py` — Parameter Calibration

Calibrates thermodynamic EOS from observed training data.

```python
from thermorg.calibration.thermo_calibrator import ThermoCalibrator, get_default_calibration_data

calibrator = ThermoCalibrator(verbose=True)
result = calibrator.calibrate(get_default_calibration_data())
print(result)
```

### 3. `thermorg/analytical_predictor.py` — Loss Prediction

Pure mathematical prediction. **No training, no backward pass.**

```python
from thermorg.analytical_predictor import AnalyticalPredictor

predictor = AnalyticalPredictor()
loss = predictor.predict(width=64, depth=5, norm_type='bn', J_topo=0.75)
print(f"Predicted loss: {loss:.4f}")
```

---

## Minimal Example

```python
import torch
import torch.nn as nn

from thermorg.topology_calculator import compute_J_topo
from thermorg.analytical_predictor import AnalyticalPredictor


class ThermoNet(nn.Module):
    def __init__(self, width, depth, norm_type='bn'):
        super().__init__()
        self.conv1 = nn.Conv2d(3, width, 3, padding=1)
        self.norm1 = nn.BatchNorm2d(width) if norm_type == 'bn' else nn.Identity()
        self.conv2 = nn.Conv2d(width, width, 3, padding=1)
        self.norm2 = nn.BatchNorm2d(width) if norm_type == 'bn' else nn.Identity()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(width, 10)

    def forward(self, x):
        x = self.norm1(self.conv1(x))
        x = self.norm2(self.conv2(x))
        return self.fc(self.pool(x).flatten(1))


# 1. Compute J_topo (zero-cost, on initialized weights)
model = ThermoNet(width=64, depth=5, norm_type='bn')
J_topo, eta_list = compute_J_topo(model)
print(f"J_topo = {J_topo:.4f}")

# 2. Predict training loss
predictor = AnalyticalPredictor()
loss = predictor.predict(width=64, depth=5, norm_type='bn', J_topo=J_topo)
print(f"Predicted loss: {loss:.4f}")
```

---

## Phase Status

| Phase | Status | Key Result |
|-------|--------|------------|
| S0 (RFF Simulation) | ✅ Complete | α phase transition, universal J_topo |
| A (ThermoNet Training) | ✅ Complete | 87 runs, 9 architectures |
| 4A1 (Stride-2 RG) | ✅ Complete | φ task-dependent |
| S1 (Cooling β(γ)) | ✅ Complete | β(γ) validated for BN, LN, None |
| B1 (LN Cooling) | ✅ Complete | β_LN=0.219, γ≈0.41, sub-critical |
| B3 (HBO_revised) | ✅ Complete | Width-first + J_topo HIGH: won 0.703 vs 0.781 |

**Full results:** [`experiments/results/EXPERIMENTS_SUMMARY.md`](experiments/results/EXPERIMENTS_SUMMARY.md)

**Notebooks:**
- [`experiments/notebooks/phase_a_v2.ipynb`](experiments/notebooks/phase_a_v2.ipynb) — Phase A: D-scaling law
- [`experiments/notebooks/phase_s1_tpu.ipynb`](experiments/notebooks/phase_s1_tpu.ipynb) — Phase S1: Cooling theory
- [`experiments/notebooks/phase_4a1_tpu.ipynb`](experiments/notebooks/phase_4a1_tpu.ipynb) — Phase 4A1: Stride-2 RG
- [`experiments/notebooks/phase_b_hbo_revised.ipynb`](experiments/notebooks/phase_b_hbo_revised.ipynb) — Phase B3: HBO vs Random

---

## References

- Theory: [`theory/THEORY.md`](theory/THEORY.md)
- Papers: [`papers/`](papers/)
