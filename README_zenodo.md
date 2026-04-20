# ThermoRG-NN: Thermodynamic Theory of Neural Architecture Scaling

A clean 3-module framework for zero-cost architecture scoring and loss prediction based on the ThermoRG theory.

## Installation

```bash
pip install -r requirements.txt
```

**Note:** This code was developed with PyTorch 2.0+. Using exactly `torch==2.0.0` is recommended for reproducibility:

```bash
pip install torch==2.0.0 numpy==1.24.0 scipy==1.10.0
```

## The 3 Modules

### 1. `thermorg/topology_calculator.py` - J_topo Computation

Computes the topological participation ratio J_topo from initialized network weights using Power Iteration (PI-20).

**Key features:**
- Zero-cost (runs on randomly initialized networks)
- PI-20: 20 iterations of power iteration for D_eff estimation
- ~23× faster than full SVD with ~2.5% error
- Fixed random seed (42) for reproducibility

**Important note on random seeds:**
Different random seeds give slightly different λ_max estimates (due to the random starting vector in power iteration), but the **J_topo ranking is stable** across seeds. This is because J_topo measures relative layer uniformity, not absolute D_eff values.

```python
from thermorg.topology_calculator import compute_J_topo

# Compute J_topo for any initialized model
model = ThermoNet(width=64, depth=5, norm_type='bn')
J_topo, eta_list = compute_J_topo(model)
print(f"J_topo = {J_topo:.4f}")
```

### 2. `thermorg/calibration/thermo_calibrator.py` - Parameter Calibration

Calibrates the thermodynamic equation of state from observed training data.

**Input:** 8 calibration architectures with early training losses  
**Output:** Parameters (α_type, β, C, B) for the scaling law

```python
from thermorg.calibration.thermo_calibrator import ThermoCalibrator, get_default_calibration_data

calibrator = ThermoCalibrator(verbose=True)
calibration_data = get_default_calibration_data()
result = calibrator.calibrate(calibration_data)
print(result)
```

### 3. `thermorg/analytical_predictor.py` - Loss Prediction

Pure mathematical prediction of training loss. **No training code, no backward pass.**

**Contains:**
- D-scaling law: L(D) = α·D^(-β) + E_floor
- E_floor decomposition: C/D + B·J_topo^ν  
- β(γ) cooling law: β = 0.425·ln(γ/γ_c) + 0.893

```python
from thermorg.analytical_predictor import AnalyticalPredictor

predictor = AnalyticalPredictor()
loss = predictor.predict(width=64, depth=5, norm_type='bn', J_topo=0.75)
print(f"Predicted loss: {loss:.4f}")
```

## Minimal Example

```python
import torch
import torch.nn as nn

from thermorg.topology_calculator import compute_J_topo
from thermorg.analytical_predictor import AnalyticalPredictor


# 1. Create a model
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


# 2. Compute J_topo (zero-cost, runs on initialized weights)
model = ThermoNet(width=64, depth=5, norm_type='bn')
J_topo, eta_list = compute_J_topo(model)
print(f"J_topo = {J_topo:.4f}")


# 3. Predict training loss
predictor = AnalyticalPredictor()
loss = predictor.predict(width=64, depth=5, norm_type='bn', J_topo=J_topo)
print(f"Predicted loss: {loss:.4f}")
```

## File Structure

```
ThermoRG-NN/
├── thermorg/
│   ├── __init__.py
│   ├── topology_calculator.py   # Module A: J_topo computation
│   ├── analytical_predictor.py   # Module C: Loss prediction
│   ├── calibration/
│   │   ├── __init__.py
│   │   └── thermo_calibrator.py  # Module B: Parameter calibration
│   └── synflow_scoring.py        # SynFlow (NASWOT) zero-cost scoring
├── requirements.txt
└── README_zenodo.md
```

## Mathematical Framework

The ThermoRG theory predicts training loss as:

```
L(D) = α · D^(-β) + C/D + B·J_topo^ν
```

where:
- D is effective degrees of freedom (proportional to width × depth)
- α depends on normalization type (α_bn ≈ 0.45, α_none ≈ 0.68)
- β is the depth exponent (cooling law: β = 0.425·ln(γ/γ_c) + 0.893)
- J_topo is the topology quality metric (0 to 1, higher is better)

## References

- ThermoRG Theory Framework v5-v8
- Power Iteration for D_eff: ~23× faster than SVD, ~2.5% error
- SynFlow: "Pruning Neural Networks without Data" (arXiv:1906.04326)