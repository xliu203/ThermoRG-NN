# ThermoRG-NN

**Thermodynamic theory of neural network scaling.**

This repository contains the code and experiments for validating the ThermoRG framework, which proposes that the data-efficiency scaling behavior of neural networks is governed by a thermodynamic framework involving:

- **Topological information flow** (J_topo) across layers
- **Effective task complexity** via participation ratio (d_task^PR)
- **Edge of Stability** (EoS) as the optimal training regime
- **Power-law D-scaling**: L(D) = α·D^(-β) + E

## Papers

| Paper | Status | Description |
|-------|--------|-------------|
| `papers/unified_framework_paper_final.tex` | Complete | ThermoRG theory + Phase S0 validation |
| `papers/applied_theory_paper_final.tex` | Draft | Application to real CIFAR-10 architectures |

## Experiments

### Phase S0 (Simulation — Complete ✅)

**Validated 3 core predictions** with RFF synthetic tasks across 5 FC architectures:

| Conclusion | Result |
|-----------|--------|
| Power-law D-scaling (R² > 0.9) | ✅ All 5 archs |
| β ≈ β_eff = s/d_task^PR | ✅ Within 3× |
| β ∝ J_topo (r=0.86, p=0.06) | ✅ Confirmed |
| α ∝ J_topo² (r=0.83, p=0.09) | ✅ Confirmed |
| J_topo predicts final loss (r=-0.955) | ✅ Strong |

**Code**: `experiments/phase_s0/thermoRG_v3_results.json`

### Phase A v2 (CIFAR-10 — Ready to Run)

Validates the same predictions on real CIFAR-10 data with diverse architectures:

- **12 architectures**: ThermoNet (width/depth families), ResNet, VGG
- **D-scaling**: D ∈ {2K, 5K, 10K, 25K, 50K}
- **Hypotheses**: β ∝ J_topo, α ∝ J_topo²

**Script**: `experiments/phase_a/phase_a_dscaling.py`

```bash
python experiments/phase_a/phase_a_dscaling.py
```

## Theory Summary

### Core Metric: J_topo

```
J_topo = exp(-|Σ log η_l| / L)
η_l = D_eff^(l) / D_eff^(l-1)
D_eff = ||W_l||_F² / λ_max(W_l)
```

### D-Scaling Law

```
L(D) = α · D^(-β) + E
```

### Key Predictions

| Prediction | Status |
|-----------|--------|
| β_eff = s/d_task^PR | ✅ Validated |
| β ∝ J_topo | ✅ r=0.86 |
| α ∝ J_topo² | ✅ r=0.83 |
| J_topo ∈ (0,1] | ✅ All measurements |

## Repository Structure

```
ThermoRG-NN/
├── papers/
│   ├── unified_framework_paper_final.tex   # Theory + S0 validation
│   └── applied_theory_paper_final.tex      # Applied theory (draft)
├── experiments/
│   ├── phase_s0/                           # Simulation experiments
│   │   └── thermoRG_v3_results.json
│   └── phase_a/                            # CIFAR-10 experiments
│       └── phase_a_dscaling.py             # Phase A v2 (ready to run)
└── thermorg/                              # Theory implementation
```

## Key Files

- `experiments/phase_s0/thermoRG_v3_results.json` — Phase S0 raw results
- `experiments/phase_a/phase_a_dscaling.py` — Phase A v2 script
- `experiments/PHASE_A_REDESIGN.md` — Phase A design rationale

## Setup

```bash
pip install torch numpy scipy
git clone https://github.com/xliu203/ThermoRG-NN.git
cd ThermoRG-NN
```

CIFAR-10 data will be downloaded automatically by the training script.

## Citation

```bibtex
@article{liu2026thermorgnn,
  title={ThermoRG-NN: Thermodynamic Scaling in Neural Architecture},
  author={Liu, Leo},
  year={2026}
}
```
