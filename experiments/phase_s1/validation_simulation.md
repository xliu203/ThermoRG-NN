# Phase S1: Cooling Theory Validation Simulation

**Date**: 2026-04-04
**Goal**: Validate the thermodynamic normalization theory before integrating into THEORY.md

## Theory to Validate

### Extended Scaling Law
```
β(J, γ) = β₀(J) · φ(γ_total)
E(J, γ) = E₀(J) · exp(-κ · γ_total)
γ_total = γ_norm + γ_skip
φ(γ) = (γ_c / (γ_c + |γ|)) · exp(-|γ|/γ_c)
```

### Key Predictions to Test
1. **Normalization cooling**: γ_norm(BatchNorm) < γ_norm(LayerNorm) ≈ γ_norm(None) ≈ large
   - BN/LN **reduce** variance fluctuation → act as cooling mechanisms
2. **Skip cooling**: γ_skip < γ_none (skip connections stabilize variance)
3. **Additivity**: γ_total = γ_norm + γ_skip (for combined configs)
4. **Variance relation**: γ = (1/L)Σ|ln(Var_final/Var_init)|

## Experimental Design

### Architecture (Fixed)
- Simple ConvNet: [3, 64, 128, 256, 128] channels
- 3 conv blocks + adaptive pool + FC
- CIFAR-10 (or synthetic task for speed)

### Variables (controlled)
| Factor | Options | Values |
|--------|---------|--------|
| **Normalization** | None / LayerNorm / BatchNorm | γ_norm: 0 / 0 / 0.22·γ_c |
| **Skip connections** | No / Yes | γ_skip: 0 / 0.0125·γ_c |
| **Activation** | GELU / ReLU | (optional, for interaction) |
| **D (model size)** | D ∈ {500, 1000, 2000, 5000} | For scaling law fit |

### Configuration Matrix

| Config | Norm | Skip | Expected γ_total |
|--------|------|------|-----------------|
| S1_A | None | No | Large (baseline heating) |
| S1_B | None | Yes | < S1_A (skip stabilizes) |
| S1_C | LayerNorm | No | < S1_A (LN cools) |
| S1_D | LayerNorm | Yes | < S1_C (combined) |
| S1_E | BatchNorm | No | < S1_A (BN cools) |
| S1_F | BatchNorm | Yes | Smallest (max cooling) |

### Expected Outcomes

| Config | Predicted γ | Predicted β | Predicted E |
|--------|-------------|-------------|-------------|
| S1_A (None, No) | Highest | Lowest | Highest |
| S1_B (None, Yes) | < S1_A | > S1_A | < S1_A |
| S1_C (LN, No) | < S1_A, ≈ S1_E | > S1_A, ≈ S1_E | < S1_A, ≈ S1_E |
| S1_D (LN, Yes) | < S1_C | > S1_C | < S1_C |
| S1_E (BN, No) | < S1_A | > S1_A | < S1_A |
| S1_F (BN, Yes) | Lowest | Highest | Lowest |

## Measurements

### 1. J_topo (at initialization)
- Compute from effective weight matrices
- W_eff = W_main + W_skip (for skip configs)
- Should be same across norm variants (norm doesn't change W_eff)

### 2. Activation Variance Ratio
- Hook into forward pass
- Record Var(x) before and after normalization
- Compute γ_norm = -k · ln(Var_out/Var_in)

### 3. Scaling Law Fit
For each config × D:
- Train to convergence (300 epochs or plateau)
- Record final loss L(D)
- Fit L(D) = α·D^(-β) + E
- Extract β and E

### 4. Verify Cooling Factor
- Compute φ_measured = β_measured / β_base (where β_base from config with γ ≈ 0)
- Compare with φ_predicted from γ

## Implementation

### Network Class Extension
```python
class ValidationNet(nn.Module):
    def __init__(self, norm_type='none', use_skip=False, activation='relu'):
        # norm_type: 'none' | 'layernorm' | 'batchnorm'
        # use_skip: bool
        # activation: 'relu' | 'gelu'
```

### Training
- Dataset: CIFAR-10 (or synthetic regression for speed)
- Optimizer: SGD, lr=0.01, momentum=0.9
- Epochs: 300 (or until plateau)
- Seeds: 2 (for averaging)

### D Variation
- Control model size via width multiplier: w ∈ {0.5, 1.0, 1.5, 2.0}
- Or via hidden dim: D ∈ {500, 1000, 2000, 5000}

## Estimated Compute

- 6 configurations × 4 D values × 2 seeds = 48 runs
- Per run: ~300 epochs on CIFAR-10
- Estimated: ~20-40 GPU hours on Kaggle (T4)

## Success Criteria

1. **γ_norm ordering**: γ_norm(BatchNorm) < γ_norm(LayerNorm) ≈ γ_norm(None) (BN/LN reduce γ)
2. **γ_skip detection**: Config with skip has lower γ than without skip
3. **β ordering**: β(BN) > β(LN) > β(None) (cooler → larger β)
4. **E_floor ordering**: E_floor(BN) < E_floor(LN) < E_floor(None) (cooler → lower floor)
5. **Form of φ(γ)**: β vs 1/γ follows the predicted functional form

## Files

- `experiments/phase_s1/phase_s1_validation.py`: Main simulation script
- `experiments/phase_s1/cooling_network.py`: Network class with norm/skip options
- `experiments/phase_s1/results/`: Output directory

## Next Steps

1. Implement ValidationNet class with norm/skip options
2. Implement variance tracking hooks
3. Write training loop with D variation
4. Run on Kaggle or local GPU
5. Analyze results vs predictions
6. If validated: integrate into THEORY.md as Section 8 (Normalization Extension)
