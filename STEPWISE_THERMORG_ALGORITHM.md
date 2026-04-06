# Stepwise ThermoRG Algorithm
## Adaptive Architecture Composition via Signal-Based Diagnosis

**Date**: 2026-04-05
**Based on**: THERMO RG THEORY.md v7
**Status**: Design Complete

---

## 1. Core Idea

Replace combinatorial search over discrete architectures with **adaptive composition**: start with a task-appropriate minimal baseline, monitor training-dynamics signals, and add modules only when signals indicate a specific problem.

**Key principle**: The algorithm **diagnoses and fixes** rather than **searches all combinations**.

---

## 2. Task-Adaptive Step 0

**Problem with fixed baseline**: Different tasks have different optimal starting points.

**Solution**: Initialize based on task type:

| Task Type | Minimal Baseline |
|-----------|----------------|
| Image (CIFAR-10, ImageNet) | Conv-GELU-Conv-Pool-Linear |
| Language (text) | Small Transformer (2 layers, 4 heads) |
| Sequential (time series) | LSTM (hidden=64) |
| Regression (tabular) | MLP (64-64) |

**Why this matters**: A Conv-ReLU-Conv baseline is inappropriate for language tasks. The baseline must match the data modality.

---

## 3. Normal Training Phases (ThermoRG Perspective)

Training loss has three phases:

### Phase 1: Rapid Descent
- **Characteristics**: Fast L1 loss decrease, low γ, high β
- **Meaning**: Easy patterns learned quickly
- **Signal**: L1 improving >10% per epoch

### Phase 2: Slow Descent
- **Characteristics**: Moderate L1 loss decrease, γ increasing, β decreasing
- **Meaning**: Complex manifold structure being learned
- **Signal**: L1 improving <2% per epoch

### Phase 3: Plateau
Two types must be distinguished:

| Type | Train | Val | Meaning | Action |
|------|-------|-----|---------|--------|
| **Honest learning** | ↓ slowly | ↓ slowly | Learning difficult manifold | Continue training |
| **True overfitting** | ↓ | ↑ | memorizing | Add regularization |
| **Difficulty plateau** | ↓ slowly | plateau | May need more capacity | Consider wider/deeper |

**Critical insight**: A plateau can mean "learning difficult manifold structure" NOT overfitting.

---

## 4. Signal-to-Module Mapping

### Key Signals

| Signal | Measurement | Interpretation |
|--------|------------|---------------|
| **γ (variance instability)** | `abs(log(σ_final/σ_init))` averaged over layers | High γ → activations variance-unstable → needs cooling |
| **L1 loss (5-epoch)** | Loss after 5 epochs | Slow descent → poor gradient flow or bad initialization |
| **J_topo** | `exp(-|Σ log η_l| / L)` at initialization | Predicts E_floor, but NOT monotonically bad |
| **Gradient magnitude** | Mean ‖∇W‖ across layers | Vanishing gradients → need skip connections |
| **Val/Train gap** | val_loss - train_loss | Overfitting vs honest learning |

### Signal → Action Rules

| Signals Observed | Diagnosis | Action |
|----------------|----------|--------|
| High γ + high L1 | Variance unstable + poor optimization | Add BN/LN (cooling) |
| High γ + low L1 + small grad | Good variance but gradient flow poor | Add skip connection |
| Low J_topo + low β | Stable but slow learning | May need more capacity |
| High J_topo + good γ + good L1 | Efficient learning (ResNet-18 style) | Good combo - keep going |
| Plateau, train↑ val↓ | TRUE overfitting | Add regularization (dropout, wd) |
| Plateau, both ↓ slowly | Honest learning | Continue training |
| Plateau, train↓ val plateau | Difficulty plateau | Consider wider/deeper |

---

## 5. J_topo: Guide, Not Minimizer

**Critical insight from ResNet-18**: High J_topo (4.16) can perform well with skip + BN.

J_topo is NOT monotonically bad. There are TWO paths to good E_floor:

1. **Low J_topo + moderate β**: Stable, slow learning
2. **High J_topo + skip + BN**: Unstable but efficient, fast learning

**Target J_topo range** (not single value):
```
J_target = 0.5 + 0.4 · tanh(s / d_manifold)
```
- Adjust only when |J_topo - J_target| > 0.2
- Within range: keep architecture, don't force adjustment

---

## 6. Algorithm Steps

### Step 0: Task-Adaptive Baseline
```
1. Identify task type (image/language/sequential/regression)
2. Initialize minimal baseline for that task
3. Estimate data properties: d_manifold, s (sample size)
4. Compute initial J_topo
```

### Step 1: Diagnose Optimization Stability
```
1. Train for 5 epochs (L1 fidelity)
2. Compute γ, L1 loss, gradient norms
3. If γ > THRESHOLD_γ → add BN/LN
   If grad_norm < THRESHOLD_grad → add skip
   Otherwise → optimization stable
```

### Step 2: Adjust Capacity via J_topo
```
1. Compute J_topo at initialization
2. If J_topo < J_target - 0.2 → increase width/depth
   If J_topo > J_target + 0.2 → decrease width/depth
   Otherwise → J_topo in optimal range
```

### Step 3: Monitor Plateau Behavior
```
1. Track train_loss and val_loss trajectories
2. If val_loss rising while train↓ → true overfitting → add regularization
   If both ↓ slowly → honest learning → continue
   If train↓ val plateau → difficulty → consider more capacity
```

### Step 4: Fine-Tune Hyperparameters
```
1. Adjust learning rate based on T_eff vs T_opt
2. Adjust batch size to control gradient noise
3. Set cooling scheduler based on observed γ
```

### Step 5: Convergence Check
```
Stop when:
- Architecture unchanged for 3 iterations
- Validation loss plateau (honest) with <1% improvement
- J_topo stable within target range
```

---

## 7. Module Library

| Module | Effect | When to Add |
|--------|--------|-------------|
| BatchNorm (BN) | Cools variance instability, reduces γ | γ > THRESHOLD_γ |
| LayerNorm (LN) | Alternative cooling, stable for small batches | γ > THRESHOLD_γ, small batch |
| Skip connection | Improves gradient flow, raises J_topo | grad_norm < THRESHOLD_grad |
| GELU/SiLU | Smoother activation, better gradient | L1 slow improvement |
| Dropout | Regularization against overfitting | val↑ train↓ |
| Weight decay | Explicit L2 regularization | val↑ train↓ |

---

## 8. Why Stepwise vs Discrete Search?

| Aspect | Discrete HBO | Stepwise Active |
|--------|-------------|----------------|
| Search space | 96 discrete combos | ~10 modules, adaptive |
| Sample efficiency | Each combo tested separately | Signals reused across steps |
| Interpretability | Black-box ranking | "Added BN because γ was high" |
| Computational cost | 96 × multi-fidelity | ≤10 steps × L1 |
| Task adaptation | Universal baseline | Task-specific baseline |

---

## 9. Validation via Simulation

The algorithm can be validated by simulating training dynamics:

1. **Simulate Phase 1→2→3**: Given architecture, simulate L1 trajectory
2. **Detect plateau type**: Simulate train↑ val↓ vs both↓ behaviors
3. **Test action effects**: Simulate adding BN/skip and observe γ changes

See `experiments/phase_b/stepwise_simulation.py` for implementation.

---

## 10. References

- ThermoRG Theory: `theory/THEORY.md` (v7)
- HBO Design: `experiments/phase_b/HIERARCHICAL_BAYESIAN_OPTIMIZATION.md`
- Universal Algorithm: `UNIVERSAL_THERMORG_ALGORITHM.md`
