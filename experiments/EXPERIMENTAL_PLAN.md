# ThermoRG Experimental Plan: Three-Phase Validation

## Philosophy
**"Phenomenological theory with strong predictive power"**
- J_topo, T_eff, ψ, φ are computable quantities
- Testable predictions: J_topo ordering, β ∝ J_topo, EoS tracking
- NOT claiming first-principles derivation

---

## Phase 0: Simulation (Theory Validation)

### Purpose
Validate the mathematical structure before any real experiments. Show that the theory's predictions are internally consistent and physically plausible.

### What to Simulate

**S0.1: J_topo Computation from Synthetic Architectures**
```python
# Create networks with varying J_topo (via eta_l products)
# Verify J_topo = exp(-|Σlog η_l|/L) ∈ (0,1]
networks = [
    ("Ideal", [1.0]*10),           # J_topo = exp(0) = 1
    ("Good", [0.9]*10),            # J_topo = exp(-|log 0.9|*10/10)
    ("Poor", [0.5, 0.6, 0.7]*3),   # J_topo << 1
]
```

**S0.2: Bounded Flow Equation**
```python
# Verify d_eff^realized = d_task + (d_manifold - d_task)*exp(-∫γ dt)
# When ∫γ dt → ∞: d_eff → d_task
# When ∫γ dt → 0: d_eff → d_manifold
```

**S0.3: ψ(T_eff) Shape**
```python
# Verify ψ(T_c) = 1, ψ(0) = 0, ψ(∞) = 0
# Verify peak at EoS condition (T_eff/T_c = 1)
```

**S0.4: T_eff/T_c = sharpness/2**
```python
# For various (η, λ_max) pairs:
# Verify T_eff/T_c = η*λ_max/2 matches sharpness/2
```

**Deliverable**: Synthetic validation notebook showing theory is self-consistent.

---

## Phase A: Architecture Validation (Kaggle)

### Purpose
Demonstrate that J_topo is measurable, architecture-dependent, and correlates with scaling exponents across diverse architectures.

### Current Status
- 12/15 architectures complete (VGG-11, DenseNet-40 still running)
- Code: `experiments/phase_a/phase_a_analysis.py`

### What Phase A Proves

**A1. J_topo is Computable**
- Measure η_l = D_eff^(l)/D_eff^(l-1) for each layer of trained network
- Compute J_topo = exp(-|Σlog η_l|/L)
- **Result type**: Scatter plot of J_topo vs architecture quality

**A2. J_topo Ordering (Main Evidence)**
- Expected ordering: ThermoNet > ResNet > VGG
- If true: strong evidence that J_topo captures architectural efficiency
- **Result type**: Bar chart of J_topo by architecture

**A3. β vs J_topo Linear Relationship**
- Fit β_hat from D-scaling for each architecture
- Plot β_hat vs J_topo
- **Result type**: Linear regression with R²

**A4. d_eff from Hessian**
- Compute d_eff^Hess = count(eigenvalues > ε·λ_max)
- Compare to d_task estimate
- Verify d_eff^realized ≈ d_task/J_topo

### Phase A Pipeline
```
train.py → CSV (acc, params, final loss)
         → phase_a_analysis.py
                 → J_topo (from trained nets)
                 → β_hat (from D-scaling fit)
                 → d_eff^Hess (from Hessian)
                 → plots + JSON results
```

### Deliverables
- `phase_a_results.csv`: architecture | J_topo | β_hat | d_eff^Hess | R²
- 3 publication figures:
  1. J_topo ordering bar chart
  2. β vs J_topo linear fit
  3. d_eff comparison

---

## Phase B: Dynamic Training Validation

### Purpose
Track J_topo, T_eff, ψ, sharpness during training to verify:
1. J_topo evolves during training
2. Sharpness → 2 (EoS) during training
3. ψ peaks at EoS point

### B1: Training Trajectory Tracking
```python
# During training, every N steps:
metrics = {
    'step': step,
    'J_topo': compute_J_topo(model),      # from eta_l at each layer
    'T_eff': compute_T_eff(grad_noise),     # = η*B/σ²  
    'sharpness': η * λ_max(H),             # track during training
    'loss': loss.item(),
    'lambda_max': compute_lambda_max(H),    # from power iteration
}
```

### B2: J_topo Evolution
- Does J_topo increase during training? (network compresses better)
- Does final J_topo correlate with final accuracy?
- **Result type**: J_topo vs training step curves

### B3: Edge of Stability Verification
- Track sharpness = η·λ_max(H) during training
- Verify it approaches 2 and stays there
- **Result type**: Sharpness vs step (should plateau at 2)

### B4: ψ(T_eff) Peak at EoS
- At each step, compute T_eff/T_c
- Plot ψ(T_eff) vs training step
- Verify ψ peaks when sharpness ≈ 2
- **Result type**: ψ(T_eff) trajectory

### Phase B Pipeline
```python
phase_b_main.py
    → Train ThermoNet + ResNet-18 on CIFAR-10
    → Every 100 steps: record J_topo, T_eff, sharpness, loss
    → Save trajectory to JSON
    → Plot training dynamics
```

### Deliverables
- `phase_b_trajectories.json`: time series of all metrics
- 2 publication figures:
  1. J_topo evolution during training
  2. Sharpness → 2 (EoS) + ψ peak

---

## Summary: What Each Phase Proves

| Phase | Theory Part | Evidence Type | Strength |
|-------|------------|---------------|---------|
| **S0** | All formulas | Internal consistency | N/A (validation) |
| **A1** | J_topo computable | Direct measurement | ✅ Strong |
| **A2** | J_topo ∝ quality | Architecture ordering | ✅ Strong if ordering holds |
| **A3** | β ∝ J_topo | Regression fit | ⚠️ Needs R² > 0.8 |
| **A4** | d_eff = d_task/J_topo | Hessian comparison | ⚠️ d_task unknown |
| **B1** | J_topo evolves | Time series | ✅ Direct |
| **B2** | EoS = sharpness → 2 | Time series | ✅ Cohen already showed |
| **B3** | ψ peaks at EoS | Time series | ⚠️ Indirect |

---

## Next Actions

1. **Now**: Complete Phase A (wait for VGG + DenseNet)
2. **Now**: Write Phase S0 (synthetic validation)
3. **After A**: Design Phase B training tracking code
4. **After all**: Write paper Results section
