# SU-HBO: Stepwise Utility-guided Hierarchical Bayesian Optimization

**Version**: 1.0
**Date**: 2026-04-06
**Based on**: ThermoRG Theory v8, Phase A, Phase S1, HBO design

---

## 1. Overview

SU-HBO combines three approaches into one coherent algorithm:

| Component | Role |
|-----------|------|
| **Stepwise Composition** | Outer loop: starts minimal, monitors β(t)/γ(t), triggers modifications on plateau |
| **Utility Function** | Objective: `U(A) = -E_floor(J, γ) + λ·β(γ)` |
| **HBO (Multi-fidelity GP)** | Surrogate model + acquisition for exploration/exploitation |

### Why This Unification?

- **Stepwise** provides interpretable, incremental architecture building
- **Utility function** provides quantitative objective grounded in thermodynamics
- **HBO** provides principled exploration with multi-fidelity efficiency
- **E_floor formula** provides physics-based prior, reducing sample complexity

---

## 2. Core Equations

### 2.1 E_floor Formula (Derived from Non-Equilibrium Thermodynamics)

$$E_\mathrm{floor}(J, \gamma) = \frac{k \cdot \gamma}{1 - B \cdot J \cdot \gamma}$$

Where:
- $J$ = J_topo (topological complexity)
- $\gamma$ = variance instability parameter
- $k \approx 0.06$, $B \approx 0.15$ (fitted from Phase A)

**Properties**:
- E_floor **increases** with J (higher topology → harder optimization)
- E_floor **decreases** with cooling (lower γ → better final error)
- **Critical condition**: $\gamma < 1/(B \cdot J)$ — beyond this, training diverges

### 2.2 Cooling Factor

$$\phi(\gamma) = \frac{\gamma_c}{\gamma_c + \gamma} \exp\!\Bigl(-\frac{\gamma}{\gamma_c}\Bigr)$$

### 2.3 Beta (Learning Speed)

$$\beta(\gamma) = \beta_0 \cdot \phi(\gamma)$$

### 2.4 Utility Function

$$U(A) = -E_\mathrm{floor}(J, \gamma) + \lambda \cdot \beta(\gamma)$$

Where $\lambda$ balances trade-off between asymptotic error and learning speed.

---

## 3. Algorithm

### 3.1 Initialization

```
INPUT: Dataset D, baseline architecture A₀, budget B, parameters k, B, λ, γ_c

1. Compute J_topo(A₀) via PI-20 (zero-cost, L0 fidelity)
2. Train A₀ for 5 epochs on 10% of D → estimate γ₀, β₀ (L1 fidelity)
3. Initialize GP surrogate with prior mean m(x) = U(A)
4. Set U_best = U(A₀), A_current = A₀
5. budget_remaining = B
```

### 3.2 Main Loop

```
WHILE budget_remaining > 0:

    # 1. Monitor training
    train(A_current, epochs=5)  # L1 evaluation
    track β(t), γ(t) over recent epochs

    # 2. Plateau detection
    IF variance(β(t)) < ε_β AND variance(γ(t)) < ε_γ AND loss_decorrelation < ε_loss:
        # Trigger architecture search
        candidates = generate_candidate_actions(A_current)

        # 3. Score candidates
        scored = []
        FOR each candidate A' in candidates:
            J' = predict_J_topo(A')  # cheap (L0)
            γ' = predict_γ(A', γ_current)  # from action library
            U_pred = -E_floor(J', γ') + λ · β(γ')

            # GP prediction at full fidelity
            μ, σ = gp.predict(A', fidelity=L3)
            ei = expected_improvement(μ, σ, U_best)

            scored.append((ei, A', U_pred))

        # 4. Select and evaluate
        best_ei, best_A, U_pred = max(scored)
        loss_L1 = evaluate(best_A, fidelity=L1)
        gp.update(best_A, loss_L1, fidelity=L1)

        # 5. Commit decision
        IF U_pred > U_best + threshold:
            A_current = best_A
            U_best = U_pred
            log(f"Accepted: {best_A}, U={U_best:.4f}")

    budget_remaining -= cost(L1)

    # 6. Periodic high-fidelity refinement
    IF iteration % N == 0:
        loss_L3 = evaluate(A_current, fidelity=L3)
        gp.update(A_current, loss_L3, fidelity=L3)

    # 7. Stopping criteria
    IF no_improvement_for_K_consecutive_steps:
        BREAK
    IF budget_remaining <= 0:
        BREAK

OUTPUT: A_current, U_best
```

### 3.3 Candidate Action Generation

Actions modify architecture incrementally:

| Action | ΔJ | Δγ | Notes |
|--------|-----|-----|-------|
| Add BatchNorm | +0.05 | -1.0 | Strong cooling |
| Add LayerNorm | +0.02 | -0.7 | Moderate cooling |
| Remove norm | -0.05 | +0.8 | |
| Add skip | +0.35 | -0.2 | Improves gradient flow |
| Remove skip | -0.35 | +0.2 | |
| Increase width (+32) | -0.15 | 0.0 | |
| Decrease width (-32) | +0.15 | 0.0 | |
| Increase depth (+2) | +0.04 | +0.1 | |
| Decrease depth (-2) | -0.04 | -0.1 | |

### 3.4 GP Surrogate Model

**Features**: (J_topo, γ, width, depth, has_skip, has_bn, has_ln)

**Kernel**: RBF + linear kernel for fidelity dimension

**Prior mean**: m(x) = U(A) = -E_floor(J, γ) + λ·β(γ)

### 3.5 Plateau Detection

```
plateau_detected():
    window = 10 epochs
    IF var(β[last window]) < ε_β AND var(γ[last window]) < ε_γ:
        # Check if loss is still improving
        IF (loss[t-10] - loss[t]) / loss[t-10] < ε_loss:
            RETURN True
    RETURN False
```

---

## 4. Parameters

### 4.1 Fixed Parameters (from Theory)

| Parameter | Value | Source |
|-----------|-------|--------|
| k | 0.06 | Fitted from Phase A |
| B | 0.15 | Fitted from Phase A |
| γ_c | 2.0 | From Phase S1 |
| β_BN | 0.368 | Measured |
| β_None | 0.180 | Measured |
| γ_BN | 2.36 | Measured |
| γ_None | 3.36 | Measured |

### 4.2 Tunable Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| λ | 10.0 | Trade-off weight in utility |
| ε_β | 0.01 | Plateau detection threshold for β |
| ε_γ | 0.05 | Plateau detection threshold for γ |
| ε_loss | 0.001 | Minimum loss improvement rate |
| threshold | 0.01 | Minimum utility improvement to accept |
| N | 5 | Steps between high-fidelity evaluations |
| K | 3 | Consecutive steps without improvement to stop |

---

## 5. Fidelity Levels

| Level | Cost | Information |
|-------|------|-------------|
| L0 | ~0 (PI-20 computation) | J_topo only |
| L1 | 5 epochs | γ, β, early loss |
| L2 | 50 epochs | Mid-training dynamics |
| L3 | 200 epochs | Full convergence |

---

## 6. Comparison to Original Approaches

| Approach | Role in SU-HBO |
|----------|----------------|
| HBO | Surrogate model + multi-fidelity acquisition engine |
| Stepwise | Outer loop + plateau detection + action triggering |
| Utility Function | Objective + GP prior mean |

---

## 7. Implementation Notes

### 7.1 Action Library Calibration

Before running SU-HBO, calibrate action effects:

```python
action_effects = {
    'add_bn': {'delta_gamma': -1.0, 'delta_j': 0.05},
    'add_ln': {'delta_gamma': -0.7, 'delta_j': 0.02},
    'add_skip': {'delta_gamma': -0.2, 'delta_j': 0.35},
    # ...
}
```

### 7.2 GP Implementation

Use BoTorch or scikit-learn GP with:
- Multi-output kernel for different fidelities
- Custom mean function: `mean(x) = -E_floor(x['j'], x['gamma']) + λ * beta(x['gamma'])`

### 7.3 Budget Allocation

Recommended budget split:
- 50% for L1 evaluations
- 30% for L2 evaluations
- 20% for L3 refinements

---

## 8. References

- ThermoRG Theory: `theory/THEORY.md` v8
- HBO Design: `experiments/phase_b/HIERARCHICAL_BAYESIAN_OPTIMIZATION.md`
- Phase A Results: `experiments/phase_a/`
- Phase S1 Results: `experiments/phase_s1/`
- Simulation: `experiments/phase_b/stepwise_simulation_v*.py`
