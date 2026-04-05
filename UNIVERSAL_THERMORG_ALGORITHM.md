# Universal ThermoRG Algorithm
## Architecture Search Without Full Training

**Date**: 2026-04-05
**Status**: Design Complete
**Based on**: THERMO RG THEORY.md v7

---

## 1. Overview

The Universal ThermoRG Algorithm enables efficient neural architecture search by avoiding full training until necessary. It leverages:

1. **J_topo** — zero-cost topology quality metric from initialization
2. **Cooling Theory** — normalization effects on training dynamics
3. **Scaling Laws** — universal power-law form
4. **HBO** — multi-fidelity Bayesian optimization

**Core idea**: Compute cheap proxies (J_topo, early loss) → decide which architectures are worth full training.

---

## 2. Data-Agnostic vs Data-Dependent

### Data-Agnostic (Universal — No Data Required)
- **J_topo formula**: exp(−mean|log η_l|) — computed from weights only
- **Cooling mechanism**: γ = variance fluctuation, φ(γ) = e^(-γ/γ_c)/(1+γ/γ_c)
- **Scaling law form**: L(D) = α·D^(-β) + E_floor
- **HBO framework**: multi-fidelity acquisition loop
- **Layer definitions**: Convolutional, linear, attention projection matrices

### Data-Dependent (Requires Calibration Per Dataset)
- **J_topo → E_floor correlation** (r = 0.83 on CIFAR-10; must remeasure)
- **Cooling factors**: φ_BN, φ_LN (≈2.0 for CIFAR-10)
- **Scaling parameters**: α, β, E_floor (dataset-specific)
- **Early-loss predictive power**: L1 → final loss correlation
- **Manifold dimension**: d_manifold (data complexity)

---

## 3. Three-Phase Workflow

### Phase 0: Calibration (Per Dataset, One-Time Cost)

**Goal**: Fit dataset-specific parameters using minimal experiments.

```
INPUT: Dataset D, budget B_cal (≈1 GPU-hour)
OUTPUT: Calibrated parameters Θ

1. Sample 5-10 diverse architectures (vary width, depth, skip, norm)
2. For each architecture A_i:
   a. Compute J_topo(A_i) via PI-20 (zero-cost)
   b. Train A_i for 5 epochs on 10% of D
   c. Record early loss L1_i
3. Fit linear regression: E_floor ≈ a·J_topo + b
4. Fit scaling law: L(D) = α·D^(-β) + E_floor
5. Estimate cooling factors φ_BN, φ_LN
6. Return Θ = {r_JtoE, f_JtoE, α, β, E_floor, φ_BN, φ_LN, d_manifold}
```

**Cost**: < 1 GPU-hour (5-10 archs × 5 epochs × 10% data)

---

### Phase 1: Hierarchical Bayesian Optimization (HBO)

**Goal**: Find optimal architecture using calibrated parameters.

```
INPUT: Search space S, calibrated Θ, budget B
OUTPUT: Optimal architecture A*

# --- Initialization ---
GP = MultiTaskGP(features, fidelity_levels)
candidates = LatinHypercube(S, N=100)

for arch in candidates:
    arch.J = compute_J_topo(arch)           # L0: zero-cost
    arch.E_prior = Θ.f_JtoE(arch.J)        # E_floor from J_topo
    arch.β_prior = Θ.β_from_norm(arch.norm)  # β from norm type

# --- Level-1 Screening (5-epoch training) ---
top_K = select_top_K(candidates, K=20, acquisition_score)
for arch in top_K:
    arch.L1 = train(arch, D_subset, epochs=5)  # L1: ~0.5 GPU-min
    update_GP(GP, arch, fidelity=1)

# --- Active Loop ---
while budget > minimum_cost:
    # Propose new candidates using GP
    new_candidates = propose_candidates(GP, S, N=10)
    
    for arch in new_candidates:
        arch.J = compute_J_topo(arch)
        arch.E_prior = Θ.f_JtoE(arch.J)
        arch.β_prior = Θ.β_from_norm(arch.norm)
    
    # Select by acquisition score
    selected = top_M(acquisition_score(new_candidates), M=5)
    
    for arch in selected:
        arch.L1 = train(arch, D_subset, epochs=5)
        update_GP(GP, arch, fidelity=1)
    
    # Periodic refinement
    if iteration % 5 == 0:
        best = get_best_expected(GP)
        best.L2 = train(best, D_subset, epochs=50)  # L2
        if budget permits:
            best.params = fit_scaling_law(best)      # L3
    
    budget -= cost_of_iteration

return best_architecture(GP)
```

**Cost**: ~10-100 GPU-hours (vs 1000+ for random search)

---

### Phase 2: Deployment

- Full training of selected architecture on entire dataset
- Optional hyperparameter tuning
- Final evaluation

---

## 4. Acquisition Function

The acquisition score combines E_floor prediction and β prior:

```
score(arch) = -E_floor_pred(arch, J_topo, L1)
              - α·D_max^(-β_pred(arch, norm_type, L1))
              + λ · sqrt(σ²_E + (α·D_max^(-β))² · σ²_β)

where:
- E_floor_pred = μ_E(J_topo) + δ_E(L1)     # J_topo prior + L1 correction
- β_pred = μ_β(norm_type) + δ_β(L1)         # norm-type prior + L1 refinement
```

**Key**: J_topo (static topology) and L1 (training dynamics) provide complementary information.

---

## 5. Modality Handling

### Vision (CIFAR-10, ImageNet)
- **Architecture**: ConvNet / ResNet variants
- **J_topo**: Convolutional filters as matrices; skip: Ŵ = S + W
- **Normalization**: BatchNorm (standard), LayerNorm (ViT)
- **Manifold**: PCA on flattened image patches

### Language (Text)
- **Architecture**: Transformer blocks
- **J_topo**: Linear layers in attention & MLP; LayerNorm excluded (η=1)
- **Normalization**: LayerNorm (standard)
- **Manifold**: PCA on token embeddings

### Video / Multimodal
- **Architecture**: 3D ConvNet or Video Transformer
- **J_topo**: Spatiotemporal layers; same skip-connection rule
- **Normalization**: BatchNorm3D or LayerNorm
- **Manifold**: PCA on spatiotemporal patches

### Universal Layer Rules
- Any new layer type: define effective-dimension computation
- Normalization layers: excluded from J_topo (η=1)
- Skip connections: Ŵ = S + W

---

## 6. Efficiency Analysis

### Time Comparison

| **Method** | **Cost Formula** | **Example (N=1000)** |
|------------|------------------|---------------------|
| Random Search | N × T_full | 1000 × 7 days = 7000 GPU-days |
| Grid Search | N × T_full | Massive |
| **ThermoRG-HBO** | N × T_init + K × T_short | 1000 × 1s + 20 × 30min ≈ 10 GPU-hours |

### Scaling with Model Size

| **Model Size** | **T_full** | **T_short (L1)** | **HBO Speedup** |
|----------------|------------|-------------------|------------------|
| 1M params | 1 hour | 6 minutes | ~100× |
| 1B params | 7 days | 30 minutes | ~1400× |
| 70B params | months | hours | ~1000× |

**Key assumption**: J_topo and/or L1 must correlate with final performance on the target dataset.

---

## 7. Minimal Validation Protocol

Before deploying on a new dataset, run mini-validation:

```
1. Sample 10-20 diverse architectures
2. Compute J_topo for all (zero-cost)
3. Train each for full convergence (or 90% of T_full)
4. Compute correlation r = corr(J_topo, final_loss)
5. If |r| > 0.5 → HBO will likely save time
   If |r| < 0.3 → HBO benefit is limited
```

This ~10 full-training runs establishes whether HBO is worthwhile.

---

## 8. Implementation Structure

```
thermorg/
├── UN IVERSAL_ALGORITHM.md      # This document
├── theory/
│   └── THEORY.md                # v7: Core theory + Section 7
├── phase_b/
│   ├── HIERARCHICAL_BAYESIAN_OPTIMIZATION.md  # HBO design
│   └── hbo_implementation.py    # HBO code
└── calibration/
    └── calibrate.py            # Phase 0 calibration
```

---

## 9. Key Takeaways

1. **Universal core**: J_topo, cooling theory, scaling-law form, and HBO framework work for **any dataset** without modification.

2. **Minimal calibration**: Only 5-10 architectures trained for 5 epochs each (~1 GPU-hour) needed to fit dataset-specific constants.

3. **Massive efficiency**: HBO reduces search cost by 100-1000× compared to random search on large models.

4. **Modality-agnostic**: Same algorithm applies to vision, language, and video by adapting layer-type definitions.

5. **Implementation-ready**: Design is concrete enough to implement directly.

---

## 10. References

- ThermoRG Theory: `theory/THEORY.md` (v7)
- HBO Design: `experiments/phase_b/HIERARCHICAL_BAYESIAN_OPTIMIZATION.md`
- Phase S1 (Cooling): `experiments/phase_s1/`
- Phase B Session 1: `experiments/phase_b/`
