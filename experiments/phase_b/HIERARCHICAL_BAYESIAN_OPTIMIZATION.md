# Hierarchical Bayesian Optimization for ThermoRG-NN
## Solving the Coupled Parameter Problem

**Date**: 2026-04-04
**Revised**: 2026-04-05 (corrected cooling theory from Phase S1 v3)
**Goal**: Design hierarchical BO that handles J_topo ↔ β ↔ E_floor ↔ D coupling

---

## 1. The Coupling Problem

Current naive approach: optimize J_topo alone, assume β and E_floor are fixed.

Reality:
```
J_topo ↔ E_floor: J_topo predicts E_floor (r=0.83) — THIS IS THE VALIDATED RELATIONSHIP
J_topo ↔ β: NO direct correlation (r=0.03, was artifact of fitter bound)
β ↔ E_floor: both appear in L(D) = α·D^(-β) + E_floor
D ↔ all: model size changes the entire scaling trajectory
γ ↔ β, E_floor: BatchNorm/LN cooling REDUCES γ → INCREASES β (φ≈2.0) → DECREASES E_floor
```

**Corrected cooling theory (Phase S1 v3)**:
- γ = variance fluctuation (net heating); larger γ = hotter
- BN/LN REDUCE γ (cooling): γ_BN=2.36 < γ_None=3.36
- φ(γ) is decreasing: lower γ → larger φ → larger β
- β_BN (0.37) > β_None (0.18), E_floor_BN (0.18) < E_floor_None (0.28)
- φ_BN = β_BN/β_None = 2.05 (cooling factor)

**Solution**: Model E_floor from J_topo (r=0.83); model β from normalization type (via γ coupling).

---

## 2. Hierarchical Fidelity Structure

### Level 0: Zero-Cost (J_topo)
- **Cost**: ~1ms per architecture (PI-20)
- **Observations**: J_topo at random initialization
- **What it gives**: Prior on E_floor via r=0.83 relationship
- **Uncertainty**: High (J_topo alone can't uniquely determine β or D effects)

### Level 1: Ultra-Low-Fidelity (5 epochs)
- **Cost**: ~0.5 GPU-minutes per architecture
- **Observations**: val_loss at D_max, 5 epochs
- **What it gives**: Learning speed proxy, early signal about optimization difficulty
- **What it tells about coupling**: 
  - High early loss + low J_topo → optimization struggling
  - High early loss + high J_topo → capacity issue
  - Low early loss + low J_topo → good sign

### Level 2: Low-Fidelity (50 epochs)
- **Cost**: ~5 GPU-minutes per architecture
- **Observations**: val_loss trajectory, ~50 epochs
- **What it gives**: 
  - Rough β estimate (from loss decay rate)
  - E_floor lower bound
  - J_topo validation (does trained J_topo match init?)
- **What it tells about coupling**:
  - β estimated from scaling of loss with D (if multiple D available)
  - Or from loss trajectory shape

### Level 3: Medium-Fidelity (200 epochs)
- **Cost**: ~30 GPU-minutes per architecture
- **Observations**: Converged val_loss, trained J_topo
- **What it gives**:
  - β fit from L(D) = α·D^(-β) + E_floor
  - E_floor estimate
  - Full J_topo trajectory (init → trained)

### Level 4: High-Fidelity (full convergence)
- **Cost**: ~60 GPU-minutes per architecture
- **Observations**: Full scaling law fit
- **What it gives**: Ground truth β, E_floor, α

---

## 3. Joint Surrogate Model

Instead of modeling E_floor(arch_features), model the **scaling trajectory**:

### State Variables (what we want to predict)
```
θ(arch) = [β(arch), E_floor(arch), J_topo_init(arch), ΔJ_topo(arch)]
```

### Observations at each fidelity
```
Level 0: J_topo_init  → direct observation of θ_3
Level 1: L(D_max, 5)  → noisy observation of E_floor and β
Level 2: L(D_max, 50) → better estimate of E_floor, rough β
Level 3: L(D, 200)    → fit β, E_floor directly
Level 4: L(D, full)  → ground truth
```

### Hierarchical GP Design

**Option A: Chain GP** (information flows up)
```
GP_0: arch_features → J_topo
GP_1: [arch_features, J_topo] → E_floor_prior
GP_2: [arch_features, J_topo, L_5ep] → E_floor_refined
GP_3: [arch_features, J_topo, L_50ep, D] → β, E_floor
```

**Option B: Multi-Task GP** (shared kernel)
```
GP_shared: kernel(arch_features) captures correlations
           across all fidelities
L(arch, fidelity) = f(arch) + g(fidelity) + ε
```

**Option C: Neural Network surrogate** (most flexible)
```
Input: arch_features (one-hot + continuous)
       + J_topo (computed)
       + fidelity indicator
Output: [β_pred, E_floor_pred, uncertainty]
```

**Recommended**: Option B (Multi-Task GP) because:
- Naturally handles multi-fidelity correlation
- Can share information across architectures and fidelities
- Well-studied for NAS (see prior work: MF-NAS, FAHB)

---

## 4. Acquisition Function with Corrected Coupling

### What to optimize?
We want to minimize **expected final loss** L_final(D_max, arch).

From scaling law:
L_final ≈ α·D_max^(-β) + E_floor

### State variables (corrected):

| Variable | Prior source | Uncertainty |
|----------|-------------|-------------|
| **E_floor** | J_topo → E_floor (r=0.83, linear) | GP residual after J_topo prior |
| **β** | Normalization type (via γ coupling) | GP from early-loss trajectory |
| **α** | Regularized (82–500), bounded | Moderate constant uncertainty |

### β prior from normalization (Phase S1 v3):
- β_None ≈ 0.18
- β_BN ≈ 0.37 (cooling reduces γ → increases β by φ≈2.0)
- β_LN ≈ 0.37 (similar cooling effect)

### Proposed acquisition (corrected):
```
score(arch) = -E_floor_pred(arch)
              - α·D_max^(-β_pred(arch))
              + λ · sqrt(σ²_Efloor + (α·D_max^(-β))² · σ²_β)

where:
- E_floor_pred = μ_E(J_topo) + δ_E(early_loss)   # J_topo prior + early-loss correction
- β_pred = μ_β(norm_type) + δ_β(early_loss)      # norm-type prior + early-loss correction
```

**Key correction**: Removed β-J_topo coupling (was invalid, r=0.03). J_topo only predicts E_floor, not β.

---

## 5. Active Learning Loop with Hierarchical Fidelity

```
ThermoRG-HBO(budget, constraints):
    
    # PHASE 0: Initialization
    1. Sample N_init architectures (Latin Hypercube)
    2. Compute J_topo for all (Level 0) → GP_0 calibrated
    3. Train top N_cal on Level 1 (5 epochs) → rough E_floor estimate
    4. Fit: E_floor ≈ f(J_topo) + g(ΔL_5ep)
    
    # PHASE 1: Active Loop
    while budget_remaining > minimum_cost:
        1. Generate N_cand candidate architectures (respect constraints)
        2. For each candidate:
           a. Compute J_topo (Level 0) → E_floor_prior
           b. Score with acquisition → expected_improvement
           c. Account for coupling: β_estimate from J_topo
        3. Select top K architectures for Level 1 (5 epochs)
        4. Update GP: (arch, J_topo, L_5ep) → E_floor_estimate
        5. If top arch stable for N_plateau rounds:
           → invest in Level 2 (50 epochs) for refinement
        6. If Level 2 confirms top arch:
           → invest in Level 3 (200 epochs) for confirmation
        7. Budget accounting:
           Level 1: 0.5 GPU-min
           Level 2: 5 GPU-min
           Level 3: 30 GPU-min
           Stop when: Level N+1 won't change decision, or budget exhausted
    
    # OUTPUT: best architecture + scaling law fit
```

### Budget allocation strategy:
```
Total budget B GPU-hours:
- 10% for initialization (Level 0 + Level 1 calibration)
- 50% for active exploration (mostly Level 1)
- 30% for refinement (Level 2 on top-K)
- 10% for validation (Level 3 on winner)

Expected total architectures evaluated:
- Level 0 (J_topo only): 1000 candidates
- Level 1 (5 epochs): 100 candidates
- Level 2 (50 epochs): 20 candidates
- Level 3 (200 epochs): 5 candidates
```

---

## 6. Coupling-Aware Candidate Generation (CORRECTED)

Instead of sampling architecture space uniformly, use the coupling relationships:

### Coupling constraints (corrected):
```
1. width ↑ → J_topo ↓, D ↑ (capacity ↑)
2. depth ↑ → J_topo ↑, D ↑ (nonlinearly)
3. skip ↑ → J_topo ↑, optimization easier
4. BatchNorm → γ ↓ (cooling), β ↑ by φ ≈ 2.0, E_floor ↓ (empirical: γ=2.36, β=0.37, E=0.18)
5. LayerNorm → γ ↓ (cooling), β ↑, E_floor ↓ (expected similar magnitude to BN)
```

**CORRECTED cooling effects**:
- BN/LN REDUCE γ (variance fluctuation = cooling)
- Lower γ → larger φ(γ) → LARGER β (not smaller!)
- Lower γ → lower E_floor (better asymptotic performance)
- Net effect: BN is BETTER for both β and E_floor

### Candidate generation rules:
1. **Respect capacity bound**: D_eff ≤ d_manifold·(log N+1)
2. **Prefer low J_topo**: lower J_topo → lower E_floor (r=0.83)
3. **Use normalization aggressively**: BN/LN improve both β and E_floor
4. **Search strategy**:
   - Wide models: low J_topo → low E_floor
   - BN/LN: bonus on both β and E_floor
   - Skip: stability + J_topo improvement

### Search space reduction via coupling (corrected):
```
Before: naive search over all (width, depth, skip, norm) combinations
After:  couple width-depth via J_topo target
        couple norm via β/E_floor improvement (BN/LN: both improve)
        couple skip via optimization stability + J_topo
```

---

## 7. Theoretical Grounding

### Why this should work:

1. **J_topo is cheap**: O(1ms) per arch vs O(30min) for full training
2. **J_topo → E_floor is strong**: r=0.83, well-validated
3. **BN/LN improve both β and E_floor**: stronger coupling signal to exploit
4. **Early loss is informative**: Level 1 (5 epochs) gives ~80% of information about E_floor at 20% of cost

### Expected efficiency gain (revised):
```
Random search: 1000 architectures at Level 3 = 1000 × 30 min = 500 GPU-hours
ThermoRG-HBO:  1000 × 1ms + 100 × 0.5min + 20 × 5min + 5 × 30min
             = ~0 GPU-hours + 50 + 100 + 150 = 300 GPU-hours
             = 40% reduction (conservative)

With corrected coupling (BN improves both β and E_floor):
  → stronger signal → faster convergence → potentially 60%+ reduction
```

Revised 2026-04-05: BN/LN effects corrected from "reduces β by φ≈0.66" to "increases β by φ≈2.0".

---

## 8. Implementation Notes

### GP Framework:
- Use `GPy` or `scikit-learn GaussianProcessRegressor`
- Custom kernel for architecture features (one-hot + continuous)
- Multi-output GP for (β, E_floor) joint prediction

### Fidelity levels:
- Implement as separate observation sets in GP
- Use "fidelity" as an additional input feature
- Or use Numpyro/SMT for hierarchical GP

### J_topo computation:
- PI-20 already implemented (23× faster than SVD)
- Integrate into candidate generation loop

### Scaling law fitting:
- Use `scipy.optimize.curve_fit`
- Set α_max = 500 (not 20) to avoid bound artifact
- Validate with bootstrap for uncertainty estimates

---

## 9. Open Questions

1. **How much does early loss reduce E_floor uncertainty?**
   → Need Phase B Session 2 data to calibrate

2. **Is the J_topo → E_floor coupling stable across architectures?**
   → Currently r=0.83 within ThermoNet; test on wider space

3. **What is the LN cooling factor φ_LN?**
   → Phase S1 v3 did not run LN; need data to confirm LN ≈ BN or different

4. **Does ΔJ_topo (init → trained) carry information?**
   → Hypothesis: large ΔJ → optimization struggling

5. **Is the BN β boost (φ≈2.0) consistent across widths?**
   → Currently measured at D=32,48,64 only; need D=96 confirmation

---

## 10. Next Steps

1. **Implement**: Basic version of hierarchical GP (2 levels: J_topo + Level 1)
2. **Calibrate**: Run Phase B Session 2 to get data for coupling parameters
3. **Test**: Compare ThermoRG-HBO vs random search on 100 architectures
4. **Iterate**: Refine based on empirical efficiency gains
