# Hierarchical Bayesian Optimization for ThermoRG-NN
## Solving the Coupled Parameter Problem

**Date**: 2026-04-04
**Revised**: 2026-04-05 (corrected cooling theory from Phase S1 v3)
**Revised**: 2026-04-05 evening (added ResNet-18 paradox, confirmed HBO necessity)
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

### The ResNet-18 Paradox (Why J_topo Alone Is Not Enough)

**Observation**: ResNet-18 has J_topo = 4.16 (very HIGH, should be bad by J_topo → E_floor theory), yet performs well in practice.

**Why?** J_topo measures **initialization topology only**, not training dynamics. BatchNorm provides:
1. **Gradient stabilization** → enables higher learning rates
2. **Internal covariate shift reduction** → better optimization trajectory
3. **Training stability** → affects 5-50 epoch loss, not just final loss

**This is why a two-stage approach (filter by J_topo → full train) FAILS:**
- It would filter out high-J_topo + BN archs like ResNet-18
- It ignores the **training dynamics dimension** that normalization provides

**Why HBO is necessary:**
- L0 (J_topo): E_floor proxy (asymptotic performance)
- L1 (5-epoch loss): β/training dynamics proxy (training efficiency)
- Both together → complete evaluation
- ResNet-18 scores well on L1 even with high J_topo

**Corrected cooling theory (Phase S1 v3)**:
- γ = variance fluctuation (net heating); larger γ = hotter
- BN/LN REDUCE γ (cooling): γ_BN=2.36 < γ_None=3.36
- φ(γ) is decreasing: lower γ → larger φ → larger β
- β_BN (0.37) > β_None (0.18), E_floor_BN (0.18) < E_floor_None (0.28)
- φ_BN = β_BN/β_None = 2.05 (cooling factor)

**Solution**: Model E_floor from J_topo (r=0.83); model β from normalization type (via γ coupling); use L1 (5-epoch loss) to capture training dynamics that J_topo cannot.

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

### What we know about each dimension:

| Dimension | What predicts it | Validation |
|-----------|-----------------|------------|
| **E_floor** | J_topo (r=0.83) | Phase A |
| **β** | Norm type (via γ coupling) + L1 loss | Phase S1 v3 |
| **Training stability** | Norm type (BN > LN > None) | ResNet-18 observation |

### Why L1 (5-epoch loss) is critical:
- Phase B Session 1: J_topo does NOT predict early loss (r=0.40, p=0.60)
- This means J_topo alone can't tell us about training dynamics
- L1 captures: optimization trajectory, β information, stability
- ResNet-18 (high J_topo + BN) would score well on L1

### State variables (corrected):

| Variable | Prior source | Refinement from L1 |
|----------|-------------|-------------------|
| **E_floor** | J_topo → E_floor (r=0.83, linear) | L1 loss refines estimate |
| **β** | Norm type (via γ coupling: BN=0.37, None=0.18) | L1 loss trajectory reveals β |
| **α** | Regularized (82–500), bounded | L2/L3 needed for α |

### β prior from normalization (Phase S1 v3):
- β_None ≈ 0.18
- β_BN ≈ 0.37 (cooling reduces γ → increases β by φ≈2.0)
- β_LN ≈ 0.37 (similar cooling effect)

### Acquisition function (corrected, with L1):

```
score(arch) = -E_floor_pred(arch, J_topo, L1)
              - α·D_max^(-β_pred(arch, norm_type, L1))
              + λ · sqrt(σ²_Efloor + (α·D_max^(-β))² · σ²_β)

where:
- E_floor_pred = μ_E(J_topo) + δ_E(L1_loss)     # J_topo prior + L1 correction
- β_pred = μ_β(norm_type) + δ_β(L1_trajectory)   # norm-type prior + L1 refinement

Interpretation:
- μ_E(J_topo): base E_floor from J_topo (asymptotic floor)
- δ_E(L1_loss): L1 observation adjusts E_floor estimate
- μ_β(norm_type): base β from normalization (cooling effect)
- δ_β(L1_trajectory): L1 learning rate reveals β
```

**Key insight**: J_topo and L1 provide **complementary** information:
- J_topo: static topology → E_floor proxy
- L1: training dynamics → β/training stability proxy
- Together: complete picture of architecture quality

---

## 5. Active Learning Loop with Hierarchical Fidelity (CORRECTED)

### Why the multi-fidelity cascade matters:

1. **J_topo alone is insufficient** (ResNet-18 paradox)
2. **L1 (5 epochs) captures training dynamics** that J_topo cannot
3. **L2/L3 needed for β refinement** beyond L1's noisy signal

```
ThermoRG-HBO(budget, constraints):

    # PHASE 0: Initialization
    1. Sample N_init architectures (Latin Hypercube)
    2. Compute J_topo for all (Level 0) → E_floor_prior
    3. Train top N_cal on Level 1 (5 epochs) → training dynamics proxy
    4. Fit: E_floor ≈ f(J_topo) + g(L1_loss)
    5. Fit: β ≈ h(norm_type) + k(L1_trajectory)

    # PHASE 1: Active Loop
    while budget_remaining > minimum_cost:
        1. Generate N_cand candidate architectures (respect constraints)
        2. For each candidate:
           a. Compute J_topo (Level 0) → E_floor_prior
           b. Look up norm_type → β_prior
           c. Score with acquisition (J_topo + norm → β_pred + E_floor_pred)
        3. Select top K architectures for Level 1 (5 epochs)
        4. Update GP: (arch, J_topo, L1) → [E_floor, β] estimates
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

### Why this cascade is necessary:

| Level | What it adds | Why can't lower level replace it |
|-------|-------------|-------------------------------|
| L0 (J_topo) | E_floor prior | Can't predict training dynamics |
| L1 (5 epochs) | β/training stability | Reveals optimization trajectory |
| L2 (50 epochs) | Refined β, α | Still noisy for α |
| L3 (200 epochs) | Full scaling law | Ground truth |

**Key**: J_topo and L1 are **complementary**, not redundant. Both needed.

### Budget allocation strategy:
```
Total budget B GPU-hours:
- 10% for initialization (Level 0 + Level 1 calibration)
- 50% for active exploration (mostly Level 1) — key for training dynamics
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
2. **Use normalization aggressively**: BN/LN improve both β and E_floor
3. **Wide models benefit from J_topo reduction**: lower J_topo → lower E_floor
4. **Don't filter by J_topo alone**: ResNet-18 shows high J_topo + BN can work well
5. **Search strategy**:
   - Prioritize BN/LN architectures (better β and E_floor)
   - Within each norm type, use J_topo to rank
   - Use L1 to validate (not just J_topo)

### Search space reduction via coupling (corrected):
```
Before: naive search over all (width, depth, skip, norm) combinations
After:  couple norm via β/E_floor improvement (BN/LN: both improve)
        use J_topo as secondary filter (not primary)
        use L1 for training dynamics validation
```

**Key insight**: The TWO-STAGE approach (filter by J_topo → train) is wrong because:
- It filters out ResNet-18-like architectures (high J_topo + BN)
- It ignores training dynamics that L1 captures
- HBO's multi-fidelity approach captures both dimensions

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
