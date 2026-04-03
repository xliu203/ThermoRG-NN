# ThermoRG Active Learning Loop — Design Document

> **Status**: Design draft (v5, 2026-04-03)
> **Author**: ThermoRG Team (theory + DeepSeek Reasoner)
> **Purpose**: Complete algorithmic design for automated architecture search via J_topo-driven active learning

---

## 1. Motivation

The ThermoRG framework establishes:

$$L(D) = \alpha \cdot D^{-\beta} + E$$

**What J_topo actually controls** (corrected from Phase A re-analysis, 2026-04-03):

| Parameter | Controlled by J_topo? | Evidence |
|-----------|---------------------|---------|
| $\alpha$ | ❌ No (RFF only) | Phase S0: diverges near J_c; ThermoNet: regularized (82–500), not J-controlled |
| $\beta$ | ❌ No (ThermoNet) | Phase A: r(β, J) = 0.03 (was artifact of fitter bound) |
| **$E_\mathrm{floor}$** | **✅ YES** | Phase A: **r(J, E) = 0.83** |

**Key insight for Phase B**: The actionable relationship is **$J_\mathrm{topo} \to E_\mathrm{floor}$** — lower J_topo → lower E_floor → better final performance. This is the **primary optimization target** for architecture search.

---

## 2. Goal

Design an automated loop that:
1. Explores architecture space under resource constraints (params, FLOPs, latency)
2. Uses J_topo as a cheap surrogate to predict E_floor (zero training cost)
3. Trains only the most promising candidates
4. Learns from training feedback to improve predictions

**Efficiency target**: Find architectures within 5% of optimal, at <20% of the training cost of random search.

---

## 3. Core Hypothesis (Phase A validated, v5)

**H1 (corrected)**: $J_\mathrm{topo}$ predicts $E_\mathrm{floor}$ for ThermoNet:
- r(J, E) = 0.83 across ThermoNet families ✅
- Lower J_topo → lower E_floor → better performance

**H2 (corrected)**: $\beta$ is NOT directly J-controlled in real architectures:
- Original Phase A r(β, J) = 0.66 was an **artifact of restrictive α bound = 20**
- Corrected r(β, J) = 0.03 — no real correlation
- $\beta$ is architecture-dependent but not J_topo-controlled

**H3 (alpha phase transition)**: Validated in RFF networks only:
- Phase S0: α jumps 220× (15→22000) near J_c ≈ 0.35
- Phase A: α is regularized by skip connections, LayerNorm → bounded (82–500)
- Not actionable for ThermoNet architecture search

**H4 (architecture families)**:
- **ThermoNet** (ideal topology): Follow ThermoRG scaling law (R² > 0.95), J_topo → E_floor validated
- **ResNet** (real topology): Subject to "van der Waals correction" from stride-2 downsampling. Forms a separate family with own parameters.

---

## 4. Algorithm

### 4.1 Overview

```
ThermoRG-AL(D_dataset, budget, constraints)
│
├─ PHASE 1: Initialization (cold start)
│   ├─ Sample N_init architectures (Latin Hypercube)
│   ├─ Compute J_topo at initialization for all
│   ├─ Train N_cal subset for N_cal_epochs (establish J_topo→E mapping)
│   └─ Fit GP surrogate: arch_features → (J_topo, E_floor)
│
├─ PHASE 2: Active Loop (while budget remains)
│   ├─ Generate N_cand candidate architectures (respect constraints)
│   ├─ Compute J_topo at initialization for all candidates
│   ├─ Score each with acquisition function (EI + constraint penalty)
│   ├─ Select top-K for short training (N_short epochs)
│   ├─ Observe actual loss, update GP
│   └─ If top arch hasn't improved in N_plateau rounds → stop
│
└─ OUTPUT: best architecture found
```

### 4.2 Architecture Encoding

Each architecture is encoded as a feature vector:

| Feature | Type | Range/Values |
|---------|------|-------------|
| depth | int | 3–20 |
| width multiplier | float | 0.5–4.0 |
| skip connections | binary per layer | 0/1 |
| layer type | categorical | FC, Conv2d, Bottleneck |
| width per layer | list of ints | e.g., [64, 128, 256] |
| activation | categorical | relu, gelu, silu |
| normalization | categorical | BN, LN, GN, None |

### 4.3 J_topo Computation (Zero-Training-Cost)

```python
def J_topo_at_init(arch):
    """Compute J_topo from random initialization weights."""
    weights = sample_init_weights(arch)  # random init, no training
    return compute_J_topo(weights, input_dim=arch.input_dim)
```

Cost: ~1ms per architecture (using PI-20).

### 4.4 Surrogate Model

**Model**: Gaussian Process with two outputs:
1. **J_topo** (directly observed from weights)
2. **E_floor** (observed after sufficient training)

**Kernel**: Matern-5/2 for continuous features + Hamming distance for categorical features.

**Training data structure**:
- `(arch_encoding, J_topo, E_floor)` for each evaluated architecture
- J_topo is available for all architectures (trained or not)
- E_floor is observed only after enough training (D ≥ 2000, 50+ epochs)

**Warm-start**: The relationship $J_\mathrm{topo} \to E_\mathrm{floor}$ provides a **theoretical prior**:

$$E_\mathrm{floor} \approx E_0 + k_E \cdot (J_\mathrm{topo} - J_0)$$

where $E_0, J_0, k_E$ are calibrated from Phase A data ($k_E \approx 0.83$ correlation).

### 4.5 Acquisition Function

**Objective**: Minimize expected asymptotic loss $E_\mathrm{floor}$.

**Expected Improvement (EI)**:

$$\mathrm{EI}(x) = \mathbb{E}\left[\max(0, E_\mathrm{best} - E(x))\right]$$

**Components**:
1. **Predicted E_floor** $\hat{E}(x)$ from GP, using $J_\mathrm{topo} \to E$ relationship
2. **Uncertainty** $\sigma(x)$ from GP posterior
3. **Constraint penalty** $P_{\mathrm{constraint}}(x)$: 0 if satisfied, $-\infty$ if violated

**Final score**:
$$\mathrm{score}(x) = -\hat{E}(x) + \lambda \cdot \sigma(x) - \gamma \cdot P_{\mathrm{constraint}}(x)$$

where $\lambda$ is the exploration coefficient (e.g., $\lambda = 0.1$).

**Theoretical grounding**:
- $\hat{E}(x) = \hat{\alpha} \cdot D^{-\hat{\beta}} + E$ with $\hat{E}$ predicted from $J_\mathrm{topo}(x)$
- $\sigma(x)$ encodes exploration value: uncertain architectures that might have lower E
- Minimizing E = minimizing asymptotic loss = maximizing architecture quality

### 4.6 Training & Feedback

**Multi-fidelity evaluation**:

| Fidelity | Epochs | Cost | Use case |
|----------|--------|------|----------|
| RFF-proxy | ~50 | ~1 GPU-hour | Calibrate J_topo→E mapping |
| Low | 5–10 | ~0.5 GPU-hour | Quick filter in active loop |
| Medium | 20–30 | ~2 GPU-hour | Refine top candidates |
| Full | 50–200 | ~10 GPU-hour | Final validation |

**E_floor estimation**: Fit $L(D) = \alpha \cdot D^{-\beta} + E$ with corrected $\alpha_\max = 500$. Use E at D=50000 as proxy for $E_\mathrm{floor}$.

**Feedback to GP**:
- After training at fidelity $f$, fit E_floor
- Update GP with $(x, J_\mathrm{topo}(x), E_\mathrm{floor-est})$

### 4.7 Constraint Handling

**Hard constraints** (filter candidates):
- Max parameters: $N_\mathrm{params} \leq N_\max$
- Max FLOPs: $\mathrm{FLOPs} \leq \mathrm{FLOPs}_\max$
- Max latency: $\mathrm{latency} \leq \mathrm{latency}_\max$

**Soft constraints** (penalize in score):
- Prefer smaller models when J_topo is similar
- Regularization: $P_\mathrm{size}(x) = \log N_\mathrm{params}(x) / \log N_\max$

**Constraint-aware candidate generation**:
- Sample architectures conditioned on satisfying hard constraints
- Use inverse-weighting to bias toward constraint-satisfying regions

### 4.8 Initialization Strategy

**Goal**: Establish J_topo→E_floor mapping with minimal training.

**Recommended**:
- $N_\mathrm{init}} = 20$ architectures sampled via Latin Hypercube
- Compute J_topo for all 20
- Train top-5 and bottom-5 (diverse J_topo range) for $N_\mathrm{cal}} = 50$ epochs
- Fit E_floor from scaling law to establish regression J_topo → E

### 4.9 Termination Conditions

Stop when ANY of:
1. **Budget exhausted**: $B_\mathrm{remaining}} < \mathrm{cost}(N_\mathrm{short\_train}})$
2. **Plateau**: No improvement in top architecture E_floor over $N_\mathrm{plateau}} = 5$ consecutive selections
3. **Uncertainty too high**: Average GP uncertainty $\bar{\sigma} > \sigma_\max$ (means model is unreliable)
4. **Theoretical bound**: Achieved $J_\mathrm{topo}}$ within $\epsilon$ of theoretical minimum (lower J → lower E)

---

## 5. Theoretical Predictions Guiding the Loop

### 5.1 E_floor Prediction from J_topo

Given J_topo at initialization:

$$\hat{E}_\mathrm{floor} = E_0 + k_E \cdot (J_\mathrm{topo} - J_0)$$

where $E_0, J_0, k_E$ are calibrated from Phase A data.

**Phase A calibration (corrected)**:
- $k_E \approx 0.83$ (strong positive correlation)
- Lower J_topo → lower E_floor → better final performance
- E_floor ≈ 0.84–1.25 for ThermoNet at D=50000

### 5.2 Optimal J_topo

**Prediction**: $J_\mathrm{topo}} \to 0$ is optimal (no information bottlenecks, perfect flow).

**Practical target**: $J_\mathrm{topo}} < 0.2$ likely indicates well-designed architecture.

**Note**: This is opposite to the original β ∝ J prediction. The corrected relationship $J \to E$ means lower J is better.

### 5.3 Width Profile from J_topo

From $J_\mathrm{topo}}$ constraint, we can derive the width profile:

$$\eta_l = \frac{D_\mathrm{eff}^{(l)}}{D_\mathrm{eff}^{(l-1)}} \approx \mathrm{const}$$

For $J_\mathrm{topo}} \approx 1$, we need $\eta_l \approx 1$ for all layers, meaning:

$$\|W_l\|_F^2 / \lambda_\max(W_l) \approx \mathrm{const} \cdot \lambda_\mathrm{prev}$$

This provides a **targeted width profile** for architecture generation.

---

## 6. Failure Modes & Mitigations

| Failure Mode | Detection | Mitigation |
|-------------|-----------|------------|
| J_topo not predictive of E | GP residuals large after N_cal | Increase N_cal; use random search baseline comparison |
| GP poor on categorical arch space | Cross-val RMSE high | Use hierarchical GPs; separate models for depth vs width |
| Constraints too restrictive | Few candidates pass filter | Relax hard constraints; use soft constraints instead |
| Training feedback noisy | Loss variance high across seeds | Use more seeds per evaluation; fit scaling law instead of point loss |
| J_topo changes significantly after training | Init J_topo vs final J_topo low correlation | Use early-stopping J_topo instead of init J_topo |
| Catastrophic forgetting in GP | New data contradicts old | Use online GP or weight by recency |

---

## 7. Implementation Roadmap

### Phase B1: Critical Gap Experiments (COMPLETED)
- [x] CPU experiments: architectures at J_c ≈ 0.40 (fill critical region) ✅
- [x] Compute J_topo for candidate architectures ✅
- [x] Validate alpha behavior in critical region ✅

### Phase B2: Core Infrastructure
- [ ] Architecture encoding class (`ArchSpace`, `Architecture`)
- [ ] J_topo computation from random init weights (PI-20, 23× speedup)
- [ ] Constraint calculators (params, FLOPs, latency)
- [ ] Latin Hypercube sampler

### Phase B3: Surrogate Model
- [ ] GP implementation (scikit-learn or GPy)
- [ ] Mixed kernel (Matern + Hamming)
- [ ] Theoretical prior: J_topo → E_floor (r=0.83)
- [ ] Acquisition function (Expected Improvement)

### Phase B4: Active Loop
- [ ] Main AL loop with logging
- [ ] Multi-fidelity training integration
- [ ] Early stopping / plateau detection
- [ ] Checkpointing and resume

### Phase B5: Validation
- [ ] Synthetic test (verify AL > random)
- [ ] CIFAR-10: AL vs random search vs Phase A baseline
- [ ] ResNet family validation (ResNet-34/50)
- [ ] Scaling to ImageNet (future)

---

## 8. Relationship to Phase A

Phase A results (87 runs, 9 architectures on CIFAR-10) serve as:
1. **Calibration data**: $J_\mathrm{topo} \to E_\mathrm{floor}$ regression (r=0.83, validated)
2. **Validation baseline**: Compare AL-discovered architectures against Phase A results
3. **Family-specific priors**: ThermoNet family (ideal) vs ResNet family (real gas)

**Phase A key findings (v5 corrected)**:
- $J_\mathrm{topo} \to E_\mathrm{floor}$ confirmed: r = 0.83 ✅
- β ∝ J was fitter artifact (r = 0.03 after correction) ❌
- α phase transition confirmed in RFF (Phase S0) ✅
- α regularized in ThermoNet (bounded 82–500) ✅
- E_floor determined by capacity (N) + optimization difficulty (J_topo)

**Phase B priorities**:
1. **Primary**: AL search using J_topo → E_floor as sole metric
2. Validate ResNet family line (ResNet-34/50)
3. Explore wider architecture space

---

## 9. Open Questions

1. **How many initial architectures (N_cal) are needed?** Empirical — start with 10.
2. **Is GP sufficient, or do we need Bayesian Neural Networks?** GP is simpler; try first.
3. **How many epochs for short training?** Needs calibration.建议 10–20 epochs.
4. **What acquisition function is best?** EI vs UCB vs Thompson sampling — compare empirically.
5. **How to handle variable-depth architectures in GP kernel?** Use padding or special categorical encoding.
6. **Can we use J_topo gradient (dJ_topo/d_architecture) for local search?** Theoretical but computationally expensive.
7. **Why does r(J, E) = 0.83 but not higher?** Residual variance comes from architecture family differences (width vs depth families have different E intercepts).

---

## 10. References

- ThermoRG Theory: `theory/THEORY.md` (v5)
- Phase A: `experiments/phase_a/` (CIFAR-10 validation, complete)
- Active Learning: Bayesian Optimization principles (Mockus 1978, Snoek 2012)
- NAS: enas, DARTS, Once-for-All — connection is that we replace differentiable relaxation with J_topo surrogate
