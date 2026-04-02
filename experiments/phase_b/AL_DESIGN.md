# ThermoRG Active Learning Loop — Design Document

> **Status**: Design draft (2026-04-02)  
> **Author**: ThermoRG Team (theory + DeepSeek Reasoner)  
> **Purpose**: Complete algorithmic design for automated architecture search via J_topo-driven active learning

---

## 1. Motivation

The ThermoRG framework establishes:

$$L(D) = \alpha \cdot D^{-\beta} + E$$

where:
- $\beta \propto J_{\mathrm{topo}}$ (scaling exponent) — **fully identifiable in D ≥ 2000**
- $\alpha$ — represents initial complexity penalty, shows **critical divergence** near $J_c \approx 0.35$ (topological phase transition). Statistically **unidentifiable in D ≥ 2000** (not a physical bound — the asymptotic regime makes α's contribution too small to measure).
- $J_{\mathrm{topo}} = \exp\bigl(-\frac{1}{L}\sum_l |\log\eta_l|\bigr)$ is **computable at initialization** (zero training cost)

This means we can **predict architecture quality before training**, enabling efficient active learning.

**Key insight for Phase B**: Since α is unidentifiable in the practical D range, **β is the sole optimization target**. The relationship β ∝ J_topo is confirmed within families (r = 0.97).

---

## 2. Goal

Design an automated loop that:
1. Explores architecture space under resource constraints (params, FLOPs, latency)
2. Uses J_topo as a cheap surrogate to rank architectures
3. Trains only the most promising candidates
4. Learns from training feedback to improve predictions

**Efficiency target**: Find architectures within 5% of optimal, at <20% of the training cost of random search.

---

## 3. Core Hypothesis (Phase A validated for β)

**H1 (confirmed)**: Within architecture families, β ∝ J_topo:
- Width family: r = 0.976 ✅
- Depth family: r = 0.973 ✅

**H2 (phase transition)**: α shows critical divergence near J_c ≈ 0.35, but is statistically unidentifiable in D ≥ 2000. This is a measurement limitation, not a theoretical failure.

**Architecture family effects**:
- **ThermoNet** (ideal topology): Follow ThermoRG scaling law precisely (R² > 0.97 within families)
- **ResNet** (real topology): Subject to "van der Waals correction" from stride-2 downsampling. Forms a separate family with lower β intercept (β ≈ 0.089·J + 0.384 for ThermoNet; ResNet family is lower).

**Core hypothesis for Phase B**: J_topo computed from **random initialization weights** is a strong predictor of trained β. This enables zero-training-cost architecture search.

---

## 4. Algorithm

### 4.1 Overview

```
ThermoRG-AL(D_dataset, budget, constraints)
│
├─ PHASE 1: Initialization (cold start)
│   ├─ Sample N_init architectures (Latin Hypercube)
│   ├─ Compute J_topo at initialization for all
│   ├─ Train N_cal subset for N_cal_epochs (establish J_topo→loss mapping)
│   └─ Fit GP surrogate: arch_features → (J_topo, loss)
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
2. **Loss** (observed only after training)

**Kernel**: Matern-5/2 for continuous features + Hamming distance for categorical features.

**Training data structure**:
- `(arch_encoding, J_topo, loss_at_D)` for each evaluated architecture
- J_topo is available for all architectures (trained or not)
- Loss is available only after short training

**Warm-start**: The relationship $\beta \propto J_{\mathrm{topo}}$ and $\alpha \propto J_{\mathrm{topo}}^2$ provides a **theoretical prior** for the GP. Use this to initialize the mean function:

$$f_{\mathrm{prior}}(J) = \alpha_0 \cdot J^2 \cdot D^{-\beta_0 \cdot J}$$

where $\alpha_0, \beta_0$ are calibrated from Phase A data.

### 4.5 Acquisition Function

**Objective**: Minimize expected loss $L(D_{\mathrm{target}})$ at the target dataset size.

**Expected Improvement (EI)**:

$$\mathrm{EI}(x) = \mathbb{E}\left[\max(0, L_{\mathrm{best}} - L(x))\right]$$

**Components**:
1. **Predicted loss** $\hat{L}(x)$ from GP, using D-scaling law with predicted $\hat{\beta}(x), \hat{\alpha}(x)$
2. **Uncertainty** $\sigma(x)$ from GP posterior
3. **Constraint penalty** $P_{\mathrm{constraint}}(x)$: 0 if satisfied, $-\infty$ if violated

**Final score**:
$$\mathrm{score}(x) = -\hat{L}(x) + \lambda \cdot \sigma(x) - \gamma \cdot P_{\mathrm{constraint}}(x)$$

where $\lambda$ is the exploration coefficient (e.g., $\lambda = 0.1$).

**Theoretical grounding**:
- $\hat{L}(x) = \hat{\alpha} \cdot D^{-\hat{\beta}} + E$ with $\hat{\beta} \propto J_{\mathrm{topo}}(x)$
- $\sigma(x)$ encodes exploration value: uncertain architectures that might be good
- Maximizing score = minimizing loss while balancing explore/exploit

### 4.6 Training & Feedback

**Multi-fidelity evaluation**:

| Fidelity | Epochs | Cost | Use case |
|----------|--------|------|----------|
| RFF-proxy | ~50 | ~1 GPU-hour | Calibrate J_topo→loss mapping |
| Low | 5–10 | ~0.5 GPU-hour | Quick filter in active loop |
| Medium | 20–30 | ~2 GPU-hour | Refine top candidates |
| Full | 50–200 | ~10 GPU-hour | Final validation |

**Feedback to GP**:
- After training at fidelity $f$, observe loss $L_f$
- Convert to equivalent full-fidelity estimate using scaling law:
  $$L_{\mathrm{full}} \approx L_f \cdot \left(\frac{D_{\mathrm{full}}}{D_f}\right)^{\hat{\beta}}$$
- Update GP with $(x, J_{\mathrm{topo}}(x), L_{\mathrm{full-est}})$

### 4.7 Constraint Handling

**Hard constraints** (filter candidates):
- Max parameters: $N_{\mathrm{params}} \leq N_{\max}$
- Max FLOPs: $\mathrm{FLOPs} \leq \mathrm{FLOPs}_{\max}$
- Max latency: $\mathrm{latency} \leq \mathrm{latency}_{\max}$

**Soft constraints** (penalize in score):
- Prefer smaller models when J_topo is similar
- Regularization: $P_{\mathrm{size}}(x) = \log N_{\mathrm{params}}(x) / \log N_{\max}$

**Constraint-aware candidate generation**:
- Sample architectures conditioned on satisfying hard constraints
- Use inverse-weighting to bias toward constraint-satisfying regions

### 4.8 Initialization Strategy

**Goal**: Establish J_topo→loss mapping with minimal training.

**Recommended**:
- $N_{\mathrm{init}} = 20$ architectures sampled via Latin Hypercube
- Compute J_topo for all 20
- Train top-5 and bottom-5 (diverse J_topo range) for $N_{\mathrm{cal}} = 20$ epochs
- This gives a regression J_topo → loss with $N=10$ data points

**Alternative (if Phase A shows strong correlation)**:
- Just train $N_{\mathrm{cal}} = 5$ architectures to validate the theoretical $\beta \propto J_{\mathrm{topo}}$ prior

### 4.9 Termination Conditions

Stop when ANY of:
1. **Budget exhausted**: $B_{\mathrm{remaining}} < \mathrm{cost}(N_{\mathrm{short\_train}})$
2. **Plateau**: No improvement in top architecture loss over $N_{\mathrm{plateau}} = 5$ consecutive selections
3. **Uncertainty too high**: Average GP uncertainty $\bar{\sigma} > \sigma_{\max}$ (means model is unreliable)
4. **Theoretical bound**: Achieved $J_{\mathrm{topo}}$ within $\epsilon$ of theoretical maximum

---

## 5. Theoretical Predictions Guiding the Loop

### 5.1 Loss Prediction from J_topo

Given J_topo at initialization:

$$\hat{\beta} = \beta_0 \cdot \frac{J_{\mathrm{topo}}}{\bar{J}_{\mathrm{topo}}}$$

$$\hat{\alpha} = \alpha_0 \cdot \left(\frac{J_{\mathrm{topo}}}{\bar{J}_{\mathrm{topo}}}\right)^2$$

where $\beta_0, \alpha_0$ are calibrated from Phase A data.

Then:
$$\hat{L}(D) = \hat{\alpha} \cdot D^{-\hat{\beta}} + E$$

### 5.2 Optimal J_topo

Theoretical prediction: $J_{\mathrm{topo}} \to 1$ is optimal (stable information flow).

**Practical target**: $J_{\mathrm{topo}} > 0.8$ likely indicates good architecture.

### 5.3 Width Profile from J_topo

From $J_{\mathrm{topo}}$ constraint, we can derive the width profile:

$$\eta_l = \frac{D_{\mathrm{eff}}^{(l)}}{D_{\mathrm{eff}}^{(l-1)}} \approx \mathrm{const}$$

For $J_{\mathrm{topo}} \approx 1$, we need $\eta_l \approx 1$ for all layers, meaning:

$$\|W_l\|_F^2 / \lambda_{\max}(W_l) \approx \mathrm{const} \cdot \lambda_{\mathrm{prev}}$$

This provides a **targeted width profile** for architecture generation.

---

## 6. Failure Modes & Mitigations

| Failure Mode | Detection | Mitigation |
|-------------|-----------|------------|
| J_topo not predictive of loss | GP residuals large after N_cal | Increase N_cal; use random search baseline comparison |
| GP poor on categorical arch space | Cross-val RMSE high | Use hierarchical GPs; separate models for depth vs width |
| Constraints too restrictive | Few candidates pass filter | Relax hard constraints; use soft constraints instead |
| Training feedback noisy | Loss variance high across seeds | Use more seeds per evaluation; fit scaling law instead of point loss |
| J_topo changes significantly after training | Init J_topo vs final J_topo low correlation | Use early-stopping J_topo instead of init J_topo |
| Catastrophic forgetting in GP | New data contradicts old | Use online GP or weight by recency |

---

## 7. Implementation Roadmap

### Phase B1: Critical Gap Experiments (NOW)
- [ ] CPU experiments: architectures at J_c ≈ 0.40 (fill critical region)
- [ ] Compute J_topo for candidate architectures
- [ ] Validate alpha phase transition near J_c

### Phase B2: Core Infrastructure
- [ ] Architecture encoding class (`ArchSpace`, `Architecture`)
- [ ] J_topo computation from random init weights (PI-20, 23× speedup)
- [ ] Constraint calculators (params, FLOPs, latency)
- [ ] Latin Hypercube sampler

### Phase B3: Surrogate Model
- [ ] GP implementation (scikit-learn or GPy)
- [ ] Mixed kernel (Matern + Hamming)
- [ ] Theoretical prior: β ∝ J_topo (family-specific for ResNet)
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
1. **Calibration data**: β_0 for the theoretical prior (β ∝ J_topo, validated within families)
2. **Validation baseline**: Compare AL-discovered architectures against Phase A results
3. **Family-specific priors**: ThermoNet family (ideal) vs ResNet family (real gas)

**Phase A key findings**:
- β ∝ J_topo confirmed within families (r = 0.97)
- α phase transition near J_c ≈ 0.35 confirmed in Phase S0
- α unidentifiable in D ≥ 2000 (statistical, not physical)
- E floor determined by capacity (N) + optimization difficulty (J_topo)

**Phase B priorities**:
1. Fill J_c ≈ 0.40 gap (critical region for alpha validation)
2. Validate ResNet family line (ResNet-34/50)
3. ThermoNet AL search using J_topo → β as sole metric

---

## 9. Open Questions

1. **How many initial architectures (N_cal) are needed?** Empirical — start with 10.
2. **Is GP sufficient, or do we need Bayesian Neural Networks?** GP is simpler; try first.
3. **How many epochs for short training?** Needs calibration.建议 10–20 epochs.
4. **What acquisition function is best?** EI vs UCB vs Thompson sampling — compare empirically.
5. **How to handle variable-depth architectures in GP kernel?** Use padding or special categorical encoding.
6. **Can we use J_topo gradient (dJ_topo/d_architecture) for local search?** Theoretical but computationally expensive.

---

## 10. References

- ThermoRG Theory: `theory/THEORY.md` (v3)
- Phase A: `experiments/phase_a/` (CIFAR-10 validation, in progress)
- Active Learning: Bayesian Optimization principles (Mockus 1978, Snoek 2012)
- NAS:enas, DARTS, Once-for-All — connection is that we replace differentiable relaxation with J_topo surrogate
