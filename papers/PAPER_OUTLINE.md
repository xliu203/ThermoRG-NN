# Paper Preparation — ThermoRG-NN

## Paper Target
**"ThermoRG: Thermodynamic Theory of Neural Architecture Scaling"**
Phenomenological theory with strong empirical grounding.

---

## Proposed Structure

### 1. Introduction
- Neural architecture scaling: empirical scaling laws (Chinchilla, etc.)
- Missing: first-principles understanding of why some architectures scale better
- Contribution: ThermoRG framework — thermodynamic + RG principles applied to architecture

### 2. Background
- D-scaling law: E(D) = α·D^(-β) + E_floor
- Edge of Stability (EOS) as critical point
- Normalization and training dynamics

### 3. Theory: ThermoRG Framework

#### 3.1 J_topo: Topological Quality Metric
- Definition: J_topo = exp(-(1/L)∑|log η_l|)
- η_l = D_eff^l/D_eff^(l-1) × ζ_l
- D_eff = ||W||_F² / λ_max²
- Stride correction
- Skip connection augmentation
- **Figure 1**: J_topo vs D_scaling (width family, depth family, cross-architecture)
- **Figure 2**: J_topo computation illustration

#### 3.2 H1: β ∝ J_topo
- Within-family correlation: r ≈ 0.97
- Physical interpretation: J_topo measures information flow uniformity
- **Figure 3**: β vs J_topo within families

#### 3.3 J_topo(D) Scaling
- J_topo(D) ∝ D^(-α_GELU/L), α_GELU ≈ 0.45
- GELU nonlinearity causes sublinear D_eff scaling
- BN/LN excluded from J_topo (η=1)
- **Figure 4**: J_topo vs D (log-log), different depths

#### 3.4 Cooling Theory: β(γ)
- γ = variance fluctuation
- β(γ) = 0.425·ln(γ/γ_c) + 0.893, γ_c ≈ 2.0
- EOS critical point explains the functional form
- **Figure 5**: β(γ) cooling curve with data points (LN, BN, None)

#### 3.5 E_floor Decomposition
- E_floor = capacity + optimization_difficulty(J_topo)
- Two independent channels: width (capacity) vs topology (optimization)
- **Figure 6**: E_floor vs J_topo (partial correlation diagram)

#### 3.6 Stride-2 as RG Blocking
- Stride-2 as coarse-graining operation
- φ correction factor
- Task-dependent (CIFAR-10 vs ImageNet)

### 4. Experiments

#### 4.1 Setup
- ThermoNet architecture
- CIFAR-10 (+ ImageNet subset for stride-2)
- TPU training

#### 4.2 D-Scaling Law Validation
- Power law fits across architecture families
- **Figure 7**: L(D) for multiple families

#### 4.3 J_topo Zero-Cost Prediction
- J_topo computed from initialization
- Correlates with final performance
- **Figure 8**: J_topo vs final val loss across families

#### 4.4 Cooling Theory
- Phase S1: BN, None (200 epochs)
- Phase B1: LN (200 epochs)
- β(γ) validated for γ ∈ [0.41, 3.39]
- **Figure 9**: LN scaling law fit (β=0.219, R²=0.9997)

#### 4.5 Architecture Selection: HBO vs Random

**Round 1 (Negative Result)**
- HBO selected HIGH J_topo globally → lost
- Random won (best: 0.386 vs 0.605)
- **Figure 10**: Random vs HBO L2 results
- **Figure 11**: Architecture distribution (HBO picked narrow-deep)

**Confounding Analysis**
- Simple vs partial correlation
- Simpson's paradox resolved
- Width channel dominates; J_topo channel secondary
- **Figure 12**: Confounding diagram (3-panel)
- **Figure 13**: Partial correlation bars

**Round 2 (HBO_revised)**
- Width-first (W≥48) + J_topo HIGH within groups
- Expected to win
- *(pending)*

#### 4.6 J_topo(D) Scaling Validation
- CPU validation with 5-seed averaging
- depth=3: slope=-0.150 ✅
- depth=5: slope=-0.087 ✅
- **Figure 14**: J_topo(D) log-log with error bars

### 5. Discussion
- Thermodynamic interpretation: information flow as RG fixed point
- J_topo as order parameter
- Capacity vs optimization duality
- Why J_topo alone is insufficient for screening
- Practical implications for architecture search

### 6. Conclusion
- ThermoRG: a complete phenomenological theory
- Validated across scaling, topology, and cooling
- Negative result is informative (not a failure)
- Future directions

---

## Figures Checklist

| Fig | Title | Data Source | Status |
|-----|-------|-------------|--------|
| 1 | J_topo universality across families | Phase A data | ✅ |
| 2 | J_topo computation illustration | Theory | ✅ |
| 3 | β vs J_topo within families | Phase A data | ✅ |
| 4 | J_topo(D) log-log | CPU validation | ✅ |
| 5 | β(γ) cooling curve | Phase S1 + B1 | ✅ |
| 6 | E_floor decomposition | Theory | ✅ |
| 7 | L(D) scaling across families | Phase A | ✅ |
| 8 | J_topo vs final loss | Phase A + B2 | ✅ |
| 9 | LN scaling law | Phase B1 | ✅ |
| 10 | Round 1: Random vs HBO | Phase B2 | ✅ |
| 11 | Architecture distribution | Phase B2 | ✅ |
| 12 | Confounding: 3-panel | Phase B2 | ✅ |
| 13 | Partial correlation bars | Phase B2 | ✅ |
| 14 | J_topo(D) with error bars | CPU validation | ✅ |
| 15 | Round 2 results (HBO_revised) | Pending | ⏳ |

---

## Key Claims (Paper-Ready)

1. **J_topo is a universal zero-cost architecture quality metric** (r = 0.97 within families)
2. **H1: β ∝ J_topo** — scaling exponent proportional to topological quality
3. **β(γ) cooling theory** — EOS-inspired formula validated across [0.41, 3.39]
4. **J_topo(D) ∝ D^(-0.45/L)** — GELU nonlinearity causes sublinear width dependence
5. **Two-channel E_floor** — capacity (width) and optimization difficulty (J_topo) are independent
6. **Negative result is informative** — HBO failure reveals width as dominant factor
7. **Practical screener** — width-first + J_topo within groups (pending Round 2)

---

## Sections That Need Writing

- [ ] Introduction (1 page)
- [ ] Theory section (3-4 pages)
- [ ] Methods (1 page)
- [ ] Results (3-4 pages)
- [ ] Discussion (1-2 pages)
- [ ] Conclusion (0.5 page)

## Pending Items Before Submission

1. ✅ **Round 2 results** (HBO_revised) — HBO WON (0.703 vs 0.781)
2. [ ] **Figure generation scripts** — all figures from data
3. [ ] **Figure captions** — detailed, self-contained
4. [ ] **Related work** — positioning against Chinchilla, NAS literature
5. [ ] **Appendix** — supplementary proofs, additional experiments

---

## Next Steps

1. ✅ HBO_revised ran and WON
2. Generate all figures from existing data
3. Write first draft of theory section (DONE: paper_draft_v1.tex)
4. Write results section (can parallelize)

## Critical Fix (2026-04-11)

**J_topo definition was inconsistent between Phase A and code:**
- Phase A used: J_topo = ln(η_product) (unbounded, ~23 for DenseNet)
- Code uses: J_topo = exp(-mean|log η_l|) (bounded in (0,1])

**Fixed all values to code definition:**
- ResNet-18: 4.16 → 0.678
- VGG-11: 2.08 → 0.399
- DenseNet-40: 23.03 → 0.888
- ThermoNet family: all recomputed

**Ordering now makes sense:** DenseNet (0.888) > ResNet (0.678) > ThermoNet-6 (0.51-0.60) > VGG (0.40)

*Last updated: 2026-04-11*
