# ThermoRG-NN Experimental Data

## Data Files Overview

| File | Phase | Contents | Status |
|------|-------|---------|--------|
| `phase_a_summary.csv` | A | 15 configs, alpha/J_topo | ✅ Valid |
| `experiments/phase_s1/phase_s1_fast_results.json` | S1 | 6 configs, beta (bug period) | ⚠️ Use correct values below |
| `experiments/phase_b/critical_region_results.json` | B | 4 candidates, J_topo | ✅ Valid |
| `experiments/phase_b/stepwise_v3_summary.json` | B | 6 exps, beta evolution | ✅ Valid |
| `experiments/phase_b/training_phase_evolution.json` | B | Training dynamics | ✅ Valid |
| `phase_b_session2_results.json` | B | Session 2 calibration | ✅ Valid |

---

## Phase S1: Cooling Theory — Corrected Beta Values

**Source:** TPU runs (200 epochs, CIFAR-10, 4 D values per config)

| Config | norm | skip | γ | β (fitted) | R² | Regime |
|--------|------|------|---|------------|-----|--------|
| None_NoSkip | none | False | 3.39 | **1.117** | >0.995 | super-critical |
| None_Skip | none | True | 3.39 | **1.117** | >0.995 | super-critical |
| BN_NoSkip | batchnorm | False | 2.29 | **0.950** | >0.995 | super-critical |
| BN_Skip | batchnorm | True | 2.29 | **0.950** | >0.995 | super-critical |
| LN_NoSkip | layernorm | False | ~0.41 | **0.219** | 0.9997 | **sub-critical** |
| LN_Skip | layernorm | True | ~0.41 | **0.219** | 0.9997 | **sub-critical** |

**β(γ) formula:** β = 0.425·ln(γ/2.0) + 0.893, γ_c ≈ 2.0

---

## Phase B1: LN Validation (8 runs × 200 epochs)

**Source:** TPU run, depth=3, skip=False, widths=[32,48,64,96], 2 seeds each

| D | seed=42 loss | seed=123 loss | avg loss |
|----|-------------|--------------|---------|
| 32 | 1.0958 | 1.1056 | 1.1007 |
| 48 | 1.0212 | 1.0274 | 1.0243 |
| 64 | 0.9626 | 0.9757 | 0.9692 |
| 96 | 0.9035 | 0.9045 | 0.9040 |

**Fit:** β = 0.219, R² = 0.9997
**γ_LN ≈ 0.41** (sub-critical, < γ_c = 2.0)

---

## Phase B2: HBO vs Random — Round 1 (Negative Result)

**Source:** TPU run, 50 epochs, BatchNorm only

### L2 Top-5 (both arms)

**Random arm:**
| rank | val_loss | J_topo | width | depth | skip |
|------|----------|--------|-------|-------|------|
| 1 | 0.3858 | 0.7739 | 96 | 6 | False |
| 2 | 0.5014 | 0.8062 | 64 | 5 | False |
| 3 | 0.6047 | 0.7838 | 64 | 5 | True |
| 4 | 0.6268 | 0.7538 | 64 | 4 | False |
| 5 | 0.6937 | 0.8027 | 48 | 4 | False |

**HBO arm (J_topo HIGH globally):**
| rank | val_loss | J_topo | width | depth | skip |
|------|----------|--------|-------|-------|------|
| 1 | 0.6051 | 0.8627 | 32 | 6 | False |
| 2 | 0.6812 | 0.8774 | 24 | 6 | False |
| 3 | 0.7479 | 0.8455 | 32 | 5 | True |
| 4 | 0.7821 | 0.8727 | 24 | 6 | True |
| 5 | 0.8378 | 0.8701 | 24 | 5 | True |

**Result:** Random best=0.3858, HBO best=0.6051, Δ = -0.2193

---

## Confounding Analysis (Phase B2, n=10)

**Simple Spearman correlations:**
| Relationship | r | p |
|------------|---|---|
| J_topo → loss | +0.588 | 0.074 |
| Width → loss | -0.829 | 0.003 |
| Depth → loss | -0.143 | 0.694 |
| Width → J_topo | -0.922 | — |

**Partial correlations (controlling for Width):**
| Relationship | r | p | Interpretation |
|-------------|---|---|---------------|
| J_topo → loss \| Width | **-0.794** | **0.006** | Within same width: higher J_topo → lower loss |
| Width → loss \| J_topo | -0.891 | 0.0005 | Width dominates even when J_topo controlled |

**L1→L2 Spearman:** r = 1.000 (perfect correlation)

---

## Phase A: Architecture Survey

**Source:** CIFAR-10 training, various epochs (see individual configs)

| name | group | params (M) | alpha | J_topo | eta_product |
|------|-------|-----------|-------|--------|------------|
| ThermoNet-3 | G1 | 1.06 | 0.2310 | 1.3863 | 4.0000 |
| ThermoNet-5 | G1 | 2.13 | 0.0446 | 0.4463 | 1.5625 |
| ThermoNet-7 | G1 | 2.71 | 0.0372 | 0.4463 | 1.5625 |
| ThermoNet-9 | G1 | 1.31 | 0.0496 | 0.4463 | 1.5625 |
| ThermoBot-3 | G2 | 1.01 | 0.3157 | 2.5257 | 12.5000 |
| ThermoBot-5 | G2 | 1.43 | 0.0446 | 0.4463 | 1.5625 |
| ThermoBot-7 | G2 | 2.40 | 0.0876 | 1.1394 | 3.1250 |
| ThermoBot-9 | G2 | 2.81 | 0.0814 | 1.1394 | 3.1250 |
| ReLUFurnace-3 | G3 | 0.08 | 0.1116 | 0.4463 | 1.5625 |
| ReLUFurnace-5 | G3 | 0.15 | 0.0744 | 0.4463 | 1.5625 |
| ReLUFurnace-7 | G3 | 0.22 | 0.0558 | 0.4463 | 1.5625 |
| ReLUFurnace-9 | G3 | 0.30 | 0.0446 | 0.4463 | 1.5625 |
| ResNet-18-CIFAR | G4 | 11.17 | 0.1980 | 4.1589 | 64.0000 |
| VGG-11-CIFAR | G4 | 4.51 | 0.2971 | 2.0794 | 8.0000 |
| DenseNet-40-CIFAR | G4 | 0.49 | 0.3030 | 23.0259 | 0.0000 |

---

## J_topo(D) Scaling — CPU Validation

**Source:** Local CPU, 5-seed averaged, depth=[3,5], norm=batchnorm

### depth=3, batchnorm
| D | ⟨J_topo⟩ | std |
|----|-----------|-----|
| 16 | 0.8429 | 0.016 |
| 24 | 0.8126 | 0.016 |
| 32 | 0.7807 | 0.013 |
| 48 | 0.7351 | 0.015 |
| 64 | 0.6944 | 0.007 |
| 96 | 0.6472 | 0.009 |

**Log-log slope:** -0.150 (theory: -0.45/3 = -0.150) ✅

### depth=5, batchnorm
| D | ⟨J_topo⟩ | std |
|----|-----------|-----|
| 16 | 0.8684 | — |
| 24 | 0.8653 | — |
| 32 | 0.8442 | — |
| 48 | 0.8149 | — |
| 64 | 0.7768 | — |
| 96 | 0.7515 | — |

**Log-log slope:** -0.087 (theory: -0.45/5 = -0.090) ✅

---

## Phase B3: HBO_revised — Pending

**Design:** Width-first (W≥48) → top-30 by J_topo HIGH → L1 10ep → top-5 → L2 50ep
**Notebook:** `experiments/phase_b/notebooks/phase_b_hbo_revised.ipynb`
**Status:** Leo to run on Kaggle TPU next week

---

*Last updated: 2026-04-09*
