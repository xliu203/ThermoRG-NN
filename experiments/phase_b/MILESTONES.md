# Phase B Milestones

## Overview
**Goal**: Validate SU-HBO algorithm for automated neural architecture design using ThermoRG theory.

---

## Milestone 1: Session 2 Calibration ✅ Design
**Estimated**: 2026-04-06
**Status**: Design ready

### Objectives
1. Calibrate utility function parameter λ
2. Validate action library effects (ΔJ, Δγ)
3. Verify utility function ranking matches early-loss ranking

### Experimental Design
```
Architectures: 8 diverse configs
  - width: {32, 64}
  - depth: {3, 5}
  - skip: {True, False}
  - norm: {none, bn}
  → 2×2×2×2 = 16 possible, select 8

Training: 5 epochs × 10% CIFAR-10
Budget: ~1 GPU hour (T4)
```

### Metrics to Collect
- [ ] J_topo for each architecture (PI-20)
- [ ] γ after 5 epochs
- [ ] β estimated from loss curve
- [ ] L1 loss (5-epoch)
- [ ] Ranking by utility vs ranking by L1 loss

### Expected Outcomes
- [ ] λ calibrated to best match ranking
- [ ] Action effects (ΔJ, Δγ) validated
- [ ] If mismatch > 20%, identify cause

---

## Milestone 2: SU-HBO Code Framework ✅
**Estimated**: 2026-04-06
**Status**: Design complete (SU_HBO_ALGORITHM.md)

### Deliverables
- [ ] `thermorg_hbo/su_hbo.py`: Main SU-HBO class
- [ ] `thermorg_hbo/gp_surrogate.py`: GP model with custom mean
- [ ] `thermorg_hbo/action_library.py`: Action definitions + effects
- [ ] `thermorg_hbo/plateau_detector.py`: Plateau detection
- [ ] `thermorg_hbo/acquisition.py`: Expected Improvement

### Implementation Notes
- Use BoTorch or GPy for surrogate
- Custom mean function: U(A) = -E_floor(J,γ) + λ·β(γ)
- Multi-fidelity support

---

## Milestone 3: Session 2 Execution 🔴 IN PROGRESS
**Estimated**: ~1 GPU hour on Kaggle T4
**Status**: Pending

### Run Order
1. Execute calibration experiment
2. Fit λ parameter
3. Update action library
4. Validate ranking match

### Success Criteria
- Utility ranking correlates with L1 loss ranking (r > 0.7)
- Action effects match expected directions

---

## Milestone 4: Full SU-HBO Validation 🔴 IN PROGRESS
**Estimated**: ~8 GPU hours
**Status**: Pending

### Experiment
```
Baseline A: Random architecture search (N=20 archs, 200 epochs each)
Baseline B: Discrete HBO (N=20 archs, multi-fidelity)
OURS: SU-HBO (N=20, multi-fidelity + stepwise + utility)

Total: ~30 GPU hours
```

### Metrics
- [ ] Final validation loss
- [ ] Sample efficiency (GPU hours to reach target loss)
- [ ] Correlation: predicted U vs actual performance
- [ ] Architecture complexity vs performance

### Success Criteria
- [ ] SU-HBO achieves comparable/better final loss than baselines
- [ ] SU-HBO uses <50% of GPU hours vs brute-force
- [ ] Stepwise modifications are interpretable

---

## Milestone 5: Analysis & Paper 🔴 IN PROGRESS
**Estimated**: 2026-04-07+
**Status**: Pending

### Tasks
- [ ] Analyze SU-HBO decisions (which actions were taken)
- [ ] Ablation: SU-HBO vs HBO-only vs utility-only
- [ ] Update THEORY.md with Phase B findings
- [ ] Update paper draft

### Paper Structure
```
6. Phase B: Automated Architecture Design
   6.1 SU-HBO Algorithm
   6.2 Calibration (Session 2)
   6.3 Experimental Results
   6.4 Comparison to Baselines
```

---

## Resource Estimate

| Phase | GPU Hours | Notes |
|-------|-----------|-------|
| Session 2 (Calibration) | ~1 | 8 archs × 5 epochs |
| Session 3 (Full HBO) | ~8 | 20 archs × mixed fidelities |
| Ablation studies | ~4 | HBO-only, utility-only |
| **Total Phase B** | **~13** | Within weekly quota |

---

## Risks & Mitigations

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| λ calibration fails | Low | Use simulation values (λ=10) |
| Action effects wrong | Medium | Update library after Session 2 |
| GP overfits | Medium | Use multi-fidelity to regularize |
| Plateau detection triggers too early | Low | Tune ε_β, ε_γ thresholds |

---

## Next Action
**Immediate**: Start Session 2 calibration on Kaggle

