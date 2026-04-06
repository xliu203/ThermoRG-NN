# Phase B Milestones

## Overview
**Goal**: Validate SU-HBO algorithm for automated neural architecture design using ThermoRG theory.

---

## Key Design Principle: Avoid Overfitting Calibration

**Risk**: If we calibrate λ and action effects on the SAME data used for main search, we overfit.

**Solution**: Calibration validates the FORMULA, not fit to data.

### What We Calibrate vs What is Theory

| Component | Source | Can Overfit? |
|-----------|--------|--------------|
| **λ** | Theory (trade-off weight) | NO - from β and E_floor units |
| **Action ΔJ, Δγ** | Empirical (Phase A) | YES - need held-out validation |
| **k, B** | Phase A fit | NO - physical constants |

### Session 2 Goal

**NOT to fit λ** — λ ≈ 10 is from dimensional analysis.

**Goal**: Validate that the FORMULA works:
- Utility ranking correlates with actual L1 loss ranking
- Action effects (ΔJ, Δγ) have correct directions
- Formula predictions are reasonable, not perfectly calibrated

---

## Session 2: Calibration (5% Data) - RUN TWICE

**Data Split**:
```
CIFAR-10: 50,000 images
├── 5% calibration Run A (seed=42, 2,500 images)
├── 5% calibration Run B (seed=123, 2,500 images)
└── 90% main search (45,000 images) → Session 3
```

**Why twice?**
- Validate robustness across random subsets
- If both runs pass, formula is robust
- If one fails, investigate why

**Epochs Decision**:
- **5 epochs** for ranking (sufficient per DeepSeek analysis)
- 5 epochs gives stable γ (per-batch) and usable β (noisy but preserves relative order)
- **If Spearman r < 0.5**: increase to 10-20 epochs and re-validate

**What we measure vs what we need**:
| Parameter | 5 epochs enough? | Use case |
|----------|----------------|----------|
| γ (variance) | ✅ Yes | Ranking |
| β (learning speed) | ⚠️ Noisy but OK | Ranking only |
| L1 loss | ✅ Yes | Ranking |
| E_floor absolute | ❌ No | Needs 20-50 epochs |

### Why 5%?

- Sufficient to estimate β, γ from 5 epochs
- Small enough to be "held-out" from main search
- Large enough to get stable estimates

### What We Validate

1. **Ranking Validation**:
   - Train 8 diverse architectures for 5 epochs on calibration set
   - Rank by utility U = -E_floor + λ·β
   - Rank by actual L1 loss
   - Check: correlation r > 0.7?

2. **Action Effect Validation**:
   - Verify ΔJ, Δγ directions are correct
   - e.g., add_BN → γ decreases? (yes, expected)
   - e.g., add_skip → J increases? (yes, expected)

3. **Formula Sanity Check**:
   - E_floor values are in reasonable range (0.1 - 1.0)
   - β values are in reasonable range (0.1 - 0.5)
   - Stability margin γ < 1/(B·J) holds

### Success Criteria

| Check | Threshold | Pass/Fail |
|-------|-----------|-----------|
| Utility vs L1 ranking correlation | r > 0.7 | TBD |
| add_BN reduces γ | Δγ < 0 | TBD |
| add_skip increases J | ΔJ > 0 | TBD |
| E_floor range | 0.1 < E < 1.0 | TBD |

---

## Session 3: Full SU-HBO (95% Data)

**Data**: 95% of CIFAR-10 (47,500 images)

**Design**:
```
SU-HBO (N=20 architectures, multi-fidelity)
vs
Random Search (N=20, full training)
vs  
Discrete HBO (N=20, multi-fidelity)
```

### Success Criteria

| Metric | Threshold | Notes |
|--------|-----------|-------|
| SU-HBO final loss | Comparable to best baseline | Within 5% |
| Sample efficiency | <50% of Random Search GPU hours | Key advantage |
| Action interpretability | >80% actions make sense | Check logs |

---

## Session 4: Ablation Studies (~4 GPU hours)

### Ablations

1. **λ sweep**: λ = {1, 5, 10, 50, 100}
   - Does utility ranking change significantly?
   - Is λ robust?

2. **Action effects**: 
   - Calibrated vs default effects
   - Does calibration improve results?

3. **Utility vs greedy**:
   - SU-HBO vs greedy (always accept best candidate)
   - Is GP acquisition worth it?

---

## Resource Estimate

| Phase | GPU Hours | Data |
|-------|-----------|------|
| Session 2 Run A (Calibration) | ~10 min | 5% CIFAR-10 |
| Session 2 Run B (Calibration) | ~10 min | 5% CIFAR-10 (diff seed) |
| Session 3 (Full SU-HBO) | ~8 | 90% CIFAR-10 |
| Session 4 (Ablation) | ~4 | 90% CIFAR-10 |
| **Total Phase B** | **~9** | Within weekly quota |

---

## Risks & Mitigations

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Calibration overfits to 5% | Low | Use multiple random 5% subsamples |
| λ wrong | Medium | λ from theory, not fit; test λ sweep in ablation |
| Action effects wrong | Medium | Update after Session 2 if directions wrong |
| GP overfits | Medium | Multi-fidelity + regularization |

---

## Next Actions

1. **Session 2 Notebook**: Create calibration experiment on Kaggle
2. **Run Session 2**: ~1 hour on T4
3. **Analyze**: Check ranking correlation
4. **Session 3**: If Session 2 passes, proceed to full SU-HBO

---

## Updated: 2026-04-06

## Status

- [x] SU-HBO Algorithm designed
- [x] SU-HBO Package implemented (tests pass)
- [ ] Session 2: Calibration (PENDING)
- [ ] Session 3: Full SU-HBO (PENDING)
- [ ] Session 4: Ablation (PENDING)
