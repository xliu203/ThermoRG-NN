# ThermoRG v4 Kaggle Notebooks

This directory contains Jupyter notebooks for running the ThermoRG v4 experiments on Kaggle.

## Overview

The experiments are split into 3 phases:

| Phase | Notebook | Purpose | Duration |
|-------|----------|---------|----------|
| 0.1 | `phase_0_1_eos_verification.ipynb` | EOS condition verification | ~20 min |
| 0.2 | `phase_0_2_parallel_lines.ipynb` | BN vs LN parallel lines test | ~4 hrs |
| 1.1 | `phase_1_1_full_sweep.ipynb` | Full normalization sweep | ~8 hrs |

## Phase 0.1: EOS Verification

**Notebook:** `phase_0_1_eos_verification.ipynb`

**Protocol:**
- Model: ConvNet L=5, D=256
- Normalization types: BatchNorm, LayerNorm
- Learning rates: 8 values [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
- Seeds: [42, 43, 44]
- Epochs: 20

**Purpose:** Verify the EOS condition λ_max · η_crit ≈ 2

**What to expect:**
- Run time: ~20 minutes on GPU
- Scatter plot of λ_max · η vs η showing convergence boundary
- Target ratio is 2.0 with tolerance [1.5, 3.0]

## Phase 0.2: BN vs LN Parallel Lines Core Test

**Notebook:** `phase_0_2_parallel_lines.ipynb`

**Protocol:**
- Model: ConvNet L=5
- Normalization types: BatchNorm, LayerNorm
- D values: [64, 256, 1024]
- Learning rates: 6 values [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
- Seeds: [42, 43, 44]
- Epochs: 200

**Purpose:** Test the core ThermoRG hypothesis that BN and LN should have the same slope β(ln(γ)) - the parallel lines hypothesis

**What to expect:**
- Run time: ~4 hours on GPU
- β vs ln(γ) plot with parallel lines for BN and LN
- F-test for equal slopes

## Phase 1.1: Full TPU Sweep

**Notebook:** `phase_1_1_full_sweep.ipynb`

**Protocol:**
- Model: ConvNet L=5
- Normalization types: BatchNorm, LayerNorm, GroupNorm, None
- D values: [32, 64, 128, 256, 512, 1024]
- Learning rates: 8 values [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
- Seeds: [42, 43, 44, 45, 46]
- Epochs: 40

**Purpose:** Complete sweep over all normalization types for comprehensive β(ln(γ)) analysis

**What to expect:**
- Run time: ~8 hours on GPU (or use TPU for faster training)
- Full β vs ln(γ) plot for all 4 normalization types
- D-scaling collapse visualization
- F-test for equal slopes across all norm types

## Running on Kaggle

1. **Upload notebooks** to Kaggle
2. **Add data source**: CIFAR-10 dataset (Kaggle has this built-in)
3. **Enable GPU/TPU**: Kaggle Notebook settings → Accelerator → GPU T4 or TPU
4. **Run All** cells in each notebook

## Checkpoint/Resume

All notebooks support resume from partial results:
- Results are saved incrementally to `/kaggle/working/phase_X/`
- If you restart the notebook, it will detect existing results and skip completed runs
- Each run's result is saved as `result_{run_id}.json`

## Output Files

Each phase saves:
- `phase_X_results.json` - Final aggregated results
- `result_*.json` - Individual run results
- `*.png` - Generated figures

## Hardware Recommendations

| Phase | GPU | TPU | Estimated Time |
|-------|-----|-----|----------------|
| 0.1 | T4 | v2 | 15-20 min |
| 0.2 | T4 | v2 | 3-4 hrs |
| 1.1 | T4 | v2 | 6-8 hrs |
| 1.1 | A100 | - | 1-2 hrs |

For Phase 1.1, using a TPU or A100 is strongly recommended due to the large number of runs (4 × 6 × 8 × 5 = 960 runs).
