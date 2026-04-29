# ThermoRG v4 Experiment Codebase

Implementation of the ThermoRG v4 parallel lines verification experiment protocol.

## Project Structure

```
phase1_experiments/
├── main.py                    # Main execution script
├── requirements.txt           # Dependencies
├── README.md                  # This file
│
├── configs/                   # Experiment configurations
│   ├── phase_0.1.yaml          # EOS verification (BN vs LN)
│   ├── phase_0.2.yaml         # BN vs LN parallel lines core test
│   ├── phase_0.3.yaml         # GN + None baseline
│   └── phase_1.1.yaml         # Full TPU sweep
│
├── models/
│   └── convnet.py             # ConvNet L=5 architecture
│
├── data/
│   └── datasets.py            # CIFAR-10 data loading
│
├── experiments/
│   ├── train.py               # Training loop with measurements
│   └── __init__.py
│
├── utils/
│   └── measurements.py        # Core measurement functions
│       ├── measure_gamma()     # γ = (1/L)Σ|ln(σ_final/σ_init)|
│       ├── measure_lambda_max()# Power iteration for λ_max
│       ├── fit_beta()         # D-scaling law fit
│       ├── is_stationary()    # 85% monotonicity check
│       └── f_test_equal_slopes()
│
├── analysis/
│   ├── statistics.py          # Aggregation, regression, F-test
│   ├── figures.py             # Plot generation
│   └── __init__.py
│
├── scripts/
│   └── generate_report.py     # Markdown report generator
│
└── outputs/                   # Results directory (created at runtime)
    ├── figures/
    ├── tables/
    └── reports/
```

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Phase 0.1: EOS verification (6 hours GPU)
python main.py --phase 0.1

# Phase 0.2: BN vs LN parallel lines core test (12 hours GPU)
python main.py --phase 0.2

# Phase 0.3: GN + None baseline (6 hours GPU)
python main.py --phase 0.3

# Phase 1.1: Full TPU sweep (~22 hours TPU)
python main.py --phase 1.1

# Resume interrupted run
python main.py --phase 0.2 --resume

# Generate report from results
python scripts/generate_report.py \
    --phase-0.1 outputs/phase_0.1/results.json \
    --phase-0.2 outputs/phase_0.2/results.json \
    --output outputs/reports/report.md
```

## Core Measurement Functions

### γ (Representational Shift)
```python
gamma = measure_gamma(activations_init, activations_final, norm_type)
# γ = (1/L) * Σ |ln(σ_final / σ_init)|
# Uses normalized activations (no affine transform) for cross-norm consistency
```

### λ_max (Maximum Singular Value)
```python
lambda_max = measure_lambda_max_mean(model, device)
# Power iteration with ~20 iterations per layer
```

### β (D-scaling Exponent)
```python
beta, alpha, r_squared = fit_beta(losses, D_values)
# Fits ℒ = α·D^(-β) + E_floor
# Returns None if R² < 0.995
```

### Stationarity Detection
```python
is_stat, epoch = is_stationary(loss_history)
# Returns True if 20 consecutive epochs show 85% monotonicity
```

## Protocol Specifications

| Parameter | Value |
|-----------|-------|
| Architecture | ConvNet L=5, kernel=3×3, no skip, GELU |
| Dataset | CIFAR-10 |
| Stationary window | 20 epochs |
| Monotonicity threshold | 85% |
| D-scaling R² threshold | 0.995 |
| Power iteration steps | 20 |
| EOS tolerance | λ_max · η ∈ [1.5, 3.0] |

## Key Design Decisions

1. **γ measurement uses normalized activations (no affine)**: This ensures consistency across normalization types, as the affine parameters (γ, β) are learned second-order effects.

2. **λ_max measured at stationarity**: For Phase 0.2+, λ_max is measured as the mean over the stationary window, not just initialization.

3. **Stationarity = 85% monotonicity**: Loss can be monotonically decreasing OR increasing, just needs to be consistent.

4. **R² ≥ 0.995 for D-scaling**: Strict threshold ensures only high-quality β values are used in the parallel lines analysis.
