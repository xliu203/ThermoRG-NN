# Phase A Redesign: Architecture Validation Protocol

## Current Status
- Current Phase A code uses OLD formula (pre-v3 theory)
- No results saved yet — training incomplete
- Phase A redesign needed before running experiments

---

## Phase A Purpose
Validate that **J_topo is measurable, architecture-dependent, and correlates with scaling exponents** across diverse architectures.

---

## What Phase A Must Measure (v3 Theory)

### Core Metric: J_topo
```python
def compute_J_topo(model, input_batch):
    """
    J_topo = exp(-|Σ log η_l| / L)
    η_l = D_eff^(l) / D_eff^(l-1)
    D_eff = ||J_l||_F² / ||J_l||_2²  (stable rank)
    
    J_topo ∈ (0, 1]
    - J_topo = 1: perfect compression (ideal architecture)
    - J_topo → 0: severe compression bottleneck (poor architecture)
    """
    # For each layer, compute Jacobian w.r.t. input
    # Record D_eff^(l) and compute η_l = D_eff^(l)/D_eff^(l-1)
    # J_topo = exp(-|Σ log η_l| / L)
```

### Derived Metrics

**For each trained network:**
1. `J_topo` — from spectral analysis of trained network
2. `β_hat` — from fitting L vs D curve (D-scaling experiment)
3. `d_eff^Hess` — from Hessian eigenvalue spectrum
4. `sharpness` — η_lr × λ_max(H) during training
5. `final_accuracy` — test accuracy after training

### What Phase A Should Prove

| Prediction | Evidence | How to Measure |
|-----------|----------|----------------|
| **J_topo is computable** | J_topo ∈ (0,1] for all archs | Direct computation |
| **J_topo reflects architecture quality** | ThermoNet > ResNet > VGG in J_topo | Ranking |
| **β ∝ J_topo** | β_hat vs J_topo linear fit, R² > 0.8 | Regression |
| **d_eff^realized = d_task/J_topo** | Compare Hessian d_eff to J_topo | Correlation |
| **EoS: sharpness → 2** | Track η·λ_max during training | Time series |

---

## Redesigned Phase A Pipeline

### Step 1: Data Generation
```python
# Use CIFAR-10 (or synthetic manifold data)
# Fixed: d_task estimated from data manifold analysis
train_data = CIFAR10(root='./data', train=True, download=True)
test_data  = CIFAR10(root='./data', train=False)
```

### Step 2: Train Each Architecture
For each architecture (ThermoNet-5, ResNet-18, VGG-11, DenseNet-40, etc.):
```python
# Train to convergence (or fixed epochs)
model = build_architecture(name)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(T_max=200)

# Training loop with metrics recording
for epoch in range(200):
    train_loss = train_one_epoch(model, train_loader)
    test_loss, test_acc = evaluate(model, test_loader)
    
    # Record every N epochs:
    if epoch % 10 == 0:
        J_topo = compute_J_topo(model, batch)
        sharpness = compute_sharpness(model, batch)
        metrics['epoch'].append(epoch)
        metrics['J_topo'].append(J_topo)
        metrics['sharpness'].append(sharpness)
        metrics['test_acc'].append(test_acc)
        metrics['test_loss'].append(test_loss)
```

### Step 3: D-Scaling Experiment (for β)
For each architecture, train with varying dataset sizes:
```python
dataset_sizes = [1000, 2000, 5000, 10000, 20000, 50000]
for D in dataset_sizes:
    subset = random_subset(train_data, D)
    model = build_architecture(name)
    train model on subset
    final_loss = evaluate(model, test_data)
    # Fit: L(D) = E + B·D^(-β)
```

### Step 4: Analysis
```python
# For each architecture, compute:
results = {
    'name': arch_name,
    'J_topo': final_J_topo,
    'beta_hat': fit_beta(dataset_sizes, losses),
    'd_eff_Hess': count_hessian_eigenvalues(model),
    'final_acc': final_accuracy,
    'params_M': count_parameters(model) / 1e6,
}

# Create plots:
# 1. Bar chart: J_topo by architecture (ordering)
# 2. Scatter: β_hat vs J_topo (linear fit)
# 3. Scatter: final_acc vs J_topo (correlation)
# 4. Time series: sharpness during training (EoS verification)
```

---

## Architecture List (Same as Current)
```
ThermoNet-3, ThermoNet-5, ThermoNet-7, ThermoNet-9
ThermoBottleneck-3, ThermoBottleneck-5, ThermoBottleneck-7, ThermoBottleneck-9
ResNet-18, ResNet-34
VGG-11, VGG-16
DenseNet-40
RandomFix-3, RandomFix-5, RandomFix-7, RandomFix-9
```

---

## Key Changes from Current Phase A

| Aspect | Old (Current) | New (v3 Theory) |
|--------|--------------|-----------------|
| J_topo formula | J_topo = \|Σlog η_l\| | J_topo = exp(-\|Σlog η_l\|/L) |
| β formula | β = s/d_eff | β = s·J_topo/d_task |
| T_c | T_c = c·d_task/Tr(H) [wrong units!] | T_c = Tr(Σ)/(B·λ_max) |
| ψ function | ψ = T·(1-T/T_c)^γ · I(T<T_c) | ψ = (T/T_c)·exp(1-T/T_c) |
| EoS tracking | Not included | Track sharpness = η·λ_max during training |

---

## Output Format
```json
{
  "architectures": [
    {
      "name": "ThermoNet-5",
      "params_M": 1.23,
      "J_topo": 0.847,
      "beta_hat": 0.234,
      "d_eff_Hess": 12.5,
      "final_acc": 0.723,
      "sharpness_trajectory": [0.5, 1.2, 1.8, 2.1, 2.0, 1.9, 2.0]
    }
  ],
  "d_scaling": {
    "ThermoNet-5": {
      "dataset_sizes": [1000, 2000, 5000, 10000, 20000],
      "losses": [2.45, 1.89, 1.34, 0.98, 0.72],
      "beta_fit": 0.234,
      "R2": 0.94
    }
  }
}
```

---

## Next Action
1. Wait for DeepSeek simulation design (Phase S0 end-to-end)
2. Incorporate simulation insights into Phase A redesign
3. Rewrite Phase A pipeline with v3 formulas
4. Run on Kaggle with corrected code
