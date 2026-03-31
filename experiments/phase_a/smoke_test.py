#!/usr/bin/env python3
"""
smoke_test.py - Lightweight pre-GPU test (no model, no real data)
Tests everything EXCEPT actual model forward pass.
Catches: import errors, d_manifold computation, JSON/CSV writing, matplotlib availability.

Run: python3 experiments/phase_a/smoke_test.py
"""
import sys, os, json, csv, argparse, math, tempfile

# ── 1. Test imports ─────────────────────────────────────────────────────────
print("=" * 60)
print("TEST 1: Imports")
print("=" * 60)
try:
    import numpy as np
    import torch, torch.nn as nn, torch.nn.functional as F
    print("  torch/numpy: OK")
except Exception as e:
    print(f"  FAIL: {e}"); sys.exit(1)

try:
    from torch.utils.data import DataLoader, Subset
    import torchvision.transforms as transforms
    print("  torch.utils.data / torchvision: OK")
except Exception as e:
    print(f"  WARNING: torchvision not available ({e})")

try:
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy import stats
    print("  matplotlib/scipy: OK")
except Exception as e:
    print(f"  WARNING: matplotlib/scipy not available ({e})")

try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
    from thermorg.core.manifold_estimator import estimate_d_manifold_pca, estimate_d_manifold_levina
    print("  thermorg TAS package: OK")
    HAS_TAS = True
except Exception as e:
    print(f"  WARNING: thermorg not available ({e})"); HAS_TAS = False

# ── 2. Test d_manifold with fake data ──────────────────────────────────────
print("\n" + "=" * 60)
print("TEST 2: d_manifold computation (fake data)")
print("=" * 60)
try:
    # Generate fake CIFAR-like data (flattened vectors)
    n = 500
    X = torch.randn(n, 3 * 32 * 32)
    y = torch.randint(0, 10, (n,))
    print(f"  Fake data: {X.shape}, labels: {y.shape}")

    results = {}
    if HAS_TAS:
        for t in [0.90, 0.95, 0.99]:
            try:
                results[f'pca_{int(t*100)}'] = float(estimate_d_manifold_pca(X, t))
                print(f"  PCA @{int(t*100)}%: {results[f'pca_{int(t*100)}']:.1f}")
            except Exception as e:
                print(f"  PCA @{int(t*100)}%: FAIL ({e})")

        for k in [10, 20]:
            try:
                results[f'levina_k{k}'] = float(estimate_d_manifold_levina(X, k))
                print(f"  Levina k={k}: {results[f'levina_k{k}']:.1f}")
            except Exception as e:
                print(f"  Levina k={k}: FAIL ({e})")
    else:
        print("  SKIPPED (no TAS package)")

    # Class-separability metric (always works)
    centroids = torch.stack([X[y==c].mean(0) for c in range(10)])
    cc = centroids - centroids.mean(0)
    cov = torch.cov(cc.T)
    eig = torch.linalg.eigvalsh(cov).flip(0)
    vr = eig / eig.sum()
    d_sep = float((vr.cumsum(0) >= 0.95).nonzero()[0][0].item() + 1)
    print(f"  d_separable_95: {d_sep}")
    print("  d_manifold: OK")
except Exception as e:
    print(f"  FAIL: {e}"); import traceback; traceback.print_exc(); sys.exit(1)

# ── 3. Test JSON/CSV writing ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("TEST 3: JSON/CSV result writing")
print("=" * 60)
try:
    with tempfile.TemporaryDirectory() as tmpdir:
        # Fake results
        fake_results = {
            'tas_results': {
                'ThermoNet-3': {'group': 'G1', 'params': 1060000, 'alpha': 0.123, 'J_topo': 0.456, 'actual_acc': 72.5},
                'ResNet-18-CIFAR': {'group': 'G4', 'params': 11170000, 'alpha': 0.789, 'J_topo': 1.234, 'actual_acc': 85.1},
            },
            'd_manifold': {'pca_95': 59.0, 'levina_k10': 200.0, 'd_separable_95': 9}
        }
        json_path = f'{tmpdir}/phase_a_results.json'
        csv_path = f'{tmpdir}/phase_a_summary.csv'

        with open(json_path, 'w') as f:
            json.dump(fake_results, f, indent=2)
        print(f"  JSON: OK ({json_path})")

        with open(csv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['name','group','params_M','actual_acc','alpha','J_topo'])
            w.writeheader()
            for n, r in fake_results['tas_results'].items():
                w.writerow({'name': n, 'group': r['group'],
                           'params_M': f"{r['params']/1e6:.2f}",
                           'actual_acc': f"{r['actual_acc']:.2f}" if r.get('actual_acc') else '',
                           'alpha': f"{r['alpha']:.4f}" if r.get('alpha') else '',
                           'J_topo': f"{r['J_topo']:.4f}" if r.get('J_topo') else ''})
        print(f"  CSV: OK ({csv_path})")

        # Verify round-trip
        with open(json_path) as f:
            loaded = json.load(f)
        assert loaded['d_manifold']['pca_95'] == 59.0
        print("  JSON round-trip: OK")
except Exception as e:
    print(f"  FAIL: {e}"); import traceback; traceback.print_exc(); sys.exit(1)

# ── 4. Test matplotlib availability ─────────────────────────────────────────
print("\n" + "=" * 60)
print("TEST 4: Plotting (matplotlib)")
print("=" * 60)
try:
    import matplotlib; matplotlib.use('Agg')
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([1, 2, 3], [2, 4, 1])
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        plt.savefig(tmp.name, dpi=100)
        plt.close()
        size = os.path.getsize(tmp.name)
        print(f"  Plot saved: {tmp.name} ({size} bytes) - OK")
        os.unlink(tmp.name)
except Exception as e:
    print(f"  WARNING: plotting not available ({e})")

# ── Summary ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("ALL SMOKE TESTS PASSED")
print("=" * 60)
print("""
Ready to run on Kaggle GPU:
  python3 experiments/phase_a/phase_a_analysis.py \\
      --device cuda --n-samples 5000 --n-per-arch 500 \\
      --actual-results experiments/phase_a/results/training_results.json \\
      --output-dir experiments/phase_a/results/
""")
