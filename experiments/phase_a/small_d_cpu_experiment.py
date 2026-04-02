#!/usr/bin/env python3
"""
Small-D CPU Experiment for ThermoRG Phase A
==========================================
Goal: Break the alpha/beta degeneracy by adding data points at D=100-2000.
      The power law curvature is most visible at small D values.

Strategy:
- Use TN-L3 with wm=0.1 (~76K params, ~9s/epoch on CPU)
- D values: 100, 200, 500, 1000, 2000 (extend below Phase A's D=2000 minimum)
- 30 epochs, 2 seeds (42, 123)
- Total runtime: ~15 min on CPU

Expected outcome:
- Loss at D=100 should be significantly higher than D=2000
- This gives the power law "left side" that Phase A lacks
- Alpha will be measurable from the combined D=100-2000 + D=2000-50000 data
"""

import torch, numpy as np, time, json, sys
from pathlib import Path

# Import from phase_a_dscaling
from experiments.phase_a.phase_a_dscaling import (
    build_TN3, train_run, load_cifar10,
    compute_J_topo, get_layer_weights_combined
)

# Configuration
EPOCHS = 20  # 20 epochs per run keeps total time ~20-25 min on CPU
D_VALUES = [100, 200, 500, 1000, 2000]
SEEDS = [42, 123]
ARCH_CFG = {'Name': 'TN-L3-S', 'arch': 'TN3', 'width_mult': 0.1}
OUTPUT_FILE = Path('experiments/phase_a/results_small_d.json')

def run_small_d_experiment():
    print("="*60)
    print("Small-D CPU Experiment for ThermoRG")
    print("="*60)
    print(f"D values: {D_VALUES}")
    print(f"Epochs: {EPOCHS}")
    print(f"Seeds: {SEEDS}")
    print(f"Architecture: {ARCH_CFG['Name']} (wm={ARCH_CFG['width_mult']})")
    print()

    # Load data
    print("Loading CIFAR-10...")
    X_train, Y_train, X_test, Y_test = load_cifar10()

    # First, compute J_topo for this architecture
    model_init = build_TN3(wm=ARCH_CFG['width_mult'])
    weights_init = get_layer_weights_combined(model_init, ARCH_CFG['arch'], ARCH_CFG['width_mult'])
    J_topo_init, _ = compute_J_topo(weights_init)
    print(f"J_topo (init): {J_topo_init:.4f}")
    print(f"Total params: {sum(p.numel() for p in model_init.parameters())/1e3:.1f}K")
    print()

    results = []
    total_runs = len(D_VALUES) * len(SEEDS)
    run_idx = 0

    for D in D_VALUES:
        for seed in SEEDS:
            run_idx += 1
            print(f"[{run_idx}/{total_runs}] D={D}, seed={seed}...", end=" ", flush=True)
            t0 = time.time()

            result = train_run(
                ARCH_CFG, D, seed,
                X_train, Y_train, X_test, Y_test,
                epochs=EPOCHS, ckpt_path=None
            )

            elapsed = time.time() - t0
            print(f"loss={result['test_loss'][-1]:.4f}, acc={result['test_acc'][-1]:.3f}, {elapsed:.0f}s")

            results.append({
                'arch': ARCH_CFG['Name'],
                'base_arch': ARCH_CFG['arch'],
                'width_mult': ARCH_CFG['width_mult'],
                'D': D,
                'seed': seed,
                'final_val_loss': result['test_loss'][-1],
                'final_val_acc': result['test_acc'][-1],
                'J_topo_init': J_topo_init,
                'J_topo_final': result['J_topo_final'],
                'params_M': result['params_M'],
                'epochs_recorded': EPOCHS,
            })

    # Fit power law
    print("\n" + "="*60)
    print("Power Law Fit (combined with existing Phase A data)")
    print("="*60)

    Ds = np.array([r['D'] for r in results], dtype=float)
    losses = np.array([r['final_val_loss'] for r in results])

    try:
        from scipy.optimize import curve_fit
        def pl(D, alpha, beta, E):
            return alpha * D**(-beta) + E
        popt, pcov = curve_fit(pl, Ds, losses, p0=[5.0, 0.5, 1.0],
                               bounds=([0, 0, 0], [50, 3, 5]), maxfev=10000)
        alpha, beta, E = popt
        pred = pl(Ds, alpha, beta, E)
        ss_res = np.sum((losses - pred)**2)
        ss_tot = np.sum((losses - np.mean(losses))**2)
        r2 = 1 - ss_res / ss_tot
        print(f"  alpha = {alpha:.4f}")
        print(f"  beta  = {beta:.4f}")
        print(f"  E     = {E:.4f}")
        print(f"  R^2   = {r2:.4f}")
        print(f"\n  Note: alpha={alpha:.2f} {'(bounded - need larger D range)' if alpha > 40 else '(free - good fit!)'}")
    except Exception as e:
        print(f"  Fit failed: {e}")
        alpha, beta, E = None, None, None

    # Save results
    output_data = {
        'experiment': 'small_d_cpu',
        'description': 'D=100-2000 CPU experiment to extend Phase A data',
        'architecture': ARCH_CFG,
        'j_topo': J_topo_init,
        'epochs': EPOCHS,
        'results': results,
        'power_law': {
            'alpha': float(alpha) if alpha else None,
            'beta': float(beta) if beta else None,
            'E': float(E) if E else None,
        } if alpha is not None else None
    }

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to {OUTPUT_FILE}")

    return results

if __name__ == '__main__':
    run_small_d_experiment()
