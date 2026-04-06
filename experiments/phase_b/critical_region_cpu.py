#!/usr/bin/env python3
"""
Critical Region CPU Experiment for ThermoRG Phase B
================================================
Goal: Fill the J_c ≈ 0.40 gap in the alpha phase transition.
Writes checkpoint after every architecture to survive process interruption.
"""

import numpy as np, time, json, sys, os
from pathlib import Path

# Block torchvision to avoid nms error
sys.modules['torchvision'] = None
sys.modules['torchvision.datasets'] = None
sys.modules['torchvision.transforms'] = None

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.phase_a.phase_a_dscaling import (
    build_TN3, build_TN5, build_TN7,
    get_layer_weights_combined, compute_J_topo,
    train_one_epoch, subset_loader, evaluate
)

OUTPUT = Path(__file__).parent / 'critical_region_results.json'
TARGET_ARCHES = [
    {'builder': build_TN3, 'wm': 0.6, 'key': 'TN-L3', 'name': 'TN-L3(w0.6)'},
    {'builder': build_TN5, 'wm': 0.5, 'key': 'TN-L5', 'name': 'TN-L5(w0.5)'},
    {'builder': build_TN7, 'wm': 1.0, 'key': 'TN-L7', 'name': 'TN-L7(w1.0)'},
    {'builder': build_TN7, 'wm': 0.9, 'key': 'TN-L7', 'name': 'TN-L7(w0.9)'},
]
D_VALUES = [100, 200, 500, 1000]
SEEDS = [42, 123]
EPOCHS = 30

def load_cifar10_binary():
    all_data, labels = [], []
    for i in range(1, 6):
        path = f"/tmp/cifar-10-batches-bin/data_batch_{i}.bin"
        with open(path, 'rb') as f:
            while True:
                record = f.read(3073)
                if len(record) < 3073: break
                all_data.append(np.frombuffer(record[1:], dtype=np.uint8))
                labels.append(record[0])
    X = np.stack(all_data, axis=0).reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    Y = np.array(labels, dtype=np.int64)
    all_data, labels = [], []
    path = "/tmp/cifar-10-batches-bin/test_batch.bin"
    with open(path, 'rb') as f:
        while True:
            record = f.read(3073)
            if len(record) < 3073: break
            all_data.append(np.frombuffer(record[1:], dtype=np.uint8))
            labels.append(record[0])
    X_test = np.stack(all_data, axis=0).reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    Y_test = np.array(labels, dtype=np.int64)
    return X, Y, X_test, Y_test

def load_checkpoint():
    if OUTPUT.exists():
        with open(OUTPUT) as f:
            return json.load(f)
    return None

def save_checkpoint(data):
    with open(OUTPUT, 'w') as f:
        json.dump(data, f, indent=2)

def main():
    import torch
    from scipy.optimize import curve_fit

    print("=" * 65)
    print("CRITICAL REGION CPU EXPERIMENT")
    print("=" * 65)

    # Load data
    print("\nLoading CIFAR-10...")
    X_train, Y_train, X_test, Y_test = load_cifar10_binary()
    X_train_t = torch.from_numpy(X_train)
    Y_train_t = torch.from_numpy(Y_train)
    X_test_t = torch.from_numpy(X_test)
    Y_test_t = torch.from_numpy(Y_test)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # Compute J_topo
    print("\nComputing J_topo:")
    for cfg in TARGET_ARCHES:
        model = cfg['builder'](wm=cfg['wm'])
        weights = get_layer_weights_combined(model, cfg['key'], cfg['wm'])
        J, _ = compute_J_topo(weights)
        cfg['J_topo'] = J
        print(f"  {cfg['name']}: J_topo = {J:.4f}")

    # Load checkpoint
    ckpt = load_checkpoint()
    if ckpt:
        results = ckpt.get('results', [])
        print(f"\nResuming from checkpoint: {len(results)} runs already complete")
    else:
        results = []
        ckpt = {
            'experiment': 'critical_region_cpu',
            'candidates': [{'name': c['name'], 'J_topo': c['J_topo']} for c in TARGET_ARCHES],
            'D_range': D_VALUES,
            'epochs': EPOCHS,
            'results': results,
            'fits': {}
        }

    # Training loop
    total_runs = len(TARGET_ARCHES) * len(D_VALUES) * len(SEEDS)
    done = {f"{r['arch']}:{r['D']}:{r['seed']}" for r in results}

    for cfg in TARGET_ARCHES:
        for D in D_VALUES:
            for seed in SEEDS:
                key = f"{cfg['name']}:{D}:{seed}"
                if key in done:
                    continue
                print(f"\n[{len(results)+1}/{total_runs}] {cfg['name']} D={D} seed={seed}...", end=" ", flush=True)
                t0 = time.time()

                model = cfg['builder'](wm=cfg['wm'])
                optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
                loader = subset_loader(X_train_t, Y_train_t, D, 128, seed)

                for ep in range(EPOCHS):
                    loss = train_one_epoch(model, loader, optimizer)

                t1 = time.time()
                test_loss, test_acc = evaluate(model, X_test_t, Y_test_t)
                elapsed = t1 - t0
                print(f"loss={test_loss:.4f} acc={test_acc:.3f} {elapsed:.0f}s")

                results.append({
                    'arch': cfg['name'],
                    'D': D,
                    'seed': seed,
                    'final_val_loss': test_loss,
                    'final_val_acc': test_acc,
                    'J_topo': cfg['J_topo'],
                    'epochs': EPOCHS,
                    'time_s': elapsed,
                })
                ckpt['results'] = results
                save_checkpoint(ckpt)

    # Fit power laws
    print("\n" + "=" * 65)
    print("POWER LAW FITS")
    print("=" * 65)

    def power_law(D, alpha, beta, E):
        return alpha * np.array(D, dtype=float)**(-np.array(beta)) + E

    fits = {}
    for cfg in TARGET_ARCHES:
        name = cfg['name']
        J = cfg['J_topo']
        runs = [r for r in results if r['arch'] == name]
        D_arr = np.array([r['D'] for r in runs], dtype=float)
        loss_arr = np.array([r['final_val_loss'] for r in runs])
        try:
            popt, _ = curve_fit(power_law, D_arr, loss_arr, p0=[5.0, 0.5, 0.5],
                                bounds=([0, 0, 0], [100, 3, 5]), maxfev=10000)
            alpha, beta, E = popt
            pred = power_law(D_arr, alpha, beta, E)
            ss_res = np.sum((loss_arr - pred)**2)
            ss_tot = np.sum((loss_arr - np.mean(loss_arr))**2)
            r2 = 1 - ss_res / (ss_tot + 1e-12)
            fits[name] = {'alpha': alpha, 'beta': beta, 'E': E, 'J_topo': J, 'r2': r2}
            print(f"  {name} (J={J:.4f}): alpha={alpha:.2f}, beta={beta:.4f}, E={E:.4f}, R2={r2:.4f}")
        except Exception as e:
            print(f"  {name}: FIT FAILED - {e}")
            fits[name] = {'alpha': None, 'beta': None, 'E': None, 'J_topo': J, 'r2': None}

    ckpt['fits'] = fits
    save_checkpoint(ckpt)
    print(f"\nSaved to {OUTPUT}")
    return results, fits

if __name__ == '__main__':
    main()
