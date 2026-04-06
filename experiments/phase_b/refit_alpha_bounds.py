#!/usr/bin/env python3
"""
Re-fit Phase B critical region data with varying alpha bounds.
Demonstrates that the alpha hitting bound=20 is an artifact of restrictive bounds.
"""
import numpy as np
from scipy.optimize import minimize
import math, json
from pathlib import Path

def fit_pl_multi_init(Ds_arr, Ls_arr, alpha_max=20.0, n_init=50):
    """Power law fit with multiple random initializations."""
    Ds = np.array(Ds_arr, dtype=float)
    Ls = np.array(Ls_arr, dtype=float)
    
    best = None
    for _ in range(n_init):
        a0 = np.random.uniform(0.1, alpha_max * 0.9)
        b0 = np.random.uniform(0.05, 2.0)
        e0 = np.random.uniform(0.0, 2.0)
        try:
            def obj(p):
                a, b, E = p
                return np.sum((Ls - (a * Ds**(-b) + E))**2)
            res = minimize(obj, x0=[a0, b0, e0],
                          bounds=[(1e-6, alpha_max), (0.005, 5.0), (1e-6, 10.0)],
                          method='L-BFGS-B')
            a, b, E = res.x
            ss_r = np.sum((Ls - (a*Ds**(-b)+E))**2)
            ss_t = np.sum((Ls - np.mean(Ls))**2)
            R2 = 1 - ss_r / (ss_t + 1e-10)
            if best is None or R2 > best[3]:
                best = (a, b, E, R2)
        except: pass
    return best  # (alpha, beta, E, R2)

def main():
    with open('experiments/phase_b/critical_region_results.json') as f:
        d = json.load(f)
    
    results = d['results']
    
    print("="*70)
    print("RE-FIT WITH VARYING ALPHA BOUNDS — Phase B Critical Region")
    print("="*70)
    
    # Test different alpha bounds
    bounds_to_test = [5.0, 10.0, 20.0, 50.0, 100.0, 500.0]
    
    print(f"\n{'Arch':15s}  {'J':>6s}  {'Bound':>6s}  {'alpha':>8s}  {'beta':>7s}  {'E':>6s}  {'R2':>6s}")
    print("-"*70)
    
    for cfg in d['candidates']:
        name = cfg['name']
        J = cfg['J_topo']
        runs = [r for r in results if r['arch'] == name]
        Ds = np.array([r['D'] for r in runs], dtype=float)
        Ls = np.array([r['final_val_loss'] for r in runs])
        
        prev_a = None
        for bound in bounds_to_test:
            a, b, E, R2 = fit_pl_multi_init(Ds, Ls, alpha_max=bound, n_init=100)
            hit = " ←HIT" if abs(a - bound) < 0.01 * bound else ""
            changed = " ***" if prev_a is not None and (a - prev_a)/prev_a > 0.05 else ""
            print(f"  {name:13s}  {J:6.4f}  {bound:6.0f}  {a:8.2f}  {b:7.4f}  {E:6.4f}  {R2:6.4f}{hit}{changed}")
            prev_a = a
        print()
    
    print("="*70)
    print("ANALYSIS")
    print("="*70)
    print("""
KEY:
  ←HIT = fitter reached the upper bound (alpha is ABOVE bound)
  ***  = alpha CHANGED when bound was raised (bound was restrictive)

If alpha changes significantly when bound goes from 20 → 50 → 100,
then the original bound=20 was restrictive and the true alpha > 20.

If alpha stays the same across all bounds,
then alpha is well-constrained and ≈ that value.
""")

if __name__ == '__main__':
    main()
