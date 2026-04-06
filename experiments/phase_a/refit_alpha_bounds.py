#!/usr/bin/env python3
"""
Refit Phase A power laws with varying alpha bounds.
Reads downloaded .pt checkpoints, extracts per-epoch test_loss curves,
and re-fits with alpha_upper_bound = [5, 10, 20, 50, 100, 200, 500].
"""
import os, glob, json, re, gc
import numpy as np
import torch
from scipy.optimize import minimize

# ============== 配置 ==============
# Leo 的本地 checkpoint 路径（Mac Desktop 下载的）
CKPT_DIR = os.path.expanduser("~/Desktop/ThermoRG-NN/experiments/phase_a/checkpoints_v2")
OUTPUT_FILE = os.path.expanduser("~/Desktop/phase_a_alpha_bound_results.json")

# Power law fitting
def fit_power_law(D_arr, loss_arr, alpha_max=20.0, n_init=300):
    """
    Fit L(D) = alpha * D^(-beta) + E
    Returns (alpha, beta, E, R2) or None if fit fails.
    """
    Ds = np.array(D_arr, dtype=float)
    Ls = np.array(loss_arr, dtype=float)
    
    # Aggregate by D (mean across seeds/runs)
    D_unique = sorted(set(Ds))
    D_mean = np.array(D_unique, dtype=float)
    L_mean = np.array([np.mean(Ls[Ds == d]) for d in D_unique])
    L_std  = np.array([np.std(Ls[Ds == d])  for d in D_unique])
    
    if len(D_unique) < 3:
        return None
    
    best = None
    for _ in range(n_init):
        a0 = np.random.uniform(0.1, alpha_max * 0.7)
        b0 = np.random.uniform(0.05, 3.0)
        e0 = np.random.uniform(0.0, min(L_mean) * 0.9)
        
        try:
            def obj(p):
                a, b, E = p
                return np.sum((L_mean - (a * D_mean ** (-b) + E)) ** 2)
            
            res = minimize(obj, x0=[a0, b0, e0],
                          bounds=[(1e-6, alpha_max), (0.01, 5.0), (1e-6, 10.0)],
                          method='L-BFGS-B')
            a, b, E = res.x
            
            pred = a * D_mean ** (-b) + E
            ss_res = np.sum((L_mean - pred) ** 2)
            ss_tot = np.sum((L_mean - np.mean(L_mean)) ** 2)
            R2 = 1 - ss_res / (ss_tot + 1e-12)
            
            if best is None or R2 > best[3]:
                best = (a, b, E, R2, D_mean, L_mean, L_std, pred)
        except:
            pass
    
    if best is None:
        return None
    
    a, b, E, R2, D_mean, L_mean, L_std, pred = best
    return {
        'alpha': float(a),
        'beta': float(b),
        'E': float(E),
        'R2': float(R2),
        'alpha_bound': alpha_max,
        'D_values': D_mean.tolist(),
        'loss_mean': L_mean.tolist(),
        'loss_std': L_std.tolist(),
        'loss_pred': pred.tolist(),
    }


def main():
    print("=" * 70)
    print("Phase A Alpha Bound Sensitivity Analysis")
    print("=" * 70)
    print(f"\nCheckpoint dir: {CKPT_DIR}")
    
    if not os.path.exists(CKPT_DIR):
        print(f"❌ 目录不存在: {CKPT_DIR}")
        print("请修改 CKPT_DIR 为你的本地路径")
        return
    
    pt_files = sorted(glob.glob(os.path.join(CKPT_DIR, "*.pt")))
    print(f"找到 {len(pt_files)} 个 .pt 文件\n")
    
    if not pt_files:
        print("⚠️ 未找到 .pt 文件")
        return
    
    # Parse filenames
    pattern = re.compile(r'(?P<arch>.+)_D(?P<D>\d+)_s(?P<seed>\d+)\.pt')
    
    # Load all results
    all_runs = []
    skipped = []
    
    for file_path in pt_files:
        fname = os.path.basename(file_path)
        m = pattern.match(fname)
        if not m:
            skipped.append(fname)
            continue
        
        arch = m.group('arch')
        D = int(m.group('D'))
        seed = int(m.group('seed'))
        
        try:
            ckpt = torch.load(file_path, map_location='cpu', weights_only=False)
            r = ckpt.get('result', {})
            
            test_loss = r.get('test_loss', [])
            test_acc  = r.get('test_acc', [])
            
            if not test_loss:
                skipped.append(f"{fname} (no test_loss)")
                del ckpt; gc.collect()
                continue
            
            # Use FINAL epoch (last recorded) as the target
            final_loss = test_loss[-1]
            final_acc  = test_acc[-1] if test_acc else None
            n_epochs   = len(test_loss)
            
            all_runs.append({
                'arch': arch,
                'D': D,
                'seed': seed,
                'final_loss': float(final_loss),
                'final_acc': float(final_acc) if final_acc is not None else None,
                'n_epochs': n_epochs,
                'loss_curve': [float(x) for x in test_loss],
            })
            del ckpt
            gc.collect()
        except Exception as e:
            skipped.append(f"{fname} (error: {e})")
    
    if skipped:
        print(f"跳过 {len(skipped)} 个文件:")
        for s in skipped[:5]:
            print(f"  - {s}")
        if len(skipped) > 5:
            print(f"  ... 还有 {len(skipped)-5} 个")
    
    print(f"成功加载 {len(all_runs)} 个 run\n")
    
    # Group by arch
    archs = sorted(set(r['arch'] for r in all_runs))
    print(f"Architecture: {archs}\n")
    
    # Bounds to test
    bounds_to_test = [5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0]
    
    # Results storage
    results_by_arch = {}
    summary_table = []
    
    print("=" * 70)
    print(f"{'Arch':15s}  {'Bound':>6s}  {'alpha':>8s}  {'beta':>7s}  {'E':>6s}  {'R2':>6s}")
    print("-" * 70)
    
    for arch in archs:
        runs = [r for r in all_runs if r['arch'] == arch]
        Ds   = np.array([r['D'] for r in runs], dtype=float)
        Ls   = np.array([r['final_loss'] for r in runs], dtype=float)
        
        arch_fits = {}
        prev_alpha = None
        
        for bound in bounds_to_test:
            fit = fit_power_law(Ds, Ls, alpha_max=bound, n_init=300)
            
            if fit is None:
                continue
            
            a, b, E, R2 = fit['alpha'], fit['beta'], fit['E'], fit['R2']
            hit = " ←HIT" if abs(a - bound) / bound < 0.05 else ""
            changed = " ***" if (prev_alpha is not None and 
                                  abs(a - prev_alpha) / (prev_alpha + 1e-6) > 0.05) else ""
            
            print(f"  {arch:15s}  {bound:6.0f}  {a:8.2f}  {b:7.4f}  {E:6.4f}  {R2:6.4f}{hit}{changed}")
            
            arch_fits[f"bound_{int(bound)}"] = fit
            prev_alpha = a
        
        results_by_arch[arch] = arch_fits
        
        # Save best (unbounded) fit summary
        best_fit = arch_fits.get(f"bound_{int(max(bounds_to_test))}", {})
        summary_table.append({
            'arch': arch,
            'alpha_200': arch_fits.get('bound_200', {}).get('alpha', None),
            'beta_200':  arch_fits.get('bound_200', {}).get('beta', None),
            'E_200':     arch_fits.get('bound_200', {}).get('E', None),
            'R2_200':    arch_fits.get('bound_200', {}).get('R2', None),
            'alpha_20':  arch_fits.get('bound_20', {}).get('alpha', None),
            'beta_20':   arch_fits.get('bound_20', {}).get('beta', None),
            'bound_affected': "YES" if arch_fits.get('bound_20', {}).get('alpha', 0) >= 19.9 else "NO",
        })
        print()
    
    # Save
    output = {
        'fits_by_arch': results_by_arch,
        'summary': summary_table,
        'all_runs': all_runs,
        'bounds_tested': bounds_to_test,
    }
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print("=" * 70)
    print("SUMMARY: alpha=20 (bound) vs alpha=200 (raised)")
    print("=" * 70)
    print(f"\n{'Arch':15s}  {'alpha_20':>8s}  {'beta_20':>8s}  {'alpha_200':>9s}  {'beta_200':>9s}  {'Changed?'}")
    print("-" * 70)
    
    for s in summary_table:
        changed = "*** YES" if s['bound_affected'] == "YES" else "no"
        print(f"  {s['arch']:15s}  "
              f"{s.get('alpha_20', 0) or 0:8.2f}  "
              f"{s.get('beta_20', 0) or 0:8.4f}  "
              f"{s.get('alpha_200', 0) or 0:9.2f}  "
              f"{s.get('beta_200', 0) or 0:9.4f}  "
              f"{changed}")
    
    print(f"\n✅ 结果已保存: {OUTPUT_FILE}")
    print("\n解读:")
    print("  *** YES = alpha hit bound=20, raising bound changes the fit")
    print("  如果 alpha_200 vs alpha_20 差异大，说明真实 alpha > 20")
    print("  如果 beta_200 vs beta_20 差异也大，说明 alpha-beta 耦合显著")


if __name__ == '__main__':
    main()
