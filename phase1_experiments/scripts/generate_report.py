#!/usr/bin/env python3
"""
Generate analysis report from experiment results.
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path


def generate_report(
    phase_0_1_results: dict = None,
    phase_0_2_results: dict = None,
    phase_1_1_results: dict = None,
    output_path: str = './outputs/report.md'
):
    """Generate markdown report from experiment results."""
    
    report = f"""# ThermoRG v4 Experiment Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## Executive Summary

This report summarizes the results of the ThermoRG v4 parallel lines verification experiment.

---

## Phase 0.1: EOS Boundary Verification

### Objective
Verify that the EOS condition λ_max · η_crit ≈ 2 holds for both BatchNorm and LayerNorm.

### Results

"""
    
    if phase_0_1_results:
        report += "### EOS Ratio Measurements\n\n"
        report += "| Normalization | LR | λ_max | λ_max·η | Converged |\n"
        report += "|--------------|-----|-------|---------|-----------|\n"
        
        eos_data = phase_0_1_results.get('eos_results', {})
        for norm_type, lr_data in eos_data.items():
            for lr, seeds in lr_data.items():
                for seed, data in seeds.items():
                    eos_ratio = data.get('eos_ratio', 0)
                    converged = '✓' if data.get('is_converged') else '✗'
                    report += f"| {norm_type} | {lr} | {data.get('lambda_max_init', 0):.3f} | {eos_ratio:.3f} | {converged} |\n"
    else:
        report += "*No data available*\n"
    
    report += """

### Conclusion
"""
    
    if phase_0_1_results:
        report += "- EOS verification results\n"
    else:
        report += "- No data collected\n"
    
    report += f"""

---

## Phase 0.2: BN vs LN Parallel Lines Core Test

### Objective
Measure β vs ln(γ) for BatchNorm and LayerNorm. Test if they form parallel lines (same slope).

### Results

"""
    
    if phase_0_2_results:
        regression = phase_0_2_results.get('regression', {})
        
        report += "### Linear Regression: β = m · ln(γ) + c\n\n"
        report += "| Normalization | Slope m | Intercept c | R² | n_points |\n"
        report += "|--------------|---------|-------------|-----|----------|\n"
        
        for norm_type, res in regression.items():
            report += f"| {norm_type} | {res.get('m', 'N/A'):.4f} | {res.get('c', 'N/A'):.4f} | {res.get('r_squared', 0):.4f} | {res.get('n_points', 0)} |\n"
        
        f_test = phase_0_2_results.get('f_test')
        if f_test and f_test.get('f_statistic'):
            report += f"\n### F-test for Equal Slopes\n"
            report += f"- F-statistic: {f_test['f_statistic']:.4f}\n"
            report += f"- p-value: {f_test['p_value']:.4f}\n"
            report += f"- Reject H0 (slopes differ)? {f_test.get('reject_null', 'N/A')}\n"
    else:
        report += "*No data available*\n"
    
    report += """

### Conclusion
"""
    
    if phase_0_2_results:
        if f_test:
            if not f_test.get('reject_null'):
                report += "- ✓ Slopes are approximately equal (parallel lines hypothesis supported)\n"
            else:
                report += "- ✗ Slopes are significantly different (parallel lines hypothesis rejected)\n"
    else:
        report += "- No data collected\n"
    
    report += f"""

---

## Phase 1.1: Full Sweep

### Objective
Complete grid scan with all 4 normalization types × 6 D values × 8 learning rates × 5 seeds.

### Status
*In progress or not yet run*

---

## Physical Constants Extracted

| Constant | Value | Description |
|----------|-------|-------------|
| β_spec | TBD | Universal slope of β vs ln(γ) |
| A_norm | TBD per type | Normalization-specific intercept parameter |

---

## Figures

1. **Figure 1:** β vs ln(γ) parallel lines plot
2. **Figure 2:** D-scaling collapse
3. **Figure 3:** EOS verification

---

## Appendix

### Protocol Details
- Architecture: ConvNet L=5, kernel=3×3, no skip, GELU
- Dataset: CIFAR-10
- Stationarity: 20 epochs with 85% monotonicity
- D-scaling fit threshold: R² ≥ 0.995

### Reproducibility
- Seeds: 42, 43, 44, 45, 46
- Batch size: 128
- Optimizer: SGD with momentum=0.9
"""
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(report)
    
    print(f"Report saved to {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase-0.1', type=str, help='Path to phase 0.1 results JSON')
    parser.add_argument('--phase-0.2', type=str, help='Path to phase 0.2 results JSON')
    parser.add_argument('--phase-1.1', type=str, help='Path to phase 1.1 results JSON')
    parser.add_argument('--output', type=str, default='./outputs/reports/report.md')
    args = parser.parse_args()
    
    phase_0_1 = None
    phase_0_2 = None
    phase_1_1 = None
    
    if args.phase_0_1 and os.path.exists(args.phase_0_1):
        with open(args.phase_0_1) as f:
            phase_0_1 = json.load(f)
    
    if args.phase_0_2 and os.path.exists(args.phase_0_2):
        with open(args.phase_0_2) as f:
            phase_0_2 = json.load(f)
    
    if args.phase_1_1 and os.path.exists(args.phase_1_1):
        with open(args.phase_1_1) as f:
            phase_1_1 = json.load(f)
    
    generate_report(phase_0_1, phase_0_2, phase_1_1, args.output)


if __name__ == '__main__':
    main()
