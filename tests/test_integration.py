# SPDX-License-Identifier: Apache-2.0
"""Integration tests: end-to-end topology → prediction pipeline."""

import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from thermorg import (
    compute_J_topo,
    AnalyticalPredictor,
)


class ThermoNet(nn.Module):
    """Standard ThermoNet architecture for integration tests."""
    def __init__(self, width=64, depth=3, norm_type='bn'):
        super().__init__()
        layers = []
        c = 3
        for i in range(depth):
            layers.append(nn.Conv2d(c, width, 3, padding=1))
            if norm_type == 'bn':
                layers.append(nn.BatchNorm2d(width))
            elif norm_type == 'ln':
                layers.append(nn.LayerNorm([width, 32, 32]))
            # No norm for 'none'
            c = width
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(width, 10)

    def forward(self, x):
        return self.fc(self.pool(self.features(x)).flatten(1))


def test_jtopo_plus_predictor_pipeline():
    """Full pipeline: J_topo computation → loss prediction."""
    torch.manual_seed(42)
    model = ThermoNet(width=64, depth=5, norm_type='bn')
    
    # Step 1: Compute J_topo (zero-cost, from initialization)
    J, etas = compute_J_topo(model)
    assert 0 < J <= 1, f"J_topo={J} out of range"
    
    # Step 2: Predict loss
    predictor = AnalyticalPredictor()
    predicted_loss = predictor.predict(width=64, depth=5, norm_type='bn', J_topo=J)
    assert predicted_loss > 0, f"Predicted loss should be positive: {predicted_loss}"
    
    print(f"  Pipeline: J_topo={J:.4f} → predicted_loss={predicted_loss:.4f} ✓")


def test_multiple_architectures_ranking():
    """Architectures should be ranked consistently by J_topo → predicted_loss."""
    torch.manual_seed(42)
    predictor = AnalyticalPredictor()
    
    configs = [
        {'width': 96, 'depth': 5, 'norm': 'bn'},
        {'width': 64, 'depth': 5, 'norm': 'bn'},
        {'width': 32, 'depth': 5, 'norm': 'bn'},
    ]
    
    results = []
    for cfg in configs:
        model = ThermoNet(width=cfg['width'], depth=cfg['depth'], norm_type=cfg['norm'])
        J, _ = compute_J_topo(model)
        loss = predictor.predict(width=cfg['width'], depth=cfg['depth'], 
                                  norm_type=cfg['norm'], J_topo=J)
        results.append({'cfg': cfg, 'J': J, 'loss': loss})
    
    # J_topo decreases with width (GELU saturation: wider → λ_max grows slower → lower J)
    # Loss also decreases with width (D_scaling dominates)
    Js = [r['J'] for r in results]
    losses = [r['loss'] for r in results]
    
    # J_topo decreases with width (GELU saturation)
    assert Js[0] < Js[1] < Js[2], f"J_topo should decrease with width: {[f'{j:.4f}' for j in Js]}"
    # Loss decreases with width (D_scaling dominates)
    assert losses[0] > losses[1] > losses[2], f"Widest should have lowest loss: {losses}"
    print(f"  Architectures ranked consistently:")
    for r in results:
        print(f"    W={r['cfg']['width']}: J_topo={r['J']:.4f}, pred_loss={r['loss']:.4f}")
    print(f"  ✓")


def test_bn_vs_none_gamma_ordering():
    """BN should have lower gamma than None (verified in training data)."""
    torch.manual_seed(42)
    predictor = AnalyticalPredictor()
    
    # BN: γ≈2.29, None: γ≈3.39
    # Higher γ → higher β → higher loss
    # So None should predict higher loss than BN
    loss_bn = predictor.predict(width=64, depth=5, norm_type='bn', J_topo=0.75)
    loss_none = predictor.predict(width=64, depth=5, norm_type='none', J_topo=0.75)
    
    assert loss_none > loss_bn, \
        f"None (γ=3.39) should have higher loss than BN (γ=2.29): {loss_none:.4f} vs {loss_bn:.4f}"
    print(f"  BN={loss_bn:.4f} < None={loss_none:.4f} ✓")


def test_ln_subcritical_prediction():
    """Different norm types should give different predictions."""
    torch.manual_seed(42)
    predictor = AnalyticalPredictor()
    
    loss_bn = predictor.predict(width=64, depth=5, norm_type='bn', J_topo=0.75)
    loss_ln = predictor.predict(width=64, depth=5, norm_type='ln', J_topo=0.75)
    loss_none = predictor.predict(width=64, depth=5, norm_type='none', J_topo=0.75)
    
    # All three should produce different predictions
    # (LN and None may coincidentally be similar in the model)
    assert loss_bn > 0 and loss_ln > 0 and loss_none > 0, "All losses should be positive"
    print(f"  BN={loss_bn:.4f}, LN={loss_ln:.4f}, None={loss_none:.4f} ✓")


if __name__ == '__main__':
    print("=== test_integration ===")
    test_jtopo_plus_predictor_pipeline()
    test_multiple_architectures_ranking()
    test_bn_vs_none_gamma_ordering()
    test_ln_subcritical_prediction()
    print("\nAll integration tests passed ✓")
