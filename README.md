# ThermoRG-NN

ThermoRG‑NN is a unified framework for neural architecture design grounded in thermodynamics and spectral momentum conservation. It formulates network training as a thermodynamic process, yielding a Unified Scaling Law that predicts the scaling exponent α from architectural and training parameters. The framework provides thermogeometric optimality criteria for architecture selection. It supports multi‑modality (tabular, text, video, audio) via embeddings. Additionally, it employs a memory‑efficient VJP‑based Jacobian for effective‑dimensionality computation.

We welcome collaboration in four directions:

1. **Theory verification and consolidation** – rigorous checking and solidification of the thermodynamic and geometric foundations.
2. **Validation on diverse datasets** – testing the scaling law across different data modalities and scales.
3. **Code optimization** – improving computational efficiency, scalability, and usability.
4. **Multi-modality applications** – extending the framework to integrate heterogeneous data streams for world-modeling and complex tasks.

The project aims to bridge theoretical principles with practical design, offering a principled alternative to heuristic architecture search.

## Overview


ThermoRG-NN provides a first-principles approach to understanding and designing neural network architectures. By viewing network training as a thermodynamic process, we derive a Unified Scaling Law that predicts the scaling exponent α from architectural and training parameters.

### Key Features

- **Unified Scaling Law**: α = k_α · |log∏η_l| · (2s/d) · ψ(T_eff) · φ(γ_cool)
- **Thermogeometric Optimality**: Formal criteria for optimal architecture selection
- **Multi-modality Support**: Tabular, Text, Video, Audio via embeddings
- **VJP-based Jacobian**: Memory-efficient D_eff computation

## Installation

```bash
pip install git+https://github.com/USER/ThermoRG-NN.git
```

For optional dependencies (vision, text):

```bash
pip install git+https://github.com/USER/ThermoRG-NN.git[vision,text]
```

## Quick Start

```python
from thermorg.tas import TASProfiler, TASConfig

# Basic usage
profiler = TASProfiler()
result = profiler.profile(X, y, architecture={'widths': [64, 128, 256]}, 
                         train_config={'lr': 1e-3, 'batch_size': 32})
print(f"α = {result.alpha:.4f}")

# With optimality verification
optimality = profiler.verify_and_profile(X, y, architecture, train_config)
print(f"Feasible: {optimality.optimality_result.is_feasible}")
```

## Theoretical Background

The Unified Scaling Law relates the scaling exponent α to:

| Term | Description |
|------|-------------|
| \|log∏η_l\| | Topological compression |
| 2s/d | Data manifold geometry |
| ψ(T_eff) | Thermal exploration phase |
| φ(γ_cool) | Cooling dynamics |

See our papers for full theory:
- [Unified Framework Paper](papers/unified_framework_paper_final.pdf)
- [Applied Theory Paper](papers/applied_theory_paper_final.pdf)

## API Reference

### TASProfiler

Main class for architecture profiling.

### TASConfig

Configuration for TAS profiling.

### OptimalityResult

Result of Phase 6 optimality verification.

## Citation

If you use ThermoRG-NN in your research, please cite:

```bibtex
@article{liu2026thermorgnn,
  title={ThermoRG-NN: Thermogeometric Neural Architecture Search},
  author={Liu, Leo},
  year={2026}
}
```

## License

Apache License 2.0 - see [LICENSE](LICENSE)
