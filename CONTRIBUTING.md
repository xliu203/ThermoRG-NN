# Contributing to ThermoRG-NN

Thank you for your interest in contributing to ThermoRG-NN!

## Collaboration Directions

We welcome contributions in four areas:

1. **Theory verification and consolidation** – rigorous checking and solidification of the thermodynamic and geometric foundations.

2. **Validation on diverse datasets** – testing the scaling law across different data modalities and scales.

3. **Code optimization** – improving computational efficiency, scalability, and usability.

4. **Multi-modality applications** – extending the framework to integrate heterogeneous data streams for world-modeling and complex tasks.

## How to Contribute

### Bug Reports

Please report bugs via GitHub Issues with:
- Clear description of the issue
- Minimal code to reproduce
- Expected vs actual behavior
- Environment details (Python version, OS, etc.)

### Feature Requests

We welcome feature requests! Please describe:
- The problem you're trying to solve
- Proposed solution
- Alternative solutions considered

### Code Contributions

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes
4. Run tests: `pytest tests/`
5. Commit with clear messages
6. Push and create a Pull Request

### Code Style

- Follow PEP 8
- Use type hints where possible
- Add docstrings to new functions/classes
- Keep changes focused and atomic

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/ThermoRG-NN.git
cd ThermoRG-NN

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/ -v
```

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.
