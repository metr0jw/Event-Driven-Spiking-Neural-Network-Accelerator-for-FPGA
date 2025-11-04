# Contributing to Event-Driven SNN FPGA Accelerator

Thank you for your interest in contributing to this project! This guide will help you get started with contributing to the Event-Driven Spiking Neural Network (SNN) FPGA Accelerator.

## Table of Contents
- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Environment Setup](#development-environment-setup)
- [Project Structure](#project-structure)
- [Contributing Guidelines](#contributing-guidelines)
- [Hardware Development](#hardware-development)
- [Software Development](#software-development)
- [Testing Requirements](#testing-requirements)
- [Documentation Standards](#documentation-standards)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Community](#community)

## Code of Conduct

This project adheres to a code of conduct that all contributors are expected to follow:

- **Be respectful**: Treat everyone with respect and consideration
- **Be collaborative**: Work together constructively
- **Be professional**: Focus on technical merit and project goals
- **Be inclusive**: Welcome contributors of all backgrounds and experience levels

## Getting Started

### Prerequisites

Before contributing, ensure you have:

1. **For Software Development**:
   - Python 3.13 or higher
   - PyTorch 2.9.0 or higher
   - NumPy, pytest, and other Python dependencies

2. **For Hardware Development**:
   - Xilinx Vivado 2025.1 or compatible version
   - Xilinx Vitis HLS 2025.1 or compatible version
   - Icarus Verilog 11.0+ (for open-source simulation)
   - PYNQ-Z2 board (for hardware testing)

3. **General Tools**:
   - Git for version control
   - A GitHub account
   - Text editor or IDE (VS Code, PyCharm, etc.)

### Initial Setup

1. **Fork the repository** on GitHub
2. **Clone your fork**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/Spiking-Neural-Network-on-FPGA.git
   cd Spiking-Neural-Network-on-FPGA
   ```

3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/metr0jw/Spiking-Neural-Network-on-FPGA.git
   ```

4. **Run setup script**:
   ```bash
   ./setup.sh
   ```

## Development Environment Setup

See the [Developer Guide](docs/developer_guide.md) for detailed setup instructions.

### Quick Setup

#### Python Environment
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install in development mode
cd software/python
pip install -e .
pip install pytest pytest-cov black flake8 mypy
```

#### Hardware Environment
```bash
# Source Xilinx tools
source /opt/Xilinx/Vivado/2025.1/settings64.sh
source /opt/Xilinx/Vitis_HLS/2025.1/settings64.sh
```

For detailed environment setup, build procedures, and development workflows, refer to the [Developer Guide](docs/developer_guide.md).

## Project Structure

Understanding the project structure will help you navigate and contribute effectively:

```
├── hardware/                    # FPGA implementation
│   ├── hdl/                    # Hardware Description Language
│   │   ├── rtl/                # Synthesizable Verilog-2001/SystemVerilog
│   │   │   ├── common/         # Utilities (FIFO, synchronizers)
│   │   │   ├── neurons/        # LIF neuron implementations
│   │   │   ├── synapses/       # Synaptic arrays
│   │   │   ├── router/         # Spike routing logic
│   │   │   ├── layers/         # Conv1D, pooling, etc.
│   │   │   ├── interfaces/     # AXI wrapper
│   │   │   └── top/            # Top-level modules
│   │   ├── sim/                # Simulation scripts
│   │   └── tb/                 # Testbenches
│   ├── hls/                    # High-Level Synthesis (C++)
│   │   ├── src/                # HLS source files
│   │   ├── include/            # Headers
│   │   ├── scripts/            # Build automation
│   │   └── test/               # HLS testbenches
│   ├── constraints/            # Timing and pin constraints
│   └── scripts/                # Build scripts (TCL)
│
├── software/python/            # Python software stack
│   └── snn_fpga_accelerator/   # Main package
│       ├── accelerator.py      # FPGA interface
│       ├── pytorch_interface.py # PyTorch integration
│       ├── spike_encoding.py   # Spike encoders/decoders
│       ├── learning.py         # STDP/R-STDP
│       └── pytorch_snn_layers.py # Custom PyTorch layers
│
├── examples/                   # Usage examples
│   ├── pytorch/                # PyTorch training examples
│   └── notebooks/              # Jupyter notebooks
│
├── tests/                      # Test suite
└── docs/                       # Documentation
```

## Contributing Guidelines

### Types of Contributions

We welcome various types of contributions:

1. **Bug Fixes**: Fix issues in existing code
2. **New Features**: Add new functionality
3. **Documentation**: Improve or add documentation
4. **Tests**: Add or improve test coverage
5. **Performance**: Optimize existing code
6. **Examples**: Add new usage examples

### Finding Issues to Work On

- Check the [GitHub Issues](https://github.com/metr0jw/Spiking-Neural-Network-on-FPGA/issues) page
- Look for issues tagged `good first issue` or `help wanted`
- Feel free to propose new features by opening an issue first

### Before Starting Work

1. **Check for existing work**: Search issues and PRs to avoid duplicate effort
2. **Discuss major changes**: For significant features, open an issue first
3. **Create a branch**: Use a descriptive branch name
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/issue-number-description
   ```

## Hardware Development

For comprehensive hardware development guidelines, RTL coding standards, HLS best practices, and detailed examples, see the [Developer Guide](docs/developer_guide.md).

### Quick Reference

**RTL Guidelines**:
- Use Verilog-2001 for portability
- Add `ifndef __ICARUS__` guards for SystemVerilog features
- Follow naming conventions: `snake_case` for signals, `UPPER_CASE` for parameters

**HLS Guidelines**:
- Include appropriate `#pragma HLS` directives
- Write comprehensive testbenches
- Test with C simulation and cosimulation

See [Developer Guide - Hardware Development](docs/developer_guide.md#hardware-development) for complete details.

## Software Development

For comprehensive Python development guidelines, see the [Developer Guide](docs/developer_guide.md).

### Coding Style

Follow **PEP 8** and use type hints for all functions. See [Developer Guide - Python Development](docs/developer_guide.md#python-development) for complete guidelines and examples.

### Quick Reference
- Use type hints for function signatures
- Write Google-style docstrings
- Provide clear error messages
- Follow naming conventions

Example:
```python
def encode_spikes(
    data: np.ndarray,
    max_rate: float = 100.0
) -> np.ndarray:
    """Encode data as spike trains.
    
    Args:
        data: Input data normalized to [0, 1]
        max_rate: Maximum firing rate in Hz
        
    Returns:
        Spike trains array
    """
    pass
```

## Testing Requirements

### Python Tests

All Python code must have tests using **pytest**:

```python
# test_spike_encoding.py
import pytest
import numpy as np
from snn_fpga_accelerator.spike_encoding import PoissonEncoder

def test_poisson_encoder_reproducibility():
    """Test that encoder produces reproducible results with seed."""
    encoder1 = PoissonEncoder(num_neurons=10, duration=0.1, seed=42)
    encoder2 = PoissonEncoder(num_neurons=10, duration=0.1, seed=42)
    
    data = np.random.rand(10)
    spikes1 = encoder1.encode(data)
    spikes2 = encoder2.encode(data)
    
    assert np.array_equal(spikes1, spikes2), "Results not reproducible"

def test_poisson_encoder_spike_rate():
    """Test that encoder respects max_rate."""
    encoder = PoissonEncoder(num_neurons=10, duration=1.0, max_rate=100.0)
    data = np.ones(10)  # Maximum input
    spikes = encoder.encode(data)
    
    spike_rate = np.sum(spikes, axis=1).mean()
    assert spike_rate <= 100.0, f"Spike rate {spike_rate} exceeds max_rate"

@pytest.mark.parametrize("num_neurons", [1, 10, 100])
def test_poisson_encoder_shapes(num_neurons):
    """Test encoder output shapes for various configurations."""
    encoder = PoissonEncoder(num_neurons=num_neurons, duration=0.1)
    data = np.random.rand(num_neurons)
    spikes = encoder.encode(data)
    
    assert spikes.shape[0] == num_neurons
```

### Running Tests

```bash
# Run all tests
pytest software/python/tests -v

# Run with coverage
pytest software/python/tests --cov=snn_fpga_accelerator --cov-report=html

# Run specific test file
pytest software/python/tests/test_spike_encoding.py

# Run tests matching pattern
pytest -k "encoder"
```

### Hardware Tests

```bash
# Run RTL simulations
cd hardware/hdl/sim
./run_sim_working.sh tb_simple_lif

# Run HLS tests
cd hardware/hls/test
./run_tests.sh
```

## Documentation Standards

### Code Comments

- **What, not how**: Explain why, not what the code does
- **Document assumptions**: Especially for hardware constraints
- **Update with changes**: Keep comments in sync with code

```python
# Good comment
# Use exponential moving average to smooth reward signal
# This prevents weight oscillations during RL training
reward_ema = 0.9 * reward_ema + 0.1 * current_reward

# Bad comment
# Multiply reward_ema by 0.9 and add 0.1 times current_reward
```

### Markdown Documentation

- Use clear headings and structure
- Include code examples
- Add links to related documentation
- Keep language simple and concise

### API Documentation

All public APIs must be documented:

```python
class SNNAccelerator:
    """FPGA-based spiking neural network accelerator.
    
    This class provides a high-level interface for configuring and
    running SNN inference on FPGA hardware.
    
    Attributes:
        num_neurons: Number of neurons in the network
        bitstream_path: Path to FPGA bitstream file
        is_configured: Whether the network is configured
        
    Example:
        >>> accelerator = SNNAccelerator(bitstream_path="snn.bit")
        >>> accelerator.configure_network(network_config)
        >>> output = accelerator.infer(input_spikes)
    """
    pass
```

## Pull Request Process

### 1. Prepare Your Changes

```bash
# Ensure your branch is up to date
git fetch upstream
git rebase upstream/main

# Run tests
pytest software/python/tests
cd hardware/hdl/sim && ./run_sim_working.sh tb_simple_lif

# Format code
black software/python/snn_fpga_accelerator
black software/python/tests

# Check style
flake8 software/python/snn_fpga_accelerator
```

### 2. Commit Your Changes

Use clear, descriptive commit messages:

```bash
# Good commit messages
git commit -m "Add temporal spike encoder for rate-based encoding"
git commit -m "Fix width mismatch in spike router neuron ID signals"
git commit -m "Update README with hardware testing instructions"

# Bad commit messages
git commit -m "fix bug"
git commit -m "updates"
git commit -m "WIP"
```

Follow this format for detailed commits:
```
Short summary (50 chars or less)

More detailed explanation if needed. Wrap at 72 characters.
Explain the problem this commit solves and why this approach
was chosen.

- Bullet points are okay
- Use present tense: "Add feature" not "Added feature"

Fixes #123
```

### 3. Push and Create PR

```bash
# Push to your fork
git push origin feature/your-feature-name

# Create PR on GitHub
# Use the PR template and fill in all sections
```

### 4. PR Guidelines

Your PR should include:

- **Clear title**: Summarize the change
- **Description**: Explain what and why
- **Tests**: Include tests for new functionality
- **Documentation**: Update docs as needed
- **Changelog**: Note any breaking changes

**PR Template**:
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Testing
- [ ] All existing tests pass
- [ ] New tests added for new functionality
- [ ] Hardware simulation tested (if applicable)

## Documentation
- [ ] README updated
- [ ] API documentation updated
- [ ] Code comments added/updated

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] No compiler warnings
- [ ] Changes are backwards compatible (or breaking changes noted)
```

### 5. Code Review Process

- Be responsive to feedback
- Make requested changes promptly
- Discuss concerns constructively
- Update PR based on reviews

## Coding Standards

### General Principles

1. **Readability**: Code is read more than written
2. **Simplicity**: Prefer simple solutions
3. **Modularity**: Keep functions/modules focused
4. **DRY**: Don't Repeat Yourself
5. **Testing**: Test your code thoroughly

### Performance Considerations

- **Profile first**: Don't optimize prematurely
- **Document tradeoffs**: Explain performance-related decisions
- **Benchmark**: Provide measurements for optimizations

### Version Control

- **Small commits**: One logical change per commit
- **Meaningful messages**: Explain what and why
- **Clean history**: Rebase/squash when appropriate

## Community

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and ideas
- **Pull Requests**: Code review and collaboration
- **Email**: [jwlee@linux.com](mailto:jwlee@linux.com)

### Getting Help

- Check existing documentation and examples
- Search GitHub issues for similar problems
- Ask questions in GitHub Discussions
- Be specific about your problem and what you've tried

### Acknowledgments

Contributors will be acknowledged in:
- Repository contributors page
- Release notes for significant contributions
- README.md (for major features)

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to the Event-Driven SNN FPGA Accelerator project! Your contributions help make neuromorphic computing more accessible to researchers and developers.
