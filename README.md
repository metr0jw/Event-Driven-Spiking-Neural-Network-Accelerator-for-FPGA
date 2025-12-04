# Event-Driven Spiking Neural Network Accelerator for FPGA

**Energy-efficient Event-driven Spiking Neural Network accelerator for FPGA with PyTorch integration**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Development Environment
- **FPGA Tools**: Vivado 2025.2, Vitis HLS 2025.2  
- **Target Hardware**: PYNQ-Z2 (Xilinx Zynq-7000, xc7z020clg400-1)
- **Software**: Python 3.13, PyTorch 2.9.0, PYNQ 2.7
- **HDL Standard**: Verilog-2001 (IEEE 1364-2001)
- **Simulation**: Icarus Verilog, Cocotb

## Description

This project implements a complete event-driven spiking neural network (SNN) accelerator on FPGA that integrates seamlessly with PyTorch for training and deployment. Unlike traditional clock-driven accelerators, this design uses event-driven processing that is more familiar to software engineers and more efficient for sparse neural computations.

### Key Features

**Biologically-Inspired Neurons**
- Leaky Integrate-and-Fire (LIF) neuron model
- Support for both excitatory and inhibitory neurons
- Configurable membrane dynamics and refractory periods
- Hardware-optimized fixed-point arithmetic

**Energy-Efficient AC-Based Architecture**
- Accumulate-only (AC) operations instead of Multiply-Accumulate (MAC)
- ~5x energy reduction per synaptic operation (0.9pJ vs 4.6pJ at 45nm)
- Exploits binary spike nature: spike x weight = weight (when spike=1)
- Sparse processing: 90-99% operations skipped when spike=0
- Real-time energy monitoring module

**Advanced Learning Algorithms**
- Standard STDP (Spike-Timing Dependent Plasticity)
- R-STDP (Reward-modulated STDP) for reinforcement learning
- Triplet STDP for enhanced stability (TODO)
- Adaptive learning rate mechanisms (TODO)

**Event-Driven Architecture**
- Asynchronous spike processing
- AXI-Stream interfaces for efficient data flow
- Time-multiplexed neuron arrays for scalability
- Optimized for sparse neural activity

**PyTorch Integration**
- Seamless model conversion from PyTorch to SNN
- Multiple spike encoding schemes (Poisson, temporal, rate-based)
- Hardware-in-the-loop training support
- Gradient-free STDP training loops for quick experimentation
- Comprehensive visualization tools

**Simulation Mode**
- Full software simulation without FPGA hardware
- Cycle-accurate behavioral model for development and debugging
- Seamless transition to hardware deployment

## Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   PyTorch       │    │   FPGA SNN       │    │   Output        │
│   Training      │───▶│   Accelerator    │───▶│   Processing    │
│                 │    │                  │    │                 │
│ ┌─────────────┐ │    │ ┌──────────────┐ │    │ ┌─────────────┐ │
│ │ SNN Models  │ │    │ │ LIF Neurons  │ │    │ │ Spike       │ │
│ │ STDP/R-STDP │ │    │ │ AC Synapses  │ │    │ │ Decoding    │ │
│ │ Encoding    │ │    │ │ Conv/FC/Pool │ │    │ │ Analysis    │ │
│ └─────────────┘ │    │ └──────────────┘ │    │ └─────────────┘ │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Energy Efficiency: AC vs MAC

This accelerator uses **Accumulate-only (AC)** operations instead of traditional **Multiply-Accumulate (MAC)** operations, significantly reducing energy consumption.

### Why AC Works for SNNs

In SNNs, spikes are binary (0 or 1):
- When spike = 0: No computation needed (sparse optimization)
- When spike = 1: `spike × weight = weight` (no multiplication required)

### Energy Comparison (45nm Technology)

| Operation | Energy | Relative |
|-----------|--------|----------|
| MAC (32-bit FP) | ~4.6 pJ | 1.0x |
| AC (16-bit INT) | ~0.9 pJ | 0.2x |

### Total Energy Savings

```
ANN Energy:  N_ops × 4.6pJ (MAC)
SNN Energy:  N_ops × sparsity × 0.9pJ (AC)

With 5% spike activity: ~100x energy reduction
With 10% spike activity: ~50x energy reduction
```

## Implementation Status

### Completed Features
- [x] **LIF Neuron Core**: Hardware implementation with excitatory/inhibitory support
- [x] **AC-Based Neurons**: Energy-efficient accumulate-only neuron arrays
- [x] **AC-Based Synapses**: Sparse synapse arrays with sparsity exploitation
- [x] **Layer Support**: Conv1D, Conv2D, FC, MaxPool, AvgPool (all AC-based)
- [x] **Energy Monitor**: Real-time energy estimation module
- [x] **STDP Learning**: Standard spike-timing dependent plasticity (HLS)
- [x] **R-STDP Learning**: Reward-modulated plasticity for RL (HLS)
- [x] **PyTorch Interface**: Model conversion and weight loading
- [x] **Spike Encoding**: Poisson, temporal, and rate-based encoders
- [x] **Comprehensive Testbenches**: 12/12 testbenches passing
- [x] **Python Software Stack**: Complete API and utilities
- [x] **Training Examples**: MNIST classification and RL navigation
- [x] **Communication Interface**: AXI-based PC communication
- [x] **Bitstream Generation**: Successfully synthesized for PYNQ-Z2
- [x] **PYNQ Driver**: Python driver for hardware control
- [x] **Dynamic Weight Setting**: Runtime weight configuration via AXI

### Hardware Build Status
| Metric | Value | Status |
|--------|-------|--------|
| **WNS (Setup Slack)** | +0.159 ns | PASS |
| **WHS (Hold Slack)** | +0.057 ns | PASS |
| **Clock Frequency** | 100 MHz | OK |
| **Timing Violations** | 0 | OK |
| **LUT Utilization** | 4,689 / 53,200 (8.81%) | OK |
| **Register Utilization** | 3,212 / 106,400 (3.02%) | OK |
| **Slice Utilization** | 1,620 / 13,300 (12.18%) | OK |

### In Progress
- [ ] **Advanced Connectivity**: Recurrent layer support
- [ ] **Power Optimization**: Further power reduction
- [ ] **Real-time Demos**: Live inference applications

## Quick Start

### Installation
```bash
# Clone the repository
git clone https://github.com/metr0jw/Event-Driven-Spiking-Neural-Network-Accelerator-for-FPGA.git
cd Event-Driven-Spiking-Neural-Network-Accelerator-for-FPGA

# Activate environment (if using provided virtualenv)
source venv/bin/activate

# Run automated setup
./setup.sh
```

### Basic Usage
```bash
# Quick demonstration
python quick_start.py

# PyTorch training example (STDP-based)
python examples/pytorch/mnist_training_example.py

# R-STDP learning example
python examples/pytorch/r_stdp_learning_example.py

# Run all RTL testbenches (12 tests)
cd hardware/hdl/sim && ./run_all_tests.sh
```

### Hardware Deployment (PYNQ-Z2)
```python
# On PYNQ board
from snn_driver import SNNAccelerator

# Load bitstream
snn = SNNAccelerator('snn_accelerator.bit')

# Configure neurons
snn.configure(threshold=100, leak_rate=16, refractory_period=5)

# Enable and run
snn.enable()
status = snn.get_status()
print(f"Spike count: {status['spike_count']}")

snn.close()
```

For detailed usage instructions, see the [User Guide](docs/user_guide.md).

## Usage Examples

See the [User Guide](docs/user_guide.md) for comprehensive usage examples and tutorials.

### Quick Example
```python
from snn_fpga_accelerator import SNNAccelerator

# Initialize and configure
accelerator = SNNAccelerator(bitstream_path="hardware/bitstream.bit")
accelerator.configure_network(network_config)

# Run inference
output = accelerator.infer(input_spikes)
```

More examples available in the `examples/` directory.

## Python Library (PyTorch-like API)

The `snn_fpga_accelerator` package provides a **PyTorch-like API** for building, training, and deploying SNNs, similar to snnTorch and SpikingJelly.

### Key Features
- **Surrogate Gradient Training**: Backprop through spiking neurons using FastSigmoid, ATan, SuperSpike, etc.
- **LIF Neurons as Activations**: Use `snn.LIF()` like `nn.ReLU()`
- **Hardware-Aware Training**: `hw_mode=True` enforces 8-bit weight constraints during training
- **FPGA Deployment**: One-line export with quantization

### Quick Example
```python
import snn_fpga_accelerator as snn
import torch
import torch.nn as nn

# Build model (just like PyTorch!)
class MySNN(nn.Module):
    def __init__(self, num_steps=25):
        super().__init__()
        self.num_steps = num_steps
        self.fc1 = nn.Linear(784, 256)
        self.lif1 = snn.LIF(thresh=1.0, tau=0.9)  # LIF as activation
        self.fc2 = nn.Linear(256, 10)
        self.lif2 = snn.LIF(thresh=1.0, tau=0.9)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        self.lif1.reset_state()
        self.lif2.reset_state()
        
        spk_rec = []
        for t in range(self.num_steps):
            spk1 = self.lif1(self.fc1(x))
            spk2 = self.lif2(self.fc2(spk1))
            spk_rec.append(spk2)
        return torch.stack(spk_rec).sum(0)

# Train with standard PyTorch!
model = MySNN()
optimizer = torch.optim.Adam(model.parameters())
loss = nn.CrossEntropyLoss()(model(data), labels)
loss.backward()  # Surrogate gradients flow through spikes!
optimizer.step()
```

### Available Neurons
```python
snn.LIF(thresh=1.0, tau=0.9)  # Leaky Integrate-and-Fire
snn.IF(thresh=1.0)            # Integrate-and-Fire (no leak)
snn.PLIF()                    # Parametric LIF (learnable tau)
snn.ALIF()                    # Adaptive LIF (adaptive threshold)
snn.Izhikevich()              # Biologically realistic
```

### Surrogate Gradient Options
```python
snn.LIF(surrogate='fast_sigmoid')  # Default, fastest
snn.LIF(surrogate='atan')          # Arctangent
snn.LIF(surrogate='super_spike')   # SuperSpike
snn.LIF(surrogate='sigmoid')       # Sigmoid
snn.LIF(surrogate='pwl')           # Piecewise Linear
```

### FPGA Deployment
```python
# Quantize and export
weights = snn.quantize(model.state_dict(), bits=8)
snn.deploy.export(model, 'weights.npz')
```

See `software/python/README.md` for full API documentation.

## Project Structure

```
├── hardware/                    # FPGA implementation
│   ├── hdl/rtl/                # Verilog RTL sources
│   │   ├── neurons/            # LIF neuron implementations (AC-based)
│   │   ├── synapses/           # Synaptic weight memory (AC-based)
│   │   ├── layers/             # Conv1D/2D, FC, Pooling layers
│   │   ├── router/             # Spike routing logic
│   │   ├── common/             # Utilities, Energy monitor
│   │   └── top/                # Top-level integration
│   ├── hdl/tb/                 # Comprehensive testbenches
│   ├── hls/                    # High-Level Synthesis
│   │   ├── src/                # Learning algorithms
│   │   ├── include/            # Header files
│   │   └── test/               # HLS testbenches
│   └── scripts/                # Build automation
├── software/python/            # Python software stack
│   └── snn_fpga_accelerator/   # Main package
│       ├── accelerator.py      # FPGA interface
│       ├── pytorch_interface.py # PyTorch integration
│       ├── spike_encoding.py   # Encoding algorithms
│       ├── learning.py         # STDP/R-STDP
│       └── utils.py            # Utilities
├── examples/                   # Usage examples
│   ├── pytorch/                # PyTorch examples
│   └── notebooks/              # Jupyter notebooks
└── docs/                       # Documentation
```

## Key Components

For detailed architecture information, see the [Architecture Documentation](docs/architecture.md).

### LIF Neuron Model (AC-Based)
- Membrane integration with shift-based leak (no multiply)
- Accumulate-only synaptic integration
- Configurable refractory periods
- Saturation arithmetic for fixed-point implementation

### AC-Based Synapse Array
- Sparse processing: skips spike=0 inputs entirely
- Weight accumulation without multiplication
- Built-in energy monitoring counters

### Layer Implementations
- **snn_conv1d_ac.v**: 1D convolution for temporal data
- **snn_conv2d_ac.v**: 2D convolution for image processing
- **snn_fc_ac.v**: Fully connected layer for classification
- **snn_maxpool1d/2d.v**: Max pooling layers
- **snn_avgpool1d/2d.v**: Average pooling layers

### STDP Learning Rule
- Long-term potentiation (LTP) and depression (LTD)
- Eligibility traces for reward-modulated learning
- Configurable time constants and learning rates

### Energy Monitor
- Real-time AC operation counting
- Memory access tracking
- ANN vs SNN energy comparison

## Performance Characteristics

| Metric | Current Implementation | Notes |
|--------|------------------------|-------|
| **Max Neurons** | 64 neurons (scalable via time-multiplexing) | BRAM-limited, expandable to 1024+ |
| **Clock Frequency** | 100 MHz | Timing verified with +0.159ns slack |
| **LUT Utilization** | 8.81% (4,689 / 53,200) | Plenty of room for expansion |
| **Register Utilization** | 3.02% (3,212 / 106,400) | Minimal register usage |
| **BRAM Usage** | 2 × BRAM36K | Weight memory + spike FIFO |
| **Power (Board)** | ~2 W on PYNQ-Z2 | Estimate from Vivado power analysis |
| **Energy Efficiency** | ~5x vs MAC-based, ~100x with sparsity | AC operations + sparse processing |
| **Numeric Precision** | 8-bit weights, 16-bit membrane | Fixed-point format |

### Documentation
- **User Guide**: Updated with CNN conversion examples and step mode usage
- **IMPROVEMENTS.md**: Detailed summary of architectural decisions
- **AXI_INTERFACE_BEST_PRACTICES.md**: Guide for safer AXI interface development
- **COCOTB_INTEGRATION_GUIDE.md**: RTL verification strategy with Python testbenches

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Workflow
1. **Fork** the repository
2. **Create** a feature branch
3. **Implement** your changes with tests
4. **Submit** a pull request

## Documentation

- [User Guide](docs/user_guide.md): Comprehensive usage instructions and tutorials
- [Architecture](docs/architecture.md): System design and component details
- [API Reference](docs/api_reference.md): Complete API documentation
- [Developer Guide](docs/developer_guide.md): Development setup and guidelines
- [Contributing](CONTRIBUTING.md): Contribution guidelines

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{snn_fpga_accelerator,
  title={Event-Driven Spiking Neural Network Accelerator for FPGA},
  author={Jiwoon Lee},
  year={2025},
  howpublished={\url{https://github.com/metr0jw/Event-Driven-Spiking-Neural-Network-Accelerator-for-FPGA}}
}
```

## References

- Energy efficiency analysis based on: Horowitz, M. "Computing's Energy Problem (and what we can do about it)" ISSCC 2014
- SNN sparsity benefits: Roy, K. et al. "Towards spike-based machine intelligence with neuromorphic computing" Nature 2019

## Troubleshooting

For common issues and solutions, see the [User Guide - Troubleshooting](docs/user_guide.md#troubleshooting) and [Developer Guide - Common Issues](docs/developer_guide.md#troubleshooting).

## Support

- **Issues**: [GitHub Issues](https://github.com/metr0jw/Event-Driven-Spiking-Neural-Network-Accelerator-for-FPGA/issues)
- **Email**: [jwlee@linux.com](mailto:jwlee@linux.com)
- **Documentation**: [Project Wiki](https://github.com/metr0jw/Event-Driven-Spiking-Neural-Network-Accelerator-for-FPGA/wiki)

## License
[MIT License](LICENSE)

## Author
[Jiwoon Lee](https://github.com/metr0jw)
