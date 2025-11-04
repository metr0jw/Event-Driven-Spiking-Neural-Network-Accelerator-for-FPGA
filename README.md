# Event-Driven Spiking Neural Network Accelerator for FPGA

**Event-driven Spiking Neural Network accelerator for FPGA with PyTorch integration**

## Development Environment
- **FPGA Tools**: Vivado 2025.1, Vitis HLS 2025.1  
- **Target Hardware**: PYNQ-Z2 (Xilinx Zynq-7000)
- **Software**: Python 3.13+, PyTorch 2.9.0+, PYNQ 2.7+

## Description

This project implements a complete event-driven spiking neural network (SNN) accelerator on FPGA that integrates seamlessly with PyTorch for training and deployment. Unlike traditional clock-driven accelerators, this design uses event-driven processing that is more familiar to software engineers and more efficient for sparse neural computations.

### Key Features

 **Biologically-Inspired Neurons**
- Leaky Integrate-and-Fire (LIF) neuron model
- Support for both excitatory and inhibitory neurons
- Configurable membrane dynamics and refractory periods
- Hardware-optimized fixed-point arithmetic

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
- Comprehensive visualization tools

## Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   PyTorch       │    │   FPGA SNN       │    │   Output        │
│   Training      │───▶│   Accelerator    │───▶│   Processing    │
│                 │    │                  │    │                 │
│ ┌─────────────┐ │    │ ┌──────────────┐ │    │ ┌─────────────┐ │
│ │ SNN Models  │ │    │ │ LIF Neurons  │ │    │ │ Spike       │ │
│ │ STDP/R-STDP │ │    │ │ STDP Engine  │ │    │ │ Decoding    │ │
│ │ Encoding    │ │    │ │ AXI Interface│ │    │ │ Analysis    │ │
│ └─────────────┘ │    │ └──────────────┘ │    │ └─────────────┘ │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Implementation Status

### Completed Features
- [x] **LIF Neuron Core**: Hardware implementation with excitatory/inhibitory support
- [x] **Neuron Arrays**: Time-multiplexed arrays for scalable processing
- [x] **STDP Learning**: Standard spike-timing dependent plasticity
- [x] **R-STDP Learning**: Reward-modulated plasticity for RL
- [x] **PyTorch Interface**: Model conversion and weight loading
- [x] **Spike Encoding**: Poisson, temporal, and rate-based encoders
- [x] **Comprehensive Testbenches**: Event-driven verification
- [x] **Python Software Stack**: Complete API and utilities
- [x] **Training Examples**: MNIST classification and RL navigation
- [x] **Communication Interface**: AXI-based PC communication

### In Progress
- [ ] **Fix bugs in existing features**
- [ ] **Triplet STDP**: Enhanced learning stability
- [ ] **Advanced Connectivity**: Recurrent and convolutional layer support
- [ ] **Optimized Bitstreams**: Power and performance optimization
- [ ] **Real-time Demos**: Live inference applications

## Quick Start

### Installation
```bash
# Clone the repository
git clone https://github.com/metr0jw/Spiking-Neural-Network-on-FPGA.git
cd Spiking-Neural-Network-on-FPGA

# Run automated setup
./setup.sh

# Activate environment (if using provided virtualenv)
source venv/bin/activate
```

### Basic Usage
```bash
# Quick demonstration
python quick_start.py

# PyTorch training example
python examples/pytorch/mnist_training_example.py

# R-STDP learning example
python examples/pytorch/r_stdp_learning_example.py
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

## Project Structure

```
├── hardware/                    # FPGA implementation
│   ├── hdl/rtl/                # Verilog RTL sources
│   │   ├── neurons/            # LIF neuron implementations
│   │   ├── synapses/           # Synaptic weight memory
│   │   ├── router/             # Spike routing logic
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

### LIF Neuron Model
- Membrane integration with leak and spike generation
- Configurable refractory periods
- Saturation arithmetic for fixed-point implementation

### STDP Learning Rule
- Long-term potentiation (LTP) and depression (LTD)
- Eligibility traces for reward-modulated learning
- Configurable time constants and learning rates

### Event-Driven Processing
- Asynchronous spike-based computation
- AXI-Stream protocol for data flow
- Temporal priority queues

## Performance Characteristics

| Metric | Current Implementation | Planned/Projected | Notes |
|--------|------------------------|-------------------|-------|
| **Max Neurons** | 64 neurons synthesized and verified in RTL simulation | ≥1024 neurons via time-multiplexed expansion | Scaling gated by BRAM budget and router partitioning |
| **Spike Throughput** | Characterization in progress under Icarus/Vivado sim benches | ≥100K spikes/s at 100 MHz fabric clock | Projection assumes one spike accepted per cycle after router optimizations |
| **End-to-End Latency** | Microsecond-scale in-cycle pipeline (pending hardware timestamping) | <10 µs per spike including AXI transfer | Latency dominated by AXI ingress + neuron update stages |
| **Power (Board)** | TBD (hardware bring-up scheduled) | ~2 W on PYNQ-Z2 during inference workload | Estimate from Vivado power analysis with 20% toggle rate |
| **Numeric Precision** | 8-bit weights, 16-bit membrane potential | Higher precision paths under evaluation | Fixed-point format matches current RTL + software stack |

Bench characterization will be updated once on-board measurements complete.

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Workflow
1. **Fork** the repository
2. **Create** a feature branch
3. **Implement** your changes with tests
4. **Submit** a pull request

## Documentation

- 📖 **[User Guide](docs/user_guide.md)**: Comprehensive usage instructions and tutorials
- 🏗️ **[Architecture](docs/architecture.md)**: System design and component details
- 📚 **[API Reference](docs/api_reference.md)**: Complete API documentation
- 🔧 **[Developer Guide](docs/developer_guide.md)**: Development setup and guidelines
- 🤝 **[Contributing](CONTRIBUTING.md)**: Contribution guidelines

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{snn_fpga_accelerator,
  title={Event-Driven Spiking Neural Network Accelerator for FPGA},
  author={Jiwoon Lee},
  year={2025},
  howpublished={\url{https://github.com/metr0jw/Spiking-Neural-Network-on-FPGA}}
}
```

## Troubleshooting

For common issues and solutions, see the [User Guide - Troubleshooting](docs/user_guide.md#troubleshooting) and [Developer Guide - Common Issues](docs/developer_guide.md#troubleshooting).

## Support

- **Issues**: [GitHub Issues](https://github.com/metr0jw/Spiking-Neural-Network-on-FPGA/issues)
- **Email**: [jwlee@linux.com](mailto:jwlee@linux.com)
- **Documentation**: [Project Wiki](https://github.com/metr0jw/Spiking-Neural-Network-on-FPGA/wiki)

## License
[MIT License](LICENSE)

## Author
[Jiwoon Lee](https://github.com/metr0jw)
