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

### 1. One-Command Setup
```bash
# Clone and setup everything
git clone https://github.com/metr0jw/Event-Driven-Spiking-Neural-Network-Accelerator-for-FPGA.git
cd Event-Driven-Spiking-Neural-Network-Accelerator-for-FPGA
./setup.sh
```

### 2. Activate Environment (Optional, create your own virtualenv if needed)
```bash
source venv/bin/activate
```

### 3. Run Examples
```bash
# Quick demonstration
python quick_start.py

# Complete integration example
python examples/complete_integration_example.py --simulation-mode

# PyTorch training and deployment
python examples/pytorch/mnist_snn_training.py

# R-STDP reinforcement learning
python examples/pytorch/rstdp_navigation_task.py
```

## Usage Examples

### Basic Usage
```python
from snn_fpga_accelerator import SNNAccelerator

# Initialize accelerator
accelerator = SNNAccelerator(bitstream_path="hardware/bitstream.bit")

# Load your trained PyTorch model
network_config = pytorch_to_snn(your_pytorch_model)
accelerator.configure_network(network_config)

# Run inference
output = accelerator.infer(input_spikes)
```

### Advanced PyTorch Integration
```python
import torch
from snn_fpga_accelerator import pytorch_to_snn, SNNAccelerator

# Train your PyTorch SNN model
model = create_your_snn_model()
train_model(model, train_loader)

# Convert and deploy to FPGA
accelerator = SNNAccelerator()
network_config = pytorch_to_snn(model)
accelerator.configure_network(network_config)
accelerator.load_weights(network_config['weights'])

# Run real-time inference
for data, target in test_loader:
    spikes = spike_encoding.rate_encode(data.numpy())
    output = accelerator.infer(spikes[0])
    prediction = np.argmax(output)
```

### Online Learning with R-STDP
```python
from snn_fpga_accelerator.learning import RSTDPLearning

# Initialize R-STDP learning
rstdp = RSTDPLearning(learning_rate=0.01, eligibility_decay=0.95)
accelerator.configure_learning(rstdp.get_config())

# Training loop with rewards
for episode in range(num_episodes):
    for input_data, target in training_data:
        # Forward pass
        output = accelerator.infer_with_learning(input_data)
        
        # Calculate reward
        reward = calculate_reward(output, target)
        
        # Apply reward signal for learning
        accelerator.apply_reward(reward)
```

### Manual Setup (Alternative)

### 1. Clone Repository
```bash
git clone https://github.com/metr0jw/Spiking-Neural-Network-on-FPGA.git
cd Spiking-Neural-Network-on-FPGA
```

### 2. Install Software Dependencies
```bash
# Install Python package
cd software/python
pip install -e .

# Install additional dependencies for examples
pip install torch torchvision tqdm matplotlib h5py
```

### 3. Build Hardware (FPGA)

#### Quick Build (Automated)

The project includes a comprehensive Makefile for automated builds:

```bash
# Build everything (HLS + Vivado + Programming)
make all

# Build HLS components only
make hls

# Build Vivado project and generate bitstream
make vivado

# Program FPGA (PYNQ-Z2)
make program

# Clean all build files
make clean

# Clean and rebuild everything
make clean-all
```

**Available Make Targets:**
- `make hls-clean`: Clean HLS build files
- `make hls-build`: Build all HLS IP cores
- `make hls-test`: Run HLS testbenches
- `make vivado-clean`: Clean Vivado build files  
- `make vivado-synth`: Run synthesis only
- `make vivado-impl`: Run implementation only
- `make vivado-bitstream`: Generate bitstream only
- `make program-jtag`: Program via JTAG cable
- `make program-pynq`: Program via PYNQ network
- `make test-hardware`: Run hardware validation tests

#### Detailed FPGA Development Guide

For detailed FPGA development, you can use the provided TCL scripts for step-by-step project creation:

##### High-Level Synthesis (HLS) Project Setup
```bash
# Create HLS project for learning algorithms
vivado_hls -f create_hls_project.tcl

# Or manually create HLS project
cd hardware/hls
vivado_hls -f scripts/create_project.tcl
vivado_hls -f scripts/run_synthesis.tcl
vivado_hls -f scripts/export_ip.tcl
```

The HLS project includes:
- **Spike-Timing Dependent Plasticity (STDP)** learning engine
- **R-STDP** reward-modulated learning 
- **Weight update** algorithms
- **Spike encoding/decoding** modules

##### Vivado Project Setup
```bash
# Create complete Vivado project
vivado -mode batch -source create_vivado_project.tcl

# Or step-by-step project creation
cd hardware/scripts
vivado -mode batch -source create_project.tcl
vivado -mode batch -source create_block_design.tcl
vivado -mode batch -source build_bitstream.tcl
```

##### Manual FPGA Development Workflow

**Step 1: Environment Setup**
```bash
# Source Vivado tools (adjust path for your installation)
source /opt/Xilinx/Vivado/2025.1/settings64.sh
source /opt/Xilinx/Vitis_HLS/2025.1/settings64.sh

# Set environment variables
export XILINX_VIVADO=/opt/Xilinx/Vivado/2025.1
export XILINX_HLS=/opt/Xilinx/Vitis_HLS/2025.1
```

**Step 2: HLS Component Generation**
```bash
cd hardware/hls

# Build STDP learning engine
vivado_hls -f scripts/create_project.tcl -tclargs stdp_learning
vivado_hls -f scripts/run_synthesis.tcl -tclargs stdp_learning
vivado_hls -f scripts/export_ip.tcl -tclargs stdp_learning

# Build spike encoder
vivado_hls -f scripts/create_project.tcl -tclargs spike_encoder
vivado_hls -f scripts/run_synthesis.tcl -tclargs spike_encoder  
vivado_hls -f scripts/export_ip.tcl -tclargs spike_encoder

# Build weight updater
vivado_hls -f scripts/create_project.tcl -tclargs weight_updater
vivado_hls -f scripts/run_synthesis.tcl -tclargs weight_updater
vivado_hls -f scripts/export_ip.tcl -tclargs weight_updater
```

**Step 3: RTL Synthesis and Implementation**
```bash
cd hardware/scripts

# Create Vivado project
vivado -mode batch -source create_project.tcl

# Add RTL sources and constraints  
vivado -mode batch -source add_sources.tcl

# Create block design with HLS IP
vivado -mode batch -source create_block_design.tcl

# Run synthesis
vivado -mode batch -source run_synthesis.tcl

# Run implementation  
vivado -mode batch -source run_implementation.tcl

# Generate bitstream
vivado -mode batch -source build_bitstream.tcl
```

**Step 4: Programming and Testing**
```bash
# Program FPGA via JTAG
vivado -mode batch -source program_board.tcl

# Or program via PYNQ
scp hardware/build/snn_accelerator.bit xilinx@pynq_ip:/home/xilinx/
ssh xilinx@pynq_ip
sudo python3 -c "
from pynq import Overlay
overlay = Overlay('/home/xilinx/snn_accelerator.bit')
print('FPGA programmed successfully!')
"
```

##### Project Structure for FPGA Development
```
hardware/
├── create_hls_project.tcl        # HLS project creation script
├── create_vivado_project.tcl     # Vivado project creation script  
├── hdl/
│   ├── rtl/                      # RTL source files
│   │   ├── common/               # Common utilities (FIFO, sync, etc.)
│   │   ├── neurons/              # LIF neuron implementations
│   │   ├── synapses/             # Synaptic weight memory
│   │   ├── router/               # Spike routing logic
│   │   ├── interfaces/           # AXI wrapper interfaces
│   │   ├── convolution/          # Conv1D/Conv2D layers
│   │   ├── pooling/              # Pooling layers
│   │   ├── layers/               # Layer management
│   │   └── top/                  # Top-level integration
│   ├── sim/                      # Simulation scripts
│   └── tb/                       # Testbenches
├── hls/                          # High-Level Synthesis
│   ├── src/                      # HLS C++ sources
│   ├── include/                  # Header files
│   ├── scripts/                  # HLS automation scripts
│   └── test/                     # HLS testbenches
├── constraints/                  # Timing and pin constraints
│   ├── pynq_z2_pins.xdc         # PYNQ-Z2 pin assignments
│   ├── timing.xdc               # Timing constraints
│   └── bitstream.xdc            # Bitstream generation settings
└── scripts/                     # Build automation
    ├── create_project.tcl       # Project creation
    ├── create_block_design.tcl  # Block design automation
    ├── build_bitstream.tcl      # Bitstream generation
    ├── program_board.tcl        # FPGA programming
    └── run_all.sh               # Complete build script
```

##### Key TCL Scripts Usage

**create_hls_project.tcl**: Creates HLS projects for learning algorithms
```bash
# Create all HLS IP cores
vivado_hls -f create_hls_project.tcl

# Create specific IP core
vivado_hls -f create_hls_project.tcl -tclargs [IP_NAME]

# Available IP cores:
# - snn_learning_engine    # STDP/R-STDP learning algorithms
# - spike_encoder          # Input spike encoding
# - spike_decoder          # Output spike decoding  
# - weight_updater         # Synaptic weight updates
# - network_controller     # Network control logic
```

**create_vivado_project.tcl**: Creates complete Vivado project with all sources
```bash
# Create for PYNQ-Z2
vivado -mode batch -source create_vivado_project.tcl

# Create for different target boards
vivado -mode batch -source create_vivado_project.tcl -tclargs [BOARD]

# Supported boards:
# - pynq-z2       # PYNQ-Z2 development board
# Support for other boards is not planned since I don't own any other boards.
```

**Configuration Examples:**
```bash
# High-performance configuration (more resources)
vivado -mode batch -source create_vivado_project.tcl \
  -tclargs pynq-z2 high_perf

# Low-power configuration (fewer resources)
vivado -mode batch -source create_vivado_project.tcl \
  -tclargs pynq-z2 low_power

# Debug configuration (with ILA cores)
vivado -mode batch -source create_vivado_project.tcl \
  -tclargs pynq-z2 debug
```

##### Hardware Configuration Options

**Neuron Array Configuration**:
- Number of neurons: 64-1024 (configurable)
- LIF parameters: threshold, leak, refractory period
- Precision: 8-bit weights, 16-bit membrane potential

**Learning Engine Configuration**:
- STDP time constants: τ+ = 20ms, τ- = 20ms  
- Learning rates: A+ = 0.1, A- = 0.12
- R-STDP eligibility trace decay: 0.95
- Weight bounds: signed 8-bit (-127 to +127)

**Memory Configuration**:
- Neuron memory: BRAM-based, dual-port
- Weight memory: BRAM/UltraRAM for large networks
- Spike buffers: FIFO-based with configurable depth

**Interface Configuration**:
- AXI-Stream for spike data (32-bit width)
- AXI-Lite for control and status registers
- DMA for bulk data transfer (optional)

##### Simulation and Verification

**RTL Simulation**:
The project includes comprehensive testbenches for verifying the hardware design:

```bash
cd hardware/hdl/sim

# Run simple LIF neuron simulation (Verilog-2001 compatible)
./run_sim_working.sh tb_simple_lif

# Try other testbenches (may require SystemVerilog support)
./run_sim_working.sh tb_lif_neuron
./run_sim_working.sh tb_spike_router

# For advanced simulation with Vivado Simulator (if available)
./run_sim.sh tb_top --simulator vivado --gui
```

**Available Simulators**:
- **Icarus Verilog**: Good open-source option (some SystemVerilog limitations)
- **Vivado Simulator**: Full SystemVerilog support with advanced debugging
- **Verilator**: Fast simulation (basic support)

**Note**: Some testbenches use SystemVerilog features that may not be fully supported by Icarus Verilog. Use simplified testbenches or Vivado Simulator for full compatibility.

**HLS Co-simulation**:
```bash
cd hardware/hls
vivado_hls -f scripts/run_cosim.tcl
```

**Hardware-in-the-Loop Testing**:
```bash
# After FPGA programming
python examples/hardware_validation.py
```

##### Performance Optimization Tips

1. **Resource Utilization**:
   - Monitor BRAM usage for large neuron arrays
   - Use UltraRAM for weight storage on newer devices
   - Optimize DSP usage for learning algorithms

2. **Timing Closure**:
   - Adjust clock frequency in constraints
   - Pipeline critical paths in HLS code
   - Use timing-driven placement and routing

3. **Power Optimization**:
   - Enable clock gating for unused neurons
   - Use power-optimized synthesis strategies
   - Implement dynamic voltage scaling

### 4. Run Examples

#### PyTorch Training Example
```bash
cd examples/pytorch
python mnist_training_example.py --train --epochs 20
```

#### R-STDP Learning Example  
```bash
python r_stdp_learning_example.py
```

#### Basic SNN Inference
```python
from snn_fpga_accelerator import SNNAccelerator, PoissonEncoder

# Initialize accelerator
with SNNAccelerator() as accelerator:
    # Load your trained model
    accelerator.configure_network(num_neurons=100, topology=network_config)
    
    # Encode input data
    encoder = PoissonEncoder(num_neurons=784, duration=0.1)
    input_spikes = encoder.encode(mnist_image)
    
    # Run inference
    output_spikes = accelerator.run_simulation(duration=0.1, input_spikes=input_spikes)
```

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

### LIF Neuron Model
- **Membrane Integration**: `V(t+1) = V(t) * (1 - leak) + Σ(w_i * spike_i)`
- **Spike Generation**: `spike = (V > threshold)`
- **Refractory Period**: Configurable non-responsive period after spike
- **Saturation Arithmetic**: Prevents overflow in fixed-point implementation

### STDP Learning Rule
- **LTP (Long-term Potentiation)**: `Δw = A+ * exp(-Δt/τ+)` for pre-before-post
- **LTD (Long-term Depression)**: `Δw = -A- * exp(Δt/τ-)` for post-before-pre
- **Eligibility Traces**: For delayed reward association in R-STDP

### Event-Driven Processing
- **Spike Events**: `(neuron_id, timestamp, weight)`
- **AXI-Stream Protocol**: For efficient FPGA communication
- **Priority Queues**: For temporal spike ordering
- **Backpressure Handling**: Flow control for high spike rates

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

-  **[User Guide](docs/user_guide.md)**: Detailed usage instructions
-  **[Architecture](docs/architecture.md)**: System design overview  
-  **[API Reference](docs/api_reference.md)**: Complete API documentation
-  **[Developer Guide](docs/developer_guide.md)**: Development setup and guidelines

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

### Common Issues

**Q: FPGA bitstream loading fails**
A: Ensure PYNQ-Z2 is properly connected and powered. Check that Vivado generated the bitstream successfully.

**Q: Spike encoding produces no output**  
A: Verify input data is normalized to [0,1] range. Check encoder parameters (max_rate, duration).

**Q: Learning convergence is slow**
A: Try adjusting learning rate, STDP time constants, or reward signal strength.

### FPGA Development Issues

**Q: HLS synthesis fails with timing violations**
A: 
```bash
# Increase clock period in HLS TCL script
set_directive_interface -mode ap_ctrl_hs "top_function"
set_directive_pipeline "loop_name" -II 2
```

**Q: Vivado implementation fails to meet timing**
A:
```bash
# Check timing report
open_checkpoint post_route.dcp
report_timing_summary -file timing_summary.rpt

# Try different synthesis strategies
set_property strategy Performance_ExplorePostRoutePhysOpt [get_runs impl_1]
```

**Q: Resource utilization exceeds device capacity**
A:
- Reduce neuron array size in configuration
- Use UltraRAM instead of BRAM for weight storage
- Enable resource sharing in HLS directives

**Q: create_hls_project.tcl fails**
A:
```bash
# Check Vivado HLS installation
which vivado_hls
source /opt/Xilinx/Vitis_HLS/2025.1/settings64.sh

# Run with verbose output
vivado_hls -f create_hls_project.tcl -l hls_build.log
```

**Q: create_vivado_project.tcl fails**
A:
```bash
# Check Vivado installation and license
vivado -version
echo $XILINXD_LICENSE_FILE

# Clean previous builds
rm -rf hardware/build/*
vivado -mode batch -source create_vivado_project.tcl -log vivado_build.log
```

**Q: Block design creation fails**
A:
```bash
# Ensure HLS IP cores are exported correctly
ls hardware/hls/*/solution1/impl/ip/

# Regenerate IP cores
cd hardware/hls
find . -name "*.zip" -delete
vivado_hls -f scripts/build_all.tcl
```

**Q: PYNQ board not detected**
A:
```bash
# Check USB connection
lsusb | grep Xilinx

# Install/update PYNQ drivers
sudo apt install libusb-1.0-0-dev
pip install pynq --upgrade
```

**Q: Bitstream programming hangs**
A:
```bash
# Reset FPGA and try again
sudo python3 -c "
import pynq
from pynq import GPIO
# Toggle FPGA reset if available
"

# Use hardware manager
vivado -mode batch -source program_board.tcl
```

**Q: Simulation shows no activity**
A:
```bash
# Check testbench stimulus
cd hardware/hdl/tb
grep -n "stimulus\|input" tb_*.v

# Use simple testbench if SystemVerilog issues occur
cd hardware/hdl/sim
./run_sim_working.sh tb_simple_lif

# Run with waveform dump
./run_sim_working.sh tb_simple_lif
# Then view with: gtkwave work/waves.vcd
```

**Q: Compilation errors with "SystemVerilog features not supported"**
A:
```bash
# Use simplified testbenches
./run_sim_working.sh tb_simple_lif

# Or install Vivado for full SystemVerilog support
# and use: ./run_sim.sh tb_top --simulator vivado
```

## Support

- **Issues**: [GitHub Issues](https://github.com/metr0jw/Spiking-Neural-Network-on-FPGA/issues)
- **Email**: [jwlee@linux.com](mailto:jwlee@linux.com)
- **Documentation**: [Project Wiki](https://github.com/metr0jw/Spiking-Neural-Network-on-FPGA/wiki)

## License
[MIT License](LICENSE)

## Author
[Jiwoon Lee](https://github.com/metr0jw)
