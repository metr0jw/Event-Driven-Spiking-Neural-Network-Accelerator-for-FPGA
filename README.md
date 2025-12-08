# Event-Driven Spiking Neural Network Accelerator for FPGA

Energy-efficient Event-driven Spiking Neural Network accelerator for FPGA with PyTorch integration.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Development Environment

| Component | Version/Specification |
|-----------|----------------------|
| FPGA Tools | Vivado 2025.2, Vitis 2025.2 (v++ compiler) |
| Target FPGA | Xilinx Zynq-7000 (xc7z020clg400-1) |
| Target Board | PYNQ-Z2 |
| Software | Python 3.13, PyTorch 2.9.0, PYNQ 2.7 |
| HDL Standard | Verilog-2001 (IEEE 1364-2001) |
| Simulation | Icarus Verilog, Cocotb |

## Overview

This project implements a complete event-driven spiking neural network (SNN) accelerator on FPGA with seamless PyTorch integration for training and deployment. The design uses Accumulate-only (AC) operations instead of traditional Multiply-Accumulate (MAC) operations, achieving significant energy reduction.

### Architecture

```
+------------------+     +-------------------+     +------------------+
|   PS (ARM)       |     |   PL (FPGA)       |     |   Output         |
|   PyTorch/PYNQ   |<--->|   SNN Accelerator |<--->|   Processing     |
+------------------+     +-------------------+     +------------------+
        |                         |
        v                         v
  AXI4-Lite (Ctrl)         LIF Neuron Arrays
  AXI4-Stream (Data)       AC-Based Synapses
                           Spike Router
                           Per-Neuron STDP/R-STDP
```

### Key Features

- **Leaky Integrate-and-Fire (LIF) Neurons**: Hardware-optimized fixed-point arithmetic with configurable membrane dynamics
- **AC-Based Architecture**: Accumulate-only operations (~5x energy reduction per synaptic operation)
- **Per-Neuron STDP/R-STDP Learning**: Memory-efficient O(N+M) trace storage (vs O(N×M) per-synapse)
- **Lazy Update with LUT**: Exponential decay computed on-demand using 16-entry lookup table
- **PyTorch Integration**: Direct model conversion and weight loading
- **Multiple Spike Encoders**: Rate (Poisson), Latency (intensity-to-latency), Delta-Sigma modulation, or direct spike input (4-bit encoding type)

---

## Build Status

### HLS Synthesis Results (Per-Neuron Trace Architecture)

| Metric | Value | Notes |
|--------|-------|-------|
| **Estimated Fmax** | **138.10 MHz** | Exceeds 100MHz target |
| Target Clock | 100 MHz (10ns) | PYNQ-Z2 system clock |
| Timing Slack | +0.06 ns | All timing constraints met |
| BRAM | 56 (20%) | Weight memory + Per-Neuron traces |
| FF | 2,850 (2%) | Synchronous registers |
| LUT | 11,503 (21%) | Combinational logic |
| DSP | 0 | Shift-based arithmetic (no DSP) |

### Memory Architecture

| Storage | Size | Complexity | Description |
|---------|------|------------|-------------|
| Weight Memory | 64×64×8bit | O(N×M) | Synaptic weights |
| Pre-Neuron Traces | 64×(8+16)bit | O(N) | Trace + timestamp |
| Post-Neuron Traces | 64×(8+16)bit | O(M) | Trace + timestamp |
| Eligibility Traces | 64×2×8bit | O(N+M) | Pre/Post eligibility |

### Verified Simulation Results

| Test Suite | Tests | Status |
|------------|-------|--------|
| Verilog Testbenches | 17 | All Pass ✅ |
| Python Unit Tests | 6+ | All Pass ✅ |
| HW-Python Identity | 6 | All Pass ✅ |
| HLS Synthesis | 1 | All Pass ✅ |

---

## Project Structure

```
hardware/
  hdl/
    rtl/
      neurons/          # LIF neuron (AC-based)
      synapses/         # Synapse arrays (AC-based)
      layers/           # Conv1D/2D, FC, Pooling
      router/           # Spike routing
      interfaces/       # AXI wrapper, weight controller
      top/              # Top-level integration
    tb/                 # 17 Verilog testbenches
  hls/
    src/                # STDP/R-STDP learning (v++ HLS)
      snn_top_hls.cpp   # Per-Neuron Trace implementation
      snn_top_hls.h     # HLS headers
    scripts/
      build_hls.sh      # v++ build script (Vitis 2025.2+)
    hls_output/         # Generated IP (after synthesis)
  ip_repo/              # Packaged IP cores
  scripts/              # Vivado build automation

software/python/
  snn_fpga_accelerator/
    accelerator.py           # FPGA interface
    hw_accurate_simulator.py # Bit-accurate simulator
    pytorch_interface.py     # PyTorch integration
    spike_encoding.py        # Encoding algorithms
    learning.py              # STDP/R-STDP

examples/
  pytorch/              # Training examples (MNIST, R-STDP)

outputs/
  snn_accelerator.bit   # FPGA bitstream
  snn_accelerator.hwh   # Hardware handoff for PYNQ
```

---

## Installation

### Prerequisites

```bash
# Required
pip install torch numpy

# Optional (for PYNQ board deployment)
pip install pynq
```

### Setup

```bash
git clone https://github.com/metr0jw/Event-Driven-Spiking-Neural-Network-Accelerator-for-FPGA.git
cd Event-Driven-Spiking-Neural-Network-Accelerator-for-FPGA

# Install Python package
cd software/python
pip install -e .
```

---

## Usage

### 1. Simulation Mode (No FPGA Required)

```python
from snn_fpga_accelerator import SNNAccelerator

# Initialize in simulation mode
accelerator = SNNAccelerator(simulation_mode=True)
accelerator.connect()

# Configure network parameters
accelerator.configure(
    threshold=100,      # Spike threshold
    leak_rate=51,       # Shift-based leak (tau ~ 0.86)
    refractory_period=5 # Refractory period in timesteps
)

# Load weights from trained PyTorch model
accelerator.load_weights(weights)

# Run inference
input_spikes = encode_input(data)  # Shape: [timesteps, neurons]
output_spikes = accelerator.infer(input_spikes, timesteps=100)

accelerator.disconnect()
```

### 2. Hardware Deployment (PYNQ-Z2)

```python
from snn_fpga_accelerator import SNNAccelerator

# Initialize with bitstream
accelerator = SNNAccelerator(
    bitstream_path='outputs/snn_accelerator.bit',
    simulation_mode=False
)
accelerator.connect()

# Same API as simulation mode
accelerator.configure(threshold=100, leak_rate=51)
accelerator.load_weights(weights)
output = accelerator.infer(input_spikes, timesteps=100)

accelerator.disconnect()
```

### 3. PyTorch Model Training

```python
import torch
import torch.nn as nn
import snn_fpga_accelerator as snn

# Define SNN model (PyTorch-like API)
class MySNN(nn.Module):
    def __init__(self, num_steps=25):
        super().__init__()
        self.num_steps = num_steps
        self.fc1 = nn.Linear(784, 256)
        self.lif1 = snn.LIF(thresh=1.0, tau=0.9)
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

# Train with standard PyTorch
model = MySNN()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

for data, labels in dataloader:
    optimizer.zero_grad()
    output = model(data)
    loss = loss_fn(output, labels)
    loss.backward()  # Surrogate gradients
    optimizer.step()

# Export for FPGA
weights = snn.quantize(model.state_dict(), bits=8)
snn.deploy.export(model, 'weights.npz')
```

### 4. Spike Encoding

```python
from snn_fpga_accelerator import PoissonEncoder, TemporalEncoder, RateEncoder

# Poisson encoding (stochastic)
encoder = PoissonEncoder(timesteps=100, rate_scale=1.0)
spikes = encoder.encode(image)  # [100, 784]

# Temporal encoding (first spike time)
encoder = TemporalEncoder(timesteps=100)
spikes = encoder.encode(image)

# Rate encoding (deterministic)
encoder = RateEncoder(timesteps=100)
spikes = encoder.encode(image)
```

### 5. Running RTL Testbenches

```bash
cd hardware/hdl/sim

# Run all testbenches
./run_all_tests.sh

# Run specific testbench
./run_sim.sh tb_lif_neuron
```

### 6. Building HLS IP (v++ Compiler - Vitis 2025.2+)

```bash
# Source Vitis environment
source /tools/Xilinx/2025.2/Vitis/settings64.sh

# Build HLS IP using v++ (recommended)
cd hardware/hls
./scripts/build_hls.sh --clean

# Or run v++ directly
v++ -c --mode hls \
    --part xc7z020clg400-1 \
    --work_dir ./hls_output \
    --hls.clock 10ns \
    --hls.syn.top snn_top_hls \
    --hls.syn.file "src/snn_top_hls.cpp" \
    --hls.flow_target vivado

# Output: hls_output/hls/impl/ip/
```

> **Note**: The legacy `vitis_hls -f script.tcl` workflow is deprecated in Vitis 2025.2+.
> Use the `v++` compiler instead.

### 7. Building Vivado Bitstream

```bash
# Source Vivado environment
source /tools/Xilinx/2025.2/Vivado/settings64.sh

# Build bitstream
cd hardware/scripts
vivado -mode batch -source build_pynq_with_hls.tcl

# Output: outputs/snn_accelerator.bit, outputs/snn_accelerator.hwh
```

---

## Configuration Parameters

### Neuron Configuration

| Parameter | Range | Description |
|-----------|-------|-------------|
| threshold | 0-65535 | Membrane potential threshold for spike generation |
| leak_rate | 0-255 | Shift-based leak configuration (see below) |
| refractory_period | 0-255 | Post-spike refractory period in timesteps |

### Shift-Based Leak Encoding

The leak_rate parameter encodes two shift values for efficient exponential decay:

```
leak_rate[2:0] = shift1 (primary leak, 1-7)
leak_rate[7:3] = shift2 (secondary leak, 0=disabled)

decay = 1 - 2^(-shift1) - 2^(-shift2)
```

Examples:

| leak_rate | shift1 | shift2 | Effective tau |
|-----------|--------|--------|---------------|
| 3 | 3 | 0 | 0.875 |
| 51 | 3 | 6 | 0.859 |
| 4 | 4 | 0 | 0.9375 |

### Register Map (AXI-Lite)

| Offset | Register | Access | Description |
|--------|----------|--------|-------------|
| 0x00 | CTRL | R/W | Control (enable, reset, clear) |
| 0x04 | STATUS | R | Status (ready, busy, error) |
| 0x08 | CONFIG | R/W | Configuration |
| 0x0C | LEAK_RATE | R/W | Leak rate setting |
| 0x10 | THRESHOLD | R/W | Spike threshold |
| 0x14 | REFRACTORY | R/W | Refractory period |
| 0x18 | SPIKE_COUNT | R | Output spike counter |
| 0x1C | VERSION | R | IP version (0x20241203) |

---

## Architecture (HLS-Based)

This project uses HLS (Vitis HLS) for all AXI interfaces and learning engines, enabling on-chip STDP/R-STDP learning.

### System Architecture

```
+------------------+                     +------------------+
|   PS (ARM)       |                     |   PL (FPGA)      |
|   Python/PYNQ    |                     |                  |
+--------+---------+                     +--------+---------+
         |                                        |
         | AXI4-Lite (Ctrl)                       |
         | AXI4-Stream (Spikes)                   |
         v                                        v
+--------+-----------------------------------------+--------+
|                    snn_top_hls (HLS)                     |
|  +-------------+  +-------------+  +---------------+     |
|  | AXI4-Lite   |  | STDP/R-STDP |  | Weight Memory |     |
|  | Registers   |  | Learning    |  | (BRAM)        |     |
|  +-------------+  +-------------+  +---------------+     |
|         |               |                 |              |
|         v               v                 v              |
|  +--------------------------------------------------+   |
|  |           Wire Interface to Verilog              |   |
|  +--------------------------------------------------+   |
+---------------------------+------------------------------+
                            |
                            v
              +-------------+-------------+
              |    Verilog SNN Core       |
              |  (LIF Neurons, Synapses)  |
              +---------------------------+
```

### HLS Modules

| Module | File | Description |
|--------|------|-------------|
| snn_top_hls | snn_top_hls.cpp | Unified top-level with learning |
| snn_learning_engine | snn_learning_engine.cpp | STDP/R-STDP algorithms |
| axi_hls_wrapper | axi_hls_wrapper.cpp | AXI protocol handling |
| weight_updater | weight_updater.cpp | Weight update logic |

### Learning Parameters

```python
# Configure STDP parameters via AXI registers
learning_params = {
    'a_plus': 0.1,        # LTP amplitude
    'a_minus': 0.12,      # LTD amplitude
    'tau_plus': 20,       # LTP time constant (timesteps)
    'tau_minus': 20,      # LTD time constant (timesteps)
    'stdp_window': 50,    # STDP window size
    'learning_rate': 0.01,
    'rstdp_enable': True, # Enable R-STDP
    'trace_decay': 0.99,  # Eligibility trace decay
}
```

### Control Register Map (HLS)

| Bit | Name | Description |
|-----|------|-------------|
| 0 | ENABLE | SNN enable |
| 1 | RESET | System reset |
| 2 | CLEAR_COUNTERS | Clear spike counters |
| 3 | LEARNING_EN | Enable STDP learning |
| 4 | WEIGHT_READ | Enable weight readback |
| 5 | APPLY_REWARD | Apply R-STDP reward |

### Building HLS IP

```bash
# Set up Vitis environment
source /tools/Xilinx/Vitis/2025.2/settings64.sh

# Run synthesis with v++ CLI (recommended)
cd hardware/hls/scripts
./build_hls.sh           # Full build
./build_hls.sh syn       # Synthesis only
./build_hls.sh csim      # C-simulation only

# Output IP: hardware/hls/hls_output/
```

> **Note**: The legacy `vitis_hls -f script.tcl` workflow is deprecated in Vitis 2025.2+.

---

## Energy Efficiency

### AC vs MAC Comparison

| Operation | Energy (45nm) | Relative |
|-----------|---------------|----------|
| MAC (32-bit FP) | ~4.6 pJ | 1.0x |
| AC (16-bit INT) | ~0.9 pJ | 0.2x |

### Total Energy Savings

```
ANN Energy:  N_ops x 4.6pJ (MAC)
SNN Energy:  N_ops x sparsity x 0.9pJ (AC)

With 5% spike activity:  ~100x energy reduction
With 10% spike activity: ~50x energy reduction
```

---

## Implementation Status

### Completed

| Category | Feature |
|----------|---------|
| Hardware | LIF Neuron Core (AC-based) |
| Hardware | Conv1D/2D, FC, MaxPool, AvgPool Layers |
| Hardware | Spike Router |
| Hardware | Energy Monitor |
| Hardware | Bitstream (4.0MB, timing met) |
| HLS | STDP/R-STDP Learning Engine |
| HLS | Unified Top-Level (snn_top_hls) |
| HLS | AXI4-Lite/Stream Interfaces |
| HLS | On-Chip Weight Memory (BRAM) |
| HLS | Eligibility Traces (R-STDP) |
| Software | Python snn_fpga_accelerator Package |
| Software | PyTorch Interface |
| Software | HW-Accurate Simulator |
| Testing | 17 Verilog Testbenches |
| Testing | Python Unit Tests |
| Testing | HW-Python Identity Verification |
| Testing | HLS C-Simulation |
| Docs | Architecture, User Guide, API Reference |

### TODO

| Priority | Task | Description |
|----------|------|-------------|
| High | On-Board Validation | Test bitstream on actual PYNQ-Z2 hardware |
| High | HLS IP Integration | Integrate snn_top_hls into Vivado block design |
| High | End-to-End Learning Test | Verify STDP/R-STDP on hardware |
| Medium | Performance Benchmark | Measure actual latency and throughput |
| Medium | PYNQ Package | Create installable overlay package |
| Low | Recurrent Layers | Feedback connection support |
| Low | Multi-Board Support | Adapt to ZCU104, Ultra96 |

---

## Examples

### MNIST Training

```bash
python examples/pytorch/mnist_training_example.py
```

### R-STDP Learning

```bash
python examples/pytorch/r_stdp_learning_example.py
```

### Complete Integration

```bash
python examples/complete_integration_example.py --simulation-mode
```

---

## Documentation

- [Architecture](docs/architecture.md): System design and component details
- [User Guide](docs/user_guide.md): Comprehensive usage instructions
- [API Reference](docs/api_reference.md): Complete API documentation
- [Developer Guide](docs/developer_guide.md): Development setup and guidelines
- **[HLS IP Integration Guide](docs/HLS_IP_INTEGRATION_GUIDE.md)**: CLI workflow for updating Vivado IP and register maps

---

## References

- Energy efficiency analysis: Horowitz, M. "Computing's Energy Problem" ISSCC 2014
- SNN sparsity benefits: Roy, K. et al. "Towards spike-based machine intelligence" Nature 2019

---

## Citation

```bibtex
@misc{snn_fpga_accelerator,
  title={Event-Driven Spiking Neural Network Accelerator for FPGA},
  author={Jiwoon Lee},
  year={2025},
  howpublished={\url{https://github.com/metr0jw/Event-Driven-Spiking-Neural-Network-Accelerator-for-FPGA}}
}
```

---

## License

[MIT License](LICENSE)

## Author

Jiwoon Lee ([@metr0jw](https://github.com/metr0jw))
- Organization: Kwangwoon University, Seoul, South Korea
- Contact: jwlee@linux.com
