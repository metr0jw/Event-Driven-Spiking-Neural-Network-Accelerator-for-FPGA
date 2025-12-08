# SNN FPGA Accelerator - Python Library

A PyTorch-like library for building, training, and deploying Spiking Neural Networks (SNNs) on FPGAs. Provides surrogate gradient training like snnTorch/SpikingJelly with hardware-aware quantization.

## Installation

```bash
cd software/python
pip install -e .
```

## Quick Start

```python
import snn_fpga_accelerator as snn
import torch
import torch.nn as nn

# Define SNN model (like PyTorch)
class SimpleSNN(nn.Module):
    def __init__(self, num_steps=25):
        super().__init__()
        self.num_steps = num_steps
        
        # FC layers + LIF neurons as activation functions
        self.fc1 = nn.Linear(784, 256)
        self.lif1 = snn.LIF(thresh=1.0, tau=0.9)
        
        self.fc2 = nn.Linear(256, 10)
        self.lif2 = snn.LIF(thresh=1.0, tau=0.9)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        
        # Reset neuron states
        self.lif1.reset_state()
        self.lif2.reset_state()
        
        # Time loop
        spk_rec = []
        for t in range(self.num_steps):
            x1 = self.lif1(self.fc1(x))
            x2 = self.lif2(self.fc2(x1))
            spk_rec.append(x2)
        
        return torch.stack(spk_rec).sum(0)

# Create model
model = SimpleSNN()

# Train with standard PyTorch (surrogate gradients enabled by default)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# Forward + backward
output = model(torch.randn(32, 1, 28, 28))
loss = loss_fn(output, torch.randint(10, (32,)))
loss.backward()  # Surrogate gradients flow through!
optimizer.step()

## XRT Backend (optional)

When targeting a Vitis/XRT flow (e.g., Alveo), you can drive the new mode/time-step
registers directly via pyxrt:

```python
from snn_fpga_accelerator import SNNAccelerator

accel = SNNAccelerator(simulation_mode=True)  # skip PYNQ
accel.configure_xrt("snn_top_hls.xclbin")

# Set mode: 0=infer, 1=STDP train, 2=checkpoint; enable encoder with encoder_enable=True
accel.set_mode(1, encoder_enable=True)
accel.set_simulation_steps(25)
accel.set_reward_signal(16)

# TODO: add DMA streaming once spike/raw stream names are finalised
```
```

## Features

### Spiking Neurons (like activation functions)

```python
# LIF - Leaky Integrate-and-Fire (most common)
lif = snn.LIF(thresh=1.0, tau=0.9)

# IF - Integrate-and-Fire (no leak)
if_neuron = snn.IF(thresh=1.0)

# PLIF - Parametric LIF (learnable tau/thresh)
plif = snn.PLIF(thresh=1.0, learn_tau=True, learn_thresh=True)

# ALIF - Adaptive LIF (adaptive threshold)
alif = snn.ALIF(thresh=1.0, beta=0.01)

# Izhikevich (biologically realistic)
izh = snn.Izhikevich()
```

### Surrogate Gradients

```python
# Choose surrogate gradient (default: fast_sigmoid)
lif = snn.LIF(surrogate='fast_sigmoid', scale=25.0)  # Fast, default
lif = snn.LIF(surrogate='atan')                       # Arctangent
lif = snn.LIF(surrogate='super_spike')                # SuperSpike
lif = snn.LIF(surrogate='sigmoid')                    # Sigmoid
lif = snn.LIF(surrogate='pwl')                        # Piecewise Linear
```

### Spike Encoding

```python
# Rate coding
encoder = snn.Rate(T=100)
spikes = encoder(images)  # (batch, time, features)

# Poisson encoding
encoder = snn.Poisson(T=100)

# Latency encoding (time-to-first-spike)
encoder = snn.Latency(T=100)

# Delta encoding (event-based, DVS-like)
encoder = snn.Delta(thresh=0.1)
```

### Combined Layers (Layer + Neuron)

```python
# SLinear = Linear + LIF
layer = snn.SLinear(784, 256, neuron=snn.LIF())

# SConv2d = Conv2d + LIF
layer = snn.SConv2d(1, 32, kernel_size=3, neuron=snn.LIF())

# SRNN = Recurrent + LIF
layer = snn.SRNN(256, 128, neuron=snn.LIF())
```

### Training Utilities

```python
# Loss functions
loss = snn.CrossEntropy()        # CE on spike counts
loss = snn.MSE()                 # MSE on spike rates
loss = snn.SpikeCount(target=5)  # Target spike count
loss = snn.SpikeRate(rate=0.1)   # Target spike rate

# STDP learning
stdp = snn.STDP(A_plus=0.01, A_minus=0.012, tau_plus=20, tau_minus=20)
stdp.update(pre_spikes, post_spikes, weights)

# Reward-modulated STDP
rstdp = snn.RSTDP(A_plus=0.01, A_minus=0.012)
rstdp.update(pre_spikes, post_spikes, weights, reward=1.0)
```

### HW-Constrained Training

```python
# Enable hardware constraints (8-bit weights, 16-bit membrane)
lif = snn.LIF(hw_mode=True)

# Full HW-aware model
model = DeepSNN(hw_mode=True)
model.train()
# ... training ...

# Quantize for FPGA
weights = snn.quantize(model.state_dict(), bits=8)
```

### FPGA Deployment

```python
# Export quantized weights
snn.deploy.export(model, 'weights.npz')

# Generate FPGA configuration
config = snn.deploy.gen_config(model)
config.save('network_config.yaml')

# Connect to PYNQ board
fpga = snn.deploy.PYNQ(bitstream='snn.bit', weights='weights.npz')
output = fpga.run(test_data)
```

## Module Structure

```
snn_fpga_accelerator/
├── __init__.py          # Main API exports
├── neuron.py            # LIF, IF, PLIF, ALIF, Izhikevich
├── layer.py             # SLinear, SConv2d, SRNN
├── encoder.py           # Rate, Poisson, Latency, Delta
├── training.py          # STDP, RSTDP, losses, Trainer
├── deploy.py            # Quantization, FPGA export
├── hw_accurate_simulator.py  # RTL-accurate Python simulator
└── accelerator.py       # FPGA interface
```

## Examples

- `examples/pytorch/mnist_snn_training.py` - MNIST with surrogate gradients
- `examples/pytorch/mnist_training_example.py` - Basic MNIST
- `examples/pytorch/r_stdp_learning_example.py` - Reward-modulated STDP

## API Reference

### snn.LIF

```python
snn.LIF(
    thresh: float = 1.0,          # Firing threshold
    tau: float = 0.9,             # Membrane time constant (leak)
    reset: str = 'subtract',      # 'subtract' or 'zero'
    surrogate: str = 'fast_sigmoid',  # Surrogate gradient type
    scale: float = 25.0,          # Surrogate gradient scale
    hw_mode: bool = False,        # Enable HW constraints
    learn_tau: bool = False,      # Learnable tau
    learn_thresh: bool = False,   # Learnable threshold
)

# Forward pass
spikes = lif(input)  # Returns spikes, maintains membrane state

# Reset state
lif.reset_state()

# Detach for TBPTT
lif.detach_state()
```

### snn.SLinear

```python
snn.SLinear(
    in_features: int,
    out_features: int,
    bias: bool = True,
    neuron: SpikingNeuron = None,  # Uses LIF() if None
    hw_mode: bool = False,
)
```

### snn.Rate (Encoder)

```python
snn.Rate(
    T: int = 100,        # Number of time steps
    max_rate: float = 1.0,  # Maximum spike rate
    normalize: bool = True,
)

# Encode
spikes = encoder(images)  # (batch, T, features)
```

## Hardware Accurate Simulation

For RTL-level simulation matching the Verilog implementation:

```python
from snn_fpga_accelerator import (
    HWAccurateLIFNeuron,
    HWAccurateSTDPEngine,
    HWAccurateSNNSimulator,
)

# Create HW-accurate neuron
params = LIFNeuronParams(tau_mem=230, v_th=256, v_reset=0)
neuron = HWAccurateLIFNeuron(params)

# Simulate
for t in range(100):
    spike = neuron.step(current[t], learn=True)
```

## License

MIT License

## Author

Jiwoon Lee (@metr0jw)
Kwangwoon University

