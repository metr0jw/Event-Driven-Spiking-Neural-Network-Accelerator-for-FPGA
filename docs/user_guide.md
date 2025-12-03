# User Guide

Complete guide for using the Event-Driven SNN FPGA Accelerator.

## Table of Contents
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Basic Usage](#basic-usage)
- [Step Modes](#step-modes)
- [Spike Encoding](#spike-encoding)
- [PyTorch Integration](#pytorch-integration)
- [Learning Algorithms](#learning-algorithms)
- [Hardware Deployment](#hardware-deployment)
- [Configuration](#configuration)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

## Installation

### Prerequisites

- Python 3.13 or higher
- PyTorch 2.9.0 or higher
- NumPy 1.24+
- PYNQ 2.7+ (for hardware deployment)

### Automated Installation

The easiest way to get started:

```bash
# Clone repository
git clone https://github.com/metr0jw/Spiking-Neural-Network-on-FPGA.git
cd Spiking-Neural-Network-on-FPGA

# Activate environment (if using provided virtualenv)
source venv/bin/activate

# Run setup script
./setup.sh
```

### Manual Installation

```bash
# Install Python package
cd software/python
pip install -e .

# Install additional dependencies
pip install torch torchvision tqdm matplotlib h5py pynq
```

### Verify Installation

```python
import snn_fpga_accelerator
print(snn_fpga_accelerator.__version__)

# Check if PyTorch is available
import torch
print(f"PyTorch version: {torch.__version__}")
```

## Quick Start

### Simple Inference Example

```python
from snn_fpga_accelerator import SNNAccelerator
from snn_fpga_accelerator.spike_encoding import PoissonEncoder
import numpy as np

# Create encoder
encoder = PoissonEncoder(num_neurons=784, duration=0.1, max_rate=100.0)

# Encode MNIST image (28x28 = 784 pixels)
image = np.random.rand(784)  # Replace with actual image data
spikes = encoder.encode(image)

# Initialize accelerator (simulation mode)
accelerator = SNNAccelerator(simulation_mode=True)

# Configure network
network_config = {
    'num_neurons': 100,
    'layers': [
        {'type': 'input', 'size': 784},
        {'type': 'hidden', 'size': 100},
        {'type': 'output', 'size': 10}
    ]
}
accelerator.configure_network(network_config)

# Run inference
output_spikes = accelerator.infer(spikes)
prediction = np.argmax(output_spikes.sum(axis=1))
print(f"Predicted class: {prediction}")
```

### Running Examples

The `examples/` directory contains various usage examples:

```bash
# MNIST classification with PyTorch training
python examples/pytorch/mnist_training_example.py

# R-STDP reinforcement learning
python examples/pytorch/r_stdp_learning_example.py

# Complete integration example
python examples/complete_integration_example.py
```

## Basic Usage

### Network Configuration

Define your SNN architecture:

```python
from snn_fpga_accelerator import SNNAccelerator

# Basic configuration
config = {
    'num_neurons': 200,
    'neuron_type': 'LIF',
    'threshold': 1.0,
    'leak': 0.9,
    'refractory_period': 2,  # timesteps
    'layers': [
        {'type': 'input', 'size': 784},
        {'type': 'hidden', 'size': 200},
        {'type': 'output', 'size': 10}
    ]
}

accelerator = SNNAccelerator()
accelerator.configure_network(config)
```

### Loading Weights

Load pre-trained weights:

```python
import numpy as np

# Load weights from file
weights = np.load('trained_weights.npy')
accelerator.load_weights(weights)

# Or set weights directly
weight_matrix = np.random.randn(784, 200) * 0.1
accelerator.set_layer_weights(layer_id=0, weights=weight_matrix)
```

### Running Inference

```python
# Prepare input spikes
input_spikes = prepare_spikes(input_data)

# Run simulation
output_spikes = accelerator.infer(
    input_spikes,
    duration=0.1,  # seconds
    timestep=0.001  # 1ms timesteps
)

# Process output
spike_counts = output_spikes.sum(axis=1)
prediction = np.argmax(spike_counts)
```

## Step Modes

The accelerator supports two simulation modes similar to SpikingJelly framework:

### Multi-Step Mode (Default)

Process the entire simulation time at once:

```python
from snn_fpga_accelerator import SNNAccelerator

accelerator = SNNAccelerator(simulation_mode=True)
accelerator.set_step_mode("multi")  # Default mode

# Encode input data
spikes = encoder.encode(input_data)

# Process entire simulation (e.g., 100ms)
output = accelerator.infer(spikes, duration=0.1)

# Output contains firing rates for entire duration
print(f"Output shape: {output.shape}")  # (num_output_neurons,)
```

### Single-Step Mode

Process one timestep at a time for fine-grained control:

```python
accelerator.set_step_mode("single", timestep_dt=0.001)  # 1ms timesteps

# Reset state before starting
accelerator.reset()

# Process timestep by timestep
num_timesteps = 100
for t in range(num_timesteps):
    # Get spikes for this timestep
    timestep_spikes = [s for s in spikes if int(s.timestamp * 1000) == t]
    
    # Process one timestep
    output = accelerator.single_step(timestep_spikes)
    
    # Output contains firing activity for this timestep only
    print(f"Timestep {t}: Output = {output}")

# Get complete spike history
history = accelerator.get_spike_history()
print(f"Total timesteps processed: {len(history)}")
```

### Single-Step with Events

Get detailed spike events from each timestep:

```python
accelerator.set_step_mode("single", timestep_dt=0.001)
accelerator.reset()

all_events = []
for t in range(num_timesteps):
    timestep_spikes = get_spikes_for_timestep(t)
    
    # Return spike events instead of firing rates
    events = accelerator.single_step(timestep_spikes, return_events=True)
    all_events.extend(events)
    
    # Process events
    for event in events:
        print(f"Neuron {event.neuron_id} spiked at {event.timestamp:.3f}s")
```

### Mode Comparison

```python
# Multi-step: Fast, processes entire simulation
accelerator.set_step_mode("multi")
output_multi = accelerator.infer(spikes, duration=0.1)
# Result: Aggregated firing rates for entire duration

# Single-step: Fine control, iterate timestep by timestep
accelerator.set_step_mode("single", timestep_dt=0.001)
accelerator.reset()
outputs_single = []
for t in range(100):
    output = accelerator.single_step(get_spikes_for_timestep(t))
    outputs_single.append(output)
# Result: List of outputs, one per timestep

# Verify equivalence (may differ slightly due to temporal resolution)
print(f"Multi-step sum: {output_multi.sum():.3f}")
print(f"Single-step sum: {sum(o.sum() for o in outputs_single):.3f}")
```

### Use Cases

**Multi-Step Mode:**
- Fast inference for classification tasks
- Batch processing of inputs
- When only final output matters
- Backward-compatible with existing code

**Single-Step Mode:**
- Online learning with real-time feedback
- Debugging and visualization
- Fine-grained control over simulation
- State inspection between timesteps
- Implementing custom temporal dynamics

### State Management

```python
# Check current mode
mode = accelerator.get_step_mode()
print(f"Current mode: {mode}")

# Switch modes preserves network configuration
accelerator.set_step_mode("single")
# ... do single-step processing ...
accelerator.set_step_mode("multi")
# Network weights and configuration unchanged

# Reset clears temporal state but preserves configuration
accelerator.reset()  # Clears membrane potentials, spike history
```

## Spike Encoding

Convert conventional data to spike trains using various encoding schemes.

### Poisson Encoding

Rate-based encoding with stochastic spike generation:

```python
from snn_fpga_accelerator.spike_encoding import PoissonEncoder

# Create encoder
encoder = PoissonEncoder(
    num_neurons=784,
    duration=0.1,  # 100ms
    max_rate=100.0,  # Hz
    seed=42  # For reproducibility
)

# Encode data (normalized to [0, 1])
data = np.random.rand(784)
spikes = encoder.encode(data)

# spikes shape: (num_neurons, num_timesteps)
print(f"Spike train shape: {spikes.shape}")
print(f"Total spikes: {spikes.sum()}")
```

### Temporal Encoding

Time-to-first-spike encoding:

```python
from snn_fpga_accelerator.spike_encoding import TemporalEncoder

encoder = TemporalEncoder(
    num_neurons=784,
    duration=0.1,
    encoding='linear'  # or 'logarithmic'
)

# Higher values spike earlier
data = np.random.rand(784)
spikes = encoder.encode(data)
```

### Rate Encoding

Fixed-rate encoding based on input intensity:

```python
from snn_fpga_accelerator.spike_encoding import RateEncoder

encoder = RateEncoder(
    num_neurons=784,
    duration=0.1,
    max_rate=100.0
)

spikes = encoder.encode(data)
```

### Population Encoding

Encode scalar values using neuron populations:

```python
from snn_fpga_accelerator.spike_encoding import PopulationEncoder

encoder = PopulationEncoder(
    num_neurons_per_value=10,
    num_values=1,
    duration=0.1
)

# Encode single value
value = 0.75
spikes = encoder.encode([value])
```

### Spike Decoding

Convert spike trains back to values:

```python
from snn_fpga_accelerator.spike_encoding import PopulationDecoder

decoder = PopulationDecoder(
    num_neurons_per_class=10,
    num_classes=10
)

# Decode output spikes
output_spikes = accelerator.infer(input_spikes)
prediction = decoder.decode(output_spikes)
```

## PyTorch Integration

Seamlessly integrate with PyTorch for training and deployment.

### Gradient-Free SNN Training

All distributed examples train spiking networks with reward-modulated STDP instead of backpropagation. Layers such as `TorchSNNLayer` run their forward pass under `torch.no_grad()`, so gradients are never recorded. Weight updates are driven by spike statistics that are cached during the forward sweep and adjusted with a reward signal.

```python
import torch
import torch.nn as nn
from snn_fpga_accelerator.pytorch_interface import TorchSNNLayer

class SpikingMLP(nn.Module):
    def __init__(self, hidden_sizes=(256, 128), num_classes=10):
        super().__init__()
        layers = []
        in_features = 784
        for hidden in hidden_sizes:
            layers.append(TorchSNNLayer(in_features, hidden))
            in_features = hidden
        layers.append(TorchSNNLayer(in_features, num_classes))
        self.layers = nn.ModuleList(layers)
        self.last_input_spikes = None
        self.last_layer_spikes = []
        self.last_output_spikes = None

    def forward(self, encoded_input: torch.Tensor) -> torch.Tensor:
        spike_history = []
        layer_histories = [[] for _ in self.layers]
        current = encoded_input
        for t in range(encoded_input.shape[2]):
            timestep_input = encoded_input[:, :, t]
            for layer_idx, layer in enumerate(self.layers):
                timestep_input = layer(timestep_input)
                layer_histories[layer_idx].append(timestep_input)
            spike_history.append(timestep_input)

        self.last_input_spikes = encoded_input.detach()
        self.last_layer_spikes = [torch.stack(history, dim=2).detach() for history in layer_histories]
        self.last_output_spikes = self.last_layer_spikes[-1]
        return torch.stack(spike_history, dim=2)

def stdp_update(model: SpikingMLP, reward: torch.Tensor, learning_rate: float = 0.01) -> None:
    input_rate = model.last_input_spikes.mean(dim=2)
    layer_rates = [spikes.mean(dim=2) for spikes in model.last_layer_spikes]
    with torch.no_grad():
        for idx, layer in enumerate(model.layers):
            pre = input_rate if idx == 0 else layer_rates[idx - 1]
            post = layer_rates[idx]
            delta_w = torch.zeros_like(layer.weight)
            delta_b = torch.zeros_like(layer.bias)
            for b in range(pre.size(0)):
                delta_w += reward[b] * torch.outer(post[b], pre[b])
                delta_b += reward[b] * (post[b] - 0.5)
            layer.weight += learning_rate * (delta_w / pre.size(0))
            layer.bias += learning_rate * (delta_b / pre.size(0))
            layer.weight.clamp_(-1.0, 1.0)
            layer.bias.clamp_(-1.0, 1.0)
```

The training loop computes rewards from task-specific feedback (e.g., classification accuracy) and calls the STDP update helper instead of `loss.backward()`. See `examples/pytorch/mnist_training_example.py` and `examples/complete_integration_example.py` for end-to-end workflows.

### Converting Fully Connected Models

```python
import torch
import torch.nn as nn
from snn_fpga_accelerator.pytorch_interface import pytorch_to_snn

# Define PyTorch SNN model
class SNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 200)
        self.fc2 = nn.Linear(200, 10)
        
    def forward(self, x):
        x = self.fc1(x)
        # Add spiking nonlinearity
        x = torch.relu(x)  # Will be converted to LIF
        x = self.fc2(x)
        return x

# Train model
model = SNNModel()
# ... training code ...

# Convert to FPGA-compatible format
network_config = pytorch_to_snn(
    model, 
    input_shape=(784,),
    neuron_params={'threshold': 1.0, 'leak': 0.9}
)
accelerator.configure_network(network_config)
```

### Converting Convolutional Neural Networks

The accelerator supports Conv2d layers with automatic spatial dimension tracking:

```python
import torch.nn as nn
from snn_fpga_accelerator.pytorch_interface import pytorch_to_snn

class ConvSNN(nn.Module):
    def __init__(self):
        super().__init__()
        # MNIST: 1×28×28 input
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        # Output: 16×28×28
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2)
        # Output: 32×14×14
        self.fc1 = nn.Linear(32 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Train your model
model = ConvSNN()
# ... training code ...

# Convert to SNN format
network_config = pytorch_to_snn(
    model,
    input_shape=(1, 28, 28),  # Channels, Height, Width
    neuron_params={'threshold': 1.0, 'leak': 0.95}
)

# The converter automatically:
# 1. Extracts Conv2d weights (out_channels × in_channels × kH × kW)
# 2. Calculates spatial dimensions: out_size = (in_size + 2×padding - kernel) / stride + 1
# 3. Tracks flattening for first Linear layer
# 4. Validates weight shapes at each layer
```

**Important Notes for CNN Conversion:**

1. **Pooling Layers**: MaxPool2d and AvgPool2d are not yet tracked during conversion. Avoid pooling or manually calculate the flattened size.

2. **Spatial Dimension Tracking**: The converter tracks height and width through conv layers:
   ```python
   # Example: Input 28×28, Conv(k=5, s=2, p=2)
   # Output = (28 + 2×2 - 5) / 2 + 1 = 14×14
   ```

3. **Weight Validation**: Conv2d layers validate 4D weight tensors:
   ```python
   # Expected shape: (out_channels, in_channels, kernel_h, kernel_w)
   assert len(conv_weights.shape) == 4
   ```

4. **Mixed Architectures**: You can combine Conv2d and Linear layers freely:
   ```python
   # Conv → Conv → Flatten → FC → FC
   # All spatial dimensions are tracked automatically
   ```

### Example: MNIST CNN End-to-End

```python
from snn_fpga_accelerator import SNNAccelerator
from snn_fpga_accelerator.pytorch_interface import pytorch_to_snn
from snn_fpga_accelerator.spike_encoding import PoissonEncoder
import torch
import torch.nn as nn

# 1. Define and train model
class MNISTConvSNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 28 * 28, 128)  # No pooling: 28×28
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = MNISTConvSNN()
# Train model on MNIST...

# 2. Convert to SNN format
snn_model = pytorch_to_snn(
    model,
    input_shape=(1, 28, 28),
    neuron_params={'threshold': 1.0, 'leak': 0.95, 'refractory_period': 5}
)

# 3. Save model for later use
snn_model.save_weights('mnist_cnn_snn.h5')

# 4. Deploy to accelerator
accelerator = SNNAccelerator(simulation_mode=True)
accelerator.configure_network(snn_model)

# 5. Encode MNIST image and infer
encoder = PoissonEncoder(max_rate=100.0, duration=0.05)
mnist_image = ...  # Shape: (28, 28), values [0, 1]
input_spikes = encoder.encode(mnist_image.flatten())

output_spikes = accelerator.infer(input_spikes, duration=0.05)

# 6. Decode output
spike_counts = [0] * 10
for spike in output_spikes:
    if spike.neuron_id >= snn_model.total_neurons - 10:
        class_id = spike.neuron_id - (snn_model.total_neurons - 10)
        spike_counts[class_id] += 1

predicted_class = spike_counts.index(max(spike_counts))
print(f"Predicted digit: {predicted_class}")
```

### Weight Scaling for Fixed-Point Hardware

The converter automatically scales floating-point weights to 8-bit integers:

```python
# PyTorch weights: float32 range [-∞, +∞]
# FPGA weights: int8 range [-128, 127]

# Automatic scaling:
weight_min, weight_max = torch_weights.min(), torch_weights.max()
scale_factor = 127.0 / max(abs(weight_min), abs(weight_max))
fpga_weights = (torch_weights * scale_factor).astype(np.int8)
```

**Note**: This preserves weight ratios but may reduce precision. For better accuracy:
1. Use quantization-aware training
2. Clip outlier weights before conversion
3. Tune neuron threshold to match scaled weights

### Using Custom PyTorch Layers

```python
from snn_fpga_accelerator.pytorch_snn_layers import LIFLayer, STDPLayer

class CustomSNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.lif1 = LIFLayer(784, 200, threshold=1.0, leak=0.9)
        self.lif2 = LIFLayer(200, 10, threshold=1.0, leak=0.9)
        
    def forward(self, x):
        x, spikes1 = self.lif1(x)
        x, spikes2 = self.lif2(x)
        return x, (spikes1, spikes2)

model = CustomSNN()
```

### Training with STDP

```python
from snn_fpga_accelerator.learning import STDP

# Configure STDP learning
stdp = STDP(
    tau_plus=20.0,   # ms
    tau_minus=20.0,  # ms
    a_plus=0.1,
    a_minus=0.12,
    w_min=-1.0,
    w_max=1.0
)

# Training loop
for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        # Encode input
        input_spikes = encoder.encode(data.numpy())
        
        # Forward pass
        output_spikes = accelerator.infer_with_learning(
            input_spikes,
            learning_rule=stdp
        )
        
        # Compute loss and accuracy
        loss = compute_loss(output_spikes, target)
        print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss:.4f}")
```

## Learning Algorithms

### Standard STDP

Spike-Timing Dependent Plasticity:

```python
from snn_fpga_accelerator.learning import STDP

stdp = STDP(
    tau_plus=20.0,    # LTP time constant (ms)
    tau_minus=20.0,   # LTD time constant (ms)
    a_plus=0.1,       # LTP learning rate
    a_minus=0.12,     # LTD learning rate
    w_min=-1.0,       # Minimum weight
    w_max=1.0         # Maximum weight
)

# Apply to accelerator
accelerator.configure_learning(stdp.get_config())

# Run learning
for input_data, target in training_data:
    spikes = encoder.encode(input_data)
    output = accelerator.infer_with_learning(spikes)
```

### Reward-Modulated STDP (R-STDP)

For reinforcement learning tasks:

```python
from snn_fpga_accelerator.learning import RSTDPLearning

rstdp = RSTDPLearning(
    tau_plus=20.0,
    tau_minus=20.0,
    a_plus=0.1,
    a_minus=0.12,
    eligibility_decay=0.95,  # Eligibility trace decay
    learning_rate=0.01
)

accelerator.configure_learning(rstdp.get_config())

# Training with reward signal
for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0
    
    for step in range(max_steps):
        # Get action from SNN
        spikes = encoder.encode(state)
        output = accelerator.infer_with_learning(spikes)
        action = np.argmax(output.sum(axis=1))
        
        # Take action
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        
        # Apply reward signal
        accelerator.apply_reward(reward)
        
        if done:
            break
        state = next_state
    
    print(f"Episode {episode}, Total reward: {total_reward}")
```

### Online vs. Offline Learning

```python
# Online learning (weight updates after each sample)
accelerator.configure_learning(stdp.get_config(), mode='online')

# Offline learning (accumulate weight changes)
accelerator.configure_learning(stdp.get_config(), mode='offline')

# Update weights at end of epoch
accelerator.apply_accumulated_weight_updates()
```

## Hardware Deployment

Deploy your trained SNN on FPGA hardware.

### Programming the FPGA

```python
from snn_fpga_accelerator import SNNAccelerator

# Initialize with bitstream
accelerator = SNNAccelerator(
    bitstream_path='hardware/bitstream.bit',
    device='pynq-z2'
)

# Or program separately
accelerator = SNNAccelerator()
accelerator.program_fpga('hardware/bitstream.bit')
```

### PYNQ Board Setup

```python
from pynq import Overlay

# Load overlay
overlay = Overlay('hardware/bitstream.bit')

# Initialize accelerator with overlay
accelerator = SNNAccelerator(overlay=overlay)
```

### Performance Monitoring

```python
# Enable performance counters
accelerator.enable_monitoring()

# Run inference
output = accelerator.infer(input_spikes)

# Get statistics
stats = accelerator.get_performance_stats()
print(f"Inference time: {stats['inference_time_ms']:.2f} ms")
print(f"Spike throughput: {stats['spikes_per_second']:.0f} spikes/s")
print(f"Power consumption: {stats['power_watts']:.2f} W")
```

### Real-Time Inference

```python
import time

# Configure for real-time operation
accelerator.set_realtime_mode(True)

while True:
    # Capture input (e.g., from camera)
    frame = capture_frame()
    
    # Encode and infer
    spikes = encoder.encode(frame)
    output = accelerator.infer(spikes)
    
    # Process result
    prediction = np.argmax(output.sum(axis=1))
    print(f"Detected: {prediction}")
    
    time.sleep(0.01)  # 100Hz update rate
```

## Configuration

### Network Parameters

```python
config = {
    # Neuron parameters
    'num_neurons': 200,
    'neuron_type': 'LIF',  # Leaky Integrate-and-Fire
    'threshold': 1.0,      # Spike threshold
    'leak': 0.9,           # Membrane leak factor
    'refractory_period': 2, # Timesteps
    
    # Network topology
    'layers': [
        {
            'type': 'input',
            'size': 784,
            'connectivity': 'full'
        },
        {
            'type': 'hidden',
            'size': 200,
            'connectivity': 'full',
            'inhibitory_ratio': 0.2  # 20% inhibitory neurons
        },
        {
            'type': 'output',
            'size': 10,
            'connectivity': 'full'
        }
    ],
    
    # Simulation parameters
    'timestep': 0.001,     # 1ms
    'duration': 0.1,       # 100ms
    
    # Learning parameters (if using on-chip learning)
    'learning_enabled': True,
    'learning_rule': 'STDP',
    'tau_plus': 20.0,
    'tau_minus': 20.0,
    'a_plus': 0.1,
    'a_minus': 0.12
}

accelerator.configure_network(config)
```

### Loading Configuration from File

```python
import yaml

# Load from YAML
with open('network_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

accelerator.configure_network(config)

# Save configuration
accelerator.save_config('saved_config.yaml')
```

### Dynamic Reconfiguration

```python
# Update neuron parameters at runtime
accelerator.set_neuron_threshold(layer_id=1, threshold=1.5)
accelerator.set_leak_factor(layer_id=1, leak=0.85)

# Update learning rates
accelerator.set_learning_rate('a_plus', 0.15)
```

## Examples

### MNIST Classification

Complete example for digit recognition:

```python
import torch
from torchvision import datasets, transforms
from snn_fpga_accelerator import SNNAccelerator
from snn_fpga_accelerator.spike_encoding import PoissonEncoder

# Load MNIST data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
test_dataset = datasets.MNIST('./data', train=False, 
                             download=True, transform=transform)

# Initialize
accelerator = SNNAccelerator(simulation_mode=True)
encoder = PoissonEncoder(num_neurons=784, duration=0.1, max_rate=100.0)

# Configure network
config = {
    'layers': [
        {'type': 'input', 'size': 784},
        {'type': 'hidden', 'size': 400},
        {'type': 'output', 'size': 10}
    ]
}
accelerator.configure_network(config)

# Test
correct = 0
total = 0

for image, label in test_dataset:
    # Flatten and encode
    data = image.view(-1).numpy()
    spikes = encoder.encode(data)
    
    # Infer
    output = accelerator.infer(spikes)
    prediction = np.argmax(output.sum(axis=1))
    
    if prediction == label:
        correct += 1
    total += 1

accuracy = 100 * correct / total
print(f"Accuracy: {accuracy:.2f}%")
```

### Navigation Task with R-STDP

```python
import gym
from snn_fpga_accelerator.learning import RSTDPLearning

# Create environment
env = gym.make('CartPole-v1')

# Setup
accelerator = SNNAccelerator(simulation_mode=True)
encoder = PoissonEncoder(num_neurons=4, duration=0.05)
rstdp = RSTDPLearning(learning_rate=0.01, eligibility_decay=0.95)

# Configure network
config = {
    'layers': [
        {'type': 'input', 'size': 4},   # State space
        {'type': 'hidden', 'size': 20},
        {'type': 'output', 'size': 2}   # Actions
    ]
}
accelerator.configure_network(config)
accelerator.configure_learning(rstdp.get_config())

# Training
for episode in range(500):
    state = env.reset()
    total_reward = 0
    
    for t in range(200):
        # Encode state
        spikes = encoder.encode(state)
        
        # Get action
        output = accelerator.infer_with_learning(spikes)
        action = 1 if output[1].sum() > output[0].sum() else 0
        
        # Step environment
        state, reward, done, _ = env.step(action)
        total_reward += reward
        
        # Apply reward
        accelerator.apply_reward(reward)
        
        if done:
            break
    
    if episode % 50 == 0:
        print(f"Episode {episode}, Reward: {total_reward}")
```

### Custom Data Processing Pipeline

```python
from snn_fpga_accelerator import SNNAccelerator
from snn_fpga_accelerator.spike_encoding import TemporalEncoder
from snn_fpga_accelerator.utils import raster_plot
import matplotlib.pyplot as plt

# Custom preprocessing
def preprocess_audio(audio_data):
    # Apply FFT, normalization, etc.
    features = extract_features(audio_data)
    return features

# Setup pipeline
accelerator = SNNAccelerator(simulation_mode=True)
encoder = TemporalEncoder(num_neurons=128, duration=0.2)

# Load pre-trained weights
accelerator.load_weights('audio_classifier_weights.npy')

# Process audio file
audio = load_audio('sample.wav')
features = preprocess_audio(audio)
spikes = encoder.encode(features)

# Inference
output_spikes = accelerator.infer(spikes)

# Visualize
raster_plot(output_spikes, title='Output Spike Raster')
plt.show()

# Classification
class_probabilities = output_spikes.sum(axis=1) / output_spikes.sum()
predicted_class = np.argmax(class_probabilities)
print(f"Predicted class: {predicted_class}")
```

## Troubleshooting

### Common Issues

#### No Output Spikes

**Symptoms**: Network produces no output spikes

**Solutions**:
1. Check input encoding:
   ```python
   print(f"Input spikes: {input_spikes.sum()}")
   # Should be > 0
   ```

2. Verify neuron parameters:
   ```python
   # Lower threshold or increase input strength
   accelerator.set_neuron_threshold(layer_id=0, threshold=0.5)
   ```

3. Check weights:
   ```python
   weights = accelerator.get_layer_weights(layer_id=0)
   print(f"Weight range: [{weights.min()}, {weights.max()}]")
   # Weights should not all be zero or negative
   ```

#### Low Accuracy

**Symptoms**: Classification accuracy below expected

**Solutions**:
1. Increase simulation duration:
   ```python
   output = accelerator.infer(spikes, duration=0.2)  # Try longer
   ```

2. Adjust encoding parameters:
   ```python
   encoder = PoissonEncoder(
       num_neurons=784,
       duration=0.1,
       max_rate=150.0  # Increase max rate
   )
   ```

3. Retrain with better parameters:
   ```python
   # Adjust learning rates
   stdp.a_plus = 0.15
   stdp.a_minus = 0.18
   ```

#### FPGA Connection Issues

**Symptoms**: Cannot connect to PYNQ board

**Solutions**:
1. Check network connection:
   ```bash
   ping 192.168.2.99  # Default PYNQ IP
   ```

2. Verify bitstream:
   ```python
   try:
       accelerator.program_fpga('bitstream.bit')
   except Exception as e:
       print(f"Error: {e}")
   ```

3. Reset board:
   ```bash
   ssh xilinx@192.168.2.99
   sudo reboot
   ```

#### Memory Issues

**Symptoms**: Out of memory errors during training

**Solutions**:
1. Reduce batch size:
   ```python
   train_loader = DataLoader(dataset, batch_size=16)  # Smaller
   ```

2. Use offline learning:
   ```python
   accelerator.configure_learning(stdp, mode='offline')
   ```

3. Clear memory between epochs:
   ```python
   accelerator.reset_state()
   ```

### Performance Issues

#### Slow Inference

Check simulation vs hardware mode:
```python
# Use hardware for speed
accelerator = SNNAccelerator(
    bitstream_path='bitstream.bit',
    simulation_mode=False
)
```

Optimize encoding:
```python
# Use faster encoding schemes
encoder = RateEncoder(num_neurons=784, duration=0.05)
```

#### High Power Consumption

Reduce activity:
```python
# Lower spike rates
encoder = PoissonEncoder(max_rate=50.0)  # Reduce from 100Hz
```

Enable clock gating:
```python
accelerator.set_power_mode('low_power')
```

### Debugging Tools

Enable verbose logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)

accelerator = SNNAccelerator(verbose=True)
```

Visualize spike activity:
```python
from snn_fpga_accelerator.utils import raster_plot, spike_rate_plot

# Plot input spikes
raster_plot(input_spikes, title='Input Spikes')

# Plot firing rates over time
spike_rate_plot(output_spikes)
```

Check internal state:
```python
# Get membrane potentials
membrane_voltages = accelerator.get_membrane_state(layer_id=1)

# Plot
import matplotlib.pyplot as plt
plt.plot(membrane_voltages[0])  # First neuron
plt.xlabel('Time step')
plt.ylabel('Membrane Potential')
plt.title('Neuron Membrane Dynamics')
plt.show()
```

### Getting Help

If you encounter issues not covered here:

1. Check the [FAQ](https://github.com/metr0jw/Spiking-Neural-Network-on-FPGA/wiki/FAQ)
2. Search [existing issues](https://github.com/metr0jw/Spiking-Neural-Network-on-FPGA/issues)
3. Open a [new issue](https://github.com/metr0jw/Spiking-Neural-Network-on-FPGA/issues/new) with:
   - Detailed description of the problem
   - Code to reproduce the issue
   - Error messages and logs
   - System information (Python version, OS, etc.)

## Next Steps

- Explore the [API Reference](api_reference.md) for detailed function documentation
- Read the [Architecture Guide](architecture.md) to understand system internals
- Check the [Developer Guide](developer_guide.md) if you want to contribute
- Try the examples in the `examples/` directory
