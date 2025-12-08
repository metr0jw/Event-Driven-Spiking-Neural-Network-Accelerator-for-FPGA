# API Reference

Complete Python API documentation for the SNN FPGA Accelerator package.

## Table of Contents
- [PYNQ Driver](#pynq-driver)
- [Core Classes](#core-classes)
- [Spike Encoding](#spike-encoding)
- [Learning Algorithms](#learning-algorithms)
- [PyTorch Integration](#pytorch-integration)
- [Utilities](#utilities)
- [FPGA Controller](#fpga-controller)

## PYNQ Driver

Low-level driver for direct hardware control on PYNQ boards.

### SNNAccelerator (Hardware Driver)

Direct hardware interface using PYNQ overlay.

**Location**: `software/python/snn_driver.py`

```python
from snn_driver import SNNAccelerator

snn = SNNAccelerator(bitstream_path='snn_accelerator.bit')
```

#### Constructor

```python
SNNAccelerator(bitstream_path='snn_accelerator.bit')
```

**Parameters**:
- `bitstream_path` (str): Path to `.bit` file (requires matching `.hwh` file)

#### Methods

##### reset()
```python
reset() -> None
```
Reset the SNN accelerator hardware.

##### enable() / disable()
```python
enable() -> None
disable() -> None
```
Enable or disable the accelerator.

##### configure()
```python
configure(threshold=None, leak_rate=None, refractory_period=None) -> None
```
Configure neuron parameters.

**Parameters**:
- `threshold` (int): Spike threshold (0-65535)
- `leak_rate` (int): Membrane leak rate (0-65535)
- `refractory_period` (int): Refractory period in clock cycles

##### get_status()
```python
get_status() -> dict
```
Get accelerator status.

**Returns**:
```python
{
    'status_raw': int,      # Raw status register
    'spike_count': int,     # Output spike counter
    'enabled': bool         # Accelerator enabled state
}
```

##### clear_counters()
```python
clear_counters() -> None
```
Clear spike counters.

##### close()
```python
close() -> None
```
Disable accelerator and release resources.

#### Properties

- `threshold` (int): Get/set spike threshold
- `leak_rate` (int): Get/set membrane leak rate
- `refractory_period` (int): Get/set refractory period

#### Example Usage

```python
from snn_driver import SNNAccelerator

# Initialize
snn = SNNAccelerator('snn_accelerator.bit')

# Configure
snn.configure(threshold=100, leak_rate=16, refractory_period=5)

# Enable and monitor
snn.enable()
for _ in range(10):
    status = snn.get_status()
    print(f"Spikes: {status['spike_count']}")
    time.sleep(0.1)

# Cleanup
snn.close()
```

### SNNAcceleratorDMA

Extended driver with DMA support for high-throughput operations.

```python
from snn_driver import SNNAcceleratorDMA

snn = SNNAcceleratorDMA('snn_accelerator.bit')
```

#### Additional Methods

##### send_spikes_dma()
```python
send_spikes_dma(spike_data: np.ndarray) -> int
```
Send multiple spikes via DMA.

**Parameters**:
- `spike_data` (np.ndarray): Array of spike packets (uint32)
  - Format: `[neuron_id(8) | weight(8) | reserved(16)]`

**Returns**: Number of spikes sent

##### receive_spikes_dma()
```python
receive_spikes_dma(max_spikes=1024, timeout=1.0) -> np.ndarray
```
Receive output spikes via DMA.

##### process_input_spike_train()
```python
process_input_spike_train(spike_times, neuron_ids, weights) -> dict
```
Process a complete spike train.

**Returns**:
```python
{
    'output_spikes': np.ndarray,
    'total_output_spikes': int,
    'processing_time_ms': float,
    'spikes_per_second': float
}
```

---

## Hardware-Accurate Simulation

### HWAccurateLIFNeuron

Bit-accurate Python simulation of the Verilog LIF neuron.

**Location**: `software/python/snn_fpga_accelerator/hw_accurate_simulator.py`

```python
from snn_fpga_accelerator.hw_accurate_simulator import HWAccurateLIFNeuron, LIFNeuronParams
```

#### LIFNeuronParams

Configuration parameters matching hardware exactly.

```python
@dataclass
class LIFNeuronParams:
    threshold: int = 1000          # 16-bit threshold
    leak_rate: int = 3             # Encoded shift values (see below)
    leak_shift1: int = 3           # Primary leak shift
    leak_shift2: int = 0           # Secondary leak shift (0 = disabled)
    leak_shift2_enabled: bool = False
    refractory_period: int = 5     # 8-bit
    reset_potential: int = 0       # 16-bit
    reset_potential_en: bool = True
    tau: float = 0.875             # Effective decay factor
```

**leak_rate Encoding**:
- Bits [3:0]: shift1 (primary leak, 1-8)
- Bits [7:4]: shift2 (secondary leak, 0 = disabled, 1-8 if enabled)

**Tau Formula**: `tau = 1 - 2^(-shift1) - 2^(-shift2)`

##### from_tau() Class Method
```python
@classmethod
def from_tau(cls, tau: float, threshold: int = 1000, ...) -> LIFNeuronParams
```
Create parameters from target tau value.

**Example**:
```python
# Create parameters for tau=0.875
params = LIFNeuronParams.from_tau(tau=0.875, threshold=1000)
print(f"leak_rate={params.leak_rate}, shift1={params.leak_shift1}")
# Output: leak_rate=3, shift1=3
```

#### HWAccurateLIFNeuron

```python
class HWAccurateLIFNeuron:
    def __init__(self, neuron_id: int, params: LIFNeuronParams)
```

##### tick()
```python
def tick(
    self,
    syn_valid: bool = False,
    syn_weight: int = 0,
    syn_excitatory: bool = True,
    enable: bool = True
) -> bool
```
Execute one hardware clock cycle.

**Parameters**:
- `syn_valid` (bool): Synaptic input present
- `syn_weight` (int): 8-bit weight value (0-255)
- `syn_excitatory` (bool): True=excitatory (+weight), False=inhibitory (-weight)
- `enable` (bool): Neuron enable signal

**Returns**: True if spike generated

**Hardware Behavior**:
```
if syn_valid:
    v_mem_next = saturate_16bit(v_mem + weight)  # +/- based on exc/inh
else:
    leak = (v_mem >> shift1) + (v_mem >> shift2)
    v_mem_next = saturate_16bit(v_mem - leak)
    
if v_mem_next >= threshold:
    spike = True, v_mem = reset_potential
```

**Example**:
```python
params = LIFNeuronParams.from_tau(tau=0.875, threshold=1000)
neuron = HWAccurateLIFNeuron(neuron_id=0, params=params)

# Apply synaptic input
spike = neuron.tick(syn_valid=True, syn_weight=500, syn_excitatory=True)
print(f"v_mem={neuron.state.v_mem}, spike={spike}")

# Leak cycle (no input)
spike = neuron.tick(syn_valid=False, enable=True)
print(f"After leak: v_mem={neuron.state.v_mem}")
```

### Tau Conversion Utilities

```python
from snn_fpga_accelerator.hw_accurate_simulator import tau_to_hw_leak_rate, hw_leak_rate_to_tau
```

##### tau_to_hw_leak_rate()
```python
def tau_to_hw_leak_rate(tau: float) -> Tuple[int, int, int, bool]
```
Convert floating-point tau to hardware shift parameters.

**Returns**: `(leak_rate, shift1, shift2, shift2_enabled)`

##### hw_leak_rate_to_tau()
```python
def hw_leak_rate_to_tau(leak_rate: int) -> float
```
Convert hardware leak_rate encoding to effective tau.

**Common Tau Values**:
| tau   | leak_rate | shift1 | shift2 | Error |
|-------|-----------|--------|--------|-------|
| 0.500 | 1         | 1      | 0      | 0.000 |
| 0.750 | 2         | 2      | 0      | 0.000 |
| 0.875 | 3         | 3      | 0      | 0.000 |
| 0.900 | 44        | 4      | 5      | 0.006 |
| 0.9375| 4         | 4      | 0      | 0.000 |
| 0.950 | 53        | 5      | 6      | 0.003 |

---

## Core Classes

### SNNAccelerator

Main interface for FPGA-based SNN acceleration.

```python
class SNNAccelerator(bitstream_path=None, simulation_mode=False, device='pynq-z2', overlay=None, verbose=False)
```

**Parameters**:
- `bitstream_path` (str, optional): Path to FPGA bitstream file
- `simulation_mode` (bool): If True, run in software simulation mode
- `device` (str): Target device ('pynq-z2', 'zcu104', etc.)
- `overlay` (pynq.Overlay, optional): Existing PYNQ overlay object
- `verbose` (bool): Enable verbose logging

**Attributes**:
- `num_neurons` (int): Total number of neurons in network
- `is_configured` (bool): Whether network is configured
- `learning_enabled` (bool): Whether on-chip learning is active

#### Methods

##### configure_network()
```python
configure_network(config: Dict[str, Any]) -> None
```
Configure network topology and parameters.

**Parameters**:
- `config` (dict): Network configuration dictionary

**Configuration Keys**:
```python
{
    'num_neurons': int,              # Total neurons
    'neuron_type': str,              # 'LIF', 'Izhikevich', etc.
    'threshold': float,              # Spike threshold
    'leak': float,                   # Leak factor [0, 1]
    'refractory_period': int,        # Timesteps
    'layers': List[Dict],            # Layer configurations
    'timestep': float,               # Simulation timestep (s)
    'duration': float                # Default duration (s)
}
```

**Example**:
```python
config = {
    'num_neurons': 200,
    'threshold': 1.0,
    'leak': 0.9,
    'layers': [
        {'type': 'input', 'size': 784},
        {'type': 'hidden', 'size': 200},
        {'type': 'output', 'size': 10}
    ]
}
accelerator.configure_network(config)
```

##### load_weights()
```python
load_weights(weights: Union[np.ndarray, str], layer_id: Optional[int] = None) -> None
```
Load synaptic weights into FPGA memory.

**Parameters**:
- `weights` (np.ndarray or str): Weight matrix or path to .npy file
- `layer_id` (int, optional): Specific layer to load weights for

**Example**:
```python
# From array
weights = np.random.randn(784, 200) * 0.1
accelerator.load_weights(weights, layer_id=0)

# From file
accelerator.load_weights('trained_weights.npy')
```

##### infer()
```python
infer(
    input_spikes: np.ndarray,
    duration: Optional[float] = None,
    timestep: Optional[float] = None,
    return_membrane: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]
```
Run inference on input spike trains.

**Parameters**:
- `input_spikes` (np.ndarray): Input spikes, shape `(num_neurons, num_timesteps)`
- `duration` (float, optional): Simulation duration in seconds
- `timestep` (float, optional): Timestep in seconds
- `return_membrane` (bool): If True, also return membrane potentials

**Returns**:
- `output_spikes` (np.ndarray): Output spike trains
- `membrane_potentials` (np.ndarray, optional): Membrane voltage traces

**Example**:
```python
# Basic inference
output = accelerator.infer(input_spikes)

# With membrane potentials
output, membrane = accelerator.infer(input_spikes, return_membrane=True)
```

##### infer_with_learning()
```python
infer_with_learning(
    input_spikes: np.ndarray,
    learning_rule: Optional[LearningRule] = None,
    reward: Optional[float] = None
) -> np.ndarray
```
Run inference with online learning enabled.

**Parameters**:
- `input_spikes` (np.ndarray): Input spike trains
- `learning_rule` (LearningRule, optional): Learning algorithm to use
- `reward` (float, optional): Reward signal for R-STDP

**Returns**:
- `output_spikes` (np.ndarray): Output spike trains

**Example**:
```python
from snn_fpga_accelerator.learning import STDP

stdp = STDP(tau_plus=20.0, a_plus=0.1)
output = accelerator.infer_with_learning(input_spikes, learning_rule=stdp)
```

##### apply_reward()
```python
apply_reward(reward: float) -> None
```
Apply reward signal for R-STDP learning.

**Parameters**:
- `reward` (float): Reward value (can be negative for punishment)

**Example**:
```python
# After inference
output = accelerator.infer_with_learning(input_spikes)
action = get_action_from_output(output)
reward = environment.step(action)
accelerator.apply_reward(reward)
```

##### reset_state()
```python
reset_state() -> None
```
Reset all neuron states and eligibility traces.

##### get_performance_stats()
```python
get_performance_stats() -> Dict[str, float]
```
Get hardware performance statistics.

**Returns**:
- dict with keys: `inference_time_ms`, `spikes_per_second`, `power_watts`, `spike_count_in`, `spike_count_out`

**Example**:
```python
stats = accelerator.get_performance_stats()
print(f"Latency: {stats['inference_time_ms']:.2f} ms")
print(f"Throughput: {stats['spikes_per_second']:.0f} spikes/s")
```

##### save_config()
```python
save_config(filepath: str) -> None
```
Save current configuration to file.

##### load_config()
```python
load_config(filepath: str) -> None
```
Load configuration from file.

## Spike Encoding

### PoissonEncoder

Poisson-process rate encoding.

```python
class PoissonEncoder(num_neurons, duration, max_rate=100.0, min_rate=0.1, rng=None, seed=None)
```

**Parameters**:
- `num_neurons` (int): Number of output neurons
- `duration` (float): Encoding duration in seconds
- `max_rate` (float): Maximum firing rate in Hz
- `min_rate` (float): Minimum firing rate in Hz
- `rng` (numpy.random.Generator, optional): Random number generator
- `seed` (int, optional): Random seed for reproducibility

#### Methods

##### encode()
```python
encode(data: np.ndarray) -> np.ndarray
```
Encode data as Poisson spike trains.

**Parameters**:
- `data` (np.ndarray): Input data normalized to [0, 1], shape `(num_neurons,)`

**Returns**:
- `spikes` (np.ndarray): Spike trains, shape `(num_neurons, num_timesteps)`

**Example**:
```python
encoder = PoissonEncoder(num_neurons=784, duration=0.1, seed=42)
data = np.random.rand(784)
spikes = encoder.encode(data)
print(f"Generated {spikes.sum()} spikes")
```

### TemporalEncoder

Time-to-first-spike encoding.

```python
class TemporalEncoder(num_neurons, duration, encoding='linear')
```

**Parameters**:
- `num_neurons` (int): Number of output neurons
- `duration` (float): Maximum spike time in seconds
- `encoding` (str): Encoding scheme ('linear' or 'logarithmic')

#### Methods

##### encode()
```python
encode(data: np.ndarray) -> np.ndarray
```
Encode data using temporal coding (higher values spike earlier).

**Example**:
```python
encoder = TemporalEncoder(num_neurons=784, duration=0.1, encoding='linear')
data = np.random.rand(784)
spikes = encoder.encode(data)
```

### RateEncoder

Fixed-rate encoding based on input intensity.

```python
class RateEncoder(num_neurons, duration, max_rate=100.0)
```

#### Methods

##### encode()
```python
encode(data: np.ndarray) -> np.ndarray
```
Generate regular spike trains with rate proportional to input.

### PopulationEncoder

Encode scalar values using neuron populations.

```python
class PopulationEncoder(num_neurons_per_value, num_values, duration, min_val=0.0, max_val=1.0)
```

**Parameters**:
- `num_neurons_per_value` (int): Neurons per encoded value
- `num_values` (int): Number of values to encode
- `duration` (float): Encoding duration
- `min_val` (float): Minimum value range
- `max_val` (float): Maximum value range

#### Methods

##### encode()
```python
encode(values: List[float]) -> np.ndarray
```
Encode values using population coding.

**Example**:
```python
encoder = PopulationEncoder(num_neurons_per_value=10, num_values=2, duration=0.1)
values = [0.3, 0.7]
spikes = encoder.encode(values)
```

### PopulationDecoder

Decode spike trains from population-encoded outputs.

```python
class PopulationDecoder(num_neurons_per_class, num_classes)
```

#### Methods

##### decode()
```python
decode(spike_train: np.ndarray) -> int
```
Decode class from spike counts.

**Returns**:
- `class_id` (int): Predicted class

**Example**:
```python
decoder = PopulationDecoder(num_neurons_per_class=10, num_classes=10)
output_spikes = accelerator.infer(input_spikes)
prediction = decoder.decode(output_spikes)
```

## Learning Algorithms

### STDP

Spike-Timing Dependent Plasticity.

```python
class STDP(tau_plus=20.0, tau_minus=20.0, a_plus=0.1, a_minus=0.12, w_min=-1.0, w_max=1.0)
```

**Parameters**:
- `tau_plus` (float): LTP time constant in ms
- `tau_minus` (float): LTD time constant in ms
- `a_plus` (float): LTP magnitude
- `a_minus` (float): LTD magnitude
- `w_min` (float): Minimum weight
- `w_max` (float): Maximum weight

#### Methods

##### get_config()
```python
get_config() -> Dict[str, Any]
```
Get configuration dictionary for hardware.

##### compute_weight_update()
```python
compute_weight_update(
    pre_spike_times: np.ndarray,
    post_spike_times: np.ndarray,
    current_weight: float
) -> float
```
Compute weight change based on spike timing.

**Example**:
```python
stdp = STDP(tau_plus=20.0, tau_minus=20.0, a_plus=0.1, a_minus=0.12)

# Apply to accelerator
accelerator.configure_learning(stdp.get_config())

# Or use manually
pre_times = np.array([10, 25, 40])  # ms
post_times = np.array([15, 30])     # ms
weight_delta = stdp.compute_weight_update(pre_times, post_times, current_weight=0.5)
```

### RSTDPLearning

Reward-modulated STDP for reinforcement learning.

```python
class RSTDPLearning(
    tau_plus=20.0,
    tau_minus=20.0,
    a_plus=0.1,
    a_minus=0.12,
    eligibility_decay=0.95,
    learning_rate=0.01,
    w_min=-1.0,
    w_max=1.0
)
```

**Parameters**:
- Standard STDP parameters (inherited)
- `eligibility_decay` (float): Eligibility trace decay factor [0, 1]
- `learning_rate` (float): Global learning rate

#### Methods

##### get_config()
```python
get_config() -> Dict[str, Any]
```
Get configuration for hardware R-STDP.

##### update_eligibility()
```python
update_eligibility(stdp_delta: float, current_eligibility: float) -> float
```
Update eligibility trace.

##### apply_reward()
```python
apply_reward(
    reward: float,
    eligibility_trace: float,
    current_weight: float
) -> float
```
Apply reward signal to compute final weight update.

**Example**:
```python
rstdp = RSTDPLearning(
    tau_plus=20.0,
    tau_minus=20.0,
    a_plus=0.1,
    a_minus=0.12,
    eligibility_decay=0.95,
    learning_rate=0.01
)

accelerator.configure_learning(rstdp.get_config())

# Training loop
for episode in range(num_episodes):
    state = env.reset()
    for step in range(max_steps):
        spikes = encoder.encode(state)
        output = accelerator.infer_with_learning(spikes)
        action = get_action(output)
        state, reward, done, _ = env.step(action)
        accelerator.apply_reward(reward)
        if done:
            break
```

## PyTorch Integration

### pytorch_to_snn()

```python
pytorch_to_snn(model: nn.Module, input_shape: Optional[Tuple] = None) -> Dict[str, Any]
```
Convert PyTorch model to SNN configuration.

**Parameters**:
- `model` (nn.Module): PyTorch model to convert
- `input_shape` (tuple, optional): Input shape for tracing

**Returns**:
- `config` (dict): Network configuration compatible with `configure_network()`

**Example**:
```python
import torch.nn as nn
from snn_fpga_accelerator.pytorch_interface import pytorch_to_snn

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 200)
        self.fc2 = nn.Linear(200, 10)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleNet()
config = pytorch_to_snn(model, input_shape=(1, 784))
accelerator.configure_network(config)
```

### LIFLayer

PyTorch-compatible LIF neuron layer.

```python
class LIFLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        threshold: float = 1.0,
        leak: float = 0.9,
        refractory_period: int = 2
    )
```

**Parameters**:
- `in_features` (int): Input dimension
- `out_features` (int): Output dimension
- `threshold` (float): Spike threshold
- `leak` (float): Membrane leak factor
- `refractory_period` (int): Refractory period in timesteps

#### Methods

##### forward()
```python
forward(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
```
Forward pass with LIF dynamics.

**Returns**:
- `membrane` (torch.Tensor): Membrane potentials
- `spikes` (torch.Tensor): Output spikes

**Example**:
```python
from snn_fpga_accelerator.pytorch_snn_layers import LIFLayer

class SNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lif1 = LIFLayer(784, 200, threshold=1.0, leak=0.9)
        self.lif2 = LIFLayer(200, 10, threshold=1.0, leak=0.9)
        
    def forward(self, x):
        membrane1, spikes1 = self.lif1(x)
        membrane2, spikes2 = self.lif2(spikes1)
        return spikes2

model = SNNModel()
```

### LIF (Neuron Class with HW Mode)

PyTorch-compatible LIF neuron with hardware-accurate mode.

**Location**: `software/python/snn_fpga_accelerator/neuron.py`

```python
from snn_fpga_accelerator.neuron import LIF

class LIF(nn.Module):
    def __init__(
        self,
        thresh: float = 1.0,
        tau: float = 0.9,
        reset: str = 'zero',      # 'zero' or 'subtract'
        hw_mode: bool = False
    )
```

**Parameters**:
- `thresh` (float): Spike threshold
- `tau` (float): Membrane decay factor (0 < tau < 1)
- `reset` (str): Reset mechanism ('zero' for hard reset, 'subtract' for soft reset)
- `hw_mode` (bool): If True, use shift-based leak matching hardware

**Hardware Mode (`hw_mode=True`)**:

When `hw_mode=True`, the neuron uses shift-based exponential decay matching
the Verilog RTL implementation:

```python
# Standard mode (float multiplication):
mem_next = tau * mem + input

# Hardware mode (shift-based leak):
leak = (mem >> shift1) + (mem >> shift2)
mem_next = mem - leak + input
```

##### forward()
```python
forward(x: torch.Tensor) -> torch.Tensor
```
Forward pass with LIF dynamics.

**Returns**: Spike tensor (same shape as input)

##### get_hw_config()
```python
def get_hw_config(self) -> Dict[str, Any]
```
Get hardware configuration for deployment.

**Returns**:
```python
{
    'leak_rate': int,        # Encoded shift values
    'shift1': int,           # Primary leak shift
    'shift2': int,           # Secondary leak shift
    'effective_tau': float,  # Actual tau achieved
    'threshold': int         # 16-bit threshold
}
```

##### reset_state()
```python
def reset_state(self) -> None
```
Reset membrane potential to zero.

**Example**:
```python
from snn_fpga_accelerator.neuron import LIF
import torch

# Standard mode (for training)
lif = LIF(thresh=1.0, tau=0.9, hw_mode=False)

# Hardware mode (for validation)
lif_hw = LIF(thresh=1.0, tau=0.875, hw_mode=True)
config = lif_hw.get_hw_config()
print(f"HW config: {config}")
# Output: {'leak_rate': 3, 'shift1': 3, 'shift2': 0, 'effective_tau': 0.875, ...}

# Forward pass
x = torch.randn(1, 10) * 2
spikes = lif_hw(x)
print(f"Membrane: {lif_hw.mem}, Spikes: {spikes}")
```

### STDPLayer

STDP learning in PyTorch.

```python
class STDPLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        tau_plus: float = 20.0,
        tau_minus: float = 20.0,
        a_plus: float = 0.1,
        a_minus: float = 0.12
    )
```

## Utilities

### load_weights()

```python
load_weights(filepath: str) -> np.ndarray
```
Load weights from file.

**Parameters**:
- `filepath` (str): Path to .npy or .h5 file

**Returns**:
- `weights` (np.ndarray): Weight array

### save_weights()

```python
save_weights(weights: np.ndarray, filepath: str) -> None
```
Save weights to file.

### visualize_spikes()

```python
visualize_spikes(
    spike_train: np.ndarray,
    title: str = "Spike Raster Plot",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> None
```
Create raster plot of spike trains.

**Parameters**:
- `spike_train` (np.ndarray): Spikes, shape `(num_neurons, num_timesteps)`
- `title` (str): Plot title
- `figsize` (tuple): Figure size
- `save_path` (str, optional): Path to save figure

**Example**:
```python
from snn_fpga_accelerator.utils import visualize_spikes

spikes = encoder.encode(data)
visualize_spikes(spikes, title="Input Spikes", save_path="input_raster.png")
```

### raster_plot()

```python
raster_plot(spike_train: np.ndarray, ax: Optional[plt.Axes] = None, **kwargs) -> plt.Axes
```
Low-level raster plot function.

### spike_rate_plot()

```python
spike_rate_plot(
    spike_train: np.ndarray,
    window_size: int = 10,
    ax: Optional[plt.Axes] = None
) -> plt.Axes
```
Plot average firing rate over time.

**Example**:
```python
from snn_fpga_accelerator.utils import spike_rate_plot
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
raster_plot(spikes, ax=ax1)
spike_rate_plot(spikes, window_size=20, ax=ax2)
plt.show()
```

### convert_to_raster()

```python
convert_to_raster(spike_times: List[np.ndarray], num_timesteps: int) -> np.ndarray
```
Convert spike time lists to raster format.

**Parameters**:
- `spike_times` (list): List of spike time arrays per neuron
- `num_timesteps` (int): Total number of timesteps

**Returns**:
- `raster` (np.ndarray): Binary spike raster

## FPGA Controller

### FPGAController

Low-level FPGA control interface.

```python
class FPGAController(bitstream_path: str, device: str = 'pynq-z2')
```

#### Methods

##### write_register()
```python
write_register(offset: int, value: int) -> None
```
Write to AXI-Lite register.

##### read_register()
```python
read_register(offset: int) -> int
```
Read from AXI-Lite register.

##### write_memory()
```python
write_memory(address: int, data: np.ndarray) -> None
```
Write to FPGA memory (DMA).

##### read_memory()
```python
read_memory(address: int, size: int) -> np.ndarray
```
Read from FPGA memory (DMA).

##### send_spikes()
```python
send_spikes(spike_data: np.ndarray) -> None
```
Send spikes via AXI-Stream.

##### receive_spikes()
```python
receive_spikes(timeout: float = 1.0) -> np.ndarray
```
Receive spikes from AXI-Stream.

**Example**:
```python
from snn_fpga_accelerator.fpga_controller import FPGAController

controller = FPGAController('bitstream.bit')

# Configure neuron threshold
THRESHOLD_REG = 0x10
controller.write_register(THRESHOLD_REG, int(1.0 * 256))

# Read status
STATUS_REG = 0x04
status = controller.read_register(STATUS_REG)
print(f"Status: 0x{status:08x}")
```

## Constants

### Register Offsets

```python
# Control and status
CTRL_REG = 0x00
STATUS_REG = 0x04
NUM_NEURONS_REG = 0x08
TIMESTEP_REG = 0x0C

# Neuron parameters
THRESHOLD_GLOBAL_REG = 0x10
LEAK_GLOBAL_REG = 0x14

# Learning
LEARNING_EN_REG = 0x18
LEARNING_RATE_REG = 0x1C

# Performance counters
SPIKE_COUNT_IN_REG = 0x20
SPIKE_COUNT_OUT_REG = 0x24
```

### Control Bits

```python
CTRL_START = 0x01
CTRL_STOP = 0x02
CTRL_RESET = 0x04
CTRL_LEARN_EN = 0x08
```

### Status Bits

```python
STATUS_BUSY = 0x01
STATUS_DONE = 0x02
STATUS_ERROR = 0x04
STATUS_LEARNING = 0x08
```

## Error Handling

All functions raise appropriate exceptions:

- `ValueError`: Invalid parameters or configuration
- `RuntimeError`: FPGA communication or hardware errors
- `FileNotFoundError`: Missing bitstream or weight files
- `TimeoutError`: FPGA operation timeout

**Example**:
```python
try:
    accelerator = SNNAccelerator(bitstream_path='bitstream.bit')
    accelerator.configure_network(config)
except FileNotFoundError as e:
    print(f"Bitstream not found: {e}")
except RuntimeError as e:
    print(f"Hardware error: {e}")
```

## Type Hints

The package uses type hints throughout. Import types:

```python
from typing import Optional, List, Dict, Tuple, Union
import numpy as np
import torch
```

## Version Information

```python
import snn_fpga_accelerator
print(snn_fpga_accelerator.__version__)  # e.g., "0.1.0"
```

## Next Steps

- See the [User Guide](user_guide.md) for practical examples
- Read the [Architecture Documentation](architecture.md) for system internals
- Check the [Developer Guide](developer_guide.md) for contributing
