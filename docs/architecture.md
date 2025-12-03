# Architecture Documentation

Detailed system architecture of the Event-Driven SNN FPGA Accelerator.

## Table of Contents
- [System Overview](#system-overview)
- [Hardware Architecture](#hardware-architecture)
- [Software Architecture](#software-architecture)
- [Data Flow](#data-flow)
- [Memory Organization](#memory-organization)
- [Communication Interfaces](#communication-interfaces)
- [Learning Engine](#learning-engine)

## System Overview

### High-Level Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    Software Layer (Python)                    │
├──────────────────────────────────────────────────────────────┤
│  PyTorch Interface │ Spike Encoding │ Learning Algorithms    │
│  Model Conversion  │ Visualization  │ Configuration          │
└──────────────────────────────────────────────────────────────┘
                              ↕ AXI Bus
┌──────────────────────────────────────────────────────────────┐
│                   FPGA Hardware (PYNQ-Z2)                     │
├──────────────────────────────────────────────────────────────┤
│  ┌────────────┐  ┌────────────┐  ┌──────────────────────┐  │
│  │ AXI        │  │ Spike      │  │  Learning Engine     │  │
│  │ Interface  │→ │ Router     │→ │  (STDP/R-STDP)       │  │
│  └────────────┘  └────────────┘  └──────────────────────┘  │
│         ↓               ↓                    ↓               │
│  ┌────────────────────────────────────────────────────────┐ │
│  │           LIF Neuron Arrays (Time-Multiplexed)         │ │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐             │ │
│  │  │ Neuron   │  │ Neuron   │  │ Neuron   │   ...       │ │
│  │  │  Array   │  │  Array   │  │  Array   │             │ │
│  │  └──────────┘  └──────────┘  └──────────┘             │ │
│  └────────────────────────────────────────────────────────┘ │
│         ↓               ↓                    ↓               │
│  ┌────────────────────────────────────────────────────────┐ │
│  │         Synaptic Weight Memory (BRAM/UltraRAM)         │ │
│  └────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

### Design Principles

1. **Event-Driven Processing**: Asynchronous spike-based computation
2. **Time Multiplexing**: Efficient resource utilization for scalability
3. **Fixed-Point Arithmetic**: Optimized for FPGA implementation
4. **Modular Design**: Configurable layers and learning algorithms
5. **Low Latency**: Microsecond-scale spike processing

## Hardware Architecture

### Core Components

#### 1. LIF Neuron Implementation

**File**: `hardware/hdl/rtl/neurons/lif_neuron.v`

The Leaky Integrate-and-Fire neuron is the fundamental computational unit.

**State Variables**:
- `membrane_potential`: 16-bit signed fixed-point (Q8.8 format)
- `refractory_counter`: 8-bit unsigned counter
- `neuron_type`: 1-bit (excitatory/inhibitory)

**Operations**:
```
V[t+1] = V[t] × leak_factor + Σ(w_i × spike_i)

if V[t+1] > threshold:
    spike_out = 1
    V[t+1] = reset_potential
    refractory_counter = refractory_period
```

**Parameters**:
- Threshold: Configurable, typical range [0.5, 2.0]
- Leak factor: 8-bit unsigned, typical 0.9 (230/256)
- Refractory period: 1-15 timesteps
- Reset potential: Typically 0 or small negative value

**Interface**:
```verilog
module lif_neuron #(
    parameter WEIGHT_WIDTH = 8,
    parameter MEMBRANE_WIDTH = 16
) (
    input wire clk,
    input wire rst_n,
    
    // Spike input
    input wire spike_in_valid,
    input wire signed [WEIGHT_WIDTH-1:0] weight,
    
    // Configuration
    input wire signed [MEMBRANE_WIDTH-1:0] threshold,
    input wire [7:0] leak_factor,
    input wire [3:0] refractory_period,
    input wire neuron_type,  // 0: excitatory, 1: inhibitory
    
    // Spike output
    output reg spike_out,
    output reg signed [MEMBRANE_WIDTH-1:0] membrane_potential
);
```

#### 2. Neuron Array (Time-Multiplexed)

**File**: `hardware/hdl/rtl/neurons/lif_neuron_array.v`

Time-multiplexed array for scalable neuron processing.

**Key Features**:
- Processes N neurons using M physical neuron cores (M < N)
- State memory for all neurons stored in BRAM
- Round-robin scheduling for fair processing
- Configurable array size via parameters

**Architecture**:
```
┌─────────────────────────────────────┐
│     Neuron State Memory (BRAM)      │
│  ┌───┬───┬───┬───┬───┬───┬───┬───┐ │
│  │N0 │N1 │N2 │N3 │...│N62│N63│...│ │
│  └───┴───┴───┴───┴───┴───┴───┴───┘ │
└──────────────┬──────────────────────┘
               ↓
     ┌─────────────────────┐
     │  Scheduler/Arbiter  │
     └─────────────────────┘
               ↓
┌──────────────────────────────────┐
│   Physical Neuron Cores (×4)     │
│  ┌────┐ ┌────┐ ┌────┐ ┌────┐    │
│  │LIF │ │LIF │ │LIF │ │LIF │    │
│  │ 0  │ │ 1  │ │ 2  │ │ 3  │    │
│  └────┘ └────┘ └────┘ └────┘    │
└──────────────────────────────────┘
```

**Timing**:
- One neuron processed per clock cycle per core
- Full array update: ⌈N/M⌉ clock cycles
- At 100 MHz: 640ns for 64 neurons (M=4)

#### 3. Spike Router

**File**: `hardware/hdl/rtl/router/spike_router.v`

Routes spikes from source neurons to target neurons based on connectivity.

**Functionality**:
- AXI-Stream input interface for incoming spikes
- Priority-based routing for temporal ordering
- Fanout handling for one-to-many connections
- Configurable routing table (stored in BRAM)

**Routing Entry Format**:
```
[63:48] - Reserved
[47:40] - Target Neuron ID (8-bit)
[39:32] - Source Neuron ID (8-bit)
[31:24] - Reserved
[23:16] - Weight Index
[15:8]  - Priority
[7:0]   - Flags (valid, inhibitory, etc.)
```

**Performance**:
- Throughput: One spike routed per clock cycle
- Latency: 2-5 clock cycles depending on fanout
- Max fanout: Configurable, typical 1-100

#### 4. Synaptic Weight Memory

**File**: `hardware/hdl/rtl/synapses/synapse_array.v`, `weight_memory.v`

Stores synaptic weights for neuron connections.

**Memory Organization**:
```
Address: [Src_Neuron_ID][Dst_Neuron_ID]
Data:    8-bit signed weight (-127 to +127)

Example for 64 neurons:
- Total connections: 64 × 64 = 4,096
- Memory size: 4,096 × 8 bits = 4 KB
- Implementation: 2× BRAM18K (dual-port)
```

**Access Pattern**:
- Port A: Weight updates (from learning engine)
- Port B: Weight reads (for spike processing)
- Dual-port allows simultaneous learning and inference

**Weight Format**:
- 8-bit signed fixed-point
- Range: [-1.0, +1.0] mapped to [-127, +127]
- Saturation arithmetic on updates

#### 5. Convolutional Layers

**File**: `hardware/hdl/rtl/layers/snn_conv1d.v`, `snn_conv2d.v`

Event-driven convolution for spatial/temporal feature extraction.

**1D Convolution Architecture**:
```
Input Spike → Kernel Application → Membrane Update → Threshold → Output Spike
    ↓                ↓                    ↓              ↓            ↓
[neuron_id]    [weight_mem]        [accumulator]   [comparator]  [FIFO]
```

**Key Features**:
- Weight loading via AXI-Lite interface
- Sequential kernel application across channels
- Per-spike processing (event-driven)
- Output spike FIFO for rate matching
- Configurable kernel size, stride, padding

**Parameters**:
- `INPUT_CHANNELS`: Number of input feature maps
- `OUTPUT_CHANNELS`: Number of output feature maps
- `KERNEL_SIZE`: Convolution kernel size (1D: 1-16, 2D: 1x1 to 7x7)
- `STRIDE`: Step size for kernel application
- `WEIGHT_WIDTH`: 8-bit (signed)

#### 6. AXI Interface Wrapper

**File**: `hardware/hdl/rtl/interfaces/axi_wrapper.v`

Bridges between PS (ARM processor) and PL (FPGA fabric).

**Interface Types**:

**AXI-Lite (Control/Status)**:
- Base address: 0x4000_0000
- Register map:
  - 0x00: Control register
  - 0x04: Status register
  - 0x08: Configuration
  - 0x0C: Neuron count
  - 0x10-0xFF: Layer-specific registers

**AXI-Stream (Data)**:
- Spike input stream
- Spike output stream
- Format: {timestamp[31:16], neuron_id[15:8], weight[7:0]}

**DMA (Optional)**:
- Bulk weight transfer
- Batch spike injection
- Output spike collection

### Resource Utilization (Verified Build)

Actual utilization for PYNQ-Z2 block design with PS (XC7Z020):

| Resource    | Used  | Available | Utilization |
|-------------|-------|-----------|-------------|
| LUTs        | 4,689 | 53,200    | 8.81%       |
| Registers   | 3,212 | 106,400   | 3.02%       |
| Slices      | 1,620 | 13,300    | 12.18%      |
| BRAM36K     | 2     | 140       | 1.4%        |
| DSPs        | 0     | 220       | 0%          |

**Timing Results (100 MHz)**:
| Metric | Value | Status |
|--------|-------|--------|
| WNS (Setup) | +0.159 ns | PASS |
| WHS (Hold) | +0.057 ns | PASS |
| WPWS (Pulse Width) | +3.750 ns | PASS |
| Timing Violations | 0 | OK |

### Clock Domains

- **System Clock**: 100 MHz (main processing)
- **AXI Clock**: 100 MHz (synchronized with system)
- **Optional High-Speed Clock**: 200 MHz (for learning engine)

All clock domain crossings handled with synchronizers in `hardware/hdl/rtl/common/sync_pulse.v`.

## Software Architecture

### Python Package Structure

```
snn_fpga_accelerator/
├── __init__.py              # Package initialization
├── accelerator.py           # Main FPGA interface class
├── pytorch_interface.py     # PyTorch model conversion
├── pytorch_snn_layers.py    # Custom PyTorch SNN layers
├── spike_encoding.py        # Encoding/decoding algorithms
├── learning.py              # STDP/R-STDP implementations
├── fpga_controller.py       # Low-level FPGA control
├── cli.py                   # Command-line interface
└── utils.py                 # Utilities and visualization
```

### Key Classes

#### SNNAccelerator

Main interface for FPGA interaction.

```python
class SNNAccelerator:
    """FPGA-based SNN accelerator interface."""
    
    def __init__(
        self,
        bitstream_path: Optional[str] = None,
        simulation_mode: bool = False,
        device: str = 'pynq-z2'
    ):
        """Initialize accelerator."""
        
    def configure_network(self, config: Dict) -> None:
        """Configure network topology and parameters."""
        
    def load_weights(self, weights: np.ndarray) -> None:
        """Load synaptic weights."""
        
    def infer(
        self,
        input_spikes: np.ndarray,
        duration: float = 0.1,
        timestep: float = 0.001
    ) -> np.ndarray:
        """Run inference and return output spikes."""
        
    def infer_with_learning(
        self,
        input_spikes: np.ndarray,
        learning_rule: Optional[LearningRule] = None
    ) -> np.ndarray:
        """Run inference with online learning."""
```

#### Spike Encoders

Convert conventional data to spike trains.

```python
class PoissonEncoder:
    """Poisson-process spike encoding."""
    
    def encode(self, data: np.ndarray) -> np.ndarray:
        """
        Encode data as Poisson spike trains.
        
        Args:
            data: Input array normalized to [0, 1]
            
        Returns:
            Spike trains: shape (num_neurons, num_timesteps)
        """
```

#### Learning Rules

```python
class STDPLearning:
    """Spike-Timing Dependent Plasticity."""
    
    def __init__(
        self,
        tau_plus: float = 20.0,
        tau_minus: float = 20.0,
        a_plus: float = 0.1,
        a_minus: float = 0.12
    ):
        """Initialize STDP parameters."""
        
    def compute_weight_update(
        self,
        pre_spike_times: np.ndarray,
        post_spike_times: np.ndarray
    ) -> float:
        """Compute weight change based on spike timing."""
```

### PyTorch Integration

#### Model Conversion Pipeline

```python
def pytorch_to_snn(model: nn.Module) -> Dict:
    """
    Convert PyTorch model to SNN configuration.
    
    1. Extract layer structure
    2. Convert weights to fixed-point
    3. Generate connectivity map
    4. Create configuration dictionary
    """
```

#### Custom PyTorch Layers

```python
class LIFLayer(nn.Module):
    """PyTorch-compatible LIF neuron layer."""
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with membrane dynamics.
        
        Returns:
            membrane_potentials, spikes
        """
```

## Data Flow

### Inference Pipeline

```
1. Input Data
      ↓
2. Spike Encoding (Software)
      ↓ [Transfer via AXI]
3. Spike Router (Hardware)
      ↓
4. Neuron Processing
   - Weight lookup
   - Membrane update
   - Threshold detection
      ↓
5. Output Spikes
      ↓ [Transfer via AXI]
6. Spike Decoding (Software)
      ↓
7. Result
```

### Learning Pipeline

```
1. Forward Pass (Inference)
      ↓
2. Spike Timing Capture
      ↓
3. Weight Update Calculation
   - STDP traces
   - Eligibility traces (R-STDP)
   - Reward signal (R-STDP)
      ↓
4. Weight Memory Update
      ↓
5. Next Iteration
```

## Memory Organization

### Neuron State Memory

```
Offset  | Field                    | Width | Description
--------|--------------------------|-------|---------------------------
0x00    | Membrane Potential       | 16b   | Signed Q8.8 fixed-point
0x02    | Threshold                | 16b   | Signed Q8.8 fixed-point
0x04    | Leak Factor              | 8b    | Unsigned [0, 255] → [0, 1]
0x05    | Refractory Period        | 4b    | Timesteps
0x05.4  | Refractory Counter       | 4b    | Current refractory state
0x06    | Neuron Type              | 1b    | 0: excit., 1: inhib.
0x06.1  | Flags                    | 7b    | Reserved
0x07    | Reserved                 | 8b    | Future use
```

Total: 8 bytes per neuron
- 64 neurons: 512 bytes (1× BRAM18K)
- 1024 neurons: 8 KB (1× BRAM18K)

### Weight Memory Layout

**Fully Connected**:
```
Address = (src_neuron_id << log2(num_neurons)) | dst_neuron_id
Data = 8-bit signed weight
```

**Convolutional**:
```
Address = (output_channel << K) | (input_channel << K') | kernel_index
Data = 8-bit signed weight
```

### Configuration Memory

```
Offset  | Register                 | Access | Description
--------|--------------------------|--------|---------------------------
0x00    | CTRL                     | RW     | Control register (enable/reset/counter control/IRQ enable)
0x04    | STATUS                   | RO     | Consolidated fabric status flags
0x08    | CONFIG                   | RW     | Write-gating bits for fabric configuration paths
0x0C    | SPIKE_COUNT              | RO     | Accumulated output spike count
0x10    | LEAK_RATE                | RW     | Global neuron leak factor (16-bit)
0x14    | THRESHOLD                | RW     | Global neuron firing threshold (16-bit)
0x18    | REFRACTORY               | RW     | Global refractory period (16-bit)
0x1C    | VERSION                  | RO     | Firmware version signature
0x20+   | Reserved                 | --     | Future expansion
```

## Communication Interfaces

### AXI-Lite Register Interface

**Control Register (0x00)**:
```
Bit  | Field        | Description
-----|--------------|--------------------------------
0    | ENABLE       | Assert to enable the accelerator fabric
1    | SOFT_RESET   | Active-high soft reset into sub-modules
2    | CLEAR_CNTS   | Pulse high to clear spike/performance counters
3    | IRQ_ENABLE   | Enable spike-count threshold interrupt
4-31 | Reserved     |
```

**Status Register (0x04)**:
```
Bit  | Field            | Description
-----|------------------|--------------------------------
0    | ENABLED          | Mirrors CTRL.ENABLE
1    | INPUT_SEEN       | At least one input spike observed
2    | NEURON_SPIKE     | Neuron array has pending spike
3    | OUTPUT_VALID     | Output spike pending toward PS
4-6  | Reserved         |
7    | ACTIVITY         | Any neuron fired in the recent window
8-11 | Reserved         |
12   | Reserved         |
13   | ARRAY_BUSY       | Neuron array currently integrating
14   | ROUTER_BUSY      | Spike router has pending traffic
15   | FIFO_OVERFLOW    | Router FIFO overflow detected
16-31| Reserved         |
```

**Configuration Register (0x08)**:
```
Bit  | Field            | Description
-----|------------------|--------------------------------
0-7  | Reserved         |
8    | WEIGHT_WE        | Gate writes into synapse weight memory (AW[15:12]==0x1)
9    | NEURON_CFG_WE    | Gate writes into neuron configuration space (AW[15:12]==0x2)
10   | ROUTER_CFG_WE    | Gate writes into router configuration space (AW[15:12]==0x3)
11   | SPIKE_THRESH_WE  | Allow writes to interrupt threshold register (AW[7:0]==0x20)
12-31| Reserved         |
```

**Version Register (0x1C)**: Constant `0x2024_0100` identifying the RTL drop.

### AXI-Stream Spike Interface

**Input Spike Format**:
```
[31:24] - Timestamp (upper 8 bits)
[23:16] - Timestamp (lower 8 bits)
[15:8]  - Neuron ID
[7:0]   - Weight/Intensity (optional)
```

**Output Spike Format**:
```
[31:24] - Timestamp (upper 8 bits)
[23:16] - Timestamp (lower 8 bits)
[15:8]  - Neuron ID
[7:0]   - Reserved
```

## Learning Engine

### STDP Implementation

**Trace Update (Hardware)**:
```verilog
// Pre-synaptic trace
if (pre_spike)
    pre_trace <= MAX_TRACE;
else
    pre_trace <= (pre_trace * decay_factor) >> 8;

// Post-synaptic trace
if (post_spike)
    post_trace <= MAX_TRACE;
else
    post_trace <= (post_trace * decay_factor) >> 8;

// Weight update
if (pre_spike && post_trace > 0)
    weight_delta <= (a_plus * post_trace) >>> 8;  // LTP
else if (post_spike && pre_trace > 0)
    weight_delta <= -(a_minus * pre_trace) >>> 8;  // LTD
```

**Parameters**:
- `tau_plus`: 20ms (default)
- `tau_minus`: 20ms (default)
- `a_plus`: 0.1 (LTP magnitude)
- `a_minus`: 0.12 (LTD magnitude, typically > a_plus)
- `decay_factor`: exp(-dt/tau) ≈ 230/256 for dt=1ms, tau=20ms

### R-STDP Extension

**Eligibility Traces**:
```
e[t+1] = e[t] × decay + Δw_STDP[t]

where:
- e[t]: eligibility trace
- decay: 0.95 (typical)
- Δw_STDP[t]: standard STDP weight change
```

**Reward Modulation**:
```
Δw_final[t] = learning_rate × reward[t] × e[t]
```

Applied when reward signal arrives (potentially delayed).

### Hardware Learning Engine

**File**: `hardware/hls/src/snn_learning_engine.cpp`

HLS-based learning engine for complex algorithms.

**Interface**:
```cpp
void snn_learning_engine(
    // Spike streams
    hls::stream<spike_t> &pre_spikes,
    hls::stream<spike_t> &post_spikes,
    
    // Weight memory
    weight_t *weight_mem,
    
    // Configuration
    learning_config_t *config,
    
    // Reward signal (R-STDP)
    hls::stream<reward_t> &reward_stream
);
```

**Pipelining**:
- Initiation Interval (II): 1 cycle
- Latency: 10-15 cycles
- Throughput: ~10 million weight updates/second @ 100 MHz

## Performance Characteristics

### Latency Breakdown

| Stage                    | Latency      | Notes                        |
|--------------------------|--------------|------------------------------|
| AXI Transfer (input)     | 100-500 ns   | Depends on spike count       |
| Spike Routing            | 20-50 ns     | 2-5 clock cycles             |
| Neuron Update            | 10 ns        | 1 clock cycle per neuron     |
| Spike Output             | 10 ns        | 1 clock cycle                |
| AXI Transfer (output)    | 100-500 ns   | Depends on spike count       |
| **Total (per spike)**    | **~1 μs**    | Event-driven, scales w/ activity |

### Throughput

- **Spike Processing**: 100M spikes/second @ 100 MHz (theoretical)
- **Actual Throughput**: 10-50M spikes/second (depends on spike density)
- **Neuron Updates**: 6.4M neuron-updates/second (64 neurons × 100 kHz)

### Power Consumption

Measured on PYNQ-Z2:
- Idle: ~0.5 W
- Inference (low activity): ~1.5 W
- Inference (high activity): ~2.5 W
- With learning: ~3.0 W

Power breakdown:
- PS (ARM): ~40%
- PL (FPGA fabric): ~50%
- Memory I/O: ~10%

### Scalability

| Configuration      | Neurons | Synapses | BRAM | Throughput    |
|--------------------|---------|----------|------|---------------|
| Small              | 64      | 4K       | 18   | 10M spikes/s  |
| Medium             | 256     | 64K      | 48   | 30M spikes/s  |
| Large              | 1024    | 1M       | 120  | 80M spikes/s  |

## Next Steps

- Explore the [API Reference](api_reference.md) for detailed function documentation
- Read the [Developer Guide](developer_guide.md) for implementation details
- Check the [User Guide](user_guide.md) for practical usage examples
