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
- `membrane_potential`: 16-bit unsigned fixed-point
- `refractory_counter`: 8-bit unsigned counter
- `spike_out`: 1-bit output spike

**Operations (Event-Driven)**:
```
// When synaptic input arrives:
V[t+1] = saturate(V[t] + weight)  // Excitatory: +weight, Inhibitory: -weight

// When no input (leak cycle):
V[t+1] = V[t] - (V[t] >> shift1) - (V[t] >> shift2)  // Shift-based exponential decay

if V[t+1] >= threshold:
    spike_out = 1
    V[t+1] = reset_potential
    refractory_counter = refractory_period
```

**Shift-Based Leak (Power-Optimized)**:

The leak operation uses shift operations instead of multiplication for power efficiency.
The effective tau (decay factor) is: `tau = 1 - 2^(-shift1) - 2^(-shift2)`

Common tau approximations:
| Target tau | shift1 | shift2 | Actual tau | Error |
|------------|--------|--------|------------|-------|
| 0.500      | 1      | 0      | 0.5000     | 0.000 |
| 0.750      | 2      | 0      | 0.7500     | 0.000 |
| 0.875      | 3      | 0      | 0.8750     | 0.000 |
| 0.900      | 4      | 5      | 0.9062     | 0.006 |
| 0.9375     | 4      | 0      | 0.9375     | 0.000 |
| 0.950      | 5      | 6      | 0.9531     | 0.003 |

**leak_rate Encoding**:
- Bits [3:0]: shift1 (primary leak shift)
- Bits [7:4]: shift2 (secondary leak shift, 0 = disabled)

**Parameters**:
- Threshold: 16-bit unsigned, typical range [100, 2000]
- Leak rate: 8-bit encoded shift values (see above)
- Refractory period: 8-bit, 0-255 timesteps
- Reset potential: 16-bit, typically 0

**Data Widths**:
- Membrane potential: 16-bit unsigned
- Synaptic weights: 8-bit signed (-128 to +127)
- Threshold: 16-bit unsigned

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

### Per-Neuron Trace Architecture (v2.0)

The learning engine uses a **Per-Neuron Trace** architecture, optimized for FPGA resource efficiency. This approach reduces memory complexity from O(N×M) to O(N+M), making it feasible for resource-constrained devices like PYNQ-Z2.

**Memory Complexity Comparison**:

| Architecture | Trace Memory | Example (256 pre × 64 post) |
|--------------|--------------|----------------------------|
| Per-Synapse  | O(N×M)      | 16,384 traces (512 KB)     |
| Per-Neuron   | O(N+M)      | 320 traces (1.25 KB)       |
| **Savings**  | **~400x**   | **~400x less memory**      |

This architecture is based on academic best practices from Intel Loihi and SpiNNaker.

### STDP with Per-Neuron Traces

**Lazy Trace Update**:
```c
// Decay LUT for efficient computation (4-bit index → decay factor)
static const trace_t EXP_DECAY_LUT[16] = {
    256, 240, 225, 211, 198, 186, 174, 163,  // 0-7 timesteps
    153, 143, 134, 126, 118, 111, 104, 97    // 8-15 timesteps
};

// Lazy update: compute trace value at current time
trace_value = trace.value * EXP_DECAY_LUT[time_diff] >> 8;

// On spike: reset to maximum
if (spike) {
    trace.value = MAX_TRACE_VALUE;
    trace.last_spike_time = current_time;
}
```

**Weight Update Rule**:
```c
// Pre-synaptic spike arrives: check post-trace (LTD)
if (pre_spike) {
    post_trace = get_trace_lazy(post_traces[post_id], current_time);
    delta_w = -(A_MINUS * post_trace) >> 8;  // LTD
}

// Post-synaptic spike occurs: check pre-trace (LTP)
if (post_spike) {
    pre_trace = get_trace_lazy(pre_traces[pre_id], current_time);
    delta_w = (A_PLUS * pre_trace) >> 8;     // LTP
}
```

### R-STDP Extension (Reward-Modulated STDP)

R-STDP extends STDP with eligibility traces for reinforcement learning.

**Per-Neuron Eligibility Traces**:
```c
// Eligibility trace structure (per neuron, not per synapse)
typedef struct {
    int32_t accumulated_stdp;  // Accumulated STDP delta
    uint16_t last_update_time; // For lazy decay
} eligibility_trace_t;

// Accumulate STDP changes
eligibility[neuron_id].accumulated_stdp += delta_w_stdp;

// On reward signal: apply modulated weight update
delta_w_final = (reward * eligibility[neuron_id].accumulated_stdp) >> 8;
```

**Reward Modulation**:
- Eligibility traces decay over time (τ_eligibility ~ 100-1000ms)
- Reward signal modulates all accumulated eligibility traces
- Weight update: `Δw = learning_rate × reward × eligibility`

### Hardware Learning Engine (HLS)

**File**: `hardware/hls/src/snn_top_hls.cpp`

HLS-based learning engine with Per-Neuron Trace optimization.

**Key Data Structures**:
```cpp
// Per-neuron trace (not per-synapse)
typedef struct {
    trace_t value;           // 8-bit trace value
    timestamp_t last_spike;  // Last spike timestamp
} neuron_trace_t;

// Separate arrays for pre and post neurons
neuron_trace_t pre_traces[MAX_NEURONS];   // Pre-synaptic traces
neuron_trace_t post_traces[MAX_NEURONS];  // Post-synaptic traces
```

**Top-Level Interface**:
```cpp
void snn_top_hls(
    // DMA Streams
    hls::stream<input_packet_t> &input_stream,
    hls::stream<output_packet_t> &output_stream,
    // Configuration
    config_reg_t config,
    // Memory Interfaces
    weight_t *weight_mem,
    potential_t *membrane_mem
);
```

**Performance (Vitis HLS 2025.2)**:
- **Target Frequency**: 100 MHz
- **Achieved Fmax**: 138.10 MHz (+38% margin)
- **Initiation Interval**: II=1 for inner loops
- **Resource Utilization**:
  - BRAM: 56 (20%)
  - LUT: 11,503 (21%)
  - FF: 2,850 (2%)
  - DSP: 0 (0%)

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
- Idle: TBD
- Inference (low activity): TBD
- Inference (high activity): TBD
- With learning: TBD

Power breakdown:
- PS (ARM): TBD
- PL (FPGA fabric): TBD
- Memory I/O: TBD

### Scalability

| Configuration      | Neurons | Synapses | BRAM | Throughput    |
|--------------------|---------|----------|------|---------------|
| Small              | 64      | 4K       | 18   | 10M spikes/s  |
| Medium             | 256     | 64K      | 48   | 30M spikes/s  |
| Large              | 1024    | 1M       | 120  | 80M spikes/s  |

## Hardware Optimization (v2.0)

### Optimized Architecture Overview

The latest hardware revision focuses on maximizing FPGA resource utilization while respecting timing constraints. Based on synthesis analysis showing underutilized resources (LUTs 8.81%, BRAM 1.4%, DSP 0%), the architecture was scaled up significantly.

### Key Optimizations

#### 1. LIF Neuron Array Optimization

**File**: `hardware/hdl/rtl/neurons/lif_neuron_array.v`

**BRAM-Based State Storage**:
```verilog
(* ram_style = "block" *)
reg [DATA_WIDTH-1:0] membrane_bram [0:NUM_NEURONS-1];

(* ram_style = "block" *)
reg [REFRAC_WIDTH-1:0] refrac_bram [0:NUM_NEURONS-1];
```

**DSP-Assisted Synaptic Accumulation**:
```verilog
(* use_dsp = "yes" *)
reg signed [DATA_WIDTH+WEIGHT_WIDTH-1:0] synaptic_sum;
```

**3-Stage Pipeline Architecture**:
```
Stage 1 (read_addr_s1): Address calculation and BRAM read initiation
Stage 2 (membrane_s2):  BRAM data available, perform membrane calculation
Stage 3 (new_membrane_s3): Write back results, threshold comparison
```

**Parallel Processing Units**:
- 8 parallel units process neurons simultaneously
- Each unit has independent pipeline stages
- Reduces full array update time from 64 cycles to 8 cycles

#### 2. Synapse Array Optimization

**File**: `hardware/hdl/rtl/synapses/synapse_array.v`

**Parallel BRAM Banking**:
```verilog
// 8 independent BRAM banks for parallel weight access
(* ram_style = "block" *)
reg [WEIGHT_WIDTH:0] weight_mem_0 [0:BANK_DEPTH-1];
// ... (weight_mem_1 through weight_mem_7)
```

**Bank Address Organization**:
- Bank b handles neurons: b, b+8, b+16, b+24, ...
- Allows 8 weights to be read in parallel per spike
- Reduces spike propagation time by 8×

**Batch Write Interface**:
```verilog
input wire [NUM_READ_PORTS*(WEIGHT_WIDTH+1)-1:0] batch_weights;
input wire [NEURON_ID_WIDTH-1:0] batch_start_neuron;
```
- Writes 8 weights per clock cycle during initialization
- Reduces weight loading time by 8×

#### 33. Top Module Integration

**File**: `hardware/hdl/rtl/top/snn_accelerator_top.v`

**New Parameters**:
```verilog
parameter NUM_NEURONS = 256;
parameter NUM_AXONS = 256;
parameter NUM_PARALLEL_UNITS = 8;
parameter ROUTER_BUFFER_DEPTH = 512;
parameter USE_BRAM = 1;
parameter USE_DSP = 1;
```

**Enhanced Status Monitoring**:
```verilog
wire [31:0] throughput_counter;  // Processing cycles per spike
wire [7:0]  active_neurons;      // Currently processing neurons
wire        synapse_busy;        // Synapse array busy signal
```

### Resource Utilization Targets

| Resource | PYNQ-Z2 Total | Original Usage | Optimized Target |
|----------|---------------|----------------|------------------|
| LUT      | 53,200        | 4,688 (8.81%)  | ~18,620 (35%)    |
| BRAM     | 140 (36Kb)    | 2 (1.4%)       | ~28 (20%)        |
| DSP      | 220           | 0 (0%)         | ~22 (10%)        |
| FF       | 106,400       | ~2,000         | ~15,000          |

### Performance Improvements

| Metric                    | Original    | Optimized     | Improvement |
|---------------------------|-------------|---------------|-------------|
| Network Size              | 64×64       | 256×256       | 16×         |
| Parallel Processing       | 4 units     | 8 units       | 2×          |
| Spike Propagation         | 64 cycles   | 8 cycles      | 8×          |
| Weight Lookup             | Sequential  | 8-way BRAM    | 8×          |
| Memory Efficiency         | Distributed | Block RAM     | Better PPA  |
| DSP Utilization           | 0%          | ~10%          | -           |

### Shift-Based Leak Implementation

The optimized leak mechanism uses configurable shift operations:

```verilog
// leak_rate[7:4] = shift2, leak_rate[3:0] = shift1
// tau = 1 - 2^(-shift1) - 2^(-shift2)

wire [DATA_WIDTH-1:0] leak_amount_s1 = (shift1_val > 0) ? 
                                        (membrane_s2 >> shift1_val) : 0;
wire [DATA_WIDTH-1:0] leak_amount_s2 = (shift2_val > 0) ? 
                                        (membrane_s2 >> shift2_val) : 0;
wire [DATA_WIDTH-1:0] leaked_membrane = membrane_s2 - leak_amount_s1 - leak_amount_s2;
```

**Benefits**:
- No multiplier required (saves DSP resources)
- Single-cycle leak computation
- Configurable decay rates via software
- Hardware-Python identity verified through extensive testing

## Next Steps

- Explore the [API Reference](api_reference.md) for detailed function documentation
- Read the [Developer Guide](developer_guide.md) for implementation details
- Check the [User Guide](user_guide.md) for practical usage examples
