# Integrated SNN System (HLS + Verilog RTL)

## ✅ Build Status: COMPLETE

**Bitstream:** `outputs/snn_integrated.bit` (3.9 MB, ready for PYNQ-Z2)  
**Build Date:** December 10, 2025  
**Build Time:** ~20 minutes

## Overview

This integrated system combines:
- **HLS IP**: STDP/R-STDP learning engine, spike encoders, weight management
- **Verilog RTL**: AC-based LIF neurons, spike router, energy-efficient processing

## Architecture

```
External Input → Block Design (PS + HLS IP) → Verilog RTL (Neurons) → Output
                      ↑                              ↓
                      └──────── Feedback ───────────┘
                         (STDP Learning)
```

### Components

#### Block Design (design_1_wrapper)
- **Zynq PS**: ARM Cortex-A9, DDR controller, peripherals
- **HLS IP** (`snn_top_hls`):
  - STDP/R-STDP learning algorithms
  - Rate/Temporal/Phase/Delta-Sigma encoders
  - 256×256 weight memory (BRAM)
  - AXI4-Lite control interface
  - AXI4-Stream spike I/O
- **AXI Infrastructure**:
  - GP0 interconnect (control path)
  - HP0 interconnect (data path)
  - AXI DMA (spike streaming)
  - Reset management

#### Verilog RTL
- **spike_router** (`spike_router.v`):
  - AER (Address Event Representation) routing
  - Configurable connectivity matrix
  - 512-deep spike FIFO
  - Weight-based spike distribution
  
- **lif_neuron_array** (`lif_neuron_array.v`):
  - 256 LIF neurons
  - **AC-based** (Accumulate only, no DSP multipliers)
  - LUT RAM-based state storage (optimized from BRAM)
  - Parallel processing (8 units)
  - Shift-based exponential leak

#### Integration Layer (`snn_integrated_top.v`)
- Instantiates Block Design
- Instantiates Verilog RTL modules
- Clock/Reset distribution from PS
- Internal spike routing (recurrent connections)

## Build Instructions

### Prerequisites
- Vivado 2025.2
- Vitis HLS 2025.2
- PYNQ-Z2 board files

### Build Steps

1. **Build HLS IP** (if not already done):
```bash
cd hardware/hls
make clean
make all
```

2. **Build Integrated System**:
```bash
cd hardware/scripts
./build_integrated.sh
```

This will:
- Create Vivado project
- Add all Verilog RTL files
- Create Block Design with exported ports
- Synthesize and implement
- Generate bitstream

**Build time**: ~20 minutes

**Output**: `outputs/snn_integrated.bit`

## Actual Resource Utilization

**Post-Place & Route Results (Vivado 2025.2):**

| Resource | Used | Available | Utilization |
|----------|------|-----------|-------------|
| **LUTs** | 26,700 | 53,200 | **50.19%** |
| **FFs** | 24,276 | 106,400 | **22.82%** |
| **BRAM** | 16.5 | 140 | **11.79%** |
| **DSP** | 38 | 220 | **17.27%** |
| **Slice** | 8,742 | 13,300 | **65.73%** |

### Comparison with HLS-Only Build

| Resource | HLS-Only | Integrated | Delta | Notes |
|----------|----------|------------|-------|-------|
| LUT | 12,396 | 26,700 | +14,304 | RTL neurons/router added |
| FF | 10,616 | 24,276 | +13,660 | RTL state registers |
| BRAM | 54.5 | 16.5 | **-38** | State optimized to LUT RAM |
| DSP | 58 | 38 | **-20** | AC-based (no multipliers) |

### Timing Results

| Metric | Value |
|--------|-------|
| Clock | 100 MHz (10 ns period) |
| WNS | **+0.845 ns** ✅ |
| TNS | 0.000 ns |
| Failing Endpoints | 0 |

### Power Analysis

| Component | Power (W) |
|-----------|-----------|
| PS7 (ARM) | 1.532 |
| HLS IP | 0.564 |
| AXI/DMA | 0.009 |
| Static | 0.156 |
| **Total** | **2.261** |

## Signal Flow

### Input Path (PS → PL)
1. **PS** sends raw spike data via AXI DMA
2. **HLS encoder** converts to temporal/rate/phase encoding
3. **HLS → RTL**: Encoded spikes forwarded to router
4. **Router** distributes spikes to target neurons based on connectivity
5. **Neurons** integrate spikes, generate output spikes

### Output Path (PL → PS)
1. **Neurons** generate output spikes
2. **RTL → HLS**: Spikes sent to learning engine
3. **HLS learning** applies STDP/R-STDP rules
4. **Weight updates** sent back to router configuration
5. **HLS** streams results to PS via AXI DMA

### Feedback Loop (Learning)
- Neuron spike timing captured by timestamp counter
- Pre/post-synaptic spike pairs processed by STDP
- Weight updates applied to router connectivity matrix
- Closed-loop learning without PS intervention

## Key Interfaces

### External (Top-Level)
```verilog
input  wire         sys_clk         // System clock (from PS)
input  wire         sys_rst_n       // System reset
input  wire         ext_spike_valid // External spike input
input  wire [31:0]  ext_spike_data  // Spike data
output wire         ext_spike_ready // Ready signal
output wire         ext_result_valid // Result output
output wire [31:0]  ext_result_data  // Result data
input  wire         ext_result_ready // Result ready
output wire [31:0]  debug_spike_count
output wire [31:0]  debug_neuron_active
output wire         debug_learning_active
```

### HLS → RTL (Internal)
```verilog
wire                         hls_encoded_spike_valid
wire [NEURON_ID_WIDTH-1:0]   hls_encoded_spike_id
wire [DATA_WIDTH-1:0]        hls_encoded_spike_data
wire                         hls_encoded_spike_ready
```

### RTL → HLS (Internal)
```verilog
wire                         rtl_spike_feedback_valid
wire [NEURON_ID_WIDTH-1:0]   rtl_spike_feedback_id
wire [DATA_WIDTH-1:0]        rtl_spike_feedback_timestamp
wire                         rtl_spike_feedback_ready
```

## Energy Efficiency

### AC-Based Architecture Benefits
- **No DSP multipliers**: Accumulate-only operations
- **Energy savings**: ~5× lower than MAC-based
  - MAC operation: ~4.6 pJ
  - AC operation: ~0.9 pJ
- **Shift-based leak**: Uses barrel shifters instead of multipliers

### Power Breakdown (Estimated)
| Component | Power (mW) | Percentage |
|-----------|------------|------------|
| HLS IP    | ~108       | 65%        |
| Verilog RTL | ~35      | 21%        |
| PS Interface | ~15     | 9%         |
| Clocking  | ~8         | 5%         |
| **Total** | **~166**   | **100%**   |

**Note**: Verilog RTL power is event-driven - actual power depends on spike activity.

## Testing

### Simulation
```bash
cd hardware/hdl/sim
./run_sim.sh
```

### Hardware Test (PYNQ-Z2)
1. **Copy bitstream to board**:
```bash
scp outputs/snn_integrated.bit xilinx@192.168.2.99:~/
scp outputs/design_1.hwh xilinx@192.168.2.99:~/snn_integrated.hwh
```

2. **Load on PYNQ**:
```python
from pynq import Overlay
import numpy as np

# Load bitstream
ol = Overlay('/home/xilinx/snn_integrated.bit')

# Access components
hls_ip = ol.snn_top_hls_0  # HLS learning engine
# Verilog RTL is integrated internally

# Send spike pattern
spike_data = np.array([1, 2, 3, 4], dtype=np.uint32)
# ... (see examples/ for complete usage)
```

## Debugging

### Vivado ILA (Integrated Logic Analyzer)
Add ILA cores to monitor internal signals:
- HLS → RTL data path
- RTL neuron states
- Spike router activity
- Weight update events

### Debug Ports
The integrated top exposes debug signals:
- `debug_spike_count`: Total routed spikes
- `debug_neuron_active`: Number of active neurons
- `debug_learning_active`: HLS learning engine status

## Known Limitations

1. **Block Design Ports**: Current implementation requires manual port connections in Block Design TCL. Some HLS IP ports may need adjustment based on actual HLS synthesis.

2. **Address Conflicts**: Ensure AXI address ranges don't overlap:
   - HLS IP: 0x43C00000 (128 bytes)
   - AXI DMA: Auto-assigned by Vivado

3. **Timing**: Combined design may require timing constraints adjustment. Target 100 MHz achievable but verify with actual implementation.

## Future Enhancements

1. **Multi-Layer Support**: Add layer manager for hierarchical SNNs
2. **Conv/Pool Layers**: Integrate CNN-style SNN layers
3. **Dynamic Routing**: Runtime-configurable connectivity
4. **Power Gating**: Clock/power gating for inactive neurons
5. **Compression**: Spike compression for AXI bandwidth reduction

## References

- `docs/BUILD_STATUS.md` - Build status and detailed reports
- `docs/VERILOG_RTL_SPECIFICATIONS.md` - RTL module specifications
- `docs/HARDWARE_SPECIFICATIONS.md` - Hardware specs and resource analysis
- `examples/` - Usage examples and test scripts

## Support

**Author**: Jiwoon Lee (@metr0jw)  
**Organization**: Kwangwoon University, Seoul, South Korea  
**Contact**: jwlee@linux.com  
**Repository**: [GitHub](https://github.com/metr0jw/Event-Driven-Spiking-Neural-Network-Accelerator-for-FPGA)
