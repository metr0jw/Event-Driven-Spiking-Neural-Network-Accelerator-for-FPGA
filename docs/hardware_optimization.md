# Hardware Optimization Guide

## Overview

This document describes the hardware optimization strategy for the SNN accelerator targeting the PYNQ-Z2 board (Zynq XC7Z020).

## Baseline Resource Analysis

From the original synthesis:

| Resource   | Used   | Available | Utilization |
|------------|--------|-----------|-------------|
| LUT        | 4,652  | 53,200    | 8.81%       |
| Register   | 3,195  | 106,400   | 3.02%       |
| BRAM       | 2.0    | 140       | 1.43%       |
| DSP        | 0      | 220       | 0.00%       |

**Timing (100 MHz):**
- WNS: +0.159ns (passes, but tight margin)

## Optimization Strategy

### 1. Scale Up Network Size (4x)

**Original:** 64 neurons × 64 axons
**Optimized:** 256 neurons × 256 axons

This increases the network capacity significantly while utilizing more FPGA resources efficiently.

### 2. Increase Parallelism (2x)

**Original:** 4 parallel processing units (TIME_MULTIPLEX_FACTOR)
**Optimized:** 8 parallel processing units

Benefits:
- 2x higher throughput
- Better resource utilization
- Improved spike processing latency

### 3. BRAM-based State Storage

**Original:** Distributed RAM for membrane potentials
**Optimized:** Block RAM with registered outputs

```verilog
// BRAM inference attributes
(* ram_style = "block" *)
reg [DATA_WIDTH-1:0] membrane_bram [0:NUM_NEURONS-1];
```

Benefits:
- Reduced LUT usage for storage
- Better timing due to registered BRAM outputs
- Efficient use of dedicated BRAM resources

### 4. Pipelined Memory Access

3-stage pipeline for BRAM access:
1. **Stage 1:** Address generation
2. **Stage 2:** BRAM read (1-cycle latency)
3. **Stage 3:** Compute and write-back

This improves timing closure by breaking long combinatorial paths.

### 5. DSP Usage (Optional)

For applications requiring higher precision leak computation:

```verilog
// DSP-based multiplication for leak
(* use_dsp = "yes" *)
wire [DATA_WIDTH+LEAK_WIDTH-1:0] leak_product;
```

Note: The default shift-based leak is more power-efficient and sufficient for most applications.

## Optimized Modules

### lif_neuron_array_optimized.v

Key features:
- 256 neurons (configurable)
- 8 parallel processing units
- BRAM-based membrane and refractory storage
- 3-stage pipeline
- Configurable DSP usage

Parameters:
```verilog
parameter NUM_NEURONS           = 256,
parameter NUM_PARALLEL_UNITS    = 8,
parameter USE_BRAM              = 1,
parameter USE_DSP               = 1
```

### synapse_array_optimized.v

Key features:
- 256×256 synapse matrix (65,536 weights)
- 8 parallel read ports via BRAM banking
- Batch weight initialization support
- Pipelined spike distribution

### snn_accelerator_top_optimized.v

Top-level integration with:
- Scaled interfaces
- Enhanced status monitoring
- Performance counters
- Configurable interrupt thresholds

## Expected Resource Utilization

| Resource   | Original | Optimized | Target  |
|------------|----------|-----------|---------|
| LUT        | 8.81%    | ~35%      | <50%    |
| Register   | 3.02%    | ~15%      | <25%    |
| BRAM       | 1.43%    | ~20%      | <30%    |
| DSP        | 0.00%    | ~10%      | <15%    |
| WNS        | +0.159ns | >+1.0ns   | >+0.5ns |

## Build Instructions

### Synthesis Only (Resource Estimation)

```bash
cd hardware/scripts
vivado -mode batch -source synth_optimized.tcl
```

### Full Implementation

```bash
cd hardware/scripts
vivado -mode batch -source run_all.sh
```

## Simulation

### Run Testbench

```bash
cd hardware/hdl
iverilog -g2012 -o sim/tb_lif_neuron_array_optimized.vvp \
    -I rtl/common -I rtl/neurons \
    tb/tb_lif_neuron_array_optimized.v \
    rtl/neurons/lif_neuron_array_optimized.v
    
cd sim
vvp tb_lif_neuron_array_optimized.vvp
```

### View Waveforms

```bash
gtkwave tb_lif_neuron_array_optimized.vcd
```

## Configuration Options

### Reduce Resource Usage

If resource utilization is too high:

```verilog
parameter NUM_NEURONS           = 128,  // Reduce neuron count
parameter NUM_PARALLEL_UNITS    = 4,    // Reduce parallelism
parameter USE_DSP               = 0     // Disable DSP usage
```

### Increase Performance

For maximum throughput:

```verilog
parameter NUM_NEURONS           = 256,
parameter NUM_PARALLEL_UNITS    = 16,   // Max parallelism
parameter SPIKE_BUFFER_DEPTH    = 128   // Larger buffers
```

## Timing Optimization Tips

1. **Use BRAM outputs registered:** Already implemented
2. **Pipeline long paths:** 3-stage pipeline in place
3. **Reduce fan-out:** Use replication for high-fanout signals
4. **Placement constraints:** Add Pblock constraints for critical modules

## Power Optimization

The shift-based leak mechanism is already power-optimized:
- No multipliers required
- Simple shift operations
- DSP blocks remain available for other uses

For ultra-low power:
```verilog
parameter USE_DSP = 0,          // Disable DSPs
parameter NUM_PARALLEL_UNITS = 2 // Minimum parallelism
```

## Verification Checklist

- [x] lif_neuron_array_optimized.v simulation passes
- [x] Shift-based leak matches Python simulator
- [x] Parallel processing verified
- [x] Spike generation correct
- [ ] Full top-level integration test
- [ ] FPGA implementation and board test

## Files

| File | Description |
|------|-------------|
| `rtl/neurons/lif_neuron_array_optimized.v` | Optimized neuron array |
| `rtl/synapses/synapse_array_optimized.v` | Optimized synapse array |
| `rtl/top/snn_accelerator_top_optimized.v` | Optimized top module |
| `tb/tb_lif_neuron_array_optimized.v` | Neuron array testbench |
| `scripts/synth_optimized.tcl` | Synthesis script |
