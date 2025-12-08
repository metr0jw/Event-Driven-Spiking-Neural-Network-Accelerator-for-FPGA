# Project Status - SNN FPGA Accelerator

**Last Updated**: January 2025

## Overview

Event-Driven Spiking Neural Network (SNN) Accelerator for PYNQ-Z2 FPGA with on-chip learning capabilities.

---

## 🎉 HLS Synthesis Complete!

### Latest Results (Per-Neuron Trace Architecture v2.0)

| Metric | Value | Notes |
|--------|-------|-------|
| **Target Frequency** | 100 MHz | 10ns clock period |
| **Achieved Fmax** | **138.10 MHz** | +38% timing margin |
| **Timing Slack** | +0.06 ns | Meets timing with margin |

### Resource Utilization (xc7z020-clg400-1)
| Resource | Used | Available | Utilization |
|----------|------|-----------|-------------|
| **BRAM** | 56 | 280 | 20% |
| **DSP** | 0 | 220 | 0% |
| **FF** | 2,850 | 106,400 | 2% |
| **LUT** | 11,503 | 53,200 | 21% |

### Architecture Highlights
- **Per-Neuron Trace**: O(N+M) memory instead of O(N×M) per-synapse
- **Lazy Update**: Compute-on-demand trace decay
- **LUT-based Decay**: 16-entry exponential decay lookup table
- **Build Tool**: Vitis HLS 2025.2 with **v++ CLI** (TCL deprecated)

---

## ✅ Completed Tasks

### 1. Verilog RTL Design
| Module | Status | Tests |
|--------|--------|-------|
| LIF Neuron (`lif_neuron.v`) | ✅ Complete | 12/12 passed |
| LIF Neuron Array (`lif_neuron_array.v`) | ✅ Complete | All passed |
| Synapse Array (`synapse_array.v`) | ✅ Complete | All passed |
| Spike Router (`spike_router.v`) | ✅ Complete | All passed |
| SNN Core (`snn_core.v`) | ✅ Complete | All passed |
| SNN Accelerator Top (`snn_accelerator_top.v`) | ✅ Complete | All passed |
| Conv1D Layer (`snn_conv1d_layer.v`) | ✅ Complete | All passed |
| Pooling Layer (`snn_pooling_layer.v`) | ✅ Complete | All passed |
| Temporal Module (`temporal_coding.v`) | ✅ Complete | All passed |
| Reset Sync (`reset_sync.v`) | ✅ Complete | All passed |
| Dual-Port BRAM (`dp_bram.v`) | ✅ Complete | All passed |

### 2. HLS Implementation
| Module | Status | Notes |
|--------|--------|-------|
| `snn_top_hls.cpp` | ✅ Complete | **Per-Neuron Trace architecture**, 138.10 MHz |
| `learning_engine.cpp` | ✅ Complete | STDP/R-STDP with eligibility traces |
| `spike_encoder.cpp` | ✅ Complete | Rate/Temporal encoding |
| `weight_update.cpp` | ✅ Complete | Batch weight updates |
| `pc_interface.cpp` | ✅ Complete | Spike/Weight/Membrane potential I/O |

> **Build System**: Uses `v++` CLI (Vitis 2025.2). Legacy TCL workflow deprecated.

### 3. Python Software
| Component | Status | Tests |
|-----------|--------|-------|
| SNN FPGA Accelerator Package | ✅ Complete | 87/90 passed, 3 skipped |
| MNIST Training Example | ✅ Complete | Working |
| R-STDP Learning Example | ✅ Complete | Working |
| Network Configuration | ✅ Complete | YAML-based |

### 4. Test Infrastructure
- **Verilog RTL Tests**: 12/12 passed (iverilog/vvp)
- **Python Tests**: 87/90 passed, 3 skipped (pytest)
- **HLS C-Simulation**: All tests passed (GCC with Vitis includes)

---

## 🔄 In Progress / Pending

### 1. Vivado Integration
- **Status**: Ready for integration
- **HLS IP**: Successfully synthesized with v++
- **Project Location**: `hardware/vivado/`

### 2. Bitstream Generation
- **Status**: Pending Vivado implementation
- **Target**: PYNQ-Z2 (xc7z020clg400-1)

### 3. Hardware Testing
- **Status**: Pending bitstream
- **Goal**: Validate on PYNQ-Z2 board

---

## Project Structure

```
Event-Driven-Spiking-Neural-Network-Accelerator-for-FPGA/
├── hardware/
│   ├── hdl/
│   │   ├── rtl/           # Verilog RTL modules
│   │   │   ├── top/       # Top-level modules
│   │   │   ├── neurons/   # LIF neuron implementations
│   │   │   ├── synapses/  # Synapse array with BRAM
│   │   │   ├── layers/    # Conv1D, pooling, temporal
│   │   │   ├── router/    # Spike routing
│   │   │   └── common/    # Utilities (BRAM, reset sync)
│   │   └── tb/            # Testbenches
│   ├── hls/
│   │   ├── src/           # HLS C++ source
│   │   ├── include/       # HLS headers
│   │   ├── test/          # HLS testbenches
│   │   └── scripts/       # v++ build scripts (build_hls.sh)
│   ├── constraints/       # XDC constraint files
│   ├── scripts/           # Vivado TCL scripts
│   └── ip_repo/           # Generated IP repository
├── software/
│   └── python/            # Python driver package
├── docs/                  # Documentation
├── examples/              # Example applications
├── build/                 # Build outputs (generated)
└── outputs/               # Final bitstreams (generated)
```

---

## Next Steps

1. **Run Vivado Implementation**:
   ```bash
   cd hardware/vivado
   vivado -mode batch -source ../scripts/create_project.tcl
   ```

2. **Generate Bitstream** for PYNQ-Z2

3. **Test on Hardware** with Python driver

---

## Build Commands

### HLS Synthesis (v++ CLI)
```bash
cd hardware/hls/scripts
./build_hls.sh           # Full build
./build_hls.sh csim      # C-simulation only
./build_hls.sh syn       # Synthesis only
```

### Vivado Build
```bash
cd hardware/vivado
vivado -mode batch -source ../scripts/build_bitstream.tcl
```

---

## Test Results Summary

### RTL Tests (December 5, 2025)
```
============================================================
 SNN RTL Test Suite
============================================================

TEST SUMMARY
Total Tests:    12
Passed:         12
Failed:         0
High-Z Warnings: 0

ALL TESTS PASSED - NO HIGH-Z ISSUES
```

### Python Tests (December 5, 2025)
```
87 passed, 3 skipped
```

### HLS Synthesis Results (January 2025)
```
============================================================
 Vitis HLS 2025.2 Synthesis Results (v++ CLI)
============================================================

Target: xc7z020clg400-1 (PYNQ-Z2)
Clock: 10ns (100 MHz target)

TIMING:
- Achieved Fmax: 138.10 MHz
- Timing Slack: +0.06 ns
- Status: PASS ✅

RESOURCES:
- BRAM: 56 / 280 (20%)
- LUT:  11,503 / 53,200 (21%)
- FF:   2,850 / 106,400 (2%)
- DSP:  0 / 220 (0%)

ARCHITECTURE: Per-Neuron Trace (O(N+M) memory)
```

---

## Known Issues

1. **PYNQ-Z2 Board Part**: Not installed in Vivado board store
   - Using direct part specification (xc7z020clg400-1) instead

---

## Contact

- **Author**: Jiwoon Lee (@metr0jw)
- **Organization**: Kwangwoon University, Seoul, South Korea
- **Email**: jwlee@linux.com
