# Project Status - SNN FPGA Accelerator

**Last Updated**: December 8, 2025

## Overview

Event-Driven Spiking Neural Network (SNN) Accelerator for PYNQ-Z2 FPGA with on-chip learning capabilities.

---

## 🎉 HLS Synthesis Complete!

### Resource Utilization (xc7z020-clg400-1)
| Resource | Used | Available | Utilization |
|----------|------|-----------|-------------|
| **BRAM** | 32 | 280 | 11% |
| **DSP** | 39 | 220 | 17% |
| **FF** | 58,184 | 106,400 | 54% |
| **LUT** | 56,644 | 53,200 | 106% ⚠️ |

- **Estimated Fmax**: 126.49 MHz (target: 100 MHz)
- **IP Location**: `hardware/ip_repo/snn_top_hls/`

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
| `snn_top_hls.cpp` | ✅ Complete | HLS synthesized with v++ (LUT optimized) |
| `learning_engine.cpp` | ✅ Complete | STDP/R-STDP with eligibility traces |
| `spike_encoder.cpp` | ✅ Complete | Rate/Temporal encoding |
| `weight_update.cpp` | ✅ Complete | Batch weight updates |
| `pc_interface.cpp` | ✅ Complete | Spike/Weight/Membrane potential I/O |

> Note: `spike_decoder.cpp` removed - decoding handled in software on PC side

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

### 1. Vitis HLS Synthesis
- **Status**: Blocked by Tcl environment issue
- **Issue**: Tcl 8.6.13 vs 8.6.11 version conflict in Vitis HLS 2025.2
- **Workaround**: Use Vitis HLS GUI mode for synthesis
- **Required**: Generate `snn_top_hls` IP for Vivado integration

### 2. AXI Wrapper
- **Status**: Pending HLS synthesis
- **Note**: Original RTL `axi_wrapper.v` was removed, replaced by HLS implementation
- **Required**: HLS IP generation for `axi_wrapper`

### 3. Vivado Integration
- **Status**: Project created, synthesis blocked
- **Blocker**: Missing `axi_wrapper` module
- **Project Location**: `build/vivado/snn_accelerator_pynq.xpr`

### 4. Bitstream Generation
- **Status**: Pending successful synthesis
- **Target**: PYNQ-Z2 (xc7z020clg400-1)

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
│   │   └── scripts/       # TCL synthesis scripts
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

1. **Run Vitis HLS in GUI mode** to synthesize:
   - `snn_top_hls`
   - `axi_wrapper` (if separate)

2. **Export HLS IP** to `hardware/ip_repo/`

3. **Run Vivado synthesis**:
   ```bash
   vivado -mode batch -source hardware/scripts/run_synthesis.tcl
   ```

4. **Generate bitstream** for PYNQ-Z2

5. **Test on hardware** with Python driver

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

### HLS C-Simulation (December 5, 2025)
```
snn_top_hls: All tests passed
- Forward propagation: PASS
- STDP learning: PASS
- R-STDP with eligibility traces: PASS
- Spike encoding/decoding: PASS
```

---

## Known Issues

1. **Vitis HLS Tcl Environment**: Tcl version conflict prevents batch mode execution
   - System has Tcl 8.6.13, Vitis expects 8.6.11
   - Solution: Use GUI mode or resolve system Tcl version

2. **PYNQ-Z2 Board Part**: Not installed in Vivado board store
   - Using direct part specification (xc7z020clg400-1) instead

---

## Contact

- **Author**: Jiwoon Lee (@metr0jw)
- **Organization**: Kwangwoon University, Seoul, South Korea
- **Email**: jwlee@linux.com
