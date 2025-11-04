# Developer Guide

In-depth guide for developing and extending the Event-Driven SNN FPGA Accelerator.

## Table of Contents
- [Development Environment](#development-environment)
- [Hardware Development](#hardware-development)
- [Software Development](#software-development)
- [Build System](#build-system)
- [Testing](#testing)
- [Debugging](#debugging)
- [Performance Optimization](#performance-optimization)
- [Troubleshooting](#troubleshooting)

## Development Environment

### Prerequisites

**For Hardware Development**:
- Xilinx Vivado 2025.1 or compatible
- Xilinx Vitis HLS 2025.1 or compatible
- Icarus Verilog 11.0+ (for open-source simulation)
- GTKWave (for waveform viewing)
- PYNQ-Z2 board (for hardware testing)

**For Software Development**:
- Python 3.13 or higher
- PyTorch 2.9.0 or higher
- NumPy, pytest, black, flake8, mypy
- Git for version control

### Setup Instructions

#### 1. Clone Repository
```bash
git clone https://github.com/metr0jw/Spiking-Neural-Network-on-FPGA.git
cd Spiking-Neural-Network-on-FPGA
```

#### 2. Python Environment
```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
cd software/python
pip install -e .

# Install development tools
pip install pytest pytest-cov black flake8 mypy sphinx
```

#### 3. Hardware Tools
```bash
# Source Xilinx tools (adjust paths for your installation)
source /opt/Xilinx/Vivado/2025.1/settings64.sh
source /opt/Xilinx/Vitis_HLS/2025.1/settings64.sh

# Verify installation
vivado -version
vivado_hls -version
iverilog -v

# Set environment variables
export XILINX_VIVADO=/opt/Xilinx/Vivado/2025.1
export XILINX_HLS=/opt/Xilinx/Vitis_HLS/2025.1
```

## Hardware Development

### RTL Development

#### Project Structure
```
hardware/hdl/rtl/
├── common/           # Utilities (FIFO, synchronizers, etc.)
├── neurons/          # LIF neuron implementations
├── synapses/         # Weight memory and synapse arrays
├── router/           # Spike routing logic
├── layers/           # Conv1D, pooling, layer management
├── interfaces/       # AXI wrapper and communication
└── top/              # Top-level integration modules
```

#### Coding Standards

**Module Template**:
```verilog
////////////////////////////////////////////////////////////////////////////////
// Module: module_name
// Description: Brief description of module functionality
//
// Parameters:
//   PARAM1 - Description of parameter 1
//   PARAM2 - Description of parameter 2
//
// Ports:
//   clk        - System clock
//   rst_n      - Active-low asynchronous reset
//   input_sig  - Description of input signal
//   output_sig - Description of output signal
//
// Author: Your Name
// Date: YYYY-MM-DD
////////////////////////////////////////////////////////////////////////////////

module module_name #(
    parameter PARAM1 = 8,
    parameter PARAM2 = 16
) (
    // Clock and reset
    input  wire                 clk,
    input  wire                 rst_n,
    
    // Input signals
    input  wire [PARAM1-1:0]    input_sig,
    input  wire                 input_valid,
    
    // Output signals
    output reg  [PARAM2-1:0]    output_sig,
    output reg                  output_valid
);

    // Internal signals
    reg [PARAM2-1:0] internal_reg;
    wire [PARAM1-1:0] internal_wire;
    
    // Sequential logic
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            output_sig <= {PARAM2{1'b0}};
            output_valid <= 1'b0;
        end else begin
            // Implementation
        end
    end
    
    // Combinational logic
    always @(*) begin
        // Implementation
    end

endmodule
```

**Naming Conventions**:
- **Modules**: `snake_case` (e.g., `lif_neuron`, `spike_router`)
- **Signals**: `snake_case` (e.g., `spike_valid`, `neuron_id`)
- **Parameters**: `UPPER_SNAKE_CASE` (e.g., `NUM_NEURONS`, `DATA_WIDTH`)
- **Active-low**: Suffix with `_n` (e.g., `rst_n`, `enable_n`)
- **Clocks**: `clk` or `clk_<domain>` for multiple domains

**Best Practices**:
1. Use non-blocking assignments (`<=`) in sequential blocks
2. Use blocking assignments (`=`) in combinational blocks
3. Always include reset logic for registers
4. Avoid combinational loops
5. Document complex state machines
6. Use parameters for configurability

#### Creating a New Module

**Example: Simple Buffer**:
```verilog
module spike_buffer #(
    parameter DATA_WIDTH = 32,
    parameter DEPTH = 8,
    parameter ADDR_WIDTH = $clog2(DEPTH)
) (
    input  wire                     clk,
    input  wire                     rst_n,
    
    // Write interface
    input  wire [DATA_WIDTH-1:0]    wr_data,
    input  wire                     wr_valid,
    output wire                     wr_ready,
    
    // Read interface
    output reg  [DATA_WIDTH-1:0]    rd_data,
    output reg                      rd_valid,
    input  wire                     rd_ready
);

    // Memory array
    reg [DATA_WIDTH-1:0] mem [0:DEPTH-1];
    
    // Pointers
    reg [ADDR_WIDTH-1:0] wr_ptr;
    reg [ADDR_WIDTH-1:0] rd_ptr;
    reg [ADDR_WIDTH:0] count;  // Extra bit for full detection
    
    // Full/empty flags
    wire full = (count == DEPTH);
    wire empty = (count == 0);
    
    assign wr_ready = !full;
    
    // Write logic
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wr_ptr <= {ADDR_WIDTH{1'b0}};
        end else if (wr_valid && wr_ready) begin
            mem[wr_ptr] <= wr_data;
            wr_ptr <= wr_ptr + 1'b1;
        end
    end
    
    // Read logic
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rd_ptr <= {ADDR_WIDTH{1'b0}};
            rd_valid <= 1'b0;
        end else begin
            if (rd_valid && rd_ready) begin
                rd_valid <= 1'b0;
            end
            
            if (!empty && (!rd_valid || rd_ready)) begin
                rd_data <= mem[rd_ptr];
                rd_valid <= 1'b1;
                rd_ptr <= rd_ptr + 1'b1;
            end
        end
    end
    
    // Count logic
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            count <= {(ADDR_WIDTH+1){1'b0}};
        end else begin
            case ({wr_valid && wr_ready, rd_valid && rd_ready})
                2'b10: count <= count + 1'b1;
                2'b01: count <= count - 1'b1;
                default: count <= count;
            endcase
        end
    end

endmodule
```

#### Simulation

**Using Icarus Verilog**:
```bash
cd hardware/hdl/sim

# Run simulation
./run_sim_working.sh tb_simple_lif

# View waveforms
gtkwave work/waves.vcd
```

**Creating Testbenches**:
```verilog
`timescale 1ns/1ps

module tb_spike_buffer;
    // Parameters
    parameter CLK_PERIOD = 10;  // 100 MHz
    parameter DATA_WIDTH = 32;
    parameter DEPTH = 8;
    
    // Signals
    reg clk;
    reg rst_n;
    reg [DATA_WIDTH-1:0] wr_data;
    reg wr_valid;
    wire wr_ready;
    wire [DATA_WIDTH-1:0] rd_data;
    wire rd_valid;
    reg rd_ready;
    
    // DUT instantiation
    spike_buffer #(
        .DATA_WIDTH(DATA_WIDTH),
        .DEPTH(DEPTH)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .wr_data(wr_data),
        .wr_valid(wr_valid),
        .wr_ready(wr_ready),
        .rd_data(rd_data),
        .rd_valid(rd_valid),
        .rd_ready(rd_ready)
    );
    
    // Clock generation
    initial clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;
    
    // Test sequence
    initial begin
        $dumpfile("work/tb_spike_buffer.vcd");
        $dumpvars(0, tb_spike_buffer);
        
        // Initialize
        rst_n = 0;
        wr_data = 0;
        wr_valid = 0;
        rd_ready = 0;
        
        // Reset
        #(CLK_PERIOD*5);
        rst_n = 1;
        #(CLK_PERIOD*2);
        
        // Test 1: Write data
        $display("Test 1: Writing data");
        repeat (5) begin
            @(posedge clk);
            wr_data = $random;
            wr_valid = 1;
            wait(wr_ready);
        end
        @(posedge clk);
        wr_valid = 0;
        
        // Test 2: Read data
        $display("Test 2: Reading data");
        rd_ready = 1;
        repeat (5) begin
            @(posedge clk);
            if (rd_valid) begin
                $display("  Read: 0x%08x", rd_data);
            end
        end
        
        // Test 3: Simultaneous read/write
        $display("Test 3: Simultaneous operations");
        fork
            begin
                repeat (10) begin
                    @(posedge clk);
                    wr_data = $random;
                    wr_valid = 1;
                end
            end
            begin
                repeat (10) begin
                    @(posedge clk);
                    rd_ready = 1;
                end
            end
        join
        
        #(CLK_PERIOD*10);
        $display("All tests passed!");
        $finish;
    end
    
    // Timeout
    initial begin
        #(CLK_PERIOD*1000);
        $display("ERROR: Timeout!");
        $finish;
    end

endmodule
```

### HLS Development

#### HLS Project Structure
```
hardware/hls/
├── include/          # Header files
│   ├── snn_types.h
│   ├── snn_config.h
│   └── axi_interfaces.h
├── src/              # Source files
│   ├── snn_learning_engine.cpp
│   ├── spike_encoder.cpp
│   └── weight_updater.cpp
├── test/             # Testbenches
│   ├── tb_snn_learning_engine.cpp
│   └── test_utils.h
└── scripts/          # Build scripts
    ├── create_project.tcl
    ├── run_synthesis.tcl
    └── run_cosim.tcl
```

#### HLS Coding Guidelines

**Function Template**:
```cpp
#include "snn_types.h"
#include "hls_stream.h"
#include "ap_int.h"

void hls_function(
    // AXI-Stream inputs
    hls::stream<spike_t> &input_spikes,
    
    // AXI-Stream outputs
    hls::stream<spike_t> &output_spikes,
    
    // AXI-Lite configuration
    ap_uint<16> threshold,
    ap_uint<8> leak_factor,
    
    // Memory interfaces
    weight_t *weight_memory
) {
    // Interface pragmas
    #pragma HLS INTERFACE axis port=input_spikes
    #pragma HLS INTERFACE axis port=output_spikes
    #pragma HLS INTERFACE s_axilite port=threshold
    #pragma HLS INTERFACE s_axilite port=leak_factor
    #pragma HLS INTERFACE m_axi depth=4096 port=weight_memory
    #pragma HLS INTERFACE ap_ctrl_none port=return
    
    // Implementation
    spike_t spike_in;
    if (input_spikes.read_nb(spike_in)) {
        // Process spike
        ap_int<16> membrane = 0;
        weight_t weight = weight_memory[spike_in.neuron_id];
        
        membrane += weight;
        
        if (membrane > threshold) {
            spike_t spike_out;
            spike_out.neuron_id = spike_in.neuron_id;
            spike_out.timestamp = spike_in.timestamp;
            output_spikes.write(spike_out);
        }
    }
}
```

**Performance Directives**:

```cpp
// Pipeline loops for throughput
void process_array(int data[100]) {
    #pragma HLS PIPELINE II=1
    for (int i = 0; i < 100; i++) {
        data[i] = data[i] * 2;
    }
}

// Dataflow for task-level parallelism
void dataflow_example(hls::stream<int> &in, hls::stream<int> &out) {
    #pragma HLS DATAFLOW
    
    hls::stream<int> temp;
    
    stage1(in, temp);
    stage2(temp, out);
}

// Array partitioning for bandwidth
void array_partition_example() {
    static int buffer[1024];
    #pragma HLS ARRAY_PARTITION variable=buffer cyclic factor=8
    
    // Operations on buffer
}
```

#### Building HLS Projects

```bash
cd hardware/hls

# Create project
vivado_hls -f scripts/create_project.tcl

# Run C simulation
vivado_hls -f scripts/run_csim.tcl

# Run synthesis
vivado_hls -f scripts/run_synthesis.tcl

# Run co-simulation
vivado_hls -f scripts/run_cosim.tcl

# Export IP
vivado_hls -f scripts/export_ip.tcl
```

### Vivado Project Creation

#### Using TCL Scripts

**Create HLS Project**:
```bash
vivado_hls -f create_hls_project.tcl
```

**Create Vivado Project**:
```bash
vivado -mode batch -source create_vivado_project.tcl
```

#### Manual Project Creation

```tcl
# create_project.tcl
create_project snn_accelerator ./build/vivado -part xc7z020clg400-1 -force

# Set project properties
set_property board_part tul.com.tw:pynq-z2:part0:1.0 [current_project]
set_property target_language Verilog [current_project]
set_property simulator_language Mixed [current_project]

# Add RTL sources
add_files -fileset sources_1 [glob hardware/hdl/rtl/**/*.v]
add_files -fileset constrs_1 [glob hardware/constraints/*.xdc]

# Add IP repositories
set_property ip_repo_paths {hardware/hls/*/solution1/impl/ip} [current_project]
update_ip_catalog

# Create block design
source hardware/scripts/create_block_design.tcl

# Generate wrapper
make_wrapper -files [get_files snn_bd.bd] -top
add_files -norecurse ./build/vivado/snn_accelerator.srcs/sources_1/bd/snn_bd/hdl/snn_bd_wrapper.v

# Run synthesis
launch_runs synth_1 -jobs 4
wait_on_run synth_1

# Run implementation
launch_runs impl_1 -jobs 4
wait_on_run impl_1

# Generate bitstream
launch_runs impl_1 -to_step write_bitstream
wait_on_run impl_1
```

## Software Development

### Python Package Structure

```
software/python/snn_fpga_accelerator/
├── __init__.py              # Package initialization
├── accelerator.py           # Main FPGA interface
├── fpga_controller.py       # Low-level hardware control
├── pytorch_interface.py     # PyTorch integration
├── pytorch_snn_layers.py    # Custom PyTorch layers
├── spike_encoding.py        # Encoding algorithms
├── learning.py              # STDP/R-STDP
├── utils.py                 # Utilities
└── cli.py                   # Command-line interface
```

### Coding Standards

Follow PEP 8 and use type hints throughout.

**Module Template**:
```python
"""
Module: module_name.py
Description: Brief description of module functionality.

Author: Your Name
Date: YYYY-MM-DD
"""

from typing import Optional, List, Dict, Tuple, Union
import numpy as np
import logging

logger = logging.getLogger(__name__)

class ClassName:
    """Brief class description.
    
    Detailed description of class purpose and usage.
    
    Attributes:
        attr1: Description of attribute 1
        attr2: Description of attribute 2
        
    Example:
        >>> obj = ClassName(param1='value')
        >>> result = obj.method()
    """
    
    def __init__(self, param1: str, param2: int = 0) -> None:
        """Initialize class.
        
        Args:
            param1: Description of param1
            param2: Description of param2
        """
        self.attr1 = param1
        self.attr2 = param2
        
    def method(self, arg: float) -> np.ndarray:
        """Method description.
        
        Detailed explanation of what the method does.
        
        Args:
            arg: Description of argument
            
        Returns:
            Description of return value
            
        Raises:
            ValueError: Description of when this is raised
            
        Example:
            >>> result = obj.method(1.5)
            >>> print(result.shape)
            (10,)
        """
        if arg < 0:
            raise ValueError(f"arg must be non-negative, got {arg}")
            
        result = np.zeros(10)
        # Implementation
        return result
```

### Adding New Features

#### Example: Adding a New Encoder

```python
"""New spike encoding scheme."""

from typing import Optional
import numpy as np
from numpy.random import Generator, default_rng

class CustomEncoder:
    """Custom spike encoding algorithm.
    
    Description of encoding approach and when to use it.
    
    Args:
        num_neurons: Number of output neurons
        duration: Encoding duration in seconds
        custom_param: Description of custom parameter
        seed: Random seed for reproducibility
        
    Example:
        >>> encoder = CustomEncoder(num_neurons=100, duration=0.1)
        >>> spikes = encoder.encode(data)
    """
    
    def __init__(
        self,
        num_neurons: int,
        duration: float,
        custom_param: float = 1.0,
        seed: Optional[int] = None
    ) -> None:
        self.num_neurons = num_neurons
        self.duration = duration
        self.custom_param = custom_param
        self._rng: Generator = default_rng(seed)
        
        # Precompute values
        self.dt = 0.001  # 1ms timesteps
        self.num_timesteps = int(duration / self.dt)
        
    def encode(self, data: np.ndarray) -> np.ndarray:
        """Encode data as spike trains.
        
        Args:
            data: Input data, shape (num_neurons,), normalized to [0, 1]
            
        Returns:
            Spike trains, shape (num_neurons, num_timesteps)
            
        Raises:
            ValueError: If data shape doesn't match num_neurons
        """
        if data.shape[0] != self.num_neurons:
            raise ValueError(
                f"Data shape {data.shape[0]} doesn't match "
                f"num_neurons {self.num_neurons}"
            )
            
        spikes = np.zeros((self.num_neurons, self.num_timesteps), dtype=bool)
        
        # Custom encoding logic
        for i in range(self.num_neurons):
            spike_prob = data[i] * self.custom_param
            random_values = self._rng.random(self.num_timesteps)
            spikes[i] = random_values < spike_prob
            
        return spikes
```

#### Adding Tests

```python
"""Test custom encoder."""

import pytest
import numpy as np
from snn_fpga_accelerator.spike_encoding import CustomEncoder

def test_custom_encoder_output_shape():
    """Test that encoder produces correct output shape."""
    encoder = CustomEncoder(num_neurons=10, duration=0.1)
    data = np.random.rand(10)
    spikes = encoder.encode(data)
    
    assert spikes.shape == (10, 100), f"Expected (10, 100), got {spikes.shape}"

def test_custom_encoder_reproducibility():
    """Test that encoder produces reproducible results with seed."""
    encoder1 = CustomEncoder(num_neurons=10, duration=0.1, seed=42)
    encoder2 = CustomEncoder(num_neurons=10, duration=0.1, seed=42)
    
    data = np.random.rand(10)
    spikes1 = encoder1.encode(data)
    spikes2 = encoder2.encode(data)
    
    assert np.array_equal(spikes1, spikes2), "Results not reproducible"

def test_custom_encoder_invalid_input():
    """Test that encoder raises ValueError for invalid input."""
    encoder = CustomEncoder(num_neurons=10, duration=0.1)
    data = np.random.rand(5)  # Wrong size
    
    with pytest.raises(ValueError, match="doesn't match"):
        encoder.encode(data)

@pytest.mark.parametrize("num_neurons", [1, 10, 100])
def test_custom_encoder_various_sizes(num_neurons):
    """Test encoder with various neuron counts."""
    encoder = CustomEncoder(num_neurons=num_neurons, duration=0.1)
    data = np.random.rand(num_neurons)
    spikes = encoder.encode(data)
    
    assert spikes.shape[0] == num_neurons
```

## Build System

### Makefile Targets

```bash
# Build everything
make all

# HLS targets
make hls                    # Build all HLS components
make hls-clean              # Clean HLS builds
make hls-test               # Run HLS testbenches

# Vivado targets
make vivado                 # Create Vivado project and generate bitstream
make vivado-synth           # Run synthesis only
make vivado-impl            # Run implementation only
make vivado-bitstream       # Generate bitstream only
make vivado-clean           # Clean Vivado builds

# Programming targets
make program                # Program FPGA
make program-jtag           # Program via JTAG
make program-pynq           # Program via PYNQ network

# Testing targets
make test                   # Run all tests
make test-python            # Run Python tests
make test-hardware          # Run hardware tests

# Cleanup
make clean                  # Clean build files
make clean-all              # Clean everything including dependencies
```

### Continuous Integration

The project uses GitHub Actions for CI. See `.github/workflows/ci.yml`:

```yaml
name: CI

on: [push, pull_request]

jobs:
  test-python:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.13
      - name: Install dependencies
        run: |
          pip install -e software/python
          pip install pytest pytest-cov
      - name: Run tests
        run: pytest software/python/tests --cov
        
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
      - name: Lint with flake8
        run: |
          pip install flake8
          flake8 software/python/snn_fpga_accelerator
```

## Testing

### Python Unit Tests

```bash
# Run all tests
pytest software/python/tests -v

# Run with coverage
pytest software/python/tests --cov=snn_fpga_accelerator --cov-report=html

# Run specific test file
pytest software/python/tests/test_spike_encoding.py

# Run tests matching pattern
pytest -k "encoder"
```

### Hardware Tests

```bash
# RTL simulation
cd hardware/hdl/sim
./run_sim_working.sh tb_simple_lif

# HLS tests
cd hardware/hls/test
./run_tests.sh
```

## Debugging

### Python Debugging

**Enable verbose logging**:
```python
import logging
logging.basicConfig(level=logging.DEBUG)

from snn_fpga_accelerator import SNNAccelerator
accelerator = SNNAccelerator(verbose=True)
```

**Using pdb**:
```python
import pdb

def problematic_function(data):
    pdb.set_trace()  # Debugger will stop here
    result = process(data)
    return result
```

### Hardware Debugging

**Waveform Analysis**:
```bash
# Generate VCD file
cd hardware/hdl/sim
./run_sim_working.sh tb_module

# View with GTKWave
gtkwave work/tb_module.vcd
```

**Vivado ILA (Integrated Logic Analyzer)**:
1. Add ILA IP to design
2. Connect signals to probe
3. Generate bitstream
4. Open Hardware Manager
5. Program FPGA and capture signals

**Print Debugging in Verilog**:
```verilog
always @(posedge clk) begin
    if (spike_valid) begin
        $display("Time=%0t: Spike from neuron %0d", $time, neuron_id);
    end
end
```

## Performance Optimization

### Python Optimization

**Use NumPy vectorization**:
```python
# Slow (loop)
for i in range(len(data)):
    result[i] = data[i] * weight[i]

# Fast (vectorized)
result = data * weight
```

**Profile code**:
```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Code to profile
accelerator.infer(input_spikes)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(10)
```

### Hardware Optimization

**Pipeline critical paths**:
```verilog
// Before (combinational)
assign result = (a + b) * (c + d);

// After (pipelined)
reg [15:0] stage1_sum1, stage1_sum2;
reg [31:0] stage2_product;

always @(posedge clk) begin
    stage1_sum1 <= a + b;
    stage1_sum2 <= c + d;
    stage2_product <= stage1_sum1 * stage1_sum2;
end
```

**Resource sharing**:
```cpp
// HLS resource sharing
void optimized_function(int data[100]) {
    #pragma HLS RESOURCE variable=multiplier core=DSP48
    
    for (int i = 0; i < 100; i++) {
        #pragma HLS PIPELINE
        data[i] = data[i] * coefficient;
    }
}
```

## Troubleshooting

### Common Development Issues

#### Python Import Errors

**Problem**: `ModuleNotFoundError: No module named 'snn_fpga_accelerator'`

**Solution**:
```bash
cd software/python
pip install -e .
```

#### Vivado License Issues

**Problem**: License error when running Vivado

**Solution**:
```bash
# Check license
echo $XILINXD_LICENSE_FILE

# Set license server
export XILINXD_LICENSE_FILE=port@server
```

#### Simulation Hangs

**Problem**: Testbench runs indefinitely

**Solution**:
- Add timeout in testbench:
```verilog
initial begin
    #1000000;  // 1ms timeout
    $display("ERROR: Timeout!");
    $finish;
end
```

#### Build Failures

**Problem**: Vivado synthesis fails

**Solution**:
1. Check synthesis log: `build/vivado/snn_accelerator.runs/synth_1/runme.log`
2. Fix timing violations
3. Reduce clock frequency
4. Add pipeline stages

### Getting Help

1. Check documentation and examples
2. Search existing GitHub issues
3. Enable verbose logging
4. Create minimal reproducible example
5. Open new issue with details

## Next Steps

- Read the [Architecture Documentation](architecture.md) for system design
- See the [API Reference](api_reference.md) for function details
- Check [CONTRIBUTING.md](../CONTRIBUTING.md) for contribution guidelines
