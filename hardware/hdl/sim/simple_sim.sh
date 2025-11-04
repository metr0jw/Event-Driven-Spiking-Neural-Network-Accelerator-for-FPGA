#!/bin/bash

##-----------------------------------------------------------------------------
## Simple Simulation Script for SNN Accelerator
##-----------------------------------------------------------------------------

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_msg() { echo -e "${GREEN}[SIM]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }

# Get script directory and project paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "DEBUG: Script directory: $SCRIPT_DIR"

# Set paths relative to script location  
TB_DIR="$SCRIPT_DIR/../tb"
RTL_DIR="$SCRIPT_DIR/../rtl"
WORK_DIR="$SCRIPT_DIR/work"

echo "DEBUG: TB_DIR = $TB_DIR"
echo "DEBUG: RTL_DIR = $RTL_DIR"

# Default testbench
TB_TOP="${1:-tb_lif_neuron}"

print_msg "Starting simple simulation"
print_msg "Testbench: $TB_TOP"

# Create work directory  
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

# Check if files exist
if [[ ! -f "$TB_DIR/${TB_TOP}.v" ]]; then
    print_error "Testbench not found: $TB_DIR/${TB_TOP}.v"
    ls -la "$TB_DIR/"
    exit 1
fi

# Simple compilation for LIF neuron
if [[ "$TB_TOP" == "tb_lif_neuron" ]]; then
    print_msg "Compiling LIF neuron testbench..."
    
    # Check required files
    if [[ -f "$RTL_DIR/neurons/lif_neuron.v" ]]; then
        echo "Found: $RTL_DIR/neurons/lif_neuron.v"
    else
        print_error "LIF neuron module not found"
        exit 1
    fi
    
    # Simple compilation
    iverilog -g2012 -o sim_exe \
        "$TB_DIR/tb_lif_neuron.v" \
        "$RTL_DIR/neurons/lif_neuron.v" \
        "$RTL_DIR/common/reset_sync.v" 2>&1
        
    if [[ $? -eq 0 ]]; then
        print_msg "Compilation successful! Running simulation..."
        ./sim_exe
        print_msg "Simulation completed!"
    else
        print_error "Compilation failed"
        exit 1
    fi
    
elif [[ "$TB_TOP" == "tb_spike_router" ]]; then
    print_msg "Compiling spike router testbench..."
    
    iverilog -g2012 -o sim_exe \
        "$TB_DIR/tb_spike_router.v" \
        "$RTL_DIR/router/spike_router.v" \
        "$RTL_DIR/common/"*.v 2>&1
        
    if [[ $? -eq 0 ]]; then
        print_msg "Compilation successful! Running simulation..."
        ./sim_exe
        print_msg "Simulation completed!"
    else
        print_error "Compilation failed"
        exit 1
    fi
    
elif [[ "$TB_TOP" == "tb_simple_lif" ]]; then
    print_msg "Compiling simple LIF neuron testbench..."
    
    iverilog -g2012 -o sim_exe \
        "$TB_DIR/tb_simple_lif.v" \
        "$RTL_DIR/neurons/lif_neuron.v" \
        "$RTL_DIR/common/reset_sync.v" 2>&1
        
    if [[ $? -eq 0 ]]; then
        print_msg "Compilation successful! Running simulation..."
        ./sim_exe
        print_msg "Simulation completed!"
        
        if [[ -f "waves.vcd" ]]; then
            print_msg "VCD waveform file generated: waves.vcd"
            if command -v gtkwave >/dev/null 2>&1; then
                print_msg "Opening waveforms in GTKWave..."
                gtkwave waves.vcd &
            fi
        fi
    else
        print_error "Compilation failed"
        exit 1
    fi
    
else
    print_error "Unsupported testbench: $TB_TOP"
    print_msg "Supported testbenches: tb_lif_neuron, tb_spike_router, tb_simple_lif"
    exit 1
fi
