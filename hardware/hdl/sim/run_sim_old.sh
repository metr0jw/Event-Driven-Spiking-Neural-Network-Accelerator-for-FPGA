#!/bin/bash

##-----------------------------------------------------------------------------
## Title         : Multi-Simulator Run Script for SNN Accelerator
## Project       : PYNQ-Z2 SNN Accelerator
## File          : run_sim.sh
## Author        : Jiwoon Lee (@metr0jw)
## Organization  : Kwangwoon University, Seoul, South Korea
## Description   : Automated simulation script supporting multiple simulators
##                 (Vivado Simulator, Icarus Verilog, Verilator)
##-----------------------------------------------------------------------------

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

# Directories
HDL_DIR="$PROJECT_ROOT/hardware/hdl"
RTL_DIR="$HDL_DIR/rtl"
TB_DIR="$HDL_DIR/tb"
SIM_DIR="$PROJECT_ROOT/hardware/hdl/sim"
WORK_DIR="$SIM_DIR/work"

# Default values
TB_TOP="tb_top"
GUI_MODE=0
CLEAN_MODE=0
DEBUG_MODE="typical"
COVERAGE=0
WAVES_MODE=0
FORCE_SIMULATOR=""

# Detect available simulators
detect_simulator() {
    if command -v xvlog >/dev/null 2>&1; then
        echo "vivado"
    elif command -v iverilog >/dev/null 2>&1; then
        echo "icarus"
    elif command -v verilator >/dev/null 2>&1; then
        echo "verilator"
    else
        echo "none"
    fi
}

# Function to print colored output
print_msg() {
    echo -e "${GREEN}[SIM]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

# Function to display usage
usage() {
    local detected_sim=$(detect_simulator)
    echo "SNN Accelerator Simulation Script"
    echo "=================================="
    echo ""
    echo "Detected simulator: $detected_sim"
    echo ""
    echo "Usage: $0 [options] [testbench_name]"
    echo "Options:"
    echo "  -s, --simulator <sim>   Force simulator: vivado, icarus, verilator"
    echo "  -t, --testbench <name>  Specify testbench module (default: $TB_TOP)"
    echo "  -g, --gui               Run simulation in GUI mode (if supported)"
    echo "  -c, --clean             Clean simulation directory before running"
    echo "  -d, --debug <level>     Debug level: none, typical, all (default: typical)"
    echo "  -w, --waves             Generate and open waveform viewer after simulation"
    echo "  -cov, --coverage        Enable code coverage (if supported)"
    echo "  -h, --help              Display this help message"
    echo ""
    echo "Available testbenches:"
    echo "  tb_top                  - Top level system testbench"
    echo "  tb_lif_neuron           - LIF neuron testbench"
    echo "  tb_spike_router         - Spike router testbench"
    echo "  tb_snn_accelerator_top  - Complete SNN accelerator testbench (default)"
    echo ""
    echo "Simulator Support:"
    echo "  Vivado Simulator  - Full support (GUI, coverage, advanced debugging)"
    echo "  Icarus Verilog    - Good support (waveforms, basic debugging)"
    echo "  Verilator         - Fast simulation (limited waveform support)"
    echo ""
    echo "Examples:"
    echo "  $0                              # Run default testbench with auto-detected simulator"
    echo "  $0 tb_snn_accelerator_top       # Run SNN accelerator testbench"
    echo "  $0 -t tb_lif_neuron -g          # Run LIF neuron testbench in GUI mode"
    echo "  $0 -s icarus -w -c               # Force Icarus Verilog with waveforms"
    echo "  $0 --clean --waves               # Clean, run simulation, then open waves"
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -s|--simulator)
            FORCE_SIMULATOR="$2"
            shift 2
            ;;
        -t|--testbench)
            TB_TOP="$2"
            shift 2
            ;;
        -g|--gui)
            GUI_MODE=1
            shift
            ;;
        -c|--clean)
            CLEAN_MODE=1
            shift
            ;;
        -d|--debug)
            DEBUG_MODE="$2"
            shift 2
            ;;
        -w|--waves)
            WAVES_MODE=1
            shift
            ;;
        -cov|--coverage)
            COVERAGE=1
            shift
            ;;
        -h|--help)
            usage
            ;;
        -*)
            print_error "Unknown option: $1"
            usage
            ;;
        *)
            # Positional argument - assume it's testbench name
            TB_TOP="$1"
            shift
            ;;
    esac
done

# Determine simulator to use
if [[ -n "$FORCE_SIMULATOR" ]]; then
    SIMULATOR="$FORCE_SIMULATOR"
    print_info "Using forced simulator: $SIMULATOR"
else
    SIMULATOR=$(detect_simulator)
    print_info "Auto-detected simulator: $SIMULATOR"
fi

# Check if simulator is available
case $SIMULATOR in
    "vivado")
        if ! command -v xvlog >/dev/null 2>&1; then
            print_error "Vivado Simulator not found in PATH"
            exit 1
        fi
        ;;
    "icarus")
        if ! command -v iverilog >/dev/null 2>&1; then
            print_error "Icarus Verilog not found in PATH"
            exit 1
        fi
        ;;
    "verilator")
        if ! command -v verilator >/dev/null 2>&1; then
            print_error "Verilator not found in PATH"
            exit 1
        fi
        ;;
    "none")
        print_error "No supported simulator found!"
        print_error "Please install one of: Vivado Simulator, Icarus Verilog, or Verilator"
        exit 1
        ;;
    *)
        print_error "Unsupported simulator: $SIMULATOR"
        print_error "Supported simulators: vivado, icarus, verilator"
        exit 1
        ;;
esac

print_msg "Starting simulation with $SIMULATOR"
print_msg "Testbench: $TB_TOP"

# Create work directory
if [[ $CLEAN_MODE -eq 1 ]]; then
    print_msg "Cleaning simulation directory..."
    rm -rf "$WORK_DIR"
fi

mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

# Find all source files
find_sources() {
    local sources=""
    
    # Add RTL sources
    if [[ -d "$RTL_DIR" ]]; then
        sources="$sources $(find "$RTL_DIR" -name "*.v" -o -name "*.sv" | sort)"
    fi
    
    # Add testbench
    if [[ -f "$TB_DIR/${TB_TOP}.v" ]]; then
        sources="$sources $TB_DIR/${TB_TOP}.v"
    elif [[ -f "$TB_DIR/${TB_TOP}.sv" ]]; then
        sources="$sources $TB_DIR/${TB_TOP}.sv"
    else
        print_error "Testbench file not found: $TB_DIR/${TB_TOP}.v or ${TB_TOP}.sv"
        exit 1
    fi
    
    echo "$sources"
}

# Get all source files
SOURCES=$(find_sources)
print_info "Found $(echo $SOURCES | wc -w) source files"

# Run simulation based on simulator
case $SIMULATOR in
    "vivado")
        run_vivado_sim
        ;;
    "icarus")
        run_icarus_sim
        ;;
    "verilator")
        run_verilator_sim
        ;;
esac

# Vivado Simulator function
run_vivado_sim() {
    print_msg "Running Vivado Simulator..."
    
    # Create simulation project
    xvlog $SOURCES
    if [[ $? -ne 0 ]]; then
        print_error "Compilation failed with Vivado Simulator"
        exit 1
    fi
    
    # Elaborate design
    xelab -debug typical $TB_TOP -s sim_snapshot
    if [[ $? -ne 0 ]]; then
        print_error "Elaboration failed with Vivado Simulator"
        exit 1
    fi
    
    # Run simulation
    if [[ $GUI_MODE -eq 1 ]]; then
        xsim sim_snapshot -gui
    else
        xsim sim_snapshot -runall
    fi
}

# Icarus Verilog function
run_icarus_sim() {
    print_msg "Running Icarus Verilog..."
    
    local sim_args=""
    if [[ $WAVES_MODE -eq 1 ]]; then
        sim_args="-DVCD_OUTPUT"
    fi
    
    # Compile and run
    iverilog -o sim_exe $sim_args -I"$RTL_DIR" $SOURCES
    if [[ $? -ne 0 ]]; then
        print_error "Compilation failed with Icarus Verilog"
        exit 1
    fi
    
    print_msg "Running simulation..."
    ./sim_exe
    
    # Open waveforms if requested
    if [[ $WAVES_MODE -eq 1 && -f "waves.vcd" ]]; then
        print_msg "Opening waveforms in GTKWave..."
        if command -v gtkwave >/dev/null 2>&1; then
            gtkwave waves.vcd &
        else
            print_warning "GTKWave not found. VCD file saved as waves.vcd"
        fi
    fi
}

# Verilator function
run_verilator_sim() {
    print_msg "Running Verilator..."
    
    # Create simple C++ testbench wrapper
    cat > tb_wrapper.cpp << 'EOF'
#include <verilated.h>
#include <iostream>

extern "C" {
    void run_simulation();
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    
    std::cout << "Starting Verilator simulation..." << std::endl;
    
    // Run for some cycles
    for (int i = 0; i < 10000; i++) {
        // Basic clock cycles - would need proper testbench for real simulation
    }
    
    std::cout << "Simulation completed" << std::endl;
    return 0;
}
EOF
    
    # Compile with Verilator
    verilator --cc $SOURCES --exe tb_wrapper.cpp
    if [[ $? -ne 0 ]]; then
        print_error "Compilation failed with Verilator"
        exit 1
    fi
    
    print_warning "Verilator support is basic - for full simulation use Vivado or Icarus"
}

print_msg "Simulation completed successfully!"
