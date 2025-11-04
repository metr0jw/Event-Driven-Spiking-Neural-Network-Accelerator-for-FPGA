#!/bin/bash

##-----------------------------------------------------------------------------
## Title         : Simulation Run Script for SNN Accelerator
## Project       : PYNQ-Z2 SNN Accelerator
## File          : run_sim.sh
## Author        : Jiwoon Lee (@metr0jw)
## Organization  : Kwangwoon University, Seoul, South Korea
## Description   : Automated simulation script for multiple simulators
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

SIMULATOR=$(detect_simulator)
GUI_MODE=0
CLEAN_MODE=0
DEBUG_MODE="typical"
COVERAGE=0

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

# Function to display usage
usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -t, --testbench <name>  Specify testbench module (default: tb_top)"
    echo "  -g, --gui               Run simulation in GUI mode"
    echo "  -c, --clean             Clean simulation directory before running"
    echo "  -d, --debug <level>     Debug level: none, typical, all (default: typical)"
    echo "  -w, --waves             Open waveform viewer after simulation"
    echo "  -cov, --coverage        Enable code coverage"
    echo "  -h, --help              Display this help message"
    echo ""
    echo "Available testbenches:"
    echo "  tb_top          - Top level system testbench"
    echo "  tb_lif_neuron   - LIF neuron testbench"
    echo "  tb_spike_router - Spike router testbench"
    echo ""
    echo "Examples:"
    echo "  $0                      # Run default testbench (tb_top)"
    echo "  $0 -t tb_lif_neuron -g  # Run LIF neuron testbench in GUI mode"
    echo "  $0 -c -w                # Clean, run simulation, then open waves"
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
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
        *)
            print_error "Unknown option: $1"
            usage
            ;;
    esac
done

# Create work directory
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

# Clean if requested
if [ $CLEAN_MODE -eq 1 ]; then
    print_msg "Cleaning simulation directory..."
    rm -rf xsim.dir .Xil *.log *.jou *.pb *.wdb *.vcd
fi

# Source file lists
print_msg "Collecting source files..."

# Common modules
COMMON_SOURCES=(
    "$RTL_DIR/common/reset_sync.v"
    "$RTL_DIR/common/fifo.v"
    "$RTL_DIR/common/sync_pulse.v"
)

# Core modules
CORE_SOURCES=(
    "$RTL_DIR/neurons/lif_neuron.v"
    "$RTL_DIR/neurons/lif_neuron_array.v"
    "$RTL_DIR/synapses/synapse_array.v"
    "$RTL_DIR/synapses/weight_memory.v"
    "$RTL_DIR/router/spike_router.v"
    "$RTL_DIR/interfaces/axi_wrapper.v"
    "$RTL_DIR/top/snn_accelerator_top.v"
)

# Select testbench
case $TB_TOP in
    tb_top)
        TB_FILE="$TB_DIR/tb_top.v"
        SNAPSHOT_NAME="tb_top_snapshot"
        ;;
    tb_lif_neuron)
        TB_FILE="$TB_DIR/tb_lif_neuron.v"
        SNAPSHOT_NAME="tb_lif_neuron_snapshot"
        # Only need neuron-related files
        CORE_SOURCES=("$RTL_DIR/neurons/lif_neuron.v")
        ;;
    tb_spike_router)
        TB_FILE="$TB_DIR/tb_spike_router.v"
        SNAPSHOT_NAME="tb_spike_router_snapshot"
        # Only need router and FIFO
        CORE_SOURCES=(
            "$RTL_DIR/router/spike_router.v"
            "$RTL_DIR/common/fifo.v"
        )
        ;;
    *)
        print_error "Unknown testbench: $TB_TOP"
        exit 1
        ;;
esac

# Compilation options
XVLOG_OPTS="--incr --relax"
XELAB_OPTS="-debug $DEBUG_MODE"

if [ $COVERAGE -eq 1 ]; then
    XVLOG_OPTS="$XVLOG_OPTS -cover all"
    XELAB_OPTS="$XELAB_OPTS -cc_db"
fi

# Compile SystemVerilog/Verilog files
print_msg "### COMPILING VERILOG SOURCES ###"
echo ""

# Compile common modules
for src in "${COMMON_SOURCES[@]}"; do
    if [ -f "$src" ]; then
        echo "Compiling: $(basename $src)"
        xvlog $XVLOG_OPTS "$src" 2>&1 | tee -a compile.log
        if [ ${PIPESTATUS[0]} -ne 0 ]; then
            print_error "Compilation failed for $(basename $src)"
            exit 1
        fi
    else
        print_warning "Source file not found: $src"
    fi
done

# Compile core modules
for src in "${CORE_SOURCES[@]}"; do
    if [ -f "$src" ]; then
        echo "Compiling: $(basename $src)"
        xvlog $XVLOG_OPTS "$src" 2>&1 | tee -a compile.log
        if [ ${PIPESTATUS[0]} -ne 0 ]; then
            print_error "Compilation failed for $(basename $src)"
            exit 1
        fi
    else
        print_warning "Source file not found: $src"
    fi
done

# Compile testbench
if [ -f "$TB_FILE" ]; then
    echo "Compiling testbench: $(basename $TB_FILE)"
    xvlog $XVLOG_OPTS "$TB_FILE" 2>&1 | tee -a compile.log
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        print_error "Testbench compilation failed"
        exit 1
    fi
else
    print_error "Testbench file not found: $TB_FILE"
    exit 1
fi

echo ""
print_msg "### ELABORATING DESIGN ###"
echo ""

# Elaborate design
xelab $XELAB_OPTS -top $TB_TOP -snapshot $SNAPSHOT_NAME 2>&1 | tee -a elaborate.log
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    print_error "Elaboration failed"
    exit 1
fi

# Create TCL script for simulation
cat > xsim_run.tcl << EOF
# Simulation run script
open_vcd ${TB_TOP}.vcd
log_vcd *
log_vcd [get_objects -r *]

# Run simulation
run all

# Close VCD
close_vcd

# Check for errors
if {[string match "*ERROR*" [exec grep -i error xsim.log]]} {
    puts "Simulation completed with errors"
} else {
    puts "Simulation completed successfully"
}

exit
EOF

# Create TCL script for GUI mode
cat > xsim_gui.tcl << EOF
# GUI mode script
if {[file exists ../wave_config.do]} {
    source ../wave_config.do
}

# Run for a limited time in GUI mode
run 10us
EOF

echo ""
print_msg "### RUNNING SIMULATION ###"
echo ""

# Run simulation
if [ $GUI_MODE -eq 1 ]; then
    print_msg "Starting simulation in GUI mode..."
    xsim $SNAPSHOT_NAME --gui --tclbatch xsim_gui.tcl 2>&1 | tee -a xsim.log
else
    print_msg "Starting simulation in batch mode..."
    xsim $SNAPSHOT_NAME --tclbatch xsim_run.tcl 2>&1 | tee -a xsim.log
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        print_error "Simulation failed"
        exit 1
    fi
fi

# Open waveform viewer if requested
if [ $WAVES_MODE -eq 1 ] && [ $GUI_MODE -eq 0 ]; then
    print_msg "### OPENING WAVEFORM VIEWER ###"
    if [ -f "${TB_TOP}.vcd" ]; then
        print_msg "Opening VCD file: ${TB_TOP}.vcd"
        gtkwave "${TB_TOP}.vcd" ../wave_config.gtkw &
    elif [ -f "${SNAPSHOT_NAME}.wdb" ]; then
        print_msg "Opening WDB file: ${SNAPSHOT_NAME}.wdb"
        xsim --gui "${SNAPSHOT_NAME}.wdb" &
    else
        print_warning "No waveform file found"
    fi
fi

# Coverage report
if [ $COVERAGE -eq 1 ]; then
    print_msg "### GENERATING COVERAGE REPORT ###"
    xcrg -report_dir coverage_report -cc_db ccdb
fi

# Summary
echo ""
print_msg "### SIMULATION SUMMARY ###"
echo "Testbench: $TB_TOP"
echo "Snapshot: $SNAPSHOT_NAME"
echo "Log files: compile.log, elaborate.log, xsim.log"
if [ -f "${TB_TOP}.vcd" ]; then
    echo "VCD file: ${TB_TOP}.vcd"
fi

# Check for errors or warnings
if grep -q "ERROR" xsim.log 2>/dev/null; then
    print_error "Simulation completed with errors. Check xsim.log for details."
    exit 1
else
    print_msg "Simulation completed successfully!"
fi

cd "$SCRIPT_DIR"
