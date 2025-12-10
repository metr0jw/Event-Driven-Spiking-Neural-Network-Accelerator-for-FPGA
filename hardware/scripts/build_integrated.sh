#!/bin/bash
##-----------------------------------------------------------------------------
## Title         : Build Integrated SNN System
## Project       : PYNQ-Z2 SNN Accelerator
## File          : build_integrated.sh
## Author        : Jiwoon Lee (@metr0jw)
## Description   : Builds complete system (HLS + Verilog RTL)
##-----------------------------------------------------------------------------

set -e  # Exit on error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJ_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "===================================================================="
echo "Building Integrated SNN Accelerator (HLS + Verilog RTL)"
echo "===================================================================="
echo "Project root: $PROJ_ROOT"
echo ""

# Find Vivado
VIVADO_BIN=""
if command -v vivado &> /dev/null; then
    VIVADO_BIN="vivado"
elif [ -f "/tools/Xilinx/2025.2/Vivado/bin/vivado" ]; then
    VIVADO_BIN="/tools/Xilinx/2025.2/Vivado/bin/vivado"
else
    echo "ERROR: Vivado not found"
    echo "Checked locations:"
    echo "  - System PATH"
    echo "  - /tools/Xilinx/2025.2/Vivado/bin/vivado"
    exit 1
fi
echo "Using Vivado: $VIVADO_BIN"

echo "Step 1: Checking HLS IP..."
if [ ! -f "$PROJ_ROOT/hardware/ip_repo/snn_top_hls/component.xml" ]; then
    echo "ERROR: HLS IP not found at hardware/ip_repo/snn_top_hls/"
    echo "Please run HLS synthesis first:"
    echo "  cd hardware/hls"
    echo "  v++ -c -t hw --platform xilinx_u250_gen3x16_xdma_4_1_202210_1 ..."
    exit 1
else
    echo "✓ HLS IP found at hardware/ip_repo/snn_top_hls/"
fi

echo ""
echo "Step 2: Cleaning previous build..."
rm -rf "$PROJ_ROOT/build/vivado_integrated"
mkdir -p "$PROJ_ROOT/build/vivado_integrated"
mkdir -p "$PROJ_ROOT/outputs"

echo ""
echo "Step 3: Running Vivado build..."
cd "$SCRIPT_DIR"
$VIVADO_BIN -mode batch -source build_integrated_system.tcl -log integrated_build.log -journal integrated_build.jou

if [ $? -eq 0 ]; then
    echo ""
    echo "===================================================================="
    echo "✅ BUILD SUCCESSFUL"
    echo "===================================================================="
    echo "Bitstream: $PROJ_ROOT/outputs/snn_integrated.bit"
    echo ""
    echo "Utilization Report: $PROJ_ROOT/outputs/integrated_utilization.rpt"
    echo "Timing Report:      $PROJ_ROOT/outputs/integrated_timing.rpt"
    echo "Power Report:       $PROJ_ROOT/outputs/integrated_power.rpt"
    echo ""
    echo "To deploy to PYNQ-Z2:"
    echo "  scp outputs/snn_integrated.bit xilinx@192.168.2.99:~/"
    echo "===================================================================="
else
    echo ""
    echo "===================================================================="
    echo "❌ BUILD FAILED"
    echo "===================================================================="
    echo "Check logs: $SCRIPT_DIR/integrated_build.log"
    exit 1
fi
