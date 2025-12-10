#!/bin/bash
#
# Run Vivado build with proper environment sourcing
#

set -e

cd /home/jwlee/workspace/Event-Driven-Spiking-Neural-Network-Accelerator-for-FPGA/hardware/build

echo "[INFO] Sourcing Vivado environment..."
source /home/jwlee/tools/2025.2/Vitis/.settings64-Vitis.sh

echo "[INFO] Running Vivado in batch mode..."
vivado -mode batch -source build_vivado.tcl

echo "[INFO] Build complete"
