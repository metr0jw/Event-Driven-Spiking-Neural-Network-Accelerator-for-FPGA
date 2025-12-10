#!/bin/bash

#-----------------------------------------------------------------------------
## Title         : HLS Test Runner Script
## Project       : PYNQ-Z2 SNN Accelerator
## File          : run_tests.sh
## Author        : Jiwoon Lee (@metr0jw)
## Organization  : Kwangwoon University, Seoul, South Korea
## Contact       : jwlee@linux.com
## Description   : Runs all HLS testbenches
#-----------------------------------------------------------------------------

# Source Vitis HLS environment for ap_int.h and other HLS headers
if [ -f "/tools/Xilinx/2025.2/Vitis/settings64.sh" ]; then
    source /tools/Xilinx/2025.2/Vitis/settings64.sh
else
    echo "Warning: Vitis HLS environment not found. Tests may fail."
fi

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

# Test results
PASSED=0
FAILED=0

echo "======================================"
echo "Running HLS Testbenches"
echo "======================================"

# Function to run a test
run_test() {
    local test_name="$1"
    local tb_file="$2"
    local log_file="${test_name// /_}_output.log"
    
    echo -e "\nRunning $test_name..."
    
    # Run the testbench
    "./$tb_file" > "$log_file" 2>&1
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}[PASS]${NC} $test_name"
        ((PASSED++))
    else
        echo -e "${RED}[FAIL]${NC} $test_name"
        ((FAILED++))
        echo "  See $log_file for details"
    fi
}

# Compile and run each testbench
echo "Compiling testbenches..."

# Get HLS include path
HLS_INCLUDE=""
if [ -n "$XILINX_HLS" ]; then
    HLS_INCLUDE="-I$XILINX_HLS/include"
elif [ -n "$XILINX_VITIS" ]; then
    HLS_INCLUDE="-I$XILINX_VITIS/include"
fi

# Learning Engine Test
echo "Compiling Learning Engine testbench..."
g++ -std=c++11 -I../include $HLS_INCLUDE tb_snn_learning_engine.cpp ../src/snn_learning_engine.cpp -o tb_learning
if [ $? -eq 0 ]; then
    run_test "Learning Engine" tb_learning
else
    echo -e "${RED}[FAIL]${NC} Learning Engine (compilation failed)"
    ((FAILED++))
fi

# Spike Encoder Test
echo "Compiling Spike Encoder testbench..."
g++ -std=c++11 -I../include $HLS_INCLUDE tb_spike_encoder.cpp ../src/spike_encoder.cpp -o tb_encoder
if [ $? -eq 0 ]; then
    run_test "Spike Encoder" tb_encoder
else
    echo -e "${RED}[FAIL]${NC} Spike Encoder (compilation failed)"
    ((FAILED++))
fi

# Weight Updater Test
echo "Compiling Weight Updater testbench..."
g++ -std=c++11 -I../include $HLS_INCLUDE -DNUM_NEURONS=256 tb_weight_updater.cpp ../src/weight_updater.cpp -o tb_weight
if [ $? -eq 0 ]; then
    run_test "Weight Updater" tb_weight
else
    echo -e "${RED}[FAIL]${NC} Weight Updater (compilation failed)"
    ((FAILED++))
fi

# Summary
echo ""
echo "======================================"
echo "Test Summary"
echo "======================================"
echo -e "Passed: ${GREEN}$PASSED${NC}"
echo -e "Failed: ${RED}$FAILED${NC}"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}Some tests failed!${NC}"
    exit 1
fi
