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
    local test_name=$1
    local tb_file=$2
    
    echo -e "\nRunning $test_name..."
    
    # Run the testbench
    ./$tb_file > ${test_name}_output.log 2>&1
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}[PASS]${NC} $test_name"
        ((PASSED++))
    else
        echo -e "${RED}[FAIL]${NC} $test_name"
        ((FAILED++))
        echo "  See ${test_name}_output.log for details"
    fi
}

# Compile and run each testbench
echo "Compiling testbenches..."

# Learning Engine Test
g++ -std=c++11 -I../include tb_snn_learning_engine.cpp -o tb_learning
run_test "Learning Engine" tb_learning

# Spike Encoder Test
g++ -std=c++11 -I../include tb_spike_encoder.cpp -o tb_encoder
run_test "Spike Encoder" tb_encoder

# Weight Updater Test
g++ -std=c++11 -I../include tb_weight_updater.cpp -o tb_weight
run_test "Weight Updater" tb_weight

# Spike Decoder Test
g++ -std=c++11 -I../include tb_spike_decoder.cpp -o tb_decoder
run_test "Spike Decoder" tb_decoder

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
