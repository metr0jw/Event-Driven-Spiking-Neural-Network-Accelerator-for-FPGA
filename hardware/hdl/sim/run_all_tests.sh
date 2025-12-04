#!/bin/bash
# Comprehensive testbench runner with high-z detection

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TB_DIR="$SCRIPT_DIR/../tb"
RTL_DIR="$SCRIPT_DIR/../rtl"
RESULT_DIR="$SCRIPT_DIR/test_results"

mkdir -p "$RESULT_DIR"

total_tests=0
passed_tests=0
failed_tests=0
highz_warnings=0

run_test() {
    local name=$1
    local tb_file=$2
    local rtl_files=$3
    
    total_tests=$((total_tests + 1))
    echo -e "\n${YELLOW}=== Test $total_tests: $name ===${NC}"
    
    # Compile
    iverilog -g2012 -o "$RESULT_DIR/${name}.vvp" \
        -I"$RTL_DIR/common" -I"$RTL_DIR/layers" -I"$RTL_DIR/neurons" \
        -I"$RTL_DIR/synapses" -I"$RTL_DIR/router" -I"$RTL_DIR/interfaces" \
        $tb_file $rtl_files 2>&1 | tee "$RESULT_DIR/${name}_compile.log"
    
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo -e "${RED}COMPILE FAILED: $name${NC}"
        failed_tests=$((failed_tests + 1))
        return 1
    fi
    
    # Run simulation with timeout
    timeout 60 vvp "$RESULT_DIR/${name}.vvp" 2>&1 | tee "$RESULT_DIR/${name}_sim.log"
    local exit_code=${PIPESTATUS[0]}
    
    # Check for high-z (z, Z patterns in output values)
    # Pattern: looks for z/Z in signal values, excluding normal hex like 0x00
    if grep -E "=[[:space:]]*'?[bhd]?[0-9_]*[zZ]+|status:[[:space:]]*[zZ]|membrane.*=[[:space:]]*z" "$RESULT_DIR/${name}_sim.log" > /dev/null 2>&1; then
        echo -e "${YELLOW}WARNING: High-Z detected in $name${NC}"
        grep -E "=[[:space:]]*'?[bhd]?[0-9_]*[zZ]+|status:[[:space:]]*[zZ]|membrane.*=[[:space:]]*z" "$RESULT_DIR/${name}_sim.log" | head -10
        highz_warnings=$((highz_warnings + 1))
    fi
    
    # Check for X (undefined) - only standalone X values, not 0x prefix
    # Pattern: looks for xxxx or 'bx or =x patterns, not 0x hex prefix
    if grep -E "=[[:space:]]*'?[bhd]?[0-9_]*[xX]{2,}|=[[:space:]]*[xX]+[[:space:]]*$|state.*=[[:space:]]*x" "$RESULT_DIR/${name}_sim.log" > /dev/null 2>&1; then
        echo -e "${YELLOW}WARNING: Undefined (X) detected in $name${NC}"
        grep -E "=[[:space:]]*'?[bhd]?[0-9_]*[xX]{2,}|=[[:space:]]*[xX]+[[:space:]]*$|state.*=[[:space:]]*x" "$RESULT_DIR/${name}_sim.log" | head -10
        highz_warnings=$((highz_warnings + 1))
    fi
    
    # Check for PASS/FAIL in output
    # Look for "ALL TESTS PASSED" or check that there are no actual test failures
    if grep -i "ALL TESTS PASSED\|All tests PASSED" "$RESULT_DIR/${name}_sim.log" > /dev/null 2>&1; then
        echo -e "${GREEN}PASSED: $name${NC}"
        passed_tests=$((passed_tests + 1))
        return 0
    fi
    
    # Also pass if there are PASS entries but no "FAIL:" failure messages (ignore "Failed: 0")
    if grep -i "PASS:" "$RESULT_DIR/${name}_sim.log" > /dev/null 2>&1; then
        if ! grep -E "FAIL:|check_fail|SOME TESTS FAILED" "$RESULT_DIR/${name}_sim.log" > /dev/null 2>&1; then
            echo -e "${GREEN}PASSED: $name${NC}"
            passed_tests=$((passed_tests + 1))
            return 0
        fi
    fi
    
    if [ $exit_code -eq 124 ]; then
        echo -e "${RED}TIMEOUT: $name${NC}"
    else
        echo -e "${RED}FAILED: $name${NC}"
    fi
    failed_tests=$((failed_tests + 1))
    return 1
}

echo "=========================================="
echo "SNN Accelerator - Full Test Suite"
echo "=========================================="
echo "Date: $(date)"
echo ""

# Test 1: LIF Neuron
run_test "lif_neuron" \
    "$TB_DIR/tb_lif_neuron.v" \
    "$RTL_DIR/neurons/lif_neuron.v $RTL_DIR/common/reset_sync.v"

# Test 2: Simple LIF
run_test "simple_lif" \
    "$TB_DIR/tb_simple_lif.v" \
    "$RTL_DIR/neurons/lif_neuron.v $RTL_DIR/common/reset_sync.v"

# Test 3: LIF Neuron Array
run_test "lif_neuron_array" \
    "$TB_DIR/tb_lif_neuron_array.v" \
    "$RTL_DIR/neurons/lif_neuron_array.v $RTL_DIR/neurons/lif_neuron.v $RTL_DIR/common/reset_sync.v"

# Test 4: Spike Router
run_test "spike_router" \
    "$TB_DIR/tb_spike_router.v" \
    "$RTL_DIR/router/spike_router.v $RTL_DIR/common/*.v"

# Test 5: Conv1D (AC)
run_test "conv1d" \
    "$TB_DIR/tb_snn_conv1d.v" \
    "$RTL_DIR/layers/snn_conv1d.v"

# Test 6: Conv2D (AC)
run_test "conv2d" \
    "$TB_DIR/tb_snn_conv2d.v" \
    "$RTL_DIR/layers/snn_conv2d.v"

# Test 7: MaxPool1D
run_test "maxpool1d" \
    "$TB_DIR/tb_snn_maxpool1d.v" \
    "$RTL_DIR/layers/snn_maxpool1d.v"

# Test 8: MaxPool2D
run_test "maxpool2d" \
    "$TB_DIR/tb_snn_maxpool2d.v" \
    "$RTL_DIR/layers/snn_maxpool2d.v"

# Test 9: AvgPool1D
run_test "avgpool1d" \
    "$TB_DIR/tb_snn_avgpool1d.v" \
    "$RTL_DIR/layers/snn_avgpool1d.v"

# Test 10: AvgPool2D
run_test "avgpool2d" \
    "$TB_DIR/tb_snn_avgpool2d.v" \
    "$RTL_DIR/layers/snn_avgpool2d.v"

# Test 11: Layer Manager
run_test "layer_manager" \
    "$TB_DIR/tb_snn_layer_manager.v" \
    "$RTL_DIR/layers/snn_layer_manager.v"

# Test 12: Synapse Array
run_test "synapse_array" \
    "$TB_DIR/tb_synapse_array.v" \
    "$RTL_DIR/synapses/synapse_array.v $RTL_DIR/synapses/weight_memory.v $RTL_DIR/common/*.v"

echo ""
echo "=========================================="
echo "TEST SUMMARY"
echo "=========================================="
echo -e "Total Tests:    $total_tests"
echo -e "Passed:         ${GREEN}$passed_tests${NC}"
echo -e "Failed:         ${RED}$failed_tests${NC}"
echo -e "High-Z Warnings: ${YELLOW}$highz_warnings${NC}"
echo "=========================================="

if [ $failed_tests -eq 0 ] && [ $highz_warnings -eq 0 ]; then
    echo -e "${GREEN}ALL TESTS PASSED - NO HIGH-Z ISSUES${NC}"
    exit 0
elif [ $failed_tests -eq 0 ]; then
    echo -e "${YELLOW}ALL TESTS PASSED - BUT HIGH-Z WARNINGS FOUND${NC}"
    exit 1
else
    echo -e "${RED}SOME TESTS FAILED${NC}"
    exit 1
fi
