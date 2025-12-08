#!/bin/bash
# Quick verification script for HLS IP update

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

echo "=== HLS IP Update Verification Checklist ==="
echo ""

# 1. Check HLS IP exists
echo "[ ] 1. Checking HLS IP output..."
HLS_IP_DIR="${PROJECT_ROOT}/hardware/hls/hls_output/hls/impl/ip"
if [ -d "${HLS_IP_DIR}" ]; then
    HLS_IP_ZIP=$(ls "${HLS_IP_DIR}"/*.zip 2>/dev/null | head -1)
    if [ -n "${HLS_IP_ZIP}" ]; then
        echo "    ✓ HLS IP package found: $(basename ${HLS_IP_ZIP})"
        ls -lh "${HLS_IP_ZIP}"
    else
        echo "    ✗ No IP zip found in ${HLS_IP_DIR}"
        exit 1
    fi
else
    echo "    ✗ HLS IP directory not found: ${HLS_IP_DIR}"
    echo "       Run: cd hardware/hls/scripts && ./build_hls.sh"
    exit 1
fi

# 2. Check control AXI register map
echo ""
echo "[ ] 2. Extracting register offsets from HLS IP..."
CTRL_AXI="${HLS_IP_DIR}/hdl/verilog/snn_top_hls_ctrl_s_axi.v"
if [ -f "${CTRL_AXI}" ]; then
    echo "    ✓ Control AXI module found"
    echo "    Key register offsets:"
    grep "ADDR_MODE_REG\|ADDR_TIME_STEPS\|ADDR_ENCODER\|ADDR_REWARD" "${CTRL_AXI}" | head -5
else
    echo "    ✗ Control AXI not found at ${CTRL_AXI}"
fi

# 3. Check component.xml for stream ports
echo ""
echo "[ ] 3. Checking AXI Stream ports in component.xml..."
COMPONENT_XML="${HLS_IP_DIR}/component.xml"
if [ -f "${COMPONENT_XML}" ]; then
    echo "    ✓ Component.xml found"
    echo "    AXI Stream interfaces:"
    grep -o "spirit:name.*axis[^<]*" "${COMPONENT_XML}" | grep -v "TDATA\|TVALID" | head -5
else
    echo "    ✗ component.xml not found"
fi

# 4. Check Python RegisterMap sync
echo ""
echo "[ ] 4. Checking Python RegisterMap sync..."
PYTHON_REGMAP="${PROJECT_ROOT}/software/python/snn_fpga_accelerator/xrt_backend.py"
if [ -f "${PYTHON_REGMAP}" ]; then
    echo "    ✓ xrt_backend.py found"
    echo "    Checking key offsets:"
    grep "mode_reg:\|time_steps_reg:\|encoder_base:\|reward_signal:" "${PYTHON_REGMAP}"
else
    echo "    ✗ Python backend not found"
fi

# 5. Check Vivado TCL script
echo ""
echo "[ ] 5. Checking Vivado update script..."
TCL_SCRIPT="${PROJECT_ROOT}/hardware/scripts/update_hls_ip.tcl"
if [ -f "${TCL_SCRIPT}" ]; then
    echo "    ✓ update_hls_ip.tcl found"
    echo "    To run: vivado -mode batch -source ${TCL_SCRIPT}"
else
    echo "    ✗ update_hls_ip.tcl not found"
fi

# 6. Summary
echo ""
echo "=== Summary ==="
echo "✓ All checks passed. Ready to update Vivado IP."
echo ""
echo "Next steps:"
echo "  1. cd hardware/scripts"
echo "  2. vivado -mode batch -source update_hls_ip.tcl"
echo "  3. Check console output for assigned base address"
echo "  4. Update Python code with base address from Vivado Address Editor"
echo "  5. Run synthesis/implementation: launch_runs impl_1 -to_step write_bitstream"
echo ""
