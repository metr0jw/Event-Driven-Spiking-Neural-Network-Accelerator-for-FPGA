#!/bin/bash
##-----------------------------------------------------------------------------
## Title         : HLS Build Script (v++ CLI - Vitis 2025.2+)
## Project       : PYNQ-Z2 SNN Accelerator
## File          : build_hls.sh
## Author        : Jiwoon Lee (@metr0jw)
## Organization  : Kwangwoon University, Seoul, South Korea
## Contact       : jwlee@linux.com
## Description   : Modern v++ compiler based HLS build script
##                 Replaces deprecated vitis_hls -f script.tcl workflow
##-----------------------------------------------------------------------------

set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# Default configuration
PART="xc7z020clg400-1"
CLOCK="10ns"
TOP_FUNCTION="snn_top_hls"
SRC_FILE="src/snn_top_hls.cpp"
WORK_DIR="./hls_output"

# Script location
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
HLS_DIR="$(dirname "$SCRIPT_DIR")"

# Parse arguments
CLEAN=0
VERBOSE=0
while [[ $# -gt 0 ]]; do
    case $1 in
        --clean)
            CLEAN=1
            shift
            ;;
        --verbose|-v)
            VERBOSE=1
            shift
            ;;
        --part)
            PART="$2"
            shift 2
            ;;
        --clock)
            CLOCK="$2"
            shift 2
            ;;
        --top)
            TOP_FUNCTION="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --clean        Clean previous build before starting"
            echo "  --verbose, -v  Verbose output"
            echo "  --part PART    Target FPGA part (default: xc7z020clg400-1)"
            echo "  --clock PERIOD Clock period (default: 10ns)"
            echo "  --top NAME     Top function name (default: snn_top_hls)"
            echo "  --help, -h     Show this message"
            echo ""
            echo "Example:"
            echo "  $0 --clean --clock 8ns"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Change to HLS directory
cd "$HLS_DIR"

echo -e "${CYAN}============================================${NC}"
echo -e "${CYAN}  SNN Accelerator HLS Build (v++ CLI)${NC}"
echo -e "${CYAN}============================================${NC}"
echo -e "Part:         ${GREEN}$PART${NC}"
echo -e "Clock:        ${GREEN}$CLOCK${NC}"
echo -e "Top Function: ${GREEN}$TOP_FUNCTION${NC}"
echo -e "Source:       ${GREEN}$SRC_FILE${NC}"
echo -e "Work Dir:     ${GREEN}$WORK_DIR${NC}"
echo ""

# Clean if requested
if [[ $CLEAN -eq 1 ]]; then
    echo -e "${YELLOW}Cleaning previous build...${NC}"
    rm -rf "$WORK_DIR"
fi

# Check if source file exists
if [[ ! -f "$SRC_FILE" ]]; then
    echo -e "${RED}Error: Source file not found: $SRC_FILE${NC}"
    exit 1
fi

# Build command
echo -e "${GREEN}Starting HLS synthesis with v++ compiler...${NC}"
echo ""

V_CMD="v++ -c --mode hls \
    --part $PART \
    --work_dir $WORK_DIR \
    --hls.clock $CLOCK \
    --hls.syn.top $TOP_FUNCTION \
    --hls.syn.file \"$SRC_FILE\" \
    --hls.flow_target vivado"

if [[ $VERBOSE -eq 1 ]]; then
    echo -e "${CYAN}Command: $V_CMD${NC}"
    echo ""
fi

# Run v++ synthesis
eval $V_CMD

# Check result
if [[ $? -eq 0 ]]; then
    echo ""
    echo -e "${GREEN}============================================${NC}"
    echo -e "${GREEN}  HLS Synthesis Completed Successfully!${NC}"
    echo -e "${GREEN}============================================${NC}"
    
    # Show summary if report exists
    REPORT_FILE="$WORK_DIR/hls/syn/report/csynth.rpt"
    if [[ -f "$REPORT_FILE" ]]; then
        echo ""
        echo -e "${CYAN}Performance Summary:${NC}"
        grep -E "Estimated Fmax|BRAM|DSP|FF|LUT" "$REPORT_FILE" 2>/dev/null | head -5 || true
    fi
    
    echo ""
    echo -e "IP Package: ${GREEN}$WORK_DIR/hls/impl/ip/${NC}"
    echo -e "Reports:    ${GREEN}$WORK_DIR/hls/syn/report/${NC}"
else
    echo ""
    echo -e "${RED}============================================${NC}"
    echo -e "${RED}  HLS Synthesis FAILED${NC}"
    echo -e "${RED}============================================${NC}"
    exit 1
fi
