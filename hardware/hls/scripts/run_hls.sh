#!/bin/bash

##-----------------------------------------------------------------------------
## Title         : HLS Build Shell Script
## Project       : PYNQ-Z2 SNN Accelerator
## File          : run_hls.sh
## Author        : Jiwoon Lee (@metr0jw)
## Organization  : Kwangwoon University, Seoul, South Korea
## Contact       : jwlee@linux.com
## Description   : Shell wrapper for HLS build process
##-----------------------------------------------------------------------------

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Default options
VITIS_HLS_PATH="/tools/Xilinx/Vitis_HLS/2023.2/bin/vitis_hls"
DO_COSIM=0
CLEAN_BUILD=0

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --cosim)
            DO_COSIM=1
            shift
            ;;
        --clean)
            CLEAN_BUILD=1
            shift
            ;;
        --path)
            VITIS_HLS_PATH="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --cosim    Run co-simulation (slow)"
            echo "  --clean    Clean before build"
            echo "  --path     Path to vitis_hls executable"
            echo "  --help     Show this message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check if Vitis HLS exists
if [ ! -f "$VITIS_HLS_PATH" ]; then
    echo -e "${RED}Error: Vitis HLS not found at $VITIS_HLS_PATH${NC}"
    echo "Please specify correct path with --path option"
    exit 1
fi

echo -e "${GREEN}===============================================${NC}"
echo -e "${GREEN}PYNQ-Z2 SNN Accelerator HLS Build${NC}"
echo -e "${GREEN}===============================================${NC}"

# Prepare build options
BUILD_OPTS=""
if [ $CLEAN_BUILD -eq 1 ]; then
    BUILD_OPTS="$BUILD_OPTS -clean"
fi

if [ $DO_COSIM -eq 1 ]; then
    BUILD_OPTS="$BUILD_OPTS -cosim"
fi

# Create log directory
mkdir -p logs

# Run HLS build
echo -e "${YELLOW}Starting HLS build...${NC}"
echo "Options: $BUILD_OPTS"
echo ""

$VITIS_HLS_PATH -f build_all.tcl $BUILD_OPTS 2>&1 | tee logs/hls_build.log

# Check result
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo -e "\n${GREEN}HLS build completed successfully!${NC}"
    
    # Show IP repository location
    echo -e "\n${GREEN}Generated IP cores location:${NC}"
    echo "$(pwd)/../../ip/custom_ip"
    
    # List generated IPs
    if [ -d "../../ip/custom_ip" ]; then
        echo -e "\n${GREEN}Available IP cores:${NC}"
        ls -1 ../../ip/custom_ip/
    fi
else
    echo -e "\n${RED}HLS build failed! Check logs/hls_build.log for details${NC}"
    exit 1
fi

echo -e "\n${GREEN}===============================================${NC}"
