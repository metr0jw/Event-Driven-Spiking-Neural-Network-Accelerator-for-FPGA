#!/bin/bash

##-----------------------------------------------------------------------------
## Title         : Master Build Script
## Project       : PYNQ-Z2 SNN Accelerator
## File          : run_all.sh
## Author        : Jiwoon Lee (@metr0jw)
## Organization  : Kwangwoon University, Seoul, South Korea
## Contact       : jwlee@linux.com
## Description   : Runs complete hardware build flow
##-----------------------------------------------------------------------------

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Default Vivado path
VIVADO_PATH="/tools/Xilinx/Vivado/2023.1/bin/vivado"

# Script options
CREATE_PROJECT=1
BUILD_BITSTREAM=1
PACKAGE_IP=0
PROGRAM_BOARD=0

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --vivado-path)
            VIVADO_PATH="$2"
            shift 2
            ;;
        --skip-create)
            CREATE_PROJECT=0
            shift
            ;;
        --skip-build)
            BUILD_BITSTREAM=0
            shift
            ;;
        --package-ip)
            PACKAGE_IP=1
            shift
            ;;
        --program)
            PROGRAM_BOARD=1
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --vivado-path PATH   Path to Vivado executable"
            echo "  --skip-create        Skip project creation"
            echo "  --skip-build         Skip bitstream build"
            echo "  --package-ip         Package as IP core"
            echo "  --program            Program FPGA after build"
            echo "  --help               Show this message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check Vivado
if [ ! -f "$VIVADO_PATH" ]; then
    echo -e "${RED}Error: Vivado not found at $VIVADO_PATH${NC}"
    exit 1
fi

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}PYNQ-Z2 SNN Accelerator Build${NC}"
echo -e "${GREEN}========================================${NC}"

# Create project
if [ $CREATE_PROJECT -eq 1 ]; then
    echo -e "\n${YELLOW}Creating Vivado project...${NC}"
    $VIVADO_PATH -mode batch -source create_project.tcl -tclargs
    if [ $? -ne 0 ]; then
        echo -e "${RED}Project creation failed!${NC}"
        exit 1
    fi
fi

# Build bitstream
if [ $BUILD_BITSTREAM -eq 1 ]; then
    echo -e "\n${YELLOW}Building bitstream...${NC}"
    $VIVADO_PATH -mode batch -source build_bitstream.tcl
    if [ $? -ne 0 ]; then
        echo -e "${RED}Bitstream build failed!${NC}"
        exit 1
    fi
fi

# Package IP
if [ $PACKAGE_IP -eq 1 ]; then
    echo -e "\n${YELLOW}Packaging IP core...${NC}"
    $VIVADO_PATH -mode batch -source package_ip.tcl
    if [ $? -ne 0 ]; then
        echo -e "${RED}IP packaging failed!${NC}"
        exit 1
    fi
fi

# Program board
if [ $PROGRAM_BOARD -eq 1 ]; then
    echo -e "\n${YELLOW}Programming FPGA...${NC}"
    $VIVADO_PATH -mode batch -source program_board.tcl
    if [ $? -ne 0 ]; then
        echo -e "${RED}FPGA programming failed!${NC}"
        exit 1
    fi
fi

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}Build completed successfully!${NC}"
echo -e "${GREEN}========================================${NC}"

# Show output location
if [ -d "../../build/bitstreams" ]; then
    echo -e "\n${GREEN}Output files:${NC}"
    ls -la ../../build/bitstreams/
fi
