#!/bin/bash

##-----------------------------------------------------------------------------
## Title         : Master Build Script
## Project       : PYNQ-Z2 SNN Accelerator
## File          : run_all.sh
## Author        : Jiwoon Lee (@metr0jw)
## Organization  : Kwangwoon University, Seoul, South Korea
## Contact       : jwlee@linux.com
## Description   : Runs complete hardware build flow (HLS + Vivado)
##-----------------------------------------------------------------------------

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Default paths
VIVADO_PATH="/tools/Xilinx/2025.2/Vivado/bin/vivado"
VITIS_SETTINGS="/tools/Xilinx/2025.2/Vitis/settings64.sh"

# Get number of CPU cores for parallel builds
NUM_JOBS=$(nproc)
echo "Detected ${NUM_JOBS} CPU cores for parallel builds"

# Script options
BUILD_HLS=1
CREATE_PROJECT=1
BUILD_BITSTREAM=1
PACKAGE_IP=1
PROGRAM_BOARD=0

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --vivado-path)
            VIVADO_PATH="$2"
            shift 2
            ;;
        --skip-hls)
            BUILD_HLS=0
            shift
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
            echo "  --skip-hls           Skip HLS IP synthesis"
            echo "  --skip-create        Skip project creation and bitstream build"
            echo "  --skip-build         Skip bitstream build (same as --skip-create)"
            echo "  --package-ip         Package as IP core"
            echo "  --program            Program FPGA after build"
            echo "  --help               Show this message"
            echo ""
            echo "Note: Project creation and bitstream generation are done together by build_pynq_with_hls.tcl"
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

# Check Vitis settings
if [ ! -f "$VITIS_SETTINGS" ]; then
    echo -e "${RED}Error: Vitis settings not found at $VITIS_SETTINGS${NC}"
    exit 1
fi

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}PYNQ-Z2 SNN Accelerator Build${NC}"
echo -e "${GREEN}========================================${NC}"

# Build HLS IP
if [ $BUILD_HLS -eq 1 ]; then
    echo -e "\n${YELLOW}Building HLS IP (snn_top_hls)...${NC}"
    cd ../hls
    source $VITIS_SETTINGS
    
    # Build using correct v++ HLS syntax from build_hls.sh
    v++ -c --mode hls \
        --part xc7z020clg400-1 \
        --work_dir hls_output \
        --hls.clock 10ns \
        --hls.syn.top snn_top_hls \
        --hls.syn.file "src/snn_top_hls.cpp" \
        --hls.flow_target vivado
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}HLS synthesis failed!${NC}"
        exit 1
    fi
    cd ../scripts
    echo -e "${GREEN}HLS IP generated successfully${NC}"
fi

# Create project and build bitstream (build_pynq_with_hls.tcl does both)
if [ $CREATE_PROJECT -eq 1 ] || [ $BUILD_BITSTREAM -eq 1 ]; then
    echo -e "\n${YELLOW}Creating Vivado project with HLS IP and building bitstream using ${NUM_JOBS} parallel jobs...${NC}"
    echo -e "${YELLOW}This will take 10-30 minutes depending on hardware...${NC}"
    source $VITIS_SETTINGS
    $VIVADO_PATH -mode batch -source build_pynq_with_hls.tcl -tclargs -jobs ${NUM_JOBS}
    if [ $? -ne 0 ]; then
        echo -e "${RED}Build failed!${NC}"
        exit 1
    fi
    echo -e "${GREEN}Project creation and bitstream generation complete${NC}"
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
PROJ_ROOT="$(cd .. && pwd)"
echo -e "\n${GREEN}Output files:${NC}"
if [ -f "${PROJ_ROOT}/outputs/snn_accelerator_hls.bit" ]; then
    echo -e "  Bitstream: ${GREEN}${PROJ_ROOT}/outputs/snn_accelerator_hls.bit${NC}"
fi
if [ -f "${PROJ_ROOT}/outputs/design_1.hwh" ]; then
    echo -e "  Hardware Handoff: ${GREEN}${PROJ_ROOT}/outputs/design_1.hwh${NC}"
fi
if [ -f "${PROJ_ROOT}/outputs/snn_accelerator_hls.xsa" ]; then
    echo -e "  Hardware Platform: ${GREEN}${PROJ_ROOT}/outputs/snn_accelerator_hls.xsa${NC}"
fi
if [ -d "${PROJ_ROOT}/hls/hls_output" ]; then
    echo -e "  HLS IP: ${GREEN}${PROJ_ROOT}/hls/hls_output/${NC}"
fi
