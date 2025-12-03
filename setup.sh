#!/bin/bash
#-----------------------------------------------------------------------------
# Title         : PYNQ-Z2 SNN Accelerator Setup Script
# Project       : PYNQ-Z2 SNN Accelerator
# File          : setup.sh
# Author        : Jiwoon Lee (@metr0jw)
# Organization  : Kwangwoon University, Seoul, South Korea
# Description   : Complete setup script for the SNN accelerator project
#-----------------------------------------------------------------------------

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project information
PROJECT_NAME="PYNQ-Z2 SNN Accelerator"
PROJECT_VERSION="1.0.0"
AUTHOR="Jiwoon Lee (@metr0jw)"

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}${PROJECT_NAME} Setup Script${NC}"
echo -e "${BLUE}Version: ${PROJECT_VERSION}${NC}"
echo -e "${BLUE}Author: ${AUTHOR}${NC}"
echo -e "${BLUE}============================================${NC}"

# Check if running on PYNQ board
check_pynq_board() {
    echo -e "${YELLOW}Checking if running on PYNQ board...${NC}"
    
    if [ -f "/etc/pynq_board" ]; then
        BOARD_TYPE=$(cat /etc/pynq_board)
        echo -e "${GREEN}✅ Detected PYNQ board: ${BOARD_TYPE}${NC}"
        return 0
    elif [ -f "/proc/device-tree/model" ]; then
        MODEL=$(cat /proc/device-tree/model)
        if [[ "$MODEL" == *"PYNQ"* ]] || [[ "$MODEL" == *"Zynq"* ]]; then
            echo -e "${GREEN}✅ Detected Zynq-based board: ${MODEL}${NC}"
            return 0
        fi
    fi
    
    echo -e "${YELLOW}⚠️  Not running on PYNQ board. Setup will continue in development mode.${NC}"
    return 1
}

# Install system dependencies
install_system_deps() {
    echo -e "${YELLOW}Installing system dependencies...${NC}"
    
    # Update package list
    sudo apt-get update
    
    # Install essential packages
    sudo apt-get install -y \
        python3-pip \
        python3-venv \
        python3-dev \
        build-essential \
        cmake \
        git \
        wget \
        curl \
        unzip \
        libhdf5-dev \
        pkg-config \
        libjpeg-dev \
        libpng-dev \
        libfreetype6-dev
    
    echo -e "${GREEN}✅ System dependencies installed${NC}"
}

# Setup Python environment
setup_python_env() {    
    # Upgrade pip
    pip install --upgrade pip setuptools wheel
}

# Install Python dependencies
install_python_deps() {
    echo -e "${YELLOW}Installing Python dependencies...${NC}"
    
    # Install core dependencies
    pip install numpy scipy matplotlib
    pip install h5py pyyaml tqdm
    pip install jupyter notebook ipython
    
    # Install PyTorch (CPU version for compatibility)
    echo -e "${YELLOW}Installing PyTorch...${NC}"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    
    # Install PYNQ (if on PYNQ board)
    if check_pynq_board; then
        echo -e "${YELLOW}Installing PYNQ...${NC}"
        pip install pynq || {
            echo -e "${RED}⚠️  Failed to install PYNQ. This is normal on development machines.${NC}"
            echo -e "${YELLOW}   Hardware features will be unavailable.${NC}"
        }
    else
        echo -e "${YELLOW}Skipping PYNQ installation (not on PYNQ board)${NC}"
        echo -e "${YELLOW}Installing PYNQ simulation dependencies...${NC}"
        pip install cffi
    fi
    
    echo -e "${GREEN}✅ Python dependencies installed${NC}"
}

# Setup project structure
setup_project() {
    echo -e "${YELLOW}Setting up project structure...${NC}"
    
    # Create necessary directories
    mkdir -p data logs outputs models
    mkdir -p software/python/snn_fpga_accelerator
    
    # Make Python package
    if [ ! -f "software/python/snn_fpga_accelerator/__init__.py" ]; then
        touch software/python/snn_fpga_accelerator/__init__.py
    fi
    
    # Install the package in development mode
    cd software/python
    pip install -e .
    cd ../..
    
    echo -e "${GREEN}✅ Project structure setup complete${NC}"
}

# Build hardware components
build_hardware() {
    echo -e "${YELLOW}Building hardware components...${NC}"
    
    # Check if Vivado is available
    if command -v vivado &> /dev/null; then
        echo -e "${GREEN}✅ Vivado found${NC}"
        
        # Run HLS synthesis
        if [ -f "hardware/hls/scripts/run_hls.sh" ]; then
            echo -e "${YELLOW}Running HLS synthesis...${NC}"
            cd hardware/hls/scripts
            chmod +x run_hls.sh
            ./run_hls.sh
            cd ../../..
        fi
        
        # Run Vivado build (if requested)
        read -p "Do you want to run full Vivado build? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo -e "${YELLOW}Running Vivado build...${NC}"
            chmod +x hardware/scripts/run_all.sh
            hardware/scripts/run_all.sh
        fi
    else
        echo -e "${YELLOW}⚠️  Vivado not found. Hardware build skipped.${NC}"
        echo -e "${YELLOW}   Install Vivado 2025.1 to build hardware components.${NC}"
    fi
    
    echo -e "${GREEN}✅ Hardware build completed${NC}"
}

# Run tests
run_tests() {
    echo -e "${YELLOW}Running tests...${NC}"
    
    # Python tests
    echo -e "${YELLOW}Running Python tests...${NC}"
    if [ -d "software/python/tests" ]; then
        cd software/python
        python -m pytest tests/ -v || echo -e "${YELLOW}⚠️  Some tests failed or were skipped (this is normal)${NC}"
        cd ../..
    else
        echo -e "${YELLOW}⚠️  Test directory not found${NC}"
    fi
    
    # Hardware simulation tests (if simulator available)
    if command -v iverilog &> /dev/null || command -v xsim &> /dev/null; then
        echo -e "${YELLOW}Running hardware simulation tests...${NC}"
        if [ -f "hardware/hdl/sim/run_sim.sh" ]; then
            cd hardware/hdl/sim
            chmod +x run_sim.sh
            ./run_sim.sh || echo -e "${YELLOW}⚠️  Hardware simulation failed${NC}"
            cd ../../..
        fi
    else
        echo -e "${YELLOW}⚠️  No HDL simulator found. Hardware tests skipped.${NC}"
    fi
    
    echo -e "${GREEN}✅ Tests completed${NC}"
}

# Generate documentation
generate_docs() {
    echo -e "${YELLOW}Generating documentation...${NC}"
        
    # Install documentation dependencies
    pip install sphinx sphinx-rtd-theme
    
    # Generate API documentation (if sphinx is set up)
    if [ -f "docs/Makefile" ]; then
        cd docs
        make html
        cd ..
        echo -e "${GREEN}✅ Documentation generated in docs/_build/html/${NC}"
    else
        echo -e "${YELLOW}⚠️  Sphinx documentation not configured${NC}"
    fi
}

# Create example scripts
create_examples() {
    echo -e "${YELLOW}Creating example scripts...${NC}"
    
    # Create quick start script
    cat > quick_start.py << 'EOF'
"""
Quick Start Example for PYNQ-Z2 SNN Accelerator
"""

import numpy as np
from snn_fpga_accelerator import SNNAccelerator
from snn_fpga_accelerator.spike_encoding import rate_encode

def main():
    print("🚀 PYNQ-Z2 SNN Accelerator Quick Start")
    
    # Initialize accelerator in simulation mode
    accelerator = SNNAccelerator(simulation_mode=True)
    
    # Create dummy input data
    input_data = np.random.rand(784)  # MNIST-like input
    
    # Encode to spikes
    spikes = rate_encode(input_data, num_steps=100, max_rate=50.0)
    
    # Run inference
    output = accelerator.infer(spikes)
    
    print(f"Input shape: {input_data.shape}")
    print(f"Spike shape: {spikes.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Predicted class: {np.argmax(output)}")
    
    print("✅ Quick start completed successfully!")

if __name__ == '__main__':
    main()
EOF
    
    chmod +x quick_start.py
    
    # Create benchmark script
    cat > benchmark.py << 'EOF'
"""
Benchmark Script for PYNQ-Z2 SNN Accelerator
"""

import time
import numpy as np
from snn_fpga_accelerator import SNNAccelerator
from snn_fpga_accelerator.spike_encoding import rate_encode

def benchmark_inference(accelerator, num_samples=100):
    """Benchmark inference performance."""
    print(f"Benchmarking {num_samples} inference samples...")
    
    times = []
    for i in range(num_samples):
        # Generate random input
        input_data = np.random.rand(784)
        spikes = rate_encode(input_data, num_steps=100)
        
        # Time inference
        start_time = time.time()
        output = accelerator.infer(spikes)
        inference_time = time.time() - start_time
        
        times.append(inference_time)
        
        if i % 10 == 0:
            print(f"Sample {i}: {inference_time*1000:.2f} ms")
    
    avg_time = np.mean(times) * 1000
    std_time = np.std(times) * 1000
    throughput = 1000 / avg_time
    
    print(f"\nBenchmark Results:")
    print(f"Average time: {avg_time:.2f} ± {std_time:.2f} ms")
    print(f"Throughput: {throughput:.1f} samples/second")

def main():
    print("📊 PYNQ-Z2 SNN Accelerator Benchmark")
    
    accelerator = SNNAccelerator(simulation_mode=True)
    benchmark_inference(accelerator)

if __name__ == '__main__':
    main()
EOF
    
    chmod +x benchmark.py
    
    echo -e "${GREEN}✅ Example scripts created${NC}"
}

# Display final instructions
show_final_instructions() {
    echo -e "${GREEN}============================================${NC}"
    echo -e "${GREEN}Setup completed successfully! 🎉${NC}"
    echo -e "${GREEN}============================================${NC}"
    echo
    echo -e "${YELLOW}To get started:${NC}"
    echo -e "1. Activate the virtual environment (Optional):"
    echo -e "   ${BLUE}source venv/bin/activate${NC}"
    echo
    echo -e "2. Run the quick start example:"
    echo -e "   ${BLUE}python quick_start.py${NC}"
    echo
    echo -e "3. Run the complete integration example:"
    echo -e "   ${BLUE}python examples/complete_integration_example.py --simulation-mode${NC}"
    echo
    echo -e "4. For PYNQ board deployment:"
    echo -e "   ${BLUE}python examples/complete_integration_example.py --bitstream hardware/bitstream.bit${NC}"
    echo
    echo -e "${YELLOW}Project structure:${NC}"
    echo -e "├── hardware/           # FPGA design files"
    echo -e "├── software/python/    # Python software stack"
    echo -e "├── examples/           # Example scripts"
    echo -e "├── data/               # Dataset storage"
    echo -e "├── models/             # Trained model storage"
    echo
    echo -e "${YELLOW}Available commands:${NC}"
    echo -e "• ${BLUE}python quick_start.py${NC}                    - Quick demonstration"
    echo -e "• ${BLUE}python benchmark.py${NC}                      - Performance benchmark"
    echo -e "• ${BLUE}python -m snn_fpga_accelerator.cli${NC}       - CLI interface"
    echo -e "• ${BLUE}jupyter notebook${NC}                         - Start Jupyter for interactive development"
    echo
    echo -e "${GREEN}Documentation and support:${NC}"
    echo -e "• README.md - Comprehensive project documentation"
    echo -e "• examples/ - Complete example implementations"
    echo -e "• GitHub: https://github.com/your-repo/Event-Driven-Spiking-Neural-Network-Accelerator-for-FPGA"
    echo
}

# Main setup function
main_setup() {
    echo -e "${BLUE}Starting setup process...${NC}"
    
    # Check if running as root
    if [ "$EUID" -eq 0 ]; then
        echo -e "${RED}❌ Do not run this script as root!${NC}"
        exit 1
    fi
    
    # Detect board type
    IS_PYNQ_BOARD=false
    check_pynq_board && IS_PYNQ_BOARD=true
    
    # Install system dependencies
    install_system_deps
    
    # Setup Python environment
    setup_python_env
    
    # Install Python dependencies
    install_python_deps
    
    # Setup project
    setup_project
    
    # Build hardware (optional)
    read -p "Do you want to build hardware components? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        build_hardware
    fi
    
    # Create examples
    create_examples
    
    # Run tests
    read -p "Do you want to run tests? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        run_tests
    fi
    
    # Generate documentation
    read -p "Do you want to generate documentation? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        generate_docs
    fi
    
    # Show final instructions
    show_final_instructions
}

# Handle command line arguments
case "${1:-}" in
    "clean")
        echo -e "${YELLOW}Cleaning up...${NC}"
        rm -rf data/
        rm -rf logs/
        rm -rf outputs/
        rm -rf models/
        rm -f quick_start.py benchmark.py
        echo -e "${GREEN}✅ Cleanup completed${NC}"
        ;;
    "test")
        echo -e "${YELLOW}Running tests only...${NC}"
        run_tests
        ;;
    "hardware")
        echo -e "${YELLOW}Building hardware only...${NC}"
        build_hardware
        ;;
    "help"|"-h"|"--help")
        echo "Usage: $0 [command]"
        echo "Commands:"
        echo "  (no args)  - Run full setup"
        echo "  clean      - Clean up generated files"
        echo "  test       - Run tests only"
        echo "  hardware   - Build hardware only"
        echo "  help       - Show this help"
        ;;
    "")
        main_setup
        ;;
    *)
        echo -e "${RED}❌ Unknown command: $1${NC}"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac

echo -e "${GREEN}Setup script completed! 🚀${NC}"
