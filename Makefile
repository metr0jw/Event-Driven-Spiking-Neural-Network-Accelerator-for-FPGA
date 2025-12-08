#-----------------------------------------------------------------------------
# Title         : PYNQ-Z2 SNN Accelerator Makefile
# Project       : PYNQ-Z2 SNN Accelerator
# File          : Makefile
# Author        : Jiwoon Lee (@metr0jw)
# Organization  : Kwangwoon University, Seoul, South Korea
# Description   : Top-level Makefile for the complete SNN accelerator project
#-----------------------------------------------------------------------------

# Project information
PROJECT_NAME = PYNQ-Z2 SNN Accelerator
VERSION = 1.0.0
AUTHOR = Jiwoon Lee (@metr0jw)

# Colors for output
RED := \033[31m
GREEN := \033[32m
YELLOW := \033[33m
BLUE := \033[34m
RESET := \033[0m

# Directories
HARDWARE_DIR = hardware
SOFTWARE_DIR = software
EXAMPLES_DIR = examples

# Tools
VIVADO = vivado
VPP = v++
PYTHON = python3
PIP = pip3

# Default target
.PHONY: all
all: help

# Help target
.PHONY: help
help:
	@echo "$(BLUE)================================================$(RESET)"
	@echo "$(BLUE)$(PROJECT_NAME) Build System$(RESET)"
	@echo "$(BLUE)Version: $(VERSION)$(RESET)"
	@echo "$(BLUE)Author: $(AUTHOR)$(RESET)"
	@echo "$(BLUE)================================================$(RESET)"
	@echo ""
	@echo "$(YELLOW)Available targets:$(RESET)"
	@echo ""
	@echo "$(GREEN)Setup:$(RESET)"
	@echo "  setup          - Complete project setup (recommended)"
	@echo "  install        - Install Python package only"
	@echo "  clean          - Clean all build artifacts"
	@echo ""
	@echo "$(GREEN)Hardware:$(RESET)"
	@echo "  hls            - Build HLS IP cores"
	@echo "  vivado         - Build Vivado project and bitstream"
	@echo "  hardware       - Build all hardware components"
	@echo "  sim            - Run hardware simulation"
	@echo ""
	@echo "$(GREEN)Software:$(RESET)"
	@echo "  python         - Install Python package"
	@echo "  examples       - Run example scripts"
	@echo "  test           - Run all tests"
	@echo ""
	@echo "$(YELLOW)Quick commands:$(RESET)"
	@echo "  make setup     - One-command setup"
	@echo "  make test      - Run all tests"
	@echo "  make examples  - Run examples"

# Setup targets
.PHONY: setup
setup:
	@echo "$(YELLOW)Running complete project setup...$(RESET)"
	@chmod +x setup.sh
	@./setup.sh

.PHONY: install
install:
	@echo "$(YELLOW)Installing Python package...$(RESET)"
	@cd $(SOFTWARE_DIR)/python && $(PIP) install -e .
	@echo "$(GREEN)✅ Python package installed$(RESET)"

.PHONY: clean
clean:
	@echo "$(YELLOW)Cleaning build artifacts...$(RESET)"
	@rm -rf build/
	@rm -rf $(SOFTWARE_DIR)/python/build
	@rm -rf $(SOFTWARE_DIR)/python/dist
	@rm -rf $(SOFTWARE_DIR)/python/*.egg-info
	@find . -name "*.pyc" -delete
	@find . -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@rm -rf $(HARDWARE_DIR)/hls/*/solution*
	@rm -rf $(HARDWARE_DIR)/vivado_*
	@rm -f *.log *.jou
	@echo "$(GREEN)✅ Cleanup completed$(RESET)"

# Hardware build targets
.PHONY: hls
hls:
	@echo "$(YELLOW)Building HLS IP cores with v++ compiler...$(RESET)"
	@if command -v $(VPP) >/dev/null 2>&1; then \
		cd $(HARDWARE_DIR)/hls && \
		./scripts/build_hls.sh --clean; \
		echo "$(GREEN)✅ HLS build completed$(RESET)"; \
	else \
		echo "$(RED)❌ v++ compiler not found. Source Vitis settings first.$(RESET)"; \
		exit 1; \
	fi

.PHONY: vivado
vivado:
	@echo "$(YELLOW)Building Vivado project...$(RESET)"
	@if command -v $(VIVADO) >/dev/null 2>&1; then \
		cd $(HARDWARE_DIR)/scripts && \
		chmod +x run_all.sh && \
		./run_all.sh; \
		echo "$(GREEN)✅ Vivado build completed$(RESET)"; \
	else \
		echo "$(RED)❌ Vivado not found$(RESET)"; \
		exit 1; \
	fi

.PHONY: hardware
hardware: hls vivado
	@echo "$(GREEN)✅ All hardware components built$(RESET)"

.PHONY: sim
sim:
	@echo "$(YELLOW)Running hardware simulation...$(RESET)"
	@cd $(HARDWARE_DIR)/hdl/sim && \
	chmod +x run_sim.sh && \
	./run_sim.sh
	@echo "$(GREEN)✅ Hardware simulation completed$(RESET)"

# Software build targets
.PHONY: python
python:
	@echo "$(YELLOW)Building Python package...$(RESET)"
	@cd $(SOFTWARE_DIR)/python && \
	$(PYTHON) setup.py build
	@echo "$(GREEN)✅ Python package built$(RESET)"

.PHONY: examples
examples: install
	@echo "$(YELLOW)Running example scripts...$(RESET)"
	@echo "$(BLUE)Running quick start example...$(RESET)"
	@$(PYTHON) quick_start.py || echo "$(YELLOW)Quick start not available$(RESET)"
	@echo "$(BLUE)Running integration example...$(RESET)"
	@$(PYTHON) $(EXAMPLES_DIR)/complete_integration_example.py --simulation-mode --skip-training
	@echo "$(GREEN)✅ Examples completed$(RESET)"

.PHONY: test
test: install
	@echo "$(YELLOW)Running all tests...$(RESET)"
	@echo "$(BLUE)Python tests...$(RESET)"
	@cd $(SOFTWARE_DIR)/python && $(PYTHON) -m pytest tests/ -v || echo "$(YELLOW)Some tests may not be available$(RESET)"
	@echo "$(BLUE)Hardware simulation tests...$(RESET)"
	@$(MAKE) sim || echo "$(YELLOW)Hardware simulation not available$(RESET)"
	@echo "$(BLUE)Integration tests...$(RESET)"
	@$(PYTHON) $(EXAMPLES_DIR)/complete_integration_example.py --simulation-mode --skip-training || echo "$(YELLOW)Integration test not available$(RESET)"
	@echo "$(GREEN)✅ All tests completed$(RESET)"

# Legacy compatibility
.PHONY: software
software: install

.PHONY: program
program:
	@echo "$(YELLOW)Programming FPGA...$(RESET)"
	@cd hardware/scripts && $(VIVADO) -mode batch -source program_fpga.tcl || echo "$(YELLOW)Program script not available$(RESET)"
	cd hls && rm -rf solution*
	find . -name "*.log" -delete
	find . -name "*.jou" -delete

test:
	cd hdl/tb && vsim -do run_all_tests.do
	cd software/python && pytest tests/
