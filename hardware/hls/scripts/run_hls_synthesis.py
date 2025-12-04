#!/usr/bin/env python3
#-----------------------------------------------------------------------------
# Vitis HLS Synthesis Script for AXI HLS Wrapper (Python API)
# Target: PYNQ-Z2 (xc7z020clg400-1)
# Vitis 2025.2 Python API
#-----------------------------------------------------------------------------

import os
import sys
import shutil

# Project parameters
PROJECT_NAME = "axi_hls_wrapper"
TOP_FUNCTION = "axi_hls_wrapper"
PART = "xc7z020clg400-1"
CLOCK_PERIOD = "10"  # 100 MHz

# Get directories
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HLS_DIR = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(HLS_DIR, "output")
CFG_FILE = os.path.join(SCRIPT_DIR, "hls_config.cfg")

print("=" * 60)
print("Vitis HLS Synthesis for AXI HLS Wrapper")
print("=" * 60)
print(f"Project: {PROJECT_NAME}")
print(f"Top Function: {TOP_FUNCTION}")
print(f"Target Part: {PART}")
print(f"Clock Period: {CLOCK_PERIOD} ns")
print(f"Output Dir: {OUTPUT_DIR}")
print(f"Config File: {CFG_FILE}")
print("=" * 60)

# Clean and create output directory
project_path = os.path.join(OUTPUT_DIR, PROJECT_NAME)
if os.path.exists(project_path):
    print(f"Removing existing project: {project_path}")
    shutil.rmtree(project_path)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Source files
src_file = os.path.join(HLS_DIR, "src", "axi_hls_wrapper.cpp")
include_dir = os.path.join(HLS_DIR, "include")

print(f"\nSource file: {src_file}")
print(f"Include dir: {include_dir}")

# Try to import Vitis API
try:
    import vitis
    print("\nVitis API imported successfully")
except ImportError as e:
    print(f"Error: Vitis Python API not available: {e}")
    print("Please run this script using: vitis -s run_hls_synthesis.py")
    sys.exit(1)

try:
    # Create Vitis client
    print("\n[1/4] Creating Vitis client...")
    client = vitis.create_client()
    client.set_workspace(OUTPUT_DIR)
    
    # Create HLS component with config file
    print("[2/4] Creating HLS component...")
    hls_comp = client.create_hls_component(
        name=PROJECT_NAME,
        part=PART,
        cfg_file=CFG_FILE
    )
    
    # Run C Synthesis
    print("[3/4] Running C Synthesis...")
    hls_comp.run(operation="SYNTHESIS")
    
    print("\n" + "=" * 60)
    print("C Synthesis Complete!")
    print("=" * 60)
    
    # Export IP
    print("[4/4] Exporting IP catalog...")
    hls_comp.run(operation="EXPORT")
    
    ip_path = os.path.join(OUTPUT_DIR, PROJECT_NAME, "impl", "ip")
    print("\n" + "=" * 60)
    print("IP Export Complete!")
    print(f"IP Location: {ip_path}")
    print("=" * 60)
    
    # Close client
    client.close()
    
except Exception as e:
    print(f"\nError during HLS synthesis: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nHLS synthesis completed successfully!")

