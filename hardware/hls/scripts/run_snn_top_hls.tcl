#-----------------------------------------------------------------------------
# Title         : SNN Top-Level HLS Synthesis Script
# Project       : PYNQ-Z2 SNN Accelerator
# File          : run_snn_top_hls.tcl
# Author        : Jiwoon Lee (@metr0jw)
# Organization  : Kwangwoon University, Seoul, South Korea
# Contact       : jwlee@linux.com
# Description   : Vitis HLS synthesis script for on-chip learning module
#-----------------------------------------------------------------------------

# Project settings
set project_name "snn_top_hls"
set solution_name "solution1"
set top_function "snn_top_hls"
set part_name "xc7z020clg400-1"
set clock_period 10

# Get script directory
set script_dir [file dirname [file normalize [info script]]]
set hls_root [file normalize "$script_dir/.."]
set proj_root [file normalize "$hls_root/../.."]

# Create project
open_project -reset $project_name

# Add source files
add_files "$hls_root/src/snn_top_hls.cpp" -cflags "-I$hls_root/include"
add_files "$hls_root/include/snn_top_hls.h"

# Add testbench
add_files -tb "$hls_root/test/tb_snn_top_hls.cpp" -cflags "-I$hls_root/include"

# Set top function
set_top $top_function

# Create solution
open_solution -reset $solution_name -flow_target vivado

# Set target device (PYNQ-Z2)
set_part $part_name

# Set clock (100 MHz)
create_clock -period $clock_period -name default

# Configuration directives
config_compile -pipeline_loops 64
config_schedule -effort high
config_bind -effort high
config_export -format ip_catalog \
              -rtl verilog \
              -library hls \
              -vendor kwu \
              -version 1.0 \
              -description "SNN Accelerator with On-Chip Learning"

#-----------------------------------------------------------------------------
# Run C Simulation
#-----------------------------------------------------------------------------
puts "=============================================="
puts "Running C Simulation..."
puts "=============================================="
csim_design -clean

#-----------------------------------------------------------------------------
# Run Synthesis
#-----------------------------------------------------------------------------
puts "=============================================="
puts "Running HLS Synthesis..."
puts "=============================================="
csynth_design

#-----------------------------------------------------------------------------
# Run Co-simulation (optional - takes longer)
#-----------------------------------------------------------------------------
# puts "=============================================="
# puts "Running Co-simulation..."
# puts "=============================================="
# cosim_design -rtl verilog -trace_level all

#-----------------------------------------------------------------------------
# Export IP
#-----------------------------------------------------------------------------
puts "=============================================="
puts "Exporting IP..."
puts "=============================================="
export_design -format ip_catalog \
              -rtl verilog \
              -output "$proj_root/hardware/ip_repo/snn_top_hls_1_0"

#-----------------------------------------------------------------------------
# Summary
#-----------------------------------------------------------------------------
puts ""
puts "=============================================="
puts "Synthesis Complete!"
puts "=============================================="
puts ""
puts "Project: $project_name"
puts "Top Function: $top_function"
puts "Target: $part_name"
puts "Clock: [expr 1000.0/$clock_period] MHz"
puts ""
puts "IP exported to: $proj_root/hardware/ip_repo/snn_top_hls_1_0"
puts ""

exit 0
