#=============================================================================
# Title         : Vitis HLS Build Script for AXI Wrapper
# Project       : PYNQ-Z2 SNN Accelerator  
# File          : run_hls_wrapper.tcl
# Author        : Jiwoon Lee (@metr0jw)
# Description   : TCL script to synthesize HLS AXI wrapper IP
#=============================================================================

# Project settings
set project_name "proj_axi_wrapper"
set solution_name "solution1"
set top_function "axi_hls_wrapper"
set part_name "xc7z020clg400-1"  ;# PYNQ-Z2
set clock_period 10  ;# 100MHz

# Paths
set script_dir [file dirname [info script]]
set hls_dir [file dirname $script_dir]
set src_dir "${hls_dir}/src"
set inc_dir "${hls_dir}/include"
set test_dir "${hls_dir}/test"

puts "============================================"
puts "HLS AXI Wrapper Build Script"
puts "============================================"
puts "Project: $project_name"
puts "Part: $part_name"
puts "Clock: ${clock_period}ns (100MHz)"
puts "============================================"

# Create project
open_project -reset $project_name

# Add source files
add_files "${src_dir}/axi_hls_wrapper.cpp" -cflags "-I${inc_dir}"

# Add testbench
add_files -tb "${test_dir}/tb_axi_hls_wrapper.cpp" -cflags "-I${inc_dir}"

# Set top function
set_top $top_function

# Create solution
open_solution -reset $solution_name

# Set target part
set_part $part_name

# Set clock
create_clock -period $clock_period -name default

# Configure solution
config_interface -m_axi_latency 0
config_interface -s_axilite_sw_reset

# Set optimization directives
set_directive_pipeline -II 1 $top_function

#=============================================================================
# C Simulation
#=============================================================================
puts "\n>>> Running C Simulation..."
if {[catch {csim_design} result]} {
    puts "WARNING: C Simulation failed or skipped"
    puts "Error: $result"
} else {
    puts "C Simulation completed"
}

#=============================================================================
# Synthesis
#=============================================================================
puts "\n>>> Running Synthesis..."
csynth_design

# Print synthesis report summary
puts "\n>>> Synthesis Report Summary:"
puts "============================================"

#=============================================================================
# C/RTL Co-simulation (Optional)
#=============================================================================
# Uncomment to run co-simulation (takes longer)
# puts "\n>>> Running C/RTL Co-simulation..."
# cosim_design -rtl verilog -tool xsim

#=============================================================================
# Export IP
#=============================================================================
puts "\n>>> Exporting IP..."
export_design -rtl verilog -format ip_catalog \
    -description "HLS AXI Wrapper for SNN Accelerator" \
    -vendor "kwu.edu" \
    -library "snn" \
    -version "1.0" \
    -display_name "AXI HLS Wrapper"

puts "\n============================================"
puts "Build Complete!"
puts "============================================"
puts "IP exported to: ${project_name}/${solution_name}/impl/ip/"
puts ""
puts "Next steps:"
puts "1. Add IP repository to Vivado project"
puts "2. Add axi_hls_wrapper IP to block design"
puts "3. Connect to Verilog SNN core modules"
puts "============================================"

exit
