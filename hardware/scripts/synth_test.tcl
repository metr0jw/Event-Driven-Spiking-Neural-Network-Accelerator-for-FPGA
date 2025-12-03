##-----------------------------------------------------------------------------
## Title         : Vivado Synthesis Test Script
## Project       : PYNQ-Z2 SNN Accelerator
## File          : synth_test.tcl
## Description   : Simple synthesis test to verify RTL is synthesizable
##-----------------------------------------------------------------------------

# Get the directory where this script is located
set script_dir [file dirname [file normalize [info script]]]
set proj_root [file normalize "$script_dir/../.."]

# Project settings
set part_name "xc7z020clg400-1"
set output_dir "$proj_root/build/synth_test"

# Create output directory
file mkdir $output_dir

# Create in-memory project
create_project -in_memory -part $part_name

# Set HDL directories
set hdl_dir "$proj_root/hardware/hdl"

# Read all Verilog files
puts "Reading RTL files..."

# Common modules
read_verilog [glob -nocomplain $hdl_dir/rtl/common/*.v]

# Neurons
read_verilog [glob -nocomplain $hdl_dir/rtl/neurons/*.v]

# Synapses
read_verilog [glob -nocomplain $hdl_dir/rtl/synapses/*.v]

# Router
read_verilog [glob -nocomplain $hdl_dir/rtl/router/*.v]

# Layers
read_verilog [glob -nocomplain $hdl_dir/rtl/layers/*.v]

# Interfaces
read_verilog [glob -nocomplain $hdl_dir/rtl/interfaces/*.v]

# Top modules
read_verilog [glob -nocomplain $hdl_dir/rtl/top/*.v]

# IP repo sources
read_verilog [glob -nocomplain $proj_root/hardware/ip_repo/*/src/*.v]

# Set top module
set_property top snn_accelerator_top [current_fileset]

# Run synthesis with verbose output
puts "Running synthesis..."
synth_design -top snn_accelerator_top -part $part_name -mode out_of_context

# Generate reports
puts "Generating reports..."
report_utilization -file $output_dir/utilization_report.txt
report_timing_summary -file $output_dir/timing_report.txt
report_drc -file $output_dir/drc_report.txt

# Write checkpoint
write_checkpoint -force $output_dir/post_synth.dcp

puts ""
puts "=============================================="
puts "Synthesis Test Complete"
puts "=============================================="
puts "Reports saved to: $output_dir"
puts ""

# Print summary
puts "Resource Utilization Summary:"
report_utilization -hierarchical -hierarchical_depth 2

exit
