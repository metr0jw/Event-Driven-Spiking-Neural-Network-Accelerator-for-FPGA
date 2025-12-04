# ============================================================================
# Synthesis Script for Optimized SNN Accelerator
# Target: PYNQ-Z2 (xc7z020clg400-1)
# Description: Compare resource utilization between original and optimized designs
# ============================================================================

# Project settings
set project_name "snn_optimized_synth"
set part_number "xc7z020clg400-1"
set top_module "lif_neuron_array_optimized"

# Get script directory
set script_dir [file dirname [info script]]
set hdl_dir [file join $script_dir ".." "hdl" "rtl"]

# Create project
create_project -force $project_name ./vivado_synth_opt -part $part_number

# Add source files - Optimized modules
add_files [file join $hdl_dir "neurons" "lif_neuron_array_optimized.v"]
add_files [file join $hdl_dir "synapses" "synapse_array_optimized.v"]

# Add common modules
add_files [file join $hdl_dir "common" "reset_sync.v"]

# Set top module
set_property top $top_module [current_fileset]

# Update compile order
update_compile_order -fileset sources_1

# ============================================================================
# Synthesis Settings for Resource Optimization
# ============================================================================

# Enable DSP inference
set_property -name {STEPS.SYNTH_DESIGN.ARGS.MORE OPTIONS} -value {-max_dsp 220} -objects [get_runs synth_1]

# Enable BRAM inference
set_property -name {STEPS.SYNTH_DESIGN.ARGS.MORE OPTIONS} -value {-max_bram 140} -objects [get_runs synth_1]

# Optimization strategy: Performance
set_property strategy Performance_Explore [get_runs synth_1]

# ============================================================================
# Run Synthesis
# ============================================================================
puts "=============================================="
puts " Running Synthesis for Optimized SNN Design"
puts "=============================================="

launch_runs synth_1 -jobs 4
wait_on_run synth_1

# Check synthesis completion
if {[get_property PROGRESS [get_runs synth_1]] != "100%"} {
    puts "ERROR: Synthesis failed!"
    exit 1
}

# ============================================================================
# Generate Reports
# ============================================================================
open_run synth_1

# Resource utilization report
puts "\n=============================================="
puts " Resource Utilization Report"
puts "=============================================="

report_utilization -file [file join ./vivado_synth_opt "optimized_utilization.rpt"]
report_utilization

# Timing report
puts "\n=============================================="
puts " Timing Report"
puts "=============================================="

# Create clock constraint (100 MHz)
create_clock -period 10.000 -name clk [get_ports clk]

report_timing_summary -file [file join ./vivado_synth_opt "optimized_timing.rpt"]
report_timing_summary -max_paths 10

# Power estimate
puts "\n=============================================="
puts " Power Estimate Report"
puts "=============================================="

report_power -file [file join ./vivado_synth_opt "optimized_power.rpt"]

# DSP utilization detail
puts "\n=============================================="
puts " DSP Utilization Detail"
puts "=============================================="

report_utilization -cells [get_cells -hierarchical -filter {REF_NAME =~ DSP*}] -file [file join ./vivado_synth_opt "dsp_utilization.rpt"]

# BRAM utilization detail
puts "\n=============================================="
puts " BRAM Utilization Detail"
puts "=============================================="

report_utilization -cells [get_cells -hierarchical -filter {REF_NAME =~ RAMB*}] -file [file join ./vivado_synth_opt "bram_utilization.rpt"]

# ============================================================================
# Summary Comparison
# ============================================================================
puts "\n=============================================="
puts " Optimization Summary"
puts "=============================================="
puts ""
puts "Original Design (baseline):"
puts "  - LUTs:      4,652 (8.81%)"
puts "  - Registers: 3,195 (3.02%)"
puts "  - BRAM:      2.0   (1.43%)"
puts "  - DSP:       0     (0.00%)"
puts "  - WNS:       +0.159ns"
puts ""
puts "Target (optimized):"
puts "  - LUTs:      ~35%"
puts "  - Registers: ~15%"
puts "  - BRAM:      ~20%"
puts "  - DSP:       ~10%"
puts "  - WNS:       >+1.0ns"
puts ""
puts "Key optimizations:"
puts "  1. 4x more neurons (64 -> 256)"
puts "  2. 2x more parallel units (4 -> 8)"
puts "  3. BRAM-based state storage"
puts "  4. Pipelined memory access"
puts "  5. Improved timing with registered paths"
puts "=============================================="

# Close project
close_project

puts "\n=============================================="
puts " Synthesis Complete!"
puts " Reports saved to: ./vivado_synth_opt/"
puts "=============================================="
